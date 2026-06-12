"""Multimodal content blocks (text + image) for LLM conversations.

Every class here is a small, picklable data class with a ``build``/``autocast``
helper to construct itself from loosely typed input. These primitives are the
minimal layer used by the optimizers to send text and images to an LLM.
"""
from typing import List, Dict, Any, Optional, Literal, Union, Iterable
from dataclasses import dataclass
import base64
from pathlib import Path

from PIL import Image
import io


# Placeholder used when rendering an image as plain text.
DEFAULT_IMAGE_PLACEHOLDER = "\n[IMAGE]\n"


@dataclass
class ContentBase:
    """Abstract base class for all content blocks."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def build(cls, value: Any, **kwargs) -> "ContentBase":
        raise NotImplementedError("Subclasses must implement this method")

    def is_empty(self) -> bool:
        raise NotImplementedError("Subclasses must implement this method")


class ContentBlockList(list):
    """List of content blocks with automatic type conversion.

    Supports automatic conversion from str (-> TextContent), a single
    ContentBlock, a list of ContentBlocks, or None (-> empty list). May contain
    mixed types (text and images).
    """

    def __init__(self, content: Union[str, "ContentBase", List["ContentBase"], None] = None):
        super().__init__()
        if content is not None:
            self.extend(self._normalize(content))

    @staticmethod
    def _normalize(content: Union[str, "ContentBase", List["ContentBase"], None]) -> List["ContentBase"]:
        if content is None:
            return []
        if isinstance(content, str):
            return [TextContent(text=content)] if content else []
        if isinstance(content, list):
            return content
        return [content]

    @classmethod
    def ensure(cls, content: Union[str, "ContentBase", List["ContentBase"], None]) -> "ContentBlockList":
        """Return content as a ContentBlockList, converting if needed."""
        if isinstance(content, cls):
            return content
        return cls(content)

    def __getitem__(self, key: Union[int, slice]) -> Union["ContentBase", "ContentBlockList"]:
        if isinstance(key, slice):
            return ContentBlockList(list.__getitem__(self, key))
        return list.__getitem__(self, key)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "list", "blocks": [b.to_dict() for b in self]}

    def append(self, item: Union[str, "ContentBase", "ContentBlockList"]) -> "ContentBlockList":
        """Append a string or ContentBlock, merging consecutive text blocks."""
        if isinstance(item, str):
            if self and isinstance(self[-1], TextContent):
                self[-1] = TextContent(text=self[-1].text + " " + item)
            else:
                super().append(TextContent(text=item))
        elif isinstance(item, TextContent):
            if self and isinstance(self[-1], TextContent):
                self[-1] = TextContent(text=self[-1].text + " " + item.text)
            else:
                super().append(item)
        elif isinstance(item, ContentBlockList):
            super().extend(item)
        else:
            super().append(item)
        return self

    def extend(self, blocks: Union[str, "ContentBase", List["ContentBase"], "ContentBlockList", None]) -> "ContentBlockList":
        """Extend with blocks, merging consecutive TextContent."""
        for block in self._normalize(blocks):
            self.append(block)
        return self

    def __add__(self, other) -> "ContentBlockList":
        if isinstance(other, (ContentBlockList, list)):
            result = ContentBlockList(list(self))
            result.extend(other)
            return result
        if isinstance(other, str):
            result = ContentBlockList(list(self))
            result.append(TextContent(text=other))
            return result
        return NotImplemented

    def __radd__(self, other) -> "ContentBlockList":
        if isinstance(other, str):
            result = ContentBlockList([TextContent(text=other)])
            result.extend(self)
            return result
        return NotImplemented

    def is_empty(self) -> bool:
        if len(self) == 0:
            return True
        return all(block.is_empty() for block in self)

    def has_images(self) -> bool:
        return any(isinstance(block, ImageContent) for block in self)

    def has_text(self) -> bool:
        return any(isinstance(block, TextContent) for block in self)

    def to_text(self, image_placeholder: str = DEFAULT_IMAGE_PLACEHOLDER) -> str:
        """Text representation where images are replaced with a placeholder.

        Nested ContentBlockLists are handled recursively.
        """
        parts = []
        for block in self:
            if isinstance(block, TextContent):
                parts.append(block.text)
            elif isinstance(block, ImageContent):
                parts.append(image_placeholder)
            elif isinstance(block, ContentBlockList):
                nested = block.to_text(image_placeholder)
                if nested:
                    parts.append(nested)
        return " ".join(parts)

    def __bool__(self) -> bool:
        for block in self:
            if isinstance(block, ImageContent):
                return True
            if isinstance(block, TextContent) and block.text.strip():
                return True
        return False

    def __repr__(self) -> str:
        return self.to_text()

    def to_content_blocks(self) -> "ContentBlockList":
        """Return self (interface compatibility with composite classes)."""
        return self

    def to_litellm_format(self, role: Optional[str] = None) -> List[Dict[str, Any]]:
        """Convert content blocks to LiteLLM/OpenAI Response API format."""
        if role is None:
            role = "user"
        content = []
        for block in self:
            if block.is_empty():
                continue
            if isinstance(block, TextContent):
                content.append(block.to_litellm_format(role=role))
            elif isinstance(block, ImageContent):
                content.append(block.to_litellm_format())
            elif hasattr(block, "to_litellm_format"):
                content.append(block.to_litellm_format())
            else:
                content.append(block.to_dict())
        return content


class Content(ContentBlockList):
    """User-facing multimodal content builder for the optimizer.

    Creation patterns:
    - Variadic: ``Content("text", image, "more text")`` (strings auto-detected
      as text or image paths/URLs)
    - Template: ``Content("See [IMAGE] here", images=[img])``
    - Empty: ``Content()``
    """

    def __init__(self, *args, images: Optional[List[Any]] = None, format: str = "PNG"):
        super().__init__()
        if images is not None:
            if len(args) != 1 or not isinstance(args[0], str):
                raise ValueError(
                    "Template mode requires exactly one template string as the first "
                    f"argument. Got {len(args)} arguments."
                )
            self._build_from_template(args[0], images=images, format=format)
        elif args:
            self._build_from_variadic(*args)

    def _build_from_variadic(self, *args) -> None:
        for arg in args:
            image_content = ImageContent.build(arg)
            if not image_content.is_empty():
                self.append(image_content)
            else:
                self.append(arg)

    def _build_from_template(self, template: str, images: List[Any], format: str = "PNG") -> None:
        placeholder = DEFAULT_IMAGE_PLACEHOLDER
        placeholder_count = template.count(placeholder)
        if placeholder_count != len(images):
            raise ValueError(
                f"Number of {placeholder} placeholders ({placeholder_count}) "
                f"does not match number of images ({len(images)})"
            )
        parts = template.split(placeholder)
        for i, part in enumerate(parts):
            if part:
                self.append(part)
            if i < len(images):
                image_content = ImageContent.build(images[i], format=format)
                if image_content is None:
                    raise ValueError(
                        f"Could not convert image at index {i} to ImageContent: {type(images[i])}"
                    )
                self.append(image_content)


@dataclass
class TextContent(ContentBase):
    """Text content block."""
    type: Literal["text"] = "text"
    text: str = ""

    def __init__(self, text: str = ""):
        super().__init__(text=text)

    def is_empty(self) -> bool:
        return not self.text

    @classmethod
    def build(cls, value: Any = "", **kwargs) -> "TextContent":
        return cls(text=value if isinstance(value, str) else str(value))

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "text": self.text}

    def to_litellm_format(self, role: str = "user") -> Dict[str, Any]:
        """Response API format: input_text for user, output_text for assistant."""
        text_type = "input_text" if role == "user" else "output_text"
        return {"type": text_type, "text": self.text}

    def __add__(self, other) -> "TextContent":
        if isinstance(other, str):
            return TextContent(text=self.text + " " + other)
        if isinstance(other, TextContent):
            return TextContent(text=self.text + " " + other.text)
        return NotImplemented

    def __radd__(self, other) -> "TextContent":
        if isinstance(other, str):
            return TextContent(text=other + " " + self.text)
        return NotImplemented


@dataclass
class ImageContent(ContentBase):
    """Image content block - supports URLs, base64, file paths, bytes, PIL, numpy.

    Storage: ``image_url`` (http/https or data URL), ``image_data`` (base64), or
    ``image_bytes`` (raw bytes; Gemini prefers these). Use :meth:`build` to
    construct from any supported value.
    """
    type: Literal["image"] = "image"
    image_url: Optional[str] = None
    image_data: Optional[str] = None  # base64 encoded
    image_bytes: Optional[bytes] = None
    media_type: str = "image/jpeg"
    detail: Optional[str] = None  # OpenAI: "auto", "low", "high"

    def __init__(self, value: Any = None, format: str = "PNG", **kwargs):
        if kwargs:
            kwargs.setdefault("type", "image")
            kwargs.setdefault("media_type", "image/jpeg")
            super().__init__(**kwargs)
        else:
            super().__init__(**self.autocast(value, format=format))

    def __repr__(self) -> str:
        data = f"{self.image_data[:10]}..." if self.image_data and len(self.image_data) > 10 else self.image_data
        raw = f"{str(self.image_bytes[:10])}..." if self.image_bytes and len(self.image_bytes) > 10 else self.image_bytes
        return f"ImageContent(image_url={self.image_url}, image_data={data}, image_bytes={raw}, media_type={self.media_type})"

    __str__ = __repr__

    def is_empty(self) -> bool:
        return not self.image_url and not self.image_data and not self.image_bytes

    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.type, "media_type": self.media_type}
        if self.image_url:
            result["image_url"] = self.image_url
        if self.image_data:
            result["image_data"] = self.image_data
        if self.image_bytes:
            result["image_bytes"] = self.image_bytes
        if self.detail:
            result["detail"] = self.detail
        return result

    def to_litellm_format(self) -> Dict[str, Any]:
        """Response API format: {"type": "input_image", "image_url": "..."}."""
        if self.image_url:
            url = self.image_url
        elif self.image_data:
            url = f"data:{self.media_type};base64,{self.image_data}"
        elif self.image_bytes:
            b64 = base64.b64encode(self.image_bytes).decode("utf-8")
            url = f"data:{self.media_type};base64,{b64}"
        else:
            return {"type": "input_image", "image_url": ""}
        result = {"type": "input_image", "image_url": url}
        if self.detail:
            result["detail"] = self.detail
        return result

    @staticmethod
    def autocast(value: Any, format: str = "PNG") -> Dict[str, Any]:
        """Auto-detect value type and return image field values.

        Accepts: None, ImageContent, URL/data-URL/file-path strings, raw bytes,
        PIL Images, and numpy arrays.
        """
        empty = {"image_url": None, "image_data": None, "image_bytes": None, "media_type": "image/jpeg"}
        if value is None:
            return dict(empty)

        if isinstance(value, ImageContent):
            return {
                "image_url": value.image_url,
                "image_data": value.image_data,
                "image_bytes": value.image_bytes,
                "media_type": value.media_type,
            }

        if isinstance(value, str):
            if not value.strip():
                return dict(empty)
            if value.startswith("data:image/"):
                try:
                    header, b64 = value.split(",", 1)
                    media_type = header.split(":")[1].split(";")[0]
                    return {"image_url": None, "image_data": b64, "image_bytes": None, "media_type": media_type}
                except (ValueError, IndexError):
                    return {"image_url": None, "image_data": value.split(",")[-1], "image_bytes": None, "media_type": "image/jpeg"}
            if value.startswith("http://") or value.startswith("https://"):
                return {"image_url": value, "image_data": None, "image_bytes": None, "media_type": "image/jpeg"}
            # Treat short strings as possible file paths.
            if len(value) < 4096:
                path = Path(value)
                try:
                    if path.exists():
                        media_type = _ext_to_media_type(path.suffix)
                        with open(value, "rb") as f:
                            image_data = base64.b64encode(f.read()).decode("utf-8")
                        return {"image_url": None, "image_data": image_data, "image_bytes": None, "media_type": media_type}
                except (OSError, IOError):
                    pass

        if isinstance(value, bytes):
            return {"image_url": None, "image_data": base64.b64encode(value).decode("utf-8"), "image_bytes": None, "media_type": "image/jpeg"}

        if isinstance(value, Image.Image):
            buffer = io.BytesIO()
            img_format = value.format or format.upper()
            value.save(buffer, format=img_format)
            buffer.seek(0)
            return {
                "image_url": None,
                "image_data": base64.b64encode(buffer.getvalue()).decode("utf-8"),
                "image_bytes": None,
                "media_type": f"image/{img_format.lower()}",
            }

        try:
            import numpy as np
            if isinstance(value, np.ndarray) or hasattr(value, "__array__"):
                arr = value if isinstance(value, np.ndarray) else np.array(value)
                if arr.dtype in (np.float32, np.float64):
                    arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
                elif arr.dtype != np.uint8:
                    arr = arr.astype(np.uint8)
                image = Image.fromarray(arr)
                buffer = io.BytesIO()
                image.save(buffer, format=format.upper())
                buffer.seek(0)
                return {
                    "image_url": None,
                    "image_data": base64.b64encode(buffer.getvalue()).decode("utf-8"),
                    "image_bytes": None,
                    "media_type": f"image/{format.lower()}",
                }
        except ImportError:
            pass

        return dict(empty)

    @classmethod
    def build(cls, value: Any, format: str = "PNG") -> "ImageContent":
        """Construct an ImageContent from any supported value."""
        if isinstance(value, cls):
            return value
        return cls(**cls.autocast(value, format=format))

    def as_image(self) -> Image.Image:
        """Return the image as a PIL Image, fetching from URL if needed."""
        image_bytes = self.get_bytes()
        if image_bytes:
            return Image.open(io.BytesIO(image_bytes))
        if self.image_url:
            if self.image_url.startswith(("http://", "https://")):
                try:
                    import requests
                    response = requests.get(self.image_url, timeout=30)
                    response.raise_for_status()
                    return Image.open(io.BytesIO(response.content))
                except ImportError:
                    from urllib.request import urlopen
                    with urlopen(self.image_url, timeout=30) as response:
                        return Image.open(io.BytesIO(response.read()))
            return Image.open(self.image_url)
        raise ValueError("No image data available to convert to PIL Image")

    def get_bytes(self) -> Optional[bytes]:
        if self.image_bytes:
            return self.image_bytes
        if self.image_data:
            return base64.b64decode(self.image_data)
        return None

    def get_base64(self) -> Optional[str]:
        if self.image_data:
            return self.image_data
        if self.image_bytes:
            return base64.b64encode(self.image_bytes).decode("utf-8")
        return None


def _ext_to_media_type(suffix: str) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix.lower(), "image/jpeg")


# Union type alias for the supported content types (for type hints).
ContentBlock = Union[TextContent, ImageContent]

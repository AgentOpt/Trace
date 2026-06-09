"""Multimodal content blocks (text + image) for LLM conversations.

Every class here is a small data class that is picklable / JSON-able and offers
an ``autocast``/``build`` helper to construct itself from loosely typed input.
"""
from typing import List, Dict, Any, Optional, Literal, Union, Iterable
from dataclasses import dataclass, field
import base64
from pathlib import Path
import warnings

from PIL import Image
import io


# Default placeholder for images that cannot be rendered as text
DEFAULT_IMAGE_PLACEHOLDER = "\n[IMAGE]\n"


@dataclass
class ContentBase:
    """Abstract base class for all content blocks."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the content block to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the content block
        """
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def build(cls, value: Any, **kwargs) -> 'ContentBase':
        """Build a content block from a value with auto-detection.
        
        Args:
            value: The value to build from (type depends on subclass)
            **kwargs: Additional keyword arguments for building
        
        Returns:
            ContentBase: The built content block
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def is_empty(self) -> bool:
        """Check if the content block is empty (has no meaningful content).
        
        Returns:
            bool: True if the block is empty, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")

class ContentBlockList(list):
    """List of content blocks with automatic type conversion.
    
    Supports automatic conversion from:
    - str -> [TextContent(text=str)]
    - TextContent -> [TextContent]
    - ImageContent -> [ImageContent]
    - List[ContentBlock] -> ContentBlockList
    - None/empty -> []
    
    Note: This list can contain mixed types of ContentBlocks (text, images, PDFs, etc.).
    Type annotations like ContentBlockList[TextContent] are used for documentation
    purposes in specialized methods but don't restrict the actual content.
    """

    def __init__(self, content: Union[str, 'ContentBase', List['ContentBase'], None] = None):
        """Initialize ContentBlockList with automatic type conversion.
        
        Args:
            content: Can be a string (converted to TextContent), a single ContentBlock,
                    a list of ContentBlocks, or None (empty list).
        """
        super().__init__()
        if content is not None:
            self.extend(self._normalize(content))
    
    @staticmethod
    def _normalize(content: Union[str, 'ContentBase', List['ContentBase'], None]) -> List['ContentBase']:
        """Normalize content to a list of ContentBlocks."""
        if content is None:
            return []
        if isinstance(content, str):
            return [TextContent(text=content)] if content else []
        if isinstance(content, list):
            return content
        # Single ContentBlock
        return [content]
    
    @classmethod
    def ensure(cls, content: Union[str, 'ContentBase', List['ContentBase'], None]) -> 'ContentBlockList':
        """Ensure content is a ContentBlockList with automatic conversion.
        
        Args:
            content: String, ContentBlock, list of ContentBlocks, or None
            
        Returns:
            ContentBlockList with the content
        """
        if isinstance(content, cls):
            return content
        return cls(content)
    
    def __getitem__(self, key: Union[int, slice]) -> Union['ContentBase', 'ContentBlockList']:
        """Support indexing and slicing.
        
        Args:
            key: Integer index or slice object
            
        Returns:
            ContentBlock for single index, ContentBlockList for slices
        """
        if isinstance(key, slice):
            # Return a new ContentBlockList with the sliced items
            return ContentBlockList(list.__getitem__(self, key))
        else:
            # Return the single item for integer index
            return list.__getitem__(self, key)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "list", "blocks": [b.to_dict() for b in self]}
    
    def append(self, item: Union[str, 'ContentBase', 'ContentBlockList']) -> 'ContentBlockList':
        """Append a string or ContentBlock, merging consecutive text.
        
        Args:
            item: String (auto-converted to TextContent) or ContentBlock.
                  If the last item is TextContent and item is also text,
                  they are merged into a single TextContent.
        """
        if isinstance(item, str):
            # String: merge with last TextContent or create new one (with a separation mark " ")
            if self and isinstance(self[-1], TextContent):
                self[-1] = TextContent(text=self[-1].text + " " + item)
            else:
                super().append(TextContent(text=item))
        elif isinstance(item, TextContent):
            # TextContent: merge with last TextContent or add (with a separation mark " ")
            if self and isinstance(self[-1], TextContent):
                self[-1] = TextContent(text=self[-1].text + " " + item.text)
            else:
                super().append(item)
        elif isinstance(item, ContentBlockList):
            # we silently call extend here
            super().extend(item)
        else:
            # Other ContentBlock types (ImageContent, etc.): just add
            super().append(item)
        return self
    
    def extend(self, blocks: Union[str, 'ContentBase', List[
        'ContentBase'], 'ContentBlockList', None]) -> 'ContentBlockList':
        """Extend with blocks, merging consecutive TextContent.
        
        Args:
            blocks: String, ContentBlock, list of ContentBlocks, or None.
                    Strings are auto-converted. Consecutive text is merged.
        """
        normalized = self._normalize(blocks)
        for block in normalized:
            self.append(block)
        return self
    
    def __add__(self, other) -> 'ContentBlockList':
        """Concatenate content block lists with other content block lists or strings.
        
        Args:
            other: ContentBlockList, List[ContentBlock], or string to concatenate
        """
        if isinstance(other, (ContentBlockList, list)):
            result = ContentBlockList(list(self))
            result.extend(other)
            return result
        elif isinstance(other, str):
            result = ContentBlockList(list(self))
            result.append(TextContent(text=other))
            return result
        else:
            return NotImplemented
    
    def __radd__(self, other) -> 'ContentBlockList':
        """Right-side concatenation (when string is on the left).
        """
        if isinstance(other, str):
            result = ContentBlockList([TextContent(text=other)])
            result.extend(self)
            return result
        else:
            return NotImplemented

    def is_empty(self) -> bool:
        """Check if the content block list is empty."""
        if len(self) == 0:
            return True
        return all(block.is_empty() for block in self)
    
    def has_images(self) -> bool:
        """Check if the content block list contains any images."""
        return any(isinstance(block, ImageContent) for block in self)

    def has_text(self) -> bool:
        """Check if the content block list contains any text."""
        return any(isinstance(block, TextContent) for block in self)

    # --- Multimodal utilities ---
    @staticmethod
    def blocks_to_text(blocks: Iterable['ContentBase'],
                       image_placeholder: str = DEFAULT_IMAGE_PLACEHOLDER) -> str:
        """Convert any iterable of ContentBlocks to text representation.
        
        This is a utility that can be used by composite classes containing
        multiple ContentBlockLists. Handles nested ContentBlockLists recursively.
        
        Args:
            blocks: Iterable of ContentBlock objects (may include nested ContentBlockLists)
            image_placeholder: Placeholder string for images (default: "[IMAGE]")
            
        Returns:
            str: Text representation where images are replaced with placeholder.
        """
        text_parts = []
        for block in blocks:
            if isinstance(block, TextContent):
                text_parts.append(block.text)
            elif isinstance(block, ImageContent):
                text_parts.append(image_placeholder)
            elif isinstance(block, ContentBlockList):
                # Recursively handle nested ContentBlockList
                nested_text = ContentBlockList.blocks_to_text(block, image_placeholder)
                if nested_text:
                    text_parts.append(nested_text)
        return " ".join(text_parts)
        
    def to_text(self, image_placeholder: str = DEFAULT_IMAGE_PLACEHOLDER) -> str:
        """Convert this list to text representation.
        
        Args:
            image_placeholder: Placeholder string for images (default: "[IMAGE]")
            
        Returns:
            str: Text representation where images are replaced with placeholder.
        """
        return self.blocks_to_text(self, image_placeholder)
    
    def __bool__(self) -> bool:
        """Check if there's any actual content (not just empty text).
        
        Returns:
            bool: True if content is non-empty (has images or non-whitespace text).
        """
        for block in self:
            if isinstance(block, ImageContent):
                return True
            if isinstance(block, TextContent) and block.text.strip():
                return True
        return False
    
    def __repr__(self) -> str:
        """Return text-only representation for logging.
        
        Images are represented as "[IMAGE]" placeholder.
        
        Returns:
            str: Text representation of the content.
        """
        return self.to_text()
    
    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks."""
        try:
            from opto.utils.display.jupyter import render_content_block_list
            return render_content_block_list(self)
        except ImportError:
            # Fallback to text representation if display module unavailable
            return None
    
    def to_content_blocks(self) -> 'ContentBlockList':
        """Return self (for interface compatibility with composites).
        
        This allows ContentBlockList and classes that inherit from it
        to be used interchangeably with composite classes that have
        a to_content_blocks() method.
        
        Returns:
            ContentBlockList: Self reference.
        """
        return self
    
    def count_blocks(self) -> Dict[str, int]:
        """Count blocks by type, including nested structures.
        
        Recursively traverses the content block structure and counts
        each block type by its class name.
        
        Returns:
            Dict[str, int]: Dictionary mapping block class names to counts.
                           Example: {"TextContent": 3, "ImageContent": 1}
        """
        counts: Dict[str, int] = {}
        
        def _count_recursive(item: Any) -> None:
            """Recursively count blocks in nested structures."""
            if isinstance(item, ContentBase):
                # Count this block
                class_name = item.__class__.__name__
                counts[class_name] = counts.get(class_name, 0) + 1
                
                # Check if this block has any attributes that might contain nested blocks
                if hasattr(item, '__dict__'):
                    for attr_value in item.__dict__.values():
                        if isinstance(attr_value, (ContentBlockList, list)):
                            for nested_item in attr_value:
                                _count_recursive(nested_item)
                        elif isinstance(attr_value, ContentBase):
                            _count_recursive(attr_value)
            elif isinstance(item, (ContentBlockList, list)):
                # Recursively count items in lists
                for nested_item in item:
                    _count_recursive(nested_item)
        
        # Count all blocks in this list
        for block in self:
            _count_recursive(block)
        
        return counts
    
    def to_litellm_format(self, role: Optional[str] = None) -> List[Dict[str, Any]]:
        """Convert content blocks to LiteLLM Response API format.
        
        Args:
            role: Optional role context ("user" or "assistant") to determine the correct type.
                  If not provided, defaults to "user" for backward compatibility.
        
        Returns:
            List[Dict[str, Any]]: List of content block dictionaries in Response API format
        """
        if role is None:
            role = "user"
        
        content = []
        for block in self:
            # Skip empty content blocks
            if block.is_empty():
                continue
            
            # Handle different content block types
            if isinstance(block, TextContent):
                # Pass role context to TextContent for proper type selection
                content.append(block.to_litellm_format(role=role))
            elif isinstance(block, ImageContent):
                # ImageContent always uses input_image for user messages
                content.append(block.to_litellm_format())
            elif hasattr(block, 'to_litellm_format'):
                # Fallback: use block's own to_litellm_format method
                content.append(block.to_litellm_format())
            else:
                # Last resort: use to_dict()
                content.append(block.to_dict())
        
        return content


class Content(ContentBlockList):
    """Semantic wrapper providing multi-modal content for the optimizer agent.

    The goal is to provide a flexible interface for user to add mixed text and image content to the optimizer agent.

    Inherits all ContentBlockList functionality (append, extend, has_images,
    to_text, __bool__, __repr__, etc.) with a flexible constructor that
    supports multiple input patterns.

    Primary use cases:
    - Building problem context for the optimizer agent
    - Providing user feedback

    Creation patterns:
    - Variadic: Content("text", image, "more text")
    - Template: Content("See [IMAGE] here", images=[img])
    - Empty: Content()

    Examples:
        # Text-only content
        ctx = Content("Important background information")

        # Image content
        ctx = Content(ImageContent.build("diagram.png"))

        # Mixed content (variadic mode)
        ctx = Content(
            "Here's the diagram:",
            "diagram.png",  # auto-detected as image file
            "And the analysis."
        )

        # Template mode with placeholders
        ctx = Content(
            "Compare [IMAGE] with [IMAGE]:",
            images=[img1, img2]
        )

        # Manual building
        ctx = Content()
        ctx.append("Here's the relevant diagram:")
        ctx.append(ImageContent.build("diagram.png"))
    """

    def __init__(
            self,
            *args,
            images: Optional[List[Any]] = None,
            format: str = "PNG"
    ):
        """Initialize a Content from various input patterns.

        Supports two usage modes:

        **Mode 1: Variadic (images=None)**
        Pass any mix of text and image sources as arguments.
        Strings are auto-detected as text or image paths/URLs.

            Content("Hello", some_image, "World")
            Content("Check this:", "path/to/image.png")

        **Mode 2: Template (images provided)**
        Pass a template string with [IMAGE] placeholders and a list of images.

            Content(
                "Compare [IMAGE] with [IMAGE]",
                images=[img1, img2]
            )

        Args:
            *args: Variable arguments - text strings and/or image sources (Mode 1),
                   or a single template string (Mode 2)
            images: Optional list of images for template mode. When provided,
                    expects exactly one template string in args.
            format: Image format for numpy arrays (PNG, JPEG, etc.). Default: PNG

        Raises:
            ValueError: In template mode, if placeholder count doesn't match image count,
                       or if args is not a single template string.
        """
        # Initialize empty list first
        super().__init__()

        # Build content based on mode
        if images is not None:
            if len(args) != 1 or not isinstance(args[0], str):
                raise ValueError(
                    "Template mode requires exactly one template string as the first argument. "
                    f"Got {len(args)} arguments."
                )
            self._build_from_template(args[0], images=images, format=format)
        elif args:
            self._build_from_variadic(*args)

    def _build_from_variadic(self, *args) -> None:
        """Populate self from variadic arguments.

        Each argument is either text (str) or an image source.
        Strings are auto-detected: if they look like image paths/URLs,
        they're converted to ImageContent; otherwise treated as text.

        Args:
            *args: Alternating text and image sources
            format: Image format for numpy arrays
        """
        for arg in args:
            # for Future expansion, we can check if the string is any special content type
            # by is_empty() on special ContentBlock subclasses
            image_content = ImageContent.build(arg)
            if not image_content.is_empty():
                self.append(image_content)
            else:
                self.append(arg)

    def _build_from_template(
            self,
            template: str,
            images: List[Any],
            format: str = "PNG"
    ) -> None:
        """Populate self from template with [IMAGE] placeholders.

        The template string contains [IMAGE] placeholders that are replaced
        by images from the images list in order.

        Args:
            template: Template string containing [IMAGE] placeholders
            images: List of image sources to insert at placeholders
            format: Image format for numpy arrays

        Raises:
            ValueError: If placeholder count doesn't match the number of images.
        """
        placeholder = DEFAULT_IMAGE_PLACEHOLDER

        # Count placeholders
        placeholder_count = template.count(placeholder)
        if placeholder_count != len(images):
            raise ValueError(
                f"Number of {placeholder} placeholders ({placeholder_count}) "
                f"does not match number of images ({len(images)})"
            )

        # Split template by placeholder and interleave with images
        parts = template.split(placeholder)

        for i, part in enumerate(parts):
            if part:  # Add text part if non-empty
                self.append(part)

            # Add image after each part except the last
            if i < len(images):
                image_content = ImageContent.build(images[i], format=format)
                if image_content is None:
                    raise ValueError(
                        f"Could not convert image at index {i} to ImageContent: {type(images[i])}"
                    )
                self.append(image_content)


@dataclass
class TextContent(ContentBase):
    """Text content block"""
    type: Literal["text"] = "text"
    text: str = ""

    def __init__(self, text: str = ""):
        super().__init__(text=text)

    def is_empty(self) -> bool:
        """Check if the text content is empty."""
        return not self.text

    @classmethod
    def build(cls, value: Any = "", **kwargs) -> 'TextContent':
        """Build a text content block from a value.
        
        Args:
            value: String or any value to convert to text
            **kwargs: Unused, for compatibility with base class
        
        Returns:
            TextContent: Text content block with the value as text
        """
        if isinstance(value, str):
            return cls(text=value)
        return cls(text=str(value))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"type": self.type, "text": self.text}
    
    def to_litellm_format(self, role: str = "user") -> Dict[str, Any]:
        """Convert to LiteLLM/OpenAI Response API compatible format.
        
        Args:
            role: The role context ("user" or "assistant") to determine the correct type
        
        Returns dict in format: 
            - {"type": "input_text", "text": "..."} for user messages
            - {"type": "output_text", "text": "..."} for assistant messages
        """
        text_type = "input_text" if role == "user" else "output_text"
        return {"type": text_type, "text": self.text}
    
    def __add__(self, other) -> 'TextContent':
        """Concatenate text content with strings or other TextContent objects.
        
        Args:
            other: String or TextContent to concatenate
            
        Returns:
            TextContent: New TextContent with concatenated text
        """
        if isinstance(other, str):
            return TextContent(text=self.text + " " + other)
        elif isinstance(other, TextContent):
            return TextContent(text=self.text + " " + other.text)
        else:
            return NotImplemented
    
    def __radd__(self, other) -> 'TextContent':
        """Right-side concatenation (when string is on the left).
        
        Args:
            other: String to concatenate
            
        Returns:
            TextContent: New TextContent with concatenated text
        """
        if isinstance(other, str):
            return TextContent(text=other + " " + self.text)
        else:
            return NotImplemented


@dataclass
class ImageContent(ContentBase):
    """Image content block - supports URLs, base64, file paths, and numpy arrays.

    OpenAI uses base64 encoded images in the image_data field and recombine it into a base64 string of the format `"image_url": f"data:image/jpeg;base64,{base64_image}"` when sending to the API.
    Gemini uses raw bytes in the image_bytes field:
    ```
    types.Part.from_bytes(
        data=image_bytes,
        mime_type='image/jpeg',
      )
    ```
    
    Supports multiple ways to create an ImageContent:
    1. Direct instantiation with image_url or image_data
    2. from_file/from_path: Load from local file path
    3. from_url: Create from HTTP/HTTPS URL
    4. from_array: Create from numpy array or array-like RGB image
    5. from_value: Auto-detect and create from various formats
    """
    type: Literal["image"] = "image"
    image_url: Optional[str] = None
    image_data: Optional[str] = None  # base64 encoded
    image_bytes: Optional[bytes] = None
    media_type: str = "image/jpeg"  # image/jpeg, image/png, image/gif, image/webp
    detail: Optional[str] = None  # OpenAI: "auto", "low", "high"

    def __init__(self, value: Any = None, format: str = "PNG", **kwargs):
        """Initialize ImageContentBlock with auto-detection of input type.
        
        Args:
            value: Can be:
                - URL string (starting with 'http://' or 'https://')
                - Data URL string (starting with 'data:image/')
                - Local file path (string)
                - Numpy array or array-like RGB image
                - PIL Image object
                - Raw bytes
                - None (empty image)
            format: Image format for numpy arrays (PNG, JPEG, etc.). Default: PNG
            **kwargs: Direct field values (image_url, image_data, media_type, detail)
        """
        # If explicit field values are provided, use them directly
        if kwargs:
            kwargs.setdefault('type', 'image')
            kwargs.setdefault('media_type', 'image/jpeg')
            super().__init__(**kwargs)
        else:
            # Use autocast to detect and convert the value
            value_dict = self.autocast(value, format=format)
            super().__init__(**value_dict)

    def __str__(self) -> str:
        # Truncate image_data and image_bytes for readability
        image_data_str = f"{self.image_data[:10]}..." if self.image_data and len(self.image_data) > 10 else self.image_data
        image_bytes_str = f"{str(self.image_bytes[:10])}..." if self.image_bytes and len(self.image_bytes) > 10 else self.image_bytes
        return f"ImageContent(image_url={self.image_url}, image_data={image_data_str}, image_bytes={image_bytes_str}, media_type={self.media_type})"
    
    def __repr__(self) -> str:
        # Truncate image_data and image_bytes for readability
        image_data_str = f"{self.image_data[:10]}..." if self.image_data and len(self.image_data) > 10 else self.image_data
        image_bytes_str = f"{str(self.image_bytes[:10])}..." if self.image_bytes and len(self.image_bytes) > 10 else self.image_bytes
        return f"ImageContent(image_url={self.image_url}, image_data={image_data_str}, image_bytes={image_bytes_str}, media_type={self.media_type})"

    def is_empty(self) -> bool:
        """Check if the image content is empty (no URL or data)."""
        return not self.image_url and not self.image_data and not self.image_bytes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (not LiteLLM format).
        
        For LiteLLM format, use to_litellm_format() instead.
        """
        result = {
            "type": self.type,
            "media_type": self.media_type
        }
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
        """Convert to LiteLLM Response API compatible format.
        
        Returns dict in format:
        {"type": "input_image", "image_url": {"url": "..."}}
        """
        # Determine the URL to use
        if self.image_url:
            url = self.image_url
        elif self.image_data:
            # Convert base64 data to data URL
            url = f"data:{self.media_type};base64,{self.image_data}"
        elif self.image_bytes:
            # Convert bytes to base64 and then to data URL
            import base64
            b64_data = base64.b64encode(self.image_bytes).decode('utf-8')
            url = f"data:{self.media_type};base64,{b64_data}"
        else:
            # Empty image
            return {"type": "input_image", "image_url": ""}
        
        # Build the result in Response API format
        result = {
            "type": "input_image",
            "image_url": url
        }
        
        # Add detail if specified (OpenAI-specific)
        if self.detail:
            result["detail"] = self.detail
            
        return result

    @classmethod
    def from_file(cls, filepath: str, media_type: Optional[str] = None):
        """Load image from file path."""
        path = Path(filepath)
        if not media_type:
            ext_to_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            media_type = ext_to_type.get(path.suffix.lower(), 'image/jpeg')

        with open(filepath, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        return cls(image_data=image_data, media_type=media_type)

    @classmethod
    def from_path(cls, filepath: str, media_type: Optional[str] = None):
        """Load image from file path. Alias for from_file."""
        return cls.from_file(filepath, media_type)

    @classmethod
    def from_url(cls, url: str, media_type: str = "image/jpeg"):
        """Create ImageContent from an HTTP/HTTPS URL.
        
        Args:
            url: HTTP or HTTPS URL pointing to an image
            media_type: MIME type of the image (default: image/jpeg)
        """
        return cls(image_url=url, media_type=media_type)

    @classmethod
    def from_array(cls, array: Any, format: str = "PNG"):
        """Create ImageContent from a numpy array or array-like RGB image.
        
        Args:
            array: numpy array representing an image (H, W, C) with values in [0, 255] or [0, 1]
            format: Image format (PNG, JPEG, etc.). Default: PNG
        
        Returns:
            ImageContent with base64-encoded image data
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy is required for from_array. Install with: pip install numpy")
        
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required for from_array. Install with: pip install Pillow")
        
        import io
        
        # Convert to numpy array if not already
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        
        # Normalize to [0, 255] if needed
        if array.dtype == np.float32 or array.dtype == np.float64:
            if array.max() <= 1.0:
                array = (array * 255).astype(np.uint8)
            else:
                array = array.astype(np.uint8)
        elif array.dtype != np.uint8:
            array = array.astype(np.uint8)
        
        # Convert to PIL Image and encode
        image = Image.fromarray(array)
        buffer = io.BytesIO()
        image.save(buffer, format=format.upper())
        buffer.seek(0)
        
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        media_type = f"image/{format.lower()}"
        
        return cls(image_data=image_data, media_type=media_type)

    @classmethod
    def from_pil(cls, image: Any, format: str = "PNG"):
        """Create ImageContent from a PIL Image.
        
        Args:
            image: PIL Image object
            format: Image format (PNG, JPEG, etc.). Default: PNG
        
        Returns:
            ImageContent with base64-encoded image data
        """
        import io
        
        buffer = io.BytesIO()
        img_format = image.format or format.upper()
        image.save(buffer, format=img_format)
        buffer.seek(0)
        
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        media_type = f"image/{img_format.lower()}"
        
        return cls(image_data=image_data, media_type=media_type)

    @classmethod
    def from_bytes(cls, data: bytes, media_type: str = "image/jpeg"):
        """Create ImageContent from raw image bytes.
        
        Args:
            data: Raw image bytes
            media_type: MIME type of the image (default: image/jpeg)
        
        Returns:
            ImageContent with base64-encoded data
        """
        image_data = base64.b64encode(data).decode('utf-8')
        return cls(image_data=image_data, media_type=media_type)

    @classmethod
    def from_base64(cls, b64_data: str, media_type: str = "image/jpeg"):
        """Create ImageContent from base64-encoded string.
        
        Args:
            b64_data: Base64-encoded image data (without data URL prefix)
            media_type: MIME type of the image (default: image/jpeg)
        
        Returns:
            ImageContent with the provided base64 data
        """
        return cls(image_data=b64_data, media_type=media_type)

    @classmethod
    def from_data_url(cls, data_url: str):
        """Create ImageContent from a data URL (data:image/...;base64,...).
        
        Args:
            data_url: Data URL string in format data:image/<type>;base64,<data>
        
        Returns:
            ImageContent with extracted base64 data and media type
        """
        try:
            header, b64_data = data_url.split(',', 1)
            media_type = header.split(':')[1].split(';')[0]  # e.g., "image/png"
            return cls(image_data=b64_data, media_type=media_type)
        except (ValueError, IndexError):
            # Fallback: assume the whole thing is base64 data
            return cls(image_data=data_url.split(',')[-1], media_type="image/jpeg")

    @staticmethod
    def autocast(value: Any, format: str = "PNG") -> Dict[str, Any]:
        """Auto-detect value type and return image field values.
        
        Args:
            value: Can be:
                - URL string (starting with 'http://' or 'https://')
                - Data URL string (starting with 'data:image/')
                - Local file path (string)
                - Numpy array or array-like RGB image
                - PIL Image object
                - Raw bytes
                - None (empty image)
            format: Image format for numpy arrays (PNG, JPEG, etc.). Default: PNG
        
        Returns:
            Dictionary with keys: image_url, image_data, image_bytes, media_type
        """
        # Handle None or empty
        if value is None:
            return {"image_url": None, "image_data": None, "image_bytes": None, "media_type": "image/jpeg"}
        
        # Handle ImageContentBlock instance
        if isinstance(value, ImageContent):
            return {
                "image_url": value.image_url, 
                "image_data": value.image_data, 
                "image_bytes": value.image_bytes,
                "media_type": value.media_type
            }
        
        # Handle string inputs
        if isinstance(value, str):
            if not value.strip():
                return {"image_url": None, "image_data": None, "image_bytes": None, "media_type": "image/jpeg"}
            
            # Data URL
            if value.startswith('data:image/'):
                try:
                    header, b64_data = value.split(',', 1)
                    media_type = header.split(':')[1].split(';')[0]
                    return {"image_url": None, "image_data": b64_data, "image_bytes": None, "media_type": media_type}
                except (ValueError, IndexError):
                    return {"image_url": None, "image_data": value.split(',')[-1], "image_bytes": None, "media_type": "image/jpeg"}
            
            # HTTP/HTTPS URL
            if value.startswith('http://') or value.startswith('https://'):
                return {"image_url": value, "image_data": None, "image_bytes": None, "media_type": "image/jpeg"}
            
            # File path - only check if string is reasonable length (< 4096 chars)
            # Long strings are clearly not file paths and would cause OS errors
            if len(value) < 4096:
                path = Path(value)
                try:
                    if path.exists():
                        ext_to_type = {
                            '.jpg': 'image/jpeg',
                            '.jpeg': 'image/jpeg',
                            '.png': 'image/png',
                            '.gif': 'image/gif',
                            '.webp': 'image/webp'
                        }
                        media_type = ext_to_type.get(path.suffix.lower(), 'image/jpeg')
                        with open(value, 'rb') as f:
                            image_data = base64.b64encode(f.read()).decode('utf-8')
                        return {"image_url": None, "image_data": image_data, "image_bytes": None, "media_type": media_type}
                except (OSError, IOError):
                    # Not a valid file path, continue to other checks
                    pass
                    
        # Handle bytes - store as base64 for portability
        if isinstance(value, bytes):
            image_data = base64.b64encode(value).decode('utf-8')
            return {"image_url": None, "image_data": image_data, "image_bytes": None, "media_type": "image/jpeg"}
        
        # Handle PIL Image
        try:
            from PIL import Image
            if isinstance(value, Image.Image):
                import io
                buffer = io.BytesIO()
                img_format = value.format or format.upper()
                value.save(buffer, format=img_format)
                buffer.seek(0)
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                media_type = f"image/{img_format.lower()}"
                return {"image_url": None, "image_data": image_data, "image_bytes": None, "media_type": media_type}
        except ImportError:
            pass
        
        # Handle numpy array or array-like
        try:
            import numpy as np
            if isinstance(value, np.ndarray) or hasattr(value, '__array__'):
                try:
                    from PIL import Image
                except ImportError:
                    raise ImportError("Pillow is required for array conversion. Install with: pip install Pillow")
                
                import io
                
                if not isinstance(value, np.ndarray):
                    value = np.array(value)
                
                # Normalize to [0, 255] if needed
                if value.dtype == np.float32 or value.dtype == np.float64:
                    if value.max() <= 1.0:
                        value = (value * 255).astype(np.uint8)
                    else:
                        value = value.astype(np.uint8)
                elif value.dtype != np.uint8:
                    value = value.astype(np.uint8)
                
                image = Image.fromarray(value)
                buffer = io.BytesIO()
                image.save(buffer, format=format.upper())
                buffer.seek(0)
                
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                media_type = f"image/{format.lower()}"
                return {"image_url": None, "image_data": image_data, "image_bytes": None, "media_type": media_type}
        except ImportError:
            pass
        
        return {"image_url": None, "image_data": None, "image_bytes": None, "media_type": "image/jpeg"}

    @classmethod
    def build(cls, value: Any, format: str = "PNG") -> 'ImageContent':
        """Auto-detect format and create ImageContent from various input types.
        
        Args:
            value: Can be:
                - URL string (starting with 'http://' or 'https://')
                - Data URL string (starting with 'data:image/')
                - Local file path (string)
                - Numpy array or array-like RGB image
                - PIL Image object
                - Raw bytes
            format: Image format for numpy arrays (PNG, JPEG, etc.). Default: PNG
        
        Returns:
            ImageContent or None if the value cannot be converted
        """
        # Handle ImageContentBlock instance directly
        if isinstance(value, cls):
            return value
        
        value_dict = cls.autocast(value, format=format)
        return cls(**value_dict)

    def set_image(self, image: Any, format: str = "PNG") -> None:
        """Set the image from various input formats (mutates self).
        
        Args:
            image: Can be:
                - URL string (starting with 'http://' or 'https://')
                - Data URL string (starting with 'data:image/')
                - Local file path (string)
                - Numpy array or array-like RGB image
                - PIL Image object
                - Raw bytes
            format: Image format for numpy arrays (PNG, JPEG, etc.). Default: PNG
        """
        result = ImageContent.build(image, format=format)
        if result:
            self.image_url = result.image_url
            self.image_data = result.image_data
            # Only copy image_bytes if it was explicitly set (e.g., from Google API)
            if result.image_bytes:
                self.image_bytes = result.image_bytes
            self.media_type = result.media_type

    def as_image(self) -> Image.Image:
        """Convert the image to a PIL Image.
        
        Fetches the image from URL if necessary (including HTTP/HTTPS URLs).
        
        Returns:
            PIL Image object
            
        Raises:
            ValueError: If no image data is available
            requests.RequestException: If fetching from URL fails
        """
        # Try to get image bytes from any available source
        image_bytes = self.get_bytes()
        
        if image_bytes:
            return Image.open(io.BytesIO(image_bytes))
        elif self.image_url:
            if self.image_url.startswith(('http://', 'https://')):
                # Fetch image from URL
                try:
                    import requests
                    response = requests.get(self.image_url, timeout=30)
                    response.raise_for_status()
                    return Image.open(io.BytesIO(response.content))
                except ImportError:
                    # Fallback to urllib if requests is not available
                    from urllib.request import urlopen
                    with urlopen(self.image_url, timeout=30) as response:
                        return Image.open(io.BytesIO(response.read()))
            else:
                # If it's a local file path
                return Image.open(self.image_url)
        else:
            raise ValueError("No image data available to convert to PIL Image")

    def show(self) -> Image.Image:
        """A convenience alias for as_image()"""
        return self.as_image()
    
    def get_bytes(self) -> Optional[bytes]:
        """Get raw image bytes.
        
        Returns image_bytes if available, otherwise decodes image_data from base64.
        
        Returns:
            Raw image bytes or None if no image data available
        """
        if self.image_bytes:
            return self.image_bytes
        elif self.image_data:
            return base64.b64decode(self.image_data)
        return None
    
    def get_base64(self) -> Optional[str]:
        """Get base64-encoded image data.
        
        Returns image_data if available, otherwise encodes image_bytes to base64.
        
        Returns:
            Base64-encoded string or None if no image data available
        """
        if self.image_data:
            return self.image_data
        elif self.image_bytes:
            return base64.b64encode(self.image_bytes).decode('utf-8')
        return None
    
    def ensure_bytes(self) -> None:
        """Ensure image_bytes is populated (converts from image_data if needed)."""
        if not self.image_bytes and self.image_data:
            self.image_bytes = base64.b64decode(self.image_data)
    
    def ensure_base64(self) -> None:
        """Ensure image_data is populated (converts from image_bytes if needed)."""
        if not self.image_data and self.image_bytes:
            self.image_data = base64.b64encode(self.image_bytes).decode('utf-8')


# Union type alias for the supported content types (for type hints).
ContentBlock = Union[TextContent, ImageContent]

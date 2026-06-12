"""Conversation turns: :class:`UserTurn` and :class:`AssistantTurn`.

``AssistantTurn.autocast`` parses raw responses from LiteLLM/OpenAI (Responses
and Completion APIs), Bedrock Converse, and Google GenAI into a uniform shape.
The :func:`to_messages` helper builds a provider-ready messages list from a
system prompt and a user content block list (the minimal replacement for the
old ``Chat`` manager).
"""
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field

from .content import ContentBlockList, TextContent, ImageContent


@dataclass
class UserTurn:
    """A user message turn (role + multimodal content)."""
    role: str = "user"
    content: ContentBlockList = field(default_factory=ContentBlockList)

    def __init__(self, content=None, **kwargs):
        if isinstance(content, UserTurn):
            self.role = content.role
            self.content = ContentBlockList(content.content)
            return
        if content is None:
            content = ContentBlockList()
        elif not isinstance(content, ContentBlockList):
            content = ContentBlockList(content) if isinstance(content, list) else ContentBlockList([content])
        self.role = kwargs.get("role", "user")
        self.content = content

    def add_text(self, text: str) -> "UserTurn":
        self.content.append(TextContent(text=text))
        return self

    def add_image(self, url: Optional[str] = None, data: Optional[str] = None,
                  media_type: str = "image/jpeg") -> "UserTurn":
        self.content.append(ImageContent(image_url=url, image_data=data, media_type=media_type))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {"role": "user", "content": [c.to_dict() for c in self.content]}

    def to_litellm_format(self) -> Dict[str, Any]:
        return {"role": "user", "content": self.content.to_litellm_format(role="user")}

    def __repr__(self) -> str:
        preview = str(self.content)
        preview = preview[:50] + "..." if len(preview) > 50 else preview
        return f"UserTurn(content={preview!r})"


@dataclass
class Turn:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class AssistantTurn(Turn):
    """An assistant message turn, parsed from a raw LLM response."""
    role: str = "assistant"
    content: ContentBlockList = field(default_factory=ContentBlockList)

    reasoning: Optional[str] = None
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    model: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, *args, **kwargs):
        """Initialize empty, from another AssistantTurn, from a raw response
        (single positional arg), or from explicit fields (kwargs)."""
        if len(args) == 1 and isinstance(args[0], AssistantTurn):
            other = args[0]
            super().__init__(
                role=other.role,
                content=ContentBlockList(other.content),
                reasoning=other.reasoning,
                finish_reason=other.finish_reason,
                prompt_tokens=other.prompt_tokens,
                completion_tokens=other.completion_tokens,
                model=other.model,
                timestamp=other.timestamp,
                metadata=dict(other.metadata),
            )
        elif len(args) > 0 and len(kwargs) == 0:
            super().__init__(**self.autocast(args[0]))
        elif len(kwargs) > 0:
            super().__init__(**kwargs)
        else:
            super().__init__(
                role="assistant",
                content=ContentBlockList(),
                reasoning=None,
                finish_reason=None,
                prompt_tokens=None,
                completion_tokens=None,
                model=None,
                timestamp=None,
                metadata={},
            )

    @staticmethod
    def from_google_genai(value: Any) -> Dict[str, Any]:
        """Parse a Google GenAI response (generate_content or Interactions API)."""
        result = {
            "role": "assistant",
            "content": ContentBlockList(),
            "reasoning": None,
            "finish_reason": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "model": None,
            "timestamp": None,
            "metadata": {},
        }

        raw_response = value.raw_response if hasattr(value, "raw_response") else value

        # Interactions API (new): has 'outputs'
        if hasattr(raw_response, "outputs"):
            interaction = raw_response
            if interaction.outputs:
                for output in interaction.outputs:
                    if hasattr(output, "text") and output.text:
                        result["content"].append(TextContent(text=output.text))
                    elif hasattr(output, "content"):
                        if isinstance(output.content, list):
                            for part in output.content:
                                if hasattr(part, "text") and part.text:
                                    result["content"].append(TextContent(text=part.text))
                        else:
                            result["content"].append(TextContent(text=str(output.content)))
            if hasattr(interaction, "model"):
                result["model"] = interaction.model
            if hasattr(interaction, "status"):
                result["finish_reason"] = interaction.status
            if hasattr(interaction, "usage"):
                usage = interaction.usage
                if hasattr(usage, "input_tokens"):
                    result["prompt_tokens"] = usage.input_tokens
                elif hasattr(usage, "prompt_token_count"):
                    result["prompt_tokens"] = usage.prompt_token_count
                if hasattr(usage, "output_tokens"):
                    result["completion_tokens"] = usage.output_tokens
                elif hasattr(usage, "candidates_token_count"):
                    result["completion_tokens"] = usage.candidates_token_count
            if hasattr(interaction, "id"):
                result["metadata"]["interaction_id"] = interaction.id
            return result

        # Legacy generate_content API
        if hasattr(raw_response, "thoughts") and raw_response.thoughts:
            result["reasoning"] = str(raw_response.thoughts)
        if hasattr(raw_response, "model_version"):
            result["model"] = raw_response.model_version
        if hasattr(raw_response, "usage_metadata"):
            usage = raw_response.usage_metadata
            if hasattr(usage, "prompt_token_count"):
                result["prompt_tokens"] = usage.prompt_token_count
            if hasattr(usage, "candidates_token_count"):
                result["completion_tokens"] = usage.candidates_token_count

        content_extracted = False
        if hasattr(raw_response, "candidates") and raw_response.candidates:
            candidate = raw_response.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        result["content"].append(TextContent(text=part.text))
                        content_extracted = True
                    elif hasattr(part, "inline_data"):
                        inline = part.inline_data
                        image_bytes = None
                        image_data = None
                        media_type = "image/jpeg"
                        data = None
                        if hasattr(inline, "data"):
                            data = inline.data
                            if isinstance(data, bytes):
                                image_bytes = data
                            elif isinstance(data, str):
                                image_data = data
                        if hasattr(inline, "mime_type"):
                            media_type = inline.mime_type
                        if image_data or image_bytes:
                            result["content"].append(ImageContent(
                                image_data=image_data,
                                image_bytes=image_bytes if isinstance(data, bytes) else None,
                                media_type=media_type,
                            ))
                            content_extracted = True
            if hasattr(candidate, "finish_reason"):
                result["finish_reason"] = str(candidate.finish_reason)

        if not content_extracted:
            if hasattr(raw_response, "text"):
                result["content"].append(TextContent(text=raw_response.text))
            elif hasattr(value, "choices"):
                result["content"].append(TextContent(text=value.choices[0].message.content))

        return result

    @staticmethod
    def from_litellm_openai_response_api(value: Any) -> Dict[str, Any]:
        """Parse a LiteLLM/OpenAI response (Responses API, Completion API, or
        Bedrock Converse)."""
        result = {
            "role": "assistant",
            "content": ContentBlockList(),
            "reasoning": None,
            "finish_reason": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "model": None,
            "timestamp": None,
            "metadata": {},
        }

        # Bedrock Converse: 'output' with nested 'message'
        is_bedrock = False
        bedrock_output = None
        bedrock_value = value
        if hasattr(value, "output"):
            output_val = value.output
            if hasattr(output_val, "message"):
                is_bedrock = True
                bedrock_output = output_val
            elif isinstance(output_val, dict) and "message" in output_val:
                is_bedrock = True
                bedrock_output = output_val
        if not is_bedrock and isinstance(value, dict) and "output" in value:
            output_val = value["output"]
            if isinstance(output_val, dict) and "message" in output_val:
                is_bedrock = True
                bedrock_output = output_val
                bedrock_value = value

        if is_bedrock and bedrock_output is not None:
            message = bedrock_output.get("message") if isinstance(bedrock_output, dict) else getattr(bedrock_output, "message", None)
            if message:
                if isinstance(message, dict):
                    result["role"] = message.get("role", "assistant")
                elif hasattr(message, "role"):
                    result["role"] = message.role
                content_list = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
                if content_list:
                    for content_item in content_list:
                        text_val = content_item.get("text") if isinstance(content_item, dict) else getattr(content_item, "text", None)
                        if text_val:
                            result["content"].append(TextContent(text=text_val))
            stop_reason = bedrock_value.get("stopReason") if isinstance(bedrock_value, dict) else getattr(bedrock_value, "stopReason", None)
            if stop_reason:
                result["finish_reason"] = stop_reason
            usage = bedrock_value.get("usage") if isinstance(bedrock_value, dict) else getattr(bedrock_value, "usage", None)
            if usage:
                if isinstance(usage, dict):
                    result["prompt_tokens"] = usage.get("inputTokens")
                    result["completion_tokens"] = usage.get("outputTokens")
                else:
                    if hasattr(usage, "inputTokens"):
                        result["prompt_tokens"] = usage.inputTokens
                    if hasattr(usage, "outputTokens"):
                        result["completion_tokens"] = usage.outputTokens
            return result

        # Responses API: 'output' list + 'object'
        if hasattr(value, "output") and hasattr(value, "object"):
            if hasattr(value, "id"):
                result["metadata"]["response_id"] = value.id
            if hasattr(value, "created_at"):
                result["timestamp"] = str(value.created_at)
            if hasattr(value, "model"):
                result["model"] = value.model
            if hasattr(value, "status"):
                result["finish_reason"] = value.status
            if value.output:
                for output_item in value.output:
                    if getattr(output_item, "type", None) == "image_generation_call":
                        if getattr(output_item, "result", None):
                            media_type = "image/jpeg"
                            if hasattr(output_item, "output_format"):
                                format_map = {
                                    "png": "image/png",
                                    "jpeg": "image/jpeg",
                                    "jpg": "image/jpeg",
                                    "webp": "image/webp",
                                    "gif": "image/gif",
                                }
                                media_type = format_map.get(output_item.output_format.lower(), "image/jpeg")
                            result["content"].append(ImageContent(image_data=output_item.result, media_type=media_type))
                            if getattr(output_item, "revised_prompt", None):
                                result["metadata"].setdefault("image_generation", []).append({
                                    "id": getattr(output_item, "id", None),
                                    "revised_prompt": output_item.revised_prompt,
                                    "size": getattr(output_item, "size", None),
                                    "quality": getattr(output_item, "quality", None),
                                    "status": getattr(output_item, "status", None),
                                })
                    elif getattr(output_item, "type", None) == "message":
                        if hasattr(output_item, "role"):
                            result["role"] = output_item.role
                        if hasattr(output_item, "status") and not result["finish_reason"]:
                            result["finish_reason"] = output_item.status
                        if getattr(output_item, "content", None):
                            for content_item in output_item.content:
                                if getattr(content_item, "type", None) == "output_text" and getattr(content_item, "text", None):
                                    result["content"].append(TextContent(text=content_item.text))
                                elif getattr(content_item, "text", None):
                                    result["content"].append(TextContent(text=str(content_item.text)))
            if hasattr(value, "reasoning"):
                if isinstance(value.reasoning, dict):
                    reasoning_parts = []
                    if value.reasoning.get("summary"):
                        reasoning_parts.append(f"Summary: {value.reasoning['summary']}")
                    if value.reasoning.get("effort"):
                        reasoning_parts.append(f"Effort: {value.reasoning['effort']}")
                    if reasoning_parts:
                        result["reasoning"] = "\n".join(reasoning_parts)
                elif value.reasoning:
                    result["reasoning"] = str(value.reasoning)
            if hasattr(value, "usage"):
                if hasattr(value.usage, "input_tokens"):
                    result["prompt_tokens"] = value.usage.input_tokens
                if hasattr(value.usage, "output_tokens"):
                    result["completion_tokens"] = value.usage.output_tokens
            return result

        # Legacy Completion API: 'choices'
        if hasattr(value, "choices") and len(value.choices) > 0:
            choice = value.choices[0]
            message = choice.message if hasattr(choice, "message") else choice
            if hasattr(message, "content") and message.content:
                result["content"].append(TextContent(text=str(message.content)))
            if hasattr(choice, "finish_reason"):
                result["finish_reason"] = choice.finish_reason
            if hasattr(message, "reasoning") and message.reasoning:
                result["reasoning"] = message.reasoning
            if hasattr(value, "usage"):
                if hasattr(value.usage, "prompt_tokens"):
                    result["prompt_tokens"] = value.usage.prompt_tokens
                if hasattr(value.usage, "completion_tokens"):
                    result["completion_tokens"] = value.usage.completion_tokens
            if hasattr(value, "model"):
                result["model"] = value.model

        return result

    @staticmethod
    def autocast(value: Any) -> Dict[str, Any]:
        """Detect the response format and parse into AssistantTurn fields."""
        raw_response = value.raw_response if hasattr(value, "raw_response") else value

        if hasattr(raw_response, "outputs") or \
           (hasattr(raw_response, "candidates") and not hasattr(value, "choices")) or \
           hasattr(raw_response, "usage_metadata"):
            return AssistantTurn.from_google_genai(value)

        has_output = hasattr(value, "output") or (isinstance(value, dict) and "output" in value)
        has_choices = hasattr(value, "choices") or (isinstance(value, dict) and "choices" in value)
        if has_output or has_choices:
            return AssistantTurn.from_litellm_openai_response_api(value)

        if hasattr(raw_response, "text"):
            return AssistantTurn.from_google_genai(value)

        return {
            "role": "assistant",
            "content": ContentBlockList(),
            "reasoning": None,
            "finish_reason": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "model": None,
            "timestamp": None,
            "metadata": {},
        }

    def add_text(self, text: str) -> "AssistantTurn":
        self.content.append(text)
        return self

    def add_image(self, url: Optional[str] = None, data: Optional[str] = None,
                  media_type: str = "image/jpeg") -> "AssistantTurn":
        self.content.append(ImageContent(image_url=url, image_data=data, media_type=media_type))
        return self

    def to_text(self) -> str:
        """All text content concatenated; images shown as placeholders."""
        return self.content.to_text()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": [c.to_dict() for c in self.content],
            "reasoning": self.reasoning,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "model": self.model,
            "metadata": self.metadata,
        }

    def get_text(self) -> ContentBlockList:
        """ContentBlockList of only the TextContent blocks."""
        blocks = ContentBlockList()
        for block in self.content:
            if isinstance(block, TextContent):
                blocks.append(block)
        return blocks

    def get_images(self) -> ContentBlockList:
        """ContentBlockList of only the ImageContent blocks."""
        blocks = ContentBlockList()
        for block in self.content:
            if isinstance(block, ImageContent):
                blocks.append(block)
        return blocks

    def to_litellm_format(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content.to_litellm_format(role=self.role)}

    def __repr__(self) -> str:
        preview = str(self.content)
        preview = preview[:50] + "..." if len(preview) > 50 else preview
        parts = [f"AssistantTurn(content={preview!r}"]
        if getattr(self, "model", None):
            parts.append(f", model={self.model!r}")
        if getattr(self, "prompt_tokens", None):
            parts.append(f", prompt_tokens={self.prompt_tokens}")
        if getattr(self, "completion_tokens", None):
            parts.append(f", completion_tokens={self.completion_tokens}")
        parts.append(")")
        return "".join(parts)


def to_messages(
    system_prompt: Optional[str],
    user_content: Union[str, ContentBlockList, TextContent, ImageContent, None] = None,
    history: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Build a LiteLLM/OpenAI-format messages list.

    This is the minimal, stateless replacement for the old ``Chat`` manager.
    Optimizers own their own ``history`` (a plain list of message dicts) and call
    this to assemble the request:

        messages = to_messages(system_prompt, user_blocks, history=self.history)

    Args:
        system_prompt: Optional system message text.
        user_content: The current user turn content (str or content blocks).
            If None, no user message is appended.
        history: Prior message dicts (already in LiteLLM format) to insert
            between the system message and the new user message.

    Returns:
        A list of message dicts suitable for ``LLM(messages=...)``.
    """
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    if user_content is not None:
        blocks = ContentBlockList.ensure(user_content)
        messages.append({"role": "user", "content": blocks.to_litellm_format(role="user")})
    return messages

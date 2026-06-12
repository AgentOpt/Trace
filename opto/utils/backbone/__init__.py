"""Minimal multimodal conversation primitives for Trace optimizers.

Provides text/image content blocks, lightweight user/assistant turns, a prompt
template, and a stateless :func:`to_messages` helper for building provider-ready
messages lists. There is no conversation manager: optimizers own their own
message history as a plain list of dicts.
"""
from .content import (
    DEFAULT_IMAGE_PLACEHOLDER,
    ContentBase,
    ContentBlockList,
    Content,
    TextContent,
    ImageContent,
    ContentBlock,
)
from .template import PromptTemplate
from .turns import Turn, UserTurn, AssistantTurn, to_messages

__all__ = [
    "DEFAULT_IMAGE_PLACEHOLDER",
    "ContentBase",
    "ContentBlockList",
    "Content",
    "ContentBlock",
    "TextContent",
    "ImageContent",
    "PromptTemplate",
    "Turn",
    "UserTurn",
    "AssistantTurn",
    "to_messages",
]

"""Multimodal conversation primitives for Trace optimizers.

This package replaces the former single-file ``backbone.py``. The public API is
re-exported here so existing imports (``from opto.utils.backbone import X``)
keep working.
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
from .turns import Turn, UserTurn, AssistantTurn
from .chat import Chat

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
    "Chat",
]

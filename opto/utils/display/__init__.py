"""
Display utilities for rendering Opto objects in various formats.

This module provides rendering functions for different output formats:
- Jupyter notebooks (HTML)
- Terminal/CLI
- Markdown export

Usage:
    from opto.utils.display import render_for_jupyter
    html = render_for_jupyter(user_turn)
"""

from .jupyter import (
    render_user_turn,
    render_assistant_turn,
    render_chat,
    render_content_block_list,
)

__all__ = [
    'render_user_turn',
    'render_assistant_turn', 
    'render_chat',
    'render_content_block_list',
]


"""
Jupyter notebook HTML rendering for Opto objects.

This module handles all HTML generation for displaying Opto objects
in Jupyter notebooks with glassmorphism styling.
"""

import html as html_module
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opto.utils.backbone import UserTurn, AssistantTurn, Chat, ContentBlockList

from .themes import get_theme


def _escape(text: str) -> str:
    """HTML escape helper."""
    return html_module.escape(str(text))


def _escape_with_linebreaks(text: str) -> str:
    """HTML escape with proper newline and tab handling."""
    # First escape HTML
    escaped = html_module.escape(str(text))
    # Convert newlines to <br> tags
    escaped = escaped.replace('\n', '<br>')
    # Convert tabs to 4 spaces (visible)
    escaped = escaped.replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')
    return escaped


def _render_image_block(block) -> str:
    """Render an image content block."""
    parts = []
    parts.append('<div style="margin: 8px 0;">')
    
    # Get image source - prioritize URL, then base64 data
    img_src = None
    if hasattr(block, 'image_url') and block.image_url:
        img_src = block.image_url
    elif hasattr(block, 'image_data') and block.image_data:
        # Base64 encoded image
        media_type = getattr(block, 'media_type', 'image/jpeg')
        img_src = f"data:{media_type};base64,{block.image_data}"
    
    theme = get_theme()
    max_height = theme['common']['image_max_height']
    
    if img_src:
        parts.append(f'<img src="{_escape(img_src)}" '
                   f'style="max-width: 100%; max-height: {max_height}; border-radius: 8px; '
                   f'box-shadow: 0 2px 8px rgba(0,0,0,0.1);" />')
    else:
        # Fallback to placeholder if no image source
        media_type = getattr(block, 'media_type', 'image/jpeg')
        parts.append('<div style="display: inline-flex; align-items: center; background: #f0f0f0; '
                   'border: 1px solid #ddd; border-radius: 4px; padding: 8px 12px; '
                   'color: #666; font-size: 0.9em;"><span style="margin-right: 6px; font-size: 1.2em;">🖼️</span>')
        parts.append(f'<span>Image ({media_type})</span></div>')
    
    parts.append('</div>')
    return ''.join(parts)


def _render_content_blocks(blocks, block_id: str) -> tuple:
    """
    Render content blocks inline and generate expandable detail view.
    
    Returns:
        tuple: (inline_html, detail_html, num_blocks, num_images)
    """
    from opto.utils.backbone import TextContent, ImageContent
    
    inline_parts = []
    detail_parts = []
    num_images = 0
    
    for block in blocks:
        if isinstance(block, TextContent):
            inline_parts.append(_escape_with_linebreaks(block.text))
            block_text = str(block)
        elif isinstance(block, ImageContent):
            inline_parts.append(_render_image_block(block))
            num_images += 1
            block_text = str(block)  # Show full repr in detail view
        else:
            # Unknown block type
            block_text = str(block)
            inline_parts.append(_escape(block_text))
        
        # Add to detail view
        block_type = type(block).__name__
        detail_parts.append(f'''
            <div style="display: flex; align-items: flex-start; padding: 8px 10px; margin: 6px 0; 
                        background: white; border-radius: 6px; border: 1px solid rgba(0, 0, 0, 0.15); 
                        box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <span style="font-weight: 600; color: #666; min-width: 100px; font-size: 0.85em;">
                    {block_type}:
                </span>
                <span style="flex: 1; color: #333; font-size: 0.9em;">
                    {_escape(block_text)}
                </span>
            </div>
        ''')
    
    return ''.join(inline_parts), ''.join(detail_parts), len(blocks), num_images


def render_user_turn(turn: 'UserTurn') -> str:
    """Render a UserTurn as Jupyter HTML."""
    theme = get_theme()
    user_theme = theme['user']
    common = theme['common']
    
    # Generate unique ID
    turn_id = str(uuid.uuid4())[:8]
    
    parts = []
    
    # Main turn container
    parts.append(f'''
    <div style="padding: 16px; margin: 12px 0; border-radius: {common['border_radius']}; 
                backdrop-filter: {common['backdrop_filter']}; 
                background: {user_theme['background']}; border: 1px solid {user_theme['border']}; 
                box-shadow: {common['box_shadow']};">
    ''')
    
    # Role header
    parts.append(f'''
        <div style="display: flex; align-items: center; font-weight: 600; color: {user_theme['text_color']}; margin-bottom: 10px;">
            <span style="font-size: 1.2em; margin-right: 8px;">{user_theme['icon']}</span>
            <span>User</span>
        </div>
    ''')
    
    # Content
    parts.append('<div style="color: #333; line-height: 1.6; margin-top: 8px;">')
    
    # Render content blocks
    inline_html, detail_html, num_blocks, num_images = _render_content_blocks(turn.content, turn_id)
    parts.append(inline_html)
    parts.append('</div>')
    
    # Metadata badges
    parts.append('<div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px;">')
    
    # Content blocks badge
    block_label = f"{num_blocks} content block{'s' if num_blocks != 1 else ''}"
    if num_images > 0:
        block_label += f" ({num_images} image{'s' if num_images != 1 else ''})"
    
    parts.append(f'''
        <span onclick="document.getElementById('blocks-{turn_id}').classList.toggle('expanded')" 
              style="display: inline-block; background: rgba(0, 0, 0, 0.05); border-radius: 12px; 
                     padding: 4px 10px; font-size: 0.85em; color: #666; cursor: pointer;">
            {block_label} ▼
        </span>
    ''')
    
    # Tools badge
    tools = getattr(turn, 'tools', [])
    if tools:
        parts.append(f'''
            <span style="display: inline-block; background: #E1F5FE; border-radius: 12px; 
                         padding: 4px 10px; font-size: 0.85em; color: #01579B;">
                🔧 {len(tools)} tool{'s' if len(tools) != 1 else ''} available
            </span>
        ''')
    
    # Settings badges
    temperature = getattr(turn, 'temperature', None)
    if temperature is not None:
        parts.append(f'''
            <span style="display: inline-block; background: rgba(0, 0, 0, 0.05); border-radius: 12px; 
                         padding: 4px 10px; font-size: 0.85em; color: #666;">
                temperature: {temperature}
            </span>
        ''')
    
    parts.append('</div>')
    
    # Content blocks detail (expandable)
    parts.append(f'''
        <div id="blocks-{turn_id}" style="display: none; margin-top: 10px; padding: 10px; 
                                          background: rgba(255, 255, 255, 0.8); border-radius: 8px; 
                                          border: 1px solid rgba(0, 0, 0, 0.15);">
    ''')
    parts.append(f'''
        <style>
            #blocks-{turn_id}.expanded {{
                display: block !important;
            }}
        </style>
    ''')
    parts.append(detail_html)
    parts.append('</div>')
    
    # Close main container
    parts.append('</div>')
    
    return ''.join(parts)


def render_assistant_turn(turn: 'AssistantTurn') -> str:
    """Render an AssistantTurn as Jupyter HTML."""
    theme = get_theme()
    assistant_theme = theme['assistant']
    common = theme['common']
    
    # Generate unique ID
    turn_id = str(uuid.uuid4())[:8]
    
    parts = []
    
    # Main turn container
    parts.append(f'''
    <div style="padding: 16px; margin: 12px 0; border-radius: {common['border_radius']}; 
                backdrop-filter: {common['backdrop_filter']}; 
                background: {assistant_theme['background']}; border: 1px solid {assistant_theme['border']}; 
                box-shadow: {common['box_shadow']};">
    ''')
    
    # Role header with token badge
    parts.append(f'''
        <div style="display: flex; align-items: center; justify-content: space-between; 
                    font-weight: 600; color: {assistant_theme['text_color']}; margin-bottom: 10px;">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 1.2em; margin-right: 8px;">{assistant_theme['icon']}</span>
                <span>Assistant</span>
            </div>
    ''')
    
    # Token count badge
    prompt_tokens = getattr(turn, 'prompt_tokens', None)
    completion_tokens = getattr(turn, 'completion_tokens', None)
    if prompt_tokens or completion_tokens:
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        parts.append(f'''
            <span style="display: inline-block; background: #FFF3E0; color: #E65100; 
                         border-radius: 12px; padding: 4px 10px; font-size: 0.85em;">
                💰 {total_tokens} tokens
            </span>
        ''')
    
    parts.append('</div>')
    
    # Reasoning section (if present)
    reasoning = getattr(turn, 'reasoning', None)
    if reasoning:
        reasoning_theme = theme['reasoning']
        parts.append(f'''
            <div style="margin-top: 10px; padding: 10px; background: {reasoning_theme['background']}; 
                        border: 1px solid {reasoning_theme['border']}; border-radius: 4px; 
                        font-style: italic; color: {reasoning_theme['text_color']};">
                <div style="font-weight: 600; color: {reasoning_theme['text_color']}; margin-bottom: 6px; font-style: normal;">
                    {reasoning_theme['icon']} Reasoning:
                </div>
                {_escape(reasoning)}
            </div>
        ''')
    
    # Content
    parts.append('<div style="color: #333; line-height: 1.6; margin-top: 8px;">')
    
    # Render content blocks
    inline_html, _, _, _ = _render_content_blocks(turn.content, turn_id)
    parts.append(inline_html)
    parts.append('</div>')
    
    # Tool calls section (if present)
    if hasattr(turn, 'tool_calls') and turn.tool_calls:
        import json
        tool_theme = theme['tool_calls']
        
        parts.append(f'''
            <div style="margin-top: 10px; padding: 10px; background: {tool_theme['background']}; 
                        border-radius: 4px; border: 1px solid {tool_theme['border']}; font-size: 0.9em;">
                <div style="font-weight: 600; margin-bottom: 6px; color: {tool_theme['text_color']};">
                    {tool_theme['icon']} Tool Calls:
                </div>
        ''')
        
        for tc in turn.tool_calls:
            func_name = tc.function.name if tc.function else "unknown"
            args_str = tc.function.arguments if tc.function else "{}"
            parts.append(f'''
                <div style="font-family: 'Monaco', 'Menlo', monospace; font-size: 0.9em; margin: 4px 0;">
                    <span style="color: #6A1B9A; font-weight: 600;">{_escape(func_name)}</span>
                    <span style="color: #666;">({_escape(args_str)})</span>
                </div>
            ''')
            
            # Show results if available
            tool_results = getattr(turn, 'tool_results', [])
            matching_results = [tr for tr in tool_results if tr.tool_call_id == tc.id]
            if matching_results:
                for tr in matching_results:
                    result_content = str(tr.content)[:200]
                    if len(str(tr.content)) > 200:
                        result_content += '...'
                    parts.append(f'''
                        <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #ddd; 
                                    font-size: 0.9em; color: #666;">
                            <strong>Result:</strong> {_escape(result_content)}
                        </div>
                    ''')
        
        parts.append('</div>')
    
    # Metadata badges
    parts.append('<div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px;">')
    
    model = getattr(turn, 'model', None)
    if model:
        parts.append(f'''
            <span style="display: inline-block; background: rgba(0, 0, 0, 0.05); border-radius: 12px; 
                         padding: 4px 10px; font-size: 0.85em; color: #666;">
                model: {_escape(model)}
            </span>
        ''')
    
    finish_reason = getattr(turn, 'finish_reason', None)
    if finish_reason:
        parts.append(f'''
            <span style="display: inline-block; background: rgba(0, 0, 0, 0.05); border-radius: 12px; 
                         padding: 4px 10px; font-size: 0.85em; color: #666;">
                finish: {_escape(finish_reason)}
            </span>
        ''')
    
    parts.append('</div>')
    
    # Close main container
    parts.append('</div>')
    
    return ''.join(parts)


def render_content_block_list(blocks: 'ContentBlockList') -> str:
    """Render a ContentBlockList as Jupyter HTML."""
    theme = get_theme()
    content_theme = theme['content_blocks']
    common = theme['common']
    
    # Generate unique ID
    list_id = str(uuid.uuid4())[:8]
    
    parts = []
    
    # Main container
    parts.append(f'''
    <div style="padding: 16px; margin: 12px 0; border-radius: {common['border_radius']}; 
                backdrop-filter: {common['backdrop_filter']}; 
                background: {content_theme['background']}; border: 1px solid {content_theme['border']}; 
                box-shadow: {common['box_shadow']};">
    ''')
    
    # Header
    parts.append(f'''
        <div style="display: flex; align-items: center; font-weight: 600; color: {content_theme['text_color']}; margin-bottom: 10px;">
            <span style="font-size: 1.2em; margin-right: 8px;">{content_theme['icon']}</span>
            <span>Content Blocks</span>
        </div>
    ''')
    
    # Content
    parts.append('<div style="color: #333; line-height: 1.6; margin-top: 8px;">')
    
    # Render content blocks
    inline_html, detail_html, num_blocks, num_images = _render_content_blocks(blocks, list_id)
    parts.append(inline_html)
    parts.append('</div>')
    
    # Metadata badges
    parts.append('<div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px;">')
    
    block_label = f"{num_blocks} block{'s' if num_blocks != 1 else ''}"
    if num_images > 0:
        block_label += f" ({num_images} image{'s' if num_images != 1 else ''})"
    
    parts.append(f'''
        <span onclick="document.getElementById('blocks-{list_id}').classList.toggle('expanded')" 
              style="display: inline-block; background: rgba(0, 0, 0, 0.05); border-radius: 12px; 
                     padding: 4px 10px; font-size: 0.85em; color: #666; cursor: pointer;">
            {block_label} ▼
        </span>
    ''')
    
    parts.append('</div>')
    
    # Content blocks detail (expandable)
    parts.append(f'''
        <div id="blocks-{list_id}" style="display: none; margin-top: 10px; padding: 10px; 
                                          background: rgba(255, 255, 255, 0.8); border-radius: 8px; 
                                          border: 1px solid rgba(0, 0, 0, 0.15);">
    ''')
    parts.append(f'''
        <style>
            #blocks-{list_id}.expanded {{
                display: block !important;
            }}
        </style>
    ''')
    parts.append(detail_html)
    parts.append('</div>')
    
    # Close main container
    parts.append('</div>')
    
    return ''.join(parts)


def render_chat(chat: 'Chat') -> str:
    """Render a Chat as Jupyter HTML."""
    theme = get_theme()
    chat_theme = theme['chat']
    system_theme = theme['system_prompt']
    common = theme['common']
    
    # Generate unique ID
    chat_id = str(uuid.uuid4())[:8]
    
    parts = []
    
    # Add inline styles for collapsed turns functionality
    parts.append(f'''
    <style>
        .hidden-turns-{chat_id} {{
            display: none;
        }}
        .hidden-turns-{chat_id}.expanded {{
            display: block;
        }}
        .collapsed-turns-{chat_id} {{
            text-align: center;
            color: #999;
            margin: 10px 0;
            padding: 8px;
            cursor: pointer;
            border-radius: 4px;
            transition: background-color 0.2s;
        }}
        .collapsed-turns-{chat_id}:hover {{
            background-color: rgba(0, 0, 0, 0.05);
            color: #666;
        }}
    </style>
    ''')
    
    # Main chat container
    parts.append(f'''
    <div style="padding: 18px; border-radius: {common['border_radius']}; 
                backdrop-filter: {common['backdrop_filter']}; 
                background: {chat_theme['background']}; border: 1px solid {chat_theme['border']}; 
                box-shadow: {common['box_shadow']};">
    ''')
    
    # Chat header
    parts.append(f'''
        <div style="display: flex; justify-content: space-between; align-items: center; 
                    padding-bottom: 12px; border-bottom: 2px solid #E0E0E0; margin-bottom: 12px;">
            <div style="font-weight: 700; font-size: 1.1em; color: {chat_theme['title_color']};">💬 Chat History</div>
            <div style="display: flex; gap: 12px; font-size: 0.9em;">
    ''')
    
    # Stats badges
    num_turns = len(chat.turns)
    parts.append(f'''
                <span style="display: inline-block; background: rgba(0, 0, 0, 0.05); 
                             border-radius: 12px; padding: 4px 10px; color: #666;">
                    {num_turns} turn{'s' if num_turns != 1 else ''}
                </span>
    ''')
    
    total_tokens = chat.get_token_count_estimate()
    if total_tokens > 0:
        parts.append(f'''
                <span style="display: inline-block; background: #FFF3E0; color: #E65100; 
                             border-radius: 12px; padding: 4px 10px;">
                    Total: ~{total_tokens} tokens
                </span>
        ''')
    
    parts.append('</div></div>')
    
    # System prompt (if present)
    if chat.system_prompt:
        parts.append(f'''
            <div style="padding: 14px 16px; margin-bottom: 16px; border-radius: {common['border_radius']}; 
                        backdrop-filter: {common['backdrop_filter']}; background: {system_theme['background']}; 
                        border: 1px solid {system_theme['border']}; box-shadow: {common['box_shadow']}; 
                        font-size: 0.95em; color: #555;">
                <div style="font-weight: 600; color: {system_theme['text_color']}; margin-bottom: 6px;">
                    {system_theme['icon']} System Prompt:
                </div>
                {_escape(chat.system_prompt)}
            </div>
        ''')
    
    # Render turns with middle collapsing for long conversations
    COLLAPSE_THRESHOLD = 20
    SHOW_FIRST = 2
    SHOW_LAST = 2
    
    if len(chat.turns) > COLLAPSE_THRESHOLD:
        # Show first few turns
        for turn in chat.turns[:SHOW_FIRST]:
            if hasattr(turn, '_repr_html_'):
                parts.append(turn._repr_html_())
        
        # Collapsed middle section
        num_hidden = len(chat.turns) - SHOW_FIRST - SHOW_LAST
        parts.append(f'''
            <div class="collapsed-turns-{chat_id}" 
                 onclick="document.getElementById('hidden-{chat_id}').classList.add('expanded'); this.style.display='none';">
                ... {num_hidden} more turn{'s' if num_hidden != 1 else ''} ... (click to expand)
            </div>
        ''')
        
        # Hidden middle turns
        parts.append(f'<div id="hidden-{chat_id}" class="hidden-turns-{chat_id}">')
        for turn in chat.turns[SHOW_FIRST:-SHOW_LAST]:
            if hasattr(turn, '_repr_html_'):
                parts.append(turn._repr_html_())
        parts.append('</div>')
        
        # Show last few turns
        for turn in chat.turns[-SHOW_LAST:]:
            if hasattr(turn, '_repr_html_'):
                parts.append(turn._repr_html_())
    else:
        # Show all turns
        for turn in chat.turns:
            if hasattr(turn, '_repr_html_'):
                parts.append(turn._repr_html_())
    
    # Close main container
    parts.append('</div>')
    
    return ''.join(parts)


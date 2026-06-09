"""
Theme configurations for display rendering.

Themes define colors, styles, and visual elements for different rendering styles.
"""

# Glassmorphism theme (default) - inspired by Trace2 branding
GLASSMORPHISM_THEME = {
    'user': {
        'background': 'rgba(236, 254, 255, 0.85)',
        'border': 'rgba(6, 182, 212, 0.3)',
        'border_color': '#06b6d4',
        'text_color': '#0891b2',
        'icon': '👤',
    },
    'assistant': {
        'background': 'rgba(238, 242, 255, 0.85)',
        'border': 'rgba(99, 102, 241, 0.3)',
        'border_color': '#6366f1',
        'text_color': '#4f46e5',
        'icon': '🤖',
    },
    'content_blocks': {
        'background': 'rgba(255, 255, 255, 0.85)',
        'border': 'rgba(158, 158, 158, 0.3)',
        'text_color': '#666',
        'icon': '📝',
    },
    'system_prompt': {
        'background': 'rgba(250, 250, 250, 0.7)',
        'border': 'rgba(158, 158, 158, 0.2)',
        'text_color': '#757575',
        'icon': '⚙️',
    },
    'chat': {
        'background': 'rgba(255, 255, 255, 0.7)',
        'border': 'rgba(158, 158, 158, 0.2)',
        'title_color': '#424242',
    },
    'reasoning': {
        'background': '#F5F5F5',
        'border': '#E0E0E0',
        'text_color': '#555',
        'icon': '💭',
    },
    'tool_calls': {
        'background': '#F5F5F5',
        'border': '#E0E0E0',
        'text_color': '#555',
        'icon': '🔧',
    },
    'common': {
        'border_radius': '12px',
        'box_shadow': '0 4px 16px rgba(0,0,0,0.06)',
        'backdrop_filter': 'blur(10px)',
        'image_max_height': '400px',
    }
}

# Active theme (can be changed)
ACTIVE_THEME = GLASSMORPHISM_THEME


def set_theme(theme_dict):
    """Set a custom theme for display rendering."""
    global ACTIVE_THEME
    ACTIVE_THEME = theme_dict


def get_theme():
    """Get the currently active theme."""
    return ACTIVE_THEME


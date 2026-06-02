# Opto Display Module

Rendering utilities for visualizing Opto objects in various formats.

## Architecture

The display module separates **visualization logic** from **data handling logic**:

```
opto/utils/
├── backbone.py          # Core data classes (UserTurn, AssistantTurn, Chat)
└── display/
    ├── __init__.py      # Public API
    ├── jupyter.py       # Jupyter notebook HTML rendering
    ├── themes.py        # Color schemes and styling
    └── README.md        # This file
```

## Usage

### Automatic Display in Jupyter

Objects automatically render with glassmorphism styling:

```python
from opto.utils.backbone import UserTurn, AssistantTurn, Chat

user_turn = UserTurn("Hello!")
user_turn  # Beautiful display! ✨
```

### Direct Rendering

Use the display module directly for more control:

```python
from opto.utils.display import render_user_turn, render_chat
from IPython.display import HTML

user_turn = UserTurn("Hello!")
html_output = render_user_turn(user_turn)
HTML(html_output)
```

## Features

- Glassmorphism Design
- Multi-modal content (images, files interleaved with text), all rendered as HTML.

### Custom Themes

You can customize the color scheme:

```python
from opto.utils.display.themes import set_theme

custom_theme = {
    'user': {
        'background': 'rgba(255, 240, 245, 0.85)',
        'border': 'rgba(233, 30, 99, 0.3)',
        'text_color': '#C2185B',
        'icon': '👤',
    },
    'assistant': {
        'background': 'rgba(232, 245, 233, 0.85)',
        'border': 'rgba(76, 175, 80, 0.3)',
        'text_color': '#388E3C',
        'icon': '🤖',
    },
    # ... other theme properties
}

set_theme(custom_theme)
```

### Custom Renderers

You can create your own rendering functions:

```python
def my_custom_renderer(user_turn):
    """Custom HTML renderer for UserTurn"""
    return f"<div style='color: red;'>{user_turn.content.to_text()}</div>"

# Use it directly
from IPython.display import HTML
HTML(my_custom_renderer(user_turn))
```

### Fallback Behavior

If the display module is unavailable (e.g., import error), classes fall back to their `__repr__()` text representation:

```python
class UserTurn:
    def _repr_html_(self):
        try:
            from opto.utils.display.jupyter import render_user_turn
            return render_user_turn(self)
        except ImportError:
            return None  # Falls back to __repr__
```

## Future Extension

Potential additions to the display module:

- **Terminal renderer**: ANSI color codes for CLI display
- **Markdown renderer**: Export conversations as markdown
- **LaTeX renderer**: For academic papers
- **HTML export**: Static HTML page generation

To add a new rendering format:

1. Create `opto/utils/display/my_format.py`
2. Implement `render_*` functions for each class
3. Export in `__init__.py`
4. Update documentation

Example:

```python
# opto/utils/display/terminal.py
def render_user_turn(turn):
    """Render UserTurn with ANSI colors for terminal"""
    from colorama import Fore, Style
    return f"{Fore.CYAN}👤 User:{Style.RESET_ALL} {turn.content.to_text()}"
```


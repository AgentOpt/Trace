"""PromptTemplate: ``str.format``-like templating that also supports
multimodal :class:`ContentBlockList` values.
"""
from typing import Union

from .content import ContentBlockList


class PromptTemplate:
    """Template for building ContentBlockLists with {placeholder} support.
    
    Similar to str.format(), but supports multimodal content (ContentBlockList).
    
    Return type depends on values:
    - All strings → returns str (backward compatible)
    - Any multimodal content → returns ContentBlockList
    
    Features:
    - Multiple placeholders: {a}, {b}, {c}
    - Escaping: {{ and }} for literal braces
    - Missing placeholders: left as-is in text
    - Extra kwargs: silently ignored (no error)
    - Nested templates: if value is PromptTemplate, formats it first
    - Mixed values: str, ContentBlockList, or objects with to_content_blocks()
    
    Examples:
        # Define template (can be class attribute)
        user_prompt_template = PromptTemplate('''
        Now you see problem instance:

        ================================
        {problem_instance}
        ================================
        ''')

        # Format with ContentBlockList (may contain images)
        content = user_prompt_template.format(
            problem_instance=problem.to_content_blocks()
        )
        # Returns ContentBlockList: [TextContent("Now you see..."), *problem_blocks, TextContent("===...")]

        # Multiple placeholders
        template = PromptTemplate("User: {user}\\nAssistant: {assistant}")
        result = template.format(user=user_blocks, assistant=assistant_blocks)

        # Nested templates
        outer = PromptTemplate("Header\\n{body}\\nFooter")
        inner = PromptTemplate("Content: {data}")
        result = outer.format(body=inner, data="some data")  # inner gets same kwargs

        # Escaping braces
        template = PromptTemplate('JSON example: {{"key": "{value}"}}')
        result = template.format(value="hello")  # {"key": "hello"}
        
        # Extra kwargs are ignored (no error)
        result = template.format(value="hello", unused_key="ignored")
        
        # Missing placeholders left as-is
        template = PromptTemplate("Hello {name}, score: {score}")
        result = template.format(name="Alice")  # "Hello Alice, score: {score}"
    """
    
    # Regex to find {placeholder} but not {{ or }}
    _PLACEHOLDER_PATTERN = None  # Lazy compiled
    
    def __init__(self, template: str):
        """Initialize with a template string.
        
        Args:
            template: Template string with {placeholder} syntax.
        """
        self.template = template
    
    @classmethod
    def _get_pattern(cls):
        """Lazily compile the placeholder regex pattern."""
        if cls._PLACEHOLDER_PATTERN is None:
            import re
            # Match {name} but not {{ or }}
            # Captures the placeholder name
            cls._PLACEHOLDER_PATTERN = re.compile(r'\{(\w+)\}')
        return cls._PLACEHOLDER_PATTERN
    
    def format(self, **kwargs) -> Union[str, 'ContentBlockList']:
        """Format the template with the given values.
        
        Similar to str.format(), but supports multimodal content.
        Extra kwargs are silently ignored.
        
        If all values are strings, returns a str (backward compatible).
        If any value is a ContentBlockList or multimodal, returns ContentBlockList.
        
        Args:
            **kwargs: Placeholder values. Each value can be:
                - str: inserted as text
                - ContentBlockList: blocks spliced in at that position
                - PromptTemplate: formatted first, then spliced in
                - Object with to_content_blocks(): method called, result spliced
                - Other: converted to str
        
        Returns:
            str: If all values are strings (backward compatible behavior).
            ContentBlockList: If any value is multimodal content.
        """
        # Check if all values are simple strings - if so, use simple string formatting
        pattern = self._get_pattern()
        placeholder_names = set(pattern.findall(self.template))
        
        # Only check values for placeholders that exist in the template
        relevant_values = {k: v for k, v in kwargs.items() if k in placeholder_names}
        
        if all(isinstance(v, str) for v in relevant_values.values()):
            # All strings: use simple string replacement, return str
            # Handle escaping and missing placeholders
            result = self.template.replace("{{", "\x00LBRACE\x00").replace("}}", "\x00RBRACE\x00")
            
            for name in placeholder_names:
                placeholder = "{" + name + "}"
                if name in kwargs:
                    result = result.replace(placeholder, kwargs[name])
                # Missing placeholders left as-is
            
            result = result.replace("\x00LBRACE\x00", "{").replace("\x00RBRACE\x00", "}")
            return result
        
        # Multimodal content: build ContentBlockList
        result = ContentBlockList()
        
        # Handle escaping: replace {{ with a sentinel, }} with another
        LBRACE_SENTINEL = "\x00LBRACE\x00"
        RBRACE_SENTINEL = "\x00RBRACE\x00"
        
        text = self.template.replace("{{", LBRACE_SENTINEL).replace("}}", RBRACE_SENTINEL)
        
        last_end = 0
        
        for match in pattern.finditer(text):
            # Add text before this placeholder
            prefix = text[last_end:match.start()]
            if prefix:
                # Restore escaped braces in prefix
                prefix = prefix.replace(LBRACE_SENTINEL, "{").replace(RBRACE_SENTINEL, "}")
                result.append(prefix)
            
            # Get placeholder name and value
            placeholder_name = match.group(1)
            
            if placeholder_name in kwargs:
                value = kwargs[placeholder_name]
                # Convert value to ContentBlockList and splice in
                content = self._value_to_content(value, **kwargs)
                result.extend(content)
            else:
                # Missing placeholder: leave as-is (restore original {name})
                result.append("{" + placeholder_name + "}")
            
            last_end = match.end()
        
        # Add remaining text after last placeholder
        suffix = text[last_end:]
        if suffix:
            suffix = suffix.replace(LBRACE_SENTINEL, "{").replace(RBRACE_SENTINEL, "}")
            result.append(suffix)
        
        return result
    
    def _value_to_content(self, value, **kwargs) -> 'ContentBlockList':
        """Convert a value to ContentBlockList.
        
        Args:
            value: The value to convert
            **kwargs: Passed to nested PromptTemplate.render()
        
        Returns:
            ContentBlockList: The value as content blocks.
        """
        if isinstance(value, ContentBlockList):
            return value
        elif isinstance(value, PromptTemplate):
            # Nested template: format it with the same kwargs
            return value.format(**kwargs)
        elif hasattr(value, 'to_content_blocks'):
            # Object with to_content_blocks method (e.g., ProblemInstance)
            return value.to_content_blocks()
        elif isinstance(value, str):
            return ContentBlockList(value)
        else:
            # Fallback: convert to string
            return ContentBlockList(str(value))
    
    def __repr__(self) -> str:
        """Return a preview of the template."""
        preview = self.template[:50] + "..." if len(self.template) > 50 else self.template
        return f"PromptTemplate({preview!r})"

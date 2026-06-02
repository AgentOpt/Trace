"""Conversation turns: :class:`UserTurn` and :class:`AssistantTurn`.

``AssistantTurn.autocast`` parses raw responses from LiteLLM/OpenAI (Responses
and Completion APIs), Bedrock Converse, and Google GenAI into a uniform shape.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .content import ContentBlockList, TextContent, ImageContent


@dataclass
class UserTurn:
    """Represents a user message turn in the conversation"""
    role: str = "user"

    content: ContentBlockList = field(default_factory=ContentBlockList)

    # Provider-specific settings
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None

    # Metadata
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, content=None, tools=None, **kwargs):
        """
        Initialize UserTurn with content and tools.
        
        Four ways to initialize:
        1. Empty: UserTurn() - creates empty turn with defaults
        2. Copy: UserTurn(existing_turn) - creates a copy of an existing UserTurn
        3. Positional args: UserTurn(content, tools) - pass content and/or tools
        4. Keyword args: UserTurn(content=..., tools=..., temperature=...) - explicit fields
        
        Args:
            content: ContentBlockList, list of content blocks, UserTurn (for copying), or None
            tools: List of ToolDefinition or None
            **kwargs: Additional fields (temperature, max_tokens, top_p, timestamp, metadata)
        """
        self.output_contains_image = False

        # Handle copy constructor: UserTurn(existing_turn)
        if isinstance(content, UserTurn):
            source = content
            self.role = source.role
            self.content = ContentBlockList(source.content)  # Deep copy the content list
            self.temperature = source.temperature
            self.max_tokens = source.max_tokens
            self.top_p = source.top_p
            self.timestamp = source.timestamp
            self.metadata = dict(source.metadata)  # Copy the metadata dict
            return
        
        # Handle content
        if content is None:
            content = ContentBlockList()
        elif not isinstance(content, ContentBlockList):
            # If it's a list, wrap it in ContentBlockList
            content = ContentBlockList(content) if isinstance(content, list) else ContentBlockList([content])
        
        
        # Set all fields
        self.role = kwargs.get('role', "user")
        self.content = content
        self.temperature = kwargs.get('temperature', None)
        self.max_tokens = kwargs.get('max_tokens', None)
        self.top_p = kwargs.get('top_p', None)
        self.timestamp = kwargs.get('timestamp', None)
        self.metadata = kwargs.get('metadata', {})

    def add_text(self, text: str) -> 'UserTurn':
        """Add text content"""
        self.content.append(TextContent(text=text))
        return self

    def add_image(self, url: Optional[str] = None, data: Optional[str] = None,
                  media_type: str = "image/jpeg") -> 'UserTurn':
        """Add image content"""
        self.content.append(ImageContent(
            image_url=url,
            image_data=data,
            media_type=media_type
        ))
        return self

    def add_image_file(self, filepath: str) -> 'UserTurn':
        """Add image from file"""
        self.content.append(ImageContent.from_file(filepath))
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "role": "user",
            "content": [c.to_dict() for c in self.content],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "metadata": self.metadata
        }

    def enable_image_generation(self):
        self.output_contains_image = True
    
    def __repr__(self) -> str:
        """Safe string representation that handles missing attributes."""
        content_preview = str(self.content)[:50] + "..." if len(str(self.content)) > 50 else str(self.content)
        parts = [f"UserTurn(content={content_preview!r}"]
        
        # Safely add optional fields if they exist
        temperature = getattr(self, 'temperature', None)
        if temperature is not None:
            parts.append(f", temperature={temperature}")
        
        parts.append(")")
        return "".join(parts)

    def to_litellm_format(self) -> Dict[str, Any]:
        """Convert to LiteLLM Response API format (OpenAI Response API compatible)"""
        return {
            "role": "user",
            "content": self.content.to_litellm_format(role="user")
        }
    
    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks with glassmorphism design."""
        try:
            from opto.utils.display.jupyter import render_user_turn
            return render_user_turn(self)
        except ImportError:
            # Fallback to text representation if display module unavailable
            return None


@dataclass
class Turn:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class AssistantTurn(Turn):
    """Represents an assistant message turn in the conversation"""
    role: str = "assistant"
    content: ContentBlockList = field(default_factory=ContentBlockList)

    # Provider-specific features
    reasoning: Optional[str] = None  # OpenAI reasoning/thinking
    finish_reason: Optional[str] = None  # "stop", "length", "tool_calls", etc.

    # Token usage
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    # Metadata
    model: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, *args, **kwargs):
        """
        Initialize AssistantTurn from a raw response or with explicit fields.
        
        Three ways to initialize:
        1. Empty: AssistantTurn() - creates empty turn with defaults
        2. From raw response: AssistantTurn(response) - autocasts the response
        3. With fields: AssistantTurn(role="assistant", content=[...]) - explicit fields
        """
        if len(args) == 1 and isinstance(args[0], AssistantTurn):
            # Case: Copy constructor - create a copy of another AssistantTurn
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
                metadata=dict(other.metadata)
            )
            return

        if len(args) > 0 and len(kwargs) == 0:
            # Case 2: Single positional arg - autocast from raw response
            value_dict = self.autocast(args[0])
            super().__init__(**value_dict)
        elif len(kwargs) > 0:
            # Case 3: Keyword arguments - use them directly
            super().__init__(**kwargs)
        else:
            # Case 1: No arguments - initialize with defaults
            super().__init__(
                role="assistant",
                content=ContentBlockList(),
                reasoning=None,
                finish_reason=None,
                prompt_tokens=None,
                completion_tokens=None,
                model=None,
                timestamp=None,
                metadata={}
            )

    @staticmethod
    def from_google_genai(value: Any) -> Dict[str, Any]:
        """Parse a Google GenAI response into a dictionary of AssistantTurn fields.
        
        Supports both the legacy generate_content API and the new Interactions API.
        
        Args:
            value: Raw response from Google GenAI API
            
        Returns:
            Dict[str, Any]: Dictionary with keys corresponding to AssistantTurn fields
        """
        # Initialize the result dictionary with default values
        result = {
            "role": "assistant",
            "content": ContentBlockList(),
            "reasoning": None,
            "finish_reason": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "model": None,
            "timestamp": None,
            "metadata": {}
        }
        
        # Check if this is a normalized response (from our GoogleGenAILLM)
        if hasattr(value, 'raw_response'):
            raw_response = value.raw_response
        else:
            raw_response = value
        
        # Handle Interactions API format (new)
        if hasattr(raw_response, 'outputs'):
            # This is an Interaction object
            interaction = raw_response
            
            # Extract text from outputs
            if interaction.outputs and len(interaction.outputs) > 0:
                for output in interaction.outputs:
                    if hasattr(output, 'text') and output.text:
                        result["content"].append(TextContent(text=output.text))
                    # Handle other output types if they exist
                    elif hasattr(output, 'content'):
                        # Content could be a list of parts
                        if isinstance(output.content, list):
                            for part in output.content:
                                if hasattr(part, 'text') and part.text:
                                    result["content"].append(TextContent(text=part.text))
                        else:
                            result["content"].append(TextContent(text=str(output.content)))
            
            # Extract model info
            if hasattr(interaction, 'model'):
                result["model"] = interaction.model
            
            # Extract status as finish_reason
            if hasattr(interaction, 'status'):
                result["finish_reason"] = interaction.status
            
            # Extract token usage from Interactions API
            if hasattr(interaction, 'usage'):
                usage = interaction.usage
                if hasattr(usage, 'input_tokens'):
                    result["prompt_tokens"] = usage.input_tokens
                elif hasattr(usage, 'prompt_token_count'):
                    result["prompt_tokens"] = usage.prompt_token_count
                    
                if hasattr(usage, 'output_tokens'):
                    result["completion_tokens"] = usage.output_tokens
                elif hasattr(usage, 'candidates_token_count'):
                    result["completion_tokens"] = usage.candidates_token_count
            
            # Extract interaction ID as metadata
            if hasattr(interaction, 'id'):
                result["metadata"]['interaction_id'] = interaction.id
        
        # Handle legacy generate_content API format
        else:
            # Extract thinking/reasoning (for Gemini 2.5+ models)
            if hasattr(raw_response, 'thoughts') and raw_response.thoughts:
                # Gemini's thinking budget feature
                result["reasoning"] = str(raw_response.thoughts)
            
            # Extract model info
            if hasattr(raw_response, 'model_version'):
                result["model"] = raw_response.model_version
            
            # Extract token usage (if available)
            if hasattr(raw_response, 'usage_metadata'):
                usage = raw_response.usage_metadata
                if hasattr(usage, 'prompt_token_count'):
                    result["prompt_tokens"] = usage.prompt_token_count
                if hasattr(usage, 'candidates_token_count'):
                    result["completion_tokens"] = usage.candidates_token_count
            
            # Handle multimodal content from Gemini (candidates with parts)
            content_extracted = False
            if hasattr(raw_response, 'candidates') and raw_response.candidates:
                candidate = raw_response.candidates[0]
                
                # Extract from parts (supports multimodal responses with text and images)
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        # Handle text parts
                        if hasattr(part, 'text') and part.text:
                            result["content"].append(TextContent(text=part.text))
                            content_extracted = True
                        # Handle inline data (images, generated images, etc.)
                        elif hasattr(part, 'inline_data'):
                            # Try to extract image data, preferring direct inline_data access
                            inline = part.inline_data
                            image_bytes = None
                            image_data = None
                            media_type = 'image/jpeg'

                            
                            # Extract from inline_data Blob (most reliable method)
                            # Google's Blob.data should be raw bytes
                            if hasattr(inline, 'data'):
                                data = inline.data
                                # Check if it's bytes or string
                                if isinstance(data, bytes):
                                    # Store raw bytes for Gemini compatibility
                                    # (Gemini prefers raw bytes when sending images)
                                    image_bytes = data
                                elif isinstance(data, str):
                                    # Already base64-encoded string
                                    image_data = data
                                    # Don't decode to bytes - keep as base64 for portability
                            
                            if hasattr(inline, 'mime_type'):
                                media_type = inline.mime_type
                            
                            # If we got the data, create ImageContent
                            # Store image_bytes only if we got raw bytes from Google
                            if image_data or image_bytes:
                                result["content"].append(ImageContent(
                                    image_data=image_data,
                                    image_bytes=image_bytes if isinstance(data, bytes) else None,
                                    media_type=media_type
                                ))
                                content_extracted = True
                
                # Extract finish reason
                if hasattr(candidate, 'finish_reason'):
                    result["finish_reason"] = str(candidate.finish_reason)
            
            # Fallback: Extract simple text content if no candidates/parts were found
            if not content_extracted:
                if hasattr(raw_response, 'text'):
                    result["content"].append(TextContent(text=raw_response.text))
                elif hasattr(value, 'choices'):
                    # Fallback to normalized format
                    result["content"].append(TextContent(text=value.choices[0].message.content))
        
        return result
    
    @staticmethod
    def from_litellm_openai_response_api(value: Any) -> Dict[str, Any]:
        """Parse a LiteLLM/OpenAI-style response into a dictionary of AssistantTurn fields.
        
        Handles both formats:
        - New Responses API: Has 'output' field with ResponseOutputMessage objects
        - Legacy Completion API: Has 'choices' field with message objects
        
        Args:
            value: Response from LiteLLM/OpenAI API (Responses API or Completion API)
            
        Returns:
            Dict[str, Any]: Dictionary with keys corresponding to AssistantTurn fields
        """
        # Initialize the result dictionary with default values
        result = {
            "role": "assistant",
            "content": ContentBlockList(),
            "reasoning": None,
            "finish_reason": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "model": None,
            "timestamp": None,
            "metadata": {}
        }
        
        # Handle Bedrock Converse API format (has 'output' field with 'message')
        # Check both attribute-based and dict-based access for robustness
        is_bedrock = False
        bedrock_output = None
        bedrock_value = value  # Keep reference to the original value for later access
        
        # Try attribute-based access first
        if hasattr(value, 'output'):
            output_val = value.output
            
            if hasattr(output_val, 'message'):
                is_bedrock = True
                bedrock_output = output_val
            # Also check dict-based access on the output attribute
            elif isinstance(output_val, dict) and 'message' in output_val:
                is_bedrock = True
                bedrock_output = output_val
        
        # If not found, try dict-based access on value itself
        if not is_bedrock and isinstance(value, dict) and 'output' in value:
            output_val = value['output']
            if isinstance(output_val, dict) and 'message' in output_val:
                is_bedrock = True
                bedrock_output = output_val
                bedrock_value = value  # Use the dict directly
        
        if is_bedrock and bedrock_output is not None:
            # Bedrock Converse API format detected
            # Get message with dict or attr access
            message = bedrock_output.get('message') if isinstance(bedrock_output, dict) else (bedrock_output.message if hasattr(bedrock_output, 'message') else None)
            
            if message:
                # Extract role
                if isinstance(message, dict):
                    result["role"] = message.get('role', 'assistant')
                elif hasattr(message, 'role'):
                    result["role"] = message.role
                
                # Extract content
                content_list = message.get('content') if isinstance(message, dict) else (message.content if hasattr(message, 'content') else None)
                
                if content_list:
                    for content_item in content_list:
                        # Handle text content (dict or attr)
                        text_val = None
                        if isinstance(content_item, dict):
                            text_val = content_item.get('text')
                        elif hasattr(content_item, 'text'):
                            text_val = content_item.text
                        
                        if text_val:
                            result["content"].append(TextContent(text=text_val))
            
            # Extract finish reason from stopReason (check both value and bedrock_value)
            stop_reason = None
            if isinstance(bedrock_value, dict):
                stop_reason = bedrock_value.get('stopReason')
            elif hasattr(bedrock_value, 'stopReason'):
                stop_reason = bedrock_value.stopReason
            if stop_reason:
                result["finish_reason"] = stop_reason
            
            # Extract token usage (check both value and bedrock_value)
            usage = None
            if isinstance(bedrock_value, dict):
                usage = bedrock_value.get('usage')
            elif hasattr(bedrock_value, 'usage'):
                usage = bedrock_value.usage
            
            if usage:
                if isinstance(usage, dict):
                    result["prompt_tokens"] = usage.get('inputTokens')
                    result["completion_tokens"] = usage.get('outputTokens')
                else:
                    if hasattr(usage, 'inputTokens'):
                        result["prompt_tokens"] = usage.inputTokens
                    if hasattr(usage, 'outputTokens'):
                        result["completion_tokens"] = usage.outputTokens
        
        # Handle Responses API format (new format with 'output' field)
        # The output field is a list of output items (messages, image generation calls, etc.)
        # NOTE: LiteLLM may set value.object to 'chat.completion' or 'response' depending on the provider
        elif hasattr(value, 'output') and hasattr(value, 'object'):
            # Extract metadata
            if hasattr(value, 'id'):
                result["metadata"]['response_id'] = value.id
            if hasattr(value, 'created_at'):
                result["timestamp"] = str(value.created_at)
            
            # Extract model info
            if hasattr(value, 'model'):
                result["model"] = value.model
            
            # Extract status as finish_reason
            if hasattr(value, 'status'):
                result["finish_reason"] = value.status
            
            # Extract content from output (list of output items)
            if value.output and len(value.output) > 0:
                for output_item in value.output:
                    # Handle ImageGenerationCall
                    if hasattr(output_item, 'type') and output_item.type == 'image_generation_call':
                        # Extract generated image
                        if hasattr(output_item, 'result') and output_item.result:
                            # Determine media type from output_format
                            media_type = 'image/jpeg'  # default
                            if hasattr(output_item, 'output_format'):
                                format_map = {
                                    'png': 'image/png',
                                    'jpeg': 'image/jpeg',
                                    'jpg': 'image/jpeg',
                                    'webp': 'image/webp',
                                    'gif': 'image/gif'
                                }
                                media_type = format_map.get(output_item.output_format.lower(), 'image/jpeg')
                            
                            # Add image to content
                            result["content"].append(ImageContent(
                                image_data=output_item.result,
                                media_type=media_type
                            ))
                            
                            # Store additional metadata about the image generation
                            if hasattr(output_item, 'revised_prompt') and output_item.revised_prompt:
                                if 'image_generation' not in result["metadata"]:
                                    result["metadata"]['image_generation'] = []
                                result["metadata"]['image_generation'].append({
                                    'id': output_item.id if hasattr(output_item, 'id') else None,
                                    'revised_prompt': output_item.revised_prompt,
                                    'size': output_item.size if hasattr(output_item, 'size') else None,
                                    'quality': output_item.quality if hasattr(output_item, 'quality') else None,
                                    'status': output_item.status if hasattr(output_item, 'status') else None
                                })
                    
                    # Handle ResponseOutputMessage
                    elif hasattr(output_item, 'type') and output_item.type == 'message':
                        # Extract role
                        if hasattr(output_item, 'role'):
                            result["role"] = output_item.role
                        
                        # Extract status for this message
                        if hasattr(output_item, 'status') and not result["finish_reason"]:
                            result["finish_reason"] = output_item.status
                        
                        # Extract content items
                        if hasattr(output_item, 'content') and output_item.content:
                            for content_item in output_item.content:
                                # Handle text content
                                if hasattr(content_item, 'type') and content_item.type == 'output_text':
                                    if hasattr(content_item, 'text') and content_item.text:
                                        result["content"].append(TextContent(text=content_item.text))
                                # Handle other content types as they become available
                                elif hasattr(content_item, 'text') and content_item.text:
                                    result["content"].append(TextContent(text=str(content_item.text)))
            
            # Extract reasoning (for models with reasoning capabilities)
            if hasattr(value, 'reasoning'):
                reasoning_parts = []
                if isinstance(value.reasoning, dict):
                    if value.reasoning.get('summary'):
                        reasoning_parts.append(f"Summary: {value.reasoning['summary']}")
                    if value.reasoning.get('effort'):
                        reasoning_parts.append(f"Effort: {value.reasoning['effort']}")
                    if reasoning_parts:
                        result["reasoning"] = "\n".join(reasoning_parts)
                elif value.reasoning:
                    result["reasoning"] = str(value.reasoning)
            
            # Extract token usage (Responses API format)
            if hasattr(value, 'usage'):
                if hasattr(value.usage, 'input_tokens'):
                    result["prompt_tokens"] = value.usage.input_tokens
                if hasattr(value.usage, 'output_tokens'):
                    result["completion_tokens"] = value.usage.output_tokens
        
        # Handle legacy Completion API format (has 'choices' field)
        elif hasattr(value, 'choices') and len(value.choices) > 0:
            choice = value.choices[0]
            message = choice.message if hasattr(choice, 'message') else choice
            
            # Extract text content
            if hasattr(message, 'content') and message.content:
                result["content"].append(TextContent(text=str(message.content)))
            
            
            # Extract finish reason
            if hasattr(choice, 'finish_reason'):
                result["finish_reason"] = choice.finish_reason
            
            # Extract reasoning/thinking (for OpenAI o1/o3 models)
            if hasattr(message, 'reasoning') and message.reasoning:
                result["reasoning"] = message.reasoning
            
            # Extract token usage (Completion API format)
            if hasattr(value, 'usage'):
                if hasattr(value.usage, 'prompt_tokens'):
                    result["prompt_tokens"] = value.usage.prompt_tokens
                if hasattr(value.usage, 'completion_tokens'):
                    result["completion_tokens"] = value.usage.completion_tokens
            
            # Extract model info
            if hasattr(value, 'model'):
                result["model"] = value.model
        
        return result
    
    @staticmethod
    def autocast(value: Any) -> Dict[str, Any]:
        """Automatically parse a response from any API into a dictionary of AssistantTurn fields.
        
        Automatically detects the response format and uses the appropriate parser:
        - Google GenAI (generate_content or Interactions API)
        - LiteLLM/OpenAI Responses API (new format with 'output' field)
        - LiteLLM/OpenAI Completion API (legacy format with 'choices' field)
        
        Args:
            value: Raw response from any supported API
            
        Returns:
            Dict[str, Any]: Dictionary with keys corresponding to AssistantTurn fields
        """
        
        # Check if this is a normalized response (from our GoogleGenAILLM)
        raw_response = value.raw_response if hasattr(value, 'raw_response') else value
        
        # Detect Google GenAI format (Interactions API or generate_content)
        # Google GenAI has 'outputs' (Google Interactions API) or 'candidates' (generate_content)
        # Note: 'outputs' is for Google's Interactions API, 'output' is for LiteLLM Responses API
        if hasattr(raw_response, 'outputs') or \
           (hasattr(raw_response, 'candidates') and not hasattr(value, 'choices')) or \
           hasattr(raw_response, 'usage_metadata'):
            return AssistantTurn.from_google_genai(value)
        
        # Detect LiteLLM/OpenAI/Bedrock format (Responses API, Completion API, or Bedrock Converse)
        # Responses API has 'output' field and object='response'
        # Completion API has 'choices' field
        # Bedrock Converse API has 'output' field with nested 'message'
        # Check both attribute and dict-based access
        has_output = hasattr(value, 'output') or (isinstance(value, dict) and 'output' in value)
        has_choices = hasattr(value, 'choices') or (isinstance(value, dict) and 'choices' in value)
        
        if has_output or has_choices:
            return AssistantTurn.from_litellm_openai_response_api(value)
        
        # Fallback: if has 'text' attribute, might be a simple Google response
        elif hasattr(raw_response, 'text'):
            return AssistantTurn.from_google_genai(value)
        
        # Default to empty result if format is not recognized
        else:
            return {
                "role": "assistant",
                "content": ContentBlockList(),
                "tool_calls": [],
                "unparsed_tool_calls": [],
                "tool_results": [],
                "reasoning": None,
                "finish_reason": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "model": None,
                "timestamp": None,
                "metadata": {}
            }

    def add_text(self, text: str) -> 'AssistantTurn':
        """Add text content"""
        self.content.append(text)
        return self

    def add_image(self, url: Optional[str] = None, data: Optional[str] = None,
                  media_type: str = "image/jpeg") -> 'AssistantTurn':
        """Add image content (some models can generate images)"""
        self.content.append(ImageContent(
            image_url=url,
            image_data=data,
            media_type=media_type
        ))
        return self

    def to_text(self) -> str:
        """Get all text content concatenated. Images will be presented as placeholder text."""
        return self.content.to_text()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "role": self.role,
            "content": [c.to_dict() for c in self.content],
            "reasoning": self.reasoning,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "model": self.model,
            "metadata": self.metadata
        }

    def get_text(self) -> ContentBlockList:
        """Get all text content blocks.
        
        Returns:
            ContentBlockList: List containing only TextContent blocks
        """
        text_blocks = ContentBlockList()
        for block in self.content:
            if isinstance(block, TextContent):
                text_blocks.append(block)
        return text_blocks
    
    def get_images(self) -> ContentBlockList:
        """Get all image content blocks.
        
        Returns:
            ContentBlockList: List containing only ImageContent blocks
        """
        image_blocks = ContentBlockList()
        for block in self.content:
            if isinstance(block, ImageContent):
                image_blocks.append(block)
        return image_blocks

    def __repr__(self) -> str:
        """Safe string representation that handles missing attributes."""
        content_preview = str(self.content)[:50] + "..." if len(str(self.content)) > 50 else str(self.content)
        parts = [f"AssistantTurn(content={content_preview!r}"]
        
        # Safely add optional fields if they exist
        if hasattr(self, 'model') and self.model:
            parts.append(f", model={self.model!r}")
        if hasattr(self, 'prompt_tokens') and self.prompt_tokens:
            parts.append(f", prompt_tokens={self.prompt_tokens}")
        if hasattr(self, 'completion_tokens') and self.completion_tokens:
            parts.append(f", completion_tokens={self.completion_tokens}")
        
        parts.append(")")
        return "".join(parts)
    
    def to_litellm_format(self) -> Dict[str, Any]:
        """Convert to LiteLLM Response API format (OpenAI Response API compatible)"""
        result = {"role": self.role}

        # Handle content blocks (text, images, etc.) - delegate to ContentBlockList
        result["content"] = self.content.to_litellm_format(role=self.role)

        return result
    
    
    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks with glassmorphism design."""
        try:
            from opto.utils.display.jupyter import render_assistant_turn
            return render_assistant_turn(self)
        except ImportError:
            # Fallback to text representation if display module unavailable
            return None

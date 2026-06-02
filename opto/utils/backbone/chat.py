"""Chat: multi-turn conversation manager that renders to provider formats."""
from typing import List, Dict, Any, Optional, Literal, Union
from dataclasses import dataclass, field
import json

from .content import ContentBlockList, TextContent, ImageContent, Content
from .turns import UserTurn, AssistantTurn


@dataclass
class Chat:
    """Manages conversation history across multiple turns using LiteLLM unified format"""
    turns: List[Union[UserTurn, AssistantTurn]] = field(default_factory=list)
    system_prompt: Optional[str] = None
    protected_rounds: int = 0  # Initial rounds to never truncate (task definition)

    def add_user_turn(self, turn: Union[str, ContentBlockList, 'TextContent', 'ImageContent', 'Content', UserTurn]) -> 'Chat':
        """Add a user turn
        
        Args:
            turn: Can be:
                - str: Plain text message
                - ContentBlockList: List of content blocks
                - TextContent: Single text content block
                - ImageContent: Single image content block
                - Content: Multi-modal content wrapper
                - UserTurn: Complete user turn object
        
        Returns:
            Chat: Self for method chaining
            
        Raises:
            TypeError: If turn is not one of the accepted types
        """
        # Accept UserTurn directly
        if isinstance(turn, UserTurn):
            self.turns.append(turn)
            return self

        assert isinstance(
            turn, (str, ContentBlockList, TextContent, ImageContent, Content)
        ), "turn must be a string, ContentBlockList, TextContent, ImageContent, or Content object"
        user_turn = UserTurn(content=turn)
        self.turns.append(user_turn)
        return self

    def add_assistant_turn(self, turn: AssistantTurn) -> 'Chat':
        """Add an assistant turn. AssistantTurn parses the response from the LLM."""
        assert isinstance(turn, AssistantTurn), "turn must be an AssistantTurn object"
        self.turns.append(turn)
        return self

    def get_last_user_turn(self) -> Optional[UserTurn]:
        """Get the most recent user turn"""
        for turn in reversed(self.turns):
            if isinstance(turn, UserTurn):
                return turn
        return None

    def get_last_assistant_turn(self) -> Optional[AssistantTurn]:
        """Get the most recent assistant turn"""
        for turn in reversed(self.turns):
            if isinstance(turn, AssistantTurn):
                return turn
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "system_prompt": self.system_prompt,
            "protected_rounds": self.protected_rounds,
            "turns": [turn.to_dict() for turn in self.turns]
        }

    def to_litellm_format(
        self, 
        n: int = -1,
        truncate_strategy: Literal["from_start", "from_end"] = "from_start",
        protected_rounds: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Convert to LiteLLM messages format (OpenAI-compatible, works with all providers)
        
        Args:
            n: Number of historical rounds (user+assistant pairs) to include.
               -1 means all history (default: -1).
               The current (potentially incomplete) round is always included.
            truncate_strategy: How to truncate when n is specified:
                - "from_start": Remove oldest rounds, keep the most recent n rounds (default)
                - "from_end": Remove newest rounds, keep the oldest n rounds
            protected_rounds: Number of initial rounds to never truncate (task definition).
                If None, uses self.protected_rounds. These rounds count towards n, so
                if n=5 and protected_rounds=1, you get 1 protected + 4 truncatable rounds.
        
        Returns:
            List of message dictionaries in LiteLLM format
        """
        # Determine protected rounds
        n_protected = protected_rounds if protected_rounds is not None else self.protected_rounds
        protected_turns = n_protected * 2  # Each round = user + assistant
        
        # Apply truncation to turns
        if n == -1:
            selected_turns = self.turns
        else:
            # Protected rounds count towards N
            # So if N=5 and protected_rounds=1, we keep 1 protected + 4 from truncatable
            remaining_rounds = max(0, n - n_protected)
            
            # Split into protected and truncatable turns
            protected_part = self.turns[:protected_turns]
            truncatable_part = self.turns[protected_turns:]
            
            # remaining_rounds = number of rounds (pairs) from the truncatable part
            # Each round = 2 turns (user + assistant)
            # Plus include current incomplete round (if last turn is user, +1)
            has_incomplete_round = len(truncatable_part) > 0 and isinstance(truncatable_part[-1], UserTurn)
            n_turns = remaining_rounds * 2 + (1 if has_incomplete_round else 0)
            
            if truncate_strategy == "from_start":
                # Keep last n_turns from truncatable part (remove from start)
                truncated_part = truncatable_part[-n_turns:] if n_turns > 0 else []
            elif truncate_strategy == "from_end":
                # Keep first n_turns from truncatable part (remove from end)
                truncated_part = truncatable_part[:n_turns] if n_turns > 0 else []
            else:
                raise ValueError(f"Unknown truncate_strategy: {truncate_strategy}. Use 'from_start' or 'from_end'")
            
            # Combine protected + truncated
            selected_turns = protected_part + truncated_part
        
        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for turn in selected_turns:
            messages.append(turn.to_litellm_format())

        return messages
    
    def to_messages(
        self,
        n: int = -1,
        truncate_strategy: Literal["from_start", "from_end"] = "from_start",
        protected_rounds: Optional[int] = None,
        model_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Smart message format conversion that auto-detects the appropriate format.
        
        This method automatically chooses between Gemini format and LiteLLM format based on
        the model name. Detection priority:
        1. If model_name argument is provided and contains "gemini", uses Gemini format
        2. Otherwise, checks if any AssistantTurn has a model name containing "gemini"
        3. If no Gemini model detected, uses LiteLLM format (default)
        
        Note: This detection may not work for custom LLM backends with Gemini model names.
        In such cases, call to_gemini_format() or to_litellm_format() explicitly.
        
        Args:
            n: Number of historical rounds (user+assistant pairs) to include.
               -1 means all history (default: -1).
               The current (potentially incomplete) round is always included.
            truncate_strategy: How to truncate when n is specified:
                - "from_start": Remove oldest rounds, keep the most recent n rounds (default)
                - "from_end": Remove newest rounds, keep the oldest n rounds
            protected_rounds: Number of initial rounds to never truncate (task definition).
                If None, uses self.protected_rounds. Counts towards n.
            model_name: Optional model name to use for format detection. If provided and
                contains "gemini" (case-insensitive), forces Gemini format.
        
        Returns:
            List of message dictionaries in the appropriate format
        
        Example:
            # Automatically uses Gemini format if model is Gemini
            history = Chat()
            history.system_prompt = "You are helpful."
            history.add_user_turn(UserTurn().add_text("Hello"))
            
            # Force Gemini format by providing model name
            messages = history.to_messages(model_name="gemini-2.5-flash")
            
            # Or be explicit:
            messages = history.to_gemini_format()  # Force Gemini format
            messages = history.to_litellm_format()  # Force LiteLLM format
        """
        # Check if model_name argument indicates Gemini (highest priority)
        use_gemini_format = False
        if model_name and 'gemini' in model_name.lower():
            use_gemini_format = True
        else:
            # Check if any AssistantTurn has a Gemini model
            for turn in self.turns:
                if isinstance(turn, AssistantTurn) and turn.model:
                    if 'gemini' in turn.model.lower():
                        use_gemini_format = True
                        break
        
        # Use the appropriate format
        if use_gemini_format:
            return self.to_gemini_format(
                n=n,
                truncate_strategy=truncate_strategy,
                protected_rounds=protected_rounds
            )
        else:
            return self.to_litellm_format(
                n=n,
                truncate_strategy=truncate_strategy,
                protected_rounds=protected_rounds
            )
    
    def to_gemini_format(
        self,
        n: int = -1,
        truncate_strategy: Literal["from_start", "from_end"] = "from_start",
        protected_rounds: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Convert to Google Gemini format (messages with 'model' role instead of 'assistant')
        
        This method converts the conversation history to a format compatible with Google's
        Gemini API. The main differences from LiteLLM format are:
        - Uses 'model' instead of 'assistant' for role names
        - Content is structured as 'parts' (list of text/image parts)
        - System message (if present) remains as first message with role='system'
        
        The GoogleGenAILLM class will extract the system message and convert it to
        system_instruction when making the API call.
        
        Args:
            n: Number of historical rounds (user+assistant pairs) to include.
               -1 means all history (default: -1).
               The current (potentially incomplete) round is always included.
            truncate_strategy: How to truncate when n is specified:
                - "from_start": Remove oldest rounds, keep the most recent n rounds (default)
                - "from_end": Remove newest rounds, keep the oldest n rounds
            protected_rounds: Number of initial rounds to never truncate (task definition).
                If None, uses self.protected_rounds. These rounds count towards n.
        
        Returns:
            List of message dictionaries in Gemini format with 'role' and 'parts'.
            System message (if present) is included as first message with role='system'.
        
        Example:
            from opto.utils.llm import LLM
            from opto.utils.backbone import Chat, UserTurn
            
            # Create conversation
            history = Chat()
            history.system_prompt = "You are a helpful assistant."
            history.add_user_turn(UserTurn().add_text("Hello!"))
            
            # Convert to Gemini format
            messages = history.to_gemini_format()
            
            # Use with GoogleGenAILLM
            llm = LLM(model="gemini-2.5-flash")
            response = llm(messages=messages)
        """
        # Get the LiteLLM format messages first (handles truncation logic)
        litellm_messages = self.to_litellm_format(
            n=n,
            truncate_strategy=truncate_strategy,
            protected_rounds=protected_rounds
        )
        
        # Convert messages to Google GenAI format
        gemini_messages = []
        
        for msg in litellm_messages:
            role = msg.get('role')
            content = msg.get('content')
            
            # Keep system messages as-is (will be extracted by GoogleGenAILLM)
            if role == 'system':
                gemini_messages.append({'role': 'system', 'content': content})
                continue
            
            # Map roles: user -> user, assistant -> model
            if role == 'assistant':
                role = 'model'
            elif role == 'tool':
                # Skip tool messages for now - Gemini handles these differently
                # TODO: Handle tool results properly if needed
                continue
            
            # Handle content (can be string or list of content blocks)
            if isinstance(content, str):
                gemini_messages.append({'role': role, 'parts': [{'text': content}]})
            elif isinstance(content, list):
                # Convert content blocks to parts
                parts = []
                for block in content:
                    if block.get('type') == 'text':
                        parts.append({'text': block.get('text', '')})
                    elif block.get('type') == 'image':
                        # Handle image URLs
                        image_url = block.get('image_url', '')
                        if image_url.startswith('data:'):
                            # Extract base64 data
                            import re
                            match = re.match(r'data:([^;]+);base64,(.+)', image_url)
                            if match:
                                mime_type, data = match.groups()
                                parts.append({'inline_data': {'mime_type': mime_type, 'data': data}})
                        else:
                            # External URL
                            parts.append({'file_data': {'file_uri': image_url}})
                if parts:
                    gemini_messages.append({'role': role, 'parts': parts})
        
        return gemini_messages

    def save_to_file(self, filepath: str):
        """Save conversation history to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'Chat':
        """Load conversation history from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # This is a simplified loader - you'd want more robust deserialization
        history = cls(
            system_prompt=data.get('system_prompt'),
            protected_rounds=data.get('protected_rounds', 0)
        )

        # Note: Full deserialization would require reconstructing objects from dicts
        # This is left as an exercise since it depends on your exact needs

        return history

    def clear(self):
        """Clear all turns from history"""
        self.turns.clear()

    def get_token_count_estimate(self) -> int:
        """Rough estimate of token count (actual count requires tokenizer)"""
        total = 0
        for turn in self.turns:
            if isinstance(turn, (UserTurn, AssistantTurn)):
                for block in turn.content:
                    if isinstance(block, TextContent):
                        # Very rough estimate: ~4 chars per token
                        total += len(block.text) // 4
        return total
    
    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks with glassmorphism design."""
        try:
            from opto.utils.display.jupyter import render_chat
            return render_chat(self)
        except ImportError:
            # Fallback to text representation if display module unavailable
            return None

"""
When MM (multimodal) is enabled, we primarily either use:
1. LiteLLM's response API
2. Google's Interaction API design (not supported by LiteLLM response API at all)
When MM is disabled, for backward compatibility, we use:
1. LiteLLM's completion API
"""

from typing import List, Tuple, Dict, Any, Callable, Union, Optional
import os
import time
import json
import os
import warnings
from .auto_retry import retry_with_exponential_backoff

# Import content/turn types for mm_beta mode.
# Heavy/optional SDKs (openai, google-genai) are imported lazily inside the
# backends that need them, so importing this module never requires them.
from .backbone import AssistantTurn, TextContent, ImageContent

try:
    import autogen  # We import autogen here to avoid the need of installing autogen
except ImportError:
    pass


def _is_image_generation_model(model_name: str) -> bool:
    """Detect if a model is for image generation based on its name.
    
    Detects:
    - OpenAI: gpt-image-1, gpt-image-1.5, gpt-image-1-mini, dall-e-2, dall-e-3
    - Gemini: gemini-2.5-flash-image, gemini-2.5-pro-image, etc.
    
    Args:
        model_name: The name of the model to check
        
    Returns:
        bool: True if the model is an image generation model, False otherwise
    """
    if model_name is None:
        return False
    model_lower = model_name.lower()
    return 'image' in model_lower or 'dall-e' in model_lower

class AbstractModel:
    """Abstract base class for LLM model wrappers with automatic refreshing.

    Provides a minimal abstraction for model APIs that need periodic refreshing
    for certificate renewal or memory management in long-running processes.

    Parameters
    ----------
    factory : callable
        A function that takes no arguments and returns a callable model instance.
    reset_freq : int or None, optional
        Number of seconds after which to refresh the model. If None, the model
        is never refreshed.
    mm_beta : bool, optional
        If True, returns AssistantTurn objects with rich multimodal content.
        If False (default), returns raw API responses in legacy format.
    model_name : str or None, optional
        The name of the model being used (e.g., "gpt-4o", "claude-3-5-sonnet-latest").
        If None, no model name is stored.

    Attributes
    ----------
    factory : callable
        The factory function for creating model instances.
    reset_freq : int or None
        Refresh frequency in seconds.
    mm_beta : bool
        Whether to use multimodal beta mode.
    model_name : str or None
        The name of the model being used.
    is_image_model : bool
        Whether the model is for image generation (auto-detected from model name).

    model : Any
        Property that returns the current model instance.

    Methods
    -------
    __call__(*args, **kwargs)
        Execute the model, refreshing if needed. Returns AssistantTurn if mm_beta=True,
        otherwise returns raw API response.

    Notes
    -----
    This class handles:
    1. **Automatic Refreshing**: Recreates the model instance periodically
       to prevent issues with long-running connections.
    2. **Serialization**: Supports pickling by recreating the model on load.
    3. **Response Formats**: 
       - Legacy (mm_beta=False): `response['choices'][0]['message']['content']`
       - Multimodal (mm_beta=True): AssistantTurn object with .content, .tool_calls, etc.

    Subclasses should override the `model` property to customize behavior.

    See Also
    --------
    AutoGenLLM : Concrete implementation using AutoGen
    LiteLLM : Concrete implementation using LiteLLM
    """

    def __init__(self, factory: Callable, reset_freq: Union[int, None] = None, 
                 mm_beta: bool = False, model_name: Union[str, None] = None) -> None:
        """
        Args:
            factory: A function that takes no arguments and returns a model that is callable.
            reset_freq: The number of seconds after which the model should be
                refreshed. If None, the model is never refreshed.
            mm_beta: If True, returns AssistantTurn objects with rich multimodal content.
                If False (default), returns raw API responses in legacy format.
            model_name: The name of the model being used (e.g., "gpt-4o", "claude-3-5-sonnet-latest").
                If None, no model name is stored.
        """
        self.factory = factory
        self._model = self.factory()
        self.reset_freq = reset_freq
        self._init_time = time.time()
        self.mm_beta = mm_beta
        self.model_name = model_name

    # Overwrite this `model` property when subclassing.
    @property
    def model(self):
        """When self.model is called, text responses should always be available at `response['choices'][0]['message']['content']`"""
        return self._model
    
    @property
    def is_image_model(self) -> bool:
        """Check if this model is for image generation based on model name.
        
        Returns True if the model name contains 'image' or 'dall-e', False otherwise.
        """
        return _is_image_generation_model(self.model_name)

    # This is the main API
    def __call__(self, *args, **kwargs) -> Any:
        """ The call function handles refreshing the model if needed.
        
        Returns:
            If mm_beta=False: Raw completion API response (backward compatible)
            If mm_beta=True: AssistantTurn object with parsed multimodal content
        """
        if self.reset_freq is not None and time.time() - self._init_time > self.reset_freq:
            self._model = self.factory()
            self._init_time = time.time()
        
        response = self.model(*args, **kwargs)
        
        # Parse to AssistantTurn if mm_beta mode is enabled
        if self.mm_beta:
            return AssistantTurn(response)
        
        return response

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._model = self.factory()


class AutoGenLLM(AbstractModel):
    """LLM wrapper using AutoGen's OpenAIWrapper for model interactions.

    This class provides integration with AutoGen for accessing various LLM APIs.
    It handles configuration, caching, and provides a consistent interface for
    the Trace framework.

    Parameters
    ----------
    config_list : list, optional
        List of model configurations. If None, attempts to load from
        'OAI_CONFIG_LIST' environment variable or auto-constructs from
        individual API keys.
    filter_dict : dict, optional
        Dictionary to filter configurations based on model properties.
    reset_freq : int or None, optional
        Number of seconds after which to refresh the model connection.

    Methods
    -------
    create(**config)
        Make a completion request with the given configuration.
    _factory(config_list)
        Class method to create the underlying AutoGen wrapper.

    Notes
    -----
    Configuration sources (in priority order):
    1. Explicitly provided config_list
    2. OAI_CONFIG_LIST environment variable or file
    3. Auto-construction from individual API keys (OPENAI_API_KEY, etc.)

    The create() method supports AutoGen's full configuration options including:
    - Templating with context
    - Response caching
    - Custom filter functions
    - API version specification

    For models not supported by AutoGen, subclass and override _factory()
    and create() methods.

    See Also
    --------
    AbstractModel : Base class for model wrappers
    auto_construct_oai_config_list_from_env : Helper for config construction

    Examples
    --------
    >>> # Using with explicit configuration
    >>> llm = AutoGenLLM(config_list=[{"model": "gpt-4", "api_key": "..."}])
    >>> 
    >>> # Using with environment variables
    >>> llm = AutoGenLLM()  # Auto-loads from environment
    >>> 
    >>> # Making a completion
    >>> response = llm(messages=[{"role": "user", "content": "Hello"}])
    """

    def __init__(self, config_list: List = None, filter_dict: Dict = None, 
                 reset_freq: Union[int, None] = None, mm_beta: bool = False) -> None:
        if config_list is None:
            try:
                config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")
            except:
                config_list = auto_construct_oai_config_list_from_env()
                if len(config_list) > 0:
                    os.environ.update({"OAI_CONFIG_LIST": json.dumps(config_list)})
                config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")
        if filter_dict is not None:
            config_list = autogen.filter_config(config_list, filter_dict)

        # Extract model name from config_list if available
        model_name = config_list[0].get('model') if config_list and len(config_list) > 0 else None

        factory = lambda *args, **kwargs: self._factory(config_list)
        super().__init__(factory, reset_freq, mm_beta=mm_beta, model_name=model_name)

    @classmethod
    def _factory(cls, config_list):
        return autogen.OpenAIWrapper(config_list=config_list)

    @property
    def model(self):
        return lambda **kwargs: self.create(**kwargs)

    # This is main API. We use the API of autogen's OpenAIWrapper
    def create(self, **config: Any):
        """Make a completion for a given config using available clients.
        Besides the kwargs allowed in openai's [or other] client, we allow the following additional kwargs.
        The config in each client will be overridden by the config.

        Args:
            - context (Dict | None): The context to instantiate the prompt or messages. Default to None.
                It needs to contain keys that are used by the prompt template or the filter function.
                E.g., `prompt="Complete the following sentence: {prefix}, context={"prefix": "Today I feel"}`.
                The actual prompt will be:
                "Complete the following sentence: Today I feel".
                More examples can be found at [templating](/docs/Use-Cases/enhanced_inference#templating).
            - cache (AbstractCache | None): A Cache object to use for response cache. Default to None.
                Note that the cache argument overrides the legacy cache_seed argument: if this argument is provided,
                then the cache_seed argument is ignored. If this argument is not provided or None,
                then the cache_seed argument is used.
            - agent (AbstractAgent | None): The object responsible for creating a completion if an agent.
            - (Legacy) cache_seed (int | None) for using the DiskCache. Default to 41.
                An integer cache_seed is useful when implementing "controlled randomness" for the completion.
                None for no caching.
                Note: this is a legacy argument. It is only used when the cache argument is not provided.
            - filter_func (Callable | None): A function that takes in the context and the response
                and returns a boolean to indicate whether the response is valid. E.g.,
            - allow_format_str_template (bool | None): Whether to allow format string template in the config. Default to false.
            - api_version (str | None): The api version. Default to None. E.g., "2024-02-01".

        Example:
            >>> # filter_func example:
            >>> def yes_or_no_filter(context, response):
            >>>    return context.get("yes_or_no_choice", False) is False or any(
            >>>        text in ["Yes.", "No."] for text in client.extract_text_or_completion_object(response)
            >>>    )

        Raises:
            - RuntimeError: If all declared custom model clients are not registered
            - APIError: If any model client create call raises an APIError
        """
        return self._model.create(**config)


def embed(model: str, text: str, max_retries: int = 10, base_delay: float = 1.0):
    """Call an embedding API with automatic retry on failure.

    Uses litellm under the hood. Returns the embedding vector (list of floats),
    or None if all retries are exhausted.

    Args:
        model: Embedding model identifier accepted by litellm
            (e.g. "gemini/gemini-embedding-001", "openai/text-embedding-3-small").
        text: Text to embed.
        max_retries: Maximum number of retry attempts on failure.
        base_delay: Initial delay (in seconds) for exponential backoff.
    """
    import litellm
    try:
        response = retry_with_exponential_backoff(
            lambda: litellm.embedding(model=model, input=text),
            max_retries=max_retries,
            base_delay=base_delay,
            operation_name="Embedding API call",
        )
        return response.data[0].embedding
    except Exception as e:
        warnings.warn(f"Embedding API call failed after retries: {e}")
        return None


def auto_construct_oai_config_list_from_env() -> List:
    """
    Collect various API keys saved in the environment and return a format like:
    [{"model": "gpt-4", "api_key": xxx}, {"model": "claude-3.5-sonnet", "api_key": xxx}]

    Note this is a lazy function that defaults to gpt-40 and claude-3.5-sonnet.
    If you want to specify your own model, please provide an OAI_CONFIG_LIST in the environment or as a file
    """
    config_list = []
    if os.environ.get("OPENAI_API_KEY") is not None:
        config_list.append(
            {"model": "gpt-4o", "api_key": os.environ.get("OPENAI_API_KEY")}
        )
    if os.environ.get("ANTHROPIC_API_KEY") is not None:
        config_list.append(
            {
                "model": "claude-3-5-sonnet-latest",
                "api_key": os.environ.get("ANTHROPIC_API_KEY"),
            }
        )
    return config_list


class LiteLLM(AbstractModel):
    """
    This is an LLM backend supported by LiteLLM library.

    https://docs.litellm.ai/docs/completion/input
    https://docs.litellm.ai/docs/response_api
    https://docs.litellm.ai/docs/image_generation

    To use this, set the credentials through the environment variable as
    instructed in the LiteLLM documentation. For convenience, you can set the
    default model name through the environment variable TRACE_LITELLM_MODEL.
    
    Azure OpenAI Authentication:
        Two authentication methods are supported for Azure OpenAI:
        
        1. API Key Authentication (Recommended for most users):
           Set these environment variables:
           - AZURE_API_KEY: Your Azure OpenAI API key
           - AZURE_API_BASE: Your Azure OpenAI endpoint (e.g., https://your-resource.openai.azure.com)
           - AZURE_API_VERSION: API version (e.g., 2024-08-01-preview)
           - TRACE_LITELLM_MODEL: Model name with azure/ prefix (e.g., azure/o4-mini)
           
           Do NOT set AZURE_TOKEN_PROVIDER_SCOPE for this method.
        
        2. Azure AD Credential Authentication (For enterprise users):
           Set AZURE_TOKEN_PROVIDER_SCOPE (e.g., https://cognitiveservices.azure.com/.default)
           to use Azure Identity credential-based authentication.
           This method does NOT use AZURE_API_KEY.
    
    This class now supports storing default completion parameters (like temperature,
    top_p, max_tokens, etc.) that will be used for all calls unless overridden.
    
    Text Generation:
        When mm_beta=True, the Responses API is used for rich multimodal content.
        When mm_beta=False (default), the Completion API is used for backward compatibility.
        
        See: https://docs.litellm.ai/docs/response_api
    
    Image Generation:
        Automatically detects image generation models (containing 'image' or 'dall-e' in name).
        Uses litellm.image_generation() API for models like:
        - gpt-image-1, gpt-image-1.5, gpt-image-1-mini
        - dall-e-2, dall-e-3
        
        Image models require a single string prompt:
            llm = LLM(model="gpt-image-1.5")
            result = llm(prompt="A serene mountain landscape")
        
        Check llm.is_image_model to determine if a model is for image generation.
    """

    def __init__(self, model: Union[str, None] = None, reset_freq: Union[int, None] = None,
                 cache=True, max_retries=1, base_delay=1.0,
                 mm_beta: bool = False, **default_params) -> None:
        if model is None:
            model = os.environ.get('TRACE_LITELLM_MODEL')
            if model is None:
                # warnings.warn("TRACE_LITELLM_MODEL environment variable is not found when loading the default model for LiteLLM. Attempt to load the default model from DEFAULT_LITELLM_MODEL environment variable. The usage of DEFAULT_LITELLM_MODEL will be deprecated. Please use the environment variable TRACE_LITELLM_MODEL for setting the default model name for LiteLLM.")
                model = os.environ.get('DEFAULT_LITELLM_MODEL', 'gpt-4o')

        self.model_name = model
        self.cache = cache
        self.default_params = default_params  # Store default completion parameters

        factory = lambda: self._factory(
            self.model_name, 
            self.default_params, 
            mm_beta,
            max_retries=max_retries, 
            base_delay=base_delay
        )
        super().__init__(factory, reset_freq, mm_beta=mm_beta, model_name=model)

    @classmethod
    def _factory(cls, model_name: str, default_params: dict, mm_beta: bool,
                 max_retries=1, base_delay=1.0):
        import litellm
        
        # For Azure models, set global litellm variables as a fallback
        # (workaround for potential litellm.responses API issues)
        if model_name.startswith('azure/'):
            if os.environ.get('AZURE_API_VERSION') and not hasattr(litellm, '_azure_api_version_set'):
                try:
                    litellm.api_version = os.environ.get('AZURE_API_VERSION')
                    litellm._azure_api_version_set = True  # Mark to avoid setting multiple times
                except:
                    pass  # Ignore if litellm doesn't support this
        
        # Check if this is an image generation model
        is_image_model = _is_image_generation_model(model_name)
        
        if is_image_model:
            # Image generation API
            api_func = litellm.image_generation
            operation_name = "LiteLLM_image_generation"
            
            # Standard image generation wrapper
            def image_wrapper(prompt, **kwargs):
                assert isinstance(prompt, str), (
                    f"Image generation requires a single string prompt. "
                    f"Got {type(prompt).__name__}. "
                    f"Usage: llm(prompt='your prompt here')"
                )
                return retry_with_exponential_backoff(
                    lambda: api_func(model=model_name, prompt=prompt, **{**default_params, **kwargs}),
                    max_retries=max_retries,
                    base_delay=base_delay,
                    operation_name=operation_name
                )
            return image_wrapper
        
        # Use Responses API when mm_beta=True, otherwise use Completion API
        api_func = litellm.responses if mm_beta else litellm.completion
        operation_name = "LiteLLM_responses" if mm_beta else "LiteLLM_completion"
        
        if model_name.startswith('azure/'):  # azure model
            azure_token_provider_scope = os.environ.get('AZURE_TOKEN_PROVIDER_SCOPE', None)
            if azure_token_provider_scope is not None:
                # Azure AD credential-based authentication
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider
                credential = get_bearer_token_provider(DefaultAzureCredential(), azure_token_provider_scope)
                if mm_beta:
                    # Responses API: model as keyword argument, convert messages to input
                    def azure_responses_wrapper(*args, **kwargs):
                        # Convert 'messages' to 'input' for Responses API
                        if 'messages' in kwargs and 'input' not in kwargs:
                            kwargs['input'] = kwargs.pop('messages')
                        return retry_with_exponential_backoff(
                            lambda: api_func(model=model_name,
                                           azure_ad_token_provider=credential, **{**default_params, **kwargs}),
                            max_retries=max_retries,
                            base_delay=base_delay,
                            operation_name=operation_name
                        )
                    return azure_responses_wrapper
                else:
                    # Completion API: model as positional argument
                    return lambda *args, **kwargs: retry_with_exponential_backoff(
                        lambda: api_func(model_name, *args,
                                       azure_ad_token_provider=credential, **{**default_params, **kwargs}),
                        max_retries=max_retries,
                        base_delay=base_delay,
                        operation_name=operation_name
                    )
            else:
                # Azure API key authentication - explicitly pass Azure env vars if available
                azure_params = {}
                if 'api_key' not in default_params:
                    azure_api_key = os.environ.get('AZURE_API_KEY')
                    if azure_api_key:
                        azure_params['api_key'] = azure_api_key
                if 'api_base' not in default_params:
                    azure_api_base = os.environ.get('AZURE_API_BASE')
                    if azure_api_base:
                        azure_params['api_base'] = azure_api_base
                if 'api_version' not in default_params:
                    azure_api_version = os.environ.get('AZURE_API_VERSION')
                    if azure_api_version:
                        azure_params['api_version'] = azure_api_version
                
                if mm_beta:
                    # Responses API: model as keyword argument, convert messages to input
                    def azure_key_responses_wrapper(*args, **kwargs):
                        # Convert 'messages' to 'input' for Responses API
                        if 'messages' in kwargs and 'input' not in kwargs:
                            kwargs['input'] = kwargs.pop('messages')
                        return retry_with_exponential_backoff(
                            lambda: api_func(model=model_name, **{**azure_params, **default_params, **kwargs}),
                            max_retries=max_retries,
                            base_delay=base_delay,
                            operation_name=operation_name
                        )
                    return azure_key_responses_wrapper
                else:
                    # Completion API: model as positional argument
                    return lambda *args, **kwargs: retry_with_exponential_backoff(
                        lambda: api_func(model_name, *args, **{**azure_params, **default_params, **kwargs}),
                        max_retries=max_retries,
                        base_delay=base_delay,
                        operation_name=operation_name
                    )
        
        if mm_beta:
            # Responses API: model as keyword argument, convert messages to input
            def responses_wrapper(*args, **kwargs):
                # Convert 'messages' to 'input' for Responses API
                if 'messages' in kwargs and 'input' not in kwargs:
                    kwargs['input'] = kwargs.pop('messages')
                return retry_with_exponential_backoff(
                    lambda: api_func(model=model_name, **{**default_params, **kwargs}),
                    max_retries=max_retries,
                    base_delay=base_delay,
                    operation_name=operation_name
                )
            return responses_wrapper
        else:
            # Completion API: model as positional argument
            return lambda *args, **kwargs: retry_with_exponential_backoff(
                lambda: api_func(model_name, *args, **{**default_params, **kwargs}),
                max_retries=max_retries,
                base_delay=base_delay,
                operation_name=operation_name
            )

    @property
    def model(self):
        """
        Calls either litellm.completion() or litellm.responses() depending on mm_beta.
        
        For completion API (mm_beta=False):
            response = litellm.completion(
                model=self.model,
                messages=[{"content": message, "role": "user"}]
            )
        
        For responses API (mm_beta=True):
            response = litellm.responses(
                model=self.model,
                input="Your input text"
            )
        """
        return lambda *args, **kwargs: self._model(*args, **kwargs)


class CustomLLM(AbstractModel):
    """
    This is for Custom server's API endpoints that are OpenAI Compatible.
    Such server includes LiteLLM proxy server.
    """

    def __init__(self, model: Union[str, None] = None, reset_freq: Union[int, None] = None,
                 cache=True, mm_beta: bool = False) -> None:
        if model is None:
            model = os.environ.get('TRACE_CUSTOMLLM_MODEL', 'gpt-4o')
        base_url = os.environ.get('TRACE_CUSTOMLLM_URL', 'http://xx.xx.xxx.xx:4000/')
        server_api_key = os.environ.get('TRACE_CUSTOMLLM_API_KEY',
                                        'sk-Xhg...')  # we assume the server has an API key
        # the server API is set through `master_key` in `config.yaml` for LiteLLM proxy server

        self.model_name = model
        self.cache = cache
        factory = lambda: self._factory(base_url, server_api_key)  # an LLM instance uses a fixed model
        super().__init__(factory, reset_freq, mm_beta=mm_beta, model_name=model)

    @classmethod
    def _factory(cls, base_url: str, server_api_key: str):
        import openai
        return openai.OpenAI(base_url=base_url, api_key=server_api_key)

    @property
    def model(self):
        return lambda **kwargs: self.create(**kwargs)
        # return lambda *args, **kwargs: self._model.chat.completions.create(*args, **kwargs)

    def create(self, **config: Any):
        if 'model' not in config:
            config['model'] = self.model_name
        return self._model.chat.completions.create(**config)

class GoogleGenAILLM(AbstractModel):
    """
    This is an LLM backend using Google's GenAI SDK with the Interactions API.
    
    https://ai.google.dev/gemini-api/docs/text-generation
    https://ai.google.dev/gemini-api/docs/image-generation
    
    The Interactions API is a unified interface for interacting with Gemini models,
    similar to OpenAI's Response API. It provides better state management, tool
    orchestration, and support for long-running tasks.
    
    To use this, set the GEMINI_API_KEY environment variable with your API key.
    For convenience, you can set the default model name through the environment 
    variable TRACE_GOOGLE_GENAI_MODEL.
    
    Supported models:
    - Text: gemini-2.5-flash, gemini-2.5-pro, gemini-2.5-flash-lite
    - Image: gemini-2.5-flash-image, gemini-2.5-pro-image
    
    This class supports storing default generation parameters (like temperature,
    max_output_tokens, etc.) that will be used for all calls unless overridden.
    
    Text Generation:
        Use ConversationHistory.to_gemini_format() to convert conversation history
        to the format expected by Google GenAI.
        
        Example:
            from opto.utils.llm import LLM
            from opto.utils.backbone import ConversationHistory, UserTurn, AssistantTurn
            
            # Initialize LLM
            llm = LLM(model="gemini-2.5-flash")
            
            # Create conversation history
            history = ConversationHistory()
            history.system_prompt = "You are a helpful assistant."
            history.add_user_turn(UserTurn().add_text("What is AI?"))
            
            # Convert to Gemini format and call LLM
            messages = history.to_gemini_format()
            response = llm(messages=messages, max_tokens=100)
            
            # Parse response
            at = AssistantTurn(response)
            print(at.get_text())
    
    Image Generation:
        Automatically detects image generation models (containing 'image' in name).
        Uses client.models.generate_images() API for models like gemini-2.5-flash-image.
        
        Image models require a single string prompt:
            llm = LLM(model="gemini-2.5-flash-image")
            result = llm(prompt="A serene mountain landscape", number_of_images=2)
        
        Check llm.is_image_model to determine if a model is for image generation.
    """

    def __init__(self, model: Union[str, None] = None, reset_freq: Union[int, None] = None,
                 cache=True, mm_beta: bool = False, max_retries: int = 1,
                 base_delay: float = 1.0, **default_params) -> None:
        if model is None:
            model = os.environ.get('TRACE_GOOGLE_GENAI_MODEL', 'gemini-2.5-flash')
        
        self.model_name = model
        self.cache = cache
        self.default_params = default_params  # Store default generation parameters
        factory = lambda: self._factory(self.model_name, self.default_params,
                                        max_retries=max_retries, base_delay=base_delay)
        super().__init__(factory, reset_freq, mm_beta=mm_beta, model_name=model)

    @classmethod
    def _factory(cls, model_name: str, default_params: dict,
                 max_retries: int = 1, base_delay: float = 1.0):
        """Create a Google GenAI client wrapper using the Interactions API."""
        from google import genai
        from google.genai import types
        # Get API key from environment variable
        api_key = os.environ.get('GEMINI_API_KEY')
        if api_key:
            client = genai.Client(api_key=api_key)
        else:
            # Try without API key (will use default credentials or fail gracefully)
            client = genai.Client()

        # Check if this is an image generation model
        is_image_model = _is_image_generation_model(model_name)
        
        if is_image_model:
            # Image generation for Gemini
            def image_api_func(prompt, **kwargs):
                assert isinstance(prompt, str), (
                    f"Image generation requires a single string prompt. "
                    f"Got {type(prompt).__name__}. "
                    f"Usage: llm(prompt='your prompt here')"
                )
                
                # Gemini image generation API
                # https://ai.google.dev/gemini-api/docs/image-generation
                # Filter kwargs to only valid parameters for generate_images
                valid_params = {
                    k: v for k, v in kwargs.items() 
                    if k in ['number_of_images', 'aspect_ratio', 'safety_filter_level']
                }
                response = client.models.generate_images(
                    model=model_name,
                    prompt=prompt,
                    **valid_params
                )
                return response
            
            return lambda *args, **kwargs: retry_with_exponential_backoff(
                lambda: image_api_func(*args, **{**default_params, **kwargs}),
                max_retries=max_retries,
                base_delay=base_delay,
                operation_name=f"{model_name}_image_gen"
            )

        # Build config if there are generation parameters
        config_params = {}
        
        # Handle thinking config for Gemini 2.5+ models
        if 'thinking_budget' in default_params:
            thinking_budget = default_params.pop('thinking_budget')
            config_params['thinking_config'] = types.ThinkingConfig(
                thinking_budget=thinking_budget
            )

        def api_func(model_name, *args, **kwargs):
            # Extract system_instruction if present (needs to be at config level, not in kwargs)
            system_instruction = kwargs.pop('system_instruction', None)
            
            # Handle messages parameter (from history.to_gemini_format())
            messages = kwargs.pop('messages', None)
            contents = kwargs.pop('contents', None)
            
            if messages:
                # Detect format: OpenAI-style has 'content' key, Gemini-native has 'parts' key.
                # If OpenAI-style, convert via _gemini_messages_to_contents so the SDK accepts it.
                first_non_system = next(
                    (m for m in messages if m.get('role') != 'system'), None
                )
                is_openai_format = (
                    first_non_system is not None and
                    'content' in first_non_system and
                    'parts' not in first_non_system
                )
                if is_openai_format:
                    converted, extracted_sys = _gemini_messages_to_contents(messages)
                    if system_instruction is None:
                        system_instruction = extracted_sys
                    contents = converted
                else:
                    # Already in Gemini native format (from history.to_gemini_format())
                    if messages[0].get('role') == 'system':
                        if system_instruction is None:
                            system_instruction = messages[0].get('content')
                        contents = messages[1:]
                    else:
                        contents = messages

            # Use contents if provided, otherwise use positional args.
            # If a positional arg is a list of OpenAI-format dicts, convert it.
            if contents is None and args:
                raw = args[0]
                if (
                    isinstance(raw, list) and raw and
                    isinstance(raw[0], dict) and
                    'content' in raw[0] and 'parts' not in raw[0]
                ):
                    converted, extracted_sys = _gemini_messages_to_contents(raw)
                    if system_instruction is None:
                        system_instruction = extracted_sys
                    contents = converted
                else:
                    contents = raw
            contents_to_use = contents
            
            # Map max_tokens to max_output_tokens for Google GenAI
            if 'max_tokens' in kwargs:
                kwargs['max_output_tokens'] = kwargs.pop('max_tokens')
            
            # Remove any other parameters that shouldn't go to GenerateContentConfig
            # Keep only valid config parameters
            valid_config_params = {
                'temperature', 'max_output_tokens', 'top_p', 'top_k', 
                'stop_sequences', 'candidate_count', 'presence_penalty',
                'frequency_penalty', 'response_mime_type', 'response_schema'
            }
            config_kwargs = {k: v for k, v in kwargs.items() if k in valid_config_params}
            
            if system_instruction:
                config_params_with_system = {**config_params, 'system_instruction': system_instruction}
            else:
                config_params_with_system = config_params
            
            response = client.models.generate_content(
                model=model_name,
                contents=contents_to_use,
                config=types.GenerateContentConfig(**{**config_params_with_system, **config_kwargs})
            )

            return response
            
        return lambda *args, **kwargs: retry_with_exponential_backoff(
            lambda: api_func(model_name, *args, **{**default_params, **kwargs}),
            max_retries=max_retries,
            base_delay=base_delay,
            operation_name=f"{model_name}"
        )

    @property
    def model(self):
        """
        Wrapper that injects the model name into calls.
        
        Example:
            response = llm(contents="How does AI work?")
        """
        return lambda *args, **kwargs: self._model(model=self.model_name, *args, **kwargs)

# ---------------------------------------------------------------------------
# Helper to convert OpenAI-style messages into Gemini 'contents' format.
# ---------------------------------------------------------------------------

def _gemini_messages_to_contents(messages):
    """Convert a standard messages list to Gemini REST API ``contents`` format.

    Returns (contents, system_instruction) where system_instruction is a str
    extracted from any ``role="system"`` message, or None.
    """
    contents = []
    system_instruction = None

    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')

        if role == 'system':
            if isinstance(content, list):
                texts = [
                    item['text'] for item in content
                    if isinstance(item, dict) and item.get('type') == 'text'
                ]
                system_instruction = '\n'.join(texts)
            else:
                system_instruction = str(content)
            continue

        gemini_role = 'model' if role == 'assistant' else 'user'

        if isinstance(content, str):
            parts = [{'text': content}]
        elif isinstance(content, list):
            parts = [
                {'text': item['text']}
                for item in content
                if isinstance(item, dict) and item.get('type') == 'text'
            ]
            if not parts:
                parts = [{'text': str(content)}]
        else:
            parts = [{'text': str(content)}]

        contents.append({'role': gemini_role, 'parts': parts})

    return contents, system_instruction


# Registry of available backends
_LLM_REGISTRY = {
    "LiteLLM": LiteLLM,
    "AutoGen": AutoGenLLM,
    "CustomLLM": CustomLLM,
    "GoogleGenAI": GoogleGenAILLM,
}

class LLMFactory:
    """Factory for creating LLM instances with named profiles.

    Profiles allow you to save and reuse LLM configurations with specific settings.
    Each profile can include any LiteLLM-supported parameters like model, temperature,
    top_p, max_tokens, etc.

    The default profile uses 'gpt-4o-mini' with standard settings.

    Basic Usage:
        # Use default model (gpt-4o-mini)
        llm = LLM()
        
        # Specify a model directly
        llm = LLM(model="gpt-4o")
        
        # Use a named profile
        llm = LLM(profile="my_profile")

    Creating Custom Profiles:
        # Register a profile with full LiteLLM configuration
        LLMFactory.create_profile(
            "creative_writer",
            backend="LiteLLM",
            model="gpt-4o",
            temperature=0.9,
            top_p=0.95,
            max_tokens=2000,
            presence_penalty=0.6
        )
        
        # Register a reasoning profile
        LLMFactory.create_profile(
            "deep_thinker",
            backend="LiteLLM",
            model="o1-preview",
            max_completion_tokens=8000
        )
        
        # Register a profile with specific formatting
        LLMFactory.create_profile(
            "json_responder",
            backend="LiteLLM",
            model="gpt-4o-mini",
            temperature=0.3,
            response_format={"type": "json_object"}
        )

    Using Profiles:
        # Use your custom profile
        llm = LLM(profile="creative_writer")
        
        # In optimizers
        optimizer = OptoPrime(parameters, llm=LLM(profile="deep_thinker"))

    Profile Management:
        # List all available profiles
        profiles = LLMFactory.list_profiles()
        
        # Get profile configuration
        config = LLMFactory.get_profile_info("creative_writer")
        
        # Override existing profile
        LLMFactory.create_profile("default", "LiteLLM", model="gpt-4o", temperature=0.5)

    Supported LiteLLM Parameters:
        See https://docs.litellm.ai/docs/completion/input for full list:
        - model: Model name (required)
        - temperature: Sampling temperature (0-2)
        - top_p: Nucleus sampling parameter
        - max_tokens: Maximum tokens to generate
        - max_completion_tokens: Upper bound for completion tokens
        - presence_penalty: Penalize new tokens based on presence
        - frequency_penalty: Penalize new tokens based on frequency
        - stop: Stop sequences (string or list)
        - stream: Enable streaming responses
        - response_format: Output format specification
        - seed: Deterministic sampling seed
        - tools: Function calling tools
        - tool_choice: Control function calling behavior
        - logprobs: Return log probabilities
        - top_logprobs: Number of most likely tokens to return
        - n: Number of completions to generate
        - and many more...
    """

    # Default profile - just gpt-4o-mini with no opinionated settings
    _profiles = {
        'default': {'backend': 'LiteLLM', 'params': {'model': 'gpt-4o'}},
    }
    @classmethod
    def get_llm(cls, profile: str = 'default', model: str = None, mm_beta: bool = False, **kwargs) -> AbstractModel:
        """Get an LLM instance for the specified profile or model.
        
        Args:
            profile: Name of the profile to use. Defaults to 'default'.
            model: Model name to use directly. If provided, overrides profile.
            mm_beta: If True, returns AssistantTurn objects with rich multimodal content.
                If False (default), returns raw API responses in legacy format.
            **kwargs: Additional parameters to pass to the backend (e.g., temperature, top_p).
                     These override profile settings if both are specified.
        
        Returns:
            An LLM instance configured according to the profile/model and parameters.
        
        Examples:
            # Use default profile
            llm = LLMFactory.get_llm()
            
            # Use specific model
            llm = LLMFactory.get_llm(model="gpt-4o")
            
            # Use named profile
            llm = LLMFactory.get_llm(profile="creative_writer")
            
            # Use model with custom parameters
            llm = LLMFactory.get_llm(model="gpt-4o", temperature=0.7, max_tokens=1000)
            
            # Override profile settings
            llm = LLMFactory.get_llm(profile="creative_writer", temperature=0.5)
            
            # Use mm_beta mode for multimodal responses
            llm = LLMFactory.get_llm(model="gpt-4o", mm_beta=True)
        """
        # If model is specified directly, create a simple config
        if model is not None:
            backend = kwargs.pop('backend', None)
            
            # Determine backend with priority:
            # 1. Explicit backend kwarg (always wins) 
            # 2. Gemini model name  -> GoogleGenAI
            # 3. Default            -> LiteLLM
            if backend is not None:
                backend_cls = _LLM_REGISTRY[backend]
            elif model.startswith('gemini'):
                # Gemini models default to GoogleGenAILLM backend
                backend_cls = _LLM_REGISTRY['GoogleGenAI']
                # Strip 'gemini/' prefix if present (LiteLLM format: gemini/gemini-pro)
                if model.startswith('gemini/'):
                    model = model[len('gemini/'):]
            else:
                # Default to LiteLLM for other models
                backend_cls = _LLM_REGISTRY['LiteLLM']
            
            params = {'model': model, 'mm_beta': mm_beta, **kwargs}
            return backend_cls(**params)
        # Otherwise use profile
        if profile not in cls._profiles:
            raise ValueError(
                f"Unknown profile '{profile}'. Available profiles: {list(cls._profiles.keys())}. "
                f"Use LLMFactory.create_profile() to create custom profiles, or pass model= directly."
            )

        config = cls._profiles[profile].copy()
        backend_cls = _LLM_REGISTRY[config['backend']]
        
        # Merge profile params with any override kwargs
        params = config['params'].copy()
        params['mm_beta'] = mm_beta
        params.update(kwargs)
        
        return backend_cls(**params)

    @classmethod
    def create_profile(cls, name: str, backend: str = 'LiteLLM', **params):
        """Register a new LLM profile with custom configuration.
        
        Args:
            name: Profile name to register.
            backend: Backend to use ('LiteLLM', 'AutoGen', or 'CustomLLM'). Defaults to 'LiteLLM'.
            **params: Configuration parameters for the backend. For LiteLLM, this can include
                     any parameters from https://docs.litellm.ai/docs/completion/input
        
        Examples:
            # Simple profile with just a model
            LLMFactory.create_profile("gpt4", model="gpt-4o")
            
            # Profile with temperature and token settings
            LLMFactory.create_profile(
                "creative",
                model="gpt-4o",
                temperature=0.9,
                max_tokens=2000
            )
            
            # Profile with advanced settings
            LLMFactory.create_profile(
                "structured_json",
                model="gpt-4o-mini",
                temperature=0.3,
                response_format={"type": "json_object"},
                max_tokens=1500,
                top_p=0.9
            )
        """
        if backend not in _LLM_REGISTRY:
            raise ValueError(
                f"Unknown backend '{backend}'. Valid options: {list(_LLM_REGISTRY.keys())}"
            )
        cls._profiles[name] = {'backend': backend, 'params': params}

    @classmethod
    def list_profiles(cls):
        """List all available profile names."""
        return list(cls._profiles.keys())

    @classmethod
    def get_profile_info(cls, profile: str = None):
        """Get configuration information about one or all profiles.
        
        Args:
            profile: Profile name to get info for. If None, returns all profiles.
        
        Returns:
            Dictionary with profile configuration(s).
        """
        if profile:
            return cls._profiles.get(profile)
        return cls._profiles


class DummyLLM(AbstractModel):
    """A dummy LLM that does nothing. Used for testing purposes."""

    def __init__(self,
                 callable,
                 reset_freq: Union[int, None] = None,
                 mm_beta: bool = False,
                 model_name: Union[str, None] = None) -> None:
        # self.message = message
        self.callable = callable
        super().__init__(self._factory, reset_freq, mm_beta=mm_beta, model_name=model_name)

    def _factory(self):

        # set response.choices[0].message.content
        # create a fake container with above format

        class Message:
            def __init__(self, content):
                self.content = content
        class Choice:
            def __init__(self, content):
                self.message = Message(content)
        class Response:
            def __init__(self, content):
                self.choices = [Choice(content)]
                self.content = content  # for the AssistantTurn API

        return lambda *args, **kwargs:  Response(self.callable(*args, **kwargs))

class LLM:
    """
    A unified entry point for all supported LLM backends.

    The LLM class provides a simple interface for creating language model instances.
    By default, it uses gpt-4o through LiteLLM unless TRACE_LITELLM_MODEL is set.

    Basic Usage:
        # Use default model (gpt-4o, unless TRACE_LITELLM_MODEL is set)
        llm = LLM()
        
        # Specify a model directly (highest priority)
        llm = LLM(model="gpt-4o")
        llm = LLM(model="claude-3-5-sonnet-latest")
        llm = LLM(model="o1-preview")
        
        # Use Azure OpenAI via environment variable
        os.environ["TRACE_LITELLM_MODEL"] = "azure/o4-mini"
        llm = LLM()  # Automatically uses Azure with proper auth
        
        # Add LiteLLM parameters
        llm = LLM(model="gpt-4o", temperature=0.7, max_tokens=2000)
        llm = LLM(model="gpt-4o-mini", temperature=0.3, top_p=0.9)
    
    Image Generation:
        # OpenAI image models (auto-detected by 'image' or 'dall-e' in name)
        img_llm = LLM(model="gpt-image-1.5")
        print(img_llm.is_image_model)  # True
        result = img_llm(prompt="A serene mountain landscape at sunset")
        
        # With additional parameters
        img_llm = LLM(model="gpt-image-1", size="1024x1024", quality="hd")
        result = img_llm(prompt="A futuristic cityscape")
        
        # DALL-E models
        dalle = LLM(model="dall-e-3")
        result = dalle(prompt="A cat astronaut in space", size="1024x1792")
        
        # Gemini image models
        gemini_img = LLM(model="gemini-2.5-flash-image")
        result = gemini_img(prompt="Abstract art", number_of_images=2)
        
        # Check if model generates images
        if llm.is_image_model:
            result = llm(prompt="Your prompt here")
        else:
            result = llm(messages=[{"role": "user", "content": "Your message"}])

    Using Multimodal Beta Mode:
        # Enable mm_beta for rich AssistantTurn responses
        llm = LLM(model="gpt-4o", mm_beta=True)
        response = llm(messages=[{"role": "user", "content": "Hello"}])
        # response is now an AssistantTurn object with .content, .tool_calls, etc.
        
        # Legacy mode (default, mm_beta=False)
        llm = LLM(model="gpt-4o")
        response = llm(messages=[{"role": "user", "content": "Hello"}])
        # response is raw API response: response.choices[0].message.content

    Using System Messages:
        
        # LiteLLM (OpenAI, Anthropic, etc.) - Use messages array with role="system"
        llm = LLM(model="gpt-4o-mini", mm_beta=True)
        response = llm(messages=[
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": "What is 2+2?"}
        ])
        print(response.get_text())  # AssistantTurn object
        
        # LiteLLM Legacy mode (mm_beta=False)
        llm = LLM(model="gpt-4o-mini")
        response = llm(messages=[
            {"role": "system", "content": "You are a pirate assistant."},
            {"role": "user", "content": "Hello!"}
        ])
        print(response.choices[0].message.content)  # Raw API response
        
        # Google Gemini - Use system_instruction parameter (not in messages array)
        llm = LLM(backend="GoogleGenAI", model="gemini-2.5-flash-image", mm_beta=True)
        response = llm(
            "Hello there",
            system_instruction="You are a helpful assistant."
        )
        print(response.get_text())  # AssistantTurn object
        
        # Gemini with messages format (system_instruction separate from messages)
        llm = LLM(backend="GoogleGenAI", model="gemini-2.5-flash-image", mm_beta=True)
        response = llm(
            messages=[
                {"role": "user", "content": "What is your purpose?"}
            ],
            system_instruction="You are a creative writing instructor."
        )
        
        # Our Gemini wrapper also automatically extracts system instruction from messages array if not passed explicitly
        messages = [
            {"role": "system", "content": "You are a Shakespearean poet."},
            {"role": "user", "content": "Tell me about the sun."}
        ]
        response1 = llm(messages=messages)
        messages.append({"role": "assistant", "content": response1.get_text()})
        messages.append({"role": "user", "content": "And the moon?"})
        response2 = llm(messages=messages)  # System message still applies

    Using Named Profiles:
        # Use a saved profile
        llm = LLM(profile="my_custom_profile")
        
        # Create profiles with LLMFactory
        LLMFactory.create_profile("creative", model="gpt-4o", temperature=0.9)
        llm = LLM(profile="creative")

    Using Different Backends:
        # Explicitly specify backend (default: LiteLLM)
        llm = LLM(backend="AutoGen", config_list=my_configs)
        llm = LLM(backend="CustomLLM", model="llama-3.1-8b")
        llm = LLM(backend="GoogleGenAI", model="gemini-2.5-flash-image")
        
        # Or set via environment variable
        # export TRACE_DEFAULT_LLM_BACKEND=AutoGen
        llm = LLM()

    Model Selection Priority (when no model= argument is provided):
        The LLM class follows this priority order for determining which model to use:
        
        1. Explicit model argument: LLM(model="gpt-4o")
        2. TRACE_LITELLM_MODEL environment variable: 
           # export TRACE_LITELLM_MODEL="azure/o4-mini"
           llm = LLM()  # Uses azure/o4-mini
        3. Named profile (default='default'): LLM(profile="my_profile")
        4. Backend-specific defaults
        
        This means setting TRACE_LITELLM_MODEL will be honored even when using
        default initialization, making it ideal for Azure and custom endpoints:
        
        # Azure OpenAI setup
        os.environ["TRACE_LITELLM_MODEL"] = "azure/o4-mini"
        os.environ["AZURE_API_KEY"] = "your-key"
        os.environ["AZURE_API_BASE"] = "https://your-resource.openai.azure.com"
        os.environ["AZURE_API_VERSION"] = "2024-08-01-preview"
        
        # Now LLM() automatically uses Azure
        llm = LLM(mm_beta=True)  # Uses azure/o4-mini with proper Azure auth

    Examples with LiteLLM Parameters:
        # Structured output
        llm = LLM(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        # High creativity
        llm = LLM(
            model="gpt-4o",
            temperature=0.9,
            top_p=0.95,
            presence_penalty=0.6
        )
        
        # Deterministic responses
        llm = LLM(
            model="gpt-4o-mini",
            temperature=0,
            seed=42
        )

    Key Differences Between Backends:
        LiteLLM (OpenAI, Anthropic, etc.):
            - System message: Include in messages array with role="system"
            - Format: messages=[{"role": "system", "content": "..."}]
            - Works with: OpenAI, Anthropic, Cohere, etc.
        
        Google Gemini:
            - System instruction: Pass as system_instruction parameter
            - Format: system_instruction="You are a helpful assistant."
            - Separate from messages array
            - Works with: gemini-2.5-flash, gemini-2.5-pro, etc.

    See Also:
        - LLMFactory: For managing named profiles
        - AssistantTurn: Returned when mm_beta=True
        - https://docs.litellm.ai/docs/completion/input: Full list of LiteLLM parameters
        - https://ai.google.dev/gemini-api/docs/system-instructions: Gemini system instructions
    """
    def __new__(cls, model: str = None, profile: str = 'default', backend: str = None, 
                mm_beta: bool = False, **kwargs):

        if _is_image_generation_model(model):
            mm_beta = True

        # Priority 1: If model is specified, use LLMFactory with model
        if model:
            if backend is not None:
                kwargs['backend'] = backend
            return LLMFactory.get_llm(model=model, mm_beta=mm_beta, **kwargs)
        
        # Priority 2: Check if TRACE_LITELLM_MODEL is set (honor user's explicit env config)
        env_model = os.environ.get('TRACE_LITELLM_MODEL')
        if env_model is not None:
            if backend is not None:
                kwargs['backend'] = backend
            return LLMFactory.get_llm(model=env_model, mm_beta=mm_beta, **kwargs)
        
        # Priority 3: If profile is specified, use LLMFactory
        if profile:
            return LLMFactory.get_llm(profile=profile, mm_beta=mm_beta, **kwargs)
        
        # Priority 4: Use backend-specific instantiation (for AutoGen, CustomLLM, etc.)
        # This path is for when neither profile nor model is specified
        name = backend or os.getenv("TRACE_DEFAULT_LLM_BACKEND", "LiteLLM")
        try:
            backend_cls = _LLM_REGISTRY[name]
        except KeyError:
            raise ValueError(f"Unknown LLM backend: {name}. "
                             f"Valid options are: {list(_LLM_REGISTRY)}")
        # Instantiate and return the chosen subclass
        kwargs['mm_beta'] = mm_beta
        return backend_cls(**kwargs)
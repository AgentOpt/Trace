from typing import List, Tuple, Dict, Any, Callable, Union
import os
import time
import json
import os
import warnings

try:
    import autogen  # We import autogen here to avoid the need of installing autogen
except ImportError:
    pass

class AbstractModel:
    """
    A minimal abstraction of a model api that refreshes the model every
    reset_freq seconds (this is useful for long-running models that may require
    refreshing certificates or memory management).
    """

    def __init__(self, factory: Callable, reset_freq: Union[int, None] = None) -> None:
        """
        Args:
            factory: A function that takes no arguments and returns a model that is callable.
            reset_freq: The number of seconds after which the model should be
                refreshed. If None, the model is never refreshed.
        """
        self.factory = factory
        self._model = self.factory()
        self.reset_freq = reset_freq
        self._init_time = time.time()

    # Overwrite this `model` property when subclassing.
    @property
    def model(self):
        """ When self.model is called, text responses should always be available at ['choices'][0].['message']['content'] """
        return self._model

    # This is the main API
    def __call__(self, *args, **kwargs) -> Any:
        """ The call function handles refreshing the model if needed.
        """
        if self.reset_freq is not None and time.time() - self._init_time > self.reset_freq:
            self._model = self.factory()
            self._init_time = time.time()
        return self.model(*args, **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._model = self.factory()


class AutoGenLLM(AbstractModel):
    """ This is the main class Trace uses to interact with the model. It is a
    wrapper around autogen's OpenAIWrapper. For using models not supported by
    autogen, subclass AutoGenLLM and override the `_factory` and  `create`
    method. Users can pass instances of this class to optimizers' llm argument.
    """

    def __init__(self, config_list: List = None, filter_dict: Dict = None, reset_freq: Union[int, None] = None) -> None:
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

        factory = lambda *args, **kwargs: self._factory(config_list)
        super().__init__(factory, reset_freq)

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

    To use this, set the credentials through the environment variable as
    instructed in the LiteLLM documentation. For convenience, you can set the
    default model name through the environment variable TRACE_LITELLM_MODEL.
    When using Azure models via token provider, you can set the Azure token
    provider scope through the environment variable AZURE_TOKEN_PROVIDER_SCOPE.
    """

    def __init__(self, model: Union[str, None] = None, reset_freq: Union[int, None] = None,
                 cache=True) -> None:
        if model is None:
            model = os.environ.get('TRACE_LITELLM_MODEL')
            if model is None:
                # warnings.warn("TRACE_LITELLM_MODEL environment variable is not found when loading the default model for LiteLLM. Attempt to load the default model from DEFAULT_LITELLM_MODEL environment variable. The usage of DEFAULT_LITELLM_MODEL will be deprecated. Please use the environment variable TRACE_LITELLM_MODEL for setting the default model name for LiteLLM.")
                model = os.environ.get('DEFAULT_LITELLM_MODEL', 'gpt-4o')

        self.model_name = model
        self.cache = cache
        factory = lambda: self._factory(self.model_name)  # an LLM instance uses a fixed model
        super().__init__(factory, reset_freq)

    @classmethod
    def _factory(cls, model_name: str):
        import litellm
        if model_name.startswith('azure/'):  # azure model
            azure_token_provider_scope = os.environ.get('AZURE_TOKEN_PROVIDER_SCOPE', None)
            if azure_token_provider_scope is not None:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider
                credential = get_bearer_token_provider(DefaultAzureCredential(), azure_token_provider_scope)
                return lambda *args, **kwargs: litellm.completion(model_name, *args,
                                                                  azure_ad_token_provider=credential, **kwargs)
        return lambda *args, **kwargs: litellm.completion(model_name, *args, **kwargs)

    @property
    def model(self):
        """
        response = litellm.completion(
            model=self.model,
            messages=[{"content": message, "role": "user"}]
        )
        """
        return lambda *args, **kwargs: self._model(*args, **kwargs)


class CustomLLM(AbstractModel):
    """
    This is for Custom server's API endpoints that are OpenAI Compatible.
    Such server includes LiteLLM proxy server.
    """

    def __init__(self, model: Union[str, None] = None, reset_freq: Union[int, None] = None,
                 cache=True) -> None:
        if model is None:
            model = os.environ.get('TRACE_CUSTOMLLM_MODEL', 'gpt-4o')
        base_url = os.environ.get('TRACE_CUSTOMLLM_URL', 'http://xx.xx.xxx.xx:4000/')
        server_api_key = os.environ.get('TRACE_CUSTOMLLM_API_KEY',
                                        'sk-Xhg...')  # we assume the server has an API key
        # the server API is set through `master_key` in `config.yaml` for LiteLLM proxy server
        
        self.model_name = model
        self.cache = cache
        factory = lambda: self._factory(base_url, server_api_key)  # an LLM instance uses a fixed model
        super().__init__(factory, reset_freq)

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

# Registry of available backends
_LLM_REGISTRY = {
    "LiteLLM": LiteLLM,
    "AutoGen": AutoGenLLM,
    "CustomLLM": CustomLLM,
}

class LLMFactory:
    """Factory for creating LLM instances with predefined profiles.
    
    The code comes with these built-in profiles:

        llm_default = LLM(profile="default")     # gpt-4o-mini
        llm_premium = LLM(profile="premium")     # gpt-4  
        llm_cheap = LLM(profile="cheap")         # gpt-4o-mini
        llm_fast = LLM(profile="fast")           # gpt-3.5-turbo-mini
        llm_reasoning = LLM(profile="reasoning") # o1-mini
    
    You can override those built-in profiles:

        LLMFactory.register_profile("default", "LiteLLM", model="gpt-4o", temperature=0.5)
        LLMFactory.register_profile("premium", "LiteLLM", model="o1-preview", max_tokens=8000)
        LLMFactory.register_profile("cheap", "LiteLLM", model="gpt-3.5-turbo", temperature=0.9)
        LLMFactory.register_profile("fast", "LiteLLM", model="gpt-3.5-turbo", max_tokens=500)
        LLMFactory.register_profile("reasoning", "LiteLLM", model="o1-preview")
        
    An Example of using Different Backends

        # Register custom profiles for different use cases
        LLMFactory.register_profile("advanced_reasoning", "LiteLLM", model="o1-preview", max_tokens=4000)
        LLMFactory.register_profile("claude_sonnet", "LiteLLM", model="claude-3-5-sonnet-latest", temperature=0.3)
        LLMFactory.register_profile("custom_server", "CustomLLM", model="llama-3.1-8b")

        # Use in different contexts
        reasoning_llm = LLM(profile="advanced_reasoning")  # For complex reasoning
        claude_llm = LLM(profile="claude_sonnet")          # For Claude responses
        local_llm = LLM(profile="custom_server")           # For local deployment

        # Single LLM optimizer with custom profile
        optimizer1 = OptoPrime(parameters, llm=LLM(profile="advanced_reasoning"))

        # Multi-LLM optimizer with multiple profiles
        optimizer2 = OptoPrimeMulti(parameters, llm_profiles=["cheap", "premium", "claude_sonnet"], generation_technique="multi_llm")
    """
    
    # Default profiles for different use cases
    _profiles = {
        'default': {'backend': 'LiteLLM', 'params': {'model': 'gpt-4o-mini'}},
        'premium': {'backend': 'LiteLLM', 'params': {'model': 'gpt-4'}},
        'cheap': {'backend': 'LiteLLM', 'params': {'model': 'gpt-4o-mini'}},
        'fast': {'backend': 'LiteLLM', 'params': {'model': 'gpt-3.5-turbo-mini'}},
        'reasoning': {'backend': 'LiteLLM', 'params': {'model': 'o1-mini'}},
    }
    
    @classmethod
    def get_llm(cls, profile: str = 'default') -> AbstractModel:
        """Get an LLM instance for the specified profile."""
        if profile not in cls._profiles:
            raise ValueError(f"Unknown profile '{profile}'. Available profiles: {list(cls._profiles.keys())}")
        
        config = cls._profiles[profile]
        backend_cls = _LLM_REGISTRY[config['backend']]
        return backend_cls(**config['params'])
    
    @classmethod
    def register_profile(cls, name: str, backend: str, **params):
        """Register a new LLM profile."""
        cls._profiles[name] = {'backend': backend, 'params': params}
    
    @classmethod
    def list_profiles(cls):
        """List all available profiles."""
        return list(cls._profiles.keys())
    
    @classmethod
    def get_profile_info(cls, profile: str = None):
        """Get information about a profile or all profiles."""
        if profile:
            return cls._profiles.get(profile)
        return cls._profiles


class DummyLLM(AbstractModel):
    """A dummy LLM that does nothing. Used for testing purposes."""
    
    def __init__(self, 
                 callable,
                 reset_freq: Union[int, None] = None) -> None:
        # self.message = message
        self.callable = callable
        factory = lambda: self._factory()
        super().__init__(factory, reset_freq)

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

        return lambda *args, **kwargs:  Response(self.callable(*args, **kwargs))


class LLM:
    """
    A unified entry point for all supported LLM backends.
    
    Usage:
      # pick by env var (default: LiteLLM)
      llm = LLM()
      # or override explicitly
      llm = LLM(backend="AutoGen", config_list=my_configs)
      # or use predefined profiles
      llm = LLM(profile="premium")  # Use premium model
      llm = LLM(profile="cheap")    # Use cheaper model
      llm = LLM(profile="reasoning")    # Use reasoning/thinking model
    """
    def __new__(cls, *args, profile: str = None, backend: str = None, **kwargs):
        # New: if profile is specified, use LLMFactory
        if profile:
            return LLMFactory.get_llm(profile)
        # Decide which backend to use
        name = backend or os.getenv("TRACE_DEFAULT_LLM_BACKEND", "LiteLLM")
        try:
            backend_cls = _LLM_REGISTRY[name]
        except KeyError:
            raise ValueError(f"Unknown LLM backend: {name}. "
                             f"Valid options are: {list(_LLM_REGISTRY)}")
        # Instantiate and return the chosen subclass
        return backend_cls(*args, **kwargs)
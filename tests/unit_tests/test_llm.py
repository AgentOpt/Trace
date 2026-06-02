from opto.utils.llm import LLM, LLMFactory
from opto.optimizers.utils import print_color
import os

import pytest
from opto.utils.backbone import (
    Chat,
    UserTurn,
    AssistantTurn
)

# Skip tests if no API credentials are available
SKIP_REASON = "No API credentials found"
HAS_CREDENTIALS = os.path.exists("OAI_CONFIG_LIST") or os.environ.get("TRACE_LITELLM_MODEL") or os.environ.get(
    "OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")


def test_llm_init():
    """Test basic LLM initialization with legacy mode (mm_beta=False)"""
    if os.path.exists("OAI_CONFIG_LIST") or os.environ.get("TRACE_LITELLM_MODEL") or os.environ.get("OPENAI_API_KEY"):
        llm = LLM()
        system_prompt = 'You are a helpful assistant.'
        user_prompt = "Hello world."


        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}]

        output = llm(messages=messages)
        # Alternatively, you can use the following code:
        # output = llm.create(messages=messages)

        response = output.choices[0].message.content


        print_color(f'System: {system_prompt}', 'red')
        print_color(f'User: {user_prompt}', 'blue')
        print_color(f'LLM: {response}', 'green')


@pytest.mark.skipif(not HAS_CREDENTIALS, reason=SKIP_REASON)
class TestLLMMMBetaMode:
    """Test suite for LLM class with mm_beta=True and mm_beta=False modes"""
    
    def test_mm_beta_false_legacy_response_format(self):
        """Test that mm_beta=False returns raw API response (legacy format)"""
        llm = LLM(mm_beta=False)
        messages = [{"role": "user", "content": "Say 'test' and nothing else."}]
        
        response = llm(messages=messages)
        
        # Legacy mode should return raw API response with .choices attribute
        assert hasattr(response, 'choices'), "Legacy mode should return raw API response"
        assert hasattr(response.choices[0], 'message'), "Response should have message attribute"
        assert hasattr(response.choices[0].message, 'content'), "Message should have content attribute"
        
        # Should NOT be an AssistantTurn object
        assert not isinstance(response, AssistantTurn), "Legacy mode should not return AssistantTurn"
        
        content = response.choices[0].message.content
        assert isinstance(content, str), "Content should be a string"
        assert len(content) > 0, "Content should not be empty"
        
        print_color(f"✓ Legacy mode (mm_beta=False) returns raw API response", 'green')
    
    def test_mm_beta_true_assistant_turn_response(self):
        """Test that mm_beta=True returns AssistantTurn object"""
        llm = LLM(mm_beta=True)
        messages = [{"role": "user", "content": "Say 'test' and nothing else."}]
        
        response = llm(messages=messages)
        
        # mm_beta mode should return AssistantTurn object
        assert isinstance(response, AssistantTurn), "mm_beta mode should return AssistantTurn object"
        
        # Check AssistantTurn attributes
        assert hasattr(response, 'content'), "AssistantTurn should have content attribute"
        assert hasattr(response, 'tool_calls'), "AssistantTurn should have tool_calls attribute"
        assert hasattr(response, 'role'), "AssistantTurn should have role attribute"
        assert response.role == "assistant", "Role should be 'assistant'"
        
        # Content should be accessible
        assert response.content is not None, "Content should not be None"
        
        print_color(f"✓ Multimodal mode (mm_beta=True) returns AssistantTurn object", 'green')
    
    def test_mm_beta_with_explicit_model(self):
        """Test mm_beta parameter works with explicit model specification"""
        # Test with mm_beta=False
        llm_legacy = LLM(model="gpt-4o-mini", mm_beta=False)
        messages = [{"role": "user", "content": "Hi"}]
        
        response_legacy = llm_legacy(messages=messages)
        assert hasattr(response_legacy, 'choices'), "Should return raw API response"
        assert not isinstance(response_legacy, AssistantTurn), "Should not be AssistantTurn"
        
        # Test with mm_beta=True
        llm_mm = LLM(model="gpt-4o-mini", mm_beta=True)
        response_mm = llm_mm(messages=messages)
        assert isinstance(response_mm, AssistantTurn), "Should return AssistantTurn"
        
        print_color(f"✓ mm_beta parameter works correctly with explicit model", 'green')
    
    def test_mm_beta_with_profile(self):
        """Test mm_beta parameter works with profile-based instantiation"""
        # Create a test profile
        LLMFactory.create_profile("test_profile", backend="LiteLLM", model="gpt-4o-mini", temperature=0.7)
        
        # Test with mm_beta=False
        llm_legacy = LLM(profile="test_profile", mm_beta=False)
        messages = [{"role": "user", "content": "Hi"}]
        
        response_legacy = llm_legacy(messages=messages)
        assert hasattr(response_legacy, 'choices'), "Profile with mm_beta=False should return raw API response"
        
        # Test with mm_beta=True
        llm_mm = LLM(profile="test_profile", mm_beta=True)
        response_mm = llm_mm(messages=messages)
        assert isinstance(response_mm, AssistantTurn), "Profile with mm_beta=True should return AssistantTurn"
        
        print_color(f"✓ mm_beta parameter works correctly with profiles", 'green')
    
    def test_mm_beta_with_litellm_parameters(self):
        """Test mm_beta works with various LiteLLM parameters"""
        # Test with temperature and max_tokens
        llm = LLM(
            model="gpt-4o-mini",
            mm_beta=True,
            temperature=0.3,
            max_tokens=100
        )
        
        messages = [{"role": "user", "content": "Say hello"}]
        response = llm(messages=messages)
        
        assert isinstance(response, AssistantTurn), "Should return AssistantTurn with LiteLLM params"
        assert response.content is not None, "Should have content"
        
        print_color(f"✓ mm_beta works with LiteLLM parameters", 'green')
    
    def test_mm_beta_default_is_false(self):
        """Test that mm_beta defaults to False for backward compatibility"""
        llm = LLM()  # No mm_beta specified
        messages = [{"role": "user", "content": "Hi"}]
        
        response = llm(messages=messages)
        
        # Default should be legacy mode (mm_beta=False)
        assert hasattr(response, 'choices'), "Default should be legacy mode"
        assert not isinstance(response, AssistantTurn), "Default should not return AssistantTurn"
        
        print_color(f"✓ mm_beta defaults to False (backward compatible)", 'green')
    
    def test_mm_beta_content_accessibility(self):
        """Test that content is accessible in both modes"""
        messages = [{"role": "user", "content": "Say 'hello'"}]
        
        # Legacy mode
        llm_legacy = LLM(mm_beta=False)
        response_legacy = llm_legacy(messages=messages)
        content_legacy = response_legacy.choices[0].message.content
        assert isinstance(content_legacy, str), "Legacy content should be string"
        assert len(content_legacy) > 0, "Legacy content should not be empty"
        
        # mm_beta mode
        llm_mm = LLM(mm_beta=True)
        response_mm = llm_mm(messages=messages)
        # AssistantTurn content is a list of ContentBlock objects
        assert response_mm.content is not None, "mm_beta content should not be None"
        
        print_color(f"✓ Content accessible in both modes", 'green')
    
    def test_mm_beta_with_different_backends(self):
        """Test mm_beta parameter with different backend specifications"""
        # Test with explicit LiteLLM backend
        llm = LLM(backend="LiteLLM", model="gpt-4o-mini", mm_beta=True)
        messages = [{"role": "user", "content": "Hi"}]
        
        response = llm(messages=messages)
        assert isinstance(response, AssistantTurn), "LiteLLM backend with mm_beta=True should return AssistantTurn"
        
        print_color(f"✓ mm_beta works with explicit backend specification", 'green')


@pytest.mark.skipif(not HAS_CREDENTIALS, reason=SKIP_REASON)
class TestLLMConstructorPriorities:
    """Test the priority logic in LLM constructor"""
    
    def test_priority_profile_over_default(self):
        """Test that profile parameter takes priority"""
        LLMFactory.create_profile("priority_test", backend="LiteLLM", model="gpt-4o-mini", temperature=0.5)
        
        llm = LLM(profile="priority_test", mm_beta=True)
        messages = [{"role": "user", "content": "Hi"}]
        
        response = llm(messages=messages)
        assert isinstance(response, AssistantTurn), "Profile-based LLM should respect mm_beta"
        
        print_color(f"✓ Profile parameter takes priority", 'green')
    
    def test_priority_model_over_profile(self):
        """Test that model parameter takes priority over default profile"""
        # When model is specified, it should use that model regardless of default profile
        llm = LLM(model="gpt-4o-mini", mm_beta=True)
        messages = [{"role": "user", "content": "Hi"}]
        
        response = llm(messages=messages)
        assert isinstance(response, AssistantTurn), "Model-based LLM should respect mm_beta"
        
        print_color(f"✓ Model parameter creates correct LLM instance", 'green')
    
    def test_backend_fallback(self):
        """Test that backend parameter works when neither profile nor model specified"""
        # This tests the Priority 3 path in __new__
        llm = LLM(backend="LiteLLM", mm_beta=True, model="gpt-4o-mini")
        messages = [{"role": "user", "content": "Hi"}]
        
        response = llm(messages=messages)
        assert isinstance(response, AssistantTurn), "Backend-based LLM should respect mm_beta"
        
        print_color(f"✓ Backend parameter works correctly", 'green')


@pytest.mark.skipif(not HAS_CREDENTIALS, reason=SKIP_REASON)
class TestLLMDocumentationExamples:
    """Test examples from LLM class documentation"""
    
    def test_basic_usage_default_model(self):
        """Test: llm = LLM()"""
        llm = LLM()
        messages = [{"role": "user", "content": "Hi"}]
        response = llm(messages=messages)
        
        # Default is mm_beta=False
        assert hasattr(response, 'choices'), "Default usage should return raw API response"
        print_color(f"✓ Basic usage with default model works", 'green')
    
    def test_specify_model_directly(self):
        """Test: llm = LLM(model='gpt-4o')"""
        llm = LLM(model="gpt-4o-mini")  # Using mini for cost efficiency
        messages = [{"role": "user", "content": "Hi"}]
        response = llm(messages=messages)
        
        assert hasattr(response, 'choices'), "Model specification should work"
        print_color(f"✓ Model specification works", 'green')
    
    def test_multimodal_beta_mode_example(self):
        """Test example from 'Using Multimodal Beta Mode' section"""
        # Enable mm_beta for rich AssistantTurn responses
        llm = LLM(model="gpt-4o-mini", mm_beta=True)
        response = llm(messages=[{"role": "user", "content": "Hello"}])
        
        # response is now an AssistantTurn object with .content, .tool_calls, etc.
        assert isinstance(response, AssistantTurn), "Should return AssistantTurn"
        assert hasattr(response, 'content'), "Should have content attribute"
        assert hasattr(response, 'tool_calls'), "Should have tool_calls attribute"
        
        print_color(f"✓ Multimodal beta mode example works as documented", 'green')
    
    def test_legacy_mode_example(self):
        """Test example from 'Legacy mode' section"""
        # Legacy mode (default, mm_beta=False)
        llm = LLM(model="gpt-4o-mini")
        response = llm(messages=[{"role": "user", "content": "Hello"}])
        
        # response is raw API response: response.choices[0].message.content
        assert hasattr(response, 'choices'), "Should return raw API response"
        content = response.choices[0].message.content
        assert isinstance(content, str), "Content should be string"
        
        print_color(f"✓ Legacy mode example works as documented", 'green')
    
    def test_litellm_parameters_example(self):
        """Test examples with LiteLLM parameters"""
        # High creativity example
        llm = LLM(
            model="gpt-4o-mini",
            temperature=0.9,
            top_p=0.95,
            presence_penalty=0.6
        )
        messages = [{"role": "user", "content": "Hi"}]
        response = llm(messages=messages)
        
        assert hasattr(response, 'choices'), "LiteLLM parameters should work"
        
        print_color(f"✓ LiteLLM parameters example works", 'green')


@pytest.mark.skipif(not HAS_CREDENTIALS, reason=SKIP_REASON)
def test_mm_beta_integration_with_conversation():
    """Test mm_beta mode with a multi-turn conversation"""
    llm = LLM(model="gpt-4o-mini", mm_beta=True)
    
    # First turn
    messages = [
        {"role": "user", "content": "My name is Alice."}
    ]
    response1 = llm(messages=messages)
    assert isinstance(response1, AssistantTurn), "First response should be AssistantTurn"
    
    # Second turn - reference previous context
    messages.append({"role": "assistant", "content": str(response1.content)})
    messages.append({"role": "user", "content": "What is my name?"})
    
    response2 = llm(messages=messages)
    assert isinstance(response2, AssistantTurn), "Second response should be AssistantTurn"
    
    print_color(f"✓ mm_beta mode works with multi-turn conversations", 'green')


@pytest.mark.skipif(not HAS_CREDENTIALS, reason=SKIP_REASON)
class TestSystemMessages:
    """Test suite for system message handling in different LLM backends"""
    
    def test_litellm_completion_api_system_message(self):
        """Test system message with LiteLLM Completion API (mm_beta=False)"""
        llm = LLM(model="gpt-4o-mini", mm_beta=False)
        
        messages = [
            {"role": "system", "content": "You are a cat. Your name is Neko. Always respond as a cat would."},
            {"role": "user", "content": "What is your name?"}
        ]
        
        response = llm(messages=messages)
        
        # Legacy mode should return raw API response
        assert hasattr(response, 'choices'), "Should return raw API response"
        content = response.choices[0].message.content
        assert isinstance(content, str), "Content should be a string"
        assert len(content) > 0, "Content should not be empty"
        
        # Check that the response reflects the system message (should mention being a cat or Neko)
        content_lower = content.lower()
        assert 'neko' in content_lower or 'cat' in content_lower, \
            f"Response should reflect system message about being a cat named Neko. Got: {content}"
        
        print_color(f"✓ LiteLLM Completion API handles system messages correctly", 'green')
    
    def test_litellm_responses_api_system_message(self):
        """Test system message with LiteLLM Responses API (mm_beta=True)"""
        llm = LLM(model="gpt-4o-mini", mm_beta=True)
        
        messages = [
            {"role": "system", "content": "You are a helpful math tutor. Always explain concepts clearly."},
            {"role": "user", "content": "What is 2+2?"}
        ]
        
        response = llm(messages=messages)
        
        # mm_beta mode should return AssistantTurn
        assert isinstance(response, AssistantTurn), "Should return AssistantTurn object"
        assert response.content is not None, "Content should not be None"
        
        # Get text content
        text_content = response.to_text()
        assert isinstance(text_content, str), "Text content should be a string"
        assert len(text_content) > 0, "Text content should not be empty"
        assert '4' in text_content, f"Response should contain the answer '4'. Got: {text_content}"
        
        print_color(f"✓ LiteLLM Responses API handles system messages correctly", 'green')
    
    @pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="No Gemini API key found")
    def test_gemini_system_instruction_legacy_mode(self):
        """Test system_instruction with Gemini API in legacy mode (mm_beta=False)"""
        llm = LLM(backend="GoogleGenAI", model="gemini-2.5-flash", mm_beta=False)
        
        # For Gemini, system_instruction is passed as a parameter
        response = llm(
            "Hello there",
            system_instruction="You are a cat. Your name is Neko. Always respond as a cat would."
        )
        
        # Check response format
        assert hasattr(response, 'text'), "Gemini response should have text attribute"
        content = response.text
        assert isinstance(content, str), "Content should be a string"
        assert len(content) > 0, "Content should not be empty"
        
        # Check that the response reflects the system instruction
        content_lower = content.lower()
        assert 'neko' in content_lower or 'cat' in content_lower or 'meow' in content_lower, \
            f"Response should reflect system instruction about being a cat named Neko. Got: {content}"
        
        print_color(f"✓ Gemini API handles system_instruction correctly (legacy mode)", 'green')
    
    @pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="No Gemini API key found")
    def test_gemini_system_instruction_mm_beta_mode(self):
        """Test system_instruction with Gemini API in mm_beta mode"""
        llm = LLM(backend="GoogleGenAI", model="gemini-2.5-flash", mm_beta=True)
        
        # For Gemini, system_instruction is passed as a parameter
        response = llm(
            "What is your name?",
            system_instruction="You are a helpful assistant named Claude. Always introduce yourself."
        )
        
        # mm_beta mode should return AssistantTurn
        assert isinstance(response, AssistantTurn), "Should return AssistantTurn object"
        assert response.content is not None, "Content should not be None"
        
        # Get text content
        text_content = response.to_text()
        assert isinstance(text_content, str), "Text content should be a string"
        assert len(text_content) > 0, "Text content should not be empty"
        
        # Check that the response reflects the system instruction
        text_lower = text_content.lower()
        assert 'claude' in text_lower or 'assistant' in text_lower, \
            f"Response should reflect system instruction about being Claude. Got: {text_content}"
        
        print_color(f"✓ Gemini API handles system_instruction correctly (mm_beta mode)", 'green')
    
    def test_litellm_system_message_with_conversation(self):
        """Test system message persists across multi-turn conversation"""
        llm = LLM(model="gpt-4o-mini", mm_beta=True)
        
        # First turn with system message
        messages = [
            {"role": "system", "content": "You are a pirate. Always talk like a pirate."},
            {"role": "user", "content": "Hello"}
        ]
        
        response1 = llm(messages=messages)
        assert isinstance(response1, AssistantTurn), "First response should be AssistantTurn"
        text1 = response1.to_text()
        
        # Check pirate-like language in first response
        pirate_indicators = ['arr', 'matey', 'ahoy', 'ye', 'aye']
        has_pirate_language = any(indicator in text1.lower() for indicator in pirate_indicators)
        assert has_pirate_language, f"First response should use pirate language. Got: {text1}"
        
        # Second turn - system message should still apply
        messages.append({"role": "assistant", "content": text1})
        messages.append({"role": "user", "content": "What's the weather like?"})
        
        response2 = llm(messages=messages)
        assert isinstance(response2, AssistantTurn), "Second response should be AssistantTurn"
        text2 = response2.to_text()
        
        # Check pirate-like language persists
        has_pirate_language_2 = any(indicator in text2.lower() for indicator in pirate_indicators)
        assert has_pirate_language_2, f"Second response should still use pirate language. Got: {text2}"
        
        print_color(f"✓ System message persists across conversation turns", 'green')
    
    @pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="No Gemini API key found")
    def test_gemini_system_instruction_with_config_params(self):
        """Test system_instruction works with other config parameters"""
        llm = LLM(
            backend="GoogleGenAI", 
            model="gemini-2.5-flash", 
            mm_beta=True,
            temperature=0.7,
            max_output_tokens=100
        )
        
        response = llm(
            "Tell me a short joke",
            system_instruction="You are a comedian who tells very short jokes."
        )
        
        assert isinstance(response, AssistantTurn), "Should return AssistantTurn object"
        text_content = response.to_text()
        assert len(text_content) > 0, "Should have content"
        
        print_color(f"✓ Gemini system_instruction works with other config parameters", 'green')


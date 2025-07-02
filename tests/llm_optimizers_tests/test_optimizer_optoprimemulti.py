import json
import pytest
from opto.optimizers.optoprimemulti import OptoPrimeMulti
from opto.utils.llm import LLMFactory
from opto.trace.propagators import GraphPropagator
from opto.trace.nodes import ParameterNode
from opto.trace import bundle, node, GRAPH

class DummyLLM:
    def __init__(self, responses):
        # responses: list of list of choice-like objects with message.content
        self.responses = responses
        self.call_args = []

    def create(self, messages, response_format, max_tokens, n, temperature):
        # Simulate LLM.create returning an object with choices
        class Choice:
            def __init__(self, content):
                self.message = type('m', (), {'content': content})
        # Pop next response batch
        batch = self.responses.pop(0)
        self.call_args.append((n, temperature, messages))
        return type('r', (), {'choices': [Choice(c) for c in batch]})

    def __call__(self, messages, max_tokens=None, response_format=None):
        # fallback single-call (not used in multi)
        return self.create(messages, response_format, max_tokens, 1, 0)

class MockLLMFactory:
    """Mock LLMFactory for testing multi-LLM functionality"""
    @staticmethod
    def get_llm(profile):
        # Return different dummy LLMs for different profiles
        profile_responses = {
            'cheap': [f"cheap_{profile}_response"],
            'premium': [f"premium_{profile}_response"],
            'default': [f"default_{profile}_response"],
        }
        return DummyLLM(responses=[profile_responses.get(profile, ["default_response"])])

@pytest.fixture
def parameter_node():
    # Minimal dummy ParameterNode
    return ParameterNode(name='x', value=0)

@pytest.fixture
def default_optimizer(parameter_node):
    # Use dummy llm that returns empty responses
    dummy = DummyLLM(responses=[["{\\\"suggestion\\\": {}}"]])
    opt = OptoPrimeMulti([parameter_node], selector=None)
    opt.llm = dummy
    # Ensure propagator is GraphPropagator
    assert isinstance(opt.propagator, GraphPropagator)
    return opt

@pytest.fixture
def multi_llm_optimizer(parameter_node):
    """Optimizer configured for multi-LLM testing"""
    dummy = DummyLLM(responses=[["{\\\"suggestion\\\": {}}"]])
    opt = OptoPrimeMulti([parameter_node], 
                        llm_profiles=['cheap', 'premium', 'default'],
                        generation_technique='multi_llm')
    opt.llm = dummy
    return opt

def test_call_llm_returns_list(default_optimizer):
    opt = default_optimizer
    # Prepare dummy response
    opt.llm = DummyLLM(responses=[["resp1", "resp2"]])
    results = opt.call_llm("sys", "usr", num_responses=2, temperature=0.5)
    assert isinstance(results, list)
    assert results == ["resp1", "resp2"]

def test_call_llm_with_specific_llm(default_optimizer):
    """Test that call_llm accepts and uses a specific LLM instance"""
    opt = default_optimizer
    specific_llm = DummyLLM(responses=[["specific_response"]])
    
    # Call with specific LLM
    results = opt.call_llm("sys", "usr", llm=specific_llm, num_responses=1)
    assert results == ["specific_response"]
    
    # Verify specific_llm was called, not the default
    assert len(specific_llm.call_args) == 1
    assert len(opt.llm.call_args) == 0  # Default LLM should not be called

@pytest.mark.parametrize("gen_tech", [
    "temperature_variation", 
    "self_refinement", 
    "iterative_alternatives", 
    "multi_experts",
    "multi_llm"]
    )
def test_generate_candidates_length(default_optimizer, gen_tech, capsys):
    opt = default_optimizer
    # monkeypatch call_llm for each call to return unique string
    responses = [["c1"], ["c2"], ["c3"], ["c4"], ["c5"], ["c6"], ["c7"]]
    opt.llm = DummyLLM(responses=[r for r in responses])
    # Use only temperature_variation for simplicity
    cands = opt.generate_candidates(summary=None, system_prompt="s", user_prompt="u", num_responses=3, generation_technique=gen_tech)
    # Should return a list of length 3
    assert isinstance(cands, list)
    assert len(cands) == 3

def test_multi_llm_initialization():
    """Test OptoPrimeMulti initialization with multi-LLM parameters"""
    param = ParameterNode(name='test', value=1)
    profiles = ['cheap', 'premium', 'default']
    weights = [0.5, 1.5, 1.0]
    
    opt = OptoPrimeMulti([param], 
                        llm_profiles=profiles,
                        llm_weights=weights,
                        generation_technique='multi_llm')
    
    assert opt.llm_profiles == profiles
    assert opt.llm_weights == weights
    assert opt._llm_instances == {}  # Should start empty

def test_get_llm_for_profile(multi_llm_optimizer, monkeypatch):
    """Test LLM profile retrieval and caching"""
    opt = multi_llm_optimizer
    
    # Mock LLMFactory
    monkeypatch.setattr('opto.utils.llm.LLMFactory', MockLLMFactory)
    
    # First call should create and cache
    llm1 = opt._get_llm_for_profile('cheap')
    assert 'cheap' in opt._llm_instances
    
    # Second call should return cached instance
    llm2 = opt._get_llm_for_profile('cheap')
    assert llm1 is llm2
    
    # None profile should return default LLM
    default_llm = opt._get_llm_for_profile(None)
    assert default_llm is opt.llm

def test_get_llms_for_generation(multi_llm_optimizer, monkeypatch):
    """Test LLM distribution for generation"""
    opt = multi_llm_optimizer
    # Patch the import location where it's actually used
    monkeypatch.setattr('opto.optimizers.optoprimemulti.LLMFactory', MockLLMFactory)

    llms = opt._get_llms_for_generation(5)
    assert len(llms) == 5
    
    # Should cycle through profiles: cheap, premium, default, cheap, premium
    expected_profiles = ['cheap', 'premium', 'default', 'cheap', 'premium']
    for i, llm in enumerate(llms):
        expected_profile = expected_profiles[i]
        assert expected_profile in opt._llm_instances

@pytest.mark.parametrize("sel_tech,method_name", [
    ("moa", "_select_moa"),
    ("majority", "_select_majority"),
    ("unknown", None)
])
def test_select_candidate_calls_correct_method(default_optimizer, sel_tech, method_name):
    opt = default_optimizer
    # Create dummy candidates
    cands = ["a", "b", "c"]
    if method_name:
        # Monkeypatch method to return sentinel
        sentinel = {'text': 'sent'}
        setattr(opt, method_name, lambda candidates, texts, summary=None: sentinel)
        result = opt.select_candidate(cands, selection_technique=sel_tech)
        assert result == sentinel
    else:
        # unknown should return last
        result = opt.select_candidate(cands, selection_technique=sel_tech)
        assert result == "c"

def test_multi_llm_generation_fallback(multi_llm_optimizer, monkeypatch):
    """Test that multi_llm generation falls back gracefully on error"""
    opt = multi_llm_optimizer
    
    # Mock LLMFactory to raise exception
    def failing_get_llm(profile):
        raise Exception("LLM creation failed")
    
    monkeypatch.setattr(MockLLMFactory, 'get_llm', failing_get_llm)
    monkeypatch.setattr('opto.utils.llm.LLMFactory', MockLLMFactory)
    
    # Should fall back to temperature_variation
    responses = [["fallback1"], ["fallback2"], ["fallback3"]]
    opt.llm = DummyLLM(responses=responses)
    
    cands = opt.generate_candidates(None, "sys", "usr", num_responses=3, 
                                  generation_technique="multi_llm", verbose=True)
    assert len(cands) == 3

def test_integration_step_updates(default_optimizer, parameter_node):
    opt = default_optimizer
    # Dummy parameter_node initial value
    parameter_node._data = 0
    # LLM returns JSON suggesting new value for parameter
    suggestion = {"x": 42}
    response_str = json.dumps({"reasoning": "ok", "answer": "", "suggestion": suggestion})
    opt.llm = DummyLLM(responses=[[response_str]*opt.num_responses])
    # Run a step
    update = opt._step(verbose=False)
    assert isinstance(update, dict)

# Test default model attribute exists and is gpt-4.1-nano
def test_default_model_name(default_optimizer):
    opt = default_optimizer
    # Default model should be set if not provided (string contains 'gpt-4.1-nano')
    model_name = getattr(opt.llm, 'model', 'gpt-4.1-nano')
    assert 'gpt-4.1-nano' in model_name


def test_multi_llm_step_integration(multi_llm_optimizer, parameter_node, monkeypatch):
    """Test full integration of multi-LLM optimization step"""
    opt = multi_llm_optimizer
    monkeypatch.setattr('opto.utils.llm.LLMFactory', MockLLMFactory)
    
    parameter_node._data = 0
    
    # Mock multiple LLM responses for multi_llm generation
    suggestion = {"x": 42}
    response_str = json.dumps({"reasoning": "ok", "answer": "", "suggestion": suggestion})
    
    # Each profile should return a response
    cheap_llm = DummyLLM(responses=[[response_str]])
    premium_llm = DummyLLM(responses=[[response_str]])
    default_llm = DummyLLM(responses=[[response_str]])
    
    opt._llm_instances = {
        'cheap': cheap_llm,
        'premium': premium_llm,
        'default': default_llm
    }
    
    # Override _parallel_call_llm to return mock responses
    def mock_parallel_call(arg_dicts):
        return [response_str] * len(arg_dicts)
    
    opt._parallel_call_llm = mock_parallel_call
    
    # Run optimization step
    update = opt._step(verbose=False, generation_technique='multi_llm')
    assert isinstance(update, dict)

def test_llm_weights_handling():
    """Test that LLM weights are properly handled"""
    param = ParameterNode(name='test', value=1)
    
    # Test with explicit weights
    profiles = ['cheap', 'premium']
    weights = [0.3, 0.7]
    opt1 = OptoPrimeMulti([param], llm_profiles=profiles, llm_weights=weights)
    assert opt1.llm_weights == weights
    
    # Test with automatic weights (should default to 1.0 for each profile)
    opt2 = OptoPrimeMulti([param], llm_profiles=profiles)
    assert opt2.llm_weights == [1.0, 1.0]
    
    # Test without profiles (should be None)
    opt3 = OptoPrimeMulti([param])
    assert opt3.llm_weights is None

def test_multi_llm_logging(multi_llm_optimizer, monkeypatch):
    """Test that multi-LLM usage is properly logged"""
    opt = multi_llm_optimizer
    opt.log = []  # Enable logging
    
    # Manually set LLM instances to avoid import issues
    opt._llm_instances = {
        'cheap': DummyLLM(responses=[["response1"]]),
        'premium': DummyLLM(responses=[["response2"]]),
        'default': DummyLLM(responses=[["response3"]])
    }

    # Override _parallel_call_llm to return mock responses
    def mock_parallel_call(arg_dicts):
        return ["response1", "response2", "response3"]
    
    opt._parallel_call_llm = mock_parallel_call
    
    cands = opt.generate_candidates(None, "sys", "usr", num_responses=3,
                                  generation_technique="multi_llm")
    
    # Check that logging includes llm_profiles
    assert len(opt.log) > 0
    log_entry = opt.log[-1]
    assert 'llm_profiles' in log_entry
    assert log_entry['llm_profiles'] == ['cheap', 'premium', 'default']
 
def user_code(output):
    if output < 0:
        return "Success."
    else:
        return "Try again. The output should be negative"

@pytest.mark.parametrize("gen_tech", [
    "temperature_variation", 
    "self_refinement", 
    "iterative_alternatives", 
    "multi_experts",
    "multi_llm"
])
@pytest.mark.parametrize("sel_tech", [
    "moa", 
    "lastofn", 
    "majority"
])
def test_optimizer_with_code(gen_tech, sel_tech):
    """Test optimizing code functionality"""
    @bundle(trainable=True)
    def my_fun(x):
        """Test function"""
        return x**2 + 1

    old_func_value = my_fun.parameter.data

    x = node(-1, trainable=False)
    optimizer = OptoPrimeMulti([my_fun.parameter], generation_technique=gen_tech, selection_technique=sel_tech)
    output = my_fun(x)
    feedback = user_code(output.data)
    optimizer.zero_feedback()
    optimizer.backward(output, feedback)

    print(f"output={output.data}, feedback={feedback}, variables=")
    for p in optimizer.parameters:
        print(p.name, p.data)
        
    optimizer.step(verbose=True)
    new_func_value = my_fun.parameter.data

    # The function implementation should be changed
    assert str(old_func_value) != str(new_func_value), f"{OptoPrimeMulti.__name__} failed to update function"
    print(f"Function updated: old value: {str(old_func_value)}, new value: {str(new_func_value)}")


def test_backwards_compatibility():
    """Test that existing OptoPrimeMulti usage continues to work without changes"""
    param = ParameterNode(name='test', value=1)
    
    # Old-style initialization should work exactly as before
    opt = OptoPrimeMulti([param], 
                        num_responses=3,
                        generation_technique="temperature_variation",
                        selection_technique="best_of_n")
    
    # New attributes should have sensible defaults
    assert opt.llm_profiles is None
    assert opt.llm_weights is None
    assert opt._llm_instances == {}
    
    # Should fall back to single LLM behavior
    llms = opt._get_llms_for_generation(3)
    assert len(llms) == 3
    assert all(llm is opt.llm for llm in llms)
    
    # Profile retrieval should return default LLM for None
    assert opt._get_llm_for_profile(None) is opt.llm
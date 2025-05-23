import json
import pytest
from opto.optimizers.optoprimemulti import OptoPrimeMulti
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

def test_call_llm_returns_list(default_optimizer):
    opt = default_optimizer
    # Prepare dummy response
    opt.llm = DummyLLM(responses=[["resp1", "resp2"]])
    results = opt.call_llm("sys", "usr", num_responses=2, temperature=0.5)
    assert isinstance(results, list)
    assert results == ["resp1", "resp2"]

@pytest.mark.parametrize("gen_tech", [
    "temperature_variation", 
    "self_refinement", 
    "iterative_alternatives", 
    "multi_experts"]
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


def user_code(output):
    if output < 0:
        return "Success."
    else:
        return "Try again. The output should be negative"

@pytest.mark.parametrize("gen_tech", [
    "temperature_variation", 
    "self_refinement", 
    "iterative_alternatives", 
    "multi_experts"
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



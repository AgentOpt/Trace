import os
import pytest
from opto.trace import bundle, node, GRAPH
import opto.optimizers
from opto.optimizers import OptoPrimeMulti, OptoPrime, TextGrad
import importlib
import inspect
import json
import pickle
from opto.utils.llm import LLM

# Dynamically get all optimizer classes from opto.optimizers
def get_all_optimizers():
    """Dynamically retrieve all optimizer classes from opto.optimizers"""
    optimizers = []
    for name in dir(opto.optimizers):
        item = getattr(opto.optimizers, name)
        # Check if it's a class and has 'step' method (likely an optimizer)
        if inspect.isclass(item) and hasattr(item, 'step'):
            optimizers.append(item)
    return optimizers

ALL_OPTIMIZERS = get_all_optimizers()
# You can override for temporarly testing a specific optimizer ALL_OPTIMIZERS = [TextGrad] # [OptoPrimeMulti] ALL_OPTIMIZERS = [OptoPrime]

# Skip tests if no API credentials are available
SKIP_REASON = "No API credentials found"
HAS_CREDENTIALS = os.path.exists("OAI_CONFIG_LIST") or os.environ.get("TRACE_LITELLM_MODEL") or os.environ.get("OPENAI_API_KEY")
llm = LLM()

@pytest.fixture(autouse=True)
def clear_graph():
    """Reset the graph before each test"""
    GRAPH.clear()
    yield
    GRAPH.clear()

@pytest.fixture(params=ALL_OPTIMIZERS)
def optimizer_class(request):
    """Fixture to provide each optimizer class"""
    return request.param

def blackbox(x):
    return -x * 2

@bundle()
def bar(x):
    "This is a test function, which does negative scaling."
    return blackbox(x)

def foo(x):
    y = x + 1
    return x * y

def foobar(x):
    return foo(bar(x))

def user_number(x):
    if x < 50:
        return "The number needs to be larger."
    else:
        return "Success."

@pytest.mark.skipif(not HAS_CREDENTIALS, reason=SKIP_REASON)
def test_optimizer_with_number(optimizer_class):
    """Test optimizing a numeric input"""
    x = node(-1.0, trainable=True)
    optimizer = optimizer_class([x])
    output = foobar(x)
    feedback = user_number(output.data)
    optimizer.zero_feedback()
    optimizer.backward(output, feedback, visualize=True)
    
    # Store initial data for comparison
    initial_data = x.data
    
    optimizer.step(verbose=True)
    
    # Basic assertion - data should change after optimization
    assert x.data != initial_data, f"{optimizer_class.__name__} failed to update x value"

@bundle()
def convert_english_to_numbers(x):
    """This is a function that converts English to numbers. This function has limited ability."""
    # remove special characters, like, ", &, etc.
    x = x.replace('"', "")
    try:  # Convert string to integer
        return int(x)
    except ValueError:
        pass
    # Convert integers written in English in [-10, 10] to numbers
    mapping = {
        "negative ten": -10, "negative nine": -9, "negative eight": -8,
        "negative seven": -7, "negative six": -6, "negative five": -5,
        "negative four": -4, "negative three": -3, "negative two": -2,
        "negative one": -1, "zero": 0, "one": 1, "two": 2, "three": 3,
        "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10
    }
    return mapping.get(x, "FAIL")

def user_text(x):
    if x == "FAIL":
        return "The text cannot be converted to a number."
    if x < 50:
        return "The number needs to be larger."
    else:
        return "Success."

def foobar_text(x):
    output = convert_english_to_numbers(x)
    if output.data == "FAIL":  # This is not traced
        return output
    else:
        return foo(bar(output))

@pytest.mark.skipif(not HAS_CREDENTIALS, reason=SKIP_REASON)
def test_optimizer_with_text(optimizer_class):
    """Test optimizing a text input"""
    x = node("negative point one", trainable=True)
    optimizer = optimizer_class([x])
    output = foobar_text(x)
    feedback = user_text(output.data)
    
    # Store initial data
    initial_data = x.data
    
    optimizer.zero_feedback()
    optimizer.backward(output, feedback)
    print(f"variable={x.data}, output={output.data}, feedback={feedback}")
    optimizer.step(verbose=True)
    
    # Basic assertion - the optimizer should attempt to change the input
    assert x.data != initial_data, f"{optimizer_class.__name__} failed to update text value"

def user_code(output):
    if output < 0:
        return "Success."
    else:
        return "Try again. The output should be negative"

@pytest.mark.skipif(not HAS_CREDENTIALS, reason=SKIP_REASON)
def test_optimizer_with_code(optimizer_class):
    """Test optimizing code functionality"""
    @bundle(trainable=True)
    def my_fun(x):
        """Test function"""
        return x**2 + 1

    old_func_value = my_fun.parameter.data

    x = node(-1, trainable=False)
    optimizer = optimizer_class([my_fun.parameter])
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
    assert str(old_func_value) != str(new_func_value), f"{optimizer_class.__name__} failed to update function"
    print(f"Function updated: old value: {str(old_func_value)}, new value: {str(new_func_value)}")

@pytest.mark.skipif(not HAS_CREDENTIALS, reason=SKIP_REASON)
def test_direct_feedback(optimizer_class):
    """Test providing feedback directly to parameters"""
    x = node(-1, trainable=True)
    optimizer = optimizer_class([x])
    initial_data = x.data
    
    feedback = "This should be a positive number greater than 10"
    optimizer.zero_feedback()
    optimizer.backward(x, feedback)
    optimizer.step(verbose=True)
    
    # Basic assertion - the optimizer should attempt to change the input
    assert x.data != initial_data, f"{optimizer_class.__name__} failed to handle direct feedback"

@pytest.mark.skipif(not HAS_CREDENTIALS, reason=SKIP_REASON)
def test_log_serialization(optimizer_class):
    """Test if optimizer logs can be saved in both pickle and JSON formats"""
    x = node(-1, trainable=True)
    optimizer = optimizer_class([x])
    feedback = "test"
    optimizer.zero_feedback()
    optimizer.backward(x, feedback)
    optimizer.step(verbose=True)
    
    # Create unique filenames for each optimizer to avoid conflicts in parallel testing
    optimizer_name = optimizer_class.__name__
    json_filename = f"log_{optimizer_name}.json"
    pickle_filename = f"log_{optimizer_name}.pik"
    
    try:
        # Test JSON serialization
        json.dump(optimizer.log, open(json_filename, "w"))
        assert os.path.exists(json_filename), f"Failed to create JSON log for {optimizer_name}"
        
        # Test pickle serialization
        pickle.dump(optimizer.log, open(pickle_filename, "wb"))
        assert os.path.exists(pickle_filename), f"Failed to create pickle log for {optimizer_name}"
    finally:
        # Clean up the files
        for filename in [json_filename, pickle_filename]:
            if os.path.exists(filename):
                os.remove(filename)

@pytest.mark.skipif(not HAS_CREDENTIALS, reason=SKIP_REASON)
def test_optimizer_customization(optimizer_class):
    """Test optimizer with custom parameters"""
    x = node(-1.0, trainable=True)
    
    # Try to set custom parameters if the optimizer supports it
    try:
        if hasattr(optimizer_class, '__init__') and 'temperature' in inspect.signature(optimizer_class.__init__).parameters:
            optimizer = optimizer_class([x], temperature=0.7)
        else:
            optimizer = optimizer_class([x])
    except Exception as e:
        # Skip this test if custom parameters aren't supported
        pytest.skip(f"Optimizer {optimizer_class.__name__} doesn't support custom parameters: {str(e)}")
    
    output = foobar(x)
    feedback = user_number(output.data)
    optimizer.zero_feedback()
    optimizer.backward(output, feedback)
    
    # Store initial data
    initial_data = x.data
    
    optimizer.step(verbose=True)
    
    # Basic assertion - data should change after optimization
    assert x.data != initial_data, f"{optimizer_class.__name__} with custom params failed to update value"
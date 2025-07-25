from opto.trace.projections import BlackCodeFormatter, DocstringProjection, SuggestionNormalizationProjection
from types import SimpleNamespace

def test_black_code_formatter():
    code = """
def example_function():
                print("Hello, World!")


                print("This is a test function.")


                
    """     
    projection = BlackCodeFormatter()
    formatted_code = projection.project(code)
    assert formatted_code == 'def example_function():\n    print("Hello, World!")\n\n    print("This is a test function.")\n'


def test_docstring_projection():
    code = """
def example_function():
    \"\"\"This is an example function.\"\"\"
    print("Hello, World!")
    """
    docstring = "This is a new docstring."
    projection = DocstringProjection(docstring)
    formatted_code = projection.project(code)
    
    new_code = """
def example_function():
    \"\"\"This is a new docstring.\"\"\"
    print("Hello, World!")
    """

    assert formatted_code == new_code

    # assert '"""This is a new docstring."""' in formatted_code    
    # assert 'print("Hello, World!")' in formatted_code

def test_suggestion_normalization_projection():
    import re
    import pytest
    # Prepare a mock parameter list with various py_names, types, and trainable flags
    params = [
        # code param: key comes in as "__code:1", should alias to "__code1" and be black‑formatted
        SimpleNamespace(py_name="__code1", trainable=True, data=""),
        # learning rate param: as float, but suggestion comes as a literal string
        SimpleNamespace(py_name="__lr", trainable=True, data=0.0),
        # should be skipped because not trainable
        SimpleNamespace(py_name="__frozen", trainable=False, data=123),
        # some other param, no suggestion provided
        SimpleNamespace(py_name="__missing", trainable=True, data=1)
    ]

    raw_suggestion = {
        "__code:1": "def foo(x):return x*2",   # needs black formatting
        "__lr": "\"0.01\"",                    # needs literal‐eval → float
        "__frozen": "999",                     # should be ignored
        "unrelated": "[1,2,3]",                # not in params
    }

    proj = SuggestionNormalizationProjection(params)
    normalized = proj.project(raw_suggestion)

    # It should only contain keys for trainable params that were suggested
    assert set(normalized.keys()) == {"__code1", "__lr"}

    # 1) __code1 should be black‐formatted: 'def foo' newline indent 'return x * 2'
    code_out = normalized["__code1"]
    # check that there's exactly one indent (4 spaces) before the return,
    # and that black added a trailing newline
    assert re.search(r"def foo\(x\):\n {4}return x \* 2\n$", code_out)

    # 2) __lr should have been converted from the string "0.01" to float 0.01
    assert isinstance(normalized["__lr"], float)
    assert normalized["__lr"] == pytest.approx(0.01)

    # 3) Non‐trainable or missing params should not appear
    assert "__frozen" not in normalized
    assert "__missing" not in normalized

    # --- literal‑eval failure should be left unchanged ---
    # If ast.literal_eval raises, the original string remains
    params_bad = [SimpleNamespace(py_name="__bad", trainable=True, data=100)]
    raw_suggestion_bad = {"__bad": "not_a_number"}
    normalized_bad = SuggestionNormalizationProjection(params_bad).project(raw_suggestion_bad)
    assert normalized_bad["__bad"] == "not_a_number"
import os
import pytest
from opto.trace import bundle, node, GRAPH
import opto.optimizers
import importlib
import inspect
import json
import pickle
from opto.utils.llm import LLM

from opto import trace
from opto.trace import node, bundle
from opto.optimizers.optoprime_v2 import OptoPrimeV2, OptimizerPromptSymbolSet, OptimizerPromptSymbolSet2

# You can override for temporarly testing a specific optimizer ALL_OPTIMIZERS = [TextGrad] # [OptoPrimeMulti] ALL_OPTIMIZERS = [OptoPrime]

# Skip tests if no API credentials are available
SKIP_REASON = "No API credentials found"
HAS_CREDENTIALS = os.path.exists("OAI_CONFIG_LIST") or os.environ.get("TRACE_LITELLM_MODEL") or os.environ.get(
    "OPENAI_API_KEY")
llm = LLM()


@pytest.fixture(autouse=True)
def clear_graph():
    """Reset the graph before each test"""
    GRAPH.clear()
    yield
    GRAPH.clear()


@pytest.mark.skipif(not HAS_CREDENTIALS, reason=SKIP_REASON)
def test_response_extraction():
    pass


class CustomConstraintSymbolSet(OptimizerPromptSymbolSet):
    constraint_tag = "guard"


def test_code_section_mask_only_hides_code_section():
    num = node(1, trainable=True)
    result = num + 1
    optimizer = OptoPrimeV2([num], use_json_object_format=False)

    optimizer.zero_feedback()
    optimizer.backward(result, "make this number bigger")

    summary = optimizer.summarize()

    code_masked = optimizer.problem_instance(
        summary,
        mask=[optimizer.optimizer_prompt_symbol_set.code_section_title],
    )
    inputs_masked = optimizer.problem_instance(
        summary,
        mask=[optimizer.optimizer_prompt_symbol_set.inputs_section_title],
    )

    assert code_masked.code == "", "# Code must hide the code section."
    assert inputs_masked.code != "", "# Inputs must not hide the code section."
    assert inputs_masked.inputs == "", "# Inputs must still hide the inputs section."


def test_repr_node_value_respects_custom_constraint_tag_for_code_variables():
    num = node(1, trainable=True)
    optimizer = OptoPrimeV2(
        [num],
        use_json_object_format=False,
        optimizer_prompt_symbol_set=CustomConstraintSymbolSet(),
    )

    rendered = optimizer.repr_node_value(
        {"__code0": ("def f(x):\n    return x", "The code should start with:\ndef f(x):")},
        node_tag=optimizer.optimizer_prompt_symbol_set.variable_tag,
        value_tag=optimizer.optimizer_prompt_symbol_set.value_tag,
        constraint_tag=optimizer.optimizer_prompt_symbol_set.constraint_tag,
    )

    assert "<guard>" in rendered, "Full code rendering must use custom constraint tags."
    assert "<constraint>" not in rendered, "Full code rendering must not hardcode the default tag."


def test_tag_template_change():
    num_1 = node(1, trainable=True)
    num_2 = node(2, trainable=True, description="<=5")
    result = num_1 + num_2
    optimizer = OptoPrimeV2([num_1, num_2], use_json_object_format=False,
                            ignore_extraction_error=False,
                            include_example=True,
                            optimizer_prompt_symbol_set=OptimizerPromptSymbolSet2())

    optimizer.zero_feedback()
    optimizer.backward(result, 'make this number bigger')

    summary = optimizer.summarize()
    part1, part2 = optimizer.construct_prompt(summary)

    part1 = optimizer.replace_symbols(part1, optimizer.prompt_symbols)
    part2 = optimizer.replace_symbols(part2, optimizer.prompt_symbols)

    assert """<var name="variable_name" type="data_type">""" in part1, "Expected <var> tag to be present in part1"
    assert """<const name="y" type="int">""" in part2, "Expected <const> tag to be present in part2"

    print(part1)
    print(part2)


@bundle()
def transform(num):
    """Add number"""
    return num + 1


@bundle(trainable=True)
def multiply(num):
    return num * 5


def test_function_repr():
    num_1 = node(1, trainable=False)

    result = multiply(transform(num_1))
    optimizer = OptoPrimeV2([multiply.parameter], use_json_object_format=False,
                            ignore_extraction_error=False,
                            include_example=True)

    optimizer.zero_feedback()
    optimizer.backward(result, 'make this number bigger')

    summary = optimizer.summarize()
    part1, part2 = optimizer.construct_prompt(summary)

    part1 = optimizer.replace_symbols(part1, optimizer.prompt_symbols)
    part2 = optimizer.replace_symbols(part2, optimizer.prompt_symbols)

    function_repr = """<variable name="__code0" type="code">
<value>
def multiply(num):
    return num * 5
</value>
<constraint>
The code should start with:
def multiply(num):
</constraint>
</variable>"""

    assert function_repr in part2, "Expected function representation to be present in part2"

def test_big_data_truncation():
    num_1 = node(1, trainable=True)

    list_1 = node([1, 2, 3, 4, 5, 6, 7, 8, 9, 20] * 10, trainable=True)

    result = num_1 + list_1[30]

    optimizer = OptoPrimeV2([num_1, list_1], use_json_object_format=False,
                            ignore_extraction_error=False,
                            include_example=True, initial_var_char_limit=10)

    optimizer.zero_feedback()
    optimizer.backward(result, 'make this number bigger')

    summary = optimizer.summarize()
    part1, part2 = optimizer.construct_prompt(summary)

    part1 = optimizer.replace_symbols(part1, optimizer.prompt_symbols)
    part2 = optimizer.replace_symbols(part2, optimizer.prompt_symbols)

    truncated_repr = """<variable name="list0" type="list">
<value>
[1, 2, 3, ...(skipped due to length limit)
</value>
</variable>"""

    assert truncated_repr in part2, "Expected truncated list representation to be present in part2"

def test_extraction_pipeline():
    num_1 = node(1, trainable=True)
    num_2 = node(2, trainable=True, description="<=5")
    result = num_1 + num_2
    optimizer = OptoPrimeV2([num_1, num_2], use_json_object_format=False,
                            ignore_extraction_error=False,
                            include_example=True,
                            optimizer_prompt_symbol_set=OptimizerPromptSymbolSet2())

    optimizer.zero_feedback()
    optimizer.backward(result, 'make this number bigger')

    summary = optimizer.summarize()
    part1, part2 = optimizer.construct_prompt(summary)

    part1 = optimizer.replace_symbols(part1, optimizer.prompt_symbols)
    part2 = optimizer.replace_symbols(part2, optimizer.prompt_symbols)

    messages = [
        {"role": "system", "content": part1},
        {"role": "user", "content": part2},
    ]

    # response = optimizer.llm(messages=messages)
    # response = response.choices[0].message.content
    response = """<reason>
The instruction suggests that the output, `add0`, needs to be made bigger than it currently is (3). The code performs an addition of `int0` and `int1` to produce `add0`. To increase `add0`, we can increase the values of `int0` or `int1`, or both. Given that `int1` has a constraint of being less than or equal to 5, we can set `int0` to a higher value, since it has no explicit constraint. By adjusting `int0` to a higher value, the output can be made larger in accordance with the feedback.
</reason>

<var>
<name>int0</name>
<data>
5
</data>
</var>

<var>
<name>int1</name>
<data>
5
</data>
</var>"""
    reasoning = response
    suggestion = optimizer.extract_llm_suggestion(response)

    assert 'reasoning' in suggestion, "Expected 'reasoning' in suggestion"
    assert 'variables' in suggestion, "Expected 'variables' in suggestion"
    assert 'int0' in suggestion['variables'], "Expected 'int0' variable in suggestion"
    assert 'int1' in suggestion['variables'], "Expected 'int1' variable in suggestion"
    assert suggestion['variables']['int0'] == 5, "Expected int0 to be incremented to 5"
    assert suggestion['variables']['int1'] == 5, "Expected int1 to be incremented to 5"

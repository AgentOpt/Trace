import os
import pytest
from opto.trace import GRAPH
from opto.utils.llm import LLM

from opto.trace import node, bundle
from opto.optimizers.optoprime_v3 import (
    OptoPrimeV3, OptimizerPromptSymbolSet2, ProblemInstance,
    OptimizerPromptSymbolSet, value_to_image_content
)
from opto.utils.backbone import TextContent, ImageContent

# You can override for temporarly testing a specific optimizer ALL_OPTIMIZERS = [TextGrad] # [OptoPrimeMulti] ALL_OPTIMIZERS = [OptoPrime]

# Skip tests if no API credentials are available
SKIP_REASON = "No API credentials found"
HAS_CREDENTIALS = os.path.exists("OAI_CONFIG_LIST") or os.environ.get("TRACE_LITELLM_MODEL") or os.environ.get(
    "OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
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


def test_tag_template_change():
    num_1 = node(1, trainable=True)
    num_2 = node(2, trainable=True, description="<=5")
    result = num_1 + num_2
    optimizer = OptoPrimeV3([num_1, num_2], use_json_object_format=False,
                            ignore_extraction_error=False,
                            include_example=True,
                            optimizer_prompt_symbol_set=OptimizerPromptSymbolSet2())

    optimizer.zero_feedback()
    optimizer.backward(result, 'make this number bigger')

    summary = optimizer.summarize()
    system_prompt, user_prompt = optimizer.construct_prompt(summary)

    # system_prompt is a string, user_prompt is a ContentBlockList
    system_prompt = optimizer.replace_symbols(system_prompt, optimizer.prompt_symbols)
    
    # Convert ContentBlockList to text for symbol replacement
    user_prompt_text = "".join(block.text for block in user_prompt if isinstance(block, TextContent))
    user_prompt_text = optimizer.replace_symbols(user_prompt_text, optimizer.prompt_symbols)

    assert """<var name="variable_name" type="data_type">""" in system_prompt, "Expected <var> tag to be present in system_prompt"
    assert """<const name="y" type="int">""" in user_prompt_text, "Expected <const> tag to be present in user_prompt"

    print(system_prompt)
    print(user_prompt_text)


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
    optimizer = OptoPrimeV3([multiply.parameter], use_json_object_format=False,
                            ignore_extraction_error=False,
                            include_example=True)

    optimizer.zero_feedback()
    optimizer.backward(result, 'make this number bigger')

    summary = optimizer.summarize()
    system_prompt, user_prompt = optimizer.construct_prompt(summary)

    system_prompt = optimizer.replace_symbols(system_prompt, optimizer.prompt_symbols)
    # Convert ContentBlockList to text for symbol replacement
    user_prompt_text = "".join(block.text for block in user_prompt if isinstance(block, TextContent))
    user_prompt_text = optimizer.replace_symbols(user_prompt_text, optimizer.prompt_symbols)

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

    assert function_repr in user_prompt_text, "Expected function representation to be present in user_prompt"

def test_big_data_truncation():
    num_1 = node("**2", trainable=True)

    list_1 = node("12345691912338" * 10, trainable=False)

    result = list_1 + num_1

    optimizer = OptoPrimeV3([num_1], use_json_object_format=False,
                            ignore_extraction_error=False,
                            include_example=True, initial_var_char_limit=10)

    optimizer.zero_feedback()
    optimizer.backward(result, 'compute the expression')

    summary = optimizer.summarize()
    system_prompt, user_prompt = optimizer.construct_prompt(summary)

    system_prompt = optimizer.replace_symbols(system_prompt, optimizer.prompt_symbols)
    # Convert ContentBlockList to text for symbol replacement
    user_prompt_text = "".join(block.text for block in user_prompt if isinstance(block, TextContent))
    user_prompt_text = optimizer.replace_symbols(user_prompt_text, optimizer.prompt_symbols)

    truncated_repr = """1234569191...(skipped due to length limit)"""

    assert truncated_repr in user_prompt_text, "Expected truncated list representation to be present in user_prompt"

def test_extraction_pipeline():
    num_1 = node(1, trainable=True)
    num_2 = node(2, trainable=True, description="<=5")
    result = num_1 + num_2
    optimizer = OptoPrimeV3([num_1, num_2], use_json_object_format=False,
                            ignore_extraction_error=False,
                            include_example=True,
                            optimizer_prompt_symbol_set=OptimizerPromptSymbolSet2())

    optimizer.zero_feedback()
    optimizer.backward(result, 'make this number bigger')

    summary = optimizer.summarize()
    system_prompt, user_prompt = optimizer.construct_prompt(summary)

    # Verify construct_prompt returns expected types
    assert isinstance(system_prompt, str)
    assert isinstance(user_prompt, list)

    # Test extraction from a mock response
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
    suggestion = optimizer.extract_llm_suggestion(response)

    assert 'reasoning' in suggestion, "Expected 'reasoning' in suggestion"
    assert 'variables' in suggestion, "Expected 'variables' in suggestion"
    assert 'int0' in suggestion['variables'], "Expected 'int0' variable in suggestion"
    assert 'int1' in suggestion['variables'], "Expected 'int1' variable in suggestion"
    assert suggestion['variables']['int0'] == '5', "Expected int0 to be incremented to 5"
    assert suggestion['variables']['int1'] == '5', "Expected int1 to be incremented to 5"


# ==================== Multimodal / Content Block Tests ====================

def test_problem_instance_text_only():
    """Test that ProblemInstance with text-only content works correctly."""
    from opto.utils.backbone import ContentBlockList
    symbol_set = OptimizerPromptSymbolSet()
    
    instance = ProblemInstance(
        instruction="Test instruction",
        code="y = add(x=a, y=b)",
        documentation="[add] Adds two numbers",
        variables=ContentBlockList("<variable name='a' type='int'><value>5</value></variable>"),
        inputs=ContentBlockList("<node name='b' type='int'><value>3</value></node>"),
        others=ContentBlockList(),
        outputs=ContentBlockList("<node name='y' type='int'><value>8</value></node>"),
        feedback="Result should be 10",
        context="Some context",
        optimizer_prompt_symbol_set=symbol_set
    )
    
    # Test __repr__ returns string
    text_repr = str(instance)
    assert "Test instruction" in text_repr
    assert "y = add(x=a, y=b)" in text_repr
    assert "Result should be 10" in text_repr
    assert "Some context" in text_repr
    
    # Test to_content_blocks returns list
    blocks = instance.to_content_blocks()
    assert isinstance(blocks, list)
    assert len(blocks) > 0
    assert all(isinstance(b, (TextContent, ImageContent)) for b in blocks)
    
    # Test has_images returns False for text-only
    assert not instance.has_images()


def test_problem_instance_with_content_blocks():
    """Test ProblemInstance with ContentBlockList fields containing images."""
    from opto.utils.backbone import ContentBlockList
    symbol_set = OptimizerPromptSymbolSet()
    
    # Create content blocks with an image
    variables_blocks = ContentBlockList([
        TextContent(text="<variable name='img' type='image'><value>"),
        ImageContent(image_url="https://example.com/test.jpg"),
        TextContent(text="</value></variable>")
    ])
    
    instance = ProblemInstance(
        instruction="Analyze the image",
        code="result = analyze(img)",
        documentation="[analyze] Analyzes an image",
        variables=variables_blocks,
        inputs=ContentBlockList(),
        others=ContentBlockList(),
        outputs=ContentBlockList("<node name='result' type='str'><value>cat</value></node>"),
        feedback="Result should be 'dog'",
        context=None,
        optimizer_prompt_symbol_set=symbol_set
    )
    
    # Test __repr__ handles content blocks (should show [IMAGE] placeholder)
    text_repr = str(instance)
    assert "Analyze the image" in text_repr
    assert "[IMAGE]" in text_repr
    
    # Test to_content_blocks includes the image
    blocks = instance.to_content_blocks()
    assert isinstance(blocks, list)
    
    # Find the ImageContent block
    image_blocks = [b for b in blocks if isinstance(b, ImageContent)]
    assert len(image_blocks) == 1
    assert image_blocks[0].image_url == "https://example.com/test.jpg"
    
    # Test has_images returns True
    assert instance.has_images()


def test_problem_instance_mixed_content():
    """Test ProblemInstance with mixed text and image content in multiple fields."""
    from opto.utils.backbone import ContentBlockList
    symbol_set = OptimizerPromptSymbolSet()
    
    # Variables with image
    variables_blocks = ContentBlockList([
        TextContent(text="<variable name='prompt' type='str'><value>Hello</value></variable>\n"),
        TextContent(text="<variable name='img' type='image'><value>"),
        ImageContent(image_data="base64data", media_type="image/png"),
        TextContent(text="</value></variable>")
    ])
    
    # Inputs with image
    inputs_blocks = ContentBlockList([
        TextContent(text="<node name='reference' type='image'><value>"),
        ImageContent(image_url="https://example.com/ref.png"),
        TextContent(text="</value></node>")
    ])
    
    instance = ProblemInstance(
        instruction="Compare images",
        code="result = compare(img, reference)",
        documentation="[compare] Compares two images",
        variables=variables_blocks,
        inputs=inputs_blocks,
        others=ContentBlockList(),
        outputs=ContentBlockList("<node name='result' type='float'><value>0.8</value></node>"),
        feedback="Similarity should be higher",
        context="Context text",
        optimizer_prompt_symbol_set=symbol_set
    )
    
    # Test has_images
    assert instance.has_images()
    
    # Test to_content_blocks
    blocks = instance.to_content_blocks()
    image_blocks = [b for b in blocks if isinstance(b, ImageContent)]
    assert len(image_blocks) == 2  # One from variables, one from inputs


def test_value_to_image_content_url():
    """Test value_to_image_content with URL strings."""
    # Valid image URL
    result = value_to_image_content("https://example.com/image.jpg")
    assert result is not None
    assert isinstance(result, ImageContent)
    assert result.image_url == "https://example.com/image.jpg"
    
    # Non-image URL (no image extension) - is_image returns False for pattern check
    result = value_to_image_content("https://example.com/page.html")
    assert result is None
    
    # Non-URL string
    result = value_to_image_content("just a regular string")
    assert result is None


def test_value_to_image_content_base64():
    """Test value_to_image_content with base64 data URLs."""
    # Valid base64 data URL
    data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
    result = value_to_image_content(data_url)
    assert result is not None
    assert isinstance(result, ImageContent)
    assert result.image_data == "iVBORw0KGgoAAAANSUhEUg=="
    assert result.media_type == "image/png"


def test_value_to_image_content_non_image():
    """Test value_to_image_content with non-image values."""
    # Integer
    assert value_to_image_content(42) is None
    
    # List
    assert value_to_image_content([1, 2, 3]) is None
    
    # Dict
    assert value_to_image_content({"key": "value"}) is None
    
    # Regular string
    assert value_to_image_content("hello world") is None


def test_construct_prompt():
    """Test construct_prompt returns ContentBlockList for multimodal support."""
    num_1 = node(1, trainable=True)
    num_2 = node(2, trainable=True)
    result = num_1 + num_2
    
    optimizer = OptoPrimeV3([num_1, num_2], use_json_object_format=False)
    optimizer.zero_feedback()
    optimizer.backward(result, 'make this number bigger')
    
    summary = optimizer.summarize()
    system_prompt, user_prompt = optimizer.construct_prompt(summary)
    
    # system_prompt should be string, user_prompt should be ContentBlockList
    assert isinstance(system_prompt, str)
    assert isinstance(user_prompt, list)
    assert all(isinstance(b, (TextContent, ImageContent)) for b in user_prompt)
    
    # Check that text content contains expected info
    text_parts = [b.text for b in user_prompt if isinstance(b, TextContent)]
    full_text = "".join(text_parts)
    assert "int0" in full_text or "int1" in full_text


def test_repr_node_value_as_content_blocks():
    """Test repr_node_value_as_content_blocks method."""
    num_1 = node(1, trainable=True)
    result = num_1 + 1
    
    optimizer = OptoPrimeV3([num_1], use_json_object_format=False)
    optimizer.zero_feedback()
    optimizer.backward(result, 'test')
    
    # Test with non-image nodes
    summary = optimizer.summarize()
    blocks = optimizer.repr_node_value_as_content_blocks(
        summary.variables,
        node_tag=optimizer.optimizer_prompt_symbol_set.variable_tag,
        value_tag=optimizer.optimizer_prompt_symbol_set.value_tag,
        constraint_tag=optimizer.optimizer_prompt_symbol_set.constraint_tag
    )
    
    assert isinstance(blocks, list)
    assert len(blocks) > 0
    assert all(isinstance(b, TextContent) for b in blocks)  # No images in this case


def test_repr_node_value_compact_as_content_blocks():
    """Test repr_node_value_compact_as_content_blocks method."""
    long_string = "x" * 5000  # Long string that will be truncated
    str_node = node(long_string, trainable=True)
    result = str_node + "!"
    
    optimizer = OptoPrimeV3([str_node], use_json_object_format=False, initial_var_char_limit=100)
    optimizer.zero_feedback()
    optimizer.backward(result, 'test')
    
    summary = optimizer.summarize()
    blocks = optimizer.repr_node_value_compact_as_content_blocks(
        summary.inputs,
        node_tag=optimizer.optimizer_prompt_symbol_set.node_tag,
        value_tag=optimizer.optimizer_prompt_symbol_set.value_tag,
        constraint_tag=optimizer.optimizer_prompt_symbol_set.constraint_tag
    )
    
    # Should be truncated
    text_parts = [b.text for b in blocks if isinstance(b, TextContent)]
    full_text = "".join(text_parts)
    assert "skipped due to length limit" in full_text or len(full_text) < len(long_string)


# ==================== Real LLM Call Tests ====================

@pytest.mark.skipif(not HAS_CREDENTIALS, reason=SKIP_REASON)
def test_optimizer_step_real_llm_call():
    """Test a real optimization step with LLM call."""
    # Create a simple optimization problem
    greeting = node("Hello", trainable=True, description="A greeting message")
    
    @bundle()
    def make_sentence(word):
        """Create a sentence from a word."""
        return f"{word}, how are you today?"
    
    result = make_sentence(greeting)
    
    # Create optimizer
    optimizer = OptoPrimeV3(
        [greeting],
        use_json_object_format=False,
        ignore_extraction_error=True,
        include_example=False,
    )
    
    # Setup feedback
    optimizer.zero_feedback()
    optimizer.backward(result, "The greeting should be more formal and professional")
    
    # Execute optimization step - this makes a real LLM call
    update_dict = optimizer.step(verbose=True)
    
    # Verify the optimizer produced a suggestion
    print(f"Update dict: {update_dict}")
    
    # The LLM should have suggested a new value
    # We don't assert specific content since LLM output varies
    # but we verify the step completed without error
    assert optimizer.log is not None
    assert len(optimizer.log) > 0
    
    # Check that the log contains the expected structure
    last_log = optimizer.log[-1]
    assert "system_prompt" in last_log
    assert "user_prompt" in last_log
    assert "response" in last_log
    
    print(f"LLM Response: {last_log['response'][:500]}...")


@pytest.mark.skipif(not HAS_CREDENTIALS, reason=SKIP_REASON)
def test_optimizer_step_with_content_blocks():
    """Test optimization step using content blocks (multimodal mode)."""
    # Create trainable parameters
    num_1 = node(5, trainable=True, description="A number to optimize")
    num_2 = node(3, trainable=True, description="Another number")
    
    result = num_1 + num_2
    
    # Create optimizer
    optimizer = OptoPrimeV3(
        [num_1, num_2],
        use_json_object_format=False,
        ignore_extraction_error=True,
        include_example=False,
    )
    
    # Setup feedback
    optimizer.zero_feedback()
    optimizer.backward(result, "The sum should be exactly 100")
    
    # Test that construct_prompt returns ContentBlockList
    summary = optimizer.summarize()
    system_prompt, user_prompt = optimizer.construct_prompt(summary)
    
    # Verify content blocks structure
    from opto.utils.backbone import ContentBlockList
    assert isinstance(user_prompt, ContentBlockList)
    assert len(user_prompt) > 0
    
    # Verify text is merged (should be fewer blocks than if not merged)
    text_blocks = [b for b in user_prompt if isinstance(b, TextContent)]
    print(f"Number of text blocks after merging: {len(text_blocks)}")
    
    # Execute the step (this makes a real LLM call)
    update_dict = optimizer.step(verbose=True)
    
    print(f"Update dict: {update_dict}")
    
    # Verify the step completed
    assert optimizer.log is not None
    assert len(optimizer.log) > 0

@pytest.mark.skipif(not HAS_CREDENTIALS, reason=SKIP_REASON)
def test_optimizer_multimodal_parameter_update():
    pass
"""
Key difference to v2:
1. Use the new backbone conversation history manager
2. Support multimodal node (both trainable and non-trainable)
"""

import re
import json
from typing import List, Union, Tuple, Optional
from dataclasses import dataclass
from opto.optimizers.optoprime import OptoPrime, node_to_function_feedback
from opto.trace.utils import dedent
from opto.optimizers.utils import truncate_expression, extract_xml_like_data, is_bedrock_model
from opto.trace.nodes import ParameterNode, is_image
from opto.trace.propagators import GraphPropagator
from opto.trace.propagators.propagators import Propagator

from opto.utils.llm import AbstractModel, LLM
from opto.optimizers.buffers import FIFOBuffer
from opto.utils.backbone import (
    Chat, UserTurn, AssistantTurn, PromptTemplate,
    TextContent, ImageContent, ContentBlockList,
    DEFAULT_IMAGE_PLACEHOLDER, Content
)
import copy
import pickle
from typing import Dict, Any


def value_to_image_content(value: Any) -> Optional[ImageContent]:
    """Convert a value to ImageContent if it's an image, otherwise return None.
    
    Uses is_image() from opto.trace.nodes for validation (stricter than ImageContent.build,
    e.g., only accepts URLs with image extensions), then delegates to ImageContent.build().
    
    Supports (via is_image detection):
    - Base64 data URL strings (data:image/...)
    - HTTP/HTTPS URLs pointing to images (pattern-based, must have image extension)
    - PIL Image objects
    - Raw image bytes
    """
    if not is_image(value):
        return None
    return ImageContent.build(value)


class OptimizerPromptSymbolSet:
    """
    By inheriting this class and pass into the optimizer. People can change the optimizer documentation

    This divides into three parts:
    - Section titles: the title of each section in the prompt
    - Node tags: the tags that capture the graph structure (only tag names are allowed to be changed)
    - Output format: the format of the output of the optimizer
    """

    # Titles should be written as markdown titles (space between # and title)
    # In text, we automatically remove space in the title, so it will become `#Title`
    variables_section_title = "# Variables"
    inputs_section_title = "# Inputs"
    outputs_section_title = "# Outputs"
    others_section_title = "# Others"
    feedback_section_title = "# Feedback"
    instruction_section_title = "# Instruction"
    code_section_title = "# Code"
    documentation_section_title = "# Documentation"
    context_section_title = "# Context"

    node_tag = "node"  # nodes that are constants in the graph
    variable_tag = "variable"  # nodes that can be changed
    value_tag = "value"  # inside node, we have value tag
    constraint_tag = "constraint"  # inside node, we have constraint tag

    # output format
    # Note: we currently don't support extracting format's like "```code```" because we assume supplied tag is name-only, i.e., <tag_name></tag_name>
    reasoning_tag = "reasoning"
    improved_variable_tag = "variable"
    name_tag = "name"

    # only used by JSON format
    suggestion_tag = "suggestion"

    expect_json = False  # this will stop `enforce_json` arguments passed to LLM calls

    # custom output format
    # if this is not None, then the user needs to implement the following functions:
    # - output_response_extractor
    # - example_output
    custom_output_format_instruction = None

    @property
    def output_format(self) -> str:
        """
        This function defines the input to:
        ```
        {output_format}
        ```
        In the self.output_format_prompt_template in the OptoPrimeV2
        """
        if self.custom_output_format_instruction is None:
            # we use a default XML like format
            return dedent(f"""
                <{self.reasoning_tag}>
                reasoning
                </{self.reasoning_tag}>
                <{self.improved_variable_tag}>
                <{self.name_tag}>variable_name</{self.name_tag}>
                <{self.value_tag}>
                value
                </{self.value_tag}>
                </{self.improved_variable_tag}>
            """)
        else:
            return self.custom_output_format_instruction.strip()

    def example_output(self, reasoning, variables):
        """
        reasoning: str
        variables: format {variable_name, value}
        """
        if self.custom_output_format_instruction is not None:
            raise NotImplementedError
        else:
            # Build the output string in the same XML-like format as self.output_format
            output = []
            if reasoning != "":
                output.append(f"<{self.reasoning_tag}>")
                output.append(reasoning)
                output.append(f"</{self.reasoning_tag}>")
            for var_name, value in variables.items():
                output.append(f"<{self.improved_variable_tag}>")
                output.append(f"<{self.name_tag}>{var_name}</{self.name_tag}>")
                output.append(f"<{self.value_tag}>")
                output.append(str(value))
                output.append(f"</{self.value_tag}>")
                output.append(f"</{self.improved_variable_tag}>")
            return "\n".join(output)

    def output_response_extractor(self, response: str) -> Dict[str, Any]:
        # the response here should just be plain text

        if self.custom_output_format_instruction is None:
            extracted_data = extract_xml_like_data(response,
                                                   reasoning_tag=self.reasoning_tag,
                                                   improved_variable_tag=self.improved_variable_tag,
                                                   name_tag=self.name_tag,
                                                   value_tag=self.value_tag)

            # if the suggested value is a code, and the entire code body is empty (i.e., not even function signature is present)
            # then we remove such suggestion
            keys_to_remove = []
            for key, value in extracted_data['variables'].items():
                if "__code" in key and value.strip() == "":
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del extracted_data['variables'][key]

            return extracted_data
        else:
            raise NotImplementedError(
                "If you supplied a custom output format prompt template, you need to implement your own response extractor")

    @property
    def default_prompt_symbols(self) -> Dict[str, str]:
        return {
            "variables": self.variables_section_title,
            "inputs": self.inputs_section_title,
            "outputs": self.outputs_section_title,
            "others": self.others_section_title,
            "feedback": self.feedback_section_title,
            "instruction": self.instruction_section_title,
            "code": self.code_section_title,
            "documentation": self.documentation_section_title,
            "context": self.context_section_title,
            "reasoning": self.reasoning_tag,
            "suggestion": self.suggestion_tag
        }


class OptimizerPromptSymbolSetJSON(OptimizerPromptSymbolSet):
    """We enforce a JSON output format extraction"""

    expect_json = True

    custom_output_format_instruction = dedent("""
    {
        "reasoning": <Your reasoning>,
        "suggestion": {
            <variable_1>: <suggested_value_1>,
            <variable_2>: <suggested_value_2>,
        }
    }
    """)

    def example_output(self, reasoning, variables):
        """
        reasoning: str
        variables: format {variable_name, value}
        """

        # Build the output string in the same JSON format as described in custom_output_format_instruction
        output = {
            "reasoning": reasoning,
            "suggestion": {var_name: value for var_name, value in variables.items()}
        }
        return json.dumps(output, indent=2)

    def output_response_extractor(self, response: str) -> Dict[str, Any]:
        """
        Extracts reasoning and suggestion variables from the LLM response using OptoPrime's extraction logic.
        """
        # Use the centralized extraction logic from OptoPrime
        suggestion_tag = self.default_prompt_symbols.get("suggestion", "suggestion")
        reasoning_tag = self.default_prompt_symbols.get("reasoning", "reasoning")

        ignore_extraction_error = True

        reasoning = "(Unable to extract, possibly due to parsing failure)"

        if "```" in response:
            # First try to extract from ```json ... ``` blocks
            json_match = re.findall(r"```json\s*(.*?)```", response, re.DOTALL)
            if len(json_match) > 0:
                response = json_match[0].strip()
            else:
                # Fall back to regular ``` ... ``` blocks
                match = re.findall(r"```(.*?)```", response, re.DOTALL)
                if len(match) > 0:
                    # Remove language identifier if present (e.g., "json", "python")
                    content = match[0].strip()
                    # Check if first line is a language identifier
                    lines = content.split('\n', 1)
                    if len(lines) > 1 and lines[0].strip().isalpha() and len(lines[0].strip()) < 20:
                        response = lines[1].strip()
                    else:
                        response = content

        json_extracted = {}
        suggestion = {}
        attempt_n = 0
        while attempt_n < 2:
            try:
                json_extracted = json.loads(response)
                if isinstance(json_extracted, dict):  # trim all whitespace keys in the json_extracted
                    json_extracted = {k.strip(): v for k, v in json_extracted.items()}
                suggestion = json_extracted.get(suggestion_tag, json_extracted)
                reasoning = json_extracted.get(reasoning_tag, "")
                break
            except json.JSONDecodeError:
                response = re.findall(r"{.*}", response, re.DOTALL)
                if len(response) > 0:
                    response = response[0]
                attempt_n += 1
            except Exception:
                attempt_n += 1

        if not isinstance(suggestion, dict):
            suggestion = json_extracted if isinstance(json_extracted, dict) else {}

        if len(suggestion) == 0:
            pattern = rf'"{suggestion_tag}"\s*:\s*\{{(.*?)\}}'
            suggestion_match = re.search(pattern, str(response), re.DOTALL)
            if suggestion_match:
                suggestion = {}
                suggestion_content = suggestion_match.group(1)
                pair_pattern = r'"([a-zA-Z0-9_]+)"\s*:\s*"(.*)"'
                pairs = re.findall(pair_pattern, suggestion_content, re.DOTALL)
                for key, value in pairs:
                    suggestion[key] = value

        if len(suggestion) == 0 and not ignore_extraction_error:
            print(f"Cannot extract {suggestion_tag} from LLM's response:\n{response}")

        keys_to_remove = []
        for key, value in suggestion.items():
            if "__code" in key and value.strip() == "":
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del suggestion[key]

        return {"reasoning": reasoning, "variables": suggestion}


class OptimizerPromptSymbolSet2(OptimizerPromptSymbolSet):
    variables_section_title = "# Variables"
    inputs_section_title = "# Inputs"
    outputs_section_title = "# Outputs"
    others_section_title = "# Others"
    feedback_section_title = "# Feedback"
    instruction_section_title = "# Instruction"
    code_section_title = "# Code"
    documentation_section_title = "# Documentation"
    context_section_title = "# Context"

    node_tag = "const"  # nodes that are constants in the graph
    variable_tag = "var"  # nodes that can be changed
    value_tag = "data"  # inside node, we have value tag
    constraint_tag = "constraint"  # inside node, we have constraint tag

    # output format
    reasoning_tag = "reason"
    improved_variable_tag = "var"
    name_tag = "name"


@dataclass
class FunctionFeedback:
    """Container for structured feedback from function execution traces.

    Used by OptoPrime to organize execution traces into a format suitable
    for LLM-based optimization.

    Attributes
    ----------
    graph : list[tuple[int, str]]
        Topologically sorted function calls with (depth, representation) pairs.
    documentation : dict[str, str]
        Mapping of function names to their documentation strings.
    others : dict[str, Any]
        Intermediate variables with (data, description) tuples.
    roots : dict[str, Any]
        Input/root variables with (data, description) tuples.
    output : dict[str, Any]
        Output/leaf variables with (data, description) tuples.
    user_feedback : Union[str, ContentBlockList]
        User-provided feedback about the execution. May include images.

    Notes
    -----
    This structure separates the execution trace into logical components
    that can be formatted into prompts for LLM-based optimization.
    """

    graph: List[
        Tuple[int, str]
    ]  # Each item is is a representation of function call. The items are topologically sorted.
    documentation: Dict[str, str]  # Function name and its documentationstring
    others: Dict[str, Any]  # Intermediate variable names and their data
    roots: Dict[str, Any]  # Root variable name and its data
    output: Dict[str, Any]  # Leaf variable name and its data
    user_feedback: Union[str, ContentBlockList]  # User feedback at the leaf of the graph (may include images)


@dataclass
class ProblemInstance:
    """Problem instance with multimodal content support.
    
    A composite of multiple ContentBlockLists representing different parts
    of a problem. Uses ContentBlockList for variables, inputs, others, and
    outputs to support both text and image content in a unified way.
    
    The class provides:
    - __repr__: Returns text-only representation for logging
    - to_content_blocks(): Returns ContentBlockList for multimodal prompts
    - has_images(): Check if any field contains images
    """
    instruction: str
    code: str
    documentation: str
    variables: ContentBlockList
    inputs: ContentBlockList
    others: ContentBlockList
    outputs: ContentBlockList
    feedback: ContentBlockList  # May contain images mixed with text
    context: Optional[ContentBlockList]

    optimizer_prompt_symbol_set: OptimizerPromptSymbolSet

    def __post_init__(self):
        # Normalize content fields so callers may pass plain strings (or None).
        # ContentBlockList.ensure is idempotent for existing ContentBlockLists.
        self.variables = ContentBlockList.ensure(self.variables)
        self.inputs = ContentBlockList.ensure(self.inputs)
        self.others = ContentBlockList.ensure(self.others)
        self.outputs = ContentBlockList.ensure(self.outputs)
        self.feedback = ContentBlockList.ensure(self.feedback)
        if self.context is not None:
            self.context = ContentBlockList.ensure(self.context)

    problem_template = dedent(
        """
        # Instruction
        {instruction}

        # Code
        {code}

        # Documentation
        {documentation}

        # Variables
        {variables}

        # Inputs
        {inputs}

        # Others
        {others}

        # Outputs
        {outputs}

        # Context
        {context}

        # Feedback
        {feedback}
        """
    )

    def __repr__(self) -> str:
        """Return text-only representation for backward compatibility.
        
        Uses ContentBlockList.to_text() for fields that may contain images.
        """
        optimization_query = self.problem_template.format(
            instruction=self.instruction,
            code=self.code,
            documentation=self.documentation,
            variables=self.variables.to_text(),
            inputs=self.inputs.to_text(),
            outputs=self.outputs.to_text(),
            others=self.others.to_text(),
            context=self.context.to_text() if self.context is not None else "",
            feedback=self.feedback.to_text()
        )

        return optimization_query

    def to_content_blocks(self) -> ContentBlockList:
        """Convert the problem instance to a list of ContentBlocks.
        
        Consecutive TextContent blocks are merged into a single block for efficiency.
        Images and other non-text blocks are kept separate.
        
        Returns:
            ContentBlockList: A list containing TextContent and ImageContent blocks
                that represent the complete problem instance including any images
                from variables, inputs, others, or outputs.
        """
        blocks = ContentBlockList()

        # Header sections (always text)
        header = dedent(f"""
        # Instruction
        {self.instruction}

        # Code
        {self.code}

        # Documentation
        {self.documentation}

        # Variables
        """)
        blocks.append(header)

        # Variables section (may contain images)
        blocks.extend(self.variables)

        # Inputs section
        blocks.append("\n\n# Inputs\n")
        blocks.extend(self.inputs)

        # Others section
        blocks.append("\n\n# Others\n")
        blocks.extend(self.others)

        # Outputs section
        blocks.append("\n\n# Outputs\n")
        blocks.extend(self.outputs)

        # Context section (optional)
        if self.context is not None and self.context.to_text().strip() != "":
            blocks.append(f"\n\n# Context\n") # section name
            blocks.extend(self.context) # extend the blocks

        # Feedback section (may contain images)
        blocks.append("\n\n# Feedback\n")
        blocks.extend(self.feedback)

        return blocks

    def has_images(self) -> bool:
        """Check if this problem instance contains any images.
        
        Efficiently checks each ContentBlockList field directly
        without building full content blocks.
        
        Returns:
            bool: True if any field contains ImageContent blocks.
        """
        return any(
            field.has_images()
            for field in [self.variables, self.inputs, self.others, self.outputs, self.feedback]
        )



    

# we provide two aliases for the Content class for semantic convenience
Context = Content
Feedback = Content

class OptoPrimeV3(OptoPrime):
    # This is generic representation prompt, which just explains how to read the problem.
    representation_prompt = dedent(
        """You're tasked to solve a coding/algorithm problem. You will see the instruction, the code, the documentation of each function used in the code, and the feedback about the execution result.

        Specifically, a problem will be composed of the following parts:
        - {instruction_section_title}: the instruction which describes the things you need to do or the question you should answer.
        - {code_section_title}: the code defined in the problem.
        - {documentation_section_title}: the documentation of each function used in #Code. The explanation might be incomplete and just contain high-level description. You can use the values in #Others to help infer how those functions work.
        - {variables_section_title}: the input variables that you can change/tweak (trainable).
        - {inputs_section_title}: the values of fixed inputs to the code, which CANNOT be changed (fixed).
        - {others_section_title}: the intermediate values created through the code execution.
        - {outputs_section_title}: the result of the code output.
        - {feedback_section_title}: the feedback about the code's execution result.
        - {context_section_title}: the context information that might be useful to solve the problem.

        In `{variables_section_title}`, `{inputs_section_title}`, `{outputs_section_title}`, and `{others_section_title}`, the format is:

        For variables we express as this:
        {variable_expression_format}

        If `data_type` is `code`, it means `{value_tag}` is the source code of a python code, which may include docstring and definitions."""
    )

    # Optimization
    default_objective = "You need to change the `{value_tag}` of the variables in {variables_section_title} to improve the output in accordance to {feedback_section_title}."

    output_format_prompt_template = dedent(
        """
        Output_format: Your output should be in the following XML or JSON format:

        {output_format}

        In <{reasoning_tag}>, explain the problem: 1. what the {instruction_section_title} means 2. what the {feedback_section_title} on {outputs_section_title} means to {variables_section_title} considering how {variables_section_title} are used in {code_section_title} and other values in {documentation_section_title}, {inputs_section_title}, {others_section_title}. 3. Reasoning about the suggested changes in {variables_section_title} (if needed) and the expected result.

        If you need to suggest a change in the values of {variables_section_title}, write down the suggested values in <{improved_variable_tag}>. Remember you can change only the values in {variables_section_title}, not others. When `type` of a variable is `code`, you should write the new definition in the format of python code without syntax errors, and you should not change the function name or the function signature.

        If no changes are needed, just output TERMINATE.
        """
    )

    example_problem_template = PromptTemplate(dedent(
        """
        Here is an example of problem instance and response:

        ================================
        {example_problem}
        ================================

        Your response:
        {example_response}
        """
    ))

    user_prompt_template = PromptTemplate(dedent(
        """
        Now you see problem instance:

        ================================
        {problem_instance}
        ================================

        """
    ))

    final_prompt = dedent(
        """
        What are your suggestions on variables {names}?

        Your response:
        """
    )

    def __init__(
            self,
            parameters: List[ParameterNode],
            llm: AbstractModel = None,
            *args,
            image_llm: AbstractModel = None,
            propagator: Propagator = None,
            objective: Union[None, str] = None,
            ignore_extraction_error: bool = True,
            # ignore the type conversion error when extracting updated values from LLM's suggestion
            include_example=False,
            memory_size=0,  # Memory size to store the past feedback
            max_tokens=8192,
            log=True,
            initial_var_char_limit=2000,
            optimizer_prompt_symbol_set: OptimizerPromptSymbolSet = OptimizerPromptSymbolSet(),
            use_json_object_format=True,  # whether to use json object format for the response when calling LLM
            truncate_expression=truncate_expression,
            problem_context: Optional[ContentBlockList] = None,
            **kwargs,
    ):
        super().__init__(parameters, *args, propagator=propagator, **kwargs)

        self.truncate_expression = truncate_expression
        self.problem_context: Optional[ContentBlockList] = problem_context
        self.output_contains_image = False

        self.use_json_object_format = use_json_object_format if optimizer_prompt_symbol_set.expect_json and use_json_object_format else False
        self.ignore_extraction_error = ignore_extraction_error
        self.llm = llm or LLM(mm_beta=True)
        self.image_llm = image_llm

        assert self.llm.mm_beta, "OptoPrimeV3 enables multi-modal LLM backbone by default. Please use LLM(model='...', mm_beta=True)."

        self.objective = objective or self.default_objective.format(value_tag=optimizer_prompt_symbol_set.value_tag,
                                                                    variables_section_title=optimizer_prompt_symbol_set.variables_section_title,
                                                                    feedback_section_title=optimizer_prompt_symbol_set.feedback_section_title)
        self.initial_var_char_limit = initial_var_char_limit
        self.optimizer_prompt_symbol_set = optimizer_prompt_symbol_set

        self.example_problem_summary = FunctionFeedback(graph=[(1, 'y = add(x=a,y=b)'), (2, "z = subtract(x=y, y=c)")],
                                                        documentation={'add': 'This is an add operator of x and y.',
                                                                       'subtract': "subtract y from x"},
                                                        others={'y': (6, None)},
                                                        roots={'a': (5, "a > 0"),
                                                               'b': (1, None),
                                                               'c': (5, None)},
                                                        output={'z': (1, None)},
                                                        user_feedback='The result of the code is not as expected. The result should be 10, but the code returns 1'
                                                        )
        self.example_problem_summary.variables = {'a': (5, "a > 0")}
        self.example_problem_summary.inputs = {'b': (1, None), 'c': (5, None)}

        self.example_problem = self.problem_instance(self.example_problem_summary)
        self.example_response = self.optimizer_prompt_symbol_set.example_output(
            reasoning="In this case, the desired response would be to change the value of input a to 14, as that would make the code return 10.",
            variables={
                'a': 10,
            }
        )

        self.include_example = include_example
        self.max_tokens = max_tokens
        self.log = [] if log else None
        self.summary_log = [] if log else None
        self.memory = FIFOBuffer(memory_size)
        self.conversation_history = Chat()
        self.conversation_length = memory_size  # Number of conversation turns to keep

        self.default_prompt_symbols = self.optimizer_prompt_symbol_set.default_prompt_symbols

        self.prompt_symbols = copy.deepcopy(self.default_prompt_symbols)
        self.initialize_instruct_prompt()

    def parameter_check(self, parameters: List[ParameterNode]):
        """Check if the parameters are valid.
        This can be overloaded by subclasses to add more checks.

        Args:
            parameters: List[ParameterNode]
                The parameters to check.
        
        Raises:
            AssertionError: If more than one parameter contains image data.
        
        Notes:
            OptoPrimeV3 supports image parameters, but only one parameter can be
            an image at a time since LLMs can only generate one image per inference.
        """
        # Count image parameters
        image_params = [param for param in parameters if param.is_image]

        if len(image_params) > 1:
            param_names = ', '.join([f"'{p.name}'" for p in image_params])
            raise AssertionError(
                f"OptoPrimeV3 supports at most one image parameter, but found {len(image_params)}: "
                f"{param_names}. LLMs can only generate one image at a time."
            )
        if len(image_params)  == 1:
            self.output_contains_image = True

    def add_context(self, *args, images: Optional[List[Any]] = None, format: str = "PNG"):
        """Add context to the optimizer, supporting both text and images.

        Two usage patterns are supported:

        **Usage 1: Variadic arguments (alternating text and images)**

            optimizer.add_context("text part 1", image_link, "text part 2", image_file)

        Each argument is either a string (text) or an image source.

        **Usage 2: Template with placeholders**

            optimizer.add_context(
                "text part 1 [IMAGE] text part 2 [IMAGE]",
                images=[image_link, image_file]
            )

        The text contains `[IMAGE]` placeholders that are replaced by images
        from the `images` list in order. The number of placeholders must match
        the number of images.
        
        Args:
            *args: Variable arguments. In Usage 1, alternating text and images.
                   In Usage 2, a single template string with placeholders.
            images: Optional list of image sources for Usage 2. Each can be:
                - URL string (http/https)
                - Local file path
                - PIL Image object
                - Numpy array
            format: Image format for numpy arrays (PNG, JPEG, etc.). Default: PNG
        
        Raises:
            ValueError: If using Usage 2 and the number of placeholders doesn't
                match the number of images.
        
        Examples:
            # Usage 1: Alternating text and images
            optimizer.add_context("Here's the diagram:", "diagram.png", "And here's another:", "other.png")
            
            # Usage 2: Template with placeholders
            optimizer.add_context("See [IMAGE] and compare with [IMAGE]", images=["a.png", "b.png"])
            
            # Text-only context
            optimizer.add_context("Important background information")
        """
        ctx = Content(*args, images=images, format=format)

        # Store the context
        if self.problem_context is None:
            self.problem_context = ctx
        else:
            # Append to existing context with a newline separator
            self.problem_context.append("\n\n")
            self.problem_context.extend(ctx.to_content_blocks())

    def initialize_instruct_prompt(self):
        self.representation_prompt = self.representation_prompt.format(
            variable_expression_format=dedent(f"""
            <{self.optimizer_prompt_symbol_set.variable_tag} name="variable_name" type="data_type">
            <{self.optimizer_prompt_symbol_set.value_tag}>
            value
            </{self.optimizer_prompt_symbol_set.value_tag}>
            <{self.optimizer_prompt_symbol_set.constraint_tag}>
            constraint_expression
            </{self.optimizer_prompt_symbol_set.constraint_tag}>
            </{self.optimizer_prompt_symbol_set.variable_tag}>
        """),
            value_tag=self.optimizer_prompt_symbol_set.value_tag,
            variables_section_title=self.optimizer_prompt_symbol_set.variables_section_title.replace(" ", ""),
            inputs_section_title=self.optimizer_prompt_symbol_set.inputs_section_title.replace(" ", ""),
            outputs_section_title=self.optimizer_prompt_symbol_set.outputs_section_title.replace(" ", ""),
            feedback_section_title=self.optimizer_prompt_symbol_set.feedback_section_title.replace(" ", ""),
            instruction_section_title=self.optimizer_prompt_symbol_set.instruction_section_title.replace(" ", ""),
            code_section_title=self.optimizer_prompt_symbol_set.code_section_title.replace(" ", ""),
            documentation_section_title=self.optimizer_prompt_symbol_set.documentation_section_title.replace(" ", ""),
            others_section_title=self.optimizer_prompt_symbol_set.others_section_title.replace(" ", ""),
            context_section_title=self.optimizer_prompt_symbol_set.context_section_title.replace(" ", "")
        )
        self.output_format_prompt = self.output_format_prompt_template.format(
            output_format=self.optimizer_prompt_symbol_set.output_format,
            reasoning_tag=self.optimizer_prompt_symbol_set.reasoning_tag,
            improved_variable_tag=self.optimizer_prompt_symbol_set.improved_variable_tag,
            instruction_section_title=self.optimizer_prompt_symbol_set.instruction_section_title.replace(" ", ""),
            feedback_section_title=self.optimizer_prompt_symbol_set.feedback_section_title.replace(" ", ""),
            outputs_section_title=self.optimizer_prompt_symbol_set.outputs_section_title.replace(" ", ""),
            code_section_title=self.optimizer_prompt_symbol_set.code_section_title.replace(" ", ""),
            documentation_section_title=self.optimizer_prompt_symbol_set.documentation_section_title.replace(" ", ""),
            variables_section_title=self.optimizer_prompt_symbol_set.variables_section_title.replace(" ", ""),
            inputs_section_title=self.optimizer_prompt_symbol_set.inputs_section_title.replace(" ", ""),
            others_section_title=self.optimizer_prompt_symbol_set.others_section_title.replace(" ", ""),
        )

    def repr_node_value(self, node_dict, node_tag="node",
                        value_tag="value", constraint_tag="constraint") -> str:
        """Returns text-only representation of node values (backward compatible)."""
        temp_list = []
        for k, v in node_dict.items():
            if "__code" not in k:
                # For images, use placeholder text
                value_repr = "[IMAGE]" if is_image(v[0]) else str(v[0])
                if v[1] is not None and node_tag == self.optimizer_prompt_symbol_set.variable_tag:
                    constraint_expr = f"<{constraint_tag}>\n{v[1]}\n</{constraint_tag}>"
                    temp_list.append(
                        f"<{node_tag} name=\"{k}\" type=\"{type(v[0]).__name__}\">\n<{value_tag}>\n{value_repr}\n</{value_tag}>\n{constraint_expr}\n</{node_tag}>\n")
                else:
                    temp_list.append(
                        f"<{node_tag} name=\"{k}\" type=\"{type(v[0]).__name__}\">\n<{value_tag}>\n{value_repr}\n</{value_tag}>\n</{node_tag}>\n")
            else:
                constraint_expr = f"<constraint>\n{v[1]}\n</constraint>"
                signature = v[1].replace("The code should start with:\n", "")
                func_body = v[0].replace(signature, "")
                temp_list.append(
                    f"<{node_tag} name=\"{k}\" type=\"code\">\n<{value_tag}>\n{signature}{func_body}\n</{value_tag}>\n{constraint_expr}\n</{node_tag}>\n")
        return "\n".join(temp_list)

    def repr_node_value_compact(self, node_dict, node_tag="node",
                                value_tag="value", constraint_tag="constraint") -> str:
        """Returns text-only compact representation of node values (backward compatible)."""
        temp_list = []
        for k, v in node_dict.items():
            if "__code" not in k:
                # For images, use placeholder text
                if is_image(v[0]):
                    node_value = "[IMAGE]"
                else:
                    node_value = self.truncate_expression(v[0], self.initial_var_char_limit)
                if v[1] is not None and node_tag == self.optimizer_prompt_symbol_set.variable_tag:
                    constraint_expr = f"<{constraint_tag}>\n{v[1]}\n</{constraint_tag}>"
                    temp_list.append(
                        f"<{node_tag} name=\"{k}\" type=\"{type(v[0]).__name__}\">\n<{value_tag}>\n{node_value}\n</{value_tag}>\n{constraint_expr}\n</{node_tag}>\n")
                else:
                    temp_list.append(
                        f"<{node_tag} name=\"{k}\" type=\"{type(v[0]).__name__}\">\n<{value_tag}>\n{node_value}\n</{value_tag}>\n</{node_tag}>\n")
            else:
                constraint_expr = f"<{constraint_tag}>\n{v[1]}\n</{constraint_tag}>"
                # we only truncate the function body
                signature = v[1].replace("The code should start with:\n", "")
                func_body = v[0].replace(signature, "")
                node_value = self.truncate_expression(func_body, self.initial_var_char_limit)
                temp_list.append(
                    f"<{node_tag} name=\"{k}\" type=\"code\">\n<{value_tag}>\n{signature}{node_value}\n</{value_tag}>\n{constraint_expr}\n</{node_tag}>\n")
        return "\n".join(temp_list)

    def repr_node_value_as_content_blocks(self, node_dict, node_tag="node",
                                          value_tag="value", constraint_tag="constraint") -> ContentBlockList:
        """Returns a ContentBlockList representing node values, including images.
        
        Consecutive TextContent blocks are merged for efficiency.
        For image values, the text before and after the image are separate blocks.
        """
        blocks = ContentBlockList()

        for k, v in node_dict.items():
            value_data = v[0]
            constraint = v[1]

            if "__code" not in k:
                # Check if this is an image
                image_content = value_to_image_content(value_data)

                if image_content is not None:
                    # Image node: output XML structure, then image, then closing
                    type_name = "image"
                    constraint_expr = f"<{constraint_tag}>\n{constraint}\n</{constraint_tag}>" if constraint is not None and node_tag == self.optimizer_prompt_symbol_set.variable_tag else ""

                    xml_text = f"<{node_tag} name=\"{k}\" type=\"{type_name}\">\n<{value_tag}>\n"
                    blocks.append(xml_text)
                    blocks.append(image_content)  # Image breaks the text flow

                    closing_text = f"\n</{value_tag}>\n{constraint_expr}</{node_tag}>\n\n" if constraint_expr else f"\n</{value_tag}>\n</{node_tag}>\n\n"
                    blocks.append(closing_text)
                else:
                    # Non-image node: text representation
                    if constraint is not None and node_tag == self.optimizer_prompt_symbol_set.variable_tag:
                        constraint_expr = f"<{constraint_tag}>\n{constraint}\n</{constraint_tag}>"
                        blocks.append(
                            f"<{node_tag} name=\"{k}\" type=\"{type(value_data).__name__}\">\n<{value_tag}>\n{value_data}\n</{value_tag}>\n{constraint_expr}\n</{node_tag}>\n\n"
                        )
                    else:
                        blocks.append(
                            f"<{node_tag} name=\"{k}\" type=\"{type(value_data).__name__}\">\n<{value_tag}>\n{value_data}\n</{value_tag}>\n</{node_tag}>\n\n"
                        )
            else:
                # Code node (never an image)
                constraint_expr = f"<{constraint_tag}>\n{constraint}\n</{constraint_tag}>"
                signature = constraint.replace("The code should start with:\n", "")
                func_body = value_data.replace(signature, "")
                blocks.append(
                    f"<{node_tag} name=\"{k}\" type=\"code\">\n<{value_tag}>\n{signature}{func_body}\n</{value_tag}>\n{constraint_expr}\n</{node_tag}>\n\n"
                )

        return blocks

    def repr_node_value_compact_as_content_blocks(self, node_dict, node_tag="node",
                                                  value_tag="value", constraint_tag="constraint") -> ContentBlockList:
        """Returns a ContentBlockList with compact representation, including images.
        
        Consecutive TextContent blocks are merged for efficiency.
        Non-image values are truncated. Images break the text flow.
        """
        blocks = ContentBlockList()

        for k, v in node_dict.items():
            value_data = v[0]
            constraint = v[1]

            if "__code" not in k:
                # Check if this is an image
                image_content = value_to_image_content(value_data)

                if image_content is not None:
                    # Image node: output XML structure, then image, then closing
                    type_name = "image"
                    constraint_expr = f"<{constraint_tag}>\n{constraint}\n</{constraint_tag}>" if constraint is not None and node_tag == self.optimizer_prompt_symbol_set.variable_tag else ""

                    xml_text = f"<{node_tag} name=\"{k}\" type=\"{type_name}\">\n<{value_tag}>\n"
                    blocks.append(xml_text)
                    blocks.append(image_content)  # Image breaks the text flow

                    closing_text = f"\n</{value_tag}>\n{constraint_expr}</{node_tag}>\n\n" if constraint_expr else f"\n</{value_tag}>\n</{node_tag}>\n\n"
                    blocks.append(closing_text)
                else:
                    # Non-image node: truncated text representation
                    node_value = self.truncate_expression(value_data, self.initial_var_char_limit)
                    if constraint is not None and node_tag == self.optimizer_prompt_symbol_set.variable_tag:
                        constraint_expr = f"<{constraint_tag}>\n{constraint}\n</{constraint_tag}>"
                        blocks.append(
                            f"<{node_tag} name=\"{k}\" type=\"{type(value_data).__name__}\">\n<{value_tag}>\n{node_value}\n</{value_tag}>\n{constraint_expr}\n</{node_tag}>\n\n"
                        )
                    else:
                        blocks.append(
                            f"<{node_tag} name=\"{k}\" type=\"{type(value_data).__name__}\">\n<{value_tag}>\n{node_value}\n</{value_tag}>\n</{node_tag}>\n\n"
                        )
            else:
                # Code node (never an image)
                constraint_expr = f"<{constraint_tag}>\n{constraint}\n</{constraint_tag}>"
                signature = constraint.replace("The code should start with:\n", "")
                func_body = value_data.replace(signature, "")
                node_value = self.truncate_expression(func_body, self.initial_var_char_limit)
                blocks.append(
                    f"<{node_tag} name=\"{k}\" type=\"code\">\n<{value_tag}>\n{signature}{node_value}\n</{value_tag}>\n{constraint_expr}\n</{node_tag}>\n\n"
                )

        return blocks

    def summarize(self):
        """Aggregate feedback from parameters into a structured summary.

        Collects and organizes feedback from all trainable parameters into
        a FunctionFeedback structure suitable for problem representation.

        Returns
        -------
        FunctionFeedback
            Structured feedback containing:
            - variables: Trainable parameters with values and descriptions
            - inputs: Non-trainable root nodes
            - graph: Topologically sorted function calls
            - others: Intermediate computation values
            - output: Final output values
            - documentation: Function documentation strings
            - user_feedback: Aggregated user feedback

        Notes
        -----
        The method performs several transformations:
        1. Aggregates feedback from all trainable parameters
        2. Converts the trace graph to FunctionFeedback structure
        3. Separates root nodes into variables (trainable) and inputs (non-trainable)
        4. Preserves the computation graph and intermediate values

        Parameters without feedback (disconnected from output) are still
        included in the summary but may not receive updates.
        """
        # Aggregate feedback from all the parameters
        feedbacks = [
            self.propagator.aggregate(node.feedback)
            for node in self.parameters
            if node.trainable
        ]
        summary = sum(feedbacks)  # TraceGraph
        # Construct variables and update others
        # Some trainable nodes might not receive feedback, because they might not be connected to the output
        summary = node_to_function_feedback(summary)
        # Classify the root nodes into variables and others
        # summary.variables = {p.py_name: p.data for p in self.parameters if p.trainable and p.py_name in summary.roots}

        trainable_param_dict = {p.py_name: p for p in self.parameters if p.trainable}
        summary.variables = {
            py_name: data
            for py_name, data in summary.roots.items()
            if py_name in trainable_param_dict
        }
        summary.inputs = {
            py_name: data
            for py_name, data in summary.roots.items()
            if py_name not in trainable_param_dict
        }  # non-variable roots

        return summary

    def construct_prompt(self, summary, mask=None, *args, **kwargs):
        """Construct the system and user prompt.

        The prompt for the optimizer agent is rather complex.
        There are prompts that are automatically constructed through the Trace frontend (aka the bundle/node API).
        However, we also allow the user to provide additional context to the optimizer agent.

        We handle multimodal (MM) conversion implicitly for the automatic part (TraceGraph),
        but we handle the user-provided context explicitly.

        Args:
            summary: The FunctionFeedback summary containing graph information.
            mask: List of section titles to exclude from the problem instance.
        
        Returns:
            Tuple of (system_prompt: str, user_prompt: ContentBlockList)
            - system_prompt is always a string
            - user_prompt is a ContentBlockList for multimodal support
        """
        system_prompt = (
                self.representation_prompt + self.output_format_prompt
        )  # generic representation + output rule

        problem_inst = self.problem_instance(summary, mask=mask)

        # Build user prompt as ContentBlockList (auto-merges consecutive text)
        user_content_blocks = ContentBlockList()

        # Add example if included
        if self.include_example:
            example_text = self.example_problem_template.format(
                example_problem=str(self.example_problem),  # Example is always text
                example_response=self.example_response,
            )
            user_content_blocks.append(example_text)

        # Add problem instance template
        # context is part of the problem instance
        user_content_blocks.append(self.user_prompt_template.format(
            problem_instance=problem_inst.to_content_blocks(),
        ))

        # Add final prompt
        var_names = ", ".join(k for k in summary.variables.keys())
        user_content_blocks.append(self.final_prompt.format(
            names=var_names,
        ))

        return system_prompt, user_content_blocks

    def problem_instance(self, summary: FunctionFeedback, mask=None):
        """Create a ProblemInstance from the summary.
        
        Args:
            summary: The FunctionFeedback summary containing graph information.
            mask: List of section titles to exclude from the problem instance.
        
        Returns:
            ProblemInstance with content block fields for multimodal support.
        """
        mask = mask or []

        # Use content block representations for multimodal support
        variables_content = (
            self.repr_node_value_as_content_blocks(
                summary.variables,
                node_tag=self.optimizer_prompt_symbol_set.variable_tag,
                value_tag=self.optimizer_prompt_symbol_set.value_tag,
                constraint_tag=self.optimizer_prompt_symbol_set.constraint_tag
            )
            if self.optimizer_prompt_symbol_set.variables_section_title not in mask
            else ContentBlockList()
        )

        # we add a temporary check here to ensure no more than 1 parameter is an image
        variable_stats = variables_content.count_blocks()
        if 'ImageContent' in variable_stats:
            assert variable_stats['ImageContent'] <= 1, "Currently we do not support generating multiple images (more than 1 parameter is an image)"
            self.output_contains_image = True

        inputs_content = (
            self.repr_node_value_compact_as_content_blocks(
                summary.inputs,
                node_tag=self.optimizer_prompt_symbol_set.node_tag,
                value_tag=self.optimizer_prompt_symbol_set.value_tag,
                constraint_tag=self.optimizer_prompt_symbol_set.constraint_tag
            )
            if self.optimizer_prompt_symbol_set.inputs_section_title not in mask
            else ContentBlockList()
        )
        outputs_content = (
            self.repr_node_value_compact_as_content_blocks(
                summary.output,
                node_tag=self.optimizer_prompt_symbol_set.node_tag,
                value_tag=self.optimizer_prompt_symbol_set.value_tag,
                constraint_tag=self.optimizer_prompt_symbol_set.constraint_tag
            )
            if self.optimizer_prompt_symbol_set.outputs_section_title not in mask
            else ContentBlockList()
        )
        others_content = (
            self.repr_node_value_compact_as_content_blocks(
                summary.others,
                node_tag=self.optimizer_prompt_symbol_set.node_tag,
                value_tag=self.optimizer_prompt_symbol_set.value_tag,
                constraint_tag=self.optimizer_prompt_symbol_set.constraint_tag
            )
            if self.optimizer_prompt_symbol_set.others_section_title not in mask
            else ContentBlockList()
        )

        return ProblemInstance(
            instruction=self.objective if "#Instruction" not in mask else "",
            code=(
                "\n".join([v for k, v in sorted(summary.graph)])
                if self.optimizer_prompt_symbol_set.inputs_section_title not in mask
                else ""
            ),
            documentation=(
                "\n".join([f"[{k}] {v}" for k, v in summary.documentation.items()])
                if self.optimizer_prompt_symbol_set.documentation_section_title not in mask
                else ""
            ),
            variables=variables_content,
            inputs=inputs_content,
            outputs=outputs_content,
            others=others_content,
            feedback=Content(summary.user_feedback) if self.optimizer_prompt_symbol_set.feedback_section_title not in mask else Content(""),
            context=self.problem_context,
            optimizer_prompt_symbol_set=self.optimizer_prompt_symbol_set
        )

    def _step(
            self, verbose=False, mask=None, *args, **kwargs
    ) -> Dict[ParameterNode, Any]:
        """Execute one optimization step.
        
        Args:
            verbose: If True, print prompts and responses.
            mask: List of section titles to exclude from the problem instance.
        
        Returns:
            Dictionary mapping parameters to their updated values.
        """
        assert isinstance(self.propagator, GraphPropagator)
        summary = self.summarize()

        system_prompt, user_content_blocks = self.construct_prompt(summary, mask=mask)

        response = self.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_content_blocks,
            verbose=verbose,
            max_tokens=self.max_tokens,
        )

        if "TERMINATE" in response.to_text():
            return {}

        suggestion = self.extract_llm_suggestion(response.to_text())
        update_dict = self.construct_update_dict(suggestion['variables'])
        # suggestion has two keys: reasoning, and variables

        # for update_dict, we manually update the image according to the variable name
        if response.get_images().has_images():
            images = response.get_images()
            assert len(images) == 1, "Currently we only allow at most one image parameter"
            # find the variable name
            image_param = [param for param in self.parameters if param.is_image][0]
            update_dict[image_param] = images[0].as_image() # parameter as PIL Image

        if self.log is not None:
            # For logging, use text representation
            log_user_prompt = str(self.problem_instance(summary))
            self.log.append(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": log_user_prompt,
                    "response": response,
                }
            )
            self.summary_log.append(
                {"problem_instance": self.problem_instance(summary), "summary": summary}
            )

        return update_dict

    def extract_llm_suggestion(self, response: str):
        """Extract the suggestion from the response."""

        suggestion = self.optimizer_prompt_symbol_set.output_response_extractor(response)

        if len(suggestion) == 0:
            if not self.ignore_extraction_error:
                print("Cannot extract suggestion from LLM's response:")
                print(response)

        return suggestion

    def call_llm(
            self,
            system_prompt: str,
            user_prompt: ContentBlockList,
            verbose: Union[bool, str] = False,
            max_tokens: int = 4096,
    ) -> AssistantTurn:
        """Call the LLM with a prompt and return the response.
        
        Args:
            system_prompt: The system prompt (always a string).
            user_prompt: The user prompt as ContentBlockList for multimodal content.
            verbose: If True, print the prompt and response. If "output", only print response.
            max_tokens: Maximum tokens in the response.
        
        Returns:
            assistant_turn: AssistantTurn object
        """
        if verbose not in (False, "output"):
            # Print text portions, indicate if images present
            text_parts = [block.text for block in user_prompt if isinstance(block, TextContent)]
            has_images = any(isinstance(block, ImageContent) for block in user_prompt)
            suffix = f" [+ {DEFAULT_IMAGE_PLACEHOLDER}]" if has_images else ""
            print("Prompt\n", system_prompt + "".join(text_parts) + suffix)

        # Update system prompt in conversation history
        self.conversation_history.system_prompt = system_prompt

        # Create user turn with content
        user_turn = UserTurn(user_prompt)
        self.conversation_history.add_user_turn(user_turn)

        # Get messages with conversation length control (truncate from start)
        # conversation_length = n historical rounds (user+assistant pairs) to keep
        # The current user turn is automatically included by to_messages()
        messages = self.conversation_history.to_messages(
            n=self.conversation_length if self.conversation_length > 0 else -1,
            truncate_strategy="from_start",
            model_name=self.llm.model_name
        )

        # Bedrock doesn't support response_format natively - LiteLLM adds tools which breaks the response
        _is_bedrock = hasattr(self.llm, 'model_name') and is_bedrock_model(self.llm.model_name)
        response_format = {"type": "json_object"} if (self.use_json_object_format and not _is_bedrock) else None

        # Prepare common arguments
        llm_kwargs = {"messages": messages, "max_tokens": max_tokens, "response_format": response_format}
        
        # Add image generation tool only for non-Gemini models when output contains image
        if self.output_contains_image and 'gemini' not in self.llm.model_name:
            llm_kwargs["tools"] = [{"type": "image_generation"}]
        
        assistant_turn = self.llm(**llm_kwargs)

        if verbose:
            print("LLM response:\n", assistant_turn)

        self.conversation_history.add_assistant_turn(assistant_turn)

        return assistant_turn

    def save(self, path: str):
        """Save the optimizer state to a file."""
        with open(path, 'wb') as f:
            pickle.dump({
                "truncate_expression": self.truncate_expression,
                "use_json_object_format": self.use_json_object_format,
                "ignore_extraction_error": self.ignore_extraction_error,
                "objective": self.objective,
                "initial_var_char_limit": self.initial_var_char_limit,
                "optimizer_prompt_symbol_set": self.optimizer_prompt_symbol_set,
                "include_example": self.include_example,
                "max_tokens": self.max_tokens,
                "memory": self.memory,
                "conversation_history": self.conversation_history,
                "conversation_length": self.conversation_length,
                "default_prompt_symbols": self.default_prompt_symbols,
                "prompt_symbols": self.prompt_symbols,
                "representation_prompt": self.representation_prompt,
                "output_format_prompt": self.output_format_prompt,
            }, f)

    def load(self, path: str):
        """Load the optimizer state from a file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
            self.truncate_expression = state["truncate_expression"]
            self.use_json_object_format = state["use_json_object_format"]
            self.ignore_extraction_error = state["ignore_extraction_error"]
            self.objective = state["objective"]
            self.initial_var_char_limit = state["initial_var_char_limit"]
            self.optimizer_prompt_symbol_set = state["optimizer_prompt_symbol_set"]
            self.include_example = state["include_example"]
            self.max_tokens = state["max_tokens"]
            self.memory = state["memory"]
            self.conversation_history = state.get("conversation_history", Chat())
            self.conversation_length = state.get("conversation_length", 0)
            self.default_prompt_symbols = state["default_prompt_symbols"]
            self.prompt_symbols = state["prompt_symbols"]
            self.representation_prompt = state["representation_prompt"]
            self.output_format_prompt = state["output_format_prompt"]

import json
from typing import Any, List, Dict, Union, Tuple
from dataclasses import dataclass, asdict
from opto.optimizers.optoprime import OptoPrime, FunctionFeedback
from opto.trace.utils import dedent

from opto.trace.nodes import ParameterNode, Node, MessageNode
from opto.trace.propagators import TraceGraph, GraphPropagator
from opto.trace.propagators.propagators import Propagator

from opto.utils.llm import AbstractModel, LLM
from opto.optimizers.buffers import FIFOBuffer
import copy

import re
from typing import Dict, Any


def extract_top_level_blocks(text: str, tag: str):
    """Extract all top-level <tag>...</tag> blocks from text."""
    blocks = []
    start_tag = f'<{tag}>'
    end_tag = f'</{tag}>'
    stack = []
    start = None
    i = 0
    while i < len(text):
        if text.startswith(start_tag, i):
            if not stack:
                start = i + len(start_tag)
            stack.append(i)
            i += len(start_tag)
        elif text.startswith(end_tag, i):
            if stack:
                stack.pop()
                if not stack and start is not None:
                    blocks.append(text[start:i])
                    start = None
            i += len(end_tag)
        else:
            i += 1
    return blocks


def extract_first_top_level_block(text: str, tag: str):
    blocks = extract_top_level_blocks(text, tag)
    return blocks[0] if blocks else None


def strip_nested_blocks(text: str, tag: str) -> str:
    """Remove all nested <tag>...</tag> blocks from text, leaving only the top-level text."""
    result = ''
    start_tag = f'<{tag}>'
    end_tag = f'</{tag}>'
    stack = []
    i = 0
    last = 0
    while i < len(text):
        if text.startswith(start_tag, i):
            if not stack:
                result += text[last:i]
            stack.append(i)
            i += len(start_tag)
        elif text.startswith(end_tag, i):
            if stack:
                stack.pop()
                if not stack:
                    last = i + len(end_tag)
            i += len(end_tag)
        else:
            i += 1
    if not stack:
        result += text[last:]
    return result.strip()


def extract_reasoning_and_remainder(text: str, tag: str = "reasoning"):
    """Extract reasoning and the remainder of the text after reasoning block (if closed). Strip whitespace only if properly closed."""
    start_tag = f'<{tag}>'
    end_tag = f'</{tag}>'
    start = text.find(start_tag)
    if start == -1:
        return '', text
    start += len(start_tag)
    end = text.find(end_tag, start)
    if end == -1:
        # If not properly closed, don't strip whitespace to preserve original formatting
        return text[start:], ''
    return text[start:end].strip(), text[end + len(end_tag):]


def extract_xml_like_data(text: str, reasoning_tag: str = "reasoning",
                          improved_variable_tag: str = "variable",
                          name_tag: str = "name",
                          value_tag: str = "value") -> Dict[str, Any]:
    """
    Extract thinking content and improved variables from text containing XML-like tags.

    Args:
        text (str): Text containing <reasoning> and <variable> tags

    Returns:
        Dict containing:
        - 'reasoning': content of <reasoning> element
        - 'variables': dict mapping variable names to their values
    """
    result = {
        'reasoning': '',
        'variables': {}
    }

    # Extract reasoning and the remainder of the text
    reasoning, remainder = extract_reasoning_and_remainder(text, reasoning_tag)
    result['reasoning'] = reasoning

    # Only parse variables from the remainder (i.e., after a closed reasoning tag)
    variable_blocks = extract_top_level_blocks(remainder, improved_variable_tag)
    for var_block in variable_blocks:
        name_block = extract_first_top_level_block(var_block, name_tag)
        value_block = extract_first_top_level_block(var_block, value_tag)
        # Only add if both name and value tags are present and name is non-empty after stripping
        if name_block is not None and value_block is not None:
            var_name = name_block.strip()
            var_value = value_block.strip() if value_block is not None else ''
            if var_name:  # Only require name to be non-empty, value can be empty
                result['variables'][var_name] = var_value
    return result


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

    node_tag = "node"  # nodes that are constants in the graph
    variable_tag = "variable"  # nodes that can be changed
    value_tag = "value"  # inside node, we have value tag
    constraint_tag = "constraint"  # inside node, we have constraint tag

    # output format
    # Note: we currently don't support extracting format's like "```code```" because we assume supplied tag is name-only, i.e., <tag_name></tag_name>
    reasoning_tag = "reasoning"
    improved_variable_tag = "variable"
    name_tag = "name"

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
        }

class OptimizerPromptSymbolSetJSON(OptimizerPromptSymbolSet):
    """We enforce a JSON output format extraction"""

    expect_json = True

    custom_output_format_instruction = """
    {{
        "reasoning": <Your reasoning>,
        "suggestion": {{
            <variable_1>: <suggested_value_1>,
            <variable_2>: <suggested_value_2>,
        }}
    }}
    """

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
        reasoning = ""
        suggestion_tag = "suggestion"

        if "```" in response:
            response = response.replace("```", "").strip()

        suggestion = {}
        attempt_n = 0
        while attempt_n < 2:
            try:
                suggestion = json.loads(response)[suggestion_tag]
                reasoning = json.loads(response)[self.reasoning_tag]
                break
            except json.JSONDecodeError:
                # Remove things outside the brackets
                response = re.findall(r"{.*}", response, re.DOTALL)
                if len(response) > 0:
                    response = response[0]
                attempt_n += 1
            except Exception:
                attempt_n += 1

        if not isinstance(suggestion, dict):
            suggestion = {}

        if len(suggestion) == 0:
            # we try to extract key/value separately and return it as a dictionary
            pattern = rf'"{suggestion_tag}"\s*:\s*\{{(.*?)\}}'
            suggestion_match = re.search(pattern, str(response), re.DOTALL)
            if suggestion_match:
                suggestion = {}
                # Extract the entire content of the suggestion dictionary
                suggestion_content = suggestion_match.group(1)
                # Regex to extract each key-value pair;
                # This scheme assumes double quotes but is robust to missing commas at the end of the line
                pair_pattern = r'"([a-zA-Z0-9_]+)"\s*:\s*"(.*)"'
                # Find all matches of key-value pairs
                pairs = re.findall(pair_pattern, suggestion_content, re.DOTALL)
                for key, value in pairs:
                    suggestion[key] = value

        if len(suggestion) == 0:
            print(f"Cannot extract suggestion from LLM's response:")
            print(response)

        # if the suggested value is a code, and the entire code body is empty (i.e., not even function signature is present)
        # then we remove such suggestion
        keys_to_remove = []
        for key, value in suggestion.items():
            if "__code" in key and value.strip() == "":
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del suggestion[key]

        extracted_data = {"reasoning": reasoning,
                          "variables": suggestion}

        return extracted_data

class OptimizerPromptSymbolSet2(OptimizerPromptSymbolSet):
    variables_section_title = "# Variables"
    inputs_section_title = "# Inputs"
    outputs_section_title = "# Outputs"
    others_section_title = "# Others"
    feedback_section_title = "# Feedback"
    instruction_section_title = "# Instruction"
    code_section_title = "# Code"
    documentation_section_title = "# Documentation"

    node_tag = "const"  # nodes that are constants in the graph
    variable_tag = "var"  # nodes that can be changed
    value_tag = "data"  # inside node, we have value tag
    constraint_tag = "constraint"  # inside node, we have constraint tag

    # output format
    reasoning_tag = "reason"
    improved_variable_tag = "var"
    name_tag = "name"


@dataclass
class ProblemInstance:
    instruction: str
    code: str
    documentation: str
    variables: str
    inputs: str
    others: str
    outputs: str
    feedback: str

    optimizer_prompt_symbol_set: OptimizerPromptSymbolSet

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

        # Feedback
        {feedback}
        """
    )

    def __repr__(self) -> str:
        return self.replace_symbols(self.problem_template.format(
            instruction=self.instruction,
            code=self.code,
            documentation=self.documentation,
            variables=self.variables,
            inputs=self.inputs,
            outputs=self.outputs,
            others=self.others,
            feedback=self.feedback,
        ), self.optimizer_prompt_symbol_set.default_prompt_symbols)

    def replace_symbols(self, text: str, symbols: Dict[str, str]) -> str:
        default_prompt_symbols = {
            "variables": "# Variables",
            "constraints": "# Constraints",
            "inputs": "# Inputs",
            "outputs": "# Outputs",
            "others": "# Others",
            "feedback": "# Feedback",
            "instruction": "# Instruction",
            "code": "# Code",
            "documentation": "# Documentation",
        }

        for k, v in symbols.items():
            text = text.replace(default_prompt_symbols[k], v)
        return text

def truncate_expression(value, limit):
    # https://stackoverflow.com/questions/1436703/what-is-the-difference-between-str-and-repr
    value = str(value)
    if len(value) > limit:
        return value[:limit] + "...(skipped due to length limit)"
    return value

class OptoPrimeV2(OptoPrime):
    # This is generic representation prompt, which just explains how to read the problem.
    representation_prompt = dedent(
        """
        You're tasked to solve a coding/algorithm problem. You will see the instruction, the code, the documentation of each function used in the code, and the feedback about the execution result.

        Specifically, a problem will be composed of the following parts:
        - {instruction_section_title}: the instruction which describes the things you need to do or the question you should answer.
        - {code_section_title}: the code defined in the problem.
        - {documentation_section_title}: the documentation of each function used in #Code. The explanation might be incomplete and just contain high-level description. You can use the values in #Others to help infer how those functions work.
        - {variables_section_title}: the input variables that you can change/tweak (trainable).
        - {inputs_section_title}: the values of fixed inputs to the code, which CANNOT be changed (fixed).
        - {others_section_title}: the intermediate values created through the code execution.
        - {outputs_section_title}: the result of the code output.
        - {feedback_section_title}: the feedback about the code's execution result.

        In `{variables_section_title}`, `{inputs_section_title}`, `{outputs_section_title}`, and `{others_section_title}`, the format is:

        For variables we express as this:
        {variable_expression_format}
        
        If `data_type` is `code`, it means `{value_tag}` is the source code of a python code, which may include docstring and definitions.
        """
    )

    # Optimization
    default_objective = "You need to change the `{value_tag}` of the variables in {variables_section_title} to improve the output in accordance to {feedback_section_title}."

    output_format_prompt_template = dedent(
        """
        Output_format: Your output should be in the following XML/HTML format:
        
        ```
        {output_format}
        ```

        In <{reasoning_tag}>, explain the problem: 1. what the {instruction_section_title} means 2. what the {feedback_section_title} on {outputs_section_title} means to {variables_section_title} considering how {variables_section_title} are used in {code_section_title} and other values in {documentation_section_title}, {inputs_section_title}, {others_section_title}. 3. Reasoning about the suggested changes in {variables_section_title} (if needed) and the expected result.

        If you need to suggest a change in the values of {variables_section_title}, write down the suggested values in <{improved_variable_tag}>. Remember you can change only the values in {variables_section_title}, not others. When `type` of a variable is `code`, you should write the new definition in the format of python code without syntax errors, and you should not change the function name or the function signature.

        If no changes are needed, just output TERMINATE.
        """
    )

    example_problem_template = dedent(
        """
        Here is an example of problem instance and response:

        ================================
        {example_problem}
        ================================

        Your response:
        {example_response}
        """
    )

    user_prompt_template = dedent(
        """
        Now you see problem instance:

        ================================
        {problem_instance}
        ================================

        """
    )

    example_prompt = dedent(
        """

        Here are some feasible but not optimal solutions for the current problem instance. Consider this as a hint to help you understand the problem better.

        ================================

        {examples}

        ================================
        """
    )

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
            propagator: Propagator = None,
            objective: Union[None, str] = None,
            ignore_extraction_error: bool = True,
            # ignore the type conversion error when extracting updated values from LLM's suggestion
            include_example=False,
            memory_size=0,  # Memory size to store the past feedback
            max_tokens=4096,
            log=True,
            initial_var_char_limit=100,
            optimizer_prompt_symbol_set: OptimizerPromptSymbolSet = OptimizerPromptSymbolSet(),
            use_json_object_format=True,  # whether to use json object format for the response when calling LLM
            truncate_expression=truncate_expression,
            **kwargs,
    ):
        super().__init__(parameters, *args, propagator=propagator, **kwargs)

        self.truncate_expression = truncate_expression

        self.use_json_object_format = use_json_object_format if optimizer_prompt_symbol_set.expect_json and use_json_object_format else False
        self.ignore_extraction_error = ignore_extraction_error
        self.llm = llm or LLM()
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

        self.default_prompt_symbols = self.optimizer_prompt_symbol_set.default_prompt_symbols

        self.prompt_symbols = copy.deepcopy(self.default_prompt_symbols)
        self.initialize_prompt()

    def initialize_prompt(self):
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
            others_section_title=self.optimizer_prompt_symbol_set.others_section_title.replace(" ", "")
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
            others_section_title=self.optimizer_prompt_symbol_set.others_section_title.replace(" ", "")
        )

    @staticmethod
    def repr_node_value(node_dict):
        temp_list = []
        for k, v in node_dict.items():
            if "__code" not in k:
                constraint_expr = f"<constraint> ({type(v[0]).__name__}) {k}: {v[1]} </constraint>"
                temp_list.append(
                    f"<node name=\"{k}\" type=\"{type(v[0]).__name__}\">\n<value>{v[0]}</value>\n{constraint_expr}\n</node>\n")
            else:
                constraint_expr = f"<constraint>\n{v[1]}\n</constraint>"
                temp_list.append(
                    f"<node name=\"{k}\" type=\"code\">\n<value>\n{v[0]}\n</value>\n{constraint_expr}\n</node>\n")
        return "\n".join(temp_list)

    def repr_node_value_compact(self, node_dict, node_tag="node",
                                value_tag="value", constraint_tag="constraint"):
        temp_list = []
        for k, v in node_dict.items():
            if "__code" not in k:
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

    def construct_prompt(self, summary, mask=None, *args, **kwargs):
        """Construct the system and user prompt."""
        system_prompt = (
                self.representation_prompt + self.output_format_prompt
        )  # generic representation + output rule
        user_prompt = self.user_prompt_template.format(
            problem_instance=str(self.problem_instance(summary, mask=mask))
        )  # problem instance
        if self.include_example:
            user_prompt = (
                    self.example_problem_template.format(
                        example_problem=self.example_problem,
                        example_response=self.example_response,
                    )
                    + user_prompt
            )

        var_names = []
        for k, v in summary.variables.items():
            var_names.append(f"{k}")  # ({type(v[0]).__name__})
        var_names = ", ".join(var_names)

        user_prompt += self.final_prompt.format(names=var_names)

        # Add examples
        if len(self.memory) > 0:
            formatted_final = self.final_prompt.format(names=var_names)
            prefix = user_prompt.split(formatted_final)[0]
            examples = []
            for variables, feedback in self.memory:
                examples.append(
                    json.dumps(
                        {
                            "variables": {k: v[0] for k, v in variables.items()},
                            "feedback": feedback,
                        },
                        indent=4,
                    )
                )
            examples = "\n".join(examples)
            user_prompt = (
                    prefix
                    + f"\nBelow are some variables and their feedbacks you received in the past.\n\n{examples}\n\n"
                    + formatted_final
            )
        self.memory.add((summary.variables, summary.user_feedback))

        return system_prompt, user_prompt

    def problem_instance(self, summary, mask=None):
        mask = mask or []
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
            variables=(
                self.repr_node_value_compact(summary.variables, node_tag=self.optimizer_prompt_symbol_set.variable_tag,
                                             value_tag=self.optimizer_prompt_symbol_set.value_tag,
                                             constraint_tag=self.optimizer_prompt_symbol_set.constraint_tag)
                if self.optimizer_prompt_symbol_set.variables_section_title not in mask
                else ""
            ),
            inputs=(
                self.repr_node_value_compact(summary.inputs, node_tag=self.optimizer_prompt_symbol_set.node_tag,
                                             value_tag=self.optimizer_prompt_symbol_set.value_tag,
                                             constraint_tag=self.optimizer_prompt_symbol_set.constraint_tag) if self.optimizer_prompt_symbol_set.inputs_section_title not in mask else ""
            ),
            outputs=(
                self.repr_node_value_compact(summary.output, node_tag=self.optimizer_prompt_symbol_set.node_tag,
                                             value_tag=self.optimizer_prompt_symbol_set.value_tag,
                                             constraint_tag=self.optimizer_prompt_symbol_set.constraint_tag) if self.optimizer_prompt_symbol_set.outputs_section_title not in mask else ""
            ),
            others=(
                self.repr_node_value_compact(summary.others, node_tag=self.optimizer_prompt_symbol_set.node_tag,
                                             value_tag=self.optimizer_prompt_symbol_set.value_tag,
                                             constraint_tag=self.optimizer_prompt_symbol_set.constraint_tag) if self.optimizer_prompt_symbol_set.others_section_title not in mask else ""
            ),
            feedback=summary.user_feedback if self.optimizer_prompt_symbol_set.feedback_section_title not in mask else "",
            optimizer_prompt_symbol_set=self.optimizer_prompt_symbol_set
        )

    def _step(
            self, verbose=False, mask=None, *args, **kwargs
    ) -> Dict[ParameterNode, Any]:
        assert isinstance(self.propagator, GraphPropagator)
        summary = self.summarize()
        system_prompt, user_prompt = self.construct_prompt(summary, mask=mask)

        response = self.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            verbose=verbose,
            max_tokens=self.max_tokens,
        )

        if "TERMINATE" in response:
            return {}

        suggestion = self.extract_llm_suggestion(response)
        update_dict = self.construct_update_dict(suggestion['variables'])
        # suggestion has two keys: reasoning, and variables

        if self.log is not None:
            self.log.append(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
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
            user_prompt: str,
            verbose: Union[bool, str] = False,
            max_tokens: int = 4096,
    ):
        """Call the LLM with a prompt and return the response."""
        if verbose not in (False, "output"):
            print("Prompt\n", system_prompt + user_prompt)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response_format = {"type": "json_object"} if self.use_json_object_format else None

        response = self.llm(messages=messages, max_tokens=max_tokens, response_format=response_format)

        response = response.choices[0].message.content

        if verbose:
            print("LLM response:\n", response)
        return response

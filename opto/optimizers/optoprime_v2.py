import json
from typing import Any, List, Dict, Union, Tuple
from textwrap import dedent, indent
from dataclasses import dataclass, asdict
from opto.optimizers.optoprime import OptoPrime, ProblemInstance

from opto.trace.nodes import ParameterNode, Node, MessageNode
from opto.trace.propagators import TraceGraph, GraphPropagator
from opto.trace.propagators.propagators import Propagator

from opto.utils.llm import AbstractModel, LLM
from opto.optimizers.buffers import FIFOBuffer
import copy

class OptoPrimeV2(OptoPrime):
    # This is generic representation prompt, which just explains how to read the problem.
    representation_prompt = dedent(
        """
        You're tasked to solve a coding/algorithm problem. You will see the instruction, the code, the documentation of each function used in the code, and the feedback about the execution result.

        Specifically, a problem will be composed of the following parts:
        - #Instruction: the instruction which describes the things you need to do or the question you should answer.
        - #Code: the code defined in the problem.
        - #Documentation: the documentation of each function used in #Code. The explanation might be incomplete and just contain high-level description. You can use the values in #Others to help infer how those functions work.
        - #Variables: the input variables that you can change.
        - #Constraints: the constraints or descriptions of the variables in #Variables.
        - #Inputs: the values of other inputs to the code, which are not changeable.
        - #Others: the intermediate values created through the code execution.
        - #Outputs: the result of the code output.
        - #Feedback: the feedback about the code's execution result.

        In #Variables, #Inputs, #Outputs, and #Others, the format is:

        <node>
        (data_type) variable_name = value
        </node>

        If `(data_type)` is `code`, it means `{value}` is the source code of a python code, which may include docstring and definitions.
        """
    )

    # Optimization
    default_objective = "You need to change the `value` of the variables in #Variables to improve the output in accordance to #Feedback."

    output_format_prompt = dedent(
        """
        Output_format: Your output should be in the following XML/HTML format:
        
        <think>
        Your reasoning on why you made the decision to suggest a new value. You can also use it to explain why you didn't 
        </think>
        
        <improved_variable>
            <name>variable_1_name</name>
            <value>
                new_value
                ...
            </value>
        </improved_variable>
        
        <improved_variable>
            <name>variable_2_name</name>
            <value>
                new_value
                ...
            </value>
        </improved_variable>

        In <think>, explain the problem: 1. what the #Instruction means 2. what the #Feedback on #Output means to #Variables considering how #Variables are used in #Code and other values in #Documentation, #Inputs, #Others. 3. Reasoning about the suggested changes in #Variables (if needed) and the expected result.

        If you need to suggest a change in the values of #Variables, write down the suggested values in <improved_variable>. Remember you can change only the values in #Variables, not others. When <type> of a variable is (code), you should write the new definition in the format of python code without syntax errors, and you should not change the function name or the function signature.

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

    default_prompt_symbols = {
        "variables": "#Variables",
        "constraints": "#Constraints",
        "inputs": "#Inputs",
        "outputs": "#Outputs",
        "others": "#Others",
        "feedback": "#Feedback",
        "instruction": "#Instruction",
        "code": "#Code",
        "documentation": "#Documentation",
    }

    def __init__(
        self,
        parameters: List[ParameterNode],
        llm: AbstractModel = None,
        *args,
        propagator: Propagator = None,
        objective: Union[None, str] = None,
        ignore_extraction_error: bool = True,  # ignore the type conversion error when extracting updated values from LLM's suggestion
        include_example=False,  # TODO # include example problem and response in the prompt
        memory_size=0,  # Memory size to store the past feedback
        max_tokens=4096,
        log=True,
        prompt_symbols=None,
        **kwargs,
    ):
        super().__init__(parameters, *args, propagator=propagator, **kwargs)
        self.ignore_extraction_error = ignore_extraction_error
        self.llm = llm or LLM()
        self.objective = objective or self.default_objective
        self.example_problem = ProblemInstance.problem_template.format(
            instruction=self.default_objective,
            code="y = add(x=a,y=b)\nz = subtract(x=y, y=c)",
            documentation="add: add x and y \nsubtract: subtract y from x",
            variables="(int) a = 5",
            constraints="a: a > 0",
            outputs="(int) z = 1",
            others="(int) y = 6",
            inputs="(int) b = 1\n(int) c = 5",
            feedback="The result of the code is not as expected. The result should be 10, but the code returns 1",
            stepsize=1,
        )
        self.example_response = dedent(
            """
            <think>
            In this case, the desired response would be to change the value of input a to 14, as that would make the code return 10.
            </think>
            
            <improved_variable>
                <name>a</name>
                <value>
                    10
                </value>
            </improved_variable>
            """
        )

        self.include_example = include_example
        self.max_tokens = max_tokens
        self.log = [] if log else None
        self.summary_log = [] if log else None
        self.memory = FIFOBuffer(memory_size)
        self.prompt_symbols = copy.deepcopy(self.default_prompt_symbols)
        if prompt_symbols is not None:
            self.prompt_symbols.update(prompt_symbols)

    @staticmethod
    def repr_node_value(node_dict):
        temp_list = []
        for k, v in node_dict.items():
            if "__code" not in k:
                temp_list.append(f"<node>\n({type(v[0]).__name__}) {k}={v[0]}\n</node>")
            else:
                temp_list.append(f"<node>\n(code) {k}:{v[0]}\n</node>")
        return "\n".join(temp_list)

    @staticmethod
    def repr_node_constraint(node_dict):
        temp_list = []
        for k, v in node_dict.items():
            if "__code" not in k:
                if v[1] is not None:
                    temp_list.append(f"<constraint>\n({type(v[0]).__name__}) {k}: {v[1]}\n</constraint>")
            else:
                if v[1] is not None:
                    temp_list.append(f"<constraint>\n(code) {k}: {v[1]}\n</constraint>")
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
            prefix = user_prompt.split(self.final_prompt)[0]
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
                    + self.final_prompt
            )
        self.memory.add((summary.variables, summary.user_feedback))

        return system_prompt, user_prompt

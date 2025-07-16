import datasets
import numpy as np
from opto import trace
from opto.utils.llm import LLM, LiteLLM
from opto.optimizers import OptoPrime
from opto.trainer.algorithms.search_algorithms import UCBSearch as SearchAlgorithm
from opto.trainer.loggers import WandbLogger
from opto.trainer.guide import VerbalJudgeGuide
from typing import Any


@trace.model
class Learner:
    """ A basic LLM agent. """

    def __init__(self, system_prompt: str = "You're a helpful agent",
                 user_prompt_template: str = "Query: {message}",
                 llm: LLM = None):
        self.system_prompt = trace.node(system_prompt, trainable=True)
        self.user_prompt_template = trace.node(user_prompt_template)
        self.llm = llm or LLM()

    @trace.bundle()
    def model(self, system_prompt: str, user_prompt_template: str, message: str) -> str:
        """Call the LLM model.

        Args:
            system_prompt: the system prompt to the agent. By tuning this prompt, we can control the behavior of the agent. For example, it can be used to provide instructions to the agent (such as how to reason about the problem, how to answer the question), or provide in-context examples of how to solve the problem.
            user_prompt_template: the user prompt template to the agent. It is used as formatting the input to the agent as user_prompt_template.format(message=message).
            message: the input to the agent. It can be a query, a task, a code, etc.
        Returns:
            The response from the agent.
        """

        if '{message}' not in user_prompt_template:
            raise ValueError("user_prompt_template must contain '{message}'")

        response = self.llm(
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt_template.format(message=message)}]
        )
        return response.choices[0].message.content

    def forward(self, message: Any) -> Any:
        """ Forward pass of the agent. """
        return self.model(self.system_prompt, self.user_prompt_template, message)


Guide = VerbalJudgeGuide
Logger = WandbLogger

def main():
    # set seed
    seed = 42
    num_epochs = 1
    batch_size = 1
    eval_frequency = -1
    num_threads = 3
    verbose = True
    teacher_model = "vertex_ai/gemini-2.0-flash"  # use default model
    student_model = "vertex_ai/gemini-2.0-flash"  # use default model
    optimizer_model = "vertex_ai/gemini-2.0-flash"  # use default model

    np.random.seed(seed)

    # In this example, we use the GSM8K dataset, which is a dataset of math word problems.
    # We will look the training error of the agent on a small portion of this dataset.
    train_dataset = datasets.load_dataset('openai/gsm8k', 'main')['train'][:10]
    train_dataset = dict(inputs=train_dataset['question'], infos=train_dataset['answer'])
    test_dataset = train_dataset

    agent = Learner(llm=LLM(student_model))
    guide = Guide(llm=LLM(teacher_model))
    optimizer = OptoPrime(agent.parameters(), llm=LLM(optimizer_model))
    logger = Logger(project="gsm8k-examples", name="ucb",verbose=verbose)
             # set use_json_object_format=False if LLM does not support JSON object format

    alg = SearchAlgorithm(
            agent=agent,
            optimizer=optimizer,
            logger=logger)

    alg.train(guide,
              train_dataset,
              num_epochs=num_epochs,
              batch_size=batch_size,
              eval_frequency=eval_frequency,
              test_dataset=test_dataset,
              num_threads=num_threads,
              verbose='output' if verbose else False)


if __name__ == "__main__":
    main()

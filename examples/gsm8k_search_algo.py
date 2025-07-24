import datasets
import numpy as np
from opto import trace
from opto.utils.llm import LLM, LiteLLM
from opto.optimizers import OptoPrimeV2 as OptoPrime
from opto.trainer.algorithms.priority_search import PrioritySearch as SearchAlgorithm
from opto.trainer.loggers import TensorboardLogger
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
Logger = TensorboardLogger


def main():
    # set seed
    seed = 42
    num_epochs = 1
    batch_size = 3
    sub_batch_size = 2
    score_range = (0, 1)  # range of the score for the guide
    eval_frequency = -1
    num_eval_samples = 2
    num_threads = 10
    datasize = 5
    verbose = True
    teacher_model = None  # use default model
    student_model = None  # use default model
    optimizer_model = None  # use default model

    np.random.seed(seed)

    # In this example, we use the GSM8K dataset, which is a dataset of math word problems.
    # We will look the training error of the agent on a small portion of this dataset.
    train_dataset = datasets.load_dataset('openai/gsm8k', 'main')['train'][:datasize]
    train_dataset = dict(inputs=train_dataset['question'], infos=train_dataset['answer'])
    test_dataset = train_dataset

    agent = Learner(llm=LLM(student_model))
    guide = Guide(llm=LLM(teacher_model))
    optimizer = OptoPrime(agent.parameters(), llm=LLM(optimizer_model))
    logger = Logger(verbose=verbose)
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
              sub_batch_size=sub_batch_size,
              score_range=score_range,
              num_eval_samples=num_eval_samples,
              verbose='output' if verbose else False)


if __name__ == "__main__":
    main()

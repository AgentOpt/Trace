import datasets
import numpy as np
from opto import trace
from opto.utils.llm import LLM, LiteLLM
from opto.optimizers.utils import print_color
from opto.optimizers import OptoPrime
from opto.trainer.algorithms.basic_algorithms import BatchedFeedback
from opto.trainer.guide import VerbalJudgeGuide
from typing import Any


@trace.model
class Learner:
    # A basic LLM agent.

    def __init__(self, system_prompt: str = "You're a helpful agent",
                 user_prompt_template: str = "Query: {message}",
                 llm: LLM = None):
        self.system_prompt = trace.node(system_prompt, trainable=True)
        self.user_prompt_template = trace.node(user_prompt_template)
        self.llm = llm or LLM()

    @trace.bundle()
    def model(self, system_prompt: str, user_prompt_template: str, message: str) -> str:
        """ Call the LLM model. system_prompt specifies
        the behavior of the agent. user prompt is the input to the agent, which
        is formatted as user_prompt_template.format(message=message)."""

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

class Logger:
    def log(self, *messages, color=None, **kwargs):
        print_color(messages, color=color)


def main():
    # set seed
    seed = 42
    num_epochs = 1
    batch_size = 1
    eval_frequency = 1
    teacher_model = "gpt-4o-mini" #"gpt-4o-mini_2024-07-18"
    student_model = "gpt-35-turbo_1106"

    np.random.seed(seed)

    train_dataset = datasets.load_dataset('openai/gsm8k', 'main')['train'][
                    :10]  # NOTE for now, we train on a smaller portion
    train_dataset = dict(inputs=train_dataset['question'], infos=train_dataset['answer'])
    test_dataset = train_dataset  # NOTE for now, we just look at training error

    agent = Learner(llm=LiteLLM(model="gpt-3.5-turbo"))

    guide = VerbalJudgeGuide(model=teacher_model)

    alg = BatchedFeedback(agent=agent,
                          optimizer=OptoPrime(agent.parameters()),
                          logger=Logger())

    alg.train(guide,
              train_dataset,
              num_epochs=num_epochs,
              batch_size=batch_size,
              eval_frequency=eval_frequency,
              test_dataset=test_dataset,
              num_threads=3)


if __name__ == "__main__":
    main()

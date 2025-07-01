from opto.trainer.utils import batch_run
from opto.trace import ExecutionError
import copy
import numpy as np


def evaluate(agent, guide, inputs, infos, min_score=None, num_samples=1, num_threads=None, description=None):
    """ Evaluate the agent on the inputs and return the scores

    Args:
        agent: The agent to evaluate
        guide: The guide to use for evaluation
        inputs: List of inputs to evaluate on
        infos: List of additional information for each input
        min_score: Minimum score to return when an exception occurs
        num_samples: Number of samples to use to evaluate each input
        num_threads: Maximum number of threads to use for parallel evaluation
        description: Description to display in the progress bar
    """
    assert len(inputs) == len(infos), "Inputs and infos must have the same length"
    N = len(inputs)
    # Use provided description or generate a default one
    eval_description = description or f"Evaluating {N} examples"

    @batch_run(max_workers=num_threads, description=eval_description)
    def _evaluate(agent, guide, i):
        try:
            output = agent(inputs[i]).data
            score = guide.metric(inputs[i], output, infos[i])
        except ExecutionError as e:
            score = min_score
        return score

    # repeat each index num_samples times
    indices = [i for i in range(N) for _ in range(num_samples)]

    # Run the evaluation in parallel
    scores = _evaluate(agent, guide, indices)
    scores = np.array(scores)
    if num_samples > 1:
        # scores will be of length N * num_samples
        # Reshape scores into an array of shape (N, num_samples)        
        scores = scores.reshape(N, num_samples)
    return scores
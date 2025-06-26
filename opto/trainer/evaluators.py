from opto.trainer.utils import async_run
import copy


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

    def evaluate_single(agent, guide, i):
        try:
            output = agent(inputs[i]).data
            score = guide.metric(inputs[i], output, infos[i])
        except:
            score = min_score
        return score

    N = len(inputs)
    assert len(inputs) == len(infos), "Inputs and infos must have the same length"
    # Use asyncio if num_threads is not None and > 1
    use_asyncio = num_threads is not None and num_threads > 1

    # repeat each index num_samples times
    indices = [i for i in range(N) for _ in range(num_samples)]
    if use_asyncio:
        # Use provided description or generate a default one
        eval_description = description or f"Evaluating {N} examples"
        scores = async_run([evaluate_single] * N, [(copy.deepcopy(agent), copy.deepcopy(guide), i) for i in indices],
                          max_workers=num_threads,
                          description=eval_description) # list of tuples
    else:
        scores = [evaluate_single(agent, guide, i) for i in indices]
    return scores
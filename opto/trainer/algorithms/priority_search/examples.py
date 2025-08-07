
from opto.trainer.algorithms.priority_search import PrioritySearch
from typing import Union, Optional

# Below we define several algorithms that use the PrioritySearch class.


class SequentialUpdate(PrioritySearch):
    """ A basic algorithm that explores the parameter space and proposes new candidates one by one.

        This is realized by setting

            num_candidates = 1
            num_proposals = 1
            memory_size = 1

        This is the same as MinibatchAlgorithm when
            1. no validation set is provided
            2. sub_batch_size is None or batch_size.

        validate_proposals here acts the same as `ensure_improvement` flag in MinibatchAlgorithm
    """

    def train(self,
              guide, # guide to provide feedback
              train_dataset,  # dataset of (x, info) pairs to train the agent
              *,
              # validation
              validate_dataset = None, # same format as train_dataset; if None use the current batch.
              validate_guide = None,  #  to provide scores for the validation set
              # training loop
              batch_size = 1,  # batch size for updating the agent
              sub_batch_size = None,  # sub-batch size that each optimizer attends to
              score_range = None,  # minimum score to update the agent
              num_epochs = 1,  # number of training epochs
              num_threads = None,  # maximum number of threads to use
              verbose = False,  # whether to print the output of the agent
              # evaluation
              test_dataset = None, # dataset of (x, info) pairs to evaluate the agent
              test_frequency: Union[int, None] = 1, # frequency of evaluation
              num_eval_samples: int = 1,  # number of samples to use to evaluate each input
              # logging
              log_frequency = None,  # frequency of logging
              save_frequency: Union[int, None] = None,  # frequency of saving the agent
              save_path: str = "checkpoints/agent.pkl",  # path to save the agent
              # Priority Search specific parameters
              num_candidates: int = 10,  # number of candidates to propose for exploration
              num_proposals: int = 1,  # number of proposals to generate per optimizer
              default_score: float = float('inf'),  # default score assigned to priority queue candidates
              validate_proposals: bool = True,  # whether to validate the proposed parameters
              memory_size: Optional[int] = None,  # size of the heap memory to store the candidates; if None, no limit is set
              # Additional keyword arguments
              **kwargs
              ):

        num_candidates = 1  # SequentialSearch only proposes one candidate at a time
        num_proposals = 1  # SequentialSearch only generates one proposal at a time
        memory_size = 1  # SequentialSearch only stores one candidate at a time in the heap memory
        # validate_proposals is the same as `ensure_improvement` flag in MinibatchAlgorithm

        return super().train(guide, train_dataset,
                      validate_dataset=validate_dataset,
                      validate_guide=validate_guide,
                      batch_size=batch_size,
                      sub_batch_size=sub_batch_size,
                      score_range=score_range,
                      num_epochs=num_epochs,
                      num_threads=num_threads,
                      verbose=verbose,
                      test_dataset=test_dataset,
                      test_frequency=test_frequency,
                      num_eval_samples=num_eval_samples,
                      log_frequency=log_frequency,
                      save_frequency=save_frequency,
                      save_path=save_path,
                      num_candidates=num_candidates,
                      num_proposals=num_proposals,
                      default_score=default_score,
                      validate_proposals=validate_proposals,
                      memory_size=memory_size, **kwargs)


class SequentialSearch(PrioritySearch):
    """ A sequential search that generates one candidate in each iteration by validating multiple proposals.

        This is realized by setting
            num_proposals = 1
            memory_size = 1

        This is the same as BasicSearchAlgorithm when
            1. a validation set is provided
            2. validate_proposals is True.
            3. sub_batch_size is None or batch_size.
    """

    def train(self,
              guide, # guide to provide feedback
              train_dataset,  # dataset of (x, info) pairs to train the agent
              *,
              # validation
              validate_dataset = None, # same format as train_dataset; if None use the current batch.
              validate_guide = None,  #  to provide scores for the validation set
              # training loop
              batch_size = 1,  # batch size for updating the agent
              sub_batch_size = None,  # sub-batch size that each optimizer attends to
              score_range = None,  # minimum score to update the agent
              num_epochs = 1,  # number of training epochs
              num_threads = None,  # maximum number of threads to use
              verbose = False,  # whether to print the output of the agent
              # evaluation
              test_dataset = None, # dataset of (x, info) pairs to evaluate the agent
              test_frequency: Union[int, None] = 1, # frequency of evaluation
              num_eval_samples: int = 1,  # number of samples to use to evaluate each input
              # logging
              log_frequency = None,  # frequency of logging
              save_frequency: Union[int, None] = None,  # frequency of saving the agent
              save_path: str = "checkpoints/agent.pkl",  # path to save the agent
              # Priority Search specific parameters
              num_candidates: int = 10,  # number of candidates to propose for exploration
              num_proposals: int = 1,  # number of proposals to generate per optimizer
              default_score: float = float('inf'),  # default score assigned to priority queue candidates
              validate_proposals: bool = True,  # whether to validate the proposed parameters
              memory_size: Optional[int] = None,  # size of the heap memory to store the candidates; if None, no limit is set
              # Additional keyword arguments
              **kwargs
              ):

        num_candidates = 1  # SequentialSearch only generates one candidate at a time
        memory_size = 1  # MultiSequentialUpdate only stores one candidate at a time in the heap memory
        # validate_proposals is the same as `ensure_improvement` flag in MinibatchAlgorithm

        return super().train(guide, train_dataset,
                      validate_dataset=validate_dataset,
                      validate_guide=validate_guide,
                      batch_size=batch_size,
                      sub_batch_size=sub_batch_size,
                      score_range=score_range,
                      num_epochs=num_epochs,
                      num_threads=num_threads,
                      verbose=verbose,
                      test_dataset=test_dataset,
                      test_frequency=test_frequency,
                      num_eval_samples=num_eval_samples,
                      log_frequency=log_frequency,
                      save_frequency=save_frequency,
                      save_path=save_path,
                      num_candidates=num_candidates,
                      num_proposals=num_proposals,
                      default_score=default_score,
                      validate_proposals=validate_proposals,
                      memory_size=memory_size, **kwargs)

class BeamSearch(PrioritySearch):
    """ A beam search algorithm that explores the parameter space and proposes new candidates based on the best candidates in the priority queue.

        This is realized by setting
            num_proposals = beam_size
            memory_size = beam_size

    """

    def train(self,
              guide, # guide to provide feedback
              train_dataset,  # dataset of (x, info) pairs to train the agent
              *,
              # validation
              validate_dataset = None, # same format as train_dataset; if None use the current batch.
              validate_guide = None,  #  to provide scores for the validation set
              # training loop
              batch_size = 1,  # batch size for updating the agent
              sub_batch_size = None,  # sub-batch size that each optimizer attends to
              score_range = None,  # minimum score to update the agent
              num_epochs = 1,  # number of training epochs
              num_threads = None,  # maximum number of threads to use
              verbose = False,  # whether to print the output of the agent
              # evaluation
              test_dataset = None, # dataset of (x, info) pairs to evaluate the agent
              test_frequency: Union[int, None] = 1, # frequency of evaluation
              num_eval_samples: int = 1,  # number of samples to use to evaluate each input
              # logging
              log_frequency = None,  # frequency of logging
              save_frequency: Union[int, None] = None,  # frequency of saving the agent
              save_path: str = "checkpoints/agent.pkl",  # path to save the agent
              # Priority Search specific parameters
              num_candidates: int = 10,  # number of candidates to propose for exploration
              num_proposals: int = 1,  # number of proposals to generate per optimizer; this is beam_size in beam search.
              default_score: float = float('inf'),  # default score assigned to priority queue candidates
              validate_proposals: bool = True,  # whether to validate the proposed parameters
              memory_size: Optional[int] = None,  # size of the heap memory to store the candidates; if None, no limit is set
              **kwargs):

        # num_candidates acts as the beam size in beam search.
        memory_size = num_candidates

        return super().train(guide, train_dataset,
                       validate_dataset=validate_dataset,
                       validate_guide=validate_guide,
                       batch_size=batch_size,
                       sub_batch_size=sub_batch_size,
                       score_range=score_range,
                       num_epochs=num_epochs,
                       num_threads=num_threads,
                       verbose=verbose,
                       test_dataset=test_dataset,
                       test_frequency=test_frequency,
                       num_eval_samples=num_eval_samples,
                       log_frequency=log_frequency,
                       save_frequency=save_frequency,
                       save_path=save_path,
                       num_candidates=num_candidates,  # beam size
                       num_proposals=num_proposals,  # number of proposals to generate per optimizer
                       default_score=default_score,
                       validate_proposals=validate_proposals,
                       memory_size=memory_size, **kwargs)

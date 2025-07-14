import numpy as np
import copy
import heapq
from dataclasses import dataclass
from typing import Union, List, Tuple, Dict, Any, Optional
from opto import trace
from opto.trace.nodes import ParameterNode
from opto.trainer.utils import async_run, batch_run
from opto.optimizers.utils import print_color
from opto.trainer.algorithms.basic_algorithms import Minibatch, AlgorithmBase, batchify, standard_forward
from opto.trainer.evaluators import evaluate
from opto.trainer.loader import DataLoader


# TODO save and load SearchAlgorithm
# TODO async version
# TODO create SYNC and ASYNC versions of the base class; add an attribute to the class to indicate
# TODO a better data structure to store samples

# update_dict

# Some helper function to convert between trace.Module and update_dict


def standard_forward(agent, x, guide, info, min_score=0):
    """ Forward and compute feedback.

        Args:
            agent: trace.Module
            x: input
            guide: (question, student_answer, info) -> score, feedback
            info: additional information for the guide
            min_score: minimum score when exception happens

        Returns:
            target: output of the agent
            score: score from the guide
            feedback: feedback from the guide
        """
    try:
        target = agent(x)
        score, feedback = guide(x, target.data, info)
    except trace.ExecutionError as e:
        target = e.exception_node
        score, feedback = min_score, target.create_feedback('full')
    return target, score, feedback
def is_node_copy(a, b):
    # check if a is a copy of b or b is a copy of a
    # For int:0, its deepcopied version is int0_copy:x
    """ Check if a is a copy of b or b is a copy of a or if they are the same node."""
    if a.name == b.name:
        return True
    if '_copy' in a.name and (a.name.split(':')[0].replace('_copy', '') ==  b.py_name):
        return True
    if '_copy' in b.name and (b.name.split(':')[0].replace('_copy', '') == a.py_name):
        return True
    return False

def is_module_copy(a, b):
    """ Check if a and b (trace.Modules) are copies of each other. """
    parameters_a = a.parameters()
    parameters_b = b.parameters()
    # Check if all parameters of a are copies of b or vice versa
    for p_a in parameters_a:
        if not any(is_node_copy(p_a, p_b) for p_b in parameters_b):
            return False
    for p_b in parameters_b:
        if not any(is_node_copy(p_b, p_a) for p_a in parameters_a):
            return False
    return True

def remap_update_dict(base_module, update_dict):
    """ Remap the update dict to the agent's parameters. update_dict might have keys which are copies of the base_module's parameters or visa versa.
        This function remaps the keys in update_dict to the original parameters of the base_module.

        The return dict is empty if no keys in update_dict matched any parameters of the base_module. This condition can be used to check if the update_dict contains non-trivial updates.
    """
    parameters = base_module.parameters()  # get the parameters of the base agent
    remapped_update_dict = {}
    for k, v in update_dict.items():
        for p in parameters:
            # Check if k is a copy of p or p is a copy of k
            if is_node_copy(k, p):
                k = p  # remap k to the original parameter
                remapped_update_dict[k] = v  # set the value in the remapped update dict
                break  # stop checking once we've found a match
    # remapped_update_dict is empty if no keys in update_dict matched any parameters of the base_module
    return remapped_update_dict

def set_module_parameters(agent, update_dict):
    """ Set the parameters of the agent based on the update_dict.
        The update_dict is a dictionary of ParameterNode: value pairs.
        The agent's parameters will be updated with the values from the update_dict.
    """
    remap_update_dict = remap_update_dict(agent, update_dict)  # remap the update dict to the agent's parameters
    for k, v in remap_update_dict.items():
        k._data = v  # set the parameter's data to the value in the update_dict

def create_module_from_update_dict(agent, update_dict):
    """ Create a new agent from the update_dict.
        The update_dict is a dictionary of ParameterNode: value pairs.
        A new agent will be created with the parameters set to the values from the update_dict.
    """
    new_agent = copy.deepcopy(agent) #.copy()  # create a copy of the agent
    set_module_parameters(new_agent, update_dict)  # set the parameters of the new agent
    return new_agent  # return the new agent


# a1, a2, a3, a4
# x1, x2, x3, x4
# a11 (x1, x2)
# a12 (x3, x4)
# a21 (x1, x2)
# a22 (x3, x4)

# N agents, M inputs
# N x M

# A list (size len(agents)) with list of samples (size batchsize) for each agent,
#                 where each sample is a dict containing:
#                 - 'module': the trace.Module (proposal)
#                 - 'x': the input data
#                 - 'info': additional information about the input
#                 - 'target': the target output (if applicable)
#                 - 'score': the score of the proposal
#                 - 'feedback': the feedback from the guide

#TODO naming
@dataclass
class Rollout:
    """ A rollout is a single sample from the environment. It contains the module, input, info, target, score, and feedback.
        This is used to store the results of the agent's evaluation on a single input.
    """
    module: trace.Module  # the trace.Module (proposal)
    x: Any  # the input data
    info: Any  # additional information about the input
    target: trace.Node  # the target output (if applicable)
    score: float  # the score of the proposal
    feedback: Any  # the feedback from the guide

    def to_dict(self):
        """ Convert the rollout to a dictionary representation. """
        return {
            "module": self.module,
            "x": self.x,
            "info": self.info,
            "target": self.target.data,
            "score": self.score,
            "feedback": self.feedback,
        }

class Subgraph:
    """ A subgraph is a collection of rollouts generated by the same agent (trace.Module) on different inputs.
    """
    module: trace.Module  # the trace.Module (proposal) that generated the rollouts
    rollouts: List[Rollout]  # a list of Rollout objects generated by the module on different inputs
    def __init__(self, rollouts):
        """ Initialize a subgraph with the given rollouts. """
        # Check that all rollouts have the same module
        if not all(rollouts[0].module == r.module for r in rollouts):
            raise ValueError("All rollouts must have the same module.")
        self.module = rollouts[0].module  # the module is the same for all rollouts
        self.rollouts = rollouts

    def get_scores(self):
        """ Get the scores of the rollouts in the subgraph. """
        return [r.score for r in self.rollouts]

    def __len__(self):
        """ Get the number of rollouts in the subgraph. """
        return len(self.rollouts)

    def __iter__(self):
        """ Iterate over the rollouts in the subgraph. """
        return iter(self.rollouts)

    def extend(self, other):
        """ Extend the subgraph with another subgraph. """
        if not isinstance(other, Subgraph):
            raise ValueError("Can only extend with another Subgraph.")
        if self.module != other.module:
            raise ValueError("Cannot extend with a subgraph with a different module.")
        self.rollouts.extend(other.rollouts)

    def to_list(self):
        """ Convert the subgraph to a list of rollouts. """
        return [r.to_dict() for r in self.rollouts]





# # TODO general broadcast decorator
# def broadcast_forward(num_threads=1, description=None, sub_batch_size=None):
#     """ A decorator to broadcast the agents, xs, infos, and guides.

#         forward should be a function that takes the arguments in the following order:
#             agent: trace.Module, the agent to evaluate
#             x: input, the input to the agent
#             info: additional information for each input
#             guide: a single guide or a list of guides that provide feedback on the outputs
#             min_score: float, minimum score when exception happens
#             **kwargs: additional keyword arguments to pass to the forward function
#         Returns:
#             A wrapper function that takes agents, xs, infos, guides, min_score, and additional keyword arguments.
#             The wrapper function will broadcast the agents, inputs, infos, and guides.

#             agents is expected to be a list of trace.Modules representing the agents.
#             xs and infos are expected to be lists of the same length of batch size.
#             guide can be a single guide or a list of guides of the same length as the number of agents.

#             The return of the wrapper function is a list of Subgraph objects, where each Subgraph contains a list of Rollout objects.
#     """

#     def decorator(forward):
#         """ A decorator to broadcast the agents, inputs, infos, and guides. """
#         def wrapper(agents, xs, infos, guides, min_score=0., **kwargs):
#             """ A wrapper to broadcast the agents, inputs, infos, and guides to match the batch size. """

#             # Example:
#             # agents : a1, a2
#             # inputs: x1, x2, x3
#             # infos: i1, i2, i3
#             # sub_batch_size: 2

#             # The forward is called in this order:
#             # (a1, x1, i1, guide1),
#             # (a1, x2, i2, guide1),
#             # (deepcopy(a1), x3, i3, guide1)
#             # (a2, x1, i1, guide2),
#             # (a2, x2, i2, guide2),
#             # (deepcopy(a2), x3, i3, guide2)


#             batch_size = len(xs)
#             n_agents = len(agents)
#             assert len(infos) == batch_size, "Length of infos must match length of xs."


#             # broadcasted_agents = [proposal for proposal in agents for _ in range(batch_size)]  # [a1, a1, a2, a2, ...]

#             # Broadcast the agents to match the batch size
#             # [a1, a1,        a1,        a1,        a1, ..., a2, a2,        a2, ...] if sub_batch_size is not specified
#             # [a1, a1, a1_copy_1, a1_copy_1, a1_copy_2, ..., a2, a2, a2_copy_1, ...] if sub_batch_size of 2 is specified
#             sub_batch_size = sub_batch_size or batch_size  # if sub_batch_size is not provided, use the batch size
#             broadcasted_agents = []
#             for agent in agents:
#                 for i in range(batch_size):
#                     if i % sub_batch_size == 0 and i > 0:
#                         agent = copy.deepcopy(agent) # create a copy of the agent for the next sub-batch
#                     broadcasted_agents.append(agent)

#             # broadcast the inputs and infos to match the number of agents
#             # [x1, x2, x3, ..., x1, x2, x3, ...]
#             broadcasted_xs = [x for _ in range(n_agents) for x in xs]
#             broadcasted_infos = [info for _ in range(n_agents) for info in infos]

#             # Broadcast the guides to match the batch size
#             if isinstance(guides, list):
#                 assert len(guides) == n_agents, "If guides is a list, its length must match the number of agents."
#                 # If multiple guides are provided, broadcast each guide to match the batch size
#                 broadcasted_guides = [guide for guide in guides for _ in range(batch_size)]
#             else:  # If a single guide is provided, broadcast it to match the batch size
#                 broadcasted_guides = [guides for _ in range(n_agents * batch_size)]

#             description = description or f"Evaluating {n_agents} agents on {batch_size} inputs"

#             # Forward the agent on the inputs and compute the feedback using the guide
#             forward = batch_run(max_workers=num_threads, description=description)(forward)
#             _outputs = forward(broadcasted_agents,
#                               broadcasted_xs,
#                               broadcasted_infos,
#                               broadcasted_guides,
#                               min_score=min_score,
#                               **kwargs)  # guide will be broadcasted inside as well
#             # return list of (target, score, feedback)


#             return outputs

#         return wrapper
#     return decorator


class SearchAlgorithm(AlgorithmBase):
    """ This implements a generic template for search algorithm. """

    def __init__(self,
                 agent,
                 optimizer,
                 num_threads: int = None,   # maximum number of threads to use for parallel execution
                 logger=None,
                 *args,
                 **kwargs,
                 ):
        super().__init__(agent, num_threads=num_threads, logger=logger, *args, **kwargs)
        self.optimizer = optimizer
        self.n_iters = 0  # number of iterations

    def train(self,
              guide, # guide to provide feedback
              train_dataset,  # dataset of (x, info) pairs to train the agent
              *,
              # validation
              validate_dataset = None, # same format as train_dataset; if None use the current batch.
              validate_guide = None,  #  to provide scores for the validation set
              # training loop
              batch_size = 1,  # batch size for updating the agent
              sub_batch_size = None,  # sub-batch size for broadcasting the agents
              score_range = None,  # minimum score to update the agent
              num_epochs = 1,  # number of training epochs
              num_threads = None,  # maximum number of threads to use
              verbose = False,  # whether to print the output of the agent
              # evaluation
              test_dataset = None, # dataset of (x, info) pairs to evaluate the agent
              test_guide = None, # guide to provide scores for the test set
            #   test_frequency: Union[int, None] = 1, # frequency of evaluation
               eval_frequency: Union[int, None] = 1,  # frequency of evaluation
              num_eval_samples: int = 1,  # number of samples to use to evaluate each input
              # logging
              log_frequency = None,  # frequency of logging
              save_frequency: Union[int, None] = None,  # frequency of saving the agent
              save_path: str = "checkpoints/agent.pkl",  # path to save the agent
              **kwargs
              ):

        ## Setup
        # TODO legacy notation
        test_frequency = eval_frequency  # use eval_frequency as test_frequency

        log_frequency = log_frequency or test_frequency  # frequency of logging (default to test_frequency)
        self.num_threads = num_threads or self.num_threads  # Use provided num_threads or fall back to self.num_threads
        test_dataset = test_dataset or train_dataset  # default to train_dataset if test_dataset is not provided
        test_guide = test_guide or guide
        self.num_eval_samples = num_eval_samples  # number of samples to use to evaluate each input
        self.score_range = score_range or (0., 1.)
        # Underscore attributes are temporary attributes for the algorithm (which will not be saved)
        # They would not affect the agent's state or the training process.
        self._loader = DataLoader(train_dataset, batch_size=batch_size)  # default data loader for training
        self.sub_batch_size = sub_batch_size  # sub-batch size for broadcasting the agents
        self._guide = guide
        self._validate_dataset = validate_dataset
        self._validate_guide = validate_guide or guide

        # Evaluate the agent before learning
        # NOTE set test_frequency < 0 to skip first evaluation
        if (test_frequency is not None) and test_frequency > 0:
            info_test = self.test(test_dataset, test_guide)
            self.log(info_test)

        # Save the agent before learning if save_frequency > 0
        if (save_frequency is not None) and save_frequency > 0:
            self.save(save_path)

        samples = None
        self.n_epochs = 0 # number of epochs (full passes over the dataset) performed by the algorithm (This is incremented in sample)
        self.n_samples = 0 # number of training samples processed by the algorithm (This is incremented in sample)
        train_scores = []  # to store the scores of the agent during training

        while self.n_epochs < num_epochs :

            print(f"Epoch: {self.n_epochs}. Iteration: {self.n_iters}")

            # 1. Propose new parameters given the current state of the algorithm
            # proposals: list of trace.Modules
            update_dict, proposals, info_update = self.update(samples, verbose=verbose, **kwargs)
            self.optimizer.update(update_dict)  # update the agent with the proposed parameters

            # 2. Get feedback on the proposed parameters on the current batch
            # samples: list of list of dict(module, x, info, target, score, feedback)
            samples, info_sample = self.sample(proposals, verbose=verbose, **kwargs)

            # Evaluate the agent after update
            if (test_frequency is not None) and (self.n_iters % test_frequency == 0):
                info_test = self.test(test_dataset, test_guide)
                self.log(info_test, prefix="Test: ")

            # Save the algorithm state
            if (save_frequency is not None and save_frequency > 0) and self.n_iters % save_frequency == 0:
                self.save(save_path)

            # Log information
            train_scores.append(info_sample['mean_score'])  # so that mean can be computed
            if self.n_iters % log_frequency == 0:
                self.logger.log('Average mean score', np.mean(train_scores), self.n_iters, color='blue')
                self.log(info_update, prefix="Update: ")
                self.log(info_sample, prefix="Sample: ")
                self.n_samples += sum(len(s) for s in samples)  # update the number of samples processed
                self.logger.log('Number of samples', self.n_samples, self.n_iters, color='blue')
                # Log parameters
                for p in self.agent.parameters():
                    self.logger.log(f"Parameter: {p.name}", p.data, self.n_iters, color='red')

            # Update counters
            self.n_epochs = info_sample['n_epochs']  # update the number of epochs completed
            self.n_iters += 1
        return

    # TODO
    def evaluate(self, agent, guide, xs, infos, min_score=None, num_samples=1, num_threads=None, description=None):
        """ Evaluate the agent on the given dataset. """
        num_threads = num_threads or self.num_threads  # Use provided num_threads or fall back to self.num_threads
        test_scores = evaluate(agent, guide, xs, infos, min_score=min_score, num_threads=num_threads,
                               description=description, num_samples=self.num_eval_samples)
        if all([s is not None for s in test_scores]):
            return np.mean(test_scores)

    # TODO move it out?
    def sample(self, agents, loader=None, guide=None, **kwargs):
        """ Sample a batch of data based on the proposed parameters. All proposals are evaluated on the same batch of inputs.

        Args:
            agents (list): A list of trace.Modules (proposed parameters) to evaluate.
            **kwargs: Additional keyword arguments that may be used by the implementation.
        Returns:
            list of list of dict:
                A list (size len(agents)) with list of samples (size batchsize) for each agent,
                where each sample is a dict containing:
                - 'module': the trace.Module (proposal)
                - 'x': the input data
                - 'info': additional information about the input
                - 'target': the target output (if applicable)
                - 'score': the score of the proposal
                - 'feedback': the feedback from the guide

        NOTE: The return might not be ordered in the same way as the agents.
        """
        assert all(isinstance(a, trace.Module) for a in agents), "All agents must be trace.Modules."

        loader = loader or self._loader  # use the provided loader or the default one (train_dataset loader)
        guide = guide or self._guide  # use the provided guide or the default one (train_dataset guide)

        # Get a batch of inputs and infos from the loader
        xs, infos = loader.sample()

        # XXX hack for now
        self.xs, self.infos = xs, infos  # store the inputs and infos for later use

        # Evaluate each agent on the sampled inputs
        #
        # agents : a1, a2
        # inputs: x1, x2, x3
        # infos: i1, i2, i3
        # sub_batch_size: 2
        #
        # The forward is called in this order:
        # (a1, x1, i1, guide1),
        # (a1, x2, i2, guide1),
        # (deepcopy(a1), x3, i3, guide1)
        # (a2, x1, i1, guide2),
        # (a2, x2, i2, guide2),
        # (deepcopy(a2), x3, i3, guide2)

        num_threads = self.num_threads
        min_score = self.score_range[0]

        batch_size = len(xs)
        sub_batch_size = self.sub_batch_size or batch_size  # if sub_batch_size is not provided, use the batch size
        n_agents = len(agents)

        assert len(infos) == batch_size, "Length of infos must match length of xs."

        # Broadcast the agents to match the batch size
        # [a1, a1,        a1,        a1,        a1, ..., a2, a2,        a2, ...] if sub_batch_size is not specified
        # [a1, a1, a1_copy_1, a1_copy_1, a1_copy_2, ..., a2, a2, a2_copy_1, ...] if sub_batch_size of 2 is specified
        broadcasted_agents = []
        for agent in agents:
            for i in range(batch_size):
                if i % sub_batch_size == 0 and i > 0:
                    agent = copy.deepcopy(agent) # create a copy of the agent for the next sub-batch
                broadcasted_agents.append(agent)

        # Broadcast the inputs and infos to match the number of agents
        # [x1, x2, x3, ..., x1, x2, x3, ...]
        broadcasted_xs = [x for _ in range(n_agents) for x in xs]
        broadcasted_infos = [info for _ in range(n_agents) for info in infos]

        # Broadcast the guides to match the batch size

        description = f"Forwarding {n_agents} agents on {batch_size} inputs"

        # Forward the agent on the inputs and compute the feedback using the guide
        batched_forward = batch_run(max_workers=num_threads, description=description)(standard_forward)
        outputs = batched_forward(agent=broadcasted_agents,
                                  x=broadcasted_xs,
                                  info=broadcasted_infos,
                                  guide=guide,  # guide will be broadcasted inside
                                  min_score=min_score)
        # return list of (target, score, feedback)

        # Collect results
        results = []  # list of subgraphs (Subgraph objects) for each agent
        for i in range(n_agents):
            rollouts = []  # the compute result of each batch for a agent (trace.Module)
            _agent = broadcasted_agents[i * batch_size ] # the first agent in the batch
            for j in range(batch_size):
                rollout = Rollout(
                    module=broadcasted_agents[i * batch_size + j],
                    x=broadcasted_xs[i * batch_size + j],
                    info=broadcasted_infos[i * batch_size + j],
                    target=outputs[i * batch_size + j][0],  # target output
                    score=outputs[i * batch_size + j][1],  # score of the proposal
                    feedback=outputs[i * batch_size + j][2],  # feedback of the proposal
                )
                if _agent != rollout.module:
                    results.append(Subgraph(rollouts))  # append the subgraph to the results
                    _agent = rollout.module  # update the agent to the current one
                    rollouts = []  # reset rollouts for the new agent
                rollouts.append(rollout)

            if rollouts:
                results.append(Subgraph(rollouts))  # append the subgraph to the results

        # Log information about the sampling
        log_info = {
            'mean_score': np.mean([ g.get_scores() for g in results]),
            'batch_size': batch_size,
            'sub_batch_size': sub_batch_size,
            'n_epochs': loader.n_epochs,
        }
        return results, log_info

    def log(self, info_log, prefix=""):
        """ Log the information from the algorithm. """
        for key, value in info_log.items():
            try:
                if value is not None:
                    self.logger.log(f"{prefix}{key}", value, self.n_iters)
            except Exception as e:
                print(e)
                breakpoint()  # if logging fails, we can debug here

    def test(self, test_dataset, guide):
        min_score = self.score_range[0]
        # Test the agent's performance
        test_score = self.evaluate(self.agent, guide, test_dataset['inputs'], test_dataset['infos'],
                          min_score=min_score, num_threads=self.num_threads,
                          description=f"Evaluating agent (iteration {self.n_iters})")  # and log
        return {'test_score': test_score}

    def save(self, save_path):
        self.save_agent(save_path, self.n_iters)
        # TODO save full state of self

    # Helper methods for the algorithm
    def get_minibatch(self, samples):
        """ Get a minibatch of samples from the provided samples. """
        # Since all proposals share the same batch, we can return the first sample's x and info
        # return [s.x for s in samples[0]], [s['info'] for s in samples[0]]
        return self.xs, self.infos # XXX hack for now

    # Unimplemented methods that should be implemented by subclasses
    def update(self, samples=None, verbose=False, **kwargs):
        """ Update the agent based on the provided samples.
        Args:
            samples (list): A list of samples from the previous iteration. If None, the agent's parameters are returned without updating.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            **kwargs: Additional keyword arguments that may be used by the implementation.
        Returns:
            update_dict (dict of Parameter: Any): A dictionary containing the updated parameters of the agent.
            proposals (list of trace.Module): A list of proposed parameters (trace.Module) after the update.
            info_log (dict of str: Any): A dictionary containing logging information about the update process.

        This method updates the agent's parameters based on samples of the training dataset and validation dataset (provided by self.get_validate_dataset).
        In addition, it return new agents (proposals) that can be used for collecting data for the next iteration.
        """
        raise NotImplementedError("The update method should be implemented by subclasses.")
        # return update_dict, proposals, info_log


class ModuleCandidate:

    def __init__(self,
                 base_module: Optional[trace.Module],
                 update_dict: Optional[Dict[ParameterNode, Any]] = None,
                 ):
        """ A candidate module with its base module and update dictionary.
        Args:
            base_module (trace.Module): The base module to use as a template for the candidate.
            update_dict (dict): A dictionary of ParameterNode: value pairs to update the base module; the key can be a deep copy of the base module's parameters.
            stats (dict): A dictionary of statistics about the candidate.
        """
        assert isinstance(base_module, trace.Module), "base_module must be a trace.Module."
        self.base_module = base_module
        self.update_dict = update_dict if update_dict is not None else {}
        self.rollouts = []  # list of dicts containing the rollout information

    def get_module(self):
        """ Apply the update_dict to the base_module and return the updated module. This will not update the base_module itself."""
        return create_module_from_update_dict(self.base_module, self.update_dict) if self.update_dict else self.base_module

    def apply_update(self, base_module=None):
        """ Apply update to the base_module in place. """
        set_module_parameters(base_module or self.base_module, self.update_dict)

    def __deepcopy__(self, memo):
        """ Create a deep copy, except for the base_module which is not copied, it is the original module. """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k != 'base_module':
                setattr(result, k, deepcopy(v, memo))
            else:
                setattr(result, k, v)  # base_module is not copied, it is the original module
        return result

    def __equal__(self, other):
        """ Check if two candidates are equal based on their base_module and update_dict. """
        if not isinstance(other, ModuleCandidate):
            return False
        if self.base_module != other.base_module:
            return False
        update_dict_self = remap_update_dict(self.base_module, self.update_dict)
        update_dict_other = remap_update_dict(other.base_module, other.update_dict)
        return update_dict_self == update_dict_other

    def add_rollouts(self, rollouts: List[Dict[str, Any]]):
        """ Add rollouts to the candidate. """

        # # Convert all ParameterNode to data in the rollouts
        # _rollouts = []
        # for r in rollouts:
        #     _r = {}
        #     for k, v in r.items():
        #         if isinstance(v, trace.ParameterNode):
        #             _r[k] = v.data
        #         else:
        #             _r[k] = v

            # _rollouts.append(_r)  # convert all ParameterNode to data
        self.rollouts.extend(rollouts)
        # # XXX TODO hacky
        # self.rollouts.rollouts.extend(_rollouts)  # extend the rollouts with the

    def score(self):
        """ Compute the score of the candidate based on the rollouts. """
        if not self.rollouts:
            return None
        scores = [r['score'] for r in self.rollouts]
        return np.mean(scores) if scores else None

class PrioritySearch(SearchAlgorithm):

    # def train(self, *args,
    #           num_candidates: int = 10,  # number of candidates to propose
    #           default_score: Union[float, None] = None,  # default score for the candidates
    #           validate_proposals: bool = True,  # whether to validate the proposed parameters # TODO better naming
    #           **kwargs
    #           ):
    def train(self,
              guide, # guide to provide feedback
              train_dataset,  # dataset of (x, info) pairs to train the agent
              *,
              # validation
              validate_dataset = None, # same format as train_dataset; if None use the current batch.
              validate_guide = None,  #  to provide scores for the validation set
              # training loop
              batch_size = 1,  # batch size for updating the agent
              sub_batch_size = None,  # sub-batch size for broadcasting the agents
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
              num_candidates: int = 10,  # number of candidates to propose
              default_score: Union[float, None] = None,  # default score for the candidates
              validate_proposals: bool = True,  # whether to validate the proposed parameters
              # Additional keyword arguments
              **kwargs
              ):


        # Create agents and optimizers for search
        self.num_candidates = num_candidates  # number of candidates to propose
        self.score_range = score_range or (0., 1.) # XXX hacky now
        self.default_score = default_score if default_score is not None else self.score_range[0]  # default score for the candidates
        self.validate_proposals = validate_proposals  # whether to validate the proposed parameters
        self._queue = [(self.default_score, ModuleCandidate(self.agent))]  # priority queue of ModuleCandidates, initialized with the base agent

        super().train(guide, train_dataset,
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
                      **kwargs)

    def update(self, samples=None, verbose=False, **kwargs):

        if samples is not None:
            # 1. Propose new parameters based on running LLM optimizers on the collected samples
            candidates = self.propose(samples, verbose=verbose, **kwargs)  # List of ModuleCandidates
            # 2. Validate the proposed parameters
            validate_results = self.validate(candidates, samples, verbose=verbose, **kwargs)  # this updates the priority queue
            # 3. Update the priority queue with the validation results
            self.update_memory(validate_results, verbose=verbose, **kwargs)  # samples are provided here in case candidates do not capture full information
        # 4. Explore and exploit the priority queue
        best_candidate = self.exploit(verbose=verbose, **kwargs)  # get the best candidate (ModuleCandidate) from the priority queue
        exploration_candidates = self.explore(verbose=verbose, **kwargs)  # List of ModuleCandidates


        # TBD Log information about the update
        info_log = {
            'best_candidate_score': best_candidate.score(),
            'num_exploration_candidates': len(exploration_candidates),
        }
        return best_candidate.update_dict, [c.get_module() for c in exploration_candidates], info_log

    def propose(self, samples=None, verbose=False, n_proposals=1, **kwargs):
        """ Analyzing samples and propose new parameters using self.optimizer. An independent optimizer is used for the minibatch generated by one agent and generates n_proposals proposals.

        Args:
            samples (list): A list of samples from the previous iteration. If None, the agent's parameters are returned without updating.
            n_proposals (int): Number of proposals to generate per optimizer. Defaults to 1.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            **kwargs: Additional keyword arguments that may be used by the implementation.

        Returns:
            candidates (list of ModuleCandidate): A list of proposed candidates for the next iteration.
        """
        if samples is None:
            parameters = self.optimizer.parameters  # use the current parameters of the optimizer
            update_dict = {p: p.data for p in parameters}  # return the current parameters as the update dict
            # TODO what to do here? should we return n_proposals variations?
            return [update_dict]  # return the update dict as a list

        def _step(n, verbose=False, num_threads=None, **kwargs):
            """ Standard optimizer step for a single agent. """
            # optimizer = self._optimizers[n]  # get the optimizer for the n-th agent
            # TODO this seems slow
            optimizer = copy.deepcopy(self.optimizer)  # create a copy of the optimizer to avoid modifying the original one

            rollouts = samples[n]  # Subgraph

            # Make sure all rollouts are based on the same module, so they can be viewed as a minibatch.
            optimizer.parameters = rollouts.module.parameters()  # set the optimizer's parameters to the proposal's parameters

            targets = [r.target for r in rollouts]
            feedbacks = [r.feedback for r in rollouts]
            # batchify the targets and feedbacks
            target = batchify(*targets)
            feedback = batchify(*feedbacks).data  # str
            # standard optimizer step
            optimizer.zero_feedback()  # reset the optimizer's feedback
            optimizer.backward(target, feedback)  # compute the gradients based on the targets and feedbacks
            update_dict = optimizer.step(verbose=verbose, num_threads=num_threads, bypassing=True, **kwargs)
            # the update_dict is linked to the copied parameters of the agent, we set it back to the agent's parameters
            update_dict = remap_update_dict(self.agent, update_dict)  # remap the update dict to the agent's parameters
            return update_dict  # return the proposed parameters

        n_agents = len(samples)  # number of agents
        args_list = [(n, verbose, self.num_threads) for n in range(n_agents)]
        args_list = args_list * n_proposals  # repeat args_list n_proposals times
        kwargs_list = [kwargs] * n_agents * n_proposals  # repeat kwargs for each agent
        update_dicts = async_run([_step]*n_agents*n_proposals,  # run the optimizer step for each agent in parallel
                                  args_list=args_list,
                                  kwargs_list=kwargs_list,
                                  max_workers=self.num_threads,  # use the number of threads specified in the class
                                  description="Running optimizers on samples")
        # update_dicts is a list of dicts of length n_agents * n_proposals
        # Create ModuleCandidate objects for each proposed update_dict
        candidates = [ModuleCandidate(self.agent, update_dict) for update_dict in update_dicts]
        return candidates


    def validate(self, candidates, samples=None, verbose=False, **kwargs):
        """ Validate the proposed candidate parameters
        Args:
            candidates (list of dict): A list of ModuleCandidate objects representing the proposed parameters.
            samples (list of dict, optional): A list of samples collected in the current iteration. Defaults to None.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            **kwargs: Additional keyword arguments that may be used by the implementation.
        Returns:
            results (dict [ModuleCandidate, list of dict]): A dictionary where the keys are ModuleCandidate objects and the values are lists of rollouts (list of dicts) containing the module, x, info, target, score, feedback.
        """

        # Get the validation dataset from the samples. If no validation dataset is provided, use the current batch.
        if self._validate_dataset is None:
            # If no validation dataset is provided, use the current batch
            xs, infos = self.get_minibatch(samples)
            validate_dataset = {'inputs': xs, 'infos': infos}
        else:
            validate_dataset = self._validate_dataset


        class Loader:  # an trivial loader for the API
            def __init__(self):
                self.n_epochs = 0
            def sample(self):
                return validate_dataset['inputs'], validate_dataset['infos']
        loader = Loader()  # create a loader for the validation dataset
        candidate_agents = [c.get_module() for c in candidates]  # get the modules from the candidates
        validate_samples, _ = self.sample(candidate_agents, loader=loader, guide=self._validate_guide, **kwargs)
        # TODO log _

        if self.validate_proposals:
            if self._validate_dataset is None:
                validate_samples += samples  # if no validation dataset is provided, append the samples to the validate_samples
            else:  # validate the agents in the validate_dataset
                # TODO need a flag?
                exploration_agents = [rollouts.module for rollouts in samples]
                exploration_samples = self.sample(exploration_agents, loader=loader, guide=self._validate_guide, **kwargs)
                validate_samples += exploration_samples  # append the exploration samples to the validate_samples


        # Return a dict, key: ModuleCandidate, value: rollouts (list of dicts)
        results = {}
        for rollouts in validate_samples:
            # rollouts is subgraph
            agent = rollouts.module
            index = candidate_agents.index(agent)
            candidate = candidates[index]  # get the candidate corresponding to the agent
            # TODO delete 'module' from the rollouts dict?
            if candidate in results:
                # If the candidate already exists in results, we can append the rollouts to the existing list
                results[candidate].extend(rollouts)
            else:
                # If the candidate does not exist in results, we create a new entry
                results[candidate] = rollouts
        return results



    def update_memory(self, validate_results, **kwargs):

        """ Update the priority queue with the validation results.
        Args:
            validate_results (dict): A dictionary where the keys are ModuleCandidate objects and the values are lists of rollouts (list of dicts) containing the module, x, info, target, score, feedback.
            **kwargs: Additional keyword arguments that may be used by the implementation.
        """
        for candidate, rollouts in validate_results.items():
            candidate.add_rollouts(rollouts.to_list())  # add the rollouts to the candidate
            score = self.compute_score(candidate)  # compute the score for the candidate
            heapq.heappush(self._queue, (-score, candidate))  # add the candidate to the priority queue

    def explore(self, **kwargs):
        """ Explore the parameter space and propose new candidates.
        Args:
            **kwargs: Additional keyword arguments that may be used by the implementation.
        Returns:
            update_dict (dict of Parameter: Any): A dictionary containing the updated parameters of the agent.
            proposal_update_dicts (list of dict): A list of proposed parameter updates (dict) for the next iteration.
        """
        # pop top self.num_candidates candidates from the priority queue
        top_candidates = []
        while len(top_candidates) < self.num_candidates and self._queue:
            score, candidate = heapq.heappop(self._queue)
            top_candidates.append(candidate)  # add the candidate to the top candidates
        return top_candidates


    def exploit(self, **kwargs):
        """ Exploit the best candidate from the priority queue. This method should not change the priority queue.
        Args:
            **kwargs: Additional keyword arguments that may be used by the implementation.
        Returns:
            ModuleCandidate: The best candidate from the priority queue.
        """
        # Right now, we just return the best candidate from the priority queue
        # This function can be overridden by subclasses to implement a different exploitation strategy
        if not self._queue:
            raise ValueError("The priority queue is empty. Cannot exploit.")
        best = min(self._queue)  # (score, candidate)
        return best[1]

    def compute_score(self, candidate):
        # By default, we compute the mean score of the rollouts
        # NOTE This function can be overridden by subclasses to compute a different score
        scores = [r['score'] for r in candidate.rollouts]
        default_score = self.default_score  if self.default_score is not None else self.score_range[1]  # default score for the candidates

        return np.mean(scores) if scores else self.default_score

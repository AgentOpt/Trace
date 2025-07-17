import numpy as np
import copy
import heapq
from dataclasses import dataclass
from typing import Union, List, Tuple, Dict, Any, Optional
from opto import trace
from opto.trace.nodes import ParameterNode
from opto.trainer.utils import async_run, batch_run
from opto.optimizers.utils import print_color
from opto.trainer.algorithms.basic_algorithms import Minibatch, AlgorithmBase, batchify
from opto.trainer.evaluators import evaluate
from opto.trainer.loader import DataLoader

from opto.trainer.sampler import Sampler, RolloutsGraph

# TODO save and load SearchTemplate
# TODO async version???
# TODO create SYNC and ASYNC versions of the base class; add an attribute to the class to indicate
# TODO a better data structure to store samples

# update_dict

# Some helper function to convert between trace.Module and update_dict


def get_original_name(node):
    """Extract the original name from a node, removing all _copy suffixes."""
    py_name = node.py_name  # This removes colons: "param:0" -> "param0"

    # Find the first occurrence of "_copy" and remove it and everything after
    copy_index = py_name.find('_copy')
    if copy_index != -1:
        return py_name[:copy_index]
    else:
        return py_name

def is_node_copy(a, b):
    """Check if two nodes are copies of each other by comparing their original names.

    This function has transitivity: if A is a copy of B and B is a copy of C,
    then A is also considered a copy of C.
    """
    return get_original_name(a) == get_original_name(b)

def is_module_copy(a, b):
    """ Check if a and b (trace.Modules) are copies of each other. """
    parameters_a = a.parameters() # list of ParameterNode
    parameters_b = b.parameters() # list of ParameterNode
    # Check if all parameters of a are copies of b or vice versa
    # This might over count
    # need to check 1:1 correspondence
    matched = []
    for p_a in parameters_a:
        _matched = []
        for p_b in parameters_b:
            _matched.append(is_node_copy(p_a, p_b))
    np.array(matched)
    if np.all(np.sum(matched, axis=1) == 1) and np.all(np.sum(matched, axis=0) == 1):
        return True
    return False

def remap_update_dict(base_module, update_dict):
    """ Remap the update dict to the agent's parameters. update_dict might have keys which are copies of the base_module's parameters or visa versa.
        This function remaps the keys in update_dict to the original parameters of the base_module.

        The return dict is empty if no keys in update_dict matched any parameters of the base_module. This condition can be used to check if the update_dict contains non-trivial updates.
    """
    parameters = base_module.parameters()  # get the parameters of the base agent
    remapped_update_dict = {}
    for k, v in update_dict.items():
        for p in parameters:
            if is_node_copy(k, p): # Check if k is a copy of p or p is a copy of k
                remapped_update_dict[p] = v
                break # stop checking once we've found a match
    return remapped_update_dict

def set_module_parameters(agent, update_dict):
    """ Set the parameters of the agent based on the update_dict.
        The update_dict is a dictionary of ParameterNode: value pairs.
        The agent's parameters will be updated with the values from the update_dict.
    """
    remapped_update_dict = remap_update_dict(agent, update_dict)  # remap the update dict to the agent's parameters
    for k, v in remapped_update_dict.items():
        k._data = v  # set the parameter's data to the value in the update_dict

def create_module_from_update_dict(agent, update_dict):
    """ Create a new agent from the update_dict.
        The update_dict is a dictionary of ParameterNode: value pairs.
        A new agent will be created with the parameters set to the values from the update_dict.
    """
    new_agent = copy.deepcopy(agent) #.copy()  # create a copy of the agent
    set_module_parameters(new_agent, update_dict)  # set the parameters of the new agent
    return new_agent  # return the new agent



class Samples:
    """ A container for samples collected during the search algorithm. It contains a list of RolloutsGraph objects
    and a dataset with inputs and infos which created the list of RolloutsGraph. """

    samples: List[RolloutsGraph]
    dataset: Dict[str, List[Any]]  # contains 'inputs' and 'infos' keys

    def __init__(self, samples: List[RolloutsGraph], dataset: Dict[str, List[Any]]):
        assert isinstance(samples, list), "samples must be a list of RolloutsGraph objects."
        assert all(isinstance(s, RolloutsGraph) for s in samples), "All samples must be RolloutsGraph objects."
        assert isinstance(dataset, dict), "dataset must be a dict."
        assert 'inputs' in dataset and 'infos' in dataset, "dataset must contain 'inputs' and 'infos' keys."

        self.samples = samples
        self.dataset = dataset  # NOTE this cannot be extracted from the samples in general?

    def add_samples(self, samples):
        """ Add samples to the Samples object. """
        assert isinstance(samples, Samples), "samples must be an instance of Samples."
        samples = samples.samples  # extract the samples from the Samples object
        assert isinstance(samples, list), "samples must be a list of RolloutsGraph objects."
        assert all(isinstance(s, RolloutsGraph) for s in samples), "All samples must be RolloutsGraph objects."

        # TODO assert xs and infos are in self.minibatch
        # add a function to extract unique inputs and infos from the samples

        self.samples.extend(samples)

    def get_batch(self):
        return self.dataset #['inputs'], self.minibatch['infos']

    def __iter__(self):
        """ Iterate over the samples. """
        return iter(self.samples)

    def __len__(self):
        return sum(len(s) for s in self.samples)



class SearchTemplate(Minibatch):
    # This only uses __init__ and evaluate of Minibatch class.
    """ This implements a generic template for search algorithm. """

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
              eval_frequency: Union[int, None] = 1,  # frequency of evaluation
              num_eval_samples: int = 1,  # number of samples to use to evaluate each input
              # logging
              log_frequency = None,  # frequency of logging
              save_frequency: Union[int, None] = None,  # frequency of saving the agent
              save_path: str = "checkpoints/agent.pkl",  # path to save the agent
              **kwargs
              ):

        ## Setup

        test_frequency = eval_frequency  # use eval_frequency as test_frequency  # NOTE legacy notation
        log_frequency = log_frequency or test_frequency  # frequency of logging (default to test_frequency)
        self.num_threads = num_threads or self.num_threads  # Use provided num_threads or fall back to self.num_threads
        test_dataset = test_dataset or train_dataset  # default to train_dataset if test_dataset is not provided
        test_guide = test_guide or guide
        self.num_eval_samples = num_eval_samples  # number of samples to use to evaluate each input
        self.score_range = score_range or (0., 1.)

        self.train_sampler = Sampler(
            DataLoader(train_dataset, batch_size=batch_size),
            guide,
            num_threads=self.num_threads,
            sub_batch_size=sub_batch_size,
            score_range=self.score_range
        )
        self._validate_dataset = validate_dataset  # if None, the current batch will be used for validation
        self.validate_sampler = Sampler(
            DataLoader(validate_dataset if validate_dataset else {'inputs':[],'infos':[]}, batch_size=batch_size),
            validate_guide or guide,
            num_threads=self.num_threads,
            sub_batch_size=None,  # no sub-batch size for validation
            score_range=self.score_range
        )

        # Evaluate the agent before learning
        # NOTE set test_frequency < 0 to skip first evaluation
        if (test_frequency is not None) and test_frequency > 0:
            info_test = self.test(test_dataset, test_guide)  # test self.agent
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
            self.optimizer.update(update_dict)  # update self.agent with the proposed parameters

            # 2. Get feedback on the proposed parameters on the current batch
            # samples: Samples object containing the samples and the minibatch
            samples, info_sample = self.sample(proposals, verbose=verbose, **kwargs)

            # Evaluate the agent after update
            if (test_frequency is not None) and (self.n_iters % test_frequency == 0):
                info_test = self.test(test_dataset, test_guide)  # test self.agent
                self.log(info_test, prefix="Test: ")

            # Save the algorithm state
            if (save_frequency is not None and save_frequency > 0) and self.n_iters % save_frequency == 0:
                self.save(save_path)

            # Log information
            assert 'mean_score' in info_sample, "info_sample must contain 'mean_score'."
            assert 'n_epochs' in info_sample, "info_sample must contain 'n_epochs'."

            train_scores.append(info_sample['mean_score'])  # so that mean can be computed
            if self.n_iters % log_frequency == 0:
                self.logger.log('Average train score', np.mean(train_scores), self.n_iters, color='blue')
                self.log(info_update, prefix="Update: ")
                self.log(info_sample, prefix="Sample: ")
                self.n_samples += len(samples)  # update the number of samples processed
                self.logger.log('Number of samples', self.n_samples, self.n_iters, color='blue')
                # Log parameters
                for p in self.agent.parameters():
                    self.logger.log(f"Parameter: {p.name}", p.data, self.n_iters, color='red')

            # Update counters
            self.n_epochs = info_sample['n_epochs']  # update the number of epochs completed
            self.n_iters += 1
        return

    # Can be overridden by subclasses to implement specific sampling strategies
    def sample(self, agents, verbose=False, **kwargs):
        """ Sample a batch of data based on the proposed parameters. All proposals are evaluated on the same batch of inputs.

        Args:
            agents (list): A list of trace.Modules (proposed parameters) to evaluate.
                **kwargs: Additional keyword arguments that may be used by the implementation.
        """
        samples = Samples(*self.train_sampler.sample(agents))  # create a Samples object to store the samples and the minibatch

        # Log information about the sampling
        scores = [ g.get_scores() for g in samples.samples]  # list of list of scores for each RolloutsGraph
        scores = [item for sublist in scores for item in sublist]  # flatten the list of scores
        log_info = {
            'mean_score': np.mean(scores),
            'n_epochs': self.train_sampler.n_epochs,
        }
        return samples, log_info

    def log(self, info_log, prefix=""):
        """ Log the information from the algorithm. """
        for key, value in info_log.items():
            try:
                if value is not None:
                    self.logger.log(f"{prefix}{key}", value, self.n_iters)
            except Exception as e:
                print(e)

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


# TODO make this hashable?
class ModuleCandidate:
    """ A container used by PrioritySearch to store a candidate module as (its base module and update dictionary) and its statistics. """

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
        self.update_dict = remap_update_dict(self.base_module, self.update_dict)
        self.rollouts = []  # list of dicts containing the rollout information (not RolloutsGraph, but a list of dicts)

    def get_module(self):
        """ Apply the update_dict to the base_module and return the updated module.
        A new module is always created so the base_module is not modified.
        The new module has a new attribute _module_candidate which is this candidate."""
        module = create_module_from_update_dict(self.base_module, self.update_dict) if self.update_dict else copy.deepcopy(self.base_module)  #
        setattr(module, '__TRACE_RESERVED_module_candidate_id', id(self))
        return module  # return the updated module

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
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, v)  # base_module is not copied, it is the original module
        return result

    def __eq__(self, other):
        """ Check if two candidates are equal based on their base_module and update_dict. """
        assert isinstance(other, ModuleCandidate), "other must be an instance of ModuleCandidate."
        return self.update_dict == other.update_dict

    def __hash__(self):
        """ Hash the candidate based on its update_dict. """
        return hash(frozenset(self.update_dict.items()))

    def add_rollouts(self, rollouts: List[Dict[str, Any]]):
        """ Add rollouts to the candidate. """
        assert isinstance(rollouts, list), "rollouts must be a list of dicts."
        assert all(isinstance(r, dict) for r in rollouts), "All rollouts must be dicts."
        # Each rollout is a dict with keys: 'module', 'x', 'info', 'target', 'score', 'feedback'
        assert all('module' in r and 'x' in r and 'info' in r and 'target' in r and 'score' in r and 'feedback' in r for r in rollouts), \
            "Each rollout must contain 'module', 'x', 'info', 'target', 'score', and 'feedback' keys."

        self.rollouts.extend(rollouts)

    def score(self):
        """ Compute the score of the candidate based on the rollouts. """
        if not self.rollouts:
            return None
        scores = [r['score'] for r in self.rollouts]
        return np.mean(scores) if scores else None


class PrioritySearch(SearchTemplate):
    """ A search algorithm that uses a priority queue to explore the parameter space and propose new candidates. """

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
              num_candidates: int = 10,  # number of candidates to propose
              default_score: float = float('inf'),  # default score assigned to priority queue candidates
              validate_proposals: bool = True,  # whether to validate the proposed parameters
              # Additional keyword arguments
              **kwargs
              ):

        # Create agents and optimizers for search
        self.num_candidates = num_candidates  # number of candidates to propose by each optimizer call
        self.validate_proposals = validate_proposals  # whether to validate the proposed parameters
        self.default_score = default_score
        self.memory = [(self.default_score, ModuleCandidate(self.agent))]  # Priority queue of ModuleCandidates, initialized with the base agent
        self._exploration_candidates = None

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
        best_candidate, info_exploit = self.exploit(verbose=verbose, **kwargs)  # get the best candidate (ModuleCandidate) from the priority queue
        exploration_candidates, info_explore = self.explore(verbose=verbose, **kwargs)  # List of ModuleCandidates

        self._exploration_candidates = exploration_candidates

        # TODO Log information about the update
        info_log = {
            'best_candidate_score': best_candidate.score(),
            'num_exploration_candidates': len(exploration_candidates),
        }
        info_log.update(info_exploit)  # add the info from the exploit step
        info_log.update(info_explore)  # add the info from the explore step
        return best_candidate.update_dict, [c.get_module() for c in exploration_candidates], info_log

    def propose(self, samples, verbose=False, n_proposals=1, **kwargs):
        """ Analyzing samples and propose new parameters using self.optimizer. An independent optimizer is used for the minibatch generated by one agent and generates n_proposals proposals.

        Args:
            samples (list): A list of samples from the previous iteration. If None, the agent's parameters are returned without updating.
            n_proposals (int): Number of proposals to generate per optimizer. Defaults to 1.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            **kwargs: Additional keyword arguments that may be used by the implementation.

        Returns:
            candidates (list of ModuleCandidate): A list of proposed candidates for the next iteration.
        """

        assert isinstance(samples, Samples), "samples must be an instance of Samples."
        samples = samples.samples  # list of RolloutsGraph objects

        def _step(n, verbose=False, num_threads=None, **kwargs):
            """ Standard optimizer step for a single agent. """
            # optimizer = self._optimizers[n]  # get the optimizer for the n-th agent
            # NOTE this seems slow
            optimizer = copy.deepcopy(self.optimizer)  # create a copy of the optimizer to avoid modifying the original one

            rollouts = samples[n]  # RolloutsGraph

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
            # update_dict may only contain some of the parameters of the agent, we need to make sure it contains all the parameters
            for param in optimizer.parameters: # for all parameters
                if param not in update_dict: # update_dict misses some parameters
                    update_dict[param] = param.data # add the parameter to the update_dict
            # the update_dict is linked to the copied parameters of the agent, we set it back to the agent's parameters
            update_dict = remap_update_dict(self.agent, update_dict)  # remap the update dict to the agent's parameters
            return update_dict  # return the proposed parameters

        n_subgraphs = len(samples)  # number of subgraphs (agents) in the samples
        args_list = [(n, verbose, self.num_threads) for n in range(n_subgraphs)]
        args_list = args_list * n_proposals  # repeat args_list n_proposals times
        kwargs_list = [kwargs] * n_subgraphs * n_proposals  # repeat kwargs for each agent
        update_dicts = async_run([_step]*n_subgraphs*n_proposals,  # run the optimizer step for each agent in parallel
                                  args_list=args_list,
                                  kwargs_list=kwargs_list,
                                  max_workers=self.num_threads,  # use the number of threads specified in the class
                                  description="Running optimizers on samples")
        # update_dicts is a list of dicts of length n_agents * n_proposals
        # Create ModuleCandidate objects for each proposed update_dict
        candidates = [ModuleCandidate(self.agent, update_dict) for update_dict in update_dicts]
        return candidates

    def validate(self, candidates, samples, verbose=False, **kwargs):
        """ Validate the proposed candidate parameters
        Args:
            candidates (list of ModuleCandidate): A list of ModuleCandidate objects representing the proposed parameters.
            samples (list of dict, optional): A list of samples collected in the current iteration. Defaults to None.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            **kwargs: Additional keyword arguments that may be used by the implementation.
        Returns:
            results (dict): A dictionary where the keys are ids of ModuleCandidate objects and the values are ModuleCandidate and lists of rollouts (list of dicts) containing the module, x, info, target, score, feedback.
        """

        # Get the validation dataset from the samples. If no validation dataset is provided, use the current batch.
        if self._validate_dataset is None:
            # If no validation dataset is provided, use the current batch
            validate_dataset = samples.get_batch()  # get the batch of inputs and infos from the samples
            self.validate_sampler.dataset = validate_dataset  # set the validation dataset in the sampler
            self.validate_sampler.batch_size = len(validate_dataset['inputs'])  # set the batch size to the number of inputs in the validation dataset

        candidate_agents = [c.get_module() for c in candidates]  # get the modules from the candidates
        validate_samples = Samples(*self.validate_sampler.sample(candidate_agents))  # list of RolloutsGraph objects


        exploration_candidates = self._exploration_candidates  # exploration candidates from the previous iteration
        assert exploration_candidates is not None, "exploration_candidates must be set before calling validate."
        if self.validate_proposals:
            if self._validate_dataset is None:
                # NOTE this might contain some duplicates due to sub_batch_size < batch_size
                validate_samples.add_samples(samples)  # if no validation dataset is provided, append the samples to the validate_samples
            else:  # validate the agents in the validate_dataset
                # exploration_agents = [rollouts.module for rollouts in samples.samples]  # NOTE this might contain some duplicates due to sub_batch_size < batch_size
                exploitation_agents = [c.get_module() for c in exploration_candidates]  # get the modules from the exploration candidates
                exploration_samples = Samples(*self.validate_sampler.sample(exploration_agents))  # sample the exploration agents
                validate_samples.add_samples(exploration_samples)  # append the exploration samples to the validate_samples


        # TODO some ModuleCandidate are the same in parameters though they have different ids

        # In validate_samples, there may be multiple rollouts collected by the same agent (or their copies).
        # We need to group the rollouts by the agent (ModuleCandidate) and return a dictionary where the keys are the ModuleCandidate objects and the values are lists of rollouts (list of dicts).
        # Group the samples by the ModuleCandidate id
        _results = {}  # dict of ModuleCandidate: list of rollouts (list of dicts)
        for c in exploration_candidates + candidates:
            _results[id(c)] = []

        for rollouts in validate_samples.samples:
            module = rollouts.module  # trace.Module
            key = getattr(module, '__TRACE_RESERVED_module_candidate_id')  # use the candidate as the key
            if key not in _results:
                raise ValueError(f"ModuleCandidate with id {key} not found in results. Samples are not collected by known candidates.")
            # Append the rollouts to the list of rollouts for the key
            _results[key].extend(rollouts.to_list())

        # Merge rollouts of ModuleCandidates sharing the same parameters
        results = {}  # dict of ModuleCandidate id: (ModuleCandidate, list of rollouts)
        for c in exploration_candidates + candidates:
            rollouts_list = _results[id(c)]
            matched = False
            for k in results.keys():
                if k == c:
                    matched = True
                    if id(k) != id(c):  # merging rollouts of candidates with the same parameters
                        rollouts_list += c.rollouts
                    results[k].extend(rollouts_list)  # add the rollouts to the candidate
                    break
            if not matched:  # key not found in results
                results[c] = rollouts_list  # add the rollouts to the candidate

        # NOTE what if propose creates multiple exploration_candidates that have the same parameters and the same rollouts stats?
        # For example, it copies candidates. This would create a bug.
        return results



    def update_memory(self, validate_results, **kwargs):
        """ Update the priority queue with the validation results.
        Args:
            validate_results (dict): A dictionary where the keys are ModuleCandidate objects and the values are lists of rollouts (list of dicts) containing the module, x, info, target, score, feedback.
            **kwargs: Additional keyword arguments that may be used by the implementation.
        """
        for candidate, rollouts in validate_results.items():
            candidate.add_rollouts(rollouts)  # add the rollouts to the candidate
            score = self.compute_score(candidate)  # compute the score for the candidate
            heapq.heappush(self.memory, (-score, candidate))  # add the candidate to the priority queue


    ####
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
        while len(top_candidates) < self.num_candidates and self.memory:
            score, candidate = heapq.heappop(self.memory)
            top_candidates.append(candidate)  # add the candidate to the top candidates
        return top_candidates, {}


    def exploit(self, **kwargs):
        # NOTE This function can be overridden by subclasses to compute a different score
        """ Exploit the best candidate from the priority queue. This method should not change the priority queue.
        Args:
            **kwargs: Additional keyword arguments that may be used by the implementation.
        Returns:
            ModuleCandidate: The best candidate from the priority queue.
        """
        # Right now, we just return the best candidate from the priority queue
        # This function can be overridden by subclasses to implement a different exploitation strategy
        if not self.memory:
            raise ValueError("The priority queue is empty. Cannot exploit.")
        best = min(self.memory)  # (score, candidate)
        score, best_candidate = best
        score = -score # remember that we stored negative scores in the priority queue
        return best_candidate, {
            'best_candidate_score': score,  # remember that we stored negative scores in the priority queue
        }



    def compute_score(self, candidate):
        # NOTE This function can be overridden by subclasses to compute a different score
        """ Compute the score for the candidate based on the rollouts during the validation phase.
        It can be overridden by subclasses to implement a different scoring strategy.

        Args:
            candidate (ModuleCandidate): The candidate for which to compute the score.
        Returns:
            float: The computed score for the candidate.
        """
        if not isinstance(candidate, ModuleCandidate):
            raise TypeError("candidate must be an instance of ModuleCandidate.")
        # By default, we compute the mean score of the rollouts

        scores = [r['score'] for r in candidate.rollouts]
        default_score = self.default_score  if self.default_score is not None else self.score_range[1]  # default score for the candidates

        return np.mean(scores) if scores else self.default_score

class UCBSearch(PrioritySearch):
    """A search algorithm that keeps a buffer with candidates and their UCB scores. It does exploration according to the UCB score."""

    def __init__(self, *args, exploration_constant=1.0, **kwargs):
        """Initialize UCBSearch with an exploration constant for the UCB formula."""
        super().__init__(*args, **kwargs)
        self.exploration_constant = exploration_constant

    def compute_score(self, candidate):
        """Compute the UCB score for the candidate.
        
        UCB = mean_score + exploration_constant * sqrt(ln(total_trials) / candidate_trials)
        
        Args:
            candidate (ModuleCandidate): The candidate for which to compute the UCB score.
        Returns:
            float: The computed UCB score for the candidate.
        """
        if not isinstance(candidate, ModuleCandidate):
            raise TypeError("candidate must be an instance of ModuleCandidate.")
        
        # Get scores from rollouts
        scores = [r['score'] for r in candidate.rollouts]
        
        # If no rollouts, return a high exploration score to encourage trying this candidate
        if not scores:
            return float('inf')  # Maximum exploration for untried candidates
        
        # Calculate mean score for this candidate
        mean_score = np.mean(scores)
        candidate_trials = len(scores)
        
        # Calculate total trials across all candidates in memory
        total_trials = sum(len(c.rollouts) for _, c in self.memory) 
        
        # Handle edge case where total_trials is 0 or 1
        if total_trials <= 1:
            return mean_score
        
        # Calculate UCB score
        exploration_term = self.exploration_constant * np.sqrt(np.log(total_trials) / candidate_trials)
        ucb_score = mean_score + exploration_term
        
        return ucb_score

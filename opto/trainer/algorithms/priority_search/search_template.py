import numpy as np
from typing import Union, List, Tuple, Dict, Any, Optional
from opto import trace
from opto.trainer.algorithms.basic_algorithms import Minibatch
from opto.trainer.loader import DataLoader
from opto.trainer.sampler import Sampler, RolloutsGraph

# TODO save and load SearchTemplate
# TODO async version???
# TODO create SYNC and ASYNC versions of the base class; add an attribute to the class to indicate


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

    @property
    def n_sub_batches(self) -> int:
        """ Number of sub-batches in the samples. """
        return len(self.samples)



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
              test_dataset = None, # dataset of (x, info) pairs to evaluate the agent; if None, use train_dataset
              test_guide = None, # guide to provide scores for the test set; if None, use guide
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
        self.score_range = score_range or (-np.inf, np.inf)

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
        samples = Samples(*self.train_sampler.sample(agents, description_prefix='Sampling training minibatch: '))  # create a Samples object to store the samples and the minibatch
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
                          description=f"Evaluating agent")  # and log
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

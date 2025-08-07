from typing import Optional
from opto.trace.modules import Module
from opto.trainer.loggers import DefaultLogger
from opto.trainer.loader import DataLoader
from opto.trainer.guide import AutoGuide
from opto.optimizers.optimizer import Optimizer
import os
import pickle

class AbstractAlgorithm:
    """ Abstract base class for all algorithms. """

    def __init__(self, agent, *args, **kwargs):
        self.agent = agent

    def train(self, *args, **kwargs):
        """ Train the agent. """
        pass


class AlgorithmBase(AbstractAlgorithm):
    """
        We define the API of algorithms to train an agent from a dataset of (x, info) pairs.

        agent: trace.Module (e.g. constructed by @trace.model)
        teacher: (question, student_answer, info) -> score, feedback (e.g. info can contain the true answer)
        train_dataset: dataset of (x, info) pairs
    """

    def __init__(self,
                 agent,  # trace.model
                 num_threads: Optional[int] = None,   # maximum number of threads to use for parallel execution
                 logger=None,  # logger for tracking metrics
                 *args,
                 **kwargs):
        assert isinstance(agent, Module), "Agent must be a trace Module. Getting {}".format(type(agent))
        super().__init__(agent, *args, **kwargs)
        self.num_threads = num_threads
        # Use DefaultLogger as default if logger is None
        self.logger = logger if logger is not None else DefaultLogger()

    def _use_asyncio(self, threads=None):
        """Determine whether to use asyncio based on the number of threads.

        Args:
            threads: Number of threads to use. If None, uses self.num_threads.

        Returns:
            bool: True if parallel execution should be used, False otherwise.
        """
        effective_threads = threads or self.num_threads
        return effective_threads is not None and effective_threads > 1

    def save_agent(self, save_path, iteration=None):
        """Save the agent to the specified path.

        Args:
            save_path: Path to save the agent to.
            iteration: Current iteration number (for logging purposes).

        Returns:
            str: The path where the agent was saved.
        """
        # Create directory if it doesn't exist
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Add iteration number to filename if provided
        if iteration is not None:
            base, ext = os.path.splitext(save_path)
            # Add "_final" for the final checkpoint
            if hasattr(self, 'n_iters') and iteration == self.n_iters:
                save_path = f"{base}_iter{iteration}_final{ext}"
            else:
                save_path = f"{base}_iter{iteration}{ext}"

        # Save the agent
        self.agent.save(save_path)

        # Log if we have a logger and iteration is provided
        if hasattr(self, 'logger') and iteration is not None:
            self.logger.log('Saved agent', save_path, iteration, color='blue')

        return save_path

    def train(self,
              guide,
              train_dataset,  # dataset of (x, info) pairs
              num_threads: int = None,  # maximum number of threads to use (overrides self.num_threads)
              **kwargs
              ):
        raise NotImplementedError


    def save(self, path: str):
        """ Save the guide to a file. """
        with open(path, 'wb') as f:
            d = {}
            for key, value in self.__dict__.items():
                if isinstance(value, Module):
                    _path = path+ f"_{key}.module"
                    value.save(_path)
                    d[key] = _path
                elif isinstance(value, AutoGuide):
                    _path = path + f"_{key}.guide"
                    value.save(_path)
                    d[key] = _path
                elif isinstance(value, DataLoader):
                    _path = path + f"_{key}.dataloader"
                    value.save(_path)
                    d[key] = _path
                elif isinstance(value, Optimizer):
                    _path = path + f"_{key}.optimizer"
                    value.save(_path)
                    d[key] = _path
                else:
                    d[key] = value
            pickle.dump(d, f)

    def load(self, path: str):
        """ Load the guide from a file. """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            for key, value in data.items():
                if key not in self.__dict__:
                    warning_msg = f"Key '{key}' not found in the algorithm's attributes. Skipping loading for this key."
                    print(warning_msg)  # or use logging.warning(warning_msg)
                    continue

                # key is in the algorithm's attributes
                if isinstance(value, str):
                    if value.endswith('.module'):
                        attr = self.__dict__[key]
                        assert isinstance(attr, Module), f"Expected {key} to be a Module, got {type(attr)}"
                    elif value.endswith('.guide'):
                        attr = self.__dict__[key]
                        assert isinstance(attr, AutoGuide), f"Expected {key} to be an AutoGuide, got {type(attr)}"
                    elif value.endswith('.dataloader'):
                        attr = self.__dict__[key]
                        assert isinstance(attr, DataLoader), f"Expected {key} to be a DataLoader, got {type(attr)}"
                    elif value.endswith('.optimizer'):
                        attr = self.__dict__[key]
                        assert isinstance(attr, Optimizer), f"Expected {key} to be an Optimizer, got {type(attr)}"
                    attr.load(value)
                else:
                    self.__dict__[key] = value
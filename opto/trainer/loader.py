import numpy as np
import pickle

class DataLoader:

    def __init__(self, dataset, batch_size=1, replacement=False, shuffle=True):
        """ Initialize the data loader

        Args:
            dataset: the dataset to load (a dict of inputs and infos)
            batch_size: the number of samples to load in each batch
            replacement: whether to sample with replacement
            shuffle: whether to shuffle the dataset after each epoch
        """
        assert isinstance(dataset, dict), "Dataset must be a dict"
        assert 'inputs' in dataset and 'infos' in dataset, "Dataset must have 'inputs' and 'infos' key"
        assert len(dataset['inputs']) == len(dataset['infos']), "Inputs and infos must have the same length"

        self.dataset = dataset
        self.batch_size = batch_size
        self.replacement = replacement
        self.shuffle = shuffle
        self._indices = self._update_indices()
        self._i = 0

    def __iter__(self):
        indices = self._indices
        for i in range(self._i, len(indices), self.batch_size):
            xs = [ self.dataset['inputs'][ind]  for ind in indices[i:i + self.batch_size] ]
            infos = [self.dataset['infos'][ind] for ind in indices[i:i + self.batch_size] ]
            self._i = i + self.batch_size
            yield xs, infos
        self._i = 0

        if self.shuffle:
            self._indices = self._update_indices()

    def _update_indices(self):
        N = len(self.dataset['inputs'])
        return np.random.choice(N, size=N, replace=self.replacement)

    def save(self, path):
        """Save the dataset to a file."""
        with open(path, 'wb') as f:
            pickle.dump(
                {'_indices': self._indices,
                 '_i': self._i,
                 'batch_size': self.batch_size,
                 'replacement': self.replacement,
                 'shuffle': self.shuffle,
                 'dataset': self.dataset},
                f
            )

    def load(self, path):
        """Load the dataset from a file."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self._indices = data['_indices']
            self._i = data['_i']
            self.batch_size = data['batch_size']
            self.replacement = data['replacement']
            self.shuffle = data['shuffle']
            self.dataset = data['dataset']
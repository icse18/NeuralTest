import numpy as np
import collections
from tensorflow.python.framework import random_seed

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

class Dataset(object):
    def __init__(self, vectors, labels, seed=None):
        seed_1, seed_2 = random_seed.get_seed(seed)
        np.random.seed(seed_1 if seed is None else seed_2)
        assert len(vectors) == len(labels), ("Total num of vectors: %s != Total num of labels: %s" % (len(vectors), len(labels)))
        self._num_examples = len(vectors)
        self._vectors = vectors
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def vectors(self):
        return self._vectors

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def fetch_next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._vectors = self.vectors[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            vectors_rest_part = self._vectors[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arrange(self._num_examples)
                np.random.shuffle(perm)
                self._vectors = self.vectors[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            vectors_new_part = self._vectors[start:end]
            labels_new_part = self.labels[start:end]
            return np.concatenate((vectors_rest_part, vectors_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._vectors[start:end], self._labels[start:end]

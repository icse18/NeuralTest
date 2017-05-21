from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import random_seed
import pandas as pd
import numpy as np

class DataSet(object):

  def __init__(self, 
               vectors,
               labels,
               seed=None):
      
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    np.random.seed(seed1 if seed is None else seed2)

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

  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._vectors = self.vectors[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      vectors_rest_part = self._vectors[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._vectors = self.vectors[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      vectors_new_part = self._vectors[start:end]
      labels_new_part = self._labels[start:end]
      return np.concatenate((vectors_rest_part, vectors_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._vectors[start:end], self._labels[start:end]


def read_data_sets(validation_size=5000, seed=None):
  
    str_train_vectors = pd.read_csv("train_vectors.txt", header = None)
    str_train_labels = pd.read_csv("train_labels.txt", header = None)
  
    str_test_vectors = pd.read_csv("test_vectors.txt", header = None)
    str_test_labels = pd.read_csv("test_labels", header = None)
    if not 0 <= validation_size <= len(str_train_vectors):
        raise ValueError(
                'Validation size should be between 0 and {}. Received: {}.'
                .format(len(str_train_vectors), validation_size))
        
    train_vectors_list = []
    for _, row in str_train_vectors.iterrows():
        train_vectors_list.append([int(value) for value in row[0].split(" ")])
    train_vectors = np.array(str_train_vectors)
    
    train_labels = (str_train_labels== 1).astype(np.int_)
    
    test_vectors_list = []
    for _, row in str_test_vectors.iterrows():
        test_vectors_list.append([int(value) for value in row[0].split(" ")])
    test_vectors = np.array(test_vectors_list)    
    
    test_labels = (str_test_labels== 1).astype(np.int_)
    
    validation_vectors = train_vectors[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_vectors = train_vectors[validation_size:]
    train_labels = train_labels[validation_size:]
    
    train = DataSet(train_vectors, train_labels, seed=seed)
    validation = DataSet(validation_vectors, validation_labels, seed=seed)
    test = DataSet(test_vectors, test_labels, seed=seed)
    return base.Datasets(train=train, validation=validation, test=test)
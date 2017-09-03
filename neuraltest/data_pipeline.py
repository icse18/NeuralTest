from dataset import Dataset
from dataset import Datasets
import numpy as np
import random

# for testing
def predicate_1(vector):
    return [1 if ((vector[0] ** 2 + vector[1] ** 2 >= vector[2] ** 2)) else 0]

def synthetic_data_generator(predicate, total_data, min_val, max_val, vector_size):
    count = 0
    seen = set()
    vectors = []
    labels = []
    while count < total_data:
        vector = [random.uniform(min_val, max_val) for _ in range(vector_size)]
        sorted_vector = tuple(sorted(vector))
        if sorted_vector in seen:
            continue
        seen.add(sorted_vector)
        labels.append(predicate(vector))
        vectors.append(vector)
        count += 1
    return np.array(vectors), np.array(labels)

def prepare_data_sets(vectors, labels, validation_size, seed=None):
    if not 0 <= validation_size <= len(vectors):
        raise ValueError("Validation size should be between 0 and {}. Received {}.".format(len(train_vectors), validation_size))

    train_vectors = vectors[validation_size:]
    train_labels = labels[validation_size:]

    validation_test_split = int(validation_size / 2)

    validation_vectors = vectors[:validation_test_split]
    validation_labels = labels[:validation_test_split]

    test_vectors = vectors[validation_test_split:validation_size]
    test_labels = labels[validation_test_split:validation_size]

    train = Dataset(train_vectors, train_labels, seed)
    validation = Dataset(validation_vectors, validation_labels, seed)
    test = Dataset(test_vectors, test_labels, seed)

    return Datasets(train=train, validation=validation, test=test)

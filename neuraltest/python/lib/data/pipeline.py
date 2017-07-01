from dataset import DataSet
from dataset import Datasets
import numpy as np

def prepare_data_set(vectors, labels, train_test_split=0.8, train_size=0.8):
    train_size = int(train_size * len(vectors))    
    vectors = np.array(vectors) 
    labels = np.array(labels)
    train_vectors = vectors[:train_size]
    train_labels = labels[:train_size]
    test_vectors = vectors[train_size:]
    test_labels = labels[train_size:]
    
    train_validation_size = int(train_test_split * len(train_vectors))
    
    train = DataSet(train_vectors[:train_validation_size], train_labels[:train_validation_size])
    validation = DataSet(train_vectors[train_validation_size:], train_labels[train_validation_size:])
    test = DataSet(test_vectors, test_labels)
    return Datasets(train=train, validation=validation, test=test)
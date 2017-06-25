from dataset import DataSet
from dataset import Datasets

def prepare_data_set(vectors, labels, train_test_split=0.8, train_size=0.8):
    train_size = int(train_size * len(vectors))    
    
    train_vectors = vectors[:train_size]
    train_labels = labels[:train_size]
    test_vectors = vectors[train_size:]
    test_labels = labels[train_size:]
    
    train_validation_size = int(train_test_split * len(train_vectors))
    
    train = DataSet(train_vectors[:train_validation_size], train_labels[:train_validation_size])
    validation = DataSet(train_vectors[train_validation_size:], train_labels[train_validation_size:])
    test = DataSet(test_vectors, test_labels)
    return Datasets(train=train, validation=validation, test=test)

#listOfVectors, listOfLabels = data_generator(predicate_1, 40000, 3, -60000, 60000)
#prepare_data_set(listOfVectors, listOfLabels)
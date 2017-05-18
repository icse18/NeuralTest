import pandas as pd
import numpy as np
import tensorflow as tf
import tflearn 
from tflearn.data_utils import to_categorical

str_vectors = pd.read_csv("vectors.txt", header = None)
str_labels = pd.read_csv("labels.txt", header = None)

int_vectors = []
for _, row in str_vectors.iterrows():
    int_vectors.append([int(value) for value in row[0].split(" ")])
vectors = np.array(int_vectors)

Y = (str_labels== 1).astype(np.int_)
records = len(str_labels)
shuffle = np.arange(records)
np.random.shuffle(shuffle)
test_fraction = 0.9

train_split, test_split = shuffle[:int(records*test_fraction)], shuffle[int(records*test_fraction):]
trainX, trainY = vectors[train_split,:], to_categorical(Y.values[train_split], 2)
testX, testY = vectors[test_split,:], to_categorical(Y.values[test_split], 2)

def build_model():
    tf.reset_default_graph()
    
    # Input Layer: Create a network with 3 input units (x, y, z)
    net = tflearn.input_data([None, 3])
    
    # Hidden Layer(s)
    net = tflearn.fully_connected(net, 200, activation = "ReLU")
    net = tflearn.fully_connected(net, 20, activation = "ReLU")
    
    # Output
    net = tflearn.fully_connected(net, 2, activation = "softmax")
    net = tflearn.regression(net, optimizer = "sgd",
                             learning_rate = 0.0001, loss = "categorical_crossentropy")
    
    model = tflearn.DNN(net)
    return model

model = build_model()
model.fit(trainX, trainY, validation_set = 0.1, 
          show_metric = True, batch_size = 128, n_epoch = 100)


predictions = (np.array(model.predict(testX))[:,0] >= 0.5).astype(np.int_)
test_accuracy = np.mean(predictions == testY[:,0], axis=0)
print("Test accuracy: ", test_accuracy)
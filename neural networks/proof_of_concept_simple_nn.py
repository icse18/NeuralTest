import pandas as pd
import numpy as np
import tensorflow as tf
import tflearn 
from tflearn.data_utils import to_categorical
import matplotlib.pyplot as plt

str_vectors = pd.read_csv("vectors_50k_5d.txt", header = None)
str_labels = pd.read_csv("labels_50k_5d.txt", header = None)

int_vectors = []

for _, row in str_vectors.iterrows():
    int_vectors.append([int(value) for value in row[0].split(" ")])

test_fraction = 0.9
records = len(int_vectors)

vectors = np.array(int_vectors)
train_features_split, test_features_split = vectors[:int(records * test_fraction)], vectors[int(records * test_fraction)]

labels = np.array(str_labels)
train_targets_split, test_targets_split = labels[:int(records * test_fraction)], labels[int(records * test_fraction)]

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
                             learning_rate = 0.1, loss = "categorical_crossentropy")
    
    model = tflearn.DNN(net)
    return model

model = build_model()
model.fit(train_features_split, train_targets_split, validation_set = 0.1, 
          show_metric = True, batch_size = 100, n_epochs = 500)

"""
Build a Neural Network for Regression of Recursive Predicates
Implements the inference/loss/training pattern for model building.
"""

# Author: Joel Ong

import tensorflow as tf
import math

def weight_initialization(shape, name):
    return tf.Variable(tf.truncated_normal(
                [shape[0], shape[1]], 
                stddev = 1.0 / math.sqrt(float(shape[0]))), name=name)

def biases_initialization(shape, name):
    return tf.Variable(tf.zeros([shape[0]]), name=name)

def inference(vectors, hidden1_units, hidden2_units, dprobability=0.75):
    with tf.name_scope("hidden_layer_1"):
        shape = [1, hidden1_units]
        weights = weight_initialization(shape, "weights")
        biases = biases_initialization(shape[0], "biases")
        hidden1 = tf.nn.relu(tf.matmul(vectors, weights) + biases)
        hidden1 = tf.nn.dropout(hidden1, dprobability)
        
    with tf.name_scope("hidden_layer_2"):
        shape = [hidden1_units, hidden2_units]
        weights = weight_initialization(shape, "weights")
        biases = biases_initialization(shape[0], "biases")
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)  
        hidden2 = tf.nn.dropout(hidden2, dprobability)
        
    with tf.name_scope("output_layer"):
        shape = [hidden2_units, 1] 
        weights = weight_initialization(shape, "weights")
        biases = biases_initialization(shape[0], "biases")
        output = tf.nn.relu(tf.matmul(hidden2, weights) + biases)
    return output

def loss(output, labels): 
    # regression problem: reduce mean square error
    return tf.reduce_mean(tf.square(tf.sub(output - labels)))

def training(loss, learning_rate=0.001):
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train

def evaluation(logits, labels_placeholder):
    
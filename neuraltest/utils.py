import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
import matplotlib.pyplot as plt
import numpy as np

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def evaluate_samples(predicate, sample_vectors, batch_labels):
    correct = 0
    for index, sample_vector in enumerate(sample_vectors):
        if batch_labels[index] == predicate(sample_vector):
            correct += 1
    return correct

def visualize_results(corrects):
    epochs = [epoch + 1 for epoch in range(len(corrects))]
    plt.plot(epochs, corrects)
    plt.show()

def concat(x, y):
    return tf.concat([x, y], 1)

def batch_norm(x, is_training, scope):
    return tf.layers.batch_normalization(x, epsilon=1e-5, momentum=0.9, training=is_training, name=scope)

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)

def layer_init(input_layer, output_size, scope=None, with_w=False):
    shape = input_layer.get_shape().as_list()
    with tf.variable_scope(scope or "layer"):
        weight = tf.get_variable("weights", [shape[1], output_size], tf.float32, initializer=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(shape[0])))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(0.0))
        return tf.matmul(input_layer, weight) + bias

def layer_initialization(shape, scope=None, with_w=False):
    with tf.variable_scope(scope or "layer"):
        weight = tf.get_variable("weights", shape, tf.float32, tf.random_normal_initializer(stddev=1.0/math.sqrt(float(shape[0]))))
        bias = tf.get_variable("bias", shape[1], initializer=tf.constant_initializer(0.0))
        return weight, bias

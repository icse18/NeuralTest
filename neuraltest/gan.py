from util import layer_initializer
import numpy as np
import tensorflow as tf

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def generator(z, hidden_layer_shapes):
    hidden_layer = []
    for index, shape in enumerate(hidden_layer_shapes):
        hidden_layer_weights, hidden_layer_bias = layer_initializer(shape)
        if index == 0:
            hidden_layer[index] = tf.nn.relu(tf.matmul(z, hidden_layer_weights) + hidden_layer_bias)
        elif index != (len(hidden_layer_shapes)-1):
            hidden_layer[index] = tf.nn.relu(tf.matmul(hidden_layer[index - 1], hidden_layer_weights) + hidden_layer_bias)
        else:
            g_prob = tf.nn.sigmoid(tf.matmul(hidden_layer[index - 1], hidden_layer_weights) + hidden_layer_bias)
    return g_prob
    
def discriminator(x, hidden_layer_shapes):
    hidden_layer = []
    for index, shape in enumerate(hidden_layer_shapes):
        hidden_layer_weights, hidden_layer_bias = layer_initializer(shape)
        if index == 0:
            hidden_layer[index] = tf.nn.relu(tf.matmul(x, hidden_layer_weights) + hidden_layer_bias)
        elif index != (len(hidden_layer_shapes)-1):
            hidden_layer[index] = tf.nn.relu(tf.matmul(hidden_layer[index - 1], hidden_layer_weights) + hidden_layer_bias)
        else:
            d_logit = tf.matmul(hidden_layer[index - 1], hidden_layer_weights) + hidden_layer_bias
            d_prob = tf.nn.sigmoid(d_logit)
    return d_prob, d_logit    
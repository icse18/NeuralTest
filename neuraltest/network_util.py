import tensorflow as tf
import math

def layer_initializer(shape):
    layer_weight = weight_initializer(shape[0], shape[1])
    layer_bias = bias_initializer(shape[1])
    return layer_weight, layer_bias
    
def weight_initializer(input_units, hidden_units):
    return tf.Variable(tf.truncated_normal([input_units, hidden_units]),
                       stddev= 1.0 / math.sqrt(float(input_units)))

def bias_initializer(hidden_units):
    return tf.Variable(tf.zeros(hidden_units))
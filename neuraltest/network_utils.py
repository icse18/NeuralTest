import numpy as np
import tensorflow as tf

def weight_initializer(input_units, hidden_units):
    return tf.Variable(tf.truncated_normal([input_units, hidden_units],
                                           stddev=(1.0 / tf.sqrt(float(input_units)))))

def xavier_initializer(input_units, hidden_units):
    return tf.Variable(tf.random_normal(shape=[input_units, hidden_units],
                                        stddev=(1.0 / tf.sqrt(input_units / 2.0))))

def bias_initializer(hidden_units):
    return tf.Variable(tf.zeros(hidden_units))

def layer_initializer(shape, xavier=True):
    if xavier:
        weight = xavier_initializer(shape[0], shape[1])
    else:
        weight = weight_initializer(shape[0], shape[1])
    return weight, bias_initializer(shape[1])

def sample_z(val, size, normal=True):
    if normal:
        return np.random.normal(-val, val, size=size)
    return np.random.uniform(-val, val, size=size)
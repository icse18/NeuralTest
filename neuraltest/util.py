import tensorflow as tf
import math
import os
import random

def write_to_file(filepath, filename, data):
    print("Writting to " + filename)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    f = open(filepath + "\\" + filename, "w+")
    f.write("\n".join("%s" % value for value in data))
    f.close()
    
def uniform_vector(minimum, maximum, size):
#    Return a random floating point number N such that a <= N <= b for a <= b 
#    and b <= N <= a for b < a.
#    The end-point value b may or may not be included in the range depending on 
#    floating-point rounding in the equation a + (b-a) * random().
    count = 0
    vector = []
    while count < size:
        count += 1
        vector.append(random.uniform(minimum, maximum))
    return vector    

def layer_initializer(shape):
    layer_weight = weight_initializer(shape[0], shape[1])
    layer_bias = bias_initializer(shape[1])
    return layer_weight, layer_bias
    
def weight_initializer(input_units, hidden_units):
    return tf.Variable(tf.truncated_normal([input_units, hidden_units]),
                       stddev= 1.0 / math.sqrt(float(input_units)))

def bias_initializer(hidden_units):
    return tf.Variable(tf.zeros(hidden_units))
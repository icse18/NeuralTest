from util import weight_initializer
from util import bias_initializer
import tensorflow as tf

def generator(z):
    # dummy variables, will make implementation more generic afterwards.
    input_units = 200
    hidden_units_1 = 200
    hidden_units_2 = 200
    hidden_units_3 = 200
    hidden_layer_1 = tf.nn.relu(tf.matmul(z, 
                                          weight_initializer(input_units, 
                                                             hidden_units_1)) + 
                                            bias_initializer(hidden_units_1))
    hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, 
                                          weight_initializer(hidden_units_1, 
                                                             hidden_units_2)) + 
                                            bias_initializer(hidden_units_2))
    g_prob = tf.nn.sigmoid(tf.matmul(hidden_layer_2, 
                                          weight_initializer(hidden_units_2, 
                                                             hidden_units_3)) + 
                                            bias_initializer(hidden_units_3))
    return g_prob
    
    
    
def discriminator(x):
    input_units = 200
    hidden_units_1 = 200
    hidden_units_2 = 200
    hidden_units_3 = 200
    hidden_layer_1 = tf.nn.relu(tf.matmul(x, 
                                          weight_initializer(input_units, 
                                                             hidden_units_1)) + 
                                            bias_initializer(hidden_units_1))
    hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, 
                                          weight_initializer(hidden_units_1, 
                                                             hidden_units_2)) + 
                                            bias_initializer(hidden_units_2))
    d_logit = (tf.matmul(hidden_layer_2, 
                        weight_initializer(hidden_units_2, hidden_units_3)) + 
                        bias_initializer(hidden_units_3))
    d_prob = tf.nn.sigmoid(d_logit)
    return d_prob, d_logit
    
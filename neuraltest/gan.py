from network_util import layer_initializer
import tensorflow as tf

def model_inputs(real_vector_size, synthetic_vector_size):
    inputs_real = tf.placeholder(tf.float32, (None, real_vector_size),
                                 name="inputs_real")
    inputs_z = tf.placeholder(tf.float32, (None, synthetic_vector_size),
                              name="input_z")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    return inputs_real, inputs_z, learning_rate
    
def generator(z, hidden_layer_shapes, dprob=0.75, is_train=True):
    hidden_layer = []
    with tf.variable_scope("generator", reuse=not is_train):
        for index, shape in enumerate(hidden_layer_shapes):
            hidden_layer_weights, hidden_layer_bias = layer_initializer(shape)
            if index == 0:
                hidden_layer.append(tf.nn.dropout(tf.nn.relu(tf.matmul(z, hidden_layer_weights) + hidden_layer_bias), dprob))
            elif index != (len(hidden_layer_shapes)-1):
                hidden_layer.append(tf.nn.dropout(tf.nn.relu(tf.matmul(hidden_layer[index - 1], hidden_layer_weights) + hidden_layer_bias), dprob))
            else:
                g_prob = tf.matmul(hidden_layer[index - 1], hidden_layer_weights) + hidden_layer_bias
    return g_prob

def discriminator(x, hidden_layer_shapes, dprob=0.75, reuse=False):
    hidden_layer = []
    with tf.variable_scope("discriminator", reuse=reuse):
        for index, shape in enumerate(hidden_layer_shapes):
            hidden_layer_weights, hidden_layer_bias = layer_initializer(shape)
            if index == 0:
                hidden_layer.append(tf.nn.dropout(tf.nn.relu(tf.matmul(x, hidden_layer_weights) + hidden_layer_bias), dprob))
            elif index != (len(hidden_layer_shapes)-1):
                hidden_layer.append(tf.nn.dropout(tf.nn.relu(tf.matmul(hidden_layer[index - 1], hidden_layer_weights) + hidden_layer_bias), dprob))
            else:
                d_logit = tf.matmul(hidden_layer[index - 1], hidden_layer_weights) + hidden_layer_bias
                d_prob = tf.nn.sigmoid(d_logit)
    return d_prob, d_logit
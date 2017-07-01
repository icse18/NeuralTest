from network_util import layer_initializer
import tensorflow as tf

def model_inputs(vector_size):
    inputs_real = tf.placeholder(tf.float32, (None, vector_size),
                                 name="inputs_real")
    inputs_z = tf.placeholder(tf.float32, (None, vector_size),
                              name="inputs_z")
    inputs_label = tf.placeholder(tf.float32, (None, 1),
                                  name="inputs_label")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    return inputs_real, inputs_z, inputs_label, learning_rate
    
def generator(z, y, hidden_layer_shapes, dprob=0.75, is_train=True):
    hidden_layer = []
    inputs = tf.concat([z, y], 1)
    with tf.variable_scope("generator", reuse=not is_train):
        for index, shape in enumerate(hidden_layer_shapes):
            hidden_layer_weights, hidden_layer_bias = layer_initializer(shape)
            if index == 0:
                hidden_layer.append(tf.nn.dropout(tf.nn.relu(tf.matmul(inputs, hidden_layer_weights) + hidden_layer_bias), dprob))
            elif index != (len(hidden_layer_shapes)-1):
                hidden_layer.append(tf.nn.dropout(tf.nn.relu(tf.matmul(hidden_layer[index - 1], hidden_layer_weights) + hidden_layer_bias), dprob))
            else:
                g_prob = tf.matmul(hidden_layer[index - 1], hidden_layer_weights) + hidden_layer_bias
    return g_prob

def discriminator(x, y, hidden_layer_shapes, dprob=0.75, reuse=False):
    hidden_layer = []
    inputs = tf.concat([x, y], 1)
    with tf.variable_scope("discriminator", reuse=reuse):
        for index, shape in enumerate(hidden_layer_shapes):
            hidden_layer_weights, hidden_layer_bias = layer_initializer(shape)
            if index == 0:
                hidden_layer.append(tf.nn.dropout(tf.nn.relu(tf.matmul(inputs, hidden_layer_weights) + hidden_layer_bias), dprob))
            elif index != (len(hidden_layer_shapes)-1):
                hidden_layer.append(tf.nn.dropout(tf.nn.relu(tf.matmul(hidden_layer[index - 1], hidden_layer_weights) + hidden_layer_bias), dprob))
            else:
                d_logit = tf.matmul(hidden_layer[index - 1], hidden_layer_weights) + hidden_layer_bias
                d_prob = tf.nn.sigmoid(d_logit)
    return d_prob, d_logit
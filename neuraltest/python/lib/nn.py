import tensorflow as tf
from network_util import layer_initializer

def nn_inputs(vector_size):
    inputs_real = tf.placeholder(tf.float32, (None, vector_size),
                                 name="inputs_real")
    inputs_z = tf.placeholder(tf.float32, (None, vector_size),
                              name="inputs_z")
    inputs_label = tf.placeholder(tf.float32, (None, 1),
                                  name="inputs_label")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    return inputs_real, inputs_z, inputs_label, learning_rate

def neuralnetwork(z, y, hidden_layer_shape, dprob=0.75):
    hidden_layer = []
    inputs = tf.concat([z, y], 1)
    with tf.variable_scope("nn"):
        for index, shape in enumerate(hidden_layer_shape):
            hidden_layer_weights, hidden_layer_bias = layer_initializer(shape)
            if index == 0:
                hidden_layer.append(tf.nn.dropout(tf.nn.relu(tf.matmul(inputs, hidden_layer_weights) + hidden_layer_bias), dprob))
            elif index != (len(hidden_layer_shape)-1):
                hidden_layer.append(tf.nn.dropout(tf.nn.relu(tf.matmul(hidden_layer[index - 1], hidden_layer_weights) + hidden_layer_bias), dprob))
            else:
                logits = tf.matmul(hidden_layer[index - 1], hidden_layer_weights) + hidden_layer_bias
    return logits

def nn_loss(input_real, input_z, input_label, hidden_layer_shape):
    nn_model = neuralnetwork(input_z, input_label, hidden_layer_shape)    
    nn_loss = tf.reduce_mean(tf.square(input_real - nn_model))
    return nn_loss, nn_model

def nn_opt(nn_loss, learning_rate=0.001, beta1=0.9, beta2=0.999):
    trainable_variables = tf.trainable_variables()
    nn_vars = [n for n in trainable_variables if n.name.startswith("nn")]
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        nn_train_opt = tf.train.AdamOptimizer(learning_rate, beta1, beta2).minimize(nn_loss, var_list=nn_vars)
    return nn_train_opt
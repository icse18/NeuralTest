import network_util as nu
import tensorflow as tf
import random

def model_inputs(vector_size, label_size=1):
    inputs_real = tf.placeholder(tf.float32, 
                                 (None, vector_size),
                                 name="inputs_real")
    inputs_z = tf.placeholder(tf.float32, 
                              (None, vector_size),
                              name="inputs_z")
    inputs_label = tf.placeholder(tf.float32,
                                  (None, label_size),
                                  name="inputs_label")
    return inputs_real, inputs_z, inputs_label

def generator(z, labels, hidden_shapes, keep_prob=0.5, is_train=True):
    inputs = tf.concat([z, labels], 1)
    hidden_layers = []
    with tf.variable_scope("generator", reuse=(not is_train)):
        for index, shape in enumerate(hidden_shapes):
            weight, bias = nu.layer_initializer(shape)
            if index == 0:
                hidden_layers.append(tf.nn.dropout(tf.nn.relu(
                        tf.matmul(inputs, weight) + bias), keep_prob))
            elif index != (len(hidden_shapes) - 1):
                hidden_layers.append(tf.nn.dropout(tf.nn.relu(
                        tf.matmul(hidden_layers[index - 1], weight) + bias), keep_prob))
            else:
                g_prob = tf.matmul(hidden_layers[index - 1], weight) + bias # Not taking sigmoid because output is required to be real values
    return g_prob

def discriminator(x, labels, hidden_shapes, keep_prob=0.5, reuse=False):
    inputs = tf.concat([x, labels], 1)
    hidden_layers = []
    with tf.variable_scope("discriminator", reuse=reuse):
        for index, shape in enumerate(hidden_shapes):
            weight, bias = nu.layer_initializer(shape)
            if index == 0:
                hidden_layers.append(tf.nn.dropout(tf.nn.relu(
                        tf.matmul(inputs, weight) + bias), keep_prob))
            elif index != (len(hidden_shapes) - 1):
                hidden_layers.append(tf.nn.dropout(tf.nn.relu(
                        tf.matmul(hidden_layers[index - 1], weight) + bias), keep_prob))
            else:
                d_logit = tf.matmul(hidden_layers[index - 1], weight) + bias
                d_prob = tf.nn.sigmoid(d_logit)
    return d_prob, d_logit

def model_loss(inputs_real, inputs_z, inputs_label, hidden_g_shape, hidden_d_shape):
    g_model = generator(inputs_z, inputs_label, hidden_g_shape)
    d_model_real, d_logits_real = discriminator(inputs_real, inputs_label, hidden_d_shape)
    d_model_fake, d_logits_fake = discriminator(g_model, inputs_label, hidden_d_shape)
    
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                         labels=tf.fill(tf.shape(d_model_real),
                                                                                        random.uniform(0.7, 0.9))))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                         labels=tf.fill(tf.shape(d_model_fake),
                                                                                        random.uniform(0.0, 0.3))))
    d_loss = d_loss_real + d_loss_fake
    
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                    labels=tf.ones_like(d_model_fake)))
    return d_loss, g_loss

def model_opt(d_loss, g_loss, learning_rate=0.001):
    trainable_variables = tf.trainable_variables()
    g_vars = [g for g in trainable_variables if g.name.startswith("generator")]
    d_vars = [d for d in trainable_variables if d.name.startswith("discriminator")]
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)
        d_train_opt = tf.train.MomentumOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
    return d_train_opt, g_train_opt
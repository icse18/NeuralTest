from gan import generator
from gan import discriminator
import tensorflow as tf
import numpy as np
import random

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def model_loss(input_real, input_z, input_label, hidden_layer_shape_generator, hidden_layer_shape_discriminator):
    g_model = generator(input_z, input_label, hidden_layer_shape_generator)
    d_model_real, d_logits_real = discriminator(input_real, input_label, hidden_layer_shape_discriminator)
    d_model_fake, d_logits_fake = discriminator(g_model, input_label, hidden_layer_shape_discriminator)
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                         labels=tf.fill(tf.shape(d_model_real),
                                                                                        random.uniform(0.7, 0.9))))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                         labels=tf.fill(tf.shape(d_model_fake),
                                                                                        random.uniform(0.0, 0.3))))
    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(tf.square(input_real - g_model))
    return d_loss, g_loss

def model_opt(d_loss, g_loss, learning_rate=0.001, beta1=0.9, beta2=0.999):
    trainable_variables = tf.trainable_variables()
    g_vars = [g for g in trainable_variables if g.name.startswith("generator")]
    d_vars = [d for d in trainable_variables if d.name.startswith("discriminator")]
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1, beta2).minimize(g_loss, var_list=g_vars)
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1, beta2).minimize(d_loss, var_list=d_vars)
    return d_train_opt, g_train_opt
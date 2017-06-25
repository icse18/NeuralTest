from gan import model_inputs
from gan_util import model_loss
from gan_util import model_opt
import numpy as np
import tensorflow as tf

def train(epoch_count, batch_size, learning_rate, get_batches, vector_size, hidden_layer_shapes):
    hidden_layer_shape_generator, hidden_layer_shape_discriminator = hidden_layer_shapes
    input_real, input_z, input_label, lr = model_inputs(vector_size)
    d_loss, g_loss = model_loss(input_real, input_z, input_label,
                                hidden_layer_shape_generator, hidden_layer_shape_discriminator)
    d_opt, g_opt = model_opt(d_loss, g_loss, learning_rate)
    
    losses = []
    count = 0
    show_losses_every = 10
    # show_generator_output_every = 100
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):
            for batch_vectors in get_batches(batch_size):
                count += 1
                z = np.random.uniform(-1, 1, size=(batch_size, vector_size))
                _ = sess.run(d_opt, feed_dict={input_real: batch_vectors, input_z: z, lr: learning_rate})
                _ = sess.run(g_opt, feed_dict={input_real: batch_vectors, input_z: z, lr: learning_rate})
                if (count % show_losses_every == 0):
                    train_loss_d = d_loss.eval({input_z: z, input_real: batch_vectors})
                    train_loss_g = g_loss.eval({input_z: z})
                    print("Epoch {}/{}...".format(epoch_i + 1, epoch_count),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}...".format(train_loss_g))
                    losses.append((train_loss_d, train_loss_g))
    return losses

epoch_count = 2
batch_size = 64
learning_rate = 0.001

hidden_layer_generator_shape = [[4, 128],[128, 64],[64, 3]]
hidden_layer_discriminator_shape = [[3, 128],[128, 64],[64, 1]]
hidden_layer_shapes = (hidden_layer_generator_shape, hidden_layer_discriminator_shape)

data_shape = (3, 4)

with tf.Graph().as_default():
    train(epoch_count, batch_size, learning_rate, (get_batch), data_shape, hidden_layer_shapes)
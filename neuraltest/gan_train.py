from gan import model_inputs
from gan_util import model_loss
from gan_util import model_opt
from util import plot_loss
from pipeline import prepare_data_set
from synthetic_data_generator import data_generator
from predicates import predicate_1

import numpy as np
import tensorflow as tf

def run_training(learning_rate, max_steps, batch_size, vector_size, hidden_layer_shapes):
    vectors, labels = data_generator(predicate_1, 10000, 3, -20000, 20000)
    data_sets = prepare_data_set(vectors, labels)
    losses = []
    
    hidden_layer_shape_generator, hidden_layer_shape_discriminator = hidden_layer_shapes
    input_real, input_z, input_label, lr = model_inputs(vector_size)
    d_loss, g_loss, g_model = model_loss(input_real, input_z, input_label,
                                    hidden_layer_shape_generator, hidden_layer_shape_discriminator)
    d_opt, g_opt = model_opt(d_loss, g_loss, lr)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(max_steps):
            z = np.random.uniform(-1, 1, size=(batch_size, vector_size))
            vectors_feed, labels_feed = data_sets.train.next_batch(batch_size)  
            
            feed_dict = { input_real: vectors_feed, input_z: z, input_label: labels_feed, lr: learning_rate}
            sess.run(d_opt, feed_dict=feed_dict)
            sess.run(g_opt, feed_dict=feed_dict)
            
            if step % 100 == 0:
                train_loss_d = d_loss.eval({input_z: z, input_real: vectors_feed, input_label: labels_feed})
                train_loss_g = g_loss.eval({input_z: z, input_real: vectors_feed, input_label: labels_feed})
                print("Epoch {}/{}...".format(step, max_steps),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}...".format(train_loss_g))
                losses.append((train_loss_d, train_loss_g))
        
        
        corrects = []
        for step in range(10):
            example_z = np.random.uniform(-60000, 60000, size=[batch_size, vector_size])
            example_label = np.random.randint(2, size=[batch_size, 1])
            samples = sess.run(
                    g_model,
                    feed_dict={input_z: example_z, input_label: example_label})
            correct = 0
            for index, element in enumerate(samples):
                prediction = predicate_1(element)
                if prediction == labels_feed[index]:
                    correct += 1
            corrects.append(correct)
        total_count = 0
        for i in corrects:
            total_count += i
        print (total_count / (max_steps * 100))
        
    return losses

losses = run_training(0.0001, 10000, 100, 3, [[[4, 256],[256, 128],[128, 3]], [[4, 256],[256, 128],[128, 1]]])
plot_loss(losses)
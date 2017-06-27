from nn import nn_inputs
from nn import nn_loss
from nn import nn_opt
from util import sample_z
from pipeline import prepare_data_set
from synthetic_data_generator import data_generator
from predicates import predicate_1

import numpy as np
import tensorflow as tf

def run_training(learning_rate, max_steps, batch_size, vector_size, hidden_layer_shape):
    vectors, labels = data_generator(predicate_1, 50000, 3, -70000, 70000)
    data_sets = prepare_data_set(vectors, labels)
    losses = []
    
    input_real, input_z, input_label, lr = nn_inputs(vector_size)
    network_loss, nn_model = nn_loss(input_real, input_z, input_label, hidden_layer_shape)
    network_opt = nn_opt(network_loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(max_steps):
            z = sample_z(-60000, 60000, [batch_size, vector_size])
            vectors_feed, labels_feed = data_sets.train.next_batch(batch_size)  
            
            feed_dict = { input_real: vectors_feed, input_z: z, input_label: labels_feed, lr: learning_rate}
            sess.run(network_opt, feed_dict=feed_dict)
            
            if step % 100 == 0:
                train_loss_nn = network_loss.eval({input_z: z, input_real: vectors_feed, input_label: labels_feed})
                print("Epoch {}/{}...".format(step, max_steps),
                          "Generator Loss: {:.4f}...".format(train_loss_nn))
                losses.append(train_loss_nn)
        
        
        corrects = []
        for step in range(max_steps):
            example_z = sample_z(-60000, 60000, [batch_size, vector_size])
            example_label = np.random.randint(2, size=[batch_size, 1])
            samples = sess.run(
                    nn_model,
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

losses = run_training(0.0001, 500000, 100, 3, [[4, 256],[256, 128],[128, 3]])
# plot_loss(losses)
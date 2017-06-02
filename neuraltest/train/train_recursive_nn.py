"""
Training script for Function approximation for Recursive Predicate 
"""

# Author: Joel Ong

from ..neuralnetwork import recursive_nn
from helper import plot_loss_per_generation, save_model
import tensorflow as tf
import time

def placeholder_inputs(batch_size, vector_size):
    vectors_placeholder = tf.placeholder(tf.float32, shape=(batch_size, vector_size))
    labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size))
    return vectors_placeholder, labels_placeholder

def fill_feed_dict(data_set, batch_size, vectors_pl, labels_pl):
    vectors_feed, labels_feed = data_set.fib(batch_size)
    feed_dict = {vectors_pl: vectors_feed,
                 labels_pl: labels_feed}
    return feed_dict
    
def run_trainning(data_sets, batch_size=10, hidden_units_1=128, hidden_units_2=32, max_steps=2000, save=False):
    with tf.Graph().as_default():
        vectors_placeholder, labels_placeholder = placeholder_inputs(batch_size, 1)
        logits = recursive_nn.inference(vectors_placeholder, hidden_units_1,
                                        hidden_units_2)
        loss = recursive_nn.loss(logits, labels_placeholder)
        train_op = recursive_nn.training(loss)
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(init)
        
        loss_vec = []
        
        for step in range(max_steps):
            start_time = time.time()
            feed_dict = fill_feed_dict(data_sets, vectors_placeholder,
                                       labels_placeholder)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            duration = time.time() - start_time
            loss_vec.append(loss_value)
            
            if step % 100 == 0:
                print("Step %d: loss = %.2f (%.3f sec)" % (step, loss_value,
                      duration))
                
        plot_loss_per_generation(loss_vec)        
        
        if (save == True):
            save_model(saver, sess)                
"""
Training script for Function approximation for Recursive Predicate 
"""

# Author: Joel Ong

from .. neuralnetwork import recursive_nn
import tensorflow as tf

def placeholder_inputs(batch_size, vector_size):
    vectors_placeholder = tf.placeholder(tf.float32, shape=(batch_size, vector_size))
    labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size))
    return vectors_placeholder, labels_placeholder

def fill_feed_dict(data_set, batch_size, vectors_pl, labels_pl):
    vectors_feed, labels_feed = data_set.next_batch(batch_size)
    feed_dict = {vectors_pl: vectors_feed,
                 labels_pl: labels_feed}
    return feed_dict

def do_eval(sess, eval_correct, vectors_placeholder, labels_placeholder, 
            data_set, batch_size):
    true_count = 0
    num_examples = len(data_set)
    steps_per_epoch = num_examples // batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, batch_size, vectors_placeholder,
                                   labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print("Num examples: %d\nNum correct: %d\nPrecision @ 1: %0.04f\n" %
          (num_examples, true_count, precision))
    
def run_trainning():
    with tf.Graph().as_default():
        vectors_placeholder, labels_placeholder = placeholder_inputs(batch_size)
        logits = recursive_nn.inference(vectors_placeholder, hidden_units_1,
                                        hidden_units_2)
        loss = recursive_nn.loss(logits, labels_placeholder)
        train = recursive_nn.training(loss)
        
        
        
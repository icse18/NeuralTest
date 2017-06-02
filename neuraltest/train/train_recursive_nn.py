"""
Training script for Function approximation for Recursive Predicate 
"""

# Author: Joel Ong

from .. neuralnetwork import recursive_nn
import tensorflow as tf
import time

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
    
def run_trainning(data_sets, batch_size=100, hidden_units_1=128, hidden_units_2=32, max_steps=2000, save_file="train_model.ckpt"):
    with tf.Graph().as_default():
        vectors_placeholder, labels_placeholder = placeholder_inputs(batch_size)
        logits = recursive_nn.inference(vectors_placeholder, hidden_units_1,
                                        hidden_units_2)
        loss = recursive_nn.loss(logits, labels_placeholder)
        train_op = recursive_nn.training(loss)
        eval_correct = recursive_nn.evaluation(logits, labels_placeholder)
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(init)
        
        for step in range(max_steps):
            start_time = time.time()
            feed_dict = fill_feed_dict(data_sets.train, vectors_placeholder,
                                       labels_placeholder)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            duration = time.time() - start_time
            
            if step % 100 == 0:
                print("Step %d: loss = %.2f (%.3f sec)" % (step, loss_value,
                      duration))
                
            if (step + 1) % 1000 == 0 or (step + 1) == max_steps:
                print("Training Data Evaluation: ")
                do_eval(sess, eval_correct, vectors_placeholder, 
                        labels_placeholder, data_sets.train)
                print("Validation Data Evaluation: ")
                do_eval(sess, eval_correct, vectors_placeholder,
                        labels_placeholder, data_sets.validation)
                print("Test Data Evaluation: ")
                do_eval(sess, eval_correct, vectors_placeholder,
                        labels_placeholder, data_sets.test)
        saver.save(sess, save_file)
        print('Trained Model saved as ' + save_file)                
                
                
        
        
                
        
        
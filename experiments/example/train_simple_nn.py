from helper import read_data_sets
from simple_nn import discriminator
from simple_nn import loss
from simple_nn import trainning
import tensorflow as tf

def nn_placeholder(batch_size):
    vectors_placeholder = tf.placeholder(tf.float32, shape = (batch_size, 3))
    labels_placeholder = tf.placeholder(tf.int32, shape = (batch_size))
    return vectors_placeholder, labels_placeholder

def fill_feed_dict(data_set, batch_size, vectors_pl, labels_pl):
    vectors_feed, labels_feed = data_set.next_batch(batch_size)
    feed_dict = {vectors_pl : vectors_feed, labels_pl : labels_feed}
    return feed_dict

def do_evaluation(sess, batch_size, eval_correct, vectors_placeholder, labels_placeholder, data_set):
    num_correct_predictions = 0
    num_examples = data_set.num_examples
    steps_per_epoch = num_examples // batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, batch_size, vectors_placeholder, labels_placeholder)
        num_correct_predictions += sess.run(eval_correct, feed_dict = feed_dict)
    precision = float(num_correct_predictions) / num_examples
    print("Num examples: %d\nNum correct: %d\nPrecision @ 1 : %0.04f\n" % (num_examples, num_correct_predictions, precision))
    
def run_training(batch_size, hidden1_units, hidden2_units, learning_rate):
    data_sets = read_data_sets()
    with tf.Graph().as_default():
        vectors_placeholder, labels_placeholder = nn_placeholder(batch_size)
        logits = discriminator(vectors_placeholder, hidden1_units, hidden2_units)
        train_loss = loss(labels_placeholder, logits)
        train_optimizer = trainning(train_loss, learning_rate)
        
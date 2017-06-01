from helper import compute_stddev
import tensorflow as tf

def discriminator(input_vector, hidden1_units, hidden2_units):
    with tf.variable_scope("hidden layer 1"):
        weights = tf.Variable(tf.truncated_normal([input_vector, hidden1_units], 
                                                  stddev = compute_stddev(hidden1_units)), name = "weights")
        biases = tf.Variable(tf.truncated_normal([hidden1_units]), name = "biases")
        hidden1 = tf.nn.relu(tf.matmul(input_vector, weights) + biases)

    with tf.variable_scope("hidden layer 2"):
        weights = tf.Variable(tf.truncated_normal([hidden1, hidden2_units], 
                                                  stddev = compute_stddev(hidden2_units)), name = "weights")
        biases = tf.Variable(tf.truncated_normal([hidden2_units]), name = "biases")
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
        
    with tf.variable_scope("output layer"):
        weights = tf.Variable(tf.truncated_normal([hidden2, 1], 
                                                  stddev = compute_stddev(1)), name = "weights")
        biases = tf.Variable(tf.truncated_normal([1]), name = "biases")
        logits = tf.matmul(hidden2, weights) + biases
    return logits

def loss(labels, logits):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels,
                                                                   logits = logits,
                                                                   name = "cross_entropy")
    return tf.reduce_mean(cross_entropy, name = "cross_entropy_mean")

def trainning(loss, learning_rate):
    tf.summary.scalar("loss", loss)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    global_step = tf.Variable(0, name = "global_step", trainable = False)
    train_optimizer = optimizer.minimize(loss, global_step = global_step)
    return train_optimizer

def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
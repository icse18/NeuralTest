import tensorflow as tf

def sample_z(shape, minval, maxval, name, dtype):
    assert dtype is tf.float32
    z = tf.random_uniform(shape, minval=minval, 
                          maxval=maxval, name=name,
                          dtype=tf.float32)
    return z

def leaky_relu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)

# AdaGrad, RMSProp, and Adam optimization automatically reduce the learning rate 
# during training, it is not necesary to add an extra learning schedule.
def learning_schedule(initial_learning_rate=0.1, decay_steps=10000, decay_rate=0.1):
    global_step = tf.Variable(0, trainable=False, name="global_step")
    return tf.train.exponential_decay(initial_learning_rate,
                                      global_step,
                                      decay_steps,
                                      decay_rate)

def weight_initializer(w_shape):
    return tf.Variable(
            tf.truncated_normal(w_shape, stddev=(1.0 / tf.sqrt((float(w_shape[0]))))),
            name="weights")

def xavier_initializer(w_shape):
    return tf.Variable(
            tf.random_normal(w_shape, stddev=(1.0 / tf.sqrt(w_shape[0] / 2.0))),
            name="weights")
    
def hidden_layer(X, num_neurons, name, activation=None):
    with tf.name_scope(name):
        w_shape = (int(X.get_shape()[1]), num_neurons)
        weight = weight_initializer(w_shape)
        bias = tf.Variable(tf.zeros([num_neurons]), name="bias")
        output = tf.matmul(X, weight) + bias
        if activation is not None:
            return activation(output)
        return output

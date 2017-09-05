import os
import time
import random
import re
import tensorflow as tf
import numpy as np
from utils import *

class cgan(object):
    def __init__(self, sess, epoch, batch_size, datasets, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.model_name = "CGAN"

        # train
        self.dprob = 0.5
        self.learning_rate = 0.001
        self.beta1 = 0.5
        self.dnpl = [[4, 512],[512, 256],[256, 64], [64, 1]]
        self.gnpl = [[4, 512],[512, 256],[256, 64], [64, 3]]

        # test
        self.sample_num = 64 # number of generated vectors to saved

        # load datasets
        self.train_vectors = datasets.train._vectors
        self.train_labels = datasets.train._labels

        self.validate_vectors = datasets.validation._vectors
        self.validate_labels = datasets.validation._labels

        self.test_vectors = datasets.test._vectors
        self.test_labels = datasets.test._labels

        self.vector_dim = len(datasets.train._vectors[0])
        self.label_dim = len(datasets.train._labels[0])

        # get number of batches for a single epoch
        self.num_batches = len(self.train_vectors) // self.batch_size

    def discriminator(self, x, y, layer_shapes, is_training=True, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            hidden_layer = []
            x = concat(x, y)
            for index, shape in enumerate(layer_shapes):
                weight, bias = layer_initialization(shape, scope="d_layer_" + str(index + 1))
                if index == 0:
                    hidden_layer.append(tf.nn.dropout(tf.nn.relu(tf.matmul(x, weight) + bias), self.dprob))
                elif index != (len(layer_shapes)-1):
                    hidden_layer.append(tf.nn.dropout(tf.nn.relu(tf.matmul(hidden_layer[index - 1], weight) + bias), self.dprob))
                else:
                    d_logit = tf.matmul(hidden_layer[index - 1], weight) + bias
                    d_prob = tf.nn.sigmoid(d_logit)
            return d_prob, d_logit

    def generator(self, z, y, layer_shapes, is_training=True, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            hidden_layer = []
            z = concat(z, y)
            for index, shape in enumerate(layer_shapes):
                weight, bias = layer_initialization(shape, scope="g_layer_" + str(index + 1))
                if index == 0:
                    hidden_layer.append(tf.nn.dropout(tf.nn.relu(tf.matmul(z, weight) + bias), self.dprob))
                elif index != (len(layer_shapes)-1):
                    hidden_layer.append(tf.nn.dropout(tf.nn.relu(tf.matmul(hidden_layer[index - 1], weight) + bias), self.dprob))
                else:
                    g_prob = tf.matmul(hidden_layer[index - 1], weight) + bias
            return g_prob

    def construct_model(self):

        # vectors
        self.vectors = tf.placeholder(tf.float32, [self.batch_size, self.vector_dim], name="vectors")

        # labels
        self.labels = tf.placeholder(tf.float32, [self.batch_size, self.label_dim], name="labels")

        # noises
        self.noise = tf.placeholder(tf.float32, [self.batch_size, self.vector_dim], name="noise")

        """ Loss functions """
        # real vectors
        d_real, d_real_logits = self.discriminator(self.vectors, self.labels, self.dnpl, is_training=True, reuse=False)

        # fake vectors
        g_out = self.generator(self.noise, self.labels, self.gnpl, is_training=True, reuse=False)
        d_fake, d_fake_logits = self.discriminator(g_out, self.labels, self.dnpl, is_training=True, reuse=True)

        # get loss for discriminator
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits,
                                                                             labels=tf.fill(tf.shape(d_real), random.uniform(0.7, 0.9))))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits,
                                                                             labels=tf.fill(tf.shape(d_fake), random.uniform(0.0, 0.3))))
        self.d_loss = d_loss_real + d_loss_fake

        # get loss for generator
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits,
                                                                                labels=tf.fill(tf.shape(d_fake), random.uniform(0.7, 0.9))))

        """ Training """
        t_vars = tf.trainable_variables()
        d_vars = [d for d in t_vars if d.name.startswith("discriminator")]
        g_vars = [g for g in t_vars if g.name.startswith("generator")]

        # Optimizer
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(self.g_loss, var_list=g_vars)

        """ Testing """
        self.fake_vectors = self.generator(self.noise, self.labels, self.gnpl, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # final summary operations
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])

    def train(self):
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.vector_dim))

        # saver to save the model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + "/" + self.model_name, self.sess.graph)

        # restore check-point if it exists
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print("Restore checkpoint successfully")

        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print("Restore checkpoint unsuccessful")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_vectors = self.train_vectors[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_labels = self.train_labels[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.vector_dim]).astype(np.float32)

                # update discriminator network
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                                        feed_dict={self.vectors: batch_vectors,
                                                                   self.labels: batch_labels,
                                                                   self.noise: batch_z})
                self.writer.add_summary(summary_str, counter)

                # update generator network
                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss],
                                                        feed_dict={self.vectors: batch_vectors,
                                                                   self.labels: batch_labels,
                                                                   self.noise: batch_z})
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

                # save training result for every 100 steps

            # after an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}".format("predicate1", self.batch_size, self.vector_dim, self.label_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print("Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print("Read {} successfully...".format(ckpt_name))
            return True, counter
        else:
            print("Failed to find checkpoint...")
            return False, 0

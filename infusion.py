from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
md, _ = mnist.train.next_batch(55000)
mnist_mean = np.mean(md, axis=0)
mnist_std = np.std(md, axis=0)


def logit(x):
    eps = 1e-3
    x = tf.minimum(tf.maximum(0.0+eps, x), 1.0-eps)
    return tf.log(x) - tf.log(1-x)

def infuse(z, x, alpha):
    a = tf.random_uniform(tf.shape(z), minval=0, maxval=1)
    z = tf.to_float(a > alpha) * z + tf.to_float(a <= alpha) * x
    return z


class Diffusion(object):
    def __init__(self, steps=15, learning_rate=5e-4):
        def transition(z, step):
            with tf.variable_scope('transition') as vs:
                fc1 = tf.contrib.layers.fully_connected(z, 400, activation_fn=tf.sigmoid)
                mu = tf.contrib.layers.fully_connected(fc1, 784, activation_fn=tf.sigmoid)
                sig = tf.contrib.layers.fully_connected(fc1, 784, activation_fn=tf.sigmoid)
                sig = tf.add(tf.div(sig, step ** 2), 1e-4)
                e = tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
                z_ = tf.add(mu, tf.mul(e, sig))
                z_ = tf.minimum(tf.maximum(0.0, z_), 1.0)
            return z_, mu, sig
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.alpha = tf.placeholder(tf.float32, [])
        z = tf.random_normal(tf.shape(self.x), 0, 1, dtype=tf.float32)
        z = mnist_mean + mnist_std * z
        z = tf.minimum(tf.maximum(0.0, z), 1.0)
        z = infuse(z, self.x, self.alpha)
        self.z = [z]
        self.loss = 0.0
        for t in range(1, steps+1):
            with tf.variable_scope('T{}'.format(t)) as vs:
                z, mu, sig = transition(z, t)
                z = infuse(z, self.x, self.alpha)
                dist = tf.contrib.distributions.Normal(mu=mu, sigma=sig)
                self.loss = self.loss + tf.reduce_mean(-dist.log_pdf(self.x))
                self.z.append(z)

        self.trainer = tf.train.AdamOptimizer(learning_rate=5e-4).minimize(self.loss)
        self.sess = tf.Session()
        self.saver = tf.train.Saver(tf.all_variables())

    def train(self, alpha=0.05, num_epochs=30):
        self.sess.run(tf.initialize_all_variables())
        batch_size = 64
        plt.ion()
        start_time = time.time()
        for epoch in range(0, num_epochs):
            batch_idxs = 1093
            self.visualize()
            for idx in range(0, batch_idxs):
                bx, _ = mnist.train.next_batch(batch_size)
                loss, _ = self.sess.run([self.loss, self.trainer], feed_dict={self.x: bx, self.alpha: alpha})
                if idx % 10 == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, " %
                          (epoch, idx, batch_idxs, time.time() - start_time), end='')
                    print("loss: %4.4f" % loss)
        plt.ioff()
        self.visualize(alpha=0.0, batch_size=20)

    def visualize(self, alpha=0.05, batch_size=10):
        bx, by = mnist.train.next_batch(batch_size)
        if alpha > 0:
            print(np.where(by > 0)[1])
        z = self.sess.run(self.z, feed_dict={self.x: bx, self.alpha: alpha})
        z = [np.reshape(zm, [zm.shape[0], 28, 28]) for zm in z]
        v = np.zeros([z[0].shape[0] * 28, len(z) * 28])
        for b in range(0, z[0].shape[0]):
            for t in range(0, len(z)):
                v[b*28:(b+1)*28, t*28:(t+1)*28] = z[t][b, :]
        plt.imshow(v, cmap='gray')
        plt.show()
        plt.pause(0.01)


class DiffusionDeterministic(object):
    def __init__(self, steps=15, learning_rate=5e-4):
        def transition(z):
            # we return logits instead of sigmoids for numerical stability
            with tf.variable_scope('transition') as vs:
                fc1 = tf.contrib.layers.fully_connected(z, 400, activation_fn=tf.sigmoid)
                mu = tf.contrib.layers.fully_connected(fc1, 784, activation_fn=tf.identity)
            return mu
        self.steps = steps
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.alpha = tf.placeholder(tf.float32, [])
        z = tf.random_normal(tf.shape(self.x), 0, 1, dtype=tf.float32)
        z = mnist_mean + mnist_std * z
        z = tf.minimum(tf.maximum(0.0, z), 1.0)
        z = infuse(z, self.x, self.alpha)
        self.z = [z]
        self.loss = 0.0
        for t in range(1, steps+1):
            with tf.variable_scope('T{}'.format(t)) as vs:
                # in this loop, z are logits
                z = transition(z)
                z = infuse(z, logit(self.x), self.alpha)
                self.loss = self.loss + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(z, self.x))
                self.z.append(tf.sigmoid(z))

        self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        self.sess = tf.Session()
        self.saver = tf.train.Saver(tf.all_variables())

    def train(self, alpha=0.05, num_epochs=30):
        self.sess.run(tf.initialize_all_variables())
        batch_size = 64
        plt.ion()
        start_time = time.time()
        for epoch in range(0, num_epochs):
            batch_idxs = 1093
            self.visualize(alpha=0.05)
            for idx in range(0, batch_idxs):
                bx, _ = mnist.train.next_batch(batch_size)
                loss, _ = self.sess.run([self.loss, self.trainer], feed_dict={self.x: bx, self.alpha: alpha})
                if idx % 10 == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, " %
                          (epoch, idx, batch_idxs, time.time() - start_time), end='')
                    print("loss: %4.4f" % loss)
        plt.ioff()
        self.visualize(alpha=0.0, batch_size=20)

    def visualize(self, alpha=0.05, batch_size=10):
        bx, by = mnist.train.next_batch(batch_size)
        if alpha > 0:
            print(np.where(by > 0)[1])
        z = self.sess.run(self.z, feed_dict={self.x: bx, self.alpha: alpha})
        z = [np.reshape(zm, [zm.shape[0], 28, 28]) for zm in z]
        v = np.zeros([z[0].shape[0] * 28, len(z) * 28])
        for b in range(0, z[0].shape[0]):
            for t in range(0, len(z)):
                v[b*28:(b+1)*28, t*28:(t+1)*28] = z[t][b, :]
        plt.imshow(v, cmap='gray')
        plt.show()
        plt.pause(0.01)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, default='s')
    parser.add_argument('-s', type=int, default=15)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('-e', type=int, default=30)
    parser.add_argument('-a', type=float, default=0.05)
    args = parser.parse_args()
    if args.t == 's':
        c = Diffusion(steps=args.s, learning_rate=args.lr)
    elif args.t == 'd':
        c = DiffusionDeterministic(steps=args.s, learning_rate=args.lr)
    else:
        raise ValueError
    c.train(alpha=args.a, num_epochs=args.e)
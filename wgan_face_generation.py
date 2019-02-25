#!/usr/bin/env python
# coding: utf-8

# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import glob
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
dataset = 'lfw_new_imgs' # LFW
# dataset = 'celeba' # CelebA
images = glob.glob(os.path.join(dataset, '*.*')) 
print(len(images))


batch_size = 48
z_dim = 100
WIDTH = 64
HEIGHT = 64
LAMBDA = 10
DIS_ITERS = 3 # 5

OUTPUT_DIR = 'samples_' + dataset
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

X = tf.placeholder(dtype=tf.float32, shape=[batch_size, 12], name='X')
noise = tf.placeholder(dtype=tf.float32, shape=[batch_size, z_dim], name='noise')
is_training = tf.placeholder(dtype=tf.bool, name='is_training')

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def get_random_batch(filename, batch_size):
    batch = []
    f_lines = []
    real_data = open(filename)
    lines = real_data.readlines()
    linenum = len(lines)
    for i in lines:
        list_to_float = i.strip('\n')
        arr = list_to_float.split(' ')
        a_float_m = map(float,arr)
        f_lines.append(list(a_float_m))
    for i in range(batch_size):
        batch.append(f_lines[random.randint(0, linenum - 1)])
    return batch


def discriminator(data, reuse=None, is_training=is_training):
    momentum = 0.9
    with tf.variable_scope('discriminator', reuse=reuse):
        #h0 = lrelu(tf.layers.conv2d(image, kernel_size=5, filters=64, strides=2, padding='same'))
        h0 = lrelu(tf.layers.dense(data, units=12))
        #h1 = lrelu(tf.layers.conv2d(h0, kernel_size=5, filters=128, strides=2, padding='same'))
        h1 = lrelu(tf.layers.dense(h0,128))
        #h2 = lrelu(tf.layers.conv2d(h1, kernel_size=5, filters=256, strides=2, padding='same'))
        h2 = lrelu(tf.layers.dense(h1, 256))
        #h3 = lrelu(tf.layers.conv2d(h2, kernel_size=5, filters=512, strides=2, padding='same'))
        h3 = lrelu(tf.layers.dense(h2, 512))
        #h4 = tf.contrib.layers.flatten(h3)
        h4 = tf.layers.dense(h3, units=1)
        return h4


def generator(z, is_training=is_training):
    momentum = 0.9
    with tf.variable_scope('generator', reuse=None):
        d = 4
        h0 = tf.layers.dense(z, units=d * d * 512)
        h0 = tf.nn.relu(tf.contrib.layers.batch_norm(h0, is_training=is_training, decay=momentum))

        h1 = tf.layers.dense(h0, 512*2)
        h1 = tf.nn.relu(tf.contrib.layers.batch_norm(h1,is_training= is_training,decay=momentum))

        h2 = tf.layers.dense(h1, 512)
        h2 = tf.nn.relu(tf.contrib.layers.batch_norm(h2, is_training=is_training,decay=momentum))

        h3 = tf.layers.dense(h2, 128)
        h3 = tf.nn.relu(tf.contrib.layers.batch_norm(h3,is_training=is_training, decay=momentum))
        h4 = tf.layers.dense(h3, 12)
        return h4



g = generator(noise)
d_real = discriminator(X)
d_fake = discriminator(g, reuse=True)

loss_d_real = -tf.reduce_mean(d_real)
loss_d_fake = tf.reduce_mean(d_fake)
loss_g = -tf.reduce_mean(d_fake)
loss_d = loss_d_real + loss_d_fake

alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
interpolates = alpha * X + (1 - alpha) * g
grad = tf.gradients(discriminator(interpolates, reuse=True), [interpolates])[0]
slop = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))
gp = tf.reduce_mean((slop - 1.) ** 2)
loss_d += LAMBDA * gp

vars_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer_d = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_d, var_list=vars_d)
    optimizer_g = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_g, var_list=vars_g)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
z_samples = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
samples = []
loss = {'d': [], 'g': []}

for i in tqdm(range(60000)):
    for j in range(DIS_ITERS):
        n = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
        batch = get_random_batch('env2.txt', batch_size)
        _, d_ls = sess.run([optimizer_d, loss_d], feed_dict={X: batch, noise: n, is_training: True})
    
    _, g_ls = sess.run([optimizer_g, loss_g], feed_dict={X: batch, noise: n, is_training: True})
    
    loss['d'].append(d_ls)
    loss['g'].append(g_ls)
    if i % 500 == 0:
        print(i, d_ls, g_ls)
plt.plot(loss['d'], label='Discriminator')
plt.plot(loss['g'], label='Generator')
plt.legend(loc='upper right')
plt.savefig(os.path.join(OUTPUT_DIR, 'Loss.png'))
plt.show()


saver = tf.train.Saver()
saver.save(sess, os.path.join(OUTPUT_DIR, 'wgan_' + dataset), global_step=60000)




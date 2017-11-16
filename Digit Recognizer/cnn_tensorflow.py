# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:18:53 2017

@author: Andrei
"""
# Now trying a CNN

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Initialize output file
cnn_file = open('output/cnn_output.csv', 'w')

cnn_file.write('ImageId,Label\n')

# Read data from csv files
train_dataset = pd.read_csv('input/train.csv')
test_dataset = pd.read_csv('input/test.csv')

# Convert the test dataset
test_pixels = np.array(test_dataset)

# Divide train dataset into train and validation sets
msk = np.random.rand(len(train_dataset)) < 0.8

train = train_dataset[msk]
validation = train_dataset[~msk]

# Isolate only the pixels of the image
train_pixels = np.array(train.iloc[:,1:])
validation_pixels = np.array(validation.iloc[:,1:])

# Isolate the labels
train_labels_array = np.array(train.iloc[:,0:1])
validation_labels_array = np.array(validation.iloc[:,0:1])

# should convert the label array into one-hot array
train_labels = np.zeros((train_labels_array.shape[0], 10))
validation_labels = np.zeros((validation_labels_array.shape[0], 10))

for i in range(train_labels_array.shape[0]):
    train_labels[i][train_labels_array[i]] = 1

for i in range(validation_labels_array.shape[0]):
    validation_labels[i][validation_labels_array[i]] = 1

# Normalize the images
#mean = train_pixels.mean(0)

#for i in range(0, train_pixels.shape[0]):
#    train_pixels[i,:] = train_pixels[i,:] - mean
                 
x = tf.placeholder(tf.float32, [None, 784], name='x')
y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def batch_data(source, target, batch_size):

   # Shuffle data
   shuffle_indices = np.random.permutation(np.arange(len(target)))
   source = source[shuffle_indices]
   target = target[shuffle_indices]

   for batch_i in range(0, len(source)//batch_size):
      start_i = batch_i * batch_size
      source_batch = source[start_i:start_i + batch_size]
      target_batch = target[start_i:start_i + batch_size]

      yield np.array(source_batch), np.array(target_batch)
      print(source_batch)
      print(target_batch)


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
                  
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

softmax = tf.nn.softmax(logits=y_conv)

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        perm = np.random.permutation(train_pixels.shape[0])
        perm = perm[0:150]
        batch = [train_pixels[perm], train_labels[perm]]
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:1.0})
    print('test accuracy %g' % accuracy.eval(feed_dict={x: validation_pixels, y_: validation_labels, keep_prob: 1.0}))
    
    cnn_softmax = softmax.eval(feed_dict={x:test_pixels, y_:validation_labels, keep_prob:1.0})
    cnn_final = np.zeros((cnn_softmax.shape[0]), dtype=np.int32)
    
    for i in range(cnn_softmax.shape[0]):
        for j in range(10):
            if cnn_softmax[i][j] == 1:
                cnn_final[i] = j
   

# Output the results
for i in range(0, cnn_final.shape[0]):
    q = str(i+1) + ',' + str(cnn_final[i]) + '\n'
    cnn_file.write(q)
    


cnn_file.close()
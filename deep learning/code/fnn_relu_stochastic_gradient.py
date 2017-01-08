# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 23:08:08 2017

@author: gbans6
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython.display import display, Image
from scipy import ndimage
from six.moves import cPickle as pickle
import tensorflow as tf

image_size = 28  # Pixel width and height.
num_labels = 10
pixel_depth = 255.0  # Number of levels per pixel.

data_dir_name = 'C:\\Users\\gbans6\\gbansalmba\\mlearning\\mlpython\\data\\nminst\\'

cwd = os.getcwd()
os.chdir(data_dir_name)  

pickle_file = 'notMNIST.pickle'

try:
    f = open(data_dir_name+pickle_file, 'rb')
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    f.close()
    del save
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape) 

train_subset = 10000

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

batch_size = 128
hidden_nodes = 1000

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  hidden1_weights = tf.Variable(
    tf.truncated_normal([image_size * image_size, hidden_nodes]))
  hidden1_biases = tf.Variable(tf.zeros([hidden_nodes]))
  
  hidden1 = tf.nn.relu(tf.matmul(tf_train_dataset, hidden1_weights) + hidden1_biases)

  hidden2_weights = tf.Variable(
    tf.truncated_normal([hidden_nodes, hidden_nodes]))
  hidden2_biases = tf.Variable(tf.zeros([hidden_nodes]))
  
  hidden2 = tf.nn.relu(tf.matmul(hidden1, hidden2_weights) + hidden2_biases)
  
  # Variables.
  weights = tf.Variable(tf.truncated_normal([hidden_nodes, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))

  print("no error till here")
  # Training computation.
  logits = tf.matmul(hidden2, weights) + biases
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.04).minimize(loss)
  
  print("no error till optimization")
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  
  hidden1_output = tf.nn.relu(tf.matmul(tf_valid_dataset, hidden1_weights) + hidden1_biases)
  hidden2_output = tf.nn.relu(tf.matmul(hidden1_output, hidden2_weights) + hidden2_biases)
  valid_prediction = tf.nn.softmax(tf.matmul(hidden2_output, weights) + biases)
  
  hidden1_output = tf.nn.relu(tf.matmul(tf_test_dataset, hidden1_weights) + hidden1_biases)
  hidden2_output = tf.nn.relu(tf.matmul(hidden1_output, hidden2_weights) + hidden2_biases)
  test_prediction = tf.nn.softmax(tf.matmul(hidden2_output, weights) + biases)
  
num_steps = 30001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
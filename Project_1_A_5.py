# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 17:01:22 2018

@author: Anton
"""

#
# Project 1, Part A, Question 5
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

NUM_FEATURES = 36 # nbr of inputs per sample (4*3*3)
NUM_CLASSES = 6 # output classes
NUM_HIDDEN = 10 # 10 neurons in hidden layer

learning_rate = 0.01 # alpha
beta = 10**-6 # weight decay parameter
epochs = 1000
batch_size = 32 # 32 samples per training batch
seed = 10
np.random.seed(seed)

#read train data
train_input = np.loadtxt('sat_train.txt',delimiter=' ') # read data to rows of values (float)
trainX, train_Y = train_input[:,0:36], train_input[:,-1].astype(int) # first 36 columns are inputs, last column is class
trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0)) # rescale data with range and min-value
train_Y[train_Y == 7] = 6 # change classes where class == 7 to class 6

trainY = np.zeros((train_Y.shape[0], NUM_CLASSES)) # empty zero matrix with domensions 4435 x 6
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 # train_1-1 to reduce classes to 0-5 (0-indexing), one hot matrix

# Read testing data
test_input = np.loadtxt('sat_test.txt', delimiter=' ')
testX, test_Y = test_input[:,0:36], test_input[:,-1].astype(int)
testX = scale(testX, np.min(testX, axis=0), np.max(testX, axis=0))
test_Y[test_Y == 7] = 6
testY = np.zeros((test_Y.shape[0], NUM_CLASSES))
testY[np.arange(test_Y.shape[0]), test_Y-1] = 1
    
# experiment with small datasets
#trainX = trainX[:1000]
#trainY = trainY[:1000]
n = trainX.shape[0]

# Create the model
x = tf.placeholder(tf.float32, [None, NUM_FEATURES]) # placeholder for the input vector
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES]) # placeholder for the output vector

# Build the graph for the deep net

# model parameters
# initialize softmax_weights to truncated normal distribution, softmax_biases to 0s	
hidden_weights1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, NUM_HIDDEN], stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='softmax_weights')
hidden_biases1  = tf.Variable(tf.zeros([NUM_HIDDEN]), name='hidden_biases')

hidden_weights2 = tf.Variable(tf.truncated_normal([NUM_HIDDEN, NUM_HIDDEN], stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='softmax_weights')
hidden_biases2  = tf.Variable(tf.zeros([NUM_HIDDEN]), name='hidden_biases')

softmax_weights = tf.Variable(tf.truncated_normal([NUM_HIDDEN, NUM_CLASSES], stddev=1.0/math.sqrt(float(NUM_HIDDEN))), name='softmax_weights')
softmax_biases  = tf.Variable(tf.zeros([NUM_CLASSES]), name='softmax_biases')

z1 = tf.matmul(x, hidden_weights1) + hidden_biases1
h1 = tf.nn.sigmoid(z1)
z2 = tf.matmul(h1, hidden_weights2) + hidden_biases2
h2 = tf.nn.sigmoid(z2)
logits  = tf.matmul(h2, softmax_weights) + softmax_biases # h*softmax_weights + softmax_biases (logits = unscaled log-probabilities)
# Original loss function
cross_entropy = (tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)) # logits input to softmax
#loss = tf.reduce_mean(cross_entropy)
# L2 norms regularization for both the hidden and softmax layer
regularizers = tf.nn.l2_loss(softmax_weights) + tf.nn.l2_loss(hidden_weights1) + tf.nn.l2_loss(hidden_weights2)
# Regularized loss
loss = tf.reduce_mean(cross_entropy + beta * regularizers)

# Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate) # optimizer object, optimizes all weights and biases
train_op = optimizer.minimize(loss)

correct_prediction = tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(y_, 1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_acc = [] 
    test_acc = []
    idx = np.arange(n)
    for i in range(epochs):
        np.random.shuffle(idx) # shuffle indeces
        trainX, trainY = trainX[idx], trainY[idx] # shuffle data with new indeces
        
        # for loop with 32 data points at a time, chunks from the 32 in the data array
        for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
            train_op.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})
         
        test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY})) # sasve accurracy for every epoch   
        train_acc.append(accuracy.eval(feed_dict={x: trainX, y_: trainY})) # store acurracy for every epoch        
        if i % 100 == 0:
            print('iter %d: accuracy %g'%(i, train_acc[i]))

# plot learning curves
plt.figure(1)
plt.title('Train Accurracy for 4 layer network')
plt.plot(range(epochs), train_acc)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Train accuracy')
plt.show()

plt.figure(2)
plt.title('Test Accurracy for 4 layer network')
plt.plot(range(epochs), test_acc)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Test accuracy')
plt.show()



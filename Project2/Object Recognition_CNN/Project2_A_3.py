# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 12:28:15 2018

@author: Anton
"""

#
# Project 2, Part A, Question 3
#

#import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle
import os

if not os.path.isdir('figuresA3'):
    print('Creating the figures folder')
    os.makedirs('figuresA3')
    
#TODO: Grid search suitable parameters, vary seed size. Early stopping, try more iterations, AlexNet...
# Save figures
    
NUM_CLASSES = 10 # 10 object classes
IMG_SIZE = 32 # 32x32 pixels
NUM_CHANNELS = 3 # RGB channels
NUM_FCONNECTED = 300 # Fully connected layer
#TODO: Change to optimal nbr of features
opt_num_ftr1 = 5 #################### TO BE FOUND IN Q2!
opt_num_ftr2 = 5 #################### TO BE FOUND IN Q2!
learning_rate = 0.001 # alpha
epochs = 1
batch_size = 128
gamma = 0.1 # momentum parameter

# for discussion: https://www.reddit.com/r/MachineLearning/comments/42nnpe/why_do_i_never_see_dropout_applied_in/

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

def load_data(file):
    with open(file, 'rb') as fo: # open file in read binary mode
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  #python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')

    data, labels = samples['data'], samples['labels']
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)    
    labels_ = np.zeros([labels.shape[0], NUM_CLASSES])
    labels_[np.arange(labels.shape[0]), labels-1] = 1 # one hot matrix for classes
    
    # Experiment with smaller data size
    data = data[:250]
    labels_ = labels_[:250]
    return data, labels_

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def cnn(images, keep_prob_cnn, keep_prob_fc):

    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS]) # shape = [?, 32, 32, 3]
    
    # Convolutional layer 1, 9x9 Window
    W_conv1 = weight_variable([9, 9, NUM_CHANNELS, opt_num_ftr1]) # [9, 9, 3, 50]
    b_conv1 = bias_variable([opt_num_ftr1])
    u_conv1 = tf.nn.conv2d(images, W_conv1, strides=[1, 1, 1, 1], padding='VALID') + b_conv1
    conv_1_dropout = tf.nn.dropout(tf.nn.relu(u_conv1), keep_prob_cnn)

    # Pooling layer 1, downsampling by 2
    pool_1 = tf.nn.max_pool(conv_1_dropout, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_1')
    
    # Convolutional layer 2, 5x5 Window
    W_conv2 = weight_variable([5, 5, opt_num_ftr1, opt_num_ftr2])
    b_conv2 = bias_variable([opt_num_ftr2])
    u_conv2 = tf.nn.conv2d(pool_1, W_conv2, strides=[1, 1, 1, 1], padding='VALID') + b_conv2
    conv_2_dropout = tf.nn.dropout(tf.nn.relu(u_conv2), keep_prob_cnn)

	 # Pooling layer 2, downsampling by 2
    pool_2 = tf.nn.max_pool(conv_2_dropout, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_1')
    # Flatten pool layer
    dim = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value
    pool_2_flat = tf.reshape(pool_2, [-1, dim])
   
    # F3: Fully connected layer, size 300
    W_fc = weight_variable([dim, NUM_FCONNECTED])
    b_fc = bias_variable([NUM_FCONNECTED])
    u_fc = tf.matmul(pool_2_flat, W_fc) + b_fc
    z_fc_dropout = tf.nn.dropout(tf.nn.relu(u_fc), keep_prob_fc)

    # F4: Softmax layer for output
    W_softmax = weight_variable([NUM_FCONNECTED, NUM_CLASSES])
    b_softmax = bias_variable([NUM_CLASSES])
    logits = tf.matmul(z_fc_dropout, W_softmax) + b_softmax
    return logits

def runModel(keep_prob_cnn, keep_prob_fc):
    trainX, trainY = load_data('data_batch_1') # load data from file
    testX, testY = load_data('test_batch_trim')
    
    # Scale data
    trainX = (trainX - np.min(trainX, axis = 0))/np.max(trainX, axis = 0) # pixel scaling 0 to 1
    testX = (testX - np.min(testX, axis = 0))/np.max(testX, axis = 0) # pixel scaling 0 to 1

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE*NUM_CHANNELS]) # 32x32x3, input image
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    # Build the layers
    logits = cnn(x, 1.0, 1.0)
    
    # Loss function to minimize
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    # Class predictions and accuracy
    prediction = tf.nn.softmax(logits)
    correct_prediction = tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    # Optimizers for training
    train_op1 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    train_op2 = tf.train.MomentumOptimizer(learning_rate, gamma).minimize(loss)
    train_op3 = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    train_op4 = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    train_ops = [train_op1, train_op2, train_op3, train_op4]
    
    # Initialize for plotting purposes
    fig1 = plt.figure(1, figsize=(10,5))
    plt.xlabel(str(epochs) + ' iterations/epochs')
    plt.ylabel('Cross entropy')
    fig1.suptitle('Training Cost (Cross entropy)')

    fig2 = plt.figure(2, figsize=(10,5))
    plt.xlabel(str(epochs) + ' iterations/epochs')
    plt.ylabel('Test accuracy')
    fig2.suptitle('Top 1 Test Accurracy')

    N = len(trainX)
    idx = np.arange(N)
    
    for train_op in train_ops:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_cost = []
            test_acc = []
            
            for e in range(epochs):
                np.random.shuffle(idx)
                trainX, trainY = trainX[idx], trainY[idx]
                for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                    train_op.run(feed_dict={x: trainX[start:end,:], y_: trainY[start:end,:]})
      
                loss_ = sess.run(loss, {x: trainX, y_: trainY})
                test_acc_ = accuracy.eval(feed_dict={x: testX, y_: testY})
                train_cost.append(loss_)
                test_acc.append(test_acc_) # save accurracy for every epoch 
                print('epoch', e, 'entropy', loss_, 'Test Acc', test_acc_*100, '%')
            
            plt.figure(1)
            plt.plot(range(epochs), train_cost, label='test')
            plt.figure(2)
            plt.plot(range(epochs), test_acc)
                  
    fig1.legend(['GD','Momentum, 'r'$\gamma$ = ' + str(gamma),'RMSProp','Adam'],loc='lower right')
    fig2.legend(['GD','Momentum, 'r'$\gamma$ = ' + str(gamma),'RMSProp','Adam'],loc='lower right')
#    fig1.show()
#    fig2.show()

    fig1.savefig('./figuresA3/PartA_3_TrainError' + str(keep_prob_cnn) + str(keep_prob_fc)+'.png')
    fig2.savefig('./figuresA3/PartA_3_TestAcc' + str(keep_prob_cnn) + str(keep_prob_fc) +'.png')

def main():
    print('Running model WITHOUUT dropout')
    runModel(1, 1) # No Droupout
    print('Running model WITH dropout')
    runModel(0.7, 0.5) # Dropout
    
if __name__ == '__main__':
  main()

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 17:23:32 2018

@author: Anton
"""

#
# Project 2, Part A, Question 2
#

#import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle
import os
import itertools
import timeit

if not os.path.isdir('figuresA2'):
    print('Creating the figures folder')
    os.makedirs('figuresA2')
    
NUM_CLASSES = 10 # 10 object classes
IMG_SIZE = 32 # 32x32 pixels
NUM_CHANNELS = 3 # RGB channels
NUM_FCONNECTED = 300 # Fully connected layer
learning_rate = 0.001 # 0.001
epochs = 10
batch_size = 128

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


def cnn(images, nbr_ftrMaps1, nbr_ftrMaps2):

    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS]) # shape = [?, 32, 32, 3]
    
    # Convolutional layer 1, 9x9 Window
    W_conv1 = weight_variable([9, 9, NUM_CHANNELS, nbr_ftrMaps1]) # [9, 9, 3, 50]
    b_conv1 = bias_variable([nbr_ftrMaps1])
    u_conv1 = tf.nn.conv2d(images, W_conv1, strides=[1, 1, 1, 1], padding='VALID') + b_conv1
    conv_1 = tf.nn.relu(u_conv1)
    
    # Pooling layer 1, downsampling by 2
    pool_1 = tf.nn.max_pool(conv_1, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_1')
    
    # Convolutional layer 2, 5x5 Window
    W_conv2 = weight_variable([5, 5, nbr_ftrMaps1, nbr_ftrMaps2])
    b_conv2 = bias_variable([nbr_ftrMaps2])
    u_conv2 = tf.nn.conv2d(pool_1, W_conv2, strides=[1, 1, 1, 1], padding='VALID') + b_conv2
    conv_2 = tf.nn.relu(u_conv2)

	 # Pooling layer 2, downsampling by 2
    pool_2 = tf.nn.max_pool(conv_2, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_1')
    # Flatten pool layer
    dim = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value
    pool_2_flat = tf.reshape(pool_2, [-1, dim])
   
    # F3: Fully connected layer, size 300
    W_fc = weight_variable([dim, NUM_FCONNECTED])
    b_fc = bias_variable([NUM_FCONNECTED])
    u_fc = tf.matmul(pool_2_flat, W_fc) + b_fc
    z_fc = tf.nn.relu(u_fc)
    
    # F4: Softmax layer for output
    W_softmax = weight_variable([NUM_FCONNECTED, NUM_CLASSES])
    b_softmax = bias_variable([NUM_CLASSES])
    logits = tf.matmul(z_fc, W_softmax) + b_softmax

    return logits


def runModel(nbr_ftrMaps1, nbr_ftrMaps2):    
    trainX, trainY = load_data('data_batch_1') # load data from file
    testX, testY = load_data('test_batch_trim')
    
    # Scale data
    trainX = (trainX - np.min(trainX, axis = 0))/np.max(trainX, axis = 0) # pixel scaling 0 to 1
    testX = (testX - np.min(testX, axis = 0))/np.max(testX, axis = 0) # pixel scaling 0 to 1

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE*NUM_CHANNELS]) # 32x32x3, input image
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    # Build the layers
    logits = cnn(x, nbr_ftrMaps1, nbr_ftrMaps2)
    
    # Loss function to minimize
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    
    # Class predictions and accuracy
    prediction = tf.nn.softmax(logits)
    correct_prediction = tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    # Optimizer for training
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    N = len(trainX)
    idx = np.arange(N)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_cost = [] 
        test_acc = []
        epoch_trainingTime = []
        
        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]
            
            beginTimer = timeit.default_timer()

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_op.run(feed_dict={x: trainX[start:end,:], y_: trainY[start:end,:]})
                
            stopTimer = timeit.default_timer()
            diffTime = stopTimer-beginTimer
            epoch_trainingTime.append(diffTime)

            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY})) # save accurracy for every epoch   
            loss_ = sess.run(loss, {x: trainX, y_: trainY})
            train_cost.append(loss_)

            print('epoch', e, 'entropy', loss_)
                
        modelData = np.zeros((3,epochs))
        modelData[0, :] = train_cost
        modelData[1, :] = test_acc
        modelData[2, :] = epoch_trainingTime
    return modelData

def main():
    # Initialize for plotting purposes
    train_cost = []
    test_acc = []
    avgtime = []
    legend = []
    fig1 = plt.figure(2, figsize=(10,5))
    ax1 = fig1.add_subplot(111)
    ax1.set_title('Training Cost (Cross entropy)')
    ax1.set_xlabel(str(epochs) + ' iterations/epochs')
    ax1.set_ylabel('Cross entropy')
    
    fig2 = plt.figure(3, figsize=(10,5))
    ax2 = fig2.add_subplot(111)
    ax2.set_title('Top 1 Test Accurracy')
    ax2.set_xlabel(str(epochs) + ' iterations/epochs')
    ax2.set_ylabel('Test accuracy')
    
    c1Fmaps = np.array([5,10,25,50])# nbr of filters (feature maps)
    c2Fmaps = np.array([5,10,25,50])
    for paramSet in itertools.product(c1Fmaps,c2Fmaps): 
        print(paramSet[0], paramSet[1])
        modelData = runModel(paramSet[0], paramSet[1])
        train_cost = modelData[0]
        test_acc = modelData[1]
        print(np.mean(modelData[2]))
        avgtime.append(np.mean(modelData[2])) # average out how much time to train ONE epoch
        
        ax1.plot(range(epochs), train_cost)
        ax2.plot(range(epochs), test_acc)
        print('---------------FINAL TRAINING COST: ', round(train_cost[epochs-1],4),' -------------------')
        print('---------------FINAL TEST ACCURACY: ', round(test_acc[epochs-1],4),' -------------------')
        # Store pairs for plot legend
        stringPair = str(paramSet[0]) + ', ' + str(paramSet[1])
        legend.append(stringPair)
        
    fig1.legend(legend)
    fig1.savefig('./figuresA2/PartA_2_TrainError.png')
    fig1.show()
    
    fig2.legend(legend)
    fig2.savefig('./figuresA2/PartA_2_TestAcc.png')
    fig2.show()
    
    plt.figure(1)
    plt.title('Average time for one epoch for different nbr of feature maps')
    plt.scatter(legend, avgtime)
    plt.xlabel('Sets of feature maps')
    plt.ylabel('Average epoch training time')
    plt.savefig('./figuresA2/PartA_2_TrainTime.png')
    plt.show()
if __name__ == '__main__':
  main()

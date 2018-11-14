# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 11:47:19 2018

@author: Anton
"""

#
# Project 2, Part A, Question 1
#

#import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle
import os

if not os.path.isdir('figuresA1'):
    print('Creating the figures folder')
    os.makedirs('figuresA1')
        
NUM_CLASSES = 10 # 10 object classes
IMG_SIZE = 32 # 32x32 pixels
NUM_CHANNELS = 3 # RGB channels
NUM_FILTERS_C1 = 50# Filters in Convolution layer 1
NUM_FILTERS_C2 = 60 # Filters in Convolution layer 2
NUM_FCONNECTED = 300 # Fully connected layer
learning_rate = 0.001 # 0.001
epochs = 300
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
#    data = data[:500]
#    labels_ = labels_[:500]
    return data, labels_

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
  
def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def cnn(images):

    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS]) # shape = [?, 32, 32, 3]
    
    # Convolutional layer 1, 9x9 Window
    W_conv1 = weight_variable([9, 9, NUM_CHANNELS, NUM_FILTERS_C1]) # [9, 9, 3, 50]
    b_conv1 = bias_variable([NUM_FILTERS_C1])
    u_conv1 = tf.nn.conv2d(images, W_conv1, strides=[1, 1, 1, 1], padding='VALID') + b_conv1
    conv_1 = tf.nn.relu(u_conv1)
    
    # Pooling layer 1, downsampling by 2
    pool_1 = tf.nn.max_pool(conv_1, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_1')
    
    # Convolutional layer 2, 5x5 Window
    W_conv2 = weight_variable([5, 5, NUM_FILTERS_C1, NUM_FILTERS_C2])
    b_conv2 = bias_variable([NUM_FILTERS_C2])
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

    return logits, conv_1, conv_2, pool_1, pool_2

def main():    
    trainX, trainY = load_data('data_batch_1') # load data from file
    print('trainX shape and trainY shape')
    print(trainX.shape, trainY.shape)
    
    testX, testY = load_data('test_batch_trim')
    print('testX shape and testY shape')
    print(testX.shape, testY.shape)

    trainX = (trainX - np.min(trainX, axis = 0))/np.max(trainX, axis = 0) # pixel scaling 0 to 1
    testX = (testX - np.min(testX, axis = 0))/np.max(testX, axis = 0) # pixel scaling 0 to 1

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE*NUM_CHANNELS]) # 32x32x3, input image
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    # Build the layers
    logits, conv_1, conv_2, pool_1, pool_2 = cnn(x)
    
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
        for e in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_op.run(feed_dict={x: trainX[start:end,:], y_: trainY[start:end,:]})

            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY})) # save accurracy for every epoch   
            loss_ = sess.run(loss, {x: trainX, y_: trainY})
            train_cost.append(loss_)

            print('epoch', e, 'entropy', loss_)
            
        # After training, plot test images with corresponding convolutional and pooling feature maps
        # Predictions for the test images are printed
        nbrTestImages = 2
        for nbrimg in range(nbrTestImages):
            ind = np.random.randint(low=0, high=len(testX))
            X = testX[ind]
            img = X.reshape(NUM_CHANNELS, IMG_SIZE, IMG_SIZE).transpose(1, 2, 0) # Original image
            X = X.reshape(1, IMG_SIZE*IMG_SIZE*NUM_CHANNELS) # Transpose
            conv_1_, conv_2_, pool_1_, pool_2_ = sess.run([conv_1, conv_2, pool_1, pool_2], feed_dict={x: X})
            
            # Evaluate class prediction for the test image
            prediction_ = sess.run(prediction, feed_dict={x: X})
            # Extract the top 5 most likely classes
            idxs = np.argsort(prediction_, axis=1)
            rev = np.fliplr(idxs)
            np.set_printoptions(precision=4,suppress=True)
            print('The top 5 predicted classes, left is most probable:\n', rev[:,:5])
            
            # Plot original image'
            print('Original test image with class ', np.argmax(testY[ind]))
            plt.figure()
            plt.gray()
            plt.subplot(1, 1, 1); plt.axis('off'); plt.imshow(img)
            plt.savefig('./figuresA1/PartA_1_Img' + str(nbrimg) + '.png')
            plt.show()
            
            # Plot convolutional feature maps
            print('Convolution layer 1 feature maps')
            plt.figure(figsize=(8,8))
            plt.gray()
            for i in range(NUM_FILTERS_C1):
                plt.subplot(10, 5, i+1); plt.axis('off'); plt.imshow(conv_1_[0, :, :, i])
            plt.savefig('./figuresA1/PartA_1_Conv1' + str(nbrimg) + '.png')
            plt.show()
            
            # Plot pooling feature maps
            print('Pool 1 feature maps (MAX pooling)')
            plt.figure(figsize=(8,8))
            plt.gray()
            for i in range(NUM_FILTERS_C1):
                plt.subplot(10, 5, i+1); plt.axis('off'); plt.imshow(pool_1_[0, :, :, i])
            plt.savefig('./figuresA1/PartA_1_Pool1' + str(nbrimg) + '.png')
            plt.show()        
            
            # Convolution and pool layer 2
            print('Convolution layer 2 feature maps')
            plt.figure(figsize=(8,8))
            plt.gray()
            for i in range(NUM_FILTERS_C1):
                plt.subplot(10, 5, i+1); plt.axis('off'); plt.imshow(conv_2_[0, :, :, i])
            plt.savefig('./figuresA1/PartA_1_Conv2' + str(nbrimg) + '.png')
            plt.show()
            
            print('Pool 2 feature maps (MAX pooling)')
            plt.figure(figsize=(8,8))
            plt.gray()
            for i in range(NUM_FILTERS_C1):
                plt.subplot(10, 5, i+1); plt.axis('off'); plt.imshow(pool_2_[0, :, :, i])
            plt.savefig('./figuresA1/PartA_1_Pool2' + str(nbrimg) + '.png')
            plt.show()      

    plt.figure(1)
    plt.title('Training Cost (Cross entropy)')
    plt.plot(range(epochs), train_cost)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Cross entropy')
    plt.savefig('./figuresA1/PartA_1_TrainCost.png')
    plt.show()
    
    plt.figure(2)
    plt.title('Top 1 Test Accurracy')
    plt.plot(range(epochs), test_acc)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Test accuracy')
    plt.savefig('./figuresA1/PartA_1_TestAcc.png')
    plt.show()

if __name__ == '__main__':
  main()

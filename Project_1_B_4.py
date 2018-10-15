# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 16:04:36 2018

@author: Anton
"""

#
# Project 1, Part B, question 4
#

import tensorflow as tf
import numpy as np
import pylab as plt
import sys
import time
import math

NUM_FEATURES = 8

num_hidden1 = 100 # use optimum nbr of neurons
num_hidden_rest = 20
epochs = 1000
batch_size = 32
learning_rate = 10**-9
no_folds = 5
prob = 0.9 # Keep probability for dropout

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)
beta = 10**-3

def getData():
    print('Fetching data...')
    #read and divide data into test and train sets 
    cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
    X_data, Y_data = cal_housing[:,:8], cal_housing[:,-1]
    Y_data = (np.asmatrix(Y_data)).transpose()
    
    idx = np.arange(X_data.shape[0])
    np.random.shuffle(idx)
    X_data, Y_data = X_data[idx], Y_data[idx]
    
    m = 3* X_data.shape[0] // 10
    trainX, trainY = X_data[m:], Y_data[m:]
    
    trainX = (trainX- np.mean(trainX, axis=0))/ np.std(trainX, axis=0)
    
#    trainX = trainX[:0.7*n]
#    testX = trainX[0.7*n+1:]
#    print(str(trainX.shape[0]))
#    print(str(testX.shape[0]))
#    # experiment with small datasets
#    trainX = trainX[:500]
#    trainY = trainY[:500]
    
    # Divide data into validation and testing data
    n = trainX.shape[0]
    print('Length of trainX before dividing the data: ' + str(n))
    validationdata = n // (1.0/0.7) # divide data into 70% validation (training) data and 30% testing data
    validationdata = int(validationdata)
    testX = trainX[validationdata:]
    testY = trainY[validationdata:]
    trainX = trainX[:validationdata]
    trainY = trainY[:validationdata]
    print('Number of data points in testX: ' + str(testX.shape[0]))
    print('Number of data points in trainX: ' + str(trainX.shape[0]))
    return trainX, trainY, testX, testY

def runModel(train_X, train_Y, test_X, test_Y):
#    trainX, trainY, testX, testY = getData()
    trainX, trainY, testX, testY = train_X, train_Y, test_X, test_Y
    
    # 5-fold Cross Validation
    n = trainX.shape[0]
    trainX = trainX[:len(trainX) - n%no_folds] # making sure the data is evenly divisible by 5
    data_per_fold = trainX.shape[0] / no_folds
    
    #TODO Remove K-fold validation (according to prof email)
    for fold in range(no_folds):
        start, end = int(fold*data_per_fold), int((fold+1)*data_per_fold)
        x_validation, y_validation = trainX[start:end], trainY[start:end] # 1 / 5 of the data for validating model
        
        x_train = np.append(trainX[:start], trainX[end:], axis=0) # Rest of the data to train the model
        y_train = np.append(trainY[:start], trainY[end:], axis=0)
        keep_prob = tf.placeholder(tf.float32)

        # Create the model
        x = tf.placeholder(tf.float32, [None, NUM_FEATURES]) # 8 inputs
        y_ = tf.placeholder(tf.float32, [None, 1]) # linear output neuron
        
        # Build the graph for the deep net
        hidden_weights1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_hidden1], stddev=1.0 / np.sqrt(NUM_FEATURES), dtype=tf.float32))
        hidden_biases1 = tf.Variable(tf.zeros([num_hidden1]), dtype=tf.float32)
        
        hidden_weights2 = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden_rest], stddev=1.0 / np.sqrt(num_hidden1), dtype=tf.float32))
        hidden_biases2 = tf.Variable(tf.zeros([num_hidden_rest]), dtype=tf.float32)
        
        hidden_weights3 = tf.Variable(tf.truncated_normal([num_hidden_rest, num_hidden_rest], stddev=1.0 / np.sqrt(num_hidden_rest), dtype=tf.float32))
        hidden_biases3 = tf.Variable(tf.zeros([num_hidden_rest]), dtype=tf.float32)
        
        output_weights = tf.Variable(tf.truncated_normal([num_hidden_rest, 1], stddev=1.0 / np.sqrt(num_hidden_rest), dtype=tf.float32))
        output_bias = tf.Variable(tf.zeros([1]), dtype=tf.float32)
        
        u1 = tf.matmul(x, hidden_weights1) + hidden_biases1
        h1 = tf.nn.relu(u1)
        h1_dropout = tf.nn.dropout(h1, keep_prob)

        u2 = tf.matmul(h1_dropout, hidden_weights2) + hidden_biases2
        h2 = tf.nn.relu(u2)
        h2_dropout = tf.nn.dropout(h2, keep_prob)
        
        u3 = tf.matmul(h2_dropout, hidden_weights3) + hidden_biases3
        h3 = tf.nn.relu(u3)
        h3_dropout = tf.nn.dropout(h3, keep_prob)
        
        y = tf.matmul(h3_dropout, output_weights) + output_bias # output layer
        
        regularizers = tf.nn.l2_loss(hidden_weights1) + tf.nn.l2_loss(hidden_weights2) + tf.nn.l2_loss(hidden_weights3) + tf.nn.l2_loss(output_weights)
    
        loss = tf.reduce_mean(tf.square(y_ - y) + beta * regularizers) # reguralized loss
        
        #Create the gradient descent optimizer with the given learning rate.
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
        error = tf.reduce_mean(tf.square(y_ - y)) # MSE (mean square error)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_err = []
            test_err = []
            cross_val_error = []
            n = x_train.shape[0]
            idx = np.arange(n)
            
            for i in range(epochs):
                np.random.shuffle(idx)
                x_train, y_train = x_train[idx], y_train[idx]
                
                for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
                    train_op.run(feed_dict={x: x_train[start:end], y_: y_train[start:end], keep_prob: prob})
                
                err = error.eval(feed_dict={x: x_train, y_: y_train, keep_prob: 1.0})
                train_err.append(err)
                
                test_error = error.eval(feed_dict={x: testX, y_: testY, keep_prob: 1.0}) # errors for each epoch, to be plotted for best model
                test_err.append(test_error)
        
                if i % 100 == 0:
                    print('iter %d: current train error %g'%(i, train_err[i]))
                    
            validation_err = error.eval(feed_dict={x: x_validation, y_:y_validation, keep_prob: 1.0}) # errors after training
            cross_val_error.append(validation_err)
            
#         plot learning curves
        plt.figure(1)
        plt.title('Training error, learning rate ' + str(learning_rate))
        plt.plot(range(epochs), train_err)
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Train Error')
        plt.show()
    mean_cve = np.mean(cross_val_error) # average cross validation error over the 5 folds
#    print('The cross validation error for model with learning parameter ' + str(learning_rate) + ' : ' + str(mean_cve))
    return mean_cve, test_err

def main():
    test_err = []
    trainX, trainY, testX, testY = getData()
    
    cross_val_error, test_err = runModel(trainX, trainY, testX, testY)

    # plot test errors for best model
    plt.figure(1)
    plt.plot(range(epochs), test_err)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Test Errors model')
    plt.show()
    print('----- FINAL TEST ERROR: ' + str(test_err[epochs-1]) + ' -----')
    print('----- FINAL TEST RMSE: ' + str(math.sqrt(test_err[epochs-1])) + ' -----')

#    fig = plt.figure(2)
#    ax1 = fig.add_subplot(211)
#    ax1.set_title('Mean cross validation error')
#    ax1.scatter([1,2,3,4,5, 6], cross_val_errors)
#    ax1.xaxis.set_ticks([1,2,3,4,5,6])
#    ax1.xaxis.set_ticklabels(neuronSS)
#    ax1.set_xlabel('Number of neurons in hidden ReLU layer')
#    ax1.set_ylabel('Cross validation error')
        
if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- Total execution time: %s seconds ---" % (time.time() - start_time))

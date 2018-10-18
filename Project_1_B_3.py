# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 16:50:59 2018

@author: Anton
"""

#
# Project 1, Part B, question 3
#

import tensorflow as tf
import numpy as np
import pylab as plt
import sys
import time
import math
import os

if not os.path.isdir('figuresB3'):
    print('Creating the figures folder')
    os.makedirs('figuresB3')
    
NUM_FEATURES = 8

epochs = 500
batch_size = 32
learning_rate = 10**-7
no_folds = 5

seed = 10
np.random.seed(seed)
beta = 10**-6

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
    
#    # experiment with small datasets
#    trainX = trainX[:200]
#    trainY = trainY[:200]
    
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

def runModel(num_neuron, train_X, train_Y, test_X, test_Y):
#    trainX, trainY, testX, testY = getData()
    trainX, trainY, testX, testY = train_X, train_Y, test_X, test_Y
    
    # 5-fold Cross Validation
    n = trainX.shape[0]
    trainX = trainX[:len(trainX) - n%no_folds] # making sure the data is evenly divisible by 5
    data_per_fold = trainX.shape[0] / no_folds
    
    for fold in range(no_folds):
        start, end = int(fold*data_per_fold), int((fold+1)*data_per_fold)
        x_validation, y_validation = trainX[start:end], trainY[start:end] # 1 / 5 of the data for validating model
        x_train = np.append(trainX[:start], trainX[end:], axis=0) # Rest of the data to train the model
        y_train = np.append(trainY[:start], trainY[end:], axis=0)
        
        # Create the model
        x = tf.placeholder(tf.float32, [None, NUM_FEATURES]) # 8 inputs
        y_ = tf.placeholder(tf.float32, [None, 1]) # linear output neuron
        
        # Build the graph for the deep net
        hidden_weights = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_neuron], stddev=1.0 / np.sqrt(NUM_FEATURES), dtype=tf.float32))
        hidden_biases = tf.Variable(tf.zeros([num_neuron]), dtype=tf.float32)
        
        output_weights = tf.Variable(tf.truncated_normal([num_neuron, 1], stddev=1.0 / np.sqrt(num_neuron), dtype=tf.float32))
        output_bias = tf.Variable(tf.zeros([1]), dtype=tf.float32)
        
        u = tf.matmul(x, hidden_weights) + hidden_biases
        h = tf.nn.relu(u)
        y = tf.matmul(h, output_weights) + output_bias # output layer
        
        regularizers = tf.nn.l2_loss(hidden_weights) + tf.nn.l2_loss(output_weights)
    
        loss = tf.reduce_mean(tf.square(y_ - y) + beta * regularizers) # MSE (mean square error)
        
        #Create the gradient descent optimizer with the given learning rate.
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
        error = tf.reduce_mean(tf.square(y_ - y))
        
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
                    train_op.run(feed_dict={x: x_train[start:end], y_: y_train[start:end]})
                
                err = error.eval(feed_dict={x: x_train, y_: y_train})
                train_err.append(err)
                
                test_error = error.eval(feed_dict={x: testX, y_: testY}) # errors for each epoch, to be plotted for best model
                test_err.append(test_error)
        
                if i % 100 == 0:
                    print('iter %d: current train error %g'%(i, train_err[i]))
                    
            validation_err = error.eval(feed_dict={x: x_validation, y_:y_validation}) # errors after training
            cross_val_error.append(validation_err)
#            pred = sess.run(y, feed_dict={x: testX[:50]}) # final test predictions

            
#         plot learning curves
        plt.figure(1)
        plt.title('Training error, learning rate ' + str(learning_rate))
        plt.plot(range(epochs), train_err)
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Train Error')
        plt.show()
        
#        plt.figure(2)
#        targets = np.asarray(testY[:50])
#        fig = plt.figure(figsize=(10,5))
#        ax1 = fig.add_subplot(111)
#        ax1.set_title('Targets and preditions')
#        ax1.scatter(range(50), pred, color='blue', marker='.', label='Targets')
#        ax1.scatter(range(50), targets, color='red', marker='x', label='Predictions')
#        ax1.set_xlabel('Test number')
#        ax1.set_ylabel('Housing price')
        
    mean_cve = np.mean(cross_val_error) # average cross validation error over the 5 folds
    print('The cross validation error for model with ' + str(num_neuron) + ' hidden neurons: ' + str(mean_cve))
    return mean_cve, test_err

def main():
    neuronSS = [20, 40, 60, 80, 100]
    cross_val_errors = []
    test_err = []
    trainX, trainY, testX, testY = getData()
    lowest_error = sys.maxsize # Max value of integers, to ensure first model is always stored as lowest
    
    for num_neurons in neuronSS:
        cross_val_error, model_err = runModel(num_neurons, trainX, trainY, testX, testY)
        cross_val_errors.append(cross_val_error)
        if(cross_val_error <= lowest_error): 
            print('Diff to new lowest error: ' + str(cross_val_error - lowest_error))
            test_err = model_err # store test error for best model for plotting
            lowest_error = cross_val_error
            print('NEW LOWEST CVE: ' + str(lowest_error) + 'for model with ' + str(num_neurons) + ' neurons.')
            print('RMSE: ' + str(math.sqrt(lowest_error)))
    # plot test errors for best model
    plt.figure(3)
    plt.plot(range(epochs), test_err)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Test Errors for best model')
    plt.savefig('./figuresB3/PartB_3_TestErrBest.png')
    plt.show()
    print('----- FINAL TEST ERROR: ' + str(test_err[epochs-1]) + ' -----')
    print('----- FINAL RMSE: ' + str(math.sqrt(test_err[epochs-1])) + ' -----')
    
    fig2 = plt.figure(4)
    ax1 = fig2.add_subplot(211)
    ax1.set_title('Mean cross validation error')
    ax1.scatter([1,2,3,4,5], cross_val_errors)
    ax1.xaxis.set_ticks([1,2,3,4,5])
    ax1.xaxis.set_ticklabels(neuronSS)
    ax1.set_xlabel('Number of neurons in hidden ReLU layer')
    ax1.set_ylabel('Cross validation error')
    fig2.savefig('./figuresB3/Part2_3_CVEs.png')

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- Total execution time: %s seconds (%s minutes)---" % ((time.time() - start_time), (time.time() - start_time)/60))
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 12:07:49 2018

@author: Anton
"""

#
# Project 1, Part B, question 1
#

import tensorflow as tf
import numpy as np
import pylab as plt


NUM_FEATURES = 8

learning_rate = 10**-7
epochs = 500
batch_size = 32
num_neuron = 30

seed = 10
np.random.seed(seed)
beta = 0

def getData():
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
    trainX = trainX[:1000]
    trainY = trainY[:1000]
    
    #TODO: Random subsampling instead of same data?
    # Divide data into validation and testing data
    n = trainX.shape[0]
    print('Length of trainX before dividing the data: ' + str(n))
    validationdata = n // (1.0/0.7) # divide data into 70% validation (training) data and 30% testing data
    validationdata = int(validationdata)
    testX = trainX[validationdata:]
    testY = trainY[validationdata:]
    trainX = trainX[:validationdata]
    trainY = trainY[:validationdata]
    print('Data points in testX: ' + str(testX.shape[0]))
    print('Data points in trainX: ' + str(trainX.shape[0]))
    return trainX, trainY, testX, testY

def runModel():
    trainX, trainY, testX, testY = getData()
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
        n = trainX.shape[0]
        idx = np.arange(n)
        
        for i in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]
            
            for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
                train_op.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})
            
            err = error.eval(feed_dict={x: trainX, y_: trainY})
            train_err.append(err)
    
            if i % 100 == 0:
                print('iter %d: test error %g'%(i, train_err[i]))
                
        pred = sess.run(y, feed_dict={x: testX[:50]}) # final test predictions
        
    # plot learning curves
    plt.figure(1)
    plt.plot(range(epochs), train_err)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Train Error')
    plt.show()
    
    # plot predicted values and targets
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(111)
    ax1.set_title('Targets and preditions')
    ax1.scatter(range(50), pred, color='blue', marker='.', label='Targets')
    targets = np.asarray(testY[:50])
    ax1.scatter(range(50), targets, color='red', marker='x', label='Predictions')
    ax1.set_xlabel('Test number')
    ax1.set_ylabel('Housing price')
    return train_err, pred

def main():
    train_err, pred = runModel()

if __name__ == '__main__':
    main()
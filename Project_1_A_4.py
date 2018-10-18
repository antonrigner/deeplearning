# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 16:39:55 2018

@author: Anton
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 21:57:14 2018

@author: Anton
"""

#
# Project 1, Part A, Question 4
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt
import timeit
import os

if not os.path.isdir('figuresA4'):
    print('Creating the figures folder')
    os.makedirs('figuresA4')

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

NUM_FEATURES = 36 # nbr of inputs per sample (4*3*3)
NUM_CLASSES = 6 # output classes

learning_rate = 0.01 # alpha
#beta = 10**-6 # weight decay parameter
epochs = 1000
seed = 10
np.random.seed(seed)


# Read data
def getData():
    # Read training data
    train_input = np.loadtxt('sat_train.txt',delimiter=' ') # read data to rows of values (float)
    trainX, train_Y = train_input[:,0:36], train_input[:,-1].astype(int) # first 36 columns are inputs, last column is class
    trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0)) # rescale data with range and min-value
    train_Y[train_Y == 7] = 6 # change classes where class == 7 to class 6
    trainY = np.zeros((train_Y.shape[0], NUM_CLASSES)) # empty zero matrix with domensions 4435 x 6
    trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 # train_Y-1 to reduce classes to 0-5 (0-indexing), one hot matrix
    
    # Read testing data
    test_input = np.loadtxt('sat_test.txt', delimiter=' ')
    testX, test_Y = test_input[:,0:36], test_input[:,-1].astype(int)
    testX = scale(testX, np.min(testX, axis=0), np.max(testX, axis=0))
    test_Y[test_Y == 7] = 6
    testY = np.zeros((test_Y.shape[0], NUM_CLASSES))
    testY[np.arange(test_Y.shape[0]), test_Y-1] = 1
    
#     experiment with small datasets
#    trainX = trainX[:1000]
#    trainY = trainY[:1000]
#    testX = testX[:500]
#    testY = testY[:500]
    
    return trainX, trainY, testX, testY

def runModel(hidden_neurons, batch_size, beta):
    trainX, trainY, testX, testY = getData()
    # Create the model
    x = tf.placeholder(tf.float32, [None, NUM_FEATURES]) # placeholder for the input vector
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES]) # placeholder for the output vector
    # model parameters
    # initialize softmax_weights to truncated normal distribution, softmax_biases to 0s	
    hidden_weights = tf.Variable(tf.truncated_normal([NUM_FEATURES, hidden_neurons],
                                                     stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='softmax_weights')
    hidden_biases  = tf.Variable(tf.zeros([hidden_neurons]), name='hidden_biases')
    
    softmax_weights = tf.Variable(tf.truncated_normal([hidden_neurons, NUM_CLASSES],
                                                     stddev=1.0/math.sqrt(float(hidden_neurons))), name='softmax_weights')
    softmax_biases  = tf.Variable(tf.zeros([NUM_CLASSES]), name='softmax_biases')
    
    z = tf.matmul(x, hidden_weights) + hidden_biases
    h = tf.nn.sigmoid(z)
    logits  = tf.matmul(h, softmax_weights) + softmax_biases # h*softmax_weights + softmax_biases (logits = unscaled log-probabilities)    
    # Original loss function
    cross_entropy = (tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)) # logits input to softmax
    # L2 norms regularization for both the hidden and softmax layer
    regularizers = tf.nn.l2_loss(softmax_weights) + tf.nn.l2_loss(hidden_weights)
    # Regularized loss
    loss = tf.reduce_mean(cross_entropy + beta * regularizers)

    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate) # optimizer object, optimizes all weights and biases
    train_op = optimizer.minimize(loss)
    
    correct_prediction = tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(y_, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    error = tf.reduce_sum(tf.cast(tf.not_equal(tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(y_, 1)), tf.float32))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_err = []
        train_acc = []
        test_acc = []
        epoch_trainingTime = []
        
        n = trainX.shape[0]
        idx = np.arange(n)
        
        for i in range(epochs):
            np.random.shuffle(idx) # shuffle indeces
            trainX, trainY = trainX[idx], trainY[idx] # shuffle data with new indeces
            
            beginTimer = timeit.default_timer()

            for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
                train_op.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})
            
            stopTimer = timeit.default_timer()
            diffTime = stopTimer-beginTimer
            epoch_trainingTime.append(diffTime)
            
            train_err.append(error.eval(feed_dict={x: trainX, y_: trainY})) # save training error for every epoch
            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY})) # sasve accurracy for every epoch   
            train_acc.append(accuracy.eval(feed_dict={x: trainX, y_: trainY}))        

            if i % 100 == 0:
                print('iter %d: accuracy %g'%(i, train_acc[i]))
                
        
        returnData = np.zeros((3,epochs))
        returnData[0, :] = train_err
        returnData[1, :] = test_acc
        returnData[2, :] = epoch_trainingTime # training time per 1 epoch
    return returnData

    
def main():
    avgtime = []
    train_err = []
    test_acc = []
    final_test_acc = []
    decay_SS = [0, 10**-3, 10**-6, 10**-9, 10**-12] # decay param search space
    
    
    fig1 = plt.figure(2, figsize=(10,5))
    ax1 = fig1.add_subplot(111)
    ax1.set_title('Training Errors for different decay paramaters')# + str(batchsize))
    ax1.set_xlabel(str(epochs) + ' iterations/epochs')
    ax1.set_ylabel('Train error')

    # run the 5 models, store the validation data (training error, testing accurracy and avg time for training)
    for decayParam in decay_SS:
        modeldata = runModel(10, 32, decayParam)
        train_err = modeldata[0]
        test_acc = modeldata[1]
        avgtime.append(np.mean(modeldata[2])) # average out how much time to train ONE epoch
        
#        print('Final training error: ' + str(train_err[epochs-1]))
        ax1.plot(range(epochs), train_err)
        final_test_acc.append(test_acc[epochs-1])
       
    fig1.legend(['0', '10^-3', '10^-6', '10^-9', '10^-12'])
    fig1.savefig('./figuresA4/PartA_4_TrainError.png')
    fig1.show()
    
    fig = plt.figure(1)
    ax1 = fig.add_subplot(211)
    ax1.set_title('Final test accurracies for different values of the decay parameter')
    ax1.scatter([1,2,3,4,5], final_test_acc)
    ax1.xaxis.set_ticks([1,2,3,4,5])
    ax1.xaxis.set_ticklabels(decay_SS)
    ax1.set_xlabel('Decay parameter')
    ax1.set_ylabel('Final test accurracy')
    fig.savefig('./figuresA4/PartA_4_Final_TestAcc.png')

if __name__ == '__main__':
    main()
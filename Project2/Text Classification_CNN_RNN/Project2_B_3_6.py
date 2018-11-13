# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:12:21 2018

@author: Anton
"""

import pylab as plt
import numpy as np
import pandas
import tensorflow as tf
import csv
import time
import os

if not os.path.isdir('figuresB36'):
    print('Creating the figures folder')
    os.makedirs('figuresB36')

MAX_DOCUMENT_LENGTH = 100 # Maximum length of words / characters for inputs
MAX_LABEL = 15 # 15 Wikipedia categories in the dataset
HIDDEN_SIZE = 20

epochs = 5
lr = 0.01
batch_size = 128

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def char_rnn_model(x, keep_prob, model):
    
    if model == 'rnn':
        cell_fn = tf.nn.rnn_cell.BasicRNNCell
    elif model == 'gru':
        cell_fn = tf.nn.rnn_cell.GRUCell
    elif model == 'lstm':
        cell_fn = tf.nn.rnn_cell.LSTMCell
        
# input layer, different classes of char-s for a given input
    input_layer = tf.reshape(
            tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, 256]) # one hot layer, with 1:s at indices defined by x, depth 256 (nbr of chars)
    input_chars = tf.unstack(input_layer, axis=1) # Char sequence
    
    cell = cell_fn(HIDDEN_SIZE)
    print(cell)
    print(input_chars)
    _, state = tf.nn.dynamic_rnn(cell, input_chars, dtype=tf.float32)
    state_drop = tf.nn.dropout(state, keep_prob)
    
    logits = tf.layers.dense(state_drop, MAX_LABEL, activation=None)
    return _, logits

def read_data_chars():
    x_train, y_train, x_test, y_test = [], [], [], []
    with open('train_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_train.append(row[1])
            y_train.append(int(row[0]))

    with open('test_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_test.append(row[1])
            y_test.append(int(row[0]))

#    print('Raw')
#    print('x_train: ', x_train[10:15])
#    print('y_train: ', y_train[10:15])       
#    print('Pandas')     
    x_train = pandas.Series(x_train)
    y_train = pandas.Series(y_train)
    x_test = pandas.Series(x_test)
    y_test = pandas.Series(y_test)
#    print('x_train: ', x_train[:5])
#    print('y_train: ', y_train[:5])
#    print('Char processor')
    char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
    x_train = np.array(list(char_processor.fit_transform(x_train)))
    x_test = np.array(list(char_processor.transform(x_test)))
    y_train = y_train.values
    y_test = y_test.values
            
    x_train, y_train, x_test, y_test = x_train[:250], y_train[:250], x_test[:250], y_test[:250]
#    print('x_train: ', x_train[:5])
#    print('y_train: ', y_train[:5])
    return x_train, y_train, x_test, y_test

  
def runModel(keep_prob, model):  
    startTime = time.time()
    tf.reset_default_graph() 
    x_train, y_train, x_test, y_test = read_data_chars()

#    print(x_train.shape)
#    print(y_train.shape)
#    print(len(x_train))
#    print(len(x_test))
    
    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)
    
    inputs, logits = char_rnn_model(x, keep_prob, model)
    
    # Class predictions and accuracy
    prediction = tf.nn.softmax(logits)
    correct_prediction = tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(tf.one_hot(y_,MAX_LABEL), 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    # Optimizer
    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
    train_op = tf.train.AdamOptimizer(lr).minimize(entropy)
      
    N = len(x_train)
    idx = np.arange(N)
      
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
  
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        train_cost = [] 
        test_acc = []
  
  
        for e in range(epochs):
            np.random.shuffle(idx)
            x_train, y_train = x_train[idx], y_train[idx]
            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):  
                train_op.run(feed_dict={x: x_train, y_: y_train})

            loss_ = entropy.eval(feed_dict={x: x_train, y_: y_train})
            train_cost.append(loss_)
            test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test})) # save accurracy for every epoch   


            if e%1 == 0:
                print('iter: %d, entropy: %g'%(e, train_cost[e]))
  
    ax1.plot(range(epochs), train_cost)
    ax2.plot(range(epochs), test_acc)

    fig1.savefig('./figuresB36/PartB_36_TrainError' + str(keep_prob)+'.png')
    fig2.savefig('./figuresB36/PartB_36_TestAcc' + str(keep_prob)+'.png')
    fig1.legend(['No dropout', 'Dropout with keep prob ' + str(keep_prob)])
    fig2.legend(['No dropout', 'Dropout with keep prob ' + str(keep_prob)])
    end = time.time()
    diff = round(end - startTime, 3)
    print('Total runtime: ', diff, 'seconds.')
    return train_cost, test_acc

def main():
    plt.figure()
    print('Running GRU model WITHOUT dropout')
    train_cost, test_acc = runModel(1, 'gru')
    plt.plot(range(epochs), train_cost, label='gru')
    plt.plot(range(epochs), test_acc, label='gru')

#    print('Running GRU model WITH dropout')
#    train_cost, test_acc = runModel(0.5, 'gru')
#    plt.plot(range(epochs), train_cost, label='gru')
#    plt.plot(range(epochs), test_acc, label='gru')

    print('Running LSTM model WITHOUT dropout')
    train_cost, test_acc = runModel(1, 'lstm')
    plt.plot(range(epochs), train_cost, label='LSTM')
    plt.plot(range(epochs), test_acc, label='LSTM')

#    print('Running LSTM model WITH dropout')
#    train_cost, test_acc = runModel(0.5, 'lstm')
#    plt.plot(range(epochs), train_cost, label='gru')
#    plt.plot(range(epochs), test_acc, label='gru')

    print('Running RNN model WITHOUT dropout')
    train_cost, test_acc = runModel(1, 'rnn')
    plt.plot(range(epochs), train_cost, label='RNN')
    plt.plot(range(epochs), test_acc, label='RNN')

#    print('Running RNN model WITH dropout')
#    train_cost, test_acc = runModel(0.5, 'RNN')   
#    plt.plot(range(epochs), train_cost, label='gru')
#    plt.plot(range(epochs), test_acc, label='gru')

    plt.xlabel('epochs')
    plt.ylabel('mean square error')
    plt.legend()
    
    plt.savefig('./figures/9.1b_1.png')
    
    
    plt.show()     
if __name__ == '__main__':
    main()

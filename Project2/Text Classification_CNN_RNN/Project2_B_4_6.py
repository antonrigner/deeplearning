# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:15:08 2018

@author: Anton
"""
import pylab as plt
import numpy as np
import pandas
import tensorflow as tf
import csv
import time
import os

if not os.path.isdir('figuresB4'):
    print('Creating the figures folder')
    os.makedirs('figuresB4')

#TODO: Epochs, batch size, data size, figures, illustrate prediction

EMBEDDING_SIZE = 20
MAX_DOCUMENT_LENGTH = 100 # Maximum length of words / characters for inputs
MAX_LABEL = 15 # 15 Wikipedia categories in the dataset
HIDDEN_SIZE = 20

epochs = 25
lr = 0.005
batch_size = 128

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 1000
tf.set_random_seed(seed)

def word_rnn_model(x, keep_prob, model):
    print(model)
    if model == 'rnn' or model == '2rnn':
        print('creating cell_fn = basicRNN')
        cell_fn = tf.nn.rnn_cell.BasicRNNCell
        print('basic rnn cell created')
    elif model == 'gru':
        cell_fn = tf.nn.rnn_cell.GRUCell
    else:
        cell_fn = tf.nn.rnn_cell.LSTMCell

    if model == '2rnn':
        print('creating multiple cells')
        cell1 = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE)
        cell2 = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE)
        cells = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
    else:
        cells = cell_fn(HIDDEN_SIZE)
    print(cells)
    
    word_vectors = tf.contrib.layers.embed_sequence(
      x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

#    word_sequence = tf.unstack(word_vectors, axis=1)
    
    outputs, state = tf.nn.dynamic_rnn(cells, word_vectors, dtype=tf.float32)
    if model == 'lstm' or model == '2rnn':
        state = state[1]
        print('Using state[1], hidden state')
    state_drop = tf.nn.dropout(state, keep_prob)
    logits = tf.layers.dense(state_drop, MAX_LABEL, activation=None)

    return outputs, logits, state

def read_data_words():
  
    x_train, y_train, x_test, y_test = [], [], [], []
    with open('train_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_train.append(row[2])
            y_train.append(int(row[0]))
    
    with open("test_medium.csv", encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_test.append(row[2])
            y_test.append(int(row[0]))
  
    x_train = pandas.Series(x_train)
    y_train = pandas.Series(y_train)
    x_test = pandas.Series(x_test)
    y_test = pandas.Series(y_test)
    y_train = y_train.values
    y_test = y_test.values
  
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
            MAX_DOCUMENT_LENGTH)

    x_transform_train = vocab_processor.fit_transform(x_train)
    x_transform_test = vocab_processor.transform(x_test)

    x_train = np.array(list(x_transform_train))
    x_test = np.array(list(x_transform_test))

    x_train, y_train, x_test, y_test = x_train[:1500], y_train[:1500], x_test[:500], y_test[:500]

    no_words = len(vocab_processor.vocabulary_)
    print('Total words: %d' % no_words)
    
    return x_train, y_train, x_test, y_test, no_words
  
def runModel(keep_prob, model):  
    startTime = time.time()
    global n_words
    tf.reset_default_graph() 
    x_train, y_train, x_test, y_test, n_words= read_data_words()

#    print(x_train.shape)
#    print(y_train.shape)
#    print(len(x_train))
#    print(len(x_test))
    
    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)
    
    inputs, logits, _ = word_rnn_model(x, keep_prob, model)
    
    # Class predictions and accuracy
    prediction = tf.nn.softmax(logits)
    correct_prediction = tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(tf.one_hot(y_,MAX_LABEL), 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    # Optimizer
    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
#    train_op = tf.train.AdamOptimizer(lr).minimize(entropy)
    
    minimizer = tf.train.AdamOptimizer(lr)#minimize(entropy)
    
    grads_and_vars = minimizer.compute_gradients(entropy)


    # Gradient clipping
    grad_clipping = tf.constant(2.0, name="grad_clipping")
    clipped_grads_and_vars = []
#    print(grads_and_vars)
    for grad, var in grads_and_vars:
        clipped_grad = tf.clip_by_value(grad, -grad_clipping, grad_clipping)
        clipped_grads_and_vars.append((clipped_grad, var))
     
    train_op = minimizer.apply_gradients(clipped_grads_and_vars)
      
    N = len(x_train)
    idx = np.arange(N)
#      
#    fig1 = plt.figure(2, figsize=(10,5))
#    ax1 = fig1.add_subplot(111)
#    ax1.set_title('Training Cost (Cross entropy)')
#    ax1.set_xlabel(str(epochs) + ' iterations/epochs')
#    ax1.set_ylabel('Cross entropy')  
#    fig2 = plt.figure(3, figsize=(10,5))
#    ax2 = fig2.add_subplot(111)
#    ax2.set_title('Top 1 Test Accurracy')
#    ax2.set_xlabel(str(epochs) + ' iterations/epochs')
#    ax2.set_ylabel('Test accuracy')
  
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        train_cost = [] 
        test_acc = []
  
  
        for e in range(epochs):
            np.random.shuffle(idx)
            x_train, y_train = x_train[idx], y_train[idx]
            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):  
                train_op.run(feed_dict={x: x_train[start:end], y_: y_train[start:end]})

            loss_ = entropy.eval(feed_dict={x: x_train, y_: y_train})
            train_cost.append(loss_)
            test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test})) # save accurracy for every epoch   


            if e%1 == 0:
                print('iter: %d, entropy: %g'%(e, train_cost[e]))
  
#    ax1.plot(range(epochs), train_cost)
#    ax2.plot(range(epochs), test_acc)
#    
#    if keep_prob != 1: # Define legend once
#        fig1.legend(['No dropout', 'Dropout with keep prob ' + str(keep_prob)])
#        fig2.legend(['No dropout', 'Dropout with keep prob ' + str(keep_prob)])
#
#
#    fig1.savefig('./figuresB4/PartB_4_TrainError' + str(keep_prob)+'.png')
#    fig2.savefig('./figuresB4/PartB_4_TestAcc' + str(keep_prob)+'.png')
    end = time.time()
    diff = round(end - startTime, 3)
    print('Total runtime: ', diff, 'seconds')
    return train_cost, test_acc

def main():
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.set_random_seed(1000)
    
    plt.figure(1)
    print('Running GRU model WITHOUT dropout')
    train_cost, test_acc = runModel(1, 'gru')
    plt.plot(range(epochs), train_cost, label='GRU')
    plt.figure(2)
    plt.plot(range(epochs), test_acc, label='GRU')

#    print('Running GRU model WITH dropout')
#    train_cost, test_acc = runModel(0.5, 'gru')
#    plt.plot(range(epochs), train_cost, label='gru')
#    plt.plot(range(epochs), test_acc, label='gru')
    
    print('Running RNN model WITHOUT dropout')
    train_cost, test_acc = runModel(1, 'rnn')
    plt.figure(1)
    plt.plot(range(epochs), train_cost, label='RNN')
    plt.figure(2)
    plt.plot(range(epochs), test_acc, label='RNN')
#
#    print('Running RNN model WITH dropout')
#    train_cost, test_acc = runModel(0.5, 'rnn')   
#    plt.figure(1)
#    plt.plot(range(epochs), train_cost, label='rnn_drop')
#    plt.figure(2)
#    plt.plot(range(epochs), test_acc, label='rnn_drop')

    print('Running LSTM model WITHOUT dropout')
    train_cost, test_acc = runModel(1, 'lstm')
    plt.figure(1)
    plt.plot(range(epochs), train_cost, label='LSTM')
    plt.figure(2)
    plt.plot(range(epochs), test_acc, label='LSTM')
    
    print('Running 2RNN model WITHOUT dropout')
    train_cost, test_acc = runModel(1, '2rnn')
    plt.figure(1)
    plt.plot(range(epochs), train_cost, label='2RNN')
    plt.legend()
    plt.figure(2)
    plt.plot(range(epochs), test_acc, label='2RNN')
    plt.legend()
    
if __name__ == '__main__':
    main()

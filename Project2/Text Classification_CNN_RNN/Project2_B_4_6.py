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
N_FILTERS = 10
FILTER_SHAPE1 = [20, EMBEDDING_SIZE] # Kernel size for CNN 1
FILTER_SHAPE2 = [20, 1] # CNN 2
POOLING_WINDOW = 4 # 4x4
POOLING_STRIDE = 2 # 2x2
MAX_LABEL = 15 # 15 Wikipedia categories in the dataset

epochs = 10
lr = 0.01
batch_size = 250
keep_prob = 1.0

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def word_cnn_model(x, keep_prob):
    word_vectors = tf.contrib.layers.embed_sequence(
      x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

    word_sequence = tf.unstack(word_vectors, axis=1)
    
    cell = tf.nn.rnn_cell.GRUCell(EMBEDDING_SIZE)
    _, state = tf.nn.static_rnn(cell, word_sequence, dtype=tf.float32)
    state_drop = tf.nn.dropout(state, keep_prob)

    logits = tf.layers.dense(state_drop, MAX_LABEL, activation=None)
    return word_sequence, logits

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

    x_train, y_train, x_test, y_test = x_train[:500], y_train[:500], x_test[:250], y_test[:250]

    no_words = len(vocab_processor.vocabulary_)
    print('Total words: %d' % no_words)
    
    return x_train, y_train, x_test, y_test, no_words
  
def runModel(keep_prob):  
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
    
    inputs, logits = word_cnn_model(x, keep_prob)
    
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
    
    fig1.savefig('./figuresB4/PartB_4_TrainError' + str(keep_prob)+'.png')
    fig2.savefig('./figuresB4/PartB_4_TestAcc' + str(keep_prob)+'.png')
    if keep_prob != 1: # Define legend once
        fig1.legend(['No dropout', 'Dropout with keep prob ' + str(keep_prob)])
        fig2.legend(['No dropout', 'Dropout with keep prob ' + str(keep_prob)])

    end = time.time()
    diff = round(end - startTime, 3)
    print('Total runtime: ', diff, 'seconds')
    
def main():
    print('Running model WITHOUUT dropout')
    runModel(1)
    print('Running model WITH dropout')
    runModel(0.5)
    
if __name__ == '__main__':
    main()

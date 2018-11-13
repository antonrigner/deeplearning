#    # -*- coding: utf-8 -*-
#    """
#    Created on Mon Nov 12 10:12:09 2018
#    
#    @author: Anton
#    """
#    
import pylab as plt
import numpy as np
import pandas
import tensorflow as tf
import csv
import time
import os

if not os.path.isdir('figuresB1'):
    print('Creating the figures folder')
    os.makedirs('figuresB1')

MAX_DOCUMENT_LENGTH = 100 # Maximum length of words / characters for inputs
N_FILTERS = 10
FILTER_SHAPE1 = [20, 256] # Kernel size for CNN 1
FILTER_SHAPE2 = [20, 1] # CNN 2
POOLING_WINDOW = 4 # 4x4
POOLING_STRIDE = 2 # 2x2
MAX_LABEL = 15 # 15 Wikipedia categories in the dataset

epochs = 5
lr = 0.01
batch_size = 128
#keep_prob = 0.5

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def char_cnn_model(x, keep_prob):
# input layer, different classes of chars for a given input
    input_layer = tf.reshape(
            tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, 256, 1]) # one hot layer, with 1:s at indices defined by x, depth 256 (nbr of chars)
    
    with tf.variable_scope('CNN_Layer1'):
      conv1 = tf.layers.conv2d(
              input_layer,
              filters=N_FILTERS,
              kernel_size=FILTER_SHAPE1,
              padding='VALID',
              activation=tf.nn.relu)
      conv1_drop = tf.nn.dropout(conv1, keep_prob)

      pool1 = tf.layers.max_pooling2d(
              conv1_drop,
              pool_size=POOLING_WINDOW,
              strides=POOLING_STRIDE,
              padding='SAME')

    with tf.variable_scope('CNN_Layer2'):
      conv2 = tf.layers.conv2d(
              pool1,
              filters=N_FILTERS,
              kernel_size=FILTER_SHAPE2,
              padding='VALID',
              activation=tf.nn.relu)
      conv2_drop = tf.nn.dropout(conv2, keep_prob)
      
      pool2 = tf.layers.max_pooling2d(
              conv2_drop,
              pool_size=POOLING_WINDOW,
              strides=POOLING_STRIDE,
              padding='SAME')
      
      pool2 = tf.squeeze(tf.reduce_max(pool1, 1), squeeze_dims=[1]) # remove dimensions of size 1 from the shape of the tensor

    logits = tf.layers.dense(pool2, MAX_LABEL, activation=None)
    return input_layer, logits

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
            
#    x_train, y_train, x_test, y_test = x_train[:250], y_train[:250], x_test[:250], y_test[:250]
#    print('x_train: ', x_train[:5])
#    print('y_train: ', y_train[:5])
    return x_train, y_train, x_test, y_test

  
def runModel(keep_prob):  
    startTime = time.time()
    tf.reset_default_graph() 
    x_train, y_train, x_test, y_test = read_data_chars()

#    print(x_train.shape)
#    print(y_train.shape)
    print(len(x_train))
    print(len(x_test))
    
    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)
    
    inputs, logits = char_cnn_model(x, keep_prob)
    
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


            if e%10 == 0:
                print('iter: %d, entropy: %g'%(e, train_cost[e]))
  
    ax1.plot(range(epochs), train_cost)
    ax2.plot(range(epochs), test_acc)

    if keep_prob != 1: # Define legend once
        fig1.legend(['No dropout', 'Dropout with keep prob ' + str(keep_prob)])
        fig2.legend(['No dropout', 'Dropout with keep prob ' + str(keep_prob)])

    fig1.savefig('./figuresB1/PartB_1_TrainError' + str(keep_prob)+'.png')
    fig2.savefig('./figuresB1/PartB_1_TestAcc' + str(keep_prob)+'.png')
    end = time.time()
    diff = round(end - startTime, 3)
    print('Total runtime: ', diff, 'seconds.')

def main():
    print('Running model WITHOUUT dropout')
    runModel(1)
    print('Running model WITH dropout')
    runModel(0.5)
        
if __name__ == '__main__':
    main()

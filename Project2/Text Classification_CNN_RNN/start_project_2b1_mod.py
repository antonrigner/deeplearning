
import numpy as np
import pandas
import tensorflow as tf
import csv

#TODO: Assume 256 chars

MAX_DOCUMENT_LENGTH = 100 # Maximum length of words / characters for inputs
N_FILTERS = 10
FILTER_SHAPE1 = [20, 256] # Kernel size for CNN
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15

no_epochs = 10
lr = 0.01
batch_size = 128

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def char_cnn_model(x):
    
    # input layer, different classes of chars
    input_layer = tf.reshape(
      tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, 256, 1]) # one hot layer, with 1:s at indices defined by x, depth 256 (nbr of chars)
  
    with tf.variable_scope('CNN_Layer'):
        conv1 = tf.layers.conv2d(
                input_layer,
                filters=N_FILTERS,
                kernel_size=FILTER_SHAPE1,
                padding='VALID',
                activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(
                conv1,
                pool_size=POOLING_WINDOW,
                strides=POOLING_STRIDE,
                padding='SAME')

        print(pool1)
        print(tf.reduce_max(pool1, 1))
        pool1 = tf.squeeze(tf.reduce_max(pool1, 1), squeeze_dims=[1])
        print(pool1)
    logits = tf.layers.dense(pool1, MAX_LABEL, activation=None)
#    print(logits)

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
  
  x_train = pandas.Series(x_train)
  y_train = pandas.Series(y_train)
  x_test = pandas.Series(x_test)
  y_test = pandas.Series(y_test)
  
  
  char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
  x_train = np.array(list(char_processor.fit_transform(x_train)))
  x_test = np.array(list(char_processor.transform(x_test)))
  y_train = y_train.values
  y_test = y_test.values
  
  # Experiment with smaller data set
  x_train, y_train, x_test, y_test = x_train[:500], y_train[:500], x_test[:250], y_test[:250]
  return x_train, y_train, x_test, y_test

  
def main():
  tf.reset_default_graph()
  x_train, y_train, x_test, y_test = read_data_chars()

  print(len(x_train))
  print(len(x_test))

  # Create the model
  x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
  y_ = tf.placeholder(tf.int64)

  inputs, logits = char_cnn_model(x)
  print('inputs: ', inputs)
  print('logits: ', logits)
  print('y_: ', y_)
  # Optimizer
  entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
  train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # training
  loss = []
  for e in range(no_epochs):
    _, loss_  = sess.run([train_op, entropy], {x: x_train, y_: y_train})
    loss.append(loss_)


    if e%1 == 0:
      print('iter: %d, entropy: %g'%(e, loss[e]))
  
  sess.close()

if __name__ == '__main__':
  main()

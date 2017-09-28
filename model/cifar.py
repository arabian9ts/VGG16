"""
mnist tester (train and test accuracy)

date: 9/24
author: arabian9ts
"""

# escape matplotlib error
import matplotlib
matplotlib.use('Agg')

# escape tensorflow warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import _pickle as pickle
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt

from vgg16 import *

# global variables
DATASET_NUM = 10000
BATCH = 100
EPOCH = 50

images = []
labels = []

def gen_onehot_list(label=0):
    """
    generate one-hot label-list based on ans-index
    e.g. if ans is 3, return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    Args: answer-index
    Returns: one-hot list
    """
    return [1 if l==label else 0 for l in range(0, 10)]


def load_data():
    """
    open cifar-dataset
    segregate images-data and answers-label to images and labels
    """
    with open('dataset/data_batch_1', 'rb') as f:
        train_data = pickle.load(f, encoding='latin-1')
        images = np.array(train_data['data'])
        labels = np.array(train_data['labels'])
        reshaped_images = np.array([x.reshape([32, 32, 3]) for x in images])
        reshaped_labels = np.array([gen_onehot_list(i) for i in labels])

    return reshaped_images, reshaped_labels
        

def get_next_batch(length=BATCH):
    """
    extract next batch-images

    Returns: batch sized BATCH
    """
    indicies = np.random.choice(DATASET_NUM, length)
    next_batch = images[indicies]
    next_labels = labels[indicies]

    return np.array(next_batch), np.array(next_labels)

def test():
    # Test
    images, labels = get_next_batch(length=100)
    result = sess.run(predict, feed_dict={input: images})

    correct = 0
    total = 100

    for i in range(len(labels)):
        pred_max = result[i].argmax()
        ans = labels[i].argmax()

        if ans == pred_max:
            correct += 1

    print('Accuracy: '+str(correct)+' / '+str(total)+' = '+str(correct/total))


with tf.Session() as sess:
    """
    TensorFlow session
    """
    # use VGG16 network
    vgg = VGG16()
    # params for converting to answer-label-size
    w = tf.Variable(tf.truncated_normal([512, 10], 0.0, 1.0) * 0.01)
    b = tf.Variable(tf.truncated_normal([10], 0.0, 1.0) * 0.01)

    # input image's placeholder and output of VGG16
    input = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
    fmap = vgg.build(input, is_training=True)
    predict = tf.nn.softmax(tf.add(tf.matmul(fmap, w), b))

    # params for defining Loss-func and Training-step
    ans = tf.placeholder(shape=None, dtype=tf.float32)
    ans = tf.squeeze(tf.cast(ans, tf.float32))

    # cross-entropy
    loss = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(predict), reduction_indices=[1]))
    optimizer = tf.train.GradientDescentOptimizer(0.05)
    train_step = optimizer.minimize(loss)

    sess.run(tf.global_variables_initializer())

    # load image data
    images, labels = load_data()

    print('\nSTART LEARNING')
    print('==================== '+str(datetime.datetime.now())+' ====================')

    # Training-loop
    lossbox = []
    for e in range(EPOCH):
        for b in range(int(DATASET_NUM/BATCH)):
            batch, actuals = get_next_batch()
            sess.run(train_step, feed_dict={input: batch, ans: actuals})

            print('Batch: %3d' % int(b+1)+', \tLoss: '+str(sess.run(loss, feed_dict={input: batch, ans: actuals})))

            if (b+1) % 100 == 0:
                print('============================================')
                print('START TEST')
                test()
                print('END TEST')
                print('============================================')
            time.sleep(0)

        lossbox.append(sess.run(loss, feed_dict={input: batch, ans: actuals}))
        print('========== Epoch: '+str(e+1)+' END ==========')

    print('==================== '+str(datetime.datetime.now())+' ====================')
    print('\nEND LEARNING')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(np.array(range(EPOCH)), lossbox)
    plt.show()
    

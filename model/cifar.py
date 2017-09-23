"""
mnist tester (train and test accuracy)

date: 9/24
author: arabian9ts
"""

import tensorflow as tf
import _pickle as pickle
import numpy as np

from vgg16 import *

# global variables
DATASET_NUM = 10000
BATCH = 20
EPOCH = 1
BATCH_CNT = 0


def gen_onehot_list(label=0):
    """
    generate one-hot label-list based on ans-index
    e.g. if ans is 3, return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    Args: answer-index
    Returns: one-hot list
    """
    return [1 if l==label else 0 for l in range(0, 10)]


with open('dataset/data_batch_1', 'rb') as f:
    """
    open cifar-dataset
    segregate images-data and answers-label to images and labels
    """
    train_data = pickle.load(f, encoding='latin-1')
    images = train_data['data']
    labels = train_data['labels']
        

def get_next_batch():
    """
    extract next batch-images

    Returns: batch sized BATCH
    """
    global BATCH_CNT
    if BATCH * (BATCH_CNT+1) >= DATASET_NUM:
        BATCH_CNT = 0
    next_batch = images[BATCH*BATCH_CNT:BATCH*(BATCH_CNT+1)]
    next_labels = labels[BATCH*BATCH_CNT:BATCH*(BATCH_CNT+1)]
    reshaped_batch = [x.reshape([32, 32, 3]) for x in next_batch]
    reshaped_labels = [gen_onehot_list(i) for i in next_labels]

    return np.array(reshaped_batch), np.array(reshaped_labels)


with tf.Session() as sess:
    """
    TensorFlow session
    """
    # use VGG16 network
    vgg = Vgg16()
    # params for converting to answer-label-size
    w = tf.Variable(tf.random_uniform([1000, 10], -1.0, 1.0))
    b = tf.Variable(tf.zeros(10))

    # input image's placeholder and output of VGG16
    input = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
    fmap = vgg.build(input)
    predict = tf.add(tf.matmul(fmap, w), b)

    # params for defining Loss-func and Training-step
    ans_labels = tf.placeholder(shape=None, dtype=tf.float32)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=ans_labels))
    optimizer = tf.train.AdamOptimizer(0.05)
    train_step = optimizer.minimize(loss)

    sess.run(tf.global_variables_initializer())

    # Training-loop
    for e in range(EPOCH):
        for b in range(int(DATASET_NUM/BATCH)):
            batch, ans = get_next_batch()
            sess.run(train_step, feed_dict={input: batch, ans_labels: ans})

        print('Loss: '+str(sess.run(loss, feed_dict={input: batch, ans_labels: ans})))
        

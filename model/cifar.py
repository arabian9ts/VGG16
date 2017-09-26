"""
mnist tester (train and test accuracy)

date: 9/24
author: arabian9ts
"""

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
EPOCH = 100
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
    images = np.array(train_data['data'])
    labels = np.array(train_data['labels'])
        

def get_next_batch():
    """
    extract next batch-images

    Returns: batch sized BATCH
    """
    global BATCH_CNT
    # if BATCH * (BATCH_CNT+1) >= DATASET_NUM:
    #     BATCH_CNT = 0
    indicies = np.random.choice(DATASET_NUM, BATCH)
    next_batch = images[indicies]
    next_labels = labels[indicies]
    reshaped_batch = [x.reshape([32, 32, 3]) for x in next_batch]
    reshaped_labels = [gen_onehot_list(i) for i in next_labels]

    return np.array(reshaped_batch), np.array(reshaped_labels)

def test():
    # Test
    indicies = np.random.randint(0, DATASET_NUM, BATCH)
    total = len(indicies)
    correct = 0
    test_images = np.array(images)[indicies]
    test_labels = np.array(labels)[indicies]
    test_images = [x.reshape([32, 32, 3]) for x in test_images]
    test_predict = sess.run(predict, feed_dict={input: test_images})

    for i in range(len(indicies)):
        pred_max = test_predict[i].argmax()
        if labels[i] == pred_max:
            correct += 1

    print('Accuracy: '+str(correct)+' / '+str(total)+' = '+str(correct/total))


with tf.Session() as sess:
    """
    TensorFlow session
    """
    # use VGG16 network
    vgg = Vgg16()
    # params for converting to answer-label-size
    w = tf.Variable(tf.random_uniform([4096, 10], 0.0, 1.0) / 100)
    b = tf.Variable(tf.random_uniform([10], 0.0, 1.0) / 100)

    # input image's placeholder and output of VGG16
    input = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
    fmap = vgg.build(input, is_training=True)
    predict = tf.nn.relu(tf.add(tf.matmul(fmap, w), b))

    # params for defining Loss-func and Training-step
    ans_labels = tf.placeholder(shape=None, dtype=tf.float32)
    ans_labels = tf.squeeze(tf.cast(ans_labels, tf.int32))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=ans_labels))
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train_step = optimizer.minimize(loss)

    sess.run(tf.global_variables_initializer())

    print('\nSTART LEARNING')
    print('==================== '+str(datetime.datetime.now())+' ====================')

    # Training-loop
    lossbox = []
    for e in range(EPOCH):
        for b in range(int(DATASET_NUM/BATCH)):
            batch, ans = get_next_batch()
            sess.run(train_step, feed_dict={input: batch, ans_labels: ans})

            print('Batch: '+str(b+1)+', Loss: '+str(sess.run(loss, feed_dict={input: batch, ans_labels: ans})))

            if (b+1) % 100 == 0:
                print('============================================')
                print('START TEST')
                test()
                print('END TEST')
                print('============================================')
            time.sleep(0)

        lossbox.append(sess.run(loss, feed_dict={input: batch, ans_labels: ans}))
        print('========== Epoch: '+str(e+1)+' END ==========')

    print('==================== '+str(datetime.datetime.now())+' ====================')
    print('\nEND LEARNING')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(np.array(range(EPOCH)), lossbox)
    plt.show()
    

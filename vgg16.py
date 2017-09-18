"""
implementation of vgg network with TensorFlow

date: 9/17
author: arabian9ts
"""

import tensorflow as tf
import numpy as np
import vgg16_structure as vgg

from functools import reduce
from activation import Activation

class Vgg16:
    def __init__(self):
        pass

    def build(self, input):
        """
        input is the placeholder of tensorflow
        build() assembles vgg16 network
        """

        self.conv1_1 = self.convolution(input, 'conv1_1')
        self.conv1_2 = self.convolution(self.conv1_1, 'conv1_2')
        self.pool1 = self.pooling(self.conv1_2, 'pool1')

        self.conv2_1 = self.convolution(self.pool1, 'conv2_1')
        self.conv2_2 = self.convolution(self.conv2_1, 'conv2_2')
        self.pool2 = self.pooling(self.conv2_2, 'pool2')

        self.conv3_1 = self.convolution(self.pool2, 'conv3_1')
        self.conv3_2 = self.convolution(self.conv3_1, 'conv3_2')
        self.conv3_3 = self.convolution(self.conv3_2, 'conv3_3')
        self.pool3 = self.pooling(self.conv3_3, 'pool3')

        self.conv4_1 = self.convolution(self.pool3, 'conv4_1')
        self.conv4_2 = self.convolution(self.conv4_1, 'conv4_2')
        self.conv4_3 = self.convolution(self.conv4_2, 'conv4_3')
        self.pool4 = self.pooling(self.conv4_3, 'pool4')

        self.conv5_1 = self.convolution(self.pool4, 'conv5_1')
        self.conv5_2 = self.convolution(self.conv5_1, 'conv5_2')
        self.conv5_3 = self.convolution(self.conv5_2, 'conv5_3')
        self.pool5 = self.pooling(self.conv5_3, 'pool5')

        self.fc6 = self.fully_connection(self.pool5, Activation.relu, 'fc6')
        self.fc7 = self.fully_connection(self.fc6, Activation.relu, 'fc7')
        self.fc8 = self.fully_connection(self.fc7, Activation.softmax, 'fc8')

        self.net = self.fc8

        return 0



    def pooling(self, input, name):
        """
        Args: output of just before layer
        Return: max_pooling layer
        """
        return tf.nn.max_pool(input, ksize=vgg.ksize, strides=vgg.pool_strides, padding='SAME', name=name)

    def convolution(self, input, name):
        """
        Args: output of just before layer
        Return: convolution layer
        """
        print('Current input size in convolution layer is: '+str(input.get_shape().as_list()))
        with tf.variable_scope(name):
            size = vgg.structure[name]
            kernel = self.getWeight(size[0])
            bias = self.getBias(size[1])
            conv = tf.nn.conv2d(input, kernel, strides=vgg.conv_strides, padding='SAME', name=name)
        return tf.nn.relu(tf.add(conv, bias))

    def fully_connection(self, input, activation, name):
        """
        Args: output of just before layer
        Return: fully_connected layer
        """
        size = vgg.structure[name]
        with tf.variable_scope(name):
            shape = input.get_shape().as_list()
            dim = reduce(lambda x, y: x * y, shape[1:])
            x = tf.reshape(input, [-1, dim])

            weights = self.getWeight([dim, size[0][0]])
            biases = self.getBias(size[1])

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            fc = activation(fc)

            print('Input shape is: '+str(shape))
            print('Total nuron count is: '+str(dim))
            
            return fc

    def getWeight(self, shape):
        """
        generate weight tensor

        Args: weight size
        Return: initialized weight tensor
        """
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def getBias(self, shape):
        """
        generate bias tensor

        Args: bias size
        Return: initialized bias tensor
        """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
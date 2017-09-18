"""
implementation of vgg network with TensorFlow

date: 9/17
author: arabian9ts
"""

import tensorflow as tf
import numpy as np
import vgg16_structure as vgg

from activation import Activation

class Vgg16:
    def __init__(self):
        pass

    def build(self, input):
        """
        input is the placeholder of tensorflow
        build() assembles vgg16 network
        """

        # layer_number
        self.lcnt = 0

        self.conv1_1 = self.convolution(input)
        self.conv1_2 = self.convolution(self.conv1_1)
        self.pool1 = self.pooling(self.conv1_2)

        self.conv2_1 = self.convolution(self.pool1)
        self.conv2_2 = self.convolution(self.conv2_1)
        self.pool2 = self.pooling(self.conv2_2)

        self.conv3_1 = self.convolution(self.pool2)
        self.conv3_2 = self.convolution(self.conv3_1)
        self.conv3_3 = self.convolution(self.conv3_2)
        self.pool3 = self.pooling(self.conv3_3)

        self.conv4_1 = self.convolution(self.pool3)
        self.conv4_2 = self.convolution(self.conv4_1)
        self.conv4_3 = self.convolution(self.conv4_2)
        self.pool4 = self.pooling(self.conv4_3)

        self.conv5_1 = self.convolution(self.pool4)
        self.conv5_2 = self.convolution(self.conv5_1)
        self.conv5_3 = self.convolution(self.conv5_2)
        self.pool5 = self.pooling(self.conv5_3)

        self.fc6 = self.fully_connection(self.pool5, activation=Activation.relu)
        self.fc7 = self.fully_connection(self.fc6, activation=Activation.relu)
        self.fc8 = self.fully_connection(self.fc7, activation=Activation.softmax)


        self.net = self.fc8

        return 0



    def pooling(self, input, name='pool'):
        """
        Args: output of just before layer
        Return: max_pooling layer
        """
        return tf.nn.max_pool(input, ksize=vgg.ksize, strides=vgg.pool_strides, padding='SAME', name=name)

    def convolution(self, input, name='conv'):
        """
        Args: output of just before layer
        Return: convolution layer
        """
        print('Current input size in convolution layer is: '+str(input.get_shape().as_list()))
        with tf.variable_scope(name):
            kernel = self.getWeight(vgg.structure[self.lcnt][0])
            bias = self.getBias(vgg.structure[self.lcnt][1])
            conv = tf.nn.conv2d(input, kernel, strides=vgg.conv_strides, padding='SAME', name=name)
            self.lcnt += 1
        return tf.nn.relu(tf.add(conv, bias))

    def fully_connection(self, input, name='fc', activation=Activation.relu):
        """
        Args: output of just before layer
        Return: fully_connected layer
        """
        with tf.variable_scope(name):
            shape = input.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(input, [-1, dim])

            weights = self.getWeight([dim, vgg.structure[self.lcnt][0][0]])
            biases = self.getBias(vgg.structure[self.lcnt][1])

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            fc = activation(fc)

            print('Input shape is: '+str(shape))
            print('Total nuron count is: '+str(dim))
            self.lcnt += 1
            
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
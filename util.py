"""
utility function group
such as load_images

date: 9/17
author: arabian9ts
"""

import numpy
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf

from vgg16 import *

def load_image(path):
    """
    load specified image

    Args: image path
    Return: resized image
    """
    img = skimage.io.imread(path)
    img = img / 255.
    resized_img = skimage.transform.resize(img, (224, 224))
    return numpy.array(resized_img, dtype=numpy.float32)


### test process ###
img = load_image('./test.jpg')
print(img.shape)
img = img.reshape((1, 224, 224, 3))

with tf.Session() as sess:
    image = tf.placeholder(shape=[1, 224, 224, 3], dtype=tf.float32)
    feed_dict = {image: img}
    vgg = Vgg16()
    vgg.build(image)
    sess.run(tf.global_variables_initializer())

    prob = sess.run(vgg.net, feed_dict=feed_dict)
    print(prob)
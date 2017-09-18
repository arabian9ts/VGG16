"""
vgg16 trainer

date: 9/18
author: arabian9ts
"""

import tensorflow as tf

from util import *
from vgg16 import *

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
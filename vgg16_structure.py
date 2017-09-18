"""
network-structure of vgg16 excludes fully-connection layer

date: 9/17
author: arabian9ts
"""

# structure of convolution and pooling layer up to fully-connection layer
structure = {
    # convolution layer 1
    'conv1_1': [[3, 3, 3, 64], [64]],
    'conv1_2': [[3, 3, 64, 64], [64]],

    # convolution layer 2
    'conv2_1': [[3, 3, 64, 128], [128]],
    'conv2_2': [[3, 3, 128, 128], [128]],

    # convolution layer 3
    'conv3_1': [[3, 3, 128, 256], [256]],
    'conv3_2': [[3, 3, 256, 256], [256]],
    'conv3_3': [[3, 3, 256, 256], [256]],

    # convolution layer 4
    'conv4_1': [[3, 3, 256, 512], [512]],
    'conv4_2': [[3, 3, 512, 512], [512]],
    'conv4_3': [[3, 3, 512, 512], [512]],

    # convolution layer 5
    'conv5_1': [[3, 3, 512, 512], [512]],
    'conv5_2': [[3, 3, 512, 512], [512]],
    'conv5_3': [[3, 3, 512, 512], [512]],

    # fully-connection 6
    'fc6': [[4096, 0, 0, 0], [4096]],

    # fully-connection 7
    'fc7': [[4096, 0, 0, 0], [4096]],

    # fully-connection 8
    'fc8': [[1000, 0, 0, 0], [1000]],
}

ksize = [1, 2, 2, 1,]
conv_strides = [1, 1, 1, 1,]
pool_strides = [1, 2, 2, 1,]
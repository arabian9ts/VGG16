"""
network-structure of vgg16 excludes fully-connection layer

date: 9/17
author: arabian9ts
"""

# structure of convolution and pooling layer up to fully-connection layer
structure = [
    # convolution layer 1
    [[3, 3, 3, 64], [64]],
    [[3, 3, 64, 64], [64]],

    # convolution layer 2
    [[3, 3, 64, 128], [128]],
    [[3, 3, 128, 128], [128]],

    # convolution layer 3
    [[3, 3, 128, 256], [256]],
    [[3, 3, 256, 256], [256]],
    [[3, 3, 256, 256], [256]],

    # convolution layer 4
    [[3, 3, 256, 512], [512]],
    [[3, 3, 512, 512], [512]],
    [[3, 3, 512, 512], [512]],

    # convolution layer 5
    [[3, 3, 512, 512], [512]],
    [[3, 3, 512, 512], [512]],
    [[3, 3, 512, 512], [512]],

    # fully-connection 6
    [[4096, 0, 0, 0], [4096]],

    # fully-connection 7
    [[4096, 0, 0, 0], [4096]],

    # fully-connection 8
    [[1000, 0, 0, 0], [1000]],
]

ksize = [1, 2, 2, 1,]
conv_strides = [1, 1, 1, 1,]
pool_strides = [1, 2, 2, 1,]
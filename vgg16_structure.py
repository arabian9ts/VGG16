"""
network-structure of vgg16 excludes fully-connection layer

date: 9/17
author: arabian9ts
"""

# structure of convolution and pooling layer up to fully-connection layer
structure = [
    [[3, 3, 3, 64], [64]],
    [[3, 3, 64, 64], [64]],
    [[3, 3, 64, 128], [128]],
    [[3, 3, 128, 128], [128]],
    [[3, 3, 128, 256], [256]],
    [[3, 3, 256, 256], [256]],
    [[3, 3, 256, 256], [256]],
    [[3, 3, 256, 512], [512]],
    [[3, 3, 512, 512], [512]],
    [[3, 3, 512, 512], [512]],
    [[3, 3, 512, 512], [512]],
    [[3, 3, 512, 512], [512]],
    [[3, 3, 512, 512], [512]]
]
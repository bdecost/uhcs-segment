""" keras implementation of PixelNet architecture. """
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Layer, Concatenate, Dropout, Dense
from keras import backend as K

from keras.applications.inception_v3 import conv2d_bn

import os
import sys
sys.path.append(os.getcwd())

from uhcsseg.upsample import sparse_upsample, sparse_upsample_output_shape

def pixelnet_model(nclasses=4):
    """ Use sparse upsample implementations to define a PixelNet model

    @article{pixelnet,
      title={PixelNet: {R}epresentation of the pixels, by the pixels, and for the pixels},
      author={Bansal, Aayush
              and Chen, Xinlei,
              and  Russell, Bryan
              and Gupta, Abhinav
              and Ramanan, Deva},
      Journal={arXiv preprint arXiv:1702.06506},
      year={2017}
    }

    TODO: add batch normalization to conv layers (for training from scratch)
    TODO: consider removing dropout from conv layers

    From the paper and their notes on github, it seems like the semantic segmentation
    task should work either with linear classifier + BatchNorm, or with MLP without BatchNorm.
    """
    
    # a single input channel for grayscale micrographs...
    inputdata = Input(shape=(None,None,1))

    # coordinate tensor: (batch, index)
    # the index axis should contain (b, i, j)
    # i.e. batch index, row index, column index
    inputcoord = Input(shape=(None, 3,), dtype='float32')

    x = conv2d_bn(inputdata, 16, 3, 3, name='block1_conv1')
    x = conv2d_bn(x, 16, 3, 3, name='block1_conv2')
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = conv2d_bn(x1, 32, 3, 3, name='block2_conv1')
    x = conv2d_bn(x, 32, 3, 3, name='block2_conv2')
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = conv2d_bn(x2, 64, 3, 3, name='block3_conv1')
    x = conv2d_bn(x, 64, 3, 3, name='block3_conv2')
    x = conv2d_bn(x, 64, 3, 3, name='block3_conv3')
    x3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = conv2d_bn(x3, 128, 3, 3, name='block4_conv1')
    x = conv2d_bn(x, 128, 3, 3, name='block4_conv2')
    x = conv2d_bn(x, 128, 3, 3, name='block4_conv3')
    x4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = conv2d_bn(x4, 128, 3, 3, name='block5_conv1')
    x = conv2d_bn(x, 128, 3, 3, name='block5_conv2')
    x = conv2d_bn(x, 128, 3, 3, name='block5_conv3')
    x5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    upsample = Lambda(
        sparse_upsample,
        output_shape=sparse_upsample_output_shape,
        name='sparse_upsample'
    )
    sel1 = upsample([x1, inputcoord])
    sel2 = upsample([x2, inputcoord])
    sel3 = upsample([x3, inputcoord])
    sel4 = upsample([x4, inputcoord])
    sel5 = upsample([x5, inputcoord])

    # now we have shape (batch, sample, channel)
    x = Concatenate()([sel1, sel2, sel3, sel4, sel5])

    # flatten into pixel features
    batchsize, npix, nchannels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

    flatten = Lambda(
        lambda t: K.reshape(t, (-1, nchannels)),
        output_shape=lambda s: (-1, s[2]),
        name='flatten_pixel_features'
    )
    x = flatten(x)

    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(nclasses, activation='softmax', name='predictions')(x)

    unflatten = Lambda(
        lambda t: K.reshape(t, (batchsize, npix, nclasses)),
        output_shape=lambda s: (4, 2048, nclasses),
        name='unflatten_pixel_features'
    )
    x = unflatten(x)
    
    model = Model(inputs=[inputdata, inputcoord], outputs=x)
    return model

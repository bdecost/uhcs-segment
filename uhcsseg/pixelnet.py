""" keras implementation of PixelNet architecture. """
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Layer, Concatenate, Dropout, Dense
from keras import backend as K

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

    x = Conv2D(16, (3, 3),  activation='relu', padding='same', name='block1_conv1')(inputdata)
    x = Conv2D(16, (3, 3),  activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    x1 = Dropout(0.25)(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv1')(x1)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    x2 = Dropout(0.25)(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv1')(x2)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    x3 = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block4_conv1')(x3)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    x4 = Dropout(0.25)(x)
    
    sel1 = Lambda(sparse_upsample,
                  output_shape=sparse_upsample_output_shape)([x1, inputcoord])
    sel2 = Lambda(sparse_upsample,
                  output_shape=sparse_upsample_output_shape)([x2, inputcoord])
    sel3 = Lambda(sparse_upsample,
                  output_shape=sparse_upsample_output_shape)([x3, inputcoord])
    sel4 = Lambda(sparse_upsample,
                  output_shape=sparse_upsample_output_shape)([x4, inputcoord])

    # now we have shape (batch, sample, channel)
    x = Concatenate()([sel1, sel2, sel3, sel4])

    def batchify_pixels(x):
        # _, npix, nchannels = tf.shape(x)
        npix, nchannels = tf.shape(x)[1], tf.shape(x)[2]
        return K.reshape(x, (npix, nchannels))

    def batchify_pixels_shape(input_shape):
        # _, npix, nchannels = tf.shape(x)
        _, npix, nchannels = input_shape
        return (npix, nchannels)

    
    # flatten into pixel features
    x = Lambda(batchify_pixels, output_shape=batchify_pixels_shape)(x)

    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(nclasses, activation='softmax', name='predictions')(x)
    
    def unbatchify_pixels(x):
        # npix, nchannels = tf.shape(x)
        print(tf.shape(x))
        npix, nchannels = tf.shape(x)[0], tf.shape(x)[1]
        return K.reshape(x, (1, npix, nchannels))

    def unbatchify_pixels_shape(input_shape):
        # npix, nchannels = tf.shape(x)
        npix, nchannels = input_shape
        return (1, npix, nchannels)
    
    x = Lambda(unbatchify_pixels, output_shape=unbatchify_pixels_shape)(x)
    
    model = Model(inputs=[inputdata, inputcoord], outputs=x)
    return model

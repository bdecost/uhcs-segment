"""
implementation of sparse upsampling of keras tensors
using bilinear interpolation
"""

import numpy as np
import tensorflow as tf
from keras import backend as K

def get_values(data, batch, x, y):
    """ construct index tensor for tf.gather_nd """
    coords = tf.stack((batch, x, y), 2)
    indices = tf.cast(coords, tf.int32)
    return tf.gather_nd(data, indices)    

def sparse_upsample_bilinear(inputs, **arguments):
    """ upsample the input tensor `data` with indices in input tensor sel
    performs sparse bilinear interpolation
    indices should explicitly contain the sample index: (b, i, j)
    """
    data, coords = inputs
    h = tf.cast(tf.shape(data)[1], tf.float32)
    w = tf.cast(tf.shape(data)[2], tf.float32)
    
    # transform fractional coordinates to feature map coordinates
    batch = coords[:,:,0]
    x = h * coords[:,:,1]
    y = w * coords[:,:,2]

    x1, x2 = tf.floor(x), tf.ceil(x)
    y1, y2 = tf.floor(y), tf.ceil(y)

    # horizontal interpolation first
    top = (x2 - x) * get_values(data, batch, x1, y2)  + (x - x1) * get_values(data, batch, x2, y2)
    bottom = (x2 - x) * get_values(data, batch, x1, y1)  + (x - x1) * get_values(data, batch, x2, y1)

    # vertical interpolation
    interp = (y2 - y) * top +  (y - y1) * bottom
    return interp
    # x, y = tf.round(x), tf.round(y)
    # return get_values(data, batch, x, y)

def sparse_upsample_nearest(inputs, **arguments):
    """ 'upsample' input tensor `data` with indices in input tensor sel
    yields the value of the nearest pixel in the `data` tensor
    indices should explicitly contain the sample index: (b, i, j)
    """
    data, coords = inputs
    h = tf.cast(tf.shape(data)[1], tf.float32)
    w = tf.cast(tf.shape(data)[2], tf.float32)
    
    # transform fractional coordinates to feature map coordinates
    batch = coords[:,:,0]
    x = h * coords[:,:,1]
    y = w * coords[:,:,2]
    x, y = tf.round(x), tf.round(y)
    return get_values(data, batch, x, y)

def sparse_upsample_output_shape(input_shape):
    data_shape, index_shape = input_shape
    assert K.backend() == 'tensorflow'
    assert len(data_shape) == 4 # only valid for 4D tensors
    assert len(index_shape) == 3
    return (data_shape[0], index_shape[0], data_shape[3])


# aliases
sparse_upsample = sparse_upsample_bilinear

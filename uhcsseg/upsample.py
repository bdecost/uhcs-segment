"""
implementation of sparse upsampling of keras tensors
using bilinear interpolation
"""

import numpy as np
import tensorflow as tf
from keras import backend as K

def build_index(batch, x, y):
    coords = tf.stack((batch, x, y), 2)
    indices = tf.cast(tf.round(coords), tf.int32)
    

def get_values(inputs, **arguments):
    """ index into input tensor x with indices in input tensor sel
    indices should explicitly contain the sample index: (b, i, j)
    arguments: h, w -- dimensions of feature map to upsample
        needed since dimensions are not known at model compile time
    """
    data, coords = inputs
    h = tf.cast(tf.shape(data)[1], tf.float32)
    w = tf.cast(tf.shape(data)[2], tf.float32)
    
    # transform fractional coordinates to feature map coordinates
    batch = coords[:,:,0]
    x = coords[:,:,1] * h
    y = coords[:,:,2] * w
    
    x1, x2 = tf.floor(x), tf.ceil(x)
    y1, y2 = tf.floor(y), tf.ceil(y)
    
    coords = tf.stack((batch, x, y), 2)
    indices = tf.cast(tf.round(coords), tf.int32)

    return tf.gather_nd(data, indices)

def sparse_upsample_output_shape(input_shape):
    data_shape, index_shape = input_shape
    assert K.backend() == 'tensorflow'
    assert len(data_shape) == 4 # only valid for 4D tensors
    assert len(index_shape) == 3
    return (data_shape[0], index_shape[0], data_shape[3])

"""
implementation of sparse upsampling of keras tensors
using bilinear interpolation
"""

import numpy as np
import tensorflow as tf
from keras import backend as K

def get_values(inputs, **arguments):
    """ index into input tensor x with indices in input tensor sel
    indices should explicitly contain the sample index: (b, i, j)
    arguments: h, w -- dimensions of feature map to upsample
        needed since dimensions are not known at model compile time
    """
    x, coords = inputs

    # transform fractional coordinates to feature map coordinates
    s1 = coords[:,:,1] * arguments['h']
    s2 = coords[:,:,2] * arguments['w']
    coords = tf.stack((tf.zeros_like(s1), s1, s2), 2)
    indices = tf.cast(tf.round(coords), tf.int32)

    return tf.gather_nd(x, indices)

def sparse_upsample_output_shape(input_shape):
    data_shape, index_shape = input_shape
    assert K.backend() == 'tensorflow'
    assert len(data_shape) == 4 # only valid for 4D tensors
    assert len(index_shape) == 3
    return (data_shape[0], index_shape[0], data_shape[3])


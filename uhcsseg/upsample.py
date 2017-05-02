"""
implementation of sparse upsampling of keras tensors
using bilinear interpolation
"""

import numpy as np
import tensorflow as tf
from keras import backend as K

def get_values(inputs):
    """ index into input tensor x with indices in input tensor sel
    indices should explicitly contain the sample index: (b, i, j)
    """
    x, sel = inputs
    return tf.gather_nd(x, sel)

def sparse_upsample_output_shape(input_shape):
    data_shape, index_shape = input_shape
    assert len(data_shape) == 4 # only valid for 4D tensors
    assert len(index_shape) == 3
    return (data_shape[0], index_shape[0], data_shape[3])


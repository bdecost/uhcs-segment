import os
import numpy as np
import skimage.data
from scipy.misc import imresize

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Layer
from keras import backend as K

import sys
sys.path.append(os.getcwd())

from uhcsseg.upsample import sparse_upsample, sparse_upsample_output_shape

def upsample_model():
    
    inputdata = Input(shape=(None,None,3))

    # coordinate tensor: (batch, index)
    # the index axis should contain (b, i, j)
    # i.e. batch index, row index, column index
    inputcoord = Input(shape=(None, 3,), dtype='float32')
    sel = Lambda(sparse_upsample,
                 output_shape=sparse_upsample_output_shape)([inputdata, inputcoord])

    model = Model(inputs=[inputdata, inputcoord], outputs=sel)
    return model

if __name__ == '__main__':
    im = skimage.data.coffee()
    print(im.shape)
    scale = 0.0625
    im_s = imresize(im, scale)
    print(im_s.shape)

    im_r = imresize(im_s, 1/scale)
    
    coords = np.array([
        [0, 36, 36],
        [0, 18, 66],
        [0, 47, 31],
        [0, 73, 91],
        [0, 83, 64],
        [0, 0, 0],
        [0, 300, 300],
        [0, 399, 100],
        [0, 1, 599]]
    ).astype(np.float32)
    # print('coords')
    # print(coords)

    # r = tf.image.resize_nearest_neighbor(im_s[np.newaxis,:,:,:], (im.shape[0], im.shape[1]))
    # with tf.Session() as sess:
    #     r = sess.run(r[0])

    r = tf.image.resize_bilinear(im_s[np.newaxis,:,:,:], (im.shape[0], im.shape[1]))
    with tf.Session() as sess:
        r = sess.run(r[0])
    
    # print ground truth (tensorflow) upsampled results
    print('tensorflow upsampled:')
    for coord in coords:
        coord = coord.astype(np.int32)
        print(r[coord[1],coord[2]])
    
    # normalize sample coordinates on [0,1] interval
    coords[:,1] = coords[:,1] / im.shape[0]
    coords[:,2] = coords[:,2] / im.shape[1]
    
    model = upsample_model()

    f = model.predict([im_s[np.newaxis,:,:,:], coords[np.newaxis,:,:]])
    print('sparse upsample:')
    print(f[0])

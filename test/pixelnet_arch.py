#!/usr/bin/env python
import os
import numpy as np
import skimage.data
import skimage.color
from scipy.misc import imresize

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Layer
from keras import backend as K

import sys
sys.path.append(os.getcwd())

from uhcsseg.upsample import sparse_upsample, sparse_upsample_output_shape
from uhcsseg.pixelnet import pixelnet_model

if __name__ == '__main__':
    im = skimage.data.coffee()
    im = skimage.color.rgb2gray(im)

    coords = np.random.random((1,2048,3))
    coords *= np.array([0, im.shape[0], im.shape[1]])
    coords = coords.astype(np.int32)
    print(coords)
    
    model = pixelnet_model()

    f = model.predict([im[np.newaxis,:,:,np.newaxis], coords])
    print('sparse upsample:')
    print(f)

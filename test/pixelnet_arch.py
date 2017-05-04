#!/usr/bin/env python
import os
import numpy as np
import skimage.data
import skimage.color

import sys
sys.path.append(os.getcwd())

from uhcsseg.pixelnet import pixelnet_model

if __name__ == '__main__':
    im = skimage.data.coffee()
    im = skimage.color.rgb2gray(im)

    coords = np.random.random((1,2048,3))
    coords *= np.array([0, 1, 1])

    print('coordinates to sample:')
    print(coords)
    
    model = pixelnet_model()
    f = model.predict([im[np.newaxis,:,:,np.newaxis], coords])
    print('sparse upsample:')
    print(f)

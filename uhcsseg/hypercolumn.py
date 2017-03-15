""" reduced hypercolumn features with keras """
import cv2
import h5py
import numpy as np
import scipy as sp
from skimage.color import gray2rgb
from sklearn.decomposition import PCA

from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input

def image_tensor(image):
    image3d = gray2rgb(image).astype(np.float32)
    x = 255*image3d.transpose((2,0,1))
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)

layers=('block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3')
cnn = VGG16(include_top=False, weights='imagenet')
layer_id = {layer.name: idx for idx, layer in enumerate(cnn.layers)}

class ReducedHyperColumn():
    def __init__(self, n_components=32, layers=layers, cnn=cnn):
        self.layers = layers
        self.n_components = n_components
        self.cnn = cnn
        self.layer_id = {layer.name: idx for idx, layer in enumerate(self.cnn.layers)}

        self.model = Model(
            input=self.cnn.input,
            output=list(map(lambda layer: self.cnn.get_layer(layer).output, self.layers))
        )

        self.block_pca = {
            layer: PCA(n_components=n_components, whiten=True) 
            for layer in layers
        }

    def reduced_hypercolumn(self, images, verbose=False, train=False):
        """ 
        hypercolumn features with block-wise PCA whitening
        extract multiple feature maps for multiple images
        apply PCA with whitening to each feature map set
        interpolate each reduced feature map onto the target image size
        concatenate feature maps into whitened hypercolumn features
        """


        # extract featuremaps for multiple images
        h_target, w_target = images[0].shape

        x_in = [image_tensor(im) for im in images]
        if verbose:
            print('computing feature maps')
        xx = self.model.predict(np.vstack(x_in))

        hc = []
        for block, features in zip(self.layers, xx):
        
            if verbose:
                print('reducing {} features'.format(block))
            
            # reshape feature map into [feature, channels]
            b, nchan, h, w = features.shape
            ff = features.transpose(0,2,3,1) # to [batch, height, width, channels]
            ff = ff.reshape((-1, nchan)) # to [feature, channels]

            if train:
                if ff.shape[0] > 1e6:
                    choice = np.random.choice(ff.shape[0], size=int(1e6), replace=False)
                    self.block_pca[block].fit(ff[choice])
                else:
                    self.block_pca[block].fit(ff)
            
            xpca = self.block_pca[block].transform(ff)

            # reshape reduced feature maps back to image tensor layout
            xxpca = xpca.reshape((b,h,w,self.n_components)) # to batch, height, width, channels
            xxpca = xxpca.transpose(0,3,1,2) # to batch, channels, height, width
    
            # interpolate each channel onto target size
            if verbose:
                print('interpolating {} features'.format(block))
            # preallocate output
            hc_block = np.zeros((b, self.n_components, h_target, w_target))
    
            for image_idx in range(b):
                for channel_idx in range(self.n_components):
                    # careful: cv2 swaps image shape order...
                    hc_block[image_idx,channel_idx,:,:] = cv2.resize(xxpca[image_idx,channel_idx,:,:], (w_target, h_target))
            
            hc.append(hc_block)
    
        return np.concatenate(hc, axis=1)

    def fit(self, images, verbose=False):
        return self.reduced_hypercolumn(images, verbose=verbose, train=True)

    def predict(self, images, verbose=False):
        return self.reduced_hypercolumn(images, verbose=verbose, train=False)

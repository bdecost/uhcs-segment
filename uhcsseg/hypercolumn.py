""" reduced hypercolumn features with keras """
import cv2
import h5py
import numpy as np
import scipy as sp
from skimage.color import gray2rgb
from sklearn.decomposition import PCA

from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input

# use VGG16 model trained on imagenet
cnn = VGG16(include_top=False, weights='imagenet')
layer_id = {layer.name: idx for idx, layer in enumerate(cnn.layers)}

layers=('block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3')
model = Model(
    input=cnn.input,
    output=list(map(lambda layer: cnn.get_layer(layer).output, layers))
)

def image_tensor(image):
    image3d = gray2rgb(image).astype(np.float32)
    x = 255*image3d.transpose((2,0,1))
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)

def reduced_hypercolumn(images, n_components=32):
    """ 
    hypercolumn features with block-wise PCA whitening
    extract multiple feature maps for multiple images
    apply PCA with whitening to each feature map set
    interpolate each reduced feature map onto the target image size
    concatenate feature maps into whitened hypercolumn features
    """

    block_pca = {
        layer: PCA(n_components=n_components, whiten=True) 
        for layer in layers
    }

    # extract featuremaps for multiple images
    h_target, w_target = images[0].shape

    x_in = [image_tensor(im) for im in images]
    xx = model.predict(np.vstack(x_in))

    hc = []
    for feature_idx in range(len(layers)):
        print('block {}'.format(feature_idx))
        # reshape feature map into [feature, channels]
        b, nchan, h, w = xx[feature_idx].shape
        X = xx[feature_idx]
        ff = X.transpose(0,2,3,1) # to [batch, height, width, channels]
        ff = ff.reshape((-1, nchan)) # to [feature, channels]
    
        block = layers[feature_idx]
        if ff.shape[0] > 1e6:
            choice = np.random.choice(ff.shape[0], size=1e6, replace=False)
            block_pca[block].fit(ff[choice])
        else:
            block_pca[block].fit(ff)
            
        xpca = block_pca[block].transform(ff)

        # reshape reduced feature maps back to image tensor layout
        xxpca = xpca.reshape((b,h,w,n_components)) # to batch, height, width, channels
        xxpca = xxpca.transpose(0,3,1,2) # to batch, channels, height, width
    
        # interpolate each channel onto target size
        # preallocate output
        hc_block = np.zeros((b, n_components, h_target, w_target))
    
        for image_idx in range(b):
            for channel_idx in range(n_components):
                # careful: cv2 swaps image shape order...
                hc_block[image_idx,channel_idx,:,:] = cv2.resize(xxpca[image_idx,channel_idx,:,:], (w_target, h_target))
            
        hc.append(hc_block)
    
    hcc = np.concatenate(hc, axis=1)
    return hcc

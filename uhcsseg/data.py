import h5py
import numpy as np

def load_dataset(hfile, cropbar=None):
    """ load uhcsseg training data from hdf5 """
    
    images, labels, names = [], [], []
    with h5py.File(hfile, 'r') as f:
        for key in f:
            micrograph = f[key]
            im = micrograph['image'][...]
            l = micrograph['labels'][...]

            if cropbar is not None and cropbar > 0:
                # remove micron bar from bottom of image
                im = im[:-cropbar]
                l = l[:-cropbar]
                
            names.append(key)        
            images.append(im)
            labels.append(l)
            
    return np.array(images), np.array(labels), np.array(names)

def preprocess_images(images, normalize=False):
    """ preprocess images """
    if normalize:
        # zero-mean and unit variance scaling
        images = (images - np.mean(images)) / np.std(images)
    else:
        # just remap intensities to (-1,1)
        images = images / 255.0
        images = images - 0.5
        images = 2 * images

    return images
    

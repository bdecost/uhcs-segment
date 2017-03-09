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

            if cropbar is not None:
                # remove micron bar from bottom of image
                im = im[:-cropbar]
                l = l[:-cropbar]
                
            names.append(key)        
            images.append(im)
            labels.append(l)
            
    return images, labels, names

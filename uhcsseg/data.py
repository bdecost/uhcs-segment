import json
import h5py
import numpy as np
from scipy.misc import bytescale

from enum import IntEnum
class Microconstituent(IntEnum):
    """ Enumerate integer labels for microconstituents """
    matrix = 0
    network = 1
    spheroidite = 2
    widmanstatten = 3

# throw out these micrographs for the spheroidite task
# the input has a weird intensity distribution
exclude = {
    '800C-85H-Q-4',
    '800C-8H-Q-2',
    '800C-90M-Q-1'
}

def load_record(f, key, cropbar=None):
    micrograph = f[key]
    im = micrograph['image'][...]
    l = micrograph['labels'][...]

    im = im.astype(np.float32)

    if cropbar is not None and cropbar > 0:
        # remove micron bar from bottom of image
        im = im[:-cropbar]
        l = l[:-cropbar]
    return im, l

def load_dataset(hfile, cropbar=None):
    """ load uhcsseg training data from hdf5 """
    
    images, labels, names = [], [], []
    with h5py.File(hfile, 'r') as f:
        for key in f:
            if key in exclude:
                continue
            im, l = load_record(f, key, cropbar=cropbar)   
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
    
def load_validation_set(validation_set_path, run_id):
    with open(validation_set_path, 'r') as jf:
        jdata = json.load(jf)
        return jdata[str(run_id)]

def validation_split(validation_set, names):
    """ take a list of string keys for validation set images and index into the data array """
    train_set = list(filter(lambda s: s not in validation_set, names))
    val_idx = [names.tolist().index(k) for k in validation_set]
    train_idx = [names.tolist().index(k) for k in train_set]
    return train_idx, val_idx

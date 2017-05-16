#!/usr/bin/env python

import os
import re
import glob
import h5py
import nrrd
import zipfile
import tempfile
import numpy as np
from skimage import color
from scipy.misc import imresize
from skimage.color import label2rgb

# fill label image -- matrix: 0
label_map = {
    'background': -1,
    'matrix': 0,
    'network': 1,
    'spheroidite': 2,
    'widmanstatten': 3
}

def _read_mitk_field(archive, fieldpath):
    """ mitk stores a zip archive with one nrrd file per label mask 
    read individual nrrd files into tempfiles, pass to internal nrrd io routines """

    b = archive.read(fieldpath)
    with tempfile.TemporaryFile() as tf:
        tf.write(b)
        tf.seek(0)
        header = nrrd.read_header(tf)
        data = nrrd.read_data(header, tf, None)
        # data = np.transpose(data)
        # mitk stores column-major 3D image....
        # if data.ndim == 3:
        #     data = data[0]
        return data
    
def read_mitk_field(filepath, field):
    """ mitk stores a zip archive with one nrrd file per label mask 
    read individual nrrd files into tempfiles, pass to internal nrrd io routines """
    paths = peek_mitk(filepath)
        
    with zipfile.ZipFile(filepath, 'r') as archive:
        return _read_mitk_field(archive, paths[field])
        
def read_mitk_fields(filepath):
    """ mitk stores a zip archive with one nrrd file per label mask 
    read individual nrrd files into tempfiles, pass to internal nrrd io routines """
    paths = peek_mitk(filepath)

    with zipfile.ZipFile(filepath, 'r') as archive:
        fields = {
            field: _read_mitk_field(archive, path)
            for field, path in paths.items()
        }
        
        return fields
        
def peek_mitk(filepath, extension='nrrd'):
    fields = dict()
    with zipfile.ZipFile(filepath, 'r') as archive:
        for name in filter(lambda name: extension in name, archive.namelist()):
            # split arbitrary part out of filename
            prefix, _ = os.path.splitext(name)
            # Segmentation editor tools (Union, Difference, etc) give compound names
            prefix, *rest = prefix.split('_')
            key = '_'.join(rest)
            fields[key] = name
            
    return fields
        
def blend_images(original, colormask, alpha=0.8):
    if colormask.shape != original.shape:
        colormask = imresize(colormask, original.shape, interp='nearest')
        
    i_hsv = color.rgb2hsv(np.dstack((original, original, original)))
    mask_hsv = color.rgb2hsv(colormask)
    i_hsv[..., 0] = mask_hsv[..., 0]
    i_hsv[..., 1] = mask_hsv[..., 1] * alpha

    return color.hsv2rgb(i_hsv)

def get_label_image(mitkfile):
    micrograph_id = int( re.search(r'\d+', sourcefile).group() )
    f = read_mitk_fields(sourcefile)
    micrograph = f['micrograph{}'.format(micrograph_id)].T
    
    h, w = micrograph.shape
    labels = np.zeros((h,w), dtype=int)

    # take advantage of label ordering:
    # do widmanstatten cementite last, overwriting everything else
    for label in sorted(label_map.keys()):
        idx = label_map[label]
        try:
            labels[f[label].T[0] == 1] = idx
        except KeyError:
            continue

    # set metadata region to background: -1
    labels[-38:] = -1
    return micrograph, labels
    

if __name__ == '__main__':
    with h5py.File('data/uhcs.h5', 'w') as f:
        for sourcefile in glob.glob('data/mitk/*.mitk'):
            m = int( re.search(r'\d+', sourcefile).group() )
            g = f.create_group('micrograph{}'.format(m))
            micrograph, labels = get_label_image(sourcefile)
            
            colors=['blue', 'red', 'cyan', 'yellow', 'black']

            g['image'] = micrograph
            g['labels'] = labels
            g['overlay'] = blend_images(micrograph, label2rgb(labels, colors=colors))

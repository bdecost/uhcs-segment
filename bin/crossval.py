#!/usr/bin/env python

import os
import h5py
import click
import numpy as np
from sklearn.model_selection import LeaveOneOut

import sys
sys.path.append(os.getcwd())

from uhcsseg import data
from uhcsseg import hypercolumn, tensorsgd

@click.command()
@click.argument('hfile', type=click.Path())
@click.option('-r', '--resultsfile', default='data/segresults.h5', type=click.Path(),
              help='hdf5 file to store results.')
@click.option('-c', '--crop', default=38, type=int, help='pixels to remove from image bottom')
def crossval(hfile, resultsfile, crop):
    """Run LOOCV with reduced hypercolumn features and SGD with linear SVM loss.
    Read data from hdf5 HFILE with input images and annotations.
    """
    cv = LeaveOneOut()
    images, labels, keys = data.load_dataset(hfile, cropbar=crop)

    for train_idx, val_idx in cv.split(images):
        print('CV iteration {}'.format(val_idx[0]))

        hc = hypercolumn.ReducedHyperColumn()
        clf = tensorsgd.TensorSGD()
        
        X_train = hc.fit(images[train_idx], verbose=True)
        clf.fit(X_train, labels[train_idx])

        train_pred = clf.predict(X_train)
    
        X_val = hc.predict(images[val_idx], verbose=True)
        val_pred = clf.predict(X_val)

        if os.path.isfile(resultsfile):
            mode = 'r+'
        else:
            mode = 'w'
        
        with h5py.File(resultsfile, mode) as f:
            # save validation predictions
            for pred, key in zip(val_pred, keys[val_idx]):
                try:
                    g = f[key]
                except KeyError:
                    g = f.create_group(key)
                g['validation'] = pred

            # save training predictions
            for pred, key in zip(train_pred, keys[train_idx]):
                try:
                    g = f[key]
                except KeyError:
                    g = f.create_group(key)
                g['train{}'.format(val_idx)] = pred

            
if __name__ == '__main__':
    crossval()

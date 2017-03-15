#!/usr/bin/env python

import os
import h5py
import click
import numpy as np
from sklearn.model_selection import LeaveOneOut

import sys
sys.path.append(os.getcwd())

from uhcsseg import io
from uhcsseg import hypercolumn, tensorsgd

@click.command()
@click.argument('hfile', type=click.Path())
@click.option('-r', '--resultsfile', default='data/segresults.h5', type=click.Path())
def crossval(hfile, resultsfile):

    cv = LeaveOneOut()
    images, labels, keys = io.load_dataset(hfile, cropbar=38)

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
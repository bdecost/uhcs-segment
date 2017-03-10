#!/usr/bin/env python

import os
import h5py
import click
import numpy as np

import sys
sys.path.append(os.getcwd())

from uhcsseg import io
from uhcsseg import hypercolumn, segment

@click.command()
@click.argument('hfile', type=click.Path())
def crossval(hfile):
    
    images, labels, keys = io.load_dataset(hfile, cropbar=38)

    ntrain = 4
    t_images = images[-4:]
    t_labels = labels[-4:]
    t_keys = keys[-4:]
    
    hc = hypercolumn.ReducedHyperColumn()
    clf = segment.TensorSGD()
    
    Xtrain = hc.fit(images[:ntrain], verbose=True)
    clf.fit(Xtrain, labels[:ntrain])

    y_train = clf.predict(Xtrain)
    
    Xtest = hc.predict(t_images, verbose=True)
    y_pred = clf.predict(Xtest)

    with h5py.File('data/segresults.h5', 'w') as f:
        # save validation predictions
        for pred, key in zip(y_pred, t_keys):
            try:
                g = f[key]
            except KeyError:
                g = f.create_group(key)
            g['validation'] = pred

        # save training predictions
        for pred, key in zip(y_train, keys[:ntrain]):
            try:
                g = f[key]
            except KeyError:
                g = f.create_group(key)
            g['train0'] = pred

            
if __name__ == '__main__':
    crossval()

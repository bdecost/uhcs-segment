#!/usr/bin/env python

import os
import h5py
import click
import numpy as np

from skimage import filters

import sys
sys.path.append(os.getcwd())

from uhcsseg import data
@click.command()
@click.argument('hfile', type=click.Path())
@click.option('-r', '--resultsfile', default='data/sph_thresh.h5', type=click.Path())
def threshold(hfile, resultsfile):

    images, labels, keys = data.load_dataset(hfile)

    for image, label, key in zip(images, labels, keys):

        methods = filter(lambda f: 'threshold' in f, filters.__all__)
        for method in methods:
            if 'adaptive' in method:
                # skip adaptive for now.
                continue
            
            threshold = getattr(filters, method)

        
            # do threshold
            pred = (image > threshold(image)).astype(int)
        
            # save results.
            if os.path.isfile(resultsfile):
                mode = 'r+'
            else:
                mode = 'w'

            with h5py.File(resultsfile, mode) as f:
                # save validation predictions
                try:
                    g = f[key]
                except KeyError:
                    g = f.create_group(key)
                g[method] = pred

            
if __name__ == '__main__':
    threshold()


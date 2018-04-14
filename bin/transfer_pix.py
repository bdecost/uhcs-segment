#!/usr/bin/env python
from __future__ import division, print_function

import os
import glob
import json
import click
import numpy as np
from keras import optimizers
from keras import applications
from keras.utils.np_utils import normalize
from keras import callbacks

import sys
sys.path.append(os.getcwd())
sys.path.append('../pixelnet')


from uhcsseg import data, perf, visualization, losses, adamw
from pixelnet import pixelnet, hypercolumn
from pixelnet import utils as px_utils
from pixelnet import vgg

# suppress some of the noisier tensorflow log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@click.command()
@click.option('--dataset', default='uhcs', type=click.Choice(['uhcs', 'spheroidite']))
@click.option('--batchsize', default=4, type=int)
@click.option('--npix', default=2048, type=int)
@click.option('--max-epochs', default=25, type=int)
@click.option('--validation-steps', default=10, type=int)
@click.option('--run-id', default=0, type=int)
@click.option('--bottleneck/--no-bottleneck', default=False)
def train_pixelnet(dataset, batchsize, npix, max_epochs, validation_steps, run_id, bottleneck):

    datadir = 'data'
    datafile = os.path.join(datadir, '{}.h5'.format(dataset))
    
    validation_set_path = os.path.join(datadir, '{}-validation-sets.json'.format(dataset))
    validation_set = data.load_validation_set(validation_set_path, run_id)
    
    if dataset == 'uhcs':
        nclasses = 4
        cropbar = 38
    elif dataset == 'spheroidite':
        nclasses = 2
        cropbar = None
        
    model_dir = os.path.join('models', 'crossval', dataset, 'run{:02d}'.format(run_id))
    os.makedirs(model_dir, exist_ok=True)
    
    images, labels, names = data.load_dataset(datafile, cropbar=cropbar)
    
    # add a channel axis (of size 1 since these are grayscale inputs)
    images = images[:,:,:,np.newaxis]
    images = np.repeat(images, 3, axis=-1)
    images = applications.vgg16.preprocess_input(images)

    # train/validation split
    train_idx, val_idx = data.validation_split(validation_set, names)
    ntrain = len(train_idx)
    
    X_train, y_train, names_train = images[train_idx], labels[train_idx], names[train_idx]
    X_val, y_val, names_val = images[val_idx], labels[val_idx], names[val_idx]

    inv_freq = y_train.size / np.bincount(y_train.flat)
    class_weights = np.squeeze(normalize(np.sqrt(inv_freq), order=1))

    # write the validation set to the model directory as well...
    with open(os.path.join(model_dir, 'validation_set.txt'), 'w') as vf:
        for name in names_val:
            print(name, file=vf)

    N, h, w, _ = images.shape

    steps_per_epoch = int(ntrain / batchsize)
    print('steps_per_epoch: {}'.format(steps_per_epoch))

    max_epochs = 25
    validation_steps = 10
    
    base_model = vgg.fully_conv_model()

    layernames = [
        'block1_conv2_relu', 'block2_conv2_relu', 'block3_conv3_relu', 'block4_conv3_relu', 'block5_conv3_relu', 'fc2_relu'
    ]

    hc = hypercolumn.build_model(base_model, layernames, batchnorm=True, mode='sparse', relu=False)
    model = pixelnet.build_model(hc, width=1024, mode='sparse', dropout_rate=0.2, l2_reg=0.0)
    
    opt = adamw.Adam(lr=1e-3, weight_decay=1e-4, amsgrad=True)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(loss=losses.focal_crossentropy_loss(class_weights=class_weights), optimizer=opt, metrics=['acc'])

    csv_logger = callbacks.CSVLogger(os.path.join(model_dir, 'training-1.log'))
    checkpoint = callbacks.ModelCheckpoint(
        os.path.join(
            model_dir,
            'weights.{epoch:03d}-{val_loss:.4f}.hdf5'
        ),
        save_best_only=False,
        save_weights_only=True,
        period=25
    )

    training_data = px_utils.random_pixel_samples(
        X_train, y_train, nclasses=nclasses,
        replace_samples=False, horizontal_flip=True, vertical_flip=True
    )

    
    f = model.fit_generator(
        training_data,
        steps_per_epoch,
        epochs=max_epochs,
        callbacks=[csv_logger, checkpoint],
        validation_data=px_utils.random_pixel_samples(X_val, y_val, nclasses=nclasses, replace_samples=False),
        validation_steps=validation_steps,
    )

    for layer in base_model.layers:
        layer.trainable = True

    # fine-tune the whole network
    opt = adamw.Adam(lr=1e-4, weight_decay=1e-4, amsgrad=True)
    model.compile(loss=losses.focal_crossentropy_loss(class_weights=class_weights), optimizer=opt, metrics=['acc'])

    csv_logger = callbacks.CSVLogger(os.path.join(model_dir, 'finetune-1.log'))
    checkpoint = callbacks.ModelCheckpoint(
        os.path.join(
            model_dir,
            'weights-finetune.{epoch:03d}-{val_loss:.4f}.hdf5'
        ),
        save_best_only=False,
        save_weights_only=True,
        period=5
    )

    f = model.fit_generator(
        training_data,
        steps_per_epoch,
        epochs=max_epochs,
        callbacks=[csv_logger, checkpoint],
        validation_data=px_utils.random_pixel_samples(X_val, y_val, nclasses=nclasses, replace_samples=False),
        validation_steps=validation_steps,
    )


if __name__ == '__main__':
    train_pixelnet()

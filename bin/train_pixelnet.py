#!/usr/bin/env python
import os
import click
import numpy as np
from keras import optimizers
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

import sys
sys.path.append(os.getcwd())
sys.path.append('../pixelnet')

from uhcsseg import data
from pixelnet.pixelnet import pixelnet_model
from pixelnet.utils import random_pixel_samples

# suppress some of the noisier tensorflow log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@click.command()
@click.option('--dataset', default='uhcs', type=click.Choice(['uhcs', 'spheroidite']))
@click.option('--batchsize', default=4, type=int)
@click.option('--npix', default=2048, type=int)
@click.option('--max-epochs', default=10, type=int)
@click.option('--validation-steps', default=10, type=int)
def train_pixelnet(dataset, batchsize, npix, max_epochs, validation_steps):

    run_id = 1
    ntrain = 20
    
    datadir = 'data'
    datafile = os.path.join(datadir, '{}.h5'.format(dataset))
    if dataset == 'uhcs':
        nclasses = 4
        cropbar = 38
    elif dataset == 'spheroidite':
        nclasses = 2
        cropbar = None
        
    model_dir = os.path.join('models', 'crossval', dataset, 'run{:02d}'.format(run_id))
    os.makedirs(model_dir, exist_ok=True)
    
    images, labels, names = data.load_dataset(datafile, cropbar=cropbar)
    images = data.preprocess_images(images)

    # add a channel axis (of size 1 since these are grayscale inputs)
    images = images[:,:,:,np.newaxis]

    # train/validation split
    X_train, y_train = images[:ntrain], labels[:ntrain]
    X_val, y_val = images[ntrain:], labels[ntrain:]
    
    N, h, w, _ = images.shape

    # steps_per_epoch = 100
    steps_per_epoch = ntrain * h * w / (batchsize*npix)
    print('steps_per_epoch: {}'.format(steps_per_epoch))
    
    opt = optimizers.Adam()
    model = pixelnet_model()
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    csv_logger = CSVLogger(os.path.join(model_dir, 'training-1.log'))
    checkpoint = ModelCheckpoint(
        os.path.join(
            model_dir,
            'weights.{epoch:03d}-{val_loss:.4f}.hdf5'
        ),
        save_best_only=True,
        save_weights_only=True
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    # note: keras/engine/training.py:L132 --> is not None
    f = model.fit_generator(
        random_pixel_samples(X_train, y_train),
        steps_per_epoch,
        epochs=max_epochs,
        callbacks=[csv_logger, checkpoint, early_stopping],
        validation_data=random_pixel_samples(X_val, y_val),
        validation_steps=validation_steps
    )

if __name__ == '__main__':
    train_pixelnet()

#!/usr/bin/env python
import os
import numpy as np

import tensorflow as tf

from keras import optimizers
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

import sys
sys.path.append(os.getcwd())
sys.path.append('../pixelnet')

from pixelnet.pixelnet import pixelnet_model
from pixelnet.utils import random_pixel_samples
from uhcsseg import data

# suppress some of the noisier tensorflow log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    
    datafile = 'data/uhcs.h5'
    run_id = 1
    batchsize = 4
    ntrain = 20
    npix = 2048
    nclasses = 4
    max_epochs = 10
    validation_steps = 10

    dataset_name, ext = os.path.splitext(os.basename(datafile))
    model_dir = os.path.join('models', 'crossval', dataset_name, 'run{:02d}'.format(run_id))
    os.path.makedirs(model_dir, exist_ok=True)
    
    images, labels, names = data.load_dataset(datafile, cropbar=38)
    images = data.preprocess_images(images)

    # add a channel axis (of size 1 since these are grayscale inputs)
    images = images[:,:,:,np.newaxis]

    # train/validation split
    X_train, y_train = images[:ntrain], labels[:ntrain]
    X_val, y_val = images[ntrain:], labels[ntrain:]
    
    N, h, w, _ = images.shape
        
    steps_per_epoch = ntrain * h * w / (batchsize*npix)
    # steps_per_epoch = 100
    print('steps_per_epoch:')
    print(steps_per_epoch)
    
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

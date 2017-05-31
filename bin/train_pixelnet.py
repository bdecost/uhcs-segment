#!/usr/bin/env python
import os
import glob
import click
import numpy as np
from keras import optimizers
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

import sys
sys.path.append(os.getcwd())
sys.path.append('../pixelnet')

from uhcsseg import data, perf
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
    val_idx = np.sort(np.random.choice(range(N), size=N-ntrain, replace=False))
        
    X_train, y_train = images[:ntrain], labels[:ntrain]
    X_val, y_val = images[ntrain:], labels[ntrain:]
    
    N, h, w, _ = images.shape

    # steps_per_epoch = 100
    steps_per_epoch = ntrain * h * w / (batchsize*npix)
    print('steps_per_epoch: {}'.format(steps_per_epoch))
    
    opt = optimizers.Adam()
    model = pixelnet_model(nclasses=nclasses)
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

    # load best model and evaluate
    # sort by epoch -- use with ModelCheckpoint(..., save_best_only=True)
    # file path format should be 'weights.{epoch}...'
    weights_files = glob.glob(os.path.join(model_dir, 'weights*.hdf5'))
    best_weights = sorted(weights_files)[-1]

    # re-instantiate model because of keras requirement that tensors
    # have the same shape at train and test time
    model = pixelnet_model(nclasses=nclasses, inference=True)
    model.load_weights(best_weights)

    for X, y in [(X_train, y_train), (X_val, y_val)]:
        # run with batch_size=1 for inference due to dense feature upsampling
        p_validate = model.predict(X, batch_size=1)
        pred = np.argmax(p_validate, axis=-1)

        # measure accuracy over the whole validation set
        print('accuracy: {}'.format(perf.accuracy(pred, y)))
        print('IU_avg: {}'.format(perf.IU_avg(pred, y)))

        print('IU')
        for c in range(nclasses):
            iu = perf.IU(pred, y, c)
            print('IU({}): {}'.format(c, iu))

if __name__ == '__main__':
    train_pixelnet()

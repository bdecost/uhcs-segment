#!/usr/bin/env python
import os
import numpy as np
import skimage.data
import skimage.color

import tensorflow as tf
from keras import optimizers
from keras.models import Model
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

import sys
sys.path.append(os.getcwd())

from uhcsseg.pixelnet import pixelnet_model
from uhcsseg.io import load_dataset

# suppress some of the noisier tensorflow log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BATCHSIZE = 4
NTRAIN = 20
NPIX = 2048
NCLASSES = 4

def random_training_samples():
    """ generate random samples of pixels in batches of four training images """
    while True:
        # sample coordinates should include the batch index for tf.gather_nd
        coords = np.ones((BATCHSIZE, NPIX, 3))
        coords = coords * np.arange(BATCHSIZE)[:,np.newaxis,np.newaxis]

        # choose random pixel coordinates on (0, 1) interval
        p = np.random.random((BATCHSIZE,NPIX,2))
        coords[:,:,1:] = p

        # choose random batch of training images, with replacement
        im_idx = np.random.choice(range(NTRAIN), BATCHSIZE, replace=True)
        t_ims = images[im_idx]

        # select the corresponding label images
        target_labels = labels[im_idx]
        ind = coords * np.array([1, images.shape[1], images.shape[2]])
        ind = ind.astype(np.int32)

        # get sample pixel labels
        with tf.Session() as s:
            pixel_labels = tf.gather_nd(target_labels, ind).eval()
        # pixel_labels = np.take(target_labels, ind)
        
        # convert labels to categorical indicators for cross-entropy loss
        s = pixel_labels.shape
        pixel_labels = to_categorical(pixel_labels.flat, num_classes=NCLASSES)
        pixel_labels = pixel_labels.reshape((*s, NCLASSES))

        yield ([t_ims, coords], pixel_labels)

def random_validation_samples():
    """ generate random samples of pixels in batches of four validation images """
    while True:
        # sample coordinates should include the batch index for tf.gather_nd
        coords = np.ones((BATCHSIZE, NPIX, 3))
        coords = coords * np.arange(BATCHSIZE)[:,np.newaxis,np.newaxis]

        # choose random pixel coordinates on (0, 1) interval
        p = np.random.random((BATCHSIZE,NPIX,2))
        coords[:,:,1:] = p

        # choose random batch of training images, with replacement
        im_idx = np.random.choice(range(NTRAIN,images.shape[0]), BATCHSIZE, replace=True)
        t_ims = images[im_idx]

        # select the corresponding label images
        target_labels = labels[im_idx]
        ind = coords * np.array([1, images.shape[1], images.shape[2]])
        ind = ind.astype(np.int32)

        # get sample pixel labels
        with tf.Session() as s:
            pixel_labels = tf.gather_nd(target_labels, ind).eval()

        # pixel_labels = np.take(target_labels, ind)

        # convert labels to categorical indicators for cross-entropy loss
        s = pixel_labels.shape
        pixel_labels = to_categorical(pixel_labels.flat, num_classes=NCLASSES)
        pixel_labels = pixel_labels.reshape((*s, NCLASSES))

        yield ([t_ims, coords], pixel_labels)
        
def stratified_training_samples():
    while True:
        coords = np.random.random((1,2048,3))
        coords *= np.array([0, 1, 1])

if __name__ == '__main__':
    datafile = 'data/uhcs.h5'
    images, labels, names = load_dataset(datafile, cropbar=38)

    # normalize dataset
    images = (images - np.mean(images)) / np.std(images)
    images = images[:,:,:,np.newaxis]

    for t in random_training_samples():
        break
    
    N, h, w, _ = images.shape
    # steps_per_epoch = NTRAIN * h * w / (BATCHSIZE*NPIX)
    steps_per_epoch = 2
    print(steps_per_epoch)
    
    opt = optimizers.Adam()
    model = pixelnet_model()
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    csv_logger = CSVLogger('run/training-1.log')
    checkpoint = ModelCheckpoint('run/weights.{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True, save_weights_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    
    f = model.fit_generator(
        random_training_samples(),
        steps_per_epoch,
        epochs=10,
        callbacks=[csv_logger, checkpoint, early_stopping],
        validation_data=random_validation_samples(),
        validation_steps=2
    )
    # t = random_training_samples()
    # v = random_validation_samples()
    # f = model.fit(t
    #     random_training_samples(),
    #     steps_per_epoch,
    #     epochs=10,
    #     callbacks=[csv_logger, checkpoint, early_stopping],
    #     validation_data=random_validation_samples(),
    #     validation_steps=10
    # )

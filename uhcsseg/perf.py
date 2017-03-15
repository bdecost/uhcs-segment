""" segmentation performance metrics """

import numpy as np

def accuracy(prediction, annotation):
    """ pixel-wise accuracy (percentage) """
    return 100 * (np.sum(prediction == annotation) / annotation.size)

def IU(prediction, annotation, cls):
    """ intersection over union accuracy for class == cls """
    numerator = np.sum(np.logical_and((prediction == cls), (annotation == cls)))
    denominator = np.sum(np.logical_or((prediction == cls), (annotation == cls)))
    return 100 * (numerator / denominator)

def IU_avg(prediction, annotation, classlist=None):
    """ IU averaged over all classes """
    if classlist is None:
        classlist = np.unique(np.concatenate((prediction, annotation)))
    avg_IU =  np.mean(list(
        IU(prediction, annotation, cls) for cls in classlist
    ))
    return avg_IU

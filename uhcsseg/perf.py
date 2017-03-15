""" segmentation performance metrics """

import numpy as np
import scipy as sp
from skimage.measure import label, regionprops

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

def PSD(mask):
    """ calculate empirical particle size distribution (in pixels) """
    if np.unique(mask).size > 2:
        raise NotImplementedError("PSD expects a binary segmentation mask")
    return np.sqrt([region.area for region in regionprops(label(mask))])

def ks_PSD(prediction, annotation):
    """ Calculate particle size distributions and perform 2-sample ks test.
        Metric is the ks test p-value
        p < 0.05 -- reject 2-tailed null hypothesis --  distributions differ
    """
    ks = sp.stats.ks_2samp(PSD(annotation), PSD(prediction))
    return ks.pvalue

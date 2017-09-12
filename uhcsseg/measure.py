""" measure microstructure features """
from __future__ import division

import numpy as np
import scipy as sp
from scipy import ndimage as ndi

from skimage import morphology
from skimage import segmentation
from skimage import measure as skmeasure

from uhcsseg import data

M = data.Microconstituent

def fill_network(labels, min_size=128):
    network = labels == M.network
    network = morphology.remove_small_holes(network, min_size=min_size)
    labels[network] = M.network
    return labels
    
def remove_disconnected_matrix(labels, watershed=False):
    """ remove matrix phase disconnected from network """
    def preprocess_matrix(labels):
        
        matrix = labels == M.matrix
        matrix = morphology.binary_closing(matrix)
        # matrix = morphology.remove_small_holes(matrix, min_size=128)
        return matrix
    
    def preprocess_matrix_new(labels):
        mask = labels.copy()
        mask[...] = M.spheroidite
        
        matrix = labels == M.matrix
        network = labels == M.network
        
        matrix = morphology.binary_closing(matrix)
        network = morphology.binary_dilation(network)
        network = morphology.binary_dilation(network)

        mask[matrix] = M.matrix
        mask[network] = M.network
        return labels == M.matrix

    matrix = preprocess_matrix(labels)
        
    segments = skmeasure.label(matrix)
    
    reduced = labels.copy()

    for segment in np.unique(segments):
        if segment == 0: continue
        mask = segments == segment

        if np.any(labels[mask] == M.network):
            continue

        reduced[mask] = M.spheroidite

    return reduced

def remove_widmanstatten(labels):
    """ mark widmanstatten as particle matrix phase """
    labels[labels == M.widmanstatten] = M.spheroidite
    return labels

def widmanstatten_cleanup_heuristic(labels):
    """ Mark any matrix pixels closer to widmanstatten than network as spheroidite """
    matrix = labels == M.matrix
    d_network = ndi.distance_transform_edt(labels != M.network)
    d_widmanstatten = ndi.distance_transform_edt(labels != M.widmanstatten)
    
    labels[(d_widmanstatten < d_network) & matrix] = M.spheroidite
    
    return labels

def network_cleanup(labels):
    L = fill_network(labels)
    L = remove_disconnected_matrix(L)
    L = widmanstatten_cleanup_heuristic(L)
    L = remove_widmanstatten(L)
    return L

def denuded_zone_widths(labels):
    """ compute denuded zone widths by finding network and particle/matrix boundaries
    and calculating minimum distance to network from each interface pixel """

    L = network_cleanup(labels)

    matrix_bounds = segmentation.find_boundaries(L == M.spheroidite)
    network_bounds = segmentation.find_boundaries(L == M.network)

    # compute distance transform to network phase
    # computes distance to background (with binary input)
    d = ndi.distance_transform_edt(L != M.network)

    denuded_zone_widths = d[matrix_bounds]
    return denuded_zone_widths

import h5py
import numpy as np
import matplotlib.pyplot as plt

from skimage.color import label2rgb
from skimage.segmentation import mark_boundaries
import seaborn as sns
sns.set(style='white')
from seaborn import xkcd_palette

import data

def colorize_labels(lab, mark=False):
    # similar to default matplotlib colors
    # colors = ['blue', 'aqua', 'goldenrod', 'dark red', 'purple']
    colors = ["cobalt", "aqua", "amber", "true green", "dark red"]
    
    colors = colors[np.min(lab):]
    colors = xkcd_palette(colors)
    c = label2rgb(lab, colors=colors)
    if mark:
        c = mark_boundaries(c, label_img=lab, color=(0,0,0), mode='inner')
    return c
    
def plot_crossval_predictions(datafile, resultsfile, cropbar=None):

    fig, axes = plt.subplots(3,4, figsize=(16,9))
    first = True

    with h5py.File(resultsfile) as f:
        keys = list(f.keys())
        probs = [f[k][...] for k in keys]
    
    for ax, p, k in zip(axes.T, probs, keys):
        with h5py.File(datafile, 'r') as f:
            i, l = data.load_record(f, str(k), cropbar=cropbar)
        
        pl = np.argmax(p, axis=-1)
        ax[0].imshow(i, cmap='gray')
        ax[1].imshow(colorize_labels(l))
        ax[2].imshow(colorize_labels(pl))
    
        if first:
            axlabels = ('input', 'annotation', 'prediction')
            for a, label in zip(ax, axlabels):
                a.set_ylabel(label, size=24)
            first = False
        
    for a in axes.flat:
        a.get_xaxis().set_ticks([])
        a.get_yaxis().set_ticks([])
    
    plt.tight_layout()

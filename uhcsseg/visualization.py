import h5py
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from sklearn.decomposition import PCA

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

def colorize_pca_tensor(x):
    """ map the PCA pixel features into the range [0,1] to reinterpret as RGB values """
    # center and normalize each channel
    # this is sort of like PCA whitening...
    x = x - np.mean(x, axis=0) / np.std(x, axis=0)
    
    # rescale to roughly [-1, 1]
    x = x / np.max(np.abs(x))
    
    # shift and clip to [0, 1]
    return np.clip((x + 1) / 2, 0, 1)

def channel_pca(base_model, layername, input_data, output_channels=3, batch_size=1, colorize=False):

    model = Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(layername).output
    )
    
    y = model.predict(input_data, batch_size=batch_size)
    
    b, h, w, c = y.shape
    features = y.reshape(-1, c)
    
    if features.shape[0] > 1e5:
        idx = np.random.choice(features.shape[0], size=100000)
        pca = PCA(n_components=output_channels, whiten=True).fit(features[idx])
    else:
        pca = PCA(n_components=output_channels, whiten=True).fit(features)
        
    x = pca.transform(features)

    if colorize and output_channels == 3:
        x = colorize_pca_tensor(x)

    return x.reshape((b, h, w, output_channels))

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

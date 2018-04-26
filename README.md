uhcs-segment
Code and data for UHCS segmentation tasks

``` sh
git clone hippolyta:/home/holmlab/code/uhcs-segment
cd uhcs-segment
scp -r 'hippolyta:/home/holmlab/microstructure-datasets/uhcs-segment/*' ./
```


```
transfer_pix.py: train
forward_pix.py: forward pass for training and validation sets.
evaluate_results.sh: compute validation set performance.
plot.py ... predictions
plot.py ... overlay

```

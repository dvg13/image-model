# image-model

### Introduction

I created this code for a project to convert depth maps of faces into realistic images of faces.  More details on the project can be found [here](https://github.com/dvg13/image-model/blob/master/deep_frankenstein.md)

This implements an image-to-image GAN, with the option of adding L1 loss from supervised pairs or a reconstruction of the source image.  This code was most influenced by Pix2Pix (https://phillipi.github.io/pix2pix/) and Apple's paper on upgrading synthetic images (https://arxiv.org/pdf/1612.07828.pdf).  There is a separate implementation of a model for a semi-supervised approach, inspired by (https://arxiv.org/abs/1611.02200).

### Prerequisites
- python 3
- tensorflow 1.0

### Basic Usage

The basic usage to train an image-to-image GAN model is:
```
python train.py <FLAGS>
```
The basic flags are:
```
--synth_directory : Directory to synthetic (or first set of images)
--real_directory : Directory to real (or second set of images)
--dlr (defualt 1e-4) : The learning rate for the discriminator
--glr (default 1e-3) : The learning rate for the generator
```
There are also flags to adjust the loss measure for the GAN:
```
--L2gan : Use L2 loss instead of log loss
--wgan : use the scheme put forth in the wasserstein gan paper (This is an abbreviated version that I did not get good results with - though the discriminator loss curve was smooth and converged)
```
I had a lot of success by adding experience replay - which can be accomplished with the following flags:
```
--use_replay
--cache_path : the path to the cache file (which is a memory mapped numpy array
--reuse_replay (default False) : load the cache existing at the specified location
--cache_size (default 100,000) : the number of images to store in the cache
```
### Adding L1 Loss:

The code currently supports adding L1 loss in two ways - from supervised data and from reconstruction error.  The image is reconstructed with an model identical to the generator/refiner, but the weights are not shared.  I plan to add L1 loss between the synthetic image and the generated images as well.


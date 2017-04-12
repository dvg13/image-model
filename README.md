# image-model

### Introduction

I created this code for a project to convert depth models of faces into realistic images of faces.  Largely, this implements
a GAN model, with the option of adding L1 loss from supervised pairs, or reconstruction loss.  This code was most influenced by Pix2Pix (https://phillipi.github.io/pix2pix/) and Apple's paper on upgrading synthetic images (https://arxiv.org/pdf/1612.07828.pdf).  There is a separate implementation of a model for a semi-supervised approach, inspired by (https://arxiv.org/abs/1611.02200).

### Prerequisites
- python 3
- tensorflow 1.0

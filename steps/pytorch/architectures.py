"""
The goal of this file is to collect functions
that serve as building blocks for neural network architectures.
For example: Unet is composed from several definitions,
so that you can configure it in any way you want.
"""
from torch import nn


def build_unet_features(**kwargs):
    return nn.Sequential()


def build_unet_classifier(**kwargs):
    return nn.Sequential()


def _unet_downward_block():
    # ends after maxpool
    pass


def _unet_upward_block():
    # ends after up-conv
    pass

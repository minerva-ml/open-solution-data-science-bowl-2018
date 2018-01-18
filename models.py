import numpy as np
from sklearn.externals import joblib

from steps.base import BaseTransformer
from steps.pytorch.models import PyTorchBasic

from steps.pytorch.architectures import build_unet_features, build_unet_classifier


class MockModel(BaseTransformer):
    def __init__(self, **kwargs):
        super().__init__()

    def fit(self, datagen, validation_datagen, **kwargs):
        return self

    def transform(self, datagen, **kwargs):
        X, steps = datagen
        masks = np.zeros((X.shape[0], 256, 256))
        masks[:, 50:200, 50:200] = 1
        return {'predicted_masks': masks}

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)


class Unet(PyTorchBasic):
    def __init__(self,
                 image_width, image_height,
                 kernel, stride, padding,
                 nonlinearity,
                 repeat_blocks,
                 n_filters,
                 batch_norm,
                 dropout):
        super(Unet, self).__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.nonlinearity = nonlinearity
        self.repeat_blocks = repeat_blocks
        self.n_filters = n_filters
        self.batch_norm = batch_norm
        self.dropout = dropout

        # main part of the U-Net
        self.features = build_unet_features()
        self.classifier = build_unet_classifier()

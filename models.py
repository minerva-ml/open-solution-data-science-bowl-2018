from functools import partial

import numpy as np
from sklearn.externals import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from steps.base import BaseTransformer
from steps.pytorch.callbacks import CallbackList, TrainingMonitor, ValidationMonitor, ModelCheckpoint, \
    NeptuneMonitorSegmentation, ExperimentTiming, ExponentialLRScheduler
from steps.pytorch.models import Model, PyTorchBasic
from steps.pytorch.architectures.unet import UNet


class UnetModel(Model):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.model = UNet(**architecture_config['model_params'])
        self.weight_regularization = weight_regularization
        self.optimizer = optim.Adam(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                   **architecture_config['optimizer_params'])
        self.loss_function = nn.BCELoss()
        self.callbacks = build_callbacks_classifier(self.callbacks_config)

    def transform(self, datagen, validation_datagen=None):
        prediction_proba = self._transform(datagen, validation_datagen)
        prediction_proba_ = [np.squeeze(mask) for mask in prediction_proba]
        return {'predicted_masks': np.array(prediction_proba_)}

class MockModel(BaseTransformer):
    def __init__(self, **kwargs):
        super().__init__()

    def fit(self, datagen, validation_datagen, **kwargs):
        return self

    def transform(self, datagen, **kwargs):
        X, steps = datagen
        masks = np.ones((X.shape[0], 256, 256))
        return {'predicted_masks': masks}

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)


class LoaderTestModel(Model):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.model = PyTorchLoaderTest(**architecture_config['model_params'])
        self.weight_regularization = weight_regularization
        self.optimizer = optim.Adam(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                   **architecture_config['optimizer_params'])
        self.loss_function = nn.BCELoss()
        self.callbacks = build_callbacks_classifier(self.callbacks_config)

    def transform(self, datagen, validation_datagen=None):
        prediction_proba = self._transform(datagen, validation_datagen)
        prediction_proba_ = [np.squeeze(mask) for mask in prediction_proba]
        return {'predicted_masks': np.array(prediction_proba_)}

# class Unet(PyTorchBasic):
#     def __init__(self,
#                  image_width, image_height,
#                  kernel, stride, padding,
#                  nonlinearity,
#                  repeat_blocks,
#                  n_filters,
#                  batch_norm,
#                  dropout):
#         super(Unet, self).__init__()
#         self.image_width = image_width
#         self.image_height = image_height
#         self.kernel = kernel
#         self.stride = stride
#         self.padding = padding
#         self.nonlinearity = nonlinearity
#         self.repeat_blocks = repeat_blocks
#         self.n_filters = n_filters
#         self.batch_norm = batch_norm
#         self.dropout = dropout
#
#         # main part of the U-Net
#         self.features = build_unet_features()
#         self.classifier = build_unet_classifier()
#
#     def forward(self, x):
#         features = self.features(x)
#         out = self.classifier(features)
#         return out


class PyTorchLoaderTest(PyTorchBasic):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        out = self.classifier(features)
        return out



def weight_regularization(model, regularize, weight_decay_conv2d, weight_decay_linear):
    if regularize:
        parameter_list = [{'params': model.features.parameters(), 'weight_decay': weight_decay_conv2d},
                          {'params': model.classifier.parameters(), 'weight_decay': weight_decay_linear},
                          ]
    else:
        parameter_list = model.parameters()
    return parameter_list


def build_callbacks_classifier(callbacks_config):
    experiment_timing = ExperimentTiming()
    model_checkpoints = ModelCheckpoint(**callbacks_config['model_checkpoint'])
    lr_scheduler = ExponentialLRScheduler(**callbacks_config['lr_scheduler'])
    validation_monitor = ValidationMonitor(**callbacks_config['validation_monitor'])
    training_monitor = TrainingMonitor(**callbacks_config['training_monitor'])
    neptune_monitor = NeptuneMonitorSegmentation()

    return CallbackList(
        callbacks=[experiment_timing, model_checkpoints, lr_scheduler, training_monitor, validation_monitor,
                   neptune_monitor])

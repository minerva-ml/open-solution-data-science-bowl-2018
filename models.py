from functools import partial

import numpy as np
from sklearn.externals import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from steps.base import BaseTransformer
from steps.pytorch.callbacks import CallbackList, TrainingMonitor, ValidationMonitor, ModelCheckpoint, \
    NeptuneMonitor, ExperimentTiming, ExponentialLRScheduler
from steps.pytorch.models import Model, PyTorchBasic
from steps.pytorch.validation import mse


class LoaderTestModel(Model):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.model = PyTorchLoaderTest(**architecture_config['model_params'])
        self.weight_regularization = weight_regularization
        self.optimizer = optim.SGD(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                   **architecture_config['optimizer_params'])
        self.loss_function = partial(mse, squeeze=True)
        self.callbacks = build_callbacks_classifier(self.callbacks_config)

    def transform(self, datagen, validation_datagen=None):
        prediction_proba = self._transform(datagen, validation_datagen)
        return {'predicted_masks': np.array(prediction_proba)}


class PyTorchLoaderTest(PyTorchBasic):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Sigmoid()
        )


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
    neptune_monitor = NeptuneMonitor()

    return CallbackList(
        callbacks=[experiment_timing, model_checkpoints, lr_scheduler, training_monitor, validation_monitor,
                   neptune_monitor])


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

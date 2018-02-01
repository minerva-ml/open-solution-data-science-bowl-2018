import numpy as np
import torch.nn as nn
import torch.optim as optim

from steps.pytorch.architectures.unet import UNet
from steps.pytorch.callbacks import CallbackList, TrainingMonitor, ValidationMonitor, ModelCheckpoint, \
    NeptuneMonitorSegmentation, ExperimentTiming, ExponentialLRScheduler, EarlyStopping
from steps.pytorch.models import Model, PyTorchBasic
from steps.pytorch.validation import segmentation_loss
from utils import sigmoid


class PyTorchUNet(Model):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.model = UNet(**architecture_config['model_params'])
        self.weight_regularization = weight_regularization_unet
        self.optimizer = optim.Adam(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                    **architecture_config['optimizer_params'])
        self.loss_function = segmentation_loss
        self.callbacks = build_callbacks(self.callbacks_config)

    def transform(self, datagen, validation_datagen=None):
        prediction_proba = self._transform(datagen, validation_datagen)
        prediction_proba_ = [sigmoid(np.squeeze(mask)) for mask in prediction_proba]
        return {'predicted_masks': np.array(prediction_proba_)}


class SequentialConvNet(Model):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.model = PyTorchSequentialConvNet(**architecture_config['model_params'])
        self.weight_regularization = weight_regularization
        self.optimizer = optim.Adam(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                    **architecture_config['optimizer_params'])
        self.loss_function = nn.BCELoss()
        self.callbacks = build_callbacks(self.callbacks_config)

    def transform(self, datagen, validation_datagen=None):
        prediction_proba = self._transform(datagen, validation_datagen)
        prediction_proba_ = [np.squeeze(mask) for mask in prediction_proba]
        return {'predicted_masks': np.array(prediction_proba_)}


class PyTorchSequentialConvNet(PyTorchBasic):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding=0),
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
        parameter_list = [model.parameters()]
    return parameter_list


def weight_regularization_unet(model, regularize, weight_decay_conv2d):
    if regularize:
        parameter_list = [{'params': model.parameters(), 'weight_decay': weight_decay_conv2d},
                          ]
    else:
        parameter_list = [model.parameters()]
    return parameter_list


def build_callbacks(callbacks_config):
    experiment_timing = ExperimentTiming()
    model_checkpoints = ModelCheckpoint(**callbacks_config['model_checkpoint'])
    lr_scheduler = ExponentialLRScheduler(**callbacks_config['lr_scheduler'])
    training_monitor = TrainingMonitor(**callbacks_config['training_monitor'])
    validation_monitor = ValidationMonitor(**callbacks_config['validation_monitor'])
    neptune_monitor = NeptuneMonitorSegmentation(**callbacks_config['neptune_monitor'])
    early_stopping = EarlyStopping(**callbacks_config['early_stopping'])

    return CallbackList(
        callbacks=[experiment_timing, model_checkpoints, lr_scheduler, training_monitor, validation_monitor,
                   neptune_monitor, early_stopping])

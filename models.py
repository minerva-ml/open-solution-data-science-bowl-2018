import numpy as np
import torch.nn as nn
import torch.optim as optim

from steps.pytorch.architectures.unet import UNet, UNetMultitask
from steps.pytorch.callbacks import CallbackList, TrainingMonitor, ValidationMonitor, ModelCheckpoint, \
    NeptuneMonitorSegmentation, ExperimentTiming, ExponentialLRScheduler, EarlyStopping, NeptuneMonitor
from steps.pytorch.models import Model, ModelMultitask
from steps.pytorch.validation import segmentation_loss, segmentation_loss_multitask
from utils import sigmoid


class PyTorchUNet(Model):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.model = UNet(**architecture_config['model_params'])
        self.weight_regularization = weight_regularization_unet
        self.optimizer = optim.Adam(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                    **architecture_config['optimizer_params'])
        self.loss_function = segmentation_loss
        self.callbacks = callbacks_unet(self.callbacks_config)

    def transform(self, datagen, validation_datagen=None):
        prediction_proba = self._transform(datagen, validation_datagen)
        prediction_proba_ = [sigmoid(np.squeeze(mask)) for mask in prediction_proba]
        return {'predicted_masks': np.array(prediction_proba_)}


class PyTorchUNetMultitask(ModelMultitask):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.model = UNetMultitask(**architecture_config['model_params'])
        self.weight_regularization = weight_regularization_unet
        self.optimizer = optim.Adam(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                    **architecture_config['optimizer_params'])
        self.loss_functions = [('mask_loss', segmentation_loss),
                               ('contour_loss', segmentation_loss),
                               ('center_loss', segmentation_loss)]
        self.callbacks = callbacks_unet_multitask(self.callbacks_config)

    def transform(self, datagen, validation_datagen=None):
        prediction_proba = self._transform(datagen, validation_datagen)
        prediction_proba_ = [sigmoid(np.squeeze(mask)) for mask in prediction_proba]
        return {'predicted_masks': np.array(prediction_proba_)}


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


def callbacks_unet(callbacks_config):
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


def callbacks_unet_multitask(callbacks_config):
    experiment_timing = ExperimentTiming()
    model_checkpoints = ModelCheckpoint(**callbacks_config['model_checkpoint'])
    lr_scheduler = ExponentialLRScheduler(**callbacks_config['lr_scheduler'])
    training_monitor = TrainingMonitor(**callbacks_config['training_monitor'])
    validation_monitor = ValidationMonitor(**callbacks_config['validation_monitor'])
    neptune_monitor = NeptuneMonitor()
    early_stopping = EarlyStopping(**callbacks_config['early_stopping'])

    return CallbackList(
        callbacks=[experiment_timing, model_checkpoints, lr_scheduler, training_monitor, validation_monitor,
                   neptune_monitor, early_stopping])

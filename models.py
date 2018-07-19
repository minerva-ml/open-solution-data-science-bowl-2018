from functools import partial

import numpy as np
from torch import optim

from callbacks import NeptuneMonitorSegmentation
from steps.pytorch.architectures.unet import UNet, UNetMultitask
from steps.pytorch.callbacks import CallbackList, TrainingMonitor, ValidationMonitor, ModelCheckpoint, \
    ExperimentTiming, ExponentialLRScheduler, EarlyStopping
from steps.pytorch.models import Model
from steps.pytorch.validation import segmentation_loss
from utils import sigmoid


class PyTorchUNet(Model):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.model = UNet(**architecture_config['model_params'])
        self.weight_regularization = weight_regularization_unet
        self.optimizer = optim.Adam(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                    **architecture_config['optimizer_params'])
        self.loss_function = [('mask', segmentation_loss, 1.0)]
        self.callbacks = callbacks_unet(self.callbacks_config)

    def transform(self, datagen, validation_datagen=None):
        outputs = self._transform(datagen, validation_datagen)
        for name, prediction in outputs.items():
            prediction_ = [sigmoid(np.squeeze(mask)) for mask in prediction]
            outputs[name] = np.array(prediction_)
        return outputs


class PyTorchUNetMultitask(Model):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.model = UNetMultitask(**architecture_config['model_params'])
        self.weight_regularization = weight_regularization_unet
        self.optimizer = optim.Adam(self.weight_regularization(self.model,
                                                               **architecture_config['regularizer_params']),
                                    **architecture_config['optimizer_params'])

        mask_loss = partial(segmentation_loss,
                            weight_bce=architecture_config['loss_weights']['bce_mask'],
                            weight_dice=architecture_config['loss_weights']['dice_mask'])
        contour_loss = partial(segmentation_loss,
                               weight_bce=architecture_config['loss_weights']['bce_contour'],
                               weight_dice=architecture_config['loss_weights']['dice_contour'])
        center_loss = partial(segmentation_loss,
                              weight_bce=architecture_config['loss_weights']['bce_center'],
                              weight_dice=architecture_config['loss_weights']['dice_center'])

        self.loss_function = [('mask', mask_loss, architecture_config['loss_weights']['mask']),
                              ('contour', contour_loss, architecture_config['loss_weights']['contour']),
                              ('center', center_loss, architecture_config['loss_weights']['center'])]
        self.callbacks = callbacks_unet(self.callbacks_config)

    def transform(self, datagen, validation_datagen=None):
        outputs = self._transform(datagen, validation_datagen)
        for name, prediction in outputs.items():
            prediction_ = [sigmoid(np.squeeze(mask)) for mask in prediction]
            outputs[name] = np.array(prediction_)
        return outputs


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
    experiment_timing = ExperimentTiming(**callbacks_config['experiment_timing'])
    model_checkpoints = ModelCheckpoint(**callbacks_config['model_checkpoint'])
    lr_scheduler = ExponentialLRScheduler(**callbacks_config['lr_scheduler'])
    training_monitor = TrainingMonitor(**callbacks_config['training_monitor'])
    validation_monitor = ValidationMonitor(**callbacks_config['validation_monitor'])
    neptune_monitor = NeptuneMonitorSegmentation(**callbacks_config['neptune_monitor'])
    early_stopping = EarlyStopping(**callbacks_config['early_stopping'])

    return CallbackList(
        callbacks=[experiment_timing, training_monitor, validation_monitor,
                   model_checkpoints, lr_scheduler, neptune_monitor, early_stopping])

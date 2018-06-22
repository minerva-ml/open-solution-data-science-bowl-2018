import numpy as np
import torch.optim as optim

from .steppy.pytorch.architectures.unet import UNet, UNetMultitask
from .steppy.pytorch.callbacks import CallbackList, TrainingMonitor, ValidationMonitor, ModelCheckpoint, \
    ExperimentTiming, ExponentialLRScheduler, EarlyStopping
from .steppy.pytorch.models import Model
from .steppy.pytorch.validation import segmentation_loss

from .utils import sigmoid
from .callbacks import NeptuneMonitorSegmentation
from .unet_models import AlbuNet, UNet11, UNetVGG16, UNetResNet

PRETRAINED_NETWORKS = {'VGG11': {'model': UNet11,
                                 'model_config': {'pretrained': True},
                                 'init_weights': False},
                       'VGG16': {'model': UNetVGG16,
                                 'model_config': {'pretrained': True,
                                                  'dropout_2d': 0.0, 'is_deconv': True},
                                 'init_weights': False},
                       'AlbuNet': {'model': AlbuNet,
                                   'model_config': {'pretrained': True, 'is_deconv': True},
                                   'init_weights': False},
                       'ResNet34': {'model': UNetResNet,
                                    'model_config': {'encoder_depth': 34,
                                                     'num_filters': 32, 'dropout_2d': 0.0,
                                                     'pretrained': True, 'is_deconv': True, },
                                    'init_weights': False},
                       'ResNet101': {'model': UNetResNet,
                                     'model_config': {'encoder_depth': 101,
                                                      'num_filters': 32, 'dropout_2d': 0.0,
                                                      'pretrained': True, 'is_deconv': True, },
                                     'init_weights': False},
                       'ResNet152': {'model': UNetResNet,
                                     'model_config': {'encoder_depth': 152,
                                                      'num_filters': 32, 'dropout_2d': 0.0,
                                                      'pretrained': True, 'is_deconv': True, },
                                     'init_weights': False}
                       }


class PyTorchUNet(Model):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.set_model()
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

    def set_model(self):
        encoder = self.architecture_config['model_params']['encoder']
        if encoder == 'from_scratch':
            self.model = UNet(**self.architecture_config['model_params'])
        else:
            config = PRETRAINED_NETWORKS[encoder]
            self.model = config['model'](num_classes=self.architecture_config['model_params']['nr_unet_outputs'],
                                         **config['model_config'])
            self._initialize_model_weights = lambda: None


class PyTorchUNetMultitask(Model):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.model = UNetMultitask(**architecture_config['model_params'])
        self.weight_regularization = weight_regularization_unet
        self.optimizer = optim.Adam(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                    **architecture_config['optimizer_params'])
        self.loss_function = [('mask', segmentation_loss, 0.45),
                              ('contour', segmentation_loss, 0.45),
                              ('contour_touching', segmentation_loss, 0.0),
                              ('center', segmentation_loss, 0.1)]
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

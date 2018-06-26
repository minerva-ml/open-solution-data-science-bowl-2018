import numpy as np
import torch
import torch.optim as optim
from functools import partial

from .steppy.pytorch.architectures.unet import UNet
from .steppy.pytorch.callbacks import CallbackList, TrainingMonitor, ValidationMonitor, ModelCheckpoint, \
    ExperimentTiming, ExponentialLRScheduler, EarlyStopping
from .steppy.pytorch.models import Model
from .steppy.pytorch.validation import multiclass_segmentation_loss, DiceLoss

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
        dice_loss = partial(multiclass_dice_loss,
                            excluded_classes=[0])
        loss_function = partial(mixed_dice_cross_entropy_loss,
                                dice_loss=dice_loss,
                                cross_entropy_loss=multiclass_segmentation_loss)
        self.loss_function = [('mask', loss_function, 1.0)]
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
            self.model = config['model'](num_classes=self.architecture_config['model_params']['out_channels'],
                                         **config['model_config'])
            self._initialize_model_weights = lambda: None


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


def mixed_dice_cross_entropy_loss(output, target, dice_weight=0.5, dice_loss=None,
                                  cross_entropy_weight=0.5, cross_entropy_loss=None, smooth=0,
                                  dice_activation='softmax'):
    target = target[:, 0, :, :].long()
    if cross_entropy_loss is None:
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
    if dice_loss is None:
        dice_loss = multiclass_dice_loss
    return dice_weight * dice_loss(output, target, smooth,
                                   dice_activation) + cross_entropy_weight * cross_entropy_loss(output,
                                                                                                target)


def multiclass_dice_loss(output, target, smooth=0, activation='softmax', excluded_classes=[]):
    """Calculate Dice Loss for multiple class output.

    Args:
        output (torch.Tensor): Model output of shape (N x C x H x W).
        target (torch.Tensor): Target of shape (N x H x W).
        smooth (float, optional): Smoothing factor. Defaults to 0.
        activation (string, optional): Name of the activation function, softmax or sigmoid. Defaults to 'softmax'.
        excluded_classes (list, optional):
            List of excluded classes numbers. Dice Loss won't be calculated
            against these classes. Often used on background when it has separate output class.
            Defaults to [].

    Returns:
        torch.Tensor: Loss value.

    """
    if activation == 'softmax':
        activation_nn = torch.nn.Softmax2d()
    elif activation == 'sigmoid':
        activation_nn = torch.nn.Sigmoid()
    else:
        raise NotImplementedError('only sigmoid and softmax are implemented')

    loss = 0
    dice = DiceLoss(smooth=smooth)
    output = activation_nn(output)
    for class_nr in range(output.size(1)):
        if class_nr in excluded_classes:
            continue
        class_target = (target == class_nr)
        class_target.data = class_target.data.float()
        loss += dice(output[:, class_nr, :, :], class_target)
    return loss

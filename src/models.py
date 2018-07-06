import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from functools import partial

from .steppy.pytorch.architectures.unet import UNet
from .steppy.pytorch.callbacks import CallbackList, TrainingMonitor, ExperimentTiming, ExponentialLRScheduler
from .steppy.pytorch.models import Model
from .steppy.pytorch.validation import multiclass_segmentation_loss, DiceLoss, segmentation_loss

from .utils import sigmoid, softmax, get_list_of_image_predictions
from .callbacks import NeptuneMonitorSegmentation, ValidationMonitorSegmentation, ModelCheckpointSegmentation, \
    EarlyStoppingSegmentation
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
        self.activation_func = self.architecture_config['model_params']['activation']
        self.set_model()
        self.set_loss()
        self.weight_regularization = weight_regularization_unet
        self.optimizer = optim.Adam(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                    **architecture_config['optimizer_params'])
        self.callbacks = callbacks_unet(self.callbacks_config)

    def fit(self, datagen, validation_datagen=None, meta_valid=None):
        self._initialize_model_weights()

        self.model = nn.DataParallel(self.model)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.callbacks.set_params(self, validation_datagen=validation_datagen, meta_valid=meta_valid)
        self.callbacks.on_train_begin()

        batch_gen, steps = datagen
        for epoch_id in range(self.training_config['epochs']):
            self.callbacks.on_epoch_begin()
            for batch_id, data in enumerate(batch_gen):
                self.callbacks.on_batch_begin()
                metrics = self._fit_loop(data)
                self.callbacks.on_batch_end(metrics=metrics)
                if batch_id == steps:
                    break
            self.callbacks.on_epoch_end()
            if self.callbacks.training_break():
                break
        self.callbacks.on_train_end()
        return self

    def transform(self, datagen, validation_datagen=None, *args, **kwargs):
        outputs = self._transform(datagen, validation_datagen)
        for name, prediction in outputs.items():
            if self.activation_func == 'softmax':
                outputs[name] = [softmax(single_prediction, axis=0) for single_prediction in prediction]
            elif self.activation_func == 'sigmoid':
                outputs[name] = [sigmoid(np.squeeze(mask)) for mask in prediction]
            else:
                raise Exception('Only softmax and sigmoid activations are allowed')
        return outputs

    def _transform(self, datagen, validation_datagen=None):
        self.model.eval()

        batch_gen, steps = datagen
        outputs = {}
        for batch_id, data in enumerate(batch_gen):
            if isinstance(data, list):
                X = data[0]
            else:
                X = data

            if torch.cuda.is_available():
                X = Variable(X, volatile=True).cuda()
            else:
                X = Variable(X, volatile=True)
            outputs_batch = self.model(X)
            if len(self.output_names) == 1:
                outputs.setdefault(self.output_names[0], []).append(outputs_batch.data.cpu().numpy())
            else:
                for name, output in zip(self.output_names, outputs_batch):
                    output_ = output.data.cpu().numpy()
                    outputs.setdefault(name, []).append(output_)
            if batch_id == steps:
                break
        self.model.train()
        outputs = {'{}_prediction'.format(name): get_list_of_image_predictions(outputs_) for name, outputs_ in outputs.items()}
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

    def set_loss(self):
        if self.activation_func == 'softmax':
            loss_function = partial(mixed_dice_cross_entropy_loss,
                                    dice_loss=multiclass_dice_loss,
                                    cross_entropy_loss=nn.CrossEntropyLoss(),
                                    dice_activation='softmax')
        elif self.activation_func == 'sigmoid':
            loss_function = partial(mixed_dice_bce_loss,
                                    dice_loss=multiclass_dice_loss,
                                    bce_loss=nn.BCEWithLogitsLoss(),
                                    dice_activation='sigmoid')
        else:
            raise Exception('Only softmax and sigmoid activations are allowed')
        self.loss_function = [('mask', loss_function, 1.0)]


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
    model_checkpoints = ModelCheckpointSegmentation(**callbacks_config['model_checkpoint'])
    lr_scheduler = ExponentialLRScheduler(**callbacks_config['lr_scheduler'])
    training_monitor = TrainingMonitor(**callbacks_config['training_monitor'])
    validation_monitor = ValidationMonitorSegmentation(**callbacks_config['validation_monitor'])
    neptune_monitor = NeptuneMonitorSegmentation(**callbacks_config['neptune_monitor'])
    early_stopping = EarlyStoppingSegmentation(**callbacks_config['early_stopping'])

    return CallbackList(
        callbacks=[experiment_timing, training_monitor, validation_monitor,
                   model_checkpoints, lr_scheduler, neptune_monitor, early_stopping])


def mixed_dice_cross_entropy_loss(output, target, dice_weight=0.5, dice_loss=None,
                                  cross_entropy_weight=0.5, cross_entropy_loss=None, smooth=0,
                                  dice_activation='softmax'):
    num_classes_without_background = output.size(1) - 1
    dice_output = output[:, 1:, :, :]
    dice_target = target[:, :num_classes_without_background, :, :].long()
    cross_entropy_target = torch.zeros_like(target[:, 0, :, :]).long()
    for class_nr in range(num_classes_without_background):
        cross_entropy_target = where(target[:, class_nr, :, :], class_nr + 1, cross_entropy_target)
    if cross_entropy_loss is None:
        cross_entropy_loss = nn.CrossEntropyLoss()
    if dice_loss is None:
        dice_loss = multiclass_dice_loss
    return dice_weight * dice_loss(dice_output, dice_target, smooth,
                                   dice_activation) + cross_entropy_weight * cross_entropy_loss(output,
                                                                                                cross_entropy_target)


def multiclass_dice_loss(output, target, smooth=0, activation='softmax'):
    """Calculate Dice Loss for multiple class output.

    Args:
        output (torch.Tensor): Model output of shape (N x C x H x W).
        target (torch.Tensor): Target of shape (N x H x W).
        smooth (float, optional): Smoothing factor. Defaults to 0.
        activation (string, optional): Name of the activation function, softmax or sigmoid. Defaults to 'softmax'.

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
    num_classes = output.size(1)
    target.data = target.data.float()
    for class_nr in range(num_classes):
        loss += dice(output[:, class_nr, :, :], target[:, class_nr, :, :])
    return loss / num_classes


def mixed_dice_bce_loss(output, target, dice_weight=0.5, dice_loss=None,
                        bce_weight=0.5, bce_loss=None,
                        smooth=0, dice_activation='sigmoid'):
    num_classes = output.size(1)
    target = target[:, :num_classes, :, :].long()
    if bce_loss is None:
        bce_loss = nn.BCEWithLogitsLoss()
    if dice_loss is None:
        dice_loss = multiclass_dice_loss
    return dice_weight * dice_loss(output, target, smooth, dice_activation) + bce_weight * bce_loss(output, target)


def where(cond, x_1, x_2):
    cond = cond.long()
    return (cond * x_1) + ((1-cond) * x_2)

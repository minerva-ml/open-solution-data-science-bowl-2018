import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.externals import joblib

from steps.base import BaseTransformer
from steps.pytorch.callbacks import CallbackList, TrainingMonitor, ValidationMonitor, ModelCheckpoint, \
    NeptuneMonitorSegmentation, ExperimentTiming, ExponentialLRScheduler
from steps.pytorch.models import Model, PyTorchBasic


class PyTorchUNet(Model):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.model = UNet(**architecture_config['model_params'])
        self.weight_regularization = weight_regularization
        self.optimizer = optim.Adam(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                    **architecture_config['optimizer_params'])
        self.loss_function = nn.BCEWithLogitsLoss()
        self.callbacks = build_callbacks_classifier(self.callbacks_config)

    def transform(self, datagen, validation_datagen=None):
        prediction_proba = self._transform(datagen, validation_datagen)
        prediction_proba_ = [np.squeeze(mask) for mask in prediction_proba]
        return {'predicted_masks': np.array(prediction_proba_)}


class SequentialConvNet(Model):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.model = PyTorchSequentialConvNet(**architecture_config['model_params'])
        self.weight_regularization = weight_regularization
        self.optimizer = optim.Adam(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                    **architecture_config['optimizer_params'])
        self.loss_function = nn.BCELoss()
        self.callbacks = build_callbacks_classifier(self.callbacks_config)

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


# dirty U-Net
def pad(size):
    return nn.ReplicationPad2d(size)


def conv(inch, outch):
    return nn.Conv2d(in_channels=inch, out_channels=outch, kernel_size=(3, 3), stride=1, padding=0, bias=True)


def lrelu():
    return nn.LeakyReLU()


def bn(inch):
    return nn.BatchNorm2d(num_features=inch, momentum=0.1, affine=False)


def maxpool():
    return nn.MaxPool2d(kernel_size=(2, 2))


def upconv(inch, outch):
    return nn.ConvTranspose2d(in_channels=inch, out_channels=outch, kernel_size=(2, 2), stride=2, padding=0,
                              output_padding=0, bias=True)


def convout(inch, outch):
    return nn.Conv2d(in_channels=inch, out_channels=outch, kernel_size=(1, 1), stride=1, padding=0, bias=True)


class UNet(PyTorchBasic):
    def __init__(self):
        super(UNet, self).__init__()
        self.d512x512 = nn.Sequential(pad(1), conv(3, 32), lrelu(), bn(32), pad(1), conv(32, 32), lrelu())
        self.d256x256 = nn.Sequential(bn(32), pad(1), conv(32, 64), lrelu(), maxpool(), bn(64), pad(1), conv(64, 64),
                                      lrelu())
        self.d128x128 = nn.Sequential(bn(64), pad(1), conv(64, 64), lrelu(), maxpool(), bn(64), pad(1), conv(64, 64),
                                      lrelu())
        self.d64x64 = nn.Sequential(bn(64), pad(1), conv(64, 128), lrelu(), maxpool(), bn(128), pad(1), conv(128, 128),
                                    lrelu())
        self.d32x32 = nn.Sequential(bn(128), pad(1), conv(128, 128), lrelu(), maxpool(), bn(128), pad(1),
                                    conv(128, 128), lrelu())
        self.d16x16 = nn.Sequential(bn(128), pad(1), conv(128, 128), lrelu(), maxpool(), bn(128), pad(1),
                                    conv(128, 128), lrelu())
        self.u16x16 = nn.Sequential(bn(128), pad(1), conv(128, 128), lrelu(), maxpool(), bn(128), pad(1),
                                    conv(128, 128), lrelu(), bn(128), upconv(128, 128), lrelu())
        self.u32x32 = nn.Sequential(bn(256), pad(1), conv(256, 256), lrelu(), bn(256), upconv(256, 256), lrelu(),
                                    bn(256), pad(1), conv(256, 128), lrelu())
        self.u64x64 = nn.Sequential(bn(256), pad(1), conv(256, 256), lrelu(), bn(256), upconv(256, 256), lrelu(),
                                    bn(256), pad(1), conv(256, 128), lrelu())
        self.u128x128 = nn.Sequential(bn(256), pad(1), conv(256, 128), lrelu(), bn(128), upconv(128, 128), lrelu(),
                                      bn(128), pad(1), conv(128, 96), lrelu())
        self.u256x256 = nn.Sequential(bn(160), pad(1), conv(160, 128), lrelu(), bn(128), upconv(128, 128), lrelu(),
                                      bn(128), pad(1), conv(128, 64), lrelu())
        self.u512x512 = nn.Sequential(bn(128), pad(1), conv(128, 64), lrelu(), bn(64), upconv(64, 64), lrelu(), bn(64),
                                      pad(1), conv(64, 32), lrelu())
        self.classifier = nn.Sequential(bn(64), pad(1), conv(64, 32), lrelu(), convout(32, 1))

    def forward(self, input_x):
        d512x512 = self.d512x512(input_x)
        d256x256 = self.d256x256(d512x512)
        d128x128 = self.d128x128(d256x256)
        d64x64 = self.d64x64(d128x128)
        d32x32 = self.d32x32(d64x64)
        d16x16 = self.d16x16(d32x32)
        u16x16 = self.u16x16(d16x16)
        u32x32 = self.u32x32(torch.cat((d16x16, u16x16), dim=1))
        u64x64 = self.u64x64(torch.cat((d32x32, u32x32), dim=1))
        u128x128 = self.u128x128(torch.cat((d64x64, u64x64), dim=1))
        u256x256 = self.u256x256(torch.cat((d128x128, u128x128), dim=1))
        u512x512 = self.u512x512(torch.cat((d256x256, u256x256), dim=1))
        scores = self.classifier(torch.cat((d512x512, u512x512), dim=1))
        return scores

import numpy as np
import torch.optim as optim
import torch

from steps.pytorch.architectures.unet import UNet, UNetMultitask, DCAN
from steps.pytorch.callbacks import CallbackList, TrainingMonitor, ValidationMonitor, ModelCheckpoint, \
    ExperimentTiming, ExponentialLRScheduler, EarlyStopping, LossWeightsScheduler
from steps.pytorch.models import Model
from steps.pytorch.validation import segmentation_loss, list_segmentation_loss
from utils import sigmoid
from callbacks import NeptuneMonitorSegmentation


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
        self.optimizer = optim.Adam(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                    **architecture_config['optimizer_params'])
        self.loss_function = [('mask', segmentation_loss, architecture_config['loss_weights']['mask']),
                              ('contour', segmentation_loss, architecture_config['loss_weights']['contour']),
                              ('contour_touching', segmentation_loss,
                               architecture_config['loss_weights']['contour_touching']),
                              ('center', segmentation_loss, architecture_config['loss_weights']['center'])]
        self.callbacks = callbacks_unet(self.callbacks_config)

    def transform(self, datagen, validation_datagen=None):
        outputs = self._transform(datagen, validation_datagen)
        for name, prediction in outputs.items():
            prediction_ = [sigmoid(np.squeeze(mask)) for mask in prediction]
            outputs[name] = np.array(prediction_)
        return outputs


class PyTorchDCAN(Model):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.model = DCAN(**architecture_config['model_params'])
        self.weight_regularization = weight_regularization_unet
        self.optimizer = optim.Adam(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                    **architecture_config['optimizer_params'])
        self.loss_function = [('mask', segmentation_loss, architecture_config['loss_weights']['mask']),
                              ('contour', segmentation_loss, architecture_config['loss_weights']['contour']),
                              ('mask_auxiliary_classifiers', list_segmentation_loss,
                               architecture_config['loss_weights']['mask_auxiliary_classifiers']),
                              ('contour_auxiliary_classifiers', list_segmentation_loss,
                               architecture_config['loss_weights']['contour_auxiliary_classifiers'])]
        self.callbacks = callbacks_dcan(self.callbacks_config)

    def transform(self, datagen, validation_datagen=None):
        outputs = self._transform(datagen, validation_datagen)
        for name, prediction in outputs.items():
            if "auxiliary" in name:
                continue
            prediction_ = [sigmoid(np.squeeze(mask)) for mask in prediction]
            outputs[name] = np.array(prediction_)
        return outputs

    def _transform(self, datagen, validation_datagen=None):
        self.model.eval()
        batch_gen, steps = datagen
        outputs = {}
        for batch_id, data in enumerate(batch_gen):
            X = data[0]

            if torch.cuda.is_available():
                X = torch.autograd.Variable(X, volatile=True).cuda()
            else:
                X = torch.autograd.Variable(X, volatile=True)

            outputs_batch = self.model(X)
            for name, output in zip(self.output_names, outputs_batch):
                if "auxiliary" in name:
                    continue
                output_ = output.data.cpu().numpy()
                outputs.setdefault(name, []).append(output_)
            if batch_id == steps:
                break
        outputs = {'{}_prediction'.format(name): np.vstack(outputs_) for name, outputs_ in outputs.items() if "auxiliary" not in name}
        return outputs

    def _fit_loop(self, data):
        X = data[0]
        targets_tensors = data[1:]

        if torch.cuda.is_available():
            X = torch.autograd.Variable(X).cuda()
            targets_var = []
            for target_tensor in targets_tensors:
                targets_var.append(torch.autograd.Variable(target_tensor).cuda())
        else:
            X = torch.autograd.Variable(X)
            targets_var = []
            for target_tensor in targets_tensors:
                targets_var.append(torch.autograd.Variable(target_tensor))

        self.optimizer.zero_grad()
        outputs_batch = self.model(X)
        partial_batch_losses = {}

        for (name, loss_function, weight), output, target in zip(self.loss_function, outputs_batch, 2*targets_var[:2]):
            partial_batch_losses[name] = loss_function(output, target) * weight

        batch_loss = sum(partial_batch_losses.values())
        partial_batch_losses['sum'] = batch_loss
        batch_loss.backward()
        self.optimizer.step()

        return partial_batch_losses


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
    lr_scheduler = ExponentialLRScheduler(**callbacks_config['exp_lr_scheduler'])
    training_monitor = TrainingMonitor(**callbacks_config['training_monitor'])
    validation_monitor = ValidationMonitor(**callbacks_config['validation_monitor'])
    neptune_monitor = NeptuneMonitorSegmentation(**callbacks_config['neptune_monitor'])
    early_stopping = EarlyStopping(**callbacks_config['early_stopping'])

    return CallbackList(
        callbacks=[experiment_timing, training_monitor, validation_monitor,
                   model_checkpoints, lr_scheduler, neptune_monitor, early_stopping
                   ])


def callbacks_dcan(callbacks_config):
    experiment_timing = ExperimentTiming(**callbacks_config['experiment_timing'])
    model_checkpoints = ModelCheckpoint(**callbacks_config['model_checkpoint'])
    lr_scheduler = ExponentialLRScheduler(**callbacks_config['exp_lr_scheduler'])
    training_monitor = TrainingMonitor(**callbacks_config['training_monitor'])
    validation_monitor = ValidationMonitor(**callbacks_config['validation_monitor'])
    neptune_monitor = NeptuneMonitorSegmentation(**callbacks_config['neptune_monitor'])
    early_stopping = EarlyStopping(**callbacks_config['early_stopping'])
    lw_scheduler = LossWeightsScheduler(**callbacks_config['loss_weights_scheduler'])

    return CallbackList(
        callbacks=[experiment_timing, training_monitor, validation_monitor,
                   model_checkpoints, lr_scheduler, neptune_monitor,
                   early_stopping, #lw_scheduler
                   ])

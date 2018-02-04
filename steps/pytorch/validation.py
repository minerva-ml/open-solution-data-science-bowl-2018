import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from .utils import sigmoid


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, output, target):
        prediction = self.sigmoid(output)
        return 1 - 2 * torch.sum(prediction * target) / (torch.sum(prediction) + torch.sum(target) + 1e-7)


def segmentation_loss(output, target):
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    return bce(output, target) + dice(output, target)


def cross_entropy(output, target, squeeze=False):
    if squeeze:
        target = target.squeeze(1)
    return F.nll_loss(output, target)


def mse(output, target, squeeze=False):
    if squeeze:
        target = target.squeeze(1)
    return F.mse_loss(output, target)


def multi_output_cross_entropy(outputs, targets):
    losses = []
    for output, target in zip(outputs, targets):
        loss = cross_entropy(output, target)
        losses.append(loss)
    return sum(losses) / len(losses)


def get_prediction_masks(model, datagen):
    batch_gen, steps = datagen
    for batch_id, data in enumerate(batch_gen):
        X, targets = data

        if torch.cuda.is_available():
            X, targets_var = Variable(X).cuda(), Variable(targets).cuda()
        else:
            X, targets_var = Variable(X), Variable(targets)
        outputs = model(X)

        raw_images = np.mean(X.data.cpu().numpy(), axis=1)
        prediction_masks = sigmoid(np.squeeze(outputs.data.cpu().numpy(), axis=1))
        ground_truth_masks = np.squeeze(targets.cpu().numpy(), axis=1)
        break
    return np.stack([raw_images, prediction_masks, ground_truth_masks], axis=1)


def score_model(model, loss_function, datagen):
    """
    Todo:
    Refactor this ugglyness
    """
    batch_gen, steps = datagen
    partial_batch_losses = {}
    for batch_id, data in enumerate(batch_gen):
        X = data[0]
        targets_tensors = data[1:]

        if torch.cuda.is_available():
            X = Variable(X).cuda()
            targets_var = []
            for target_tensor in targets_tensors:
                targets_var.append(Variable(target_tensor).cuda())
        else:
            X = Variable(X)
            targets_var = []
            for target_tensor in targets_tensors:
                targets_var.append(Variable(target_tensor))

        outputs = model(X)
        for (name, loss_function_one), output, target in zip(loss_function, outputs, targets_var):
            loss = loss_function_one(output, target)
            partial_batch_losses.setdefault(name, []).append(loss)
        loss_sum = sum(partial_batch_losses.values())
        partial_batch_losses.setdefault('loss_sum', []).append(loss_sum)
        if batch_id == steps:
            break

    average_losses = {name: sum(losses) / steps for name, losses in partial_batch_losses.items()}
    return average_losses


def predict_on_batch_multi_output(model, datagen):
    batch_gen, steps = datagen

    for data in batch_gen:
        X, targets = data

        if torch.cuda.is_available():
            X = Variable(X).cuda()
        else:
            X = Variable(X)
        break

    outputs = model(X)
    predictions = []
    for output in outputs:
        prediction = output.data.cpu().numpy().argmax(axis=1)
        predictions.append(prediction)

    predictions = np.stack(predictions, axis=0)
    predictions = predictions.transpose(1, 0)
    images = X.data.cpu().numpy()
    images = images.transpose(0, 2, 3, 1)

    image_list, prediction_list, target_list = [], [], []
    for i in range(images.shape[0]):
        image_list.append(images[i, :, :, :])
        prediction_list.append(predictions[i])
        target_list.append(targets.numpy()[i, :])

    return zip(image_list, prediction_list, target_list)


def torch_acc_score(output, target):
    output = output.data.cpu().numpy()
    y_true = target.numpy()
    y_pred = output.argmax(axis=1)

    return accuracy_score(y_true, y_pred)


def torch_acc_score_multi_output(outputs, targets, take_first=None):
    accuracies = []
    for i, (output, target) in enumerate(zip(outputs, targets)):
        if i == take_first:
            break
        accuracy = torch_acc_score(output, target)
        accuracies.append(accuracy)
    avg_accuracy = sum(accuracies) / len(accuracies)
    return avg_accuracy

import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.autograd import Variable
import torch.nn.functional as F


def cross_entropy(output, target, squeeze=False):
    if squeeze:
        target = target.squeeze(1)
    return F.nll_loss(output, target)


def multi_output_cross_entropy(outputs, targets):
    loss_seq = []
    for output, target in zip(outputs, targets):
        loss = cross_entropy(output, target)
        loss_seq.append(loss)
    return sum(loss_seq) / len(loss_seq)


def score_model(model, loss_function, datagen):
    """
    Todo:
    Refactor this ugglyness
    """
    batch_gen, steps = datagen

    total_loss, total_acc = [], []
    for batch_id, data in enumerate(batch_gen):
        X, targets = data

        if torch.cuda.is_available():
            X, targets_var = Variable(X).cuda(), Variable(targets).cuda()
        else:
            X, targets_var = Variable(X), Variable(targets)
        outputs = model(X)
        batch_loss = loss_function(outputs, targets_var).data.cpu().numpy()[0]
        batch_acc = torch_acc_score(outputs, targets)

        total_loss.append(batch_loss)
        total_acc.append(batch_acc)

        if batch_id == steps:
            break

    avg_loss = sum(total_loss) / steps
    avg_acc = sum(total_acc) / steps
    return avg_loss, avg_acc


def score_model_multi_output(model, loss_function, datagen):
    """
    Todo:
    Refactor this ugglyness
    """
    batch_gen, steps = datagen

    total_loss, total_acc = [], []
    for batch_id, data in enumerate(batch_gen):
        X, targets = data

        targets = targets.transpose(0, 1)

        if torch.cuda.is_available():
            X, targets_var = Variable(X).cuda(), Variable(targets).cuda()
        else:
            X, targets_var = Variable(X), Variable(targets)
        outputs = model(X)
        batch_loss = loss_function(outputs, targets_var).data.cpu().numpy()[0]
        batch_acc = torch_acc_score_multi_output(outputs, targets)

        total_loss.append(batch_loss)
        total_acc.append(batch_acc)

        if batch_id == steps:
            break

    avg_loss = sum(total_loss) / steps
    avg_acc = sum(total_acc) / steps
    return avg_loss, avg_acc


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

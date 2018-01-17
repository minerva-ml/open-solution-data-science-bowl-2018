import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.metrics.pairwise import pairwise_distances


def decompose(mask):
    labeled, nr_true = ndimage.label(mask)
    res = []
    for i in range(1, nr_true + 1):
        msk = labeled.copy()
        msk[msk != i] = 0.
        msk[msk == i] = 255.
        res.append(msk)
    return res


def _iou(gt, pred):
    gt[gt > 0] = 1.
    pred[pred > 0] = 1.
    intersection = gt * pred
    union = gt + pred
    union[union > 0] = 1.
    intersection = np.sum(intersection)
    union = np.sum(union)
    if union == 0:
        union = 1e-09
    return intersection / union


def intersaction_over_union(y_true, y_pred):
    gt_ = decompose(y_true)
    predictions_ = decompose(y_pred)
    gt_ = np.asarray([el.flatten() for el in gt_])
    predictions_ = np.asarray([el.flatten() for el in predictions_])
    ious = pairwise_distances(X=gt_, Y=predictions_, metric=_iou)
    return ious


def compute_precision_at(ious, threshold):
    mx1 = np.max(ious, axis=0)
    mx2 = np.max(ious, axis=1)
    tp = np.sum(mx2 >= threshold)
    fp = np.sum(mx2 < threshold)
    fn = np.sum(mx1 < threshold)
    return float(tp) / (tp + fp + fn)


def intersaction_over_union_thresholds(y_true, y_pred):
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ious = intersaction_over_union(y_true, y_pred)
    precisions = [compute_precision_at(ious, th) for th in thresholds]
    return sum(precisions) / len(precisions)

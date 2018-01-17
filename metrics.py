from tqdm import tqdm

import numpy as np
from scipy import ndimage
from sklearn.metrics.pairwise import pairwise_distances


def decompose(mask):
    labeled, nr_true = ndimage.label(mask)
    masks = []
    for i in range(1, nr_true + 1):
        msk = labeled.copy()
        msk[msk != i] = 0.
        msk[msk == i] = 255.
        masks.append(msk)
    if not masks:
        return [mask]
    else:
        return masks


def iou(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return intersection.sum() / im_sum


def compute_ious(gt, predictions):
    gt_ = decompose(gt)
    predictions_ = decompose(predictions)
    gt_ = np.asarray([el.flatten() for el in gt_])
    predictions_ = np.asarray([el.flatten() for el in predictions_])
    ious = pairwise_distances(X=gt_, Y=predictions_, metric=iou)
    return ious


def compute_precision_at(ious, threshold):
    mx1 = np.max(ious, axis=0)
    mx2 = np.max(ious, axis=1)
    tp = np.sum(mx2 >= threshold)
    fp = np.sum(mx2 < threshold)
    fn = np.sum(mx1 < threshold)
    return float(tp) / (tp + fp + fn)


def compute_eval_metric(gt, predictions):
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ious = compute_ious(gt, predictions)
    precisions = [compute_precision_at(ious, th) for th in thresholds]
    return sum(precisions) / len(precisions)


def intersection_over_union(y_true, y_pred):
    ious = []
    for y_t, y_p in tqdm(list(zip(y_true, y_pred))):
        ious.append(np.mean(compute_ious(y_t, y_p)))
    return np.mean(ious)


def intersection_over_union_thresholds(y_true, y_pred):
    iouts = []
    for y_t, y_p in tqdm(list(zip(y_true, y_pred))):
        iouts.append(compute_eval_metric(y_t, y_p))
    return np.mean(iouts)

import logging
import yaml

import cv2
import numpy as np
import torch


def init_logger():
    logger = logging.getLogger('steps')
    logger.setLevel(logging.INFO)
    message_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s',
                                       datefmt='%Y-%m-%d %H-%M-%S')

    # console handler for validation info
    ch_va = logging.StreamHandler()
    ch_va.setLevel(logging.INFO)

    ch_va.setFormatter(fmt=message_format)

    # add the handlers to the logger
    logger.addHandler(ch_va)


def get_logger():
    return logging.getLogger('steps')


def denormalize_img(img):
    mean = [0.28201905, 0.37246801, 0.42341868]
    std = [0.13609867, 0.12380088, 0.13325344]
    img_ = (img * std) + mean
    return img_


def overlay_box(img, predicted_box, true_box, bin_nr):
    img_h, img_w, img_c = img.shape
    x1, y1, x2, y2 = predicted_box

    x1 = int(1.0 * x1 * img_w / bin_nr)
    y1 = int(1.0 * y1 * img_h / bin_nr)
    x2 = int(1.0 * x2 * img_h / bin_nr)
    y2 = int(1.0 * y2 * img_h / bin_nr)

    tx1, ty1, tx2, ty2 = true_box

    tx1 = int(1.0 * tx1 * img_w / bin_nr)
    ty1 = int(1.0 * ty1 * img_h / bin_nr)
    tx2 = int(1.0 * tx2 * img_h / bin_nr)
    ty2 = int(1.0 * ty2 * img_h / bin_nr)

    img_overlayed = img.copy()
    img_overlayed = (denormalize_img(img_overlayed) * 255.).astype(np.uint8)
    cv2.rectangle(img_overlayed, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.rectangle(img_overlayed, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)
    img_overlayed = (img_overlayed / 255.).astype(np.float64)
    return img_overlayed


def overlay_keypoints(img, pred_keypoints, true_keypoints, bin_nr):
    img_h, img_w, img_c = img.shape
    x1, y1, x2, y2 = pred_keypoints[:4]

    x1 = int(1.0 * x1 * img_w / bin_nr)
    y1 = int(1.0 * y1 * img_h / bin_nr)
    x2 = int(1.0 * x2 * img_h / bin_nr)
    y2 = int(1.0 * y2 * img_h / bin_nr)

    tx1, ty1, tx2, ty2 = true_keypoints[:4]

    tx1 = int(1.0 * tx1 * img_w / bin_nr)
    ty1 = int(1.0 * ty1 * img_h / bin_nr)
    tx2 = int(1.0 * tx2 * img_h / bin_nr)
    ty2 = int(1.0 * ty2 * img_h / bin_nr)

    img_overlayed = img.copy()
    img_overlayed = (denormalize_img(img_overlayed) * 255.).astype(np.uint8)

    cv2.circle(img_overlayed, (x1, y1), 5, (252, 124, 0), -1)
    cv2.circle(img_overlayed, (x2, y2), 5, (139, 46, 87), -1)

    cv2.circle(img_overlayed, (tx1, ty1), 5, (102, 255, 102), -1)
    cv2.circle(img_overlayed, (tx2, ty2), 5, (0, 204, 0), -1)

    img_overlayed = (img_overlayed / 255.).astype(np.float64)
    return img_overlayed


def save_model(model, path):
    model.eval()
    torch.save(model.state_dict(), path)
    model.train()


class Averager:
    """
    Todo:
        Rewrite as a coroutine (yield from)
    """

    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

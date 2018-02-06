import os
import glob

import cv2
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split


def train_valid_split(meta, validation_size):
    meta_train = meta[meta['is_train'] == 1]
    meta_train_split, meta_valid_split = train_test_split(meta_train, test_size=validation_size, random_state=1234)
    return meta_train_split, meta_valid_split

def overlay_masks(images_dir, subdir_name, target_dir):
    train_dir = os.path.join(images_dir, subdir_name)
    for mask_dirname in tqdm(glob.glob('{}/*/masks'.format(train_dir))):
        masks = []
        for image_filepath in glob.glob('{}/*'.format(mask_dirname)):
            masks.append(plt.imread(image_filepath))
        overlayed_masks = np.sum(masks, axis=0)
        target_filepath = '/'.join(mask_dirname.replace(images_dir, target_dir).split('/')[:-1]) + '.png'
        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        plt.imsave(target_filepath, overlayed_masks)


def overlay_contours(images_dir, subdir_name, target_dir):
    train_dir = os.path.join(images_dir, subdir_name)
    for mask_dirname in tqdm(glob.glob('{}/*/masks'.format(train_dir))):
        masks = []
        for image_filepath in glob.glob('{}/*'.format(mask_dirname)):
            image = plt.imread(image_filepath)
            masks.append(get_contour(image))
        overlayed_masks = np.where(np.sum(masks, axis=0) > 128., 255., 0.).astype(np.uint8)
        # print(overlayed_masks.min(), overlayed_masks.max())
        target_filepath = '/'.join(mask_dirname.replace(images_dir, target_dir).split('/')[:-1]) + '.png'
        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        plt.imsave(target_filepath, overlayed_masks)


def overlay_centers(images_dir, subdir_name, target_dir):
    train_dir = os.path.join(images_dir, subdir_name)
    for mask_dirname in tqdm(glob.glob('{}/*/masks'.format(train_dir))):
        masks = []
        for image_filepath in glob.glob('{}/*'.format(mask_dirname)):
            image = plt.imread(image_filepath)
            masks.append(get_center(image))
        overlayed_masks = np.where(np.sum(masks, axis=0) > 128., 255., 0.).astype(np.uint8)
        target_filepath = '/'.join(mask_dirname.replace(images_dir, target_dir).split('/')[:-1]) + '.png'
        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        plt.imsave(target_filepath, overlayed_masks)


def get_contour(img):
    img_contour = np.zeros_like(img).astype(np.uint8)
    _, contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_contour, contours, -1, (255, 255, 255), 4)
    return img_contour


def get_center(img):
    img_center = np.zeros_like(img).astype(np.uint8)
    y, x = ndi.measurements.center_of_mass(img)
    cv2.circle(img_center, (int(x), int(y)), 4, (255, 255, 255), -1)
    return img_center

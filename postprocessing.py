from itertools import product

import numpy as np
import skimage.morphology as morph
from scipy import ndimage as ndi
from scipy.stats import itemfreq
from skimage.transform import resize
from sklearn.externals import joblib
from tqdm import tqdm

from steps.base import BaseTransformer
from utils import relabel


class Resizer(BaseTransformer):
    def transform(self, images, target_sizes):
        resized_images = []
        for image, target_size in tqdm(zip(images, target_sizes)):
            resized_image = resize(image, target_size)
            resized_images.append(resized_image)
        return {'resized_images': resized_images}

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)


class Thresholder(BaseTransformer):
    def __init__(self, threshold):
        self.threshold = threshold

    def transform(self, images):
        binarized_images = []
        for image in images:
            binarized_image = (image > self.threshold).astype(np.uint8)
            binarized_images.append(binarized_image)
        return {'binarized_images': binarized_images}

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)


class WatershedCenter(BaseTransformer):
    def transform(self, images, centers):
        detached_images = []
        for image, center in tqdm(zip(images, centers)):
            detached_image = watershed_center(image, center)
            detached_images.append(detached_image)
        return {'detached_images': detached_images}

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)


class WatershedContour(BaseTransformer):
    def transform(self, images, contours):
        detached_images = []
        for image, contour in tqdm(zip(images, contours)):
            detached_image = watershed_contour(image, contour)
            detached_images.append(detached_image)
        return {'detached_images': detached_images}

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)


class WatershedCombined(BaseTransformer):
    def transform(self, images, contours, centers):
        detached_images = []
        for image, contour, center in tqdm(zip(images, contours, centers)):
            detached_image = watershed_combined(image, contour, center)
            detached_images.append(detached_image)
        return {'detached_images': detached_images}

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)


class Dropper(BaseTransformer):
    def __init__(self, min_size):
        self.min_size = min_size

    def transform(self, labels):
        labeled_images = []
        for image in tqdm(labels):
            labeled_image = drop_small(image, self.min_size)
            labeled_images.append(labeled_image)

        return {'labels': labeled_images}

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)


class NucleiLabeler(BaseTransformer):
    def transform(self, images):
        labeled_images = []
        for i, image in enumerate(images):
            labeled_image = label(image)
            labeled_images.append(labeled_image)

        return {'labeled_images': labeled_images}

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)


def cut(image, contour):
    image = np.where(contour + image == 2, 0, image)
    labeled, nr_true = ndi.label(image)
    return labeled


def drop_small(img, min_size):
    freqs = itemfreq(img)
    small_blob_id = freqs[freqs[:, 1] < min_size, 0]

    h, w = img.shape
    for i, j in product(range(h), range(w)):
        if img[i, j] in small_blob_id:
            img[i, j] = 0

    return relabel(img)


def label(mask):
    labeled, nr_true = ndi.label(mask)
    return labeled


def watershed_center(image, center):
    distance = ndi.distance_transform_edt(image)
    markers, nr_blobs = ndi.label(center)
    labeled = morph.watershed(-distance, markers, mask=image)

    dropped, _ = ndi.label(image - (labeled > 0))
    dropped = np.where(dropped > 0, dropped + nr_blobs, 0)
    correct_labeled = dropped + labeled

    return relabel(correct_labeled)


def watershed_contour(image, contour):
    mask = np.where(contour + image == 2, 0, image)

    distance = ndi.distance_transform_edt(mask)
    markers, nr_blobs = ndi.label(mask)
    labeled = morph.watershed(-distance, markers, mask=image)

    dropped, _ = ndi.label(image - (labeled > 0))
    dropped = np.where(dropped > 0, dropped + nr_blobs, 0)
    correct_labeled = dropped + labeled

    return correct_labeled


def watershed_combined(image, contour, center):
    mask = np.where(contour + image == 2, 0, image)

    distance = ndi.distance_transform_edt(mask)
    markers, nr_blobs = ndi.label(mask)
    labeled = morph.watershed(-distance, markers, mask=image)

    dropped, _ = ndi.label(image - (labeled > 0))
    dropped = np.where(dropped > 0, dropped + nr_blobs, 0)
    correct_labeled = dropped + labeled

    return correct_labeled

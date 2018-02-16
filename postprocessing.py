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


class BinaryFillHoles(BaseTransformer):
    def transform(self, images):
        filled_images = []
        for image in tqdm(images):
            filled_image = fill_holes_per_blob(image)
            filled_images.append(filled_image)
        return {'filled_images': filled_images}

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


class Postprocessor(BaseTransformer):
    def __init__(self, **kwargs):
        pass

    def transform(self, images, contours):
        labeled_images = []
        for image, contour in tqdm(zip(images, contours)):
            labeled_image = postprocess(image, contour)
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
    mask = np.where(contour == 1, 0, image)

    distance = ndi.distance_transform_edt(mask)
    markers, nr_blobs = ndi.label(mask)
    labeled = morph.watershed(-distance, markers, mask=image)

    dropped, _ = ndi.label(image - (labeled > 0))
    dropped = np.where(dropped > 0, dropped + nr_blobs, 0)
    correct_labeled = dropped + labeled
    return relabel(correct_labeled)


def postprocess(image, contour):
    image_init = (image > 0.5).astype(np.uint8)
    contour_init = (contour > 0.5).astype(np.uint8)

    image_cleaned = ndi.morphology.binary_fill_holes(image_init)
    # labels, _ = ndi.label(image_cleaned)
    # mean_r = int(np.ceil(0.05 * (np.sqrt(itemfreq(labels)[1:, 1].mean()))))
    #
    # image_cleaned = ndi.morphology.binary_closing(image_cleaned, structure=morph.disk(mean_r), iterations=1)
    # image_cleaned = ndi.morphology.binary_fill_holes(image_cleaned).astype(np.uint8)

    image_cleaned = np.where((image_init == 0) & (contour_init == 1), 0, image_cleaned)
    # if not in initial with threshold then drop

    final_labels = watershed_contour(image_cleaned, contour_init)
    final_labels = drop_small(final_labels, min_size=20)

    final_labels = fill_holes_per_blob(final_labels)
    return final_labels


def watershed_combined(image, contour, center):
    return NotImplementedError


def fill_holes_per_blob(image):
    image_cleaned = np.zeros_like(image)
    for i in range(1, image.max() + 1):
        mask = np.where(image == i, 1, 0)
        mask = ndi.morphology.binary_fill_holes(mask)
        image_cleaned = image_cleaned + mask * i
    return image_cleaned

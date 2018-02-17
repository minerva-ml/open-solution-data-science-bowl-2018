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
            resized_image = resize(image, target_size, mode='constant')
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
    img = morph.remove_small_objects(img, min_size=min_size)
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
    contour_binary = (contour > 0.5).astype(np.uint8)
    mask_binary = (image > 0.5).astype(np.uint8)

    mask_cleaned = get_clean_mask(mask_binary, contour_binary)

    mask_distance = ndi.distance_transform_edt(mask_cleaned)

    markers, nr_markers = get_markers(mask_cleaned, contour_binary)

    labels_water = morph.watershed(-mask_distance, markers, mask=mask_cleaned)
    labels_water = add_dropped_blobs(labels_water, mask_cleaned, nr_markers)

    labels_water = drop_artifacts_per_label(labels_water, mask_binary)
    final_labels = drop_small(labels_water, min_size=20)
    final_labels = fill_holes_per_blob(final_labels)
    return final_labels


def drop_artifacts_per_label(labels, initial_mask):
    labels_cleaned = np.zeros_like(labels)
    for i in range(1, labels.max() + 1):
        component = np.where(labels == i, 1, 0)
        component_initial_mask = np.where(labels == i, initial_mask, 0)
        component = drop_artifacts(component, component_initial_mask)
        labels_cleaned = labels_cleaned + component * i
    return labels_cleaned


def get_clean_mask(mask_binary, contour_binary):
    mask_cleaned = (mask_binary + contour_binary) > 0.5
    mask_cleaned = ndi.morphology.binary_fill_holes(mask_cleaned)
    mean_area, mean_radius = mean_blob_size(mask_cleaned)
    structure_size = 4 * mean_radius
    mask_cleaned_padded = pad_mask(mask_cleaned, structure_size)
    mask_cleaned_padded = ndi.morphology.binary_closing(mask_cleaned_padded,
                                                        structure=morph.disk(structure_size),
                                                        iterations=1)
    mask_cleaned = crop_mask(mask_cleaned_padded, structure_size)

    mask_cleaned = ndi.morphology.binary_fill_holes(mask_cleaned).astype(np.uint8)
    mask_cleaned = np.where((contour_binary == 1) & (mask_binary == 0), 0, mask_cleaned)
    mask_cleaned = drop_artifacts(mask_cleaned, mask_binary)
    return mask_cleaned


def get_markers(mask_cleaned, contour_binary):
    _, mean_radius = mean_blob_size(mask_cleaned)
    markers = np.where(contour_binary, 0, mask_cleaned)
    markers = morph.erosion(markers, selem=morph.disk(mean_radius))
    markers, nr_blobs = ndi.label(markers)

    return markers, nr_blobs


def add_dropped_blobs(water, mask_cleaned, nr_blobs):
    water_mask = (water > 0).astype(np.uint8)
    dropped = mask_cleaned - water_mask
    dropped, _ = ndi.label(dropped)
    dropped = np.where(dropped, dropped + nr_blobs, 0)
    water = water + dropped
    return water


def watershed_combined(image, contour, center):
    return NotImplementedError


def fill_holes_per_blob(image):
    image_cleaned = np.zeros_like(image)
    for i in range(1, image.max() + 1):
        mask = np.where(image == i, 1, 0)
        mask = ndi.morphology.binary_fill_holes(mask)
        image_cleaned = image_cleaned + mask * i
    return image_cleaned


def drop_artifacts(mask_after, mask_pre, min_coverage=0.5):
    connected, nr_connected = ndi.label(mask_after)
    mask = np.zeros_like(mask_after)
    for i in range(1, nr_connected + 1):
        conn_blob = np.where(connected == i, 1, 0)
        initial_space = np.where(connected == i, mask_pre, 0)
        blob_size = np.sum(conn_blob)
        initial_blob_size = np.sum(initial_space)
        coverage = float(initial_blob_size) / float(blob_size)
        if coverage > min_coverage:
            mask = mask + conn_blob
        else:
            mask = mask + initial_space
    return mask


def mean_blob_size(mask):
    labels, labels_nr = ndi.label(mask)
    if labels_nr < 2:
        mean_area = 1
        mean_radius = 1
    else:
        mean_area = int(np.sqrt(itemfreq(labels)[1:, 1].mean()))
        mean_radius = int(np.round(np.sqrt(mean_area) / np.pi))
    return mean_area, mean_radius


def pad_mask(mask, pad):
    if pad == 1:
        pad = 2
    h, w = mask.shape
    h_pad = h + 2 * pad
    w_pad = w + 2 * pad
    mask_padded = np.zeros((h_pad, w_pad))
    mask_padded[pad:pad + h, pad:pad + w] = mask
    mask_padded[pad - 1, :] = 1
    mask_padded[pad + h + 1, :] = 1
    mask_padded[:, pad - 1] = 1
    mask_padded[:, pad + w + 1] = 1

    return mask_padded


def crop_mask(mask, crop):
    if crop == 1:
        crop = 2
    h, w = mask.shape
    mask_cropped = mask[crop:h - crop, crop:w - crop]
    return mask_cropped

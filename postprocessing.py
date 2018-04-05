import numpy as np
import skimage.morphology as morph
from scipy import ndimage as ndi
from scipy.stats import itemfreq
from skimage.filters import threshold_otsu
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



class Thresholder(BaseTransformer):
    def __init__(self, threshold):
        self.threshold = threshold

    def transform(self, images):
        binarized_images = []
        for image in images:
            binarized_image = (image > self.threshold).astype(np.uint8)
            binarized_images.append(binarized_image)
        return {'binarized_images': binarized_images}



class WatershedCenter(BaseTransformer):
    def transform(self, images, centers):
        detached_images = []
        for image, center in tqdm(zip(images, centers)):
            detached_image = watershed_center(image, center)
            detached_images.append(detached_image)
        return {'detached_images': detached_images}



class WatershedContour(BaseTransformer):
    def transform(self, images, contours):
        detached_images = []
        for image, contour in tqdm(zip(images, contours)):
            detached_image = watershed_contour(image, contour)
            detached_images.append(detached_image)
        return {'detached_images': detached_images}



class BinaryFillHoles(BaseTransformer):
    def transform(self, images):
        filled_images = []
        for image in tqdm(images):
            filled_image = fill_holes_per_blob(image)
            filled_images.append(filled_image)
        return {'filled_images': filled_images}



class Dropper(BaseTransformer):
    def __init__(self, min_size):
        self.min_size = min_size

    def transform(self, labels):
        labeled_images = []
        for image in tqdm(labels):
            labeled_image = drop_small(image, self.min_size)
            labeled_images.append(labeled_image)

        return {'labels': labeled_images}



class NucleiLabeler(BaseTransformer):
    def transform(self, images):
        labeled_images = []
        for i, image in enumerate(images):
            labeled_image = label(image)
            labeled_images.append(labeled_image)

        return {'labeled_images': labeled_images}



class Postprocessor(BaseTransformer):
    def __init__(self, **kwargs):
        pass

    def transform(self, images, contours):
        labeled_images = []
        for image, contour in tqdm(zip(images, contours)):
            labeled_image = postprocess(image, contour)
            labeled_images.append(labeled_image)
        return {'labeled_images': labeled_images}



class CellSizer(BaseTransformer):
    def __init__(self, **kwargs):
        pass

    def transform(self, labeled_images):
        mean_sizes = []
        for image in tqdm(labeled_images):
            mean_size = mean_cell_size(image)
            mean_sizes.append(mean_size)
        return {'sizes': mean_sizes}


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
    cleaned_mask = get_clean_mask_basic(image, contour)
    good_markers = get_markers_basic(cleaned_mask, contour)
    good_distance = get_distance(cleaned_mask)

    labels = morph.watershed(-good_distance, good_markers, mask=cleaned_mask)

    labels = add_dropped_water_blobs(labels, cleaned_mask)

    # m_tresh = 0.5 #threshold_otsu(image)
    # initial_mask_binary = (image > m_tresh).astype(np.uint8)
    # labels = drop_artifacts_per_label(labels, initial_mask_binary)

    labels = connect_small(labels, fraction_of_percentile=0.1)
    min_size = min_blob_size(labels, fraction_of_percentile=0.1)
    labels = drop_small(labels, min_size=min_size)

    labels = drop_big_artifacts(labels, scale=0.01)

    return relabel(labels)


def drop_artifacts_per_label(labels, initial_mask):
    labels_cleaned = np.zeros_like(labels)
    for i in range(1, labels.max() + 1):
        component = np.where(labels == i, 1, 0)
        component_initial_mask = np.where(labels == i, initial_mask, 0)
        component = drop_artifacts(component, component_initial_mask)
        labels_cleaned = labels_cleaned + component * i
    return labels_cleaned


def get_clean_mask_basic(m, c):
    m_b = m > 0.5 #threshold_otsu(m)

    return m_b


def get_clean_mask(m, c):
    # threshold
    m_b = m > 0.5 #threshold_otsu(m)
    c_b = c > 0.5 #threshold_otsu(c)

    # combine contours and masks and fill the cells
    m_ = np.where(m_b | c_b, 1, 0)
    m_ = ndi.binary_fill_holes(m_)

    # close what wasn't closed before
    area, radius = mean_blob_size(m_b)
    if radius >= 6:
        struct_size = int(1.0 * radius)
        struct_el = morph.disk(struct_size)
        m_padded = pad_mask(m_, pad=struct_size)
        m_padded = morph.binary_closing(m_padded, selem=struct_el)
        m_padded = ndi.binary_fill_holes(m_padded)
        m_ = crop_mask(m_padded, crop=struct_size)

        # open to cut the real cells from the artifacts
        area, radius = mean_blob_size(m_b)
        struct_size = int(1.25 * radius)
        struct_el = morph.disk(struct_size)
        m_ = np.where(c_b & (~m_b), 0, m_)
        m_padded = pad_mask(m_, pad=struct_size)
        m_padded = morph.binary_opening(m_padded, selem=struct_el)
        m_ = crop_mask(m_padded, crop=struct_size)

        # join the connected cells with what we had at the beginning
        m_ = np.where(m_b | m_, 1, 0)
        m_ = ndi.binary_fill_holes(m_)

        # drop all the cells that weren't present at least in 25% of area in the initial mask
        m_ = drop_artifacts(m_, m_b, min_coverage=0.25)
    return m_


def get_markers_basic(m_b, c):
    c_b = c > 0.5 #threshold_otsu(c)
    m_ = np.where(c_b, 0, m_b)
    m_, _ = ndi.label(m_)
    return m_


def get_markers(m_b, c):
    # threshold
    area, radius = mean_blob_size(m_b)
    if radius >= 4:
        c_b = c > 0.5 #threshold_otsu(c)
        m_ = np.where(c_b, 0, m_b)

        struct_size = int(0.5 * radius)
        struct_el = morph.disk(struct_size)
        m_padded = pad_mask(m_, pad=struct_size)
        m_padded = morph.erosion(m_padded, selem=struct_el)
        m_ = crop_mask(m_padded, crop=struct_size)
    else:
        m_ = m_b
    m_, _ = ndi.label(m_)
    return m_


def get_distance(m_b):
    distance = ndi.distance_transform_edt(m_b)
    return distance


def add_dropped_water_blobs(water, mask_cleaned):
    water_mask = (water > 0).astype(np.uint8)
    dropped = mask_cleaned - water_mask
    dropped, _ = ndi.label(dropped)
    dropped = np.where(dropped, dropped + water.max(), 0)
    water = water + dropped
    return water


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
        blob_sizes = itemfreq(labels)
        blob_sizes = blob_sizes[blob_sizes[:, 0].argsort()][1:, :]
        mean_area = int(blob_sizes.mean())
        mean_radius = int(np.round(np.sqrt(mean_area) / np.pi))
    return mean_area, mean_radius


def pad_mask(mask, pad):
    if pad <= 1:
        pad = 2
    h, w = mask.shape
    h_pad = h + 2 * pad
    w_pad = w + 2 * pad
    mask_padded = np.zeros((h_pad, w_pad))
    mask_padded[pad:pad + h, pad:pad + w] = mask
    mask_padded[pad, :] = 1
    mask_padded[pad + h + 1, :] = 1
    mask_padded[:, pad] = 1
    mask_padded[:, pad + w + 1] = 1

    return mask_padded


def crop_mask(mask, crop):
    if crop <= 1:
        crop = 2
    h, w = mask.shape
    mask_cropped = mask[crop:h - crop, crop:w - crop]
    return mask_cropped


def drop_small(img, min_size):
    img = morph.remove_small_objects(img, min_size=min_size)
    return relabel(img)


def label(mask):
    labeled, nr_true = ndi.label(mask)
    return labeled


def min_blob_size(mask, percentile=25, fraction_of_percentile=0.1):
    labels, labels_nr = ndi.label(mask)
    if labels_nr < 2:
        return 0
    else:
        blob_sizes = itemfreq(labels)
        blob_sizes = blob_sizes[blob_sizes[:, 0].argsort()][1:, 1]
        return fraction_of_percentile * np.percentile(blob_sizes, percentile)


def mean_cell_size(labeled_image):
    blob_sizes = itemfreq(labeled_image)
    if blob_sizes.shape[0]==1:
        return 0
    else:
        blob_sizes = blob_sizes[blob_sizes[:, 0].argsort()][1:, 1]
        return np.mean(blob_sizes)


def find_touching_labels(labels, label_id):
    mask = np.where(labels == label_id, 0, 1)
    dist = ndi.distance_transform_edt(mask)
    neighbour_labels = np.unique(np.where(dist == 1.0, labels, 0)).tolist()
    neighbour_labels.remove(0)
    return neighbour_labels


def connect_small(labels, fraction_of_percentile):
    touching_cell_was_connected = False
    min_cell_size = min_blob_size(labels, fraction_of_percentile)
    for label_id in range(1, labels.max() + 1):
        cell_size = np.sum(labels == label_id)
        touching_labels = find_touching_labels(labels, label_id)
        for touching_label in touching_labels:
            touching_cell_mask = np.where(labels == touching_label, 1, 0)
            touching_cell_size = np.sum(touching_cell_mask)
            if touching_cell_size < min_cell_size:
                labels = np.where(labels == touching_label, label_id, labels)
                touching_cell_was_connected = True
    labels = relabel(labels)
    if touching_cell_was_connected:
        labels = connect_small(labels, fraction_of_percentile)
    return relabel(labels)


def is_slim(im, object_ar, area_ar):
    ind = np.where(im == 1)
    ydiff = np.max(ind[0]) - np.min(ind[0])
    xdiff = np.max(ind[1]) - np.min(ind[1])
    rec_area = xdiff * ydiff
    area = np.sum(im == 1)
    if xdiff / ydiff < object_ar and xdiff / ydiff > 1.0/object_ar and area / rec_area > area_ar:
        return False
    return True


def touching_edges(im, margin):
    indices = np.where(im == 1)
    edges = []
    edges.append(np.sum(indices[0] <= margin))
    edges.append(np.sum(indices[1] <= margin))
    edges.append(np.sum(indices[0] >= im.shape[0] - 1 - margin))
    edges.append(np.sum(indices[1] >= im.shape[1] - 1 - margin))
    return np.sum(np.array(edges) > 0)


def drop_big_artifacts(im, scale):
    im_cleaned = np.copy(im)
    im_size = im.shape[0] * im.shape[1]
    for label in np.unique(im):
        if label == 0:
            continue
        size = np.sum(im == label)
        if size < scale * im_size:
            continue
        if not is_slim(im == label, 2, 0.5):
            continue
        if touching_edges(im = im==label, margin=2) < 2:
            continue
        im_cleaned[im_cleaned == label] = 0
    return im_cleaned
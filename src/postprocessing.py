import numpy as np
import skimage.morphology as morph
from scipy import ndimage as ndi
from scipy.stats import itemfreq
from skimage.transform import resize
from skimage.filters import threshold_otsu

from .utils import relabel, get_crop_pad_sequence


def resize_image(image, target_size):
    """Resize image to target size

    Args:
        image (numpy.ndarray): Image of shape (C x H x W).
        target_size (tuple): Target size (H, W).

    Returns:
        numpy.ndarray: Resized image of shape (C x H x W).

    """
    n_channels = image.shape[0]
    resized_image = resize(image, (n_channels, target_size[0], target_size[1]), mode='constant')
    return resized_image


def crop_image(image, target_size):
    """Crop image to target size. Image cropped symmetrically.

    Args:
        image (numpy.ndarray): Image of shape (C x H x W).
        target_size (tuple): Target size (H, W).

    Returns:
        numpy.ndarray: Cropped image of shape (C x H x W).

    """
    top_crop, right_crop, bottom_crop, left_crop = get_crop_pad_sequence(image.shape[1] - target_size[0],
                                                                      image.shape[2] - target_size[1])
    cropped_image = image[:, top_crop:image.shape[1] - bottom_crop, left_crop:image.shape[2] - right_crop]
    return cropped_image


def categorize_image(image, activation='softmax', threshold=0.5):
    """Maps probability map to categories. Each pixel is assigned with a category with highest probability.

    Args:
        image (numpy.ndarray): Probability map of shape (C x H x W).
        activation (string): Activation function, either softmax or sigmoid. Defaults to 'softmax'.
        threshold (float or list): Single threshold for sigmoid activation or list of per class thresholds.

    Returns:
        numpy.ndarray: Categorized image of shape (C x H x W).

    """
    categorized_image = []
    if activation == 'softmax':
        class_map = np.argmax(image, axis=0)
        for class_nr in range(image.shape[0]):
            categorized_image.append((class_map == class_nr).astype(np.uint8))
    if activation == 'sigmoid':
        if isinstance(threshold, float):
            threshold = [threshold] * image.shape[0]
        for thrs, class_instance in zip(threshold, image):
            categorized_image.append((class_instance > thrs).astype(np.uint8))

    return np.stack(categorized_image)


def label_multiclass_image(image):
    labeled_image = []
    for class_instance in image:
        labeled_image.append(label(class_instance))
    return np.stack(labeled_image)


def get_channel(image, channel):
    return image[channel, :, :]


def watershed(masks, seeds, borders):
    seeds_detached = seeds * (1 - borders)
    markers = label(seeds_detached)
    labels = morph.watershed(masks, markers, mask=masks)
    return labels


def drop_artifacts_per_label(labels, initial_mask):
    labels_cleaned = np.zeros_like(labels)
    for i in range(1, labels.max() + 1):
        component = np.where(labels == i, 1, 0)
        component_initial_mask = np.where(labels == i, initial_mask, 0)
        component = drop_artifacts(component, component_initial_mask)
        labels_cleaned = labels_cleaned + component * i
    return labels_cleaned


def clean_mask(m, c):
    # threshold
    m_thresh = threshold_otsu(m)
    c_thresh = threshold_otsu(c)
    m_b = m > m_thresh
    c_b = c > c_thresh

    # combine contours and masks and fill the cells
    m_ = np.where(m_b | c_b, 1, 0)
    m_ = ndi.binary_fill_holes(m_)

    # close what wasn't closed before
    area, radius = mean_blob_size(m_b)
    struct_size = int(1.25 * radius)
    struct_el = morph.disk(struct_size)
    m_padded = pad_mask(m_, pad=struct_size)
    m_padded = morph.binary_closing(m_padded, selem=struct_el)
    m_ = crop_mask(m_padded, crop=struct_size)

    # open to cut the real cells from the artifacts
    area, radius = mean_blob_size(m_b)
    struct_size = int(0.75 * radius)
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


def get_markers(m_b, c):
    # threshold
    c_thresh = threshold_otsu(c)
    c_b = c > c_thresh

    mk_ = np.where(c_b, 0, m_b)

    area, radius = mean_blob_size(m_b)
    struct_size = int(0.25 * radius)
    struct_el = morph.disk(struct_size)
    m_padded = pad_mask(mk_, pad=struct_size)
    m_padded = morph.erosion(m_padded, selem=struct_el)
    mk_ = crop_mask(m_padded, crop=struct_size)
    mk_, _ = ndi.label(mk_)
    return mk_


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
        mean_area = int(itemfreq(labels)[1:, 1].mean())
        mean_radius = int(np.round(np.sqrt(mean_area / np.pi)))
    return mean_area, mean_radius


def pad_mask(mask, pad):
    if pad <= 1:
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
    if crop <= 1:
        crop = 2
    h, w = mask.shape
    mask_cropped = mask[crop:h - crop, crop:w - crop]
    return mask_cropped


def drop_small_unlabeled(img, min_size):
    img = morph.remove_small_objects(img.astype(np.bool), min_size=min_size)
    return img.astype(np.uint8)


def drop_small(img, min_size):
    img = morph.remove_small_objects(img, min_size=min_size)
    return relabel(img)


def label(mask):
    labeled, nr_true = ndi.label(mask)
    return labeled

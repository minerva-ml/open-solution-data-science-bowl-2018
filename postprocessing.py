from itertools import product

from tqdm import tqdm
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import itemfreq
from sklearn.externals import joblib
from skimage.transform import resize
import skimage.morphology as morph

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


class Whatershed(BaseTransformer):
    def __init__(self, **kwargs):
        pass

    def transform(self, images, centers):
        detached_images = []
        for image, center in zip(images, centers):
            detached_image = self.detach_nuclei(image, center)
            detached_images.append(detached_image)
        return {'detached_images': detached_images}

    def detach_nuclei(self, image, center):
        distance = ndi.distance_transform_edt(image)
        markers, nr_blobs = ndi.label(center)
        labeled = morph.watershed(-distance, markers, mask=image)

        dropped, _ = ndi.label(image - (labeled > 0))
        dropped = np.where(dropped > 0, dropped + nr_blobs, 0)
        correct_labeled = dropped + labeled

        return relabel(correct_labeled)

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)


class Dropper(BaseTransformer):
    def __init__(self, min_size):
        self.min_size = min_size

    def transform(self, labels):
        labeled_images = []
        for i, image in enumerate(labels):
            labeled_image = self.drop_small(image)
            labeled_images.append(labeled_image)

        return {'labels': labeled_images}

    def drop_small(self, img):
        freqs = itemfreq(img)
        small_blob_id = freqs[freqs[:, 1] < self.min_size, 0]

        h, w = img.shape
        for i, j in product(range(h), range(w)):
            if img[i, j] in small_blob_id:
                img[i, j] = 0

        return relabel(img)

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)


class NucleiLabeler(BaseTransformer):
    def transform(self, images):
        labeled_images = []
        for i, image in enumerate(images):
            labeled_image = self.label(image)
            labeled_images.append(labeled_image)

        return {'labeled_images': labeled_images}

    def label(self, mask):
        labeled, nr_true = ndi.label(mask)
        return labeled

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)

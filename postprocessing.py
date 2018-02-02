import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy import ndimage as ndi
from sklearn.externals import joblib
from skimage.transform import resize
import skimage.morphology as morph

from steps.base import BaseTransformer


class Resizer(BaseTransformer):
    def transform(self, images, target_sizes):
        resized_images = []
        for i, (image, target_size) in enumerate(tqdm(zip(images, target_sizes))):
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
        for i, image in enumerate(images):
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

    def transform(self, images):
        detached_images = []
        for i, image in enumerate(images):
            detached_image = self.detach_nuclei(image)
            detached_images.append(detached_image)
        joblib.dump((images, detached_images), '/mnt/ml-team/dsb_2018/kuba/labeler_debug.pkl')
        return {'detached_images': detached_images}

    def detach_nuclei(self, image):
        distance = ndi.distance_transform_edt(image)

        eroded = distance
        for i in range(3):
            eroded = morph.erosion(eroded, selem=morph.square(5))

        markers, nr_blobs = ndi.label(eroded)
        labeled = morph.watershed(-distance, markers, mask=image)

        dropped, _ = ndi.label(image - (labeled > 0))
        dropped = np.where(dropped > 0, dropped + nr_blobs, 0)

        correct_labeled = dropped + labeled
        return correct_labeled

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)


class NucleiLabeler(BaseTransformer):
    def __init__(self, **kwargs):
        pass

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

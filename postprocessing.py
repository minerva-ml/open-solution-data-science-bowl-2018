import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.externals import joblib
from skimage.transform import resize

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

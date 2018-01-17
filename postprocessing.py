from sklearn.externals import joblib
from skimage.transform import resize

from steps.base import BaseTransformer


class Resizer(BaseTransformer):
    def transform(self, images, target_sizes):
        resized_images = []
        for image, target_shape in zip(images, target_sizes):
            resized_image = resize(image, target_sizes)
            resized_images.append(resized_image)

        return {'resized_images': resized_images}

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)

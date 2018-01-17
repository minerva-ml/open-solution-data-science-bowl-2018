from sklearn.externals import joblib
from skimage.transform import resize

from steps.base import BaseTransformer


class Resizer(BaseTransformer):
    def transform(self, images, target_sizes):
        resized_images = []
        for image, target_size in zip(images, target_sizes):
            print(image.shape, target_size)
            resized_image = resize(image, target_size)
            resized_images.append(resized_image)

        return {'resized_images': resized_images}

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)

import numpy as np
from sklearn.externals import joblib

from steps.base import BaseTransformer


class MockModel(BaseTransformer):
    def __init__(self, **kwargs):
        super().__init__()

    def fit(self, datagen, validation_datagen, **kwargs):
        return self

    def transform(self, datagen, **kwargs):
        X, steps = datagen
        masks = np.random.random((X.shape[0], 256, 256))
        return {'predicted_masks': masks}

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)

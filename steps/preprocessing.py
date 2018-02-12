import os

from tqdm import tqdm
from PIL import Image
from sklearn.externals import joblib
from sklearn.feature_extraction import text

from .base import BaseTransformer


class XYSplit(BaseTransformer):
    def __init__(self, x_columns, y_columns):
        self.x_columns = x_columns
        self.y_columns = y_columns

    def transform(self, meta, train_mode):
        X = meta[self.x_columns].values
        if train_mode:
            y = meta[self.y_columns].values
        else:
            y = None

        return {'X': X,
                'y': y}

    def load(self, filepath):
        params = joblib.load(filepath)
        self.columns_to_get = params['x_columns']
        self.target_columns = params['y_columns']
        return self

    def save(self, filepath):
        params = {'x_columns': self.x_columns,
                  'y_columns': self.y_columns
                  }
        joblib.dump(params, filepath)


class ImageReader(BaseTransformer):
    def __init__(self, x_columns, y_columns, target_shape):
        self.x_columns = x_columns
        self.y_columns = y_columns
        self.target_shape = target_shape

    def transform(self, meta, train_mode):
        X_ = meta[self.x_columns].values

        X = self.load_images(X_, grayscale=False)
        if train_mode:
            y_ = meta[self.y_columns].values
            y = self.load_images(y_, grayscale=True)
        else:
            y = None

        return {'X': X,
                'y': y}

    def load_images(self, image_filepaths, grayscale):
        X = []
        for i in range(image_filepaths.shape[1]):
            column = image_filepaths[:, i]
            X.append([])
            for img_filepath in tqdm(column):
                img = self.load_image(img_filepath, grayscale=grayscale)
                X[i].append(img)
        return X

    def load_image(self, img_filepath, grayscale):
        image = Image.open(img_filepath, 'r')
        if not grayscale:
            image = image.convert('RGB')
        else:
            image = image.convert('L')
        image = image.resize(self.target_shape)
        return image

    def load(self, filepath):
        params = joblib.load(filepath)
        self.columns_to_get = params['x_columns']
        self.target_columns = params['y_columns']
        return self

    def save(self, filepath):
        params = {'x_columns': self.x_columns,
                  'y_columns': self.y_columns
                  }
        joblib.dump(params, filepath)


class TfidfVectorizer(BaseTransformer):
    def __init__(self, **kwargs):
        self.vectorizer = text.TfidfVectorizer(**kwargs)

    def fit(self, text):
        self.vectorizer.fit(text)
        return self

    def transform(self, text):
        return {'features': self.vectorizer.transform(text)}

    def load(self, filepath):
        self.vectorizer = joblib.load(filepath)
        return self

    def save(self, filepath):
        joblib.dump(self.vectorizer, filepath)

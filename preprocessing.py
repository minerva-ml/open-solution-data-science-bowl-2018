from sklearn.externals import joblib

from steps.base import BaseTransformer


class ImageReaderRescaler(BaseTransformer):
    def transform(self, sizes, X, y):
        # X_ = meta[self.x_columns].values

        print(X,y,sizes)

        return {'X': X,
                'y': y}

    def load(self, filepath):
        return self

    def save(self, filepath):
        params = {}
        joblib.dump(params, filepath)

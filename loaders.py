from sklearn.externals import joblib


from steps.base import BaseTransformer


class MockLoader(BaseTransformer):
    def __init__(self, loader_params):
        super().__init__()
        self.loader_params = loader_params

    def transform(self, X, y, X_valid, y_valid, train_mode):
        if train_mode:
            flow, steps = X, 10
        else:
            flow, steps = X, 10

        if X_valid is not None and y_valid is not None:
            valid_flow, valid_steps = X_valid, 10
        else:
            valid_flow = None
            valid_steps = None

        return {'datagen': (flow, steps),
                'validation_datagen': (valid_flow, valid_steps)}

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)
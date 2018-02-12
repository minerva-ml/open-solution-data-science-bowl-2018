import numpy as np
import sklearn.linear_model as lr
from sklearn.externals import joblib

from steps.base import BaseTransformer
from steps.utils import get_logger

logger = get_logger()


class LogisticRegressionMultilabel(BaseTransformer):
    def __init__(self, label_nr, **kwargs):
        self.label_nr = label_nr
        self.logistic_regressors = self._get_logistic_regressors(**kwargs)

    def _get_logistic_regressors(self, **kwargs):
        logistic_regressors = []
        for i in range(self.label_nr):
            logistic_regressors.append((i, lr.LogisticRegression(**kwargs)))
        return logistic_regressors

    def fit(self, X, y):
        for i, log_reg in self.logistic_regressors:
            logger.info('fitting regressor {}'.format(i))
            log_reg.fit(X, y[:, i])
        return self

    def transform(self, X, y=None):
        predictions = []
        for i, log_reg in self.logistic_regressors:
            prediction = log_reg.predict_proba(X)
            predictions.append(prediction)
        predictions = np.stack(predictions, axis=0)
        predictions = predictions[:, :, 1].transpose()
        return {'prediction_probability': predictions}

    def load(self, filepath):
        params = joblib.load(filepath)
        self.label_nr = params['label_nr']
        self.logistic_regressors = params['logistic_regressors']
        return self

    def save(self, filepath):
        params = {'label_nr': self.label_nr,
                  'logistic_regressors': self.logistic_regressors}
        joblib.dump(params, filepath)

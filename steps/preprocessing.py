import re
import string

import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_extraction import text

from .base import BaseTransformer


class TextCleaner(BaseTransformer):
    def __init__(self, drop_punctuation, drop_newline, drop_multispaces, all_lower_case, fill_na_with):
        self.drop_punctuation = drop_punctuation
        self.drop_newline = drop_newline
        self.drop_multispaces = drop_multispaces
        self.all_lower_case = all_lower_case
        self.fill_na_with = fill_na_with

    def transform(self, X):
        X = pd.DataFrame(X, columns=['text']).astype(str)
        X['text'] = X['text'].apply(self._transform)
        if self.fill_na_with:
            X['text'] = X['text'].fillna(self.fill_na_with).values
        return {'X': X['text'].values}

    def _transform(self, x):
        if self.all_lower_case:
            x = self._lower(x)
        if self.drop_punctuation:
            x = self._remove_punctuation(x)
        if self.drop_newline:
            x = self._remove_newline(x)
        if self.drop_multispaces:
            x = self._substitute_multiple_spaces(x)
        return x

    def _lower(self, x):
        return x.lower()

    def _remove_punctuation(self, x):
        return re.sub(r'[^\w\s]', ' ', x)

    def _remove_newline(self, x):
        x = x.replace('\n', ' ')
        x = x.replace('\n\n', ' ')
        return x
    
    def _substitute_multiple_spaces(self, x):
        return ' '.join(x.split())

    def load(self, filepath):
        params = joblib.load(filepath)
        self.drop_punctuation = params['drop_punctuation']
        self.all_lower_case = params['all_lower_case']
        self.fill_na_with = params['fill_na_with']
        return self

    def save(self, filepath):
        params = {'drop_punctuation': self.drop_punctuation,
                  'all_lower_case': self.all_lower_case,
                  'fill_na_with': self.fill_na_with,
                  }
        joblib.dump(params, filepath)


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

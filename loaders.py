from sklearn.externals import joblib

from steps.base import BaseTransformer

import matplotlib.pyplot as plt
from math import ceil
import numpy as np
from sklearn.externals import joblib
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class MetadataImageSegmentationDataset(Dataset):
    def __init__(self, X, y, image_transform, image_augment):
        super().__init__()
        self.X = X
        if y is not None:
            self.y = y
        else:
            self.y = None

        self.image_transform = image_transform
        self.image_augment = image_augment

    def load_image(self, img_filepath):
        image = plt.imread(img_filepath)
        return image

    def load_image_mask(self, mask_filepath):
        mask = plt.imread(mask_filepath)[:, :, 0]
        mask_binarized = (mask > 0.5).astype(np.uint8)
        return mask_binarized

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        img_filepath = self.X[index]
        mask_filepath = self.y[index]

        Xi = self.load_image(img_filepath)
        if self.image_augment is not None:
            Xi = self.image_augment(Xi)

        if self.image_transform is not None:
            Xi = self.image_transform(Xi)

        if self.y is not None:
            Mi = self.load_image_mask(mask_filepath)
            if self.image_augment is not None:
                Mi = self.image_augment(Mi)
            if self.image_transform is not None:
                Mi = self.image_augment(Mi)
            return Xi, Mi
        else:
            return Xi


class MetadataImageSegmentationLoader(BaseTransformer):
    def __init__(self, loader_params):
        super().__init__()
        self.loader_params = loader_params

        self.dataset = MetadataImageSegmentationDataset
        self.image_transform = transforms.ToTensor()
        self.image_augment = None

    def transform(self, X, y, validation_data, train_mode):
        if train_mode:
            flow, steps = self.get_datagen(X, y, train_mode, self.loader_params['training'])
        else:
            flow, steps = self.get_datagen(X, y, train_mode, self.loader_params['inference'])

        if validation_data is not None:
            X_valid, y_valid = validation_data
            valid_flow, valid_steps = self.get_datagen(X_valid, y_valid, False, self.loader_params['inference'])
        else:
            valid_flow = None
            valid_steps = None

        return {'datagen': (flow, steps),
                'validation_datagen': (valid_flow, valid_steps)}

    def get_datagen(self, X, y, train_mode, loader_params):
        if train_mode:
            augmantation = self.image_augment
        else:
            augmantation = None

        dataset = self.dataset(X, y,
                               image_augment=augmantation,
                               image_transform=self.image_transform)

        datagen = DataLoader(dataset, **loader_params)
        steps = ceil(X.shape[0] / loader_params['batch_size'])
        return datagen, steps

    def load(self, filepath):
        params = joblib.load(filepath)
        self.loader_params = params['loader_params']
        return self

    def save(self, filepath):
        params = {'loader_params': self.loader_params}
        joblib.dump(params, filepath)


class MockLoader(BaseTransformer):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def transform(self, X, y, X_valid=None, y_valid=None, train_mode=True):
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

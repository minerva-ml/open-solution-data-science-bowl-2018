from attrdict import AttrDict
from PIL import Image
from math import ceil
import numpy as np
from sklearn.externals import joblib
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from steps.base import BaseTransformer


class MetadataImageSegmentationDataset(Dataset):
    def __init__(self, X, y, train_mode, image_transform, mask_transform, image_augment):
        super().__init__()
        self.X = X
        if y is not None:
            self.y = y
        else:
            self.y = None

        self.train_mode = train_mode
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_augment = image_augment

    def load_image(self, img_filepath):
        image = Image.open(img_filepath, 'r')
        return image.convert('RGB')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        img_filepath = self.X[index]

        Xi = self.load_image(img_filepath)
        if self.image_augment is not None:
            Xi = self.image_augment(Xi)

        if self.image_transform is not None:
            Xi = self.image_transform(Xi)

        if self.y is not None and self.train_mode:
            mask_filepath = self.y[index]
            Mi = self.load_image(mask_filepath)
            if self.image_augment is not None:
                Mi = self.image_augment(Mi)
            if self.mask_transform is not None:
                Mi = self.mask_transform(Mi)
            return Xi, Mi
        else:
            return Xi


class MetadataImageSegmentationMultitaskDataset(Dataset):
    def __init__(self, X, y, train_mode, image_transform, mask_transform, image_augment):
        super().__init__()
        self.X = X
        if y is not None:
            self.y = y
        else:
            self.y = None

        self.train_mode = train_mode
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_augment = image_augment

    def load_image(self, img_filepath):
        image = Image.open(img_filepath, 'r')
        return image.convert('RGB')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        img_filepath = self.X[index]

        Xi = self.load_image(img_filepath)
        if self.image_augment is not None:
            Xi = self.image_augment(Xi)

        if self.image_transform is not None:
            Xi = self.image_transform(Xi)

        if self.y is not None and self.train_mode:
            mask_filepath = self.y[index, 0]
            contour_filepath = self.y[index, 1]
            center_filepath = self.y[index, 2]

            Mi = self.load_image(mask_filepath)
            CTi = self.load_image(contour_filepath)
            CRi = self.load_image(center_filepath)

            if self.image_augment is not None:
                Mi = self.image_augment(Mi)
                CTi = self.image_augment(CTi)
                CRi = self.image_augment(CRi)

            if self.mask_transform is not None:
                Mi = self.mask_transform(Mi)
                CTi = self.mask_transform(CTi)
                CRi = self.mask_transform(CRi)

            return Xi, Mi, CTi, CRi
        else:
            return Xi


class MetadataImageSegmentationLoader(BaseTransformer):
    def __init__(self, loader_params, dataset_params):
        super().__init__()
        self.loader_params = AttrDict(loader_params)
        self.dataset_params = AttrDict(dataset_params)

        self.dataset = MetadataImageSegmentationDataset
        self.image_transform = transforms.Compose([transforms.Scale((self.dataset_params.h,
                                                                     self.dataset_params.w)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                        std=[0.2, 0.2, 0.2]),
                                                   ])
        self.mask_transform = transforms.Compose([transforms.Scale((self.dataset_params.h,
                                                                    self.dataset_params.w)),
                                                  transforms.Lambda(binarize),
                                                  transforms.Lambda(to_tensor),
                                                  ])
        self.image_augment = None

    def transform(self, X, y, X_valid=None, y_valid=None, train_mode=True):
        if train_mode and y is not None:
            flow, steps = self.get_datagen(X, y, True, self.loader_params.training)
        else:
            flow, steps = self.get_datagen(X, None, False, self.loader_params.inference)

        if X_valid is not None and y_valid is not None:
            valid_flow, valid_steps = self.get_datagen(X_valid, y_valid, True, self.loader_params.inference)
        else:
            valid_flow = None
            valid_steps = None
        return {'datagen': (flow, steps),
                'validation_datagen': (valid_flow, valid_steps)}

    def get_datagen(self, X, y, train_mode, loader_params):
        if train_mode:
            dataset = self.dataset(X, y,
                                   train_mode=True,
                                   image_augment=self.image_augment,
                                   mask_transform=self.mask_transform,
                                   image_transform=self.image_transform)
        else:
            dataset = self.dataset(X, y,
                                   train_mode=False,
                                   image_augment=None,
                                   mask_transform=self.mask_transform,
                                   image_transform=self.image_transform)

        datagen = DataLoader(dataset, **loader_params)

        steps = ceil(X.shape[0] / loader_params.batch_size)
        return datagen, steps

    def load(self, filepath):
        params = joblib.load(filepath)
        self.loader_params = params['loader_params']
        return self

    def save(self, filepath):
        params = {'loader_params': self.loader_params}
        joblib.dump(params, filepath)


class MetadataImageSegmentationMultitaskLoader(MetadataImageSegmentationLoader):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)
        self.dataset = MetadataImageSegmentationMultitaskDataset


def binarize(x):
    x_ = x.convert('L')  # convert image to monochrome
    x_ = np.array(x_)
    x_ = (x_ > 125).astype(np.float32)
    return x_


def to_tensor(x):
    x_ = np.expand_dims(x, axis=0)
    x_ = torch.from_numpy(x_)
    return x_

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from attrdict import AttrDict
from sklearn.externals import joblib
from torch.utils.data import Dataset, DataLoader

from augmentation import affine_seq, color_seq
from steps.base import BaseTransformer
from steps.pytorch.utils import ImgAug
from utils import from_pil, to_pil


class MetadataImageSegmentationDataset(Dataset):
    def __init__(self, X, y, train_mode,
                 image_transform, image_augment_with_target,
                 mask_transform, image_augment):
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
        self.image_augment_with_target = image_augment_with_target

    def load_image(self, img_filepath):
        image = Image.open(img_filepath, 'r')
        return image.convert('RGB')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        img_filepath = self.X[index]
        Xi = self.load_image(img_filepath)

        if self.y is not None:
            mask_filepath = self.y[index]
            Mi = self.load_image(mask_filepath)

            if self.train_mode and self.image_augment_with_target is not None:
                Xi, Mi = from_pil(Xi, Mi)
                Xi, Mi = self.image_augment_with_target(Xi, Mi)
                Xi = self.image_augment(Xi)
                Xi, Mi = to_pil(Xi, Mi)

            if self.mask_transform is not None:
                Mi = self.mask_transform(Mi)

            if self.image_transform is not None:
                Xi = self.image_transform(Xi)
            return Xi, Mi
        else:
            if self.image_transform is not None:
                Xi = self.image_transform(Xi)
            return Xi


class ImageSegmentationDataset(Dataset):
    def __init__(self, X, y, train_mode,
                 image_transform, image_augment_with_target,
                 mask_transform, image_augment):
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
        self.image_augment_with_target = image_augment_with_target

    def __len__(self):
        return len(self.X[0])

    def __getitem__(self, index):
        Xi = self.X[0][index]

        if self.y is not None:
            Mi = self.y[0][index]

            if self.train_mode and self.image_augment_with_target is not None:
                Xi, Mi = from_pil(Xi, Mi)
                Xi, Mi = self.image_augment_with_target(Xi, Mi)
                Xi = self.image_augment(Xi)
                Xi, Mi = to_pil(Xi, Mi)

            if self.mask_transform is not None:
                Mi = self.mask_transform(Mi)

            if self.image_transform is not None:
                Xi = self.image_transform(Xi)
            return Xi, Mi
        else:
            if self.image_transform is not None:
                Xi = self.image_transform(Xi)
            return Xi


class MetadataImageSegmentationMultitaskDataset(Dataset):
    def __init__(self, X, y, train_mode,
                 image_transform, image_augment_with_target,
                 mask_transform, image_augment):
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
        self.image_augment_with_target = image_augment_with_target

    def load_image(self, img_filepath):
        image = Image.open(img_filepath, 'r')
        return image.convert('RGB')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        img_filepath = self.X[index]

        Xi = self.load_image(img_filepath)
        if self.y is not None:
            target_masks = []
            for i in range(y.shape[1]):
                filepath = self.y[index, i]
                mask = self.load_image(filepath)
                target_masks.append(mask)
            target_masks = [target[index] for target in self.y]
            data = [Xi] + target_masks

            if self.train_mode and self.image_augment_with_target is not None:
                data = from_pil(*data)
                data = self.image_augment_with_target(*data)
                data[0] = self.image_augment(data[0])
                data = to_pil(*data)

            if self.mask_transform is not None:
                data[1:] = [self.mask_transform(mask) for mask in data[1:]]

            if self.image_transform is not None:
                data[0] = self.image_transform(data[0])

            return data
        else:
            if self.image_transform is not None:
                Xi = self.image_transform(Xi)
            return Xi


class ImageSegmentationMultitaskDataset(Dataset):
    def __init__(self, X, y, train_mode,
                 image_transform, image_augment_with_target,
                 mask_transform, image_augment):
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
        self.image_augment_with_target = image_augment_with_target

    def __len__(self):
        return len(self.X[0])

    def __getitem__(self, index):
        Xi = self.X[0][index]

        if self.y is not None:
            target_masks = [target[index] for target in self.y]
            data = [Xi] + target_masks

            if self.train_mode and self.image_augment_with_target is not None:
                data = from_pil(*data)
                data = self.image_augment_with_target(*data)
                data[0] = self.image_augment(data[0])
                data = to_pil(*data)

            if self.mask_transform is not None:
                data[1:] = [self.mask_transform(mask) for mask in data[1:]]

            if self.image_transform is not None:
                data[0] = self.image_transform(data[0])

            return data
        else:
            if self.image_transform is not None:
                Xi = self.image_transform(Xi)
            return Xi


class MetadataImageSegmentationLoader(BaseTransformer):
    def __init__(self, loader_params, dataset_params):
        super().__init__()
        self.loader_params = AttrDict(loader_params)
        self.dataset_params = AttrDict(dataset_params)

        self.dataset = MetadataImageSegmentationDataset
        self.image_transform = transforms.Compose([transforms.Resize((self.dataset_params.h,
                                                                      self.dataset_params.w)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                        std=[0.2, 0.2, 0.2]),
                                                   ])
        self.mask_transform = transforms.Compose([transforms.Resize((self.dataset_params.h,
                                                                     self.dataset_params.w)),
                                                  transforms.Lambda(binarize),
                                                  transforms.Lambda(to_tensor),
                                                  ])
        self.image_augment_with_target = ImgAug(affine_seq)
        self.image_augment = ImgAug(color_seq)

    def transform(self, X, y, X_valid=None, y_valid=None, train_mode=True):
        if train_mode and y is not None:
            flow, steps = self.get_datagen(X, y, True, self.loader_params.training)
        else:
            flow, steps = self.get_datagen(X, None, False, self.loader_params.inference)

        if X_valid is not None and y_valid is not None:
            valid_flow, valid_steps = self.get_datagen(X_valid, y_valid, False, self.loader_params.inference)
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
                                   image_augment_with_target=self.image_augment_with_target,
                                   mask_transform=self.mask_transform,
                                   image_transform=self.image_transform)
        else:
            dataset = self.dataset(X, y,
                                   train_mode=False,
                                   image_augment=None,
                                   image_augment_with_target=None,
                                   mask_transform=self.mask_transform,
                                   image_transform=self.image_transform)

        datagen = DataLoader(dataset, **loader_params)
        steps = len(datagen)
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


class ImageSegmentationLoader(MetadataImageSegmentationLoader):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)
        self.dataset = ImageSegmentationDataset


class ImageSegmentationMultitaskLoader(MetadataImageSegmentationLoader):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)
        self.dataset = ImageSegmentationMultitaskDataset


def binarize(x):
    x_ = x.convert('L')  # convert image to monochrome
    x_ = np.array(x_)
    x_ = (x_ > 125).astype(np.float32)
    return x_


def to_tensor(x):
    x_ = np.expand_dims(x, axis=0)
    x_ = torch.from_numpy(x_)
    return x_

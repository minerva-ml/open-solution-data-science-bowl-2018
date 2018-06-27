import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from attrdict import AttrDict
from sklearn.externals import joblib
from torch.utils.data import Dataset, DataLoader

from .steppy.base import BaseTransformer
from .steppy.pytorch.utils import ImgAug

from .augmentation import affine_seq, color_seq, crop_seq, pad_to_fit_net
from .utils import from_pil, to_pil
from .pipeline_config import MEAN, STD


class ImageSegmentationDataset(Dataset):
    def __init__(self, X, y, train_mode,
                 image_transform, image_augment_with_target,
                 mask_transform, image_augment,
                 image_source='memory'):
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

        self.image_source = image_source

    def __len__(self):
        if self.image_source == 'memory':
            return len(self.X[0][0])
        elif self.image_source == 'disk':
            return self.X.shape[0]

    def __getitem__(self, index):
        if self.image_source == 'memory':
            load_func = self.load_from_memory
        elif self.image_source == 'disk':
            load_func = self.load_from_disk
        else:
            raise NotImplementedError("Possible loading options: 'memory' and 'disk'!")

        Xi = load_func(self.X, index)

        if self.y is not None:
            Mi = load_func(self.y, index)

            if self.image_augment_with_target is not None or self.image_augment is not None:
                Xi, Mi = from_pil(Xi, Mi)
                if self.image_augment_with_target is not None:
                    Xi, Mi = self.image_augment_with_target(Xi, Mi)
                if self.image_augment is not None:
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

    def load_from_memory(self, data_source, index):
        return data_source[0][0][index]

    def load_from_disk(self, data_source, index):
        img_filepath = data_source[index]
        return self.load_image(img_filepath)

    def load_image(self, img_filepath):
        image = Image.open(img_filepath, 'r')
        return image.convert('RGB')


class ImageSegmentationLoaderBasic(BaseTransformer):
    def __init__(self, loader_params, dataset_params):
        super().__init__()
        self.loader_params = AttrDict(loader_params)
        self.dataset_params = AttrDict(dataset_params)

        self.mask_transform = None
        self.image_transform = None

        self.image_augment_train = None
        self.image_augment_inference = None
        self.image_augment_with_target_train = None
        self.image_augment_with_target_inference = None

        self.dataset = ImageSegmentationDataset

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
                                   image_augment=self.image_augment_train,
                                   image_augment_with_target=self.image_augment_with_target_train,
                                   mask_transform=self.mask_transform,
                                   image_transform=self.image_transform,
                                   image_source=self.dataset_params.image_source)
        else:
            dataset = self.dataset(X, y,
                                   train_mode=False,
                                   image_augment=self.image_augment_inference,
                                   image_augment_with_target=self.image_augment_with_target_inference,
                                   mask_transform=self.mask_transform,
                                   image_transform=self.image_transform,
                                   image_source=self.dataset_params.image_source)

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


class ImageSegmentationLoaderCropPad(ImageSegmentationLoaderBasic):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)

        self.image_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(mean=MEAN, std=STD),
                                                   ])
        self.mask_transform = transforms.Compose([transforms.Lambda(to_array),
                                                  transforms.Lambda(to_tensor),
                                                  ])
        self.image_augment_train = ImgAug(color_seq)
        self.image_augment_with_target_train = ImgAug(crop_seq(crop_size=(self.dataset_params.h, self.dataset_params.w)))
        self.image_augment_with_target_inference = ImgAug(pad_to_fit_net(self.dataset_params.divisor, self.dataset_params.pad_method))


class ImageSegmentationLoaderResize(ImageSegmentationLoaderBasic):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)

        self.image_transform = transforms.Compose([transforms.Resize((self.dataset_params.h, self.dataset_params.w)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=MEAN, std=STD),
                                                   ])
        self.mask_transform = transforms.Compose([transforms.Resize((self.dataset_params.h, self.dataset_params.w),
                                                                    interpolation=0),
                                                  transforms.Lambda(to_array),
                                                  transforms.Lambda(to_tensor),
                                                  ])
        self.image_augment_train = ImgAug(color_seq)
        self.image_augment_with_target_train = ImgAug(affine_seq)

        self.dataset = ImageSegmentationDataset


def to_array(x):
    x_ = x.convert('L')  # convert image to monochrome
    x_ = np.array(x_)
    x_ = x_.astype(np.float32)
    return x_


def to_tensor(x):
    x_ = np.expand_dims(x, axis=0)
    x_ = torch.from_numpy(x_)
    return x_

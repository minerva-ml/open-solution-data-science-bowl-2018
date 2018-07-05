import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from attrdict import AttrDict
from sklearn.externals import joblib
from torch.utils.data import Dataset, DataLoader
from imgaug import augmenters as iaa
from functools import partial
from itertools import product
import multiprocessing as mp
from scipy.stats import gmean

from .steppy.base import BaseTransformer
from .steppy.pytorch.utils import ImgAug, reseed

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
        self.image_augment = image_augment if image_augment is not None else ImgAug(iaa.Noop())
        self.image_augment_with_target = image_augment_with_target if image_augment_with_target is not None else ImgAug(iaa.Noop())

        self.image_source = image_source

    def __len__(self):
        if self.image_source == 'memory':
            return len(self.X[0])
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
            Xi = from_pil(Xi)
            Xi = self.image_augment(Xi)
            Xi = to_pil(Xi)

            if self.image_transform is not None:
                Xi = self.image_transform(Xi)
            return Xi

    def load_from_memory(self, data_source, index):
        return data_source[0][index]

    def load_from_disk(self, data_source, index):
        img_filepath = data_source[index]
        return self.load_image(img_filepath)

    def load_image(self, img_filepath):
        image = Image.open(img_filepath, 'r')
        return image.convert('RGB')


class ImageSegmentationTTADataset(ImageSegmentationDataset):
    def __init__(self, tta_params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tta_params = tta_params

    def __getitem__(self, index):
        if self.image_source == 'memory':
            load_func = self.load_from_memory
        elif self.image_source == 'disk':
            load_func = self.load_from_disk
        else:
            raise NotImplementedError("Possible loading options: 'memory' and 'disk'!")

        Xi = load_func(self.X, index)
        Xi = from_pil(Xi)

        if self.image_augment is not None:
            Xi = self.image_augment(Xi)

        if self.tta_params is not None:
            tta_transform_specs = self.tta_params[index]
            Xi = test_time_augmentation_transform(Xi, tta_transform_specs)
        Xi = to_pil(Xi)

        if self.image_transform is not None:
            Xi = self.image_transform(Xi)

        return Xi


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

        self.dataset = None

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


class ImageSegmentationLoaderBasicTTA(ImageSegmentationLoaderBasic):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)

    def transform(self, X, tta_params, **kwargs):
        flow, steps = self.get_datagen(X, tta_params, self.loader_params.inference)
        valid_flow = None
        valid_steps = None
        return {'datagen': (flow, steps),
                'validation_datagen': (valid_flow, valid_steps)}

    def get_datagen(self, X, tta_params, loader_params):
        dataset = self.dataset(tta_params=tta_params,
                               X=X,
                               y=None,
                               train_mode=False,
                               image_augment=self.image_augment_inference,
                               image_augment_with_target=self.image_augment_with_target_inference,
                               mask_transform=self.mask_transform,
                               image_transform=self.image_transform,
                               image_source=self.dataset_params.image_source)

        datagen = DataLoader(dataset, **loader_params)
        steps = len(datagen)
        return datagen, steps


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
        self.image_augment_with_target_train = ImgAug(
            crop_seq(crop_size=(self.dataset_params.h, self.dataset_params.w)))
        self.image_augment_inference = ImgAug(
            pad_to_fit_net(self.dataset_params.divisor, self.dataset_params.pad_method))
        self.image_augment_with_target_inference = ImgAug(
            pad_to_fit_net(self.dataset_params.divisor, self.dataset_params.pad_method))

        self.dataset = ImageSegmentationDataset


class ImageSegmentationLoaderCropPadTTA(ImageSegmentationLoaderBasicTTA):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)

        self.image_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(mean=MEAN, std=STD),
                                                   ])
        self.mask_transform = transforms.Compose([transforms.Lambda(to_array),
                                                  transforms.Lambda(to_tensor),
                                                  ])
        self.image_augment_inference = ImgAug(
            pad_to_fit_net(self.dataset_params.divisor, self.dataset_params.pad_method))
        self.image_augment_with_target_inference = ImgAug(
            pad_to_fit_net(self.dataset_params.divisor, self.dataset_params.pad_method))

        self.dataset = ImageSegmentationTTADataset


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


class ImageSegmentationLoaderResizeTTA(ImageSegmentationLoaderBasicTTA):
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

        self.dataset = ImageSegmentationTTADataset


class MetaTestTimeAugmentationGenerator(BaseTransformer):
    def __init__(self, **kwargs):
        self.tta_transformations = AttrDict(kwargs)

    def transform(self, X, **kwargs):
        X_tta_rows, tta_params, img_ids = [], [], []
        for i in range(len(X)):
            rows, params, ids = self._get_tta_data(i, X[i])
            tta_params.extend(params)
            img_ids.extend(ids)
            X_tta_rows.extend(rows)
        X_tta = np.array(X_tta_rows)
        return {'X_tta': X_tta, 'tta_params': tta_params, 'img_ids': img_ids}

    def _get_tta_data(self, i, row):
        original_specs = {'ud_flip': False, 'lr_flip': False, 'rotation': 0, 'color_shift': False}
        tta_specs = [original_specs]

        ud_options = [True, False] if self.tta_transformations.flip_ud else [False]
        lr_options = [True, False] if self.tta_transformations.flip_lr else [False]
        rot_options = [0, 90, 180, 270] if self.tta_transformations.rotation else [0]
        if self.tta_transformations.color_shift_runs:
            color_shift_options = list(range(1, self.tta_transformations.color_shift_runs + 1, 1))
        else:
            color_shift_options = [False]

        for ud, lr, rot, color in product(ud_options, lr_options, rot_options, color_shift_options):
            if ud is False and lr is False and rot == 0 and color is False:
                continue
            else:
                tta_specs.append({'ud_flip': ud, 'lr_flip': lr, 'rotation': rot, 'color_shift': color})

        img_ids = [i] * len(tta_specs)
        X_rows = [row] * len(tta_specs)
        return X_rows, tta_specs, img_ids


class TestTimeAugmentationGenerator(BaseTransformer):
    def __init__(self, **kwargs):
        self.tta_transformations = AttrDict(kwargs)

    def transform(self, X, **kwargs):
        X_tta, tta_params, img_ids = [], [], []
        X = X[0]
        for i in range(len(X)):
            images, params, ids = self._get_tta_data(i, X[i])
            tta_params.extend(params)
            img_ids.extend(ids)
            X_tta.extend(images)
        return {'X_tta': [X_tta], 'tta_params': tta_params, 'img_ids': img_ids}

    def _get_tta_data(self, i, row):
        original_specs = {'ud_flip': False, 'lr_flip': False, 'rotation': 0, 'color_shift': False}
        tta_specs = [original_specs]

        ud_options = [True, False] if self.tta_transformations.flip_ud else [False]
        lr_options = [True, False] if self.tta_transformations.flip_lr else [False]
        rot_options = [0, 90, 180, 270] if self.tta_transformations.rotation else [0]
        if self.tta_transformations.color_shift_runs:
            color_shift_options = list(range(1, self.tta_transformations.color_shift_runs + 1, 1))
        else:
            color_shift_options = [False]

        for ud, lr, rot, color in product(ud_options, lr_options, rot_options, color_shift_options):
            if ud is False and lr is False and rot == 0 and color is False:
                continue
            else:
                tta_specs.append({'ud_flip': ud, 'lr_flip': lr, 'rotation': rot, 'color_shift': color})

        img_ids = [i] * len(tta_specs)
        X_rows = [row] * len(tta_specs)
        return X_rows, tta_specs, img_ids


class TestTimeAugmentationAggregator(BaseTransformer):
    def __init__(self, method, nthreads):
        self.method = method
        self.nthreads = nthreads

    @property
    def agg_method(self):
        methods = {'mean': np.mean,
                   'max': np.max,
                   'min': np.min,
                   'gmean': gmean
                   }
        return partial(methods[self.method], axis=-1)

    def transform(self, images, tta_params, img_ids, **kwargs):
        _aggregate_augmentations = partial(aggregate_augmentations,
                                           images=images,
                                           tta_params=tta_params,
                                           img_ids=img_ids,
                                           agg_method=self.agg_method)
        unique_img_ids = set(img_ids)
        threads = min(self.nthreads, len(unique_img_ids))
        with mp.pool.ThreadPool(threads) as executor:
            averages_images = executor.map(_aggregate_augmentations, unique_img_ids)
        return {'aggregated_prediction': averages_images}


def aggregate_augmentations(img_id, images, tta_params, img_ids, agg_method):
    tta_predictions_for_id = []
    for image, tta_param, ids in zip(images, tta_params, img_ids):
        if ids == img_id:
            tta_prediction = test_time_augmentation_inverse_transform(image, tta_param)
            tta_predictions_for_id.append(tta_prediction)
        else:
            continue
    tta_averaged = agg_method(np.stack(tta_predictions_for_id, axis=-1))
    return tta_averaged


def test_time_augmentation_transform(image, tta_parameters):
    if tta_parameters['ud_flip']:
        image = np.flipud(image)
    if tta_parameters['lr_flip']:
        image = np.fliplr(image)
    if tta_parameters['color_shift']:
        random_color_shift = reseed(color_seq, deterministic=False)
        image = random_color_shift.augment_image(image)
    image = rotate(image, tta_parameters['rotation'])
    return image


def test_time_augmentation_inverse_transform(image, tta_parameters):
    image = per_channel_rotation(image.copy(), -1 * tta_parameters['rotation'])

    if tta_parameters['lr_flip']:
        image = per_channel_fliplr(image.copy())
    if tta_parameters['ud_flip']:
        image = per_channel_flipud(image.copy())
    return image


def per_channel_flipud(x):
    x_ = x.copy()
    for i, channel in enumerate(x):
        x_[i, :, :] = np.flipud(channel)
    return x_


def per_channel_fliplr(x):
    x_ = x.copy()
    for i, channel in enumerate(x):
        x_[i, :, :] = np.fliplr(channel)
    return x_


def per_channel_rotation(x, angle):
    return rotate(x, angle, axes=(1, 2))


def rotate(image, angle, axes=(0, 1)):
    if angle % 90 != 0:
        raise Exception('Angle must be a multiple of 90.')
    k = angle // 90
    return np.rot90(image, k, axes=axes)


def to_array(x):
    x_ = x.convert('L')  # convert image to monochrome
    x_ = np.array(x_)
    x_ = x_.astype(np.float32)
    return x_


def to_tensor(x):
    x_ = np.expand_dims(x, axis=0)
    x_ = torch.from_numpy(x_)
    return x_

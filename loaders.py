from itertools import product
import math

from attrdict import AttrDict
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.externals import joblib
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from augmentation import affine_seq, color_seq, patching_seq
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


class ImageSegmentationLoaderBasic(BaseTransformer):
    def __init__(self, loader_params, dataset_params):
        super().__init__()
        self.loader_params = AttrDict(loader_params)
        self.dataset_params = AttrDict(dataset_params)

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
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)


class ImageSegmentationLoaderPatchingTrain(ImageSegmentationLoaderBasic):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)

        self.image_augment_with_target = ImgAug(patching_seq(crop_size=(self.dataset_params.h,
                                                                        self.dataset_params.w)))
        self.image_augment = ImgAug(color_seq)

        self.dataset = None


class ImageSegmentationLoaderPatchingInference(ImageSegmentationLoaderBasic):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)

        self.image_augment_with_target = ImgAug(patching_seq(crop_size=(self.dataset_params.h,
                                                                        self.dataset_params.w)))
        self.image_augment = ImgAug(color_seq)

        self.dataset = None

    def transform(self, X, y, X_valid=None, y_valid=None, train_mode=True):
        X, patch_ids = self.get_patches(X)

        flow, steps = self.get_datagen(X, None, False, self.loader_params.inference)
        valid_flow = None
        valid_steps = None
        return {'datagen': (flow, steps),
                'patch_ids': patch_ids,
                'validation_datagen': (valid_flow, valid_steps)}

    def get_datagen(self, X, y, train_mode, loader_params):
        dataset = self.dataset(X, None,
                               train_mode=False,
                               image_augment=None,
                               image_augment_with_target=None,
                               mask_transform=self.mask_transform,
                               image_transform=self.image_transform)

        datagen = DataLoader(dataset, **loader_params)
        steps = len(datagen)
        return datagen, steps

    def get_patches(self, X):
        patches, patch_ids, tta_angles, patch_y_coords, patch_x_coords, image_h, image_w = [], [], [], [], [], [], []
        for i, image in enumerate((X[0])):
            image = from_pil(image)
            h, w = image.shape[:2]
            for y_coord, x_coord, image_patch in generate_patches(image, self.dataset_params.h,
                                                                  self.dataset_params.patching_stride):
                for tta_rotation_angle, image_patch_tta in test_time_augmentation(image_patch):
                    image_patch_tta = to_pil(image_patch_tta)
                    patches.append(image_patch_tta)
                    patch_ids.append(i)
                    tta_angles.append(tta_rotation_angle)
                    patch_y_coords.append(y_coord)
                    patch_x_coords.append(x_coord)
                    image_h.append(h)
                    image_w.append(w)

        patch_ids = pd.DataFrame({'patch_ids': patch_ids,
                                  'tta_angles': tta_angles,
                                  'y_coordinates': patch_y_coords,
                                  'x_coordinates': patch_x_coords,
                                  'image_h': image_h,
                                  'image_w': image_w})
        return [patches], patch_ids


class MetadataImageSegmentationLoader(ImageSegmentationLoaderBasic):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)
        self.dataset = MetadataImageSegmentationDataset


class MetadataImageSegmentationMultitaskLoader(ImageSegmentationLoaderBasic):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)
        self.dataset = MetadataImageSegmentationMultitaskDataset


class ImageSegmentationLoader(ImageSegmentationLoaderBasic):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)
        self.dataset = ImageSegmentationDataset


class ImageSegmentationMultitaskLoader(ImageSegmentationLoaderBasic):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)
        self.dataset = ImageSegmentationMultitaskDataset


class ImageSegmentationMultitaskLoaderPatchingTrain(ImageSegmentationLoaderPatchingTrain):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)
        self.dataset = ImageSegmentationMultitaskDataset


class ImageSegmentationMultitaskLoaderPatchingInference(ImageSegmentationLoaderPatchingInference):
    def __init__(self, loader_params, dataset_params):
        super().__init__(loader_params, dataset_params)
        self.dataset = ImageSegmentationMultitaskDataset


class PatchCombiner(BaseTransformer):
    def __init__(self, patching_size, patching_stride):
        super().__init__()
        self.patching_size = patching_size
        self.patching_stride = patching_stride

    def transform(self, outputs, patch_ids):
        combined_outputs = {}
        for name, output in outputs.items():
            for patch_id in patch_ids['patch_ids'].unique():
                patch_meta = patch_ids[patch_ids['patch_ids'] == patch_id]
                image_patches = output[patch_meta.index]
                combined_outputs.setdefault(name, []).append(self._join_output(patch_meta, image_patches))
        return combined_outputs

    def _join_output(self, patch_meta, image_patches):
        image_h = patch_meta['image_h'].unique()[0]
        image_w = patch_meta['image_w'].unique()[0]
        prediction_image = np.zeros((image_h, image_w))
        prediction_image_padded = get_padded_image(prediction_image, self.patching_size)

        for (y_coordinate, x_coordinate, tta_angle), image_patch in zip(
                patch_meta[['y_coordinates', 'x_coordinates', 'tta_angles']].values.tolist(), image_patches):
            image_patch = np.rot90(image_patch, -1 * tta_angle / 90.)
            prediction_image_padded[
            y_coordinate * self.patching_stride:y_coordinate * self.patching_stride + self.patching_size,
            x_coordinate * self.patching_stride:x_coordinate * self.patching_stride + self.patching_size] += image_patch

        prediction_image = prediction_image_padded[self.patching_size:image_h + self.patching_size,
                           self.patching_size:image_w + self.patching_size]
        return prediction_image


def binarize(x):
    x_ = x.convert('L')  # convert image to monochrome
    x_ = np.array(x_)
    x_ = (x_ > 125).astype(np.float32)
    return x_


def to_tensor(x):
    x_ = np.expand_dims(x, axis=0)
    x_ = torch.from_numpy(x_)
    return x_


def test_time_augmentation(img):
    for i in range(4):
        yield i*90, np.rot90(img, i)


def generate_patches(img, patch_size, overlap):
    img_padded = get_padded_image(img, patch_size)
    h_pad, w_pad = img_padded.shape[:2]
    h_patch_nr = int(h_pad / overlap)
    w_patch_nr = int(w_pad / overlap)

    for y_coordinate, x_coordinate in product(range(h_patch_nr - 1), range(w_patch_nr - 1)):
        img_patch = img_padded[y_coordinate * overlap:y_coordinate * overlap + patch_size,
                    x_coordinate * overlap:x_coordinate * overlap + patch_size, :]
        yield y_coordinate, x_coordinate, img_patch


def get_padded_image(img, patch_size):
    if len(img.shape) == 2:
        h, w = img.shape

        h_pad = math.ceil((h + 2 * patch_size) / patch_size) * patch_size
        w_pad = math.ceil((w + 2 * patch_size) / patch_size) * patch_size

        img_padded = np.zeros((h_pad, w_pad))
        img_padded[patch_size:h + patch_size, patch_size:w + patch_size] = img
        return img_padded
    else:
        h, w, c = img.shape

        h_pad = math.ceil((h + 2 * patch_size) / patch_size) * patch_size
        w_pad = math.ceil((w + 2 * patch_size) / patch_size) * patch_size

        img_padded = np.zeros((h_pad, w_pad, c))
        img_padded[patch_size:h + patch_size, patch_size:w + patch_size, :] = img
        return img_padded

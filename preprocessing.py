import numpy as np
import scipy.ndimage as ndi
from PIL import Image
from skimage.transform import resize
from sklearn.externals import joblib
from tqdm import tqdm

from steps.base import BaseTransformer
from preparation import get_contour
from utils import from_pil, to_pil, clip


class ImageReader(BaseTransformer):
    def __init__(self, x_columns, y_columns):
        self.x_columns = x_columns
        self.y_columns = y_columns

    def transform(self, meta, train_mode, meta_valid=None):
        X, y = self._transform(meta, train_mode)
        if meta_valid is not None:
            X_valid, y_valid = self._transform(meta_valid, train_mode)
        else:
            X_valid, y_valid = None, None

        return {'X': X,
                'y': y,
                'X_valid': X_valid,
                'y_valid': y_valid}

    def _transform(self, meta, train_mode):
        X_ = meta[self.x_columns].values

        X = self.load_images(X_, grayscale=False)
        if train_mode:
            y_ = meta[self.y_columns].values
            y = self.load_images(y_, grayscale=True)
        else:
            y = None

        return X, y

    def load_images(self, image_filepaths, grayscale):
        X = []
        for i in range(image_filepaths.shape[1]):
            column = image_filepaths[:, i]
            X.append([])
            for img_filepath in tqdm(column):
                img = self.load_image(img_filepath, grayscale=grayscale)
                X[i].append(img)
        return X

    def load_image(self, img_filepath, grayscale):
        image = Image.open(img_filepath, 'r')
        if not grayscale:
            image = image.convert('RGB')
        else:
            image = image.convert('L')
        return image

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


class ImageReaderRescaler(BaseTransformer):
    def __init__(self, min_size, max_size, target_ratio):
        self.min_size = min_size
        self.max_size = max_size
        self.target_ratio = target_ratio

    def transform(self, sizes, X, y, sizes_valid, X_valid, y_valid):
        X, y = self._transform(sizes, X, y)
        if X_valid is not None and y_valid is not None:
            X_valid, y_valid = self._transform(sizes_valid, X_valid, y_valid)
        else:
            X_valid, y_valid = None, None

        return {'X': X,
                'y': y,
                'X_valid': X_valid,
                'y_valid': y_valid}

    def load(self, filepath):
        return self

    def save(self, filepath):
        params = {}
        joblib.dump(params, filepath)

    def _transform(self, sizes, X, y=None):
        raw_images = X[0]
        masks, contours, centers = y

        raw_images_adj, masks_adj, contours_adj, centers_adj = [], [], [], []
        for size, raw_image, mask, contour, center in zip(sizes, raw_images, masks, contours, centers):
            raw_image_adj = self._adjust_image_size(size, from_pil(raw_image))
            mask_adj = self._adjust_image_size(size, from_pil(mask))
            contour_adj = self._get_contour(from_pil(mask_adj))
            center_adj = self._adjust_image_size(size, from_pil(center))

            raw_images_adj.append(to_pil(raw_image_adj))
            masks_adj.append(to_pil(mask_adj))
            contours_adj.append(to_pil(contour_adj))
            centers_adj.append(to_pil(center_adj))

        X_adj = [raw_images_adj]
        y_adj = [masks_adj, contours_adj, centers_adj]

        joblib.dump(contours, '/mnt/ml-team/dsb_2018/kuba/debug/contours_pre_resize.pkl')
        joblib.dump(contours_adj, '/mnt/ml-team/dsb_2018/kuba/debug/contours_post_resize.pkl')
        return X_adj, y_adj

    def _adjust_image_size(self, mean_cell_size, img):
        h, w = img.shape[:2]
        img_area = h * w

        size_ratio = img_area / mean_cell_size
        adj_ratio = size_ratio / self.target_ratio

        h_adj = int(clip(self.min_size, h * adj_ratio, self.max_size))
        w_adj = int(clip(self.min_size, w * adj_ratio, self.max_size))

        img_adj = resize(img, (h_adj, w_adj), preserve_range=True).astype(np.uint8)

        return img_adj

    def _get_contour(self, mask):
        # Todo: This should be done on each mask individually
        labels, nr_labels = ndi.label(mask)

        label_contours = np.zeros_like(mask).astype(np.uint8)
        for label in range(1, nr_labels + 1):
            label_mask = np.where(labels == label, 1, 0)
            label_contour = get_contour(label_mask)
            label_contour_inside = np.where((label_contour != 0) & (label_mask != 0), 1, 0).astype(np.uint8)
            label_contours += label_contour_inside

        label_contours = np.where(label_contours > 0, 255, 0).astype(np.uint8)
        return label_contours

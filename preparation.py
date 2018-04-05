import glob
import os

import cv2
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import torch
from PIL import Image
from imageio import imwrite
from skimage.transform import resize
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from torchvision import models
from tqdm import tqdm


def train_valid_split(meta, validation_size, valid_category_ids=None, simple_split=False):
    meta_train = meta[meta['is_train'] == 1]
    
    if simple_split:
        meta_train_splittable = meta_train[meta_train['is_external'] == 0]
        external_data = meta_train[meta_train['is_external'] == 1]
        meta_train_split, meta_valid_split = train_test_split(meta_train_splittable,
                                                              test_size=validation_size,
                                                              random_state=1234)
        meta_train_split = pd.concat([meta_train_split, external_data], axis=0).sample(frac=1, random_state=1234)
    else:
        meta_train_split, meta_valid_split = split_on_column(meta_train,
                                                             column='vgg_features_clusters',
                                                             test_size=validation_size,
                                                             random_state=1234,
                                                             valid_category_ids=valid_category_ids
                                                             )
    return meta_train_split, meta_valid_split


def split_on_column(meta, column, test_size, random_state=1, valid_category_ids=None):
    if valid_category_ids is None:
        categories = meta[column].unique()
        np.random.seed(random_state)
        valid_category_ids = np.random.choice(categories, int(test_size * len(categories)))
    valid = meta[meta[column].isin(valid_category_ids)].sample(frac=1, random_state=random_state)
    train = meta[~(meta[column].isin(valid_category_ids))].sample(frac=1, random_state=random_state)
    return train, valid


def overlay_masks(images_dir, subdir_name, target_dir):
    train_dir = os.path.join(images_dir, subdir_name)
    for mask_dirname in tqdm(glob.glob('{}/*/masks'.format(train_dir))):
        overlayed_masks = overlay_masks_from_dir(mask_dirname)
        target_filepath = '/'.join(mask_dirname.replace(images_dir, target_dir).split('/')[:-1]) + '.png'
        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        imwrite(target_filepath, overlayed_masks)


def overlay_contours(images_dir, subdir_name, target_dir):
    train_dir = os.path.join(images_dir, subdir_name)
    for mask_dirname in tqdm(glob.glob('{}/*/masks'.format(train_dir))):
        overlayed_masks = overlay_contours_from_dir(mask_dirname)
        target_filepath = '/'.join(mask_dirname.replace(images_dir, target_dir).split('/')[:-1]) + '.png'
        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        imwrite(target_filepath, overlayed_masks)


def overlay_centers(images_dir, subdir_name, target_dir):
    train_dir = os.path.join(images_dir, subdir_name)
    for mask_dirname in tqdm(glob.glob('{}/*/masks'.format(train_dir))):
        overlayed_masks = overlay_centers_from_dir(mask_dirname)
        target_filepath = '/'.join(mask_dirname.replace(images_dir, target_dir).split('/')[:-1]) + '.png'
        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        imwrite(target_filepath, overlayed_masks)


def overlay_contours_from_dir(mask_dirname):
    masks = []
    for image_filepath in glob.glob('{}/*'.format(mask_dirname)):
        image = np.asarray(Image.open(image_filepath))
        image = ndi.binary_fill_holes(image)
        contour = get_contour(image)
        inside_contour = np.where(image & contour, 255, 0)
        masks.append(inside_contour)
    overlayed_masks = np.where(np.sum(masks, axis=0) > 128., 255., 0.).astype(np.uint8)
    return overlayed_masks


def overlay_masks_from_dir(mask_dirname):
    masks = []
    for image_filepath in glob.glob('{}/*'.format(mask_dirname)):
        image = np.asarray(Image.open(image_filepath))
        image = ndi.binary_fill_holes(image) * 255.
        masks.append(image)
    overlayed_masks = np.where(np.sum(masks, axis=0) > 128., 255., 0.).astype(np.uint8)
    return overlayed_masks


def overlay_centers_from_dir(mask_dirname):
    masks = []
    for image_filepath in glob.glob('{}/*'.format(mask_dirname)):
        image = np.asarray(Image.open(image_filepath))
        image = ndi.binary_fill_holes(image)
        masks.append(get_center(image))
    overlayed_masks = np.where(np.sum(masks, axis=0) > 128., 255., 0.).astype(np.uint8)
    return overlayed_masks


def get_contour(img):
    img_contour = np.zeros_like(img).astype(np.uint8)
    _, contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_contour, contours, -1, (255, 255, 255), 8)
    return img_contour


def get_center(img):
    if img.max() == 0:
        return img
    else:
        img_center = np.zeros_like(img).astype(np.uint8)
        y, x = ndi.measurements.center_of_mass(img)
        cv2.circle(img_center, (int(x), int(y)), 4, (255, 255, 255), -1)
        return img_center


def get_vgg_clusters(meta):
    img_filepaths = meta['file_path_image'].values

    extractor = vgg_extractor()

    features = []
    for filepath in tqdm(img_filepaths):
        img = np.asarray(Image.open(filepath))[:, :, :3]
        img = img / 255.0
        x = preprocess_image(img)
        feature = extractor(x)
        feature = np.ndarray.flatten(feature.cpu().data.numpy())
        features.append(feature)
    features = np.stack(features, axis=0)

    labels = cluster_features(features)

    return labels


def vgg_extractor():
    model = models.vgg16(pretrained=True)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return torch.nn.Sequential(*list(model.features.children())[:-1])


def preprocess_image(img, target_size=(128, 128)):
    img = resize(img, target_size, mode='constant')
    x = np.expand_dims(img, axis=0)
    x = x.transpose(0, 3, 1, 2)
    x = torch.FloatTensor(x)
    if torch.cuda.is_available():
        x = torch.autograd.Variable(x, volatile=True).cuda()
    else:
        x = torch.autograd.Variable(x, volatile=True)
    return x


def cluster_features(features, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=1111)
    kmeans.fit(features)
    labels = kmeans.labels_
    return labels


def build_external_dataset_metadata(external_data_dir):
    external_data_info = {}
    for subdir in os.listdir(external_data_dir):
        external_data_info.setdefault('n_nuclei', []).append('NaN')
        external_data_info.setdefault('is_external', []).append(1)
        external_data_info.setdefault('ImageId', []).append(subdir)
        external_data_info.setdefault('is_train', []).append(1)

        file_path_image = glob.glob('{}/{}/images/*'.format(external_data_dir, subdir))[0]
        file_path_masks = os.path.join(external_data_dir, subdir, 'masks')
        file_path_mask = glob.glob('{}/{}/masks_overlayed/*'.format(external_data_dir, subdir))[0]
        file_path_contour = glob.glob('{}/{}/contours_overlayed/*'.format(external_data_dir, subdir))[0]
        file_path_center = glob.glob('{}/{}/centers_overlayed/*'.format(external_data_dir, subdir))[0]

        external_data_info.setdefault('file_path_image', []).append(file_path_image)
        external_data_info.setdefault('file_path_masks', []).append(file_path_masks)
        external_data_info.setdefault('file_path_mask', []).append(file_path_mask)
        external_data_info.setdefault('file_path_contours', []).append(file_path_contour)
        external_data_info.setdefault('file_path_centers', []).append(file_path_center)
        external_data_info.setdefault('file_path_contours_touching', []).append(file_path_contour_touching)

        img = plt.imread(file_path_image)
        h, w = img.shape[:2]

        external_data_info.setdefault('width', []).append(w)
        external_data_info.setdefault('height', []).append(h)

    external_data_info = pd.DataFrame(external_data_info)
    return external_data_info

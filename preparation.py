import glob
import os

import cv2
import numpy as np
import scipy.ndimage as ndi
import torch
from PIL import Image
from imageio import imwrite
from skimage.transform import resize
from sklearn.cluster import KMeans
from torchvision import models
from tqdm import tqdm


def train_valid_split(meta, validation_size, valid_category_ids=None):
    meta_train = meta[meta['is_train'] == 1]
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
        valid_category_ids = np.random.choice(categories,
                                              int(test_size * len(categories)))
    valid = meta[meta[column].isin(valid_category_ids)].sample(frac=1, random_state=random_state)
    train = meta[~(meta[column].isin(valid_category_ids))].sample(frac=1, random_state=random_state)
    return train, valid


def overlay_masks(images_dir, subdir_name, target_dir):
    train_dir = os.path.join(images_dir, subdir_name)
    for mask_dirname in tqdm(glob.glob('{}/*/masks'.format(train_dir))):
        masks = []
        for image_filepath in glob.glob('{}/*'.format(mask_dirname)):
            image = np.asarray(Image.open(image_filepath))
            image = image / 255.0
            masks.append(image)
        overlayed_masks = np.sum(masks, axis=0)
        target_filepath = '/'.join(mask_dirname.replace(images_dir, target_dir).split('/')[:-1]) + '.png'
        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        imwrite(target_filepath, overlayed_masks)


def overlay_contours(images_dir, subdir_name, target_dir, touching_only=False):
    train_dir = os.path.join(images_dir, subdir_name)
    for mask_dirname in tqdm(glob.glob('{}/*/masks'.format(train_dir))):
        masks = []
        for image_filepath in glob.glob('{}/*'.format(mask_dirname)):
            image = np.asarray(Image.open(image_filepath))
            image = image / 255.0
            masks.append(get_contour(image))
        if touching_only:
            overlayed_masks = np.where(np.sum(masks, axis=0) > 128. + 255., 255., 0.).astype(np.uint8)
        else:
            overlayed_masks = np.where(np.sum(masks, axis=0) > 128., 255., 0.).astype(np.uint8)
        target_filepath = '/'.join(mask_dirname.replace(images_dir, target_dir).split('/')[:-1]) + '.png'
        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        imwrite(target_filepath, overlayed_masks)


def overlay_centers(images_dir, subdir_name, target_dir):
    train_dir = os.path.join(images_dir, subdir_name)
    for mask_dirname in tqdm(glob.glob('{}/*/masks'.format(train_dir))):
        masks = []
        for image_filepath in glob.glob('{}/*'.format(mask_dirname)):
            image = np.asarray(Image.open(image_filepath))
            image = image / 255.0
            masks.append(get_center(image))
        overlayed_masks = np.where(np.sum(masks, axis=0) > 128., 255., 0.).astype(np.uint8)
        target_filepath = '/'.join(mask_dirname.replace(images_dir, target_dir).split('/')[:-1]) + '.png'
        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        imwrite(target_filepath, overlayed_masks)


def get_contour(img):
    img_contour = np.zeros_like(img).astype(np.uint8)
    _, contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_contour, contours, -1, (255, 255, 255), 4)
    return img_contour


def get_center(img):
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

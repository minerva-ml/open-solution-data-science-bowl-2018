import glob
import os

import cv2
import numpy as np
import scipy.ndimage as ndi
import torch
from PIL import Image
from cv2 import imwrite
from skimage.transform import resize
from skimage.morphology import watershed, dilation, rectangle
from sklearn.cluster import KMeans
from torchvision import models
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def train_valid_split(meta, validation_size, random_state=None):
    meta_train = meta[meta['is_train'] == 1]

    meta_train_split, meta_valid_split = train_test_split(meta_train,
                                                          test_size=validation_size,
                                                          random_state=random_state)

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
            image = np.where(image > 0, 1, 0)
            masks.append(image)
        overlayed_masks = np.sum(masks, axis=0)
        target_filepath = '/'.join(mask_dirname.replace(images_dir, target_dir).split('/')[:-1]) + '.png'
        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        imwrite(target_filepath, overlayed_masks)


def overlay_cut_masks(images_dir, subdir_name, target_dir, cut_size=1):
    train_dir = os.path.join(images_dir, subdir_name)
    for mask_dirname in tqdm(glob.glob('{}/*/masks'.format(train_dir))):
        masks = []
        for ind, image_filepath in enumerate(glob.glob('{}/*'.format(mask_dirname))):
            image = np.asarray(Image.open(image_filepath))
            image = np.where(image > 0, ind + 1, 0)
            masks.append(image)
        labeled_masks = np.sum(masks, axis=0)
        overlayed_masks = np.where(labeled_masks, 1, 0)

        watershed_mask = watershed(overlayed_masks.astype(np.bool), labeled_masks, watershed_line=True)
        if watershed_mask.max() == watershed_mask.min():
            cut_masks = overlayed_masks
        else:
            borders = (watershed_mask == 0) & overlayed_masks
            selem = rectangle(cut_size, cut_size)
            dilated_borders = dilation(borders, selem=selem)
            cut_masks = np.where(dilated_borders, 0, overlayed_masks)

        target_filepath = '/'.join(mask_dirname.replace(images_dir, target_dir).split('/')[:-1]) + '.png'
        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        imwrite(target_filepath, cut_masks)
        

def overlay_masks_with_borders(images_dir, subdir_name, target_dir, borders_size=3, dilation_size=5):
    train_dir = os.path.join(images_dir, subdir_name)
    for mask_dirname in tqdm(glob.glob('{}/*/masks'.format(train_dir))):
        masks = []
        for ind, image_filepath in enumerate(glob.glob('{}/*'.format(mask_dirname))):
            image = np.asarray(Image.open(image_filepath))
            image = np.where(image > 0, ind + 1, 0)
            masks.append(image)
        labeled_masks = np.sum(masks, axis=0)
        overlayed_masks = np.where(labeled_masks, 1, 0)

        selem = rectangle(dilation_size, dilation_size)
        dilated_mask = dilation(overlayed_masks, selem=selem)
        watershed_mask = watershed((dilated_mask >= 0).astype(np.bool), labeled_masks, watershed_line=True)

        if watershed_mask.max() == watershed_mask.min():
            masks_with_borders = overlayed_masks
        else:
            borders = (watershed_mask == 0) & (dilated_mask > 0)
            selem = rectangle(borders_size, borders_size)
            dilated_borders = dilation(borders, selem=selem)
            masks_with_borders = np.where(dilated_borders, 2, overlayed_masks)

        target_filepath = '/'.join(mask_dirname.replace(images_dir, target_dir).split('/')[:-1]) + '.png'
        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        imwrite(target_filepath, masks_with_borders)


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

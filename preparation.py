import os
import glob
import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import get_logger

logger = get_logger()


def train_valid_split(meta, validation_size):
    meta_train = meta[meta['is_train'] == 1]
    meta_train_split, meta_valid_split = train_test_split(meta_train, test_size=validation_size, random_state=1234)
    return meta_train_split, meta_valid_split


def overlay_masks(images_dir, subdir_name, target_dir):
    train_dir = os.path.join(images_dir, subdir_name)
    for mask_dirname in tqdm(glob.glob('{}/*/masks'.format(train_dir))):
        masks = []
        for image_filepath in glob.glob('{}/*'.format(mask_dirname)):
            masks.append(plt.imread(image_filepath))
        overlayed_masks = np.sum(masks, axis=0)
        target_filepath = '/'.join(mask_dirname.replace(images_dir, target_dir).split('/')[:-1]) + '.png'
        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        plt.imsave(target_filepath, overlayed_masks)

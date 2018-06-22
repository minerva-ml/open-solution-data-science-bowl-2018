import glob
import logging
import os
import sys
from itertools import product

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from attrdict import AttrDict
from tqdm import tqdm


def read_yaml(filepath):
    with open(filepath) as f:
        config = yaml.load(f)
    return AttrDict(config)


def get_logger():
    logger = logging.getLogger('dsb-2018')
    logger.setLevel(logging.INFO)
    message_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s',
                                       datefmt='%Y-%m-%d %H-%M-%S')

    # console handler for validation info
    ch_va = logging.StreamHandler(sys.stdout)
    ch_va.setLevel(logging.INFO)

    ch_va.setFormatter(fmt=message_format)

    # add the handlers to the logger
    logger.addHandler(ch_va)

    return logger


def decompose(labeled):
    nr_true = labeled.max()
    masks = []
    for i in range(1, nr_true + 1):
        msk = labeled.copy()
        msk[msk != i] = 0.
        msk[msk == i] = 255.
        masks.append(msk)

    if not masks:
        return [labeled]
    else:
        return masks


def create_submission(experiments_dir, meta, predictions, logger):
    image_ids, encodings = [], []
    output = []
    for image_id, prediction in zip(meta['ImageId'].values, predictions):
        for mask in decompose(prediction):
            rle_encoded = ' '.join(str(rle) for rle in run_length_encoding(mask > 128.))
            if len(rle_encoded) != 0:
                image_ids.append(image_id)
                encodings.append(rle_encoded)
                output.append([image_id, rle_encoded])
            else:
                logger.info('*** image_id {}'.format(image_id))
                logger.info('*** rle_encoded {} is empty'.format(rle_encoded))

    submission = pd.DataFrame(output, columns=['ImageId', 'EncodedPixels']).astype(str)
    submission = submission[submission['EncodedPixels'] != 'nan']
    submission_filepath = os.path.join(experiments_dir, 'submission.csv')
    submission.to_csv(submission_filepath, index=None, encoding='utf-8')
    logger.info('submission saved to {}'.format(submission_filepath))
    logger.info('submission head \n\n{}'.format(submission.head()))


def read_masks(masks_filepaths):
    masks = []
    for mask_dir in tqdm(masks_filepaths):
        mask = []
        if len(mask_dir) == 1:
            mask_dir = mask_dir[0]
        for i, mask_filepath in enumerate(glob.glob('{}/*'.format(mask_dir))):
            blob = np.asarray(Image.open(mask_filepath))
            blob_binarized = (blob > 128.).astype(np.uint8) * i
            mask.append(blob_binarized)
        mask = np.sum(np.stack(mask, axis=0), axis=0).astype(np.uint8)
        masks.append(mask)
    return masks


def run_length_encoding(x):
    # https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
    bs = np.where(x.T.flatten())[0]

    rle = []
    prev = -2
    for b in bs:
        if (b > prev + 1): rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b

    if len(rle) != 0 and rle[-1] + rle[-2] == x.size:
        rle[-2] = rle[-2] - 1

    return rle


def read_params(ctx):
    if ctx.params.__class__.__name__ == 'OfflineContextParams':
        neptune_config = read_yaml('neptune.yaml')
        params = neptune_config.parameters
    else:
        params = ctx.params
    return params


def generate_metadata(data_dir,
                      masks_overlayed_dir,
                      contours_overlayed_dir,
                      contours_touching_overlayed_dir,
                      centers_overlayed_dir):
    def stage1_generate_metadata(train):
        df_metadata = pd.DataFrame(columns=['ImageId', 'file_path_image', 'file_path_masks', 'file_path_mask',
                                            'is_train', 'width', 'height', 'n_nuclei'])
        if train:
            tr_te = 'stage1_train'
        else:
            tr_te = 'stage1_test'

        for image_id in sorted(os.listdir(os.path.join(data_dir, tr_te))):
            p = os.path.join(data_dir, tr_te, image_id, 'images')
            if image_id != os.listdir(p)[0][:-4]:
                ValueError('ImageId mismatch ' + str(image_id))
            if len(os.listdir(p)) != 1:
                ValueError('more than one image in dir')

            file_path_image = os.path.join(p, os.listdir(p)[0])
            if train:
                is_train = 1
                file_path_masks = os.path.join(data_dir, tr_te, image_id, 'masks')
                file_path_mask = os.path.join(masks_overlayed_dir, tr_te, image_id + '.png')
                file_path_contours = os.path.join(contours_overlayed_dir, tr_te, image_id + '.png')
                file_path_contours_touching = os.path.join(contours_touching_overlayed_dir, tr_te, image_id + '.png')
                file_path_centers = os.path.join(centers_overlayed_dir, tr_te, image_id + '.png')
                n_nuclei = len(os.listdir(file_path_masks))
            else:
                is_train = 0
                file_path_masks = None
                file_path_mask = None
                file_path_contours = None
                file_path_contours_touching = None
                file_path_centers = None
                n_nuclei = None

            img = Image.open(file_path_image)
            width = img.size[0]
            height = img.size[1]
            s = df_metadata['ImageId']
            if image_id is s:
                ValueError('ImageId conflict ' + str(image_id))
            df_metadata = df_metadata.append({'ImageId': image_id,
                                              'file_path_image': file_path_image,
                                              'file_path_masks': file_path_masks,
                                              'file_path_mask': file_path_mask,
                                              'file_path_contours': file_path_contours,
                                              'file_path_contours_touching': file_path_contours_touching,
                                              'file_path_centers': file_path_centers,
                                              'is_train': is_train,
                                              'width': width,
                                              'height': height,
                                              'n_nuclei': n_nuclei}, ignore_index=True)
        return df_metadata

    train_metadata = stage1_generate_metadata(train=True)
    test_metadata = stage1_generate_metadata(train=False)
    metadata = train_metadata.append(test_metadata, ignore_index=True)
    return metadata


def squeeze_inputs(inputs):
    return np.squeeze(inputs[0], axis=1)


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def relabel(img):
    h, w = img.shape

    relabel_dict = {}

    for i, k in enumerate(np.unique(img)):
        if k == 0:
            relabel_dict[k] = 0
        else:
            relabel_dict[k] = i
    for i, j in product(range(h), range(w)):
        img[i, j] = relabel_dict[img[i, j]]
    return img


def relabel_random_colors(img, max_colours=1000):
    keys = list(range(1, max_colours, 1))
    np.random.shuffle(keys)
    values = list(range(1, max_colours, 1))
    np.random.shuffle(values)
    funky_dict = {k: v for k, v in zip(keys, values)}
    funky_dict[0] = 0

    h, w = img.shape

    for i, j in product(range(h), range(w)):
        img[i, j] = funky_dict[img[i, j]]
    return img


def from_pil(*images):
    return [np.array(image) for image in images]


def to_pil(*images):
    return [Image.fromarray((image).astype(np.uint8)) for image in images]

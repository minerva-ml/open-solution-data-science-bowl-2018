import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from attrdict import AttrDict


def read_yaml(filepath):
    with open(filepath) as f:
        config = yaml.load(f)
    return AttrDict(config)


def init_logger():
    logger = logging.getLogger('dsb-2018')
    logger.setLevel(logging.INFO)
    message_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s',
                                       datefmt='%Y-%m-%d %H-%M-%S')

    # console handler for validation info
    ch_va = logging.StreamHandler()
    ch_va.setLevel(logging.INFO)

    ch_va.setFormatter(fmt=message_format)

    # add the handlers to the logger
    logger.addHandler(ch_va)


def get_logger():
    return logging.getLogger('dsb-2018')


def create_submission(experiments_dir, meta, predictions, logger):
    encoded_predictions = [' '.join(str(p) for p in run_length_encoding(pred)) for pred in predictions]
    submission = pd.DataFrame({'ImageId': meta['ImageId'].values,
                               'EncodedPixels': encoded_predictions})
    submission_filepath = os.path.join(experiments_dir, 'submission.csv')
    submission.to_csv(submission_filepath, index=None)
    logger.info('submission saved to {}'.format(submission_filepath))
    logger.info('submission head \n\n{}'.format(submission.head()))



def read_masks(mask_filepaths):
    masks = []
    for mask_filepath in mask_filepaths:
        mask = plt.imread(mask_filepath[0])[:, :, 0]
        mask_binarized = (mask > 0.5).astype(np.uint8)
        masks.append(mask_binarized)
    return masks


def run_length_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten() == 1)[0]  # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def read_params(ctx):
    params = ctx.params
    # if params.__class__.__name__ == 'OfflineContextParams':
    neptune_config = read_yaml('neptune_config.yaml')
    params = neptune_config.parameters
    return params


def generate_metadata(data_dir, masks_overlayed_dir):
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
                n_nuclei = len(os.listdir(file_path_masks))
            else:
                is_train = 0
                file_path_masks = None
                file_path_mask = None
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
                                              'is_train': is_train,
                                              'width': width,
                                              'height': height,
                                              'n_nuclei': n_nuclei}, ignore_index=True)
        return df_metadata

    train_metadata = stage1_generate_metadata(train=True)
    test_metadata = stage1_generate_metadata(train=False)
    metadata = train_metadata.append(test_metadata, ignore_index=True)
    return metadata

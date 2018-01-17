import logging
import os

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from attrdict import AttrDict
from sklearn.metrics import log_loss


def read_yaml(filepath):
    with open(filepath) as f:
        config = yaml.load(f)
    return AttrDict(config)


def init_logger():
    logger = logging.getLogger('toxic')
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
    return logging.getLogger('toxic')


def read_data(data_dir, filename):
    meta_filepath = os.path.join(data_dir, filename)
    meta_data = pd.read_csv(meta_filepath)
    return meta_data


def create_submission(experiments_dir, meta, predictions, columns, logger):
    submission = meta[['id']]
    predictions_ = pd.DataFrame(predictions, columns=columns)
    submission = pd.concat([submission, predictions_], axis=1)
    logger.info('submission head', submission.head())

    submission_filepath = os.path.join(experiments_dir, 'submission.csv')
    submission.to_csv(submission_filepath, index=None)
    logger.info('submission saved to {}'.format(submission_filepath))


def multi_log_loss(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    columns = y_true.shape[1]
    column_losses = []
    for i in range(0, columns):
        column_losses.append(log_loss(y_true[:, i], y_pred[:, i]))
    return np.array(column_losses).mean()


def generate_metadata(data_dir):
    def stage1_generate_metadata(train):
        df_metadata = pd.DataFrame(columns=['ImageId', 'file_path_image', 'file_path_masks',
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
                n_nuclei = len(os.listdir(file_path_masks))
            else:
                is_train = 0
                file_path_masks = None
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
                                              'is_train': is_train,
                                              'width': width,
                                              'height': height,
                                              'n_nuclei': n_nuclei}, ignore_index=True)
        return df_metadata

    train_metadata = stage1_generate_metadata(train=True)
    test_metadata = stage1_generate_metadata(train=False)
    metadata = train_metadata.append(test_metadata, ignore_index=True)
    return metadata

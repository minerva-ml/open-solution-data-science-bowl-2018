import logging
import os

import numpy as np
import pandas as pd
import yaml
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



def read_meta_data(data_dir, filename):
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


def read_masks(mask_filepaths):
    return NotImplementedError

def intersection_over_union_score(y_true, y_pred):
    return NotImplementedError

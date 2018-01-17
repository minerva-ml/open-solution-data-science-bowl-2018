import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from attrdict import AttrDict


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


def create_submission(experiments_dir, meta, predictions, logger):
    submission = meta[['ImageId']]
    encoded_predictions = [run_length_encoding(pred) for pred in predictions]
    submission['EncodedPixels'] = encoded_predictions
    logger.info('submission head', submission.head())

    submission_filepath = os.path.join(experiments_dir, 'submission.csv')
    submission.to_csv(submission_filepath, index=None)
    logger.info('submission saved to {}'.format(submission_filepath))


def read_masks(mask_filepaths):
    masks = []
    for mask_filepath in mask_filepaths:
        mask = plt.imread(mask_filepath)
        masks.append(mask)
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
    if params.__class__.__name__ == 'OfflineContextParams':
        neptune_config = read_yaml('neptune_config.yaml')
        params = neptune_config.parameters
    return params

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import get_logger

logger = get_logger()


def split_train_data(data_dir, validation_size):
    meta_train_filepath = os.path.join(data_dir, 'train.csv')
    meta_train_split_filepath = meta_train_filepath.replace('train', 'train_split')
    meta_valid_split_filepath = meta_train_filepath.replace('train', 'valid_split')

    logger.info('reading data from {}'.format(meta_train_filepath))
    meta_data = pd.read_csv(meta_train_filepath)
    logger.info('splitting data')
    meta_train, meta_valid = train_test_split(meta_data, test_size=validation_size, random_state=1234)
    logger.info('saving train split data to {}'.format(meta_train_split_filepath))
    meta_train.to_csv(meta_train_split_filepath, index=None)
    logger.info('saving valid split data to {}'.format(meta_valid_split_filepath))
    meta_valid.to_csv(meta_valid_split_filepath, index=None)
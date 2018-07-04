import os
import shutil

import pandas as pd
from deepsense import neptune

from .metrics import intersection_over_union, intersection_over_union_thresholds
from .pipeline_config import SOLUTION_CONFIG, Y_COLUMNS_SCORING, SIZE_COLUMNS, SEED
from .pipelines import PIPELINES
from .preparation import train_valid_split, overlay_masks, overlay_cut_masks, overlay_masks_with_borders
from .utils import init_logger, read_masks, read_masks_from_csv, read_params, create_submission, generate_metadata


class PipelineManager():
    def __init__(self):
        self.logger = init_logger()
        self.ctx = neptune.Context()
        self.params = read_params(self.ctx)

    def prepare_metadata(self):
        prepare_metadata(self.logger, self.params)

    def prepare_masks(self):
        prepare_masks(self.logger, self.params)

    def train(self, pipeline_name, validation_size, dev_mode):
        train(pipeline_name, validation_size, dev_mode, self.logger, self.params)

    def evaluate(self, pipeline_name, validation_size, dev_mode):
        evaluate(pipeline_name, validation_size, dev_mode, self.logger, self.params, self.ctx)

    def predict(self, pipeline_name, dev_mode):
        predict(pipeline_name, dev_mode, self.logger, self.params)


def prepare_metadata(logger, params):
    logger.info('creating metadata')
    meta = generate_metadata(data_dir=params.data_dir,
                             masks_overlayed_dir=params.masks_overlayed_dir,
                             cut_masks_dir=params.cut_masks_dir,
                             masks_with_borders_dir=params.masks_with_borders_dir,
                             )
    meta.to_csv(os.path.join(params.meta_dir, 'stage1_metadata.csv'), index=None)


def prepare_masks(logger, params):
    logger.info('overlaying masks')
    overlay_masks(images_dir=params.data_dir, subdir_name='stage1_train', target_dir=params.masks_overlayed_dir)
    logger.info('cutting masks')
    overlay_cut_masks(images_dir=params.data_dir, subdir_name='stage1_train',
                      target_dir=params.cut_masks_dir, cut_size=2)
    logger.info('masks with borders')
    overlay_masks_with_borders(images_dir=params.data_dir, subdir_name='stage1_train',
                               target_dir=params.masks_with_borders_dir)


def train(pipeline_name, validation_size, dev_mode, logger, params):
    logger.info('training')
    if bool(params.overwrite) and os.path.isdir(params.experiment_dir):
        shutil.rmtree(params.experiment_dir)

    meta = pd.read_csv(os.path.join(params.meta_dir, 'stage1_metadata.csv'))
    meta_train = meta[meta['is_train'] == 1]
    meta_train_split, meta_valid_split = train_valid_split(meta_train, validation_size, random_state=SEED)

    if dev_mode:
        meta_train_split = meta_train_split.sample(params.dev_mode_size, random_state=SEED)
        meta_valid_split = meta_valid_split.sample(int(params.dev_mode_size/2), random_state=SEED)

    data = {'input': {'meta': meta_train_split,
                      'target_sizes': meta_train_split[SIZE_COLUMNS].values},
            'specs': {'train_mode': True},
            'callback_input': {'meta_valid': meta_valid_split}
            }

    pipeline = PIPELINES[pipeline_name]['train'](SOLUTION_CONFIG)
    pipeline.clean_cache()
    pipeline.fit_transform(data)
    pipeline.clean_cache()


def evaluate(pipeline_name, validation_size, dev_mode, logger, params, ctx):
    logger.info('evaluating')
    meta = pd.read_csv(os.path.join(params.meta_dir, 'stage1_metadata.csv'))
    meta_train = meta[meta['is_train'] == 1]

    try:
        validation_size = float(validation_size)
    except ValueError:
        pass

    if isinstance(validation_size, float):
        meta_train_split, meta_valid_split = train_valid_split(meta_train, validation_size, random_state=SEED)
        y_true = read_masks(meta_valid_split[Y_COLUMNS_SCORING].values)
    elif validation_size == 'test':
        meta_valid_split = meta[meta['is_train'] == 0]
        solution_dir = os.path.join(params.data_dir, 'stage1_solution.csv')
        image_ids = meta_valid_split['ImageId'].values
        y_true = read_masks_from_csv(image_ids, solution_dir)
    else:
        raise NotImplementedError

    if dev_mode:
        meta_valid_split = meta_valid_split.sample(params.dev_mode_size, random_state=SEED)

    data = {'input': {'meta': meta_valid_split,
                      'target_sizes': meta_valid_split[SIZE_COLUMNS].values},
            'specs': {'train_mode': False},
            'callback_input': {'meta_valid': None}
            }

    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
    pipeline.clean_cache()
    output = pipeline.transform(data)
    pipeline.clean_cache()
    y_pred = output['y_pred']

    logger.info('Calculating IOU and IOUT Scores')
    iou_score = intersection_over_union(y_true, y_pred)
    logger.info('IOU score on validation is {}'.format(iou_score))
    ctx.channel_send('IOU Score', 0, iou_score)

    iout_score = intersection_over_union_thresholds(y_true, y_pred)
    logger.info('IOUT score on validation is {}'.format(iout_score))
    ctx.channel_send('IOUT Score', 0, iout_score)


def predict(pipeline_name, dev_mode, logger, params):
    logger.info('predicting')
    meta = pd.read_csv(os.path.join(params.meta_dir, 'stage1_metadata.csv'))
    meta_test = meta[meta['is_train'] == 0]

    if dev_mode:
        meta_test = meta_test.sample(params.dev_mode_size, random_state=SEED)

    data = {'input': {'meta': meta_test,
                      'meta_valid': None,
                      'train_mode': False,
                      'target_sizes': meta_test[SIZE_COLUMNS].values
                      },
            }

    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
    pipeline.clean_cache()
    output = pipeline.transform(data)
    pipeline.clean_cache()
    y_pred = output['y_pred']

    create_submission(params.experiment_dir, meta_test, y_pred, logger)

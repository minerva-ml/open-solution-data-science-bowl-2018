import os
import shutil

import click
from deepsense import neptune
import pandas as pd

from pipeline_config import SOLUTION_CONFIG, Y_COLUMNS, SIZE_COLUMNS
from pipelines import PIPELINES
from preparation import train_valid_split, overlay_masks
from metrics import intersection_over_union, intersection_over_union_thresholds
from utils import init_logger, get_logger, read_masks, read_params, create_submission, generate_metadata

logger = get_logger()
ctx = neptune.Context()
params = read_params(ctx)


@click.group()
def action():
    pass


@action.command()
def prepare_metadata():
    logger.info('creating metadata')
    meta = generate_metadata(data_dir=params.data_dir, masks_overlayed_dir=params.masks_overlayed_dir)
    meta.to_csv(os.path.join(params.meta_dir, 'stage1_metadata.csv'), index=None)


@action.command()
def prepare_masks():
    logger.info('overlaying masks')
    overlay_masks(images_dir=params.data_dir, subdir_name='stage1_train', target_dir=params.masks_overlayed_dir)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-v', '--validation_size', help='percentage of training used for validation', default=0.1, required=False)
def train_pipeline(pipeline_name, validation_size):
    _train_pipeline(pipeline_name, validation_size)


def _train_pipeline(pipeline_name, validation_size):
    if bool(params.overwrite) and os.path.isdir(params.experiment_dir):
        shutil.rmtree(params.experiment_dir)

    meta = pd.read_csv(os.path.join(params.meta_dir, 'stage1_metadata.csv'))
    meta_train_split, meta_valid_split = train_valid_split(meta, validation_size)

    data = {'input': {'meta': meta_train_split,
                      'meta_valid': meta_valid_split,
                      'train_mode': True,
                      'target_sizes': meta_train_split[SIZE_COLUMNS].values
                      },
            }

    pipeline = PIPELINES[pipeline_name]['train'](SOLUTION_CONFIG)
    pipeline.fit_transform(data)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-v', '--validation_size', help='percentage of training used for validation', default=0.1, required=False)
def evaluate_pipeline(pipeline_name, validation_size):
    _evaluate_pipeline(pipeline_name, validation_size)


def _evaluate_pipeline(pipeline_name, validation_size):
    meta = pd.read_csv(os.path.join(params.meta_dir, 'stage1_metadata.csv'))
    meta_train_split, meta_valid_split = train_valid_split(meta, validation_size)

    data = {'input': {'meta': meta_valid_split,
                      'meta_valid': None,
                      'train_mode': False,
                      'target_sizes': meta_valid_split[SIZE_COLUMNS].values
                      },
            }

    y_true = read_masks(meta_valid_split[Y_COLUMNS].values)

    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
    output = pipeline.transform(data)
    y_pred = output['y_pred']

    logger.info('Calculating IOU and IOUT Scores')
    iou_score = intersection_over_union(y_true, y_pred)
    logger.info('IOU score on validation is {}'.format(iou_score))
    ctx.channel_send('IOU Score', 0, iou_score)

    iout_score = intersection_over_union_thresholds(y_true, y_pred)
    logger.info('IOUT score on validation is {}'.format(iout_score))
    ctx.channel_send('IOUT Score', 0, iout_score)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
def predict_pipeline(pipeline_name):
    _predict_pipeline(pipeline_name)


def _predict_pipeline(pipeline_name):
    meta = pd.read_csv(os.path.join(params.meta_dir, 'stage1_metadata.csv'))
    meta_test = meta[meta['is_train'] == 0]

    data = {'input': {'meta': meta_test,
                      'meta_valid': None,
                      'train_mode': False,
                      'target_sizes': meta_test[SIZE_COLUMNS].values
                      },
            }

    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
    output = pipeline.transform(data)
    y_pred = output['y_pred']

    create_submission(params.experiment_dir, meta_test, y_pred, logger)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-v', '--validation_size', help='percentage of training used for validation', default=0.1, required=False)
def train_evaluate_predict_pipeline(pipeline_name, validation_size):
    logger.info('training')
    _train_pipeline(pipeline_name, validation_size)
    logger.info('evaluating')
    _evaluate_pipeline(pipeline_name, validation_size)
    logger.info('predicting')
    _predict_pipeline(pipeline_name)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-v', '--validation_size', help='percentage of training used for validation', default=0.1, required=False)
def train_evaluate_pipeline(pipeline_name, validation_size):
    logger.info('training')
    _train_pipeline(pipeline_name, validation_size)
    logger.info('evaluating')
    _evaluate_pipeline(pipeline_name, validation_size)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-v', '--validation_size', help='percentage of training used for validation', default=0.1, required=False)
def evaluate_predict_pipeline(pipeline_name, validation_size):
    logger.info('evaluating')
    _evaluate_pipeline(pipeline_name, validation_size)
    logger.info('predicting')
    _predict_pipeline(pipeline_name)


if __name__ == "__main__":
    init_logger()
    action()

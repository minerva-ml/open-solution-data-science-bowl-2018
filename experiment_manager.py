import os
import shutil

import click
from deepsense import neptune

from pipeline_config import SOLUTION_CONFIG, Y_COLUMNS, SIZE_COLUMNS
from pipelines import PIPELINES
from preparation import split_train_data, overlay_masks
from utils import init_logger, get_logger, read_meta_data, read_masks, intersection_over_union_score, create_submission

logger = get_logger()
ctx = neptune.Context()


@click.group()
def action():
    pass


@action.command()
@click.option('-v', '--validation_size', help='percentage of training used for validation', default=0.2, required=False)
def train_valid_split(validation_size):
    params = ctx.params
    logger.info('splitting into train and valid')
    split_train_data(data_dir=params.meta_dir, validation_size=validation_size)


@action.command()
def prepare_masks():
    params = ctx.params
    logger.info('overlaying masks')
    overlay_masks(images_dir=params.images_dir)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
def train_pipeline(pipeline_name):
    _train_pipeline(pipeline_name)


def _train_pipeline(pipeline_name):
    params = ctx.params

    if bool(params.overwrite) and os.path.isdir(params.experiment_dir):
        shutil.rmtree(params.experiment_dir)

    train = read_meta_data(data_dir=params.meta_dir, filename='train_split.csv')
    valid = read_meta_data(data_dir=params.meta_dir, filename='valid_split.csv')

    data = {'input': {'meta': train,
                      'meta_valid': valid,
                      'train_mode': True,
                      },
            }

    pipeline = PIPELINES[pipeline_name]['train'](SOLUTION_CONFIG)
    pipeline.fit_transform(data)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
def evaluate_pipeline(pipeline_name):
    _evaluate_pipeline(pipeline_name)


def _evaluate_pipeline(pipeline_name):
    params = ctx.params

    valid = read_meta_data(data_dir=params.meta_dir, filename='valid_split.csv')

    data = {'input': {'meta': valid,
                      'meta_valid': None,
                      'train_mode': False,
                      'target_sizes': valid[SIZE_COLUMNS].values
                      },
            }

    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
    output = pipeline.transform(data)
    y_pred = output['y_pred']

    y_true = read_masks(valid[Y_COLUMNS].values)
    score = intersection_over_union_score(y_true, y_pred)
    logger.info('Score on validation is {}'.format(score))
    ctx.channel_send('Final Validation Score', 0, score)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
def predict_pipeline(pipeline_name):
    _predict_pipeline(pipeline_name)


def _predict_pipeline(pipeline_name):
    params = ctx.params

    test = read_meta_data(data_dir=params.meta_dir, filename='test.csv')

    data = {'input': {'meta': test,
                      'meta_valid': None,
                      'train_mode': False,
                      },
            }

    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
    output = pipeline.transform(data)
    y_pred = output['y_pred']

    create_submission(params.experiment_dir, test, y_pred, Y_COLUMNS, logger)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
def train_evaluate_predict_pipeline(pipeline_name):
    logger.info('training')
    _train_pipeline(pipeline_name)
    logger.info('evaluating')
    _evaluate_pipeline(pipeline_name)
    logger.info('predicting')
    _predict_pipeline(pipeline_name)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
def train_evaluate_pipeline(pipeline_name):
    logger.info('training')
    _train_pipeline(pipeline_name)
    logger.info('evaluating')
    _evaluate_pipeline(pipeline_name)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
def evaluate_predict_pipeline(pipeline_name):
    logger.info('evaluating')
    _evaluate_pipeline(pipeline_name)
    logger.info('predicting')
    _predict_pipeline(pipeline_name)


if __name__ == "__main__":
    init_logger()
    action()

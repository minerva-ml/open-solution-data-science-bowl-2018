import os
import shutil

import click
from deepsense import neptune

from pipeline_config import SOLUTION_CONFIG, Y_COLUMNS
from pipelines import PIPELINES
from preprocessing import split_train_data
from utils import init_logger, get_logger, read_yaml, read_data, multi_log_loss, create_submission

logger = get_logger()
ctx = neptune.Context()


@click.group()
def action():
    pass


@action.command()
@click.option('-v', '--validation_size', help='percentage of training used for validation', default=0.2, required=False)
def train_valid_split(validation_size):
    params = ctx.params
    logger.info('preprocessing training data')
    split_train_data(data_dir=params.data_dir, validation_size=validation_size)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
def train_pipeline(pipeline_name):
    _train_pipeline(pipeline_name)


def _train_pipeline(pipeline_name):
    params = ctx.params

    if bool(params.overwrite) and os.path.isdir(params.experiment_dir):
        shutil.rmtree(params.experiment_dir)

    train = read_data(data_dir=params.data_dir, filename='train_split.csv')
    valid = read_data(data_dir=params.data_dir, filename='valid_split.csv')

    data = {'input': {'meta': train,
                      'meta_valid': valid,
                      'train_mode': True,
                      },
            'input_ensemble': {'meta': valid,
                               'meta_valid': None,
                               'train_mode': True,
                               },
            }

    pipeline = PIPELINES[pipeline_name]['train'](SOLUTION_CONFIG)
    output = pipeline.fit_transform(data)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
def evaluate_pipeline(pipeline_name):
    _evaluate_pipeline(pipeline_name)


def _evaluate_pipeline(pipeline_name):
    params = ctx.params

    valid = read_data(data_dir=params.data_dir, filename='valid_split.csv')

    data = {'input': {'meta': valid,
                      'meta_valid': None,
                      'train_mode': False,
                      },
            'input_ensemble': {'meta': valid,
                               'meta_valid': None,
                               'train_mode': False,
                               },
            }

    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
    output = pipeline.transform(data)
    y_true = valid[Y_COLUMNS].values
    y_pred = output['y_pred']
    score = multi_log_loss(y_true, y_pred)
    logger.info('Score on validation is {}'.format(score))
    ctx.channel_send('Final Validation Score', 0, score)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
def predict_pipeline(pipeline_name):
    _predict_pipeline(pipeline_name)


def _predict_pipeline(pipeline_name):
    params = ctx.params

    test = read_data(data_dir=params.data_dir, filename='test.csv')

    data = {'input': {'meta': test,
                      'meta_valid': None,
                      'train_mode': False,
                      },
            'input_ensemble': {'meta': test,
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


@action.command()
@click.argument('pipeline_names', nargs=-1)
@click.option('-bn', '--blended_name', help='name of the new blended pipeline', required=True)
def blend_pipelines(pipeline_names, blended_name):
    params = ctx.params

    new_pipeline_dir = os.path.join(params.experiment_dir, blended_name, 'transformers')
    os.makedirs(new_pipeline_dir)
    for pipeline_name in pipeline_names:
        if pipeline_name is not None:
            pipeline_dir = os.path.join(params.experiment_dir, pipeline_name, 'transformers')
            for transformer_name in os.listdir(pipeline_dir):
                source_filepath = os.path.join(pipeline_dir, transformer_name)
                destination_filepath = os.path.join(new_pipeline_dir, transformer_name)
                logger.info('copying transformer from {} to {}'.format(source_filepath, destination_filepath))
                shutil.copy(source_filepath, destination_filepath)


if __name__ == "__main__":
    init_logger()
    action()

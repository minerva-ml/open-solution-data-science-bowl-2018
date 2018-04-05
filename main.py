import os
import shutil
from multiprocessing import set_start_method

set_start_method('spawn')

import click
import glob
import pandas as pd
from deepsense import neptune

from metrics import intersection_over_union, intersection_over_union_thresholds
from pipeline_config import SOLUTION_CONFIG, Y_COLUMNS_SCORING, SIZE_COLUMNS
from pipelines import PIPELINES
from preparation import train_valid_split, overlay_masks, overlay_contours, overlay_centers, get_vgg_clusters
from utils import init_logger, read_masks, read_params, create_submission, generate_metadata, set_seed, generate_data_frame_chunks

logger = init_logger()
ctx = neptune.Context()
params = read_params(ctx)

set_seed(1234)


@click.group()
def action():
    pass


@action.command()
def prepare_metadata():
    logger.info('creating metadata')
    meta = generate_metadata(data_dir=params.data_dir,
                             masks_overlayed_dir=params.masks_overlayed_dir,
                             contours_overlayed_dir=params.contours_overlayed_dir,
                             centers_overlayed_dir=params.centers_overlayed_dir)
    meta['is_external'] = 0

    for external_data_dir in glob.glob('{}/*'.format(params.external_data_dirs)):
        logger.info('adding external metadata for {}'.format(external_data_dir))
        meta_external = generate_metadata(data_dir=external_data_dir,
                                          masks_overlayed_dir=params.masks_overlayed_dir,
                                          contours_overlayed_dir=params.contours_overlayed_dir,
                                          centers_overlayed_dir=params.centers_overlayed_dir,
                                          generate_train_only=True)
        meta_external['is_external'] = 1
        meta = pd.concat([meta, meta_external], axis=0)

    logger.info('calculating clusters')
    meta_train = meta[meta['is_train'] == 1]
    meta_test = meta[meta['is_train'] == 0]

    vgg_features_clusters = get_vgg_clusters(meta_train)
    meta_train['vgg_features_clusters'] = vgg_features_clusters
    meta_test['vgg_features_clusters'] = 'NaN'
    meta = pd.concat([meta_train, meta_test], axis=0)

    meta.to_csv(os.path.join(params.meta_dir, 'stage1_metadata.csv'), index=None)


@action.command()
def prepare_masks():
    official_data_dir = params.data_dir
    external_data_dirs = glob.glob('{}/*'.format(params.external_data_dirs))
    all_data_dirs = external_data_dirs + [official_data_dir]
    for data_dir in all_data_dirs:
        logger.info('processing directory {}'.format(data_dir))
        logger.info('overlaying masks')
        overlay_masks(images_dir=data_dir, subdir_name='stage1_train', target_dir=params.masks_overlayed_dir)
        logger.info('overlaying contours')
        overlay_contours(images_dir=data_dir, subdir_name='stage1_train', target_dir=params.contours_overlayed_dir)
        logger.info('overlaying centers')
        overlay_centers(images_dir=data_dir, subdir_name='stage1_train', target_dir=params.centers_overlayed_dir)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-v', '--validation_size', help='percentage of training used for validation', default=0.1, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-s', '--simple_cv', help='use simple train test split', is_flag=True, required=False)
def train_pipeline(pipeline_name, validation_size, dev_mode, simple_cv):
    _train_pipeline(pipeline_name, validation_size, dev_mode, simple_cv)


def _train_pipeline(pipeline_name, validation_size, dev_mode, simple_cv):
    if bool(params.overwrite) and os.path.isdir(params.experiment_dir):
        shutil.rmtree(params.experiment_dir)

    meta = pd.read_csv(os.path.join(params.meta_dir, 'stage1_metadata.csv'))
    meta_train = meta[meta['is_train'] == 1]
    valid_ids = eval(params.valid_category_ids)

    if simple_cv:
        meta_train_split, meta_valid_split = train_valid_split(meta_train, validation_size, simple_split=True)
    else:
        meta_train_split, meta_valid_split = train_valid_split(meta_train, validation_size,
                                                               valid_category_ids=valid_ids)

    if dev_mode:
        meta_train_split = meta_train_split.sample(5, random_state=1234)
        meta_valid_split = meta_valid_split.sample(3, random_state=1234)

    data = {'input': {'meta': meta_train_split,
                      'meta_valid': meta_valid_split,
                      'train_mode': True,
                      'target_sizes': meta_train_split[SIZE_COLUMNS].values,
                      },
            }

    pipeline = PIPELINES[pipeline_name]['train'](SOLUTION_CONFIG)
    pipeline.clean_cache()
    pipeline.fit_transform(data)
    pipeline.clean_cache()


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-v', '--validation_size', help='percentage of training used for validation', default=0.1, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-s', '--simple_cv', help='use simple train test split', is_flag=True, required=False)
def evaluate_pipeline(pipeline_name, validation_size, dev_mode, simple_cv):
    _evaluate_pipeline(pipeline_name, validation_size, dev_mode, simple_cv)


def _evaluate_pipeline(pipeline_name, validation_size, dev_mode, simple_cv):
    meta = pd.read_csv(os.path.join(params.meta_dir, 'stage1_metadata.csv'))
    meta_train = meta[meta['is_train'] == 1]
    valid_ids = eval(params.valid_category_ids)

    if simple_cv:
        meta_train_split, meta_valid_split = train_valid_split(meta_train, validation_size, simple_split=True)
    else:
        meta_train_split, meta_valid_split = train_valid_split(meta_train, validation_size,
                                                               valid_category_ids=valid_ids)

    if dev_mode:
        meta_valid_split = meta_valid_split.sample(2, random_state=1234)

    data = {'input': {'meta': meta_valid_split,
                      'meta_valid': None,
                      'train_mode': False,
                      'target_sizes': meta_valid_split[SIZE_COLUMNS].values
                      },
            }

    y_true = read_masks(meta_valid_split[Y_COLUMNS_SCORING].values)

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


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-c', '--chunk_size', help='size of the chunks to run prediction on', type=int, default=None, required=False)
def predict_pipeline(pipeline_name, dev_mode, chunk_size):
    if chunk_size is not None:
        _predict_in_chunks_pipeline(pipeline_name, dev_mode, chunk_size)
    else:
        _predict_pipeline(pipeline_name, dev_mode)


def _predict_pipeline(pipeline_name, dev_mode):
    meta = pd.read_csv(os.path.join(params.meta_dir, 'stage1_metadata.csv'))
    meta_test = meta[meta['is_train'] == 0]

    if dev_mode:
        meta_test = meta_test.sample(2, random_state=1234)

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

    submission = create_submission(meta_test, y_pred, logger)

    submission_filepath = os.path.join(params.experiment_dir, 'submission.csv')
    submission.to_csv(submission_filepath, index=None, encoding='utf-8')
    logger.info('submission saved to {}'.format(submission_filepath))
    logger.info('submission head \n\n{}'.format(submission.head()))


def _predict_in_chunks_pipeline(pipeline_name, dev_mode, chunk_size):
    meta = pd.read_csv(os.path.join(params.meta_dir, 'stage1_metadata.csv'))
    meta_test = meta[meta['is_train'] == 0]

    if dev_mode:
        meta_test = meta_test.sample(9, random_state=1234)

    logger.info('processing metadata of shape {}'.format(meta_test.shape))

    submission_chunks = []
    for meta_chunk in generate_data_frame_chunks(meta_test, chunk_size):

        data = {'input': {'meta': meta_chunk,
                          'meta_valid': None,
                          'train_mode': False,
                          'target_sizes': meta_chunk[SIZE_COLUMNS].values
                          },
                }

        pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
        pipeline.clean_cache()
        output = pipeline.transform(data)
        pipeline.clean_cache()
        y_pred = output['y_pred']

        submission_chunk = create_submission(meta_chunk, y_pred, logger)
        submission_chunks.append(submission_chunk)

    submission = pd.concat(submission_chunks, axis=0)

    submission_filepath = os.path.join(params.experiment_dir, 'submission.csv')
    submission.to_csv(submission_filepath, index=None, encoding='utf-8')
    logger.info('submission saved to {}'.format(submission_filepath))
    logger.info('submission head \n\n{}'.format(submission.head()))


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-v', '--validation_size', help='percentage of training used for validation', default=0.1, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-s', '--simple_cv', help='use simple train test split', is_flag=True, required=False)
def train_evaluate_predict_pipeline(pipeline_name, validation_size, dev_mode, simple_cv):
    logger.info('training')
    _train_pipeline(pipeline_name, validation_size, dev_mode, simple_cv)
    logger.info('evaluating')
    _evaluate_pipeline(pipeline_name, validation_size, dev_mode, simple_cv)
    logger.info('predicting')
    _predict_pipeline(pipeline_name, dev_mode)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-v', '--validation_size', help='percentage of training used for validation', default=0.1, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-s', '--simple_cv', help='use simple train test split', is_flag=True, required=False)
def train_evaluate_pipeline(pipeline_name, validation_size, dev_mode, simple_cv):
    logger.info('training')
    _train_pipeline(pipeline_name, validation_size, dev_mode, simple_cv)
    logger.info('evaluating')
    _evaluate_pipeline(pipeline_name, validation_size, dev_mode, simple_cv)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-v', '--validation_size', help='percentage of training used for validation', default=0.1, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-s', '--simple_cv', help='use simple train test split', is_flag=True, required=False)
def evaluate_predict_pipeline(pipeline_name, validation_size, dev_mode, simple_cv):
    logger.info('evaluating')
    _evaluate_pipeline(pipeline_name, validation_size, dev_mode, simple_cv)
    logger.info('predicting')
    _predict_pipeline(pipeline_name, dev_mode)


if __name__ == "__main__":
    action()

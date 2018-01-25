"""
Implement trainable ensemble: XGBoost, random forest, Linear Regression
"""

from steps.base import Step, Dummy
from steps.preprocessing import XYSplit
from postprocessing import Resizer, Thresholder
from loaders import MockLoader, MetadataImageSegmentationLoader
from models import MockModel, SequentialConvNet
from utils import squeeze_inputs


def dummy_train(config):
    xy_train = Step(name='xy_train',
                    transformer=XYSplit(**config.xy_splitter),
                    input_data=['input'],
                    adapter={'meta': ([('input', 'meta')]),
                             'train_mode': ([('input', 'train_mode')])
                             },
                    cache_dirpath=config.env.cache_dirpath)

    xy_inference = Step(name='xy_inference',
                        transformer=XYSplit(**config.xy_splitter),
                        input_data=['input'],
                        adapter={'meta': ([('input', 'meta_valid')]),
                                 'train_mode': ([('input', 'train_mode')])
                                 },
                        cache_dirpath=config.env.cache_dirpath)

    loader_train = Step(name='loader',
                        transformer=MockLoader(**config.loader),
                        input_data=['input'],
                        input_steps=[xy_train, xy_inference],
                        adapter={'X': ([('xy_train', 'X')]),
                                 'y': ([('xy_train', 'y')]),
                                 'train_mode': ([('input', 'train_mode')]),
                                 'X_valid': ([('xy_inference', 'X')]),
                                 'y_valid': ([('xy_inference', 'y')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)

    sequential_convnet = Step(name='sequential_convnet',
                              transformer=MockModel(**config.sequential_convnet),
                              input_steps=[loader_train],
                              cache_dirpath=config.env.cache_dirpath)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[sequential_convnet],
                  adapter={'y_pred': ([('sequential_convnet', 'predicted_masks')])},
                  cache_dirpath=config.env.cache_dirpath)
    return output


def dummy_inference(config):
    xy_inference = Step(name='xy_inference',
                        transformer=XYSplit(**config.xy_splitter),
                        input_data=['input'],
                        adapter={'meta': ([('input', 'meta')]),
                                 'train_mode': ([('input', 'train_mode')])
                                 },
                        cache_dirpath=config.env.cache_dirpath)

    loader_inference = Step(name='loader',
                            transformer=MockLoader(**config.loader),
                            input_data=['input'],
                            input_steps=[xy_inference, xy_inference],
                            adapter={'X': ([('xy_inference', 'X')]),
                                     'y': ([('xy_inference', 'y')]),
                                     'train_mode': ([('input', 'train_mode')]),
                                     },
                            cache_dirpath=config.env.cache_dirpath)

    sequential_convnet = Step(name='sequential_convnet',
                              transformer=MockModel(**config.sequential_convnet),
                              input_steps=[loader_inference],
                              cache_dirpath=config.env.cache_dirpath)

    mask_resize = Step(name='mask_resize',
                       transformer=Resizer(),
                       input_data=['input'],
                       input_steps=[sequential_convnet],
                       adapter={'images': ([('sequential_convnet', 'predicted_masks')]),
                                'target_sizes': ([('input', 'target_sizes')]),
                                },
                       cache_dirpath=config.env.cache_dirpath)

    thresholding = Step(name='thresholding',
                        transformer=Thresholder(**config.thresholder),
                        input_steps=[mask_resize],
                        adapter={'images': ([('mask_resize', 'resized_images')])},
                        cache_dirpath=config.env.cache_dirpath)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[thresholding],
                  adapter={'y_pred': ([('thresholding', 'binarized_images')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def seq_conv_train(config):
    xy_train = Step(name='xy_train',
                    transformer=XYSplit(**config.xy_splitter),
                    input_data=['input'],
                    adapter={'meta': ([('input', 'meta')]),
                             'train_mode': ([('input', 'train_mode')])
                             },
                    cache_dirpath=config.env.cache_dirpath)

    xy_inference = Step(name='xy_inference',
                        transformer=XYSplit(**config.xy_splitter),
                        input_data=['input'],
                        adapter={'meta': ([('input', 'meta_valid')]),
                                 'train_mode': ([('input', 'train_mode')])
                                 },
                        cache_dirpath=config.env.cache_dirpath)

    loader_train = Step(name='loader',
                        transformer=MetadataImageSegmentationLoader(**config.loader),
                        input_data=['input'],
                        input_steps=[xy_train, xy_inference],
                        adapter={'X': ([('xy_train', 'X')], squeeze_inputs),
                                 'y': ([('xy_train', 'y')], squeeze_inputs),
                                 'train_mode': ([('input', 'train_mode')]),
                                 'X_valid': ([('xy_inference', 'X')], squeeze_inputs),
                                 'y_valid': ([('xy_inference', 'y')], squeeze_inputs),
                                 },
                        cache_dirpath=config.env.cache_dirpath)

    sequential_convnet = Step(name='sequential_convnet',
                              transformer=SequentialConvNet(**config.sequential_convnet),
                              input_steps=[loader_train],
                              cache_dirpath=config.env.cache_dirpath)

    mask_resize = Step(name='mask_resize',
                       transformer=Resizer(),
                       input_data=['input'],
                       input_steps=[sequential_convnet],
                       adapter={'images': ([('sequential_convnet', 'predicted_masks')]),
                                'target_sizes': ([('input', 'target_sizes')]),
                                },
                       cache_dirpath=config.env.cache_dirpath)

    thresholding = Step(name='thresholding',
                        transformer=Thresholder(**config.thresholder),
                        input_steps=[mask_resize],
                        adapter={'images': ([('mask_resize', 'resized_images')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[thresholding],
                  adapter={'y_pred': ([('thresholding', 'binarized_images')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def seq_conv_inference(config):
    xy_inference = Step(name='xy_inference',
                        transformer=XYSplit(**config.xy_splitter),
                        input_data=['input'],
                        adapter={'meta': ([('input', 'meta')]),
                                 'train_mode': ([('input', 'train_mode')])
                                 },
                        cache_dirpath=config.env.cache_dirpath)

    loader_inference = Step(name='loader',
                            transformer=MetadataImageSegmentationLoader(**config.loader),
                            input_data=['input'],
                            input_steps=[xy_inference, xy_inference],
                            adapter={'X': ([('xy_inference', 'X')], squeeze_inputs),
                                     'y': ([('xy_inference', 'y')], squeeze_inputs),
                                     'train_mode': ([('input', 'train_mode')]),
                                     },
                            cache_dirpath=config.env.cache_dirpath)

    sequential_convnet = Step(name='sequential_convnet',
                              transformer=SequentialConvNet(**config.sequential_convnet),
                              input_steps=[loader_inference],
                              cache_dirpath=config.env.cache_dirpath)

    mask_resize = Step(name='mask_resize',
                       transformer=Resizer(),
                       input_data=['input'],
                       input_steps=[sequential_convnet],
                       adapter={'images': ([('sequential_convnet', 'predicted_masks')]),
                                'target_sizes': ([('input', 'target_sizes')]),
                                },
                       cache_dirpath=config.env.cache_dirpath)

    thresholding = Step(name='thresholding',
                        transformer=Thresholder(**config.thresholder),
                        input_steps=[mask_resize],
                        adapter={'images': ([('mask_resize', 'resized_images')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[thresholding],
                  adapter={'y_pred': ([('thresholding', 'binarized_images')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


PIPELINES = {'dummy': {'train': dummy_train,
                       'inference': dummy_inference},
             'hello_dsb': {'train': seq_conv_train,
                           'inference': seq_conv_inference},
             }

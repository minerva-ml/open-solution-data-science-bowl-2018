"""
Implement trainable ensemble: XGBoost, random forest, Linear Regression
"""

from steps.base import Step, Dummy, stack_inputs, hstack_inputs, sparse_hstack_inputs, to_tuple_inputs
from steps.preprocessing import XYSplit
from postprocessing import Resizer
from loaders import MockLoader
from models import MockModel


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

    unet_network = Step(name='unet_network',
                        transformer=MockModel(**config.unet_network),
                        input_steps=[loader_train],
                        cache_dirpath=config.env.cache_dirpath)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[unet_network],
                  adapter={'y_pred': ([('unet_network', 'predicted_masks')]),
                           },
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

    unet_network = Step(name='unet_network',
                        transformer=MockModel(**config.unet_network),
                        input_steps=[loader_inference],
                        cache_dirpath=config.env.cache_dirpath)

    mask_resize = Step(name='mask_resize',
                       transformer=Resizer(),
                       input_data=['input'],
                       input_steps=[unet_network],
                       adapter={'images': ([('unet_network', 'predicted_masks')]),
                                'target_sizes': ([('input', 'target_sizes')]),
                                },
                       cache_dirpath=config.env.cache_dirpath)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[mask_resize],
                  adapter={'y_pred': ([('mask_resize', 'resized_images')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


PIPELINES = {'dummy': {'train': dummy_train,
                       'inference': dummy_inference},
             }

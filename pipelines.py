"""
Implement trainable ensemble: XGBoost, random forest, Linear Regression
"""

from steps.base import Step, Dummy
from steps.preprocessing import XYSplit
from postprocessing import Resizer, Thresholder, Whatershed, NucleiLabeler
from loaders import MetadataImageSegmentationLoader, MetadataImageSegmentationMultitaskLoader
from models import PyTorchUNet, PyTorchUNetMultitask
from utils import squeeze_inputs


def unet_train(config):
    """
    U-Net architecture
    :param config:
    :return:
    """
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

    loader = Step(name='loader',
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

    unet = Step(name='unet',
                transformer=PyTorchUNet(**config.unet),
                input_steps=[loader],
                cache_dirpath=config.env.cache_dirpath)

    mask_resize = Step(name='mask_resize',
                       transformer=Resizer(),
                       input_data=['input'],
                       input_steps=[unet],
                       adapter={'images': ([('unet', 'predicted_masks')]),
                                'target_sizes': ([('input', 'target_sizes')]),
                                },
                       cache_dirpath=config.env.cache_dirpath)

    thresholding = Step(name='thresholding',
                        transformer=Thresholder(**config.thresholder),
                        input_steps=[mask_resize],
                        adapter={'images': ([('mask_resize', 'resized_images')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath,
                        cache_output=True)

    labeler = Step(name='labeler',
                   overwrite_transformer=True,
                   transformer=NucleiLabeler(),
                   input_steps=[thresholding],
                   adapter={'images': ([('thresholding', 'binarized_images')]),
                            },
                   cache_dirpath=config.env.cache_dirpath)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[labeler],
                  adapter={'y_pred': ([('labeler', 'labeled_images')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def unet_inference(config):
    xy_inference = Step(name='xy_inference',
                        transformer=XYSplit(**config.xy_splitter),
                        input_data=['input'],
                        adapter={'meta': ([('input', 'meta')]),
                                 'train_mode': ([('input', 'train_mode')])
                                 },
                        cache_dirpath=config.env.cache_dirpath)

    loader = Step(name='loader',
                  transformer=MetadataImageSegmentationLoader(**config.loader),
                  input_data=['input'],
                  input_steps=[xy_inference, xy_inference],
                  adapter={'X': ([('xy_inference', 'X')], squeeze_inputs),
                           'y': ([('xy_inference', 'y')], squeeze_inputs),
                           'train_mode': ([('input', 'train_mode')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)

    unet = Step(name='unet',
                transformer=PyTorchUNet(**config.unet),
                input_steps=[loader],
                cache_dirpath=config.env.cache_dirpath)

    mask_resize = Step(name='mask_resize',
                       transformer=Resizer(),
                       input_data=['input'],
                       input_steps=[unet],
                       adapter={'images': ([('unet', 'predicted_masks')]),
                                'target_sizes': ([('input', 'target_sizes')]),
                                },
                       cache_dirpath=config.env.cache_dirpath)

    thresholding = Step(name='thresholding',
                        transformer=Thresholder(**config.thresholder),
                        input_steps=[mask_resize],
                        adapter={'images': ([('mask_resize', 'resized_images')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)

    labeler = Step(name='labeler',
                   transformer=NucleiLabeler(),
                   input_steps=[thresholding],
                   adapter={'images': ([('thresholding', 'binarized_images')]),
                            },
                   cache_dirpath=config.env.cache_dirpath)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[labeler],
                  adapter={'y_pred': ([('labeler', 'labeled_images')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)

    return output


def unet_multitask_train(config):
    """
    U-Net architecture
    :param config:
    :return:
    """
    xy_train = Step(name='xy_train',
                    transformer=XYSplit(**config.xy_splitter_multitask),
                    input_data=['input'],
                    adapter={'meta': ([('input', 'meta')]),
                             'train_mode': ([('input', 'train_mode')])
                             },
                    cache_dirpath=config.env.cache_dirpath)

    xy_inference = Step(name='xy_inference',
                        transformer=XYSplit(**config.xy_splitter_multitask),
                        input_data=['input'],
                        adapter={'meta': ([('input', 'meta_valid')]),
                                 'train_mode': ([('input', 'train_mode')])
                                 },
                        cache_dirpath=config.env.cache_dirpath)

    loader = Step(name='loader',
                  transformer=MetadataImageSegmentationMultitaskLoader(**config.loader),
                  input_data=['input'],
                  input_steps=[xy_train, xy_inference],
                  adapter={'X': ([('xy_train', 'X')], squeeze_inputs),
                           'y': ([('xy_train', 'y')]),
                           'train_mode': ([('input', 'train_mode')]),
                           'X_valid': ([('xy_inference', 'X')], squeeze_inputs),
                           'y_valid': ([('xy_inference', 'y')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)

    unet_multitask = Step(name='unet_multitask',
                          transformer=PyTorchUNetMultitask(**config.unet),
                          input_steps=[loader],
                          cache_dirpath=config.env.cache_dirpath)

    mask_resize = Step(name='mask_resize',
                       transformer=Resizer(),
                       input_data=['input'],
                       input_steps=[unet_multitask],
                       adapter={'images': ([('unet_multitask', 'mask_prediction')]),
                                'target_sizes': ([('input', 'target_sizes')]),
                                },
                       cache_dirpath=config.env.cache_dirpath)

    mask_thresholding = Step(name='mask_thresholding',
                        transformer=Thresholder(**config.thresholder),
                        input_steps=[mask_resize],
                        adapter={'images': ([('mask_resize', 'resized_images')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath,
                        cache_output=True)

    contour_resize = Step(name='contour_resize',
                          transformer=Resizer(),
                          input_data=['input'],
                          input_steps=[unet_multitask],
                          adapter={'images': ([('unet_multitask', 'contour_prediction')]),
                                   'target_sizes': ([('input', 'target_sizes')]),
                                   },
                          cache_dirpath=config.env.cache_dirpath)

    contour_thresholding = Step(name='contour_thresholding',
                        transformer=Thresholder(**config.thresholder),
                        input_steps=[contour_resize],
                        adapter={'images': ([('contour_resize', 'resized_images')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath,
                        cache_output=True)

    center_resize = Step(name='center_resize',
                         transformer=Resizer(),
                         input_data=['input'],
                         input_steps=[unet_multitask],
                         adapter={'images': ([('unet_multitask', 'center_prediction')]),
                                  'target_sizes': ([('input', 'target_sizes')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath)

    center_thresholding = Step(name='center_thresholding',
                        transformer=Thresholder(**config.thresholder),
                        input_steps=[center_resize],
                        adapter={'images': ([('center_resize', 'resized_images')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath,
                        cache_output=True)


    labeler = Step(name='labeler',
                   transformer=NucleiLabeler(),
                   input_steps=[contour_thresholding],
                   adapter={'images': ([('contour_thresholding', 'binarized_images')]),
                            },
                   cache_dirpath=config.env.cache_dirpath)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[labeler],
                  adapter={'y_pred': ([('labeler', 'labeled_images')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def unet_multitask_inference(config):
    xy_inference = Step(name='xy_inference',
                        transformer=XYSplit(**config.xy_splitter),
                        input_data=['input'],
                        adapter={'meta': ([('input', 'meta')]),
                                 'train_mode': ([('input', 'train_mode')])
                                 },
                        cache_dirpath=config.env.cache_dirpath)

    loader = Step(name='loader',
                  transformer=MetadataImageSegmentationMultitaskLoader(**config.loader),
                  input_data=['input'],
                  input_steps=[xy_inference, xy_inference],
                  adapter={'X': ([('xy_inference', 'X')], squeeze_inputs),
                           'y': ([('xy_inference', 'y')], squeeze_inputs),
                           'train_mode': ([('input', 'train_mode')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)

    unet_multitask = Step(name='unet_multitask',
                          transformer=PyTorchUNetMultitask(**config.unet),
                          input_steps=[loader],
                          cache_dirpath=config.env.cache_dirpath)

    mask_resize = Step(name='mask_resize',
                       transformer=Resizer(),
                       input_data=['input'],
                       input_steps=[unet_multitask],
                       adapter={'images': ([('unet_multitask', 'predicted_masks')]),
                                'target_sizes': ([('input', 'target_sizes')]),
                                },
                       cache_dirpath=config.env.cache_dirpath)

    mask_resize = Step(name='mask_resize',
                       transformer=Resizer(),
                       input_data=['input'],
                       input_steps=[unet_multitask],
                       adapter={'images': ([('unet_multitask', 'mask_prediction')]),
                                'target_sizes': ([('input', 'target_sizes')]),
                                },
                       cache_dirpath=config.env.cache_dirpath)

    mask_thresholding = Step(name='mask_thresholding',
                        transformer=Thresholder(**config.thresholder),
                        input_steps=[mask_resize],
                        adapter={'images': ([('mask_resize', 'resized_images')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath,
                        )

    contour_resize = Step(name='contour_resize',
                          transformer=Resizer(),
                          input_data=['input'],
                          input_steps=[unet_multitask],
                          adapter={'images': ([('unet_multitask', 'contour_prediction')]),
                                   'target_sizes': ([('input', 'target_sizes')]),
                                   },
                          cache_dirpath=config.env.cache_dirpath)

    contour_thresholding = Step(name='contour_thresholding',
                        transformer=Thresholder(**config.thresholder),
                        input_steps=[contour_resize],
                        adapter={'images': ([('contour_resize', 'resized_images')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath,
                        )

    center_resize = Step(name='center_resize',
                         transformer=Resizer(),
                         input_data=['input'],
                         input_steps=[unet_multitask],
                         adapter={'images': ([('unet_multitask', 'center_prediction')]),
                                  'target_sizes': ([('input', 'target_sizes')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath)

    center_thresholding = Step(name='center_thresholding',
                        transformer=Thresholder(**config.thresholder),
                        input_steps=[center_resize],
                        adapter={'images': ([('center_resize', 'resized_images')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath,
                        )


    labeler = Step(name='labeler',
                   transformer=NucleiLabeler(),
                   input_steps=[mask_thresholding],
                   adapter={'images': ([('mask_thresholding', 'binarized_images')]),
                            },
                   cache_dirpath=config.env.cache_dirpath)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[labeler],
                  adapter={'y_pred': ([('labeler', 'labeled_images')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)

    return output


PIPELINES = {'unet': {'train': unet_train,
                      'inference': unet_inference},
             'unet_multitask': {'train': unet_multitask_train,
                                'inference': unet_multitask_inference},
             }

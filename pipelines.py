from functools import partial

import loaders
from models import PyTorchUNet, PyTorchUNetMultitask
from postprocessing import Resizer, Thresholder, NucleiLabeler, Dropper, \
    WatershedCenter, WatershedContour, BinaryFillHoles, Postprocessor, CellSizer
from preprocessing import ImageReaderRescaler, ImageReader, StainDeconvolution
from steps.base import Step, Dummy, to_dict_inputs
from steps.preprocessing import XYSplit
from utils import squeeze_inputs


def unet(config, train_mode):
    if train_mode:
        save_output = False
        load_saved_output = False
    else:
        save_output = False
        load_saved_output = False

    loader = preprocessing(config, model_type='single', is_train=train_mode)

    unet = Step(name='unet',
                transformer=PyTorchUNet(**config.unet),
                input_steps=[loader],
                cache_dirpath=config.env.cache_dirpath,
                save_output=save_output, load_saved_output=load_saved_output)

    mask_postprocessed = mask_postprocessing(unet, config, save_output=save_output)

    detached = nuclei_labeler(mask_postprocessed, config, save_output=save_output)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[detached],
                  adapter={'y_pred': ([(detached.name, 'labels')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def patched_unet_training(config):
    save_output = False
    load_saved_output = False

    loader = preprocessing(config, model_type='multitask', is_train=True, loader_mode='patching_train')

    unet_multitask = Step(name='unet_multitask',
                          transformer=PyTorchUNetMultitask(**config.unet),
                          input_steps=[loader],
                          adapter={'datagen': ([(loader.name, 'datagen')]),
                                   'validation_datagen': ([(loader.name, 'validation_datagen')]),
                                   },
                          cache_dirpath=config.env.cache_dirpath,
                          save_output=save_output, load_saved_output=load_saved_output)
    return unet_multitask


def unet_multitask(config, train_mode):
    if train_mode:
        save_output = False
        load_saved_output = False
    else:
        save_output = False
        load_saved_output = False

    if config.loader.dataset_params.use_patching:
        loader = preprocessing(config, model_type='multitask',
                               is_train=train_mode,
                               loader_mode='patching_inference')

        unet_multitask_patches = Step(name='unet_multitask',
                                      transformer=PyTorchUNetMultitask(**config.unet),
                                      input_steps=[loader],
                                      adapter={'datagen': ([(loader.name, 'datagen')]),
                                               'validation_datagen': ([(loader.name, 'validation_datagen')]),
                                               },
                                      cache_dirpath=config.env.cache_dirpath,
                                      cache_output=False,
                                      save_output=False,
                                      load_saved_output=False)

        unet_multitask = Step(name='patch_joiner',
                              transformer=loaders.PatchCombiner(**config.patch_combiner),
                              input_steps=[unet_multitask_patches, loader],
                              adapter={'patch_ids': ([(loader.name, 'patch_ids')]),
                                       'outputs': ([(unet_multitask_patches.name, 'mask_prediction'),
                                                    (unet_multitask_patches.name, 'contour_prediction'),
                                                    (unet_multitask_patches.name, 'center_prediction')],
                                                   partial(to_dict_inputs, keys=['mask_prediction',
                                                                                 'contour_prediction',
                                                                                 'center_prediction'])),
                                       },
                              cache_dirpath=config.env.cache_dirpath,
                              cache_output=True,
                              save_output=True,
                              load_saved_output=load_saved_output)
    else:
        loader = preprocessing(config, model_type='multitask',
                               is_train=train_mode,
                               loader_mode=None)

        unet_multitask = Step(name='unet_multitask',
                              transformer=PyTorchUNetMultitask(**config.unet),
                              input_steps=[loader],
                              adapter={'datagen': ([(loader.name, 'datagen')]),
                                       'validation_datagen': ([(loader.name, 'validation_datagen')]),
                                       },
                              cache_dirpath=config.env.cache_dirpath,
                              cache_output=True,
                              save_output=True,
                              load_saved_output=False)

    mask_resize = Step(name='mask_resize',
                       transformer=Resizer(),
                       input_data=['input'],
                       input_steps=[unet_multitask],
                       adapter={'images': ([(unet_multitask.name, 'mask_prediction')]),
                                'target_sizes': ([('input', 'target_sizes')]),
                                },
                       cache_dirpath=config.env.cache_dirpath,
                       save_output=True)

    contour_resize = Step(name='contour_resize',
                          transformer=Resizer(),
                          input_data=['input'],
                          input_steps=[unet_multitask],
                          adapter={'images': ([(unet_multitask.name, 'contour_prediction')]),
                                   'target_sizes': ([('input', 'target_sizes')]),
                                   },
                          cache_dirpath=config.env.cache_dirpath,
                          save_output=True)

    detached = Step(name='detached',
                    transformer=Postprocessor(),
                    input_steps=[mask_resize, contour_resize],
                    adapter={'images': ([(mask_resize.name, 'resized_images')]),
                             'contours': ([(contour_resize.name, 'resized_images')]),
                             },
                    save_output=True,
                    cache_dirpath=config.env.cache_dirpath)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[detached],
                  adapter={'y_pred': ([(detached.name, 'labeled_images')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def two_unet_specialists(config, train_mode):
    if train_mode:
        save_output = False
        load_saved_output = False
    else:
        save_output = False
        load_saved_output = False

    loader = preprocessing(config, model_type='specialists', is_train=train_mode, loader_mode='patching_inference')

    if config.loader.dataset_params.use_patching:
        unet_mask_patches = Step(name='unet_mask',
                                 transformer=PyTorchUNetMultitask(**config.unet_mask),
                                 input_steps=[loader],
                                 adapter={'datagen': ([(loader.name, 'datagen')]),
                                          'validation_datagen': ([(loader.name, 'validation_datagen')]),
                                          },
                                 cache_dirpath=config.env.cache_dirpath,
                                 cache_output=False,
                                 save_output=save_output,
                                 load_saved_output=load_saved_output)

        unet_contour_patches = Step(name='unet_contour',
                                    transformer=PyTorchUNetMultitask(**config.unet_contour),
                                    input_steps=[loader],
                                    adapter={'datagen': ([(loader.name, 'datagen')]),
                                             'validation_datagen': ([(loader.name, 'validation_datagen')]),
                                             },
                                    cache_dirpath=config.env.cache_dirpath,
                                    cache_output=False,
                                    save_output=save_output,
                                    load_saved_output=load_saved_output)

        unet_specialists = Step(name='patch_joiner',
                                transformer=loaders.PatchCombiner(**config.patch_combiner),
                                input_steps=[unet_mask_patches, unet_contour_patches, loader],
                                adapter={'patch_ids': ([(loader.name, 'patch_ids')]),
                                         'outputs': ([(unet_mask_patches.name, 'mask_prediction'),
                                                      (unet_contour_patches.name, 'contour_prediction'),
                                                      ],
                                                     partial(to_dict_inputs, keys=['mask_prediction',
                                                                                   'contour_prediction'])),
                                         },
                                cache_dirpath=config.env.cache_dirpath,
                                load_saved_output=load_saved_output)

        mask_resize = Step(name='mask_resize',
                           transformer=Resizer(),
                           input_data=['input'],
                           input_steps=[unet_specialists],
                           adapter={'images': ([(unet_specialists.name, 'mask_prediction')]),
                                    'target_sizes': ([('input', 'target_sizes')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath,
                           save_output=True)

        contour_resize = Step(name='contour_resize',
                              transformer=Resizer(),
                              input_data=['input'],
                              input_steps=[unet_specialists],
                              adapter={'images': ([(unet_specialists.name, 'contour_prediction')]),
                                       'target_sizes': ([('input', 'target_sizes')]),
                                       },
                              cache_dirpath=config.env.cache_dirpath,
                              save_output=True)

    else:
        unet_mask = Step(name='unet_mask',
                         transformer=PyTorchUNetMultitask(**config.unet_mask),
                         input_steps=[loader],
                         adapter={'datagen': ([(loader.name, 'datagen')]),
                                  'validation_datagen': ([(loader.name, 'validation_datagen')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath,
                         save_output=save_output, load_saved_output=load_saved_output)
        unet_contour = Step(name='unet_contour',
                            transformer=PyTorchUNetMultitask(**config.unet_contour),
                            input_steps=[loader],
                            adapter={'datagen': ([(loader.name, 'datagen')]),
                                     'validation_datagen': ([(loader.name, 'validation_datagen')]),
                                     },
                            cache_dirpath=config.env.cache_dirpath,
                            save_output=save_output, load_saved_output=load_saved_output)

        mask_resize = Step(name='mask_resize',
                           transformer=Resizer(),
                           input_data=['input'],
                           input_steps=[unet_mask],
                           adapter={'images': ([(unet_mask.name, 'mask_prediction')]),
                                    'target_sizes': ([('input', 'target_sizes')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath,
                           save_output=save_output)

        contour_resize = Step(name='contour_resize',
                              transformer=Resizer(),
                              input_data=['input'],
                              input_steps=[unet_contour],
                              adapter={'images': ([(unet_contour.name, 'contour_prediction')]),
                                       'target_sizes': ([('input', 'target_sizes')]),
                                       },
                              cache_dirpath=config.env.cache_dirpath,
                              save_output=save_output)

    detached = Step(name='detached',
                    transformer=Postprocessor(),
                    input_steps=[mask_resize, contour_resize],
                    adapter={'images': ([(mask_resize.name, 'resized_images')]),
                             'contours': ([(contour_resize.name, 'resized_images')]),
                             },
                    cache_dirpath=config.env.cache_dirpath,
                    save_output=True)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[detached],
                  adapter={'y_pred': ([(detached.name, 'labeled_images')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def scale_adjusted_patched_unet_training(config):
    reader_train = Step(name='reader',
                        transformer=ImageReader(**config.reader_multitask),
                        input_data=['input'],
                        adapter={'meta': ([('input', 'meta')]),
                                 'meta_valid': ([('input', 'meta_valid')]),
                                 'train_mode': ([('input', 'train_mode')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)
    reader_valid = Step(name='reader',
                        transformer=ImageReader(**config.reader_multitask),
                        input_data=['input'],
                        adapter={'meta': ([('input', 'meta_valid')]),
                                 'train_mode': ([('input', 'train_mode')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)

    scale_estimator_train = unet_size_estimator(reader_train, config)
    scale_estimator_valid = unet_size_estimator(reader_valid, config)

    reader_rescaler_train = Step(name='rescaler',
                                 transformer=ImageReaderRescaler(**config.reader_rescaler),
                                 input_data=['input'],
                                 input_steps=[reader_train, scale_estimator_train],
                                 adapter={'sizes': ([(scale_estimator_train.name, 'sizes')]),
                                          'X': ([(reader_train.name, 'X')]),
                                          'y': ([(reader_train.name, 'y')]),
                                          'meta': ([('input', 'meta')]),
                                          },
                                 cache_dirpath=config.env.cache_dirpath)

    stain_deconvolution_train = Step(name='stain_deconvolution',
                                     transformer=StainDeconvolution(**config.stain_deconvolution),
                                     input_steps=[reader_rescaler_train],
                                     adapter={'X': ([(reader_rescaler_train.name, 'X')]),
                                              },
                                     cache_dirpath=config.env.cache_dirpath)

    reader_rescaler_valid = Step(name='rescaler',
                                 transformer=ImageReaderRescaler(**config.reader_rescaler),
                                 input_data=['input'],
                                 input_steps=[reader_valid, scale_estimator_valid],
                                 adapter={'sizes': ([(scale_estimator_valid.name, 'sizes')]),
                                          'X': ([(reader_valid.name, 'X')]),
                                          'y': ([(reader_valid.name, 'y')]),
                                          'meta': ([('input', 'meta_valid')]),
                                          },
                                 cache_dirpath=config.env.cache_dirpath)

    reader_rescaler = Step(name='rescaler_join',
                           transformer=Dummy(),
                           input_steps=[reader_rescaler_train, reader_rescaler_valid],
                           adapter={'X': ([(reader_rescaler_train.name, 'X')]),
                                    'y': ([(reader_rescaler_train.name, 'y')]),
                                    'X_valid': ([(reader_rescaler_valid.name, 'X')]),
                                    'y_valid': ([(reader_rescaler_valid.name, 'y')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath,
                           save_output=True, load_saved_output=True)

    stain_deconvolution_train = Step(name='stain_deconvolution',
                                     transformer=StainDeconvolution(**config.stain_deconvolution),
                                     input_steps=[reader_rescaler],
                                     adapter={'X': ([(reader_rescaler.name, 'X')]),
                                              },
                                     cache_dirpath=config.env.cache_dirpath)

    stain_deconvolution_valid = Step(name='stain_deconvolution',
                                     transformer=StainDeconvolution(**config.stain_deconvolution),
                                     input_steps=[reader_rescaler],
                                     adapter={'X': ([(reader_rescaler.name, 'X_valid')]),
                                              },
                                     cache_dirpath=config.env.cache_dirpath)

    rescaler_deconver = Step(name='rescaled_deconved',
                             transformer=Dummy(),
                             input_steps=[reader_rescaler,
                                          stain_deconvolution_train, stain_deconvolution_valid],
                             adapter={'X': ([(stain_deconvolution_train.name, 'X')]),
                                      'y': ([(reader_rescaler.name, 'y')]),
                                      'X_valid': ([(stain_deconvolution_valid.name, 'X')]),
                                      'y_valid': ([(reader_rescaler.name, 'y')]),
                                      },
                             cache_dirpath=config.env.cache_dirpath,
                             save_output=True, load_saved_output=True)

    unet_rescaled = unet_multitask_block(rescaler_deconver, config,
                                         loader_mode='patched_training',
                                         suffix='_rescaled',
                                         force_fitting=True)

    return unet_rescaled


def scale_adjusted_patched_unet_inference(config):
    reader = Step(name='reader',
                  transformer=ImageReader(**config.reader_multitask),
                  input_data=['input'],
                  adapter={'meta': ([('input', 'meta')]),
                           'train_mode': ([('input', 'train_mode')]),
                           },
                  cache_dirpath=config.env.cache_dirpath,
                  cache_output=True)

    scale_estimator = unet_size_estimator(reader, config, cache_output=True)

    reader_rescaler = Step(name='rescaler',
                           transformer=ImageReaderRescaler(**config.reader_rescaler),
                           input_data=['input'],
                           input_steps=[reader, scale_estimator],
                           adapter={'sizes': ([(scale_estimator.name, 'sizes')]),
                                    'X': ([(reader.name, 'X')]),
                                    'y': ([(reader.name, 'y')]),
                                    'meta': ([('input', 'meta')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath,
                           cache_output=True)

    stain_deconvolution = Step(name='stain_deconvolution',
                               transformer=StainDeconvolution(**config.stain_deconvolution),
                               input_steps=[reader_rescaler],
                               adapter={'X': ([(reader_rescaler.name, 'X')]),
                                        },
                               cache_dirpath=config.env.cache_dirpath,
                               cache_output=True)

    loader_rescaled = Step(name='loader_rescaled',
                           transformer=loaders.ImageSegmentationMultitaskLoaderPatchingInference(**config.loader),
                           input_data=['input'],
                           input_steps=[reader_rescaler, stain_deconvolution],
                           adapter={'X': ([(stain_deconvolution.name, 'X')]),
                                    'y': ([(reader_rescaler.name, 'y')]),
                                    'train_mode': ([('input', 'train_mode')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath,
                           cache_output=True)

    unet_rescaled_patches = Step(name='unet_rescaled',
                                 transformer=PyTorchUNetMultitask(**config.unet),
                                 input_steps=[loader_rescaled],
                                 adapter={'datagen': ([(loader_rescaled.name, 'datagen')]),
                                          'validation_datagen': ([(loader_rescaled.name, 'validation_datagen')]),
                                          },
                                 cache_dirpath=config.env.cache_dirpath)

    unet_rescaled = Step(name='patch_joiner',
                         transformer=loaders.PatchCombiner(**config.patch_combiner),
                         input_steps=[unet_rescaled_patches, loader_rescaled],
                         adapter={'patch_ids': ([(loader_rescaled.name, 'patch_ids')]),
                                  'outputs': ([(unet_rescaled_patches.name, 'mask_prediction'),
                                               (unet_rescaled_patches.name, 'contour_prediction'),
                                               (unet_rescaled_patches.name, 'center_prediction')],
                                              partial(to_dict_inputs, keys=['mask_prediction',
                                                                            'contour_prediction',
                                                                            'center_prediction'])),
                                  },
                         cache_dirpath=config.env.cache_dirpath,
                         cache_output=True)

    morphological_postprocessing = postprocessing(unet_rescaled, unet_rescaled, config,
                                                  suffix='_rescaled', save_output=True)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[morphological_postprocessing],
                  adapter={'y_pred': ([(morphological_postprocessing.name, 'labeled_images')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def unet_size_estimator(reader, config, suffix='_size_estimator', cache_output=False):
    unet = unet_multitask_block(reader, config, loader_mode=None, suffix=suffix, cache_output=cache_output)

    morphological_postprocessing = postprocessing(unet, unet, config, suffix=suffix, cache_output=cache_output)

    cell_sizer = Step(name='cell_sizer{}'.format(suffix),
                      transformer=CellSizer(),
                      input_steps=[morphological_postprocessing],
                      adapter={'labeled_images': ([(morphological_postprocessing.name, 'labeled_images')])},
                      cache_dirpath=config.env.cache_dirpath,
                      cache_output=cache_output
                      )
    return cell_sizer


def unet_multitask_block(reader, config, loader_mode, force_fitting=False, suffix='', cache_output=False):
    if loader_mode == 'patching_train':
        Loader = loaders.ImageSegmentationMultitaskLoaderPatchingTrain
    elif loader_mode == 'patching_inference':
        Loader = loaders.ImageSegmentationMultitaskLoaderPatchingInference
    else:
        Loader = loaders.ImageSegmentationMultitaskLoader

    loader = Step(name='loader{}'.format(suffix),
                  transformer=Loader(**config.loader),
                  input_data=['input'],
                  input_steps=[reader],
                  adapter={'X': ([(reader.name, 'X')]),
                           'y': ([(reader.name, 'y')]),
                           'train_mode': ([('input', 'train_mode')]),
                           'X_valid': ([(reader.name, 'X_valid')]),
                           'y_valid': ([(reader.name, 'y_valid')]),
                           },
                  cache_dirpath=config.env.cache_dirpath,
                  cache_output=cache_output)

    unet_multitask = Step(name='unet{}'.format(suffix),
                          transformer=PyTorchUNetMultitask(**config.unet),
                          input_steps=[loader],
                          adapter={'datagen': ([(loader.name, 'datagen')]),
                                   'validation_datagen': ([(loader.name, 'validation_datagen')]),
                                   },
                          cache_dirpath=config.env.cache_dirpath,
                          force_fitting=force_fitting,
                          cache_output=cache_output)

    return unet_multitask


def preprocessing(config, model_type, is_train, loader_mode=None):
    if config.execution.load_in_memory:
        if model_type == 'single':
            loader = _preprocessing_single_in_memory(config, is_train, loader_mode)
        elif model_type == 'multitask':
            loader = _preprocessing_multitask_in_memory(config, is_train, loader_mode, is_specialist=False)
        elif model_type == 'specialists':
            loader = _preprocessing_multitask_in_memory(config, is_train, loader_mode, is_specialist=True)
        else:
            raise NotImplementedError
    else:
        if model_type == 'single':
            loader = _preprocessing_single_generator(config, is_train, loader_mode)
        elif model_type == 'multitask':
            loader = _preprocessing_multitask_generator(config, is_train, loader_mode, is_specialist=False)
        elif model_type == 'specialists':
            loader = _preprocessing_multitask_generator(config, is_train, loader_mode, is_specialist=True)
        else:
            raise NotImplementedError
    return loader


def postprocessing(model_mask, model_contour, config, suffix='', save_output=False, cache_output=False):
    mask_resize = Step(name='mask_resize{}'.format(suffix),
                       transformer=Resizer(),
                       input_data=['input'],
                       input_steps=[model_mask],
                       adapter={'images': ([(model_mask.name, 'mask_prediction')]),
                                'target_sizes': ([('input', 'target_sizes')]),
                                },
                       cache_dirpath=config.env.cache_dirpath,
                       save_output=save_output,
                       cache_output=cache_output)

    contour_resize = Step(name='contour_resize{}'.format(suffix),
                          transformer=Resizer(),
                          input_data=['input'],
                          input_steps=[model_contour],
                          adapter={'images': ([(model_contour.name, 'contour_prediction')]),
                                   'target_sizes': ([('input', 'target_sizes')]),
                                   },
                          cache_dirpath=config.env.cache_dirpath,
                          save_output=save_output,
                          cache_output=cache_output)

    morphological_postprocessing = Step(name='morphological_postprocessing{}'.format(suffix),
                                        transformer=Postprocessor(),
                                        input_steps=[mask_resize, contour_resize],
                                        adapter={'images': ([(mask_resize.name, 'resized_images')]),
                                                 'contours': ([(contour_resize.name, 'resized_images')]),
                                                 },
                                        cache_dirpath=config.env.cache_dirpath,
                                        save_output=save_output,
                                        cache_output=cache_output)

    return morphological_postprocessing


def mask_postprocessing(model, config, save_output=True):
    mask_resize = Step(name='mask_resize',
                       transformer=Resizer(),
                       input_data=['input'],
                       input_steps=[model],
                       adapter={'images': ([(model.name, 'mask_prediction')]),
                                'target_sizes': ([('input', 'target_sizes')]),
                                },
                       cache_dirpath=config.env.cache_dirpath,
                       save_output=save_output)

    mask_thresholding = Step(name='mask_thresholding',
                             transformer=Thresholder(**config.thresholder),
                             input_steps=[mask_resize],
                             adapter={'images': ([('mask_resize', 'resized_images')]),
                                      },
                             cache_dirpath=config.env.cache_dirpath,
                             save_output=save_output)

    return mask_thresholding


def contour_postprocessing(model, config, save_output=True):
    contour_resize = Step(name='contour_resize',
                          transformer=Resizer(),
                          input_data=['input'],
                          input_steps=[model],
                          adapter={'images': ([(model.name, 'contour_prediction')]),
                                   'target_sizes': ([('input', 'target_sizes')]),
                                   },
                          cache_dirpath=config.env.cache_dirpath,
                          save_output=save_output)

    contour_thresholding = Step(name='contour_thresholding',
                                transformer=Thresholder(**config.thresholder),
                                input_steps=[contour_resize],
                                adapter={'images': ([('contour_resize', 'resized_images')]),
                                         },
                                cache_dirpath=config.env.cache_dirpath,
                                save_output=save_output)
    return contour_thresholding


def center_postprocessing(model, config, save_output=True):
    center_resize = Step(name='center_resize',
                         transformer=Resizer(),
                         input_data=['input'],
                         input_steps=[model],
                         adapter={'images': ([(model.name, 'center_prediction')]),
                                  'target_sizes': ([('input', 'target_sizes')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath,
                         save_output=save_output)

    center_thresholding = Step(name='center_thresholding',
                               transformer=Thresholder(**config.thresholder),
                               input_steps=[center_resize],
                               adapter={'images': ([('center_resize', 'resized_images')]),
                                        },
                               cache_dirpath=config.env.cache_dirpath,
                               save_output=save_output)
    return center_thresholding


def watershed_centers(mask, center, config, save_output=True):
    watershed_center = Step(name='watershed_centers',
                            transformer=WatershedCenter(),
                            input_steps=[mask, center],
                            adapter={'images': ([(mask.name, 'binarized_images')]),
                                     'contours': ([(center.name, 'binarized_images')]),
                                     },
                            cache_dirpath=config.env.cache_dirpath,
                            save_output=save_output)

    drop_smaller = Step(name='drop_smaller',
                        transformer=Dropper(**config.dropper),
                        input_steps=[watershed_center],
                        adapter={'labels': ([('watershed_center', 'detached_images')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath,
                        save_output=save_output)

    binary_fill = Step(name='binary_fill',
                       transformer=BinaryFillHoles(),
                       input_steps=[drop_smaller],
                       adapter={'images': ([('drop_smaller', 'labels')]),
                                },
                       cache_dirpath=config.env.cache_dirpath,
                       save_output=save_output)

    return binary_fill


def watershed_contours(mask, contour, config, save_output=True):
    watershed_contour = Step(name='watershed_contour',
                             transformer=WatershedContour(),
                             input_steps=[mask, contour],
                             adapter={'images': ([(mask.name, 'binarized_images')]),
                                      'contours': ([(contour.name, 'binarized_images')]),
                                      },
                             cache_dirpath=config.env.cache_dirpath,
                             save_output=save_output)

    drop_smaller = Step(name='drop_smaller',
                        transformer=Dropper(**config.dropper),
                        input_steps=[watershed_contour],
                        adapter={'labels': ([('watershed_contour', 'detached_images')]),
                                 },

                        cache_dirpath=config.env.cache_dirpath,
                        save_output=save_output)
    return drop_smaller


def nuclei_labeler(postprocessed_mask, config, save_output=True):
    labeler = Step(name='labeler',
                   transformer=NucleiLabeler(),
                   input_steps=[postprocessed_mask],
                   adapter={'images': ([(postprocessed_mask.name, 'binarized_images')]),
                            },
                   cache_dirpath=config.env.cache_dirpath,
                   save_output=save_output)
    return labeler


def _preprocessing_single_in_memory(config, is_train, use_patching):
    if use_patching:
        raise NotImplementedError
    else:
        reader = Step(name='reader',
                      transformer=ImageReader(**config.reader_single),
                      input_data=['input'],
                      adapter={'meta': ([('input', 'meta')]),
                               'meta_valid': ([('input', 'meta_valid')]),
                               'train_mode': ([('input', 'train_mode')]),
                               },
                      cache_dirpath=config.env.cache_dirpath)

        loader = Step(name='loader',
                      transformer=loaders.ImageSegmentationLoader(**config.loader),
                      input_data=['input'],
                      input_steps=[reader],
                      adapter={'X': ([('reader', 'X')]),
                               'y': ([('reader', 'y')]),
                               'train_mode': ([('input', 'train_mode')]),
                               'X_valid': ([('reader', 'X_valid')]),
                               'y_valid': ([('reader', 'y_valid')]),
                               },
                      cache_dirpath=config.env.cache_dirpath)
    return loader


def _preprocessing_single_generator(config, is_train, use_patching):
    if use_patching:
        raise NotImplementedError
    else:
        if is_train:
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
                          transformer=loaders.MetadataImageSegmentationLoader(**config.loader),
                          input_data=['input'],
                          input_steps=[xy_train, xy_inference],
                          adapter={'X': ([('xy_train', 'X')], squeeze_inputs),
                                   'y': ([('xy_train', 'y')], squeeze_inputs),
                                   'train_mode': ([('input', 'train_mode')]),
                                   'X_valid': ([('xy_inference', 'X')], squeeze_inputs),
                                   'y_valid': ([('xy_inference', 'y')], squeeze_inputs),
                                   },
                          cache_dirpath=config.env.cache_dirpath)
        else:
            xy_inference = Step(name='xy_inference',
                                transformer=XYSplit(**config.xy_splitter),
                                input_data=['input'],
                                adapter={'meta': ([('input', 'meta')]),
                                         'train_mode': ([('input', 'train_mode')])
                                         },
                                cache_dirpath=config.env.cache_dirpath)

            loader = Step(name='loader',
                          transformer=loaders.MetadataImageSegmentationLoader(**config.loader),
                          input_data=['input'],
                          input_steps=[xy_inference, xy_inference],
                          adapter={'X': ([('xy_inference', 'X')], squeeze_inputs),
                                   'y': ([('xy_inference', 'y')], squeeze_inputs),
                                   'train_mode': ([('input', 'train_mode')]),
                                   },
                          cache_dirpath=config.env.cache_dirpath)
    return loader


def _preprocessing_multitask_in_memory(config, is_train, loader_mode, is_specialist):
    if is_specialist:
        reader_config = config.reader_specialists
    else:
        reader_config = config.reader_multitask

    if loader_mode == 'patching_train':
        Loader = loaders.ImageSegmentationMultitaskLoaderPatchingTrain
    elif loader_mode == 'patching_inference':
        Loader = loaders.ImageSegmentationMultitaskLoaderPatchingInference
    else:
        Loader = loaders.ImageSegmentationMultitaskLoader

    reader = Step(name='reader',
                  transformer=ImageReader(**reader_config),
                  input_data=['input'],
                  adapter={'meta': ([('input', 'meta')]),
                           'meta_valid': ([('input', 'meta_valid')]),
                           'train_mode': ([('input', 'train_mode')]),
                           },
                  cache_dirpath=config.env.cache_dirpath,
                  save_output=False, load_saved_output=False)

    loader = Step(name='loader',
                  transformer=Loader(**config.loader),
                  input_data=['input'],
                  input_steps=[reader],
                  adapter={'X': ([('reader', 'X')]),
                           'y': ([('reader', 'y')]),
                           'train_mode': ([('input', 'train_mode')]),
                           'X_valid': ([('reader', 'X_valid')]),
                           'y_valid': ([('reader', 'y_valid')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return loader


def _preprocessing_multitask_generator(config, is_train, use_patching, is_specialist):
    if is_specialist:
        splitter_config = config.xy_splitter_specialists
    else:
        splitter_config = config.xy_splitter_multitask

    if use_patching:
        raise NotImplementedError
    else:
        if is_train:
            xy_train = Step(name='xy_train',
                            transformer=XYSplit(**splitter_config),
                            input_data=['input'],
                            adapter={'meta': ([('input', 'meta')]),
                                     'train_mode': ([('input', 'train_mode')])
                                     },
                            cache_dirpath=config.env.cache_dirpath)

            xy_inference = Step(name='xy_inference',
                                transformer=XYSplit(**config.splitter_config),
                                input_data=['input'],
                                adapter={'meta': ([('input', 'meta_valid')]),
                                         'train_mode': ([('input', 'train_mode')])
                                         },
                                cache_dirpath=config.env.cache_dirpath)

            loader = Step(name='loader',
                          transformer=loaders.MetadataImageSegmentationMultitaskLoader(**config.loader),
                          input_data=['input'],
                          input_steps=[xy_train, xy_inference],
                          adapter={'X': ([('xy_train', 'X')], squeeze_inputs),
                                   'y': ([('xy_train', 'y')]),
                                   'train_mode': ([('input', 'train_mode')]),
                                   'X_valid': ([('xy_inference', 'X')], squeeze_inputs),
                                   'y_valid': ([('xy_inference', 'y')]),
                                   },
                          cache_dirpath=config.env.cache_dirpath)
        else:
            xy_inference = Step(name='xy_inference',
                                transformer=XYSplit(**splitter_config),
                                input_data=['input'],
                                adapter={'meta': ([('input', 'meta')]),
                                         'train_mode': ([('input', 'train_mode')])
                                         },
                                cache_dirpath=config.env.cache_dirpath)

            loader = Step(name='loader',
                          transformer=loaders.MetadataImageSegmentationMultitaskLoader(**config.loader),
                          input_data=['input'],
                          input_steps=[xy_inference, xy_inference],
                          adapter={'X': ([('xy_inference', 'X')], squeeze_inputs),
                                   'y': ([('xy_inference', 'y')], squeeze_inputs),
                                   'train_mode': ([('input', 'train_mode')]),
                                   },
                          cache_dirpath=config.env.cache_dirpath)
    return loader


PIPELINES = {'unet': {'train': partial(unet, train_mode=True),
                      'inference': partial(unet, train_mode=False),
                      },
             'unet_multitask': {'train': partial(unet_multitask, train_mode=True),
                                'inference': partial(unet_multitask, train_mode=False),
                                },
             'two_unets_specialists': {'train': partial(two_unet_specialists, train_mode=True),
                                       'inference': partial(two_unet_specialists, train_mode=False),
                                       },
             'patched_unet_training': {'train': patched_unet_training},
             'scale_adjusted_patched_unet_training': {'train': scale_adjusted_patched_unet_training},
             'scale_adjusted_patched_unet': {'train': scale_adjusted_patched_unet_inference,
                                             'inference': scale_adjusted_patched_unet_inference}
             }

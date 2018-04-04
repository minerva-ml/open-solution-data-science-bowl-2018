from functools import partial

import loaders
from models import PyTorchUNet, PyTorchUNetMultitask
from postprocessing import Resizer, NucleiLabeler, Postprocessor, CellSizer
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
                  adapter={'y_pred': ([(detached.name, 'labeled_images')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


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

    morphological_postprocessing = postprocessing(unet_multitask, unet_multitask, config,
                                                  suffix='', save_output=True)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[morphological_postprocessing],
                  adapter={'y_pred': ([(morphological_postprocessing.name, 'labeled_images')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def patched_unet_training(config):
    reader_train = Step(name='reader',
                        transformer=ImageReader(**config.reader_multitask),
                        input_data=['input'],
                        adapter={'meta': ([('input', 'meta')]),
                                 'meta_valid': ([('input', 'meta_valid')]),
                                 'train_mode': ([('input', 'train_mode')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)
    deconved_reader_train = add_stain_deconvolution(reader_train, config, cache_output=True, save_output=False,
                                                    suffix='')

    reader_valid = Step(name='reader_valid',
                        transformer=ImageReader(**config.reader_multitask),
                        input_data=['input'],
                        adapter={'meta': ([('input', 'meta_valid')]),
                                 'train_mode': ([('input', 'train_mode')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)
    deconved_reader_valid = add_stain_deconvolution(reader_valid, config, cache_output=True, save_output=False,
                                                    suffix='_valid')

    reader = Step(name='reader_joined',
                  transformer=Dummy(),
                  input_steps=[reader_train, reader_valid,
                               deconved_reader_train, deconved_reader_valid],
                  adapter={'X': ([(deconved_reader_train.name, 'X')]),
                           'y': ([(deconved_reader_train.name, 'y')]),
                           'X_valid': ([(deconved_reader_valid.name, 'X')]),
                           'y_valid': ([(deconved_reader_valid.name, 'y')]),
                           },
                  cache_dirpath=config.env.cache_dirpath,
                  save_output=True, load_saved_output=True)

    unet_multitask = unet_multitask_block(reader, config, config.unet_size_estimator,
                                          loader_mode=None, suffix='_size_estimator')

    return unet_multitask


def scale_adjusted_patched_unet_training(config):
    reader_train = Step(name='reader',
                        transformer=ImageReader(**config.reader_multitask),
                        input_data=['input'],
                        adapter={'meta': ([('input', 'meta')]),
                                 'meta_valid': ([('input', 'meta_valid')]),
                                 'train_mode': ([('input', 'train_mode')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath,
                        cache_output=True)
    deconved_reader_train = add_stain_deconvolution(reader_train, config,
                                                    cache_output=True,
                                                    save_output=False,
                                                    suffix='')

    reader_valid = Step(name='reader_valid',
                        transformer=ImageReader(**config.reader_multitask),
                        input_data=['input'],
                        adapter={'meta': ([('input', 'meta_valid')]),
                                 'train_mode': ([('input', 'train_mode')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath,
                        cache_output=True)
    deconved_reader_valid = add_stain_deconvolution(reader_valid, config,
                                                    cache_output=True,
                                                    save_output=False,
                                                    suffix='_valid')

    deconved_reader_train_valid = Step(name='reader_train_valid',
                                       transformer=Dummy(),
                                       input_steps=[deconved_reader_train, deconved_reader_valid],
                                       adapter={'X': ([(deconved_reader_train.name, 'X')]),
                                                'y': ([(deconved_reader_train.name, 'y')]),
                                                'X_valid': ([(deconved_reader_valid.name, 'X')]),
                                                'y_valid': ([(deconved_reader_valid.name, 'y')]),
                                                },
                                       cache_dirpath=config.env.cache_dirpath,
                                       cache_output=True)

    scale_estimator_train = unet_size_estimator(deconved_reader_train_valid, config, config.unet_size_estimator,
                                                cache_output=True, train_mode=True)
    scale_estimator_valid = unet_size_estimator(deconved_reader_valid, config, config.unet_size_estimator,
                                                suffix='_valid', cache_output=True, train_mode=False)

    reader_rescaler_train = Step(name='rescaler',
                                 transformer=ImageReaderRescaler(**config.reader_rescaler),
                                 input_data=['input'],
                                 input_steps=[reader_train, scale_estimator_train],
                                 adapter={'sizes': ([(scale_estimator_train.name, 'sizes')]),
                                          'X': ([(reader_train.name, 'X')]),
                                          'y': ([(reader_train.name, 'y')]),
                                          'meta': ([('input', 'meta')]),
                                          },
                                 cache_dirpath=config.env.cache_dirpath,
                                 cache_output=True)
    deconved_reader_rescaler_train = add_stain_deconvolution(reader_rescaler_train, config,
                                                             cache_output=True,
                                                             save_output=False,
                                                             suffix='_rescaled')

    reader_rescaler_valid = Step(name='rescaler_valid',
                                 transformer=ImageReaderRescaler(**config.reader_rescaler),
                                 input_data=['input'],
                                 input_steps=[reader_valid, scale_estimator_valid],
                                 adapter={'sizes': ([(scale_estimator_valid.name, 'sizes')]),
                                          'X': ([(reader_valid.name, 'X')]),
                                          'y': ([(reader_valid.name, 'y')]),
                                          'meta': ([('input', 'meta_valid')]),
                                          },
                                 cache_dirpath=config.env.cache_dirpath,
                                 cache_output=True)
    deconved_reader_rescaler_valid = add_stain_deconvolution(reader_rescaler_valid, config,
                                                             cache_output=True,
                                                             save_output=False,
                                                             suffix='_rescaled_valid')

    reader_rescaler = Step(name='rescaler_join',
                           transformer=Dummy(),
                           input_steps=[deconved_reader_rescaler_train, deconved_reader_rescaler_valid],
                           adapter={'X': ([(deconved_reader_rescaler_train.name, 'X')]),

                                    'y': ([(deconved_reader_rescaler_train.name, 'y')]),
                                    'X_valid': ([(deconved_reader_rescaler_valid.name, 'X')]),
                                    'y_valid': ([(deconved_reader_rescaler_valid.name, 'y')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath,
                           cache_output=True,
                           save_output=True, load_saved_output=True)

    unet_rescaled = unet_multitask_block(reader_rescaler, config, config.unet,
                                         loader_mode='patched_training',
                                         suffix='_rescaled',
                                         force_fitting=True)

    return unet_rescaled


def scale_adjusted_patched_unet(config):
    reader = Step(name='reader',
                  transformer=ImageReader(**config.reader_multitask),
                  input_data=['input'],
                  adapter={'meta': ([('input', 'meta')]),
                           'train_mode': ([('input', 'train_mode')]),
                           },
                  cache_dirpath=config.env.cache_dirpath,
                  cache_output=True)
    deconved_reader = add_stain_deconvolution(reader, config,
                                              cache_output=True,
                                              save_output=False,
                                              suffix='')
    scale_estimator = unet_size_estimator(deconved_reader, config, config.unet_size_estimator,
                                          cache_output=True, train_mode=False)

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
    deconved_reader = add_stain_deconvolution(reader_rescaler, config,
                                              cache_output=True,
                                              save_output=False,
                                              suffix='_rescaled')

    loader_rescaled = Step(name='loader_rescaled',
                           transformer=loaders.ImageSegmentationMultitaskLoaderPatchingInference(**config.loader),
                           input_data=['input'],
                           input_steps=[deconved_reader],
                           adapter={'X': ([(deconved_reader.name, 'X')]),
                                    'y': ([(deconved_reader.name, 'y')]),
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


def add_stain_deconvolution(reader, config, cache_output=False, save_output=False, suffix=''):
    stain_deconvolution = Step(name='stain_deconvolution{}'.format(suffix),
                               transformer=StainDeconvolution(**config.stain_deconvolution),
                               input_steps=[reader],
                               adapter={'X': ([(reader.name, 'X')]),
                                        },
                               cache_dirpath=config.env.cache_dirpath)

    reader = Step(name='reader_with_deconv{}'.format(suffix),
                  transformer=Dummy(),
                  input_steps=[reader, stain_deconvolution],
                  adapter={'X': ([(stain_deconvolution.name, 'X')]),
                           'y': ([(reader.name, 'y')]),
                           },
                  cache_dirpath=config.env.cache_dirpath,
                  cache_output=cache_output,
                  save_output=save_output)

    return reader


def unet_size_estimator(reader, config, config_network, suffix='', cache_output=False, train_mode=True):
    unet = unet_multitask_block(reader, config, config_network, loader_mode=None, suffix='_size_estimator',
                                train_mode=train_mode)

    suffix = '_size_estimator{}'.format(suffix)

    morphological_postprocessing = postprocessing(unet, unet, config, suffix=suffix, cache_output=cache_output)

    cell_sizer = Step(name='cell_sizer{}'.format(suffix),
                      transformer=CellSizer(),
                      input_steps=[morphological_postprocessing],
                      adapter={'labeled_images': ([(morphological_postprocessing.name, 'labeled_images')])},
                      cache_dirpath=config.env.cache_dirpath,
                      cache_output=cache_output
                      )
    return cell_sizer


def unet_multitask_block(reader, config, config_network, loader_mode, force_fitting=False, suffix='',
                         cache_output=False, train_mode=True):
    if loader_mode == 'patching_train':
        Loader = loaders.ImageSegmentationMultitaskLoaderPatchingTrain
    elif loader_mode == 'patching_inference':
        Loader = loaders.ImageSegmentationMultitaskLoaderPatchingInference
    else:
        Loader = loaders.ImageSegmentationMultitaskLoader

    if train_mode:
        adapter_mapping = {'X': ([(reader.name, 'X')]),
                           'y': ([(reader.name, 'y')]),
                           'train_mode': ([('input', 'train_mode')]),
                           'X_valid': ([(reader.name, 'X_valid')]),
                           'y_valid': ([(reader.name, 'y_valid')]),
                           }
    else:
        adapter_mapping = {'X': ([(reader.name, 'X')]),
                           'y': ([(reader.name, 'y')]),
                           'train_mode': ([('input', 'train_mode')]),
                           }

    loader = Step(name='loader{}'.format(suffix),
                  transformer=Loader(**config.loader),
                  input_data=['input'],
                  input_steps=[reader],
                  adapter=adapter_mapping,
                  cache_dirpath=config.env.cache_dirpath,
                  cache_output=cache_output)

    unet_multitask = Step(name='unet{}'.format(suffix),
                          transformer=PyTorchUNetMultitask(**config_network),
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
            loader = _preprocessing_multitask_in_memory(config, is_train, loader_mode)
        else:
            raise NotImplementedError
    else:
        if model_type == 'single':
            loader = _preprocessing_single_generator(config, is_train, loader_mode)
        elif model_type == 'multitask':
            loader = _preprocessing_multitask_generator(config, is_train, loader_mode)
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


def _preprocessing_multitask_in_memory(config, is_train, loader_mode):
    if loader_mode == 'patching_train':
        Loader = loaders.ImageSegmentationMultitaskLoaderPatchingTrain
    elif loader_mode == 'patching_inference':
        Loader = loaders.ImageSegmentationMultitaskLoaderPatchingInference
    else:
        Loader = loaders.ImageSegmentationMultitaskLoader

    reader = Step(name='reader',
                  transformer=ImageReader(**config.reader_multitask),
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


def _preprocessing_multitask_generator(config, is_train, use_patching):
    if use_patching:
        raise NotImplementedError
    else:
        if is_train:
            xy_train = Step(name='xy_train',
                            transformer=XYSplit(**config.xy_splitter_multitask),
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

             'patched_unet_training': {'train': patched_unet_training},
             'scale_adjusted_patched_unet_training': {'train': scale_adjusted_patched_unet_training},
             'scale_adjusted_patched_unet': {'train': scale_adjusted_patched_unet,
                                             'inference': scale_adjusted_patched_unet}
             }

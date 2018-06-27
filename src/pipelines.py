from functools import partial

from .steppy.base import Step, Dummy
from .steppy.preprocessing.misc import XYSplit, ImageReader

from . import loaders
from .models import PyTorchUNet
from .postprocessing import resize_image, categorize_image, label_multiclass_image, get_channel, watershed
from .utils import squeeze_inputs_if_needed, make_apply_transformer


def unet(config, train_mode):
    if train_mode:
        save_output = False
        load_saved_output = False
        preprocessing = preprocessing_train(config, model_name='unet')
    else:
        save_output = False
        load_saved_output = False
        preprocessing = preprocessing_inference(config)

    unet = Step(name='unet',
                transformer=PyTorchUNet(**config.model['unet']),
                input_steps=[preprocessing],
                is_trainable=True,
                cache_dirpath=config.env.cache_dirpath,
                save_output=save_output, load_saved_output=load_saved_output)

    mask_postprocessed = mask_postprocessing(unet, config, save_output=save_output)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[mask_postprocessed],
                  adapter={'y_pred': ([(mask_postprocessed.name, 'nuclei_images')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def unet_masks(config):
    save_output = False
    load_saved_output = False
    preprocessing = preprocessing_train(config, model_name='unet_masks')

    unet_masks = Step(name='unet_masks',
                      transformer=PyTorchUNet(**config.model['unet_masks']),
                      input_steps=[preprocessing],
                      is_trainable=True,
                      cache_dirpath=config.env.cache_dirpath,
                      save_output=save_output, load_saved_output=load_saved_output)
    return unet_masks


def unet_borders(config):
    save_output = False
    load_saved_output = False
    preprocessing = preprocessing_train(config, model_name='unet_borders')

    unet_borders = Step(name='unet_borders',
                        transformer=PyTorchUNet(**config.model['unet_borders']),
                        input_steps=[preprocessing],
                        is_trainable=True,
                        cache_dirpath=config.env.cache_dirpath,
                        save_output=save_output, load_saved_output=load_saved_output)
    return unet_borders


def double_unet(config):
    save_output = False
    load_saved_output = False
    preprocessing = preprocessing_inference(config)

    unet_masks = Step(name='unet_masks',
                      transformer=PyTorchUNet(**config.model['unet_masks']),
                      input_steps=[preprocessing],
                      is_trainable=True,
                      cache_dirpath=config.env.cache_dirpath,
                      save_output=save_output, load_saved_output=load_saved_output)

    masks = postprocessing_masks(unet_masks, config, save_output=save_output)

    unet_borders = Step(name='unet_borders',
                        transformer=PyTorchUNet(**config.model['unet_borders']),
                        input_steps=[preprocessing],
                        is_trainable=True,
                        cache_dirpath=config.env.cache_dirpath,
                        save_output=save_output, load_saved_output=load_saved_output)

    borders = postprocessing_borders(unet_borders, config, save_output=save_output)

    watersheder = Step(name='watershed',
                       transformer=make_apply_transformer(watershed,
                                                          output_name='watersheded_images',
                                                          apply_on=['masks', 'borders']),
                       input_steps=[masks, borders],
                       adapter={'masks': ([(masks.name, 'nuclei_images')]),
                                'borders': ([(borders.name, 'nuclei_images')]),
                                },
                       cache_dirpath=config.env.cache_dirpath,
                       save_output=save_output, load_saved_output=load_saved_output)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[watersheder],
                  adapter={'y_pred': ([(watersheder.name, 'watersheded_images')]),
                           },
                  cache_dirpath=config.env.cache_dirpath,
                  )

    return output


def preprocessing_train(config, model_name='unet'):
    if config.execution.loader_mode == 'crop_and_pad':
        Loader = loaders.ImageSegmentationLoaderCropPad
    elif config.execution.loader_mode == 'resize':
        Loader = loaders.ImageSegmentationLoaderResize
    else:
        raise NotImplementedError

    if config.loader.dataset_params.image_source == 'memory':
        reader_train = Step(name='reader_train',
                            transformer=ImageReader(**config.reader[model_name]),
                            input_data=['input'],
                            adapter={'meta': ([('input', 'meta')]),
                                     'train_mode': ([('input', 'train_mode')]),
                                     },
                            cache_dirpath=config.env.cache_dirpath)

        reader_inference = Step(name='reader_inference',
                                transformer=ImageReader(**config.reader[model_name]),
                                input_data=['input'],
                                adapter={'meta': ([('input', 'meta_valid')]),
                                         'train_mode': ([('input', 'train_mode')]),
                                         },
                                cache_dirpath=config.env.cache_dirpath)

    elif config.loader.dataset_params.image_source == 'disk':
        reader_train = Step(name='xy_train',
                            transformer=XYSplit(**config.xy_splitter[model_name]),
                            input_data=['input'],
                            adapter={'meta': ([('input', 'meta')]),
                                     'train_mode': ([('input', 'train_mode')])
                                     },
                            cache_dirpath=config.env.cache_dirpath)

        reader_inference = Step(name='xy_inference',
                                transformer=XYSplit(**config.xy_splitter[model_name]),
                                input_data=['input'],
                                adapter={'meta': ([('input', 'meta_valid')]),
                                         'train_mode': ([('input', 'train_mode')])
                                         },
                                cache_dirpath=config.env.cache_dirpath)
    else:
        raise NotImplementedError

    loader = Step(name='loader',
                  transformer=Loader(**config.loader),
                  input_data=['input'],
                  input_steps=[reader_train, reader_inference],
                  adapter={'X': ([(reader_train.name, 'X')], squeeze_inputs_if_needed),
                           'y': ([(reader_train.name, 'y')], squeeze_inputs_if_needed),
                           'train_mode': ([('input', 'train_mode')]),
                           'X_valid': ([(reader_inference.name, 'X')], squeeze_inputs_if_needed),
                           'y_valid': ([(reader_inference.name, 'y')], squeeze_inputs_if_needed),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return loader


def preprocessing_inference(config, model_name='unet'):
    if config.execution.loader_mode == 'crop_and_pad':
        Loader = loaders.ImageSegmentationLoaderCropPad
    elif config.execution.loader_mode == 'resize':
        Loader = loaders.ImageSegmentationLoaderResize
    else:
        raise NotImplementedError

    if config.loader.dataset_params.image_source == 'memory':
        reader_inference = Step(name='reader_inference',
                                transformer=ImageReader(**config.reader[model_name]),
                                input_data=['input'],
                                adapter={'meta': ([('input', 'meta')]),
                                         'train_mode': ([('input', 'train_mode')]),
                                         },
                                cache_dirpath=config.env.cache_dirpath)

    elif config.loader.dataset_params.image_source == 'disk':
        reader_inference = Step(name='xy_inference',
                                transformer=XYSplit(**config.xy_splitter[model_name]),
                                input_data=['input'],
                                adapter={'meta': ([('input', 'meta')]),
                                         'train_mode': ([('input', 'train_mode')])
                                         },
                                cache_dirpath=config.env.cache_dirpath)
    else:
        raise NotImplementedError

    loader = Step(name='loader',
                  transformer=Loader(**config.loader),
                  input_data=['input'],
                  input_steps=[reader_inference],
                  adapter={'X': ([(reader_inference.name, 'X')], squeeze_inputs_if_needed),
                           'y': ([(reader_inference.name, 'y')], squeeze_inputs_if_needed),
                           'train_mode': ([('input', 'train_mode')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return loader


def mask_postprocessing(model, config, save_output=False):
    mask_resize = Step(name='mask_resize',
                       transformer=make_apply_transformer(resize_image,
                                                          output_name='resized_images',
                                                          apply_on=['images', 'target_sizes']),
                       input_data=['input'],
                       input_steps=[model],
                       adapter={'images': ([(model.name, 'mask_prediction')]),
                                'target_sizes': ([('input', 'target_sizes')]),
                                },
                       cache_dirpath=config.env.cache_dirpath,
                       save_output=save_output)

    category_mapper = Step(name='category_mapper',
                           transformer=make_apply_transformer(categorize_image,
                                                              output_name='categorized_images'),
                           input_steps=[mask_resize],
                           adapter={'images': ([('mask_resize', 'resized_images')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath,
                           save_output=save_output)

    labeler = Step(name='labeler',
                   transformer=make_apply_transformer(label_multiclass_image,
                                                      output_name='labeled_images'),
                   input_steps=[category_mapper],
                   adapter={'images': ([('category_mapper', 'categorized_images')]),
                            },
                   cache_dirpath=config.env.cache_dirpath,
                   save_output=save_output)

    nuclei_filter = Step(name='nuclei_filter',
                         transformer=make_apply_transformer(partial(get_channel,
                                                                    channel=config.postprocessor.channels.index(
                                                                        'nuclei')),
                                                            output_name='nuclei_images'),
                         input_steps=[labeler],
                         adapter={'images': ([('labeler', 'labeled_images')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath,
                         save_output=save_output)

    return nuclei_filter


def postprocessing_masks(model, config, save_output=False):
    mask_resize = Step(name='mask_resize_simple',
                       transformer=make_apply_transformer(resize_image,
                                                          output_name='resized_images',
                                                          apply_on=['images', 'target_sizes']),
                       input_data=['input'],
                       input_steps=[model],
                       adapter={'images': ([(model.name, 'mask_prediction')]),
                                'target_sizes': ([('input', 'target_sizes')]),
                                },
                       cache_dirpath=config.env.cache_dirpath,
                       save_output=save_output)

    category_mapper = Step(name='category_mapper_simple',
                           transformer=make_apply_transformer(categorize_image,
                                                              output_name='categorized_images'),
                           input_steps=[mask_resize],
                           adapter={'images': ([('mask_resize_simple', 'resized_images')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath,
                           save_output=save_output)

    nuclei_filter = Step(name='nuclei_filter_simple',
                         transformer=make_apply_transformer(partial(get_channel,
                                                                    channel=config.postprocessor.channels.index(
                                                                        'nuclei')),
                                                            output_name='nuclei_images'),
                         input_steps=[category_mapper],
                         adapter={'images': ([('category_mapper_simple', 'categorized_images')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath,
                         save_output=save_output)

    return nuclei_filter


def postprocessing_borders(model, config, save_output=False):
    mask_resize = Step(name='mask_resize_borders',
                       transformer=make_apply_transformer(resize_image,
                                                          output_name='resized_images',
                                                          apply_on=['images', 'target_sizes']),
                       input_data=['input'],
                       input_steps=[model],
                       adapter={'images': ([(model.name, 'mask_prediction')]),
                                'target_sizes': ([('input', 'target_sizes')]),
                                },
                       cache_dirpath=config.env.cache_dirpath,
                       save_output=save_output)

    category_mapper = Step(name='category_mapper_borders',
                           transformer=make_apply_transformer(categorize_image,
                                                              output_name='categorized_images'),
                           input_steps=[mask_resize],
                           adapter={'images': ([('mask_resize_borders', 'resized_images')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath,
                           save_output=save_output)

    borders_filter = Step(name='borders_filter_borders',
                          transformer=make_apply_transformer(partial(get_channel,
                                                                     channel=config.postprocessor.channels.index(
                                                                         'borders')),
                                                             output_name='nuclei_images'),
                          input_steps=[category_mapper],
                          adapter={'images': ([('category_mapper_borders', 'categorized_images')]),
                                   },
                          cache_dirpath=config.env.cache_dirpath,
                          save_output=save_output)

    return borders_filter


PIPELINES = {'unet': {'train': partial(unet, train_mode=True),
                      'inference': partial(unet, train_mode=False),
                      },
             'unet_masks': {'train': unet_masks,
                            },
             'unet_borders': {'train': unet_borders,
                              },
             'double_unet': {'inference': double_unet,
                             }
             }

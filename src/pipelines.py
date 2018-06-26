from functools import partial

from .steppy.base import Step, Dummy
from .steppy.preprocessing.misc import XYSplit, ImageReader

from .loaders import MetadataImageSegmentationLoader, MetadataImageSegmentationMultitaskLoader, \
    ImageSegmentationMultitaskLoader, ImageSegmentationLoader
from .models import PyTorchUNet, PyTorchUNetMultitask
from .postprocessing import Thresholder, NucleiLabeler, Dropper, \
    WatershedCenter, WatershedContour, BinaryFillHoles, Postprocessor, \
    resize_image, categorize_image, label_multiclass_image, get_channel, watershed
from .utils import squeeze_inputs, make_apply_transformer


def unet(config, train_mode):
    if train_mode:
        save_output = False
        load_saved_output = False
        preprocessing = preprocessing_train(config)
    else:
        save_output = False
        load_saved_output = False
        preprocessing = preprocessing_inference(config)

    unet = Step(name='unet',
                transformer=PyTorchUNet(**config.unet),
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


def double_unet(config):
    save_output = False
    load_saved_output = False
    preprocessing = preprocessing_inference(config)

    unet_simple = Step(name='unet_simple',
                       transformer=PyTorchUNet(**config.unet_simple),
                       input_steps=[preprocessing],
                       is_trainable=True,
                       cache_dirpath=config.env.cache_dirpath,
                       save_output=save_output, load_saved_output=load_saved_output)

    masks = mask_postprocessing_simple(unet_simple, config, save_output=save_output)

    unet_borders = Step(name='unet_borders',
                        transformer=PyTorchUNet(**config.unet_borders),
                        input_steps=[preprocessing],
                        is_trainable=True,
                        cache_dirpath=config.env.cache_dirpath,
                        save_output=save_output, load_saved_output=load_saved_output)

    borders = mask_postprocessing_borders(unet_borders, config, save_output=save_output)

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

    output.plot_graph('graph.png')

    return output


def unet_multitask(config, train_mode):
    if train_mode:
        save_output = True
        load_saved_output = False
        preprocessing = preprocessing_multitask_train(config)
    else:
        save_output = True
        load_saved_output = False
        preprocessing = preprocessing_multitask_inference(config)

    unet_multitask = Step(name='unet_multitask',
                          transformer=PyTorchUNetMultitask(**config.unet),
                          input_steps=[preprocessing],
                          cache_dirpath=config.env.cache_dirpath,
                          save_output=save_output, load_saved_output=load_saved_output)

    mask_resize = Step(name='mask_resize',
                       transformer=make_apply_transformer(resize_image,
                                                          output_name='resized_images',
                                                          apply_on=['images', 'target_sizes']),
                       input_data=['input'],
                       input_steps=[unet_multitask],
                       adapter={'images': ([(unet_multitask.name, 'mask_prediction')]),
                                'target_sizes': ([('input', 'target_sizes')]),
                                },
                       cache_dirpath=config.env.cache_dirpath,
                       save_output=save_output)

    contour_resize = Step(name='contour_resize',
                          transformer=make_apply_transformer(resize_image,
                                                             output_name='resized_images',
                                                             apply_on=['images', 'target_sizes']),
                          input_data=['input'],
                          input_steps=[unet_multitask],
                          adapter={'images': ([(unet_multitask.name, 'contour_prediction')]),
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
                    save_output=save_output)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[detached],
                  adapter={'y_pred': ([(detached.name, 'labeled_images')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def preprocessing_train(config):
    if config.execution.load_in_memory:
        reader_train = Step(name='reader_train',
                            transformer=ImageReader(**config.reader_single),
                            input_data=['input'],
                            adapter={'meta': ([('input', 'meta')]),
                                     'train_mode': ([('input', 'train_mode')]),
                                     },
                            cache_dirpath=config.env.cache_dirpath)

        reader_inference = Step(name='reader_inference',
                                transformer=ImageReader(**config.reader_single),
                                input_data=['input'],
                                adapter={'meta': ([('input', 'meta_valid')]),
                                         'train_mode': ([('input', 'train_mode')]),
                                         },
                                cache_dirpath=config.env.cache_dirpath)

        loader = Step(name='loader',
                      transformer=ImageSegmentationLoader(**config.loader),
                      input_data=['input'],
                      input_steps=[reader_train, reader_inference],
                      adapter={'X': ([('reader_train', 'X')]),
                               'y': ([('reader_train', 'y')]),
                               'train_mode': ([('input', 'train_mode')]),
                               'X_valid': ([('reader_inference', 'X')]),
                               'y_valid': ([('reader_inference', 'y')]),
                               },
                      cache_dirpath=config.env.cache_dirpath)
    else:
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
    return loader


def preprocessing_inference(config):
    if config.execution.load_in_memory:
        reader_inference = Step(name='reader_inference',
                                transformer=ImageReader(**config.reader_single),
                                input_data=['input'],
                                adapter={'meta': ([('input', 'meta')]),
                                         'train_mode': ([('input', 'train_mode')]),
                                         },
                                cache_dirpath=config.env.cache_dirpath)

        loader = Step(name='loader',
                      transformer=ImageSegmentationLoader(**config.loader),
                      input_data=['input'],
                      input_steps=[reader_inference],
                      adapter={'X': ([('reader_inference', 'X')]),
                               'y': ([('reader_inference', 'y')]),
                               'train_mode': ([('input', 'train_mode')]),
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
                      transformer=MetadataImageSegmentationLoader(**config.loader),
                      input_data=['input'],
                      input_steps=[xy_inference, xy_inference],
                      adapter={'X': ([('xy_inference', 'X')], squeeze_inputs),
                               'y': ([('xy_inference', 'y')], squeeze_inputs),
                               'train_mode': ([('input', 'train_mode')]),
                               },
                      cache_dirpath=config.env.cache_dirpath)
    return loader


def preprocessing_multitask_train(config):
    if config.execution.load_in_memory:
        reader_train = Step(name='reader_train',
                            transformer=ImageReader(**config.reader_multitask),
                            input_data=['input'],
                            adapter={'meta': ([('input', 'meta')]),
                                     'train_mode': ([('input', 'train_mode')]),
                                     },
                            cache_dirpath=config.env.cache_dirpath,
                            save_output=True, load_saved_output=True)

        reader_inference = Step(name='reader_inference',
                                transformer=ImageReader(**config.reader_multitask),
                                input_data=['input'],
                                adapter={'meta': ([('input', 'meta_valid')]),
                                         'train_mode': ([('input', 'train_mode')]),
                                         },
                                cache_dirpath=config.env.cache_dirpath,
                                save_output=True, load_saved_output=True)

        loader = Step(name='loader',
                      transformer=ImageSegmentationMultitaskLoader(**config.loader),
                      input_data=['input'],
                      input_steps=[reader_train, reader_inference],
                      adapter={'X': ([('reader_train', 'X')]),
                               'y': ([('reader_train', 'y')]),
                               'train_mode': ([('input', 'train_mode')]),
                               'X_valid': ([('reader_inference', 'X')]),
                               'y_valid': ([('reader_inference', 'y')]),
                               },
                      cache_dirpath=config.env.cache_dirpath)
    else:
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

    return loader


def preprocessing_multitask_inference(config):
    if config.execution.load_in_memory:
        reader_inference = Step(name='reader_inference',
                                transformer=ImageReader(**config.reader_multitask),
                                input_data=['input'],
                                adapter={'meta': ([('input', 'meta')]),
                                         'train_mode': ([('input', 'train_mode')]),
                                         },
                                cache_dirpath=config.env.cache_dirpath)

        loader = Step(name='loader',
                      transformer=ImageSegmentationMultitaskLoader(**config.loader),
                      input_data=['input'],
                      input_steps=[reader_inference],
                      adapter={'X': ([('reader_inference', 'X')]),
                               'y': ([('reader_inference', 'y')]),
                               'train_mode': ([('input', 'train_mode')]),
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
                      transformer=MetadataImageSegmentationMultitaskLoader(**config.loader),
                      input_data=['input'],
                      input_steps=[xy_inference, xy_inference],
                      adapter={'X': ([('xy_inference', 'X')], squeeze_inputs),
                               'y': ([('xy_inference', 'y')], squeeze_inputs),
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
                                                            channel=config.postprocessor.channels.index('nuclei')),
                                                            output_name='nuclei_images'),
                         input_steps=[labeler],
                         adapter={'images': ([('labeler', 'labeled_images')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath,
                         save_output=save_output)

    return nuclei_filter


def mask_postprocessing_simple(model, config, save_output=False):
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
                                                            channel=config.postprocessor.channels.index('nuclei')),
                                                            output_name='nuclei_images'),
                         input_steps=[category_mapper],
                         adapter={'images': ([('category_mapper_simple', 'categorized_images')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath,
                         save_output=save_output)

    return nuclei_filter


def mask_postprocessing_borders(model, config, save_output=False):
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
                                                             channel=config.postprocessor.channels.index('borders')),
                                                             output_name='nuclei_images'),
                          input_steps=[category_mapper],
                          adapter={'images': ([('category_mapper_borders', 'categorized_images')]),
                                   },
                          cache_dirpath=config.env.cache_dirpath,
                          save_output=save_output)

    return borders_filter


def contour_postprocessing(model, config, save_output=True):
    contour_resize = Step(name='contour_resize',
                          transformer=make_apply_transformer(resize_image,
                                                             output_name='resized_images',
                                                             apply_on=['images', 'target_sizes']),
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
                         transformer=make_apply_transformer(resize_image,
                                                            output_name='resized_images',
                                                            apply_on=['images', 'target_sizes']),
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


PIPELINES = {'unet': {'train': partial(unet, train_mode=True),
                      'inference': partial(unet, train_mode=False),
                      },
             'unet_multitask': {'train': partial(unet_multitask, train_mode=True),
                                'inference': partial(unet_multitask, train_mode=False),
                                },

             'double_unet': {'inference': double_unet,
                             }
             }

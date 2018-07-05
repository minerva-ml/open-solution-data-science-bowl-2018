from functools import partial

from .steppy.base import Step, Dummy
from .steppy.preprocessing.misc import XYSplit, ImageReader

from . import loaders
from .models import PyTorchUNet
from .utils import squeeze_inputs_if_needed, make_apply_transformer
from .postprocessing import resize_image, categorize_image, label_multiclass_image, get_channel, watershed,\
    drop_small_unlabeled, crop_image


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
                input_data=['callback_input'],
                input_steps=[preprocessing],
                is_trainable=True,
                cache_dirpath=config.env.cache_dirpath,
                save_output=save_output, load_saved_output=load_saved_output)

    if train_mode:
        return unet

    mask_postprocessed = mask_postprocessing(unet, config, save_output=save_output)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[mask_postprocessed],
                  adapter={'y_pred': ([(mask_postprocessed.name, 'nuclei_images')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def unet_tta(config):
    save_output = False
    load_saved_output = False
    preprocessing, tta_generator = preprocessing_inference_tta(config, model_name='unet')

    unet = Step(name='unet',
                transformer=PyTorchUNet(**config.model['unet']),
                input_data=['callback_input'],
                input_steps=[preprocessing],
                is_trainable=True,
                cache_dirpath=config.env.cache_dirpath,
                save_output=save_output, load_saved_output=load_saved_output)

    tta_aggregator = aggregator('tta_aggregator', unet,
                                tta_generator=tta_generator,
                                cache_dirpath=config.env.cache_dirpath,
                                save_output=save_output,
                                config=config.tta_aggregator)

    prediction_renamed = Step(name='prediction_renamed',
                              transformer=Dummy(),
                              input_steps=[tta_aggregator],
                              adapter={
                                  'mask_prediction': (
                                      [(tta_aggregator.name, 'aggregated_prediction')]), },
                              cache_dirpath=config.env.cache_dirpath,
                              save_output=save_output)

    mask_postprocessed = mask_postprocessing(prediction_renamed, config, save_output=save_output)

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
                      input_data=['callback_input'],
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
                        input_data=['callback_input'],
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
                      input_data=['callback_input'],
                      input_steps=[preprocessing],
                      is_trainable=True,
                      cache_dirpath=config.env.cache_dirpath,
                      save_output=save_output, load_saved_output=load_saved_output)

    masks, seeds = postprocessing_masks(unet_masks, config, save_output=save_output)

    unet_borders = Step(name='unet_borders',
                        transformer=PyTorchUNet(**config.model['unet_borders']),
                        input_data=['callback_input'],
                        input_steps=[preprocessing],
                        is_trainable=True,
                        cache_dirpath=config.env.cache_dirpath,
                        save_output=save_output, load_saved_output=load_saved_output)

    borders = postprocessing_borders(unet_borders, config, save_output=save_output)

    watersheder = Step(name='watershed',
                       transformer=make_apply_transformer(watershed,
                                                          output_name='watersheded_images',
                                                          apply_on=['masks', 'seeds', 'borders']),
                       input_steps=[masks, seeds, borders],
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


def double_unet_tta(config):
    save_output = False
    load_saved_output = False
    preprocessing, tta_generator = preprocessing_inference_tta(config)

    unet_masks = Step(name='unet_masks',
                      transformer=PyTorchUNet(**config.model['unet_masks']),
                      input_data=['callback_input'],
                      input_steps=[preprocessing],
                      is_trainable=True,
                      cache_dirpath=config.env.cache_dirpath,
                      save_output=save_output,
                      load_saved_output=load_saved_output)

    tta_aggregator_masks = aggregator('tta_aggregator_masks', unet_masks,
                                      tta_generator=tta_generator,
                                      cache_dirpath=config.env.cache_dirpath,
                                      save_output=save_output,
                                      config=config.tta_aggregator)

    prediction_renamed_masks = Step(name='prediction_renamed_masks',
                                    transformer=Dummy(),
                                    input_steps=[tta_aggregator_masks],
                                    adapter={
                                        'mask_prediction': (
                                            [(tta_aggregator_masks.name, 'aggregated_prediction')]), },
                                    cache_dirpath=config.env.cache_dirpath,
                                    save_output=save_output)

    masks, seeds = postprocessing_masks(prediction_renamed_masks, config, save_output=save_output)

    unet_borders = Step(name='unet_borders',
                        transformer=PyTorchUNet(**config.model['unet_borders']),
                        input_data=['callback_input'],
                        input_steps=[preprocessing],
                        is_trainable=True,
                        cache_dirpath=config.env.cache_dirpath,
                        save_output=save_output, load_saved_output=load_saved_output)

    tta_aggregator_borders = aggregator('tta_aggregator_borders', unet_borders,
                                        tta_generator=tta_generator,
                                        cache_dirpath=config.env.cache_dirpath,
                                        save_output=save_output,
                                        config=config.tta_aggregator)

    prediction_renamed_borders = Step(name='prediction_renamed_borders',
                                      transformer=Dummy(),
                                      input_steps=[tta_aggregator_borders],
                                      adapter={
                                          'mask_prediction': (
                                              [(tta_aggregator_borders.name, 'aggregated_prediction')]), },
                                      cache_dirpath=config.env.cache_dirpath,
                                      save_output=save_output)

    borders = postprocessing_borders(prediction_renamed_borders, config, save_output=save_output)

    watersheder = Step(name='watershed',
                       transformer=make_apply_transformer(watershed,
                                                          output_name='watersheded_images',
                                                          apply_on=['masks', 'seeds', 'borders']),
                       input_steps=[masks, seeds, borders],
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


def aggregator(name, model, tta_generator, cache_dirpath, save_output, config):

    tta_aggregator = Step(name=name,
                          transformer=loaders.TestTimeAugmentationAggregator(**config),
                          input_steps=[model, tta_generator],
                          adapter={'images': ([(model.name, 'mask_prediction')]),
                                   'tta_params': ([(tta_generator.name, 'tta_params')]),
                                   'img_ids': ([(tta_generator.name, 'img_ids')]),
                                   },
                          cache_dirpath=cache_dirpath,
                          save_output=save_output)
    return tta_aggregator


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
                            input_data=['input', 'specs'],
                            adapter={'meta': ([('input', 'meta')]),
                                     'train_mode': ([('specs', 'train_mode')]),
                                     },
                            cache_dirpath=config.env.cache_dirpath)

        reader_inference = Step(name='reader_inference',
                                transformer=ImageReader(**config.reader[model_name]),
                                input_data=['callback_input', 'specs'],
                                adapter={'meta': ([('callback_input', 'meta_valid')]),
                                         'train_mode': ([('specs', 'train_mode')]),
                                         },
                                cache_dirpath=config.env.cache_dirpath)

    elif config.loader.dataset_params.image_source == 'disk':
        reader_train = Step(name='xy_train',
                            transformer=XYSplit(**config.xy_splitter[model_name]),
                            input_data=['input', 'specs'],
                            adapter={'meta': ([('input', 'meta')]),
                                     'train_mode': ([('specs', 'train_mode')])
                                     },
                            cache_dirpath=config.env.cache_dirpath)

        reader_inference = Step(name='xy_inference',
                                transformer=XYSplit(**config.xy_splitter[model_name]),
                                input_data=['callback_input', 'specs'],
                                adapter={'meta': ([('callback_input', 'meta_valid')]),
                                         'train_mode': ([('specs', 'train_mode')])
                                         },
                                cache_dirpath=config.env.cache_dirpath)
    else:
        raise NotImplementedError

    loader = Step(name='loader',
                  transformer=Loader(**config.loader),
                  input_data=['specs'],
                  input_steps=[reader_train, reader_inference],
                  adapter={'X': ([(reader_train.name, 'X')], squeeze_inputs_if_needed),
                           'y': ([(reader_train.name, 'y')], squeeze_inputs_if_needed),
                           'train_mode': ([('specs', 'train_mode')]),
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
                                input_data=['input', 'specs'],
                                adapter={'meta': ([('input', 'meta')]),
                                         'train_mode': ([('specs', 'train_mode')]),
                                         },
                                cache_dirpath=config.env.cache_dirpath)

    elif config.loader.dataset_params.image_source == 'disk':
        reader_inference = Step(name='xy_inference',
                                transformer=XYSplit(**config.xy_splitter[model_name]),
                                input_data=['input', 'specs'],
                                adapter={'meta': ([('input', 'meta')]),
                                         'train_mode': ([('specs', 'train_mode')])
                                         },
                                cache_dirpath=config.env.cache_dirpath)
    else:
        raise NotImplementedError

    loader = Step(name='loader',
                  transformer=Loader(**config.loader),
                  input_data=['specs'],
                  input_steps=[reader_inference],
                  adapter={'X': ([(reader_inference.name, 'X')], squeeze_inputs_if_needed),
                           'y': ([(reader_inference.name, 'y')], squeeze_inputs_if_needed),
                           'train_mode': ([('specs', 'train_mode')]),
                           },
                  cache_dirpath=config.env.cache_dirpath,
                  cache_output=True)
    return loader


def preprocessing_inference_tta(config, model_name='unet'):
    if config.execution.loader_mode == 'crop_and_pad':
        Loader = loaders.ImageSegmentationLoaderCropPadTTA
    elif config.execution.loader_mode == 'resize':
        Loader = loaders.ImageSegmentationLoaderResizeTTA
    else:
        raise NotImplementedError

    if config.loader.dataset_params.image_source == 'memory':
        reader_inference = Step(name='reader_inference',
                                transformer=ImageReader(**config.reader[model_name]),
                                input_data=['input', 'specs'],
                                adapter={'meta': ([('input', 'meta')]),
                                         'train_mode': ([('specs', 'train_mode')]),
                                         },
                                cache_dirpath=config.env.cache_dirpath)

        tta_generator = Step(name='tta_generator',
                             transformer=loaders.TestTimeAugmentationGenerator(**config.tta_generator),
                             input_steps=[reader_inference],
                             adapter={'X': ([('reader_inference', 'X')]),
                                      },
                             cache_dirpath=config.env.cache_dirpath)

    elif config.loader.dataset_params.image_source == 'disk':
        reader_inference = Step(name='reader_inference',
                                transformer=XYSplit(**config.xy_splitter[model_name]),
                                input_data=['input', 'specs'],
                                adapter={'meta': ([('input', 'meta')]),
                                         'train_mode': ([('specs', 'train_mode')])
                                         },
                                cache_dirpath=config.env.cache_dirpath)

        tta_generator = Step(name='tta_generator',
                             transformer=loaders.MetaTestTimeAugmentationGenerator(**config.tta_generator),
                             input_steps=[reader_inference],
                             adapter={'X': ([('reader_inference', 'X')]),
                                      },
                             cache_dirpath=config.env.cache_dirpath)
    else:
        raise NotImplementedError

    loader = Step(name='loader',
                  transformer=Loader(**config.loader),
                  input_data=['specs'],
                  input_steps=[tta_generator],
                  adapter={'X': ([(tta_generator.name, 'X_tta')], squeeze_inputs_if_needed),
                           'tta_params': ([(tta_generator.name, 'tta_params')]),
                           },
                  cache_dirpath=config.env.cache_dirpath,
                  cache_output=True)
    return loader, tta_generator


def mask_postprocessing(model, config, save_output=False):
    if config.execution.loader_mode == 'crop_and_pad':
        size_adjustment_function = crop_image
    elif config.execution.loader_mode == 'resize':
        size_adjustment_function = resize_image
    else:
        raise NotImplementedError

    mask_resize = Step(name='mask_resize',
                       transformer=make_apply_transformer(size_adjustment_function,
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
                           transformer=make_apply_transformer(
                                     partial(categorize_image,
                                             activation='sigmoid',
                                             threshold=config.thresholder.threshold_masks
                                             ),
                                     output_name='categorized_images'),
                           input_steps=[mask_resize],
                           adapter={'images': ([('mask_resize', 'resized_images')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath,
                           save_output=save_output)

    dropper = Step(name='dropper',
                   transformer=make_apply_transformer(partial(drop_small_unlabeled,
                                                              min_size=config.dropper.min_mask_size),
                                                      output_name='cleaned_images'),
                   input_steps=[category_mapper],
                   adapter={'images': ([('category_mapper', 'categorized_images')]),
                            },
                   cache_dirpath=config.env.cache_dirpath,
                   save_output=save_output)

    labeler = Step(name='labeler',
                   transformer=make_apply_transformer(label_multiclass_image,
                                                      output_name='labeled_images'),
                   input_steps=[dropper],
                   adapter={'images': ([('dropper', 'cleaned_images')]),
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
    if config.execution.loader_mode == 'crop_and_pad':
        size_adjustment_function = crop_image
    elif config.execution.loader_mode == 'resize':
        size_adjustment_function = resize_image
    else:
        raise NotImplementedError

    mask_resize = Step(name='mask_resize_masks',
                       transformer=make_apply_transformer(size_adjustment_function,
                                                          output_name='resized_images',
                                                          apply_on=['images', 'target_sizes']),
                       input_data=['input'],
                       input_steps=[model],
                       adapter={'images': ([(model.name, 'mask_prediction')]),
                                'target_sizes': ([('input', 'target_sizes')]),
                                },
                       cache_dirpath=config.env.cache_dirpath,
                       save_output=save_output,
                       cache_output=True)

    category_mapper_masks = Step(name='category_mapper_masks',
                                 transformer=make_apply_transformer(
                                     partial(categorize_image,
                                             activation='sigmoid',
                                             threshold=config.thresholder.threshold_masks
                                             ),
                                     output_name='categorized_images'),
                                 input_steps=[mask_resize],
                                 adapter={'images': ([('mask_resize_masks', 'resized_images')]),
                                          },
                                 cache_dirpath=config.env.cache_dirpath,
                                 save_output=save_output)

    masks_filter = Step(name='masks_filter',
                        transformer=make_apply_transformer(partial(get_channel,
                                                                   channel=config.postprocessor.channels.index('nuclei')),
                                                           output_name='masks'),
                        input_steps=[category_mapper_masks],
                        adapter={'images': ([('category_mapper_masks', 'categorized_images')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath,
                        save_output=save_output)

    dropper_masks = Step(name='dropper_masks',
                         transformer=make_apply_transformer(partial(drop_small_unlabeled,
                                                                    min_size=config.dropper.min_mask_size),
                                                            output_name='masks'),
                         input_steps=[masks_filter],
                         adapter={'images': ([('masks_filter', 'masks')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath,
                         save_output=save_output)

    category_mapper_seeds = Step(name='category_mapper_seeds',
                                 transformer=make_apply_transformer(
                                     partial(categorize_image,
                                             activation='sigmoid',
                                             threshold=config.thresholder.threshold_seeds
                                             ),
                                     output_name='categorized_images'),
                                 input_steps=[mask_resize],
                                 adapter={'images': ([('mask_resize_masks', 'resized_images')]),
                                          },
                                 cache_dirpath=config.env.cache_dirpath,
                                 save_output=save_output)

    seeds_filter = Step(name='seeds_filter',
                        transformer=make_apply_transformer(
                            partial(get_channel,
                                    channel=config.postprocessor.channels.index('nuclei')),
                            output_name='seeds'),
                        input_steps=[category_mapper_seeds],
                        adapter={'images': ([('category_mapper_seeds', 'categorized_images')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath,
                        save_output=save_output)

    dropper_seeds = Step(name='dropper_seeds',
                         transformer=make_apply_transformer(partial(drop_small_unlabeled,
                                                            min_size=config.dropper.min_seed_size),
                                                            output_name='seeds'),
                         input_steps=[seeds_filter],
                         adapter={'images': ([('seeds_filter', 'seeds')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath,
                         save_output=save_output)

    return dropper_masks, dropper_seeds


def postprocessing_borders(model, config, save_output=False):
    if config.execution.loader_mode == 'crop_and_pad':
        size_adjustment_function = crop_image
    elif config.execution.loader_mode == 'resize':
        size_adjustment_function = resize_image
    else:
        raise NotImplementedError

    mask_resize = Step(name='mask_resize_borders',
                       transformer=make_apply_transformer(size_adjustment_function,
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
                           transformer=make_apply_transformer(
                                     partial(categorize_image,
                                             activation='sigmoid',
                                             threshold=config.thresholder.threshold_borders
                                             ),
                                     output_name='categorized_images'),
                           input_steps=[mask_resize],
                           adapter={'images': ([('mask_resize_borders', 'resized_images')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath,
                           save_output=save_output)

    borders_filter = Step(name='borders_filter_borders',
                          transformer=make_apply_transformer(partial(get_channel,
                                                             channel=config.postprocessor.channels.index('borders')),
                                                             output_name='borders'),
                          input_steps=[category_mapper],
                          adapter={'images': ([('category_mapper_borders', 'categorized_images')]),
                                   },
                          cache_dirpath=config.env.cache_dirpath,
                          save_output=save_output)

    return borders_filter


PIPELINES = {'unet': {'train': partial(unet, train_mode=True),
                      'inference': partial(unet, train_mode=False),
                      },
             'unet_tta': {'inference': unet_tta,
                          },
             'unet_masks': {'train': unet_masks,
                            },
             'unet_borders': {'train': unet_borders,
                              },
             'double_unet': {'inference': double_unet,
                             },
             'double_unet_tta': {'inference': double_unet_tta,
                                 }
             }

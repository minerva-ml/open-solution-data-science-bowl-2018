import os

from deepsense import neptune
from attrdict import AttrDict

from utils import read_params

ctx = neptune.Context()
params = read_params(ctx)

SIZE_COLUMNS = ['height', 'width']
X_COLUMNS = ['file_path_image']
Y_COLUMNS = ['file_path_mask']
Y_COLUMNS_MULTITASK = ['file_path_mask', 'file_path_contours', 'file_path_contours_touching', 'file_path_centers']
Y_COLUMNS_SPECIALISTS = ['file_path_mask', 'file_path_contours']
Y_COLUMNS_SCORING = ['file_path_masks']

GLOBAL_CONFIG = {'exp_root': params.experiment_dir,
                 'load_in_memory': params.load_in_memory,
                 'num_workers': params.num_workers,
                 'num_classes': 2,
                 'img_H-W': (params.image_h, params.image_w),
                 'batch_size_train': params.batch_size_train,
                 'batch_size_inference': params.batch_size_inference
                 }

SOLUTION_CONFIG = AttrDict({
    'env': {'cache_dirpath': params.experiment_dir},
    'execution': GLOBAL_CONFIG,
    'xy_splitter': {'x_columns': X_COLUMNS,
                    'y_columns': Y_COLUMNS,
                    },
    'xy_splitter_multitask': {'x_columns': X_COLUMNS,
                              'y_columns': Y_COLUMNS_MULTITASK
                              },
    'xy_splitter_specialists': {'x_columns': X_COLUMNS,
                                'y_columns': Y_COLUMNS_SPECIALISTS
                                },
    'reader_single': {'x_columns': X_COLUMNS,
                      'y_columns': Y_COLUMNS,
                      'target_shape': GLOBAL_CONFIG['img_H-W']
                      },
    'reader_multitask': {'x_columns': X_COLUMNS,
                         'y_columns': Y_COLUMNS_MULTITASK,
                         'target_shape': GLOBAL_CONFIG['img_H-W']
                         },
    'reader_specialists': {'x_columns': X_COLUMNS,
                           'y_columns': Y_COLUMNS_SPECIALISTS,
                           'target_shape': GLOBAL_CONFIG['img_H-W']
                           },
    'loader': {'dataset_params': {'h': params.image_h,
                                  'w': params.image_w,
                                  },
               'loader_params': {'training': {'batch_size': params.batch_size_train,
                                              'shuffle': True,
                                              'num_workers': params.num_workers,
                                              'pin_memory': params.pin_memory
                                              },
                                 'inference': {'batch_size': params.batch_size_inference,
                                               'shuffle': False,
                                               'num_workers': params.num_workers,
                                               'pin_memory': params.pin_memory
                                               },
                                 },
               },
    'unet': {
        'architecture_config': {'model_params': {'n_filters': params.n_filters,
                                                 'conv_kernel': params.conv_kernel,
                                                 'pool_kernel': params.pool_kernel,
                                                 'pool_stride': params.pool_stride,
                                                 'repeat_blocks': params.repeat_blocks,
                                                 'batch_norm': params.use_batch_norm,
                                                 'dropout': params.dropout_conv,
                                                 'in_channels': params.image_channels,
                                                 'nr_outputs': params.nr_unet_outputs
                                                 },
                                'optimizer_params': {'lr': params.lr,
                                                     },
                                'regularizer_params': {'regularize': True,
                                                       'weight_decay_conv2d': params.l2_reg_conv,
                                                       },
                                'weights_init': {'function': 'xavier',
                                                 },
                                'loss_weights': {'mask': params.mask,
                                                 'contour': params.contour,
                                                 'contour_touching': params.contour_touching,
                                                 'center': params.center,
                                                 },
                                },
        'training_config': {'epochs': params.epochs_nr,
                            'shuffle': True,
                            'batch_size': params.batch_size_train,
                            },
        'callbacks_config': {
            'model_checkpoint': {
                'filepath': os.path.join(GLOBAL_CONFIG['exp_root'], 'checkpoints', 'network', 'best.torch'),
                'epoch_every': 1},
            'lr_scheduler': {'gamma': params.gamma,
                             'epoch_every': 1},
            'training_monitor': {'batch_every': 0,
                                 'epoch_every': 1},
            'experiment_timing': {'batch_every': 0,
                                  'epoch_every': 1},
            'validation_monitor': {'epoch_every': 1},
            'neptune_monitor': {'model_name': 'unet',
                                'image_nr': 4,
                                'image_resize': 0.2},
            'early_stopping': {'patience': params.patience},
        },
    },
    'unet_mask': {
        'architecture_config': {'model_params': {'n_filters': params.mask_n_filters,
                                                 'conv_kernel': params.mask_conv_kernel,
                                                 'pool_kernel': params.mask_pool_kernel,
                                                 'pool_stride': params.mask_pool_stride,
                                                 'repeat_blocks': params.mask_repeat_blocks,
                                                 'batch_norm': params.use_batch_norm,
                                                 'dropout': params.dropout_conv,
                                                 'in_channels': params.image_channels,
                                                 'nr_outputs': params.mask_nr_unet_outputs
                                                 },
                                'optimizer_params': {'lr': params.lr,
                                                     },
                                'regularizer_params': {'regularize': True,
                                                       'weight_decay_conv2d': params.l2_reg_conv,
                                                       },
                                'weights_init': {'function': 'xavier',
                                                 },
                                'loss_weights': {'mask': params.mask_mask,
                                                 'contour': params.mask_contour,
                                                 'contour_touching': params.mask_contour_touching,
                                                 'center': params.mask_center,
                                                 },
                                },
        'training_config': {'epochs': params.epochs_nr,
                            'shuffle': True,
                            'batch_size': params.batch_size_train,
                            },
        'callbacks_config': {
            'model_checkpoint': {
                'filepath': os.path.join(GLOBAL_CONFIG['exp_root'], 'checkpoints_mask', 'network', 'best.torch'),
                'epoch_every': 1},
            'lr_scheduler': {'gamma': params.gamma,
                             'epoch_every': 1},
            'training_monitor': {'batch_every': 0,
                                 'epoch_every': 1},
            'experiment_timing': {'batch_every': 0,
                                  'epoch_every': 1},
            'validation_monitor': {'epoch_every': 1},
            'neptune_monitor': {'model_name': 'unet_mask',
                                'image_nr': 4,
                                'image_resize': 0.2},
            'early_stopping': {'patience': params.patience},
        },
    },
    'unet_contour': {
        'architecture_config': {'model_params': {'n_filters': params.contour_n_filters,
                                                 'conv_kernel': params.contour_conv_kernel,
                                                 'pool_kernel': params.contour_pool_kernel,
                                                 'pool_stride': params.contour_pool_stride,
                                                 'repeat_blocks': params.contour_repeat_blocks,
                                                 'batch_norm': params.use_batch_norm,
                                                 'dropout': params.dropout_conv,
                                                 'in_channels': params.image_channels,
                                                 'nr_outputs': params.contour_nr_unet_outputs
                                                 },
                                'optimizer_params': {'lr': params.lr,
                                                     },
                                'regularizer_params': {'regularize': True,
                                                       'weight_decay_conv2d': params.l2_reg_conv,
                                                       },
                                'weights_init': {'function': 'xavier',
                                                 },
                                'loss_weights': {'mask': params.contour_mask,
                                                 'contour': params.contour_contour,
                                                 'contour_touching': params.contour_contour_touching,
                                                 'center': params.contour_center,
                                                 },
                                },
        'training_config': {'epochs': params.epochs_nr,
                            'shuffle': True,
                            'batch_size': params.batch_size_train,
                            },
        'callbacks_config': {
            'model_checkpoint': {
                'filepath': os.path.join(GLOBAL_CONFIG['exp_root'], 'checkpoints_contour', 'network', 'best.torch'),
                'epoch_every': 1},
            'lr_scheduler': {'gamma': params.gamma,
                             'epoch_every': 1},
            'training_monitor': {'batch_every': 0,
                                 'epoch_every': 1},
            'experiment_timing': {'batch_every': 0,
                                  'epoch_every': 1},
            'validation_monitor': {'epoch_every': 1},
            'neptune_monitor': {'model_name': 'unet_contour',
                                'image_nr': 4,
                                'image_resize': 0.2},
            'early_stopping': {'patience': params.patience},
        },
    },
    'thresholder': {'threshold': params.threshold},
    'watershed': {},
    'dropper': {'min_size': params.min_nuclei_size},
    'postprocessor': {}
})

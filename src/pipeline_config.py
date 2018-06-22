import os

from deepsense import neptune
from attrdict import AttrDict

from utils import read_params

ctx = neptune.Context()
params = read_params(ctx)

SIZE_COLUMNS = ['height', 'width']
X_COLUMNS = ['file_path_image']
Y_COLUMNS = ['file_path_mask']
Y_COLUMNS_MULTITASK = ['file_path_mask',
                       'file_path_contours',
                       'file_path_contours_touching',
                       'file_path_centers']
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
    'reader_single': {'x_columns': X_COLUMNS,
                      'y_columns': Y_COLUMNS,
                      'target_shape': GLOBAL_CONFIG['img_H-W']
                      },
    'reader_multitask': {'x_columns': X_COLUMNS,
                         'y_columns': Y_COLUMNS_MULTITASK,
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
    'thresholder': {'threshold': params.threshold},
    'watershed': {},
    'dropper': {'min_size': params.min_nuclei_size},
    'postprocessor': {}
})

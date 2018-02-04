import os

from attrdict import AttrDict

from utils import read_params

params = read_params()

SIZE_COLUMNS = ['height', 'width']
X_COLUMNS = ['file_path_image']
Y_COLUMNS = ['file_path_mask']
Y_COLUMNS_MULTITASK = ['file_path_mask', 'file_path_contours', 'file_path_centers']

GLOBAL_CONFIG = {'exp_root': params.experiment_dir,
                 'num_workers': 6,
                 'num_classes': 2,
                 'img_H-W': (256, 256),
                 'batch_size_train': 32,
                 'batch_size_inference': 32,
                 }

SOLUTION_CONFIG = AttrDict({
    'env': {'cache_dirpath': params.experiment_dir},
    'xy_splitter': {'x_columns': X_COLUMNS,
                    'y_columns': Y_COLUMNS
                    },
    'xy_splitter_multitask': {'x_columns': X_COLUMNS,
                              'y_columns': Y_COLUMNS_MULTITASK
                              },
    'loader': {'dataset_params': {'h': params.image_h,
                                  'w': params.image_w,
                                  },
               'loader_params': {'training': {'batch_size': params.batch_size_train,
                                              'shuffle': True,
                                              'num_workers': params.num_workers
                                              },
                                 'inference': {'batch_size': params.batch_size_inference,
                                               'shuffle': False,
                                               'num_workers': params.num_workers
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
            'lr_scheduler': {'gamma': 0.9955,
                             'epoch_every': 1},
            'training_monitor': {'batch_every': 1,
                                 'epoch_every': 1},
            'validation_monitor': {'epoch_every': 1},
            'neptune_monitor': {'image_nr': 4,
                                'image_resize': 0.2},
            'early_stopping': {'patience': params.patience},
        },
    },
    'thresholder': {'threshold': 0.5},
    'watershed': {}
})

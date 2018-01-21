import os

from attrdict import AttrDict
from deepsense import neptune

from utils import read_params

ctx = neptune.Context()
params = read_params(ctx)

SIZE_COLUMNS = ['height', 'width']
X_COLUMNS = ['file_path_image']
Y_COLUMNS = ['file_path_mask']

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
    'unet_network': {
        'architecture_config': {'model_params': {'num_classes': params.image_width,
                                                 'in_channels': params.image_height,
                                                 'depth': params.kernel,
                                                 'start_filts': params.stride,
                                                 },
                                # 'model_params': {'image_width': params.image_width,
                                #                                      'image_height': params.image_height,
                                #                                      'kernel': params.kernel,
                                #                                      'stride': params.stride,
                                #                                      'padding': params.padding,
                                #                                      'nonlinearity': params.nonlinearity,
                                #                                      'repeat_blocks': params.repeat_blocks,
                                #                                      'n_filters': params.n_filters,
                                #                                      'batch_norm': params.batch_norm,
                                #                                      'dropout': params.dropout,
                                #                                      },
                                'optimizer_params': {'lr': params.lr,
                                                     # 'momentum': params.momentum,
                                                     # 'nesterov': True
                                                     },
                                'regularizer_params': {'regularize': True,
                                                       'weight_decay_conv2d': params.l2_reg_convo,
                                                       'weight_decay_linear': params.l2_reg_dense
                                                       },
                                'weights_init': {'function': 'xavier',
                                                 'params': {},
                                                 },
                                },
        'training_config': {'epochs': params.epochs_nr,
                            'shuffle': True,
                            'batch_size': params.batch_size_train,
                            },
        'callbacks_config': {'model_checkpoint': {
            'checkpoint_dir': os.path.join(GLOBAL_CONFIG['exp_root'], 'checkpoints', 'network'),
            'epoch_every': 1},
            'lr_scheduler': {'gamma': 0.9955,
                             'epoch_every': 1},
            'training_monitor': {'batch_every': 1,
                                 'epoch_every': 1},
            'validation_monitor': {'epoch_every': 1},
            'neptune_monitor': {},
        },

    },
    'thresholder': {'threshold': 0.5},
})

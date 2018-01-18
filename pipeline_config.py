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
    'loader': {'loader_params': {'training': {'batch_size': params.batch_size_train,
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
        'architecture_config': {'model_params': {},
                                'optimizer_params': {'lr': params.lr,
                                                     'momentum': params.momentum,
                                                     'nesterov': True
                                                     },
                                'regularizer_params': {'regularize': True,
                                                       'weight_decay_conv2d': 0.0001,
                                                       'weight_decay_linear': 0.001
                                                       },
                                'weights_init': {'function': 'normal',
                                                 'params': {'mean': 0,
                                                            'std_conv2d': 0.01,
                                                            'std_linear': 0.001
                                                            },
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
    'thresholder': {'threshold': 0.2},
})

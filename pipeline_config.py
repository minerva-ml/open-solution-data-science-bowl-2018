import os

from attrdict import AttrDict
from deepsense import neptune

from utils import read_params

ctx = neptune.Context()
params = read_params(ctx)
print(params)

SIZE_COLUMNS = ['height', 'width']
X_COLUMNS = ['file_path_image']
Y_COLUMNS = ['file_path_mask']

SOLUTION_CONFIG = AttrDict({
    'env': {'cache_dirpath': params.experiment_dir},
    'xy_splitter': {'x_columns': X_COLUMNS,
                    'y_columns': Y_COLUMNS
                    },
    'loader': {'params': {}},
    'unet_network': {
        'architecture_config': {'model_params': {'filter_nr': params.filter_nr,
                                                 'kernel_size': params.kernel_size,
                                                 'repeat_block': params.repeat_block,
                                                 'l2_reg_convo': params.l2_reg_convo,
                                                 'use_prelu': bool(params.use_prelu),
                                                 'use_batch_norm': bool(params.use_batch_norm),
                                                 'dropout_convo': params.dropout_convo,
                                                 },
                                'optimizer_params': {'lr': params.lr,
                                                     'momentum': params.momentum,
                                                     'nesterov': True
                                                     },
                                },
        'training_config': {'epochs': params.epochs_nr,
                            'shuffle': True,
                            'batch_size': params.batch_size_train,
                            },
        'callbacks_config': {'model_checkpoint': {
            'filepath': os.path.join(params.experiment_dir, 'checkpoints',
                                     'unet_network',
                                     'unet_network.h5'),
            'save_best_only': True,
            'save_weights_only': False},
            'lr_scheduler': {'gamma': params.gamma},
            'early_stopping': {'patience': params.patience},
            'neptune_monitor': {},
        },
    },
    'thresholder': {'threshold': 0.8},
})

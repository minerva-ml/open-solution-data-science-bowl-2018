import numpy as np
import torch
from PIL import Image
from deepsense import neptune
from torch.autograd import Variable
from functools import partial
from tempfile import TemporaryDirectory

from .steppy.base import Step, Dummy
from .steppy.utils import get_logger
from .steppy.pytorch.utils import save_model
from .steppy.pytorch.callbacks import NeptuneMonitor, ValidationMonitor, ModelCheckpoint

from . import postprocessing as post
from .utils import sigmoid, softmax, make_apply_transformer, read_masks, get_list_of_image_predictions
from .pipeline_config import Y_COLUMNS_SCORING, CHANNELS, SIZE_COLUMNS
from .metrics import intersection_over_union, intersection_over_union_thresholds

logger = get_logger()


class NeptuneMonitorSegmentation(NeptuneMonitor):
    def __init__(self, image_nr, image_resize, model_name):
        super().__init__(model_name)
        self.image_nr = image_nr
        self.image_resize = image_resize

    def on_epoch_end(self, *args, **kwargs):
        self._send_numeric_channels()
        # self._send_image_channels()
        self.epoch_id += 1

    def _send_image_channels(self):
        self.model.eval()
        pred_masks = self.get_prediction_masks()
        self.model.train()

        for name, pred_mask in pred_masks.items():
            for i, image_duplet in enumerate(pred_mask):
                h, w = image_duplet.shape[1:]
                image_glued = np.zeros((h, 2 * w + 10))

                image_glued[:, :w] = image_duplet[0, :, :]
                image_glued[:, (w + 10):] = image_duplet[1, :, :]

                pill_image = Image.fromarray((image_glued * 255.).astype(np.uint8))
                h_, w_ = image_glued.shape
                pill_image = pill_image.resize((int(self.image_resize * w_), int(self.image_resize * h_)),
                                               Image.ANTIALIAS)

                self.ctx.channel_send('{} {}'.format(self.model_name, name), neptune.Image(
                    name='epoch{}_batch{}_idx{}'.format(self.epoch_id, self.batch_id, i),
                    description="true and prediction masks",
                    data=pill_image))

                if i == self.image_nr:
                    break

    def get_prediction_masks(self):
        prediction_masks = {}
        batch_gen, steps = self.validation_datagen
        for batch_id, data in enumerate(batch_gen):
            if len(data) != len(self.output_names) + 1:
                raise ValueError('incorrect targets provided')
            X = data[0]
            targets_tensors = data[1:]

            if torch.cuda.is_available():
                X = Variable(X).cuda()
            else:
                X = Variable(X)

            outputs_batch = self.model(X)
            if len(outputs_batch) == len(self.output_names):
                for name, output, target in zip(self.output_names, outputs_batch, targets_tensors):
                    prediction = sigmoid(np.squeeze(output.data.cpu().numpy(), axis=1))
                    ground_truth = np.squeeze(target.cpu().numpy(), axis=1)
                    prediction_masks[name] = np.stack([prediction, ground_truth], axis=1)
            else:
                for name, target in zip(self.output_names, targets_tensors):
                    prediction = sigmoid(np.squeeze(outputs_batch.data.cpu().numpy(), axis=1))
                    ground_truth = np.squeeze(target.cpu().numpy(), axis=1)
                    prediction_masks[name] = np.stack([prediction, ground_truth], axis=1)
            break
        return prediction_masks


class ValidationMonitorSegmentation(ValidationMonitor):
    def __init__(self, data_dir, loader_mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.validation_pipeline = postprocessing__pipeline_simplified
        self.loader_mode = loader_mode
        self.validation_loss = None
        self.meta_valid = None
        self.y_true = None

    def set_params(self, transformer, validation_datagen, meta_valid=None, *args, **kwargs):
        self.model = transformer.model
        self.optimizer = transformer.optimizer
        self.loss_function = transformer.loss_function
        self.output_names = transformer.output_names
        self.validation_datagen = validation_datagen
        self.meta_valid = meta_valid
        self.validation_loss = transformer.validation_loss
        self.y_true = read_masks(self.meta_valid[Y_COLUMNS_SCORING].values)

    def get_validation_loss(self):
        return self._get_validation_loss()

    def _get_validation_loss(self):
        output, epoch_loss = self._transform()
        y_pred = self._generate_prediction(output)

        logger.info('Calculating IOU and IOUT Scores')
        iou_score = intersection_over_union(self.y_true, y_pred)
        iout_score = intersection_over_union_thresholds(self.y_true, y_pred)
        logger.info('IOU score on validation is {}'.format(iou_score))
        logger.info('IOUT score on validation is {}'.format(iout_score))

        return self.validation_loss.setdefault(self.epoch_id, {'sum': epoch_loss,
                                                               'iou': Variable(torch.Tensor([iou_score])),
                                                               'iout': Variable(torch.Tensor([iout_score]))})

    def _transform(self):
        self.model.eval()
        batch_gen, steps = self.validation_datagen
        partial_batch_losses = []
        outputs = {}
        for batch_id, data in enumerate(batch_gen):
            X = data[0]
            targets_tensors = data[1:]

            if torch.cuda.is_available():
                X = Variable(X, volatile=True).cuda()
                targets_var = []
                for target_tensor in targets_tensors:
                    targets_var.append(Variable(target_tensor, volatile=True).cuda())
            else:
                X = Variable(X, volatile=True)
                targets_var = []
                for target_tensor in targets_tensors:
                    targets_var.append(Variable(target_tensor, volatile=True))

            outputs_batch = self.model(X)
            if len(self.output_names) == 1:
                for (name, loss_function_one, weight), target in zip(self.loss_function, targets_var):
                    loss_sum = loss_function_one(outputs_batch, target) * weight
                outputs.setdefault(self.output_names[0], []).append(outputs_batch.data.cpu().numpy())
            else:
                batch_losses = []
                for (name, loss_function_one, weight), output, target in zip(self.loss_function, outputs_batch, targets_var):
                    loss = loss_function_one(output, target) * weight
                    batch_losses.append(loss)
                    partial_batch_losses.setdefault(name, []).append(loss)
                    output_ = output.data.cpu().numpy()
                    outputs.setdefault(name, []).append(output_)
                loss_sum = sum(batch_losses)
            partial_batch_losses.append(loss_sum)
            if batch_id == steps:
                break
        self.model.train()
        average_losses = sum(partial_batch_losses) / steps
        outputs = {'{}_prediction'.format(name): get_list_of_image_predictions(outputs_) for name, outputs_ in outputs.items()}
        for name, prediction in outputs.items():
            outputs[name] = [softmax(single_prediction, axis=0) for single_prediction in prediction]

        return outputs, average_losses

    def _generate_prediction(self, outputs):
        data = {'callback_input': {'meta': self.meta_valid,
                                   'meta_valid': None,
                                   'target_sizes': self.meta_valid[SIZE_COLUMNS].values,
                                   },
                'unet_output': {**outputs}
                }
        with TemporaryDirectory() as cache_dirpath:
            pipeline = self.validation_pipeline(cache_dirpath, self.loader_mode)
            output = pipeline.transform(data)
        y_pred = output['y_pred']

        return y_pred


class ModelCheckpointSegmentation(ModelCheckpoint):
    def __init__(self, metric_name='sum', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_name = metric_name

    def on_epoch_end(self, *args, **kwargs):
        if self.epoch_every and ((self.epoch_id % self.epoch_every) == 0):
            self.model.eval()
            val_loss = self.get_validation_loss()
            loss_sum = val_loss[self.metric_name]
            loss_sum = loss_sum.data.cpu().numpy()[0]

            self.model.train()

            if self.best_score is None:
                self.best_score = loss_sum

            if (self.minimize and loss_sum < self.best_score) or (not self.minimize and loss_sum > self.best_score) or (
                    self.epoch_id == 0):
                self.best_score = loss_sum
                save_model(self.model, self.filepath)
                logger.info('epoch {0} model saved to {1}'.format(self.epoch_id, self.filepath))

        self.epoch_id += 1


def postprocessing__pipeline_simplified(cache_dirpath, loader_mode):
    if loader_mode == 'crop_and_pad':
        size_adjustment_function = post.crop_image
    elif loader_mode == 'resize':
        size_adjustment_function = post.resize_image
    else:
        raise NotImplementedError

    mask_resize = Step(name='mask_resize',
                       transformer=make_apply_transformer(size_adjustment_function,
                                                          output_name='resized_images',
                                                          apply_on=['images', 'target_sizes']),
                       input_data=['unet_output', 'callback_input'],
                       adapter={'images': ([('unet_output', 'mask_prediction')]),
                                'target_sizes': ([('callback_input', 'target_sizes')]),
                                },
                       cache_dirpath=cache_dirpath)

    category_mapper = Step(name='category_mapper',
                           transformer=make_apply_transformer(post.categorize_image,
                                                              output_name='categorized_images'),
                           input_steps=[mask_resize],
                           adapter={'images': ([('mask_resize', 'resized_images')]),
                                    },
                           cache_dirpath=cache_dirpath)

    labeler = Step(name='labeler',
                   transformer=make_apply_transformer(post.label_multiclass_image,
                                                      output_name='labeled_images'),
                   input_steps=[category_mapper],
                   adapter={'images': ([(category_mapper.name, 'categorized_images')]),
                            },
                   cache_dirpath=cache_dirpath)

    nuclei_filter = Step(name='nuclei_filter',
                         transformer=make_apply_transformer(partial(post.get_channel,
                                                                    channel=CHANNELS.index('nuclei')),
                                                            output_name='nuclei_images'),
                         input_steps=[labeler],
                         adapter={'images': ([('labeler', 'labeled_images')]),
                                  },
                         cache_dirpath=cache_dirpath)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[nuclei_filter],
                  adapter={'y_pred': ([('nuclei_filter', 'nuclei_images')]),
                           },
                  cache_dirpath=cache_dirpath)

    return output

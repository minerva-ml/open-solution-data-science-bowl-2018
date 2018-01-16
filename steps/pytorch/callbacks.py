import os
import shutil
from datetime import datetime, timedelta

import names
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from deepsense import neptune
from torch.optim.lr_scheduler import ExponentialLR

from .validation import score_model, predict_on_batch_multi_output
from .utils import get_logger, Averager, save_model

logger = get_logger()


class Callback:
    def __init__(self):
        self.epoch_id = None
        self.batch_id = None

        self.model = None
        self.optimizer = None
        self.loss_function = None
        self.validation_datagen = None
        self.lr_scheduler = None

    def set_params(self, transformer, validation_datagen):
        self.model = transformer.model
        self.optimizer = transformer.optimizer
        self.loss_function = transformer.loss_function
        self.validation_datagen = validation_datagen

    def on_train_begin(self, *args, **kwargs):
        self.epoch_id = 0
        self.batch_id = 0

    def on_train_end(self, *args, **kwargs):
        pass

    def on_epoch_begin(self, *args, **kwargs):
        pass

    def on_epoch_end(self, *args, **kwargs):
        self.epoch_id += 1
        self.batch_id = 0

    def on_batch_begin(self, *args, **kwargs):
        pass

    def on_batch_end(self, *args, **kwargs):
        self.batch_id += 1


class CallbackList:
    def __init__(self, callbacks=None):
        if callbacks is None:
            self.callbacks = []
        elif isinstance(callbacks, Callback):
            self.callbacks = [callbacks]
        else:
            self.callbacks = callbacks

    def __len__(self):
        return len(self.callbacks)

    def set_params(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.set_params(*args, **kwargs)

    def on_train_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_train_begin(*args, **kwargs)

    def on_train_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_train_end(*args, **kwargs)

    def on_epoch_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_begin(*args, **kwargs)

    def on_epoch_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(*args, **kwargs)

    def on_batch_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_begin(*args, **kwargs)

    def on_batch_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_end(*args, **kwargs)


class TrainingMonitor(Callback):
    def __init__(self, epoch_every=None, batch_every=None):
        super().__init__()
        self.epoch_loss_averager = Averager()
        self.epoch_acc_averager = Averager()
        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every
        if batch_every == 0:
            self.batch_every = False
        else:
            self.batch_every = batch_every

    def on_train_begin(self, *args, **kwargs):
        self.epoch_loss_averager.reset()
        self.epoch_acc_averager.reset()
        self.epoch_id = 0
        self.batch_id = 0

    def on_epoch_end(self, *args, **kwargs):
        epoch_avg_loss = self.epoch_loss_averager.value
        epoch_avg_acc = self.epoch_acc_averager.value
        self.epoch_loss_averager.reset()
        self.epoch_acc_averager.reset()
        if self.epoch_every and ((self.epoch_id % self.epoch_every) == 0):
            logger.info('epoch {0} loss:     {1:.5f}'.format(self.epoch_id, epoch_avg_loss))
            logger.info('epoch {0} accuracy: {1:.5f}'.format(self.epoch_id, epoch_avg_acc))
        self.epoch_id += 1
        self.batch_id = 0

    def on_batch_end(self, metrics, *args, **kwargs):
        batch_loss = metrics['batch_loss']
        batch_acc = metrics['batch_acc']
        self.epoch_loss_averager.send(batch_loss)
        self.epoch_acc_averager.send(batch_acc)
        if self.batch_every and ((self.batch_id % self.batch_every) == 0):
            logger.info('epoch {0} batch {1} loss:     {2:.5f}'.format(self.epoch_id, self.batch_id, batch_loss))
            logger.info('epoch {0} batch {1} accuracy: {2:.5f}'.format(self.epoch_id, self.batch_id, batch_acc))
        self.batch_id += 1


class ValidationMonitor(Callback):
    def __init__(self, epoch_every=None, batch_every=None):
        super().__init__()
        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every
        if batch_every == 0:
            self.batch_every = False
        else:
            self.batch_every = batch_every

    def on_epoch_end(self, *args, **kwargs):
        if self.epoch_every and ((self.epoch_id % self.epoch_every) == 0):
            self.model.eval()
            val_loss, val_acc = score_model(self.model, self.loss_function, self.validation_datagen)
            self.model.train()
            logger.info('epoch {0} validation loss:     {1:.5f}'.format(self.epoch_id, val_loss))
            logger.info('epoch {0} validation accuracy: {1:.5f}'.format(self.epoch_id, val_acc))
        self.epoch_id += 1
        self.batch_id = 0

    def on_batch_end(self, metrics, *args, **kwargs):
        if self.batch_every and ((self.batch_id % self.batch_every) == 0):
            self.model.eval()
            val_loss, val_acc = score_model(self.model, self.loss_function, self.validation_datagen)
            self.model.train()
            logger.info('epoch {0} batch {1} validation loss:     {2:.5f}'.format(self.epoch_id,
                                                                                  self.batch_id,
                                                                                  val_loss))
            logger.info('epoch {0} batch {1} validation accuracy: {2:.5f}'.format(self.epoch_id,
                                                                                  self.batch_id,
                                                                                  val_acc))
        self.batch_id += 1


class PlotBoundingBoxPredictions(Callback):
    def __init__(self, img_dir, bins_nr, epoch_every=1, batch_every=None):
        super().__init__()
        self.img_dir = img_dir
        self.bins_nr = bins_nr
        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every
        if batch_every == 0:
            self.batch_every = False
        else:
            self.batch_every = batch_every
        self._create_dir()

    def on_batch_end(self, *args, **kwargs):
        if self.batch_every and ((self.batch_id % self.batch_every) == 0):
            self._save_images()
        self.batch_id += 1

    def on_epoch_end(self, *args, **kwargs):
        if self.epoch_every and ((self.epoch_id % self.epoch_every) == 0):
            self._save_images()
        self.epoch_id += 1
        self.batch_id = 0

    def _create_dir(self):
        try:
            shutil.rmtree(self.img_dir)
        except Exception:
            pass
        os.makedirs(self.img_dir, exist_ok=True)

    def _save_images(self):
        for i, (image, box_coord, true_box) in enumerate(
                predict_on_batch_multi_output(self.model, self.validation_datagen)):
            image_with_box = overlay_box(image, box_coord, true_box, self.bins_nr)
            filepath = os.path.join(self.img_dir,
                                    'epoch{}_batch{}_idx{}.png'.format(self.epoch_id, self.batch_id, i))
            plt.imsave(filepath, image_with_box)


class PlotKeyPointsPredictions(Callback):
    def __init__(self, img_dir, bins_nr, epoch_every=1, batch_every=None):
        super().__init__()
        self.img_dir = img_dir
        self.bins_nr = bins_nr
        self.batch_every = batch_every
        self.epoch_every = epoch_every

        self._create_dir()

    def on_batch_end(self, *args, **kwargs):
        if self.batch_every and self.batch_id % self.batch_every == 0 and self.batch_id > 0:
            self._save_images()
        self.batch_id += 1

    def on_epoch_end(self, *args, **kwargs):
        if self.epoch_every and self.epoch_id % self.epoch_every == 0 and self.epoch_id > 0:
            self._save_images()
        self.epoch_id += 1
        self.batch_id = 0

    def _create_dir(self):
        try:
            shutil.rmtree(self.img_dir)
        except Exception:
            pass
        os.makedirs(self.img_dir, exist_ok=True)

    def _save_images(self):
        for i, (image, box_coord, true_box) in enumerate(
                predict_on_batch_multi_output(self.model, self.validation_datagen)):
            image_with_box = overlay_box(image, box_coord, true_box, self.bins_nr)
            filepath = os.path.join(self.img_dir,
                                    'epoch{}_batch{}_idx{}.png'.format(self.epoch_id, self.batch_id, i))
            plt.imsave(filepath, image_with_box)


class ExponentialLRScheduler(Callback):
    def __init__(self, gamma, epoch_every=1, batch_every=None):
        super().__init__()
        self.gamma = gamma
        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every
        if batch_every == 0:
            self.batch_every = False
        else:
            self.batch_every = batch_every

    def set_params(self, transformer, validation_datagen):
        self.validation_datagen = validation_datagen
        self.model = transformer.model
        self.optimizer = transformer.optimizer
        self.loss_function = transformer.loss_function
        self.lr_scheduler = ExponentialLR(self.optimizer, self.gamma, last_epoch=-1)

    def on_train_begin(self, *args, **kwargs):
        self.epoch_id = 0
        self.batch_id = 0
        logger.info('initial lr: {0}'.format(self.optimizer.state_dict()['param_groups'][0]['initial_lr']))

    def on_epoch_end(self, *args, **kwargs):
        if self.epoch_every and (((self.epoch_id + 1) % self.epoch_every) == 0):
            self.lr_scheduler.step()
            logger.info('epoch {0} current lr: {1}'.format(self.epoch_id + 1,
                                                           self.optimizer.state_dict()['param_groups'][0]['lr']))
        self.epoch_id += 1
        self.batch_id = 0

    def on_batch_end(self, *args, **kwargs):
        if self.batch_every and ((self.batch_id % self.batch_every) == 0):
            self.lr_scheduler.step()
            logger.info('epoch {0} batch {1} current lr: {2}'.format(
                self.epoch_id + 1, self.batch_id + 1, self.optimizer.state_dict()['param_groups'][0]['lr']))
        self.batch_id += 1


class ModelCheckpoint(Callback):
    def __init__(self, checkpoint_dir, best_only=False, epoch_every=1, batch_every=None):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        if best_only:
            self.best_only = best_only
            raise NotImplementedError
        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every
        if batch_every == 0:
            self.batch_every = False
        else:
            self.batch_every = batch_every

    def on_train_begin(self, *args, **kwargs):
        self.epoch_id = 0
        self.batch_id = 0
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def on_epoch_end(self, *args, **kwargs):
        if self.epoch_every and ((self.epoch_id % self.epoch_every) == 0):
            full_path = os.path.join(self.checkpoint_dir, 'model_epoch{0}.torch'.format(self.epoch_id))
            save_model(self.model, full_path)
            logger.info('epoch {0} model saved to {1}'.format(self.epoch_id, full_path))
        self.epoch_id += 1
        self.batch_id = 0

    def on_batch_end(self, *args, **kwargs):
        if self.batch_every and ((self.batch_id % self.batch_every) == 0):
            full_path = os.path.join(self.checkpoint_dir,
                                     'model_epoch{0}_batch{1}.torch'.format(self.epoch_id, self.batch_id))
            save_model(self.model, full_path)
            logger.info('epoch {0} batch {1} model saved to {2}'.format(self.epoch_id, self.batch_id, full_path))
        self.batch_id += 1


class NeptuneMonitor(Callback):
    def __init__(self):
        super().__init__()
        self.ctx = neptune.Context()
        self.random_name = names.get_first_name()
        self.epoch_loss_averager = Averager()
        self.epoch_acc_averager = Averager()

    def on_train_begin(self, *args, **kwargs):
        self.epoch_loss_averager.reset()
        self.epoch_acc_averager.reset()
        self.epoch_id = 0
        self.batch_id = 0

    def on_batch_end(self, metrics, *args, **kwargs):
        batch_loss = metrics['batch_loss']
        batch_acc = metrics['batch_acc']

        self.epoch_loss_averager.send(batch_loss)
        self.epoch_acc_averager.send(batch_acc)

        logs = {'epoch_id': self.epoch_id, 'batch_id': self.batch_id, 'batch_loss': batch_loss,
                'batch_acc': batch_acc}

        self.ctx.channel_send('batch_loss {}'.format(self.random_name), x=logs['batch_id'], y=logs['batch_loss'])
        self.ctx.channel_send('batch_acc {}'.format(self.random_name), x=logs['batch_id'], y=logs['batch_acc'])

        self.batch_id += 1

    def on_epoch_end(self, *args, **kwargs):
        epoch_avg_loss = self.epoch_loss_averager.value
        epoch_avg_acc = self.epoch_acc_averager.value
        self.epoch_loss_averager.reset()
        self.epoch_acc_averager.reset()

        self.model.eval()
        val_loss, val_acc = score_model(self.model, self.loss_function, self.validation_datagen)
        self.model.train()

        logs = {'epoch_id': self.epoch_id, 'batch_id': self.batch_id,
                'epoch_loss': epoch_avg_loss,
                'epoch_acc': epoch_avg_acc, 'epoch_val_loss': val_loss, 'epoch_val_acc': val_acc}

        self._send_numeric_channels(logs)
        self.epoch_id += 1

    def _send_numeric_channels(self, logs):
        self.ctx.channel_send('epoch_loss {}'.format(self.random_name), x=logs['epoch_id'], y=logs['epoch_loss'])
        self.ctx.channel_send('epoch_acc {}'.format(self.random_name), x=logs['epoch_id'], y=logs['epoch_acc'])
        self.ctx.channel_send('epoch_val_loss {}'.format(self.random_name), x=logs['epoch_id'],
                              y=logs['epoch_val_loss'])
        self.ctx.channel_send('epoch_val_acc {}'.format(self.random_name), x=logs['epoch_id'], y=logs['epoch_val_acc'])


class NeptuneMonitorLocalizer(NeptuneMonitor):
    def __init__(self, bins_nr, img_nr):
        super().__init__()
        self.bins_nr = bins_nr
        self.img_nr = img_nr

    def on_epoch_end(self, *args, **kwargs):
        epoch_avg_loss = self.epoch_loss_averager.value
        epoch_avg_acc = self.epoch_acc_averager.value
        self.epoch_loss_averager.reset()
        self.epoch_acc_averager.reset()

        self.model.eval()
        val_loss, val_acc = score_model(self.model, self.loss_function, self.validation_datagen)
        self.model.train()

        logs = {'epoch_id': self.epoch_id, 'batch_id': self.batch_id,
                'epoch_loss': epoch_avg_loss,
                'epoch_acc': epoch_avg_acc, 'epoch_val_loss': val_loss, 'epoch_val_acc': val_acc}

        self._send_numeric_channels(logs)

        for i, (image, y_pred, y_true) in enumerate(
                predict_on_batch_multi_output(self.model, self.validation_datagen)):
            image_with_box = overlay_box(image, y_pred, y_true, self.bins_nr)
            pill_image = Image.fromarray((image_with_box * 255.).astype(np.uint8))
            self.ctx.channel_send("plotted bbox", neptune.Image(
                name='epoch{}_batch{}_idx{}'.format(self.epoch_id, self.batch_id, i),
                description="true and prediction bbox",
                data=pill_image))

            if i == self.img_nr:
                break
        self.epoch_id += 1


class NeptuneMonitorKeypoints(NeptuneMonitor):
    def __init__(self, bins_nr, img_nr):
        super().__init__()
        self.bins_nr = bins_nr
        self.img_nr = img_nr

    def on_epoch_end(self, *args, **kwargs):
        epoch_avg_loss = self.epoch_loss_averager.value
        epoch_avg_acc = self.epoch_acc_averager.value
        self.epoch_loss_averager.reset()
        self.epoch_acc_averager.reset()

        self.model.eval()
        val_loss, val_acc = score_model(self.model, self.loss_function, self.validation_datagen)
        self.model.train()

        logs = {'epoch_id': self.epoch_id, 'batch_id': self.batch_id,
                'epoch_loss': epoch_avg_loss,
                'epoch_acc': epoch_avg_acc, 'epoch_val_loss': val_loss, 'epoch_val_acc': val_acc}

        self._send_numeric_channels(logs)

        for i, (image, y_pred, y_true) in enumerate(
                predict_on_batch_multi_output(self.model, self.validation_datagen)):
            image_with_keypoints = overlay_keypoints(image, y_pred, y_true, self.bins_nr)
            pill_image = Image.fromarray((image_with_keypoints * 255.).astype(np.uint8))
            self.ctx.channel_send("plotted key points", neptune.Image(
                name='epoch{}_batch{}_idx{}'.format(self.epoch_id, self.batch_id, i),
                description="true and prediction key points",
                data=pill_image))

            if i == self.img_nr:
                break
        self.epoch_id += 1


class ExperimentTiming(Callback):
    def __init__(self):
        super().__init__()
        self.batch_start = None
        self.epoch_start = None
        self.current_sum = None
        self.current_mean = None

    def on_train_begin(self, *args, **kwargs):
        self.epoch_id = 0
        self.batch_id = 0
        logger.info('starting training...')

    def on_train_end(self, *args, **kwargs):
        logger.info('training finished...')

    def on_epoch_begin(self, *args, **kwargs):
        if self.epoch_id > 0:
            epoch_time = datetime.now() - self.epoch_start
            logger.info('epoch {0} time {1}'.format(self.epoch_id - 1, str(epoch_time)[:-7]))
        self.epoch_start = datetime.now()
        self.current_sum = timedelta()
        self.current_mean = timedelta()
        logger.info('epoch {0} ...'.format(self.epoch_id))

    def on_batch_begin(self, *args, **kwargs):
        if self.batch_id > 0:
            current_delta = datetime.now() - self.batch_start
            self.current_sum += current_delta
            self.current_mean = self.current_sum / self.batch_id
        if self.batch_id > 0 and (((self.batch_id - 1) % 10) == 0):
            logger.info('epoch {0} average batch time: {1}'.format(self.epoch_id, str(self.current_mean)[:-5]))
        if self.batch_id == 0 or self.batch_id % 10 == 0:
            logger.info('epoch {0} batch {1} ...'.format(self.epoch_id, self.batch_id))
        self.batch_start = datetime.now()


class CallbackReduceLROnPlateau(Callback):  # thank you keras
    def __init__(self):
        super().__init__()
        pass


class CallbackEarlyStopping(Callback):  # thank you keras
    def __init__(self):
        super().__init__()
        pass

import os
from datetime import datetime, timedelta

from PIL import Image
import numpy as np
from deepsense import neptune
from torch.optim.lr_scheduler import ExponentialLR

from .validation import score_model, get_prediction_masks
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

    def training_break(self, *args, **kwargs):
        return False

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

    def training_break(self, *args, **kwargs):
        callback_out = [callback.training_break(*args, **kwargs) for callback in self.callbacks]
        return any(callback_out)

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
        self.epoch_id = 0
        self.batch_id = 0

    def on_epoch_end(self, *args, **kwargs):
        epoch_avg_loss = self.epoch_loss_averager.value
        self.epoch_loss_averager.reset()
        if self.epoch_every and ((self.epoch_id % self.epoch_every) == 0):
            logger.info('epoch {0} loss:     {1:.5f}'.format(self.epoch_id, epoch_avg_loss))
        self.epoch_id += 1
        self.batch_id = 0

    def on_batch_end(self, metrics, *args, **kwargs):
        batch_loss = metrics['batch_loss']
        self.epoch_loss_averager.send(batch_loss)
        if self.batch_every and ((self.batch_id % self.batch_every) == 0):
            logger.info('epoch {0} batch {1} loss:     {2:.5f}'.format(self.epoch_id, self.batch_id, batch_loss))
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
            val_loss = score_model(self.model, self.loss_function, self.validation_datagen)
            self.model.train()
            logger.info('epoch {0} validation loss:     {1:.5f}'.format(self.epoch_id, val_loss))
        self.epoch_id += 1
        self.batch_id = 0

    def on_batch_end(self, metrics, *args, **kwargs):
        if self.batch_every and ((self.batch_id % self.batch_every) == 0):
            self.model.eval()
            val_loss = score_model(self.model, self.loss_function, self.validation_datagen)
            self.model.train()
            logger.info('epoch {0} batch {1} validation loss:     {2:.5f}'.format(self.epoch_id,
                                                                                  self.batch_id,
                                                                                  val_loss))
        self.batch_id += 1


class EarlyStopping(Callback):
    def __init__(self, patience, minimize=True):
        super().__init__()
        self.patience = patience
        self.minimize = minimize
        self.best_score = None
        self.epoch_since_best = 0

    def training_break(self, *args, **kwargs):
        self.model.eval()
        val_loss = score_model(self.model, self.loss_function, self.validation_datagen)
        self.model.train()

        if not self.best_score:
            self.best_score = val_loss

        if (self.minimize and val_loss < self.best_score) or (not self.minimize and val_loss > self.best_score):
            self.best_score = val_loss
            self.epoch_since_best = 0
        else:
            self.epoch_since_best += 1

        if self.epoch_since_best > self.patience:
            return True
        else:
            return False


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
    def __init__(self, filepath, epoch_every=1, minimize=True):
        super().__init__()
        self.filepath = filepath
        self.minimize = minimize
        self.best_score = None

        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every

    def on_train_begin(self, *args, **kwargs):
        self.epoch_id = 0
        self.batch_id = 0
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def on_epoch_end(self, *args, **kwargs):
        if self.epoch_every and ((self.epoch_id % self.epoch_every) == 0):
            self.model.eval()
            val_loss = score_model(self.model, self.loss_function, self.validation_datagen)
            self.model.train()

            if not self.best_score:
                self.best_score = val_loss

            if (self.minimize and val_loss < self.best_score) or (not self.minimize and val_loss > self.best_score):
                self.best_score = val_loss
                save_model(self.model, self.filepath)
                logger.info('epoch {0} model saved to {1}'.format(self.epoch_id, self.filepath))

        self.epoch_id += 1
        self.batch_id = 0


class NeptuneMonitor(Callback):
    def __init__(self):
        super().__init__()
        self.ctx = neptune.Context()
        self.random_name = ''
        self.epoch_loss_averager = Averager()

    def on_train_begin(self, *args, **kwargs):
        self.epoch_loss_averager.reset()
        self.epoch_id = 0
        self.batch_id = 0

    def on_batch_end(self, metrics, *args, **kwargs):
        batch_loss = metrics['batch_loss']

        self.epoch_loss_averager.send(batch_loss)

        logs = {'epoch_id': self.epoch_id, 'batch_id': self.batch_id, 'batch_loss': batch_loss}

        self.ctx.channel_send('batch_loss {}'.format(self.random_name), x=logs['batch_id'], y=logs['batch_loss'])

        self.batch_id += 1

    def on_epoch_end(self, *args, **kwargs):
        epoch_avg_loss = self.epoch_loss_averager.value
        self.epoch_loss_averager.reset()

        self.model.eval()
        val_loss = score_model(self.model, self.loss_function, self.validation_datagen)
        self.model.train()

        logs = {'epoch_id': self.epoch_id, 'batch_id': self.batch_id, 'epoch_loss': epoch_avg_loss,
                'epoch_val_loss': val_loss}
        self._send_numeric_channels(logs)
        self.epoch_id += 1

    def _send_numeric_channels(self, logs):
        self.ctx.channel_send('epoch_loss {}'.format(self.random_name), x=logs['epoch_id'], y=logs['epoch_loss'])
        self.ctx.channel_send('epoch_val_loss {}'.format(self.random_name), x=logs['epoch_id'],
                              y=logs['epoch_val_loss'])


class NeptuneMonitorSegmentation(NeptuneMonitor):
    def __init__(self, image_nr, image_resize):
        super().__init__()
        self.image_nr = image_nr
        self.image_resize = image_resize

    def on_epoch_end(self, *args, **kwargs):
        epoch_avg_loss = self.epoch_loss_averager.value
        self.epoch_loss_averager.reset()

        self.model.eval()
        val_loss = score_model(self.model, self.loss_function, self.validation_datagen)
        pred_masks = get_prediction_masks(self.model, self.validation_datagen)
        self.model.train()

        logs = {'epoch_id': self.epoch_id, 'batch_id': self.batch_id, 'epoch_loss': epoch_avg_loss,
                'epoch_val_loss': val_loss}
        self._send_numeric_channels(logs)
        self._send_image_channels(pred_masks)
        self.epoch_id += 1

    def _send_image_channels(self, pred_masks):
        for i, image_triplet in enumerate(pred_masks):
            h, w = image_triplet.shape[1:]
            image_glued = np.zeros((h, 3 * w + 20))

            image_glued[:, :w] = image_triplet[0, :, :]
            image_glued[:, w + 10:2 * w + 10] = image_triplet[1, :, :]
            image_glued[:, 2 * w + 20:] = image_triplet[2, :, :]

            pill_image = Image.fromarray((image_glued * 255.).astype(np.uint8))
            h_, w_ = image_glued.shape
            pill_image = pill_image.resize((int(self.image_resize * w_), int(self.image_resize * h_)), Image.ANTIALIAS)

            self.ctx.channel_send("masks", neptune.Image(
                name='epoch{}_batch{}_idx{}'.format(self.epoch_id, self.batch_id, i),
                description="true and prediction masks",
                data=pill_image))

            if i == self.image_nr: break


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

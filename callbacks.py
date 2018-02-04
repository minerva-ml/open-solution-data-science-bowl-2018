from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from deepsense import neptune

from steps.pytorch.callbacks import NeptuneMonitor
from utils import sigmoid


class NeptuneMonitorSegmentation(NeptuneMonitor):
    def __init__(self, image_nr, image_resize):
        super().__init__()
        self.image_nr = image_nr
        self.image_resize = image_resize

    def on_epoch_end(self, *args, **kwargs):
        self._send_numeric_channels()
        self._send_image_channels()
        self.epoch_id += 1

    def _send_image_channels(self):
        self.model.eval()
        pred_masks = self.get_prediction_masks()
        self.model.train()

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

    def get_prediction_masks(self):
        batch_gen, steps = self.validation_datagen
        for batch_id, data in enumerate(batch_gen):
            X, targets = data

            if torch.cuda.is_available():
                X, targets_var = Variable(X).cuda(), Variable(targets).cuda()
            else:
                X, targets_var = Variable(X), Variable(targets)
            outputs = self.model(X)

            raw_images = np.mean(X.data.cpu().numpy(), axis=1)
            prediction_masks = sigmoid(np.squeeze(outputs.data.cpu().numpy(), axis=1))
            ground_truth_masks = np.squeeze(targets.cpu().numpy(), axis=1)
            break
        return np.stack([raw_images, prediction_masks, ground_truth_masks], axis=1)


class NeptuneMonitorSegmentationMultitask(NeptuneMonitor):
    def __init__(self, image_nr, image_resize):
        super().__init__()
        self.image_nr = image_nr
        self.image_resize = image_resize

    def on_epoch_end(self, *args, **kwargs):
        self._send_numeric_channels()
        self._send_image_channels()
        self.epoch_id += 1

    def _send_image_channels(self):
        pass



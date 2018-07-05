import glob
import logging
import os
import sys
from itertools import product, chain
from collections import Iterable

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from attrdict import AttrDict
from tqdm import tqdm
from pycocotools import mask as cocomask

from .steppy.base import BaseTransformer


def read_yaml(filepath):
    with open(filepath) as f:
        config = yaml.load(f)
    return AttrDict(config)


def init_logger():
    logger = logging.getLogger('dsb-2018')
    logger.setLevel(logging.INFO)
    message_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s',
                                       datefmt='%Y-%m-%d %H-%M-%S')

    # console handler for validation info
    ch_va = logging.StreamHandler(sys.stdout)
    ch_va.setLevel(logging.INFO)

    ch_va.setFormatter(fmt=message_format)

    # add the handlers to the logger
    logger.addHandler(ch_va)

    return logger


def get_logger():
    return logging.getLogger('dsb-2018')


def decompose(labeled):
    nr_true = labeled.max()
    masks = []
    for i in range(1, nr_true + 1):
        msk = labeled.copy()
        msk[msk != i] = 0.
        msk[msk == i] = 255.
        masks.append(msk)

    if not masks:
        return [labeled]
    else:
        return masks


def create_submission(experiments_dir, meta, predictions, logger):
    image_ids, encodings = [], []
    output = []
    for image_id, prediction in zip(meta['ImageId'].values, predictions):
        for mask in decompose(prediction):
            rle_encoded = ' '.join(str(rle) for rle in run_length_encoding(mask > 128.))
            if len(rle_encoded) != 0:
                image_ids.append(image_id)
                encodings.append(rle_encoded)
                output.append([image_id, rle_encoded])
            else:
                logger.info('*** image_id {}'.format(image_id))
                logger.info('*** rle_encoded {} is empty'.format(rle_encoded))

    submission = pd.DataFrame(output, columns=['ImageId', 'EncodedPixels']).astype(str)
    submission = submission[submission['EncodedPixels'] != 'nan']
    submission_filepath = os.path.join(experiments_dir, 'submission.csv')
    submission.to_csv(submission_filepath, index=None, encoding='utf-8')
    logger.info('submission saved to {}'.format(submission_filepath))
    logger.info('submission head \n\n{}'.format(submission.head()))


def read_masks(masks_filepaths):
    masks = []
    for mask_dir in tqdm(masks_filepaths):
        mask = []
        if len(mask_dir) == 1:
            mask_dir = mask_dir[0]
        for i, mask_filepath in enumerate(glob.glob('{}/*'.format(mask_dir))):
            blob = np.asarray(Image.open(mask_filepath))
            blob_binarized = (blob > 128.).astype(np.uint8) * (i + 1)
            mask.append(blob_binarized)
        mask = np.sum(np.stack(mask, axis=0), axis=0).astype(np.uint8)
        masks.append(mask)
    return masks


def read_masks_from_csv(image_ids, solution_file_path):
    solution = pd.read_csv(solution_file_path)
    masks = []
    for image_id in image_ids:
        mask_shape = (solution[solution['ImageId'] == image_id]['Height'].iloc[0],
                      solution[solution['ImageId'] == image_id]['Width'].iloc[0])
        mask = np.zeros(mask_shape, dtype=np.uint8)
        for i, rle in enumerate(solution[solution['ImageId'] == image_id]['EncodedPixels']):
            mask += (i + 1) * run_length_decoding(rle, mask_shape)
        masks.append(mask)
    return masks


def run_length_encoding(x):
    # https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
    bs = np.where(x.T.flatten())[0]

    rle = []
    prev = -2
    for b in bs:
        if (b > prev + 1): rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b

    if len(rle) != 0 and rle[-1] + rle[-2] == x.size:
        rle[-2] = rle[-2] - 1

    return rle


def run_length_decoding(mask_rle, shape):
    """
    Based on https://www.kaggle.com/msl23518/visualize-the-stage1-test-solution and modified
    Args:
        mask_rle: run-length as string formatted (start length)
        shape: (height, width) of array to return

    Returns:
        numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[1] * shape[0], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((shape[1], shape[0])).T


def read_params(ctx):
    if ctx.params.__class__.__name__ == 'OfflineContextParams':
        neptune_config = read_yaml('neptune.yaml')
        params = neptune_config.parameters
    else:
        params = ctx.params
    return params


def generate_metadata(data_dir,
                      masks_overlayed_dir,
                      cut_masks_dir,
                      masks_with_borders_dir,
                      generate_test=True,
                      ):
    def stage1_generate_metadata(train):
        df_metadata = pd.DataFrame(columns=['ImageId', 'file_path_image', 'file_path_masks', 'file_path_mask',
                                            'file_path_mask_with_borders', 'file_path_cut_mask',
                                            'is_train', 'width', 'height', 'n_nuclei'])
        if train:
            tr_te = 'stage1_train'
        else:
            tr_te = 'stage1_test'

        for image_id in tqdm(sorted(os.listdir(os.path.join(data_dir, tr_te)))):
            p = os.path.join(data_dir, tr_te, image_id, 'images')
            if image_id != os.listdir(p)[0][:-4]:
                ValueError('ImageId mismatch ' + str(image_id))
            if len(os.listdir(p)) != 1:
                ValueError('more than one image in dir')

            file_path_image = os.path.join(p, os.listdir(p)[0])
            if train:
                is_train = 1
                file_path_masks = os.path.join(data_dir, tr_te, image_id, 'masks')
                file_path_mask = os.path.join(masks_overlayed_dir, tr_te, image_id + '.png')
                file_path_cut_mask = os.path.join(cut_masks_dir, tr_te, image_id + '.png')
                file_path_mask_with_borders = os.path.join(masks_with_borders_dir, tr_te, image_id + '.png')
                n_nuclei = len(os.listdir(file_path_masks))
            else:
                is_train = 0
                file_path_masks = None
                file_path_mask = None
                file_path_cut_mask = None
                file_path_mask_with_borders = None
                n_nuclei = None

            img = Image.open(file_path_image)
            width = img.size[0]
            height = img.size[1]
            s = df_metadata['ImageId']
            if image_id is s:
                ValueError('ImageId conflict ' + str(image_id))
            df_metadata = df_metadata.append({'ImageId': image_id,
                                              'file_path_image': file_path_image,
                                              'file_path_masks': file_path_masks,
                                              'file_path_mask': file_path_mask,
                                              'file_path_cut_mask': file_path_cut_mask,
                                              'file_path_mask_with_borders': file_path_mask_with_borders,
                                              'is_train': is_train,
                                              'width': width,
                                              'height': height,
                                              'n_nuclei': n_nuclei}, ignore_index=True)
        return df_metadata

    train_metadata = stage1_generate_metadata(train=True)
    if generate_test:
        test_metadata = stage1_generate_metadata(train=False)
        metadata = train_metadata.append(test_metadata, ignore_index=True)
    else:
        metadata = train_metadata
    return metadata


def squeeze_inputs_if_needed(inputs):
    if isinstance(inputs[0], np.ndarray):
        return np.squeeze(inputs[0], axis=1)
    else:
        return inputs[0]


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def softmax(X, theta=1.0, axis=None):
    """
    https://nolanbconaway.github.io/blog/2017/softmax-numpy
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def relabel(img):
    h, w = img.shape

    relabel_dict = {}

    for i, k in enumerate(np.unique(img)):
        if k == 0:
            relabel_dict[k] = 0
        else:
            relabel_dict[k] = i
    for i, j in product(range(h), range(w)):
        img[i, j] = relabel_dict[img[i, j]]
    return img


def relabel_random_colors(img, max_colours=1000):
    keys = list(range(1, max_colours, 1))
    np.random.shuffle(keys)
    values = list(range(1, max_colours, 1))
    np.random.shuffle(values)
    funky_dict = {k: v for k, v in zip(keys, values)}
    funky_dict[0] = 0

    h, w = img.shape

    for i, j in product(range(h), range(w)):
        img[i, j] = funky_dict[img[i, j]]
    return img


def from_pil(*images):
    images = [np.array(image) for image in images]
    if len(images) == 1:
        return images[0]
    else:
        return images


def to_pil(*images):
    images = [Image.fromarray((image).astype(np.uint8)) for image in images]
    if len(images) == 1:
        return images[0]
    else:
        return images


def make_apply_transformer(func, output_name='output', apply_on=None):
    class StaticApplyTransformer(BaseTransformer):
        def transform(self, *args, **kwargs):
            self.check_input(*args, **kwargs)

            if not apply_on:
                iterator = zip(*args, *kwargs.values())
            else:
                iterator = zip(*args, *[kwargs[key] for key in apply_on])

            output = []
            for func_args in tqdm(iterator, total=self.get_arg_length(*args, **kwargs)):
                output.append(func(*func_args))
            return {output_name: output}

        @staticmethod
        def check_input(*args, **kwargs):
            if len(args) and len(kwargs) == 0:
                raise Exception('Input must not be empty')

            arg_length = None
            for arg in chain(args, kwargs.values()):
                if not isinstance(arg, Iterable):
                    raise Exception('All inputs must be iterable')
                arg_length_loc = None
                try:
                    arg_length_loc = len(arg)
                except:
                    pass
                if arg_length_loc is not None:
                    if arg_length is None:
                        arg_length = arg_length_loc
                    elif arg_length_loc != arg_length:
                        raise Exception('All inputs must be the same length')

        @staticmethod
        def get_arg_length(*args, **kwargs):
            arg_length = None
            for arg in chain(args, kwargs.values()):
                if arg_length is None:
                    try:
                        arg_length = len(arg)
                    except:
                        pass
                if arg_length is not None:
                    return arg_length

    return StaticApplyTransformer()


def rle_from_binary(prediction):
    prediction = np.asfortranarray(prediction)
    return cocomask.encode(prediction)


def get_segmentations(labeled):
    nr_true = labeled.max()
    segmentations = []
    for i in range(1, nr_true + 1):
        msk = labeled == i
        segmentation = rle_from_binary(msk.astype('uint8'))
        segmentation['counts'] = segmentation['counts'].decode("UTF-8")
        segmentations.append(segmentation)
    return segmentations


def get_crop_pad_sequence(vertical, horizontal):
    top = int(vertical / 2)
    bottom = vertical - top
    right = int(horizontal / 2)
    left = horizontal - right
    return (top, right, bottom, left)


def get_list_of_image_predictions(batch_predictions):
    image_predictions = []
    for batch_pred in batch_predictions:
        image_predictions.extend(list(batch_pred))
    return image_predictions

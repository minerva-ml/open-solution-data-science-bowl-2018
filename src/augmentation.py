import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

from .utils import get_crop_pad_sequence


def _perspective_transform_augment_images(self, images, random_state, parents, hooks):
    result = images
    if not self.keep_size:
        result = list(result)

    matrices, max_heights, max_widths = self._create_matrices(
        [image.shape for image in images],
        random_state
    )

    for i, (M, max_height, max_width) in enumerate(zip(matrices, max_heights, max_widths)):
        warped = cv2.warpPerspective(images[i], M, (max_width, max_height))
        if warped.ndim == 2 and images[i].ndim == 3:
            warped = np.expand_dims(warped, 2)
        if self.keep_size:
            h, w = images[i].shape[0:2]
            warped = ia.imresize_single_image(warped, (h, w))

        result[i] = warped

    return result


iaa.PerspectiveTransform._augment_images = _perspective_transform_augment_images

affine_seq = iaa.Sequential([
    # General
    iaa.SomeOf((1, 2),
               [iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Affine(rotate=(0, 360),
                           translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, mode='symmetric'),
                iaa.CropAndPad(percent=(-0.25, 0.25), pad_mode='symmetric')
                ]),
    # Deformations
    iaa.Sometimes(0.3, iaa.PiecewiseAffine(scale=(0.02, 0.04))),
    iaa.Sometimes(0.3, iaa.PerspectiveTransform(scale=(0.05, 0.10))),
], random_order=True)

color_seq = iaa.Sequential([
    # Color
    iaa.Sometimes(0.3, iaa.ContrastNormalization((0.3, 1.0))),
    iaa.Sometimes(0.3, iaa.ElasticTransformation(alpha=(1, 5), sigma=0.1)),
    iaa.OneOf([
        iaa.Noop(),
        iaa.Sequential([
            iaa.OneOf([
                iaa.OneOf([
                    # Add in HSV or RGB
                    iaa.Sequential([
                        iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                        iaa.WithChannels(0, iaa.Add((0, 100))),
                        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
                    iaa.Sequential([
                        iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                        iaa.WithChannels(1, iaa.Add((0, 100))),
                        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
                    iaa.Sequential([
                        iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                        iaa.WithChannels(2, iaa.Add((0, 100))),
                        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
                    iaa.WithChannels(0, iaa.Add((0, 100))),
                    iaa.WithChannels(1, iaa.Add((0, 100))),
                    iaa.WithChannels(2, iaa.Add((0, 100)))
                ]),
                iaa.OneOf([
                    # Add elementwise in HSV or RGB
                    iaa.Sequential([
                        iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                        iaa.WithChannels(0, iaa.AddElementwise((0, 30))),
                        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
                    iaa.Sequential([
                        iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                        iaa.WithChannels(1, iaa.AddElementwise((0, 30))),
                        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
                    iaa.Sequential([
                        iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                        iaa.WithChannels(2, iaa.AddElementwise((0, 30))),
                        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
                    iaa.WithChannels(0, iaa.AddElementwise((0, 30))),
                    iaa.WithChannels(1, iaa.AddElementwise((0, 30))),
                    iaa.WithChannels(2, iaa.AddElementwise((0, 30)))
                ]),
                iaa.OneOf([
                    # Multiply in HSV or RGB
                    iaa.Sequential([
                        iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                        iaa.WithChannels(0, iaa.Multiply((0, 2))),
                        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
                    iaa.Sequential([
                        iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                        iaa.WithChannels(1, iaa.Multiply((0, 2))),
                        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
                    iaa.Sequential([
                        iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                        iaa.WithChannels(2, iaa.Multiply((0, 2))),
                        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
                    iaa.WithChannels(0, iaa.Multiply((0, 2))),
                    iaa.WithChannels(1, iaa.Multiply((0, 2))),
                    iaa.WithChannels(2, iaa.Multiply((0, 2)))
                ]),
                iaa.OneOf([
                    # Multiply elementwise in HSV or RGB
                    iaa.Sequential([
                        iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                        iaa.WithChannels(0, iaa.MultiplyElementwise((0, 2))),
                        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
                    iaa.Sequential([
                        iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                        iaa.WithChannels(1, iaa.MultiplyElementwise((0, 2))),
                        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
                    iaa.Sequential([
                        iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                        iaa.WithChannels(2, iaa.MultiplyElementwise((0, 2))),
                        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
                    iaa.WithChannels(0, iaa.MultiplyElementwise((0, 2))),
                    iaa.WithChannels(1, iaa.MultiplyElementwise((0, 2))),
                    iaa.WithChannels(2, iaa.MultiplyElementwise((0, 2)))
                ]),
            ]),
            iaa.OneOf([
                iaa.Noop(),
                # iaa.CoarseSaltAndPepper(p=(0, 0.1), size_px=(64, 1024), per_channel=True),
                # iaa.CoarseSaltAndPepper(p=(0, 0.1), size_px=(64, 1024), per_channel=False)
            ])
        ]),
        iaa.OneOf([
            iaa.GaussianBlur(sigma=(0.4, 8.0)),
            iaa.AverageBlur(k=(2, 21)),
            iaa.MedianBlur(k=(3, 15))
        ]),
        iaa.OneOf([
            iaa.Sharpen(alpha=0.5),
            iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))
        ])
    ])
], random_order=False)

color_seq_grey = iaa.Sequential([
    # Color
    iaa.Invert(0.3),
    iaa.Sometimes(0.3, iaa.ContrastNormalization((0.5, 1.5))),
    iaa.Sometimes(0.3, iaa.ElasticTransformation(alpha=(1, 5), sigma=0.1)),
    iaa.OneOf([
        iaa.Noop(),
        iaa.Sequential([
            iaa.OneOf([
                iaa.Add((0, 100)),
                iaa.AddElementwise((0, 100)),
                iaa.Multiply((0, 100)),
                iaa.MultiplyElementwise((0, 100)),
            ]),
            iaa.OneOf([
                iaa.Noop(),
                iaa.CoarseSaltAndPepper(p=(0, 0.1), size_px=(64, 1024), per_channel=False)
            ])
        ]),
        iaa.OneOf([
            iaa.GaussianBlur(sigma=(0.0, 8.0)),
            iaa.AverageBlur(k=(2, 21)),
            iaa.MedianBlur(k=(3, 15))
        ])
    ])
], random_order=False)


def crop_seq(crop_size):
    seq = iaa.Sequential([affine_seq,
                          RandomCropFixedSize(px=crop_size)], random_order=False)
    return seq


def padding_seq(pad_size, pad_method):
    seq = iaa.Sequential([PadFixed(pad=pad_size, pad_method=pad_method),
                          ]).to_deterministic()
    return seq


def pad_to_fit_net(divisor, pad_mode, rest_of_augs=iaa.Noop()):
    return iaa.Sequential(InferencePad(divisor, pad_mode), rest_of_augs)


class PadFixed(iaa.Augmenter):
    PAD_FUNCTION = {'reflect': cv2.BORDER_REFLECT_101,
                    'replicate': cv2.BORDER_REPLICATE,
                    }

    def __init__(self, pad=None, pad_method=None, name=None, deterministic=False, random_state=None):
        super().__init__(name, deterministic, random_state)
        self.pad = pad
        self.pad_method = pad_method

    def _augment_images(self, images, random_state, parents, hooks):
        result = []
        for i, image in enumerate(images):
            image_pad = self._pad(image)
            result.append(image_pad)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        return result

    def _pad(self, img):
        img_ = img.copy()

        if self._is_expanded_grey_format(img):
            img_ = np.squeeze(img_, axis=-1)

        h_pad, w_pad = self.pad
        img_ = cv2.copyMakeBorder(img_.copy(), h_pad, h_pad, w_pad, w_pad, PadFixed.PAD_FUNCTION[self.pad_method])

        if self._is_expanded_grey_format(img):
            img_ = np.expand_dims(img_, axis=-1)

        return img_

    def get_parameters(self):
        return []

    def _is_expanded_grey_format(self, img):
        if len(img.shape) == 3 and img.shape[2] == 1:
            return True
        else:
            return False


class RandomCropFixedSize(iaa.Augmenter):
    def __init__(self, px=None, name=None, deterministic=False, random_state=None):
        super(RandomCropFixedSize, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.px = px
        if isinstance(self.px, tuple):
            self.px_h, self.px_w = self.px
        elif isinstance(self.px, int):
            self.px_h = self.px
            self.px_w = self.px
        else:
            raise NotImplementedError

    def _augment_images(self, images, random_state, parents, hooks):

        result = []
        seeds = random_state.randint(0, 10 ** 6, (len(images),))
        for i, image in enumerate(images):
            seed = seeds[i]
            image_cr = self._random_crop(seed, image)
            result.append(image_cr)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        return result

    def _random_crop(self, seed, image):
        height, width = image.shape[:2]

        np.random.seed(seed)
        if height > self.px_h:
            crop_top = np.random.randint(height - self.px_h)
        elif height == self.px_h:
            crop_top = 0
        else:
            raise ValueError("To big crop height")
        crop_bottom = crop_top + self.px_h

        np.random.seed(seed + 1)
        if width > self.px_w:
            crop_left = np.random.randint(width - self.px_w)
        elif width == self.px_w:
            crop_left = 0
        else:
            raise ValueError("To big crop width")
        crop_right = crop_left + self.px_w

        if len(image.shape) == 2:
            image_cropped = image[crop_top:crop_bottom, crop_left:crop_right]
        else:
            image_cropped = image[crop_top:crop_bottom, crop_left:crop_right, :]
        return image_cropped

    def get_parameters(self):
        return []


class InferencePad(iaa.Augmenter):
    def __init__(self, divisor=2, pad_mode='symmetric', name=None, deterministic=False, random_state=None):
        super(InferencePad, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.divisor = divisor
        self.pad_mode = pad_mode

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def _augment_images(self, images, random_state, parents, hooks):

        result = []
        for i, image in enumerate(images):
            image_padded = self._pad_image(image)
            result.append(image_padded)
        return result

    def _pad_image(self, image):
        height = image.shape[0]
        width = image.shape[1]

        pad_sequence = self._get_pad_sequence(height, width)
        augmenter = iaa.Pad(px=pad_sequence, keep_size=False, pad_mode=self.pad_mode)
        return augmenter.augment_image(image)

    def _get_pad_sequence(self, height, width):
        pad_vertical = self._get_pad(height)
        pad_horizontal = self._get_pad(width)
        return get_crop_pad_sequence(pad_vertical, pad_horizontal)

    def _get_pad(self, dim):
        if dim % self.divisor == 0:
            return 0
        else:
            return self.divisor - dim % self.divisor

    def get_parameters(self):
        return [self.divisor, self.pad_mode]

import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

affine_seq = iaa.Sequential([
    # General
    iaa.SomeOf((1, 2),
               [iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Affine(rotate=(0, 360),
                           translate_percent=(-0.1, 0.1)),
                iaa.CropAndPad(percent=(-0.25, 0.25), pad_cval=0)
                ]),
    # Deformations
    iaa.PiecewiseAffine(scale=(0.00, 0.06))
], random_order=True)

color_seq = iaa.Sequential([
    # Color
    iaa.OneOf([
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
    ])
], random_order=True)


def patching_seq(crop_size):
    h, w = crop_size
    seq = iaa.Sequential([
        iaa.Affine(rotate=(0, 360)),
        CropFixed(px=h)
    ], random_order=False)
    return seq


class CropFixed(iaa.Augmenter):
    def __init__(self, px=None, name=None, deterministic=False, random_state=None):
        super(CropFixed, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.px = px

    def _augment_images(self, images, random_state, parents, hooks):

        result = []
        seeds = random_state.randint(0, 10 ** 6, (len(images),))
        for i, image in enumerate(images):
            seed = seeds[i]
            height, width = image.shape[:2]
            crop_top, crop_right, crop_bottom, crop_left = self._draw_sample_crops(seed, height, width)

            if len(image.shape) == 2:
                image_cr = image[crop_top:crop_bottom, crop_left:crop_right]
            else:
                image_cr = image[crop_top:crop_bottom, crop_left:crop_right, :]

            result.append(image_cr)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        return result

    def _draw_sample_crops(self, seed, height, width):
        np.random.seed(seed)
        crop_top = np.random.randint(height - self.px)
        crop_bottom = crop_top + self.px

        np.random.seed(seed + 1)
        crop_left = np.random.randint(width - self.px)
        crop_right = crop_left + self.px

        return crop_top, crop_right, crop_bottom, crop_left

    def get_parameters(self):
        return []

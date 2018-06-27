import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa


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
                           translate_percent={"x": (-0.1, 0.1), "y":(-0.1, 0.1)}),
                iaa.CropAndPad(percent=(-0.25, 0.25), pad_mode='reflect')
                ]),
    # Deformations
    iaa.Sometimes(0.3, iaa.PiecewiseAffine(scale=(0.02, 0.04))),
    iaa.Sometimes(0.3, iaa.PerspectiveTransform(scale=(0.05, 0.10))),
], random_order=True)

color_seq = iaa.Sequential([
    # Color
    iaa.Sometimes(0.3, iaa.ContrastNormalization((0.3, 1.0))),
    iaa.Sometimes(0.3, iaa.ElasticTransformation(alpha=(1,5), sigma=0.1)),
    iaa.OneOf([
        iaa.Noop(),
        iaa.Sequential([
            iaa.OneOf([
                iaa.OneOf([
                    #Add in HSV or RGB
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
                    #Add elementwise in HSV or RGB
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
                    #Multiply in HSV or RGB
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
                    #Multiply elementwise in HSV or RGB
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
                iaa.CoarseSaltAndPepper(p=(0, 0.1), size_px=(64, 1024), per_channel=True),
                iaa.CoarseSaltAndPepper(p=(0, 0.1), size_px=(64, 1024), per_channel=False)
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
    iaa.Sometimes(0.3, iaa.ElasticTransformation(alpha=(1,5), sigma=0.1)),
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
import numpy as np
from imgaug import augmenters as iaa


def conditional_to_rgb(images, random_state, parents, hooks):
    images_rgb = []
    for image in images:
        h, w, c = image.shape
        if c == 1:
            print(image.shape)
            image = np.reshape(image, (h, w, 3))
            print(image.shape)
            images_rgb.append(image)
        else:
            images_rgb.append(image)
    return images_rgb


def conditional_to_gray(images, random_state, parents, hooks):
    images_rgb = []
    for image in images:
        h, w, c = image.shape
        if c == 1:
            print(image.shape)
            if image[:, :, 0] == image[:, :, 1]:
                image = np.expand_dims(image[:, :, 0], axis=2)
            print(image.shape)
            images_rgb.append(image)
        else:
            images_rgb.append(image)
    return images_rgb


def keypoint_func(keypoints_on_images, random_state, parents, hooks):
    return keypoints_on_images


augm_seq = iaa.Sequential([
    # General
    iaa.Fliplr(0.5),  # horizontal flips
    iaa.Flipud(0.5),
    iaa.Affine(rotate=180),
    iaa.CropAndPad(percent=(-0.5, 0.5), pad_cval=(0)),

    # # Color
    # iaa.Sequential([
    #     iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
    #     iaa.WithChannels(0, iaa.Add((50, 100))),
    #     iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
    # ]),
    # iaa.Multiply((0.5, 1.5), per_channel=0.5),

    # Deformations
    iaa.Affine(shear=(-16, 16)),
    iaa.PiecewiseAffine(scale=(0.01, 0.05))

], random_order=True)

basic_seq = iaa.Sequential([
    # iaa.Lambda(conditional_to_rgb, keypoint_func),
    augm_seq,
    # iaa.Lambda(conditional_to_gray, keypoint_func),
], random_order=False)

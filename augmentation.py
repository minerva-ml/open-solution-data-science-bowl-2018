from imgaug import augmenters as iaa

affine_seq = iaa.Sequential([
    # General
    iaa.SomeOf((1, 2),
               [iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Affine(rotate=180),
                iaa.CropAndPad(percent=(-0.5, 0.5), pad_cval=(0)),
                ]),

    # Deformations
    iaa.OneOf(
        [iaa.Affine(shear=(-16, 16)),
         iaa.PiecewiseAffine(scale=(0.01, 0.05))
         ])

], random_order=True)

color_seq = iaa.Sequential([
    # Color
    iaa.OneOf(
        [iaa.Sequential([
            iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
            iaa.WithChannels(0, iaa.Add((50, 100))),
            iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
        ]),
            iaa.Multiply((0.5, 1.5), per_channel=0.5),
        ])
], random_order=True)

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


from imgaug import augmenters as iaa

affine_seq = iaa.Sequential([
    # General
    iaa.SomeOf((1, 2),
               [iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Affine(rotate=(0, 360),
                           translate_percent={"x": (-0.1, 0.1), "y":(-0.1, 0.1)}),
                iaa.CropAndPad(percent=(-0.25, 0.25), pad_cval=0)
                ]),
    # Deformations
    iaa.Sometimes(0.9, iaa.PiecewiseAffine(scale=(0.00, 0.04))),
    iaa.Sometimes(0.9, iaa.PerspectiveTransform(scale=(0.01, 0.10)))
], random_order=True)

color_seq = iaa.Sequential([
    # Color
    iaa.Invert(0.3),
    iaa.Sometimes(0.3, iaa.ContrastNormalization((0.5, 1.5))),
    iaa.Sometimes(0.3, iaa.ElasticTransformation(alpha=(1,5), sigma=0.1)),
    iaa.OneOf([
        iaa.Noop(),
        iaa.Sequential([
            iaa.OneOf([
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
                ]),
                iaa.OneOf([
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
            iaa.GaussianBlur(sigma=(0.0, 8.0)),
            iaa.AverageBlur(k=(2, 21)),
            iaa.MedianBlur(k=(3, 15))
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

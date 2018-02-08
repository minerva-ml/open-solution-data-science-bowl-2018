from imgaug import augmenters as iaa


class ImgAug:
    def __init__(self, augmenters):
        if not isinstance(augmenters, list):
            augmenters = [augmenters]
        self.augmenters = augmenters
        self.seq_det = None

    def _pre_call_hook(self):
        seq = iaa.Sequential(self.augmenters)
        self.seq_det = seq.to_deterministic()

    def transform(self, images):
        return [self.seq_det.augment_image(image) for image in images]

    def __call__(self, *args):
        self._pre_call_hook()
        return self.transform(args)

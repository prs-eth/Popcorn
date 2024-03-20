"""
Code is adapted from https://github.com/SebastianHafner/DDA_UrbanExtraction
Modified: Arno Rüegg, Nando Metzger
"""

import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np


def compose_transformations(cfg):
    transformations = []

    if cfg.AUGMENTATION.CROP_SIZE:
        transformations.append(ImageCrop(cfg.AUGMENTATION.CROP_SIZE))

    if cfg.AUGMENTATION.RANDOM_FLIP:
        transformations.append(RandomFlip())

    if cfg.AUGMENTATION.RANDOM_ROTATE:
        transformations.append(RandomRotate())

    if cfg.AUGMENTATION.COLOR_SHIFT:
        transformations.append(ColorShift())

    if cfg.AUGMENTATION.GAMMA_CORRECTION:
        transformations.append(GammaCorrection())

    transformations.append(Numpy2Torch())

    return transforms.Compose(transformations)


class Numpy2Torch(object):
    def __call__(self, args):
        img, label = args
        img_tensor = TF.to_tensor(img)
        label_tensor = TF.to_tensor(label)
        return img_tensor, label_tensor


class RandomFlip(object):
    def __call__(self, args):
        img, label = args
        horizontal_flip = np.random.choice([True, False])
        vertical_flip = np.random.choice([True, False])

        if horizontal_flip:
            img = np.flip(img, axis=1)
            label = np.flip(label, axis=1)

        if vertical_flip:
            img = np.flip(img, axis=0)
            label = np.flip(label, axis=0)

        img = img.copy()
        label = label.copy()

        return img, label


class RandomRotate(object):
    def __call__(self, args):
        img, label = args
        k = np.random.randint(1, 4) # number of 90 degree rotations
        img = np.rot90(img, k, axes=(0, 1)).copy()
        label = np.rot90(label, k, axes=(0, 1)).copy()
        return img, label


class ColorShift(object):
    def __init__(self, min_factor: float = 0.5, max_factor: float = 1.5):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, args):
        img, label = args
        factors = np.random.uniform(self.min_factor, self.max_factor, img.shape[-1])
        img_rescaled = np.clip(img * factors[np.newaxis, np.newaxis, :], 0, 1).astype(np.float32)
        return img_rescaled, label


class GammaCorrection(object):
    def __init__(self, gain: float = 1, min_gamma: float = 0.25, max_gamma: float = 2):
        self.gain = gain
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma

    def __call__(self, args):
        img, label = args
        gamma = np.random.uniform(self.min_gamma, self.max_gamma, img.shape[-1])
        img_gamma_corrected = np.clip(np.power(img,gamma[np.newaxis, np.newaxis, :]), 0, 1).astype(np.float32)
        return img_gamma_corrected, label


class ImageCrop(object):
    def __init__(self, crop_size: int):
        self.crop_size = crop_size

    def __call__(self, args):
        img, label = args
        m, n, _ = img.shape
        i = 0 if m == self.crop_size else np.random.randint(0, m - self.crop_size)
        j = 0 if n == self.crop_size else np.random.randint(0, n - self.crop_size)
        img_crop = img[i:i + self.crop_size, j:j + self.crop_size, ]
        label_crop = label[i:i + self.crop_size, j:j + self.crop_size, ]
        return img_crop, label_crop

class ImageCroptest(object):
    def __call__(self, args):
        img, label = args
        m, n, _ = img.shape
        crop_size = min(m,n)
        while crop_size%16!=0:
            crop_size-=1
        i = 0 if m == crop_size else np.random.randint(0, m - crop_size)
        j = 0 if n == crop_size else np.random.randint(0, n - crop_size)
        img_crop = img[i:i + crop_size, j:j + crop_size, ]
        label_crop = label[i:i + crop_size, j:j + crop_size, ]
        return img_crop, label_crop

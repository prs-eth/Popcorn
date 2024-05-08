# Data Augmentations

import os
import random

import torch
import torchvision.transforms.functional as TF

from utils.utils import * 


class OwnCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        if torch.is_tensor(x):
            mask = None
        else:
            x, mask = x
        for t in self.transforms:
            x = t(x)
        return x if mask is None else (x, mask)

class UnNormalize(object):
    """
    Description:
        Unnormalize the image tensor with the given mean and standard deviation.

    Args:
        mean (tuple): mean of the dataset
        std (tuple): standard deviation of the dataset

    Returns:
        Tensor: Unnormalized image tensor
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
    

class RandomVerticalFlip(object):
    """
    Description:
        Randomly flip the input tensor image vertically with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5

    Returns:
        Tensor: Randomly flipped image tensor
    """
    def __init__(self, p=0.5, allsame=False): 
        self.p = p
        self.allsame = allsame
        
    def __call__(self, x):
        if torch.is_tensor(x):
            mask = None
        else:
            x, mask = x
        
        if self.allsame:
            if torch.rand(1) < self.p:
                x = TF.vflip(x)
                if mask is not None:
                    mask = TF.vflip(mask)
                    return x, mask
                return x
            else:
                if mask is not None:
                    return x, mask
                return x
        else:
            # random horizontal flip with probability 0.5 for each sample in batch
            selection = torch.rand(x.shape[0])<self.p
            x[selection] = TF.vflip(x)[selection]
            if mask is not None:
                mask[selection] = TF.vflip(mask)[selection]
                return x, mask
            return x 
        
    def __repr__(self):
        return self.__class__.__name__ + '(p={0}'.format(self.p)


class RandomHorizontalFlip(object):
    """
    Description:
        Randomly flip the input tensor image horizontally with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5

    Returns:
        Tensor: Randomly flipped image tensor
    """
    def __init__(self, p=0.5, allsame=False): 
        self.p = p
        self.allsame = allsame
        
    def __call__(self, x):
        if torch.is_tensor(x):
            mask = None
        else:
            x, mask = x
        
        if self.allsame:
            if torch.rand(1) < self.p:
                x = TF.hflip(x)
                if mask is not None:
                    mask = TF.hflip(mask)
                    return x, mask
                return x
            else:
                if mask is not None:
                    return x, mask
                return x
        else:
            # random horizontal flip with probability 0.5 for each sample in batch
            selection = torch.rand(x.shape[0])<self.p
            x[selection] = TF.hflip(x)[selection]
            if mask is not None:
                mask[selection] = TF.hflip(mask)[selection]
                return x, mask
            return x
        
    def __repr__(self):
        return self.__class__.__name__ + '(p={0}'.format(self.p)


class RandomRotationTransform(torch.nn.Module):
    """
    Description:
        Rotate by one of the given angles.
    Args:
        angles (sequence): sequence of rotation angles
        p (float): probability of the image being flipped. Default value is 0.5
    Returns:
        Tensor: Randomly rotated image tensor
    """

    def __init__(self, angles: list, p=0.5):
        self.angles = angles
        self.p = p

    def __call__(self, x):
        """
        Description:
            Rotate the input tensor image by one of the given angles.  
        """
        if torch.is_tensor(x):
            mask = None
        else:
            x, mask = x
        if torch.rand(1) < self.p:
            angle = random.choice(self.angles)
            if mask is not None:
                return TF.rotate(x, angle, expand=True), TF.rotate(mask, angle, expand=True, fill=-1)
                # return x.permute(0,1,3,2), mask.permute(0,1,3,2)
            return TF.rotate(x, angle)
        return x if mask is None else (x, mask)


class RandomGamma(torch.nn.Module):
    """
    Description:
        Perform gamma correction on an image.
        Also known as Power Law Transform. Intensities in RGB mode are adjusted

    Args:
        gamma_limit (tuple): gamma limit range
        p (float): probability of the image being flipped. Default value is 0.5
        s2_max (int): maximum value of the image tensor

    Returns:
        Tensor: Randomly gamma corrected image tensor
    """

    def __init__(self, gamma_limit=(0.5, 2.0), p=0.5, s2_max=10000):
        self.gamma_limit = gamma_limit
        self.p = p
        self.s2_max = s2_max

    def __call__(self, x):
        if torch.rand(1) < self.p:
            gamma = random.uniform(self.gamma_limit[0], self.gamma_limit[1])
            x = torch.clip(x, min=0)

            # convert to 0-1 range
            x = x / self.s2_max

            if len(x.shape) == 3:
                # Apply gamma to each channel separately if the image is in RGB mode
                if x.shape[0] == 3:
                    x = TF.adjust_brightness(x, gamma)

                else:
                    # Apply brightness to each channel separately
                    for i in range(x.shape[1]):
                        x[i:i+1] = TF.adjust_gamma(x[i:i+1], gamma)

            elif len(x.shape) == 4:
                # Apply gamma to each channel separately if the image is in RGB mode
                if x.shape[1] == 3:
                    x = TF.adjust_brightness(x, gamma)

                else:
                    # Apply brightness to each channel separately
                    for i in range(x.shape[1]):
                        x[:,i:i+1] = TF.adjust_gamma(x[:,i:i+1], gamma)

            x = x * self.s2_max
        return x


class RandomBrightness(torch.nn.Module):
    """
    Perform random brightness on an image.
    Args:
        beta_limit (tuple): beta limit range
        p (float): probability of the image being flipped. Default value is 0.5
    Returns:
        Tensor: Randomly brightness adjusted image tensor
    """

    def __init__(self, beta_limit=(0.666, 1.5), p=0.5):
        self.beta_limit = beta_limit
        self.p = p
        self.s2_max = 10000 # for the conversion to a pillow-typical range

    def __call__(self, x):
        """
        Applies the random brightness transformation with probability p.

        :param x: Tensor, input image
        :return: Tensor, output image with brightness adjusted if the transformation was applied
        """
        if torch.rand(1) < self.p:

            # get random brightness factor
            beta = random.uniform(self.beta_limit[0], self.beta_limit[1])

            # convert to pillow-typical range
            x = x / self.s2_max


            if len(x.shape) == 3:
                if x.shape[0] == 3:
                    x = TF.adjust_brightness(x, beta)
                else:
                    # Apply brightness to each channel separately
                    for i in range(x.shape[1]):
                        x[i:i+1] = TF.adjust_brightness(x[i:i+1], beta)
            elif len(x.shape) == 4:
                if x.shape[1] == 3:
                    x = TF.adjust_brightness(x, beta)
                else:
                    # Apply brightness to each channel separately
                    for i in range(x.shape[1]):
                        x[:,i:i+1] = TF.adjust_brightness(x[:,i:i+1], beta)

            # back to the original range
            x = x * self.s2_max

        return x
    

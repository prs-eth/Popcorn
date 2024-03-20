"""
Project: 🍿POPCORN: High-resolution Population Maps Derived from Sentinel-1 and Sentinel-2 🌍🛰️
Nando Metzger, 2024
"""

import os
import glob
import csv
import random

import numpy as np
import torch
from tqdm import tqdm
from random import randrange
import time
from pylab import figure, imshow, matshow, grid, savefig, colorbar
import pandas as pd
import matplotlib.pyplot as plt
import json


def to_cuda_inplace(sample, half=False, spare=[]):
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            sample[key] = val.cuda()
            if half and val.dtype == torch.float32 and key not in spare:
                sample[key] = sample[key].half()
        elif isinstance(val, list):
            new_val = []
            for e in val:
                if isinstance(e, torch.Tensor):
                    e = e.cuda()
                    if half and val.dtype == torch.float32 and key not in spare:
                        e = e.half()
                    new_val.append(e)
                else:
                    new_val.append(e)
            sample[key] = new_val
        elif isinstance(val, dict):
            sample[key] = to_cuda_inplace(val, half=half)
        else:
            sample[key] = val
    return sample


def detach_tensors_in_dict(input_dict):
    return {key: value.detach() if torch.is_tensor(value) else value for key, value in input_dict.items()}


def seed_all(seed):
    # Fix all random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed(seed) # gpu
    torch.cuda.manual_seed_all(seed) # multi-gpu
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def new_log(folder_path, args=None):
    os.makedirs(folder_path, exist_ok=True)
    n_exp = len(os.listdir(folder_path))
    randn  = round((time.time()*1000000) % 1000)
    experiment_folder = os.path.join(folder_path, f'experiment_{n_exp}_{randn}')
    os.mkdir(experiment_folder)

    if args is not None:
        args_dict = args.__dict__
        write_params(args_dict, os.path.join(experiment_folder, 'args' + '.csv'))

    return experiment_folder, n_exp, randn


def write_params(params, path):
    with open(path, 'w') as fh:
        writer = csv.writer(fh)
        writer.writerow(['key', 'value'])
        for data in params.items():
            writer.writerow([el for el in data])


def read_params(path):
    params = {}
    with open(path, 'r') as fh:
        reader = csv.reader(fh)
        next(reader)  # skip header
        for row in reader:
            key, value = row
            params[key] = value
    return params


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def load_json(file):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


def apply_normalize(indata, dataset_stats):
    """
    :param indata: input data
    :param dataset_stats: dataset statistics for normalization
    :return: normalized data
    """

    # S2
    if "S2" in indata:
        if indata["S2"].shape[1] == 4: 
            indata["S2"] = ((indata["S2"].permute((0,2,3,1)) - dataset_stats["sen2springNIR"]['mean'].cuda() ) / dataset_stats["sen2springNIR"]['std'].cuda()).permute((0,3,1,2))
        else:  
            indata["S2"] = ((indata["S2"].permute((0,2,3,1)) - dataset_stats["sen2spring"]['mean'].cuda() ) / dataset_stats["sen2spring"]['std'].cuda()).permute((0,3,1,2))

    # S1
    if "S1" in indata: 
        indata["S1"] = ((indata["S1"].permute((0,2,3,1)) - dataset_stats["sen1"]['mean'].cuda() ) / dataset_stats["sen1"]['std'].cuda()).permute((0,3,1,2))

    # VIIRS
    if "VIIRS" in indata: 
        indata["VIIRS"] = ((indata["VIIRS"].permute((0,3,1,2)) - dataset_stats["viirs"]['mean'].cuda() ) / dataset_stats["viirs"]['std'].cuda()).permute((0,3,1,2))

    return indata


def apply_transformations_and_normalize(sample, transform, dataset_stats, buildinginput=False, segmentationinput=False):
    """
    :param sample: image to be transformed
    :param transform: transform to be applied to the image
    :param dataset_stats: dataset statistics for normalization
    :param buildinginput: whether to use building input
    :param segmentationinput: whether to use segmentation input
    :return: transformed image and mask
    """

    # Modality-wise transformations
    if transform is not None:
        if "S2" in transform and "S2" in sample:
            sample["S2"] = transform["S2"](sample["S2"])
        if "S1" in transform and "S1" in sample:
            sample["S1"] = transform["S1"](sample["S1"])
        if "VIIRS" in transform and "VIIRS" in sample:
            sample["VIIRS"] = transform["VIIRS"](sample["VIIRS"])

    # Normalizations
    sample = apply_normalize(sample, dataset_stats)

    # merge inputs
    if buildinginput:
        
        if "building_counts" in sample.keys() and "building_segmentation" not in sample.keys():
            if segmentationinput:
                sample["building_segmentation"] = sample["building_counts"]>0.5
        if not segmentationinput and "building_segmentation" in sample.keys():
            del sample["building_segmentation"]

        # merge the inputs
        keys_to_concatenate = [sample[key] for key in ["S2", "S1", "VIIRS"] if key in sample]

        if keys_to_concatenate:
            sample["input"] = torch.concatenate(keys_to_concatenate, dim=1)
        else:
            # Handle the case where no modality is available
            sample["input"] = None
            
    else:
        sample["input"] = torch.concatenate([sample[key] for key in ["S2", "S1", "VIIRS"] if key in sample], dim=1)

    # General transformations
    if transform is not None:
        # apply the transformation to the image

        if "general" in transform.keys():

            # Collect data and lengths
            data = []
            lens = []
            keys = ["admin_mask", "positional_encoding", "building_counts", "building_segmentation"]

            # Collect data and lengths
            for key in keys:
                if key in sample:
                    if key == "admin_mask":
                        data.append(sample[key].unsqueeze(1))
                    else:
                        data.append(sample[key])
                    lens.append(data[-1].shape[1])

            # Apply the transformation if there is data
            if sample["input"] is not None:
                if sum(lens) > 0:
                    
                    # Transform the datas
                    if sample["input"] is not None:
                        sample["input"], data = transform["general"]((sample["input"], torch.cat(data, dim=1)))

                    # Reassign transformed data back into sample
                    start = 0
                    for i, key in enumerate(keys):
                        if key in sample:
                            end = start + lens[i]
                            sample[key] = data[:, start:end, :, :]
                            if key == "admin_mask":
                                sample[key] = sample[key][:,0,:,:]
                            start = end
                            
                else:
                    sample["input"] = transform["general"](sample["input"])
    
    return sample



class NumberList:
    """
    Class to store a list of numbers and return the last n numbers
    """
    def __init__(self, window=400):
        self.numbers = []
        self.window = window

    def add(self, nums):
        """
        Add a list of numbers to the list
        :param nums: list of numbers or other types of objects
        :return: None
        """
        for num in nums:
            if len(self.numbers) >= self.window:
                self.numbers.pop(0)
            self.numbers.append(num)

    def get(self):
        """
        Return the last n numbers
        :return: list of numbers
        """
        return self.numbers

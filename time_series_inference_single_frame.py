

import os
import torch
from torch import nn
from datetime import datetime
import rasterio
import numpy as np
from scipy.interpolate import griddata
from pathlib import Path
import json
from tqdm import tqdm


from utils.download_gee_country_single_frame_gaza import get_sentinel2_config, get_sentinel1desc_config, get_sentinel1asc_config
from utils.plot import plot_2dmatrix

from utils.constants import dda_cfg, stage1feats, stage2feats
from model.DDA_model.utils.networks import load_checkpoint


targetregion = "gaza"
root = "/scratch2/metzgern/HAC/data/PopMapData/raw/timseries/gaza"
outputdir = "/scratch2/metzgern/HAC/data/PopMapData/raw/timseries/gaza/BuiltUp"

# load normalization stats
p = 14
p2d = (p, p, p, p)

def load_json(file: Path):
    with open(str(file)) as f:
        d = json.load(f)
    return d


dataset_stats = load_json(os.path.join("data", "config", 'dataset_stats', 'my_dataset_stats_unified_2A.json'))
for mkey in dataset_stats.keys():
    if isinstance(dataset_stats[mkey], dict):
        for key,val in dataset_stats[mkey].items():
            dataset_stats[mkey][key] = torch.tensor(val)
    else:
        dataset_stats[mkey] = torch.tensor(val)


# Function to parse date string
def parse_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d")


# Function to find the closest frame
def find_closest_frame(target_date, frames):
    """
    Find the closest frame to the target date
    Input:
        target_date: target date
        frames: dictionary containing the frame dates
    Output:
        closest_frame: name of the closest frame
    """
    closest_frame = None
    min_diff = float('inf')

    for frame_name, frame_dates in frames.items():
        start_date, end_date = map(parse_date, frame_dates)
        avg_date = start_date + (end_date - start_date) / 2
        diff = abs((avg_date - target_date).days)

        if diff < min_diff:
            min_diff = diff
            closest_frame = frame_name

    return closest_frame


# Assuming rootfolder and target_region are provided as inputs
def generate_file_paths(rootfolder, target_region):
    """
    Generate file paths for the closest frames for each Sentinel-2 frame
    Input:
        rootfolder: path to the root folder containing the data
        target_region: name of the target region
    Output:
        closest_frames_paths: dictionary containing the file paths for the closest frames
    """
    closest_frames_paths = {}
    
    closest_frames = find_closest_frames()  # From previous function

    for s2_frame, closest_frames in closest_frames.items():
        s2_path = f"{rootfolder}/S2A/S2A_{s2_frame}_{target_region}_v1.tif"
        s1_desc_path = f"{rootfolder}/S1desc/s1desc_{closest_frames['Descending']}_{target_region}_v1.tif"
        s1_asc_path = f"{rootfolder}/S1asc/s1dasc_{closest_frames['Ascending']}_{target_region}_v1.tif"

        closest_frames_paths[s2_frame] = {
            'S2A': s2_path,
            'S1_Descending': s1_desc_path,
            'S1_Ascending': s1_asc_path
        }

    return closest_frames_paths


# Adjusting the find_closest_frames function to also output the average date of all the modalities
def find_closest_frames(rootfolder, target_region):
    """
    This function finds the closest frames for each Sentinel-2 frame in the given rootfolder
    Input:
        rootfolder: path to the root folder containing the data
        target_region: name of the target region
    Output:
        closest_frames_info: dictionary containing the information about the closest frames
    """
    sentinel2_frames = get_sentinel2_config()
    sentinel1_desc_frames = get_sentinel1desc_config()
    sentinel1_asc_frames = get_sentinel1asc_config()

    closest_frames_info = {}
    for frame_name, frame_dates in sentinel2_frames.items():
        start_date, end_date = map(parse_date, frame_dates)
        s2_avg_date = start_date + (end_date - start_date) / 2

        closest_desc = find_closest_frame(s2_avg_date, sentinel1_desc_frames)
        closest_asc = find_closest_frame(s2_avg_date, sentinel1_asc_frames)

        # Get average dates for the closest frames
        s1_desc_start, s1_desc_end = sentinel1_desc_frames[closest_desc]
        s1_desc_avg_date = (parse_date(s1_desc_start) + (parse_date(s1_desc_end) - parse_date(s1_desc_start)) / 2)

        s1_asc_start, s1_asc_end = sentinel1_asc_frames[closest_asc]
        s1_asc_avg_date = (parse_date(s1_asc_start) + (parse_date(s1_asc_end) - parse_date(s1_asc_start)) / 2)

        # Construct file paths
        s2_path = f"{rootfolder}/S2A/S2A_{frame_name}_{target_region}_v1.tif"
        s1_desc_path = f"{rootfolder}/S1desc/S1desc_{closest_desc}_{target_region}_v1.tif"
        s1_asc_path = f"{rootfolder}/S1asc/S1dasc_{closest_asc}_{target_region}_v1.tif"

        closest_frames_info[frame_name] = {
            'S2_Avg_Date': s2_avg_date.strftime("%Y-%m-%d"),
            'S1_Desc_Avg_Date': s1_desc_avg_date.strftime("%Y-%m-%d"),
            'S1_Asc_Avg_Date': s1_asc_avg_date.strftime("%Y-%m-%d"),
            'Closest_S1_Desc': closest_desc,
            'Closest_S1_Asc': closest_asc,
            'S2_Path': s2_path,
            'S1_Desc_Path': s1_desc_path,
            'S1_Asc_Path': s1_asc_path
        }

    return closest_frames_info


def interpolate_nan(input_array):
    """
    Interpolate the NaN values in the input array using bicubic interpolation, extrapolating if necessary using nearest neighbor
    Input:
        input_array: input array with NaN values
    Output:
        input_array: input array with NaN values interpolated
    """

    # Create an array with True values for NaN positions (interpolation mask)
    nan_mask = np.isnan(input_array)
    known_points = np.where(~nan_mask)
    values = input_array[known_points]
    missing_points = np.where(nan_mask)

    if (~nan_mask).sum()< 4:
        print("all nan detected")
        return np.zeros_like(input_array)
    
    # interpolate the missing values
    interpolated_values = griddata(np.vstack(known_points).T, values, np.vstack(missing_points).T, method='nearest')

    # fillin the missing ones
    input_array[missing_points] = interpolated_values

    return input_array


def rearrange_channels(input_array):
    """
    Rearrange the channels of the input array to match the order of the model
    Input:
        input_array: input array with channels in the wrong order
    Output:
        input_array: input array with channels in the correct order VV-VH-RGB-NIR
    """
    
    X = torch.cat([
        input_array[:, 4:6], # S1
        torch.flip(input_array[:, :3],dims=(1,)), # S2_RGB
        input_array[:, 3:4]], # S2_NIR
    dim=1)

    return input_array

# Function to load data into memory
def load_to_memory(infodict, device="cuda"):
    """
    Load data into memory
    Input:
        infodict: dictionary with information about the frames
    Output:
        datadict: dictionary with the data loaded into memory
    """
    datadict = {}

    for frame_name, frame_info in tqdm(infodict.items()):
        s2_path = frame_info['S2_Path']
        s1_desc_path = frame_info['S1_Desc_Path']
        s1_asc_path = frame_info['S1_Asc_Path']

        # using rasterio to load data
        with rasterio.open(s2_path) as src:
            s2_data = src.read()

        with rasterio.open(s1_desc_path) as src:
            s1_desc_data = src.read()

        with rasterio.open(s1_asc_path) as src:
            s1_asc_data = src.read()

        # get a mask for the nan values, reduce to 2d (1 channel)
        s2_nan_mask = np.isnan(s2_data).any(axis=0)
        s1_desc_nan_mask = np.isnan(s1_desc_data).any(axis=0)
        s1_asc_nan_mask = np.isnan(s1_asc_data).any(axis=0)
        overall_nan_mask1 = s2_nan_mask | s1_desc_nan_mask
        overall_nan_mask2 = s2_nan_mask | s1_asc_nan_mask

        # fill nan values
        s2_data = interpolate_nan(s2_data)
        s1_desc_data = interpolate_nan(s1_desc_data)
        s1_asc_data = interpolate_nan(s1_asc_data)

        # to torch and device
        s2_data = torch.from_numpy(s2_data).unsqueeze(0).float().to(device)
        s1_desc_data = torch.from_numpy(s1_desc_data).unsqueeze(0).float().to(device)
        s1_asc_data = torch.from_numpy(s1_asc_data).unsqueeze(0).float().to(device)
        overall_nan_mask1 = torch.from_numpy(overall_nan_mask1).unsqueeze(0).float().to(device)
        overall_nan_mask2 = torch.from_numpy(overall_nan_mask2).unsqueeze(0).float().to(device)

        # Normalize data
        s2_data = ((s2_data.permute((0,2,3,1)) - dataset_stats["sen2springNIR"]['mean'].to(device) ) / dataset_stats["sen2springNIR"]['std'].to(device)).permute((0,3,1,2))
        s1_desc_data = ((s1_desc_data.permute((0,2,3,1)) - dataset_stats["sen1"]['mean'].to(device) ) / dataset_stats["sen1"]['std'].to(device)).permute((0,3,1,2))
        s1_asc_data = ((s1_asc_data.permute((0,2,3,1)) - dataset_stats["sen1"]['mean'].to(device) ) / dataset_stats["sen1"]['std'].to(device)).permute((0,3,1,2))

        # concatenate the data
        s2_s1_desc = torch.cat((s2_data, s1_desc_data), dim=1)
        s2_s1_asc = torch.cat((s2_data, s1_asc_data), dim=1)

        # rearrange channels
        s2_s1_desc = rearrange_channels(s2_s1_desc)
        s2_s1_asc = rearrange_channels(s2_s1_asc)

        # create the geotiff metadata for future saving of output, the can just copy the metadata from the input
        with rasterio.open(s2_path) as src:
            profile = src.profile
            profile.update(dtype=rasterio.float32, count=1)
        output_file = os.path.join(outputdir, f"BuiltUp_{frame_name}.tif")
       
        # save
        datadict[frame_name] = {
            "s2_s1_desc": s2_s1_desc,
            "s2_s1_asc": s2_s1_asc,
            "mask_desc": overall_nan_mask2,
            "mask_asc": overall_nan_mask2,
            "output_metadata": profile,
            "output_filename": output_file,
        }

    return datadict


def add_padding(data: torch.Tensor, force=True) -> torch.Tensor:
    """
    Description:
        - Add padding to the input data
    Input:
        - data (torch.Tensor): input data
        - force (bool): whether to force the padding
    Output:
        - data (torch.Tensor): padded data
    """
    # Add padding
    px1,px2,py1,py2 = None, None, None, None
    if force:
        data  = nn.functional.pad(data, p2d, mode='reflect')
        px1,px2,py1,py2 = p, p, p, p
    else:
        # pad to make sure it is divisible by 32
        if (data.shape[2] % 32) != 0:
            px1 = (64 - data.shape[2] % 64) //2
            px2 = (64 - data.shape[2] % 64) - px1
            # data = nn.functional.pad(data, (px1,0,px2,0), mode='reflect') 
            data = nn.functional.pad(data, (0,0,px1,px2,), mode='reflect') 
        if (data.shape[3] % 32) != 0:
            py1 = (64 - data.shape[3] % 64) //2
            py2 = (64 - data.shape[3] % 64) - py1
            data = nn.functional.pad(data, (py1,py2,0,0), mode='reflect')

    return data, (px1,px2,py1,py2)

def revert_padding(data: torch.tensor, padding: tuple) -> torch.Tensor:
    """
    Description:
        - Revert the padding of the input data
    Input:
        - data (torch.Tensor): input data
        - padding (tuple): padding parameters
    Output:
        - data (torch.Tensor): padded data
    """
    px1,px2,py1,py2 = padding
    if px1 is not None or px2 is not None:
        data = data[:,:,px1:-px2,:]
    if py1 is not None or py2 is not None:
        data = data[:,:,:,py1:-py2]
    return data


if __name__=="__main__":

    # arrage dates
    infodict = find_closest_frames(root, targetregion)
    
    debug = False
    if debug:
        infodict = {k:infodict[k] for k in list(infodict.keys())[:1]}


    # load data into memory
    datadict = load_to_memory(infodict, device="cpu")

    # load model
    unetmodel, _, _ = load_checkpoint(epoch=30, cfg=dda_cfg, device="cuda", no_disc=True)
    unetmodel.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # inference
    for frame_name, frame_data in tqdm(datadict.items()):
        s2_s1_desc = frame_data["s2_s1_desc"].to(device)
        s2_s1_asc = frame_data["s2_s1_asc"].to(device)
        mask_desc = frame_data["mask_desc"].to(device)
        mask_asc = frame_data["mask_asc"].to(device)

        # add padding
        s2_s1_desc, (px1,px2,py1,py2) = add_padding(s2_s1_desc, True)
        s2_s1_asc, (px1,px2,py1,py2) = add_padding(s2_s1_asc, True)

        # inference
        with torch.no_grad():
            logits_fusion_desc = unetmodel.sparse_forward(s2_s1_desc, torch.ones_like(mask_desc), return_features=False)
            logits_fusion_asc = unetmodel.sparse_forward(s2_s1_asc, torch.ones_like(mask_asc), return_features=False)

            # logics to probabilities
            pred_desc = torch.sigmoid(logits_fusion_desc)
            pred_asc = torch.sigmoid(logits_fusion_asc)

            pred = (pred_desc + pred_asc) / 2

        # revert padding
        pred = revert_padding(pred, (px1,px2,py1,py2)) 

        # save
        a = pred[0,0].cpu().numpy() 

        # make output dir if not exists
        os.makedirs(os.path.dirname(frame_data["output_filename"]), exist_ok=True)

        # save resut as a tif file
        with rasterio.open(frame_data["output_filename"], 'w', **frame_data["output_metadata"]) as dst:
            dst.write(a, 1)
        



    print("Done")
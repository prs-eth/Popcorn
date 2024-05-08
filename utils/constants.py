"""
Project: 🍿POPCORN: High-resolution Population Maps Derived from Sentinel-1 and Sentinel-2 🌍🛰️
Nando Metzger, 2024
"""

import os
import numpy as np
from os.path import dirname
from utils.utils import Namespace


inference_patch_size = 2048
overlap = 128

# data folder
large_file_paths = [
        "/scratch2/metzgern/HAC/data",
        "/scratch/metzgern/HAC/data",
        "/cluster/work/igp_psr/metzgern/HAC/data",
        "/cluster/scratch/metzgern"
]
for name in large_file_paths:
    if os.path.isdir(name):
        large_file_path = name
if large_file_path is None:
    raise Exception("No data folder found")
pop_map_root = os.path.join(large_file_path, os.path.join("PopMapData", "processed"))
pop_map_root_large = os.path.join("/scratch2/metzgern/HAC/data", os.path.join("PopMapData", "processed"))
pop_map_covariates = os.path.join(large_file_path, os.path.join("PopMapData", os.path.join("merged", "EE")))
print("pop_map_root", pop_map_root)

# raw data folder
raw_file_paths = [
        "/scratch2/metzgern/HAC/data",
        "/scratch/metzgern/HAC/data",
        "/cluster/work/igp_psr/metzgern/HAC/data",
        "/cluster/scratch/metzgern"
]
for name in raw_file_paths:
    if os.path.isdir(name):
        raw_file_path = name
if raw_file_path is None:
    raise Exception("No data folder found")
raw_map_root = os.path.join(raw_file_path, os.path.join("PopMapData", "raw"))
rawEE_map_root = os.path.join(raw_map_root, "EE")
print("rawEE_map_root", rawEE_map_root)

# google buildings data folder
data_paths_aux = [
        "/scratch/metzgern/HAC/data",
        "/scratch2/metzgern/HAC/data",
        "/cluster/work/igp_psr/metzgern/HAC/data",
]
for name in data_paths_aux:
    if os.path.isdir(name):
        data_path_aux = name
if large_file_path is None:
    raise Exception("No data folder found")
pop_gbuildings_path = os.path.join(data_path_aux, os.path.join("PopMapData", os.path.join("raw", "GoogleBuildings")))
print("pop_gbuildings_path", pop_gbuildings_path)

src_path = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(os.path.join(src_path, 'data'), 'config')

# Definitions of where to find the census data and the boundaries of the target areas
datalocations = {
    'pricp2': {
        'fine': {
            'boundary': "boundaries4.tif",
            'census': "census4.csv",
        },
        'fineBLOCKCE': {
            'boundary': "boundaries_BLOCKCE20.tif",
            'census': "census_BLOCKCE20.csv",
        },
        'fineCOUNTYFP': {
            'boundary': "boundaries_COUNTYFP20.tif",
            'census': "census_COUNTYFP20.csv",
        },
        'fineTRACTCE': {
            'boundary': "boundaries_TRACTCE20.tif",
            'census': "census_TRACTCE20.csv",
        },
        'coarseTRACTCE': {
            'boundary': "boundaries_coarseTRACTCE20.tif",
            'census': "census_coarseTRACTCE20.csv",
        },
        'coarse': {
            'boundary': "boundaries_TRACTCE20.tif",
            'census': "census_TRACTCE20.csv",
        }
    },
    'rwa': {
        'fine100': {
            'boundary': "boundaries_kigali100.tif",
            'census': "census_kigali100.csv",
        },
        'coarse': {
            'boundary': "boundaries_coarse.tif",
            'census': "census_coarse.csv",
        } 
    },
    "uga": {
        'coarse': {
            'boundary': "boundaries.tif",
            'census': "census.csv",
        },
        'fine': {
            'boundary': "boundaries.tif",
            'census': "census.csv",
        },
    },
    "che": {
        'coarse4': {
            'boundary': "boundaries_coarse4.tif",
            'census': "census_coarse4.csv",
        },
        'coarse3': {
            'boundary': "boundaries_coarse3.tif",
            'census': "census_coarse3.csv",
        },
        'coarse1': {
            'boundary': "boundaries_coarse1.tif",
            'census': "census_coarse1.csv",
        },
        'finezurich': {
            'boundary': "boundaries_finezurich.tif",
            'census': "census_finezurich.csv",
        },
        'finezurich2': {
            'boundary': "boundaries_finezurich2.tif",
            'census': "census_finezurich2.csv",
        },
        'fine': {
            'boundary': "boundaries_fine.tif",
            'census': "census_fine.csv",
        },
        'coarse': {
            'boundary': "boundaries_coarse4.tif",
            'census': "census_coarse4.csv",
        },
    }
}

testlevels = {
    'pricp2': ["fine", "fineTRACTCE"],
    'rwa': ["fine100", "coarse"],
    'uga': ["coarse"],
    'che': ["finezurich2", "coarse4"],
}

testlevels_eval = {
    'pricp2': ["fine", "fineTRACTCE"],
    'rwa': ["fine100", "coarse"],
    'uga': ["coarse"],
    'che': ["fine", "finezurich2", "coarse4"],
}


# inicies to skip while training
skip_indices = {
    "pricp2": [],
    "rwa": [],
    "uga": [1323],
    "che": [],
}


# DDA model definitions
stage1feats = 8
stage2feats = 16
dda_dir="model/DDA_model/checkpoints/"
MODEL = Namespace(TYPE='dualstreamunet', OUT_CHANNELS=1, IN_CHANNELS=6, TOPOLOGY=[stage1feats, stage2feats,] )
CONSISTENCY_TRAINER = Namespace(LOSS_FACTOR=0.5)
PATHS = Namespace(OUTPUT=dda_dir)
DATALOADER = Namespace(SENTINEL1_BANDS=['VV', 'VH'], SENTINEL2_BANDS=['B02', 'B03', 'B04', 'B08'])
TRAINER = Namespace(LR=1e5)
dda_cfg = Namespace(MODEL=MODEL, CONSISTENCY_TRAINER=CONSISTENCY_TRAINER, PATHS=PATHS,
                DATALOADER=DATALOADER, TRAINER=TRAINER, NAME=f"fusionda_newAug{stage1feats}_{stage2feats}")


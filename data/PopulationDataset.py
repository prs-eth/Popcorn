"""
Project: 🍿POPCORN: High-resolution Population Maps Derived from Sentinel-1 and Sentinel-2 🌍🛰️
Nando Metzger, 2024
"""

import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from scipy import ndimage
from scipy.interpolate import griddata
import rasterio
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import random

from typing import Dict, Tuple

from utils.constants import pop_map_root, pop_map_covariates, pop_gbuildings_path, rawEE_map_root, skip_indices
from utils.constants import datalocations

def load_json(file):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


class Population_Dataset(Dataset):
    """
    Population dataset for target domain
    Use this dataset to evaluate the model on the target domain and compare it the census data
    """
    def __init__(self, region, S1=False, S2=True, VIIRS=True, NIR=False, patchsize=1024, overlap=32, fourseasons=False, mode="test",
                 max_samples=None, transform=None, sentinelbuildings=True, ascfill=False, ascAug=False, train_level="fine", split="all",
                 max_pix=5e6, max_pix_box=12000000) -> None:
        """
        Input:
            region: the region identifier (e.g. "pri" for puerto rico)
            S1: whether to use sentinel 1 data
            S2: whether to use sentinel 2 data
            VIIRS: whether to use VIIRS data
            NIR: whether to use NIR data
            patchsize: the size of the patches to extract
            overlap: the overlap between patches
            fourseasons: whether to use the four seasons data
            mode: the mode to use ("weaksup" (weakly supervised training) or "test")
            split: the split to use ("all", "train", "val")
            max_samples: the maximum number of samples to use
            transform: the transform to apply to the data
            sentinelbuildings: whether to use the sentinel buildings
            ascfill: whether to use the ascending orbit data to fill the missing values
            ascAug: whether to use the ascending orbit data to augment the data
            train_level: the level of the training data ("fine" or "coarse")
            max_pix: the maximum number of pixels in the administrative region
            max_pix_box: the maximum number of pixels in the bounding box
        """
        super().__init__()

        print("---------------------")
        print("Setting up dataset", region, "with mode", mode, "and split", split)
        
        # set the parameters of the dataset 
        self.region = region
        self.S1 = S1
        self.S2 = S2
        self.NIR = NIR
        self.VIIRS = VIIRS
        self.patchsize = patchsize
        self.overlap = overlap
        self.fourseasons = fourseasons
        self.mode = mode
        self.transform = transform
        self.sentinelbuildings = sentinelbuildings
        self.ascfill = ascfill
        self.split = split
        self.ascAug = ascAug
        self.train_level = train_level

        # get the path to the data files
        region_root = os.path.join(pop_map_root, region)

        # load the boundary and census data as a path dictionary
        levels = datalocations[region].keys()
        self.file_paths = {}
        for level in levels:
            self.file_paths[level] = {}
            for data_type in ["boundary", "census"]:
                self.file_paths[level][data_type] = os.path.join(region_root, datalocations[region][level][data_type])
            
        # weaksup data specific preparation
        if self.mode == "weaksup":
            # read the census file
            self.coarse_census = pd.read_csv(self.file_paths[train_level]["census"])
            
            # kicking out samples that are ugly shaped and difficualt to handle using the skip_indices
            self.coarse_census = self.coarse_census[~self.coarse_census["idx"].isin(skip_indices[region])].reset_index(drop=True)

            # redefine indexing
            if max_samples is not None:
                # shuffle and sample the data
                self.coarse_census = self.coarse_census.sample(frac=1, random_state=1610)[-max_samples:].reset_index(drop=True)

            # check the split mode
            if split=="all":
                self.coarse_census = self.coarse_census
                print("Using", len(self.coarse_census), "samples in this dataset")

            elif split=="train":
                # shuffle and split the data
                self.coarse_census = self.coarse_census.sample(frac=1, random_state=1610)[:int(len(self.coarse_census)*0.8)].reset_index(drop=True)

                print("Using", len(self.coarse_census), "samples for weakly supervised training")
            elif split=="val":
                # shuffle and split the data
                self.coarse_census = self.coarse_census.sample(frac=1, random_state=1610)[int(len(self.coarse_census)*0.8):].reset_index(drop=True)
                print("Using", len(self.coarse_census), "samples for weakly supervised validation")

            else:
                raise ValueError("Split not recognized")

            # kicking out samples with too many pixels in the administrative region
            print("Kicking out ", (self.coarse_census["count"]>=max_pix).sum(), "samples with more than ", int(max_pix), " pixels in the administrative region")
            self.coarse_census = self.coarse_census[self.coarse_census["count"]<max_pix].reset_index(drop=True)

            # Print the number of samples exceeding the max pixels of the bounding box (12000000) and kick them out
            self.coarse_census["bbox_count"] = self.coarse_census["bbox"].apply(self.calculate_pixel_count)
            num_samples_exceeding_max = (self.coarse_census["bbox_count"] >= max_pix_box).sum()
            print(f"Kicking out {num_samples_exceeding_max} samples with more than {max_pix_box} pixels in the bounding box")
            self.coarse_census = self.coarse_census[self.coarse_census["bbox_count"] < max_pix_box].reset_index(drop=True)

            # print the number of samples
            print("Effective number of samples: ", len(self.coarse_census))

            # get the shape of the coarse regions
            with rasterio.open(self.file_paths[train_level]["boundary"], "r") as src:
                self.cr_shape = src.shape


        elif self.mode=="test":
            # testdata specific preparation
            # get the shape and metadata of the images
            with rasterio.open(self.file_paths[list(self.file_paths.keys())[0]]["boundary"], "r") as src:
                self.img_shape = src.shape
                self._meta = src.meta.copy()
            self._meta.update(count=1, dtype='float32', nodata=None, compress='lzw', BIGTIFF="IF_SAFER")

            # get a list of indices of the possible patches
            self.patch_indices = self.get_patch_indices(patchsize, overlap)
        else:
            raise ValueError("Mode not recognized")

        # get the path to the data files
        covar_root = os.path.join(pop_map_covariates, region)
        if not os.path.exists(covar_root):
            covar_root = covar_root.replace("scratch", "scratch3")
        if not os.path.exists(covar_root):
            covar_root = covar_root.replace("scratch3", "scratch2")
        print("Using covar_root: ", covar_root)
        
        # load the covariates for Sentinel 1
        S1spring_file = os.path.join(covar_root,  os.path.join("S1spring", region +"_S1spring.tif"))
        S1summer_file = os.path.join(covar_root,  os.path.join("S1summer", region +"_S1summer.tif"))
        S1autumn_file = os.path.join(covar_root,  os.path.join("S1autumn", region +"_S1autumn.tif"))
        S1winter_file = os.path.join(covar_root,  os.path.join("S1winter", region +"_S1winter.tif"))

        if ascfill:
            S1springAsc_file = os.path.join(covar_root,  os.path.join("S1springAsc", region +"_S1springAsc.tif"))
            S1summerAsc_file = os.path.join(covar_root,  os.path.join("S1summerAsc", region +"_S1summerAsc.tif"))
            S1autumnAsc_file = os.path.join(covar_root,  os.path.join("S1autumnAsc", region +"_S1autumnAsc.tif"))
            S1winterAsc_file = os.path.join(covar_root,  os.path.join("S1winterAsc", region +"_S1winterAsc.tif"))

            self.S1Asc_file = {0: S1springAsc_file, 1: S1summerAsc_file, 2: S1autumnAsc_file, 3: S1winterAsc_file}

        if not os.path.exists(S1spring_file): 
            print(S1spring_file, "Does not exist, are you sure your file paths were set correctly?")
            print("Start searching and building virtual files from the unmerged raw files...") 
            global rawEE_map_root
             
            if not os.path.exists(os.path.join(rawEE_map_root, region, "S1spring")):
                rawEE_map_root = rawEE_map_root.replace("scratch", "scratch3")
            if not os.path.exists(os.path.join(rawEE_map_root, region, "S1spring")):
                rawEE_map_root = rawEE_map_root.replace("scratch3", "scratch2")
            print("Using rawEE_map_root: ", rawEE_map_root)
        
            spring_dir = os.path.join(rawEE_map_root, region, "S1spring")
            summer_dir = os.path.join(rawEE_map_root, region, "S1summer")
            autumn_dir = os.path.join(rawEE_map_root, region, "S1autumn")
            winter_dir = os.path.join(rawEE_map_root, region, "S1winter")

            if not os.path.exists(os.path.join(rawEE_map_root, region, "S1winter_out.vrt")):
                print("VRT {} file do not exist".format(os.path.join(rawEE_map_root, region, "S1winter_out.vrt")))

                from osgeo import gdal
                _ = gdal.BuildVRT(os.path.join(rawEE_map_root, region, "S1spring_out.vrt"), [ os.path.join(spring_dir, f) for f in os.listdir(spring_dir) if f.endswith(".tif")])
                _ = gdal.BuildVRT(os.path.join(rawEE_map_root, region, "S1summer_out.vrt"), [ os.path.join(summer_dir, f) for f in os.listdir(summer_dir) if f.endswith(".tif")])
                _ = gdal.BuildVRT(os.path.join(rawEE_map_root, region, "S1autumn_out.vrt"), [ os.path.join(autumn_dir, f) for f in os.listdir(autumn_dir) if f.endswith(".tif")])
                _ = gdal.BuildVRT(os.path.join(rawEE_map_root, region, "S1winter_out.vrt"), [ os.path.join(winter_dir, f) for f in os.listdir(winter_dir) if f.endswith(".tif")])

            S1spring_file = os.path.join(rawEE_map_root, region, "S1spring_out.vrt")
            S1summer_file = os.path.join(rawEE_map_root, region, "S1summer_out.vrt")
            S1autumn_file = os.path.join(rawEE_map_root, region, "S1autumn_out.vrt")
            S1winter_file = os.path.join(rawEE_map_root, region, "S1winter_out.vrt")
            
            if ascfill:
                spring_dir = os.path.join(rawEE_map_root, region, "S1springAsc")
                summer_dir = os.path.join(rawEE_map_root, region, "S1summerAsc")
                autumn_dir = os.path.join(rawEE_map_root, region, "S1autumnAsc")
                winter_dir = os.path.join(rawEE_map_root, region, "S1winterAsc")

                if not os.path.exists(os.path.join(rawEE_map_root, region, "S1winterAsc_out.vrt")):
                    print("VRT {} file do not exist".format(os.path.join(rawEE_map_root, region, "S1winterAsc_out.vrt")))

                    from osgeo import gdal
                    _ = gdal.BuildVRT(os.path.join(rawEE_map_root, region, "S1springAsc_out.vrt"), [ os.path.join(spring_dir, f) for f in os.listdir(spring_dir) if f.endswith(".tif")])
                    _ = gdal.BuildVRT(os.path.join(rawEE_map_root, region, "S1summerAsc_out.vrt"), [ os.path.join(summer_dir, f) for f in os.listdir(summer_dir) if f.endswith(".tif")])
                    _ = gdal.BuildVRT(os.path.join(rawEE_map_root, region, "S1autumnAsc_out.vrt"), [ os.path.join(autumn_dir, f) for f in os.listdir(autumn_dir) if f.endswith(".tif")])
                    _ = gdal.BuildVRT(os.path.join(rawEE_map_root, region, "S1winterAsc_out.vrt"), [ os.path.join(winter_dir, f) for f in os.listdir(winter_dir) if f.endswith(".tif")])

                S1springAsc_file = os.path.join(rawEE_map_root, region, "S1springAsc_out.vrt")
                S1summerAsc_file = os.path.join(rawEE_map_root, region, "S1summerAsc_out.vrt")
                S1autumnAsc_file = os.path.join(rawEE_map_root, region, "S1autumnAsc_out.vrt")
                S1winterAsc_file = os.path.join(rawEE_map_root, region, "S1winterAsc_out.vrt")

                self.S1Asc_file = {0: S1springAsc_file, 1: S1summerAsc_file, 2: S1autumnAsc_file, 3: S1winterAsc_file}


        self.S1_file = {0: S1spring_file, 1: S1summer_file, 2: S1autumn_file, 3: S1winter_file}
        self.S1_file = {0: S1spring_file, 1: S1summer_file, 2: S1autumn_file, 3: S1winter_file}
        
        # load the covariates for Sentinel 2
        S2spring_file = os.path.join(covar_root,  os.path.join("S2Aspring", region +"_S2Aspring.tif"))
        S2summer_file = os.path.join(covar_root,  os.path.join("S2Asummer", region +"_S2Asummer.tif"))
        S2autumn_file = os.path.join(covar_root,  os.path.join("S2Aautumn", region +"_S2Aautumn.tif"))
        S2winter_file = os.path.join(covar_root,  os.path.join("S2Awinter", region +"_S2Awinter.tif"))
        
        # if not exists, we use the virtual rasters of the raw files, if exists, we use the preprocessed files 
        if not os.path.exists(S2spring_file):
            print(S2spring_file, "Does not exist, are you sure your file paths were set correctly?")
            print("Start searching and building virtual files from the unmerged raw files...")
            
            spring_dir = os.path.join(rawEE_map_root, region, "S2Aspring")
            summer_dir = os.path.join(rawEE_map_root, region, "S2Asummer")
            autumn_dir = os.path.join(rawEE_map_root, region, "S2Aautumn")
            winter_dir = os.path.join(rawEE_map_root, region, "S2Awinter")

            if not os.path.exists(os.path.join(rawEE_map_root, region, "S2Awinter_out.vrt")): 
                print("VRT {} file do not exist".format(os.path.join(rawEE_map_root, region, "S2Awinter_out.vrt")))

                from osgeo import gdal
                _ = gdal.BuildVRT(os.path.join(rawEE_map_root, region, "S2Aspring_out.vrt"), [ os.path.join(spring_dir, f) for f in os.listdir(spring_dir) if f.endswith(".tif")])
                _ = gdal.BuildVRT(os.path.join(rawEE_map_root, region, "S2Asummer_out.vrt"), [ os.path.join(summer_dir, f) for f in os.listdir(summer_dir) if f.endswith(".tif")])
                _ = gdal.BuildVRT(os.path.join(rawEE_map_root, region, "S2Aautumn_out.vrt"), [ os.path.join(autumn_dir, f) for f in os.listdir(autumn_dir) if f.endswith(".tif")])
                _ = gdal.BuildVRT(os.path.join(rawEE_map_root, region, "S2Awinter_out.vrt"), [ os.path.join(winter_dir, f) for f in os.listdir(winter_dir) if f.endswith(".tif")])

            S2spring_file = os.path.join(rawEE_map_root, region, "S2Aspring_out.vrt")
            S2summer_file = os.path.join(rawEE_map_root, region, "S2Asummer_out.vrt")
            S2autumn_file = os.path.join(rawEE_map_root, region, "S2Aautumn_out.vrt")
            S2winter_file = os.path.join(rawEE_map_root, region, "S2Awinter_out.vrt")

        self.S2_file = {0: S2spring_file, 1: S2summer_file, 2: S2autumn_file, 3: S2winter_file}
        self.S2_file = {0: S2spring_file, 1: S2summer_file, 2: S2autumn_file, 3: S2winter_file}

        self.season_dict = {0: "spring", 1: "summer", 2: "autumn", 3: "winter"}
        self.inv_season_dict = {v: k for k, v in self.season_dict.items()}
        self.VIIRS_file = os.path.join(covar_root,  os.path.join("viirs", region +"_viirs.tif"))

        # load the google buildings, if requested...
        if self.sentinelbuildings:
            # load sentinel buildings, (Not used anymore, will calculate on-the-fly)
            self.gbuildings_segmentation_file = ''
            self.sbuildings = True
            self.gbuildings = False
        else:
            self.sbuildings_segmentation_file = ''
            if region == "che":
                # there are no google buildings for Switzerland
                global pop_gbuildings_path
                pop_gbuildings_path = pop_gbuildings_path.replace("GoogleBuildings", "SwissBuildings")
                self.gbuildings_segmentation_file = os.path.join(pop_gbuildings_path, "SwissTLM3D", "swisstlm3d_2020-03_2056_5728", "2020_SWISSTLM3D_SHP_CHLV95_LN02", "TLM_BAUTEN", "swissTLM3D_TLM_GEBAEUDE_FOOTPRINT_segmentation_s2.tif")
                self.gbuildings_counts_file = os.path.join(pop_gbuildings_path, "SwissTLM3D", "swisstlm3d_2020-03_2056_5728", "2020_SWISSTLM3D_SHP_CHLV95_LN02", "TLM_BAUTEN", "swissTLM3D_TLM_GEBAEUDE_FOOTPRINT_count_s2.tif")
            else:
                self.gbuildings_segmentation_file = os.path.join(pop_gbuildings_path, region, "Gbuildings_" + region + "_segmentation.tif")
                self.gbuildings_counts_file = os.path.join(pop_gbuildings_path, region, "Gbuildings_" + region + "_counts.tif")
            self.gbuildings = True 
 
        print("---------------------")

    # delete the dataset
    def __del__(self):
        pass 

    def get_patch_indices(self, patchsize, overlap):
        """
        :param patchsize: size of the patch
        :param overlap: overlap between patches
        :return: list of indices of the patches
        """
        # get the indices of the main patches
        stride = patchsize - overlap*2
        h,w = self.img_shape

        # get the indices of the main patches
        x = torch.arange(0, h-patchsize, stride, dtype=int)
        y = torch.arange(0, w-patchsize, stride, dtype=int)
        main_indices = torch.cartesian_prod(x,y)

        # also cover the boarder pixels that are not covered by the main indices
        max_x = h-patchsize
        max_y = w-patchsize
        bottom_indices = torch.stack([torch.ones(len(y), dtype=int)*max_x, y]).T
        right_indices = torch.stack([x, torch.ones(len(x), dtype=int)*max_y]).T

        # add the bottom right corner
        bottom_right_idx = torch.tensor([max_x, max_y]).unsqueeze(0)

        # concatenate all the indices
        main_indices = torch.cat([main_indices, bottom_indices, right_indices, bottom_right_idx])

        # concatenate the season indices, e.g encode the season an enlongate the indices
        season_template = torch.ones(main_indices.shape[0], dtype=int).unsqueeze(1)
        if self.fourseasons:
            main_indices = torch.cat([
                torch.cat([main_indices, season_template*0], dim=1),
                torch.cat([main_indices, season_template*1], dim=1),
                torch.cat([main_indices, season_template*2], dim=1),
                torch.cat([main_indices, season_template*3], dim=1)],
                dim=0
            )
        else:
            main_indices = torch.cat([main_indices, season_template*0], dim=1)

        return main_indices


    def parse_bbox(self, bbox_str):
        """
        Parse a bounding box string and return the coordinates as integers.
        Input:
            bbox_str: the bounding box string
        Output:
            the coordinates of the bounding box as integers
        """
        bbox_values = bbox_str.strip('()[]').split(',')
        return [int(val) for val in bbox_values]

    def calculate_pixel_count(self,bbox_str):
        """
        Calculate the number of pixels inside the bounding box.
        Input:
            bbox_str: the bounding box string
        Output:
            the number of pixels inside the bounding box
            """
        x1, y1, x2, y2 = self.parse_bbox(bbox_str)
        return (y1 - x1) * (y2 - x2)


    def metadata(self):
        return self._meta

    def shape(self) -> Tuple[int, int]:
        return self.img_shape

    def __len__(self) -> int:
        if self.mode=="test":
            return len(self.patch_indices)
        elif self.mode=="weaksup":
            return len(self.coarse_census)

    def __getitem__(self, index: int) -> Dict[str, torch.FloatTensor]:
        """
        Description:
            Get the item at the given index, depending on the mode, the item is either a patch or a coarse region,
        Input:
            index: index of the item
        Output:
            item: dictionary containing the input, the mask, the coordinates of the patch, the season and the season string
        """
        if self.mode=="test":
            return self.__gettestitem__(index)
        elif self.mode=="weaksup":
            return self.__getadminitem__(index)


    def __getadminitem__(self, index: int) -> Dict[str, torch.FloatTensor]:
        """
        Description:
            Get the item at the given index, the item is a coarse region with arbitrary size, shape
        Input:
            index: index of the item
        Output:
            item: dictionary containing
                indata: the input data (S1, S2, VIIRS, NIR)
                y: the target data (population)
                admin_mask: the mask of the administrative region
                img_coords: the coordinates of the patch
                season: the season of the data as integer (0,1,2,3) for (spring, summer, autumn, winter)
                census_idx: the index of the census sample
        """
        # get the indices of the patch
        census_sample = self.coarse_census.loc[index]
        
        # get the coordinates of the patch 
        xmin, xmax, ymin, ymax = tuple(map(int, census_sample["bbox"].strip('()').strip('[]').split(',')))

        # get the season for the S2 data
        season = random.choice(['spring', 'autumn', 'winter', 'summer']) if self.fourseasons else "spring"
        descending = random.choice([True, False]) if self.ascAug else True

        # overlap for the admin mask
        ad_over = 32

        # get the data
        indata, auxdata, w = self.generate_raw_data(xmin, ymin, self.inv_season_dict[season], patchsize=(xmax-xmin, ymax-ymin), overlap=0, admin_overlap=ad_over, descending=descending)

        if "S2" in indata:
            if np.any(np.isnan(indata["S2"])): 
                # interpolate the NaN values in the input array using bicubic interpolation, extrapolating if necessary using nearest neighbor if there are not too many NaNs
                indata["S2"] = self.interpolate_nan(indata["S2"]) 
            
        if "S1" in indata:
            if np.any(np.isnan(indata["S1"])):
                # if there is a small amount of invalid pixels, we just interpolate it. if the amout is to large, we need to resort to another orbit
                
                S1tensor = torch.tensor(indata["S1"])
                if torch.isnan(S1tensor).sum() / torch.numel(S1tensor) < 0.05 and not self.ascfill:
                    # interpolate the NaN values in the input array using bicubic interpolation, extrapolating if necessary using nearest neighbor if there are not too many NaNs
                    indata["S1"] = self.interpolate_nan(indata["S1"])
                else:
                    # generate another datapatch, but with the ascending orbit
                    indataAsc, _, _ = self.generate_raw_data(xmin, ymin, self.inv_season_dict[season], patchsize=(xmax-xmin, ymax-ymin), overlap=0, admin_overlap=ad_over, descending=False)
                    indata["S1"] = indataAsc["S1"]
                    S1tensor = torch.tensor(indataAsc["S1"])
                    if torch.isnan(S1tensor).sum() / torch.numel(S1tensor) < 0.05:
                        # interpolate the NaN values in the input array using bicubic interpolation, extrapolating if necessary using nearest neighbor if there are not too many NaNs
                        indata["S1"] = self.interpolate_nan(indata["S1"])
                    else:
                        print("S1 contains too many NaNs, skipping")
                        raise Exception("No data here!")
                    
        # get admin_mask
        with rasterio.open(self.file_paths[self.train_level]["boundary"], "r") as src:
            admin_mask = torch.from_numpy(src.read(1, window=w).astype(np.float32))

        # To torch
        indata = {key:torch.from_numpy(np.asarray(val, dtype=np.float32)).type(torch.FloatTensor) for key,val in indata.items()}

        # return dictionary
        return {
                **indata,
                'y': torch.from_numpy(np.asarray(census_sample["POP20"])).type(torch.FloatTensor),
                'admin_mask': admin_mask,
                'img_coords': (xmin,ymin), 'valid_coords':  (xmin, xmax, ymin, ymax),
                'season': self.inv_season_dict[season],# 'season_str': [season],
                "census_idx": torch.tensor([census_sample["idx"]]),
                }


    def __gettestitem__(self, index: int) -> Dict[str, torch.FloatTensor]:
        """
        Description:
            Get the item at the given index, the item is a patch
        Input:
            index: index of the item
        Output:
            item: dictionary containing the input, the mask, the coordinates of the patch, the season and the season string
        """

        # get the indices of the patch
        x,y,season = self.patch_indices[index]

        # get the data
        indata, mask, _ = self.generate_raw_data(x,y,season.item())

        # check for nans and fill it if necessary
        if "S2" in indata:
            if np.any(np.isnan(indata["S2"])): 
                # interpolate the NaN values in the input array using bicubic interpolation, extrapolating if necessary using nearest neighbor if there are not too many NaNs
                indata["S2"] = self.interpolate_nan(indata["S2"]) 
            
        if "S1" in indata:
            if np.any(np.isnan(indata["S1"])):
                S1tensor = torch.tensor(indata["S1"])
                if torch.isnan(S1tensor).sum() / torch.numel(S1tensor) < 0.05 and not self.ascfill:
                    # interpolate the NaN values in the input array using bicubic interpolation, extrapolating if necessary using nearest neighbor if there are not too many NaNs
                    indata["S1"] = self.interpolate_nan(indata["S1"])
                else:
                    # generate another data patch, but with the ascending orbit
                    indataAsc, mask, _ = self.generate_raw_data(x,y,season.item(), descending=False)
                    indata["S1"] = indataAsc["S1"]
                    S1tensor = torch.tensor(indataAsc["S1"])
                    if np.any(np.isnan(indata["S1"])):
                        if torch.isnan(S1tensor).sum() / torch.numel(S1tensor) < 0.05:
                        # interpolate the NaN values in the input array using bicubic interpolation, extrapolating if necessary using nearest neighbor if there are not too many NaNs
                            indata["S1"] = self.interpolate_nan(indata["S1"])
                        else:
                            print("S1 contains too many NaNs, skipping")
                            raise Exception("No data here!")
                        
            if "S2" in indata:
                # check if S1 and S2 have the same shape
                if indata["S1"].shape[1:] != indata["S2"].shape[1:]:
                    print("S1 and S2 have different shapes")

                    raise Exception("different shapes")

        # To Torch
        indata = {key:torch.from_numpy(np.asarray(val, dtype=np.float32)).type(torch.FloatTensor) for key,val in indata.items()} 
        mask = torch.from_numpy(mask).type(torch.FloatTensor)

        # get valid coordinates of the patch
        xmin = x+self.overlap
        xmax = x+self.patchsize-self.overlap
        ymin = y+self.overlap
        ymax = y+self.patchsize-self.overlap

        # return dictionary
        return {
                'img_coords': (x,y), 'valid_coords':  (xmin, xmax, ymin, ymax),
                **indata,
                'season': season.item(), 'mask': mask, 'season_str': self.season_dict[season.item()]}
    

    def interpolate_nan(self, input_array):
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


    def generate_raw_data(self, x, y, season, patchsize=None, overlap=None, admin_overlap=0, descending=True):
        """
        Generate the data for the patch
        Input:
            x: x coordinate of the patch
            y: y coordinate of the patch
            season: season of the patch
            patchsize: size of the patch
            overlap: overlap of the patches
        Output:
            data: data of the patch
        """
        S2_RGB_channels = (3,2,1)
        S2_RGBNIR_channels = (3,2,1,4)
        S1_channels = (1,2)
        
        patchsize_x, patchsize_y, overlap, window = self._setup_patch_parameters(x, y, patchsize, overlap, admin_overlap)

        indata = {}

        # Initialize the mask
        mask = self._create_mask(patchsize_x, patchsize_y, overlap)

        # for debugging
        fake = False
        if fake:
            if self.NIR:
                indata["S2"] = np.random.randint(0, 10000, size=(4,patchsize_x,patchsize_y))
            else:
                indata["S2"] = np.random.randint(0, 10000, size=(3,patchsize_x,patchsize_y))
            indata["S1"] = np.random.randint(0, 10000, size=(2,patchsize_x,patchsize_y))
            indata["building_segmentation"] = np.random.randint(0, 1, size=(1,patchsize_x,patchsize_y))
            indata["building_counts"] = np.random.randint(0, 2, size=(1,patchsize_x,patchsize_y))
            return indata, mask, window

        ### get the input data ###
        # Sentinel 2
        if self.S2:
            S2_file = self.S2_file[season]
            if self.NIR:
                with rasterio.open(S2_file, "r") as src:
                    indata["S2"] = src.read(S2_RGBNIR_channels, window=window).astype(np.float32) 
            else:
                with rasterio.open(S2_file, "r") as src:
                    indata["S2"] = src.read(S2_RGB_channels, window=window).astype(np.float32) 

        # Sentinel 1
        if self.S1:
            S1_file = self.S1_file[season] if descending else self.S1Asc_file[season]
            with rasterio.open(S1_file, "r") as src:
                indata["S1"] = src.read(S1_channels, window=window).astype(np.float32) 

        # building data
        if self.gbuildings:
            if os.path.exists(self.gbuildings_segmentation_file): 
                with rasterio.open(self.gbuildings_segmentation_file, "r") as src:
                    indata["building_segmentation"] = src.read(1, window=window)[np.newaxis].astype(np.float32)
                with rasterio.open(self.gbuildings_counts_file, "r") as src:
                    indata["building_counts"] = src.read(1, window=window)[np.newaxis].astype(np.float32) 
            
            """
            Deprecated: now the sentinel buildings are calculated on the fly, no need to load it
            if self.sentinelbuildings:
                with rasterio.open(self.sbuildings_segmentation_file, "r") as src:
                    indata["building_counts"] = src.read(1, window=window)[np.newaxis].astype(np.float32)/255
            """

        return indata, mask, window
    

    def _setup_patch_parameters(self, x, y, patchsize, overlap, admin_overlap):
        """
        Set up the parameters for the patch.
        Input:
            x: x coordinate of the patch
            y: y coordinate of the patch
            patchsize: size of the patch (tuple or None)
            overlap: overlap of the patches (int or None)
            admin_overlap: additional overlap for administrative regions (int)
        Output:
            patchsize_x: x dimension of the patch
            patchsize_y: y dimension of the patch
            overlap: overlap of the patches
            window: tuple defining the window to read the data
        """
        patchsize_x = self.patchsize if patchsize is None else patchsize[0]
        patchsize_y = self.patchsize if patchsize is None else patchsize[1]
        overlap = self.overlap if overlap is None else overlap

        # Calculate the window for the patch, considering administrative overlap if applicable
        if admin_overlap > 0:
            new_x = max(x - admin_overlap, 0)
            new_y = max(y - admin_overlap, 0)
            x_stop = min(x + patchsize_x + admin_overlap, self.cr_shape[0])
            y_stop = min(y + patchsize_y + admin_overlap, self.cr_shape[1])
            window = ((new_x, x_stop), (new_y, y_stop))
        else:
            window = ((x, x + patchsize_x), (y, y + patchsize_y))

        return patchsize_x, patchsize_y, overlap, window
    

    def _create_mask(self, patchsize_x, patchsize_y, overlap):
        """
        Create a mask for the patch.
        Input:
            patchsize_x: x dimension of the patch
            patchsize_y: y dimension of the patch
            overlap: overlap of the patches
        :return:
            mask: a boolean mask array for the patch
        """
        # Initialize a mask with all False (not in the area of interest)
        mask = np.zeros((patchsize_x, patchsize_y), dtype=bool)

        # Set the area inside the overlap to True (area of interest)
        mask[overlap:patchsize_x - overlap, overlap:patchsize_y - overlap] = True

        return mask


    def convert_popmap_to_census(self, pred, gpu_mode=False, level="fine", details_to=None):
        """
        Converts the predicted population to the census data
        inputs:
            :param pred: predicted population
            :param gpu_mode: if aggregation is done on gpu (can use a bit more GPU memory, but is a lot faster)
        Output:
            :return: the predicted population for each census region
        """

        boundary_file = self.file_paths[level]["boundary"]
        census_file = self.file_paths[level]["census"]

        # raise NotImplementedError
        with rasterio.open(boundary_file, "r") as src:
            boundary = src.read(1)
        boundary = torch.from_numpy(boundary.astype(np.float32))

        # read the census file
        census = pd.read_csv(census_file)

        if gpu_mode: 
            pred = pred.cuda()
            boundary = boundary#.cuda() 

            # initialize more efficient version
            census_pred_i = -torch.ones(len(census), dtype=torch.float32).cuda()
            census_i = -torch.ones(len(census), dtype=torch.float32).cuda()

            # iterate over census regions and get totals
            for i, (cidx,bbox) in tqdm(enumerate(zip(census["idx"], census["bbox"])), total=len(census), disable=False, leave=False):
                if pd.isnull(bbox):
                    continue

                # append the predicted census and the true census
                xmin, xmax, ymin, ymax = tuple(map(int, bbox.strip('()').strip('[]').split(',')))
                census_pred_i[i] = pred[xmin:xmax, ymin:ymax].cuda()[boundary[xmin:xmax, ymin:ymax].cuda()==cidx].to(torch.float32).sum()
                census_i[i] = census["POP20"][i]

        else:

            pred = pred.to(torch.float32)

            census_pred_i = -torch.ones(len(census), dtype=torch.float32)
            census_i = -torch.ones(len(census), dtype=torch.float32)

            # iterate over census regions and get totals
            for i, (cidx,bbox) in enumerate(zip(census["idx"], census["bbox"])):
                xmin, xmax, ymin, ymax = tuple(map(int, bbox.strip('()').strip('[]').split(',')))
                census_pred_i[i] = pred[xmin:xmax, ymin:ymax][boundary[xmin:xmax, ymin:ymax]==cidx].to(torch.float32).sum()
                census_i[i] = census["POP20"][i]

        valid_census_i = census_pred_i>-1
        census_pred_i = census_pred_i[valid_census_i]
        census_i = census_i[valid_census_i]

        if details_to is not None:
            """
            Save the detailed maps
            maps include:
                - densities
                - totals
                - densities_gt
                - totals_gt
                - residuals
                - residuals_rel (relative area residuals)
            """

            # create directory if not exists
            if not os.path.exists(details_to):
                os.makedirs(details_to)

            # produce density map
            densities = torch.zeros_like(pred)
            pred_densities_census = census_pred_i.cpu() / census["count"]
            for i, (cidx,bbox) in enumerate(zip(census["idx"], census["bbox"])):
                xmin, xmax, ymin, ymax = tuple(map(int, bbox.strip('()').strip('[]').split(',')))
                densities[xmin:xmax, ymin:ymax][boundary[xmin:xmax, ymin:ymax]==cidx] = pred_densities_census[i]
            densities = densities.cpu()

            # total map
            totals = torch.zeros_like(pred, dtype=torch.float32)
            totals_pred_census = census_pred_i.cpu().to(torch.float32)
            for i, (cidx,bbox) in enumerate(zip(census["idx"], census["bbox"])):
                xmin, xmax, ymin, ymax = tuple(map(int, bbox.strip('()').strip('[]').split(',')))
                totals[xmin:xmax, ymin:ymax][boundary[xmin:xmax, ymin:ymax]==cidx] = totals_pred_census[i]
            totals = totals.cpu()

            # produce density map for the ground truth
            densities_gt = torch.zeros_like(pred)
            gt_densities_census = torch.tensor(census["POP20"]) / census["count"]
            for i, (cidx,bbox) in enumerate(zip(census["idx"], census["bbox"])):
                xmin, xmax, ymin, ymax = tuple(map(int, bbox.strip('()').strip('[]').split(',')))
                densities_gt[xmin:xmax, ymin:ymax][boundary[xmin:xmax, ymin:ymax]==cidx] = gt_densities_census[i]
            densities_gt = densities_gt.cpu()

            # total map
            totals_gt = torch.zeros_like(pred, dtype=torch.float32)
            totals_gt_census = torch.tensor(census["POP20"]).to(torch.float32)
            for i, (cidx,bbox) in enumerate(zip(census["idx"], census["bbox"])):
                xmin, xmax, ymin, ymax = tuple(map(int, bbox.strip('()').strip('[]').split(',')))
                totals_gt[xmin:xmax, ymin:ymax][boundary[xmin:xmax, ymin:ymax]==cidx] = totals_gt_census[i]
            totals_gt = totals_gt.cpu()

            # residual map
            residuals = torch.zeros_like(pred, dtype=torch.float32) 
            residuals_census = census_pred_i.cpu().to(torch.float32) - torch.tensor(census["POP20"]).to(torch.float32)
            for i, (cidx,bbox) in enumerate(zip(census["idx"], census["bbox"])):
                xmin, xmax, ymin, ymax = tuple(map(int, bbox.strip('()').strip('[]').split(',')))
                residuals[xmin:xmax, ymin:ymax][boundary[xmin:xmax, ymin:ymax]==cidx] = residuals_census[i]
            residuals = residuals.cpu()

            #relaltive residuals
            residuals_rel = torch.zeros_like(pred, dtype=torch.float32) 
            if gpu_mode:
                pop20 = torch.tensor(census["POP20"]).cuda().to(torch.float32)
                pix_count = torch.tensor(census["count"]).cuda().to(torch.float32)
                census_pred = census_pred_i.cuda().to(torch.float32)
            else:
                pop20 = torch.tensor(census["POP20"]).to(torch.float32)
                pix_count = torch.tensor(census["count"]).to(torch.float32)
                census_pred = census_pred_i.to(torch.float32)
            # residuals_rel_census = (census_pred - pop20) / pop20
            residuals_rel_census = (census_pred - pop20) / pix_count
            residuals_rel_census[torch.isinf(residuals_rel_census) | torch.isnan(residuals_rel_census)] = 0

            for i, (cidx,bbox) in enumerate(zip(census["idx"], census["bbox"])):
                xmin, xmax, ymin, ymax = tuple(map(int, bbox.strip('()').strip('[]').split(',')))
                residuals_rel[xmin:xmax, ymin:ymax][boundary[xmin:xmax, ymin:ymax]==cidx] = residuals_rel_census[i]
            residuals_rel = residuals_rel.cpu()

            # save the maps
            print("*"*10)
            print ("saving detailed maps to ", details_to, " folder")
            self.save(densities, details_to, "_densities")
            self.save(totals, details_to, "_totals")
            self.save(densities_gt, details_to, "_densities_gt")
            self.save(totals_gt, details_to, "_totals_gt")
            self.save(residuals, details_to, "_residuals")
            self.save(residuals_rel, details_to, "_residuals_rel")
        
        del boundary, pred
        torch.cuda.empty_cache()

        assert census_pred_i.shape[0] == len(census_i), "census_pred and census have different lengths"
        return census_pred_i, census_i
    

    def adjust_map_to_census(self, pred, gpu_mode=True):
        """
        Adjust the predicted map to the census regions via dasymmetric mapping strategy
        Inputs:
            :param pred: predicted map
            :param census: census data
        Output:
            :return: adjusted map
        """
        
        boundary_file = self.file_paths[self.train_level]["boundary"]
        census_file = self.file_paths[self.train_level]["census"]

        with rasterio.open(boundary_file, "r") as src:
            boundary = src.read(1)
        boundary = torch.from_numpy(boundary.astype(np.float32))

        # read the census file
        census = pd.read_csv(census_file)

        # iterate over census regions and adjust the totals
        for i, (cidx,bbox) in enumerate(zip(census["idx"], census["bbox"])):
            xmin, xmax, ymin, ymax = tuple(map(int, bbox.strip('()').strip('[]').split(',')))
            pred_census_count = pred[xmin:xmax, ymin:ymax][boundary[xmin:xmax, ymin:ymax]==cidx].to(torch.float32).sum()
            if pred_census_count==0:
                continue
            adj_scale = census["POP20"][i] / pred_census_count
            pred[xmin:xmax, ymin:ymax][boundary[xmin:xmax, ymin:ymax]==cidx] *= adj_scale

        return pred


    def save(self, preds, output_folder, tag="") -> None:
        """
        Saves the predictions to a tif file
        Inputs:
            :param preds: the predictions
            :param output_folder: the folder to save the predictions to (will be created if it doesn't exist)
        Output:
            :return: None
        """

        # convert to numpy array
        preds = preds.cpu().numpy()

        # create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # save the predictions
        output_file = os.path.join(output_folder, self.region + f"_predictions{tag}.tif")

        try:
            with rasterio.open(output_file, "w", CHECK_DISK_FREE_SPACE="NO", **self._meta) as dest:
                dest.write(preds,1)
        except:
            print("Error saving predictions to file, continuing...")
            pass



# collate function for the dataloader
def Population_Dataset_collate_fn(batch):
    """
    Collate function for the dataloader used in the Population_Dataset class
    to ensure that all items in the batch have the same shape
    Inputs:
        :param batch: the batch of data with irregular shapes
    Output:
        :return: the batch of data with same shapes
    """
    # Find the maximum dimensions for each item in the batch 

    # Create empty tensors with the maximum dimensions
    use_S2, use_S1 = False, False
    if 'S2' in batch[0]:
        max_x = max([item['S2'].shape[1] for item in batch])
        max_y = max([item['S2'].shape[2] for item in batch])
        input_batch_S2 = torch.zeros(len(batch), batch[0]['S2'].shape[0], max_x, max_y)
        use_S2 = True
    if 'S1' in batch[0]:
        max_x = max([item['S1'].shape[1] for item in batch])
        max_y = max([item['S1'].shape[2] for item in batch])
        input_batch_S1 = torch.zeros(len(batch), batch[0]['S1'].shape[0], max_x, max_y)
        use_S1 = True

    if 'building_counts' in batch[0]:
        max_x = max([item['building_counts'].shape[1] for item in batch])
        max_y = max([item['building_counts'].shape[2] for item in batch])
        building_counts = torch.zeros(len(batch), 1, max_x, max_y)
    
    # initialize flags
    use_building_segmentation, use_building_counts, use_positional_encoding = False, False, False
    
    # initialize tensors
    admin_mask_batch = (-1)*torch.ones(len(batch), max_x, max_y)
    y_batch = torch.zeros(len(batch))
    
    # Fill the tensors with the data from the batch
    for i, item in enumerate(batch):
        if use_S2:
            x_size, y_size = item['S2'].shape[1], item['S2'].shape[2]
            input_batch_S2[i, :, :x_size, :y_size] = item['S2']
        if use_S1:
            x_size, y_size = item['S1'].shape[1], item['S1'].shape[2]
            input_batch_S1[i, :, :x_size, :y_size] = item['S1']

        y_batch[i] = item['y']

        if "building_counts" in item:
            x_size, y_size = item['building_counts'].shape[1], item['building_counts'].shape[2]
            building_counts[i, :, :x_size, :y_size] = item['building_counts']
            use_building_counts = True

        # get the admin_mask
        x_size, y_size = item['admin_mask'].shape[0], item['admin_mask'].shape[1]
        admin_mask_batch[i, :x_size, :y_size] = item['admin_mask']

    out_dict = {
        'admin_mask': admin_mask_batch,
        'y': y_batch,
        'img_coords': [item['img_coords'] for item in batch],
        'valid_coords': [item['valid_coords'] for item in batch],
        'season': torch.tensor([item['season'] for item in batch]),
        # 'source': torch.tensor([item['source'] for item in batch], dtype=torch.bool),
        'census_idx': torch.cat([item['census_idx'] for item in batch]),
    }

    if use_S2:
        out_dict["S2"] = input_batch_S2
    if use_S1:
        out_dict["S1"] = input_batch_S1
    if use_building_counts:
        out_dict["building_counts"] = building_counts

    return out_dict


if __name__=="__main__":

    #test the dataset
    from torch.utils.data import DataLoader, ChainDataset, ConcatDataset

    input_defs = {'S1': True, 'S2': True, 'VIIRS': False, 'NIR': True}

    # Create the dataset for testing
    dataset = Population_Dataset_target("pricp2", mode="weaksup", patchsize=None, overlap=None, fourseasons=True, **input_defs) 
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True, drop_last=True)

    # Test the dataset
    for e in tqdm(range(10), leave=True):
        dataloader_iterator = iter(dataloader)
        for i in tqdm(range(5000)):
            sample = dataset[i%len(dataset)]
            print(i,sample['input'].shape)

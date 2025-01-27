"""
Project: 🍿POPCORN: High-resolution Population Maps Derived from Sentinel-1 and Sentinel-2 🌍🛰️
Nando Metzger, 2024
"""


import argparse
import os
from os.path import isfile   
from osgeo import gdal
from tqdm import tqdm

from tqdm import tqdm

import rasterio
from rasterio import windows
from rasterio.vrt import WarpedVRT

import numpy as np


def process(parent_dir, output_dir, compression=True):
    """
    This function merges all tiffs in parent_dir and saves them in output_dir

    Args:
        parent_dir (str): path to parent directory
        output_dir (str): path to output directory

    Returns:
        None
    """
    name = parent_dir.split("/")[-1]

    # get all subdirectories
    subdirs = [x[0] for x in os.walk(parent_dir)]

    for subdir in subdirs:

        # Sort out fist unusable one
        if subdir==parent_dir:
            continue
        print("Processing files in", subdir)
        
        ty = subdir.split("/")[-1]
        os.makedirs(os.path.join(output_dir, ty), exist_ok=True)

        # Collect files
        files_to_mosaic = [ os.path.join(subdir,file) for file in os.listdir(subdir)]
        
        # Create output name
        output_file = os.path.join(os.path.join(output_dir, ty), name + "_" + ty + ".tif" )

        # Skip if file already exists
        if isfile(output_file):
            print("File already exists, skipping") 

        elif ty in ["S1spring", "S1summer", "S1autumn", "S1winter","S1springAsc", "S1summerAsc", "S1autumnAsc", "S1winterAsc"]:
            print("Merging files to", output_file)
            if isfile(output_file):
                print("File already exists, skipping")
                continue
            # Merge files
            if len(files_to_mosaic) < 16:
                g = gdal.Warp(output_file, files_to_mosaic, format="GTiff", options=["COMPRESS=LZW"])
                
            else:
                if compression:
                    # compress it  Open the GeoTIFF file in read mode 
                    for filename in tqdm(files_to_mosaic):
                        with rasterio.open(filename, "r") as src:
                            profile = src.profile.copy()
                            
                            # check if file is already compressed
                            if profile["compress"] == "lzw" and profile["dtype"] == "float32":
                                print("already compressed")
                                continue

                            profile["compress"] = "lzw"
                            profile["dtype"] = "float32"

                        with rasterio.open(filename, "r") as src:
                            a = src.read()

                        with rasterio.open(filename, "w", **profile) as dst:
                            dst.write(a)


        if ty in ["S2Aspring", "S2Asummer", "S2Aautumn", "S2Awinter"]: 
            if isfile(output_file):
                print("File already exists, skipping")
                continue
            print("Merging files to", output_file)
            # Merge files
            if len(files_to_mosaic) < 16:
                g = gdal.Warp(output_file, files_to_mosaic, format="GTiff", outputType=gdal.GDT_UInt16, options=["COMPRESS=LZW,bandList=[2,3,4,8]"]) 
            else:
                if compression:
                    # Open the GeoTIFF file in read mode 
                    for filename in tqdm(files_to_mosaic):
                        with rasterio.open(filename, "r") as src:
                            profile = src.profile.copy()

                            # check if file is already compressed
                            if profile["compress"] == "lzw" and profile["dtype"] == "uint16": 
                                continue

                            profile["compress"] = "lzw"
                            profile["dtype"] = "uint16"

                        with rasterio.open(filename, "r") as src:
                            a = src.read().astype(np.uint16)

                        with rasterio.open(filename, "w", **profile) as dst:
                            dst.write(a)

    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_dir", default="/scratch/metzgern/HAC/data/PopMapData/raw/EE/pri", type=str, help="")
    parser.add_argument("output_dir", default="/scratch/metzgern/HAC/data/PopMapData/processed/EE/pri", type=str, help="")
    args = parser.parse_args()

    process(args.parent_dir, args.output_dir)


if __name__ == "__main__":
    main()
    print("Done!")



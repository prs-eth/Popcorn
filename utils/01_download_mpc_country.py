"""
Project: 🍿POPCORN: High-resolution Population Maps Derived from Sentinel-1 and Sentinel-2 🌍🛰️
Author: Antigravity (derived from Nando Metzger)
Description: MPC Data Download Pipeline (Alternative to GEE)
"""

import argparse
import os
import sys
import pandas as pd
import xarray as xr
import rioxarray
from odc.stac import load
import pystac_client
import planetary_computer
import rasterio
import datetime
import numpy as np
import time
from tqdm import tqdm
from dask.callbacks import Callback

# Configuration for Sentinel-2
CLOUD_FILTER = 60
S2_BANDS = ["B02", "B03", "B04", "B08"] # Standard 10m bands for POPCORN

class TqdmSpeedBar(Callback):
    """
    Custom Dask callback to show progress with speed (MB/s) in tqdm.
    """
    def __init__(self, total_size_mb, desc="Downloading"):
        self.pbar = tqdm(total=100, desc=desc, unit="%")
        self.total_size_mb = total_size_mb
        self.start_time = None
        self.last_frac = 0

    def _start(self, dsk):
        self.start_time = time.time()

    def _posttask(self, key, value, dsk, state, id):
        # Estimate progress based on finished tasks
        ntasks = len(dsk)
        nfinished = len(state["finished"])
        frac = nfinished / ntasks if ntasks > 0 else 0
        
        if frac - self.last_frac > 0.01 or frac == 1.0:
            elapsed = time.time() - self.start_time
            curr_mb = frac * self.total_size_mb
            speed = curr_mb / elapsed if elapsed > 0 else 0
            
            self.pbar.n = int(frac * 100)
            self.pbar.set_postfix({
                "MB": f"{curr_mb:.1f}/{self.total_size_mb:.1f}",
                "speed": f"{speed:.2f} MB/s"
            })
            self.pbar.refresh()
            self.last_frac = frac

    def _finish(self, dsk, state, errored):
        self.pbar.close()

def get_sentinel2_config(year):
    return {
        'Sen2spring': (f'{year}-03-01', f'{year}-06-01'),
        'Sen2summer': (f'{year}-06-01', f'{year}-09-01'),
        'Sen2autumn': (f'{year}-09-01', f'{year}-12-01'),
        'Sen2winter': (f'{year}-12-01', f'{year+1}-03-01'),
    }

def mask_s2_clouds(ds):
    """
    Mask clouds in Sentinel-2 data using the SCL band.
    """
    if "SCL" not in ds.data_vars:
        return ds
    
    # SCL values: 4: vegetation, 5: bare soil, 6: water, 7: unclassified
    # We exclude 3 (shadow), 8 (cloud medium), 9 (cloud high), 10 (cirrus), 11 (snow)
    mask = (ds.SCL == 4) | (ds.SCL == 5) | (ds.SCL == 6) | (ds.SCL == 7)
    return ds.where(mask)

def download_seasonal_composite(catalog, collection, bbox, start_date, end_date, bands, output_path, is_s2=True):
    """
    Search, process, and save a seasonal median composite.
    """
    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": CLOUD_FILTER}} if is_s2 else None
    )
    
    # Corrected method call (item_collection() instead of get_all_items())
    items = search.item_collection()
    if len(items) == 0:
        print(f"No items found for {collection} between {start_date} and {end_date}")
        return

    print(f"Found {len(items)} items for {collection}")
    
    # Sign URLs for MPC
    signed_items = [planetary_computer.sign(item) for item in items]
    
    # Estimate size in MB
    width = int(abs(bbox[2] - bbox[0]) / 0.0001)
    height = int(abs(bbox[3] - bbox[1]) / 0.0001)
    total_pixels = width * height
    bytes_per_pix = 2 if is_s2 else 4 # uint16 for S2, float32 for S1
    est_size_mb = (total_pixels * len(bands) * bytes_per_pix) / (1024 * 1024)
    
    # Load using odc-stac
    assets = bands + (["SCL"] if is_s2 else [])
    
    try:
        ds = load(
            signed_items,
            bands=assets,
            bbox=bbox,
            crs="EPSG:4326",
            resolution=0.0001,
            chunks={"x": 2048, "y": 2048}, # Larger chunks for country scale
        )
    except Exception as e:
        print(f"Error loading data for {collection}: {e}")
        return
    
    if is_s2:
        ds = mask_s2_clouds(ds)
        ds = ds[bands]
    
    # Compute median across time
    print(f"Processing and saving {output_path}...")
    median_composite = ds.median(dim="time", skipna=True)
    
    # Optimization: Cast Sentinel-2 to uint16 to save memory and space (values 0-10000 usually)
    if is_s2:
        median_composite = median_composite.round().astype("uint16")

    # Use custom callback for progress reporting
    with TqdmSpeedBar(est_size_mb, desc=f"Saving {os.path.basename(output_path)}"):
        # Save as multiband GeoTIFF with compression
        # to_array() merges variables (bands) into a single dimension named 'variable'
        da = median_composite.to_array(dim="band")
        
        # Explicitly pass tiling and compression to rasterio via rioxarray
        da.rio.to_raster(
            output_path, 
            tiled=True, 
            compress="LZW", 
            predictor=2, 
            num_threads="all_cpus"
        )
    
    print(f"Finished saving {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("minx", type=float)
    parser.add_argument("miny", type=float)
    parser.add_argument("maxx", type=float)
    parser.add_argument("maxy", type=float) 
    parser.add_argument("name", type=str) 
    parser.add_argument("--year", type=int, default=2020)
    args = parser.parse_args()

    # bbox = [minx, miny, maxx, maxy]
    bbox = [args.minx, args.miny, args.maxx, args.maxy]
    os.makedirs(args.name, exist_ok=True)
    
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
    )

    config = get_sentinel2_config(args.year)
    seasons = ['spring', 'summer', 'autumn', 'winter']

    for season in seasons:
        season_key = f'Sen2{season}'
        start_date, end_date = config[season_key]
        
        # Sentinel-2 L2A
        s2_filename = os.path.join(args.name, f"S2A_{season}_{args.name}.tif")
        download_seasonal_composite(
            catalog, "sentinel-2-l2a", bbox, start_date, end_date, 
            S2_BANDS, s2_filename, is_s2=True
        )
        
        # Sentinel-1 GRD (VV, VH)
        s1_filename = os.path.join(args.name, f"S1_{season}_{args.name}.tif")
        download_seasonal_composite(
            catalog, "sentinel-1-grd", bbox, start_date, end_date, 
            ["vv", "vh"], s1_filename, is_s2=False
        )

if __name__ == "__main__":
    main()

"""
Project: 🍿POPCORN: High-resolution Population Maps Derived from Sentinel-1 and Sentinel-2 🌍🛰️
Nando Metzger, 2024
"""


import os
import argparse
import torch
import geopandas as gdp
import pandas as pd
import numpy as np
from tqdm import tqdm
from shapely.affinity import translate
import rasterio
from rasterio import features
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.warp import transform_bounds
from rasterio import Affine



def process(hd_regions_path, wp_regions_path,
            census_data_path, output_path, target_col,
            template_file,
            kigali_census_data_path,
            xoff=None, yoff=None, gpu_mode=True):
    """
    Process the shapefile and create a raster file that contains the boundaries of the regions
    and the population count in each region.

    Parameters
    ----------
    hd_regions_path : str
        Path to the shapefile with humdata.org administrative boundaries information
        https://data.humdata.org/dataset/cod-ab-rwa
    wp_regions_path : str
        Path to the shapefile with WorldPop administrative boundaries information
    census_data_path : str
        Path to the CSV file containing the WorldPop ids of the regions and the population counts
    output_path : str
        Output path
    target_col : str
        Target column
    template_file : str
        Template Sentinel-2/1 file that shows the resolution of the output
    kigali_census_data_path : str
        tif containing the population counts for Kigali
        https://zenodo.org/record/7712047
    xoff : float, optional
        x offset, because the WolrdPop grid has a mysterious offset that can be resolved by local shifting
    yoff : float, optional
        y offset, because the WolrdPop grid has a mysterious offset that can be resolved by local shifting
    gpu_mode : bool, optional
        Use GPU mode, by default True
    """
    
    # read the shapefiles    
    hd_regions = gdp.read_file(hd_regions_path)
    hd_regions["idx"] = np.nan
    hd_regions['idx'] = hd_regions['idx'].astype('Int64')    
    wp_regions = gdp.read_file(wp_regions_path)[["adm_id", "geometry"]]
    
    # all_census = read_multiple_targets_from_csv(census_data_path)
    all_census = pd.read_csv(census_data_path)[["ISO","GID", target_col]]
    all_census = all_census.rename(columns={'ISO': 'ISO', 'GID': 'adm_id', target_col: "pop_count"})

    # wp_joined = pd.concat([wp_regions, all_census], axis=1, join="inner")
    wp_joined2 = wp_regions.merge(all_census, on='adm_id', how='inner')
    
    # iterate over the hd_regions and wp_regions and find the intersection
    iou_calc = np.zeros((len(hd_regions), len(wp_joined2)))
    for i,hd_row in tqdm(hd_regions.iterrows(), total=len(hd_regions)):
        hd_geometry = hd_row["geometry"]
        hd_regions.loc[i,"idx"] = int(i)
        if xoff is not None or yoff is not None:
                        
            xoff = 0 if xoff is None else xoff
            yoff = 0 if yoff is None else yoff
            hd_geometry = translate(hd_geometry, xoff=-xoff, yoff=-yoff)

        for j, wp_row in wp_joined2.iterrows():
            wp_geometry = wp_row["geometry"]
            intersection = hd_geometry.intersection(wp_geometry)
            if not intersection.is_empty:
                union = hd_geometry.union(wp_geometry)
                iou_calc[i,j] = intersection.area / union.area

    print("Mean IoU matching score", iou_calc.max(1).mean())
    print("Median IoU matching score", np.median(iou_calc.max(1)))

    iou = iou_calc.copy()
    iou_thresh = 0.66
    iou[iou<iou_thresh] = 0.

    valid_matches = iou.sum(1)>=0.5
    print("Number of valid matches", sum(valid_matches))
    
    # hardening the matches
    iou_argmax = iou.argmax(1)

    # add the population count to the hd_regions
    hd_regions["pop_count"] = hd_regions.apply(lambda row: wp_joined2["pop_count"][iou_argmax[row["idx"]]], axis=1) 
    hd_regions = hd_regions[valid_matches] 

    #(minx, miny, maxx, maxy) as a list
    hd_regions = pd.concat([hd_regions, hd_regions["geometry"].bounds], axis=1)

    # get boundaries on the template file
    # read metadata of the template file
    with rasterio.open(template_file, 'r') as tmp:
        metadata = tmp.meta.copy()
    metadata.update({"count": 1, "dtype": rasterio.int32})
    
    this_outputfile = os.path.join(output_path, 'boundaries_coarse.tif')
    this_outputfile_densities = os.path.join(output_path, 'densities_coarse.tif')
    this_outputfile_totals = os.path.join(output_path, 'totals_coarse.tif')
    this_censusfile = os.path.join(output_path, 'census_coarse.csv')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    hd_regions.reset_index(drop=True, inplace=True)
    hd_regions["idx"] = hd_regions.index+1

    # rasterize
    metadata.update({"compress": "lzw"})
    with rasterio.open(this_outputfile, 'w+', **metadata) as out:
        out_arr = out.read(1)

        # this is where we create a generator of geom, value pairs to use in rasterizing
        shapes = ((geom,i) for i, geom in zip(hd_regions["idx"], hd_regions.geometry))

        # flattens the shapefile into the raster (burns them in)
        burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
        out.write_band(1, burned)

    burned = torch.tensor(burned, dtype=torch.int32)
    if gpu_mode:
        burned = burned.cuda()

    hd_regions["bbox"] = ""
    hd_regions["count"] = 0

    # get the bounding box and count of each region to enrich the data, also add the count to the dataframe  
    for rowi, row in enumerate(tqdm(hd_regions.itertuples(), total=len(hd_regions))):
        i = row.idx
        mask = burned==i
        count = mask.sum().cpu().item()
        if count==0:
            xmin, ymax, ymin, ymax = 0, 0, 0, 0
        else:
            vertical_indices = torch.where(torch.any(mask, dim=1))[0]
            horizontal_indices = torch.where(torch.any(mask, dim=0))[0]
            xmin, xmax = vertical_indices[[0,-1]].cpu()
            ymin, ymax = horizontal_indices[[0,-1]].cpu()
            xmax, ymax = xmax+1, ymax+1
            xmin, xmax, ymin, ymax = xmin.item(), xmax.item(), ymin.item(), ymax.item()

        hd_regions.at[rowi, "bbox"] = [xmin, xmax, ymin, ymax]
        hd_regions.loc[rowi, "count"] = count

    hd_regions["POP20"] = hd_regions["pop_count"]
    hd_regions[["idx", "POP20", "bbox", "count"]].to_csv(this_censusfile)

    # create map of densities
    densities = torch.zeros_like(burned, dtype=torch.float32)
    totals = torch.zeros_like(burned, dtype=torch.float32)
    for row in hd_regions.itertuples():
        densities[burned==row.idx] = row.pop_count/row.count
        totals[burned==row.idx] = row.pop_count

    if burned.is_cuda:
        burned = burned.cpu()
        densities = densities.cpu()
        totals = totals.cpu()

    #save densities
    metadatad = metadata.copy()
    metadatad.update({"dtype": rasterio.float32, "compress": "lzw"})
    with rasterio.open(this_outputfile_densities, 'w+', **metadatad) as out:
        out.write_band(1, densities.numpy())

    #save totals
    metadatad = metadata.copy()
    metadatad.update({"dtype": rasterio.float32, "compress": "lzw"})
    with rasterio.open(this_outputfile_totals, 'w+', **metadatad) as out:
        out.write_band(1, totals.numpy())

    ##############################
    # now process the fine census data
    # read kigali census data

    for res in [100,200,400,500,1000]:

        with rasterio.open(kigali_census_data_path, 'r') as src:
            kigali_census_map = src.read(1)
            kigali_census_meta = src.meta.copy()

        this_outputfile = os.path.join(output_path, 'boundaries_kigali{}.tif'.format(res))
        this_censusfile = os.path.join(output_path, 'census_kigali{}.csv'.format(res))
        tmp_path = os.path.join(output_path, "tmp.tif")

        # pool the image to the desired resolution
        scale = int(res/100)
        if scale!=1:
            # pool the image
            fitted = kigali_census_map[:kigali_census_map.shape[0]//scale*scale, :kigali_census_map.shape[1]//scale*scale]
            reshaped = fitted.reshape(fitted.shape[0]//scale, scale, fitted.shape[1]//scale, scale)
            sum_pooled = reshaped.sum(axis=(1,3)) 

            # update the metadata
            kigali_census_meta.update({
                "height": sum_pooled.shape[0], "width": sum_pooled.shape[1],
                "transform": Affine(kigali_census_meta["transform"].a*scale, kigali_census_meta["transform"].b, kigali_census_meta["transform"].c,
                                    kigali_census_meta["transform"].d, kigali_census_meta["transform"].e*scale, kigali_census_meta["transform"].f)})
            
            kigali_census_map_pooled = sum_pooled
        else:
            kigali_census_map_pooled = kigali_census_map

        # create indices for the census map by unrolling the map into a vector (ignoring nan values) and creating a census list out of it
        validmask = ~np.isnan(kigali_census_map_pooled)
        kigali_census = kigali_census_map_pooled[validmask]
        kigali_census_ids = np.arange(1, len(kigali_census)+1)
        kigali_census_map_ids = np.zeros_like(kigali_census_map_pooled)
        kigali_census_map_ids[validmask] = kigali_census_ids
        
        # update the metadata for the temporary file
        tmp_meta = kigali_census_meta.copy()
        tmp_meta.update({"count": 1, "dtype": rasterio.int32, "nodata": 0, "compress": "lzw"})

        # write censusmap_ids to temporary file
        with rasterio.open(tmp_path, 'w+', **tmp_meta) as out:
            out.write_band(1, kigali_census_map_ids)

        # Reproject the coarse resolution raster to the fine resolution raster
        with rasterio.open(tmp_path) as src:
            # Open the fine resolution raster
            with rasterio.open(template_file) as dst:
                
                # Transform the bounds of the destination raster to the source crs
                dst_bounds = transform_bounds(dst.crs, src.crs, *dst.bounds)

                # Calculate the transformation matrix from the source (coarse) to destination (fine) crs
                transform, width, height = calculate_default_transform(src.crs, dst.crs, src.width, src.height, *dst_bounds,
                                                                        dst_width=dst.width, dst_height=dst.height)

                # Create a new dataset to store the reprojected raster
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': dst.crs,
                    # 'transform': transform,
                    'transform': dst.transform,
                    'width': width,
                    'height': height,
                    'compress': 'lzw'
                })

                # Write the reprojected raster to the new dataset
                with rasterio.open(this_outputfile, 'w', **kwargs) as reproj:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(reproj, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=dst.crs,
                            resampling=Resampling.nearest
                        )

        # read the reprojected raster
        with rasterio.open(this_outputfile, 'r') as src:
            kigali_census_map_ids_fine = src.read(1)
            kigali_census_meta = src.meta.copy()
        check = False
        if check:
            # check if it is the same as the template file
            with rasterio.open(template_file) as dst:
                meta_template = dst.meta.copy() 
            assert kigali_census_meta["width"]==meta_template["width"]
            assert kigali_census_meta["height"]==meta_template["height"]
            assert kigali_census_meta["transform"]==meta_template["transform"]
            assert kigali_census_meta["crs"]==meta_template["crs"]
        
        # turn on the gpu mode
        if gpu_mode:
            kigali_census_map_ids_fine = torch.tensor(kigali_census_map_ids_fine, dtype=torch.int32).cuda()
        else:
            kigali_census_map_ids_fine = torch.tensor(kigali_census_map_ids_fine, dtype=torch.int32)
        
        # create pandas dataframe
        thisdb = pd.DataFrame({"idx": kigali_census_ids, "POP20": kigali_census})
        thisdb["bbox"] = ""
        thisdb["count"] = 0
        
        # this might take a while, from 15 minutes up to 1.5 hours if not done on GPU
        # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 72976/72976 [1:34:32<00:00, 12.87it/s]
        # skip this for loop if the census file already exists
        if os.path.exists(this_censusfile):
            thisdb = pd.read_csv(this_censusfile)
        else:
            for rowi, row in enumerate(tqdm(thisdb.itertuples(), total=len(thisdb))):
                i = row.idx
                mask = kigali_census_map_ids_fine==i
                count = mask.sum().cpu().item()
                if count==0:
                    xmin, ymax, ymin, ymax = 0, 0, 0, 0
                else:
                    # get the bounding box of the mask by finding the first and last indices where the mask is True
                    vertical_indices = torch.where(torch.any(mask, dim=1))[0]
                    horizontal_indices = torch.where(torch.any(mask, dim=0))[0]
                    xmin, xmax = vertical_indices[[0,-1]].cpu()
                    ymin, ymax = horizontal_indices[[0,-1]].cpu()
                    xmax, ymax = xmax+1, ymax+1 # add 1 to xmax and ymax to make it inclusive
                    xmin, xmax, ymin, ymax = xmin.item(), xmax.item(), ymin.item(), ymax.item()

                thisdb.at[rowi, "bbox"] = [xmin, xmax, ymin, ymax]
                thisdb.loc[rowi, "count"] = count        
            
            thisdb[["idx", "POP20", "bbox", "count"]].to_csv(this_censusfile)

        if kigali_census_map_ids_fine.is_cuda:
            kigali_census_map_ids_fine = kigali_census_map_ids_fine.cpu()

        print("Done with Kigali resolution", res)
    print("Done with Kigali")

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hd_regions_path", type=str, help="Shapefile with humdata.org administrative boundaries information https://data.humdata.org/dataset/cod-ab-rwa")
    parser.add_argument("--wp_regions_path", type=str, help="Shapefile with WorldPop administrative boundaries information")
    parser.add_argument("--census_data_path", type=str, help="CSV file containing the WorldPop ids of the regions and the population counts")
    parser.add_argument("--kigali_census_data_path", type=str, help="tif containing the population counts for Kigali https://zenodo.org/record/7712047")
    parser.add_argument("--output_path", type=str, help="Output path")
    parser.add_argument("--target_col", type=str, help="Target column")
    parser.add_argument("--template_file", type=str, help="template Sentinel-2/1 file that shows the resolution of the output")
    args = parser.parse_args()

    process(args.hd_regions_path, args.wp_regions_path,
                    args.census_data_path, args.output_path,
                    args.target_col,
                    args.template_file,
                    args.kigali_census_data_path,
                    # yoff=-0.)
                    yoff=-0.0026)


if __name__ == "__main__":
    main()
    print("Done!")



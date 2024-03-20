"""
Code is adapted from https://github.com/SebastianHafner/DDA_UrbanExtraction
Modified: Arno Rüegg, Nando Metzger
"""
import rasterio
import json
from pathlib import Path
import numpy as np


# reading in geotiff file as numpy array
def read_tif(file: Path):

    if not file.exists():
        raise FileNotFoundError(f'File {file} not found')

    with rasterio.open(file) as dataset:
        arr = dataset.read()  # (bands X height X width)
        transform = dataset.transform
        crs = dataset.crs

    return arr.transpose((1, 2, 0)), transform, crs


# writing an array to a geo tiff file
def write_tif(file: Path, arr, transform, crs):

    if not file.parent.exists():
        file.parent.mkdir()

    if len(arr.shape) == 3:
        height, width, bands = arr.shape
    else:
        height, width = arr.shape
        bands = 1
        arr = arr[:, :, None]
    with rasterio.open(
            file,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=bands,
            dtype=arr.dtype,
            crs=crs,
            transform=transform,
    ) as dst:
        for i in range(bands):
            dst.write(arr[:, :, i], i + 1)


def get_coords(file: Path):
    file_parts = file.stem.split('_')
    coord_part = file_parts[-1]
    coords = coord_part.split('-')
    coords = [int(coord) for coord in coords]
    return coords


def basename_from_file(file: Path):
    file_parts = file.stem.split('_')
    base_parts = file_parts[:-1]
    base_name = '_'.join(base_parts)
    return base_name


def id2yx(patch_id: str) -> tuple:
    y, x = patch_id.split('-')
    return int(y), int(x)


def combine_tif_patches(folder: Path, basename: str, delete_tiles: bool = False, dtype=np.int8):
    files = [f for f in folder.glob('**/*') if f.is_file() and basename_from_file(f) == basename]
    coords = [get_coords(f) for f in files]
    i_coords = [coord[0] for coord in coords]
    j_coords = [coord[1] for coord in coords]

    max_i = max(i_coords)
    max_j = max(j_coords)

    # TODO: handle if upper left file does not exist (random file can be chosen)
    ul_file = folder / f'{basename}_{0:010d}-{0:010d}.tif'
    ul_arr, transform, crs = read_tif(ul_file)
    tile_height, tile_width, n_bands = ul_arr.shape
    assert (tile_height == tile_width)
    tile_size = tile_height

    lr_file = folder / f'{basename}_{max_i:010d}-{max_j:010d}.tif'
    lr_arr, _, _ = read_tif(lr_file)
    lr_height, lr_width, _ = lr_arr.shape

    mosaic_height = max_i + lr_height
    mosaic_width = max_j + lr_width
    mosaic = np.full((mosaic_height, mosaic_width, n_bands), fill_value=-1, dtype=dtype)

    for index, file in enumerate(files):
        tile, _, _ = read_tif(file)
        i_start, j_start = get_coords(file)
        i_end = i_start + tile_size
        j_end = j_start + tile_size
        mosaic[i_start:i_end, j_start:j_end, ] = tile
        if delete_tiles:
            file.unlink()

    output_file = folder / f'{basename}.tif'
    write_tif(output_file, mosaic, transform, crs)


def load_json(file: Path):
    with open(str(file)) as f:
        d = json.load(f)
    return d


def write_json(file: Path, data):
    with open(str(file), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

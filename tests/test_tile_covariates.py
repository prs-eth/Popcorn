"""
Checks for tools/tile_covariates.py on tiny rasters laid out like the Popcorn release
(uncompressed, 1-row strips; S1 float64 with NaNs, S2 uint16). Run from the Popcorn root:
    python -m pytest tests/test_tile_covariates.py
"""

import json
import os
import sys

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import tile_covariates as tc  # noqa: E402

H, W = 300, 520


def _write_stripped(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prof = {"driver": "GTiff", "height": H, "width": W, "count": arr.shape[0], "dtype": arr.dtype.name,
            "crs": "EPSG:4326", "transform": from_origin(29.0, -1.0, 9e-5, 9e-5),
            "blockysize": 1, "interleave": "pixel"}
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr)


def _make_release(root):
    rng = np.random.default_rng(0)
    merged = os.path.join(root, "PopMapData", "merged", "EE", "rwa")
    s1 = rng.normal(-12, 4, size=(2, H, W)) + rng.normal(0, 1e-9, size=(2, H, W))  # float64 detail
    s1[0, 10:20, 30:50] = np.nan
    s2 = rng.integers(0, 10000, size=(4, H, W), dtype=np.uint16)
    for season in tc.SEASONS:
        _write_stripped(os.path.join(merged, "S1" + season, f"rwa_S1{season}.tif"), s1)
        _write_stripped(os.path.join(merged, "S2A" + season, f"rwa_S2A{season}.tif"), s2)
    proc = os.path.join(root, "PopMapData", "processed", "rwa")
    os.makedirs(proc)
    with open(os.path.join(proc, "census_coarse.csv"), "w") as f:
        f.write(",idx,POP20,bbox,count\n0,1,10.0,\"[0, 10, 0, 10]\",100\n")


def _popcorn_read(path, bands, window):
    # the read in Population_Dataset.generate_raw_data
    with rasterio.open(path) as src:
        return src.read(bands, window=window).astype(np.float32)


def test_copies_read_identically(tmp_path, monkeypatch):
    src, dst = str(tmp_path / "src"), str(tmp_path / "dst")
    _make_release(src)
    monkeypatch.setattr(sys, "argv", ["tile_covariates.py", "--src", src, "--dst", dst, "--region", "rwa"])
    tc.main()

    window = ((5, 290), (17, 511))
    for season in tc.SEASONS:
        for folder, bands in (("S1" + season, (1, 2)), ("S2A" + season, (3, 2, 1, 4))):
            rel = os.path.join("PopMapData", "merged", "EE", "rwa", folder, f"rwa_{folder}.tif")
            a = _popcorn_read(os.path.join(src, rel), bands, window)
            b = _popcorn_read(os.path.join(dst, rel), bands, window)
            assert np.array_equal(a, b, equal_nan=True), folder
            with rasterio.open(os.path.join(dst, rel)) as cp:
                assert cp.block_shapes[0] == (256, 256)
                assert cp.profile["compress"] == "deflate"
                assert cp.dtypes[0] == ("float32" if folder.startswith("S1") else "uint16")
            with open(os.path.join(dst, rel) + ".verify.json") as f:
                rec = json.load(f)
            assert rec["identical"]
            if folder.startswith("S1"):
                assert rec["nan_src"] == rec["nan_copy"] == 200

    link = os.path.join(dst, "PopMapData", "processed", "rwa", "census_coarse.csv")
    assert os.path.islink(link)
    assert os.path.realpath(link) == os.path.realpath(os.path.join(src, "PopMapData", "processed", "rwa", "census_coarse.csv"))


def test_resume_skips_verified_copies(tmp_path, monkeypatch, capsys):
    src, dst = str(tmp_path / "src"), str(tmp_path / "dst")
    _make_release(src)
    monkeypatch.setattr(sys, "argv", ["tile_covariates.py", "--src", src, "--dst", dst, "--only", "S1spring"])
    tc.main()
    capsys.readouterr()
    tc.main()
    assert "S1spring: already done, skipping" in capsys.readouterr().out

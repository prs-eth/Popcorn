"""
Fast-to-read copies of Popcorn's merged S1/S2 covariates (see "Faster data loading" in README.md).

The released merged composites (PopMapData/merged/EE/<region>/S1<season>/..., S2A<season>/...)
are uncompressed GeoTIFFs with 1-row strips (S1 as float64). Popcorn reads one window per
census unit; with 1-row strips GDAL fetches whole raster rows, so a ~1000 x 1000 px window of
Rwanda pulls ~400 MB (S1) + ~200 MB (S2) from disk. This tool writes 256 x 256 tiled,
DEFLATE-compressed copies (predictor 2 for integers, 3 for floats) that Popcorn reads unchanged
once the data root in utils/constants.py points at the destination:

    <dst>/PopMapData/merged/EE/<region>/<S1|S2A><season>/<region>_<...>.tif   tiled copies
    <dst>/PopMapData/processed/<region>/*                                      symlinks to <src>

S1 is stored as float32. Popcorn casts every read with .astype(np.float32), and this tool casts
with the same call, so the values Popcorn sees are bit-identical. Each copy is verified before it
is published: a SHA-256 of the cast values taken while writing must equal a SHA-256 of the
values read back from the finished copy. A <file>.verify.json records both. Resumable: copies
with a matching verify.json are skipped. Source files are only read.

    python tools/tile_covariates.py --src <dir with PopMapData> --dst <new dir> --region rwa
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np
import rasterio
from rasterio.windows import Window

SEASONS = ["spring", "summer", "autumn", "winter"]


def tile_copy(src_path, dst_path, out_dtype=None, rows=512):
    """Write a tiled DEFLATE copy of src_path at dst_path; return the verification record."""
    t0 = time.time()
    tmp = dst_path + ".part"
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    h_src, h_dst = hashlib.sha256(), hashlib.sha256()
    nan_src = nan_dst = 0
    with rasterio.open(src_path) as src:
        src_dtype = src.dtypes[0]
        out_dtype = np.dtype(out_dtype or src_dtype)
        prof = src.profile.copy()
        prof.update(driver="GTiff", dtype=out_dtype.name, tiled=True, blockxsize=256, blockysize=256,
                    compress="deflate", predictor=3 if out_dtype.kind == "f" else 2, zlevel=6,
                    interleave="pixel", BIGTIFF="IF_SAFER", num_threads="ALL_CPUS")
        with rasterio.open(tmp, "w", **prof) as dst:
            for r0 in range(0, src.height, rows):
                w = Window(0, r0, src.width, min(rows, src.height - r0))
                x = src.read(window=w).astype(out_dtype)
                if out_dtype.kind == "f":
                    nan_src += int(np.isnan(x).sum())
                h_src.update(np.ascontiguousarray(x).tobytes())
                dst.write(x, window=w)
                if (r0 // rows) % 8 == 0:
                    print(f"  write {r0 + w.height}/{src.height} rows {time.time() - t0:.0f}s", flush=True)
            dst.update_tags(**src.tags())
    t_write = time.time() - t0

    t1 = time.time()
    with rasterio.open(src_path) as src, rasterio.open(tmp) as dst:
        if (src.height, src.width, src.count) != (dst.height, dst.width, dst.count):
            raise SystemExit(f"shape mismatch for {dst_path}")
        if src.transform != dst.transform or src.crs != dst.crs:
            raise SystemExit(f"georeferencing mismatch for {dst_path}")
        for r0 in range(0, dst.height, rows):
            w = Window(0, r0, dst.width, min(rows, dst.height - r0))
            y = dst.read(window=w)
            if out_dtype.kind == "f":
                nan_dst += int(np.isnan(y).sum())
            h_dst.update(np.ascontiguousarray(y).tobytes())
    rec = {"src": os.path.abspath(src_path), "dst": os.path.abspath(dst_path),
           "src_dtype": src_dtype, "out_dtype": out_dtype.name,
           "layout": f"256x256 tiles, pixel-interleaved, DEFLATE zlevel 6, predictor {prof['predictor']}",
           "write_s": round(t_write, 1), "verify_s": round(time.time() - t1, 1),
           "sha256_values_src_cast": h_src.hexdigest(), "sha256_values_copy": h_dst.hexdigest(),
           "nan_src": nan_src, "nan_copy": nan_dst,
           "identical": h_src.hexdigest() == h_dst.hexdigest()}
    if not rec["identical"]:
        print(json.dumps(rec, indent=1))
        raise SystemExit(f"MISMATCH: {tmp} differs from {src_path}; not publishing it")
    os.replace(tmp, dst_path)
    rec["src_bytes"] = os.path.getsize(src_path)
    rec["dst_bytes"] = os.path.getsize(dst_path)
    with open(dst_path + ".verify.json", "w") as f:
        json.dump(rec, f, indent=1)
    return rec


def is_done(src_path, dst_path):
    vj = dst_path + ".verify.json"
    if not (os.path.exists(dst_path) and os.path.exists(vj)):
        return False
    with open(vj) as f:
        rec = json.load(f)
    return (rec.get("identical") and rec.get("dst_bytes") == os.path.getsize(dst_path)
            and rec.get("src_bytes") == os.path.getsize(src_path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder that contains the original PopMapData/")
    ap.add_argument("--dst", required=True, help="new folder; gets PopMapData/ with the copies")
    ap.add_argument("--region", default="rwa")
    ap.add_argument("--only", nargs="*", default=None, help="subset of folders, e.g. S1spring S2Aspring")
    a = ap.parse_args()

    src_merged = os.path.join(a.src, "PopMapData", "merged", "EE", a.region)
    dst_merged = os.path.join(a.dst, "PopMapData", "merged", "EE", a.region)
    jobs = [("S1" + s, "float32") for s in SEASONS] + [("S2A" + s, None) for s in SEASONS]
    if a.only:
        jobs = [j for j in jobs if j[0] in a.only]

    # processed/<region>: symlink each file (a directory link would let writes reach the source)
    src_proc = os.path.join(a.src, "PopMapData", "processed", a.region)
    dst_proc = os.path.join(a.dst, "PopMapData", "processed", a.region)
    os.makedirs(dst_proc, exist_ok=True)
    for name in sorted(os.listdir(src_proc)):
        link = os.path.join(dst_proc, name)
        if not os.path.lexists(link):
            os.symlink(os.path.join(os.path.abspath(src_proc), name), link)

    for folder, dtype in jobs:
        fname = f"{a.region}_{folder}.tif"
        sp, dp = os.path.join(src_merged, folder, fname), os.path.join(dst_merged, folder, fname)
        if not os.path.exists(sp):
            print(f"{folder}: no source file {sp}, skipping", flush=True)
            continue
        if is_done(sp, dp):
            print(f"{folder}: already done, skipping", flush=True)
            continue
        print(f"{folder}: {sp} -> {dp}", flush=True)
        rec = tile_copy(sp, dp, out_dtype=dtype)
        print(f"{folder}: done, {rec['src_bytes'] / 1e9:.2f} GB -> {rec['dst_bytes'] / 1e9:.2f} GB, "
              f"write {rec['write_s']} s, verify {rec['verify_s']} s, identical={rec['identical']}", flush=True)
    print("all done", flush=True)


if __name__ == "__main__":
    main()

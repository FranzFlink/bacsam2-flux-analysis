#!/usr/bin/env python3
"""
build_modis_ist.py
==================
MODIS/Aqua ice surface temperature of the campaign days on a regular latitude-longitude
grid: the shipped file
``data/observations/processed/satellite_unified/MODIS_Aqua_Sea_Ice_Extent_and_IST_Daily_L3_Global_4km_EASE-Grid_Day_V061.nc``.

Product: MODIS/Aqua Sea Ice Extent and IST Daily L3 Global 4km EASE-Grid Day, version 61
(MYD29E1D, NSIDC, doi:10.5067/MODIS/MYD29E1D.061), one HDF4 granule per day, 12-24 April 2024
(``GRANULES``). The granules are downloaded from NSIDC with an Earthdata login. The script runs
on the raw-data host and needs an HDF4 reader, for example in a temporary environment:

    pixi exec --spec python=3.12 --spec pyhdf --spec rioxarray --spec netcdf4 \\
        python preprocessing/build_modis_ist.py <granule folder> <output file>

Per day, the field ``Ice_Surface_Temperature_NP`` of the 4 km EASE-Grid North is reprojected by
nearest neighbour onto a global WGS 84 grid of 0.0632 degrees (the grid GDAL proposes for this
reprojection), cut to the campaign region (74-85.5 N, 19 W-29 E) and converted to kelvin (scale
factor 0.01). Every value outside the valid range of the product (223.2-313.2 K) is a flag
(missing, no decision, land, inland water, cloud, fill) and becomes NaN. No other correction is
applied.

    python preprocessing/build_modis_ist.py --check <MOD29E1D granule> <YYYY-MM-DD> <earlier file>

repeats the steps for a MOD29E1D (Terra) granule and compares the result with a time step of the
file shipped before 2026-10-28. That file was built in the same way from MOD29E1D GeoTIFFs of the
same grid, ordered on 23 April 2024, and therefore had no data for 16, 17, 19, 20 and 23 April.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from affine import Affine
from pyhdf.SD import SD, SDC
from rasterio.crs import CRS
from rasterio.enums import Resampling

GRANULES = [
    "MYD29E1D.A2024103.061.2024104163724.hdf",   # 12 April 2024
    "MYD29E1D.A2024104.061.2024105162029.hdf",   # 13 April (flight day)
    "MYD29E1D.A2024105.061.2024106165053.hdf",
    "MYD29E1D.A2024106.061.2024107171009.hdf",   # 15 April (flight day)
    "MYD29E1D.A2024107.061.2024115072936.hdf",
    "MYD29E1D.A2024108.061.2024115065805.hdf",   # 17 April (flight day)
    "MYD29E1D.A2024109.061.2024110163639.hdf",
    "MYD29E1D.A2024110.061.2024115061742.hdf",
    "MYD29E1D.A2024111.061.2024115070213.hdf",
    "MYD29E1D.A2024112.061.2024113162007.hdf",
    "MYD29E1D.A2024113.061.2024115120807.hdf",
    "MYD29E1D.A2024114.061.2024116012201.hdf",   # 23 April (flight day)
    "MYD29E1D.A2024115.061.2024117015829.hdf",   # 24 April
]

FIELD = "Ice_Surface_Temperature_NP"
VALID_COUNTS = (22320, 31320)       # valid range of the stored counts (223.2-313.2 K)
SCALE = 0.01                        # K per count
FILL = 7                            # tile fill

# 4 km EASE-Grid North of the product (StructMetadata.0 of the granules: Lambert azimuthal
# equal-area, sphere of radius 6371228 m, 4501 x 4501 cells, upper-left corner below)
EASE_CRS = CRS.from_proj4("+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +R=6371228 +units=m +no_defs")
EASE_N, EASE_CORNER = 4501, 9026314.402
EASE_CELL = 2 * EASE_CORNER / EASE_N   # 4010.804 m
EASE_TRANSFORM = Affine(EASE_CELL, 0.0, -EASE_CORNER, 0.0, -EASE_CELL, EASE_CORNER)

# target: the global WGS 84 grid GDAL proposes for this source grid, cut to the campaign region
DST_TRANSFORM = Affine(0.06323555243281223, 0.0, -180.0, 0.0, -0.0632303894801017, 90.0)
DST_SHAPE = (2846, 5693)
LAT_MIN, LAT_MAX = 74.0, 85.5
LON_MIN, LON_MAX = -19.0, 29.0


def granule_date(path: Path) -> pd.Timestamp:
    """UTC day of a granule from its name (``.AYYYYDDD.``)."""
    stamp = path.name.split(".")[1]
    return pd.Timestamp(year=int(stamp[1:5]), month=1, day=1) + pd.Timedelta(days=int(stamp[5:8]) - 1)


def regional_ist(path: Path) -> xr.DataArray:
    """Ice surface temperature [K] of one granule on the regional latitude-longitude grid."""
    counts = SD(str(path), SDC.READ).select(FIELD).get()
    centres = (np.arange(EASE_N) + 0.5) * EASE_CELL
    src = xr.DataArray(counts, dims=("y", "x"),
                       coords={"y": EASE_CORNER - centres, "x": -EASE_CORNER + centres})
    src = src.rio.write_crs(EASE_CRS).rio.write_transform(EASE_TRANSFORM).rio.write_nodata(FILL)
    dst = src.rio.reproject("EPSG:4326", transform=DST_TRANSFORM, shape=DST_SHAPE,
                            resampling=Resampling.nearest)
    dst = dst.rename({"y": "lat", "x": "lon"})
    dst = dst.sel(lat=slice(LAT_MAX, LAT_MIN), lon=slice(LON_MIN, LON_MAX))
    raw = dst.values
    valid = (raw >= VALID_COUNTS[0]) & (raw <= VALID_COUNTS[1])
    # single precision, as CF decoding of the scale factor does
    ist = np.where(valid, raw.astype(np.float32) * np.float32(SCALE), np.float32(np.nan))
    return xr.DataArray(ist, dims=("lat", "lon"), coords={"lat": dst["lat"].values, "lon": dst["lon"].values})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(folder: Path, out: Path) -> None:
    paths = [folder / name for name in GRANULES]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        sys.exit(f"missing granules: {missing}")
    days = [granule_date(p) for p in paths]
    fields = [regional_ist(p) for p in paths]
    ist = xr.concat(fields, dim=pd.Index(days, name="time"))
    ist.attrs = {"long_name": "MODIS/Aqua ice surface temperature (daily, daytime)", "units": "K",
                 "comment": "flag values of the product (outside 223.2-313.2 K) are NaN"}
    ds = xr.Dataset({"IST": ist})
    ds["lat"].attrs = {"long_name": "latitude", "units": "degrees_north"}
    ds["lon"].attrs = {"long_name": "longitude", "units": "degrees_east"}
    ds.attrs = {
        "title": "MODIS/Aqua Sea Ice Extent and IST Daily L3 Global 4km EASE-Grid Day, version 61, "
                 "campaign region and days",
        "source": "MYD29E1D v061, NSIDC DAAC, doi:10.5067/MODIS/MYD29E1D.061",
        "granules": " ".join(GRANULES),
        "granule_sha256": " ".join(sha256(p) for p in paths),
        "processing": "field Ice_Surface_Temperature_NP of the 4 km EASE-Grid North reprojected by nearest "
                      "neighbour onto a global WGS 84 grid of 0.0632 degrees, cut to 74-85.5 N, 19 W-29 E, "
                      "counts x 0.01 = K, flag values set to NaN (preprocessing/build_modis_ist.py)",
    }
    enc = {"IST": {"zlib": True, "complevel": 4, "shuffle": True, "_FillValue": np.float32(np.nan)}}
    ds.to_netcdf(out, encoding=enc)
    print(f"wrote {out}: {len(days)} days {days[0].date()} .. {days[-1].date()}, grid {ist.shape[1:]}")


def check(granule: Path, day: str, earlier: Path) -> None:
    new = regional_ist(granule)
    old = xr.open_dataset(earlier)["IST"].sel(time=day)
    old = old.sel(lat=new["lat"], lon=new["lon"], method="nearest", tolerance=1e-9)
    same = np.array_equal(new.values, old.values.astype(np.float32), equal_nan=True)
    print(f"{granule.name} vs {earlier.name} at {day}: grid {new.shape}, "
          f"valid {int(np.isfinite(new.values).sum())} / {int(np.isfinite(old.values).sum())}, identical: {same}")
    if not same:
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", nargs=3, metavar=("GRANULE", "DATE", "EARLIER_FILE"))
    ap.add_argument("folder", nargs="?", type=Path)
    ap.add_argument("out", nargs="?", type=Path)
    a = ap.parse_args()
    if a.check:
        check(Path(a.check[0]), a.check[1], Path(a.check[2]))
    else:
        build(a.folder, a.out)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_reviewer_subsets.py
========================
Build the small, reusable Tier-2 data subsets that let every figure/table
notebook run without the full raw archives (~300 GB → tens of MB).

Run on the raw-data host (barat):

    python scripts/make_reviewer_subsets.py

Subsets are written under ``data_subsets/`` mirroring the ``data/`` tree with
IDENTICAL relative paths and variable structure, so notebooks run unchanged
once the subset tree is copied over ``data/`` on a light checkout.

Subset criteria (derived from how the notebooks actually read the files):
  * time-series files (noseboom, radiation): merged per-leg / per-flight-day
    windows from the analysis-ready leg metadata
  * gridded files: campaign-region bounding box, campaign period, and only
    the variables/levels the notebooks and the precompute use
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUT  = REPO / "data_subsets"

AR_META = DATA / "observations" / "ar" / "analysis_ready_polar6_legs_metadata.csv"

# Campaign region (leg bbox ± margin) and period
LAT_MIN, LAT_MAX = 74.0, 87.0
LON_MIN, LON_MAX = -19.0, 29.0
PERIOD = slice("2024-04-05", "2024-05-02")

ENC_FLOAT = {"zlib": True, "complevel": 4}


def _log(msg: str) -> None:
    print(f"[subsets] {msg}", flush=True)


def _out_path(rel: str) -> Path:
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _write(ds: xr.Dataset, rel: str, f32: bool = False) -> None:
    if f32:
        ds = ds.copy()
        for v in ds.data_vars:
            if ds[v].dtype == np.float64:
                ds[v] = ds[v].astype("float32")
    enc = {v: ENC_FLOAT for v in ds.data_vars
           if np.issubdtype(ds[v].dtype, np.floating)}
    p = _out_path(rel)
    ds.to_netcdf(p, mode="w", encoding=enc)
    _log(f"wrote {rel}  ({p.stat().st_size/1e6:.1f} MB)")


def _slice_dir(coord: xr.DataArray, lo: float, hi: float) -> slice:
    """Slice respecting coordinate direction (ascending or descending)."""
    if float(coord[0]) <= float(coord[-1]):
        return slice(lo, hi)
    return slice(hi, lo)


def _leg_windows(margin_s: float, merge_gap_s: float = 300.0,
                 max_altitude: float | None = None):
    """Merged (start, end) windows around every leg in the metadata table."""
    df = pd.read_csv(AR_META, sep=None, engine="python")
    if max_altitude is not None and "mean_altitude" in df.columns:
        df = df[df["mean_altitude"] < max_altitude]
    starts = pd.to_datetime(df["start_time"])
    ends   = pd.to_datetime(df["end_time"])
    m = pd.Timedelta(seconds=margin_s)
    wins = sorted(zip(starts - m, ends + m))
    if not wins:
        raise ValueError("no legs match the window criteria")
    merged = [list(wins[0])]
    for s, e in wins[1:]:
        if s <= merged[-1][1] + pd.Timedelta(seconds=merge_gap_s):
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in merged]


def _day_windows(margin_h: float):
    """One window per flight day: min(start)−margin … max(end)+margin."""
    df = pd.read_csv(AR_META, sep=None, engine="python")
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"]   = pd.to_datetime(df["end_time"])
    m = pd.Timedelta(hours=margin_h)
    out = []
    for _, g in df.groupby(df["start_time"].dt.date):
        out.append((g["start_time"].min() - m, g["end_time"].max() + m))
    return sorted(out)


def _windows_subset(ds: xr.Dataset, windows) -> xr.Dataset:
    parts = [ds.sel(time=slice(str(s), str(e))) for s, e in windows]
    parts = [p for p in parts if p.sizes.get("time", 0) > 0]
    return xr.concat(parts, dim="time")


def subset_noseboom() -> None:
    # low-level (flux) legs only — the only ones any notebook slices at 100 Hz
    rel = "observations/processed/t-bird_unified/noseboom_unified.nc"
    ds = xr.open_dataset(DATA / rel, engine="h5netcdf").sortby("time")
    # P6 flies ~40-70 m above the towed T-Bird: low-level flux legs sit at
    # P6 mean altitudes of ~65-100 m in the leg metadata.
    _write(_windows_subset(ds, _leg_windows(margin_s=120, max_altitude=150.0)),
           rel, f32=True)


def subset_radiation() -> None:
    rel = "observations/processed/polar6_unified/radiation_unified.nc"
    ds = xr.open_dataset(DATA / rel, engine="h5netcdf").sortby("time")
    _write(_windows_subset(ds, _day_windows(margin_h=1.0)), rel, f32=True)


def subset_sic() -> None:
    rel = ("observations/processed/satellite_unified/"
           "sic_modis-aqua_amsr2-gcom-w1_merged_nh_1000m_APRIL.nc")
    ds = xr.open_dataset(DATA / rel)
    ds = ds.sel(lat=_slice_dir(ds["lat"], LAT_MIN, LAT_MAX),
                lon=_slice_dir(ds["lon"], LON_MIN, LON_MAX))
    _write(ds, rel)


def subset_modis_l3() -> None:
    rel = ("observations/processed/satellite_unified/"
           "MODIS_Aqua_Sea_Ice_Extent_and_IST_Daily_L3_Global_4km_EASE-Grid_Day_V061.nc")
    ds = xr.open_dataset(DATA / rel)
    ds = ds.sel(lat=_slice_dir(ds["lat"], LAT_MIN, LAT_MAX),
                lon=_slice_dir(ds["lon"], LON_MIN, LON_MAX))
    _write(ds, rel)


def _flight_day_mask(times, buffer_days: int = 1):
    df = pd.read_csv(AR_META, sep=None, engine="python")
    days = sorted(set(pd.to_datetime(df["start_time"]).dt.date))
    keep = set()
    for d in days:
        for off in range(-buffer_days, buffer_days + 1):
            keep.add(d + pd.Timedelta(days=off))
    return np.isin(pd.DatetimeIndex(times).date, sorted(keep))


def subset_carra() -> None:
    # only ever point-sampled at leg times → flight days suffice
    rel = "reanalysis/carra/carra_east_latlon0p05.nc"
    ds = xr.open_dataset(DATA / rel)[["skt", "sp"]]
    ds = ds.sel(lat=_slice_dir(ds["lat"], LAT_MIN, LAT_MAX),
                lon=_slice_dir(ds["lon"], LON_MIN, LON_MAX))
    ds = ds.isel(time=np.where(_flight_day_mask(ds["time"].values))[0])
    _write(ds.load(), rel, f32=True)


def subset_carra_hl() -> None:
    # height-level file used by load_carra() (30 m + 75 m only). At 0.05° this
    # grid is heavy, and it is only ever point-sampled at leg times → flight
    # days without buffer and a tight leg bounding box.
    rel = "reanalysis/carra/height_level/carra_latlon0p05.nc"
    ds = xr.open_dataset(DATA / rel)[["t", "ws", "pres"]]
    ds = ds.sel(heightAboveGround=[30.0, 75.0],
                lat=_slice_dir(ds["lat"], 75.4, 85.8),
                lon=_slice_dir(ds["lon"], -16.8, 26.4))
    tname = "valid_time" if "valid_time" in ds.dims else "time"
    ds = ds.isel({tname: np.where(
        _flight_day_mask(ds[tname].values, buffer_days=0))[0]})
    _write(ds.load(), rel, f32=True)


def subset_era5_sl() -> None:
    for name in ["data_stream-oper_stepType-instant.nc",
                 "data_stream-oper_stepType-avg.nc"]:
        rel = f"reanalysis/era5/ERA5_single_levels/{name}"
        ds = xr.open_dataset(DATA / rel)
        ds = ds.sel(latitude=_slice_dir(ds["latitude"], LAT_MIN, LAT_MAX),
                    longitude=_slice_dir(ds["longitude"], LON_MIN, LON_MAX))
        ds = ds.isel(valid_time=np.where(_flight_day_mask(ds["valid_time"].values))[0])
        _write(ds.load(), rel)


def subset_era5_ml() -> None:
    """Levels 134–137 (10–54 m AGL), flight days ±1, region crop, t/u/v only."""
    rel = "reanalysis/era5/ERA5_model_levels/era5_ml.nc"
    df = pd.read_csv(AR_META, sep=None, engine="python")
    days = sorted(set(pd.to_datetime(df["start_time"]).dt.date))
    keep_days = set()
    for d in days:
        for off in (-1, 0, 1):
            keep_days.add(d + pd.Timedelta(days=off))
    ds = xr.open_dataset(DATA / rel, chunks={"time": 24})[["t", "u", "v"]]
    ds = ds.sel(model_level=[134, 135, 136, 137],
                latitude=_slice_dir(ds["latitude"], LAT_MIN, LAT_MAX),
                longitude=_slice_dir(ds["longitude"], LON_MIN, LON_MAX))
    tmask = np.isin(pd.DatetimeIndex(ds["time"].values).date, sorted(keep_days))
    ds = ds.isel(time=np.where(tmask)[0])
    _write(ds.load(), rel)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    steps = [subset_noseboom, subset_radiation, subset_sic, subset_modis_l3,
             subset_carra, subset_carra_hl, subset_era5_sl, subset_era5_ml]
    if len(sys.argv) > 1:   # optional: run only the named steps
        wanted = {f"subset_{n}" for n in sys.argv[1:]}
        steps = [fn for fn in steps if fn.__name__ in wanted]
    failed = []
    for fn in steps:
        try:
            fn()
        except Exception as exc:                      # keep going, report at end
            failed.append((fn.__name__, repr(exc)))
            _log(f"FAILED {fn.__name__}: {exc!r}")
    _log("done" if not failed else f"done with {len(failed)} failure(s): {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

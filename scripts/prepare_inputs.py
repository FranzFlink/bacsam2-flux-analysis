#!/usr/bin/env python3
"""Published PANGAEA data -> the input files of the analysis, cut to the low-level legs.

The measurements are published in the bundle "Observations from the BACSAM II aircraft
campaign with Polar 6" (Köhler et al., 2026, doi:10.1594/PANGAEA.995979). The netCDF files of
the datasets below are expected in data/pangaea/<dataset>/ (one file per research flight):

    995974  Polar 6 nose boom, 100 Hz   -> data/observations/processed/polar6_unified/noseboom_unified.nc

Each file is cut to the windows around the legs flown below 150 m (Polar 6 height in
data/observations/ar/analysis_ready_polar6_legs_metadata.csv), 120 s before and after every
leg, windows less than 300 s apart merged. Names and units are those the analysis uses
(T in degC, p in hPa, h in m); the values are the published ones (float64).

Run:  pixi run inputs
"""
import sys
from pathlib import Path

import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
PANGAEA = REPO / "data" / "pangaea"
OBS = REPO / "data" / "observations"
LEGS_FILE = OBS / "ar" / "analysis_ready_polar6_legs_metadata.csv"
POLAR6_OUT = OBS / "processed" / "polar6_unified" / "noseboom_unified.nc"

MAX_LEG_HEIGHT_M = 150.0      # Polar 6 leg height below which a leg is a low-level (flux) leg
MARGIN_S = 120.0              # time added before and after every leg
MERGE_GAP_S = 300.0           # windows closer than this are merged


def leg_windows() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Merged (start, end) windows around the low-level legs."""
    legs = pd.read_csv(LEGS_FILE, sep="\t", parse_dates=["start_time", "end_time"])
    legs = legs[legs["mean_altitude"] < MAX_LEG_HEIGHT_M]
    margin = pd.Timedelta(seconds=MARGIN_S)
    windows = sorted(zip(legs["start_time"] - margin, legs["end_time"] + margin))
    merged = [list(windows[0])]
    for start, end in windows[1:]:
        if start <= merged[-1][1] + pd.Timedelta(seconds=MERGE_GAP_S):
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in merged]


def read_published(dataset: str, pattern: str) -> xr.Dataset:
    """All files of one PANGAEA dataset, concatenated in time."""
    files = sorted((PANGAEA / dataset).glob(pattern))
    if not files:
        sys.exit(f"no files {pattern} in {PANGAEA / dataset}: download PANGAEA {dataset} first (DATA_PUBLICATION.md)")
    ds = xr.concat([xr.open_dataset(f) for f in files], dim="time").sortby("time")
    return ds.drop_vars("segment", errors="ignore")


def cut(ds: xr.Dataset, windows) -> xr.Dataset:
    parts = [ds.sel(time=slice(s, e)) for s, e in windows]
    return xr.concat([p for p in parts if p.sizes["time"] > 0], dim="time")


def write(ds: xr.Dataset, path: Path, source: str) -> None:
    ds.attrs = {"source": source, "history": "cut to the low-level legs by scripts/prepare_inputs.py"}
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, mode="w", encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    print(f"written {path.relative_to(REPO)}: {ds.sizes['time']} samples")


def polar6_noseboom(windows) -> None:
    """Polar 6 nose boom: air temperature, static pressure and height (Fig. 2a)."""
    p = cut(read_published("995974", "*_noseboom_100Hz.nc"), windows)
    out = xr.Dataset({"T": (p["t_air"] - 273.15).assign_attrs(units="degC", long_name="air temperature"),
                      "p": (p["p_air"] / 100.0).assign_attrs(units="hPa", long_name="static pressure"),
                      "h": p["alt"].assign_attrs(units="m", long_name="altitude (alt of the published file)")})
    write(out, POLAR6_OUT, "Polar 6 nose boom 100 Hz, Köhler and Hartmann (2026), doi:10.1594/PANGAEA.995974")


if __name__ == "__main__":
    w = leg_windows()
    print(f"{len(w)} windows around the legs below {MAX_LEG_HEIGHT_M:.0f} m")
    polar6_noseboom(w)

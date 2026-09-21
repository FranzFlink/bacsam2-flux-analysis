#!/usr/bin/env python3
"""Segment-placement uncertainty of the EC sensible heat flux.

Where exactly a 5 km segment is cut out of a flight leg is arbitrary. To
quantify what this choice does to the flux, a 5 km window is moved through
every low-level leg in 0.05 km steps and the EC flux is computed at each
position (same estimator as the segment fluxes: linear detrend, cospectrum
integrated over 0.1-20 Hz). For each analysis segment, the placement
uncertainty is the standard deviation of all window fluxes that overlap the
segment, weighted by their fractional overlap (triangular weights).

Ported from the notebook preprocessing/ec_segment_placement_sensitivity.ipynb.

Output: data/observations/ar/ec_segment_placement.nc
Run:    pixi run placement      (about 10 minutes)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from fluxpy.flux_processing import get_cov   # noqa: E402
from fluxpy.meteorology import add_theta     # noqa: E402

warnings.filterwarnings("ignore")

WIN_KM, STEP_KM = 5.0, 0.05
MIN_SAMP_WIN = 3000            # samples required inside a window
MIN_LEG_KM = 2 * WIN_KM
MAX_LEG_ALT_M = 100.0          # low-level legs only
FS = 100.0
MATCH_TOL_S = 60.0             # segment start times must agree within this

AR = REPO / "data" / "observations" / "ar"
ds_seg = xr.open_dataset(AR / "analysis_ready_polar6_turbulent_fluxes_latest_5km.nc")
ds_hf = xr.open_dataset(REPO / "data/observations/processed/t-bird_unified/noseboom_unified.nc").sortby("time")
legs = pd.read_csv(AR / "analysis_ready_polar6_legs_metadata.csv", sep="\t",
                   parse_dates=["start_time", "end_time"])
legs = legs[legs["platform"].str.lower().str.contains("polar", na=False)
            & (legs["mean_altitude"] < MAX_LEG_ALT_M)].reset_index(drop=True)


def _flux(ds):
    _, h = get_cov(ds.copy(), "theta", heat_flux=True, time_dim="distance", fs=FS, highpass=False)
    return float(h)


rows = []
for meta in tqdm(list(legs.itertuples()), desc="legs"):
    raw = ds_hf.sel(time=slice(meta.start_time, meta.end_time))
    if raw.time.size < MIN_SAMP_WIN:
        continue
    raw = add_theta(raw.copy())
    tas = float(raw["tas"].mean(skipna=True))
    if not np.isfinite(tas) or tas < 10:
        continue
    dist_km = tas * (raw["time"] - raw["time"][0]).dt.total_seconds().values / 1000.0
    raw = raw.assign_coords(distance=("time", dist_km)).swap_dims({"time": "distance"})
    raw = raw.dropna(dim="distance", how="any")
    if raw.distance.size < MIN_SAMP_WIN:
        continue
    d_min, d_max = float(raw.distance.min()), float(raw.distance.max())
    if d_max - d_min < MIN_LEG_KM:
        continue
    t0 = pd.to_datetime(raw["time"].values[0])

    # flux at every window position
    centres, H = [], []
    left = d_min
    while left + WIN_KM <= d_max + 1e-9:
        win = raw.sel(distance=slice(left, left + WIN_KM))
        if win.distance.size >= MIN_SAMP_WIN:
            centres.append(left + WIN_KM / 2.0)
            H.append(_flux(win))
        left += STEP_KM
    centres, H = np.asarray(centres), np.asarray(H)
    if H.size < 2:
        continue

    # overlap-weighted spread for each non-overlapping 5 km segment of the leg
    seg_s = 0.0
    while seg_s + WIN_KM <= (d_max - d_min) + 1e-9:
        lo, hi = d_min + seg_s, d_min + seg_s + WIN_KM
        overlap = np.clip(np.minimum(centres + WIN_KM / 2, hi) - np.maximum(centres - WIN_KM / 2, lo), 0, None)
        use = (overlap > 1e-9) & np.isfinite(H)
        if use.sum() >= 2:
            w = overlap[use] / WIN_KM
            mean = float((w * H[use]).sum() / w.sum())
            std = float(np.sqrt((w * (H[use] - mean) ** 2).sum() / w.sum()))
            rows.append({"t_seg_start": t0 + pd.to_timedelta(seg_s * 1000.0 / tas, unit="s"),
                         "H_placement_mean": mean, "sigma_placement": std, "n_windows": int(use.sum())})
        seg_s += WIN_KM

df = pd.DataFrame(rows)

# match to the analysis segments by start time
t_seg = pd.DatetimeIndex(ds_seg["start_time"].values)
idx = t_seg.get_indexer(pd.DatetimeIndex(df["t_seg_start"]), method="nearest")
df["seg_idx"] = idx
df["dt_s"] = np.abs((pd.DatetimeIndex(df["t_seg_start"]) - t_seg[idx]).total_seconds())
df = df[df["dt_s"] < MATCH_TOL_S].sort_values("dt_s").drop_duplicates("seg_idx")

out = xr.Dataset(coords={"start_time": ds_seg["start_time"].values})
for name, units, long_name in [
    ("sigma_placement", "W m-2", "Segment-placement uncertainty of the EC sensible heat flux (overlap-weighted std of 5 km window fluxes, 0.05 km steps)"),
    ("H_placement_mean", "W m-2", "Overlap-weighted mean EC sensible heat flux of the 5 km windows"),
    ("n_windows", "1", "Number of windows overlapping the segment"),
]:
    arr = np.full(t_seg.size, np.nan)
    arr[df["seg_idx"].values] = df[name].values
    out[name] = ("start_time", arr, {"units": units, "long_name": long_name})
out.attrs = {"title": "BACSAM-II EC segment-placement uncertainty",
             "source": "scripts/ec_segment_placement.py",
             "window_km": WIN_KM, "step_km": STEP_KM}
out.to_netcdf(AR / "ec_segment_placement.nc")
print(f"matched {len(df)} of {t_seg.size} segments; "
      f"median sigma_placement = {np.nanmedian(out['sigma_placement']):.2f} W m-2")

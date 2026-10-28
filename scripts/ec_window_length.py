#!/usr/bin/env python3
"""EC window-length experiment (manuscript Fig. 3).

Each low-level T-Bird leg is cut into consecutive sub-windows of length L
(0.5 ... 50 km). For every L the mean EC flux of the sub-windows is compared with
the EC flux of the stretch they cover (the remainder shorter than L is not part of
any window). This ratio depends on where the window grid starts, because a window
edge can cut through a strong flux event, so it is averaged over five grid
positions (starting 0, L/5, ..., 4L/5 along the leg). The random error of the
sub-window fluxes and the shares of sub-windows whose flux exceeds 1, 2 and 3 times
its random error come from the grid that starts at the beginning of the leg. Same
estimator as the segment fluxes (linear detrend, cospectrum integrated over 0.1-20 Hz).

Output: data/observations/ar/ec_leg_length_sensitivity/ec_sensitivity_summary.csv
Run:    pixi run window-length      (about 10 minutes)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from fluxpy.eddy_covariance import segment_flux   # noqa: E402
from fluxpy.util import add_theta                 # noqa: E402

warnings.filterwarnings("ignore")

LENGTHS_KM = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]
EXCLUDE = [pd.Timestamp("2024-04-25").date(), pd.Timestamp("2024-04-26").date()]   # open water / no low legs
FS = 100.0
MIN_FULL_LEG_SAMPLES = 5000
MIN_FULL_FLUX = 0.5            # W m-2; ratios are undefined for legs (and stretches) with weaker fluxes
N_GRID_POSITIONS = 5           # the flux ratio is averaged over grids starting at 0, L/5, ..., 4L/5

AR = REPO / "data" / "observations" / "ar"
ds_hf = xr.open_dataset(REPO / "data/observations/processed/t-bird_unified/noseboom_unified.nc").sortby("time")
legs = pd.read_csv(AR / "analysis_ready_polar6_legs_metadata.csv", sep="\t",
                   parse_dates=["start_time", "end_time"])
legs = legs[legs["platform"].str.lower().str.contains("polar", na=False)
            & (legs["mean_altitude"] < 100)].reset_index(drop=True)


def min_samples(L_km):
    """Half of the samples expected at 100 Hz and 60 m/s, at least 500."""
    return max(500, int(FS * L_km * 1000 / 60.0 * 0.5))


def along_track_km(leg):
    """Cumulative great-circle distance from the GPS position."""
    lat = pd.Series(leg["lat"].values).interpolate().ffill().bfill().values
    lon = pd.Series(leg["lon"].values).interpolate().ffill().bfill().values
    lat_mid = np.radians((lat[:-1] + lat[1:]) / 2.0)
    dx = np.radians(np.diff(lon)) * np.cos(lat_mid) * 6371.0088
    dy = np.radians(np.diff(lat)) * 6371.0088
    return np.cumsum(np.insert(np.hypot(dx, dy), 0, 0.0))


def flux(ds):
    """EC flux of a window, same estimator as the segment fluxes."""
    return segment_flux(ds, fs=FS).flux


detailed, parents, windows = [], [], []
for i, meta in enumerate(legs.itertuples()):
    leg = ds_hf.sel(time=slice(meta.start_time, meta.end_time))
    if leg.time.size == 0 or pd.to_datetime(leg["time"][0].item()).date() in EXCLUDE:
        continue
    leg = add_theta(leg.copy())
    dist = along_track_km(leg)
    clean = leg.dropna(dim="time", subset=["w", "theta", "T", "p"])
    if clean.time.size < MIN_FULL_LEG_SAMPLES:
        continue
    H_full, full_km = flux(clean), float(dist[-1] - dist[0])
    parents.append({"leg_id": i, "full_length_km": full_km, "full_flux": H_full})
    print(f"leg {i:02d}: {full_km:5.1f} km, full-leg flux {H_full:+.2f} W m-2")

    for L in LENGTHS_KM:
        ratios, differences, n_first = [], [], 0
        for k in range(N_GRID_POSITIONS):
            offset = k * L / N_GRID_POSITIONS
            n_windows = int(np.floor((full_km - offset) / L))
            H_sub = []
            for j in range(n_windows):
                sel = (dist >= offset + j * L) & (dist < offset + (j + 1) * L)
                sub = leg.isel(time=sel).dropna(dim="time", subset=["w", "theta", "T", "p"])
                if sub.time.size < min_samples(L):
                    continue
                # lag range of the random error: 10 s, or a quarter of a short window
                ec = segment_flux(sub, fs=FS, max_lag_s=min(10.0, 0.25 * sub.time.size / FS))
                H_sub.append(ec.flux)
                if k == 0:       # random error and fractions above 1/2/3 sigma: grid from the leg start
                    windows.append({"subleg_length_km": L, "random_error": ec.random_error,
                                    "ratio": abs(ec.flux) / ec.random_error})
            if not H_sub:
                continue
            if k == 0:
                n_first = len(H_sub)
            # reference: the flux of the stretch this grid covers (the remainder shorter than L
            # is not part of any window, so the full-leg flux would compare different air)
            covered = leg.isel(time=(dist >= offset) & (dist < offset + n_windows * L))
            H_cov = flux(covered.dropna(dim="time", subset=["w", "theta", "T", "p"]))
            differences.append(np.mean(H_sub) - H_cov)
            if abs(H_cov) >= MIN_FULL_FLUX:
                ratios.append(np.mean(H_sub) / H_cov)
        if not differences:
            continue
        detailed.append({
            "leg_id": i, "subleg_length_km": L, "n_sublegs": n_first,
            "absolute_difference": np.mean(differences),
            "relative_ratio": np.mean(ratios) if abs(H_full) >= MIN_FULL_FLUX and ratios else np.nan,
        })

par, det, win = pd.DataFrame(parents), pd.DataFrame(detailed), pd.DataFrame(windows)
rows = [{"label": "full leg", "length_km": float(par["full_length_km"].median()), "subleg_length_km": np.nan,
         "N_bins": len(par), "n_legs": len(par), "ratio_mean": 1.0, "ratio_std": 0.0}]
for L, g in det.groupby("subleg_length_km"):
    r, w = g["relative_ratio"].dropna(), win[win["subleg_length_km"] == L]
    rows.append({
        "label": f"{L:.1f} km".replace(".0", ""), "length_km": L, "subleg_length_km": L,
        "N_bins": int(g["n_sublegs"].sum()), "n_legs": int(g["leg_id"].nunique()),
        "ratio_mean": r.mean(), "ratio_std": r.std(ddof=0),
        "median_absolute_difference": g["absolute_difference"].median(),
        "random_error_median": w["random_error"].median(),
        "random_error_p25": w["random_error"].quantile(0.25), "random_error_p75": w["random_error"].quantile(0.75),
        "fraction_above_1sigma": (w["ratio"] > 1).mean(), "fraction_above_2sigma": (w["ratio"] > 2).mean(),
        "fraction_above_3sigma": (w["ratio"] > 3).mean(),
    })
out = AR / "ec_leg_length_sensitivity" / "ec_sensitivity_summary.csv"
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
print(f"\nwritten {out.relative_to(REPO)}")
print(pd.DataFrame(rows)[["label", "n_legs", "N_bins", "ratio_mean", "ratio_std",
                          "random_error_median", "fraction_above_1sigma", "fraction_above_2sigma",
                          "fraction_above_3sigma"]].round(2).to_string(index=False))

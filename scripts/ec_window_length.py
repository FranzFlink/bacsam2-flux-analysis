#!/usr/bin/env python3
"""EC window-length experiment (manuscript Fig. 3).

Each low-level T-Bird leg is cut into consecutive sub-windows of length L
(0.5 ... 50 km). For every L the EC flux of the sub-windows is compared with
the EC flux of the complete leg, and the relative sampling error and the
spectral QC verdicts of the sub-windows are collected. Same estimator as the
segment fluxes (linear detrend, cospectrum integrated over 0.1-20 Hz).

Ported from the exploratory notebook of the same name (see git history).

Output: data/observations/ar/ec_leg_length_sensitivity/ec_sensitivity_summary.csv
Run:    pixi run window-length      (about 2 minutes)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from fluxpy.aggregate import _integral_length_scale       # noqa: E402
from fluxpy.flux_processing import get_cov                # noqa: E402
from fluxpy.meteorology import add_theta                  # noqa: E402
from fluxpy.qc import classify_leg_simple, qc_leg         # noqa: E402

warnings.filterwarnings("ignore")

LENGTHS_KM = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]
EXCLUDE = [pd.Timestamp("2024-04-25").date(), pd.Timestamp("2024-04-26").date()]   # open water / no low legs
FS = 100.0
MIN_FULL_LEG_SAMPLES = 5000
MIN_FULL_FLUX = 0.5            # W m-2; ratios are undefined for weaker full-leg fluxes

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
    _, h = get_cov(ds, "theta", heat_flux=True, time_dim="time", fs=FS, highpass=False)
    return float(h)


detailed, parents = [], []
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
        H_sub, e_rel, n_fail = [], [], []
        for j in range(int(np.floor(full_km / L))):
            sel = (dist >= j * L) & (dist < (j + 1) * L)
            sub = leg.isel(time=sel).dropna(dim="time", subset=["w", "theta", "T", "p"])
            if sub.time.size < min_samples(L):
                continue
            H_sub.append(flux(sub))
            tas = float(sub["tas"].mean())
            L_int = _integral_length_scale(sub["T"].values, sub["w"].values, tas, FS)
            e_rel.append(float(np.clip(np.sqrt(2.0 * L_int / (L * 1000.0)), 0, 2)) if np.isfinite(L_int) else np.nan)
            try:        # windows below ~1 km are too short for the Welch spectra of the QC
                qc = qc_leg(w=sub.w.values, theta=sub.theta.values, u=sub.u.values, v=sub.v.values, fs=FS)
                n_fail.append(classify_leg_simple(qc, L)["n_failed"])
            except ValueError:
                pass
        if not H_sub:
            continue
        ok = abs(H_full) >= MIN_FULL_FLUX
        detailed.append({
            "leg_id": i, "subleg_length_km": L, "n_sublegs": len(H_sub),
            "absolute_difference": np.mean(H_sub) - H_full,
            "relative_ratio": np.mean(H_sub) / H_full if ok else np.nan,
            "within_leg_cv_abs": np.std(H_sub) / abs(H_full) if ok else np.nan,
            "lenschow_mean": np.nanmean(e_rel), "lenschow_std": np.nanstd(e_rel),
            "n_accepted": sum(f == 0 for f in n_fail), "n_tolerable": sum(f == 1 for f in n_fail),
            "n_discarded": sum(f >= 2 for f in n_fail),
        })

par, det = pd.DataFrame(parents), pd.DataFrame(detailed)
rows = [{"label": "full leg", "length_km": float(par["full_length_km"].median()), "subleg_length_km": np.nan,
         "N_bins": len(par), "n_legs": len(par), "median_ratio": 1.0, "mean_ratio": 1.0, "p25_ratio": 1.0,
         "p75_ratio": 1.0, "median_absolute_difference": 0.0, "RMSE_absolute_difference": 0.0,
         "median_within_leg_cv_abs": 0.0, "ratio_mean": 1.0, "ratio_std": 0.0,
         "lenschow_rel_error_mean": np.nan, "lenschow_rel_error_std": np.nan,
         "n_accepted": len(par), "n_tolerable": 0, "n_discarded": 0}]
for L, g in det.groupby("subleg_length_km"):
    r = g["relative_ratio"].dropna()
    rows.append({
        "label": f"{L:.1f} km".replace(".0", ""), "length_km": L, "subleg_length_km": L,
        "N_bins": int(g["n_sublegs"].sum()), "n_legs": int(g["leg_id"].nunique()),
        "median_ratio": r.median(), "mean_ratio": r.mean(),
        "p25_ratio": r.quantile(0.25), "p75_ratio": r.quantile(0.75),
        "median_absolute_difference": g["absolute_difference"].median(),
        "RMSE_absolute_difference": float(np.sqrt((g["absolute_difference"] ** 2).mean())),
        "median_within_leg_cv_abs": g["within_leg_cv_abs"].median(),
        "ratio_mean": r.mean(), "ratio_std": r.std(ddof=0),
        "lenschow_rel_error_mean": g["lenschow_mean"].mean(), "lenschow_rel_error_std": g["lenschow_std"].mean(),
        "n_accepted": int(g["n_accepted"].sum()), "n_tolerable": int(g["n_tolerable"].sum()),
        "n_discarded": int(g["n_discarded"].sum()),
    })
out = AR / "ec_leg_length_sensitivity" / "ec_sensitivity_summary.csv"
pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
print(f"\nwritten {out.relative_to(REPO)}")
print(pd.DataFrame(rows)[["label", "n_legs", "N_bins", "ratio_mean", "ratio_std",
                          "lenschow_rel_error_mean", "n_discarded"]].round(3).to_string(index=False))

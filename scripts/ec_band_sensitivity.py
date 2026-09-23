#!/usr/bin/env python3
"""Sensitivity of the EC reference and of the comparison to the frequency band of the flux.

Manuscript Sect. 3.1 integrates the w-theta cospectrum over 0.1-20 Hz. The window-length
experiment (Fig. 3) tests the consistency of that choice but not how much flux lies outside
the band. Here every EC flux is recomputed from the 100 Hz data with other bands, and the
comparison (slope b / correlation r of the three bulk estimates against the EC flux) is
repeated with each of them. The first row reproduces the shipped EC fluxes.

A 5 km segment lasts about 85 s, so its lowest resolved frequency is about 0.012 Hz;
bands that start below 0.05 Hz rest on one or two cycles and are not used.

Run:  pixi run ec-bands      (about 30 s)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.signal import detrend
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from fluxpy.eddy_covariance import FLUX_BAND_HZ, SAMPLING_HZ, band_covariance  # noqa: E402
from fluxpy.util import CP, add_theta, air_density  # noqa: E402

warnings.filterwarnings("ignore")
RULE = "=" * 110
BANDS = [("0.1-20 Hz (this study)", FLUX_BAND_HZ), ("0.05-20 Hz", (0.05, 20.0)), ("0.2-20 Hz", (0.2, 20.0)),
         ("0.1-10 Hz", (0.1, 10.0)), ("0.1-5 Hz", (0.1, 5.0)), ("all frequencies", (0.0, SAMPLING_HZ / 2))]
ESTIMATES = [("H_obs_bulk", "H_S,obs,bulk"), ("H_rs_bulk", "H_S,RS,bulk"), ("H_era_bulk", "H_S,ERA,bulk")]

FIG = xr.open_dataset(REPO / "data/model_output/sensitivity_figure_data.nc")
SEG = xr.open_dataset(REPO / "data/observations/ar/analysis_ready_polar6_turbulent_fluxes_latest_5km.nc"
                      ).sel(start_time=FIG["start_time"].values)
TBIRD = xr.open_dataset(REPO / "data/observations/processed/t-bird_unified/noseboom_unified.nc").sortby("time")

flux = {name: [] for name, _ in BANDS}
for t0, t1 in zip(SEG["start_time"].values, SEG["end_time"].values):
    seg = add_theta(TBIRD.sel(time=slice(t0, t1))).dropna("time", how="any")
    w, th = detrend(seg["w"].values.astype(float)), detrend(seg["theta"].values)
    k = float(np.mean(air_density(seg["T"].values, seg["p"].values))) * CP          # rho c_p: covariance -> W m-2
    for name, (f_lo, f_hi) in BANDS:
        flux[name].append(k * band_covariance(w, th, SAMPLING_HZ, f_lo, f_hi))
flux = {name: np.asarray(v) for name, v in flux.items()}

ec_shipped = np.asarray(FIG["H_obs_ec"], dtype=float)
reference = flux[BANDS[0][0]]
large = np.abs(reference) > 2.0


def slope_r(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return np.polyfit(x[ok], y[ok], 1)[0], np.corrcoef(x[ok], y[ok])[0, 1]


print(RULE)
print("EC FLUX BAND: how much flux lies outside 0.1-20 Hz, and does the comparison depend on the band?")
print(RULE)
print(f"0.1-20 Hz recomputed here vs the shipped EC fluxes: max |difference| {np.max(np.abs(reference - ec_shipped)):.3f} W m-2")
print(f"flux in the band / flux in 0.1-20 Hz for the {large.sum()} segments with |H| > 2 W m-2: median [quartiles];  "
      "then slope b / r of each bulk estimate against the EC flux of that band (n = 91)")
print(f"  {'band':<24} {'flux ratio':>20}   " + "   ".join(f"{name:>13}" for _, name in ESTIMATES))
for name, _ in BANDS:
    ratio = flux[name][large] / reference[large]
    cells = [slope_r(flux[name], np.asarray(FIG[key], dtype=float)) for key, _ in ESTIMATES]
    print(f"  {name:<24} {np.median(ratio):6.2f} [{np.percentile(ratio, 25):.2f}, {np.percentile(ratio, 75):.2f}]   "
          + "   ".join(f"{b:5.1f} / {r:5.1f}" for b, r in cells))
below =(flux["all frequencies"] - reference)[large] / reference[large]
print(f"flux outside 0.1-20 Hz relative to the flux inside, |H| > 2 W m-2: median {np.median(below) * 100:+.0f} % "
      f"(quartiles {np.percentile(below, 25) * 100:+.0f} to {np.percentile(below, 75) * 100:+.0f} %); "
      f"rank correlation of the all-frequency and the 0.1-20 Hz flux over all 91 segments: "
      f"{spearmanr(flux['all frequencies'], reference)[0]:.2f}")
print()

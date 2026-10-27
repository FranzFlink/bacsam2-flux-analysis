#!/usr/bin/env python3
"""Robustness of the comparison, printed from the pipeline output.

The checks test the statements of the manuscript against the choice of metric, errors
in the reference, the dependence between neighbouring segments, the cloud cover of the
satellite retrieval, the stability correction, the fitting of alpha_T and the
measurement height. ``pixi run check`` compares this output with
``expected_numbers.txt`` like every other number.

Blocks (in the order printed) and what each one tests
    1  more metrics ...................... the OLS slope is r times the ratio of standard deviations;
                                           intercept, sd ratio, mean absolute error, and what a
                                           zero-flux or constant forecast would score
    2  errors in the reference ........... how much the random error of the EC flux flattens
                                           the regression slopes (regression dilution)
    3  dependent segments ................ segments along a leg are not independent: confidence
                                           intervals with blocks of segments and legs as units,
                                           leave-one-day-out, the satellite gain under both
    4  satellite surface temperature ..... how many segments have a genuine MODIS value, the
                                           comparison on those only, T_s of MODIS / ERA5 vs KT-19
    5  stability ......................... sampled Ri_b, the stability factor f_h = C_h / C_h,n,
                                           and the per-segment spread of the eight schemes
    6  alpha_T on withheld days .......... amplitude matching fitted on three days, tested on the fourth
    7  measurement height ................ lower half vs upper half of the T-Bird heights

Run:  pixi run robustness      (needs pixi run precompute; about 10 s)
"""
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from fluxpy.bulk import (amplitude_matched_config, compute_bulk_shf_obs, compute_bulk_shf_remote,  # noqa: E402
                         reference_config, transfer_coefficients)
from fluxpy.sensitivity import skill_metrics  # noqa: E402
from calc_bulk_and_ec_fluxes import _read_cache, era5_forcing, load_obs  # noqa: E402

logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")        # resamples drawn from a single leg have no variance; they are dropped below
RULE = "=" * 118
N_BOOT = 2000

FIG = xr.open_dataset(REPO / "data/model_output/sensitivity_figure_data.nc")
EC = np.asarray(FIG["H_obs_ec"], dtype=float)
T = pd.to_datetime(FIG["start_time"].values)
DAY = np.asarray(T.strftime("%d %b"))
LEG = np.cumsum(np.r_[0, np.diff(T.values) / np.timedelta64(1, "m")] > 10)       # a new leg after a gap > 10 min
ESTIMATES = [("H_obs_bulk", "H_S,obs,bulk"), ("H_rs_bulk", "H_S,RS,bulk"), ("H_era_bulk", "H_S,ERA,bulk"),
             ("H_era_diag", "H_S,ERA,diag"), ("H_car_diag", "H_S,CAR,diag")]
BULK = ESTIMATES[:3]


def H(key):
    return np.asarray(FIG[key], dtype=float)


def slope_r(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return np.polyfit(x[ok], y[ok], 1)[0], np.corrcoef(x[ok], y[ok])[0, 1]


def header(title):
    print("\n" + RULE + "\n" + title + "\n" + RULE)


# ── 1  more metrics ──────────────────────────────────────────────────────────

def more_metrics():
    header("1. METRICS BEYOND SLOPE AND CORRELATION  (reference configuration; W m-2)")
    print(f"{'estimate':<14} {'slope':>5} {'= r':>6} {'x sd ratio':>10} {'intercept':>9} {'bias':>6} {'MAE':>5} {'RMSE':>5} {'n':>4}")
    for key, name in ESTIMATES:
        m = skill_metrics(EC, H(key))
        ok = np.isfinite(EC) & np.isfinite(H(key))
        mae = np.mean(np.abs(H(key)[ok] - EC[ok]))
        print(f"{name:<14} {m['slope']:5.1f} {m['r']:6.1f} {m['std_ratio']:10.1f} {m['intercept']:+9.1f} "
              f"{m['bias']:+6.1f} {mae:5.1f} {m['rmse']:5.1f} {m['n']:4d}")
    runs_obs = [str(r) for r in FIG["run_obs"].values]
    matched_cfg = amplitude_matched_config()
    matched = np.asarray(FIG["H_obs_bulk_all"])[runs_obs.index(matched_cfg.label)]
    matched_name = f"alpha_T {matched_cfg.zt_ratio:.0e}".replace("e-0", "e-")
    m = skill_metrics(EC, matched)
    print(f"{'  ' + matched_name:<14} {m['slope']:5.1f} {m['r']:6.1f} {m['std_ratio']:10.1f} {m['intercept']:+9.1f} "
          f"{m['bias']:+6.1f} {np.mean(np.abs(matched - EC)):5.1f} {m['rmse']:5.1f} {m['n']:4d}   (amplitude-matched in-situ)")
    zero_mae, zero_rmse = np.mean(np.abs(EC)), np.sqrt(np.mean(EC ** 2))
    print(f"{'zero flux':<14} {'':>5} {'':>6} {'':>10} {'':>9} {-EC.mean():+6.1f} {zero_mae:5.1f} "
          f"{zero_rmse:5.1f} {EC.size:4d}   (H = 0 everywhere)")
    print(f"{'constant mean':<14} {'':>5} {'':>6} {'':>10} {'':>9} {0.0:+6.1f} {np.mean(np.abs(EC - EC.mean())):5.1f} "
          f"{EC.std():5.1f} {EC.size:4d}   (H = mean EC flux everywhere)")
    candidates = [(name, H(key)) for key, name in ESTIMATES] + [(matched_name, matched)]
    beat = [name for name, h in candidates
            if np.sqrt(np.nanmean((h - EC) ** 2)) < zero_rmse and np.nanmean(np.abs(h - EC)) < zero_mae]
    print("estimates with a lower RMSE and MAE than the zero-flux forecast: " + (", ".join(beat) if beat else "none"))


# ── 2  errors in the reference ───────────────────────────────────────────────

def regression_dilution():
    header("2. RANDOM ERROR OF THE EC REFERENCE AND THE REGRESSION SLOPE (errors in variables)")
    sigma = H("sigma_random_ec")
    reliability = 1.0 - np.mean(sigma ** 2) / np.var(EC, ddof=1)
    print(f"variance of the EC flux {np.var(EC, ddof=1):.1f} W2 m-4, mean random-error variance {np.mean(sigma ** 2):.1f} W2 m-4 "
          f"-> reliability ratio {reliability:.2f}")
    print("slope of each estimate on the EC flux, OLS and corrected for the random error of the reference (OLS / reliability):")
    for key, name in ESTIMATES:
        b = skill_metrics(EC, H(key))["slope"]
        print(f"  {name:<14} {b:.1f} -> {b / reliability:.1f}")
    print(f"the random error of the reference flattens every OLS slope by {(1 - reliability) * 100:.0f} %; "
          "the correlations are lowered by the square root of the same factor")


# ── 3  dependent segments ────────────────────────────────────────────────────

def _blocks(length):
    """Every run of `length` consecutive segments inside one leg (moving blocks)."""
    out = []
    for leg in np.unique(LEG):
        idx = np.where(LEG == leg)[0]
        out += [idx[i:i + length] for i in range(len(idx) - length + 1)]
    return out


def _resample(scheme, rng):
    if scheme == "segments":
        return rng.integers(0, EC.size, EC.size)
    if scheme == "blocks of 3":
        blocks = _blocks(3)
        return np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), int(np.ceil(EC.size / 3)))])[:EC.size]
    legs = np.unique(LEG)
    return np.concatenate([np.where(LEG == leg)[0] for leg in rng.choice(legs, legs.size)])


def _lag1(x):
    """Lag-1 autocorrelation of x along the legs (pairs of neighbouring segments in the same leg)."""
    same = LEG[1:] == LEG[:-1]
    a, b = x[:-1][same], x[1:][same]
    ok = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[ok], b[ok])[0, 1]


def dependence():
    header("3. DEPENDENT SEGMENTS: segments along a leg share the weather, the grid cells and the satellite scene")
    rho_ec = _lag1(EC)
    sizes = np.bincount(LEG)
    print(f"{EC.size} segments on {sizes.size} legs (a new leg after a gap of more than 10 min; segments per leg "
          + ", ".join(str(s) for s in sizes) + f") of {len(pd.unique(DAY))} flight days")
    print(f"lag-1 autocorrelation along the legs: EC flux {rho_ec:.1f}, "
          f"residual of the T-Bird bulk {_lag1(H('H_obs_bulk') - EC):.1f}, of the ERA5-MODIS bulk {_lag1(H('H_rs_bulk') - EC):.1f}; "
          f"effective sample size of the EC flux n (1 - rho) / (1 + rho) = {EC.size * (1 - rho_ec) / (1 + rho_ec):.0f}")
    print("95 % bootstrap intervals of slope and r, resampling units: single segments | moving blocks of 3 segments (15 km) | whole legs")
    for key, name in BULK:
        h = H(key)
        cells = []
        for scheme in ("segments", "blocks of 3", "legs"):
            rng = np.random.default_rng(7)
            boot = np.array([slope_r(EC[i], h[i]) for i in (_resample(scheme, rng) for _ in range(N_BOOT))])
            lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
            cells.append(f"b [{lo[0]:.1f}, {hi[0]:.1f}] r [{lo[1]:.1f}, {hi[1]:.1f}]")
        print(f"  {name:<14} " + " | ".join(cells))

    print("leave one flight day out (slope / r on the other three days):")
    print(f"  {'left out':<9} {'n':>3}   " + "   ".join(f"{name:>13}" for _, name in BULK))
    for day in pd.unique(DAY):
        keep = DAY != day
        print(f"  {day:<9} {keep.sum():3d}   " + "   ".join(f"{b:5.1f} / {r:5.1f}" for b, r in
                                                             (slope_r(EC[keep], H(k)[keep]) for k, _ in BULK)))

    # the satellite gain in the heterogeneous half (manuscript Sect. 4.1) with legs and days as units
    seg = xr.open_dataset(REPO / "data/observations/ar/analysis_ready_polar6_turbulent_fluxes_latest_5km.nc")
    sd = pd.Series(np.asarray(seg["T_sfc_std"]), index=pd.to_datetime(seg["start_time"].values)
                   ).reindex(T, method="nearest", tolerance=pd.Timedelta("60s")).values
    hetero = sd > np.nanmedian(sd)
    rs, era = H("H_rs_bulk"), H("H_era_bulk")

    def gain(i):
        i = i[hetero[i]]
        return np.corrcoef(EC[i], rs[i])[0, 1] - np.corrcoef(EC[i], era[i])[0, 1]

    all_idx = np.arange(EC.size)
    print(f"satellite gain in the heterogeneous half, r(ERA5-MODIS) - r(ERA5-only) = {gain(all_idx):+.1f}; "
          f"heterogeneous segments per day: " + ", ".join(f"{d} {int((hetero & (DAY == d)).sum())}" for d in pd.unique(DAY)))
    for scheme in ("segments", "blocks of 3", "legs"):
        rng = np.random.default_rng(11)
        boot = np.array([gain(_resample(scheme, rng)) for _ in range(N_BOOT)])
        boot = boot[np.isfinite(boot)]
        print(f"  resampling {scheme:<12}: 95 % interval [{np.percentile(boot, 2.5):+.1f}, {np.percentile(boot, 97.5):+.1f}], "
              f"fraction of resamples <= 0: {np.mean(boot <= 0):.3f}")
    print("  leave one day out: " + ", ".join(f"without {d} {gain(all_idx[DAY != d]):+.1f}" for d in pd.unique(DAY)))


# ── 4  satellite surface temperature ─────────────────────────────────────────

def satellite_input():
    header("4. SATELLITE SURFACE TEMPERATURE: GENUINE MODIS VALUES, AND T_s AGAINST THE KT-19")
    cache = xr.open_dataset(REPO / "data/observations/ar/along_track_inputs.nc")
    assert np.array_equal(cache["start_time"].values, FIG["start_time"].values)
    valid = np.isfinite(np.asarray(cache["modis_IST"], dtype=float))
    print(f"MODIS IST valid (cloud-free) at {valid.sum()} of {valid.size} segments; elsewhere the ERA5 skin temperature is used. Per day: "
          + ", ".join(f"{d} {int(valid[DAY == d].sum())}/{int((DAY == d).sum())}" for d in pd.unique(DAY)))
    print(f"comparison on the {valid.sum()} segments with a genuine MODIS value (slope / r):   "
          + "   ".join(f"{name} {b:.1f} / {r:.1f}" for (key, name) in BULK for b, r in [slope_r(EC[valid], H(key)[valid])]))
    print(f"                    on the {(~valid).sum()} gap-filled segments (slope / r):             "
          + "   ".join(f"{name} {b:.1f} / {r:.1f}" for (key, name) in BULK for b, r in [slope_r(EC[~valid], H(key)[~valid])]))
    kt19 = H("T_sfc_C")
    print("surface temperature against the segment-mean KT-19 brightness temperature (K): bias / RMSE / r")
    for name, ts, m in (("MODIS, valid segments", H("T_sfc_rs_C"), valid),
                        ("ERA5, same segments", H("T_sfc_era_C"), valid),
                        ("ERA5, all segments", H("T_sfc_era_C"), np.ones_like(valid))):
        d = ts[m] - kt19[m]
        print(f"  {name:<22} {d.mean():+5.1f} / {np.sqrt(np.mean(d ** 2)):4.1f} / {np.corrcoef(ts[m], kt19[m])[0, 1]:4.1f}   (n = {m.sum()})")


# ── 5  stability ─────────────────────────────────────────────────────────────

def stability():
    header("5. STABILITY: SAMPLED Ri_b AND THE STABILITY FACTOR f_h = C_h / C_h,n OF THE REFERENCE CONFIGURATION")
    cfg = reference_config()
    obs = load_obs()
    ds_modis, era5_ml, era5_sl, _ = _read_cache(obs)
    column, surface_modis, _ = era5_forcing(era5_ml, era5_sl, ds_modis, obs)
    runs = {"in situ": (compute_bulk_shf_obs(obs, cfg), "obs", np.asarray(obs["altitude"], dtype=float)),
            "ERA5-MODIS": (compute_bulk_shf_remote(obs, column, surface_modis, cfg, z_ref=era5_ml["z30"]), "rs", era5_ml["z30"])}
    for name, (run, sfx, z) in runs.items():
        Ri = np.asarray(run[f"Ri_{sfx}"], dtype=float)
        z0, zt, Ch = (np.asarray(run[f"{v}_{sfx}"], dtype=float) for v in ("z0", "zt", "Ch"))
        _, Ch_neutral = transfer_coefficients(np.zeros_like(Ri), z, z0, zt, cfg.scheme_name)
        fh = Ch / Ch_neutral
        stable = Ri > 0
        p = np.nanpercentile(Ri, [5, 25, 50, 75, 95])
        print(f"{name:<11} Ri_b percentiles 5/25/50/75/95: " + " / ".join(f"{v:+.2f}" for v in p)
              + f";  stable {stable.sum()}, unstable {(~stable).sum()} of {Ri.size}")
        for label, m in (("stable", stable), ("unstable", ~stable)):
            q = np.nanpercentile(fh[m], [5, 50, 95])
            print(f"            f_h on the {label:<8} segments: median {q[1]:.2f}, 5-95 % {q[0]:.2f} to {q[2]:.2f}")
    runs_obs = [str(r) for r in FIG["run_obs"].values]
    schemes = np.asarray(FIG["H_obs_bulk_all"])[[i for i, r in enumerate(runs_obs) if r.endswith("_ref")]]
    spread = np.nanmax(schemes, axis=0) - np.nanmin(schemes, axis=0)
    print(f"per-segment spread of the in-situ bulk flux across the eight schemes: median {np.median(spread):.2f} W m-2, "
          f"maximum {np.max(spread):.1f} W m-2 (median |flux| {np.median(np.abs(H('H_obs_bulk'))):.1f} W m-2)")


# ── 6  alpha_T on withheld days ──────────────────────────────────────────────

def alpha_on_withheld_days():
    header("6. AMPLITUDE MATCHING ON WITHHELD DAYS: alpha_T fitted on three days (slope 1), tested on the fourth")
    runs = [str(r) for r in FIG["run_obs"].values]
    grid = sorted((float(r.split("ztr")[1]), i) for i, r in enumerate(runs) if "ztr" in r)
    log_a = np.log10([a for a, _ in grid])
    Hs = np.asarray(FIG["H_obs_bulk_all"])[[i for _, i in grid]]            # (alpha, segment), alpha ascending

    def slope_curve(mask):
        curve = np.array([np.polyfit(EC[mask], h[mask], 1)[0] for h in Hs])
        assert np.all(np.diff(curve) > 0), "the slope must rise with alpha_T for the interpolation"
        return curve

    print(f"  {'left out':<9} {'alpha_T fitted':>14}   {'slope on the left-out day':>26}   {'at alpha_T = 1e-1':>17}   "
          f"{'sd of its EC flux':>17}")
    for day in pd.unique(DAY):
        train, test = DAY != day, DAY == day
        fitted = np.interp(1.0, slope_curve(train), log_a)
        on_test = np.interp(fitted, log_a, slope_curve(test))
        print(f"  {day:<9} {10 ** fitted:14.0e}   {on_test:26.1f}   {slope_curve(test)[-1]:17.1f}   "
              f"{EC[test].std():12.1f} W m-2")
    print(f"  all days  {10 ** np.interp(1.0, slope_curve(np.ones(EC.size, bool)), log_a):14.0e}")
    print("  single-day slopes are poorly conditioned where the EC flux hardly varies (small sd)")


# ── 7  measurement height ────────────────────────────────────────────────────

def measurement_height():
    header("7. MEASUREMENT HEIGHT: lower vs upper half of the T-Bird heights, and segments that are not at one level")
    seg = xr.open_dataset(REPO / "data/observations/ar/analysis_ready_polar6_turbulent_fluxes_latest_5km.nc"
                          ).sel(start_time=FIG["start_time"].values)
    z = np.asarray(seg["altitude"], dtype=float)
    med = np.median(z)
    for label, m in ((f"below {med:.0f} m", z <= med), (f"above {med:.0f} m", z > med)):
        print(f"  {label:<12} (n = {m.sum()}, mean {z[m].mean():.0f} m, EC median {np.median(EC[m]):+.1f} W m-2): "
              + "   ".join(f"{name} {b:.1f} / {r:.1f}" for (key, name) in BULK for b, r in [slope_r(EC[m], H(key)[m])]))

    # height range of the T-Bird inside each segment, from the 100 Hz data (10 s running mean)
    tbird = xr.open_dataset(REPO / "data/observations/processed/t-bird_unified/noseboom_unified.nc").sortby("time")
    ranges = []
    for t0, t1 in zip(seg["start_time"].values, seg["end_time"].values):
        h = pd.Series(np.asarray(tbird["h"].sel(time=slice(t0, t1)), dtype=float)).rolling(1000, center=True).mean()
        ranges.append(h.max() - h.min())
    ranges = np.asarray(ranges)
    level = ranges <= 20.0
    print(f"  T-Bird height range within a segment (10 s running mean): median {np.median(ranges):.0f} m, maximum {ranges.max():.0f} m; "
          f"more than 20 m in {(~level).sum()} segment(s)"
          + "".join(f", {pd.Timestamp(t):%d %b %H:%M} ({r:.0f} m, mean height {zz:.0f} m)"
                    for t, r, zz in zip(seg["start_time"].values[~level], ranges[~level], z[~level])))
    print(f"  without them (n = {level.sum()}): "
          + "   ".join(f"{name} {b:.1f} / {r:.1f}" for (key, name) in BULK for b, r in [slope_r(EC[level], H(key)[level])])
          + f";  maximum mean height {z[level].max():.0f} m")


if __name__ == "__main__":
    more_metrics()
    regression_dilution()
    dependence()
    satellite_input()
    stability()
    alpha_on_withheld_days()
    measurement_height()
    print()

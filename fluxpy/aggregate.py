"""
fluxpy.aggregate
================
Aggregation pipeline: raw high-frequency leg data → analysis-ready DataFrame / xr.Dataset.

The pipeline mirrors the aggregation loop in
``book/notebooks/processing/analysis-ready_polar6_legs.ipynb`` but is fully
parameterised, importable, and callable from scripts or other notebooks.

Quick start
-----------
    from fluxpy.aggregate import AggConfig, process_legs, to_xarray

    cfg = AggConfig(bin_length_km=5.0, min_samples=5000)
    df  = process_legs(legs, df_legs_meta,
                       ds_radiation, ds_tbird, ds_polar6,
                       ds_sic, ds_track, ds_ar,
                       cfg=cfg)
    ds  = to_xarray(df)
    ds.to_netcdf("output.nc")
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal
from scipy.signal import correlate, correlation_lags
from tqdm import tqdm

from .flux_processing import (
    get_tke,
    add_u_star,
    get_cov,
    add_monin_obukhov_length,
    add_roughness_length,
)
from .meteorology import add_theta

# ── fluxpy sibling modules ────────────────────────────────────────────────────
from .stability import bulk_richardson_2pt
from .util      import potential_temperature
from .qc        import qc_leg, classify_leg_soft, classify_leg_simple

__all__ = ["AggConfig", "process_legs", "to_xarray"]


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class AggConfig:
    """Aggregation configuration.

    Attributes
    ----------
    bin_length_km       : Spatial bin length [km].  Changing this value
                          automatically adjusts how the leg distance coordinate
                          is partitioned.  Default 5.0.
    bin_overlap_km      : Overlap between adjacent bins [km].  0 (default) =
                          non-overlapping.  >0 triggers a sliding-window loop
                          instead of ``groupby_bins``.
    min_samples         : Minimum number of high-frequency samples required to
                          keep a bin.  Bins below this threshold are skipped.
                          Default 3000 (at 100 Hz → 30 s minimum).
    fs                  : Sampling rate of the raw high-frequency data [Hz].
                          Default 100.0.
    highpass            : Apply Butterworth high-pass filter before EC
                          calculations.  Default True.
    aerosol_zscore_thresh: Z-score threshold used to reject aerosol outliers
                          before bin-mean.  Default 5.0.
    compute_ri_b        : Compute the classic 2-pt bulk Richardson number
                          ``ri_b_mean`` (surface → T-Bird, manuscript
                          Eq. rib) plus the three-level θ/z diagnostics
                          (``theta_*``, ``z_*``, ``T_*`` columns).
                          Requires ds_tbird and ds_polar6.  Default True.
    kt19_var            : Variable name in ds_radiation for the infrared
                          surface temperature sensor.  Default ``'KT19'``.
    aerosol_vars        : List of aerosol variable names to read from ds_ar.
    verbose             : Print per-bin progress messages.  Default False.
    resume_from         : Path to a previously written output file (netCDF or
                          CSV) produced by this pipeline.  When set and the
                          file exists, bins whose ``start_time`` is already
                          present in that file are skipped and the prior rows
                          are carried forward unchanged.  This allows
                          interrupted runs to pick up where they left off
                          without reprocessing any data.  Default ``None``
                          (always process everything).
    """

    bin_length_km:        float       = 5.0
    bin_overlap_km:       float       = 0.0
    min_samples:          int         = 3000
    fs:                   float       = 100.0
    highpass:             bool        = True
    aerosol_zscore_thresh: float      = 5.0
    compute_ri_b:         bool        = True
    kt19_var:             str         = "KT19"
    aerosol_vars:         list[str]   = field(
        default_factory=lambda: ["POPS_sum_b2-b15", "CCN", "Ncpc_c2s"]
    )
    verbose:              bool        = False
    resume_from:          str | None  = None
    T_thresh_open:        float       = -1.8   # °C  KT19 > this  → open water
    T_thresh_thick:       float       = -20.0  # °C  KT19 ≤ this  → thick ice (rest = thin ice)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _scalar(x) -> float:
    """Convert xarray DataArray / numpy scalar to a Python float."""
    try:
        return float(x.item())
    except Exception:
        try:
            return float(x)
        except Exception:
            return np.nan


def _mean_slice(ds: xr.Dataset, start, end, vars_: list[str]) -> dict:
    """Return a dict of per-variable scalar means over a time slice."""
    sel = ds[vars_].sel(time=slice(start, end))
    return {v: _scalar(sel[v].mean(skipna=True)) for v in vars_}


def _robust_aerosol_means(
    ds_ar: xr.Dataset,
    aerosol_vars: list[str],
    start,
    end,
    z_thresh: float = 5.0,
) -> dict:
    """Z-score filtered per-variable means.  Output keys: CCN, POPS, Ncpc."""
    out = {"CCN": np.nan, "POPS": np.nan, "Ncpc": np.nan}
    if ds_ar is None:
        return out
    available = [v for v in aerosol_vars if v in ds_ar]
    if not available:
        return out
    aero    = ds_ar[available].sel(time=slice(start, end))
    mapping = {"CCN": "CCN", "POPS_sum_b2-b15": "POPS", "Ncpc_c2s": "Ncpc"}
    for raw_name, out_name in mapping.items():
        if raw_name not in aero:
            continue
        da         = aero[raw_name]
        mu, sd     = da.mean(skipna=True), da.std(skipna=True)
        mask       = xr.where(sd > 0, np.abs((da - mu) / sd) < z_thresh, True)
        out[out_name] = _scalar(da.where(mask).mean(skipna=True))
    return out


def _integral_length_scale(T_vals: np.ndarray, w_vals: np.ndarray, tas_mean: float, fs: float) -> float:
    """Compute the integral length scale from the autocorrelation of w'T'.

    The integral length scale L_int is the lag distance at which the
    normalised autocorrelation of the w'T' product drops to 1/e of its
    zero-lag value (Lenschow et al. 1994).

    Parameters
    ----------
    T_vals  : 1-D float array of air temperature fluctuations (or raw T).
    w_vals  : 1-D float array of vertical wind fluctuations (or raw w).
    tas_mean: Mean true airspeed [m/s] — used to convert time lags → distance.
    fs      : Sampling frequency [Hz].

    Returns
    -------
    L_int : float
        Integral length scale [m].  Falls back to NaN if the autocorrelation
        has no 1/e crossing or if there are too few valid samples.
    """
    # Remove NaNs (keep indices aligned)
    mask = ~np.isnan(T_vals) & ~np.isnan(w_vals)
    T_clean = T_vals[mask]
    w_clean = w_vals[mask]

    if len(T_clean) < 100:
        return np.nan

    # Detrend to obtain fluctuation components
    T_prime = signal.detrend(T_clean, type="linear")
    w_prime = signal.detrend(w_clean, type="linear")

    wt = T_prime * w_prime
    if len(wt) < 10:
        return np.nan

    # Full autocorrelation, then keep positive-lag half
    acf  = correlate(wt, wt, mode="full")
    lags = correlation_lags(len(wt), len(wt), mode="full")

    pos          = lags >= 0
    acf_pos      = acf[pos]
    lags_dist_m  = lags[pos] * (1.0 / fs) * tas_mean  # samples → metres

    # Find first 1/e crossing
    threshold  = acf_pos[0] / np.e
    crossings  = np.where(np.diff(np.sign(acf_pos - threshold)))[0]

    if len(crossings) == 0:
        return np.nan

    return float(lags_dist_m[crossings[0]])


def _build_distance_bins(leg_len_km: float, bin_step: float, bin_length: float) -> list[tuple[float, float]]:
    """Return a list of (left, right) bin edges in km.

    For non-overlapping bins ``bin_step == bin_length``.
    For overlapping bins ``bin_step < bin_length``.
    """
    bins = []
    left = 0.0
    while left + bin_length <= leg_len_km + 1e-9:
        bins.append((left, left + bin_length))
        left += bin_step
    return bins


# ---------------------------------------------------------------------------
# Main aggregation function
# ---------------------------------------------------------------------------

def process_legs(
    legs:         list[xr.Dataset],
    df_legs_meta: pd.DataFrame,
    ds_radiation: xr.Dataset,
    ds_tbird:     xr.Dataset,
    ds_polar6:    xr.Dataset | None,
    ds_sic:       xr.Dataset,
    ds_track:     xr.Dataset | None,
    ds_ar:        xr.Dataset | None,
    cfg:          AggConfig = AggConfig(),
) -> pd.DataFrame:
    """Aggregate high-frequency leg data into analysis-ready bins.

    Parameters
    ----------
    legs          : List of xr.Dataset leg slices (one per flight leg) sliced
                    from the **T-Bird** noseboom dataset.  All EC / turbulence
                    / altitude / position quantities are derived from these.
                    Time windows are determined by the Polar-6 flight
                    segmentation (both platforms flew together).
    df_legs_meta  : DataFrame of leg metadata (start/end times, platform) as
                    returned by ``helper.process_flight_segments`` ``['df_legs']``
                    *filtered to Polar-6 legs only* (used only for time
                    window boundaries and metadata columns).
    ds_radiation  : Radiation dataset (Polar-6 mounted) containing at least
                    ``F_dnw_sol``, ``F_upw_sol``, ``F_dnw_ter``,
                    ``F_upw_ter``, and the KT19 variable
                    (cf. ``cfg.kt19_var``).  Only radiation quantities are
                    drawn from here.
    ds_tbird      : T-Bird nose-boom dataset.  Used as the source of all
                    EC / turbulence / altitude / position data (via ``legs``)
                    and also directly for the 3-pt Richardson.
    ds_polar6     : Polar-6 nose-boom dataset, or ``None``.  Used **only** for
                    the aircraft-level diagnostics (``T_polar6``,
                    ``theta_polar6``, ``z_polar6``); these are NaN when absent.
                    The EC flux and Ri_b do not depend on it.
    ds_sic        : Sea-ice concentration gridded dataset with coords
                    ``time``, ``lat``, ``lon`` and variable ``sic``.
    ds_track      : Master track dataset with coords ``time``,
                    ``latitude``, ``longitude``, or ``None`` — the SIC
                    colocation then uses the T-Bird's own ``lat``/``lon``.
    ds_ar         : Aerosol dataset (cf. ``cfg.aerosol_vars``), or ``None``
                    (aerosol columns are NaN; not used in the paper).
    cfg           : :class:`AggConfig` controlling all tuning parameters.

    Returns
    -------
    pd.DataFrame
        One row per accepted bin.  Column layout:
        leg metadata | EC fluxes | radiation | aerosols | sea ice |
        QC metrics | Richardson diagnostics
    """
    warnings.filterwarnings("ignore")

    step_km = cfg.bin_length_km - cfg.bin_overlap_km
    if step_km <= 0:
        raise ValueError(
            f"bin_overlap_km ({cfg.bin_overlap_km}) must be < bin_length_km ({cfg.bin_length_km})"
        )

    # ── Resume: load previously processed bins ────────────────────────────
    _already_done: set[pd.Timestamp] = set()
    leg_aggregated_data: list[dict]  = []

    if cfg.resume_from and os.path.isfile(cfg.resume_from):
        try:
            _ext = os.path.splitext(cfg.resume_from)[1].lower()
            if _ext in (".nc", ".nc4", ".netcdf"):
                with xr.open_dataset(cfg.resume_from) as _ds_prior:
                    _prior_df = _ds_prior.to_dataframe().reset_index()
            else:  # CSV (tab or comma)
                _sep = "\t" if _ext == ".tsv" else ","
                _prior_df = pd.read_csv(
                    cfg.resume_from, sep=_sep,
                    parse_dates=["start_time", "end_time"],
                )
            leg_aggregated_data = _prior_df.to_dict("records")
            _already_done = set(pd.to_datetime(_prior_df["start_time"]))
            print(
                f"Resume: loaded {len(leg_aggregated_data)} previously processed bins "
                f"from {cfg.resume_from}"
            )
        except Exception as _exc:
            warnings.warn(
                f"Could not load resume file '{cfg.resume_from}': {_exc}. "
                "Starting from scratch."
            )

    N_total   = 0
    N_skipped = 0

    for i in tqdm(range(len(df_legs_meta)), total=len(df_legs_meta), desc="Aggregating legs"):
        leg_ds  = legs[i].copy()
        meta    = df_legs_meta.iloc[i]

        # Guard: skip legs where the T-Bird has no data in this window
        if leg_ds.time.size == 0:
            if cfg.verbose:
                print(f"  Skipping leg {i} — empty dataset (outside T-Bird coverage?)")
            continue

        # ── Distance coordinate ───────────────────────────────────────────
        leg_ds = add_theta(leg_ds)
        dist   = leg_ds["tas"].mean() * (leg_ds["time"] - leg_ds["time"][0]).dt.seconds / 1000.0
        leg_len_km = _scalar(dist[-1] - dist[0])

        leg_ds[cfg.kt19_var + "_interp"] = ds_radiation[cfg.kt19_var].interp_like(leg_ds) + 273.15
        leg_ds["distance"] = dist
        leg_ds = (
            leg_ds
            .assign_coords(distance=leg_ds["distance"])
            .swap_dims({"time": "distance"})
        )

        # ── Build bins (supports overlap) ──────────────────────────────────
        bins = _build_distance_bins(leg_len_km, step_km, cfg.bin_length_km)

        for (bin_left, bin_right) in bins:
            N_total += 1
            ds_leg_bin = leg_ds.sel(distance=slice(bin_left, bin_right))
            leg_data   = ds_leg_bin.dropna(dim="distance", how="any")
            N_samples  = int(leg_data["distance"].size)

            if N_samples < cfg.min_samples:
                if cfg.verbose:
                    print(f"  Skipping bin [{bin_left:.1f}, {bin_right:.1f}] km — "
                          f"only {N_samples} samples (< {cfg.min_samples})")
                continue

            start = pd.to_datetime(leg_data["time"][0].item())
            end   = pd.to_datetime(leg_data["time"][-1].item())

            # Check if altitude is sufficiently constant within the bin; if not, skip to avoid spurious fluxes
            alt_range = float(leg_data["h"].max() - leg_data["h"].min())
            # print(f"  Bin [{bin_left:.1f}, {bin_right:.1f}] km: altitude range = {alt_range:.1f} m")
            if alt_range > 100:  # Threshold for "sufficiently constant" (tunable)
                if cfg.verbose:
                    print(f"    Skipping due to large altitude variation (> 100 m)")
                continue

            # ── Resume skip ───────────────────────────────────────────────
            if start in _already_done:
                N_skipped += 1
                if cfg.verbose:
                    print(f"  Resuming: skipping bin [{bin_left:.1f}, {bin_right:.1f}] km "
                          f"(start={start} already in output)")
                continue

            z_ref = _scalar(leg_data["h"].mean())

            # ── EC fluxes ─────────────────────────────────────────────────
            tke       = _scalar(get_tke(leg_data, highpass=cfg.highpass).mean())
            ustar     = _scalar(add_u_star(leg_data)["u_star"].mean())
            # w'theta' covariance (manuscript Eq. Hec): potential temperature is
            # the conserved variable; band-limited to 0.1-20 Hz (see get_cov).
            _, hf     = get_cov(leg_data, "theta", heat_flux=True, time_dim="distance", highpass=True)

            hf_scalar = _scalar(hf.mean()) if hasattr(hf, "mean") else float(hf)
            mol       = _scalar(add_monin_obukhov_length(leg_data)["monin_obukhov_length"].mean())
            rz0       = _scalar(add_roughness_length(leg_data, z_ref)["z0"].mean())

            # ── Radiation ─────────────────────────────────────────────────
            rad_means = _mean_slice(
                ds_radiation, start, end,
                ["F_dnw_sol", "F_upw_sol", "F_dnw_ter", "F_upw_ter", cfg.kt19_var],
            )
            rad_means["t_kt19"] = rad_means.pop(cfg.kt19_var)

            # ── Aerosols (z-score filtered) ───────────────────────────────
            aerosol_means = _robust_aerosol_means(
                ds_ar, cfg.aerosol_vars, start, end, z_thresh=cfg.aerosol_zscore_thresh
            )

            # ── Sea ice ───────────────────────────────────────────────────
            try:
                if ds_track is not None:
                    track_sel = ds_track.sel(time=slice(start, end))
                    _t, _la, _lo = track_sel.time, track_sel.latitude, track_sel.longitude
                else:  # no master track shipped: T-Bird positions, 1 Hz
                    _pos = leg_data[["lat", "lon"]].swap_dims({"distance": "time"}).isel(
                        time=slice(None, None, int(cfg.fs)))
                    _t, _la, _lo = _pos.time, _pos.lat, _pos.lon
                ds_sel_sic = ds_sic.sel(time=_t, lat=_la, lon=_lo, method="nearest")
                sic = _scalar(ds_sel_sic["sic"].mean())
            except Exception:
                sic = np.nan

            # fraction of the bin where KT19 (°C) > −5°C  →  must use raw
            # time-series, NOT the scalar mean stored in rad_means
            _kt19_ts  = ds_radiation[cfg.kt19_var].sel(time=slice(start, end))
            lead_frac = float((_kt19_ts > -5).mean(skipna=True))

            # ── fractional surface type from KT19 time series ─────────────
            _kt19_arr   = _kt19_ts.values.astype(float)
            _finite_mask = np.isfinite(_kt19_arr)
            _n_fin       = int(_finite_mask.sum())
            if _n_fin > 0:
                _kt = _kt19_arr[_finite_mask]
                frac_open      = float((_kt > cfg.T_thresh_open).mean())
                frac_thick_ice = float((_kt <= cfg.T_thresh_thick).mean())
                frac_thin_ice  = float(1.0 - frac_open - frac_thick_ice)
            else:
                frac_open = frac_thin_ice = frac_thick_ice = np.nan

            # ── Integral length scale & Lenschow relative error ────────────
            T_arr   = leg_data["T"].values.flatten()
            w_arr   = leg_data["w"].values.flatten()
            tas_bin = float(leg_data["tas"].mean(skipna=True))
            L_int   = _integral_length_scale(T_arr, w_arr, tas_bin, cfg.fs)

            # Averaging distance = bin length in metres; U ≈ TAS
            bin_len_m = (bin_right - bin_left) * 1e3
            if np.isfinite(L_int) and tas_bin > 0 and bin_len_m > 0:
                T_avg_bin         = bin_len_m / tas_bin          # [s]
                lenschow_rel_err  = float(np.clip(
                    np.sqrt(2.0 * L_int / (tas_bin * T_avg_bin)), 0.0, 2.0
                ))
            else:
                lenschow_rel_err = np.nan

            # ── QC ────────────────────────────────────────────────────────
            qc_results = qc_leg(
                w=leg_data.w.values,
                theta=leg_data.theta.values,
                u=leg_data.u.values,
                v=leg_data.v.values,
                fs=cfg.fs,
                x_dist=leg_data["distance"].values,
            )
            bin_length_km  = float(bin_right - bin_left)
            qc_soft        = classify_leg_soft(qc_results, bin_length_km, kind="heat")
            qc_simple      = classify_leg_simple(qc_results, bin_length_km)

            # ── Richardson number + level diagnostics (classic 2-pt Ri_b) ─
            # Keep unaveraged slices for std computations; .mean() copies are
            # used only where a scalar input is required (Richardson, row dict).
            ds_tbird_slice = ds_tbird.sel(time=slice(start, end))
            ds_rad_slice   = ds_radiation.sel(time=slice(start, end))

            ri_diag: dict = {"ri_b_mean": np.nan}
            if cfg.compute_ri_b:
                try:
                    ds_tbird_sel  = ds_tbird_slice.mean()
                    ds_rad_sel    = ds_rad_slice.mean()

                    # surface (KT-19; T-Bird pressure as surface surrogate)
                    T_sfc = float(ds_rad_sel[cfg.kt19_var])
                    p_tb  = float(ds_tbird_sel["p"])
                    T_tb  = float(ds_tbird_sel["T"])
                    th_sfc = float(potential_temperature(T_sfc, p_tb))
                    th_tb  = float(potential_temperature(T_tb, p_tb))
                    z_tb   = float(ds_tbird_sel["h"])

                    # aircraft level: diagnostic only, NaN without the Polar 6 file
                    T_p6 = th_p6 = z_p6 = np.nan
                    if ds_polar6 is not None:
                        ds_polar6_sel = ds_polar6.sel(time=slice(start, end)).mean()
                        T_p6  = float(ds_polar6_sel["T"])
                        th_p6 = float(potential_temperature(T_p6, float(ds_polar6_sel["p"])))
                        z_p6  = float(ds_polar6_sel["h"])

                    # manuscript Eq. (rib): surface (no-slip) → T-Bird level
                    ri_b = float(bulk_richardson_2pt(
                        dtheta=th_tb - th_sfc,
                        dz=z_tb,
                        du=float(ds_tbird_sel["u"]),
                        dv=float(ds_tbird_sel["v"]),
                        theta_ref=0.5 * (th_sfc + th_tb),
                    ))
                    ri_diag = {
                        "ri_b_mean":     ri_b,
                        "T_surface":     T_sfc, "T_tbird":     T_tb,  "T_polar6":     T_p6,
                        "theta_surface": th_sfc, "theta_tbird": th_tb, "theta_polar6": th_p6,
                        "z_surface":     0.0,    "z_tbird":     z_tb,  "z_polar6":     z_p6,
                    }
                except Exception as exc:
                    if cfg.verbose:
                        print(f"  Ri_b failed for bin [{bin_left:.1f}, {bin_right:.1f}]: {exc}")

            row: dict = {
                # ── identification ──────────────────────────────────────
                "leg":             N_total,
                "start_time":      start,
                "end_time":        end,
                "platform":        meta["platform"],
                # ── meteorology ─────────────────────────────────────────
                "altitude":        _scalar(leg_data["h"].mean()),
                "T_mean":          _scalar(leg_data["T"].mean()),
                "p_mean":          _scalar(leg_data["p"].mean()),
                "u_mean":          _scalar(leg_data["u"].mean()),
                "v_mean":          _scalar(leg_data["v"].mean()),
                "w_mean":          _scalar(leg_data["w"].mean()),
                "latitude":        _scalar(leg_data["lat"].mean()),
                "longitude":       _scalar(leg_data["lon"].mean()),
                # ── EC fluxes ───────────────────────────────────────────
                "sensible_heat_flux":   hf_scalar if qc_simple["verdict"] != "discarded" else np.nan,
                "tke_mean":             tke,
                "u_star":               ustar,
                "monin_obukhov_length": mol,
                "roughness_length":     rz0,
                # ── sea ice ─────────────────────────────────────────────
                "sic":             sic,
                "lead_frac":       lead_frac,
                "frac_open":       frac_open,
                "frac_thin_ice":   frac_thin_ice,
                "frac_thick_ice":  frac_thick_ice,
                # ── QC ──────────────────────────────────────────────────
                "N_samples":       N_samples,
                "leg_length_km":   bin_length_km,
                "slope_th":        qc_results["slope_th"],
                "slope_w":         qc_results["slope_w"],
                "tv_og":           qc_results["tv_og"],
                "R2":              qc_results["R2"],
                "max_dev":         qc_results["max_dev"],
                # soft (weighted) classifier
                "qc_verdict":      qc_soft["verdict"],
                "qc_score":        qc_soft["score"],
                # simple (manuscript 3-criterion) classifier
                "qc_verdict_simple":  qc_simple["verdict"],
                "qc_n_failed_simple": qc_simple["n_failed"],
                "qc_slope_w":              qc_simple["metrics"]["slope_w"],
                "qc_og_fft_plateau_err":   qc_simple["metrics"]["og_fft_plateau_err"],
                "qc_R2":                   qc_simple["metrics"]["R2"],
                "qc_snr_fwC":              qc_simple["metrics"]["snr_fwC"],
                # ── Lenschow error metrics ──────────────────────────────
                "L_int":               L_int,
                "lenschow_rel_error":  lenschow_rel_err,
                "ri_b_std":  0.0,
                # std variables — computed from *unaveraged* time slices
                "T_sfc_std": float(ds_rad_slice[cfg.kt19_var].std(skipna=True)),
                "ws_mean":   float(np.sqrt(ds_tbird_slice["u"].mean(skipna=True)**2 + ds_tbird_slice["v"].mean(skipna=True)**2)),
                "ws_std":    float(np.sqrt(ds_tbird_slice["u"].std(skipna=True)**2  + ds_tbird_slice["v"].std(skipna=True)**2)),
                "T_air_std": float(ds_tbird_slice["T"].std(skipna=True)),
                "delta_theta_std": float((ds_tbird_slice["T"] - ds_rad_slice[cfg.kt19_var]).std(skipna=True)),
                # ── radiation ───────────────────────────────────────────
                **rad_means,
                # ── aerosols ────────────────────────────────────────────
                **aerosol_means,
                # ── Ri_b (2-pt, Eq. rib) + three-level θ/z diagnostics ──
                **ri_diag,
            }
            leg_aggregated_data.append(row)

    N_new = len(leg_aggregated_data) - len(_already_done)
    print(
        f"\nAggregation complete.  "
        f"bins evaluated: {N_total}  |  "
        f"resumed (skipped): {N_skipped}  |  "
        f"newly accepted: {N_new}  |  "
        f"total in output: {len(leg_aggregated_data)}"
    )
    return pd.DataFrame(leg_aggregated_data)


# ---------------------------------------------------------------------------
# Post-processing / export
# ---------------------------------------------------------------------------

def to_xarray(df: pd.DataFrame) -> xr.Dataset:
    """Convert the aggregation DataFrame to an xr.Dataset.

    * Sets ``start_time`` as the primary dimension.
    * Derives ``delta_T``, ``class`` (sic classes), and clips
      ``roughness_length`` to 10 m (physical upper guard).
    * Drops non-numeric / non-serialisable columns (e.g. ``class``) that
      would break netCDF serialisation.

    Parameters
    ----------
    df : DataFrame returned by :func:`process_legs`.

    Returns
    -------
    xr.Dataset indexed by ``start_time``.
    """
    df = df.copy()

    # Derived quantities
    if "T_mean" in df.columns and "t_kt19" in df.columns:
        df["delta_T"] = df["T_mean"] - df["t_kt19"]

    if "roughness_length" in df.columns:
        df["roughness_length"] = df["roughness_length"].where(df["roughness_length"] < 10.0)

    if "sic" in df.columns:
        df["sic_class"] = pd.cut(
            df["sic"],
            bins=[-1, 15, 80, 100],
            labels=["open_water", "marginal_ice_zone", "closed_ice"],
        ).astype(str)

    # Encode verdict strings as integers so they survive netCDF serialisation.
    # 0 = accepted, 1 = tolerable, 2 = discarded  (stored with flag_values attr)
    _VERDICT_INT = {"accepted": 0, "tolerable": 1, "discarded": 2}
    for col in ("qc_verdict", "qc_verdict_simple"):
        if col in df.columns:
            # fillna(-1) before casting: pandas nullable Int8 is not netCDF-compatible
            df[col + "_int"] = df[col].map(_VERDICT_INT).fillna(-1).astype(np.int8)
            df = df.drop(columns=[col])

    ds = df.set_index("start_time").to_xarray().sortby("start_time")

    # Attach human-readable flag metadata to the encoded verdict variables
    _flag_attrs = dict(
        flag_values=[0, 1, 2],
        flag_meanings="accepted tolerable discarded",
        missing_value=np.int8(-1),
    )
    for col in ("qc_verdict_int", "qc_verdict_simple_int"):
        if col in ds:
            ds[col].attrs.update(_flag_attrs)

    # Drop any remaining object-dtype columns that aren't netCDF-serialisable
    drop = [v for v in ds.data_vars if ds[v].dtype == object]
    if drop:
        ds = ds.drop_vars(drop)

    return ds

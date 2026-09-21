#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_bulk_and_ec_fluxes.py
==========================
Pre-computation pipeline for the BACSAM-II bulk SHF sensitivity analysis.
Saves sensitivity_figure_data.nc and sensitivity_summary_table.csv which are
read by figures_manuscript/*.py, manuscript_numbers.py and make_manuscript_tables.py.

All physics come exclusively from **fluxpy** (bulk, stability, util, sensitivity).

Sensitivity axes
----------------
  A  MOST scheme              (GLGS20, BD, HB88, BH91, CB05, GAFGP07, ZE07-*)
  B  zt_ratio                 (scalar roughness ratio alpha_T: 1e-10 … 1e-1; reference 1e-1)
  C  Roughness set            (lit-lo / lit-hi / SHEBA)
  C' Constant roughness       (z0_constant; surface-type-independent)
  D  IST classification threshold (T_thresh_thick combos)
  E  T_sfc source             (ERA5 skt [MODIS gap-fill]  /  MODIS IST)

Atmospheric columns for reanalysis bulk:
  ERA5 ML  – model levels 137 (~30 m) / 136 (~50 m AGL)
  CARRA    – height levels 30 m / 75 m AGL

Perturbation sweeps (output variables H_obs_dTsfc, H_obs_dTair, H_obs_dRi, H_obs_dU):
  P6 in-situ, ERA5 ML column (ERA5 skt), CARRA column (CARRA skt + MODIS IST)

Usage
-----
    python scripts/calc_bulk_and_ec_fluxes.py          # full run
    python scripts/calc_bulk_and_ec_fluxes.py --verbose

Physics sign convention
-----------------------
    H > 0  →  upward flux (surface to atmosphere); consistent with EC archive.
    H_bulk = ρ cp Ch U (θ_sfc − θ_air)

Author: Joshua Müller, BACSAM-II 2024 / 2026
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from fluxpy.bulk import (  # noqa: E402
    compute_bulk_shf_obs,
    compute_bulk_shf_remote,
    reference_config,
)
from fluxpy.sensitivity import sensitivity_matrix, run_sensitivity, skill_metrics
from fluxpy.stability import Cd_Ch_noniterative, bulk_richardson_2pt, effective_wind_speed
from fluxpy.util import (
    potential_temperature,
    air_density,
    classify_surface,
    assign_roughness,
    CP,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
LOG_DATE   = "%Y-%m-%dT%H:%M:%S"
logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE, level=logging.INFO)
log = logging.getLogger("calc_bulk_and_ec_fluxes")

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
_AR   = _REPO_ROOT / "data" / "observations" / "ar"
_SAT  = _REPO_ROOT / "data" / "observations" / "processed" / "satellite_unified"
_ERA5_SL = _REPO_ROOT / "data" / "reanalysis" / "era5" / "ERA5_single_levels"
_ERA5_ML = _REPO_ROOT / "data" / "reanalysis" / "era5" / "ERA5_model_levels"
_CARRA   = _REPO_ROOT / "data" / "reanalysis" / "carra"
_OUT     = _REPO_ROOT / "data" / "model_output"

# The paper analyses the four low-level flight days 13, 15, 17 and 23 April.
# 25 April: 32 segments pass the < 50 m filter, but at 47 m mean altitude
# (analysed days: 21-29 m), and the merged SIC product is unusable that day
# (100 % everywhere). 26 April has no segment below 50 m; listed for clarity.
EXCLUDE = [pd.Timestamp("2024-04-25").date(), pd.Timestamp("2024-04-26").date()]
LAT_SLICE = slice(76.0, 87.0)
LON_SLICE = slice(4.0, 36.0)


# ===========================================================================
# Data loaders
# ===========================================================================

def load_obs() -> xr.Dataset:
    """Load and filter the P6 turbulent-flux archive."""
    fp = _AR / "analysis_ready_polar6_turbulent_fluxes_latest_5km.nc"
    ds = xr.open_dataset(fp)
    ds["ws_mean"] = np.sqrt(ds["u_mean"] ** 2 + ds["v_mean"] ** 2)
    ds = ds.where(ds["altitude"] < 50, drop=True)
    ds = ds.sel(start_time=~ds["start_time"].dt.date.isin(EXCLUDE))
    log.info("P6 legs: %d", ds.sizes["start_time"])
    return ds


def load_modis(ds_obs: xr.Dataset) -> xr.Dataset:
    """Load MODIS IST nearest-neighbour colocated to P6 legs."""
    fp = _SAT / "MODIS_Aqua_Sea_Ice_Extent_and_IST_Daily_L3_Global_4km_EASE-Grid_Day_V061.nc"
    ds_raw = xr.open_dataset(fp)
    if "time" in ds_raw.dims:
        ds_raw = ds_raw.rename({"time": "start_time"})
    t_da   = xr.DataArray(pd.to_datetime(ds_obs["start_time"].values), dims="leg")
    lat_da = xr.DataArray(np.asarray(ds_obs["latitude"]),               dims="leg")
    lon_da = xr.DataArray(np.asarray(ds_obs["longitude"]),              dims="leg")
    ds_mod = ds_raw.sel(lat=lat_da, lon=lon_da, start_time=t_da, method="nearest")
    ds_mod["start_time"] = ds_obs["start_time"].values
    # mask physically implausible values (fill / cloud / land)
    ist    = ds_mod["IST"]
    valid  = (ist > 150) & (ist < 280)
    # NOTE: a +3 K adjustment of cold (< -10 °C) MODIS IST, suggested by buoy
    # comparisons, was tested and is deliberately NOT applied; the bulk flux is
    # strongly sensitive to IST, so any such correction must be introduced as a
    # documented sensitivity experiment rather than silently.
    ds_mod = ds_mod.assign({"IST": ist.where(valid)})
    log.info("MODIS IST: %d/%d legs valid", int(valid.sum()), ds_obs.sizes["start_time"])
    return ds_mod


def load_era5_ml(ds_obs: xr.Dataset) -> dict:
    """
    Colocate ERA5 model-level data to the P6 legs.

    Standard-atmosphere geometric altitudes of the lowest L137 levels
    (ECMWF L137 model level definitions): L137 = 10.0 m, L136 = 31.0 m,
    L135 = 53.9 m. The bulk column uses L136 (~31 m; closest to the mean
    T-Bird altitude of 24 m and the 30 m stated in the manuscript); Ri_b
    is the classic 2-pt bulk Richardson number surface → L136. L135
    (~54 m) and L137 (10 m) are retained as diagnostics.
    """
    era5 = xr.open_dataset(_ERA5_ML / "era5_ml.nc")
    t_da   = xr.DataArray(pd.to_datetime(ds_obs["start_time"].values), dims="leg")
    lat_da = xr.DataArray(np.asarray(ds_obs["latitude"]),               dims="leg")
    lon_da = xr.DataArray(np.asarray(ds_obs["longitude"]),              dims="leg")

    def _lev(k):
        return era5.sel(model_level=k).sel(
            time=t_da, latitude=lat_da, longitude=lon_da, method="nearest")

    L137, L136, L135 = _lev(137), _lev(136), _lev(135)
    n = ds_obs.sizes["start_time"]
    return {
        # column level for the bulk fluxes (L136, z ≈ 31 m)
        "T30_K": L136["t"].values.astype(float),
        "ws30":  np.hypot(L136["u"].values, L136["v"].values).astype(float),
        "z30":   np.full(n, 30.96),
        # upper level (L135, z ≈ 54 m) — diagnostic only
        "T50_K": L135["t"].values.astype(float),
        "ws50":  np.hypot(L135["u"].values, L135["v"].values).astype(float),
        "z50":   np.full(n, 53.92),
        # lowest model level (L137, z ≈ 10 m) — diagnostic only
        "T10_K": L137["t"].values.astype(float),
        "ws10":  np.hypot(L137["u"].values, L137["v"].values).astype(float),
        "z10":   np.full(n, 10.0),
    }


def load_era5_sl(ds_obs: xr.Dataset) -> dict:
    """Load ERA5 single-level skt and avg_ishf colocated to P6 legs."""
    t_da   = xr.DataArray(pd.to_datetime(ds_obs["start_time"].values), dims="leg")
    lat_da = xr.DataArray(np.asarray(ds_obs["latitude"]),               dims="leg")
    lon_da = xr.DataArray(np.asarray(ds_obs["longitude"]),              dims="leg")
    ds_inst = xr.open_dataset(_ERA5_SL / "data_stream-oper_stepType-instant.nc")
    skt_K = ds_inst["skt"].sel(
        valid_time=t_da, latitude=lat_da, longitude=lon_da, method="nearest"
    ).values.astype(float)
    surface_pressure = ds_inst["sp"].sel(
        valid_time=t_da, latitude=lat_da, longitude=lon_da, method="nearest"
    ).values.astype(float)
    ds_avg = xr.open_dataset(_ERA5_SL / "data_stream-oper_stepType-avg.nc")
    avg_ishf = ds_avg["avg_ishf"].sel(
        valid_time=t_da, latitude=lat_da, longitude=lon_da, method="nearest"
    ).values.astype(float)
    log.info("ERA5 skt: mean=%.2f K  (%d/%d valid)",
             np.nanmean(skt_K), int(np.isfinite(skt_K).sum()), ds_obs.sizes["start_time"])
    return {"skt_K": skt_K, "avg_ishf": avg_ishf, "sp_Pa": surface_pressure}


def load_carra(ds_obs: xr.Dataset) -> dict:
    """Load CARRA surface and height-level data colocated to P6 legs."""
    t_da   = xr.DataArray(pd.to_datetime(ds_obs["start_time"].values), dims="leg")
    lat_da = xr.DataArray(np.asarray(ds_obs["latitude"]),               dims="leg")
    lon_da = xr.DataArray(np.asarray(ds_obs["longitude"]),              dims="leg")
    ds_sf = (
        xr.open_dataset(_CARRA / "carra_east_latlon0p05.nc")[["skt", "sp"]]
        .sel(lat=LAT_SLICE, lon=LON_SLICE)
        .load()
    )
    ds_hl = xr.open_dataset(_CARRA / "height_level" / "carra_latlon0p05.nc").sel(
        lat=LAT_SLICE, lon=LON_SLICE
    )
    hl30 = ds_hl.sel(heightAboveGround=30.0)[["t", "ws", "pres"]].load()
    hl75 = ds_hl.sel(heightAboveGround=75.0)[["t", "ws", "pres"]].load()
    sf_col   = ds_sf.sel(time=t_da,       lat=lat_da, lon=lon_da, method="nearest")
    hl30_col = hl30.sel(valid_time=t_da,  lat=lat_da, lon=lon_da, method="nearest")
    hl75_col = hl75.sel(valid_time=t_da,  lat=lat_da, lon=lon_da, method="nearest")
    n_ok = int(np.isfinite(sf_col["skt"].values).sum())
    log.info("CARRA: %d/%d legs with valid skt", n_ok, ds_obs.sizes["start_time"])
    return {
        "skt_K":  sf_col["skt"].values.astype(float),
        "sp_Pa":  sf_col["sp"].values.astype(float),
        "T30_K":  hl30_col["t"].values.astype(float),
        "ws30":   hl30_col["ws"].values.astype(float),
        "p30_Pa": hl30_col["pres"].values.astype(float),
        "T75_K":  hl75_col["t"].values.astype(float),
        "ws75":   hl75_col["ws"].values.astype(float),
        "p75_Pa": hl75_col["pres"].values.astype(float),
    }


def load_carra_direct_shf(ds_obs: xr.Dataset) -> np.ndarray:
    """Load CARRA model-diagnostic SHF (sshf [J m⁻² h⁻¹] → W m⁻²)."""
    fp = _CARRA / "carra_east_forecast_sf_latlon0p05.nc"
    if not fp.exists():
        log.warning("CARRA forecast SHF not found: %s", fp)
        return np.full(ds_obs.sizes["start_time"], np.nan)
    t_da   = xr.DataArray(pd.to_datetime(ds_obs["start_time"].values), dims="leg")
    lat_da = xr.DataArray(np.asarray(ds_obs["latitude"]),               dims="leg")
    lon_da = xr.DataArray(np.asarray(ds_obs["longitude"]),              dims="leg")
    sshf = (
        xr.open_dataset(fp)["sshf"]
        .sel(time=t_da, lat=lat_da, lon=lon_da, method="nearest")
        .values.astype(float)
    )
    return sshf / 3600.0   # J m⁻² h⁻¹ → W m⁻²


def _load_tbird(ds_obs: xr.Dataset):
    """Try to load the T-Bird noseboom unified file; return None if absent."""
    fp = _REPO_ROOT / "data" / "observations" / "processed" / "t-bird_unified" / "noseboom_unified.nc"
    if not fp.exists():
        log.warning("T-Bird file not found (%s) — 2-pt Ri falls back to P6 height", fp)
        return None
    try:
        ds_raw = xr.open_dataset(fp)
        ds_tb  = (
            ds_raw
            .sel(time=ds_obs["start_time"].values, method="nearest")
            .rename({"time": "start_time"})
        )
        ds_tb["start_time"] = ds_obs["start_time"].values
        log.info("T-Bird loaded: %d legs  → Ri_b evaluated at the T-Bird level", ds_tb.sizes["start_time"])
        return ds_tb
    except Exception as exc:
        log.warning("T-Bird load failed (%s) — 2-pt Ri only", exc)
        return None


# ===========================================================================
# Along-track input cache (extract/compute split)
# ===========================================================================
# All raw inputs above (36 GB ERA5 ML, CARRA rasters, MODIS L3, 574 MB
# noseboom archive) are only ever used leg-colocated. The cache stores those
# per-leg extractions (~100 kB), so the compute stage — including the full
# sensitivity sweep — runs on any machine without the raw archives.
# Workflow: run ``--extract-only`` once on the raw-data host, ship the cache,
# then ``pixi run precompute`` anywhere.

_CACHE = _AR / "along_track_inputs.nc"

_ERA5ML_KEYS = ["T30_K", "ws30", "z30", "T50_K", "ws50", "z50", "T10_K", "ws10", "z10"]
_ERA5SL_KEYS = ["skt_K", "avg_ishf", "sp_Pa"]
_TB_KEYS     = ["T", "p", "u", "v", "h"]


def _write_cache(ds_obs, ds_modis, era5_ml: dict, era5_sl: dict,
                 carra_sshf: np.ndarray, ds_tbird) -> None:
    """Persist all leg-colocated raw-data extractions to one small NetCDF."""
    n = ds_obs.sizes["start_time"]
    coords = {"leg": ("leg", np.arange(n, dtype="int16")),
              "start_time": ("leg", ds_obs["start_time"].values)}
    data = {"modis_IST": ("leg", np.asarray(ds_modis["IST"], dtype=float))}
    for k in _ERA5ML_KEYS:
        data[f"era5ml_{k}"] = ("leg", np.asarray(era5_ml[k], dtype=float))
    for k in _ERA5SL_KEYS:
        data[f"era5sl_{k}"] = ("leg", np.asarray(era5_sl[k], dtype=float))
    data["carra_sshf_Wm2"] = ("leg", np.asarray(carra_sshf, dtype=float))
    if ds_tbird is not None:
        for k in _TB_KEYS:
            data[f"tb_{k}"] = ("leg", np.asarray(ds_tbird[k], dtype=float))
    ds = xr.Dataset(data, coords=coords, attrs={
        "title":   "BACSAM-II along-track (leg-colocated) raw-data extractions",
        "summary": ("Per-leg extractions of MODIS IST, ERA5 ML/SL, CARRA sshf and "
                    "the T-Bird state, written by calc_bulk_and_ec_fluxes.py "
                    "--extract-only on the raw-data host. Allows the full "
                    "precompute (incl. sensitivity sweeps) without raw archives."),
        "source":  "scripts/calc_bulk_and_ec_fluxes.py (fluxpy)",
        "history": time.strftime("%Y-%m-%dT%H:%M:%SZ created", time.gmtime()),
    })
    ds.to_netcdf(_CACHE, mode="w")
    log.info("Along-track input cache written: %s (%d legs)", _CACHE, n)


def _read_cache(ds_obs):
    """Load the leg-colocated extractions; returns the loader outputs."""
    cache = xr.open_dataset(_CACHE)
    if not np.array_equal(cache["start_time"].values, ds_obs["start_time"].values):
        raise RuntimeError(
            f"{_CACHE} is stale: leg set differs from the flux archive. "
            "Re-run 'python scripts/calc_bulk_and_ec_fluxes.py --extract-only' "
            "on the raw-data host.")
    ds_modis = xr.Dataset({"IST": ("leg", cache["modis_IST"].values)})
    era5_ml  = {k: cache[f"era5ml_{k}"].values.astype(float) for k in _ERA5ML_KEYS}
    era5_sl  = {k: cache[f"era5sl_{k}"].values.astype(float) for k in _ERA5SL_KEYS}
    carra_sshf = cache["carra_sshf_Wm2"].values.astype(float)
    if all(f"tb_{k}" in cache for k in _TB_KEYS):
        coord = {"start_time": ds_obs["start_time"]}
        ds_tbird = xr.Dataset(
            {k: xr.DataArray(cache[f"tb_{k}"].values.astype(float),
                             dims="start_time", coords=coord) for k in _TB_KEYS},
            coords=coord)
        log.info("T-Bird state from cache → Ri_b evaluated at the T-Bird level")
    else:
        ds_tbird = None
        log.warning("Cache has no T-Bird state — 2-pt Ri falls back to P6 height")
    log.info("Along-track input cache read: %s", _CACHE)
    return ds_modis, era5_ml, era5_sl, carra_sshf, ds_tbird


# ===========================================================================
# ERA5 ML classic 2-pt bulk Richardson (direct, for diagnostic output only)
# ===========================================================================

def _compute_era5_ml_ri(era5_ml: dict, era5_sl: dict, ds_obs: xr.Dataset, ref_cfg) -> tuple:
    """
    Classic 2-pt bulk Richardson number and GLGS20 bulk SHF for ERA5 ML column.

    Ri_b per manuscript Eq. (rib): surface (ERA5 skt, no-slip) → ML 136
    (~31 m), layer-mean theta_ref; ML 135 (~54 m) is retained for
    diagnostics only. Returns (Ri_era, H_era5_direct_ml).

    Note: This diagnostic H uses ERA5 skt directly (no MODIS gap-fill) and is
    stored as ``H_era_bulk_ml`` in the output NC.  Sensitivity-sweep H uses the
    gap-filled MODIS IST via ``run_sensitivity``.
    """
    n = ds_obs.sizes["start_time"]
    T_sfc_C   = era5_sl["skt_K"] - 273.15
    T30era_C    = era5_ml["T30_K"] - 273.15

    p_sfc_hPa = era5_sl["sp_Pa"] / 100.0
    # Full-level pressure from the official L137 half-level coefficients
    # (a in Pa, hence /100): pf(k) = (a(k-1)+a(k))/2 + (b(k-1)+b(k))/2 * ps
    p_30era_hPa = (3.757813 + 0.0) / 2 / 100.0 + ((0.995003 + 0.997630) / 2) * p_sfc_hPa   # L136
    theta_sfc = potential_temperature(T_sfc_C, p_sfc_hPa)
    theta_30era = potential_temperature(T30era_C,  p_30era_hPa)

    # Effective wind (Jordan 1999 stable / Fairall 1996 unstable; the
    # Zhang et al. 2025 treatment) prevents the 1/U² divergence physically;
    # the ±ri_clip analysis bound is then applied as in every pathway.
    dtheta_era = theta_30era - theta_sfc
    theta_bar  = 0.5 * (theta_sfc + theta_30era)
    S_era = effective_wind_speed(era5_ml["ws30"].copy(), dtheta_era,
                                 era5_ml["z30"].copy(), theta_bar)
    Ri_era = bulk_richardson_2pt(
        dtheta=dtheta_era,
        dz=era5_ml["z30"].copy(),
        theta_ref=theta_bar,
        eps_wind2=0.0,
        wspd=S_era,                    # no-slip surface: ΔU = S(L136)
    )
    Ri_era = np.clip(Ri_era, -ref_cfg.ri_clip, ref_cfg.ri_clip)
    log.info("Ri_era (ML 2-pt, effective wind, |Ri|<=%.0f): mean=%.3f  range=[%.3f, %.3f]",
             ref_cfg.ri_clip,
             float(np.nanmean(Ri_era)), float(np.nanmin(Ri_era)), float(np.nanmax(Ri_era)))

    stype = classify_surface(T_sfc_C, ref_cfg.T_thresh_open, ref_cfg.T_thresh_thick)
    z0, zt = assign_roughness(stype, ref_cfg.z0_ocean, ref_cfg.z0_thin_ice, ref_cfg.z0_thick_ice, ref_cfg.zt_ratio)
    z_arr = era5_ml["z30"].copy()
    rho   = air_density(T30era_C, p_30era_hPa)
    dTheta = theta_sfc - theta_30era

    Ch = np.full(n, np.nan)
    for i in range(n):
        zi, z0i, zti, ri = z_arr[i], z0[i], zt[i], Ri_era[i]
        if not (np.isfinite(ri) and z0i > 0 and zti > 0 and zi > z0i and zi > zti):
            continue
        try:
            _, Ch[i] = Cd_Ch_noniterative(float(ri), float(zi), float(z0i), float(zti), "GLGS20")
        except Exception:
            pass

    H_era_bulk = rho * CP * Ch * era5_ml["ws30"] * dTheta
    log.info("H_era_bulk (ML bulk): mean=%.2f W/m²  (n=%d valid)", np.nanmean(H_era_bulk), int(np.isfinite(H_era_bulk).sum()))
    return Ri_era, H_era_bulk


# ===========================================================================
# Build ds_era5_col / ds_era5_sfc for fluxpy.sensitivity.run_sensitivity
# ===========================================================================

def _build_era5_col_sfc(era5_ml: dict, era5_sl: dict, ds_modis: xr.Dataset,
                         ds_obs: xr.Dataset) -> tuple:
    """
    Assemble the two pseudo-Datasets consumed by ``compute_bulk_shf_remote``:

        ds_era5_col  →  atmospheric column (ERA5 ML level 136 ≈ 31 m)
        ds_era5_sfc  →  surface temperature (MODIS IST, gap-filled with ERA5 skt)

    Returns (ds_era5_col, ds_era5_sfc, era5_z_col, era5_skt_C, era5_T30_C).
    """
    n     = ds_obs.sizes["start_time"]
    coord = {"start_time": ds_obs["start_time"]}

    # Gap-fill MODIS IST with ERA5 skt where MODIS is cloud-masked / missing
    ist_modis  = np.asarray(ds_modis["IST"]).astype(float)
    era5_skt_K = era5_sl["skt_K"]
    ist_filled = np.where(np.isnan(ist_modis), era5_skt_K, ist_modis)
    n_filled   = int(np.sum(np.isnan(ist_modis) & np.isfinite(era5_skt_K)))
    n_valid    = int(np.isfinite(ist_filled).sum())
    log.info("ERA5 skt gap-fill: +%d legs  →  %d/%d valid IST", n_filled, n_valid, n)

    T30era_C = era5_ml["T30_K"] - 273.15
    T50era_C = era5_ml["T50_K"] - 273.15
    p_sfc_Pa = era5_sl["sp_Pa"]

    ds_era5_col = xr.Dataset(
        {
            "t":    xr.DataArray(T30era_C,                dims="start_time", coords=coord, attrs={"units": "degC",  "long_name": "ERA5 ML level 136 (~31 m) air T"}),
            "ws":   xr.DataArray(era5_ml["ws30"],      dims="start_time", coords=coord, attrs={"units": "m s-1", "long_name": "ERA5 ML level 136 (~31 m) wind speed"}),
            "pres": xr.DataArray(np.full(n, p_sfc_Pa),  dims="start_time", coords=coord, attrs={"units": "Pa",    "long_name": "approximate surface pressure (1000 hPa)"}),
        },
        coords=coord,
    )

    ds_era5_sfc = xr.Dataset(
        {
            "IST": xr.DataArray(
                ist_filled, dims="start_time", coords=coord,
                attrs={"units": "K", "long_name": "MODIS IST gap-filled with ERA5 skt"},
            ),
        },
        coords=coord,
    )

    era5_z_col  = era5_ml["z30"].copy()   # per-leg column height [m] (L136 ≈ 31 m)
    era5_skt_C  = ist_filled - 273.15
    return ds_era5_col, ds_era5_sfc, era5_z_col, era5_skt_C, T30era_C, T50era_C


# ===========================================================================
# Perturbation sweeps
# ===========================================================================

def _perturbation_sweeps(ds_obs: xr.Dataset, ds_era5_col: xr.Dataset,
                          ds_era5_sfc: xr.Dataset, stacked: xr.Dataset,
                          ref_cfg, ds_tbird, era5_ml: dict) -> tuple:
    """
    Four perturbation sweeps (T_sfc, T_air, Ri, U) for P6 and ERA5 column.

    Returns a 11-tuple:
        dT_levels, dRi_levels, dU_levels,
        H_obs_dTsfc, H_rs_dTsfc,
        H_obs_dTair, H_rs_dTair,
        H_obs_dRi,   H_rs_dRi,
        H_obs_dU,    H_rs_dU
    """
    n      = ds_obs.sizes["start_time"]
    z_col  = era5_ml["z30"]

    dT_levels  = np.array([-5., -3., -2., -1., -0.5, 0., 0.5, 1., 2., 3., 5.])
    dRi_levels = np.array([-5., -3., -2., -1., -0.5, 0., 0.5, 1., 2., 3., 5.])
    dU_levels  = np.array([-4., -3., -2., -1., -0.5, 0., 0.5, 1., 2., 3., 4.])

    # The ΔRi sweep is a mathematical sensitivity probe over ±5, so it must NOT
    # be truncated by the (much tighter) analysis bound ref_cfg.ri_clip.
    RI_PERT_CLIP = 10.0

    H_obs_dTsfc      = np.full((len(dT_levels), n), np.nan)
    H_rs_dTsfc = np.full((len(dT_levels), n), np.nan)
    H_obs_dTair      = np.full((len(dT_levels), n), np.nan)
    H_rs_dTair = np.full((len(dT_levels), n), np.nan)
    H_obs_dRi        = np.full((len(dRi_levels), n), np.nan)
    H_rs_dRi   = np.full((len(dRi_levels), n), np.nan)
    H_obs_dU         = np.full((len(dU_levels), n), np.nan)
    H_rs_dU    = np.full((len(dU_levels), n), np.nan)

    # ── Sweeps A & B: T_sfc and T_air ────────────────────────────────────────
    log.info("Sweep A: ΔT_sfc …")
    for k, dT in enumerate(dT_levels):
        _ds = ds_obs.assign({"t_kt19": ds_obs["t_kt19"] + float(dT)})
        H_obs_dTsfc[k] = np.asarray(
            compute_bulk_shf_obs(_ds, ref_cfg, ds_tbird=ds_tbird)["H_obs_bulk"]
        )
        _dm = ds_era5_sfc.assign({"IST": ds_era5_sfc["IST"] + float(dT)})
        H_rs_dTsfc[k] = np.asarray(
            compute_bulk_shf_remote(ds_obs, ds_era5_col, _dm, ref_cfg,
                                    ds_tbird=None, z_ref=z_col)["H_rs_bulk"]
        )

    log.info("Sweep B: ΔT_air …")
    for k, dT in enumerate(dT_levels):
        _ds = ds_obs.assign({"T_mean": ds_obs["T_mean"] + float(dT)})
        H_obs_dTair[k] = np.asarray(
            compute_bulk_shf_obs(_ds, ref_cfg, ds_tbird=ds_tbird)["H_obs_bulk"]
        )
        _dc = ds_era5_col.assign({"t": ds_era5_col["t"] + float(dT)})
        H_rs_dTair[k] = np.asarray(
            compute_bulk_shf_remote(ds_obs, _dc, ds_era5_sfc, ref_cfg,
                                    ds_tbird=None, z_ref=z_col)["H_rs_bulk"]
        )

    # ── Reference arrays for analytical sweeps C & D ─────────────────────
    ref_run    = stacked.sel(sensitivity_run="GLGS20_ref")
    Ri_ref     = np.asarray(ref_run["Ri_obs"])
    z0_ref     = np.asarray(ref_run["z0_obs"])
    zt_ref     = np.asarray(ref_run["zt_obs"])
    rho_ref    = np.asarray(ref_run["rho_obs"])
    dT_ref     = np.asarray(ref_run["dtheta_obs"])
    U_ref      = np.sqrt(np.asarray(ds_obs["u_mean"]) ** 2 + np.asarray(ds_obs["v_mean"]) ** 2)
    z_arr      = np.asarray(ds_obs["altitude"])

    Ri_rs_ref  = np.asarray(ref_run["Ri_rs"])
    z0_rs_ref  = np.asarray(ref_run["z0_rs"])
    zt_rs_ref  = np.asarray(ref_run["zt_rs"])
    rho_rs_ref = np.asarray(ref_run["rho_rs"])
    dT_rs_ref  = np.asarray(ref_run["dtheta_rs"])
    U_rs       = era5_ml["ws30"]
    z_rs_arr   = z_col

    # ── Sweep C: ΔRi_b ────────────────────────────────────────────────────
    log.info("Sweep C: ΔRi_b …")
    for k, dRi in enumerate(dRi_levels):
        Ri_pert = np.clip(Ri_ref + dRi, -RI_PERT_CLIP, RI_PERT_CLIP)
        for i in range(n):
            zi, z0i, zti, ri = z_arr[i], z0_ref[i], zt_ref[i], Ri_pert[i]
            if not (np.isfinite(ri) and np.isfinite(zi) and z0i > 0
                    and zti > 0 and zi > z0i and zi > zti):
                continue
            try:
                _, Ch_i = Cd_Ch_noniterative(float(ri), float(zi), float(z0i), float(zti), "GLGS20")
                H_obs_dRi[k, i] = -rho_ref[i] * CP * Ch_i * U_ref[i] * dT_ref[i]
            except Exception:
                pass

        Ri_pert_rs = np.clip(Ri_rs_ref + dRi, -RI_PERT_CLIP, RI_PERT_CLIP)
        for i in range(n):
            z0i, zti, ri = z0_rs_ref[i], zt_rs_ref[i], Ri_pert_rs[i]
            _z = float(z_rs_arr[i])
            if not (np.isfinite(ri) and np.isfinite(_z) and z0i > 0
                    and zti > 0 and _z > z0i and _z > zti):
                continue
            try:
                _, Ch_i = Cd_Ch_noniterative(float(ri), _z, float(z0i), float(zti), "GLGS20")
                H_rs_dRi[k, i] = -rho_rs_ref[i] * CP * Ch_i * U_rs[i] * dT_rs_ref[i]
            except Exception:
                pass

    # ── Sweep D: ΔU ───────────────────────────────────────────────────────
    log.info("Sweep D: ΔU …")
    Ch_ref_obs = np.full(n, np.nan)
    for i in range(n):
        zi, z0i, zti, ri = z_arr[i], z0_ref[i], zt_ref[i], Ri_ref[i]
        if not (np.isfinite(ri) and np.isfinite(zi) and z0i > 0
                and zti > 0 and zi > z0i and zi > zti):
            continue
        try:
            _, Ch_ref_obs[i] = Cd_Ch_noniterative(float(ri), float(zi), float(z0i), float(zti), "GLGS20")
        except Exception:
            pass

    for k, dU in enumerate(dU_levels):
        U_pert = np.maximum(U_ref + dU, 0.0)
        for i in range(n):
            if np.isfinite(Ch_ref_obs[i]) and np.isfinite(rho_ref[i]) and np.isfinite(dT_ref[i]):
                H_obs_dU[k, i] = -rho_ref[i] * CP * Ch_ref_obs[i] * U_pert[i] * dT_ref[i]

    Ch_ref_rs = np.full(n, np.nan)
    for i in range(n):
        z0i, zti, ri = z0_rs_ref[i], zt_rs_ref[i], Ri_rs_ref[i]
        _z = float(z_rs_arr[i])
        if not (np.isfinite(ri) and np.isfinite(_z) and z0i > 0
                and zti > 0 and _z > z0i and _z > zti):
            continue
        try:
            _, Ch_ref_rs[i] = Cd_Ch_noniterative(float(ri), _z, float(z0i), float(zti), "GLGS20")
        except Exception:
            pass

    for k, dU in enumerate(dU_levels):
        U_pert_rs = np.maximum(U_rs + dU, 0.0)
        for i in range(n):
            if (np.isfinite(Ch_ref_rs[i]) and np.isfinite(rho_rs_ref[i])
                    and np.isfinite(dT_rs_ref[i])):
                H_rs_dU[k, i] = -rho_rs_ref[i] * CP * Ch_ref_rs[i] * U_pert_rs[i] * dT_rs_ref[i]

    return (dT_levels, dRi_levels, dU_levels,
            H_obs_dTsfc, H_rs_dTsfc,
            H_obs_dTair, H_rs_dTair,
            H_obs_dRi,   H_rs_dRi,
            H_obs_dU,    H_rs_dU)


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_all(verbose: bool = False, extract_only: bool = False,
            no_cache: bool = False) -> None:
    log.info("BACSAM-II bulk SHF pipeline  (start)")
    t_wall = time.perf_counter()

    # ── Load data ──────────────────────────────────────────────────────────
    ds_obs   = load_obs()
    n_legs  = ds_obs.sizes["start_time"]
    t_obs    = ds_obs["start_time"].values
    H_obs_ec    = np.asarray(ds_obs["sensible_heat_flux"]).astype(float)
    e_rel   = ds_obs["lenschow_rel_error"].values.astype("float32")

    use_cache = _CACHE.exists() and not no_cache and not extract_only
    if use_cache:
        ds_modis, era5_ml_dat, era5_sl_dat, carra_sshf, ds_tbird = _read_cache(ds_obs)
    else:
        # The CARRA forecast surface-flux file is not part of the repository.
        # Rebuilding the cache without it would replace the shipped CARRA
        # flux (H_car_diag) by NaN, so refuse instead of overwriting.
        _carra_fc = _CARRA / "carra_east_forecast_sf_latlon0p05.nc"
        if _CACHE.exists() and not _carra_fc.exists():
            raise SystemExit(
                f"Refusing to rebuild {_CACHE.name}: {_carra_fc.name} is not available, "
                "so the shipped CARRA surface flux would be lost. The MODIS and ERA5 "
                "entries of the shipped cache are reproducible from the shipped gridded "
                "files (checked 2026-09-21: identical).")
        ds_modis    = load_modis(ds_obs)
        era5_ml_dat = load_era5_ml(ds_obs)
        era5_sl_dat = load_era5_sl(ds_obs)
        carra_sshf  = load_carra_direct_shf(ds_obs)
        ds_tbird    = _load_tbird(ds_obs)
        _write_cache(ds_obs, ds_modis, era5_ml_dat, era5_sl_dat, carra_sshf, ds_tbird)
        if extract_only:
            log.info("Extract-only run complete (%.1f s) — cache ready to ship.",
                     time.perf_counter() - t_wall)
            return

    # Model-diagnosed fluxes are positive-DOWNWARD in the raw products; flip
    # here ONCE so every H_* in the pipeline is positive-upward (see
    # fluxpy.naming). Notebooks must NOT negate again.
    H_era_diag = -era5_sl_dat["avg_ishf"]
    H_car_diag = -carra_sshf

    # ── Reference config ───────────────────────────────────────────────────
    ref_cfg = reference_config()   # GLGS20, classic 2-pt Ri_b

    # ── ERA5 ML 2-pt Ri_b + direct bulk H (diagnostic-only ML implementation) ─
    # The CANONICAL H_era_bulk comes from sensitivity sweep B's reference run
    # (same algorithm as H_rs_bulk, only T_s swapped); this quick ML variant is
    # kept as H_era_bulk_ml for cross-checking.
    Ri_era, H_era_bulk_ml = _compute_era5_ml_ri(era5_ml_dat, era5_sl_dat, ds_obs, ref_cfg)

    # ── ERA5 column datasets for sensitivity sweep ─────────────────────────
    ds_era5_col, ds_era5_sfc, era5_z_col, era5_skt_C, era5_T30_C, era5_T50_C = (
        _build_era5_col_sfc(era5_ml_dat, era5_sl_dat, ds_modis, ds_obs)
    )

    # ── Sensitivity sweep A: MODIS IST gap-filled with ERA5 skt (~25 configs) ──
    configs = sensitivity_matrix()
    log.info("Running sensitivity sweep A (MODIS gap-fill): %d configs …", len(configs))
    import logging as _lg
    _lg.getLogger("fluxpy.bulk").setLevel(_lg.ERROR)
    _lg.getLogger("fluxpy.sensitivity").setLevel(_lg.ERROR)
    stacked, summary = run_sensitivity(
        ds_obs, ds_era5_col, ds_era5_sfc,
        configs=configs,
        ds_tbird=ds_tbird,
        z_ref_rs=era5_z_col,
    )

    # ── Sensitivity sweep B: ERA5 skt only (no MODIS substitution, ~25 configs) ──
    # Surface temperature = ERA5 skt directly; MODIS IST is never used.
    _coord_sfc = {"start_time": ds_obs["start_time"]}
    ds_era5_sfc_era5only = xr.Dataset(
        {
            "IST": xr.DataArray(
                era5_sl_dat["skt_K"],
                dims="start_time", coords=_coord_sfc,
                attrs={"units": "K",
                       "long_name": "ERA5 skt – no MODIS substitution (era5only)"},
            ),
        },
        coords=_coord_sfc,
    )
    configs_era5only = [replace(c, label=c.label + "_era5only") for c in configs]
    log.info("Running sensitivity sweep B (ERA5 skt only): %d configs …", len(configs_era5only))
    stacked_era5only, summary_era5only = run_sensitivity(
        ds_obs, ds_era5_col, ds_era5_sfc_era5only,
        configs=configs_era5only,
        ds_tbird=ds_tbird,
        z_ref_rs=era5_z_col,
    )
    _lg.getLogger("fluxpy.bulk").setLevel(_lg.INFO)
    _lg.getLogger("fluxpy.sensitivity").setLevel(_lg.INFO)

    # ── Combine both sweeps into a joint ~50-run stack ─────────────────────
    stacked_combined = xr.concat([stacked, stacked_era5only], dim="sensitivity_run")
    configs_all      = configs + configs_era5only

    # ── Reference run outputs ──────────────────────────────────────────────
    # Every headline series is taken from the reference configuration;
    # configuration-ensemble means are stored separately (*_ensmean) and
    # used for spread bands only.
    _ref = stacked.sel(sensitivity_run="GLGS20_ref")
    _ref_era = stacked_era5only.sel(sensitivity_run="GLGS20_ref_era5only")
    H_obs_bulk = np.asarray(_ref["H_obs_bulk"]).astype(float)
    H_rs_bulk  = np.asarray(_ref["H_rs_bulk"]).astype(float)
    H_era_bulk = np.asarray(_ref_era["H_rs_bulk"]).astype(float)
    Ri_obs     = np.asarray(_ref["Ri_obs"]).astype(float)
    Ri_rs      = np.asarray(_ref["Ri_rs"]).astype(float)
    dtheta_obs = np.asarray(_ref["dtheta_obs"]).astype(float)
    dtheta_rs  = np.asarray(_ref["dtheta_rs"]).astype(float)
    U_obs      = np.asarray(ds_obs["ws_mean"]).astype(float)
    U_era      = np.asarray(ds_era5_col["ws"]).astype(float)

    # ── Configuration-ensemble arrays ──────────────────────────────────────
    # The concatenated sweeps already carry a ``sensitivity_run`` dimension in
    # configuration order; transpose once instead of re-stacking per label.
    # In the combined stack, runs 0..N-1 use MODIS IST (gap-filled with ERA5
    # skt) and runs N..2N-1 use ERA5 skt only (``_era5only`` suffix).
    H_obs_bulk_all = stacked["H_obs_bulk"].transpose("sensitivity_run", "start_time").values
    H_rs_bulk_all  = stacked_combined["H_rs_bulk"].transpose("sensitivity_run", "start_time").values
    Ri_rs_all      = stacked_combined["Ri_rs"].transpose("sensitivity_run", "start_time").values
    Ri_obs_all     = stacked["Ri_obs"].transpose("sensitivity_run", "start_time").values

    # Ensemble means per T_sfc source
    H_rs_bulk_ensmean          = np.nanmean(H_rs_bulk_all[:len(configs)], axis=0)   # MODIS-fill mean
    H_era_bulk_ensmean = np.nanmean(H_rs_bulk_all[len(configs):], axis=0)   # ERA5-skt-only mean

    config_labels     = [c.label for c in configs]
    config_labels_all = [c.label for c in configs_all]

    # ── Perturbation sweeps ────────────────────────────────────────────────
    (dT_levels, dRi_levels, dU_levels,
     H_obs_dTsfc, H_rs_dTsfc,
     H_obs_dTair, H_rs_dTair,
     H_obs_dRi,   H_rs_dRi,
     H_obs_dU,    H_rs_dU) = _perturbation_sweeps(
        ds_obs, ds_era5_col, ds_era5_sfc, stacked, ref_cfg, ds_tbird, era5_ml_dat
    )

    # ── Quick skill summary (headline = reference-run series) ──────────────
    for label, H in [("H_obs_bulk (GLGS20_ref)",     H_obs_bulk),
                     ("H_rs_bulk (GLGS20_ref)",      H_rs_bulk),
                     ("H_era_bulk (ref, era5only)",  H_era_bulk),
                     ("H_era_bulk_ml (diagnostic)",  H_era_bulk_ml),
                     ("H_era_diag",                  H_era_diag),
                     ("H_car_diag",                  H_car_diag)]:
        m = skill_metrics(H_obs_ec, np.asarray(H, dtype=float), label=label)
        log.info("  %-30s  RMSE=%6.2f  bias=%+6.2f  R=%+.3f  N=%d",
                 label, m["rmse"], m["bias"], m["r"], m["n"])

    # ── Pair-completeness diagnostic (n must be 91 legs / 81 QC-retained) ──
    _fin_ec = np.isfinite(H_obs_ec)
    for _lbl, _H in [("H_obs_bulk", H_obs_bulk), ("H_rs_bulk", H_rs_bulk),
                     ("H_era_bulk", H_era_bulk)]:
        _lost = _fin_ec & ~np.isfinite(np.asarray(_H, dtype=float))
        if _lost.any():
            log.warning("%s: %d EC-valid legs have no finite bulk value: %s",
                        _lbl, int(_lost.sum()),
                        [str(t)[:16] for t in np.asarray(t_obs)[_lost]])

    # ── Segment ancillaries for the published (Tier-1) file ───────────────
    # The segment-placement uncertainty lives in the bin-resolved archive;
    # align it once here so that every downstream notebook needs only this
    # file. Selection identical to load_obs().
    _bins = xr.open_dataset(_AR / "analysis_ready_polar6_turbulent_fluxes_with_bins.nc")
    _bins = _bins.where(_bins["altitude"] < 50, drop=True)
    _keepb = ~np.isin(pd.DatetimeIndex(_bins["start_time"].values).date, EXCLUDE)
    _bins = _bins.isel(start_time=np.where(_keepb)[0])
    assert _bins.sizes["start_time"] == n_legs, (_bins.sizes, n_legs)
    sigma_placement = _bins["sensible_heat_flux_bin_std"].values.astype(float)
    qc_retained = np.isfinite(H_obs_ec)

    # ── Build output NetCDF (self-describing; naming per fluxpy.naming) ────
    def _v(dims, data, units, long_name, **extra):
        """Float32 DataArray with mandatory units/long_name attributes."""
        return xr.DataArray(
            np.asarray(data, dtype="float32"),
            dims=dims, attrs={"units": units, "long_name": long_name, **extra},
        )

    _UP = "positive upward (surface to atmosphere)"
    ds_fig = xr.Dataset(
        {
            # headline flux series (reference configuration)
            "H_obs_ec":   _v("leg", H_obs_ec,   "W m-2", "Sensible heat flux, airborne eddy covariance", comment=_UP),
            "H_obs_bulk": _v("leg", H_obs_bulk, "W m-2", "Bulk SHF, T-Bird/P6 in-situ inputs + KT-19 Ts", comment=_UP),
            "H_rs_bulk":  _v("leg", H_rs_bulk,  "W m-2", "Bulk SHF, ERA5 column + MODIS Ts", comment=_UP),
            "H_era_bulk": _v("leg", H_era_bulk, "W m-2", "Bulk SHF, ERA5 column + ERA5 Ts", comment=_UP),
            "H_era_diag": _v("leg", np.asarray(H_era_diag, dtype=float), "W m-2", "ERA5 model-diagnosed surface SHF", comment=_UP + "; sign converted from the ERA5 convention"),
            "H_car_diag": _v("leg", np.asarray(H_car_diag, dtype=float), "W m-2", "CARRA model-diagnosed surface SHF", comment=_UP + "; sign converted from the CARRA convention"),
            "H_era_bulk_ml": _v("leg", H_era_bulk_ml, "W m-2", "Bulk SHF, ERA5 (independent model-level implementation, diagnostic)"),
            # meteorological inputs
            "U_obs":      _v("leg", U_obs,      "m s-1", "Wind speed, P6 segment mean"),
            "U_era":      _v("leg", U_era,      "m s-1", "Wind speed, ERA5 model level 136 (z ~ 31 m)"),
            "dtheta_obs": _v("leg", dtheta_obs, "K", "theta_air - theta_s, observations (T-Bird / KT-19)"),
            "dtheta_rs":  _v("leg", dtheta_rs,  "K", "theta_air - theta_s, ERA5 column / MODIS Ts"),
            "T_sfc_C":    _v("leg", ds_obs['t_kt19'].values, "degC", "Surface skin temperature, KT-19"),
            "era5_skt_C": _v("leg", era5_skt_C, "degC", "Surface skin temperature, ERA5"),
            "T_air_C":    _v("leg", ds_obs["T_mean"].values, "degC", "Air temperature, P6 segment mean"),
            "era5_T30_C": _v("leg", era5_T30_C, "degC", "Air temperature, ERA5 model level 136 (z ~ 31 m)"),
            "era5_T50_C": _v("leg", era5_T50_C, "degC", "Air temperature, ERA5 model level 135 (z ~ 54 m)"),
            # segment ancillaries
            "sigma_placement": _v("leg", sigma_placement, "W m-2", "EC segment-placement uncertainty (start/end-time perturbation, 1-sigma)"),
            "lead_frac":  _v("leg", np.asarray(ds_obs["lead_frac"], dtype=float), "1", "Lead fraction within the 5 km segment"),
            "sic":        _v("leg", np.asarray(ds_obs["sic"], dtype=float), "1", "Sea-ice concentration (MODIS/AMSR-2 merged)"),
            "qc_retained": xr.DataArray(qc_retained.astype("int8"), dims="leg",
                            attrs={"long_name": "Spectral-QC verdict",
                                   "flag_values": np.array([0, 1], dtype="int8"),
                                   "flag_meanings": "discarded retained"}),
            # ensemble means (spread bands only; not for skill metrics)
            "H_rs_bulk_ensmean":  _v("leg", H_rs_bulk_ensmean, "W m-2", "Bulk SHF, ERA5+MODIS, configuration-ensemble mean"),
            "H_era_bulk_ensmean": _v("leg", H_era_bulk_ensmean, "W m-2", "Bulk SHF, ERA5-only, configuration-ensemble mean"),
            # configuration ensembles
            "H_obs_bulk_all": _v(("run_obs", "leg"), H_obs_bulk_all, "W m-2", "Bulk SHF, in-situ pathway, all configurations"),
            "H_rs_bulk_all":  _v(("run", "leg"), H_rs_bulk_all, "W m-2", "Bulk SHF, remote pathway, all configurations"),
            "Ri_rs_all":      _v(("run", "leg"), Ri_rs_all, "1", "Bulk Richardson number, remote pathway, all configurations"),
            "Ri_obs_all":     _v(("run_obs", "leg"), Ri_obs_all, "1", "Bulk Richardson number, in-situ pathway, all configurations"),
            # input-perturbation sweeps (reference configuration)
            "H_obs_dTsfc": _v(("dT", "leg"), H_obs_dTsfc, "W m-2", "Bulk SHF, in-situ, perturbed surface temperature"),
            "H_rs_dTsfc":  _v(("dT", "leg"), H_rs_dTsfc, "W m-2", "Bulk SHF, remote, perturbed surface temperature"),
            "H_obs_dTair": _v(("dT", "leg"), H_obs_dTair, "W m-2", "Bulk SHF, in-situ, perturbed air temperature"),
            "H_rs_dTair":  _v(("dT", "leg"), H_rs_dTair, "W m-2", "Bulk SHF, remote, perturbed air temperature"),
            "H_obs_dRi":   _v(("dRi", "leg"), H_obs_dRi, "W m-2", "Bulk SHF, in-situ, perturbed Richardson number"),
            "H_rs_dRi":    _v(("dRi", "leg"), H_rs_dRi, "W m-2", "Bulk SHF, remote, perturbed Richardson number"),
            "H_obs_dU":    _v(("dU", "leg"), H_obs_dU, "W m-2", "Bulk SHF, in-situ, perturbed wind speed"),
            "H_rs_dU":     _v(("dU", "leg"), H_rs_dU, "W m-2", "Bulk SHF, remote, perturbed wind speed"),
            # uncertainties and stability diagnostics
            "e_rel_ec":  _v("leg", e_rel, "1", "Lenschow relative sampling error of the EC flux"),
            "Ri_era":    _v("leg", Ri_era, "1", "Bulk Richardson number, ERA5 column (2-point, surface to model level 136; effective wind, no additional bound)"),
            "Ri_obs":    _v("leg", Ri_obs, "1", "Bulk Richardson number, observations (2-point, surface to T-Bird)"),
            "Ri_rs":     _v("leg", Ri_rs, "1", "Bulk Richardson number, remote pathway (2-point, surface to model level 136; effective wind, naturally bounded; reference configuration)"),
        },
        coords={
            "leg":        ("leg", np.arange(n_legs, dtype="int16")),
            "start_time": ("leg", t_obs),
            "run":        ("run", config_labels_all),
            "run_obs":    ("run_obs", config_labels),
            "dT":         ("dT", dT_levels.astype("float32")),
            "dRi":        ("dRi", dRi_levels.astype("float32")),
            "dU":         ("dU", dU_levels.astype("float32")),
        },
        attrs={
            "title":        "BACSAM-II sensible heat flux comparison - segment-level flux estimates and sensitivity sweep",
            "summary":      ("Six sensible-heat-flux estimates on common 5 km flight segments "
                             "(airborne eddy covariance, in-situ bulk, satellite/reanalysis bulk, "
                             "model-diagnosed), with configuration ensembles and input-perturbation "
                             "sweeps. Naming convention: fluxpy.naming; all fluxes positive upward."),
            "campaign":     "BACSAM-II, Longyearbyen (Svalbard), 5 April - 2 May 2024",
            "institution":  "Leipzig Institute for Meteorology (LIM), Leipzig University",
            "creator_name": "Joshua J. Mueller",
            "source":       "scripts/calc_bulk_and_ec_fluxes.py (fluxpy)",
            "references":   "Mueller et al. (2026), Atmos. Meas. Tech.",
            "history":      time.strftime("%Y-%m-%dT%H:%M:%SZ created", time.gmtime()),
            "Conventions":  "CF-1.8",
            "era5_column":  "ERA5 ML L136 (30.96 m) column; Ri_b is the classic 2-pt bulk Richardson number surface->L136; L135 (53.92 m) and L137 (10.0 m) diagnostic; heights per the ECMWF L137 standard-atmosphere definitions",
            "run_dim_info": "run[0:N]=MODIS-fill; run[N:2N]=ERA5-skt-only (_era5only); run_obs=in-situ configs",
        },
    )

    _OUT.mkdir(parents=True, exist_ok=True)
    _nc  = _OUT / "sensitivity_figure_data.nc"
    _csv = _OUT / "sensitivity_summary_table.csv"
    log.info("Writing %s", _nc)
    ds_fig.to_netcdf(_nc, mode="w")
    log.info("Writing %s", _csv)
    summary_combined = pd.concat([summary, summary_era5only], ignore_index=True)
    summary_combined.to_csv(_csv, index=False)
    log.info("Pipeline complete in %.1f s", time.perf_counter() - t_wall)
    log.info("Variables: %s", list(ds_fig.data_vars))
    log.info("Summary rows: %d  cols: %s", len(summary_combined), list(summary_combined.columns))


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="BACSAM-II bulk SHF pre-computation pipeline")
    p.add_argument("--verbose", action="store_true", help="Set log level to DEBUG")
    p.add_argument("--extract-only", action="store_true",
                   help="Only extract the leg-colocated inputs from the raw archives "
                        "and write data/observations/ar/along_track_inputs.nc (run on the raw-data host)")
    p.add_argument("--no-cache", action="store_true",
                   help="Ignore the along-track input cache and read the raw archives")
    args = p.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    run_all(verbose=args.verbose, extract_only=args.extract_only, no_cache=args.no_cache)


if __name__ == "__main__":
    main()
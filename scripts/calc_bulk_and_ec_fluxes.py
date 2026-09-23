#!/usr/bin/env python3
"""Bulk fluxes for every forcing, the sensitivity sweep and the input perturbations.

Manuscript: Sect. 3.2 (bulk method), Sect. 4.1 (comparison, Table 4, Figs. 4-5),
Sect. 4.2 (sensitivity, Table 3, Fig. 6), Table 2 (uncertainty budget), Appendix C.

Writes
    data/model_output/sensitivity_figure_data.nc      every flux estimate on the 91 segments
    data/model_output/sensitivity_summary_table.csv   skill of every configuration

Stages
    1  load_obs             the 91 low-level segments over sea ice with the EC reference
    2  along-track inputs   MODIS IST, ERA5 level 136 (T, wind, height), ERA5 skin temperature,
                            surface pressure and surface flux, CARRA surface flux, at each segment.
                            Cached in along_track_inputs.nc so that the bulk computation
                            runs without the gridded files (pixi run verify-cache re-extracts).
    3  sensitivity sweep    every configuration of Table C1, once with the ERA5-MODIS forcing
                            (sweep A) and once with the ERA5-only forcing (sweep B); the in-situ
                            estimate is part of every run
    4  input perturbations  the reference configuration with T_s, T_air and U shifted by fixed
                            amounts, for the uncertainty budget and the input sensitivities
    5  output

All physics comes from fluxpy (bulk, stability, sensitivity). The model-diagnosed
fluxes of ERA5 and CARRA are positive downward in the products and are sign-flipped
here, once, so that every H_* is positive upward.

Run:  pixi run precompute                       (about 15 s with the cache)
      python scripts/calc_bulk_and_ec_fluxes.py --extract-only   (raw-data host: rebuild the cache)
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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from fluxpy.bulk import compute_bulk_shf_obs, compute_bulk_shf_remote, reference_config  # noqa: E402
from fluxpy.sensitivity import run_sensitivity, sensitivity_matrix, skill_metrics  # noqa: E402
from fluxpy.stability import Cd_Ch_noniterative  # noqa: E402
from fluxpy.util import CP  # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger("calc_bulk_and_ec_fluxes")

# ── files ─────────────────────────────────────────────────────────────────────
_AR = REPO / "data" / "observations" / "ar"
_SAT = REPO / "data" / "observations" / "processed" / "satellite_unified"
_ERA5_SL = REPO / "data" / "reanalysis" / "era5" / "ERA5_single_levels"
_ERA5_ML = REPO / "data" / "reanalysis" / "era5" / "ERA5_model_levels"
_CARRA_FORECAST = REPO / "data" / "reanalysis" / "carra" / "carra_east_forecast_sf_latlon0p05.nc"  # not shipped
_CACHE = _AR / "along_track_inputs.nc"
_OUT = REPO / "data" / "model_output"

# The paper analyses the four low-level flight days over sea ice (13, 15, 17, 23 April).
# 25 April: the 32 segments below 50 m were flown over open water (KT-19 mean +4.1 degC,
# open-water fraction 1.00). 26 April has no segment below 50 m.
EXCLUDE = [pd.Timestamp("2024-04-25").date(), pd.Timestamp("2024-04-26").date()]
MAX_TBIRD_HEIGHT_M = 50.0

# ERA5 model level 136 is the level closest to the T-Bird (standard-atmosphere height
# 30.96 m; ECMWF L137 definitions).
ERA5_LEVEL = 136
ERA5_LEVEL_HEIGHT_M = 30.96

# perturbations of the reference configuration (stage 4)
DT_LEVELS_K = np.array([-5., -3., -2., -1., -0.5, 0., 0.5, 1., 2., 3., 5.])
DU_LEVELS_MS = np.array([-4., -3., -2., -1., -0.5, 0., 0.5, 1., 2., 3., 4.])


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: the segments
# ══════════════════════════════════════════════════════════════════════════════

def load_obs() -> xr.Dataset:
    """The 91 low-level segments over sea ice (T-Bird below 50 m; 25 and 26 April excluded)."""
    ds = xr.open_dataset(_AR / "analysis_ready_polar6_turbulent_fluxes_latest_5km.nc")
    ds["ws_mean"] = np.sqrt(ds["u_mean"] ** 2 + ds["v_mean"] ** 2)
    ds = ds.where(ds["altitude"] < MAX_TBIRD_HEIGHT_M, drop=True)
    ds = ds.sel(start_time=~ds["start_time"].dt.date.isin(EXCLUDE))
    log.info("segments: %d", ds.sizes["start_time"])
    return ds


def _segment_points(ds_obs: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """(time, latitude, longitude) of the segments as 1-D indexers for nearest-neighbour extraction."""
    return (xr.DataArray(pd.to_datetime(ds_obs["start_time"].values), dims="leg"),
            xr.DataArray(np.asarray(ds_obs["latitude"]), dims="leg"),
            xr.DataArray(np.asarray(ds_obs["longitude"]), dims="leg"))


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: along-track inputs (nearest grid cell and time step of each product)
# ══════════════════════════════════════════════════════════════════════════════

def load_modis(ds_obs: xr.Dataset) -> xr.Dataset:
    """MODIS ice surface temperature ``IST`` [K] at the segments; fill, cloud and land values
    (outside 150-280 K) are masked. No correction is applied to the product."""
    t, lat, lon = _segment_points(ds_obs)
    raw = xr.open_dataset(_SAT / "MODIS_Aqua_Sea_Ice_Extent_and_IST_Daily_L3_Global_4km_EASE-Grid_Day_V061.nc")
    if "time" in raw.dims:
        raw = raw.rename({"time": "start_time"})
    at_segments = raw.sel(lat=lat, lon=lon, start_time=t, method="nearest")
    ist = at_segments["IST"]
    valid = (ist > 150) & (ist < 280)
    log.info("MODIS IST valid at %d of %d segments", int(valid.sum()), ds_obs.sizes["start_time"])
    return xr.Dataset({"IST": ("leg", ist.where(valid).values.astype(float))})


def load_era5_ml(ds_obs: xr.Dataset) -> dict:
    """ERA5 model level 136 at the segments: temperature ``T30_K`` [K], wind speed ``ws30`` [m/s]
    and the level height ``z30`` [m]."""
    t, lat, lon = _segment_points(ds_obs)
    level = xr.open_dataset(_ERA5_ML / "era5_ml.nc").sel(model_level=ERA5_LEVEL)
    level = level.sel(time=t, latitude=lat, longitude=lon, method="nearest")
    return {"T30_K": level["t"].values.astype(float),
            "ws30": np.hypot(level["u"].values, level["v"].values).astype(float),
            "z30": np.full(ds_obs.sizes["start_time"], ERA5_LEVEL_HEIGHT_M)}


def load_era5_sl(ds_obs: xr.Dataset) -> dict:
    """ERA5 single levels at the segments: skin temperature ``skt_K`` [K], surface pressure
    ``sp_Pa`` [Pa] and the time-averaged surface sensible heat flux ``avg_ishf`` [W m-2,
    positive downward as in the product]."""
    t, lat, lon = _segment_points(ds_obs)
    inst = xr.open_dataset(_ERA5_SL / "data_stream-oper_stepType-instant.nc")
    avg = xr.open_dataset(_ERA5_SL / "data_stream-oper_stepType-avg.nc")
    pick = dict(valid_time=t, latitude=lat, longitude=lon, method="nearest")
    return {"skt_K": inst["skt"].sel(**pick).values.astype(float),
            "sp_Pa": inst["sp"].sel(**pick).values.astype(float),
            "avg_ishf": avg["avg_ishf"].sel(**pick).values.astype(float)}


def load_carra_direct_shf(ds_obs: xr.Dataset) -> np.ndarray:
    """CARRA surface sensible heat flux at the segments [W m-2, positive downward as in the
    product; ``sshf`` is an hourly accumulation in J m-2]. The forecast file is not part of
    the repository; the values are carried in the cache."""
    if not _CARRA_FORECAST.exists():
        log.warning("CARRA forecast file not available: %s", _CARRA_FORECAST)
        return np.full(ds_obs.sizes["start_time"], np.nan)
    t, lat, lon = _segment_points(ds_obs)
    sshf = xr.open_dataset(_CARRA_FORECAST)["sshf"].sel(time=t, lat=lat, lon=lon, method="nearest")
    return sshf.values.astype(float) / 3600.0


_ERA5ML_KEYS = ["T30_K", "ws30", "z30"]
_ERA5SL_KEYS = ["skt_K", "avg_ishf", "sp_Pa"]


def _write_cache(ds_obs, ds_modis, era5_ml, era5_sl, carra_sshf) -> None:
    """Persist the along-track inputs (about 100 kB) so that the bulk computation runs anywhere."""
    n = ds_obs.sizes["start_time"]
    data = {"modis_IST": ("leg", np.asarray(ds_modis["IST"], dtype=float))}
    data.update({f"era5ml_{k}": ("leg", np.asarray(era5_ml[k], dtype=float)) for k in _ERA5ML_KEYS})
    data.update({f"era5sl_{k}": ("leg", np.asarray(era5_sl[k], dtype=float)) for k in _ERA5SL_KEYS})
    data["carra_sshf_Wm2"] = ("leg", np.asarray(carra_sshf, dtype=float))
    ds = xr.Dataset(data, coords={"leg": ("leg", np.arange(n, dtype="int16")),
                                  "start_time": ("leg", ds_obs["start_time"].values)},
                    attrs={"title": "BACSAM-II along-track (segment-colocated) inputs of the bulk fluxes",
                           "summary": "MODIS IST, ERA5 level 136, ERA5 single levels and CARRA sshf at the "
                                      "91 segments, written by calc_bulk_and_ec_fluxes.py --extract-only.",
                           "history": time.strftime("%Y-%m-%dT%H:%M:%SZ created", time.gmtime())})
    ds.to_netcdf(_CACHE, mode="w")
    log.info("cache written: %s", _CACHE)


def _read_cache(ds_obs):
    cache = xr.open_dataset(_CACHE)
    if not np.array_equal(cache["start_time"].values, ds_obs["start_time"].values):
        raise RuntimeError(f"{_CACHE} is stale: its segments differ from the segment file. "
                           "Re-run 'python scripts/calc_bulk_and_ec_fluxes.py --extract-only'.")
    ds_modis = xr.Dataset({"IST": ("leg", cache["modis_IST"].values)})
    era5_ml = {k: cache[f"era5ml_{k}"].values.astype(float) for k in _ERA5ML_KEYS}
    era5_sl = {k: cache[f"era5sl_{k}"].values.astype(float) for k in _ERA5SL_KEYS}
    log.info("along-track inputs read from %s", _CACHE.name)
    return ds_modis, era5_ml, era5_sl, cache["carra_sshf_Wm2"].values.astype(float)


def load_inputs(ds_obs: xr.Dataset, use_cache: bool, extract_only: bool):
    """Stage 2: the along-track inputs, from the cache or from the gridded files."""
    if use_cache:
        return _read_cache(ds_obs)
    ds_modis = load_modis(ds_obs)
    era5_ml = load_era5_ml(ds_obs)
    era5_sl = load_era5_sl(ds_obs)
    if _CARRA_FORECAST.exists() or not _CACHE.exists():
        carra_sshf = load_carra_direct_shf(ds_obs)
    else:   # the CARRA forecast file is not shipped: keep the values extracted on the raw-data host
        with xr.open_dataset(_CACHE) as old:
            carra_sshf = old["carra_sshf_Wm2"].values.astype(float)
        log.warning("CARRA forecast file absent: CARRA flux carried over from the existing cache")
    _write_cache(ds_obs, ds_modis, era5_ml, era5_sl, carra_sshf)
    return ds_modis, era5_ml, era5_sl, carra_sshf


def era5_forcing(era5_ml: dict, era5_sl: dict, ds_modis: xr.Dataset, ds_obs: xr.Dataset):
    """The inputs of the two reanalysis-driven bulk estimates (Table 1, rows 3-4):
    the ERA5 column at level 136, and the surface temperature either from MODIS
    (gaps filled with the ERA5 skin temperature; H_S,RS,bulk) or from ERA5 alone (H_S,ERA,bulk)."""
    coord = {"start_time": ds_obs["start_time"]}
    n = ds_obs.sizes["start_time"]

    def series(values, units, long_name):
        return xr.DataArray(values, dims="start_time", coords=coord, attrs={"units": units, "long_name": long_name})

    ds_column = xr.Dataset({
        "t":    series(era5_ml["T30_K"] - 273.15, "degC", "ERA5 air temperature, model level 136 (~31 m)"),
        "ws":   series(era5_ml["ws30"], "m s-1", "ERA5 wind speed, model level 136"),
        "pres": series(np.full(n, era5_sl["sp_Pa"]), "Pa", "ERA5 surface pressure"),
    })
    ist_modis = np.asarray(ds_modis["IST"], dtype=float)
    ist_filled = np.where(np.isnan(ist_modis), era5_sl["skt_K"], ist_modis)
    log.info("MODIS IST gap-filled with ERA5 skt at %d segments", int(np.isnan(ist_modis).sum()))
    ds_surface_modis = xr.Dataset({"IST": series(ist_filled, "K", "MODIS IST, gaps filled with ERA5 skt")})
    ds_surface_era5 = xr.Dataset({"IST": series(era5_sl["skt_K"], "K", "ERA5 skin temperature")})
    return ds_column, ds_surface_modis, ds_surface_era5


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4: input perturbations of the reference configuration
# ══════════════════════════════════════════════════════════════════════════════

def perturbation_sweeps(ds_obs, ds_column, ds_surface_modis, reference_run, ref_cfg, era5_ml) -> dict:
    """H of the reference configuration with T_s, T_air (both pathways: ``dT`` levels) and
    U (``dU`` levels) shifted. T_s and T_air are shifted in the inputs and the bulk flux is
    recomputed; U is shifted analytically with the transfer coefficient of the unperturbed
    run (the wind enters Eq. 2 linearly). Returns arrays of shape (n_levels, n_segments)."""
    n = ds_obs.sizes["start_time"]
    z_col = era5_ml["z30"]
    out = {k: np.full((len(DT_LEVELS_K), n), np.nan) for k in ("H_obs_dTsfc", "H_rs_dTsfc", "H_obs_dTair", "H_rs_dTair")}
    out.update({k: np.full((len(DU_LEVELS_MS), n), np.nan) for k in ("H_obs_dU", "H_rs_dU")})

    log.info("perturbations: surface temperature")
    for k, dT in enumerate(DT_LEVELS_K):
        obs = ds_obs.assign(t_kt19=ds_obs["t_kt19"] + float(dT))
        out["H_obs_dTsfc"][k] = compute_bulk_shf_obs(obs, ref_cfg)["H_obs_bulk"]
        sfc = ds_surface_modis.assign(IST=ds_surface_modis["IST"] + float(dT))
        out["H_rs_dTsfc"][k] = compute_bulk_shf_remote(ds_obs, ds_column, sfc, ref_cfg, z_ref=z_col)["H_rs_bulk"]

    log.info("perturbations: air temperature")
    for k, dT in enumerate(DT_LEVELS_K):
        obs = ds_obs.assign(T_mean=ds_obs["T_mean"] + float(dT))
        out["H_obs_dTair"][k] = compute_bulk_shf_obs(obs, ref_cfg)["H_obs_bulk"]
        col = ds_column.assign(t=ds_column["t"] + float(dT))
        out["H_rs_dTair"][k] = compute_bulk_shf_remote(ds_obs, col, ds_surface_modis, ref_cfg, z_ref=z_col)["H_rs_bulk"]

    log.info("perturbations: wind speed")
    for pathway, U, z in (("obs", np.asarray(ds_obs["ws_mean"], dtype=float), np.asarray(ds_obs["altitude"], dtype=float)),
                          ("rs", era5_ml["ws30"], z_col)):
        Ri, z0, zt = (np.asarray(reference_run[f"{v}_{pathway}"]) for v in ("Ri", "z0", "zt"))
        rho, dtheta = (np.asarray(reference_run[f"{v}_{pathway}"]) for v in ("rho", "dtheta"))
        Ch = np.full(n, np.nan)
        for i in range(n):
            if np.isfinite(Ri[i]) and np.isfinite(z[i]) and z0[i] > 0 and zt[i] > 0 and z[i] > z0[i] and z[i] > zt[i]:
                try:
                    _, Ch[i] = Cd_Ch_noniterative(float(Ri[i]), float(z[i]), float(z0[i]), float(zt[i]), ref_cfg.scheme_name)
                except Exception:
                    pass
        for k, dU in enumerate(DU_LEVELS_MS):
            U_shifted = np.maximum(U + dU, 0.0)
            ok = np.isfinite(Ch) & np.isfinite(rho) & np.isfinite(dtheta)
            out[f"H_{pathway}_dU"][k, ok] = -rho[ok] * CP * Ch[ok] * U_shifted[ok] * dtheta[ok]   # dtheta = theta_air - theta_s
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run_all(extract_only: bool = False, no_cache: bool = False, verbose: bool = False) -> None:
    t_wall = time.perf_counter()
    if not verbose:
        logging.getLogger("fluxpy").setLevel(logging.WARNING)

    # stage 1 and 2 ---------------------------------------------------------------
    ds_obs = load_obs()
    n = ds_obs.sizes["start_time"]
    ds_modis, era5_ml, era5_sl, carra_sshf = load_inputs(
        ds_obs, use_cache=_CACHE.exists() and not no_cache and not extract_only, extract_only=extract_only)
    if extract_only:
        log.info("extract-only run complete (%.1f s)", time.perf_counter() - t_wall)
        return
    ds_column, ds_surface_modis, ds_surface_era5 = era5_forcing(era5_ml, era5_sl, ds_modis, ds_obs)

    H_obs_ec = np.asarray(ds_obs["sensible_heat_flux"], dtype=float)
    H_era_diag = -era5_sl["avg_ishf"]          # products are positive downward: flip once, here
    H_car_diag = -carra_sshf

    # stage 3: sensitivity sweep -------------------------------------------------
    ref_cfg = reference_config()
    configs = sensitivity_matrix()
    log.info("sweep A (ERA5-MODIS forcing): %d configurations", len(configs))
    stacked_A, summary_A = run_sensitivity(ds_obs, ds_column, ds_surface_modis, configs, z_ref_rs=era5_ml["z30"])
    configs_B = [replace(c, label=c.label + "_era5only") for c in configs]
    log.info("sweep B (ERA5-only forcing): %d configurations", len(configs_B))
    stacked_B, summary_B = run_sensitivity(ds_obs, ds_column, ds_surface_era5, configs_B, z_ref_rs=era5_ml["z30"])

    reference_run = stacked_A.sel(sensitivity_run="GLGS20_ref")
    H_obs_bulk = np.asarray(reference_run["H_obs_bulk"], dtype=float)
    H_rs_bulk = np.asarray(reference_run["H_rs_bulk"], dtype=float)
    H_era_bulk = np.asarray(stacked_B.sel(sensitivity_run="GLGS20_ref_era5only")["H_rs_bulk"], dtype=float)
    stacked_rs = xr.concat([stacked_A, stacked_B], dim="sensitivity_run")   # remote pathway: both forcings

    # stage 4: perturbations -----------------------------------------------------
    sweeps = perturbation_sweeps(ds_obs, ds_column, ds_surface_modis, reference_run, ref_cfg, era5_ml)

    # segment-placement spread of the EC flux (scripts/ec_segment_placement.py)
    with xr.open_dataset(_AR / "ec_segment_placement.nc") as placement:
        sigma_placement = placement["sigma_placement"].sel(start_time=ds_obs["start_time"].values).values.astype(float)

    for label, H in [("H_obs_bulk", H_obs_bulk), ("H_rs_bulk", H_rs_bulk), ("H_era_bulk", H_era_bulk),
                     ("H_era_diag", H_era_diag), ("H_car_diag", H_car_diag)]:
        m = skill_metrics(H_obs_ec, np.asarray(H, dtype=float), label)
        log.info("  %-11s slope %.2f  r %.2f  bias %+.2f  n %d", label, m["slope"], m["r"], m["bias"], m["n"])

    # stage 5: output ------------------------------------------------------------
    def var(dims, data, units, long_name, **extra):
        return xr.DataArray(np.asarray(data, dtype="float32"), dims=dims,
                            attrs={"units": units, "long_name": long_name, **extra})

    up = "positive upward (surface to atmosphere)"
    ds_out = xr.Dataset(
        {
            # the six flux estimates of Table 1 (reference configuration)
            "H_obs_ec":   var("leg", H_obs_ec, "W m-2", "Sensible heat flux, airborne eddy covariance", comment=up),
            "H_obs_bulk": var("leg", H_obs_bulk, "W m-2", "Bulk SHF, T-Bird in-situ inputs and KT-19 T_s", comment=up),
            "H_rs_bulk":  var("leg", H_rs_bulk, "W m-2", "Bulk SHF, ERA5 column and MODIS T_s", comment=up),
            "H_era_bulk": var("leg", H_era_bulk, "W m-2", "Bulk SHF, ERA5 column and ERA5 T_s", comment=up),
            "H_era_diag": var("leg", H_era_diag, "W m-2", "ERA5 model-diagnosed surface SHF", comment=up + "; sign converted"),
            "H_car_diag": var("leg", H_car_diag, "W m-2", "CARRA model-diagnosed surface SHF", comment=up + "; sign converted"),
            # inputs of the bulk estimates
            "U_obs":      var("leg", ds_obs["ws_mean"], "m s-1", "Wind speed, T-Bird segment mean"),
            "U_era":      var("leg", ds_column["ws"], "m s-1", "Wind speed, ERA5 model level 136 (~31 m)"),
            "dtheta_obs": var("leg", reference_run["dtheta_obs"], "K", "theta_air - theta_s, T-Bird / KT-19"),
            "dtheta_rs":  var("leg", reference_run["dtheta_rs"], "K", "theta_air - theta_s, ERA5 column / MODIS T_s"),
            "T_sfc_C":    var("leg", ds_obs["t_kt19"], "degC", "Surface skin temperature, KT-19"),
            "T_sfc_rs_C": var("leg", ds_surface_modis["IST"] - 273.15, "degC", "Surface skin temperature, MODIS (gaps filled with ERA5 skt)"),
            "T_sfc_era_C": var("leg", ds_surface_era5["IST"] - 273.15, "degC", "Surface skin temperature, ERA5"),
            "T_air_C":    var("leg", ds_obs["T_mean"], "degC", "Air temperature, T-Bird segment mean"),
            "T_air_era_C": var("leg", ds_column["t"], "degC", "Air temperature, ERA5 model level 136"),
            "Ri_obs":     var("leg", reference_run["Ri_obs"], "1", "Bulk Richardson number, surface to T-Bird (Eq. B2)"),
            "Ri_rs":      var("leg", reference_run["Ri_rs"], "1", "Bulk Richardson number, surface to ERA5 level 136, effective wind (Eq. B2)"),
            "lead_frac":  var("leg", ds_obs["lead_frac"], "1", "Fraction of KT-19 samples warmer than -5 degC"),
            "sic":        var("leg", ds_obs["sic"], "%", "Sea-ice concentration along the segment (merged MODIS-AMSR2)"),
            # uncertainties of the EC flux (Table 2)
            "sigma_random_ec":    var("leg", ds_obs["sensible_heat_flux_random_error"], "W m-2", "Random (sampling) error of the EC flux, 1 sigma (Finkelstein and Sims, 2001)"),
            "ec_flux_over_sigma": var("leg", ds_obs["flux_over_random_error"], "1", "|H_obs_ec| / sigma_random_ec (all fluxes are kept, whatever this ratio)"),
            "sigma_placement":    var("leg", sigma_placement, "W m-2", "Segment-placement spread of the EC flux, 1 sigma (cross-check of the random error)"),
            # every configuration (Table C1); run_obs = in-situ pathway, run = remote pathway with both forcings
            "H_obs_bulk_all": var(("run_obs", "leg"), stacked_A["H_obs_bulk"].transpose("sensitivity_run", "start_time"), "W m-2", "Bulk SHF, in-situ pathway, all configurations"),
            "H_rs_bulk_all":  var(("run", "leg"), stacked_rs["H_rs_bulk"].transpose("sensitivity_run", "start_time"), "W m-2", "Bulk SHF, remote pathway, all configurations; runs with suffix _era5only use the ERA5 skin temperature"),
            # input perturbations of the reference configuration (Table 2, Sect. 4.2)
            "H_obs_dTsfc": var(("dT", "leg"), sweeps["H_obs_dTsfc"], "W m-2", "Bulk SHF, in-situ, surface temperature shifted by dT"),
            "H_rs_dTsfc":  var(("dT", "leg"), sweeps["H_rs_dTsfc"], "W m-2", "Bulk SHF, ERA5-MODIS, surface temperature shifted by dT"),
            "H_obs_dTair": var(("dT", "leg"), sweeps["H_obs_dTair"], "W m-2", "Bulk SHF, in-situ, air temperature shifted by dT"),
            "H_rs_dTair":  var(("dT", "leg"), sweeps["H_rs_dTair"], "W m-2", "Bulk SHF, ERA5-MODIS, air temperature shifted by dT"),
            "H_obs_dU":    var(("dU", "leg"), sweeps["H_obs_dU"], "W m-2", "Bulk SHF, in-situ, wind speed shifted by dU"),
            "H_rs_dU":     var(("dU", "leg"), sweeps["H_rs_dU"], "W m-2", "Bulk SHF, ERA5-MODIS, wind speed shifted by dU"),
        },
        coords={
            "leg": ("leg", np.arange(n, dtype="int16")),
            "start_time": ("leg", ds_obs["start_time"].values),
            "run_obs": ("run_obs", [c.label for c in configs]),
            "run": ("run", [c.label for c in configs + configs_B]),
            "dT": ("dT", DT_LEVELS_K.astype("float32"), {"units": "K"}),
            "dU": ("dU", DU_LEVELS_MS.astype("float32"), {"units": "m s-1"}),
        },
        attrs={
            "title": "BACSAM-II sensible heat flux comparison: flux estimates on the 91 segments and the sensitivity sweep",
            "summary": "Six sensible-heat-flux estimates on common 5 km flight segments (airborne eddy covariance, "
                       "in-situ bulk, satellite/reanalysis bulk, model-diagnosed), all configurations of the "
                       "sensitivity analysis and the input perturbations. Names follow fluxpy.naming; all fluxes positive upward.",
            "campaign": "BACSAM-II, Longyearbyen (Svalbard), April 2024",
            "institution": "Leipzig Institute for Meteorology (LIM), Leipzig University",
            "creator_name": "Joshua J. Mueller",
            "source": "scripts/calc_bulk_and_ec_fluxes.py (fluxpy)",
            "references": "Mueller et al. (2026), Atmos. Meas. Tech.",
            "history": time.strftime("%Y-%m-%dT%H:%M:%SZ created", time.gmtime()),
            "Conventions": "CF-1.8",
        },
    )
    _OUT.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(_OUT / "sensitivity_figure_data.nc", mode="w")
    pd.concat([summary_A, summary_B], ignore_index=True).to_csv(_OUT / "sensitivity_summary_table.csv",
                                                                index=False, lineterminator="\n")
    log.info("written %s and sensitivity_summary_table.csv (%.1f s)", _OUT / "sensitivity_figure_data.nc",
             time.perf_counter() - t_wall)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--extract-only", action="store_true",
                   help="only extract the along-track inputs from the gridded files into "
                        "data/observations/ar/along_track_inputs.nc (raw-data host)")
    p.add_argument("--no-cache", action="store_true", help="ignore the cache and read the gridded files")
    p.add_argument("--verbose", action="store_true", help="log every configuration run")
    args = p.parse_args()
    run_all(extract_only=args.extract_only, no_cache=args.no_cache, verbose=args.verbose)


if __name__ == "__main__":
    main()

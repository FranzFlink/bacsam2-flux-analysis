r"""
fluxpy.sensitivity
==================
Sensitivity of the bulk fluxes to the configuration, and the skill metrics
(manuscript Sect. 4.1, Table 4; Sect. 4.2, Table 3, Fig. 7; Appendix C, Table C1).

* :func:`sensitivity_matrix` builds the tested configurations, one axis at a time
  about the reference: stability scheme, scalar roughness ratio alpha_T,
  momentum roughness set, constant momentum roughness.
* :func:`run_sensitivity` evaluates every configuration for the in-situ and the
  remote pathway and stacks the results along ``sensitivity_run``.
* :func:`skill_metrics` are the numbers of the comparison: regression slope
  (amplitude), correlation, bias, RMSE against the EC reference. The slope and
  the correlation are the headline metrics because RMSE in this low-flux regime
  rewards estimates that shrink towards zero (Sect. 4.1).
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from .bulk import BulkConfig, compute_bulk_shf_obs, compute_bulk_shf_remote, reference_config

log = logging.getLogger(__name__)

__all__ = ["sensitivity_matrix", "skill_metrics", "amplitude_metrics", "slope_bootstrap_ci",
           "run_sensitivity"]

#: the seven schemes tested against the reference GLGS20 (Table C1)
ALTERNATIVE_SCHEMES = ["BD", "HB88", "BH91", "CB05", "GAFGP07", "ZE07-I", "ZE07-II"]
#: alpha_T values of the sweep (Table C1 lists 1e-7 ... 1e-1; Fig. 7 and Table 3 use 1e-4 ... 1e-1)
ZT_RATIOS = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
#: alternative momentum roughness sets (open water, thin ice, snow-covered ice) [m]
ROUGHNESS_SETS = {
    "mosaic":   dict(z0_ocean=3.27e-4, z0_thin_ice=2e-4, z0_thick_ice=3.27e-4),
    "nschnier": dict(z0_ocean=1e-4,    z0_thin_ice=1e-4, z0_thick_ice=5e-3),
}
#: surface-type-independent z0m values [m]
CONSTANT_Z0 = [1e-4, 2e-4, 4e-4]


def sensitivity_matrix() -> list[BulkConfig]:
    """All configurations of Table C1, the reference first. Labels are the run
    names used in the data files and tables (``GLGS20_ref``, ``BD_ref``,
    ``GLGS20_ztr1e-04``, ``GLGS20_mosaic``, ``GLGS20_z0c2e-04``, ...)."""
    ref = reference_config()
    at_reference = dict(z0_ocean=ref.z0_ocean, z0_thin_ice=ref.z0_thin_ice, z0_thick_ice=ref.z0_thick_ice,
                        T_thresh_open=ref.T_thresh_open, T_thresh_thick=ref.T_thresh_thick)
    configs = [ref]
    for scheme in ALTERNATIVE_SCHEMES:                                  # stability scheme
        configs.append(BulkConfig(scheme_name=scheme, zt_ratio=ref.zt_ratio, label=f"{scheme}_ref", **at_reference))
    for ratio in ZT_RATIOS:                                              # alpha_T
        configs.append(BulkConfig(scheme_name="GLGS20", zt_ratio=ratio, label=f"GLGS20_ztr{ratio:.0e}", **at_reference))
    for name, z0_set in ROUGHNESS_SETS.items():                          # momentum roughness set
        configs.append(BulkConfig(scheme_name="GLGS20", zt_ratio=ref.zt_ratio, label=f"GLGS20_{name}",
                                  T_thresh_open=ref.T_thresh_open, T_thresh_thick=ref.T_thresh_thick, **z0_set))
    for z0c in CONSTANT_Z0:                                              # constant z0m
        configs.append(BulkConfig(scheme_name="GLGS20", zt_ratio=ref.zt_ratio, z0_constant=z0c,
                                  label=f"GLGS20_z0c{z0c:.0e}", **at_reference))
    return configs


# ── skill metrics (Sect. 4.1) ─────────────────────────────────────────────────

def amplitude_metrics(x: np.ndarray, y: np.ndarray) -> dict:
    """Amplitude of an estimate ``y`` relative to the EC reference ``x``:
    ordinary-least-squares ``slope`` of y on x (the paper's amplitude metric),
    ``intercept``, and ``std_ratio`` = sigma_y / sigma_x."""
    sxx = float(np.var(x, ddof=1))
    syy = float(np.var(y, ddof=1))
    sxy = float(np.cov(x, y, ddof=1)[0, 1])
    if sxx <= 0 or sxy == 0.0:
        return {"slope": np.nan, "intercept": np.nan, "std_ratio": np.nan}
    slope = sxy / sxx
    return {"slope": float(slope), "intercept": float(np.mean(y) - slope * np.mean(x)),
            "std_ratio": float(np.sqrt(syy / sxx))}


def skill_metrics(H_ref: np.ndarray, H_pred: np.ndarray, label: str = "") -> dict:
    """Slope, intercept, std_ratio, Pearson r, bias, RMSE, centred RMSE and n of an
    estimate against the reference, over the pairs where both are finite."""
    mask = np.isfinite(H_ref) & np.isfinite(H_pred)
    n = int(mask.sum())
    if n < 3:
        log.warning("%s: only %d valid pairs - metrics unreliable", label, n)
        return {"rmse": np.nan, "bias": np.nan, "crmse": np.nan, "r": np.nan,
                "slope": np.nan, "intercept": np.nan, "std_ratio": np.nan, "n": n}
    x, y = H_ref[mask], H_pred[mask]
    out = {"rmse": float(np.sqrt(np.mean((y - x) ** 2))),
           "bias": float(np.mean(y - x)),
           "crmse": float(np.sqrt(np.mean(((y - y.mean()) - (x - x.mean())) ** 2))),
           "r": float(np.corrcoef(x, y)[0, 1]),
           "n": n, **amplitude_metrics(x, y)}
    log.info("%s: slope=%.2f  r=%.3f  bias=%+.2f  RMSE=%.2f  n=%d",
             label, out["slope"], out["r"], out["bias"], out["rmse"], n)
    return out


def slope_bootstrap_ci(H_ref: np.ndarray, H_pred: np.ndarray, n_boot: int = 5000, seed: int = 42,
                       key: str = "slope") -> tuple[float, float]:
    """Percentile-bootstrap 95 % confidence interval of the regression slope."""
    mask = np.isfinite(H_ref) & np.isfinite(H_pred)
    x, y = H_ref[mask], H_pred[mask]
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        values[i] = amplitude_metrics(x[idx], y[idx])[key]
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


# ── the sweep ─────────────────────────────────────────────────────────────────

def run_sensitivity(ds_obs: xr.Dataset, ds_rs: xr.Dataset, ds_modis: xr.Dataset,
                    configs: Optional[list[BulkConfig]] = None, z_ref_rs=30.0) -> tuple[xr.Dataset, pd.DataFrame]:
    """Evaluate every configuration for both pathways.

    Returns
    -------
    stacked : Dataset with a ``sensitivity_run`` dimension (select a run with
              ``stacked.sel(sensitivity_run="GLGS20_ref")``).
    summary : one row per run with the skill metrics of ``H_obs_bulk`` (``obs_*``)
              and ``H_rs_bulk`` (``rs_*``) against the EC flux.
    """
    configs = sensitivity_matrix() if configs is None else configs
    H_ec = np.asarray(ds_obs["sensible_heat_flux"], dtype=float)
    results, rows = [], []
    t0 = time.perf_counter()
    for k, cfg in enumerate(configs):
        log.info("run %d/%d: %s", k + 1, len(configs), cfg.label)
        out_obs = compute_bulk_shf_obs(ds_obs, cfg)
        out_rs = compute_bulk_shf_remote(ds_obs, ds_rs, ds_modis, cfg, z_ref=z_ref_rs)
        results.append(xr.merge([out_obs, out_rs]).expand_dims({"sensitivity_run": [cfg.label]}))
        row = {"label": cfg.label, "scheme": cfg.scheme_name, "zt_ratio": cfg.zt_ratio,
               "z0_ocean": cfg.z0_ocean, "z0_constant": cfg.z0_constant}
        row.update({f"obs_{m}": v for m, v in skill_metrics(H_ec, np.asarray(out_obs["H_obs_bulk"]), cfg.label + "/obs").items()})
        row.update({f"rs_{m}": v for m, v in skill_metrics(H_ec, np.asarray(out_rs["H_rs_bulk"]), cfg.label + "/rs").items()})
        rows.append(row)
    log.info("sensitivity sweep: %d runs in %.1f s", len(results), time.perf_counter() - t0)
    return xr.concat(results, dim="sensitivity_run"), pd.DataFrame(rows)

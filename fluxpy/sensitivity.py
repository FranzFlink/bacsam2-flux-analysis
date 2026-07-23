"""
fluxpy.sensitivity
==================
Sensitivity sweep over the bulk-flux parameter space.

    configs = sensitivity_matrix()          # ~20 named BulkConfig objects
    stacked, summary = run_sensitivity(ds_obs, ds_rs, ds_modis, configs)

The stacked Dataset has a ``sensitivity_run`` dimension keyed by config labels
for easy ``ds.sel(sensitivity_run="GLGS20_ref")``.

Lead-fraction dependence (not a perturbation — stratification analysis):

    df_lf = lead_frac_dependence(ds_obs, stacked, configs)
    # df_lf has columns: lead_frac_mid, H_ec_mean, H_obs_mean,
    #   dH_obs_mean, dH_obs_std, H_rs_mean, dH_rs_mean, dH_rs_std  (per run × bin)
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from .bulk import BulkConfig, reference_config, compute_bulk_shf_obs, compute_bulk_shf_remote

log = logging.getLogger(__name__)

__all__ = ["sensitivity_matrix", "skill_metrics", "run_sensitivity", "lead_frac_dependence"]


def sensitivity_matrix() -> list[BulkConfig]:
    """
    Build ~20 named BulkConfig objects spanning four parameter axes.

    Axis 1 – MOST scheme (roughness/thresholds at reference)
    Axis 2 – zt_ratio (scheme=GLGS20)
    Axis 3 – roughness sets: lit-lo, lit-hi, SHEBA-based
    Axis 4 – IST classification thresholds (2 × 2)
    """
    ref = reference_config()    # GLGS20, zt_ratio=1e-5 (reference, 2026-07-23), classic 2-pt Ri_b (surface -> T-Bird)
    configs: list[BulkConfig] = [ref]

    # ── Axis 1: MOST scheme ────────────────────────────────────────────────
    for scheme in ["BD", "HB88", "BH91", "CB05", "GAFGP07", "ZE07-I", "ZE07-II"]:
        configs.append(BulkConfig(
            scheme_name=scheme, zt_ratio=ref.zt_ratio,
            z0_ocean=ref.z0_ocean, z0_thin_ice=ref.z0_thin_ice, z0_thick_ice=ref.z0_thick_ice,
            T_thresh_open=ref.T_thresh_open, T_thresh_thick=ref.T_thresh_thick,
            label=f"{scheme}_ref",
        ))

    # ── Axis 2: zt_ratio ──────────────────────────────────────────────────
    for ratio in [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        configs.append(BulkConfig(
            scheme_name="GLGS20", zt_ratio=ratio,
            z0_ocean=ref.z0_ocean, z0_thin_ice=ref.z0_thin_ice, z0_thick_ice=ref.z0_thick_ice,
            T_thresh_open=ref.T_thresh_open, T_thresh_thick=ref.T_thresh_thick,
            label=f"GLGS20_ztr{ratio:.0e}",
        ))

    # ── Axis 3: roughness sets ────────────────────────────────────────────
    roughness_sets = {
        "mosaic": dict(z0_ocean=3.27e-4, z0_thin_ice=2e-4,  z0_thick_ice=3.27e-4),
        "sheba":  dict(z0_ocean=3.27e-4, z0_thin_ice=2e-4,  z0_thick_ice=2.3e-4),
        "nschnier": dict(z0_ocean=1e-4, z0_thin_ice=1e-4, z0_thick_ice=5e-3),
    }
    # zt_ratio is held at the reference value so that the momentum-roughness
    # experiment is not confounded with a change of the scalar-roughness ratio.
    for rname, rvals in roughness_sets.items():
        configs.append(BulkConfig(
            scheme_name="GLGS20", zt_ratio=ref.zt_ratio,
            T_thresh_open=ref.T_thresh_open, T_thresh_thick=ref.T_thresh_thick,
            label=f"GLGS20_{rname}", **rvals,
        ))

    # ── Axis 3b: constant (surface-type-independent) roughness ───────────
    # z0_constant overrides the fractional/T-class z0 with a single value.
    for z0c in [1e-4, 2e-4, 4e-4]:
        configs.append(BulkConfig(
            scheme_name="GLGS20", zt_ratio=ref.zt_ratio,
            z0_ocean=ref.z0_ocean, z0_thin_ice=ref.z0_thin_ice, z0_thick_ice=ref.z0_thick_ice,
            T_thresh_open=ref.T_thresh_open, T_thresh_thick=ref.T_thresh_thick,
            z0_constant=z0c,
            label=f"GLGS20_z0c{z0c:.0e}",
        ))

    # ── Axis 4: classification thresholds ────────────────────────────────
    for t_open, t_thick, tlabel in [(-1.8, -10.0, "tT-10"),
                                     ( 0.0, -10.0, "tO0_tT-10"),
                                     ( 0.0, -20.0, "tO0_tT-20")]:
        configs.append(BulkConfig(
            scheme_name="GLGS20", zt_ratio=ref.zt_ratio,
            z0_ocean=ref.z0_ocean, z0_thin_ice=ref.z0_thin_ice, z0_thick_ice=ref.z0_thick_ice,
            T_thresh_open=t_open, T_thresh_thick=t_thick, label=f"GLGS20_{tlabel}",
        ))

    # ── Axis 5: Aerodynamic & low-wind gustiness models ─────────────────
    for zt_m in ["dynamic", "andreas", "fairall"]:
        configs.append(BulkConfig(
            scheme_name="GLGS20", zt_ratio=ref.zt_ratio,
            z0_ocean=ref.z0_ocean, z0_thin_ice=ref.z0_thin_ice, z0_thick_ice=ref.z0_thick_ice,
            T_thresh_open=ref.T_thresh_open, T_thresh_thick=ref.T_thresh_thick,
            zt_model=zt_m, use_gustiness=False,
            label=f"GLGS20_zt_{zt_m}",
        ))

    # Add best engineering approaches (dynamic with scaled-down zt_factor)
    for factor in [0.1, 0.01, 0.002]:
        configs.append(BulkConfig(
            scheme_name="GLGS20", zt_ratio=ref.zt_ratio,
            z0_ocean=ref.z0_ocean, z0_thin_ice=ref.z0_thin_ice, z0_thick_ice=ref.z0_thick_ice,
            T_thresh_open=ref.T_thresh_open, T_thresh_thick=ref.T_thresh_thick,
            zt_model="dynamic", zt_factor=factor, use_gustiness=True,
            label=f"GLGS20_zt_dynamic_scaled_{factor}",
        ))

    configs.append(BulkConfig(
        scheme_name="GLGS20", zt_ratio=ref.zt_ratio,
        z0_ocean=ref.z0_ocean, z0_thin_ice=ref.z0_thin_ice, z0_thick_ice=ref.z0_thick_ice,
        T_thresh_open=ref.T_thresh_open, T_thresh_thick=ref.T_thresh_thick,
        zt_model="constant", use_gustiness=True,
        label="GLGS20_gustiness",
    ))

    configs.append(BulkConfig(
        scheme_name="GLGS20", zt_ratio=ref.zt_ratio,
        z0_ocean=ref.z0_ocean, z0_thin_ice=ref.z0_thin_ice, z0_thick_ice=ref.z0_thick_ice,
        T_thresh_open=ref.T_thresh_open, T_thresh_thick=ref.T_thresh_thick,
        zt_model="dynamic", use_gustiness=True,
        label="GLGS20_coupled_adv",
    ))

    log.info("Sensitivity matrix: %d configs", len(configs))
    return configs


def skill_metrics(
    H_ref: np.ndarray,
    H_pred: np.ndarray,
    label: str = "",
    sign_agreement: bool = False,
    sign_floor: float = 0.0,
) -> dict:
    """
    RMSE, bias and Pearson R between two flux arrays (NaN-safe).

    Parameters
    ----------
    sign_agreement : bool
        If True, restrict the metrics to segments where reference and
        prediction agree on the sign of the flux. Parameter inference
        (e.g. of the scalar-roughness ratio) is only meaningful on this
        subset: where the two estimates disagree on the flux direction,
        the mismatch originates in the inputs rather than in the transfer
        coefficient, and tuning the roughness against it would attribute
        an input error to the surface.
    sign_floor : float
        With ``sign_agreement=True``, segments where both fluxes are
        below this magnitude [W m⁻²] are kept regardless of sign, since
        near-zero fluxes carry no meaningful sign information.
    """
    mask = np.isfinite(H_ref) & np.isfinite(H_pred)
    if sign_agreement:
        agree = np.sign(H_ref) == np.sign(H_pred)
        near_zero = (np.abs(H_ref) < sign_floor) & (np.abs(H_pred) < sign_floor)
        mask &= agree | near_zero
    n = int(np.sum(mask))
    if n < 3:
        log.warning("%s: only %d valid pairs – metrics unreliable", label, n)
        return {"rmse": np.nan, "bias": np.nan, "r": np.nan, "n": n}
    x, y  = H_ref[mask], H_pred[mask]
    rmse  = float(np.sqrt(np.mean((y - x) ** 2)))
    bias  = float(np.mean(y - x))
    r     = float(np.corrcoef(x, y)[0, 1])
    log.info("%s%s: RMSE=%.2f  bias=%.2f  R=%.3f  N=%d",
             label, " [sign-agree]" if sign_agreement else "", rmse, bias, r, n)
    return {"rmse": rmse, "bias": bias, "r": r, "n": n}


def run_sensitivity(
    ds_obs:    xr.Dataset,
    ds_rs:    xr.Dataset,
    ds_modis: xr.Dataset,
    configs:  Optional[list[BulkConfig]] = None,
    ds_tbird: Optional[xr.Dataset] = None,
    z_ref_rs: "float | np.ndarray" = 30.0,
) -> tuple[xr.Dataset, pd.DataFrame]:
    """
    Run every config in the sensitivity matrix and stack along ``sensitivity_run``.

    Parameters
    ----------
    z_ref_rs : float or ndarray
        Reference height [m] passed to ``compute_bulk_shf_remote``.
        Use a scalar (default 30 m) or a per-leg array (e.g. ERA5 geopotential
        height at 1000 hPa) when the column height varies by leg.

    Returns
    -------
    stacked : xr.Dataset
        All runs merged; select with ``stacked.sel(sensitivity_run="GLGS20_ref")``.
    summary : pd.DataFrame
        Per-run skill metrics (RMSE, bias, R) vs. the EC flux.
    """
    if configs is None:
        configs = sensitivity_matrix()

    results, metrics_rows = [], []
    t_wall = time.perf_counter()

    H_obs_ec = np.asarray(ds_obs["sensible_heat_flux"])

    for k, cfg in enumerate(configs):
        log.info("── run %d/%d: %s ──", k + 1, len(configs), cfg.label)
        try:
            out_obs = compute_bulk_shf_obs(ds_obs, cfg, ds_tbird=ds_tbird)
            out_rs = compute_bulk_shf_remote(
                ds_obs, ds_rs, ds_modis, cfg, ds_tbird=None, z_ref=z_ref_rs  # never T-Bird for RS/ERA5 column
            )

            run_ds = xr.merge([out_obs, out_rs])
            results.append(run_ds.expand_dims({"sensitivity_run": [cfg.label]}))

            H_obs_bulk = np.asarray(out_obs["H_obs_bulk"])
            H_rs = np.asarray(out_rs["H_rs_bulk"])
            row = {"label": cfg.label, "scheme": cfg.scheme_name,
                   "zt_ratio": cfg.zt_ratio, "z0_ocean": cfg.z0_ocean,
                   "z0_constant": cfg.z0_constant}
            row.update({f"obs_{k}": v for k, v in skill_metrics(H_obs_ec, H_obs_bulk, cfg.label + "/obs").items()})
            row.update({f"rs_{k}": v for k, v in skill_metrics(H_obs_ec, H_rs, cfg.label + "/rs").items()})
            # Sign-agreement subset: used for the alpha_T / roughness
            # inference only; the headline comparison uses all segments.
            row.update({f"obs_sgn_{k}": v for k, v in
                        skill_metrics(H_obs_ec, H_obs_bulk, cfg.label + "/obs",
                                      sign_agreement=True).items()})
            row.update({f"rs_sgn_{k}": v for k, v in
                        skill_metrics(H_obs_ec, H_rs, cfg.label + "/rs",
                                      sign_agreement=True).items()})
            metrics_rows.append(row)
        except Exception as exc:
            log.error("Run '%s' failed: %s", cfg.label, exc, exc_info=True)

    log.info("Sensitivity sweep done in %.1f s  (%d/%d runs succeeded)",
             time.perf_counter() - t_wall, len(results), len(configs))

    stacked = xr.concat(results, dim="sensitivity_run")
    summary = pd.DataFrame(metrics_rows)
    return stacked, summary


def lead_frac_dependence(
    ds_obs:    xr.Dataset,
    stacked:  xr.Dataset,
    configs:  Optional[list[BulkConfig]] = None,
    lf_bins:  Optional[np.ndarray] = None,
    lf_var:   str = "lead_frac",
) -> pd.DataFrame:
    """
    Stratify bulk-flux skill by per-leg lead fraction.

    For every sensitivity run and every ``lead_frac`` bin this function computes:

    * ``H_obs_ec``     – mean EC sensible heat flux  [W m⁻²]
    * ``H_obs_bulk``     – mean P6 bulk SHF
    * ``dH_obs``    – mean ΔH = H_obs_bulk − H_EC           (bias in bin)
    * ``dH_obs_std``– std of ΔH_obs in bin
    * ``H_rs``     – mean RS bulk SHF  (NaN where not available)
    * ``dH_rs``    – mean ΔH = H_rs_bulk − H_EC
    * ``dH_rs_std``– std of ΔH_rs in bin
    * ``n``        – number of legs in bin

    This is **not** a perturbation sweep — ``lead_frac`` is taken as-is from
    the aggregation output (``ds_obs[lf_var]``) and used only for stratification.

    Parameters
    ----------
    ds_obs    : Analysis-ready P6 dataset (must contain ``lf_var`` and
               ``sensible_heat_flux``).
    stacked  : Output of :func:`run_sensitivity` — Dataset with a
               ``sensitivity_run`` dimension.
    configs  : List of :class:`BulkConfig` used to build ``stacked``.
               If ``None``, only the runs present in ``stacked`` are used.
    lf_bins  : Bin edges for lead fraction (0–1).  Default: 0, 0.05, 0.10, …, 1.05
               (steps of 0.05).
    lf_var   : Variable name for lead fraction in ``ds_obs``.
               Default ``'lead_frac'``.

    Returns
    -------
    pd.DataFrame
        Columns: ``lead_frac_mid``, ``lead_frac_left``, ``lead_frac_right``,
        ``sensitivity_run``, ``H_obs_ec``, ``H_obs_bulk``, ``dH_obs``, ``dH_obs_std``,
        ``H_rs``, ``dH_rs``, ``dH_rs_std``, ``n``.
    """
    if lf_bins is None:
        lf_bins = np.arange(0.0, 1.05, 0.05)

    # ── Guard: lead_frac must exist ──────────────────────────────────────────
    if lf_var not in ds_obs:
        raise KeyError(
            f"'{lf_var}' not found in ds_obs.  "
            f"Available variables: {list(ds_obs.data_vars)}"
        )

    lf       = np.asarray(ds_obs[lf_var], dtype=float)
    H_obs_ec     = np.asarray(ds_obs["sensible_heat_flux"], dtype=float)

    run_labels = list(stacked["sensitivity_run"].values)
    if configs is not None:
        run_labels = [c.label for c in configs if c.label in run_labels]

    bin_lefts  = lf_bins[:-1]
    bin_rights = lf_bins[1:]
    bin_mids   = 0.5 * (bin_lefts + bin_rights)

    rows: list[dict] = []
    for run_label in run_labels:
        run = stacked.sel(sensitivity_run=run_label)
        H_obs_bulk = np.asarray(run["H_obs_bulk"], dtype=float)
        H_rs = (
            np.asarray(run["H_rs_bulk"], dtype=float)
            if "H_rs_bulk" in run
            else np.full_like(H_obs_bulk, np.nan)
        )

        for lo, hi, mid in zip(bin_lefts, bin_rights, bin_mids):
            mask = (lf >= lo) & (lf < hi) & np.isfinite(lf)
            n = int(mask.sum())
            if n == 0:
                rows.append(dict(
                    lead_frac_mid=mid, lead_frac_left=lo, lead_frac_right=hi,
                    sensitivity_run=run_label,
                    H_obs_ec=np.nan, H_obs_bulk=np.nan,
                    dH_obs=np.nan, dH_obs_std=np.nan,
                    H_rs=np.nan, dH_rs=np.nan, dH_rs_std=np.nan,
                    n=0,
                ))
                continue

            ec_bin   = H_obs_ec[mask]
            p6_bin   = H_obs_bulk[mask]
            rs_bin   = H_rs[mask]
            dH_obs_b  = p6_bin - ec_bin
            dH_rs_b  = rs_bin - ec_bin

            rows.append(dict(
                lead_frac_mid=mid,
                lead_frac_left=lo,
                lead_frac_right=hi,
                sensitivity_run=run_label,
                H_obs_ec=float(np.nanmean(ec_bin)),
                H_obs_bulk=float(np.nanmean(p6_bin)),
                dH_obs=float(np.nanmean(dH_obs_b)),
                dH_obs_std=float(np.nanstd(dH_obs_b)),
                H_rs=float(np.nanmean(rs_bin)),
                dH_rs=float(np.nanmean(dH_rs_b)),
                dH_rs_std=float(np.nanstd(dH_rs_b)),
                n=n,
            ))

    return pd.DataFrame(rows)

"""
fluxpy.naming
=============
Canonical variable names for the BACSAM-II pipeline (single source of truth).

The scheme follows the manuscript notation:

===================  =========================  =====================================
code / NetCDF name   manuscript symbol          meaning
===================  =========================  =====================================
``H_obs_ec``         :math:`H_{S,obs,EC}`       Airborne eddy-covariance reference
``H_obs_bulk``       :math:`H_{S,obs,bulk}`     Bulk, T-Bird/P6 in-situ inputs + KT-19
``H_rs_bulk``        :math:`H_{S,RS,bulk}`      Bulk, ERA5 column + MODIS T_s
``H_era_bulk``       :math:`H_{S,ERA,bulk}`     Bulk, ERA5 column + ERA5 T_s
``H_era_diag``       :math:`H_{S,ERA,diag}`     ERA5 model-diagnosed surface SHF
``H_car_diag``       :math:`H_{S,CAR,diag}`     CARRA model-diagnosed surface SHF
===================  =========================  =====================================

Conventions
-----------
* **Sign**: every ``H_*`` variable is stored positive-UPWARD
  (surface → atmosphere). The raw ERA5/CARRA products are positive-downward;
  they are sign-flipped ONCE, in ``scripts/calc_bulk_and_ec_fluxes.py`` — never
  again in notebooks.
* Ensembles carry ``_all`` (configuration dimension) and ``_ensmean``
  (configuration mean). Ensemble means illustrate the parameter spread;
  all skill metrics refer to the reference-configuration series.
* Diagnostics use the same pathway tokens: ``Ri_obs``, ``Ri_rs``, ``Ri_era``,
  ``dtheta_obs``, ``dtheta_rs``, ``U_obs``, ``U_era`` …  Every ``Ri_*`` is the
  classic 2-pt bulk Richardson number (manuscript Eq. rib). CARRA appears
  only as the model-diagnosed ``H_car_diag`` (no CARRA bulk pathway).
* Input archives (``analysis_ready_*``) keep their historical names
  (``sensible_heat_flux`` = EC flux, …); map them on load with
  :func:`rename_to_canonical`.

Matplotlib labels for figures are provided in :data:`PLOT_LABEL` so every
figure uses the identical legend strings.
"""
from __future__ import annotations

import xarray as xr

__all__ = ["PATHWAYS", "PLOT_LABEL", "LEGACY_ALIASES", "rename_to_canonical"]

#: pathway tokens in canonical plotting order
PATHWAYS = ["obs_ec", "obs_bulk", "rs_bulk", "era_bulk", "era_diag", "car_diag"]

#: legend/axis labels (mathtext) — identical in every figure and the manuscript
PLOT_LABEL = {
    "H_obs_ec":   r"$H_\mathrm{S,obs,EC}$",
    "H_obs_bulk": r"$H_\mathrm{S,obs,bulk}$",
    "H_rs_bulk":  r"$H_\mathrm{S,RS,bulk}$",
    "H_era_bulk": r"$H_\mathrm{S,ERA,bulk}$",
    "H_era_diag": r"$H_\mathrm{S,ERA,diag}$",
    "H_car_diag": r"$H_\mathrm{S,CAR,diag}$",
}

#: old → canonical names (covers every historical generation of the
#: precompute NetCDFs and the analysis-ready archives)
LEGACY_ALIASES = {
    # EC reference
    "H_ec": "H_obs_ec",
    "sensible_heat_flux": "H_obs_ec",           # analysis-ready archives
    # bulk pathways
    "H_p6_ref": "H_obs_bulk",
    "sensible_heat_flux_bulk_p6": "H_obs_bulk",
    "sensible_heat_flux_bulk_rs": "H_rs_bulk",
    "H_era5_col": "H_rs_bulk_ensmean",          # NB: the OLD H_era5_col was an ensemble mean!
    "H_era5_col_era5only": "H_era_bulk_ensmean",
    "H_era5_only": "H_era_bulk",
    "H_era5": "H_era_bulk",
    "H_carra": "H_car_bulk",
    # model-diagnosed (old files: positive-downward!)
    "H_era5_direct": "H_era_diag",
    "H_carra_direct": "H_car_diag",
    # ensembles
    "H_p6_all": "H_obs_bulk_all",
    "H_era5_col_all": "H_rs_bulk_all",
    "H_rs_all": "H_rs_bulk_all",
    # Richardson numbers
    "Ri_b_p6": "Ri_obs", "Ri_p6_ref": "Ri_obs", "Ri_p6_3pt": "Ri_obs_3pt",
    "Ri_b_rs": "Ri_rs", "Ri_era5": "Ri_era", "Ri_carra": "Ri_car",
    "Ri_b_p6_all": "Ri_obs_all", "Ri_b_rs_all": "Ri_rs_all",
    # perturbation sweeps
    "H_dTsfc": "H_obs_dTsfc", "H_dTsfc_era5": "H_rs_dTsfc",
    "H_dTair": "H_obs_dTair", "H_dTair_era5": "H_rs_dTair",
    "H_dU": "H_obs_dU", "H_dU_era5": "H_rs_dU",
    "H_dRi": "H_obs_dRi", "H_dRi_era5": "H_rs_dRi",
}


def rename_to_canonical(ds: xr.Dataset) -> xr.Dataset:
    """
    Return *ds* with every recognised legacy variable renamed to its
    canonical name. Unknown variables pass through unchanged. Safe to call
    on already-canonical datasets (no-op).

    Note: this only renames — it does NOT fix the sign convention of legacy
    ``H_era5_direct`` / ``H_carra_direct`` (positive-downward). When loading
    OLD precompute files, flip those manually; files written by the current
    ``calc_bulk_and_ec_fluxes.py`` are already positive-upward.
    """
    mapping = {k: v for k, v in LEGACY_ALIASES.items()
               if k in ds and v not in ds}
    return ds.rename(mapping) if mapping else ds

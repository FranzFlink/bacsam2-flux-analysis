r"""
fluxpy.uncertainty
==================
Flux-uncertainty budget of manuscript Table 2 (Sect. 3.1 and 3.2).

* **EC flux**: total = random error (Finkelstein and Sims, 2001; Appendix A2)
  (+) instrument error, added in quadrature. The instrument error is taken as
  5 % of the flux (five-hole-probe flow calibration). The segment-placement
  spread (``scripts/ec_segment_placement.py``) is an independent, empirical
  estimate of the same sampling variability; it is reported as a cross-check and
  NOT added.
* **Bulk fluxes**: first-order Gaussian propagation of the assumed 1-sigma input
  uncertainties (Table 2) with the derivatives dH/dx taken as central
  differences from the perturbation experiments stored in
  ``sensitivity_figure_data.nc`` (Sect. 4.2).
"""
from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = ["REL_INSTRUMENT", "INPUT_UNCERTAINTY", "central_difference", "uncertainty_budget"]

#: systematic instrument error of the EC flux, relative (manuscript Sect. 3.1)
REL_INSTRUMENT = 0.05

#: assumed 1-sigma input uncertainties (T_s [K], T_air [K], U [m s-1]) of Table 2
INPUT_UNCERTAINTY = {
    "obs": (0.5, 0.3, 0.5),    # KT-19, T-Bird Pt100, T-Bird five-hole probe
    "rs":  (1.5, 1.2, 1.7),    # MODIS IST, ERA5 temperature, ERA5 wind
}


def central_difference(fig: xr.Dataset, variable: str, axis: str) -> np.ndarray:
    """dH/dx per segment: central difference of the perturbation sweep ``variable``
    (e.g. ``H_obs_dTair``) about zero perturbation along ``axis`` (``dT`` or ``dU``)."""
    sweep = np.asarray(fig[variable], dtype=float)
    x = np.asarray(fig[axis], dtype=float)
    i0 = int(np.where(x == 0.0)[0][0])
    return (sweep[i0 + 1] - sweep[i0 - 1]) / (x[i0 + 1] - x[i0 - 1])


def uncertainty_budget(fig: xr.Dataset) -> dict[str, np.ndarray]:
    """Per-segment 1-sigma uncertainties [W m-2] of Table 2: ``ec_random``,
    ``ec_instrument``, ``ec_total``, ``ec_placement`` (cross-check), ``obs_bulk``, ``rs_bulk``."""
    H_ec = np.asarray(fig["H_obs_ec"], dtype=float)
    out = {
        "ec_random":     np.asarray(fig["sigma_random_ec"], dtype=float),
        "ec_instrument": np.abs(H_ec) * REL_INSTRUMENT,
        "ec_placement":  np.asarray(fig["sigma_placement"], dtype=float),
    }
    out["ec_total"] = np.sqrt(out["ec_random"] ** 2 + out["ec_instrument"] ** 2)
    for key, (u_ts, u_ta, u_wind) in INPUT_UNCERTAINTY.items():
        out[f"{key}_bulk"] = np.sqrt(
            (central_difference(fig, f"H_{key}_dTsfc", "dT") * u_ts) ** 2
            + (central_difference(fig, f"H_{key}_dTair", "dT") * u_ta) ** 2
            + (central_difference(fig, f"H_{key}_dU", "dU") * u_wind) ** 2)
    return out

"""
fluxpy.uncertainty
==================
Flux-uncertainty budget of manuscript Table ``tab:bulk_unc``.

* EC flux: quadrature sum of (i) the random sampling error after Lenschow et
  al. (1994), (ii) a 5 % instrument error from the five-hole-probe flow
  calibration and (iii) the segment-placement error
  (``scripts/ec_segment_placement.py``).
* Bulk fluxes: first-order Gaussian propagation of the assumed 1-sigma input
  uncertainties, with the derivatives dH/dx taken as central differences from
  the perturbation experiments stored in ``sensitivity_figure_data.nc``.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

REL_INSTRUMENT = 0.05

#: assumed 1-sigma input uncertainties: (T_s [K], T_air [K], U [m s-1])
INPUT_UNCERTAINTY = {
    "obs": (0.5, 0.3, 0.5),    # KT-19, T-Bird Pt100, T-Bird five-hole probe
    "rs":  (1.5, 1.2, 1.7),    # MODIS IST, ERA5 temperature, ERA5 wind
}


def _central(fig: xr.Dataset, vname: str, axname: str) -> np.ndarray:
    """Central-difference dH/dx at zero perturbation, per segment."""
    sweep = np.asarray(fig[vname], dtype=float)
    ax = np.asarray(fig[axname], dtype=float)
    i0 = int(np.where(ax == 0.0)[0][0])
    return (sweep[i0 + 1] - sweep[i0 - 1]) / (ax[i0 + 1] - ax[i0 - 1])


def uncertainty_budget(fig: xr.Dataset) -> dict[str, np.ndarray]:
    """Per-segment 1-sigma uncertainties [W m-2] of the EC and the two bulk fluxes."""
    H_ec = np.asarray(fig["H_obs_ec"], dtype=float)
    out = {
        "ec_sampling":   np.abs(H_ec) * np.asarray(fig["e_rel_ec"], dtype=float),
        "ec_instrument": np.abs(H_ec) * REL_INSTRUMENT,
        "ec_placement":  np.asarray(fig["sigma_placement"], dtype=float),
    }
    out["ec_total"] = np.sqrt(out["ec_sampling"] ** 2 + out["ec_instrument"] ** 2
                              + out["ec_placement"] ** 2)
    for key, (u_ts, u_ta, u_u) in INPUT_UNCERTAINTY.items():
        out[f"{key}_bulk"] = np.sqrt(
            (_central(fig, f"H_{key}_dTsfc", "dT") * u_ts) ** 2
            + (_central(fig, f"H_{key}_dTair", "dT") * u_ta) ** 2
            + (_central(fig, f"H_{key}_dU", "dU") * u_u) ** 2)
    return out

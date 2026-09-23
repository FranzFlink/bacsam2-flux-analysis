r"""
fluxpy.util
===========
Physical constants and the small thermodynamic helpers every other module uses.

This is the ONLY place where a physical constant is defined. The values are
those stated in the manuscript (Appendix B1: c_p = 1004 J kg-1 K-1; Appendix B3:
kappa = 0.40).
"""
from __future__ import annotations

import numpy as np
import xarray as xr

# ── constants ────────────────────────────────────────────────────────────────
CP = 1004.0        #: J kg-1 K-1  specific heat of dry air at constant pressure (manuscript App. B1)
RD = 287.05        #: J kg-1 K-1  specific gas constant of dry air
G = 9.80665        #: m s-2       gravitational acceleration
KAPPA = 0.4        #: von Kármán constant (manuscript App. B3)
P0 = 100000.0      #: Pa          reference pressure of the potential temperature (1000 hPa)
RD_OVER_CP = RD / CP   #: exponent of the Poisson equation, 0.2859


# ── unit helpers ─────────────────────────────────────────────────────────────
# The shipped files store temperature in degC and pressure in hPa; reanalysis
# fields come in K and Pa. The helpers accept both and always return float64.

def _to_kelvin(T) -> np.ndarray:
    """degC -> K; values above 200 are taken to be in K already."""
    T = np.asarray(T, dtype=float)
    return np.where(T > 200.0, T, T + 273.15)


def _to_pa(p) -> np.ndarray:
    """hPa -> Pa; values above 2000 are taken to be in Pa already."""
    p = np.asarray(p, dtype=float)
    return np.where(p > 2000.0, p, p * 100.0)


# ── thermodynamics ───────────────────────────────────────────────────────────

def potential_temperature(T, p, p0: float = P0) -> np.ndarray:
    r"""Dry potential temperature (Poisson equation; manuscript App. B1, Eq. B1),

    $$ \theta = T \left( \frac{p_0}{p} \right)^{R_d / c_p} \quad [\mathrm{K}], $$

    ``T`` in degC or K, ``p`` in hPa or Pa (see the unit helpers above).
    """
    return _to_kelvin(T) * (p0 / _to_pa(p)) ** RD_OVER_CP


def air_density(T, p) -> np.ndarray:
    r"""Dry-air density from the ideal-gas law, $\rho = p / (R_d T)$ [kg m-3].

    Humidity corrections are negligible for the cold, dry conditions sampled
    during BACSAM-II (manuscript Sect. 3.1). ``T`` in degC or K, ``p`` in hPa or Pa.
    """
    return _to_pa(p) / (RD * _to_kelvin(T))


def add_theta(ds: xr.Dataset, T: str = "T", p: str = "p") -> xr.Dataset:
    """Return a copy of ``ds`` with the potential temperature ``theta`` [K] added
    (float64), computed from the temperature ``T`` and the static pressure ``p``."""
    out = ds.copy()
    theta = xr.apply_ufunc(potential_temperature, ds[T], ds[p], keep_attrs=False)
    out["theta"] = theta.assign_attrs(long_name="Potential temperature", units="K")
    return out


# ── surface classes of the bulk method (manuscript App. B2, Table B1) ─────────

def classify_surface(T_s, T_open: float = -1.8, T_thick: float = -20.0) -> np.ndarray:
    """Surface class of each sample from the KT-19 surface temperature ``T_s`` [degC]:
    ``"ocean"`` above ``T_open``, ``"thick_snow_ice"`` at or below ``T_thick``,
    ``"thin_ice"`` in between (Table B1)."""
    T_s = np.asarray(T_s, dtype=float)
    return np.where(T_s > T_open, "ocean",
                    np.where(T_s <= T_thick, "thick_snow_ice", "thin_ice"))


def assign_roughness(surface_type, z0_ocean: float, z0_thin_ice: float, z0_thick_ice: float,
                     zt_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Momentum roughness ``z0`` [m] of each surface class and the scalar roughness
    ``zt = alpha_T * z0`` (manuscript App. B2). Used when no surface fractions are
    available; the paper's fluxes use the fraction-weighted blend in ``fluxpy.bulk``."""
    z0 = np.where(surface_type == "ocean", z0_ocean,
                  np.where(surface_type == "thick_snow_ice", z0_thick_ice, z0_thin_ice)).astype(float)
    return z0, z0 * zt_ratio

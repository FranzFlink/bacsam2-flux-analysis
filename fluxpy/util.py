"""
fluxpy.util
===========
Physical constants and lightweight helpers shared across the library.
Keep functions here small and unit-agnostic.
"""
from __future__ import annotations
import numpy as np

# ── constants ────────────────────────────────────────────────────────────────
CP     = 1004.0    # J kg⁻¹ K⁻¹  specific heat, dry air
RD     = 287.05    # J kg⁻¹ K⁻¹  gas constant,  dry air
G      = 9.80665   # m s⁻²
KAPPA  = 0.4       # von Kármán constant
RDCP   = RD / CP   # ≈ 0.2857  exponent in Poisson formula
P0     = 100000.0  # Pa  reference pressure (1000 hPa)


# ── unit helpers (private) ───────────────────────────────────────────────────
def _to_kelvin(T: np.ndarray) -> np.ndarray:
    """°C → K; passes through values already in K (anything > 200)."""
    T = np.asarray(T, dtype=float)
    return np.where(T > 200.0, T, T + 273.15)


def _to_pa(p: np.ndarray) -> np.ndarray:
    """hPa → Pa; passes through values already in Pa (> 2000)."""
    p = np.asarray(p, dtype=float)
    return np.where(p > 2000.0, p, p * 100.0)


# ── public helpers ────────────────────────────────────────────────────────────
def potential_temperature(T, p, p0: float = P0) -> np.ndarray:
    r"""
    Dry potential temperature (Poisson equation),

    $$ \theta = T \left( \frac{p_0}{p} \right)^{R_d / c_p} , $$

    the temperature an air parcel would attain if brought adiabatically to
    the reference pressure $p_0$. Using $\theta$ rather than $T$ makes
    temperatures at different flight levels directly comparable.

    Parameters
    ----------
    T  : air temperature — °C or K (auto-detected by a 200 K threshold)
    p  : pressure — hPa or Pa (auto-detected by a 2000 threshold)
    p0 : reference pressure [Pa], default 100 000 Pa
    """
    T_K  = _to_kelvin(T)
    p_Pa = _to_pa(p)
    return T_K * (p0 / p_Pa) ** RDCP


def air_density(T, p) -> np.ndarray:
    r"""
    Dry-air density from the ideal-gas law,

    $$ \rho = \frac{p}{R_d\,T} \quad [\mathrm{kg\,m^{-3}}]. $$

    Humidity corrections are negligible for the cold, dry conditions
    sampled during BACSAM-II. $T$ in °C or K, $p$ in hPa or Pa.
    """
    return _to_pa(p) / (RD * _to_kelvin(T))


def classify_surface(
    ist: np.ndarray,
    T_open:  float = -1.8,
    T_thick: float = -20.0,
) -> np.ndarray:
    """
    Map IST [°C] to surface class strings.

    Returns string ndarray: ``"ocean"`` | ``"thin_ice"`` | ``"thick_snow_ice"``
    """
    ist = np.asarray(ist, dtype=float)
    return np.where(
        ist > T_open,  "ocean",
        np.where(ist <= T_thick, "thick_snow_ice", "thin_ice"),
    )


def assign_roughness(
    surface_type:  np.ndarray,
    z0_ocean:      float,
    z0_thin_ice:   float,
    z0_thick_ice:  float,
    zt_ratio:      float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign momentum (z0) and scalar (zt) roughness lengths [m] from surface class.

    zt = z0 * zt_ratio   (ratio method only; kB⁻¹ method removed)
    """
    z0 = np.where(
        surface_type == "ocean",          z0_ocean,
        np.where(surface_type == "thick_snow_ice", z0_thick_ice, z0_thin_ice),
    ).astype(np.float64)
    zt = z0 * zt_ratio
    return z0, zt

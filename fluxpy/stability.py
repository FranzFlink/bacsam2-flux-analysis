r"""
fluxpy.stability
================
Near-surface stability of the bulk method (manuscript Appendix B1, Eq. B3).

* :func:`bulk_richardson_2pt` is THE Richardson number of the paper: the classic
  two-point bulk Richardson number between the surface (no-slip) and the
  measurement level, with the layer-mean potential temperature,

      Ri_b = (g / theta_bar) * dtheta * dz / (du^2 + dv^2 + eps dz^2).

  Every flux pathway uses it; there is no other Richardson formulation in the
  code. ``Ri_b > 0`` is stable, ``Ri_b < 0`` unstable.
* :func:`effective_wind_speed` is the wind that enters Ri_b for the
  reanalysis-driven pathways (Jordan et al., 1999, stable; Fairall et al., 1996,
  unstable; following Zhang et al., 2025). It bounds Ri_b at vanishing wind, so
  no regularisation and no clip are needed there.

The transfer-coefficient machinery (``SCHEMES``, ``Cd_Ch_noniterative``) lives
in :mod:`fluxpy.stability_lib` and is re-exported here for convenience.
"""
from __future__ import annotations

import numpy as np

from .stability_lib import SCHEMES, MOSTScheme, Cd_Ch_iterative, Cd_Ch_noniterative  # noqa: F401
from .util import G

__all__ = ["SCHEMES", "MOSTScheme", "Cd_Ch_noniterative", "Cd_Ch_iterative",
           "EPSILON_DEFAULT", "bulk_richardson_2pt", "effective_wind_speed"]

#: shear regularisation eps [m2 s-2] of the in-situ Ri_b (manuscript Appendix B1)
EPSILON_DEFAULT = 1e-4


def bulk_richardson_2pt(dtheta, dz, du=None, dv=None, theta_ref=None,
                        eps_wind2: float = EPSILON_DEFAULT, *, wspd=None) -> np.ndarray:
    r"""Two-point bulk Richardson number (manuscript Eq. B3),

    $$ Ri_b = \frac{g}{\bar\theta}\,
              \frac{\Delta\theta / \Delta z}{(\Delta U / \Delta z)^2 + \varepsilon}. $$

    Parameters
    ----------
    dtheta    : theta(top) - theta(surface) [K]
    dz        : height of the top level above the surface [m]
    du, dv    : wind components at the top level [m s-1] (surface wind = 0, no-slip);
                shear^2 = (du^2 + dv^2) / dz^2
    theta_ref : layer-mean potential temperature [K]
    eps_wind2 : regularisation of the shear denominator [m2 s-2]; 1e-4 for the
                measured (in-situ) wind, 0 with the effective wind
    wspd      : keyword alternative to ``du, dv``: a wind-speed difference [m s-1],
                shear^2 = (wspd / dz)^2 (used with :func:`effective_wind_speed`)

    All inputs broadcast; returns a float64 array.
    """
    if theta_ref is None:
        raise TypeError("bulk_richardson_2pt() missing required argument: 'theta_ref'")
    if wspd is not None:
        shear2 = (np.asarray(wspd, dtype=float) / dz) ** 2 + eps_wind2
    else:
        if du is None or dv is None:
            raise TypeError("bulk_richardson_2pt(): supply both 'du' and 'dv', or 'wspd'")
        shear2 = (du ** 2 + dv ** 2) / dz ** 2 + eps_wind2
    return (G / theta_ref) * (dtheta / dz) / shear2


def effective_wind_speed(U, dtheta, dz, theta_ref, *, beta: float = 1.25, zi: float = 600.0,
                         z0: float = 2e-4, zt_ratio: float = 1e-3, scheme: str = "GLGS20",
                         n_iter: int = 12) -> np.ndarray:
    r"""Effective wind speed $S$ for the reanalysis-driven Ri_b (manuscript Appendix B1).

    * stable (``dtheta >= 0``): windless exchange of Jordan et al. (1999),
      $S = U + 0.5 / \cosh U$, a smooth floor of 0.5 m/s at U = 0;
    * unstable (``dtheta < 0``): convective gustiness of Fairall et al. (1996),
      $S = \sqrt{U^2 + (\beta w_*)^2}$ with
      $w_* = (g/\bar\theta \cdot \overline{w'\theta'}_s \cdot z_i)^{1/3}$; the surface
      heat flux is obtained by a damped fixed-point iteration over $C_h$
      (``beta = 1.25``, ``zi = 600`` m as in COARE; representative roughness
      ``z0``, ``zt_ratio``).

    This is the treatment of Zhang et al. (2025) for ERA5-driven fluxes over sea
    ice. Parameters: ``U`` wind speed at ``dz`` [m/s]; ``dtheta`` = theta(dz) -
    theta_s [K]; ``theta_ref`` layer-mean theta [K]. Returns $S$ [m/s]
    (``S = U`` where inputs are NaN).
    """
    U = np.asarray(U, dtype=float)
    dtheta = np.asarray(dtheta, dtype=float) * np.ones_like(U)
    dz = np.asarray(dz, dtype=float) * np.ones_like(U)
    theta_ref = np.asarray(theta_ref, dtype=float) * np.ones_like(U)

    S = U.copy()
    ok = np.isfinite(U) & np.isfinite(dtheta) & np.isfinite(dz) & np.isfinite(theta_ref)

    stable = ok & (dtheta >= 0)                                   # Jordan et al. (1999)
    S[stable] = U[stable] + 0.5 / np.cosh(U[stable])

    zt = z0 * zt_ratio                                            # Fairall et al. (1996)
    for i in np.where(ok & (dtheta < 0))[0]:
        S_i = max(U[i], 0.01)
        for _ in range(n_iter):
            ri = (G / theta_ref[i]) * dtheta[i] * dz[i] / S_i ** 2
            try:
                _, ch = Cd_Ch_noniterative(float(np.clip(ri, -50.0, 50.0)), float(dz[i]), z0, zt, scheme)
            except Exception:
                ch = np.nan
            wth = -ch * S_i * dtheta[i] if np.isfinite(ch) else 0.0
            w_star = (G / theta_ref[i] * max(wth, 0.0) * zi) ** (1.0 / 3.0)
            S_new = float(np.hypot(U[i], beta * w_star))
            S_i = 0.5 * S_i + 0.5 * max(S_new, 0.01)
        S[i] = S_i
    return S

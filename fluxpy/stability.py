r"""
fluxpy.stability
================
Monin–Obukhov stability functions and Richardson-number diagnostics.

Monin–Obukhov similarity theory (Monin and Obukhov, 1954) relates the
turbulent fluxes in the surface layer to the mean vertical gradients of
wind and potential temperature. The near-surface stability is expressed
here through the bulk Richardson number $Ri_b$, the ratio of buoyant
suppression (or production) of turbulence to its production by wind shear.
The stability-scheme objects (``SCHEMES``) implement the transfer-coefficient
formulations compared in the manuscript: Businger–Dyer
(Businger et al., 1971, doi:10.1175/1520-0469(1971)028<0181:FPRITA>2.0.CO;2;
Dyer, 1974, doi:10.1007/BF00240838), Holtslag and de Bruin (1988),
Beljaars and Holtslag (1991), Cheng and Brutsaert (2005),
Grachev et al. (2007), Zilitinkevich and Esau (2007), and the
SHEBA-based functions of Gryanik et al. (2020,
doi:10.1175/JAS-D-19-0255.1), following the implementation of
Gryanik and Lüpkes (2023, doi:10.1007/s10546-022-00730-9).

Scheme objects and transfer-coefficient functions are re-exported from
``lib/stability_lib.py``; this module adds the vectorised two-point bulk
Richardson number — the $Ri_b$ of the manuscript (Eq. rib), used by every
flux pathway. No other Richardson formulation exists in the code base;
the historical three-point gradient estimator lives only in the dedicated
analysis notebook ``book/notebooks/data_analysis/essential_ri_b_analysis.ipynb``.
"""
from __future__ import annotations

import numpy as np

from .stability_lib import (
    SCHEMES,
    MOSTScheme,
    bulk_richardson,                  # original scalar 2-pt Ri
    fm_fh_iterative,
    fm_fh_noniterative,
    Cd_Ch_noniterative,
    Cd_Ch_iterative,
    epsilons,
    kinematic_viscosity,
    zt_open_water,
    zt_sea_ice,
)

from .util import G

__all__ = [
    "SCHEMES", "MOSTScheme",
    "bulk_richardson",
    "fm_fh_iterative", "fm_fh_noniterative",
    "Cd_Ch_noniterative", "Cd_Ch_iterative",
    "epsilons",
    "bulk_richardson_2pt",
    "effective_wind_speed",
    "kinematic_viscosity",
    "zt_open_water",
    "zt_sea_ice",
]

EPSILON_DEFAULT = 1e-4


# ── vectorised 2-pt bulk Richardson ──────────────────────────────────────────
def bulk_richardson_2pt(
    dtheta:    np.ndarray,
    dz:        np.ndarray,
    du:        np.ndarray | None = None,
    dv:        np.ndarray | None = None,
    theta_ref: np.ndarray = None,
    eps_wind2: float = EPSILON_DEFAULT,
    *,
    wspd: np.ndarray | None = None,
) -> np.ndarray:
    r"""
    Bulk Richardson number from finite differences,

    $$ Ri_b = \frac{g}{\theta_\mathrm{ref}}\,
              \frac{\Delta\theta/\Delta z}
                   {\left(\Delta U/\Delta z\right)^2 + \varepsilon} , $$

    the ratio of buoyant damping (or generation) of turbulence to shear
    production between two levels. $Ri_b > 0$ indicates stable
    stratification, $Ri_b < 0$ unstable conditions. The regularisation
    $\varepsilon$ (``eps_wind2``, m² s⁻²) prevents division by a vanishing
    shear in near-calm conditions. All inputs are broadcast-compatible;
    returns $Ri_b$ as a float64 array.

    Wind shear can be supplied in two ways:
        * **Component mode**: pass ``du`` (Δu) and ``dv`` (Δv).
          Shear² = (Δu² + Δv²) / Δz²
        * **Scalar mode**: pass ``wspd`` (Δ|U|, absolute wind-speed difference).
          ``du`` and ``dv`` are not used.
          Shear² = (Δ|U| / Δz)²

    Parameters
    ----------
    dtheta    : potential-temperature difference  Δθ = θ_top − θ_bot  [K]
    dz        : height difference  Δz = z_top − z_bot  [m]
    du        : Δu wind component  [m/s]  (component mode only)
    dv        : Δv wind component  [m/s]  (component mode only)
    theta_ref : reference potential temperature  [K]
    eps_wind2 : small regularisation term  [m²/s²]
    wspd      : absolute wind-speed difference  Δ|U|  [m/s]  (keyword-only, scalar mode)
    """
    if theta_ref is None:
        raise TypeError("bulk_richardson_2pt() missing required argument: 'theta_ref'")

    if wspd is not None:
        # scalar (absolute wind-speed) mode — du/dv not needed
        shear2 = (np.asarray(wspd, dtype=float) / dz) ** 2 + eps_wind2
    else:
        # component (u / v) mode — both du and dv required
        if du is None or dv is None:
            raise TypeError(
                "bulk_richardson_2pt(): supply both 'du' and 'dv' (component mode) "
                "or 'wspd' (scalar mode)"
            )
        shear2 = (du ** 2 + dv ** 2) / dz ** 2 + eps_wind2

    return (G / theta_ref) * (dtheta / dz) / shear2


# ── effective wind speed: natural calm-limit bound for Ri_b ──────────────────
def effective_wind_speed(
    U:         np.ndarray,
    dtheta:    np.ndarray,
    dz:        np.ndarray,
    theta_ref: np.ndarray,
    *,
    beta:     float = 1.25,
    zi:       float = 600.0,
    z0:       float = 2e-4,
    zt_ratio: float = 1e-3,
    scheme:   str   = "GLGS20",
    n_iter:   int   = 12,
) -> np.ndarray:
    r"""
    Effective wind speed $S$ that bounds the calm limit of the bulk formulas
    — the treatment adopted for reanalysis-driven bulk fluxes over Arctic
    sea ice by Zhang et al. (2025, doi:10.1016/j.accre.2025.04.017, their
    Sect. 3.2.2), and by operational models in place of any $Ri_b$ clipping:

    * **stable** ($\Delta\theta \ge 0$): windless exchange of Jordan et al.
      (1999, doi:10.1029/1999JC900011),

      $$ S = U + 0.5 / \cosh(U), $$

      a smooth floor of 0.5 m s⁻¹ at $U = 0$ that decays to $S \approx U$
      in moderate winds;
    * **unstable** ($\Delta\theta < 0$): convective gustiness of Fairall
      et al. (1996, doi:10.1029/95JC03205),

      $$ S = \sqrt{U^2 + (\beta w_*)^2}, \qquad
         w_* = \left(\tfrac{g}{\bar\theta}\,\overline{w'\theta'}_s\,
                     z_i\right)^{1/3}, $$

      with the surface kinematic heat flux obtained by a damped fixed-point
      iteration over $C_h$ (defaults $\beta = 1.25$, $z_i = 600$ m as in
      COARE; representative roughness ``z0``/``zt_ratio``).

    Using $S$ in the shear of the two-point bulk Richardson number bounds
    $Ri_b$ naturally on both branches, so reanalysis-driven pathways need
    neither the $\varepsilon$ regularisation nor an explicit clip.

    Parameters: ``U`` wind speed at ``dz`` [m/s]; ``dtheta`` = θ(dz) − θ_s
    [K]; ``dz`` evaluation height [m]; ``theta_ref`` layer-mean θ [K].
    Returns $S$ [m/s] as a float64 array (``S = U`` where inputs are NaN).
    """
    U         = np.asarray(U, dtype=float)
    dtheta    = np.asarray(dtheta, dtype=float) * np.ones_like(U)
    dz        = np.asarray(dz, dtype=float) * np.ones_like(U)
    theta_ref = np.asarray(theta_ref, dtype=float) * np.ones_like(U)

    S = U.copy()
    ok = np.isfinite(U) & np.isfinite(dtheta) & np.isfinite(dz) & np.isfinite(theta_ref)

    # stable / neutral: closed form (Jordan et al. 1999)
    stab = ok & (dtheta >= 0)
    S[stab] = U[stab] + 0.5 / np.cosh(U[stab])

    # unstable: iterate w* with C_h (Fairall et al. 1996)
    zt = z0 * zt_ratio
    for i in np.where(ok & (dtheta < 0))[0]:
        S_i = max(U[i], 0.01)
        for _ in range(n_iter):
            ri = (G / theta_ref[i]) * dtheta[i] * dz[i] / S_i ** 2
            try:
                _, ch = Cd_Ch_noniterative(float(np.clip(ri, -50.0, 50.0)),
                                           float(dz[i]), z0, zt, scheme)
            except Exception:
                ch = np.nan
            wth = -ch * S_i * dtheta[i] if np.isfinite(ch) else 0.0
            w_star = (G / theta_ref[i] * max(wth, 0.0) * zi) ** (1.0 / 3.0)
            S_new = float(np.hypot(U[i], beta * w_star))
            S_i = 0.5 * S_i + 0.5 * max(S_new, 0.01)
        S[i] = S_i
    return S

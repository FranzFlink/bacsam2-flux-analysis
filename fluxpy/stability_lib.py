"""
Implementation of:
Gryanik & Lüpkes (2023), Boundary-Layer Meteorology 187:41–72
"A Package of Momentum and Heat Transfer Coefficients for the Stable Surface Layer
 Extended by New Coefficients over Sea Ice"
doi:10.1007/s10546-022-00730-9

This module:
- Defines stability correction functions (SCFs) ψ_m, ψ_h for all schemes in Table 1
- Implements the non-iterative universal parametrization for ζ(Ri_b, ε_m, ε_t)
- Computes normalized transfer coefficients f_m, f_h
- Computes bulk transfer coefficients C_d, C_h (momentum, heat)
- Provides an iterative ζ solver for verification against the exact MOST equation

 Notes:
- Stable surface layer primary focus; Ri_b < 0 is handled with standard Businger-Dyer unstable ψ functions.
- ε_m = z/z0, ε_t = z/z_t
- κ = 0.4; g = 9.80665 m/s²
- ψ_h uses the "multiplicative" convention (Pr0 absorbed), as in the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict
import logging


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

KAPPA = 0.4
G = 9.80665  # m s^-2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Micro-meteorological equations (Andreas 2002, Fairall 2003)
# ---------------------------------------------------------------------

def kinematic_viscosity(T_c: float) -> float:
    """
    Kinematic viscosity of air [m^2/s] as a function of temperature T_c [C].
    Following TOGA COARE or standard meteorological parameterizations.
    """
    return 1.326e-5 * (1.0 + 6.542e-3 * T_c + 8.301e-6 * (T_c ** 2) - 4.84e-9 * (T_c ** 3))


def zt_open_water(u_star: float, z0: float, T_c: float) -> float:
    """
    Converting the roughness length of momentum to that of heat over open water.
    Following COARE 3.0 / Fairall et al. (2003).
    """
    if u_star <= 0.0 or z0 <= 0.0:
        return 1.1e-4
    nu = kinematic_viscosity(T_c)
    Re_star = u_star * z0 / nu
    if Re_star <= 0.0:
        return 1.1e-4
    return min(1.1e-4, 5.5e-5 * (Re_star ** -0.6))


def zt_sea_ice(u_star: float, z0: float, T_c: float) -> float:
    """
    Converting the roughness length of momentum to that of heat over sea ice.
    Following Andreas (2002).
    """
    if u_star <= 0.0 or z0 <= 0.0:
        return z0 * math.exp(1.250)
    nu = kinematic_viscosity(T_c)
    Re_star = u_star * z0 / nu
    if Re_star <= 0.0:
        return z0 * math.exp(1.250)

    # Andreas 2002 empirical coefficients for sensible heat
    if Re_star <= 0.135:
        b0, b1, b2 = 1.250, 0.0, 0.0
    elif Re_star < 2.5:
        b0, b1, b2 = 0.149, -0.550, 0.0
    else:
        b0, b1, b2 = 0.317, -0.565, -0.183

    ln_Re = math.log(Re_star)
    ln_ratio = b0 + b1 * ln_Re + b2 * (ln_Re ** 2)
    # Clamp to prevent extreme out-of-bound numerical artifacts
    return z0 * math.exp(min(max(ln_ratio, -15.0), 10.0))


# ---------------------------------------------------------------------
# Data structure for a MOST scheme
# ---------------------------------------------------------------------

PsiFunc = Callable[[float], float]


@dataclass
class MOSTScheme:
    name: str
    psi_m: PsiFunc
    psi_h: PsiFunc
    Pr0: float
    gamma: float   # from Table 1, used in Eq. (22a)
    zeta_a: float  # from Table 1, used in Eq. (22a)
    zeta_max: float | None = None  # nominal validity range (optional)


SCHEMES: Dict[str, MOSTScheme] = {}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _clip_zeta(zeta: float) -> float:
    # Ensure non-negative and finite
    if not math.isfinite(zeta):
        return 0.0
    return max(zeta, 0.0)


def _clip_zeta_signed(zeta: float, min_z: float = -20.0, max_z: float = 50.0) -> float:
    """Clip ζ to a finite range but preserve sign for unstable cases."""
    if not math.isfinite(zeta):
        return 0.0
    return min(max(zeta, min_z), max_z)


def epsilons(z: float, z0: float, zt: float) -> tuple[float, float]:
    if z <= 0 or z0 <= 0 or zt <= 0:
        raise ValueError("z, z0, zt must be positive.")
    return z / z0, z / zt


def bulk_richardson(
    U: float,
    theta_v_z: float,
    theta_v_surface: float,
    z: float,
    z0: float,
    zt: float,
) -> float:
    """
    Bulk Richardson number Ri_b following Eq. (21).

    U: mean wind speed at height z (m/s)
    theta_v_z: virtual potential temperature at z (K)
    theta_v_surface: virtual potential temperature at (quasi-)surface level zt (K)
    z: reference height (m)
    z0: roughness length for momentum (m)
    zt: roughness/measurement height for temperature (m)
    """
    if z <= z0 or z <= zt:
        raise ValueError("Require z > z0 and z > zt.")
    if U == 0:
        # Formally Ri_b → ∞; return large number
        return float("inf")

    num = (G / theta_v_surface) * ((theta_v_z - theta_v_surface) / (z - zt))
    den = (U / (z - z0)) ** 2
    return num / den

# ---------------------------------------------------------------------
# SCF definitions (ψ_m, ψ_h) for each scheme
# ---------------------------------------------------------------------

def _make_bd() -> MOSTScheme:
    # Businger-Dyer, Eq. (11)
    am = 5.0
    ah = 5.0
    Pr0 = 1.0

    def psi_m(zeta: float) -> float:
        return -am * zeta

    def psi_h(zeta: float) -> float:
        return -Pr0 * ah * zeta

    return MOSTScheme("BD", psi_m, psi_h, Pr0, gamma=4.42, zeta_a=2.5, zeta_max=1.0)


def _make_bd_unstable() -> MOSTScheme:
    # Standard Businger-Dyer unstable forms
    Pr0 = 1.0

    def psi_m(zeta: float) -> float:
        if zeta >= 0.0:
            return -5.0 * zeta
        x = (1.0 - 16.0 * zeta) ** 0.25
        return (
            2.0 * math.log((1.0 + x) / 2.0)
            + math.log((1.0 + x * x) / 2.0)
            - 2.0 * math.atan(x)
            + math.pi / 2.0
        )

    def psi_h(zeta: float) -> float:
        if zeta >= 0.0:
            return -Pr0 * 5.0 * zeta
        x = (1.0 - 16.0 * zeta) ** 0.25
        return 2.0 * math.log((1.0 + x * x) / 2.0)

    return MOSTScheme("BD-unstable", psi_m, psi_h, Pr0, gamma=4.42, zeta_a=2.5)


def _make_hb88() -> MOSTScheme:
    # Holtslag & De Bruin (1988), Eq. (12)
    a, b, c, d = 0.7, 0.75, 5.0, 0.35
    Pr0 = 1.0

    def psi(zeta: float) -> float:
        return -a * zeta - b * (zeta - c / d) * math.exp(-d * zeta) - (b * c / d)

    return MOSTScheme("HB88", psi, psi, Pr0, gamma=2.14, zeta_a=4.0, zeta_max=10.0)


def _make_bh91() -> MOSTScheme:
    # Beljaars & Holtslag (1991):
    # ψ_m: Eq. (12) with a=1, b=0.667, c=5, d=0.35
    # ψ_h: Eq. (13) with same constants
    a, b, c, d = 1.0, 0.667, 5.0, 0.35
    Pr0 = 1.0

    def psi_m(zeta: float) -> float:
        return -a * zeta - b * (zeta - c / d) * math.exp(-d * zeta) - (b * c / d)

    def psi_h(zeta: float) -> float:
        term1 = 1.0 - (1.0 + (2.0 * a / 3.0) * zeta) ** 1.5
        term2 = -b * (zeta - c / d) * math.exp(-d * zeta) - (b * c / d)
        return term1 + term2

    return MOSTScheme("BH91", psi_m, psi_h, Pr0, gamma=1.71, zeta_a=4.8, zeta_max=10.0)


def _make_bh91_ecmwf() -> MOSTScheme:
    # ECMWF variant: ψ_m uses HB88 form with BH91 constants; ψ_h as BH91
    a, b, c, d = 1.0, 0.667, 5.0, 0.35
    Pr0 = 1.0

    def psi_m(zeta: float) -> float:
        return -a * zeta - b * (zeta - c / d) * math.exp(-d * zeta) - (b * c / d)

    def psi_h(zeta: float) -> float:
        term1 = 1.0 - (1.0 + (2.0 * a / 3.0) * zeta) ** 1.5
        term2 = -b * (zeta - c / d) * math.exp(-d * zeta) - (b * c / d)
        return term1 + term2

    return MOSTScheme(
        "BH91/ECMWF",
        psi_m,
        psi_h,
        Pr0,
        gamma=1.81,
        zeta_a=5.2,
        zeta_max=10.0,
    )


def _make_cb05() -> MOSTScheme:
    # Cheng & Brutsaert (2005), Eq. (14)
    a_m, b_m = 6.1, 2.5
    a_h, b_h = 5.3, 1.1
    Pr0 = 1.0

    def psi_m(zeta: float) -> float:
        return -a_m * math.log(zeta + (1.0 + zeta ** b_m) ** (1.0 / b_m))

    def psi_h(zeta: float) -> float:
        return -a_h * math.log(zeta + (1.0 + zeta ** b_h) ** (1.0 / b_h))

    return MOSTScheme("CB05", psi_m, psi_h, Pr0, gamma=2.28, zeta_a=4.5, zeta_max=5.0)


def _make_gafgp07() -> MOSTScheme:
    # Grachev et al. (2007); ψ_m, ψ_h are lengthy (Eq. 15a,b).
    # Implemented directly from published form.
    am, bm = 5.0, 0.77
    ah, bh, ch = 5.0, 5.0, 3.0
    Pr0 = 1.0

    Bm = (1.0 / bm - 1.0) ** (1.0 / 3.0)
    Bh = math.sqrt(ch ** 2 - 4.0)

    # Helper for ψ_m
    def psi_m(zeta: float) -> float:
        if zeta < 0.0:
            return 0.0
        x = (1.0 + zeta) ** (1.0 / 3.0)
        # This expression follows Eq. (15a) in compact code form
        term1 = x - 1.0
        # Log/atan composite term:
        # Carefully grouped to avoid exceeding text limits / direct copying.
        num1 = (x ** 2 - x * Bm + Bm ** 2)
        den1 = (1.0 - Bm + Bm ** 2)
        log_part = math.log(num1 / den1)
        atan_part = (
            math.atan((2.0 * x - Bm) / (math.sqrt(3.0) * Bm))
            - math.atan((2.0 - Bm) / (math.sqrt(3.0) * Bm))
        )
        bracket = (
            term1
            - (Bm / 6.0) * (2.0 * math.log(x) + log_part - 2.0 * math.sqrt(3.0) * atan_part)
        )
        return -3.0 * am / bm * bracket

    def psi_h(zeta: float) -> float:
        if zeta < 0.0:
            return 0.0
        # Eq. (15b)
        term_log = math.log(
            (2.0 * zeta + ch - Bh) / (2.0 * zeta + ch + Bh)
            * ((ch + Bh) / (ch - Bh))
        )
        first = (ah / Bh - bh * ch / (2.0 * Bh)) * term_log
        second = -0.5 * bh * math.log(1.0 + ch * zeta + zeta ** 2)
        return Pr0 * (first + second)

    return MOSTScheme(
        "GAFGP07", psi_m, psi_h, Pr0, gamma=2.91, zeta_a=3.6, zeta_max=100.0
    )


def _make_glgs20() -> MOSTScheme:
    # Gryanik et al. (2020), Eq. (19a,b)
    am, ah = 5.0, 5.0
    bm, bh = 0.3, 0.4
    Pr0 = 0.98

    def psi_m(zeta: float) -> float:
        return -3.0 * am / bm * ((1.0 + bm * zeta) ** (1.0 / 3.0) - 1.0)

    def psi_h(zeta: float) -> float:
        return -Pr0 * (ah / bh) * math.log(1.0 + bh * zeta)

    return MOSTScheme(
        "GLGS20", psi_m, psi_h, Pr0, gamma=3.66, zeta_a=10.2, zeta_max=100.0
    )


def _make_ze07_I(Pr0: float = 0.85, am: float = 5.0, ah: float = 4.0, bh: float = 1.25,
                 gamma: float = 1.80, zeta_a: float = 10.0,
                 name: str = "ZE07-I") -> MOSTScheme:
    # ZE07-I, Eq. (16)
    def psi_m(zeta: float) -> float:
        return -am * zeta

    def psi_h(zeta: float) -> float:
        return -Pr0 * (ah * zeta + 0.5 * bh * zeta ** 2)

    return MOSTScheme(name, psi_m, psi_h, Pr0, gamma=gamma, zeta_a=zeta_a, zeta_max=1.0)


def _make_ze07_II(Pr0: float = 0.85, cm: float = 6.44, ch: float = 5.2,
                  gamma: float = 4.64, zeta_a: float = 14.6,
                  name: str = "ZE07-II") -> MOSTScheme:
    # ZE07-II, Eq. (17)
    def psi_m(zeta: float) -> float:
        return -cm * (zeta ** (5.0 / 6.0))

    def psi_h(zeta: float) -> float:
        return -Pr0 * ch * (zeta ** (4.0 / 5.0))

    return MOSTScheme(name, psi_m, psi_h, Pr0, gamma=gamma, zeta_a=zeta_a, zeta_max=125.0)


def _make_zekre13(Pr0: float = 0.8,
                  am: float = 4.0,
                  ah: float = 4.5,
                  bh: float = 1.13,
                  ch_c: float = -0.0062,
                  dh: float = 3.55,
                  gamma: float = 1.21,
                  zeta_a: float = 9.7,
                  name: str = "ZEKRE13") -> MOSTScheme:
    # ZEKRE13, Eq. (18a,b)
    def psi_m(zeta: float) -> float:
        return -am * zeta

    def psi_h(zeta: float) -> float:
        # Here ch_c is c_h in the paper; named to avoid clash.
        lin = (ah - ch_c * dh) * zeta
        quad = 0.5 * bh * zeta ** 2
        logt = ch_c * math.log(1.0 + dh * zeta)
        return -Pr0 * (lin + quad + logt)

    return MOSTScheme(name, psi_m, psi_h, Pr0, gamma=gamma, zeta_a=zeta_a, zeta_max=1.0)


# Modified variants (ZE07-I/GL, ZE07-II/GL, ZEKRE13/GL), Sect. 6.4
def _make_ze07_I_GL() -> MOSTScheme:
    # Eq. (31), (34)
    return _make_ze07_I(
        Pr0=0.98,
        am=5.0,
        ah=5.0,
        bh=1.25,
        gamma=1.70,
        zeta_a=11.0,
        name="ZE07-I/GL",
    )


def _make_ze07_II_GL() -> MOSTScheme:
    # Eq. (32), (35)
    return _make_ze07_II(
        Pr0=0.98,
        cm=5.5,
        ch=5.2,
        gamma=4.30,
        zeta_a=14.0,
        name="ZE07-II/GL",
    )


def _make_zekre13_GL() -> MOSTScheme:
    # Eq. (33), (36)
    return _make_zekre13(
        Pr0=0.7,
        am=4.0,
        ah=4.5,
        bh=1.13,
        ch_c=-0.0062,
        dh=3.55,
        gamma=1.71,
        zeta_a=11.3,
        name="ZEKRE13/GL",
    )


# ---------------------------------------------------------------------
# Register schemes
# ---------------------------------------------------------------------

def _register_schemes():
    for scheme in [
        _make_bd(),
        _make_hb88(),
        _make_bh91(),
        _make_bh91_ecmwf(),
        _make_cb05(),
        _make_gafgp07(),
        _make_glgs20(),
        _make_ze07_I(),
        _make_ze07_II(),
        _make_zekre13(),
        _make_ze07_I_GL(),
        _make_ze07_II_GL(),
        _make_zekre13_GL(),
    ]:
        SCHEMES[scheme.name] = scheme


_register_schemes()
UNSTABLE_SCHEME = _make_bd_unstable()


def _scheme_for_ri(Ri_b: float, scheme_name: str) -> MOSTScheme:
    if Ri_b < 0.0:
        if scheme_name not in SCHEMES and scheme_name != UNSTABLE_SCHEME.name:
            raise KeyError(f"Unknown scheme '{scheme_name}'")
        return UNSTABLE_SCHEME
    if scheme_name not in SCHEMES:
        raise KeyError(f"Unknown scheme '{scheme_name}'")
    return SCHEMES[scheme_name]


def _scheme_for_zeta(zeta: float, scheme_name: str) -> MOSTScheme:
    if zeta < 0.0:
        if scheme_name not in SCHEMES and scheme_name != UNSTABLE_SCHEME.name:
            raise KeyError(f"Unknown scheme '{scheme_name}'")
        return UNSTABLE_SCHEME
    if scheme_name not in SCHEMES:
        raise KeyError(f"Unknown scheme '{scheme_name}'")
    return SCHEMES[scheme_name]


# ---------------------------------------------------------------------
# Non-iterative ζ(Ri_b, ε_m, ε_t) following Eq. (22a–c)
# ---------------------------------------------------------------------

def zeta_noniterative(
    Ri_b: float,
    eps_m: float,
    eps_t: float,
    scheme_name: str,
) -> float:
    """
    Semi-analytical non-iterative approximation ζ(Ri_b, ε_m, ε_t).

    Implements Eq. (22a–c) using ψ_m, ψ_h, Pr0, γ, ζ_a of the given scheme.
    """
    if Ri_b <= 0.0:
        return 0.0

    if scheme_name not in SCHEMES:
        raise KeyError(f"Unknown scheme '{scheme_name}'")

    s = SCHEMES[scheme_name]

    ln_em = math.log(eps_m)
    ln_et = math.log(eps_t)

    # Equivalent bulk Richardson number, Eq. (22c)
    hat_Ri = (Ri_b / s.Pr0) * ((1.0 - 1.0 / eps_t) / ((1.0 - 1.0 / eps_m) ** 2))

    # ψ_ka, Eq. (22b); note: paper uses ε_t for both k = m,h
    psi_ma = s.psi_m(s.zeta_a) - s.psi_m(s.zeta_a / eps_t)
    psi_ha = s.psi_h(s.zeta_a) - s.psi_h(s.zeta_a / eps_t)

    # First (linear) term in hat_Ri
    term1 = (ln_em ** 2 / ln_et) * hat_Ri

    # Second (nonlinear) term ~ hat_Ri^γ
    num = (ln_em - psi_ma) ** (2.0 * (s.gamma - 1.0))
    den = (s.zeta_a ** (s.gamma - 1.0)) * (
        (ln_et - psi_ha / s.Pr0) ** (s.gamma - 1.0)
    )

    if den == 0.0:
        return _clip_zeta(term1)

    bracket = (
        ((ln_em - psi_ma) ** 2) / (ln_et - psi_ha / s.Pr0)
        - (ln_em ** 2) / ln_et
    )

    term2 = (num / den) * bracket * (hat_Ri ** s.gamma)

    return _clip_zeta(term1 + term2)


# ---------------------------------------------------------------------
# f_m, f_h from ζ (non-iterative or iterative)
# ---------------------------------------------------------------------

def fm_fh_from_zeta(
    zeta: float,
    eps_m: float,
    eps_t: float,
    scheme_name: str,
    verbose: bool = False,
) -> tuple[float, float]:
    """
    Compute normalized transfer coefficients f_m, f_h from ζ using Eq. (7)-(8).
    """
    s = _scheme_for_zeta(zeta, scheme_name)
    ln_em = math.log(eps_m)
    ln_et = math.log(eps_t)

    psi_m_z = s.psi_m(zeta)
    psi_m_zeps = s.psi_m(zeta / eps_m)

    A = 1.0 - (psi_m_z - psi_m_zeps) / ln_em
    fm = 1.0 / (A * A)

    psi_h_z = s.psi_h(zeta)
    psi_h_zeps = s.psi_h(zeta / eps_t)

    B = 1.0 - (psi_h_z - psi_h_zeps) / (s.Pr0 * ln_et)

    fh = 1.0 / (A * B)

    if verbose:
        logger.info(
            "fm/fh from zeta: scheme=%s zeta=%.4f eps_m=%.3f eps_t=%.3f fm=%.4f fh=%.4f",
            s.name,
            zeta,
            eps_m,
            eps_t,
            fm,
            fh,
        )

    return fm, fh


def fm_fh_noniterative(
    Ri_b: float,
    z: float,
    z0: float,
    zt: float,
    scheme_name: str,
    verbose: bool = False,
) -> tuple[float, float]:
    """
    Convenience wrapper:
    - builds ε_m, ε_t from z, z0, zt
    - computes ζ via the universal non-iterative formula
    - returns f_m, f_h
    """
    eps_m, eps_t = epsilons(z, z0, zt)
    if Ri_b < 0.0:
        zeta = zeta_iterative(Ri_b, eps_m, eps_t, scheme_name, verbose=verbose)
        return fm_fh_from_zeta(zeta, eps_m, eps_t, scheme_name, verbose=verbose)
    if Ri_b == 0.0:
        return 1.0, 1.0
    zeta = zeta_noniterative(Ri_b, eps_m, eps_t, scheme_name)
    return fm_fh_from_zeta(zeta, eps_m, eps_t, scheme_name, verbose=verbose)


# ---------------------------------------------------------------------
# Iterative solver for ζ (reference "exact" MOST solution)
# ---------------------------------------------------------------------

def zeta_iterative(
    Ri_b: float,
    eps_m: float,
    eps_t: float,
    scheme_name: str,
    max_iter: int = 80,
    tol: float = 1e-6,
    verbose: bool = False,
) -> float:
    """
    Solve Eq. (20) iteratively for ζ using Newton's method with a finite-difference derivative.

    Intended for verification of the non-iterative parametrization. For Ri_b < 0 the
    standard Businger-Dyer unstable ψ functions are used.
    """
    if Ri_b == 0.0:
        return 0.0

    s = _scheme_for_ri(Ri_b, scheme_name)

    ln_em = math.log(eps_m)
    ln_et = math.log(eps_t)

    # Start from non-iterative estimate as initial guess (stable), or a small
    # negative magnitude (unstable).
    if Ri_b > 0.0:
        zeta = zeta_noniterative(Ri_b, eps_m, eps_t, scheme_name)
        if zeta <= 0.0:
            zeta = 0.05
    else:
        zeta = max(-0.5, -0.1 * abs(Ri_b))

    # Precompute constant prefactor from Eq. (20)
    pref = ((1.0 - 1.0 / eps_m) ** 2) / (1.0 - 1.0 / eps_t)
    clip_fn = _clip_zeta if Ri_b > 0.0 else _clip_zeta_signed

    for _ in range(max_iter):
        psi_m = s.psi_m(zeta)
        psi_m_eps = s.psi_m(zeta / eps_m)
        psi_h = s.psi_h(zeta)
        psi_h_eps = s.psi_h(zeta / eps_t)

        num = pref * zeta * (s.Pr0 * ln_et - psi_h + psi_h_eps)
        den = (ln_em - psi_m + psi_m_eps) ** 2

        if den == 0.0:
            break

        f = num / den - Ri_b

        if abs(f) < tol:
            break

        # Finite-difference derivative
        dz = 1e-4 * max(1.0, zeta)
        z2 = zeta + dz
        psi_m2 = s.psi_m(z2)
        psi_m2_eps = s.psi_m(z2 / eps_m)
        psi_h2 = s.psi_h(z2)
        psi_h2_eps = s.psi_h(z2 / eps_t)

        num2 = pref * z2 * (s.Pr0 * ln_et - psi_h2 + psi_h2_eps)
        den2 = (ln_em - psi_m2 + psi_m2_eps) ** 2
        if den2 == 0.0:
            break

        f2 = num2 / den2 - Ri_b
        df = (f2 - f) / dz
        if df == 0.0 or not math.isfinite(df):
            break

        zeta_new = zeta - f / df
        zeta_new = clip_fn(zeta_new)

        if abs(zeta_new - zeta) < tol:
            zeta = zeta_new
            break

        zeta = zeta_new

    zeta = clip_fn(zeta)
    if verbose:
        logger.info(
            "zeta_iterative: scheme=%s Ri=%.4f eps_m=%.3f eps_t=%.3f zeta=%.4f",
            s.name,
            Ri_b,
            eps_m,
            eps_t,
            zeta,
        )
    return zeta


def fm_fh_iterative(
    Ri_b: float,
    z: float,
    z0: float,
    zt: float,
    scheme_name: str,
    verbose: bool = False,
    **kwargs,
) -> tuple[float, float]:
    """
    f_m, f_h using iterative ζ solution of Eq. (20).
    """
    eps_m, eps_t = epsilons(z, z0, zt)
    if Ri_b <= 0.0:
        zeta = zeta_iterative(Ri_b, eps_m, eps_t, scheme_name, verbose=verbose, **kwargs)
        return fm_fh_from_zeta(zeta, eps_m, eps_t, scheme_name, verbose=verbose)
    zeta = zeta_iterative(Ri_b, eps_m, eps_t, scheme_name, verbose=verbose, **kwargs)
    return fm_fh_from_zeta(zeta, eps_m, eps_t, scheme_name, verbose=verbose)


# ---------------------------------------------------------------------
# Bulk transfer coefficients C_d, C_h
# ---------------------------------------------------------------------

def Cd_Ch_ECHAM6(
    Ri_b: float,
    z: float,
    z0: float,
    zt: float,
    verbose: bool = False,
) -> tuple[float, float]:
    """
    Direct Bulk stability correction functions according to Giorgetta et al. (2013) / ECHAM6.
    """
    eps_m, eps_t = epsilons(z, z0, zt)
    ln_em = math.log(eps_m)
    ln_et = math.log(eps_t)
    C_dn = KAPPA ** 2 / (ln_em ** 2)
    C_hn = KAPPA ** 2 / (ln_em * ln_et)

    c_const = 5.0
    a_m = 10.0
    a_h = 15.0

    if Ri_b >= 0.0:
        # Stable
        fm = 1.0 / (1.0 + a_m * Ri_b * math.sqrt(1.0 + Ri_b))
        fh = 1.0 / (1.0 + a_h * Ri_b * math.sqrt(1.0 + Ri_b))
    else:
        # Unstable
        denom_term = 3.0 * (c_const ** 2) * C_dn * math.sqrt(-Ri_b * (z / z0 + 1.0))
        fm = 1.0 - (a_m * Ri_b) / (1.0 + denom_term)
        fh = 1.0 - (a_h * Ri_b) / (1.0 + denom_term)

    Cd, Ch = C_dn * fm, C_hn * fh
    if verbose:
        logger.info(
            "Cd/Ch ECHAM6: Ri=%.4f Cd=%.6f Ch=%.6f",
            Ri_b,
            Cd,
            Ch,
        )
    return Cd, Ch


def Cd_Ch_noniterative(
    Ri_b: float,
    z: float,
    z0: float,
    zt: float,
    scheme_name: str,
    verbose: bool = False,
) -> tuple[float, float]:
    """
    Bulk transfer coefficients (C_d, C_h) using non-iterative NTCs.

    C_dn, C_hn from Eq. (6); f_m, f_h from fm_fh_noniterative.
    """
    if scheme_name == "ECHAM6":
        return Cd_Ch_ECHAM6(Ri_b, z, z0, zt, verbose=verbose)

    s = _scheme_for_ri(Ri_b, scheme_name)
    eps_m, eps_t = epsilons(z, z0, zt)

    ln_em = math.log(eps_m)
    ln_et = math.log(eps_t)

    C_dn = KAPPA ** 2 / (ln_em ** 2)
    C_hn = KAPPA ** 2 / (s.Pr0 * ln_em * ln_et)

    fm, fh = fm_fh_noniterative(Ri_b, z, z0, zt, scheme_name, verbose=verbose)
    Cd, Ch = C_dn * fm, C_hn * fh
    if verbose:
        logger.info(
            "Cd/Ch noniterative: scheme=%s Ri=%.4f Cd=%.6f Ch=%.6f",
            s.name,
            Ri_b,
            Cd,
            Ch,
        )
    return Cd, Ch


def Cd_Ch_iterative(
    Ri_b: float,
    z: float,
    z0: float,
    zt: float,
    scheme_name: str,
    verbose: bool = False,
    **kwargs,
) -> tuple[float, float]:
    """
    Bulk transfer coefficients (C_d, C_h) using iterative ζ solution.
    """
    if scheme_name == "ECHAM6":
        return Cd_Ch_ECHAM6(Ri_b, z, z0, zt, verbose=verbose)

    s = _scheme_for_ri(Ri_b, scheme_name)
    eps_m, eps_t = epsilons(z, z0, zt)

    ln_em = math.log(eps_m)
    ln_et = math.log(eps_t)

    C_dn = KAPPA ** 2 / (ln_em ** 2)
    C_hn = KAPPA ** 2 / (s.Pr0 * ln_em * ln_et)

    fm, fh = fm_fh_iterative(Ri_b, z, z0, zt, scheme_name, verbose=verbose, **kwargs)
    Cd, Ch = C_dn * fm, C_hn * fh
    if verbose:
        logger.info(
            "Cd/Ch iterative: scheme=%s Ri=%.4f Cd=%.6f Ch=%.6f",
            s.name,
            Ri_b,
            Cd,
            Ch,
        )
    return Cd, Ch

r"""
fluxpy.stability_lib
====================
Stability functions and transfer coefficients (manuscript Appendix B3).

Implementation of Gryanik and Lüpkes (2023, Boundary-Layer Meteorol. 187,
41-72, doi:10.1007/s10546-022-00730-9): "A Package of Momentum and Heat
Transfer Coefficients for the Stable Surface Layer Extended by New Coefficients
over Sea Ice". Equation numbers below refer to that paper.

Given the bulk Richardson number ``Ri_b`` (Appendix B1, ``fluxpy.stability``)
and the roughness ratios ``eps_m = z / z0m`` and ``eps_t = z / z0T``:

    C_d = C_dn * f_m,   C_h = C_hn * f_h                             (Eq. 6-8)
    C_dn = kappa^2 / ln(eps_m)^2,  C_hn = kappa^2 / (Pr0 ln(eps_m) ln(eps_t))

* **Stable** (Ri_b > 0): the stability parameter ``zeta = z/L`` follows from the
  non-iterative universal formula ``zeta(Ri_b, eps_m, eps_t)`` (Eq. 22a-c) and
  ``f_m, f_h`` from the scheme's ``psi_m, psi_h`` (Eq. 7-8). This is the
  "Louis-type" evaluation of the manuscript: no iteration over the Obukhov length.
* **Unstable** (Ri_b < 0): the Businger-Dyer functions (Businger et al., 1971;
  Dyer, 1974); ``zeta`` is found by Newton iteration of the exact MOST relation
  (Eq. 20).
* **Neutral** (Ri_b = 0): ``f_m = f_h = 1``.

Schemes (Table 1 of Gryanik and Lüpkes, 2023; the eight compared in the paper):
BD (Businger-Dyer), HB88, BH91, CB05, GAFGP07 (SHEBA), GLGS20 (the reference
configuration), ZE07-I, ZE07-II. ``psi_h`` uses the multiplicative convention
(Pr0 absorbed), as in the paper. Constants: kappa = 0.4 (``fluxpy.util``).

Entry point used by the bulk flux: :func:`Cd_Ch_noniterative`.
:func:`Cd_Ch_iterative` solves Eq. (20) for the stable branch as well and
serves as the exact reference in ``tests/test_most_schemes.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict

from .util import KAPPA

__all__ = ["MOSTScheme", "SCHEMES", "UNSTABLE_SCHEME",
           "zeta_noniterative", "zeta_iterative", "fm_fh_from_zeta",
           "fm_fh_noniterative", "fm_fh_iterative",
           "Cd_Ch_noniterative", "Cd_Ch_iterative"]


# ---------------------------------------------------------------------
# A stability scheme = its integrated stability functions psi_m, psi_h
# ---------------------------------------------------------------------

PsiFunc = Callable[[float], float]


@dataclass
class MOSTScheme:
    name: str
    psi_m: PsiFunc
    psi_h: PsiFunc
    Pr0: float                   #: neutral turbulent Prandtl number
    gamma: float                 #: Table 1, used in the non-iterative formula (Eq. 22a)
    zeta_a: float                #: Table 1, used in the non-iterative formula (Eq. 22a)
    zeta_max: float | None = None   #: nominal validity range of the scheme (documentation)


SCHEMES: Dict[str, MOSTScheme] = {}


def _make_bd() -> MOSTScheme:
    """Businger-Dyer, stable branch psi = -5 zeta (Eq. 11)."""
    am = ah = 5.0
    Pr0 = 1.0
    return MOSTScheme("BD", lambda z: -am * z, lambda z: -Pr0 * ah * z, Pr0,
                      gamma=4.42, zeta_a=2.5, zeta_max=1.0)


def _make_bd_unstable() -> MOSTScheme:
    """Businger-Dyer with the Paulson (1970) unstable forms; used for every scheme when Ri_b < 0."""
    Pr0 = 1.0

    def psi_m(zeta: float) -> float:
        if zeta >= 0.0:
            return -5.0 * zeta
        x = (1.0 - 16.0 * zeta) ** 0.25
        return (2.0 * math.log((1.0 + x) / 2.0) + math.log((1.0 + x * x) / 2.0)
                - 2.0 * math.atan(x) + math.pi / 2.0)

    def psi_h(zeta: float) -> float:
        if zeta >= 0.0:
            return -Pr0 * 5.0 * zeta
        x = (1.0 - 16.0 * zeta) ** 0.25
        return 2.0 * math.log((1.0 + x * x) / 2.0)

    return MOSTScheme("BD-unstable", psi_m, psi_h, Pr0, gamma=4.42, zeta_a=2.5)


def _make_hb88() -> MOSTScheme:
    """Holtslag and De Bruin (1988), Eq. (12); the same function for momentum and heat."""
    a, b, c, d = 0.7, 0.75, 5.0, 0.35
    Pr0 = 1.0

    def psi(zeta: float) -> float:
        return -a * zeta - b * (zeta - c / d) * math.exp(-d * zeta) - (b * c / d)

    return MOSTScheme("HB88", psi, psi, Pr0, gamma=2.14, zeta_a=4.0, zeta_max=10.0)


def _make_bh91() -> MOSTScheme:
    """Beljaars and Holtslag (1991): psi_m Eq. (12), psi_h Eq. (13), a=1, b=0.667, c=5, d=0.35."""
    a, b, c, d = 1.0, 0.667, 5.0, 0.35
    Pr0 = 1.0

    def psi_m(zeta: float) -> float:
        return -a * zeta - b * (zeta - c / d) * math.exp(-d * zeta) - (b * c / d)

    def psi_h(zeta: float) -> float:
        term1 = 1.0 - (1.0 + (2.0 * a / 3.0) * zeta) ** 1.5
        term2 = -b * (zeta - c / d) * math.exp(-d * zeta) - (b * c / d)
        return term1 + term2

    return MOSTScheme("BH91", psi_m, psi_h, Pr0, gamma=1.71, zeta_a=4.8, zeta_max=10.0)


def _make_cb05() -> MOSTScheme:
    """Cheng and Brutsaert (2005), Eq. (14)."""
    a_m, b_m = 6.1, 2.5
    a_h, b_h = 5.3, 1.1
    Pr0 = 1.0

    def psi_m(zeta: float) -> float:
        return -a_m * math.log(zeta + (1.0 + zeta ** b_m) ** (1.0 / b_m))

    def psi_h(zeta: float) -> float:
        return -a_h * math.log(zeta + (1.0 + zeta ** b_h) ** (1.0 / b_h))

    return MOSTScheme("CB05", psi_m, psi_h, Pr0, gamma=2.28, zeta_a=4.5, zeta_max=5.0)


def _make_gafgp07() -> MOSTScheme:
    """Grachev et al. (2007, SHEBA), stable branch.

    phi_m = 1 + a_m z (1+z)^(1/3) / (1 + b_m z),  a_m = 5, b_m = a_m / 6.5
    phi_h = 1 + (a_h z + b_h z^2) / (1 + c_h z + z^2),  a_h = b_h = 5, c_h = 3

    psi_m, psi_h are the closed-form integrals of (1 - phi) / z; they are checked
    against numerical quadrature of phi in tests/test_most_schemes.py.
    """
    am = 5.0
    bm = am / 6.5
    ah, bh, ch = 5.0, 5.0, 3.0
    Pr0 = 1.0

    Bm = ((1.0 - bm) / bm) ** (1.0 / 3.0)
    Bh = math.sqrt(ch ** 2 - 4.0)
    sqrt3 = math.sqrt(3.0)

    def psi_m(zeta: float) -> float:
        if zeta < 0.0:
            return 0.0
        x = (1.0 + zeta) ** (1.0 / 3.0)
        log1 = math.log((x + Bm) / (1.0 + Bm))
        log2 = math.log((x ** 2 - x * Bm + Bm ** 2) / (1.0 - Bm + Bm ** 2))
        atan_part = (math.atan((2.0 * x - Bm) / (sqrt3 * Bm))
                     - math.atan((2.0 - Bm) / (sqrt3 * Bm)))
        return (-3.0 * am / bm * (x - 1.0)
                + am * Bm / (2.0 * bm) * (2.0 * log1 - log2 + 2.0 * sqrt3 * atan_part))

    def psi_h(zeta: float) -> float:
        if zeta < 0.0:
            return 0.0
        term_log = (math.log((2.0 * zeta + ch - Bh) / (2.0 * zeta + ch + Bh))
                    - math.log((ch - Bh) / (ch + Bh)))
        first = (-ah / Bh + bh * ch / (2.0 * Bh)) * term_log
        second = -0.5 * bh * math.log(1.0 + ch * zeta + zeta ** 2)
        return Pr0 * (first + second)

    return MOSTScheme("GAFGP07", psi_m, psi_h, Pr0, gamma=2.91, zeta_a=3.6, zeta_max=100.0)


def _make_glgs20() -> MOSTScheme:
    """Gryanik et al. (2020), Eq. (19a, b): the reference scheme of the manuscript."""
    am, ah = 5.0, 5.0
    bm, bh = 0.3, 0.4
    Pr0 = 0.98

    def psi_m(zeta: float) -> float:
        return -3.0 * am / bm * ((1.0 + bm * zeta) ** (1.0 / 3.0) - 1.0)

    def psi_h(zeta: float) -> float:
        return -Pr0 * (ah / bh) * math.log(1.0 + bh * zeta)

    return MOSTScheme("GLGS20", psi_m, psi_h, Pr0, gamma=3.66, zeta_a=10.2, zeta_max=100.0)


def _make_ze07_I() -> MOSTScheme:
    """Zilitinkevich and Esau (2007), form I, Eq. (16)."""
    Pr0, am, ah, bh = 0.85, 5.0, 4.0, 1.25
    return MOSTScheme("ZE07-I", lambda z: -am * z, lambda z: -Pr0 * (ah * z + 0.5 * bh * z ** 2),
                      Pr0, gamma=1.80, zeta_a=10.0, zeta_max=1.0)


def _make_ze07_II() -> MOSTScheme:
    """Zilitinkevich and Esau (2007), form II, Eq. (17)."""
    Pr0, cm, ch = 0.85, 6.44, 5.2
    return MOSTScheme("ZE07-II", lambda z: -cm * z ** (5.0 / 6.0), lambda z: -Pr0 * ch * z ** (4.0 / 5.0),
                      Pr0, gamma=4.64, zeta_a=14.6, zeta_max=125.0)


for _scheme in (_make_bd(), _make_hb88(), _make_bh91(), _make_cb05(),
                _make_gafgp07(), _make_glgs20(), _make_ze07_I(), _make_ze07_II()):
    SCHEMES[_scheme.name] = _scheme
UNSTABLE_SCHEME = _make_bd_unstable()


def _scheme_for(stability: float, scheme_name: str) -> MOSTScheme:
    """The named scheme for stable stratification, Businger-Dyer for unstable."""
    if scheme_name not in SCHEMES:
        raise KeyError(f"Unknown scheme '{scheme_name}'. Available: {sorted(SCHEMES)}")
    return UNSTABLE_SCHEME if stability < 0.0 else SCHEMES[scheme_name]


def _epsilons(z: float, z0: float, zt: float) -> tuple[float, float]:
    if z <= 0 or z0 <= 0 or zt <= 0:
        raise ValueError("z, z0, zt must be positive.")
    return z / z0, z / zt


# ---------------------------------------------------------------------
# zeta from Ri_b
# ---------------------------------------------------------------------

def zeta_noniterative(Ri_b: float, eps_m: float, eps_t: float, scheme_name: str) -> float:
    """Stability parameter zeta = z/L from Ri_b without iteration, Eq. (22a-c); stable branch only."""
    if Ri_b <= 0.0:
        return 0.0
    if scheme_name not in SCHEMES:
        raise KeyError(f"Unknown scheme '{scheme_name}'")
    s = SCHEMES[scheme_name]
    ln_em, ln_et = math.log(eps_m), math.log(eps_t)

    # equivalent bulk Richardson number, Eq. (22c)
    hat_Ri = (Ri_b / s.Pr0) * ((1.0 - 1.0 / eps_t) / ((1.0 - 1.0 / eps_m) ** 2))
    # psi_ka, Eq. (22b); the paper uses eps_t for both k = m, h
    psi_ma = s.psi_m(s.zeta_a) - s.psi_m(s.zeta_a / eps_t)
    psi_ha = s.psi_h(s.zeta_a) - s.psi_h(s.zeta_a / eps_t)

    term1 = (ln_em ** 2 / ln_et) * hat_Ri                                 # linear term
    num = (ln_em - psi_ma) ** (2.0 * (s.gamma - 1.0))
    den = (s.zeta_a ** (s.gamma - 1.0)) * ((ln_et - psi_ha / s.Pr0) ** (s.gamma - 1.0))
    if den == 0.0:
        return _clip_stable(term1)
    bracket = ((ln_em - psi_ma) ** 2) / (ln_et - psi_ha / s.Pr0) - (ln_em ** 2) / ln_et
    term2 = (num / den) * bracket * (hat_Ri ** s.gamma)                  # nonlinear term
    return _clip_stable(term1 + term2)


def _clip_stable(zeta: float) -> float:
    return max(zeta, 0.0) if math.isfinite(zeta) else 0.0


def zeta_iterative(Ri_b: float, eps_m: float, eps_t: float, scheme_name: str,
                   max_iter: int = 80, tol: float = 1e-6) -> float:
    """Stability parameter zeta from Ri_b by Newton iteration of the exact relation, Eq. (20).

    Used for the unstable branch (Businger-Dyer psi functions) and as the exact
    reference for the non-iterative stable formula. zeta is kept within
    [-20, 50] (unstable) or >= 0 (stable).
    """
    if Ri_b == 0.0:
        return 0.0
    s = _scheme_for(Ri_b, scheme_name)
    ln_em, ln_et = math.log(eps_m), math.log(eps_t)

    if Ri_b > 0.0:
        zeta = zeta_noniterative(Ri_b, eps_m, eps_t, scheme_name) or 0.05
        clip = _clip_stable
    else:
        zeta = max(-0.5, -0.1 * abs(Ri_b))
        clip = lambda z: min(max(z, -20.0), 50.0) if math.isfinite(z) else 0.0   # noqa: E731

    pref = ((1.0 - 1.0 / eps_m) ** 2) / (1.0 - 1.0 / eps_t)

    def residual(z: float) -> float | None:
        num = pref * z * (s.Pr0 * ln_et - s.psi_h(z) + s.psi_h(z / eps_t))
        den = (ln_em - s.psi_m(z) + s.psi_m(z / eps_m)) ** 2
        return None if den == 0.0 else num / den - Ri_b

    for _ in range(max_iter):
        f = residual(zeta)
        if f is None or abs(f) < tol:
            break
        dz = 1e-4 * max(1.0, zeta)
        f2 = residual(zeta + dz)
        if f2 is None:
            break
        df = (f2 - f) / dz
        if df == 0.0 or not math.isfinite(df):
            break
        zeta_new = clip(zeta - f / df)
        if abs(zeta_new - zeta) < tol:
            zeta = zeta_new
            break
        zeta = zeta_new
    return clip(zeta)


# ---------------------------------------------------------------------
# f_m, f_h and the transfer coefficients
# ---------------------------------------------------------------------

def fm_fh_from_zeta(zeta: float, eps_m: float, eps_t: float, scheme_name: str) -> tuple[float, float]:
    """Normalised transfer coefficients f_m, f_h from zeta, Eq. (7)-(8)."""
    s = _scheme_for(zeta, scheme_name)
    ln_em, ln_et = math.log(eps_m), math.log(eps_t)
    A = 1.0 - (s.psi_m(zeta) - s.psi_m(zeta / eps_m)) / ln_em
    B = 1.0 - (s.psi_h(zeta) - s.psi_h(zeta / eps_t)) / (s.Pr0 * ln_et)
    return 1.0 / (A * A), 1.0 / (A * B)


def fm_fh_noniterative(Ri_b: float, z: float, z0: float, zt: float, scheme_name: str) -> tuple[float, float]:
    """f_m, f_h for the bulk flux: non-iterative when stable, Businger-Dyer iteration when unstable."""
    eps_m, eps_t = _epsilons(z, z0, zt)
    if Ri_b < 0.0:
        zeta = zeta_iterative(Ri_b, eps_m, eps_t, scheme_name)
    elif Ri_b == 0.0:
        return 1.0, 1.0
    else:
        zeta = zeta_noniterative(Ri_b, eps_m, eps_t, scheme_name)
    return fm_fh_from_zeta(zeta, eps_m, eps_t, scheme_name)


def fm_fh_iterative(Ri_b: float, z: float, z0: float, zt: float, scheme_name: str) -> tuple[float, float]:
    """f_m, f_h with the iterative zeta solution on both branches (reference solution)."""
    eps_m, eps_t = _epsilons(z, z0, zt)
    zeta = zeta_iterative(Ri_b, eps_m, eps_t, scheme_name)
    return fm_fh_from_zeta(zeta, eps_m, eps_t, scheme_name)


def _neutral_coefficients(z: float, z0: float, zt: float, Pr0: float) -> tuple[float, float]:
    """C_dn, C_hn of the manuscript (Appendix B3): kappa^2 / ln^2, kappa^2 / (Pr0 ln ln)."""
    eps_m, eps_t = _epsilons(z, z0, zt)
    ln_em, ln_et = math.log(eps_m), math.log(eps_t)
    return KAPPA ** 2 / (ln_em ** 2), KAPPA ** 2 / (Pr0 * ln_em * ln_et)


def Cd_Ch_noniterative(Ri_b: float, z: float, z0: float, zt: float, scheme_name: str) -> tuple[float, float]:
    """Transfer coefficients (C_d, C_h) at height ``z`` for roughness lengths ``z0``, ``zt``
    and bulk Richardson number ``Ri_b``: C = C_n * f (manuscript Appendix B3).
    This is the function the bulk flux calls."""
    s = _scheme_for(Ri_b, scheme_name)
    C_dn, C_hn = _neutral_coefficients(z, z0, zt, s.Pr0)
    fm, fh = fm_fh_noniterative(Ri_b, z, z0, zt, scheme_name)
    return C_dn * fm, C_hn * fh


def Cd_Ch_iterative(Ri_b: float, z: float, z0: float, zt: float, scheme_name: str) -> tuple[float, float]:
    """(C_d, C_h) with the iterative zeta solution on both branches (test reference)."""
    s = _scheme_for(Ri_b, scheme_name)
    C_dn, C_hn = _neutral_coefficients(z, z0, zt, s.Pr0)
    fm, fh = fm_fh_iterative(Ri_b, z, z0, zt, scheme_name)
    return C_dn * fm, C_hn * fh

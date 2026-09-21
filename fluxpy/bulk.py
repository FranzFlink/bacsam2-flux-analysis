r"""
fluxpy.bulk
===========
Bulk sensible heat flux from Monin–Obukhov similarity theory,

$$ H_\mathrm{S} = \rho\, c_p\, C_h\, U\,
   (\theta_\mathrm{s} - \theta_\mathrm{air})
   \quad [\mathrm{W\,m^{-2}};\ \text{positive upward}], $$

where the heat transfer coefficient combines a neutral part and a
stability correction,

$$ C_h = \frac{\kappa^2}
         {\mathrm{Pr}_{t,0}\, \ln(z/z_{0,m})\, \ln(z/z_{0,T})}\;
         f_h\!\left(Ri_b, \tfrac{z}{z_{0,m}}, \tfrac{z}{z_{0,T}}\right). $$

The stability functions $f_h$ are Louis-type, i.e. formulated directly in
the bulk Richardson number so that no iteration over the Obukhov length is
required (Louis, 1979, doi:10.1007/BF00117978). The reference configuration
uses the SHEBA-based functions of Gryanik et al. (2020,
doi:10.1175/JAS-D-19-0255.1) as implemented by Gryanik and Lüpkes (2023,
doi:10.1007/s10546-022-00730-9), with Businger–Dyer functions in unstable
conditions. The scalar roughness length is tied to the momentum roughness
through the ratio $\alpha_T = z_{0,T}/z_{0,m}$.

Two entry points:

    ``compute_bulk_shf_obs``
        In-situ pathway: T-Bird/P6 meteorology and KT-19 skin temperature.
        Pass ``ds_tbird`` to evaluate the Richardson number at the T-Bird
        level.

    ``compute_bulk_shf_remote``
        Remote-sensing pathway: reanalysis column and satellite (or
        reanalysis) skin temperature.

Sign convention
---------------
$\theta_\mathrm{s} > \theta_\mathrm{air}$ — surface warmer than the air —
gives $H_\mathrm{S} > 0$ (upward, e.g. over a lead); the reverse gives
$H_\mathrm{S} < 0$ (downward, e.g. over cold ice).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import xarray as xr

from .util import (
    CP, RD, G, P0,
    potential_temperature,
    air_density,
    classify_surface,
    assign_roughness,
)
from .stability import (
    SCHEMES,
    bulk_richardson_2pt,      # classic 2-pt Ri_b (manuscript Eq. rib) — the only Ri in the code base
    effective_wind_speed,     # Jordan (1999) / Fairall (1996) calm-limit wind (Zhang et al. 2025)
    Cd_Ch_noniterative,
    fm_fh_iterative,
    zt_open_water,
    zt_sea_ice,
    kinematic_viscosity,
)

log = logging.getLogger(__name__)

__all__ = ["BulkConfig", "reference_config", "roughness_from_fractions", "compute_bulk_shf_obs", "compute_bulk_shf_remote", "solve_coupled_bulk"]


# ── configuration ─────────────────────────────────────────────────────────────
@dataclass
class BulkConfig:
    """All free parameters for one bulk flux run."""
    scheme_name:    str   = "GLGS20"
    zt_ratio:       float = 0.1     # zt = z0 * zt_ratio
    z0_ocean:       float = 2.0e-4  # m
    z0_thin_ice:    float = 1.0e-4
    z0_thick_ice:   float = 4.0e-4
    z0_constant:    Optional[float] = None   # m; when set, overrides fractional/T-class (sensitivity sweeps)
    T_thresh_open:  float = -1.8    # °C  ocean if IST > threshold
    T_thresh_thick: float = -20.0   # °C  thick ice if IST <= threshold
    ri_eps_wind2:   float = 1e-4    # m² s⁻²; shear regularisation ε in Ri_b (prevents 1/U² blow-up in near-calm conditions)
    ri_clip:        float = np.inf  # |Ri_b| analysis bound, applied to every pathway AFTER the wind treatment
                                    # (np.inf disables). Bounded stability ranges are standard practice: the
                                    # revised WRF surface layer floors u*/U to keep Ri_B from becoming
                                    # "inordinately high" (Jiménez et al. 2012, doi:10.1175/MWR-D-11-00056.1),
                                    # CLM caps the stability parameter (ζ ≤ 2 in CLM4.5, ≤ 0.5 in CLM5;
                                    # Lawrence et al. 2019, doi:10.1029/2018MS001583), COARE saturates ζ(Ri_b)
                                    # smoothly (Grachev and Fairall 1997), and our stability_lib clips ζ to
                                    # [-20, 50] so C_h is saturated beyond |Ri_b| ~ 1 anyway. Sweep
                                    # (scripts/ri_clip_experiments.py): flux skill changes < 0.03 W m⁻² for
                                    # any bound between 1 and ∞.
    ri_wind_remote: str   = "effective"  # Ri_b wind for the remote (no-T-Bird) pathway: "effective" =
                                    # Jordan (1999) windless exchange (stable) + Fairall (1996) convective
                                    # gustiness (unstable) — bounds Ri_b NATURALLY, no ε and no clip applied
                                    # (the approach of Zhang et al. 2025 for ERA5-driven fluxes over sea ice);
                                    # "raw" = reanalysis wind as given, with ri_eps_wind2 + ri_clip.
    zt_model:       str   = "constant"  # "constant", "andreas", "fairall", "dynamic"
    zt_factor:      float = 1.0     # scalar factor to multiply dynamic zt with (best engineering approach)
    use_gustiness:  bool  = False
    label:          str   = "default"

    def __post_init__(self):
        if self.scheme_name not in SCHEMES:
            raise ValueError(f"Unknown scheme '{self.scheme_name}'. Available: {sorted(SCHEMES)}")
        if not (0 < self.zt_ratio <= 1):
            raise ValueError("zt_ratio must be in (0, 1]")
        if self.T_thresh_thick >= self.T_thresh_open:
            raise ValueError("T_thresh_thick must be < T_thresh_open")
        if self.z0_constant is not None and self.z0_constant <= 0:
            raise ValueError("z0_constant must be > 0")
        if self.zt_factor <= 0:
            raise ValueError("zt_factor must be > 0")
        if self.ri_eps_wind2 < 0:
            raise ValueError("ri_eps_wind2 must be >= 0")
        if self.ri_clip <= 0:
            raise ValueError("ri_clip must be > 0 (use np.inf to disable clipping)")
        if self.ri_wind_remote not in ("effective", "raw"):
            raise ValueError(f"Unknown ri_wind_remote '{self.ri_wind_remote}'. Available: ('effective', 'raw')")
        if self.zt_model not in ("constant", "andreas", "fairall", "dynamic"):
            raise ValueError(f"Unknown zt_model '{self.zt_model}'. Available: ('constant', 'andreas', 'fairall', 'dynamic')")


def reference_config() -> BulkConfig:
    """Reference run: GLGS20, zt_ratio=1e-1 -- the CONVENTIONAL scalar roughness
    ratio used by models and LES over sea ice (author decision 2026-07-27;
    supersedes the tuned 1e-5, which was an artefact of minimising RMSE in a
    low-flux regime: RMSE rewards shrinking the amplitude and alpha_T is the
    amplitude knob). Amplitude skill is now reported as a regression slope;
    see amplitude_matched_config() for the slope-matched counterpart.
    Classic 2-pt Ri_b (surface → T-Bird level, manuscript Eq. rib)."""
    return BulkConfig(scheme_name="GLGS20", zt_ratio=1e-1, label="GLGS20_ref")


def amplitude_matched_config() -> BulkConfig:
    """Amplitude-matched run: GLGS20, zt_ratio=1e-4 -- the value at which the
    in-situ bulk estimate reproduces the observed flux amplitude (regression
    slope of unity), coinciding with the airborne EC estimate of Schroeder et
    al. (2003). Reported ALONGSIDE the conventional reference, never as a
    roughness retrieval: alpha_T enters C_h only logarithmically and leaves the
    correlation unchanged."""
    return BulkConfig(scheme_name="GLGS20", zt_ratio=1e-4, label="GLGS20_ampmatch")


# ── roughness from fractional surface type ────────────────────────────────────
def roughness_from_fractions(
    f_open:  np.ndarray,
    f_thin:  np.ndarray,
    f_thick: np.ndarray,
    cfg:     BulkConfig,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Effective roughness lengths from the surface-type fractions of each
    segment, blended log-linearly (a fraction-weighted geometric mean),

    $$ \ln z_{0,m}^\mathrm{eff} = \sum_i f_i \ln z_{0,m,i},
       \qquad \sum_i f_i = 1, $$

    over the classes open water, thin ice, and snow-covered thick ice.
    Logarithmic averaging is the appropriate blend because the roughness
    enters the transfer coefficient through $\ln(z/z_0)$; averaging
    $z_0$ itself would overweight the roughest class. The scalar roughness
    follows as $z_{0,T}^\mathrm{eff} = \alpha_T\, z_{0,m}^\mathrm{eff}$.
    Fractions are clipped to $[0, 1]$ and normalised per segment; returns
    ``(z0_eff, zt_eff)`` arrays.
    """
    f = np.vstack([f_open, f_thin, f_thick]).T.astype(float)
    f = np.where(np.isfinite(f), f, 0.0)
    f = np.clip(f, 0.0, 1.0)

    s = f.sum(axis=1)
    valid = s > 0
    f_norm = np.zeros_like(f)
    f_norm[valid] = f[valid] / s[valid, None]

    z0_vals   = np.array([cfg.z0_ocean, cfg.z0_thin_ice, cfg.z0_thick_ice], dtype=float)
    ln_z0_eff = np.full(len(f_open), np.nan)
    ln_z0_eff[valid] = f_norm[valid] @ np.log(z0_vals)

    z0_eff = np.exp(ln_z0_eff)
    zt_eff = z0_eff * cfg.zt_ratio
    return z0_eff, zt_eff


def _roughness_from_ds(
    ds:  xr.Dataset,
    cfg: BulkConfig,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Read ``frac_open``, ``frac_thin_ice``, ``frac_thick_ice`` from *ds* and
    return weighted (z0, zt).  Returns ``(None, None)`` if any variable is absent.
    """
    req = ("frac_open", "frac_thin_ice", "frac_thick_ice")
    if not all(v in ds for v in req):
        return None, None
    return roughness_from_fractions(
        np.asarray(ds["frac_open"],      dtype=float),
        np.asarray(ds["frac_thin_ice"],  dtype=float),
        np.asarray(ds["frac_thick_ice"], dtype=float),
        cfg,
    )


def solve_coupled_bulk(
    theta_sfc: np.ndarray,
    theta_air: np.ndarray,
    U_u: np.ndarray,
    U_v: np.ndarray,
    z_arr: np.ndarray,
    cfg: BulkConfig,
    f_open: Optional[np.ndarray] = None,
    f_thin: Optional[np.ndarray] = None,
    f_thick: Optional[np.ndarray] = None,
    surface_type: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Coupled iterative solver for the bulk-flux system.

    When the scalar roughness is made a function of the flow itself —
    $z_{0,T} = f(Re_*)$ with the roughness Reynolds number
    $Re_* = u_* z_{0,m}/\nu$ (Andreas, 1987, doi:10.1007/BF00121562;
    Andreas et al., 2010, doi:10.1175/2009JHM1102.1) — or when a
    convective gustiness contribution is added to the wind speed, the
    system becomes implicit and is solved by damped fixed-point iteration
    over: effective wind speed, roughness lengths, Richardson number,
    transfer coefficients $(C_d, C_h)$, and friction velocity. Ten
    iterations are sufficient for convergence at the accuracy of the
    input data. Used only in the corresponding sensitivity experiments;
    the reference configuration is non-iterative by construction.

    Returns
    -------
    Cd, Ch, z0, zt, Ri, U_eff : ndarrays per segment
    """
    n = len(theta_sfc)
    Cd = np.full(n, np.nan)
    Ch = np.full(n, np.nan)
    z0 = np.full(n, np.nan)
    zt = np.full(n, np.nan)
    Ri = np.full(n, np.nan)
    U_eff_arr = np.full(n, np.nan)

    # Pre-parse surface fractions
    if f_open is None or f_thin is None or f_thick is None:
        if surface_type is not None:
            f_open = np.where(surface_type == "ocean", 1.0, 0.0)
            f_thick = np.where(surface_type == "thick_snow_ice", 1.0, 0.0)
            f_thin = np.where((surface_type != "ocean") & (surface_type != "thick_snow_ice"), 1.0, 0.0)
        else:
            f_open = np.zeros(n)
            f_thin = np.zeros(n)
            f_thick = np.zeros(n)

    # Normalize fractions
    f = np.vstack([f_open, f_thin, f_thick]).T.astype(float)
    f = np.where(np.isfinite(f), f, 0.0)
    f = np.clip(f, 0.0, 1.0)
    s = f.sum(axis=1)
    f_norm = np.zeros_like(f)
    valid_frac = s > 0
    f_norm[valid_frac] = f[valid_frac] / s[valid_frac, None]

    # Precompute constant momentum roughness z0 if z0_constant is not set
    z0_blended = np.full(n, np.nan)
    if cfg.z0_constant is not None:
        z0_blended = np.full(n, cfg.z0_constant)
    else:
        z0_vals = np.array([cfg.z0_ocean, cfg.z0_thin_ice, cfg.z0_thick_ice], dtype=float)
        ln_z0_eff = np.full(n, np.nan)
        ln_z0_eff[valid_frac] = f_norm[valid_frac] @ np.log(z0_vals)
        z0_blended = np.exp(ln_z0_eff)

    for i in range(n):
        theta_sfc_i = theta_sfc[i]
        theta_air_i = theta_air[i]
        U_u_i = U_u[i]
        U_v_i = U_v[i]
        zi = z_arr[i]
        z0i = z0_blended[i]

        if not (np.isfinite(theta_sfc_i) and np.isfinite(theta_air_i) and 
                np.isfinite(U_u_i) and np.isfinite(U_v_i) and np.isfinite(zi) and 
                np.isfinite(z0i) and z0i > 0 and zi > z0i):
            continue

        U_spd_val = float(np.sqrt(U_u_i ** 2 + U_v_i ** 2))
        dT_val = theta_air_i - theta_sfc_i
        T_sfc_C = theta_sfc_i - 273.15

        # Initialize iteratives
        U_eff_val = max(U_spd_val, 0.001)
        Cd_val = 1.5e-3
        Ch_val = 1.5e-3
        u_star_val = max(1e-4, np.sqrt(Cd_val) * U_eff_val)
        zt_val = z0i * cfg.zt_ratio

        # Iterative Loop (10 steps for excellent convergence)
        for _ in range(10):
            # 1. Update Scalar Roughness length (zt)
            if cfg.zt_model == "constant":
                zt_val = z0i * cfg.zt_ratio
            else:
                if cfg.zt_model in ("fairall", "dynamic"):
                    zt_ocean_i = zt_open_water(u_star_val, cfg.z0_ocean, T_sfc_C)
                elif cfg.zt_model == "andreas":
                    zt_ocean_i = zt_sea_ice(u_star_val, cfg.z0_ocean, T_sfc_C)
                else:
                    zt_ocean_i = cfg.z0_ocean * cfg.zt_ratio

                if cfg.zt_model in ("andreas", "dynamic"):
                    zt_thin_i = zt_sea_ice(u_star_val, cfg.z0_thin_ice, T_sfc_C)
                elif cfg.zt_model == "fairall":
                    zt_thin_i = zt_open_water(u_star_val, cfg.z0_thin_ice, T_sfc_C)
                else:
                    zt_thin_i = cfg.z0_thin_ice * cfg.zt_ratio

                if cfg.zt_model in ("andreas", "dynamic"):
                    zt_thick_i = zt_sea_ice(u_star_val, cfg.z0_thick_ice, T_sfc_C)
                elif cfg.zt_model == "fairall":
                    zt_thick_i = zt_open_water(u_star_val, cfg.z0_thick_ice, T_sfc_C)
                else:
                    zt_thick_i = cfg.z0_thick_ice * cfg.zt_ratio

                if cfg.z0_constant is not None:
                    if cfg.zt_model in ("andreas", "dynamic"):
                        zt_val = zt_sea_ice(u_star_val, cfg.z0_constant, T_sfc_C)
                    elif cfg.zt_model == "fairall":
                        zt_val = zt_open_water(u_star_val, cfg.z0_constant, T_sfc_C)
                else:
                    ln_zt_eff = (f_norm[i, 0] * np.log(zt_ocean_i) + 
                                 f_norm[i, 1] * np.log(zt_thin_i) + 
                                 f_norm[i, 2] * np.log(zt_thick_i))
                    zt_val = np.exp(ln_zt_eff)

                zt_val *= cfg.zt_factor

            zt_val = max(1e-15, zt_val)

            if zi <= zt_val:
                zt_val = z0i * cfg.zt_ratio

            # 2. Update Effective Wind Speed with Gustiness
            if cfg.use_gustiness:
                if dT_val > 0.0:  # Stable
                    U_eff_val = U_spd_val + 0.5 / np.cosh(U_spd_val)
                else:  # Unstable
                    W_star = max(0.0, - (9.80665 / theta_air_i) * Ch_val * U_eff_val * dT_val * 600.0) ** (1.0 / 3.0)
                    U_g = 1.25 * W_star
                    U_eff_val = np.sqrt(U_spd_val ** 2 + U_g ** 2)
            else:
                U_eff_val = max(U_spd_val, 0.001)

            # 3. Richardson number update — classic 2-pt Ri_b (manuscript Eq. rib)
            # at the evaluation height, with the gustiness-modified wind speed.
            Ri_val = float(bulk_richardson_2pt(
                dtheta=dT_val,
                dz=zi,
                theta_ref=0.5 * (theta_sfc_i + theta_air_i),
                eps_wind2=cfg.ri_eps_wind2,
                wspd=U_eff_val,
            ))

            if not np.isfinite(Ri_val):
                break

            Ri_val = np.clip(Ri_val, -cfg.ri_clip, cfg.ri_clip)

            # 4. Recompute drag & heat transfer coefficients
            try:
                Cd_new, Ch_new = Cd_Ch_noniterative(Ri_val, zi, z0i, zt_val, cfg.scheme_name)
            except Exception:
                break

            if not (np.isfinite(Cd_new) and np.isfinite(Ch_new)):
                break

            Cd_val = Cd_new
            Ch_val = Ch_new

            # 5. Recompute u_star with damping
            u_star_new = max(1e-4, np.sqrt(Cd_val) * U_eff_val)
            u_star_val = 0.5 * u_star_val + 0.5 * u_star_new

        if np.isfinite(Cd_val) and np.isfinite(Ch_val):
            Cd[i] = Cd_val
            Ch[i] = Ch_val
            z0[i] = z0i
            zt[i] = zt_val
            Ri[i] = Ri_val
            U_eff_arr[i] = U_eff_val

    return Cd, Ch, z0, zt, Ri, U_eff_arr


# ── transfer coefficients (per-sample loop) ───────────────────────────────────
def _coefs_loop(
    Ri:    np.ndarray,
    z0:    np.ndarray,
    zt:    np.ndarray,
    z:     np.ndarray,
    cfg:   BulkConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute (Cd, Ch) per sample via ``Cd_Ch_noniterative``.  NaN on invalid geometry.
    Returns (Cd, Ch) arrays.
    """
    n = len(Ri)
    Cd = np.full(n, np.nan)
    Ch = np.full(n, np.nan)
    n_skip = 0
    for i in range(n):
        zi, z0i, zti, ri = z[i], z0[i], zt[i], Ri[i]
        if not (np.isfinite(ri) and np.isfinite(zi) and zi > z0i and zi > zti
                and z0i > 0 and zti > 0):
            n_skip += 1
            continue
        try:
            Cd[i], Ch[i] = Cd_Ch_noniterative(ri, zi, z0i, zti, cfg.scheme_name)
        except Exception as exc:
            log.debug("[%s] sample %d: %s", cfg.label, i, exc)
            n_skip += 1
    if n_skip:
        log.warning("[%s] %d/%d samples skipped in transfer-coef loop", cfg.label, n_skip, n)
    return Cd, Ch


# ── shared quick sanity check ────────────────────────────────────────────────
def _sanity(H: np.ndarray, dT: np.ndarray, label: str) -> None:
    """Log sign inconsistencies and extreme values (never raises)."""
    fin = np.isfinite(H) & np.isfinite(dT) & (np.abs(dT) > 0.5)
    if np.any(fin):
        bad = fin & (np.sign(H) == np.sign(dT))   # H and dT should have opposite sign
        if np.any(bad):
            log.warning("[%s] sign(H) ≠ sign(θ_sfc−θ_air) in %d/%d samples",
                        label, int(np.sum(bad)), int(np.sum(fin)))
    large = np.abs(H) > 300
    if np.any(large):
        log.warning("[%s] |H| > 300 W/m² in %d samples", label, int(np.sum(large)))
    log.info("[%s] H  mean=%.1f  std=%.1f  range=[%.1f, %.1f] W/m²",
             label, float(np.nanmean(H)), float(np.nanstd(H)),
             float(np.nanmin(H)), float(np.nanmax(H)))


# ── helper: compute Ri with optional 3-pt method ─────────────────────────────
def _compute_ri(
    theta_sfc:  np.ndarray,  # K
    theta_air:  np.ndarray,  # K
    U_u:        np.ndarray,  # m/s  u-component
    U_v:        np.ndarray,  # m/s  v-component
    z0:         np.ndarray,
    zt:         np.ndarray,
    z_obs:       np.ndarray,
    cfg:        BulkConfig,
    ds_tbird:   Optional[xr.Dataset] = None,
) -> np.ndarray:
    r"""
    Per-sample classic two-point bulk Richardson number, manuscript Eq. (rib):

    $$ Ri_b = \frac{g/\bar\theta\,\Delta\theta\,\Delta z}
                   {\Delta u^2 + \Delta v^2 + \varepsilon\,\Delta z^2}, $$

    with layer-mean $\bar\theta$ and finite differences between the surface
    (no-slip, $u = v = 0$) and the T-Bird level when ``ds_tbird`` is given
    (in-situ pathway) or the reference level ``z_obs`` otherwise (remote
    pathway). This is the formulation of Gryanik and Lüpkes (2023) used by
    the transfer-coefficient schemes; it does NOT depend on the roughness
    lengths.

    Calm-limit treatment differs by wind source: the in-situ pathway
    (measured winds) uses the regularisation ``cfg.ri_eps_wind2``; the
    remote pathway (reanalysis winds, biased low in calm conditions) by
    default uses the EFFECTIVE wind speed of Jordan et al. (1999) /
    Fairall et al. (1996) — the approach of Zhang et al. (2025) — which
    prevents the $1/U^2$ divergence physically
    (``cfg.ri_wind_remote = "raw"`` restores the ε treatment). No analysis
    bound is applied by default (``cfg.ri_clip = inf``); the clip-sweep
    experiments showed the flux statistics are insensitive to any bound ≥ 1.
    """
    n = len(theta_sfc)

    if ds_tbird is not None:
        # Classic 2-pt Ri_b, surface → T-Bird (manuscript Eq. rib):
        # Δθ = θ_TB − θ_s, Δz = z_TB, Δu/Δv = T-Bird wind (no-slip surface).
        # All inputs are the 5 km SEGMENT MEANS (the same θ, wind and height
        # that enter the flux). Until 2026-09 this branch used the single
        # 100 Hz T-Bird sample nearest to the segment start instead, which
        # contradicted the manuscript ("all quantities are evaluated on the
        # same 5 km segments"). ``ds_tbird`` now only selects this branch.
        log.info("[%s] Classic 2-pt bulk Ri_b (surface → T-Bird level; segment means)", cfg.label)
        Ri = bulk_richardson_2pt(
            dtheta=theta_air - theta_sfc,
            dz=z_obs,
            du=U_u, dv=U_v,
            theta_ref=0.5 * (theta_sfc + theta_air),
            eps_wind2=cfg.ri_eps_wind2,
        )
    else:
        # No T-Bird (remote pathway): classic 2-pt Ri_b, surface → z_obs.
        # Reanalysis winds are biased low in calm conditions, so by default
        # the shear uses the EFFECTIVE wind speed (Jordan 1999 stable /
        # Fairall 1996 unstable, as in Zhang et al. 2025) — this bounds
        # Ri_b naturally on both branches, so no ε and no clip are applied.
        dtheta    = theta_air - theta_sfc
        theta_bar = 0.5 * (theta_sfc + theta_air)
        U_spd     = np.hypot(U_u, U_v)
        if cfg.ri_wind_remote == "effective":
            log.info("[%s] Classic 2-pt bulk Ri_b (surface → z_ref; effective wind)", cfg.label)
            S = effective_wind_speed(U_spd, dtheta, z_obs, theta_bar,
                                     scheme=cfg.scheme_name)
            Ri = bulk_richardson_2pt(
                dtheta=dtheta, dz=z_obs, theta_ref=theta_bar,
                eps_wind2=0.0, wspd=S,
            )
        else:
            log.info("[%s] Classic 2-pt bulk Ri_b (surface → z_ref; raw wind)", cfg.label)
            Ri = bulk_richardson_2pt(
                dtheta=dtheta, dz=z_obs, theta_ref=theta_bar,
                eps_wind2=cfg.ri_eps_wind2, wspd=U_spd,
            )
    return np.clip(Ri, -cfg.ri_clip, cfg.ri_clip)


# ── P6 bulk flux ──────────────────────────────────────────────────────────────
def compute_bulk_shf_obs(
    ds:        xr.Dataset,
    cfg:       BulkConfig,
    ds_tbird:  Optional[xr.Dataset] = None,
) -> xr.Dataset:
    """
    Bulk SHF using P6 in-situ state and KT-19 brightness temperature.

    Required variables in ``ds``:
        ``t_kt19``  [°C]   KT-19 surface brightness temperature
        ``T_mean``  [°C]   mean air temperature
        ``p_mean``  [hPa]  mean air pressure
        ``u_mean``  [m/s]  mean eastward wind
        ``v_mean``  [m/s]  mean northward wind
        ``altitude``[m]    flight altitude

    Parameters
    ----------
    ds_tbird : xr.Dataset, optional
        T-Bird dataset **already averaged to the P6 leg time grid** (same length as ds).
        Must contain ``T`` [°C], ``p`` [hPa|Pa], ``u`` [m/s], ``v`` [m/s], ``h`` [m].
        When supplied, $Ri_b$ is evaluated surface → T-Bird level (manuscript
        Eq. rib); otherwise surface → flight altitude.
    """
    t0 = time.perf_counter()
    log.info("[%s] compute_bulk_shf_obs  (scheme=%s  zt_ratio=%.3f)",
             cfg.label, cfg.scheme_name, cfg.zt_ratio)

    T_sfc  = np.asarray(ds["t_kt19"],   dtype=float)
    T_air  = np.asarray(ds["T_mean"],   dtype=float)
    p_hPa  = np.asarray(ds["p_mean"],   dtype=float)
    U_u    = np.asarray(ds["u_mean"],   dtype=float)
    U_v    = np.asarray(ds["v_mean"],   dtype=float)
    z_arr  = np.asarray(ds["altitude"], dtype=float)
    n      = len(T_sfc)

    theta_sfc = potential_temperature(T_sfc, p_hPa)
    theta_air = potential_temperature(T_air, p_hPa)
    dT        = theta_air - theta_sfc         # > 0 when atm. warmer (stable)
    rho       = air_density(T_air, p_hPa)

    # ── roughness: constant (sensitivity sweep) → fractional (default) → T-class fallback
    stype = classify_surface(T_sfc, cfg.T_thresh_open, cfg.T_thresh_thick)  # always for diagnostics
    if cfg.z0_constant is not None:
        z0 = np.full(n, cfg.z0_constant)
        zt = z0 * cfg.zt_ratio
        roughness_method = "constant"
        log.info("[%s] constant roughness: z0=%.2e m", cfg.label, cfg.z0_constant)
    else:
        z0, zt = _roughness_from_ds(ds, cfg)
        if z0 is not None:
            roughness_method = "fractional"
        else:
            log.info("[%s] frac_* not found in ds – falling back to T-based classification", cfg.label)
            z0, zt = assign_roughness(stype, cfg.z0_ocean, cfg.z0_thin_ice, cfg.z0_thick_ice, cfg.zt_ratio)
            roughness_method = "temperature_class"
    log.info("[%s] roughness method: %s", cfg.label, roughness_method)

    if cfg.zt_model == "constant" and not cfg.use_gustiness:
        Ri = _compute_ri(theta_sfc, theta_air, U_u, U_v, z0, zt, z_arr, cfg, ds_tbird)
        Cd, Ch = _coefs_loop(Ri, z0, zt, z_arr, cfg)
        U_eff = np.sqrt(U_u ** 2 + U_v ** 2)
    else:
        f_open = np.asarray(ds["frac_open"], dtype=float) if "frac_open" in ds else None
        f_thin = np.asarray(ds["frac_thin_ice"], dtype=float) if "frac_thin_ice" in ds else None
        f_thick = np.asarray(ds["frac_thick_ice"], dtype=float) if "frac_thick_ice" in ds else None
        Cd, Ch, z0, zt, Ri, U_eff = solve_coupled_bulk(
            theta_sfc=theta_sfc,
            theta_air=theta_air,
            U_u=U_u,
            U_v=U_v,
            z_arr=z_arr,
            cfg=cfg,
            f_open=f_open,
            f_thin=f_thin,
            f_thick=f_thick,
            surface_type=stype,
        )

    log.info("[%s] Ri  mean=%.3f  range=[%.3f, %.3f]",
             cfg.label, float(np.nanmean(Ri)), float(np.nanmin(Ri)), float(np.nanmax(Ri)))

    H = rho * CP * Ch * U_eff * (theta_sfc - theta_air)
    _sanity(H, dT, label=cfg.label + " P6")

    _v = lambda data, units, lname: xr.DataArray(
        data, dims=["start_time"],
        coords={"start_time": ds["start_time"]},
        attrs={"units": units, "long_name": lname},
    )
    out = xr.Dataset(
        {
            "H_obs_bulk": _v(H,   "W m-2",  "Bulk SHF P6; positive upward"),
            "Ri_obs":   _v(Ri,  "-",     "Richardson number (P6)"),
            "z0_obs":     _v(z0,  "m",     "Momentum roughness length"),
            "zt_obs":     _v(zt,  "m",     "Scalar roughness length"),
            "Cd_obs":     _v(Cd,  "-",     "Drag coefficient"),
            "Ch_obs":     _v(Ch,  "-",     "Heat transfer coefficient"),
            "U_eff_obs":  _v(U_eff, "m s-1", "Effective wind speed"),
            "rho_obs":    _v(rho, "kg m-3","Air density"),
            "dtheta_obs":     _v(dT,  "K",     "θ_air − θ_sfc"),
            "surface_type_obs": xr.DataArray(
                stype, dims=["start_time"],
                coords={"start_time": ds["start_time"]},
            ),
        },
        attrs={**asdict(cfg),
               "ri_method": ("2pt_surface_tbird" if ds_tbird is not None
                             else f"2pt_surface_zref_{cfg.ri_wind_remote}_wind"),
               "roughness_method": roughness_method},
    )
    log.info("[%s] P6 done in %.1f s  |  H mean=%.1f W/m²",
             cfg.label, time.perf_counter() - t0, float(np.nanmean(H)))
    return out


# ── Remote bulk flux (reanalysis sfc T + atmospheric column) ─────────────────
def compute_bulk_shf_remote(
    ds_obs:    xr.Dataset,
    ds_rs:    xr.Dataset,
    ds_modis: xr.Dataset,
    cfg:      BulkConfig,
    ds_tbird: Optional[xr.Dataset] = None,
    z_ref:    "float | np.ndarray" = 30.0,
) -> xr.Dataset:
    """
    Bulk SHF using a surface temperature field and an atmospheric column dataset.

    ``ds_rs`` variables: ``t`` [°C], ``pres`` [Pa or hPa, auto-detected], ``ws`` [m/s].
    ``ds_modis`` variable: ``IST`` [K]; values outside [150, 280] K are masked.
    ``z_ref`` : reference height [m] at which the flux is evaluated.  Can be a
        scalar (default 30 m) or a per-leg 1-D array (e.g. ERA5 geopotential height).
    """
    t0 = time.perf_counter()
    log.info("[%s] compute_bulk_shf_remote  (scheme=%s)", cfg.label, cfg.scheme_name)

    T_sfc_K = np.asarray(ds_modis["IST"], dtype=float)
    T_sfc   = np.where((T_sfc_K > 150) & (T_sfc_K < 280), T_sfc_K - 273.15, np.nan)
    n_masked = int(np.sum(~np.isfinite(T_sfc)))
    if n_masked:
        log.warning("[%s] %d surface T values masked (outside [150, 280] K)", cfg.label, n_masked)

    T_air    = np.asarray(ds_rs["t"],    dtype=float)
    pres_raw = np.asarray(ds_rs["pres"], dtype=float)
    # auto-detect Pa vs hPa
    p_hPa    = pres_raw / 100.0 if np.nanmedian(pres_raw) > 2000 else pres_raw
    U_spd    = np.asarray(ds_rs["ws"],   dtype=float)
    # z_ref may be scalar or per-leg array; broadcast to (n_legs,)
    z_arr    = np.broadcast_to(np.asarray(z_ref, dtype=float), len(T_sfc)).copy()

    theta_sfc = potential_temperature(T_sfc, p_hPa)
    theta_air = potential_temperature(T_air, p_hPa)
    dT        = theta_air - theta_sfc
    rho       = air_density(T_air, p_hPa)

    # ── roughness: constant (sensitivity sweep) → fractional (default) → T-class fallback
    stype = classify_surface(T_sfc, cfg.T_thresh_open, cfg.T_thresh_thick)  # always for diagnostics
    if cfg.z0_constant is not None:
        z0 = np.full(len(T_sfc), cfg.z0_constant)
        zt = z0 * cfg.zt_ratio
        roughness_method = "constant"
        log.info("[%s] constant roughness: z0=%.2e m", cfg.label, cfg.z0_constant)
    else:
        # prefer fractions from MODIS aggregate, then P6 aggregate, then T-class
        z0, zt = _roughness_from_ds(ds_modis, cfg)
        if z0 is None:
            z0, zt = _roughness_from_ds(ds_obs, cfg)
        if z0 is not None:
            roughness_method = "fractional"
        else:
            log.info("[%s] frac_* not found – falling back to T-based classification", cfg.label)
            z0, zt = assign_roughness(stype, cfg.z0_ocean, cfg.z0_thin_ice, cfg.z0_thick_ice, cfg.zt_ratio)
            roughness_method = "temperature_class"
    log.info("[%s] roughness method: %s", cfg.label, roughness_method)

    # RS product carries only scalar wind speed; split into u/v for 3pt Ri
    # (use ws as u, zero v – conservative approximation when direction unavailable)
    U_u = U_spd
    U_v = np.zeros_like(U_spd)

    if cfg.zt_model == "constant" and not cfg.use_gustiness:
        Ri = _compute_ri(theta_sfc, theta_air, U_u, U_v, z0, zt, z_arr, cfg, ds_tbird)
        Cd, Ch = _coefs_loop(Ri, z0, zt, z_arr, cfg)
        U_eff = U_spd
    else:
        # Check fractions from ds_modis first, then ds_obs
        f_open = np.asarray(ds_modis["frac_open"], dtype=float) if "frac_open" in ds_modis else (np.asarray(ds_obs["frac_open"], dtype=float) if "frac_open" in ds_obs else None)
        f_thin = np.asarray(ds_modis["frac_thin_ice"], dtype=float) if "frac_thin_ice" in ds_modis else (np.asarray(ds_obs["frac_thin_ice"], dtype=float) if "frac_thin_ice" in ds_obs else None)
        f_thick = np.asarray(ds_modis["frac_thick_ice"], dtype=float) if "frac_thick_ice" in ds_modis else (np.asarray(ds_obs["frac_thick_ice"], dtype=float) if "frac_thick_ice" in ds_obs else None)
        Cd, Ch, z0, zt, Ri, U_eff = solve_coupled_bulk(
            theta_sfc=theta_sfc,
            theta_air=theta_air,
            U_u=U_u,
            U_v=U_v,
            z_arr=z_arr,
            cfg=cfg,
            f_open=f_open,
            f_thin=f_thin,
            f_thick=f_thick,
            surface_type=stype,
        )

    log.info("[%s] Ri  mean=%.3f  range=[%.3f, %.3f]",
             cfg.label, float(np.nanmean(Ri)), float(np.nanmin(Ri)), float(np.nanmax(Ri)))

    H = rho * CP * Ch * U_eff * (theta_sfc - theta_air)
    _sanity(H, dT, label=cfg.label + " RS")

    coords = {"start_time": ds_obs["start_time"]}
    _v = lambda data, units, lname: xr.DataArray(
        data, dims=["start_time"],
        coords=coords,
        attrs={"units": units, "long_name": lname},
    )
    out = xr.Dataset(
        {
            "H_rs_bulk": _v(H,   "W m-2",  "Bulk SHF RS; positive upward"),
            "Ri_rs":   _v(Ri,  "-",     "Richardson number (RS)"),
            "z0_rs":     _v(z0,  "m",     "Momentum roughness length"),
            "zt_rs":     _v(zt,  "m",     "Scalar roughness length"),
            "Cd_rs":     _v(Cd,  "-",     "Drag coefficient"),
            "Ch_rs":     _v(Ch,  "-",     "Heat transfer coefficient"),
            "U_eff_rs":  _v(U_eff, "m s-1", "Effective wind speed"),
            "rho_rs":    _v(rho, "kg m-3","Air density"),
            "dtheta_rs":     _v(dT,  "K",     "θ_air − θ_sfc"),
            "surface_type_rs": xr.DataArray(stype, dims=["start_time"], coords=coords),
        },
        attrs={**asdict(cfg),
               "ri_method": ("2pt_surface_tbird" if ds_tbird is not None
                             else f"2pt_surface_zref_{cfg.ri_wind_remote}_wind"),
               "roughness_method": roughness_method},
    )
    log.info("[%s] RS done in %.1f s  |  H mean=%.1f W/m²",
             cfg.label, time.perf_counter() - t0, float(np.nanmean(H)))
    return out

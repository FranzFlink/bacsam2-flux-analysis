r"""
fluxpy.bulk
===========
Bulk sensible heat flux (manuscript Sect. 3.2, Eq. 2, and Appendix B),

    H_S = rho * c_p * C_h * U * (theta_s - theta_air)     [W m-2, positive upward]

evaluated on the same 5 km segments as the EC reference. The three bulk
estimates of the paper differ only in their inputs (Table 1):

    compute_bulk_shf_obs      H_S,obs,bulk   KT-19 T_s; T-Bird theta, U, z          (in situ)
    compute_bulk_shf_remote   H_S,RS,bulk    MODIS T_s; ERA5 theta, U, p (level 136)
                              H_S,ERA,bulk   ERA5 T_s;  ERA5 theta, U, p (level 136)

Steps inside each (Appendix B):
  B1  potential temperatures from T and p; bulk Richardson number Ri_b between the
      surface (no-slip) and the measurement level (:mod:`fluxpy.stability`)
  B2  effective roughness of the segment from its surface composition,
      z0m_eff = fraction-weighted geometric mean (Eq. B3); z0T = alpha_T * z0m
  B3  transfer coefficient C_h = C_hn * f_h(Ri_b, z/z0m, z/z0T) (:mod:`fluxpy.stability_lib`)

Each potential temperature is computed with the pressure at its own height. In
situ, the surface pressure is reconstructed from the T-Bird pressure with the
hypsometric equation (no surface pressure is measured; fluxpy.util.
surface_pressure_hypsometric); using the T-Bird pressure at the surface as well
would make theta_s too warm by about g z / c_p = 0.23 K at 24 m. Remote, theta_s
uses the ERA5 surface pressure and theta_air the pressure of model level 136
(fluxpy.util.era5_level_pressure).

Sign: theta_s > theta_air (warm surface, e.g. a lead) gives H_S > 0, upward.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import xarray as xr

from .stability import Cd_Ch_noniterative, SCHEMES, bulk_richardson_2pt, effective_wind_speed
from .util import (CP, air_density, assign_roughness, classify_surface, potential_temperature,
                   surface_pressure_hypsometric)

log = logging.getLogger(__name__)

__all__ = ["BulkConfig", "reference_config", "amplitude_matched_config",
           "effective_roughness", "bulk_richardson_in_situ", "bulk_richardson_remote",
           "transfer_coefficients", "compute_bulk_shf_obs", "compute_bulk_shf_remote"]


# ── configuration ─────────────────────────────────────────────────────────────

@dataclass
class BulkConfig:
    """The free parameters of one bulk flux run (manuscript Table B1 and Sect. 4.2)."""
    scheme_name: str = "GLGS20"      #: stability functions (one of fluxpy.stability_lib.SCHEMES)
    zt_ratio: float = 0.1            #: scalar roughness ratio alpha_T = z0T / z0m
    z0_ocean: float = 2.0e-4         #: momentum roughness of open water [m]        (Table B1)
    z0_thin_ice: float = 1.0e-4      #: ... of thin ice [m]
    z0_thick_ice: float = 4.0e-4     #: ... of cold snow-covered ice [m]
    z0_constant: Optional[float] = None   #: when set, one z0m for every segment (sensitivity experiment)
    T_thresh_open: float = -1.8      #: KT-19 T_s above this: open water [degC]     (Table B1)
    T_thresh_thick: float = -20.0    #: KT-19 T_s at or below this: snow-covered ice [degC]
    ri_eps_wind2: float = 1e-4       #: shear regularisation eps of the in-situ Ri_b [m2 s-2] (App. B1)
    ri_clip: float = np.inf          #: |Ri_b| bound; inf = none, as in the paper ("no further limitation").
                                     #: The flux statistics change by < 0.03 W m-2 for any bound >= 1 (App. B1).
    label: str = "default"

    def __post_init__(self):
        if self.scheme_name not in SCHEMES:
            raise ValueError(f"Unknown scheme '{self.scheme_name}'. Available: {sorted(SCHEMES)}")
        if not (0 < self.zt_ratio <= 1):
            raise ValueError("zt_ratio must be in (0, 1]")
        if self.T_thresh_thick >= self.T_thresh_open:
            raise ValueError("T_thresh_thick must be < T_thresh_open")
        if self.z0_constant is not None and self.z0_constant <= 0:
            raise ValueError("z0_constant must be > 0")
        if self.ri_eps_wind2 < 0:
            raise ValueError("ri_eps_wind2 must be >= 0")
        if self.ri_clip <= 0:
            raise ValueError("ri_clip must be > 0 (np.inf disables the bound)")


def reference_config() -> BulkConfig:
    """Reference configuration of the paper: GLGS20, alpha_T = 1e-1 (the conventional
    value of large-eddy simulations over sea ice), Table B1 roughness set."""
    return BulkConfig(scheme_name="GLGS20", zt_ratio=1e-1, label="GLGS20_ref")


def amplitude_matched_config() -> BulkConfig:
    """Amplitude-matched configuration: alpha_T = 1e-4, the value at which the in-situ
    bulk estimate reproduces the observed flux amplitude (regression slope 1;
    manuscript Sect. 4.2). Reported alongside the reference, never as a roughness retrieval."""
    return BulkConfig(scheme_name="GLGS20", zt_ratio=1e-4, label="GLGS20_ztr1e-04")


# ── B2  roughness ─────────────────────────────────────────────────────────────

def effective_roughness(f_open, f_thin, f_thick, cfg: BulkConfig) -> tuple[np.ndarray, np.ndarray]:
    r"""Effective roughness lengths of each segment from its surface fractions (Eq. B3),

    $$ \ln z_{0,m}^\mathrm{eff} = \sum_i f_i \ln z_{0,m,i}, \qquad
       z_{0,T}^\mathrm{eff} = \alpha_T z_{0,m}^\mathrm{eff}, $$

    i.e. a fraction-weighted geometric mean over open water, thin ice and
    snow-covered ice. The logarithm is the right average because z0 enters the
    transfer coefficient through ln(z/z0). Fractions are clipped to [0, 1] and
    renormalised; segments without valid fractions get NaN.
    """
    f = np.vstack([f_open, f_thin, f_thick]).T.astype(float)
    f = np.clip(np.where(np.isfinite(f), f, 0.0), 0.0, 1.0)
    total = f.sum(axis=1)
    valid = total > 0
    ln_z0 = np.full(f.shape[0], np.nan)
    ln_z0[valid] = (f[valid] / total[valid, None]) @ np.log([cfg.z0_ocean, cfg.z0_thin_ice, cfg.z0_thick_ice])
    z0 = np.exp(ln_z0)
    return z0, z0 * cfg.zt_ratio


def _roughness(ds_fractions: xr.Dataset, T_sfc_degC: np.ndarray, cfg: BulkConfig) -> tuple[np.ndarray, np.ndarray, str]:
    """(z0m, z0T, method): a constant z0m if the configuration says so, otherwise the
    fraction-weighted blend of Eq. (B3); if the fractions are missing, the class of
    the segment-mean surface temperature."""
    n = len(T_sfc_degC)
    if cfg.z0_constant is not None:
        z0 = np.full(n, cfg.z0_constant)
        return z0, z0 * cfg.zt_ratio, "constant"
    if all(v in ds_fractions for v in ("frac_open", "frac_thin_ice", "frac_thick_ice")):
        z0, zt = effective_roughness(np.asarray(ds_fractions["frac_open"], dtype=float),
                                     np.asarray(ds_fractions["frac_thin_ice"], dtype=float),
                                     np.asarray(ds_fractions["frac_thick_ice"], dtype=float), cfg)
        return z0, zt, "fractional"
    surface_class = classify_surface(T_sfc_degC, cfg.T_thresh_open, cfg.T_thresh_thick)
    z0, zt = assign_roughness(surface_class, cfg.z0_ocean, cfg.z0_thin_ice, cfg.z0_thick_ice, cfg.zt_ratio)
    return z0, zt, "temperature_class"


# ── B1  bulk Richardson number ────────────────────────────────────────────────

def bulk_richardson_in_situ(theta_sfc, theta_air, u, v, z, cfg: BulkConfig) -> np.ndarray:
    """Ri_b of the in-situ pathway (Eq. B2): surface (no-slip) to T-Bird level, from the
    segment-mean theta, wind components and height that also enter the flux; the
    measured wind with the regularisation eps in the shear denominator."""
    Ri = bulk_richardson_2pt(dtheta=theta_air - theta_sfc, dz=z, du=u, dv=v,
                             theta_ref=0.5 * (theta_sfc + theta_air), eps_wind2=cfg.ri_eps_wind2)
    return np.clip(Ri, -cfg.ri_clip, cfg.ri_clip)


def bulk_richardson_remote(theta_sfc, theta_air, wind_speed, z, cfg: BulkConfig) -> np.ndarray:
    """Ri_b of the reanalysis-driven pathways (Eq. B2): surface to the ERA5 level, with the
    effective wind speed of Jordan et al. (1999) / Fairall et al. (1996) in the shear,
    which bounds Ri_b at vanishing wind; no eps, no further limitation."""
    dtheta = theta_air - theta_sfc
    theta_bar = 0.5 * (theta_sfc + theta_air)
    S = effective_wind_speed(wind_speed, dtheta, z, theta_bar, scheme=cfg.scheme_name)
    Ri = bulk_richardson_2pt(dtheta=dtheta, dz=z, theta_ref=theta_bar, eps_wind2=0.0, wspd=S)
    return np.clip(Ri, -cfg.ri_clip, cfg.ri_clip)


# ── B3  transfer coefficients ─────────────────────────────────────────────────

def transfer_coefficients(Ri, z, z0, zt, scheme_name: str) -> tuple[np.ndarray, np.ndarray]:
    """(C_d, C_h) per segment from ``fluxpy.stability_lib.Cd_Ch_noniterative``;
    NaN where the inputs are missing or the geometry is invalid (z <= z0)."""
    n = len(Ri)
    Cd, Ch = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        if not (np.isfinite(Ri[i]) and np.isfinite(z[i]) and z0[i] > 0 and zt[i] > 0
                and z[i] > z0[i] and z[i] > zt[i]):
            continue
        try:
            Cd[i], Ch[i] = Cd_Ch_noniterative(float(Ri[i]), float(z[i]), float(z0[i]), float(zt[i]), scheme_name)
        except Exception as exc:                       # pragma: no cover - defensive
            log.debug("segment %d: %s", i, exc)
    return Cd, Ch


# ── the two entry points ──────────────────────────────────────────────────────

def _as_dataset(fields: dict, start_time, suffix: str, cfg: BulkConfig, method: str) -> xr.Dataset:
    """Pack the per-segment arrays into a Dataset with units; variable names get the pathway suffix."""
    units = {"H": ("W m-2", "Bulk sensible heat flux, positive upward"),
             "Ri": ("1", "Bulk Richardson number, surface to measurement level (Eq. B2)"),
             "z0": ("m", "Effective momentum roughness length"),
             "zt": ("m", "Scalar roughness length, alpha_T * z0"),
             "Cd": ("1", "Drag coefficient"), "Ch": ("1", "Heat transfer coefficient"),
             "rho": ("kg m-3", "Air density"), "dtheta": ("K", "theta_air - theta_s")}
    def variable_name(name: str) -> str:
        return f"H_{suffix}_bulk" if name == "H" else f"{name}_{suffix}"      # H_obs_bulk, Ri_obs, ...

    data = {variable_name(name): xr.DataArray(values, dims=["start_time"], coords={"start_time": start_time},
                                              attrs=dict(zip(("units", "long_name"), units[name])))
            for name, values in fields.items()}
    return xr.Dataset(data, attrs={**asdict(cfg), "roughness_method": method})


def compute_bulk_shf_obs(ds: xr.Dataset, cfg: BulkConfig) -> xr.Dataset:
    """H_S,obs,bulk: the in-situ bulk estimate (manuscript Table 1, row 2).

    ``ds`` is the segment file: ``t_kt19`` [degC] KT-19 surface temperature,
    ``T_mean`` [degC], ``p_mean`` [hPa], ``u_mean``, ``v_mean`` [m/s], ``altitude``
    [m] of the T-Bird, and the surface fractions ``frac_*``. theta_air uses the
    T-Bird pressure; theta_s the surface pressure reconstructed from it with the
    hypsometric equation (no surface pressure is measured).
    Returns ``H_obs_bulk``, ``Ri_obs``, ``z0_obs``, ``zt_obs``, ``Cd_obs``,
    ``Ch_obs``, ``rho_obs``, ``dtheta_obs``.
    """
    T_sfc = np.asarray(ds["t_kt19"], dtype=float)
    T_air = np.asarray(ds["T_mean"], dtype=float)
    p_hPa = np.asarray(ds["p_mean"], dtype=float)
    u = np.asarray(ds["u_mean"], dtype=float)
    v = np.asarray(ds["v_mean"], dtype=float)
    z = np.asarray(ds["altitude"], dtype=float)

    p_sfc_hPa = surface_pressure_hypsometric(p_hPa, z, T_mean=0.5 * (T_sfc + T_air))   # B1, layer-mean T
    theta_sfc = potential_temperature(T_sfc, p_sfc_hPa)
    theta_air = potential_temperature(T_air, p_hPa)
    rho = air_density(T_air, p_hPa)
    z0, zt, method = _roughness(ds, T_sfc, cfg)                           # B2
    Ri = bulk_richardson_in_situ(theta_sfc, theta_air, u, v, z, cfg)     # B1
    Cd, Ch = transfer_coefficients(Ri, z, z0, zt, cfg.scheme_name)        # B3
    U = np.sqrt(u ** 2 + v ** 2)
    H = rho * CP * Ch * U * (theta_sfc - theta_air)                       # Eq. (2)

    log.info("[%s] H_obs_bulk mean %.1f W m-2 (roughness: %s)", cfg.label, np.nanmean(H), method)
    return _as_dataset(dict(H=H, Ri=Ri, z0=z0, zt=zt, Cd=Cd, Ch=Ch, rho=rho, dtheta=theta_air - theta_sfc),
                       ds["start_time"], "obs", cfg, method)


def compute_bulk_shf_remote(ds_obs: xr.Dataset, ds_rs: xr.Dataset, ds_modis: xr.Dataset,
                            cfg: BulkConfig, z_ref=30.0) -> xr.Dataset:
    """H_S,RS,bulk or H_S,ERA,bulk: the reanalysis-driven bulk estimate (Table 1, rows 3-4).

    ``ds_rs``    the ERA5 column at the level closest to the T-Bird: ``t`` [degC],
                 ``ws`` [m/s], ``pres`` [Pa or hPa] (surface pressure, for theta_s) and
                 ``pres_air`` [Pa or hPa] (pressure of the level of ``t``, for theta_air
                 and the air density).
    ``ds_modis`` the surface temperature ``IST`` [K] (MODIS, gap-filled with ERA5
                 skin temperature, or ERA5 skin temperature alone); values outside
                 150-280 K are masked.
    ``ds_obs``   the segment file, for the surface fractions of the roughness blend.
    ``z_ref``    height of the ERA5 level [m], scalar or per segment.
    Returns ``H_rs_bulk``, ``Ri_rs``, ``z0_rs``, ``zt_rs``, ``Cd_rs``, ``Ch_rs``,
    ``rho_rs``, ``dtheta_rs``.
    """
    T_sfc_K = np.asarray(ds_modis["IST"], dtype=float)
    T_sfc = np.where((T_sfc_K > 150) & (T_sfc_K < 280), T_sfc_K - 273.15, np.nan)
    T_air = np.asarray(ds_rs["t"], dtype=float)
    p_sfc = np.asarray(ds_rs["pres"], dtype=float)
    p_air = np.asarray(ds_rs["pres_air"], dtype=float)
    U = np.asarray(ds_rs["ws"], dtype=float)
    z = np.broadcast_to(np.asarray(z_ref, dtype=float), len(T_sfc)).copy()

    theta_sfc = potential_temperature(T_sfc, p_sfc)                               # B1, each at its own pressure
    theta_air = potential_temperature(T_air, p_air)
    rho = air_density(T_air, p_air)
    z0, zt, method = _roughness(ds_obs, T_sfc, cfg)                           # B2
    Ri = bulk_richardson_remote(theta_sfc, theta_air, U, z, cfg)              # B1
    Cd, Ch = transfer_coefficients(Ri, z, z0, zt, cfg.scheme_name)            # B3
    H = rho * CP * Ch * U * (theta_sfc - theta_air)                           # Eq. (2)

    log.info("[%s] H_rs_bulk mean %.1f W m-2 (%d of %d T_s valid)", cfg.label, np.nanmean(H),
             int(np.isfinite(T_sfc).sum()), len(T_sfc))
    return _as_dataset(dict(H=H, Ri=Ri, z0=z0, zt=zt, Cd=Cd, Ch=Ch, rho=rho, dtheta=theta_air - theta_sfc),
                       ds_obs["start_time"], "rs", cfg, method)

r"""
fluxpy.aggregate
================
100 Hz T-Bird legs  ->  one row per 5 km segment.

Manuscript: Sect. 2.1 (airborne measurements), Sect. 3.1 (EC method) and
Appendix B2 (surface composition from the KT-19).

For every straight low-level leg (start and end times in
``data/observations/ar/analysis_ready_polar6_legs_metadata.csv``) the
along-track distance is computed from the true airspeed, the leg is cut into
consecutive 5 km segments, and for each segment one row holds

* the EC flux, its random error and the integrity check
  (:mod:`fluxpy.eddy_covariance`, :mod:`fluxpy.qc`),
* the segment means that drive the T-Bird bulk estimate: air temperature,
  pressure, wind, height and the KT-19 surface temperature,
* the surface composition of the segment from the 1 Hz KT-19 (open water,
  thin ice, snow-covered ice; Table B1) and the merged MODIS-AMSR2 sea-ice
  concentration.

Entry points: :func:`process_legs` (list of legs -> DataFrame) and
:func:`to_xarray` (DataFrame -> self-describing Dataset). The command-line
wrapper is ``scripts/run_ec_aggregation.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from .eddy_covariance import SAMPLING_HZ, segment_flux
from .qc import segment_spectra, turbulence_resolved
from .util import add_theta

__all__ = ["SegmentConfig", "process_legs", "to_xarray"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SegmentConfig:
    """How the legs are cut and classified (defaults = the paper)."""
    length_km: float = 5.0          #: segment length along the track [km]
    min_samples: int = 100          #: skip a segment with fewer 100 Hz samples (a full
                                    #: 5 km segment has about 8300)
    max_height_range_m: float = 30.0    #: skip a segment whose T-Bird height (raw 100 Hz,
                                        #: max - min) varies more than this: constant flight
                                        #: level, Appendix A1. Level segments vary by at most
                                        #: 22 m (altimeter over rough ice, towed-body motion);
                                        #: the one segment flown while the T-Bird was winched
                                        #: up (13 April 10:21 UTC) by 53 m.
    fs: float = SAMPLING_HZ         #: sampling frequency of the T-Bird [Hz]
    T_open_degC: float = -1.8       #: KT-19 T_s above this: open water      (Table B1)
    T_thick_degC: float = -20.0     #: KT-19 T_s at or below this: snow-covered ice
    T_lead_degC: float = -5.0       #: "lead fraction": share of KT-19 samples warmer than this
    kt19_var: str = "KT19"          #: name of the KT-19 variable in the radiation file [degC]
    verbose: bool = False


# ---------------------------------------------------------------------------
# Helpers for one leg / one segment
# ---------------------------------------------------------------------------

#: T-Bird variables a segment needs (the 100 Hz file has no gaps inside the legs,
#: but the flux must never be computed across a missing sample)
_REQUIRED = ["w", "theta", "T", "p", "u", "v", "h", "lat", "lon"]


def _add_along_track_distance(leg: xr.Dataset) -> xr.Dataset:
    """Distance along the track [km] from the mean true airspeed, as the new dimension.

    The elapsed time is taken in whole seconds (``dt.seconds``), so the distance
    coordinate advances in steps of one second of flight (about 60 m) and the
    segment boundaries fall on whole seconds.
    """
    elapsed_s = (leg["time"] - leg["time"][0]).dt.seconds
    distance_km = leg["tas"].mean() * elapsed_s / 1000.0
    return (leg.assign_coords(distance=("time", distance_km.values))
               .swap_dims({"time": "distance"}))


def _segment_edges(leg_length_km: float, length_km: float) -> list[tuple[float, float]]:
    """Consecutive, non-overlapping [left, right] edges in km; a trailing remainder is dropped."""
    edges, left = [], 0.0
    while left + length_km <= leg_length_km + 1e-9:
        edges.append((left, left + length_km))
        left += length_km
    return edges


def _surface_composition(kt19_1hz: np.ndarray, cfg: SegmentConfig) -> dict:
    """Fractions of open water / thin ice / snow-covered ice from the 1 Hz KT-19 (Table B1).

    The class fractions are taken over the valid samples; the lead fraction over
    all samples of the segment (a missing sample counts as "not a lead").
    """
    lead_frac = float((kt19_1hz > cfg.T_lead_degC).mean()) if kt19_1hz.size else np.nan
    T_s = kt19_1hz[np.isfinite(kt19_1hz)]
    if T_s.size == 0:
        return dict(frac_open=np.nan, frac_thin_ice=np.nan, frac_thick_ice=np.nan, lead_frac=lead_frac)
    frac_open = float((T_s > cfg.T_open_degC).mean())
    frac_thick = float((T_s <= cfg.T_thick_degC).mean())
    return dict(frac_open=frac_open,
                frac_thin_ice=float(1.0 - frac_open - frac_thick),
                frac_thick_ice=frac_thick,
                lead_frac=lead_frac)


def _sea_ice_concentration(segment: xr.Dataset, ds_sic: xr.Dataset, fs: float) -> float:
    """Mean sea-ice concentration along the segment (nearest grid cell to the 1 Hz T-Bird position)."""
    try:
        pos = segment[["lat", "lon"]].swap_dims({"distance": "time"}).isel(time=slice(None, None, int(fs)))
        along = ds_sic.sel(time=pos.time, lat=pos.lat, lon=pos.lon, method="nearest")
        return float(along["sic"].mean())
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def process_legs(legs: list[xr.Dataset], ds_radiation: xr.Dataset, ds_sic: xr.Dataset,
                 cfg: SegmentConfig = SegmentConfig()) -> pd.DataFrame:
    """Cut every leg into segments and compute one row per segment.

    Parameters
    ----------
    legs         : T-Bird 100 Hz time slices, one per leg (``T`` degC, ``p`` hPa,
                   ``u``, ``v``, ``w``, ``tas`` m/s, ``h`` m, ``lat``, ``lon``).
    ds_radiation : Polar 6 radiation file with the 1 Hz KT-19 surface temperature [degC].
    ds_sic       : gridded sea-ice concentration (``time``, ``lat``, ``lon``; ``sic`` in %).
    cfg          : :class:`SegmentConfig`.

    Returns
    -------
    DataFrame with one row per segment (columns documented in :func:`to_xarray`).
    """
    rows: list[dict] = []
    n_evaluated = 0

    for leg in tqdm(legs, desc="legs"):
        if leg.time.size == 0:
            continue
        leg = _add_along_track_distance(add_theta(leg))
        leg_length_km = float(leg.distance[-1] - leg.distance[0])

        for left_km, right_km in _segment_edges(leg_length_km, cfg.length_km):
            n_evaluated += 1
            segment = leg.sel(distance=slice(left_km, right_km)).dropna("distance", how="any", subset=_REQUIRED)
            n_samples = int(segment.distance.size)
            if n_samples < cfg.min_samples:
                if cfg.verbose:
                    print(f"  skipped [{left_km:.0f}, {right_km:.0f}] km: {n_samples} samples")
                continue
            height_range = float(segment["h"].max() - segment["h"].min())
            if height_range > cfg.max_height_range_m:
                if cfg.verbose:
                    print(f"  skipped [{left_km:.0f}, {right_km:.0f}] km: height varies by {height_range:.0f} m")
                continue

            start = pd.to_datetime(segment["time"][0].item())
            end = pd.to_datetime(segment["time"][-1].item())

            # -- EC flux, random error, integrity check (Sect. 3.1, Appendix A) --
            ec = segment_flux(segment, fs=cfg.fs)
            spectra = segment_spectra(segment["w"].values, segment["theta"].values,
                                      segment["u"].values, segment["v"].values, fs=cfg.fs)
            resolved = turbulence_resolved(spectra["slope_w"])

            # -- surface: KT-19 at 1 Hz, sea-ice concentration --------------------
            kt19 = ds_radiation[cfg.kt19_var].sel(time=slice(start, end))
            composition = _surface_composition(kt19.values.astype(float), cfg)

            rows.append({
                "start_time": start,
                "end_time": end,
                "N_samples": n_samples,
                "segment_length_km": float(right_km - left_km),
                # T-Bird segment means (inputs of the T-Bird bulk estimate)
                "altitude": float(segment["h"].mean()),
                "T_mean": float(segment["T"].mean()),
                "p_mean": float(segment["p"].mean()),
                "u_mean": float(segment["u"].mean()),
                "v_mean": float(segment["v"].mean()),
                "latitude": float(segment["lat"].mean()),
                "longitude": float(segment["lon"].mean()),
                # EC reference
                "sensible_heat_flux": ec.flux,                  # kept whatever the integrity check says
                "sensible_heat_flux_random_error": ec.random_error,
                "flux_over_random_error": ec.flux_over_random_error,
                "qc_slope_w": spectra["slope_w"],
                "turbulence_resolved": float(resolved),
                # surface
                "t_kt19": float(kt19.mean(skipna=True)),
                "T_sfc_std": float(kt19.std(skipna=True)),
                "sic": _sea_ice_concentration(segment, ds_sic, cfg.fs),
                **composition,
            })

    print(f"segments evaluated: {n_evaluated}   kept: {len(rows)}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

#: units and meaning of every column (these become the NetCDF attributes)
VARIABLES = {
    "end_time":              (None, "End of the segment"),
    "N_samples":             ("1", "Number of 100 Hz samples in the segment"),
    "segment_length_km":     ("km", "Segment length along the track"),
    "altitude":              ("m", "Segment-mean height of the T-Bird above the surface"),
    "T_mean":                ("degC", "Segment-mean air temperature (T-Bird)"),
    "p_mean":                ("hPa", "Segment-mean static pressure (T-Bird)"),
    "u_mean":                ("m s-1", "Segment-mean eastward wind (T-Bird)"),
    "v_mean":                ("m s-1", "Segment-mean northward wind (T-Bird)"),
    "latitude":              ("degrees_north", "Segment-mean latitude"),
    "longitude":             ("degrees_east", "Segment-mean longitude"),
    "sensible_heat_flux":    ("W m-2", "Eddy-covariance sensible heat flux, 0.1-20 Hz band; positive upward"),
    "sensible_heat_flux_random_error": ("W m-2", "Random (sampling) error of the EC flux, 1 sigma (Finkelstein and Sims, 2001)"),
    "flux_over_random_error": ("1", "|flux| divided by its random error; the paper counts the fluxes above 1, 2 and 3 (all fluxes are kept)"),
    "qc_slope_w":            ("1", "Log-log slope of the w power spectrum, 0.1-20 Hz"),
    "turbulence_resolved":   ("1", "1 if |qc_slope_w + 5/3| <= 0.4 (integrity check, Appendix A1; a diagnostic, no segment is excluded by it)"),
    "t_kt19":                ("degC", "Segment-mean KT-19 surface brightness temperature"),
    "T_sfc_std":             ("K", "Standard deviation of the 1 Hz KT-19 surface temperature within the segment"),
    "sic":                   ("%", "Sea-ice concentration along the segment (merged MODIS-AMSR2)"),
    "lead_frac":             ("1", "Fraction of KT-19 samples warmer than -5 degC"),
    "frac_open":             ("1", "Fraction of KT-19 samples classified as open water (T_s > -1.8 degC)"),
    "frac_thin_ice":         ("1", "Fraction of KT-19 samples classified as thin ice"),
    "frac_thick_ice":        ("1", "Fraction of KT-19 samples classified as snow-covered ice (T_s <= -20 degC)"),
}


def to_xarray(df: pd.DataFrame) -> xr.Dataset:
    """DataFrame of :func:`process_legs` -> Dataset indexed by ``start_time``, with units."""
    ds = df.set_index("start_time").to_xarray().sortby("start_time")
    for name, (units, long_name) in VARIABLES.items():
        if name in ds:
            ds[name].attrs["long_name"] = long_name
            if units is not None:
                ds[name].attrs["units"] = units
    ds.attrs.update(
        title="BACSAM-II T-Bird eddy-covariance fluxes and segment means on 5 km segments",
        source="scripts/run_ec_aggregation.py (fluxpy.aggregate)",
    )
    return ds

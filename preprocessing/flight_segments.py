"""Flight segmentation: turn the flight-separation JSON into leg/profile slices.

Extracted unchanged from the former ``lib/helper.py`` (process_flight_segments
and its private helpers).  Only needed to REBUILD the leg table from the full
Polar 6 archive; the shipped ``analysis_ready_polar6_legs_metadata.csv`` makes
this step unnecessary for reproducing the paper.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# ----------------------------------------------------------------------------- #
# Logger
# ----------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)  # configure in user space via basicConfig

# ----------------------------------------------------------------------------- #
# Types and helpers
# ----------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SegmentKey:
    """Immutable identifier for deduplication: start and end ISO times."""
    start: str
    end: str


def _open_ds(maybe_path: xr.Dataset | str | Path, engine: Optional[str] = None) -> xr.Dataset:
    """Open an xarray Dataset from path or pass-through if already a Dataset."""
    if isinstance(maybe_path, xr.Dataset):
        return maybe_path
    p = Path(maybe_path).expanduser()
    logger.debug("Opening dataset: %s", p)
    ds = xr.open_dataset(p, engine=engine)  # engine left flexible for NetCDF/Zarr
    if "time" in ds.coords and not ds.indexes["time"].is_monotonic_increasing:
        logger.warning("Dataset %s has non-monotonic time coord. Sorting by time.", p)
        ds = ds.sortby("time")
    return ds


def _safe_sel(ds: xr.Dataset, start: str, stop: str) -> xr.Dataset:
    """Slice dataset by time and ensure non-empty result; returns empty dataset if no overlap."""
    try:
        sub = ds.sel(time=slice(start, stop))
        if "time" in sub.coords and sub.time.size == 0:
            logger.debug("Empty selection for interval [%s, %s]", start, stop)
        return sub
    except Exception as e:
        logger.debug("Selection failed for [%s, %s]: %s", start, stop, e)
        return ds.isel(time=slice(0, 0)) if "time" in ds.coords else ds


def _bin_profile(ds: xr.Dataset, var_alt: str, bin_edges: np.ndarray) -> xr.Dataset:
    """Group a dataset into altitude bins using xarray.groupby_bins."""
    if var_alt not in ds:
        raise KeyError(f"Altitude variable '{var_alt}' not found in dataset.")
    binned = ds.groupby_bins(var_alt, bin_edges).mean()
    # propagate a compact coordinate for day if available
    if "time" in ds:
        try:
            day_str = str(ds["time"].dt.strftime("%Y-%m-%d").values[0])
            binned = binned.assign_coords(day=("day", [day_str]))
        except Exception:
            pass
    return binned


def _segment_stats(
    ds: xr.Dataset,
    *,
    platform: str,
    kind: str,
    start: str,
    end: str,
    var_alt_candidates: Tuple[str, ...] = ("h", "altitude", "z"),
    var_T_candidates: Tuple[str, ...] = ("T", "temp", "temperature"),
    var_u_candidates: Tuple[str, ...] = ("u",),
    var_v_candidates: Tuple[str, ...] = ("v",),
    var_w_candidates: Tuple[str, ...] = ("w", "ww"),
) -> Dict[str, object]:
    """
    Compute summary stats for a segment from the *unbinned* time-slice.

    Returns a dict with keys:
      day, platform, type, start_time, end_time, N_points, mean_altitude, mean_T, mean_u, mean_v, mean_w
    Missing variables are filled with NaN.
    """
    # Helpers to pick first available variable
    def _pick(ds: xr.Dataset, candidates: Tuple[str, ...]) -> Optional[str]:
        for c in candidates:
            if c in ds:
                return c
        return None

    var_alt = _pick(ds, var_alt_candidates)
    var_T = _pick(ds, var_T_candidates)
    var_u = _pick(ds, var_u_candidates)
    var_v = _pick(ds, var_v_candidates)
    var_w = _pick(ds, var_w_candidates)

    def _nanmean(var: Optional[str]) -> float:
        if var is None or var not in ds:
            return float("nan")
        try:
            return float(ds[var].mean(skipna=True).item())
        except Exception:
            try:
                return float(ds[var].mean(skipna=True).values)
            except Exception:
                return float("nan")

    N = int(ds.time.size) if "time" in ds.coords else 0
    try:
        day = str(ds.time.dt.strftime("%Y-%m-%d").values[0]) if N > 0 else ""
    except Exception:
        day = ""

    stats = {
        "day": day,
        "platform": platform,
        "type": kind,
        "start_time": start,
        "end_time": end,
        "N_points": N,
        "mean_altitude": _nanmean(var_alt),
        "mean_T": _nanmean(var_T),
        "mean_u": _nanmean(var_u),
        "mean_v": _nanmean(var_v),
        "mean_w": _nanmean(var_w),
    }
    return stats


# ----------------------------------------------------------------------------- #
# Main high-level function
# ----------------------------------------------------------------------------- #

def process_flight_segments(
    flight_sep_json: str | Path,
    polar6_path: str | Path,
    ar_path: str | Path,
    tbird_path: Optional[str | Path] = None,
    *,
    leg_regex: str = r"L\d+",
    profile_regex: str = r"MD\d+",
    profile_bin_edges: np.ndarray | Iterable[float] = np.arange(0, 4000, 10),
    polar6_alt_var: str = "h",
    ar_alt_var: str = "altitude",
    deduplicate_legs: bool = True,
) -> Dict[str, object]:
    """
    Process flight separation metadata into segment datasets and per-segment overviews.

    Parameters
    ----------
    flight_sep_json : str | Path
        Path to JSON file mapping day → {segment_key → {'start_time', 'end_time', ...}}.
    polar6_path : str | Path
        Path to Polar-6 noseboom-unified NetCDF.
    ar_path : str | Path
        Path to analysis-ready (AR) NetCDF covering Polar-6 / T-Bird window.
    tbird_path : optional str | Path
        Path to T-Bird noseboom-unified NetCDF. If provided, legs will include an
        additional platform row in `df_legs` when overlap exists.
    leg_regex : str, default r"L\\d+"
        Regular expression to identify leg keys.
    profile_regex : str, default r"MD\\d+"
        Regular expression to identify profile keys.
    profile_bin_edges : array-like, default np.arange(0, 4000, 10)
        Altitude bin edges (meters) for profile binning.
    polar6_alt_var : str, default "h"
        Altitude variable in Polar-6 dataset.
    ar_alt_var : str, default "altitude"
        Altitude variable in AR dataset.
    deduplicate_legs : bool, default True
        If True, suppress adding duplicate Polar-6 leg slices with identical
        (start, end).

    Returns
    -------
    Dict[str, object] with keys:
      - profiles:        List[xr.Dataset]  (Polar-6 binned profiles)
      - ar_profiles:     List[xr.Dataset]  (AR binned profiles)
      - legs:            List[xr.Dataset]  (Polar-6 leg time-slices; T-Bird legs appended when available)
      - ar_legs:         List[xr.Dataset]  (AR leg time-slices)
      - df_profiles:     pandas.DataFrame  (one row per profile; Polar-6 stats)
      - df_legs:         pandas.DataFrame  (one row per leg per platform)
      - days:            np.ndarray[str]   (unique days found in Polar-6)
      - patterns:        Dict[str, str]    (compiled patterns used)

    Notes
    -----
    - Leg deduplication compares Polar-6 leg start/end ISO strings. T-Bird rows
      are independent (only added if overlap returns non-empty selection).
    - Overview means are computed over the *unbinned* time slices for each segment.
    - Missing variables (e.g., `w`) are filled with NaN in the overviews.
    """
    # Open inputs
    polar6 = _open_ds(polar6_path)
    ar = _open_ds(ar_path)
    tbird = _open_ds(tbird_path) if tbird_path is not None else None

    # Load flight separation
    fp = Path(flight_sep_json).expanduser()
    logger.info("Reading flight separation JSON: %s", fp)
    with fp.open() as f:
        flight_sep_data = json.load(f)

    # Days from Polar-6 timeline
    if "time" not in polar6.coords:
        raise KeyError("Polar-6 dataset must contain a 'time' coordinate.")
    days = np.unique(polar6.time.dt.strftime("%Y-%m-%d"))

    # Outputs
    profiles: List[xr.Dataset] = []
    legs: List[xr.Dataset] = []
    ar_profiles: List[xr.Dataset] = []
    ar_legs: List[xr.Dataset] = []
    prof_rows: List[Dict[str, object]] = []
    leg_rows: List[Dict[str, object]] = []

    # Patterns
    leg_pattern = re.compile(leg_regex)
    profile_pattern = re.compile(profile_regex) # extend profile pattern by adding LYR_P\d+ if needed
    profile_pattern = re.compile(f"{profile_regex}|LYR_P\\d+")

    # For deduplication of Polar-6 legs
    seen_legs: set[SegmentKey] = set()

    logger.info("Processing %d day(s).", len(days))
    # Iterate all available days present in JSON (use intersection with Polar-6 days)
    for day in days:
        day_str = str(day)
        if day_str not in flight_sep_data:
            logger.debug("Day %s not in flight separation JSON; skipping.", day_str)
            continue
        sub_dict = flight_sep_data[day_str]
        logger.debug("Day %s: %d segment entries.", day_str, len(sub_dict))

        for key, value in sub_dict.items():
            start = value.get("start_time")
            stop = value.get("end_time")
            if not (start and stop):
                logger.debug("Segment %s missing start/stop; skipping.", key)
                continue

            # --------------------------- Profiles --------------------------- #
            if profile_pattern.search(key):
                # Polar-6 time slice and binned profile
                ds_sel = _safe_sel(polar6, start, stop)
                ds_sel['start_time'] = np.datetime64(start)
                if "time" in ds_sel and ds_sel.time.size > 0:
                    try:
                        ds_profile = _bin_profile(ds_sel, polar6_alt_var, np.asarray(profile_bin_edges))
                        ds_profile['start_time'] = np.datetime64(start)
                        profiles.append(ds_profile)
                    except Exception as e:
                        logger.debug("Binning profile (Polar-6) failed: %s", e)

                    # Overview (Polar-6 only; AR analog not included in df_profiles)
                    prof_rows.append(
                        _segment_stats(
                            ds_sel,
                            platform="polar6",
                            kind="profile",
                            start=start,
                            end=stop,
                        )
                    )
                else:
                    logger.debug("Empty Polar-6 profile slice for %s [%s, %s].", key, start, stop)

                # AR analog (binned by AR altitude)
                ds_ar_sel = _safe_sel(ar, start, stop)
                if "time" in ds_ar_sel and ds_ar_sel.time.size > 0:
                    try:
                        ds_ar_profile = _bin_profile(ds_ar_sel, ar_alt_var, np.asarray(profile_bin_edges))
                        ar_profiles.append(ds_ar_profile)
                    except Exception as e:
                        logger.debug("Binning profile (AR) failed: %s", e)

            # ----------------------------- Legs ---------------------------- #
            if leg_pattern.search(key):
                seg_key = SegmentKey(start=start, end=stop)
                ds_leg = _safe_sel(polar6, start, stop)
                if "time" in ds_leg and ds_leg.time.size > 0:
                    # Deduplicate Polar-6 legs
                    if (not deduplicate_legs) or (seg_key not in seen_legs):
                        seen_legs.add(seg_key)
                        legs.append(ds_leg)
                        # Overview for Polar-6 leg
                        leg_rows.append(
                            _segment_stats(
                                ds_leg,
                                platform="polar6",
                                kind="leg",
                                start=start,
                                end=stop,
                            )
                        )
                else:
                    logger.debug("Empty Polar-6 leg slice for %s [%s, %s].", key, start, stop)

                # AR leg analog
                ds_ar_leg = _safe_sel(ar, start, stop)
                if "time" in ds_ar_leg and ds_ar_leg.time.size > 0:
                    ar_legs.append(ds_ar_leg)
                    leg_rows.append(
                        _segment_stats(
                            ds_ar_leg,
                            platform="ar",
                            kind="leg",
                            start=start,
                            end=stop,
                        )
                    )

                # Optional T-Bird leg (added after Polar-6 leg for same interval)
                if tbird is not None:
                    ds_tbird_leg = _safe_sel(tbird, start, stop)
                    if "time" in ds_tbird_leg and ds_tbird_leg.time.size > 0:
                        legs.append(ds_tbird_leg)  # append to common legs list as in original snippet
                        leg_rows.append(
                            _segment_stats(
                                ds_tbird_leg,
                                platform="t-bird",
                                kind="leg",
                                start=start,
                                end=stop,
                            )
                        )

    # Build DataFrames
    df_profiles = pd.DataFrame(prof_rows, columns=[
        "day", "platform", "type", "start_time", "end_time",
        "N_points", "mean_altitude", "mean_T", "mean_u", "mean_v", "mean_w",
    ]).sort_values(["day", "start_time"], ignore_index=True)

    df_legs = pd.DataFrame(leg_rows, columns=[
        "day", "platform", "type", "start_time", "end_time",
        "N_points", "mean_altitude", "mean_T", "mean_u", "mean_v", "mean_w",
    ]).sort_values(["day", "start_time", "platform"], ignore_index=True)

    logger.info(
        "Complete: profiles=%d (polar6=%d, ar=%d), legs=%d (polar6/tbird list), ar_legs=%d",
        len(profiles), len(profiles), len(ar_profiles), len(legs), len(ar_legs)
    )

    return {
        "profiles": profiles,
        "ar_profiles": ar_profiles,
        "legs": legs,
        "ar_legs": ar_legs,
        "df_profiles": df_profiles,
        "df_legs": df_legs,
        "days": days,
        "patterns": {"leg": leg_regex, "profile": profile_regex},
    }

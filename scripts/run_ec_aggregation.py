#!/usr/bin/env python3
"""100 Hz T-Bird data  ->  the 5 km EC segment file (manuscript Sect. 2.1 and 3.1).

Reads
    data/observations/processed/t-bird_unified/noseboom_unified.nc    100 Hz T-Bird
    data/observations/processed/polar6_unified/kt19_unified.nc       50 Hz KT-19 surface temperature
    data/observations/processed/satellite_unified/sic_*.nc           sea-ice concentration
    data/observations/ar/analysis_ready_polar6_legs_metadata.csv     start/end time of every leg
Writes
    data/observations/ar/analysis_ready_polar6_turbulent_fluxes_latest_5km.nc

The physics is in fluxpy.aggregate (segments), fluxpy.eddy_covariance (flux and
random error) and fluxpy.qc (integrity check); this script only loads the files.

Run:  pixi run ec-segments            (about 30 s)
      pixi run verify-ec              recomputes into a temporary file and compares
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from fluxpy.aggregate import SegmentConfig, process_legs, to_xarray  # noqa: E402

OBS = REPO / "data" / "observations"
TBIRD_FILE = OBS / "processed" / "t-bird_unified" / "noseboom_unified.nc"
KT19_FILE = OBS / "processed" / "polar6_unified" / "kt19_unified.nc"
SIC_FILE = OBS / "processed" / "satellite_unified" / "sic_modis-aqua_amsr2-gcom-w1_merged_nh_1000m_APRIL.nc"
LEGS_FILE = OBS / "ar" / "analysis_ready_polar6_legs_metadata.csv"
OUTPUT_FILE = OBS / "ar" / "analysis_ready_polar6_turbulent_fluxes_latest_5km.nc"


def load_legs(ds_tbird: xr.Dataset) -> list[xr.Dataset]:
    """T-Bird time slices of the Polar 6 legs that fall into the T-Bird record."""
    legs = pd.read_csv(LEGS_FILE, sep="\t", parse_dates=["start_time", "end_time"])
    legs = legs[legs["platform"] == "polar6"]
    t_first, t_last = pd.Timestamp(ds_tbird.time.values[0]), pd.Timestamp(ds_tbird.time.values[-1])
    covered = (legs["end_time"] >= t_first) & (legs["start_time"] <= t_last)
    print(f"{covered.sum()} of {len(legs)} legs lie within the T-Bird record "
          f"({t_first.date()} to {t_last.date()})")
    return [ds_tbird.sel(time=slice(row.start_time, row.end_time))
            for row in legs[covered].itertuples()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUTPUT_FILE, help="output NetCDF file")
    parser.add_argument("--length-km", type=float, default=5.0, help="segment length")
    parser.add_argument("--min-samples", type=int, default=100, help="minimum 100 Hz samples per segment")
    parser.add_argument("--verbose", action="store_true", help="print every skipped segment")
    args = parser.parse_args()

    cfg = SegmentConfig(length_km=args.length_km, min_samples=args.min_samples, verbose=args.verbose)
    print(f"segments of {cfg.length_km} km, at least {cfg.min_samples} samples, "
          f"height range <= {cfg.max_height_range_m:.0f} m")

    ds_tbird = xr.open_dataset(TBIRD_FILE).sortby("time")
    ds_kt19 = xr.open_dataset(KT19_FILE)
    ds_sic = xr.open_dataset(SIC_FILE)
    # The product flags land and missing data with values of 101-127; as in the
    # original processing, exactly 100 % is masked as well.
    ds_sic = ds_sic.where(ds_sic["sic"] < 100)

    df = process_legs(load_legs(ds_tbird), ds_kt19, ds_sic, cfg=cfg)
    ds = to_xarray(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(args.out, mode="w")
    print(f"written {args.out}  ({ds.sizes['start_time']} segments)")


if __name__ == "__main__":
    main()

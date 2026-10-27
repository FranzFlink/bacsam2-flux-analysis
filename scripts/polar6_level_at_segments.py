#!/usr/bin/env python3
"""Polar 6 (aircraft-level) air temperature, potential temperature and height per 5 km segment.

Fig. 2a shows the potential-temperature profile surface - T-Bird - Polar 6 of every segment.
For each segment of the segment file, the Polar 6 nose-boom data (prepared from PANGAEA
995974 by scripts/prepare_inputs.py) are averaged over the segment's time window; the
potential temperature is taken from the mean temperature and the mean pressure at the
aircraft (fluxpy.util.potential_temperature).

Reads   data/observations/processed/polar6_unified/noseboom_unified.nc
        data/observations/ar/analysis_ready_polar6_turbulent_fluxes_latest_5km.nc
Writes  data/observations/ar/polar6_level_at_segments.nc

Run:  pixi run polar6-level
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from fluxpy.util import potential_temperature  # noqa: E402

OBS = REPO / "data" / "observations"
POLAR6 = OBS / "processed" / "polar6_unified" / "noseboom_unified.nc"
SEGMENTS = OBS / "ar" / "analysis_ready_polar6_turbulent_fluxes_latest_5km.nc"
OUT = OBS / "ar" / "polar6_level_at_segments.nc"

p6 = xr.open_dataset(POLAR6).sortby("time")
seg = xr.open_dataset(SEGMENTS)
T, theta, z = [], [], []
for t0, t1 in zip(seg["start_time"].values, seg["end_time"].values):
    m = p6.sel(time=slice(t0, t1)).mean()
    T.append(float(m["T"]))
    theta.append(float(potential_temperature(float(m["T"]), float(m["p"]))))
    z.append(float(m["h"]))

out = xr.Dataset(
    {"T_polar6": ("start_time", np.array(T), {"units": "degC", "long_name": "Segment-mean air temperature at the Polar 6 nose boom"}),
     "theta_polar6": ("start_time", np.array(theta), {"units": "K", "long_name": "Potential temperature from the segment-mean temperature and pressure at the Polar 6 nose boom"}),
     "z_polar6": ("start_time", np.array(z), {"units": "m", "long_name": "Segment-mean altitude of the Polar 6 aircraft"})},
    coords={"start_time": seg["start_time"].values},
    attrs={"title": "Polar 6 (aircraft-level) segment means for the BACSAM-II 5 km segments",
           "source": "Polar 6 nose boom 100 Hz, Köhler and Hartmann (2026), doi:10.1594/PANGAEA.995974",
           "history": "written by scripts/polar6_level_at_segments.py"})
out.to_netcdf(OUT, mode="w")
print(f"written {OUT.relative_to(REPO)}: {out.sizes['start_time']} segments, "
      f"{int(np.isfinite(out['theta_polar6']).sum())} with Polar 6 data")

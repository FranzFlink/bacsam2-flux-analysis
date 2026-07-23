#!/usr/bin/env python3
"""
scripts/run_ec_aggregation.py
==============================
CLI entry-point for the BACSAM-II Polar-6 EC / bulk aggregation pipeline.

Example usage
-------------
Run with defaults (5 km bins, 5000 sample min):
    python scripts/run_ec_aggregation.py

Custom bin length (2 km) and output path:
    python scripts/run_ec_aggregation.py --bin-length 2.0 --out /tmp/fluxes_2km.nc

Overlapping bins (5 km bins, 2.5 km step → 50 % overlap):
    python scripts/run_ec_aggregation.py --bin-length 5.0 --bin-overlap 2.5

Skip the bulk Richardson number (faster, e.g. for quick sensitivity runs):
    python scripts/run_ec_aggregation.py --no-ri-b --bin-length 5.0

Outputs
-------
    analysis_ready_polar6_turbulent_fluxes_latest_5km.nc  – analysis-ready xr.Dataset
    analysis_ready_polar6_legs_metadata.csv               – leg metadata table (tab-separated)

All options
-----------
    --bin-length     BIN_LENGTH_KM      default: 5.0
    --bin-overlap    BIN_OVERLAP_KM     default: 0.0
    --min-samples    MIN_SAMPLES        default: 5000
    --fs             SAMPLE_FREQ_HZ     default: 100.0
    --aerosol-zscore ZSCORE_THRESH      default: 5.0
    --no-ri-b                           disable 2-pt bulk Ri_b diagnostics (faster)
    --out            OUTPUT_NC_PATH     netCDF output path
    --out-csv        OUTPUT_CSV_PATH    leg metadata CSV path ('' to skip)
    --verbose                           print per-bin skips
    --dry-run                           load data only, skip aggregation loop

Data files are resolved relative to the project root (detected automatically
from this script's location: ``<project_root>/scripts/run_ec_aggregation.py``).
Override any path with an environment variable, e.g.
    POLAR6_PATH=/my/custom/path.nc  python scripts/run_ec_aggregation.py
"""

from __future__ import annotations

import argparse
import os
import sys

# ── Resolve project root ──────────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

# Make sure lib/ and fluxpy/ are importable
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "lib"))
sys.path.insert(0, _PROJECT_ROOT)

# ── Default data paths ────────────────────────────────────────────────────────
_OBS = os.path.join(_PROJECT_ROOT, "data", "observations", "processed")
_AR  = os.path.join(_PROJECT_ROOT, "data", "observations", "ar")

DEFAULTS = dict(
    polar6_nc    = os.environ.get("POLAR6_PATH",    os.path.join(_OBS, "polar6_unified",    "noseboom_unified.nc")),
    tbird_nc     = os.environ.get("TBIRD_PATH",     os.path.join(_OBS, "t-bird_unified",    "noseboom_unified.nc")),
    radiation_nc = os.environ.get("RADIATION_PATH", os.path.join(_OBS, "polar6_unified",    "radiation_unified.nc")),
    sic_nc       = os.environ.get("SIC_PATH",       os.path.join(_OBS, "satellite_unified", "sic_modis-aqua_amsr2-gcom-w1_merged_nh_1000m_APRIL.nc")),
    track_nc     = os.environ.get("TRACK_PATH",     os.path.join(_OBS, "polar6_unified",    "mastertracks_unified.nc")),
    ar_nc        = os.environ.get("AR_PATH",        os.path.join(_AR,                       "analysis_ready_polar6_t_bird.nc")),
    flight_json  = os.environ.get("FLIGHT_JSON",    os.path.join(_PROJECT_ROOT, "lib",      "flight_sep_BACSAM-II_MASTER.json")),
    output_nc    = os.environ.get("OUTPUT_PATH",    os.path.join(_AR, "analysis_ready_polar6_turbulent_fluxes_latest_5km.nc")),
    output_csv   = os.environ.get("OUTPUT_CSV",     os.path.join(_AR, "analysis_ready_polar6_legs_metadata.csv")),
)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BACSAM-II Polar-6 EC aggregation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bin-length",    type=float, default=5,    metavar="KM",
                   help="Spatial bin length [km]")
    p.add_argument("--bin-overlap",   type=float, default=0,    metavar="KM",
                   help="Bin overlap [km] (0 = non-overlapping)")
    p.add_argument("--min-samples",   type=int,   default=100,   metavar="N",
                   help="Minimum high-frequency samples required per bin")
    p.add_argument("--fs",            type=float, default=100.0,  metavar="HZ",
                   help="Sampling frequency of raw data [Hz]")
    p.add_argument("--aerosol-zscore", type=float, default=5.0,   metavar="Z",
                   help="Z-score threshold for aerosol outlier removal")
    p.add_argument("--no-ri-b",       action="store_true",
                   help="Skip the 2-pt bulk Richardson number and level diagnostics")
    p.add_argument("--out",           type=str,   default=DEFAULTS["output_nc"], metavar="PATH",
                   help="Output netCDF path")
    p.add_argument("--out-csv",       type=str,   default=DEFAULTS["output_csv"], metavar="PATH",
                   help="Output CSV path for leg metadata (pass '' to skip)")
    p.add_argument("--verbose",       action="store_true",
                   help="Print per-bin skipping messages")
    p.add_argument("--dry-run",       action="store_true",
                   help="Load data only; do not run aggregation loop")
    p.add_argument("--resume",        type=str, default=None, metavar="PATH",
                   help="Resume from a previous output NC or CSV; already-processed "
                        "bins (matched by start_time) are skipped")

    # Optional data path overrides
    g = p.add_argument_group("data paths (override defaults)")
    g.add_argument("--polar6",     default=DEFAULTS["polar6_nc"],    metavar="NC",  help="Polar-6 nose-boom file")
    g.add_argument("--tbird",      default=DEFAULTS["tbird_nc"],     metavar="NC",  help="T-Bird nose-boom file")
    g.add_argument("--radiation",  default=DEFAULTS["radiation_nc"], metavar="NC",  help="Radiation file")
    g.add_argument("--sic",        default=DEFAULTS["sic_nc"],       metavar="NC",  help="Sea-ice concentration file")
    g.add_argument("--track",      default=DEFAULTS["track_nc"],     metavar="NC",  help="Master track file")
    g.add_argument("--ar",         default=DEFAULTS["ar_nc"],        metavar="NC",  help="Analysis-ready (aerosol) file")
    g.add_argument("--flight-json", default=DEFAULTS["flight_json"], metavar="JSON",help="Flight separation JSON")
    g.add_argument(
        "--legs-csv", type=str, default=DEFAULTS["output_csv"], metavar="PATH",
        help="Pre-saved legs metadata CSV (default: output_csv).  If the file "
             "exists, flight re-segmentation is skipped entirely and legs are "
             "reconstructed by slicing the raw Polar-6 NC.  Pass '' to force "
             "re-running process_flight_segments.",
    )
    return p


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_legs_from_csv(
    csv_path: str,
    ds_tbird: "xr.Dataset",
) -> "tuple[list, pd.DataFrame]":
    """Fast path: reconstruct legs by slicing ds_tbird via the saved CSV.

    Returns (legs, df_legs) where legs[i] matches df_legs.iloc[i].
    Legs are sliced from T-Bird because all EC / turbulence computations
    use T-Bird data.
    """
    import pandas as pd

    # auto-detect separator
    with open(csv_path) as _f:
        first = _f.readline()
    sep = "\t" if "\t" in first else ","

    df = pd.read_csv(
        csv_path, sep=sep,
        parse_dates=["start_time", "end_time"],
    )
    # Keep only Polar-6 legs (segmentation was Polar-6 based; T-Bird flew together)
    df = df[df["platform"] == "polar6"].reset_index(drop=True)

    # Drop legs outside T-Bird coverage (T-Bird wasn't deployed for all flights)
    tb_start = pd.Timestamp(ds_tbird.time.values[0])
    tb_end   = pd.Timestamp(ds_tbird.time.values[-1])
    mask = (df["end_time"] >= tb_start) & (df["start_time"] <= tb_end)
    n_dropped = (~mask).sum()
    if n_dropped:
        print(f"  Dropping {n_dropped} leg(s) outside T-Bird coverage "
              f"({tb_start.date()} – {tb_end.date()})")
    df = df[mask].reset_index(drop=True)

    legs = [
        ds_tbird.sel(time=slice(
            pd.Timestamp(row["start_time"]),
            pd.Timestamp(row["end_time"]),
        ))
        for _, row in df.iterrows()
    ]
    return legs, df


def load_data(args) -> dict:
    """Open all required datasets and return them in a dict.

    Fast path (default): if ``--legs-csv`` points to an existing file the
    expensive ``process_flight_segments`` call is skipped entirely.  Legs are
    reconstructed by slicing the raw Polar-6 NC with the start/end timestamps
    already stored in the CSV.

    Slow path: if ``--legs-csv`` is empty or the file does not exist,
    ``process_flight_segments`` is called as before.
    """
    import pandas as pd
    import xarray as xr

    print("Loading auxiliary datasets …")
    ds_radiation = xr.open_dataset(args.radiation)
    ds_tbird     = xr.open_dataset(args.tbird)
    if "time" in ds_tbird.coords and not ds_tbird.indexes["time"].is_monotonic_increasing:
        print("  Sorting T-Bird by time (detected non-monotonic times) …")
        ds_tbird = ds_tbird.sortby("time")
    ds_polar6    = xr.open_dataset(args.polar6)
    if "time" in ds_polar6.coords and not ds_polar6.indexes["time"].is_monotonic_increasing:
        print("  Sorting Polar-6 by time (detected non-monotonic times) …")
        ds_polar6 = ds_polar6.sortby("time")
    ds_sic       = xr.open_dataset(args.sic)
    ds_sic       = ds_sic.where(ds_sic["sic"] < 100)
    ds_track     = xr.open_dataset(args.track)
    ds_ar        = xr.open_dataset(args.ar).sel(platform="t-bird")

    legs_csv = getattr(args, "legs_csv", "") or ""
    if legs_csv and os.path.isfile(legs_csv):
        print(f"Loading leg boundaries from saved CSV: {legs_csv}")
        legs, df_legs = _load_legs_from_csv(legs_csv, ds_tbird)
        print(f"  {len(df_legs)} Polar-6 legs reconstructed (T-Bird slices).")
    else:
        from helper import process_flight_segments
        if legs_csv and not os.path.isfile(legs_csv):
            print(f"  Legs CSV not found ({legs_csv}); running full flight segmentation …")
        else:
            print("Running full flight segmentation (--legs-csv not set) …")
        out_dict = process_flight_segments(
            flight_sep_json=args.flight_json,
            polar6_path=args.polar6,
            ar_path=args.ar,
            tbird_path=args.tbird,
        )
        df_legs = out_dict["df_legs"]
        df_legs = df_legs[df_legs["platform"] == "polar6"]
        # Drop legs outside T-Bird coverage
        tb_start = pd.Timestamp(ds_tbird.time.values[0])
        tb_end   = pd.Timestamp(ds_tbird.time.values[-1])
        mask = (df_legs["end_time"] >= tb_start) & (df_legs["start_time"] <= tb_end)
        n_dropped = (~mask).sum()
        if n_dropped:
            print(f"  Dropping {n_dropped} leg(s) outside T-Bird coverage "
                  f"({tb_start.date()} – {tb_end.date()})")
        df_legs = df_legs[mask]
        # Use T-Bird time slices (same time windows, T-Bird data) for EC
        # df_legs.index still holds the original positions into out_dict["legs"]
        p6_legs = [out_dict["legs"][i] for i in df_legs.index]
        df_legs = df_legs.reset_index(drop=True)
        legs = [
            ds_tbird.sel(time=slice(
                pd.Timestamp(lg.time[0].item()),
                pd.Timestamp(lg.time[-1].item()),
            ))
            for lg in p6_legs
        ]
        print(f"  {len(df_legs)} Polar-6 legs found (T-Bird slices).")

    return dict(
        legs=legs,
        df_legs=df_legs,
        ds_radiation=ds_radiation,
        ds_tbird=ds_tbird,
        ds_polar6=ds_polar6,
        ds_sic=ds_sic,
        ds_track=ds_track,
        ds_ar=ds_ar,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    # ── Import after sys.path is set ──────────────────────────────────────
    from fluxpy.aggregate import AggConfig, process_legs, to_xarray

    # ── Config ────────────────────────────────────────────────────────────
    cfg = AggConfig(
        bin_length_km=args.bin_length,
        bin_overlap_km=args.bin_overlap,
        min_samples=args.min_samples,
        fs=args.fs,
        aerosol_zscore_thresh=args.aerosol_zscore,
        compute_ri_b=not args.no_ri_b,
        verbose=args.verbose,
        resume_from=args.resume,
    )

    print(f"\n{'─'*60}")
    print(f"  BACSAM-II EC Aggregation Pipeline")
    print(f"{'─'*60}")
    print(f"  bin_length_km  : {cfg.bin_length_km} km")
    print(f"  bin_overlap_km : {cfg.bin_overlap_km} km")
    print(f"  min_samples    : {cfg.min_samples}")
    print(f"  fs             : {cfg.fs} Hz")
    print(f"  compute_ri_b   : {cfg.compute_ri_b}")
    print(f"  output         : {args.out}")
    if args.resume:
        print(f"  resume_from    : {args.resume}")
    if args.out_csv:
        print(f"  output CSV     : {args.out_csv}")
    print(f"{'─'*60}\n")

    # ── Load data ─────────────────────────────────────────────────────────
    data = load_data(args)

    # ── Save leg metadata CSV (mirrors notebook cell 4) ───────────────────
    if args.out_csv and not args.dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        data["df_legs"].to_csv(args.out_csv, index=False, header=True, sep='\t')
        print(f"Leg metadata CSV saved → {args.out_csv} ({len(data['df_legs'])} rows)")

    if args.dry_run:
        print("Dry run — data loaded successfully.  Skipping aggregation loop.")
        return

    # ── Run pipeline ──────────────────────────────────────────────────────
    df = process_legs(
        legs=data["legs"],
        df_legs_meta=data["df_legs"],
        ds_radiation=data["ds_radiation"],
        ds_tbird=data["ds_tbird"],
        ds_polar6=data["ds_polar6"],
        ds_sic=data["ds_sic"],
        ds_track=data["ds_track"],
        ds_ar=data["ds_ar"],
        cfg=cfg,
    )

    print(f"\nBuilding xarray Dataset …")
    ds = to_xarray(df)

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    ds.to_netcdf(args.out, mode="w")
    print(f"Saved → {args.out}")
    print(f"Dataset shape: {dict(ds.dims)}")


if __name__ == "__main__":
    main()

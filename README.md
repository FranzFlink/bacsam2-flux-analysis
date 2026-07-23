# BACSAM-II sensible-heat-flux analysis

Analysis code and segment-level data for **Müller et al. (2026), Atmos. Meas. Tech.**:
airborne eddy-covariance (EC) sensible heat flux over Arctic sea ice from the T-Bird
towed platform (BACSAM-II campaign, Svalbard, April 2024), compared with bulk estimates
driven by in-situ, combined satellite–reanalysis, and reanalysis-only forcings.

All physics live in the **`fluxpy`** package; notebooks only orchestrate and plot.

## Quickstart

Requires [pixi](https://pixi.sh) (pinned conda-forge environment, linux-64 + win-64):

```
pixi install
pixi run test          # unit tests (fluxpy physics, naming, Ri_b treatment)
pixi run precompute    # bulk-flux reference run + full sensitivity sweep
pixi run figures       # publication figures (see data tiers below)
```

`pixi run precompute` reproduces the paper's central results
(`data/model_output/sensitivity_figure_data.nc` and the skill summary CSV)
from the shipped segment-level inputs in a few minutes on a laptop.

## Data tiers

**Tier 1 — shipped in this repository (segment-level, ~70 MB):**

| Path | Content |
|------|---------|
| `data/observations/ar/analysis_ready_polar6_turbulent_fluxes_latest_5km.nc` | EC fluxes + in-situ meteorology on the 5 km segments (91 segments, 81 after spectral QC) |
| `data/observations/ar/along_track_inputs.nc` | Leg-colocated extractions of MODIS IST, ERA5 (model + single levels), CARRA sshf, and the T-Bird state — written by `calc_bulk_and_ec_fluxes.py --extract-only` on the raw-data host |
| `data/model_output/` | Precompute outputs (reference + sensitivity sweep) and derived diagnostics |
| `data/reanalysis/era5/ERA5_model_levels/era5_ml_along_track_P6.nc` | ERA5 model-level along-track extract |

**Tier 2 — not shipped (raw archives, ~300 GB):** 100 Hz noseboom data, full ERA5
model-level fields, CARRA rasters, MODIS L3 granules, unified radiation/SIC files.
These are only needed for the preprocessing/extraction stages (`--pregame`,
`--extract-only`) and for the map/case-study figure notebooks
(`figure01_meteo`, `figure02`, `figure04`, `figure06`, `figure07`,
`figure_meteo_alongtrack`); those steps skip cleanly when the data are absent.
Raw data are available from the authors on request and will be archived with a DOI
at submission.

## Layout

| Path | Purpose |
|------|---------|
| `fluxpy/` | Physics package: bulk fluxes, MOST stability schemes, Ri_b, naming registry |
| `scripts/calc_bulk_and_ec_fluxes.py` | Precompute: reference run + sensitivity sweeps (`--extract-only` writes the along-track cache) |
| `run_full_analysis.py` | Pipeline runner (pregame → EC aggregation → essentials → figures → tables) |
| `book/notebooks/` | Figure, table, and diagnostic notebooks (MyST/Jupyter book) |
| `tests/` | Pytest suite |

## Conventions

- Flux naming per `fluxpy/naming.py` (`H_obs_ec`, `H_obs_bulk`, `H_rs_bulk`,
  `H_era_bulk`, `H_era_diag`, `H_car_diag`); all fluxes positive upward.
- Reference configuration: GLGS20 stability, scalar roughness ratio
  α_T = z₀,T/z₀,m = 1×10⁻⁵, classic 2-pt bulk Richardson number
  (effective wind for reanalysis forcings, no analysis bound).

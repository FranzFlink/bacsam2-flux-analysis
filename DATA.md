# Data files

All paths are relative to `data/`. Units are those the code assumes when it reads the file
(most of these files carry no `units` attributes themselves).

No data file is tracked by git. `pixi run fetch-pangaea` downloads the published measurements and master
tracks from PANGAEA, `pixi run fetch-zenodo` the gridded products, the GIBS images and the leg table from the
Zenodo data record (doi:10.5281/zenodo.23729591); every other file below is written by the steps of `pixi run all`.
`data/manifest.tsv` lists every file with its size, SHA-256 checksum and origin.

## Measurements

| File | Content |
|------|---------|
| `observations/processed/t-bird_unified/noseboom_unified.nc` | 100 Hz T-Bird time series in the windows around the low-level legs. `T` air temperature (°C), `p` static pressure (hPa), `u`, `v`, `w` wind components (m s⁻¹), `tas` true airspeed (m s⁻¹), `h` altitude (m), `lat`, `lon` (°). `pitch`, `roll`, `gs` and `UTC` are carried along but not used. Cut by `pixi run inputs` (`scripts/prepare_inputs.py`) from the published dataset of Köhler and Hartmann (2026), doi:10.1594/PANGAEA.995975, whose files `pixi run fetch-pangaea` downloads into `pangaea/995975/`. Not in git; the first steps of `pixi run all` write it. |
| `observations/processed/polar6_unified/kt19_unified.nc` | 50 Hz KT-19 surface brightness temperature `KT19` (°C) with the Polar 6 position `latitude`, `longitude`, from 1 h before the first to 1 h after the last leg of every flight day. Cut by `pixi run inputs` from the published dataset of Köhler and Jurányi (2026), doi:10.1594/PANGAEA.995980, whose files `pixi run fetch-pangaea` downloads into `pangaea/995980/`. Not in git; the first steps of `pixi run all` write it. |
| `observations/processed/polar6_unified/noseboom_unified.nc` | 100 Hz Polar 6 nose-boom air temperature `T` (°C), static pressure `p` (hPa) and altitude `h` (m) in the windows around the low-level legs. Cut by `pixi run inputs` (`scripts/prepare_inputs.py`) from the published dataset of Köhler and Hartmann (2026), doi:10.1594/PANGAEA.995974, whose files `pixi run fetch-pangaea` downloads into `pangaea/995974/`. Not in git; the first steps of `pixi run all` write it. |
| `observations/raw/polar6/mastertracks/*.tab` | Polar 6 master tracks of all research flights (flight paths in Fig. 1), one PANGAEA dataset per flight in the series of Jurányi et al. (2024), doi:10.1594/PANGAEA.971763, downloaded by `pixi run fetch-pangaea`. |
| `observations/ar/analysis_ready_polar6_legs_metadata.csv` | Start and end time of every straight flight leg, with its number of samples, mean altitude, temperature and wind. Cut once from the Polar 6 archive by `preprocessing/flight_segments.py`; from the Zenodo data record (`pixi run fetch-zenodo`). |

## Segment level (5 km)

| File | Content |
|------|---------|
| `observations/ar/analysis_ready_polar6_turbulent_fluxes_latest_5km.nc` | One row per 5 km segment (156 segments of all heights flown at one level; the paper uses the 91 below 50 m over sea ice). EC sensible heat flux `sensible_heat_flux` (W m⁻², positive upward), its random error `sensible_heat_flux_random_error` (W m⁻², 1σ) and the ratio `flux_over_random_error` (the paper counts the fluxes above 1, 2 and 3; all fluxes are kept), the integrity check `turbulence_resolved` with the spectral slope `qc_slope_w` (a diagnostic; no segment is excluded by it); the T-Bird segment means `T_mean` (°C), `p_mean` (hPa), `u_mean`, `v_mean` (m s⁻¹), `altitude` (m), `latitude`, `longitude`; the KT-19 surface temperature `t_kt19` (°C) with its within-segment standard deviation `T_sfc_std`, the surface-type fractions `frac_open`, `frac_thin_ice`, `frac_thick_ice`, `lead_frac` and the sea-ice concentration `sic`; `N_samples`, `segment_length_km`, `end_time`. Written by `scripts/run_ec_aggregation.py` from the 100 Hz file above (`pixi run ec-segments`). Every variable carries units and a long name. |
| `observations/ar/ec_segment_placement.nc` | Segment-placement spread of the EC flux (5 km window moved in 0.05 km steps), the empirical cross-check of the random error, written by `scripts/ec_segment_placement.py`. |
| `observations/ar/along_track_inputs.nc` | Values of the gridded products at each of the 91 segments: MODIS ice surface temperature, ERA5 model-level temperature/wind/height, ERA5 skin temperature, surface pressure and surface sensible heat flux, CARRA surface sensible heat flux. Written by `pixi run extract` (`scripts/calc_bulk_and_ec_fluxes.py --extract-only`) from the gridded files below. With this file present the bulk computation does not open the gridded files. |
| `observations/ar/ec_leg_length_sensitivity/ec_sensitivity_summary.csv` | Result of the EC window-length experiment (Fig. 3), written by `scripts/ec_window_length.py`. |
| `observations/ar/polar6_level_at_segments.nc` | Polar 6 aircraft-level air temperature `T_polar6` (°C), potential temperature `theta_polar6` (K) and height `z_polar6` (m) per segment, for the profiles of Fig. 2a: means over the segment's time window, θ from the mean temperature and pressure. Written by `pixi run polar6-level` from the Polar 6 nose-boom file above. |

## Gridded products (campaign region and period only; from the Zenodo data record, `pixi run fetch-zenodo`)

| File | Product |
|------|---------|
| `reanalysis/era5/ERA5_model_levels/era5_ml.nc` | ERA5 model levels (Hersbach et al., 2020), Copernicus Climate Change Service |
| `reanalysis/era5/ERA5_single_levels/*.nc` | ERA5 single levels, instantaneous and time-averaged fields |
| `reanalysis/carra/carra_east_latlon0p05.nc` | CARRA (Copernicus Arctic Regional Reanalysis, east domain) skin temperature and surface pressure, regridded to 0.05°; used for Fig. 2 only |
| `reanalysis/carra/height_level/carra_latlon0p05.nc` | CARRA temperature, wind speed and pressure at 30 and 75 m above ground, regridded to 0.05°; used for Fig. 2 only |
| `reanalysis/carra/carra_east_forecast_sf_latlon0p05.nc` | CARRA forecast surface sensible heat flux `sshf` (hourly accumulation, J m⁻²), regridded to 0.05°, on the flight days 13, 15, 17, 23 and 25 April 2024 within 74–86° N, 20° W–30° E; the model-diagnosed CARRA flux |
| `observations/processed/satellite_unified/MODIS_Aqua_Sea_Ice_Extent_and_IST_Daily_L3_Global_4km_EASE-Grid_Day_V061.nc` | MODIS/Aqua Sea Ice Extent and IST Daily L3 Global 4 km EASE-Grid Day, version 61 (MYD29E1D; reference: see the paper), one grid per UTC day, 12–24 April 2024, reprojected onto a 0.063° latitude–longitude grid by `preprocessing/build_modis_ist.py` (granule names and checksums in the file attributes). Each segment takes the grid of its own date. |
| `observations/processed/satellite_unified/sic_modis-aqua_amsr2-gcom-w1_merged_nh_1000m_APRIL.nc` | Merged MODIS–AMSR2 sea-ice concentration, 1 km (Ludwig et al., 2020), provided by the University of Bremen |
| `../figures_manuscript/_gibs_cache/*.png` | MODIS imagery from NASA GIBS / Worldview (background of Fig. 5) |

These are subsets of third-party products and remain subject to the licences of their
providers; please cite the original products, not this repository, when using them.

## Output

| File | Content |
|------|---------|
| `model_output/sensitivity_figure_data.nc` | All flux estimates, inputs and perturbation experiments on the 91 segments (self-describing: units and long names on every variable) |
| `model_output/sensitivity_summary_table.csv` | Skill of every sensitivity configuration against the EC reference |

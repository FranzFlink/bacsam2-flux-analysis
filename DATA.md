# Data files

All paths are relative to `data/`. Units are those the code assumes when it reads the file
(most shipped files carry no `units` attributes themselves).

## Measurements

| File | Content |
|------|---------|
| `observations/processed/t-bird_unified/noseboom_unified.nc` | 100 Hz T-Bird nose-boom time series, restricted to the time windows of the flux legs. `T` air temperature (°C), `p` static pressure (hPa), `u`, `v`, `w` wind components (m s⁻¹), `tas` true airspeed (m s⁻¹), `h` height (m), `lat`, `lon` (°). `pitch`, `roll`, `gs` and `UTC` are carried along but not used. |
| `observations/processed/polar6_unified/radiation_unified.nc` | Polar 6 broadband radiation and KT-19 surface brightness temperature for the same windows. `F_*` irradiances (W m⁻²). **`KT19` is in °C** — its `units` attribute says "K", which is wrong; the code adds 273.15. |
| `observations/raw/polar6/mastertracks/*.tab` | Polar 6 master tracks of all research flights (flight paths in Fig. 1). |
| `observations/ar/analysis_ready_polar6_legs_metadata.csv` | Start and end time of every straight flight leg. |

## Segment level (5 km)

| File | Content |
|------|---------|
| `observations/ar/analysis_ready_polar6_turbulent_fluxes_latest_5km.nc` | One row per 5 km segment: EC sensible heat flux `sensible_heat_flux` (W m⁻², positive upward; NaN if discarded by the spectral QC), segment means `T_mean` (°C), `p_mean` (hPa), `u_mean`, `v_mean`, `w_mean` (m s⁻¹), `altitude` (m), `t_kt19` (°C), surface-type fractions `frac_open`, `frac_thin_ice`, `frac_thick_ice`, QC metrics `qc_slope_w`, `qc_og_fft_plateau_err`, `qc_R2`, `qc_snr_fwC` and verdict `qc_verdict_simple_int` (0 accepted, 1 tolerable, 2 discarded), `lenschow_rel_error` (relative sampling error), `L_int` (m), `u_star` (m s⁻¹), `monin_obukhov_length` (m). Written by `scripts/run_ec_aggregation.py` on the raw-data host; the `bulk3_*_polar6` variables are the aircraft-level values shown in Fig. 2a. `pixi run verify-ec` recomputes the EC flux and the QC verdict from the 100 Hz file above. The aerosol columns are not used in the paper. |
| `observations/ar/analysis_ready_polar6_turbulent_fluxes_with_bins.nc` | Same processing with a 5 km window moved in 0.05 km steps; used only for the segment-placement uncertainty of the EC flux. |
| `observations/ar/along_track_inputs.nc` | Values of the gridded products at each of the 91 segments: MODIS ice surface temperature, ERA5 model-level temperature/wind/height, ERA5 skin temperature, surface pressure and surface sensible heat flux, CARRA surface sensible heat flux. Written by `scripts/calc_bulk_and_ec_fluxes.py --extract-only`. With this file present the bulk computation does not open the gridded files. |
| `observations/ar/ec_leg_length_sensitivity/ec_sensitivity_summary.csv` | Result of the EC window-length experiment (Fig. 3), from `preprocessing/ec_leg_length_sensitivity.ipynb`. |

## Gridded products (campaign region and period only)

| File | Product |
|------|---------|
| `reanalysis/era5/ERA5_model_levels/era5_ml.nc` | ERA5 model levels (Hersbach et al., 2020), Copernicus Climate Change Service |
| `reanalysis/era5/ERA5_single_levels/*.nc` | ERA5 single levels, instantaneous and time-averaged fields |
| `reanalysis/carra/*.nc`, `reanalysis/carra/height_level/*.nc` | CARRA (Copernicus Arctic Regional Reanalysis), Copernicus Climate Change Service |
| `observations/processed/satellite_unified/MODIS_Aqua_Sea_Ice_Extent_and_IST_Daily_L3_Global_4km_EASE-Grid_Day_V061.nc` | MODIS/Aqua Sea Ice Extent and IST Daily L3 Global 4 km EASE-Grid Day, version 61 (reference: see the paper) |
| `observations/processed/satellite_unified/sic_modis-aqua_amsr2-gcom-w1_merged_nh_1000m_APRIL.nc` | Merged MODIS–AMSR2 sea-ice concentration, 1 km (Ludwig et al., 2020), provided by the University of Bremen |
| `../figures_manuscript/_gibs_cache/*.png` | MODIS imagery from NASA GIBS / Worldview (background of Fig. 5) |

These are subsets of third-party products and remain subject to the licences of their
providers; please cite the original products, not this repository, when using them.

## Output

| File | Content |
|------|---------|
| `model_output/sensitivity_figure_data.nc` | All flux estimates, inputs and perturbation experiments on the 91 segments (self-describing: units and long names on every variable) |
| `model_output/sensitivity_summary_table.csv` | Skill of every sensitivity configuration against the EC reference |

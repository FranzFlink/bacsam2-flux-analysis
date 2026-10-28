# preprocessing — how the shipped data files were made

These notebooks and scripts document how the raw archives were turned into the files under
`data/`. They ran on the institute's data server and need the raw data (not shipped); file
paths inside them refer to that server. They are **not** part of `pixi run all`. Stored cell
outputs were removed.

The T-Bird, Polar 6 nose-boom and KT-19 inputs are no longer made here: `scripts/prepare_inputs.py`
(`pixi run inputs`) cuts them from the published PANGAEA datasets 995975, 995974 and 995980. The two
entries marked *superseded* below document the earlier inputs, converted from the ASCII files of the raw-data
server.

| File | Produces |
|------|----------|
| `preprocess_polar6_noseboom.ipynb` | *superseded:* unified 100 Hz nose-boom time series (`noseboom_unified.nc`) from the ASCII files |
| `preprocess_mastertracks.ipynb` | unified master track |
| `preprocess_sic.ipynb` | merged sea-ice concentration file |
| `build_modis_ist.py` | MODIS/Aqua ice surface temperature (MYD29E1D) of 12–24 April 2024 from the NSIDC granules; replaces the earlier MODIS file, which was made from Terra granules (MOD29E1D) ordered on 23 April 2024 and lacked 16, 17, 19, 20 and 23 April |
| `preprocess_carra.ipynb`, `carra_write_subset.py` | CARRA crops |
| `ERA5_STN_dl.py` | ERA5 download (Copernicus Climate Data Store; needs your own CDS account) |
| `make_reviewer_subsets.py` | cuts the raw archives down to the campaign region and period shipped here (gridded products); its time-series part (nose boom, radiation) and its MODIS ice-surface-temperature part are *superseded* |
| `flight_segments.py` | how the leg table `analysis_ready_polar6_legs_metadata.csv` was cut from the flight-separation file `config/flight_sep_BACSAM-II_MASTER.json` and the full Polar 6 archive |

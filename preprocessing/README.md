# preprocessing — how the shipped data files were made

These notebooks and scripts document how the raw archives were turned into the files under
`data/`. They ran on the institute's data server and need the raw data (not shipped); file
paths inside them refer to that server. They are **not** part of `pixi run all`. Stored cell
outputs were removed.

| File | Produces |
|------|----------|
| `preprocess_polar6_noseboom.ipynb` | unified 100 Hz nose-boom time series (`noseboom_unified.nc`) |
| `preprocess_mastertracks.ipynb` | unified master track |
| `preprocess_sic.ipynb` | merged sea-ice concentration file |
| `preprocess_carra.ipynb`, `carra_write_subset.py` | CARRA crops |
| `ERA5_STN_dl.py` | ERA5 download (Copernicus Climate Data Store; needs your own CDS account) |
| `make_reviewer_subsets.py` | cuts the raw archives down to the campaign region, period and flux-leg time windows shipped here |
| `flight_segments.py` | how the leg table `analysis_ready_polar6_legs_metadata.csv` was cut from the flight-separation file `config/flight_sep_BACSAM-II_MASTER.json` and the full Polar 6 archive |

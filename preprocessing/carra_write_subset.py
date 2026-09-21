import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster

if __name__ == "__main__":

    cluster = LocalCluster(
        n_workers=32,
        threads_per_worker=1,
        memory_limit="16GB",
        dashboard_address=":8708",
    )
    client = Client(cluster)

    # 1) Open with auto-chunking to avoid the “separate stored chunks” warnings
    ds = xr.open_dataset(
        "/projekt_agmwend/home_rad/Joshua/CARRA/CARRA_March_April_2024_full_model.nc",
        chunks="auto",
    ).sel(valid_time="2024-04")

    # 2) Make a proper mask DataArray with (y, x) dims
    lat = ds["latitude"]
    lon = ds["longitude"] #
    # reformat longitude to -180 to 180 if needed
    lon = ((lon + 180) % 360) - 180
    mask = (lat >= 75) & (lat <= 82) & (lon >= -30) & (lon <= 30)

    # 3) Apply mask with drop=True so coords get cropped too
    ds_sub = ds.where(mask)
    ds_sub = ds_sub.dropna(dim="y", how="all")
    ds_sub = ds_sub.dropna(dim="x", how="all")

    # 4) Clean up chunking: after drop, unify and set friendly chunks
    ds_sub = ds_sub.unify_chunks()
    # (Optional) keep time chunked; make spatial unchunked or modest
    ds_sub = ds_sub.chunk({"valid_time": 100, "y": -1, "x": -1})

    # 5) Avoid fragile encodings on 2-D coordinate variables
    for coord in ["latitude", "longitude"]:
        if coord in ds_sub:
            ds_sub[coord].encoding.clear()  # removes stale chunksizes/compression
            ds_sub[coord].encoding.update({"_FillValue": None})  # coords shouldn’t have fill

    # (Optional) also clear encodings on data vars if you had custom ones before
    # for v in ds_sub.data_vars:
    #     ds_sub[v].encoding.pop("chunksizes", None)

    # 6) Write
    out = "/projekt_agmwend/home_rad/Joshua/BACSAM-II/data/reanalysis/carra/height_level/carra_east_subset.nc"
    ds_sub.to_netcdf(out, mode="w")

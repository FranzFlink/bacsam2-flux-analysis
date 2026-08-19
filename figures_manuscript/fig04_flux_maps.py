"""Fig. 4 (fig:flux_map_overview) — four low-level measurement-area maps.

2x2 North-Polar-Stereo panels, one per low-level flight day (13/15/17/23
April 2024): MODIS/AMSR-2 daily sea-ice concentration as background, 5 km
EC segments coloured by H_S,obs,EC (positive = upward), ERA5-level wind
barbs, and a coloured frame matching the campaign-map focus rectangles.
Ported from book/notebooks/figures/figure04.ipynb.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fluxpy import figstyle as fs

fs.apply()

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cmcrameri import cm as cm_crameri
from cmocean import cm as cm_ocean

# ── Data ────────────────────────────────────────────────────────────────────
DATA = REPO / "data" / "observations"
ds = xr.open_dataset(DATA / "ar" / "analysis_ready_polar6_turbulent_fluxes_latest_5km.nc")
ds_sic = (
    xr.open_dataset(DATA / "processed" / "satellite_unified" /
                    "sic_modis-aqua_amsr2-gcom-w1_merged_nh_1000m_APRIL.nc")
    .squeeze()
    .sortby("time")
    .drop_sel(time="2024-04-25")   # 25 Apr: 100 % SIC everywhere -> bad data
)

low = ds.where(ds.altitude <= 50, drop=True)   # T-Bird low-level legs
low_alt_days = np.unique(low.start_time.dt.strftime("%Y-%m-%d").values)

# colocate the daily SIC map onto the segments (drives the marker edge colour)
sic_pts = (
    ds_sic.rename(time="start_time")
    .sel(start_time=low.start_time, lat=low.latitude, lon=low.longitude,
         method="nearest")
)
low["sic"] = ("start_time",), sic_pts.sic.values
low["sic"] = low.sic.where(low.sic <= 100)     # mask invalid SIC

# drop the spurious first 13-Apr segment
first_ts_0413 = low.sel(start_time="2024-04-13").start_time.min().values
low = low.where(low.start_time != first_ts_0413, drop=True)

# frame colours = campaign-map focus-rectangle colours (batlow over all days)
frame_colors = dict(zip(low_alt_days,
                        cm_crameri.batlow(np.linspace(0, 1, len(low_alt_days)))))
detail_days = [d for d in low_alt_days if d != "2024-04-25"]

# ── Figure ──────────────────────────────────────────────────────────────────
# MID_W tried first: the fixed-point-size wind barbs then overlap and hide the
# flux-coloured markers in (a)/(b) -> full text width, as the notebook original.
WIDTH = fs.FULL_W
fig = plt.figure(figsize=(WIDTH, WIDTH * 0.82))
fig.get_layout_engine().set(hspace=0.06)   # room for the row-2 panel letters
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.045])
axs = [fig.add_subplot(gs[i // 2, i % 2], projection=ccrs.NorthPolarStereo())
       for i in range(4)]

sic_norm = mpl.colors.Normalize(0, 100)
flux_norm = mpl.colors.TwoSlopeNorm(vmin=-15, vcenter=0, vmax=15)

for ax, day, label in zip(axs, detail_days, ["(a)", "(b)", "(c)", "(d)"]):
    ax.set_aspect("auto")          # fill the grid cell (as in the notebook)

    sic = ds_sic.sel(time=day, method="nearest").sic
    im_sic = sic.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(),
                                 cmap=cm_ocean.ice, norm=sic_norm,
                                 add_colorbar=False, rasterized=True)
    ax.add_feature(cfeature.LAND, zorder=1, facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE, zorder=2, linewidth=0.4)

    ds_day = low.sel(start_time=day)
    x, y = ds_day.longitude.values, ds_day.latitude.values

    x0, x1 = np.nanmin(x) - 1.0, np.nanmax(x) + 1.0    # lon buffer
    y0, y1 = np.nanmin(y) - 0.12, np.nanmax(y) + 0.12  # lat buffer
    ax.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())

    edgecolors = np.array(["w" if v < 50 else "k" for v in ds_day.sic.values])
    im_flux = ax.scatter(x, y, c=ds_day.sensible_heat_flux.values, s=30,
                         linewidths=0.5, edgecolor=edgecolors,
                         cmap=cm_ocean.balance, norm=flux_norm,
                         transform=ccrs.PlateCarree(), zorder=3)

    barbs = ax.barbs(x[::2], y[::2],
                     ds_day.u_mean.values[::2], ds_day.v_mean.values[::2],
                     transform=ccrs.PlateCarree(), length=5.5, linewidth=0.9,
                     barb_increments=dict(half=2.5, full=5, flag=25),
                     color="white", zorder=5)
    barbs.set_path_effects([pe.Stroke(linewidth=2.2, foreground="black"),
                            pe.Normal()])

    # coloured frame, same colour as the campaign-map focus rectangle
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
            color=frame_colors[day], linewidth=3,
            transform=ccrs.PlateCarree(), zorder=6)

    ax.spines["geo"].set_visible(False)
    ax.set_title("")
    fs.panel_label(ax, label, x=0.0, y=1.01)

# stacked equal-height colourbars (flux above, SIC below)
cbar = fig.colorbar(im_flux, cax=fig.add_subplot(gs[0, 2]),
                    ticks=np.arange(-15, 16, 5))
cbar.set_label(f"{fs.L['H_obs_ec']} {fs.FLUX_UNIT}")
cbar_sic = fig.colorbar(im_sic, cax=fig.add_subplot(gs[1, 2]))
cbar_sic.set_label("Sea-ice concentration (%)")

fs.save(fig, "fig04_flux_maps")

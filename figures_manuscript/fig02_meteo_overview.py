"""Fig. 2 — meteorological overview (manuscript label ``fig:bulk_methods_overview``).

(a) Per-flight-day potential-temperature profiles (surface / T-Bird / Polar 6)
    from observations, ERA5, ERA5-MODIS, and CARRA;
(b) surface-temperature histograms per data source;
(c) near-surface stratification (theta_30m - theta_sfc) histograms.

Ported from book/notebooks/figures/figure01_meteo.ipynb (cells 1-3 only).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import logging

logging.getLogger("fontTools").setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from fluxpy import figstyle as fs

fs.apply()

sys.path.insert(0, str(REPO / "scripts"))
from calc_bulk_and_ec_fluxes import load_carra  # noqa: E402

# ── Semantic colours (single source: figstyle) ───────────────────────────────
C_OBS = fs.C["H_obs_bulk"]   # Polar-6 KT-19 / in-situ
C_MOD = fs.C["H_rs_bulk"]    # ERA5-MODIS surface temperature
C_ERA = fs.C["H_era_diag"]   # standard ERA5 skin temperature
C_CAR = fs.C["H_car_diag"]   # standard CARRA skin temperature

EXCLUDE = [pd.Timestamp("2024-04-25").date(), pd.Timestamp("2024-04-26").date()]

# ═════════════════════════════════════════════════════════════════════════════
# Data loading & colocation
# ═════════════════════════════════════════════════════════════════════════════
_AR = REPO / "data" / "observations" / "ar"
_SAT = REPO / "data" / "observations" / "processed" / "satellite_unified"
_E5_SL = REPO / "data" / "reanalysis" / "era5" / "ERA5_single_levels"
_E5_ML = REPO / "data" / "reanalysis" / "era5" / "ERA5_model_levels"

ds_fig = xr.open_dataset(REPO / "data" / "model_output" / "sensitivity_figure_data.nc")

# P6 low-level legs (altitude < 50 m, exclude high-level-only days)
ds_obs_full = xr.open_dataset(_AR / "analysis_ready_polar6_turbulent_fluxes_latest_5km.nc")
_times_full = pd.DatetimeIndex(ds_obs_full["start_time"].values)
_mask = (np.asarray(ds_obs_full["altitude"]) < 50) & ~np.isin(_times_full.date, EXCLUDE)
ds_obs = ds_obs_full.sel(start_time=ds_obs_full["start_time"][_mask])

T_kt19 = np.asarray(ds_obs["t_kt19"]).astype(float)          # KT-19 surface T [degC]
start_times = pd.DatetimeIndex(ds_obs["start_time"].values)
flight_days = sorted(set(start_times.date))


def _ar_col(ds, name):
    """Archive column with legacy fallback (older exports used a bulk3_ prefix)."""
    key = name if name in ds else "bulk3_" + name
    return np.asarray(ds[key]).astype(float)


z_sfc = _ar_col(ds_obs, "z_surface")
z_tb = _ar_col(ds_obs, "z_tbird")
z_p6 = _ar_col(ds_obs, "z_polar6")
th_sfc = _ar_col(ds_obs, "theta_surface")   # K
th_tb = _ar_col(ds_obs, "theta_tbird")
th_p6 = _ar_col(ds_obs, "theta_polar6")

# Leg-colocation index arrays (times from the sensitivity file, positions from P6)
_t_da = xr.DataArray(pd.to_datetime(ds_fig["start_time"].values), dims="leg")
_lat_da = xr.DataArray(np.asarray(ds_obs["latitude"]), dims="leg")
_lon_da = xr.DataArray(np.asarray(ds_obs["longitude"]), dims="leg")

# ERA5 single levels: skin temperature + surface pressure
_ds_sl = xr.open_dataset(_E5_SL / "data_stream-oper_stepType-instant.nc")
_kw_sl = dict(valid_time=_t_da, latitude=_lat_da, longitude=_lon_da, method="nearest")
skt_era5_K = _ds_sl["skt"].sel(**_kw_sl).values.astype(float)
p_sfc_hPa = _ds_sl["sp"].sel(**_kw_sl).values.astype(float) / 100.0

# MODIS IST: nearest-neighbour colocation, physical-range mask, ERA5 gap-fill
ds_modis = xr.open_dataset(
    _SAT / "MODIS_Aqua_Sea_Ice_Extent_and_IST_Daily_L3_Global_4km_EASE-Grid_Day_V061.nc"
)
if "time" in ds_modis.dims:
    ds_modis = ds_modis.rename({"time": "start_time"})
ist_K = (
    ds_modis["IST"]
    .sel(lat=ds_obs["latitude"], lon=ds_obs["longitude"],
         start_time=ds_obs["start_time"], method="nearest")
    .values.astype(float)
)
ist_K = np.where((ist_K > 150.0) & (ist_K < 280.0), ist_K, np.nan)  # cloud/fill -> NaN
ist_K = np.where(np.isnan(ist_K), skt_era5_K, ist_K)                # ERA5 gap-fill

# ERA5 model levels 137-134 (~10/31/54/79 m)
era5_ml = xr.open_dataset(_E5_ML / "era5_ml.nc").sel(
    time=_t_da, latitude=_lat_da, longitude=_lon_da, method="nearest"
)

# ═════════════════════════════════════════════════════════════════════════════
# Potential temperatures
# ═════════════════════════════════════════════════════════════════════════════
def pot_temp_K(T_K, p_hPa):
    return T_K * (1000.0 / p_hPa) ** 0.286


# ERA5 model-level pressures from surface pressure (hybrid-coefficient ratios)
_ML_SIGMA = {137: 0.998815, 136: 0.996317, 135: 0.993618, 134: 0.990448}
theta_ml = {
    lvl: pot_temp_K(era5_ml["t"].sel(model_level=lvl).values, sig * p_sfc_hPa)
    for lvl, sig in _ML_SIGMA.items()
}
theta_sfc_era5 = pot_temp_K(skt_era5_K, p_sfc_hPa)
theta_sfc_modis = pot_temp_K(ist_K, p_sfc_hPa)

# CARRA (surface / 30 m / 75 m)
carra = load_carra(ds_obs)
theta_sfc_carra = pot_temp_K(carra["skt_K"], carra["sp_Pa"] / 100.0)
theta_30_carra = pot_temp_K(carra["T30_K"], carra["p30_Pa"] / 100.0)
theta_75_carra = pot_temp_K(carra["T75_K"], carra["p75_Pa"] / 100.0)

# Histogram series: surface temperature [degC] and stratification [K]
T_hist = [
    (T_kt19, C_OBS),
    (ist_K - 273.15, C_MOD),
    (skt_era5_K - 273.15, C_ERA),
    (carra["skt_K"] - 273.15, C_CAR),
]
dth_hist = [
    (th_tb - th_sfc, C_OBS),
    (theta_ml[136] - theta_sfc_modis, C_MOD),
    (theta_ml[136] - theta_sfc_era5, C_ERA),
    (theta_30_carra - theta_sfc_carra, C_CAR),
]

# ═════════════════════════════════════════════════════════════════════════════
# Figure: 4 profile columns (a) on top, histograms (b) and (c) stacked below
# (single column: side-by-side histograms would crush the region labels)
# ═════════════════════════════════════════════════════════════════════════════
fig = fs.gridfig(width=fs.COL_W, aspect=1.75)
gs = fig.add_gridspec(3, 4, height_ratios=[2.1, 1, 1])

axs_prof = []
for i in range(4):
    axs_prof.append(fig.add_subplot(gs[0, i], sharey=axs_prof[0] if i else None))
ax_tsfc = fig.add_subplot(gs[1, :])
ax_dth = fig.add_subplot(gs[2, :])

# ── (a) Three-point theta(z) profiles per flight day ─────────────────────────
_ylim_top = 100
_ok = (z_p6 > 0) & (z_p6 < 200) & (z_tb > 0) & (z_tb < 200)
_ok &= np.isfinite(th_sfc) & np.isfinite(th_tb) & np.isfinite(th_p6)
z_tb_mean, z_p6_mean = np.nanmean(z_tb[_ok]), np.nanmean(z_p6[_ok])

Z_ERA = [0, 10, 31, 54, 79]
# (colour, theta columns, indiv z (fixed | per-leg columns), median z (fixed |
#  None = per-leg median), marker, indiv ls, median ls, zorder).  NB: as in the
# notebook, MODIS individual profiles stop at 10 m but the median is drawn to 30 m.
prof_sources = [
    (C_OBS, [th_sfc, th_tb, th_p6], [z_sfc, z_tb, z_p6], None, "o", "-", "--", 4),
    (C_ERA, [theta_sfc_era5] + [theta_ml[l] for l in (137, 136, 135, 134)],
     Z_ERA, Z_ERA, "^", "-", "--", 3),
    (C_MOD, [theta_sfc_modis, theta_ml[137]], [0, 10], [0, 30], "d", ":", ":", 2),
    (C_CAR, [theta_sfc_carra, theta_30_carra, theta_75_carra],
     [0, 30, 75], [0, 30, 75], "s", "-", "--", 2.5),
]
x_lims = [(240, 255), (240, 260), (245, 260), (265, 275)]

for i_day, (ax, day) in enumerate(zip(axs_prof, flight_days)):
    ax.axhline(z_tb_mean, color="0.75", lw=0.6, ls="--", zorder=0)
    ax.axhline(z_p6_mean, color="0.75", lw=0.6, ls="--", zorder=0)

    _dm = _ok & (start_times.date == day)
    for c, th_cols, z_ind, z_med, mk, ls_i, ls_m, zo in prof_sources:
        _per_leg_z = not isinstance(z_ind[0], (int, float))
        # individual leg profiles
        for j in np.where(_dm)[0]:
            z_j = [z[j] for z in z_ind] if _per_leg_z else z_ind
            ax.plot([th[j] for th in th_cols], z_j, color=c, lw=0.5, ls=ls_i,
                    alpha=0.1, zorder=1)
        # daily median profile
        th_med = [np.nanmedian(th[_dm]) for th in th_cols]
        if z_med is None:
            z_med = [np.nanmedian(z[_dm]) for z in z_ind]
        ax.plot(th_med, z_med, marker=mk, color=c, lw=1.6, ls=ls_m, alpha=0.9,
                markeredgewidth=0.25, markeredgecolor="0.3", markerfacecolor=c,
                zorder=zo)

    ax.set_ylim(0, _ylim_top)
    ax.set_xlim(*x_lims[i_day])
    ax.set_xlabel(r"$\theta$ (K)")
    ax.set_title(pd.Timestamp(day).strftime("%d %b"), fontweight="bold")
    # major (labelled) ticks every 10 K, minor every 5 K; blank a label that
    # falls on the shared column edge so adjacent columns do not collide
    x0, x1 = x_lims[i_day]
    _major = np.arange(x0, x1 + 1, 10)
    ax.set_xticks(_major)
    ax.set_xticks(np.arange(x0, x1 + 1, 5), minor=True)
    _last = i_day < 3  # last column has free space to its right
    ax.set_xticklabels(["" if (_last and t == x1) else f"{t:.0f}" for t in _major])
    if i_day > 0:
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)

axs_prof[0].set_ylabel("Height (m)")
# one legend row above the whole figure (an in-panel box would cover the
# ~1.6 cm wide profile columns at single-column width)
fig.legend(
    handles=[
        Patch(facecolor=C_OBS, edgecolor="0.3", lw=0.4, label="Obs"),
        Patch(facecolor=C_MOD, edgecolor="0.3", lw=0.4, label="MODIS"),
        Patch(facecolor=C_ERA, edgecolor="0.3", lw=0.4, label="ERA5"),
        Patch(facecolor=C_CAR, edgecolor="0.3", lw=0.4, label="CARRA"),
    ],
    loc="outside upper center", ncol=4, columnspacing=1.0, handlelength=1.1,
    handletextpad=0.4,
)
fs.panel_label(axs_prof[0], "(a)")

# flight-level annotations inside the first column, on the dashed height lines
# (there is no margin room left of the axis at single-column width)
for _z, _txt, _dy in [(z_tb_mean, "T-Bird", 4), (z_p6_mean, "Polar 6", 4)]:
    axs_prof[0].text(0.06, (_z + _dy) / _ylim_top, _txt,
                     transform=axs_prof[0].transAxes, va="bottom", ha="left",
                     color="0.5", fontsize=fs.FONT_SIZE - 1, zorder=5,
                     bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))

# ── (b, c) Histograms: surface temperature and stratification ────────────────
_kw_fill = dict(density=True, alpha=0.2, linewidth=0, zorder=1)
_kw_step = dict(density=True, histtype="step", linewidth=0.8, alpha=0.9, zorder=2)

# (axis, series, bins, xlim, vlines, shaded bands, region labels, xlabel, panel)
hist_panels = [
    (ax_tsfc, T_hist, np.linspace(-35, 5, 20), (-35, 5), [-1.8, -20],
     [(-1.8, 5, "#fffde7"), (-20, -1.8, "#e3f2fd"), (-35, -20, "#e8eaf6")],
     [(-27.5, "snow-covered\nsea-ice"), (-10.9, "thin\nsea-ice"), (1.6, "sea-ice\nfree")],
     r"$T_\mathrm{sfc}$ (°C)", "(b)"),
    (ax_dth, dth_hist, np.linspace(-6, 6, 20), (-8, 8), [0],
     [(-8, 0, "#fff3e0"), (0, 8, "#e8eaf6")],
     [(-4, "unstable\nstratification"), (4, "stable\nstratification")],
     r"$\theta_\mathrm{air} - \theta_{\mathrm{sfc}}$ (K)", "(c)"),
]

for ax, series, bins, xlim, vlines, bands, labels, xlabel, letter in hist_panels:
    for x0, x1, c in bands:
        ax.axvspan(x0, x1, color=c, alpha=0.25, zorder=0, lw=0)
    for x, c in series:
        x = x[np.isfinite(x)]
        ax.hist(x, bins=bins, color=c, **_kw_fill)
        ax.hist(x, bins=bins, color=c, **_kw_step)
    for xv in vlines:
        ax.axvline(xv, color="0.5", lw=0.8, ls=":", zorder=4)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)  # headroom for the region labels
    for xc, lbl in labels:
        _x = min(max((xc - xlim[0]) / (xlim[1] - xlim[0]), 0.13), 0.85)
        ax.text(_x, 0.97, lbl, transform=ax.transAxes, va="top", ha="center",
                color="0.45", multialignment="center", zorder=5)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    fs.panel_label(ax, letter)

fs.save(fig, "fig02_meteo_overview")

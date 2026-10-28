"""Fig. 5 (fig:flux_timeseries) — flux along the low-level legs and MODIS scenes (manuscript Sect. 4.1).

Six rows, one per low-level leg:
(a-f) EC flux and the five estimates (T-Bird bulk, ERA5-MODIS bulk, ERA5-only bulk,
      ERA5 and CARRA model-diagnosed) along the leg, reference configuration;
(g-l) MODIS Terra ice-surface temperature of the day (NASA GIBS) with the flight track;
(m-r) MODIS Terra true-colour image of the same area, the track coloured by the EC flux.
The GIBS images are downloaded once and cached in figures_manuscript/_gibs_cache/.

Run:  pixi run python figures_manuscript/fig05_flux_timeseries.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fluxpy import figstyle as fs

fs.apply()

import os
from urllib.parse import urlencode
from urllib.request import urlretrieve

import cmocean
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

_PROJ_DIR = Path(sys.executable).resolve().parents[1] / "share" / "proj"
if _PROJ_DIR.exists():
    os.environ.setdefault("PROJ_LIB", str(_PROJ_DIR))
from pyproj import Transformer

# ── Constants ─────────────────────────────────────────────────────────────────
CMAP_IST = cmocean.cm.thermal
CMAP_TRACK = cmocean.cm.balance
GIBS_IST_SOURCE_RANGE_K = (223.2, 313.2)
IST_RANGE_K = (245.0, 278.0)
IST_TICKS_K = (250.0, 260.0, 270.0)     # sparse: the colour bar is only ~2.5 cm wide
GIBS_IST_LAYER = "MODIS_Terra_Ice_Surface_Temp_Day"
GIBS_RGB_LAYER = "MODIS_Terra_CorrectedReflectance_TrueColor"
GIBS_IST_LEGEND_URL = "https://gibs.earthdata.nasa.gov/legends/MODIS_Ice_Surface_Temp_H.png"
GAP_THRESH = pd.Timedelta("20min")
XPAD = pd.Timedelta("3min")

_TO_3413 = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)

CACHE_DIR = REPO / "figures_manuscript" / "_gibs_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# One dict per figure row; panel letters follow from the row index
# (a-f time series, g-l IST, m-r true colour).
ROWS = [
    dict(title="13 Apr 2024", window=("2024-04-13T11:45", "2024-04-13T12:15"),
         key="20240413", min_km=180, legend=True),
    dict(title="15 Apr 2024", window=("2024-04-15T10:10", "2024-04-15T10:52"),
         key="20240415a", min_km=140, legend=False),
    dict(title="17 Apr 2024  (I)", window=("2024-04-17T14:25", "2024-04-17T14:52"),
         key="20240417a", min_km=140, legend=False),
    dict(title="17 Apr 2024  (II)", window=("2024-04-17T15:08", "2024-04-17T15:32"),
         key="20240417b", min_km=140, legend=False),
    dict(title="23 Apr 2024  (I)", window=("2024-04-23T09:30", "2024-04-23T09:52"),
         key="20240423a", min_km=140, legend=False),
    dict(title="23 Apr 2024  (II)", window=("2024-04-23T11:17", "2024-04-23T11:40"),
         key="20240423b", min_km=140, legend=False),
]

# ── Load flux data ────────────────────────────────────────────────────────────
ds_turb = xr.open_dataset(
    REPO / "data" / "observations" / "ar" / "analysis_ready_polar6_turbulent_fluxes_latest_5km.nc"
)
ds_turb = ds_turb.where(ds_turb["altitude"] < 50, drop=True)  # high-altitude artefacts
EXCLUDE = [pd.Timestamp("2024-04-25").date(), pd.Timestamp("2024-04-26").date()]
ds_turb = ds_turb.sel(start_time=~ds_turb["start_time"].dt.date.isin(EXCLUDE))

ds_sens_ref = xr.open_dataset(
    REPO / "data" / "model_output" / "sensitivity_figure_data.nc"
).sel(run="GLGS20_ref", dT=0, dU=0)

starts = pd.DatetimeIndex(ds_turb.start_time.values)
ends = pd.DatetimeIndex(ds_turb["end_time"].values)
end_times = ends.values
_dur_ns = np.where(~pd.isnull(ends), (ends.values - starts.values).astype("int64"), 0)
mid_times = pd.DatetimeIndex(starts.values.astype("int64") + _dur_ns // 2, dtype="datetime64[ns]")

# Fluxes (sign: positive = upward), every analysed segment.
H_obs_ec = ds_turb["sensible_heat_flux"].values.astype(float)
H_obs_bulk = ds_sens_ref["H_obs_bulk"].values.astype(float)
# H_rs_bulk is the ERA5-MODIS estimate, H_era_bulk the ERA5-only estimate.
H_rs_bulk = ds_sens_ref["H_rs_bulk"].values.astype(float)
H_era_bulk = ds_sens_ref["H_era_bulk"].values.astype(float)
# H_era_diag / H_car_diag are stored positive-upward (sign flipped ONCE in the
# precompute, see calc_bulk_and_ec_fluxes.py) — do NOT negate here.
H_era_diag = ds_sens_ref["H_era_diag"].values.astype(float)
H_car_diag = ds_sens_ref["H_car_diag"].values.astype(float)

# Aircraft position (50 Hz, from the KT-19 file) for the track overlays.
ds_rad = xr.open_dataset(
    REPO / "data" / "observations" / "processed" / "polar6_unified" / "kt19_unified.nc"
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _plot_with_gaps(ax, times, values, gap_thresh=GAP_THRESH, **kw):
    times = pd.DatetimeIndex(times)
    values = np.asarray(values, dtype=float)
    if len(times) == 0:
        return
    t_out, v_out = [], []
    for i in range(len(times)):
        if i > 0 and (times[i] - times[i - 1]) > gap_thresh:
            t_out.append(times[i - 1] + gap_thresh)
            v_out.append(np.nan)
        t_out.append(times[i])
        v_out.append(values[i])
    ax.plot(pd.DatetimeIndex(t_out), v_out, **kw)


def _track_xy(window):
    rad = ds_rad.sel(time=slice(window[0], window[1]))
    lats = rad["latitude"].values
    lons = rad["longitude"].values
    ok = np.isfinite(lats) & np.isfinite(lons)
    tx, ty = _TO_3413.transform(lons[ok], lats[ok])
    return tx, ty, rad.time.values[ok]


def _square_extent(window, min_km, pad_km=15, pad_frac=0.08):
    tx, ty, _ = _track_xy(window)
    pad = pad_km * 1000
    xmin, xmax = tx.min() - pad, tx.max() + pad
    ymin, ymax = ty.min() - pad, ty.max() + pad
    side = max(xmax - xmin, ymax - ymin, min_km * 1000) * (1 + pad_frac)
    xc, yc = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    return xc - side / 2, xc + side / 2, yc - side / 2, yc + side / 2


def _add_scalebar(ax, length_km=50, pad_frac=0.05):
    xl, yl = ax.get_xlim(), ax.get_ylim()
    xspan, yspan = xl[1] - xl[0], yl[1] - yl[0]
    bar_m = length_km * 1000
    if bar_m > 0.45 * xspan:
        bar_m = round(0.30 * xspan / 1e4) * 1e4
        length_km = max(int(bar_m / 1000), 10)
    x0 = xl[0] + pad_frac * xspan
    y0 = yl[0] + pad_frac * yspan
    ax.plot([x0, x0 + bar_m], [y0, y0], "-", color="k", lw=1.5, solid_capstyle="butt", zorder=20)
    ax.text(x0 + bar_m / 2, y0 + 0.03 * yspan, f"{length_km} km", ha="center", va="bottom",
            fontsize=fs.FONT_SIZE - 1, color="k", zorder=20)


def _draw_track(ax, window, zorder=10):
    tx, ty, tt = _track_xy(window)
    if len(tx) < 2:
        return
    ax.plot(tx, ty, color="white", lw=2.2, zorder=zorder - 1, solid_capstyle="round", rasterized=True)
    ax.plot(tx, ty, color=fs.GREY, lw=0.8, zorder=zorder - 1, solid_capstyle="round", rasterized=True)

    day_mask = starts.normalize() == pd.Timestamp(window[0]).normalize()
    d_starts, d_ends, d_ec = starts[day_mask], end_times[day_mask], H_obs_ec[day_mask]
    tw0, tw1 = np.datetime64(pd.Timestamp(window[0])), np.datetime64(pd.Timestamp(window[1]))
    for i in range(len(d_starts)):
        t0 = d_starts[i].to_datetime64()
        t1 = t0 + np.timedelta64(5, "m") if np.isnat(d_ends[i]) else d_ends[i]
        if t1 < tw0 or t0 > tw1:
            continue
        seg = (tt >= t0) & (tt <= t1)
        if seg.sum() < 2:
            continue
        ax.scatter(tx[seg], ty[seg], color=CMAP_TRACK(FLUX_NORM(float(d_ec[i]))), s=3.2, zorder=zorder)


def _gibs_wms_url(layer, extent, date_string, width=1200, height=1200):
    params = {
        "SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.3.0", "LAYERS": layer,
        "CRS": "EPSG:3413", "BBOX": f"{extent[0]},{extent[2]},{extent[1]},{extent[3]}",
        "WIDTH": str(width), "HEIGHT": str(height), "FORMAT": "image/png", "TIME": date_string,
    }
    return "https://gibs.earthdata.nasa.gov/wms/epsg3413/best/wms.cgi?" + urlencode(params)


def _load_cached_png(cache_path, url):
    if not cache_path.exists():
        urlretrieve(url, cache_path)
    return plt.imread(cache_path)


def _extract_legend_ramp(legend_img):
    """Longest coloured pixel run per row of the published legend PNG -> RGB ramp."""
    legend_rgb = np.asarray(legend_img, dtype=float)[..., :3]
    mask = np.any(np.abs(legend_rgb - 1.0) > 0.02, axis=2)
    runs = []
    for row in range(legend_rgb.shape[0]):
        cols = np.where(mask[row])[0]
        if len(cols) < 60:
            continue
        splits = np.where(np.diff(cols) > 1)[0]
        starts_, ends_ = np.r_[0, splits + 1], np.r_[splits, len(cols) - 1]
        c0, c1 = max(((int(cols[s]), int(cols[e])) for s, e in zip(starts_, ends_)),
                     key=lambda item: item[1] - item[0])
        runs.append((row, c0, c1, c1 - c0 + 1))
    if not runs:
        raise RuntimeError("Could not isolate the MODIS IST legend ramp.")
    best_row, best_c0, best_c1, best_len = max(runs, key=lambda item: item[3])
    ramp_rows = [row for row, c0, c1, n in runs
                 if n >= 0.98 * best_len and abs(c0 - best_c0) <= 2 and abs(c1 - best_c1) <= 2]
    return legend_rgb[ramp_rows, best_c0:best_c1 + 1].mean(axis=0)


def _remap_gibs_ist(image, legend_ramp):
    """GIBS serves IST as a pre-coloured PNG; remap it through the published legend."""
    arr = np.asarray(image, dtype=float)
    out = np.array(arr, copy=True)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return out
    alpha = arr[..., 3] if arr.shape[2] > 3 else np.ones(arr.shape[:2], dtype=float)
    preserve = np.all(arr[..., :3] > 0.985, axis=2)
    valid = (alpha > 0) & (~preserve)
    if not np.any(valid):
        return out
    rgb8 = np.clip(np.round(arr[..., :3] * 255.0), 0, 255).astype(np.uint8)
    unique_rgb, inverse = np.unique(rgb8[valid].reshape(-1, 3), axis=0, return_inverse=True)
    ramp8 = np.clip(np.round(legend_ramp * 255.0), 0, 255).astype(np.int32)
    dist2 = np.sum((unique_rgb.astype(np.int32)[:, None, :] - ramp8[None, :, :]) ** 2, axis=2)
    idx = np.argmin(dist2, axis=1)
    source_temps = np.linspace(*GIBS_IST_SOURCE_RANGE_K, legend_ramp.shape[0])
    mapped = CMAP_IST(IST_NORM(source_temps[idx]))[:, :3]
    flat_rgb = out[..., :3].reshape(-1, 3)
    flat_rgb[valid.reshape(-1)] = mapped[inverse]
    out[..., :3] = flat_rgb.reshape(arr.shape[0], arr.shape[1], 3)
    return out


def _plot_flux_row(ax, window, title, legend=False):
    tw0, tw1 = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    mask = (starts >= tw0) & (starts <= tw1)
    t = mid_times[mask]
    # legend labels = the estimate names of the manuscript's Table 1
    _plot_with_gaps(ax, t, H_obs_bulk[mask], color=fs.C["H_obs_bulk"], lw=1.2, label="T-Bird bulk")
    _plot_with_gaps(ax, t, H_rs_bulk[mask], color=fs.C["H_rs_bulk"], lw=1.2, label="ERA5–MODIS bulk")
    _plot_with_gaps(ax, t, H_era_bulk[mask], color=fs.C["H_era_bulk"], lw=1.2, label="ERA5-only bulk")
    _plot_with_gaps(ax, t, H_era_diag[mask], color=fs.C["H_era_diag"], lw=1.0, ls="--",
                    label="ERA5 model-diagnosed")
    _plot_with_gaps(ax, t, H_car_diag[mask], color=fs.C["H_car_diag"], lw=1.0, ls="--",
                    label="CARRA model-diagnosed")
    _plot_with_gaps(ax, t, H_obs_ec[mask], color=fs.C["H_obs_ec"], lw=1.5, label="EC")
    ax.axhline(0, color=fs.GREY, lw=0.8, ls=":", zorder=1)       # zero line, below the data
    ax.set_ylabel(fs.FLUX_AXIS)
    # Row title above the axes (in-axes titles collide with the 9 pt legend).
    ax.text(0.02, 1.02, title, transform=ax.transAxes, fontweight="bold", va="bottom", ha="left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))
    ax.set_xlim(tw0 - XPAD, tw1 + XPAD)


def _map_panel(ax, image, ext, window):
    ax.imshow(image, extent=ext, origin="upper", zorder=1, rasterized=True)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    _draw_track(ax, window)
    ax.set_aspect("equal", adjustable="box")
    _add_scalebar(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _add_colorbar_below(ax, cmap, norm, label, ticks, ticklabels,
                        bounds=(0.0, -0.13, 1.0, 0.055)):
    """Horizontal colour bar under the bottom map of a column, outside the imagery.

    The bar is an inset of the map axes placed below it (axes coordinates of the
    aspect-equal map, so it spans the image width); constrained layout reserves
    the space because inset axes count towards the parent's tight bounding box.
    """
    cax = ax.inset_axes(bounds)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = ax.figure.colorbar(sm, cax=cax, orientation="horizontal", ticks=ticks)
    cbar.ax.set_xticklabels(ticklabels)
    cbar.ax.tick_params(axis="x", which="major", length=2.0, width=0.5, pad=1.5)
    cbar.ax.tick_params(axis="x", which="minor", length=0)
    cbar.set_label(label, labelpad=1.5)
    cbar.outline.set_linewidth(0.5)


# ── Norms (IST fixed range; EC flux symmetric about 0 at the 97th pctile) ────
IST_NORM = mcolors.Normalize(vmin=IST_RANGE_K[0], vmax=IST_RANGE_K[1], clip=True)
IST_LEGEND_RAMP = _extract_legend_ramp(
    _load_cached_png(CACHE_DIR / "gibs_modis_ist_legend.png", GIBS_IST_LEGEND_URL)
)
_fabs = float(np.nanpercentile(np.abs(H_obs_ec[np.isfinite(H_obs_ec)]), 97))
FLUX_NORM = mcolors.TwoSlopeNorm(vcenter=0, vmin=-_fabs, vmax=_fabs)

# ── Figure: 6 rows x [time series | IST map | true-colour map] ───────────────
fig = plt.figure(figsize=(fs.FULL_W, fs.MAX_H))
gs = fig.add_gridspec(6, 3, width_ratios=[2, 1, 1])
ts_axes = []

for i, row in enumerate(ROWS):
    ax_ts = fig.add_subplot(gs[i, 0])
    ax_ist = fig.add_subplot(gs[i, 1])
    ax_rgb = fig.add_subplot(gs[i, 2])
    ts_axes.append(ax_ts)

    _plot_flux_row(ax_ts, row["window"], row["title"], legend=row["legend"])
    ax_ts.set_xlabel("Time (UTC)" if i == len(ROWS) - 1 else "")
    fs.panel_label(ax_ts, f"({chr(ord('a') + i)})")

    ext = _square_extent(row["window"], row["min_km"])
    date_string = row["window"][0][:10]
    for ax, layer, suffix, letter in (
        (ax_ist, GIBS_IST_LAYER, "ist", chr(ord("g") + i)),
        (ax_rgb, GIBS_RGB_LAYER, "truecolor", chr(ord("m") + i)),
    ):
        img = _load_cached_png(CACHE_DIR / f"gibs_{row['key']}_{suffix}.png",
                               _gibs_wms_url(layer, ext, date_string))
        if suffix == "ist":
            img = _remap_gibs_ist(img, IST_LEGEND_RAMP)
        _map_panel(ax, img, ext, row["window"])
        fs.panel_label(ax, f"({letter})")

    if i == len(ROWS) - 1:      # one shared colour bar per map column, below it
        # (g-l) show the GIBS layer MODIS_Terra_Ice_Surface_Temp_Day, not the
        # MODIS/Aqua L3 T_s,MOD of the analysis -> labelled as what it is
        _add_colorbar_below(ax_ist, CMAP_IST, IST_NORM, "MODIS/Terra IST (K)",
                            ticks=IST_TICKS_K,
                            ticklabels=[f"{t:.0f}" for t in IST_TICKS_K])
        _add_colorbar_below(ax_rgb, CMAP_TRACK, FLUX_NORM, fs.L["H_obs_ec"] + " " + fs.FLUX_UNIT,
                            ticks=[-10, 0, 10],
                            ticklabels=["\N{MINUS SIGN}10", "0", "10"])

fig.align_ylabels(ts_axes)
handles, labels = ts_axes[0].get_legend_handles_labels()
# two rows, read row by row: bulk estimates on top, model-diagnosed and EC below
# (legend columns fill column-major, hence the interleaved order)
ORDER = [0, 3, 1, 4, 2, 5]
fig.legend([handles[k] for k in ORDER], [labels[k] for k in ORDER],
           loc="outside upper center", ncol=3,
           columnspacing=1.5, handlelength=2.0, handletextpad=0.5)
fs.save(fig, "fig05_flux_timeseries")

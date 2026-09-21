"""Figure A1 -- spectral quality-control diagnostics (fig:spectral_qc).

Four spectral QC diagnostics applied to each 5-km T-Bird segment of the
low-level legs on 15 April 2024, coloured by QC verdict (accepted /
tolerable / discarded); black line = mean of accepted segments.

(a-d) power spectra of theta, w, u, v with -5/3 guides;
(e)   normalised single-realisation FFT ogive O_wtheta;
(f)   frequency-weighted cospectrum f*C_wtheta;
(g)   normalised cumulative covariance vs fractional distance.

Ported from book/notebooks/figures/figure02.ipynb.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lib"))          # meteorology.add_theta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D

from fluxpy import figstyle as fs
from fluxpy.qc import classify_leg_simple, qc_leg
from fluxpy.meteorology import add_theta

fs.apply()

# ── Configuration ────────────────────────────────────────────────────────────
DATE_STR  = "2024-04-15"
ALT_MAX_M = 70.0     # keep legs where T-Bird mean altitude < this [m]
BIN_KM    = 5.0      # spectral analysis bin size [km]
FS        = 100.0    # sampling frequency [Hz]
MIN_SAMP  = 1000     # minimum samples per bin

# Verdict line styles = CONTENT (QC classes), not house styling.
# Colours are lib/custom_style_lib.VERDICT_STYLES: cmocean.cm.ice C1/C3
# and THERMAL_RED, hard-coded here so the script is standalone.
VERDICT_STYLES = {
    "accepted":  dict(color=(0.4189584, 0.69487019, 0.79630958),
                      linewidth=1.2, linestyle="-",  zorder=0),
    "tolerable": dict(color=(0.73861316, 0.89262124, 0.90445285),
                      linewidth=1.1, linestyle="-",  zorder=1),
    "discarded": dict(color=(0.87936081, 0.44295096, 0.38747682),
                      linewidth=1.0, linestyle="--", zorder=2),
}

# ── Load data and build per-bin QC dicts ─────────────────────────────────────
ds_tb = xr.open_dataset(
    REPO / "data" / "observations" / "processed" / "t-bird_unified" /
    "noseboom_unified.nc", engine="h5netcdf").sortby("time")
ds_radiation = xr.open_dataset(
    REPO / "data" / "observations" / "processed" / "polar6_unified" /
    "radiation_unified.nc", engine="h5netcdf")

df_legs = pd.read_csv(
    REPO / "data" / "observations" / "ar" /
    "analysis_ready_polar6_legs_metadata.csv",
    index_col=0, sep=None, engine="python")
df_legs["start_time"] = pd.to_datetime(df_legs["start_time"])
df_legs["end_time"]   = pd.to_datetime(df_legs["end_time"])

df_day = df_legs[df_legs["start_time"].dt.date.astype(str) == DATE_STR]

list_of_qcs = []
for _, row in df_day.iterrows():
    l = ds_tb.sel(time=slice(str(row.start_time), str(row.end_time))).copy()
    if len(l.time) == 0 or float(l.h.mean()) >= ALT_MAX_M:
        continue

    # KT19 is not plotted, but it joins the dataset BEFORE dropna(how="any")
    # below and therefore co-determines which samples survive -- keep it.
    r = ds_radiation.sel(time=slice(str(row.start_time), str(row.end_time)))
    if "KT19" in r:
        l["kt19"] = r["KT19"].interp_like(l)

    # along-track distance coordinate [km] from mean true airspeed
    tas_mean = float(l.tas.mean())
    dt_s = (l["time"] - l["time"][0]).dt.seconds.values.astype(float)
    dist = tas_mean * dt_s / 1000.0
    l = l.assign_coords({"distance": ("time", dist)})
    l = l.swap_dims({"time": "distance"})
    l = add_theta(l)

    distance_bins = np.arange(0, dist[-1] + BIN_KM, BIN_KM)
    for _, ds_bin in l.groupby_bins("distance", distance_bins):
        data = ds_bin.dropna(dim="distance", how="any")
        if data.sizes["distance"] < MIN_SAMP:
            continue
        list_of_qcs.append(qc_leg(
            w=data["w"].values, theta=data["theta"].values,
            u=data["u"].values, v=data["v"].values,
            fs=FS, x_dist=data["distance"].values,
        ))

qc_simple = [classify_leg_simple(qc, BIN_KM) for qc in list_of_qcs]
verdicts = [s["verdict"] for s in qc_simple]
print(f"{len(list_of_qcs)} QC bins:",
      {v: verdicts.count(v) for v in ("accepted", "tolerable", "discarded")})

# ── Plot helpers ─────────────────────────────────────────────────────────────
def _add_inertial_band(ax, f1=0.1, f2=20.0):
    ax.axvspan(f1, f2, color="0.92", zorder=-5)
    ax.axvline(f1, color="k", linestyle="-", linewidth=0.7)
    ax.axvline(f2, color="k", linestyle="-", linewidth=0.7)


def _add_minus53_guides(ax, f, anchor_idx=3):
    fa = f[anchor_idx]
    with np.errstate(divide="ignore"):       # f[0] = 0 -> inf, not drawn
        for k in range(-5, 2):
            ax.loglog(f, 10.0 ** k * (f / fa) ** (-5 / 3),
                      ls="--", color="0.6", linewidth=0.5)


def _avg_accepted(key, f_key="f"):
    """Nanmean of `key` across accepted legs, trimmed to common length."""
    acc = [qc for qc, s in zip(list_of_qcs, qc_simple) if s["verdict"] == "accepted"]
    if not acc:
        return None, None
    n = min(len(qc[key]) for qc in acc)
    avg = np.nanmean(np.array([qc[key][:n] for qc in acc]), axis=0)
    return acc[0][f_key][:n], avg


def _plot_verdict_lines(ax, x_key, y_key, scale=1.0, skip_dc=True):
    """One line per segment, coloured by verdict, plus black accepted mean."""
    i0 = 1 if skip_dc else 0
    for qc, simple in zip(list_of_qcs, qc_simple):
        style = VERDICT_STYLES[simple["verdict"]]
        n = min(len(qc[x_key]), len(qc[y_key]))
        ax.plot(qc[x_key][i0:n], qc[y_key][i0:n] * scale, **style)
    x_avg, avg = _avg_accepted(y_key, f_key=x_key)
    if avg is not None:
        ax.plot(x_avg[i0:], avg[i0:] * scale, color="k", lw=1.0, ls="-", zorder=10)


# ── Figure: (a-d) power spectra | (e) ogive, (f) f-cospectrum | (g) cumsum ──
mosaic = """
AABB
CCDD
GGFF
EEHH
"""
fig, axs = plt.subplot_mosaic(mosaic, figsize=(fs.FULL_W, 18.5 * fs.CM))

power_panels = [
    ("A", "Pth", r"$\theta$", r"$P_\theta\;(K^2\,s)$"),
    ("B", "Pw",  r"$w$",      r"$P_w\;(m^2\,s^{-1})$"),
    ("C", "Pu",  r"$u$",      r"$P_u\;(m^2\,s^{-1})$"),
    ("D", "Pv",  r"$v$",      r"$P_v\;(m^2\,s^{-1})$"),
]
for key, p_key, var_label, ylabel in power_panels:
    ax = axs[key]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-2, 100.0)
    ax.set_ylim(1e-5, 1e-1)
    ax.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    _add_inertial_band(ax)
    _add_minus53_guides(ax, list_of_qcs[0]["f"])
    _plot_verdict_lines(ax, "f", p_key)
    ax.text(0.17, 0.92, var_label, transform=ax.transAxes, fontweight="bold")
    ax.set_ylabel(ylabel)

# (e) normalised FFT ogive; goldenrod line = 0.1 Hz plateau boundary
ax = axs["G"]
ax.set_xscale("log")
ax.axhline(0.0, color="0.3", lw=0.7)
ax.axhline(1.0, color="0.3", lw=0.7)
ax.axvline(0.1, color="goldenrod", lw=1.0, ls="--")
_plot_verdict_lines(ax, "f_fft", "og_fft_norm")
ax.set_xlim(1e-2, 100.0)
_add_inertial_band(ax)
ax.set_xlabel("f (Hz)")
ax.set_ylabel(r"norm. $O_{w\theta}$")

# (f) frequency-weighted cospectrum (x1000 -> m K s-1)
ax = axs["F"]
ax.set_xscale("log")
ax.set_xlim(1e-2, 100.0)
_add_inertial_band(ax)
ax.axhline(0, color="0.3", lw=0.7)
_plot_verdict_lines(ax, "f", "fwCwth", scale=1000.0)
ax.set_xlabel("f (Hz)")
ax.set_ylabel(r"$f\,C_{w\theta}\;(m\,K\,s^{-1})$")

# (g) cumulative covariance vs fractional distance; dashed 1:1 = stationary
ax = axs["E"]
ax.set_xlim(0, 1)
ax.plot([0, 1], [0, 1], color="0.3", lw=0.7, ls="--", zorder=5)
_plot_verdict_lines(ax, "cum_frac", "cum_cov_norm", skip_dc=False)
ax.set_xlabel("fractional distance")
ax.set_ylabel(r"$cumsum_\mathrm{w\theta}$")

for key, lab in [("A", "(a)"), ("B", "(b)"), ("C", "(c)"), ("D", "(d)"),
                 ("G", "(e)"), ("F", "(f)"), ("E", "(g)")]:
    fs.panel_label(axs[key], lab)

# (H) legend panel
axs["H"].axis("off")
handles = [Line2D([0], [0], **VERDICT_STYLES[v])
           for v in ("accepted", "tolerable", "discarded")]
labels = ["Accepted", "Tolerable", "Discarded"]
handles.append(Line2D([0], [0], color="k", lw=1.0, ls="-"))
labels.append("mean (accepted)")
axs["H"].legend(handles, labels, ncol=1, loc="center")

fs.save(fig, "figA1_spectral_qc")

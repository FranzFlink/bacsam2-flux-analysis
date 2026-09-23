"""Figure A1 -- spectral diagnostics of the EC segments (fig:spectral_qc).

One line per 5-km T-Bird segment of the four low-level flight days, coloured
by whether the segment's flux exceeds twice its random error; black line = median over the plotted segments.

(a-d) power spectra of theta, w, u, v with -5/3 guides;
(e)   frequency-weighted cospectrum f*C_wtheta;
(f)   normalised ogive O_wtheta (fluxes above twice their random error only:
      the normalisation by the total covariance is meaningless for a flux
      that cannot be distinguished from zero).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from scipy.signal import detrend

from fluxpy import figstyle as fs
from fluxpy.eddy_covariance import band_covariance, band_covariance_random_error
from fluxpy.qc import segment_spectra
from fluxpy.util import add_theta

fs.apply()

# ── Configuration ────────────────────────────────────────────────────────────
FS = 100.0           # sampling frequency [Hz]

# Line styles = CONTENT (detectability classes), not house styling.
STYLES = {
    True:  dict(color=(0.4189584, 0.69487019, 0.79630958), linewidth=0.8, linestyle="-", zorder=1),
    False: dict(color="0.75", linewidth=0.6, linestyle="-", zorder=0),
}

# ── Segments of the paper and their 100 Hz data ──────────────────────────────
FIG = xr.open_dataset(REPO / "data/model_output/sensitivity_figure_data.nc")
SEG = xr.open_dataset(
    REPO / "data/observations/ar/analysis_ready_polar6_turbulent_fluxes_latest_5km.nc"
).sel(start_time=FIG["start_time"].values)
ds_tb = xr.open_dataset(
    REPO / "data/observations/processed/t-bird_unified/noseboom_unified.nc").sortby("time")

list_of_qcs, detectable = [], []
for t0, t1 in zip(SEG["start_time"].values, SEG["end_time"].values):
    data = add_theta(ds_tb.sel(time=slice(t0, t1))).dropna(dim="time", how="any")
    list_of_qcs.append(segment_spectra(w=data["w"].values, theta=data["theta"].values,
                                       u=data["u"].values, v=data["v"].values, fs=FS))
    w, th = detrend(data["w"].values), detrend(data["theta"].values)
    detectable.append(bool(abs(band_covariance(w, th, FS))
                           > 2.0 * band_covariance_random_error(w, th, FS)))
print(f"{len(list_of_qcs)} segments, {sum(detectable)} above twice their random error")


# ── Plot helpers ─────────────────────────────────────────────────────────────
def _add_flux_band(ax, f1=0.1, f2=20.0):
    ax.axvspan(f1, f2, color="0.92", zorder=-5)
    ax.axvline(f1, color="k", linestyle="-", linewidth=0.7)
    ax.axvline(f2, color="k", linestyle="-", linewidth=0.7)


def _add_minus53_guides(ax, f, anchor_idx=3):
    fa = f[anchor_idx]
    with np.errstate(divide="ignore"):       # f[0] = 0 -> inf, not drawn
        for k in range(-5, 2):
            ax.loglog(f, 10.0 ** k * (f / fa) ** (-5 / 3),
                      ls="--", color="0.6", linewidth=0.5)


def _plot_lines(ax, x_key, y_key, scale=1.0, only_detectable=False):
    """One line per segment plus the black median over the plotted segments."""
    rows = [(qc, d) for qc, d in zip(list_of_qcs, detectable) if d or not only_detectable]
    n = min(min(len(qc[x_key]), len(qc[y_key])) for qc, _ in rows)
    for qc, d in rows:
        ax.plot(qc[x_key][1:n], qc[y_key][1:n] * scale, **STYLES[d])
    med = np.nanmedian(np.array([qc[y_key][:n] for qc, _ in rows]), axis=0)
    ax.plot(rows[0][0][x_key][1:n], med[1:] * scale, color="k", lw=1.0, zorder=10)


# ── Figure: (a-d) power spectra | (e) f-cospectrum, (f) ogive ────────────────
fig, axs = plt.subplot_mosaic("AABB\nCCDD\nEEFF", figsize=(fs.FULL_W, 15.0 * fs.CM))

for key, p_key, var_label, ylabel in [
    ("A", "P_theta", r"$\theta$", r"$P_\theta\;(K^2\,s)$"),
    ("B", "P_w",     r"$w$",      r"$P_w\;(m^2\,s^{-1})$"),
    ("C", "P_u",     r"$u$",      r"$P_u\;(m^2\,s^{-1})$"),
    ("D", "P_v",     r"$v$",      r"$P_v\;(m^2\,s^{-1})$"),
]:
    ax = axs[key]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-2, 100.0)
    ax.set_ylim(1e-5, 1e0)
    ax.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    _add_flux_band(ax)
    _add_minus53_guides(ax, list_of_qcs[0]["f"])
    _plot_lines(ax, "f", p_key)
    ax.text(0.17, 0.92, var_label, transform=ax.transAxes, fontweight="bold")
    ax.set_ylabel(ylabel)

# (e) frequency-weighted cospectrum (x1000 -> m K s-1)
ax = axs["E"]
ax.set_xscale("log")
ax.set_xlim(1e-2, 100.0)
_add_flux_band(ax)
ax.axhline(0, color="0.3", lw=0.7)
_plot_lines(ax, "f", "f_cospectrum", scale=1000.0)
ax.set_xlabel("f (Hz)")
ax.set_ylabel(r"$f\,C_{w\theta}\;(m\,K\,s^{-1})$")

# (f) normalised ogive, detectable fluxes only
ax = axs["F"]
ax.set_xscale("log")
ax.set_xlim(1e-2, 100.0)
ax.set_ylim(-0.5, 2.0)
_add_flux_band(ax)
ax.axhline(0.0, color="0.3", lw=0.7)
ax.axhline(1.0, color="0.3", lw=0.7)
_plot_lines(ax, "f_fft", "ogive_norm", only_detectable=True)
ax.set_xlabel("f (Hz)")
ax.set_ylabel(r"norm. $O_{w\theta}$")

for key, lab in zip("ABCDEF", ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]):
    fs.panel_label(axs[key], lab)

axs["E"].legend(
    [Line2D([0], [0], **STYLES[True]), Line2D([0], [0], **STYLES[False]), Line2D([0], [0], color="k", lw=1.0)],
    [r"$|H_\mathrm{S}| > 2\,\sigma_\mathrm{rand}$", r"$|H_\mathrm{S}| \leq 2\,\sigma_\mathrm{rand}$", "median"], loc="lower left")

fs.save(fig, "figA1_spectral_qc")

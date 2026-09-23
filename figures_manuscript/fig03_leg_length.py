"""Fig. 3 (fig:ec_leg_length_sensitivity) -- EC flux vs the length of the averaging window.

Manuscript Sect. 3.1: why 5 km. Three stacked panels over the six low-level legs:
(a) sub-window flux relative to the full-leg flux, H_sub/H_full (mean, +-1 sigma),
(b) random error sigma_rand of the sub-window fluxes (median, quartiles),
(c) fraction of sub-windows whose flux exceeds 1, 2 and 3 sigma_rand.

Data: scripts/ec_window_length.py -> data/observations/ar/ec_leg_length_sensitivity/
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib.ticker as mticker
import pandas as pd
import cmcrameri.cm as cmc
import numpy as np

from fluxpy import figstyle as fs

fs.apply()

# ── Data ─────────────────────────────────────────────────────────────────────
CSV = (REPO / "data" / "observations" / "ar" / "ec_leg_length_sensitivity"
       / "ec_sensitivity_summary.csv")
df_raw = pd.read_csv(CSV)
df_sub = (df_raw[(df_raw["label"] != "full leg") & (df_raw["length_km"] >= 1.0)]
          .sort_values("length_km").reset_index(drop=True))

# ── Colours (batlow: ratio, random error, and the 1/2/3 sigma curves) ────────
C_BATLOW = cmc.batlow(np.linspace(0, 1, 4))
C_RATIO = C_BATLOW[0]
C_SIGMA = C_BATLOW[1]
C_REF = "goldenrod"      # 5 km reference marker

REF_LENGTH_KM = 5.0

# ── Layout ───────────────────────────────────────────────────────────────────
fig, axes = fs.figure(width=fs.COL_W, aspect=1.85, nrows=3)

kw_eb = dict(lw=1.0, ms=3.5, mew=0.6, markeredgecolor="k", elinewidth=0.8,
             capsize=2.5, capthick=0.8, zorder=0, alpha=0.5, ecolor="k")


def _fmt_xax(ax, xlim=(0.9, 31.0)):
    ax.set_xscale("log")
    ax.set_xlim(xlim)
    ax.set_xticks([1.0, 2.5, 5.0, 10.0, 20.0, 30])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Averaging window (km)")
    ax.axvline(REF_LENGTH_KM, color=C_REF, lw=3.0, ls="--", zorder=-1)


# ── (a) Flux-capture ratio H_sub / H_full ────────────────────────────────────
ax = axes[0]
valid = df_sub["ratio_mean"].notna() & df_sub["ratio_std"].notna()
x = df_sub.loc[valid, "length_km"].where(df_sub.loc[valid, "length_km"] <= 30.0)
y = df_sub.loc[valid, "ratio_mean"]
yerr = df_sub.loc[valid, "ratio_std"]

ax.plot(x, y, color=C_RATIO)
ax.errorbar(x, y, yerr=yerr, color=C_RATIO, fmt="o",
            label=r"$H_\mathrm{S,sub}/H_\mathrm{S,full}$  $\pm1\sigma$", **kw_eb)
ax.axhline(1.0, color="k", lw=1.5, ls="--", alpha=0.6, label="Full-leg = 1")
_fmt_xax(ax)
ax.set_ylabel(r"$H_\mathrm{S,sub}\,/\,H_\mathrm{S,full}$")
ax.legend(loc="lower right")
fs.panel_label(ax, "(a)")

# ── (b) random error of the sub-window flux (Finkelstein & Sims, 2001) ─────
ax = axes[1]
valid = df_sub["random_error_median"].notna()
x = df_sub.loc[valid, "length_km"]
y = df_sub.loc[valid, "random_error_median"]
yerr = [y - df_sub.loc[valid, "random_error_p25"], df_sub.loc[valid, "random_error_p75"] - y]

ax.plot(x, y, color=C_SIGMA)
ax.errorbar(x, y, yerr=yerr, fmt="s", color=C_SIGMA,
            label=r"$\sigma_\mathrm{rand}$  (median, quartiles)", **kw_eb)
_fmt_xax(ax)
ax.set_ylim(bottom=0)
ax.set_ylabel(r"random error $\sigma_\mathrm{rand}$ (W m$^{-2}$)")
ax.legend(loc="upper right")
fs.panel_label(ax, "(b)")

# ── (c) share of sub-windows whose flux exceeds 1, 2, 3 x its random error ──
ax = axes[2]
for i, k in enumerate((1, 2, 3)):
    col = f"fraction_above_{k}sigma"
    valid = df_sub[col].notna()
    ax.plot(df_sub.loc[valid, "length_km"], df_sub.loc[valid, col], marker="o", ms=3, ls="-", color=C_BATLOW[i],
            label=rf"$|H_\mathrm{{S}}| > {k}\,\sigma_\mathrm{{rand}}$")
_fmt_xax(ax)
ax.set_ylim(0, 1)
ax.set_ylabel("fraction of sub-windows")
ax.legend(loc="lower right", handlelength=2.2)
fs.panel_label(ax, "(c)")
ax.axvline(REF_LENGTH_KM, color=C_REF, lw=3.0, ls="--", zorder=10)

fs.save(fig, "fig03_leg_length")

"""Manuscript Fig. 7 (fig:sensitivity_summary) — sensitivity in the slope/r framework.

(a) amplitude skill: regression slope b on the EC reference (ideal b = 1);
(b) variability skill: correlation r.
Representative configurations only (full matrix: Table C1); the model-diagnosed
reanalysis fluxes are grey-diamond benchmarks.

Run:  pixi run python figures_manuscript/fig07_sensitivity.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fluxpy import figstyle as fs  # noqa: E402

fs.apply()

df = pd.read_csv(REPO / "data/model_output/sensitivity_summary_table.csv").set_index("label")
FIG = xr.open_dataset(REPO / "data/model_output/sensitivity_figure_data.nc")
EC = np.asarray(FIG["H_obs_ec"])


def diag_metrics(key):
    """Slope and r of a model-diagnosed flux against the EC reference."""
    h = np.asarray(FIG[key])
    m = np.isfinite(h) & np.isfinite(EC)
    return float(np.polyfit(EC[m], h[m], 1)[0]), float(np.corrcoef(EC[m], h[m])[0, 1])


# Essential rows only; None = italic section header, "DIAG:" = benchmark diamond.
ROWS = [
    ("Reference ($\\alpha_T = 10^{-1}$)", "GLGS20_ref"),
    ("Amplitude-matched ($\\alpha_T = 10^{-4}$)", "GLGS20_ztr1e-04"),
    ("Stability scheme", None),
    ("Businger–Dyer", "BD_ref"),
    ("SHEBA (GAFGP07)", "GAFGP07_ref"),
    ("Zilitinkevich (ZE07-I)", "ZE07-I_ref"),
    ("Momentum roughness", None),
    ("MOSAiC set", "GLGS20_mosaic"),
    ("Schnierstein set", "GLGS20_nschnier"),
    ("$z_{0,m} = 4{\\times}10^{-4}$ m (uniform)", "GLGS20_z0c4e-04"),
    ("Model-diagnosed benchmark", None),
    ("ERA5", "DIAG:H_era_diag"),
    ("CARRA$^{*}$", "DIAG:H_car_diag"),
]

C_DIAG = "0.30"
fig, (ax_a, ax_b) = fs.figure(width=fs.FULL_W, aspect=0.50, ncols=2, sharey=True)
ypos = [-float(i) for i in range(len(ROWS))]

for ax, (co, cr, xlab) in [(ax_a, ("obs_slope", "rs_slope", "Regression slope $b$")),
                           (ax_b, ("obs_r", "rs_r", "Correlation $r$"))]:
    ax.axvline(1.0, color="0.35", lw=1.0, ls="--", zorder=1)
    for yy, (lbl, key) in zip(ypos, ROWS):
        if key is None:
            continue
        if key.startswith("DIAG:"):
            sl, r = diag_metrics(key.split(":")[1])
            ax.plot(sl if ax is ax_a else r, yy, "D", ms=4.5, color=C_DIAG,
                    mfc=C_DIAG, mec="white", mew=0.6, zorder=3)
            continue
        if key in ("GLGS20_ref", "GLGS20_ztr1e-04"):
            ax.axhspan(yy - 0.5, yy + 0.5, color="0.92", zorder=0)
        vo, vr = df.loc[key, co], df.loc[key, cr]
        ax.plot([vo, vr], [yy, yy], color="0.75", lw=1.0, zorder=2)
        ax.plot(vo, yy, "o", ms=5, color=fs.C["H_obs_bulk"], mec="white", mew=0.6, zorder=3)
        ax.plot(vr, yy, "o", ms=5, color=fs.C["H_rs_bulk"], mec="white", mew=0.6, zorder=3)
    ax.set_xlabel(xlab)
    ax.grid(axis="x", lw=0.4, alpha=0.5)
    ax.set_ylim(min(ypos) - 0.7, max(ypos) + 0.7)

ax_a.set_yticks(ypos)
ax_a.set_yticklabels([lbl for lbl, _ in ROWS])
for tick, (_, key) in zip(ax_a.get_yticklabels(), ROWS):
    if key is None:
        tick.set_fontstyle("italic")
        tick.set_color("0.35")
ax_b.set_xlim(0.0, 1.05)

fig.legend(handles=[
    Line2D([0], [0], marker="o", lw=0, color=fs.C["H_obs_bulk"], ms=5,
           label=fs.L["H_obs_bulk"] + " (T-Bird)"),
    Line2D([0], [0], marker="o", lw=0, color=fs.C["H_rs_bulk"], ms=5,
           label=fs.L["H_rs_bulk"] + " (ERA5$-$MODIS)"),
    Line2D([0], [0], marker="D", lw=0, color=C_DIAG, ms=4.5,
           label="model-diagnosed")],
    loc="outside upper center", ncol=3)

fs.panel_label(ax_a, "(a)")
fs.panel_label(ax_b, "(b)")
fs.save(fig, "fig07_sensitivity")

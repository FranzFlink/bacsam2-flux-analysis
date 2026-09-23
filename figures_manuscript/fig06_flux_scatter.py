"""Fig. 6 (fig:flux_scatter_comparison) -- bulk vs EC, 2x2 (manuscript Sect. 4.1).

(a) H_S,obs,bulk against H_S,obs,EC;
(b) histograms of the EC, T-Bird bulk and ERA5-MODIS bulk fluxes;
(c) H_S,RS,bulk against H_S,obs,EC;
(d) histograms of the EC flux and the reanalysis-driven estimates.

Scatter points are coloured by the bulk Richardson number; the error bars are the
1-sigma uncertainties of Table 2 (fluxpy.uncertainty). The regression slope b,
correlation r and bias are printed for the caption. All fluxes are positive
upward; the model-diagnosed fluxes are already stored with this sign.

Run:  pixi run python figures_manuscript/fig06_flux_scatter.py
"""
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import gaussian_kde

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fluxpy import figstyle as fs  # noqa: E402
from fluxpy.uncertainty import uncertainty_budget  # noqa: E402

fs.apply()

# ── Data: reference-configuration fluxes ─────────────────────────────────────
ds = xr.open_dataset(REPO / "data/model_output/sensitivity_figure_data.nc")
ref = ds.sel(run="GLGS20_ref", dT=0, dU=0)
H = {k: ref[k].values.astype(float)
     for k in ["H_obs_ec", "H_obs_bulk", "H_rs_bulk", "H_era_bulk",
               "H_era_diag", "H_car_diag"]}
RI = ds["Ri_obs"].values.astype(float)
ec = H["H_obs_ec"]
ec_valid = np.isfinite(ec)

# ── 1-sigma uncertainties of Table 2 (same function as the table) ────────────
BUDGET = uncertainty_budget(ds)
SIGMA = {"H_obs_ec": BUDGET["ec_total"], "H_obs_bulk": BUDGET["obs_bulk"], "H_rs_bulk": BUDGET["rs_bulk"]}


def metrics(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return (float(np.polyfit(x[m], y[m], 1)[0]),
            float(stats.pearsonr(x[m], y[m])[0]),
            float(np.mean(y[m] - x[m])))

# ── Layout (single column: ~3 cm panels — sparse major ticks, minors carry
#    the 10 W m-2 raster; metric annotations reduced to b / r / bias) ─────────
SC_LIM, SC_TICKS = (-15, 35), [0, 20]
HI_LIM, HI_TICKS = (-25, 35), [-20, 0, 20]
HI_BINS = np.arange(-26, 36.1, 2.0)
cmap_ri = plt.cm.coolwarm_r
norm_ri = mcolors.Normalize(vmin=0.2, vmax=-1)     # red = unstable, blue = stable

fig, axes = fs.figure(width=fs.COL_W, aspect=1.1, nrows=2, ncols=2)

# ── (a) + (c): scatter vs EC, coloured by Ri_b ───────────────────────────────
for ax, key in [(axes[0, 0], "H_obs_bulk"), (axes[1, 0], "H_rs_bulk")]:
    y = H[key]
    m = np.isfinite(ec) & np.isfinite(y) & np.isfinite(RI)
    ax.errorbar(ec[m], y[m],
                xerr=np.nan_to_num(SIGMA["H_obs_ec"][m]),
                yerr=np.nan_to_num(SIGMA[key][m]),
                fmt="none", ecolor="0.35", elinewidth=0.3, alpha=0.8, zorder=2)
    ax.scatter(ec[m], y[m], c=RI[m], cmap=cmap_ri, norm=norm_ri,
               s=10, alpha=0.85, linewidths=0, rasterized=True, zorder=3)
    ax.plot(SC_LIM, SC_LIM, color=fs.GREY, lw=0.7, alpha=0.5, zorder=1)
    b, r, bias = metrics(ec, y)
    ax.plot(np.asarray(SC_LIM), np.polyval(np.polyfit(ec[m], y[m], 1), SC_LIM),
            color=fs.C[key], lw=1.2, ls="--", zorder=4)
    # metrics live in the CAPTION at single-column size — print for provenance
    print(f"caption numbers {key}: b={b:.2f}  r={r:.2f}  bias={bias:+.1f} W m-2")
    ax.set_xlim(SC_LIM), ax.set_ylim(SC_LIM)
    # NB no set_box_aspect here: it silently breaks constrained layout (axes
    # overflow the canvas without warning); near-square comes from the figure
    # aspect instead.
    ax.set_xticks(SC_TICKS), ax.set_yticks(SC_TICKS)
    ax.set_xticks(np.arange(-10, 31, 10), minor=True)
    ax.set_yticks(np.arange(-10, 31, 10), minor=True)
    ax.axhline(0, color=fs.GRID, lw=0.4, zorder=0)
    ax.axvline(0, color=fs.GRID, lw=0.4, zorder=0)
    ax.set_ylabel(fs.L[key] + " " + fs.FLUX_UNIT)

axes[0, 0].tick_params(labelbottom=False)
axes[1, 0].set_xlabel(fs.L["H_obs_ec"] + " " + fs.FLUX_UNIT)

cax = axes[0, 0].inset_axes([0.06, 0.56, 0.05, 0.38])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_ri, norm=norm_ri), cax=cax,
                  ticks=[0, -1])
cb.ax.set_title(r"$Ri_\mathrm{b}$", fontsize=fs.FONT_SIZE - 1, pad=2)
cb.ax.tick_params(labelsize=fs.FONT_SIZE - 1, length=2)
cb.outline.set_linewidth(0.5)

# ── (b) + (d): histograms + KDE over the EC-valid segments ───────────────────
def hist_panel(ax, keys):
    xg = np.linspace(*HI_LIM, 500)
    for k in keys:
        v = H[k][ec_valid & np.isfinite(H[k])]
        lbl = fs.L[k] + (f"  (n={len(v)})" if len(v) < int(ec_valid.sum()) else "")
        ax.hist(v, bins=HI_BINS, density=True, histtype="stepfilled",
                alpha=0.12, color=fs.C[k], zorder=1)
        ax.hist(v, bins=HI_BINS, density=True, histtype="step",
                alpha=0.75, color=fs.C[k], lw=0.9, zorder=2)
        kde = gaussian_kde(v, bw_method=0.4)(xg)
        ax.plot(xg, kde, color=fs.C[k], label=lbl,
                lw=1.2 if k == "H_obs_ec" else 0.8,       # EC heavier: the reference
                zorder=12 if k == "H_obs_ec" else 10)
        ax.fill_between(xg, kde, alpha=0.08, color=fs.C[k], zorder=9)
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlim(HI_LIM), ax.set_ylim(bottom=0)
    ax.set_xticks(HI_TICKS)
    ax.set_xticks(np.arange(-25, 36, 5), minor=True)
    ax.set_ylabel("Density")


hist_panel(axes[0, 1], ["H_obs_ec", "H_obs_bulk", "H_rs_bulk"])
hist_panel(axes[1, 1], ["H_obs_ec", "H_era_bulk", "H_era_diag", "H_car_diag"])
axes[0, 1].tick_params(labelbottom=False)
# short symbol label: the spelled-out one overhangs the canvas under a ~3 cm
# panel (constrained layout ignores xlabel WIDTH, so it would be clipped)
axes[1, 1].set_xlabel(fs.FLUX_AXIS)

# one legend for all series (per-panel legends would cover the ~3 cm panels).
# ncol=2: three math-symbol columns are wider than the 8.3 cm canvas and make
# constrained layout overflow. Column-major fill -> left column = panel (b)
# series, right column = panel (d) series.
_leg = [("H_obs_ec", 2.0), ("H_obs_bulk", 1.6), ("H_rs_bulk", 1.6),
        ("H_era_bulk", 1.6), ("H_era_diag", 1.6), ("H_car_diag", 1.6)]
fig.legend(handles=[Line2D([0], [0], color=fs.C[k], lw=lw, label=fs.L[k])
                    for k, lw in _leg],
           loc="outside upper center", ncol=3, columnspacing=1.2,
           handlelength=1.0, handletextpad=0.3)

for ax, lbl in zip(axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
    fs.panel_label(ax, lbl)

fs.save(fig, "fig06_flux_scatter")

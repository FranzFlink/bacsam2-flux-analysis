"""Fig. 3 — EC sensitivity to the averaging-window length (fig:ec_leg_length_sensitivity).

Three stacked panels over the six low-level legs:
(a) flux-capture ratio H_sub/H_full (median, ±1σ),
(b) Lenschow relative random sampling error ε_rel (mean, ±1σ),
(c) stacked QC-verdict fractions (accepted / tolerable / discarded).

Ported from book/notebooks/figures/figure03.ipynb (last-run cell).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib.ticker as mticker
import pandas as pd

from fluxpy import figstyle as fs

fs.apply()

# ── Data ─────────────────────────────────────────────────────────────────────
CSV = (REPO / "data" / "observations" / "ar" / "ec_leg_length_sensitivity"
       / "ec_sensitivity_summary.csv")
df_raw = pd.read_csv(CSV)
df_sub = (df_raw[(df_raw["label"] != "full leg") & (df_raw["length_km"] >= 1.0)]
          .sort_values("length_km").reset_index(drop=True))

# ── Colours ──────────────────────────────────────────────────────────────────
C_RATIO = "#ff7f00"      # amber   – flux-capture ratio
C_LENSCHOW = "#984ea3"   # purple  – Lenschow ε_rel
C_REF = "goldenrod"      # 5-km reference marker
# QC verdicts (values of cmocean.cm.ice(linspace(.6, 1, 5))[1|3] and THERMAL_RED)
C_ACC = (0.418958, 0.694870, 0.796310)    # accepted  – mid ice blue
C_TOL = (0.738613, 0.892621, 0.904453)    # tolerable – pale ice blue
C_DISC = (0.879361, 0.442951, 0.387477)   # discarded – thermal red

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
y = df_sub.loc[valid, "median_ratio"]
yerr = df_sub.loc[valid, "ratio_std"]

ax.plot(x, y, color=C_RATIO)
ax.errorbar(x, y, yerr=yerr, color=C_RATIO, fmt="o",
            label=r"$H_\mathrm{S,sub}/H_\mathrm{S,full}$  $\pm1\sigma$", **kw_eb)
ax.axhline(1.0, color="k", lw=1.5, ls="--", alpha=0.6, label="Full-leg = 1")
_fmt_xax(ax)
ax.set_ylabel(r"$H_\mathrm{S,sub}\,/\,H_\mathrm{S,full}$")
ax.legend(loc="lower right")
fs.panel_label(ax, "(a)")

# ── (b) Lenschow relative sampling error ─────────────────────────────────────
ax = axes[1]
valid = (df_sub["lenschow_rel_error_mean"].notna()
         & df_sub["lenschow_rel_error_std"].notna())
x = df_sub.loc[valid, "length_km"]
y = df_sub.loc[valid, "lenschow_rel_error_mean"] * 100
yerr = df_sub.loc[valid, "lenschow_rel_error_std"] * 100

ax.plot(x, y, color=C_LENSCHOW)
ax.errorbar(x, y, yerr=yerr, fmt="s", color=C_LENSCHOW,
            label=r"$\varepsilon_\mathrm{rel}$  $\pm1\sigma$", **kw_eb)
_fmt_xax(ax)
ax.set_ylim(bottom=0)
ax.set_ylabel(r"rel. error $\varepsilon_\mathrm{rel}$  (%)")
ax.legend(loc="upper right")
fs.panel_label(ax, "(b)")

# ── (c) QC verdict fractions ─────────────────────────────────────────────────
ax = axes[2]
valid = df_sub[["n_accepted", "n_tolerable", "n_discarded"]].notna().all(axis=1)
dq = df_sub[valid]
n_total = dq["N_bins"].values.astype(float)
ax.stackplot(dq["length_km"].values,
             dq["n_discarded"].values / n_total,
             dq["n_tolerable"].values / n_total,
             dq["n_accepted"].values / n_total,
             labels=["Discarded", "Tolerable", "Accepted"],
             colors=[C_DISC, C_TOL, C_ACC], alpha=0.85)
_fmt_xax(ax)
ax.set_ylim(0, 1)
ax.set_ylabel("Fraction of bins")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="upper right")
fs.panel_label(ax, "(c)")
ax.axvline(REF_LENGTH_KM, color=C_REF, lw=3.0, ls="--", zorder=10)

fs.save(fig, "fig03_leg_length")

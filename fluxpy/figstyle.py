"""
fluxpy.figstyle — THE single styling file for every manuscript figure.
=====================================================================

All manuscript figure scripts (``figures_manuscript/fig*.py``) import ONLY
from here.  No figure script sets rcParams, font sizes, or colours itself.

Design contract
---------------
1. **Physical size.**  Figures are built at their REAL printed size
   (Copernicus/AMT two-column layout on DIN A4).  The .tex must include them
   without scaling (``\\includegraphics{...}``, no ``width=``), then fonts
   print at exactly the size set here.  Widths:

   =========  ========  ==============================
   constant   cm        Copernicus figure class
   =========  ========  ==============================
   COL_W      8.3       one column  (the default)
   MID_W      12.0      1.5 columns
   FULL_W     17.0      full text width (two columns)
   =========  ========  ==============================

2. **One font size.**  ``FONT_SIZE`` is used for EVERYTHING (labels, ticks,
   legends, annotations) and equals the journal text size.  Tune the whole
   manuscript here, never per figure.

3. **Exact width.**  ``save()`` writes without a tight bounding box, so the
   PDF keeps the exact physical width.  Use ``layout="constrained"`` (the
   default from ``figure()``) so nothing overflows.

Typical figure script::

    from fluxpy import figstyle as fs
    fs.apply()
    fig, ax = fs.figure(width=fs.COL_W, aspect=0.8)
    ax.plot(x, y, color=fs.C["H_obs_bulk"], label=fs.L["H_obs_bulk"])
    fs.panel_label(ax, "(a)")
    fs.save(fig, "fig06_flux_scatter")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Physical geometry (inches; matplotlib wants inches) ─────────────────────
CM = 1.0 / 2.54
COL_W = 8.3 * CM        # 3.27 in — single column, the DEFAULT
MID_W = 12.0 * CM       # 4.72 in — 1.5 column
FULL_W = 17.0 * CM      # 6.69 in — full text width
MAX_H = 23.0 * CM       # keep every figure below this (A4 page minus caption)

# ── The one font size (pt).  Copernicus final print body ≈ 9 pt ─────────────
FONT_SIZE = 9.0

# ── Semantic colours, keyed by the fluxpy.naming variable names ─────────────
C = {
    "H_obs_ec":   "#222222",   # EC reference — near-black
    "H_obs_bulk": "#1b9e75",   # T-Bird bulk — CVD green
    "H_rs_bulk":  "#d95f02",   # ERA5–MODIS bulk — CVD orange
    "H_era_bulk": "#377eb8",   # ERA5-only bulk — CVD blue
    "H_era_diag": "#e7298a",   # ERA5 model-diagnosed — CVD magenta
    "H_car_diag": "#7570b3",   # CARRA model-diagnosed — CVD purple
}
GREY = "0.45"               # 1:1 lines, reference lines
GRID = "0.82"               # zero lines / faint guides

# Math-text legend/axis labels (single source: fluxpy.naming.PLOT_LABEL)
from fluxpy.naming import PLOT_LABEL as L          # noqa: E402, N811

FLUX_UNIT = r"(W m$^{-2}$)"
FLUX_AXIS = r"$H_\mathrm{S}$ (W m$^{-2}$)"

# ── Output directory (repo-root/figures_manuscript/output) ──────────────────
OUT_DIR = Path(__file__).resolve().parents[1] / "figures_manuscript" / "output"

# ── rcParams: everything styling lives HERE ──────────────────────────────────
RC = {
    # fonts — ONE size for everything
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "figure.labelsize": FONT_SIZE,
    "mathtext.fontset": "dejavusans",
    "mathtext.default": "regular",

    # house style: open axes, ticks out
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,

    # lines / markers
    "lines.linewidth": 1.2,
    "lines.markersize": 4,

    # legend
    "legend.frameon": False,
    "legend.borderpad": 0.3,
    "legend.labelspacing": 0.35,
    "legend.handlelength": 1.4,

    # layout: constrained by default → figures keep their EXACT physical size
    "figure.constrained_layout.use": True,
    "figure.dpi": 150,
    "savefig.dpi": 300,

    # vector output: embed TrueType fonts
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply() -> None:
    """Activate the manuscript style (call once, right after the imports)."""
    mpl.rcParams.update(RC)


def figure(width: float = COL_W, aspect: float = 0.75, **kwargs):
    """``plt.subplots`` at physical size: ``figsize=(width, width*aspect)``.

    Any extra kwargs (``nrows``, ``ncols``, ``sharex`` …) pass through.
    For a custom GridSpec layout use :func:`gridfig` instead — this function
    already creates axes, and adding a gridspec on top leaves a stray empty
    axes underneath that breaks constrained layout.
    """
    return plt.subplots(figsize=(width, width * aspect), **kwargs)


def gridfig(width: float = COL_W, aspect: float = 0.75, **kwargs):
    """Bare figure (NO axes) at physical size, for ``fig.add_gridspec`` layouts.

    Constrained layout is on via the style, so ``add_gridspec`` +
    ``add_subplot`` compose cleanly::

        fig = fs.gridfig(width=fs.COL_W, aspect=1.7)
        gs = fig.add_gridspec(3, 4, height_ratios=[2, 1, 1])
        ax = fig.add_subplot(gs[0, 0])
    """
    return plt.figure(figsize=(width, width * aspect), **kwargs)


def panel_label(ax, label: str, x: float = -0.02, y: float = 1.02) -> None:
    """Bold panel letter '(a)' at the top-left corner, outside the axes."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=FONT_SIZE,
            fontweight="bold", va="bottom", ha="right")


def save(fig, name: str, png: bool = True) -> Path:
    """Save ``OUT_DIR/name.pdf`` (+ ``.png``) at EXACT physical size.

    Deliberately NOT ``bbox_inches='tight'`` — a tight crop changes the
    physical width and silently rescales the fonts in print.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(pdf, metadata={"Creator": "BACSAM-II figures_manuscript"})
    if png:
        fig.savefig(OUT_DIR / f"{name}.png")
    w, h = fig.get_size_inches()
    print(f"{name}: {w / CM:.1f} x {h / CM:.1f} cm  ->  {pdf}")
    return pdf

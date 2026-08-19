"""Generate the manuscript LaTeX tables from the pipeline output.

Writes table bodies that are pasted into manuscript.tex, so the printed tables
can never drift from the analysis:

  * ``table_sensitivity_summary.tex`` -- Table 3 (summary of experiments),
    reported in the slope/correlation framework.
  * ``table_sensitivity_full.tex``    -- Table C1 (full matrix), now carrying
    slope and r instead of RMSE/bias alone.

Run:  pixi run python scripts/make_manuscript_tables.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from fluxpy.sensitivity import skill_metrics  # noqa: E402

OUT = REPO / "book" / "notebooks" / "figures"
df = pd.read_csv(REPO / "data/model_output/sensitivity_summary_table.csv")
FIG = xr.open_dataset(REPO / "data/model_output/sensitivity_figure_data.nc")
EC = np.asarray(FIG["H_obs_ec"])


def rng_str(sub, col, fmt="{:.2f}"):
    """min--max across a set of runs."""
    v = sub[col].astype(float)
    return f"{fmt.format(v.min())}--{fmt.format(v.max())}"


# ── Table 3: summary of the sensitivity experiments ──────────────────────────
schemes = df[df.label.str.endswith("_ref") & ~df.label.str.contains("era5only")]
alpha = df[df.label.str.contains("ztr")& ~df.label.str.contains("era5only")]
alpha = alpha[alpha.zt_ratio.between(1e-4, 1e-1)]
rough = df[df.label.str.contains("mosaic|sheba|nschnier") & ~df.label.str.contains("era5only")]
const = df[df.label.str.contains("z0c") & ~df.label.str.contains("era5only")]

rows = [
    ("Stability scheme (8 schemes)", schemes),
    (r"$\alpha_T$ ($10^{-4}$--$10^{-1}$)", alpha),
    ("Momentum roughness set", rough),
    (r"Constant $z_{0,m}$", const),
]
lines = []
for name, sub in rows:
    lines.append(
        f"{name} & {rng_str(sub, 'obs_slope')} & {rng_str(sub, 'obs_r')} & "
        f"{rng_str(sub, 'rs_slope')} & {rng_str(sub, 'rs_r')} \\\\")
tbl3 = "\n".join(lines)
(OUT / "table_sensitivity_summary.tex").write_text(tbl3 + "\n", encoding="utf-8")
print("── Table 3 body ──\n" + tbl3 + "\n")

# ── Table C1: full matrix with slope and r ───────────────────────────────────
GROUPS = [
    ("Reference configuration (conventional $\\alpha_T$)", ["GLGS20_ref"]),
    ("Amplitude-matched configuration", ["GLGS20_ztr1e-04"]),
    ("Stability scheme (all other settings at reference)",
     ["BD_ref", "HB88_ref", "BH91_ref", "CB05_ref", "GAFGP07_ref", "ZE07-I_ref", "ZE07-II_ref"]),
    ("Scalar roughness ratio $\\alpha_T$ (GLGS20)",
     [f"GLGS20_ztr1e-0{k}" for k in (2, 3, 5, 6, 7)]),
    ("Momentum roughness set (GLGS20)", ["GLGS20_mosaic", "GLGS20_sheba", "GLGS20_nschnier"]),
    ("Constant (surface-type-independent) $z_{0,m}$ (GLGS20)",
     ["GLGS20_z0c1e-04", "GLGS20_z0c2e-04", "GLGS20_z0c4e-04"]),
]
SCHEME_CITE = {
    "GLGS20_ref": r"\citet{gryanik_new_2020}", "GLGS20_ztr1e-04": "GLGS20",
    "BD_ref": r"\citet{businger_flux-profile_1971,dyer_review_1974}",
    "HB88_ref": r"\citet{holtslag_applied_1988}",
    "BH91_ref": r"\citet{beljaars_flux_1991}",
    "CB05_ref": r"\citet{chenge_flux-profile_2005}",
    "GAFGP07_ref": r"\citet{grachev_sheba_2007}",
    "ZE07-I_ref": r"\citet{zilitinkevich_similarity_2007}",
    "ZE07-II_ref": r"\citet{zilitinkevich_similarity_2007}",
}
out = []
for title, labels in GROUPS:
    out.append(r"\multicolumn{8}{l}{\textit{" + title + r"}} \\")
    for lab in labels:
        r = df[df.label == lab]
        if r.empty:
            print(f"  ! missing run: {lab}")
            continue
        r = r.iloc[0]
        bold = lab in ("GLGS20_ref", "GLGS20_ztr1e-04")
        w = (lambda s: r"\textbf{" + s + "}") if bold else (lambda s: s)
        scheme = SCHEME_CITE.get(lab, "GLGS20")
        a = f"{r.zt_ratio:.0e}".replace("e-0", "e-")
        z0 = "Ref." if lab not in ("GLGS20_ref",) and "z0c" not in lab and lab not in (
            "GLGS20_mosaic", "GLGS20_sheba", "GLGS20_nschnier") else ""
        if lab == "GLGS20_ref":
            z0 = r"$2{\times}10^{-4}$ / $1{\times}10^{-4}$ / $4{\times}10^{-4}$"
        elif "z0c" in lab:
            exp = int(np.floor(np.log10(r.z0_constant)))
            mant = r.z0_constant / 10.0 ** exp
            z0 = (rf"$z_{{0,m}} = {mant:.0f}{{\times}}10^{{{exp}}}$" + r"\,m (uniform)")
        elif lab in ("GLGS20_mosaic", "GLGS20_sheba", "GLGS20_nschnier"):
            z0 = {"GLGS20_mosaic": r"$3.3{\times}10^{-4}$ / $2{\times}10^{-4}$ / $3.3{\times}10^{-4}$",
                  "GLGS20_sheba": r"$3.3{\times}10^{-4}$ / $2{\times}10^{-4}$ / $2.3{\times}10^{-4}$",
                  "GLGS20_nschnier": r"$1{\times}10^{-4}$ / $1{\times}10^{-4}$ / $5{\times}10^{-3}$"}[lab]
        out.append(
            f"\\texttt{{{lab.replace('_', chr(92) + '_')}}} & {scheme} & {w(a)} & {z0} & "
            f"{w(f'{r.obs_slope:.2f}')} & {w(f'{r.obs_r:.2f}')} & "
            f"{w(f'{r.rs_slope:.2f}')} & {w(f'{r.rs_r:.2f}')} \\\\")
    out.append(r"\hline")
tblC = "\n".join(out)
(OUT / "table_sensitivity_full.tex").write_text(tblC + "\n", encoding="utf-8")
print("-- Table C1 body written --")
print(tblC[:1200])

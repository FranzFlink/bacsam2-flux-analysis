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

OUT = REPO / "tables_manuscript"
OUT.mkdir(exist_ok=True)
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


# ── Table (tab:bulk_unc): flux-uncertainty budget ────────────────────────────
# Ported unchanged from book/notebooks/tables/table02_instrument_uncertainties.ipynb.
H_ec = np.asarray(FIG["H_obs_ec"], dtype=float)
e_rel = np.asarray(FIG["e_rel_ec"], dtype=float)      # Lenschow relative sampling error
REL_INSTR = 0.05                                      # five-hole-probe flow calibration

sigma_sampling = np.abs(H_ec) * e_rel
sigma_instr = np.abs(H_ec) * REL_INSTR
sigma_placement = np.asarray(FIG["sigma_placement"], dtype=float)
sigma_ec = np.sqrt(sigma_sampling**2 + sigma_instr**2 + sigma_placement**2)

# assumed 1-sigma input uncertainties (manuscript Table tab:bulk_unc)
U_TS_KT19, U_TS_MODIS = 0.5, 1.5     # K
U_TA_INSITU, U_TA_ERA = 0.3, 1.2     # K
U_U_INSITU, U_U_ERA = 0.5, 1.7       # m s-1


def _central(vname, axname):
    """Central-difference dH/dx at zero perturbation from the perturbation sweeps."""
    sweep = np.asarray(FIG[vname], dtype=float)
    ax = np.asarray(FIG[axname], dtype=float)
    i0 = int(np.where(ax == 0.0)[0][0])
    return (sweep[i0 + 1] - sweep[i0 - 1]) / (ax[i0 + 1] - ax[i0 - 1])


sigma_obs_bulk = np.sqrt((_central("H_obs_dTsfc", "dT") * U_TS_KT19) ** 2
                         + (_central("H_obs_dTair", "dT") * U_TA_INSITU) ** 2
                         + (_central("H_obs_dU", "dU") * U_U_INSITU) ** 2)
sigma_rs_bulk = np.sqrt((_central("H_rs_dTsfc", "dT") * U_TS_MODIS) ** 2
                        + (_central("H_rs_dTair", "dT") * U_TA_ERA) ** 2
                        + (_central("H_rs_dU", "dU") * U_U_ERA) ** 2)


def _stat(x):
    v = x[np.isfinite(x)]
    return f"{np.median(v):.2f} [{np.percentile(v, 25):.2f}--{np.percentile(v, 75):.2f}]"


def _rel_stat(sig, flux, floor=2.0):
    rel = sig / np.abs(flux) * 100
    v = rel[np.isfinite(rel) & (np.abs(flux) >= floor)]
    return f"{np.median(v):.1f} [{np.percentile(v, 25):.1f}--{np.percentile(v, 75):.1f}]"


unc_rows = [
    (r"$H_\mathrm{S,obs,EC}$ total", sigma_ec, H_ec),
    (r"\quad Sampling", sigma_sampling, H_ec),
    (r"\quad Instrument", sigma_instr, H_ec),
    (r"\quad Segment placement", sigma_placement, H_ec),
    (r"$H_\mathrm{S,obs,bulk}$", sigma_obs_bulk, np.asarray(FIG["H_obs_bulk"], dtype=float)),
    (r"$H_\mathrm{S,RS,bulk}$", sigma_rs_bulk, np.asarray(FIG["H_rs_bulk"], dtype=float)),
]
tblU = "\n".join(f"{name} & {_stat(sig)} & {_rel_stat(sig, flux)} \\\\" for name, sig, flux in unc_rows)
(OUT / "table_uncertainty_budget.tex").write_text(tblU + "\n", encoding="utf-8")
print("\n% tab:bulk_unc  (sigma median [p25--p75] in W m-2 | sigma/|H| in %, |H| > 2 W m-2)")
print(tblU)

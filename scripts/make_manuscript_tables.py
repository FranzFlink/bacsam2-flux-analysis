"""The data-derived table bodies of the manuscript, written to tables_manuscript/.

    table_sensitivity_summary.tex   Table 4  (Sect. 4.2): range of slope and r over each sensitivity axis
    table_sensitivity_full.tex      Table C1 (Appendix C): every configuration, slope and r of both pathways
    table_uncertainty_budget.tex    Table 2  (Sect. 3.2): flux-uncertainty budget

The bodies are pasted into manuscript.tex, so the printed tables cannot drift
from the analysis. (Table A1 is written by scripts/ec_quality.py.)

Run:  pixi run tables
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from fluxpy.bulk import amplitude_matched_config  # noqa: E402
from fluxpy.uncertainty import uncertainty_budget  # noqa: E402

MATCHED = amplitude_matched_config().label          # the amplitude-matched row of Table C1 (bold)

OUT = REPO / "tables_manuscript"
OUT.mkdir(exist_ok=True)
SUMMARY = pd.read_csv(REPO / "data/model_output/sensitivity_summary_table.csv")   # one row per configuration
FIG = xr.open_dataset(REPO / "data/model_output/sensitivity_figure_data.nc")


def write(name: str, body: str) -> None:
    (OUT / name).write_text(body + "\n", encoding="utf-8", newline="\n")


# ── Table 4: range over each sensitivity axis ─────────────────────────────────

def value_range(rows, column, fmt="{:.2f}"):
    """min--max of a metric across a set of configurations."""
    v = rows[column].astype(float)
    return f"{fmt.format(v.min())}--{fmt.format(v.max())}"


modis_forcing = SUMMARY[~SUMMARY.label.str.contains("era5only")]        # Table 4 uses the ERA5-MODIS runs
axes = [
    ("Stability scheme (8 schemes)", modis_forcing[modis_forcing.label.str.endswith("_ref")]),
    (r"$\alpha_T$ ($10^{-4}$--$10^{-1}$)", modis_forcing[modis_forcing.label.str.contains("ztr")
                                                          & modis_forcing.zt_ratio.between(1e-4, 1e-1)]),
    ("Momentum roughness set", modis_forcing[modis_forcing.label.str.contains("mosaic|nschnier")]),
    (r"Constant $z_{0,m}$", modis_forcing[modis_forcing.label.str.contains("z0c")]),
]
table3 = "\n".join(f"{name} & {value_range(rows, 'obs_slope')} & {value_range(rows, 'obs_r')} & "
                   f"{value_range(rows, 'rs_slope')} & {value_range(rows, 'rs_r')} \\\\" for name, rows in axes)
write("table_sensitivity_summary.tex", table3)
print("── Table 4 body ──\n" + table3 + "\n")


# ── Table C1: every configuration ─────────────────────────────────────────────

GROUPS = [
    ("Reference configuration (conventional $\\alpha_T$)", ["GLGS20_ref"]),
    ("Amplitude-matched configuration", [MATCHED]),
    ("Stability scheme (all other settings at reference)",
     ["BD_ref", "HB88_ref", "BH91_ref", "CB05_ref", "GAFGP07_ref", "ZE07-I_ref", "ZE07-II_ref"]),
    ("Scalar roughness ratio $\\alpha_T$ (GLGS20)",
     [f"GLGS20_ztr1e-0{k}" for k in (2, 3, 4, 5, 6, 7) if f"GLGS20_ztr1e-0{k}" != MATCHED]),
    ("Momentum roughness set (GLGS20)", ["GLGS20_mosaic", "GLGS20_nschnier"]),
    ("Constant (surface-type-independent) $z_{0,m}$ (GLGS20)",
     ["GLGS20_z0c1e-04", "GLGS20_z0c2e-04", "GLGS20_z0c4e-04"]),
]
SCHEME_CITATION = {
    "GLGS20_ref": r"\citet{gryanik_new_2020}",
    "BD_ref": r"\citet{businger_flux-profile_1971,dyer_review_1974}",
    "HB88_ref": r"\citet{holtslag_applied_1988}",
    "BH91_ref": r"\citet{beljaars_flux_1991}",
    "CB05_ref": r"\citet{chenge_flux-profile_2005}",
    "GAFGP07_ref": r"\citet{grachev_sheba_2007}",
    "ZE07-I_ref": r"\citet{zilitinkevich_similarity_2007}",
    "ZE07-II_ref": r"\citet{zilitinkevich_similarity_2007}",
}
ROUGHNESS_COLUMN = {      # z0m of open water / thin ice / snow-covered ice, as printed
    "GLGS20_ref": r"$2{\times}10^{-4}$ / $1{\times}10^{-4}$ / $4{\times}10^{-4}$",
    "GLGS20_mosaic": r"$3.3{\times}10^{-4}$ / $2{\times}10^{-4}$ / $3.3{\times}10^{-4}$",
    "GLGS20_nschnier": r"$1{\times}10^{-4}$ / $1{\times}10^{-4}$ / $5{\times}10^{-3}$",
}


def roughness_cell(label: str, z0_constant) -> str:
    if label in ROUGHNESS_COLUMN:
        return ROUGHNESS_COLUMN[label]
    if "z0c" in label:
        exponent = int(np.floor(np.log10(z0_constant)))
        mantissa = z0_constant / 10.0 ** exponent
        return rf"$z_{{0,m}} = {mantissa:.0f}{{\times}}10^{{{exponent}}}$" + r"\,m (uniform)"
    return "Ref."


lines = []
for title, labels in GROUPS:
    lines.append(r"\multicolumn{8}{l}{\textit{" + title + r"}} \\")
    for label in labels:
        match = SUMMARY[SUMMARY.label == label]
        if match.empty:
            print(f"  ! missing run: {label}")
            continue
        row = match.iloc[0]
        bold = label in ("GLGS20_ref", MATCHED)
        emph = (lambda s: r"\textbf{" + s + "}") if bold else (lambda s: s)
        alpha = f"{row.zt_ratio:.0e}".replace("e-0", "e-")
        lines.append(
            f"\\texttt{{{label.replace('_', chr(92) + '_')}}} & {SCHEME_CITATION.get(label, 'GLGS20')} & {emph(alpha)} & "
            f"{roughness_cell(label, row.z0_constant)} & "
            f"{emph(f'{row.obs_slope:.2f}')} & {emph(f'{row.obs_r:.2f}')} & "
            f"{emph(f'{row.rs_slope:.2f}')} & {emph(f'{row.rs_r:.2f}')} \\\\")
    lines.append(r"\hline")
tableC1 = "\n".join(lines)
write("table_sensitivity_full.tex", tableC1)
print("-- Table C1 body written --")
print(tableC1[:1200])


# ── Table 2: flux-uncertainty budget ─────────────────────────────────────────

BUDGET = uncertainty_budget(FIG)
H_ec = np.asarray(FIG["H_obs_ec"], dtype=float)


def absolute(sigma):
    v = sigma[np.isfinite(sigma)]
    return f"{np.median(v):.2f} [{np.percentile(v, 25):.2f}--{np.percentile(v, 75):.2f}]"


def relative(sigma, flux, floor=2.0):
    """sigma / |H| in %, over the segments with |H| >= 2 W m-2."""
    rel = sigma / np.abs(flux) * 100
    v = rel[np.isfinite(rel) & (np.abs(flux) >= floor)]
    return f"{np.median(v):.0f} [{np.percentile(v, 25):.0f}--{np.percentile(v, 75):.0f}]"


budget_rows = [
    (r"$H_\mathrm{S,obs,EC}$ total", BUDGET["ec_total"], H_ec),
    (r"\quad Random (sampling) error", BUDGET["ec_random"], H_ec),
    (r"\quad Instrument", BUDGET["ec_instrument"], H_ec),
    (r"\quad \textit{cross-check: segment placement}", BUDGET["ec_placement"], H_ec),
    (r"$H_\mathrm{S,obs,bulk}$", BUDGET["obs_bulk"], np.asarray(FIG["H_obs_bulk"], dtype=float)),
    (r"$H_\mathrm{S,RS,bulk}$", BUDGET["rs_bulk"], np.asarray(FIG["H_rs_bulk"], dtype=float)),
]
table2 = "\n".join(f"{name} & {absolute(sigma)} & {relative(sigma, flux)} \\\\" for name, sigma, flux in budget_rows)
write("table_uncertainty_budget.tex", table2)
print("\n% tab:bulk_unc  (sigma median [p25--p75] in W m-2 | sigma/|H| in %, |H| > 2 W m-2)")
print(table2)

---
title: Paper and code
---

Each module, script and figure script names the manuscript section it implements in its first lines. The table gives
the same mapping from the side of the paper; module and script names link to the [API reference](api/index.md).

| Paper | Code |
|-------|------|
| Sect. 2.1, 5 km segments | [`fluxpy.aggregate`](api/fluxpy.aggregate.md), [`scripts/run_ec_aggregation.py`](#scripts/run_ec_aggregation.py); leg table: [`preprocessing/flight_segments.py`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/preprocessing/flight_segments.py) |
| Sect. 2.2–2.3, reanalysis and satellite inputs, Fig. 2 | stage 2 of [`scripts/calc_bulk_and_ec_fluxes.py`](#scripts/calc_bulk_and_ec_fluxes.py); [`figures_manuscript/fig02_meteo_overview.py`](#figures_manuscript/fig02_meteo_overview.py) |
| Sect. 3.1, eddy covariance, Eq. (1) | [`fluxpy.eddy_covariance`](api/fluxpy.eddy_covariance.md) (band-limited covariance, random error, [`segment_flux`](#fluxpy.eddy_covariance.segment_flux)) |
| Sect. 3.1, window length (Fig. 3), placement spread | [`scripts/ec_window_length.py`](#scripts/ec_window_length.py), [`scripts/ec_segment_placement.py`](#scripts/ec_segment_placement.py) |
| Sect. 3.2 and Appendix B, bulk method, Eq. (2) | [`fluxpy.bulk`](api/fluxpy.bulk.md) (B1 Richardson number, B2 roughness, B3 transfer coefficient), [`fluxpy.stability`](api/fluxpy.stability.md), [`fluxpy.stability_lib`](api/fluxpy.stability_lib.md); constants and pressures in [`fluxpy.util`](api/fluxpy.util.md) |
| Table 2, uncertainty budget | [`fluxpy.uncertainty`](api/fluxpy.uncertainty.md), [`scripts/make_manuscript_tables.py`](#scripts/make_manuscript_tables.py) |
| Sect. 4.1, comparison, Table 4, Figs. 4–6 | [`scripts/manuscript_numbers.py`](#scripts/manuscript_numbers.py), [`fluxpy.sensitivity`](api/fluxpy.sensitivity.md) (skill metrics), [figure scripts](api/scripts.md) `fig04`–`fig06` |
| Sect. 4.2, sensitivity, Table 3, Fig. 7, Appendix C | [`fluxpy.sensitivity`](api/fluxpy.sensitivity.md), [`scripts/make_manuscript_tables.py`](#scripts/make_manuscript_tables.py), [`figures_manuscript/fig07_sensitivity.py`](#figures_manuscript/fig07_sensitivity.py) |
| Appendix A, quality of the EC reference, Fig. A1, Table A1 | [`fluxpy.qc`](api/fluxpy.qc.md), [`scripts/ec_quality.py`](#scripts/ec_quality.py), [`figures_manuscript/figA1_spectral_qc.py`](#figures_manuscript/figA1_spectral_qc.py) |

## Printed numbers

No number in the paper is transcribed by hand. Every number of the text is printed by `scripts/manuscript_numbers.py`
(text and Table 4) or `scripts/ec_quality.py` (Appendix A); the bodies of Tables 2, 3, C1 and A1 are written by
`scripts/make_manuscript_tables.py` and `scripts/ec_quality.py` into `tables_manuscript/`. The output of the number
scripts, together with that of the robustness checks, is committed as `expected_numbers.txt`; `pixi run check` fails if
any number deviates from it by more than 0.02.

## Annotated manuscript

[`paper/manuscript_clean.tex`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/paper/manuscript_clean.tex)
is the manuscript without comments and template text, with cross-references and citations resolved as in the PDF, and
with a `% [code]` line after each section, equation, figure and table that names the code producing it.
[`AGENTS.md`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/AGENTS.md) describes how to check a
statement of the paper against the code and lists the conventions of the analysis.

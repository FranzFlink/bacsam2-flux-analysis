---
title: Review the paper against the code
short_title: Paper → code
---

Every module, script and figure file names the manuscript section it implements in its first lines. The
table gives the same map from the side of the paper. Paths link to the repository on GitHub.

| Paper | Code |
|-------|------|
| Sect. 2.1, 5 km segments | [`fluxpy/aggregate.py`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/fluxpy/aggregate.py), [`scripts/run_ec_aggregation.py`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/scripts/run_ec_aggregation.py); leg table: [`preprocessing/flight_segments.py`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/preprocessing/flight_segments.py) |
| Sect. 2.2–2.3 reanalysis and satellite inputs, Fig. 2 | stage 2 of [`scripts/calc_bulk_and_ec_fluxes.py`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/scripts/calc_bulk_and_ec_fluxes.py); [`figures_manuscript/fig02_meteo_overview.py`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/figures_manuscript/fig02_meteo_overview.py) |
| Sect. 3.1 eddy covariance, Eq. (1) | [`fluxpy/eddy_covariance.py`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/fluxpy/eddy_covariance.py) (band-limited covariance, random error, `segment_flux`) |
| Sect. 3.1 window length (Fig. 3), placement spread | [`scripts/ec_window_length.py`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/scripts/ec_window_length.py), [`scripts/ec_segment_placement.py`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/scripts/ec_segment_placement.py) |
| Sect. 3.2 and Appendix B bulk method, Eq. (2) | [`fluxpy/bulk.py`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/fluxpy/bulk.py) (B1 Richardson number, B2 roughness, B3 transfer coefficient), [`fluxpy/stability.py`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/fluxpy/stability.py), [`fluxpy/stability_lib.py`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/fluxpy/stability_lib.py), constants and pressures in [`fluxpy/util.py`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/fluxpy/util.py) |
| Table 2 uncertainty budget | [`fluxpy/uncertainty.py`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/fluxpy/uncertainty.py), [`scripts/make_manuscript_tables.py`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/scripts/make_manuscript_tables.py) |
| Sect. 4.1 comparison, Table 4, Figs. 4–6 | [`scripts/manuscript_numbers.py`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/scripts/manuscript_numbers.py), `figures_manuscript/fig04`–`fig06` |
| Sect. 4.2 sensitivity, Table 3, Fig. 7, Appendix C | [`fluxpy/sensitivity.py`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/fluxpy/sensitivity.py), [`scripts/make_manuscript_tables.py`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/scripts/make_manuscript_tables.py), [`figures_manuscript/fig07_sensitivity.py`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/figures_manuscript/fig07_sensitivity.py) |
| Appendix A quality of the EC reference, Fig. A1, Table A1 | [`fluxpy/qc.py`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/fluxpy/qc.py), [`scripts/ec_quality.py`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/scripts/ec_quality.py), [`figures_manuscript/figA1_spectral_qc.py`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/figures_manuscript/figA1_spectral_qc.py) |
| robustness of the comparison ([Chapter 6](../book/06_robustness.ipynb)) | [`scripts/robustness_checks.py`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/scripts/robustness_checks.py), [`scripts/ec_band_sensitivity.py`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/scripts/ec_band_sensitivity.py) |

## The number contract

Nothing in the paper is typed by hand from a notebook. Every number of the text is printed by
`scripts/manuscript_numbers.py` (text and Table 4) or `scripts/ec_quality.py` (Appendix A); the bodies
of Tables 2, 3, C1 and A1 are written by `scripts/make_manuscript_tables.py` and `scripts/ec_quality.py`
into `tables_manuscript/`. The output of the number scripts, together with the robustness checks, is
committed as `expected_numbers.txt`, and `pixi run check` fails if any number moves by more than 0.02.
A change of the code that moves a number is committed on its own, with the numbers before and after in
the commit message and in the [changelog](changelog.md).

## A reading copy of the manuscript

[`paper/manuscript_clean.tex`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/paper/manuscript_clean.tex)
is the manuscript without comments or template boilerplate, with cross-references and citations resolved as
in the PDF, and with a `% [code]` line after every section, equation, figure and table that names the code
producing it. It is written for reviewers and for AI agents that compare the paper with the code;
[`AGENTS.md`](https://github.com/FranzFlink/bacsam2-flux-analysis/blob/main/AGENTS.md) explains how to
check a claim and which conventions are deliberate.

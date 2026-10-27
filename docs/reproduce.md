---
title: Reproduction
---

The repository contains the data (about 300 MB) and a locked Python environment. A single command recomputes every
data-derived number, table and figure of the paper and compares the numbers with the committed reference
`expected_numbers.txt`.

## Procedure

1. **Obtain the repository**: on GitHub, *Code* → *Download ZIP* and unpack, or
   `git clone https://github.com/FranzFlink/bacsam2-flux-analysis.git`.
2. **Install [pixi](https://pixi.sh)**, which sets up the exact environment of the paper without modifying the rest of
   the system. The environment is locked for Windows, Linux and macOS.
3. **Run** in a terminal in the repository folder (on Windows, enter `cmd` in the address bar of the Explorer window):

   ```
   pixi run all
   ```

   The first run downloads the environment (a few minutes); Figures 1, 4 and 5 download map data once. The run stops
   with an error if any number differs from the reference.

## Steps

| Step | Content | Time |
|------|---------|------|
| `test` | The physics against independent references (numerical integration of the stability functions, log-law limit, Parseval, a Monte-Carlo test of the random error, the hydrostatic equation, the ECMWF level table) | 5 s |
| `verify-ec` | The eddy-covariance fluxes follow from the 100 Hz T-Bird data | 30 s |
| `verify-cache` | The MODIS and ERA5 values at the segments follow from the shipped gridded fields | 5 s |
| `precompute` | Bulk fluxes for every forcing and every configuration of the sensitivity analysis | 5 s |
| `check` | Every printed number equals `expected_numbers.txt` (text, Table 4, Appendix A, robustness checks) | 1 min |
| `tables` | LaTeX bodies of Tables 2, 3 and C1 → `tables_manuscript/` | 5 s |
| `figures` | All manuscript figures at print size → `figures_manuscript/output/` | 3 min |

Individual steps can be run separately, for example `pixi run numbers` (numbers of the text), `pixi run ec-quality`
(Appendix A), `pixi run robustness` and `pixi run ec-bands` (robustness checks),
`pixi run book` (re-execute the chapters of this site), `pixi run api` (regenerate the
[API reference](api/index.md) and the [figure pages](figures/index.md)) and `pixi run site` (build this site into `_build/html`). `pixi run all-from-raw`
additionally regenerates the 5 km segments, the segment-placement spread and the window-length experiment from the
100 Hz data (about 10 minutes). The [scripts page](api/scripts.md) describes each step.

## Data not included

* `preprocessing/` documents how the shipped files were extracted from the raw archives and requires those archives.

Raw data are available from the authors on request.

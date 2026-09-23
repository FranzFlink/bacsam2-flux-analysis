---
title: Reproduce every number
---

The repository contains the data (about 300 MB) and a locked Python environment. One command rebuilds
every number, table and figure of the paper that is derived from data, and checks the numbers against the
committed reference `expected_numbers.txt`.

## Run it

1. **Get the repository**: on GitHub, *Code* → *Download ZIP* and unpack, or
   `git clone https://github.com/FranzFlink/bacsam2-flux-analysis.git`.
2. **Install [pixi](https://pixi.sh)**, a small program that sets up the exact environment of the paper
   without touching anything else on the computer. The environment is locked for Windows, Linux and macOS.
3. **Open a terminal in the folder** (Windows: type `cmd` in the address bar of the Explorer window) and run

   ```
   pixi run all
   ```

   The first call downloads the environment (a few minutes). Figures 1, 4 and 5 fetch map data once.
   The run stops with an error if any number differs from the reference.

## What `pixi run all` does

| Step | What it shows | Time |
|------|---------------|------|
| `test` | The physics against independent references (numerical integration of the stability functions, log-law limit, Parseval, a Monte-Carlo test of the random error, the hydrostatic equation, the ECMWF level table) | 5 s |
| `verify-ec` | The eddy-covariance fluxes follow from the 100 Hz T-Bird data | 30 s |
| `verify-cache` | The MODIS and ERA5 values at the segments follow from the shipped gridded fields | 5 s |
| `precompute` | Bulk fluxes for every forcing and every configuration of the sensitivity analysis | 5 s |
| `check` | Every printed number equals `expected_numbers.txt` (text, Table 4, Appendix A, robustness checks) | 1 min |
| `tables` | LaTeX bodies of Tables 2, 3, C1 → `tables_manuscript/` | 5 s |
| `figures` | All manuscript figures at print size → `figures_manuscript/output/` | 3 min |

Single steps run on their own, for example `pixi run numbers` (print the numbers of the text),
`pixi run ec-quality` (Appendix A), `pixi run robustness` and `pixi run ec-bands` (the checks of
[Chapter 6](../book/06_robustness.ipynb)), `pixi run book` (re-execute the chapters of this site) and
`pixi run site` (build this site into `_build/html`). `pixi run all-from-raw` additionally regenerates the
5 km segments, the segment-placement spread and the window-length experiment from the 100 Hz data (about
10 minutes).

## What cannot be reproduced from the repository

* The CARRA surface sensible heat flux at the segments is shipped in `along_track_inputs.nc`; the CARRA
  forecast file it was extracted from is not included.
* The Polar 6 aircraft-level temperature and height per segment (Fig. 2a) are shipped in
  `polar6_level_at_segments.nc`; the Polar 6 nose-boom archive is not included.
* `preprocessing/` documents how the shipped files were cut from the raw archives; it needs those archives.

Raw data are available from the authors on request.

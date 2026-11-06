---
title: Reproduction
---

The repository contains the code and a locked Python environment, but no data: the first steps download the
measurements from PANGAEA and every other input from a Zenodo data record (doi:10.5281/zenodo.23729591). A single command
recomputes every data-derived number, table and figure of the paper and compares the numbers with the committed
reference `expected_numbers.txt`.

## Procedure

1. **Obtain the repository**: on GitHub, *Code* → *Download ZIP* and unpack, or
   `git clone https://github.com/FranzFlink/bacsam2-flux-analysis.git`.
2. **Install [pixi](https://pixi.sh)**, which sets up the exact environment of the paper without modifying the rest of
   the system. The environment is locked for Windows, Linux and macOS.
3. **Run** in a terminal in the repository folder (on Windows, enter `cmd` in the address bar of the Explorer window):

   ```
   pixi run all
   ```

   The first run downloads the environment (a few minutes), the published measurements (about 1.5 GB) and the
   Zenodo data record (about 0.2 GB); Figures 1, 4 and 5 download map data once. The run stops with an error if
   any number differs from the reference.

## Steps

| Step | Content | Time |
|------|---------|------|
| `fetch-pangaea` | The published T-Bird, nose-boom and KT-19 files (doi:10.1594/PANGAEA.995979) → `data/pangaea/`, and the Polar 6 master tracks (doi:10.1594/PANGAEA.971763), checked against their checksums; files already present are only checked | download, then 10 s |
| `fetch-zenodo` | The Zenodo data record (ERA5 and CARRA subsets, MODIS ice surface temperature, sea-ice concentration, NASA GIBS images, leg table) → the folders the scripts read, checked against `data/manifest.tsv` | download |
| `inputs` | These files cut to the windows around the low-level legs → `data/observations/processed/` | 1 min |
| `test` | The physics against independent references (numerical integration of the stability functions, log-law limit, Parseval, a Monte-Carlo test of the random error, the hydrostatic equation, the ECMWF level table) | 5 s |
| `ec-segments` | The 5 km segments and their eddy-covariance fluxes from the 100 Hz T-Bird data | 1 min |
| `placement`, `window-length` | Segment-placement spread and window-length experiment (Fig. 3) | 5 and 2 min |
| `polar6-level` | Polar 6 temperature and height per segment (Fig. 2a) | 10 s |
| `extract` | MODIS, ERA5 and CARRA values at the segments from the gridded fields | 5 s |
| `precompute` | Bulk fluxes for every forcing and every configuration of the sensitivity analysis | 5 s |
| `check` | Every printed number equals `expected_numbers.txt` (text, Table 3, Appendix A, robustness checks) | 1 min |
| `tables` | LaTeX bodies of Tables 2, 3 and C1 → `tables_manuscript/` | 5 s |
| `figures` | All manuscript figures at print size → `figures_manuscript/output/` | 3 min |

Individual steps can be run separately, for example `pixi run numbers` (numbers of the text), `pixi run ec-quality`
(Appendix A), `pixi run robustness` and `pixi run ec-bands` (robustness checks),
`pixi run book` (re-execute the chapters of this site), `pixi run api` (regenerate the
[API reference](api/index.md) and the [figure pages](figures/index.md)) and `pixi run site` (build this site into `_build/html`).
The [scripts page](api/scripts.md) describes each step.

## Data not included

* `preprocessing/` documents how the files of the Zenodo data record were extracted from the raw archives and requires
  those archives.

Raw data are available from the authors on request.

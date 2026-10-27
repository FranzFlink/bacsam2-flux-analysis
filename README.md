# BACSAM-II sensible-heat-flux analysis

Code and data for **Müller et al., "Evaluation of reanalysis and satellite products of
sensible heat flux over Arctic sea ice against airborne eddy-covariance measurements"**
(submitted to *Atmospheric Measurement Techniques*).

The repository reproduces, from the shipped data, every number, table and figure of the
paper that is derived from data:

| Step | What it shows | Time |
|------|---------------|------|
| `verify-ec` | The eddy-covariance (EC) reference fluxes follow from the 100 Hz T-Bird measurements | ~30 s |
| `placement`, `window-length` | EC segment-placement uncertainty and the window-length experiment (Fig. 3); part of `all-from-raw` | ~2 min each |
| `ec-quality` | Every number of Appendix A: robustness to screening, random-error cross-checks, physical consistency, platform resonance | ~1 min |
| `verify-cache` | The MODIS and ERA5 values at the segments follow from the shipped gridded fields | ~5 s |
| `precompute` | Bulk fluxes for all forcings + the full sensitivity sweep | ~5 s |
| `robustness`, `ec-bands` | Robustness of the comparison: dependent segments, zero-flux baseline, the genuine-MODIS subset, stability correction, the frequency band of the EC flux | ~10 s |
| `check` | Every printed number (text, Table 4, Appendix A, robustness checks) equals the committed reference `expected_numbers.txt` | ~1 min |
| `tables` | LaTeX bodies of the data-derived tables → `tables_manuscript/` | ~5 s |
| `figures` | All manuscript figures → `figures_manuscript/output/` | ~3 min |

**Website.** The electronic supplement of the paper (the method, the figures, the reproduction guide,
a map from the paper to the code and the API reference of `fluxpy` and the scripts) is a website:
`pixi run site` builds it into `_build/html` (`pixi run site-serve` for a live preview); on GitHub it is
published from `.github/workflows/site.yml`. `pixi run api` regenerates the API reference and the
figure pages from the code.

**Checking the paper against the code** (also with an AI agent): start with [`AGENTS.md`](AGENTS.md); a
clean reading copy of the manuscript, annotated with the code that produces each part, is in
[`paper/`](paper/README.md).

## How to run it (no programming knowledge needed)

1. **Get the repository** (about 300 MB). On GitHub: green *Code* button → *Download ZIP*,
   then unpack. Or, with git: `git clone https://github.com/FranzFlink/bacsam2-flux-analysis.git`
2. **Install pixi** (a small program that sets up the exact Python environment used for
   the paper; nothing else on your computer is changed). Follow the one-line instruction
   for your system at <https://pixi.sh>. The environment is locked for Windows, Linux and
   macOS (tested on Windows).
3. **Open a terminal in the unpacked folder**
   (Windows: open the folder in Explorer, type `cmd` in the address bar, press Enter).
4. **Run everything:**

   ```
   pixi run all
   ```

   The first call downloads the Python environment (a few minutes). `verify-ec` and `check`
   end with a `RESULT:` line; the run stops with an error if anything does not match.
   Figures 1, 4 and 5 download map data on first use, so an internet connection is needed once.

Single steps: `pixi run test`, `pixi run verify-ec`, `pixi run precompute`, `pixi run check`,
`pixi run numbers`, `pixi run robustness`, `pixi run tables`, `pixi run figures`, `pixi run book`,
`pixi run api`, `pixi run site`. `pixi run all-from-raw` additionally
rewrites all EC products (segments, placement uncertainty, window-length experiment) from the
100 Hz data before everything else (~10 min).

Re-running rewrites the output files with a new time stamp; `git status` will then list
them as modified although the numbers are identical.

## Where is what in the paper?

| Paper | Code |
|-------|------|
| Sect. 3.1 Eddy-covariance method, Eq. (1) | `fluxpy/eddy_covariance.py` (band-limited covariance, random error, `segment_flux`); the 5 km segments: `fluxpy/aggregate.py`, `scripts/run_ec_aggregation.py` |
| Sect. 2.3 thermodynamic inputs of the bulk estimates (Fig. 2) | `figures_manuscript/fig02_meteo_overview.py` |
| Sect. 3.1 window length (Fig. 3), placement spread | `scripts/ec_window_length.py`, `scripts/ec_segment_placement.py` |
| Sect. 3.2 and Appendix B Bulk method, Eq. (2) | `fluxpy/bulk.py` (B1 Richardson number, B2 roughness, B3 transfer coefficient), `fluxpy/stability.py` (Eq. B2, effective wind), `fluxpy/stability_lib.py` (stability functions) |
| Table 2 uncertainty budget | `fluxpy/uncertainty.py`, `scripts/make_manuscript_tables.py` |
| Sect. 4.1 comparison, Table 4, Figs. 4-6 | `scripts/calc_bulk_and_ec_fluxes.py` (all estimates on the segments), `scripts/manuscript_numbers.py`, `figures_manuscript/fig04-06` |
| Sect. 4.2 sensitivity, Table 3, Fig. 7, Appendix C | `fluxpy/sensitivity.py` (configurations, skill metrics), `scripts/make_manuscript_tables.py`, `figures_manuscript/fig07_sensitivity.py` |
| Appendix A quality of the EC reference, Fig. A1, Table A1 | `fluxpy/qc.py` (integrity check), `scripts/ec_quality.py`, `figures_manuscript/figA1_spectral_qc.py` |
| every number in the text | `scripts/manuscript_numbers.py` and `scripts/ec_quality.py`, whose output is committed as `expected_numbers.txt` |
| robustness of the comparison | `scripts/robustness_checks.py`, `scripts/ec_band_sensitivity.py` |

Every module starts with a docstring that names the manuscript section it implements;
`paper/manuscript_clean.tex` carries the same map inside the text of the paper.

## What is where

| Path | Content |
|------|---------|
| `fluxpy/` | **All physics**: EC flux and random error (`eddy_covariance.py`), integrity check and spectra (`qc.py`), 5 km segments (`aggregate.py`), bulk flux (`bulk.py`), bulk Richardson number (`stability.py`), stability functions and transfer coefficients (`stability_lib.py`), sensitivity sweep and skill metrics (`sensitivity.py`), uncertainty budget (`uncertainty.py`) |
| `scripts/run_ec_aggregation.py` | 100 Hz data → 5 km EC segments (`fluxpy/aggregate.py`) |
| `scripts/ec_segment_placement.py`, `scripts/ec_window_length.py` | EC segment-placement spread; EC window-length experiment (Fig. 3) |
| `scripts/ec_quality.py` | Quality of the EC reference (Appendix A) → `tables_manuscript/table_qc_robustness.tex` |
| `scripts/verify_ec_reproduction.py` | Recomputes the EC segments and compares with the shipped segment file |
| `scripts/calc_bulk_and_ec_fluxes.py` | Bulk fluxes + sensitivity sweep → `data/model_output/` |
| `scripts/manuscript_numbers.py` | Prints the numbers quoted in the text and in the comparison table |
| `scripts/make_manuscript_tables.py` | Sensitivity tables and the uncertainty-budget table |
| `scripts/robustness_checks.py`, `scripts/ec_band_sensitivity.py` | Robustness of the comparison |
| `figures_manuscript/` | One script per manuscript figure (see its README) |
| `tests/` | Unit tests; `test_most_schemes.py` checks the physics against independent references (numerical integration of the published stability functions, log-law limit, sign convention, integrity threshold of Appendix A, Monte-Carlo check of the random error); `test_constants_and_theta.py` pins the constants to the values stated in the paper |
| `preprocessing/` | How the shipped data files were produced from the raw archives, including the leg table (documentation; needs the raw data) |
| `book/` | Executable chapters on the method (see below) |
| `docs/`, `myst.yml` | The pages of the website around the book (landing page, reproduction guide, paper-to-code map, glossary); `docs/build_api.py` writes the API reference into `docs/api/` and the figure pages into `docs/figures/` |
| `paper/` | Annotated reading copy of the manuscript, and the script that writes it |
| `AGENTS.md` | How to check a claim of the paper against the code; conventions and rules for changes |
| `DATA.md` | Description, units and origin of every data file |

## Method chapters

The [book](book/README.md) consists of five notebooks that lead from the 100 Hz measurements to the
paper's conclusions and compute one segment by hand at each step. `pixi run book` re-executes them;
`pixi run site` turns them into the website.

## Reference configuration

GLGS20 stability functions (Gryanik et al., 2020), scalar roughness ratio
α_T = z₀,T/z₀,m = 10⁻¹, surface-type-dependent momentum roughness, classic two-point bulk
Richardson number between the surface and the measurement level. The amplitude-matched
configuration uses α_T = 10⁻⁴. All fluxes are positive upward. 90 low-level 5 km segments over
sea ice. All pass the integrity check (resolved inertial subrange). Every EC flux carries a measured random error
(median 0.75 W m⁻²); 67, 46 and 33 of the 90 fluxes exceed 1, 2 and 3 times this error. All fluxes are kept: those that do
not stand out from their random error are valid measurements of a small flux.
Statistics (slopes, correlations, biases, errors and uncertainties, sensitivities) are reported with two decimals, so that
related numbers stay consistent (e.g. r = 0.36 and r² = 0.13); flux values, temperatures and Ri_b with one.

Two implementation details worth knowing:

* Every potential temperature uses the pressure at its own height. No surface pressure is
  measured in situ, so the surface pressure of the in-situ estimate is reconstructed from the
  T-Bird pressure (≈ 24 m above the surface) with the hypsometric equation
  (`fluxpy.util.surface_pressure_hypsometric`). For ERA5, the air temperature of model level 136
  uses the pressure of that level (`fluxpy.util.era5_level_pressure`), the surface temperature the
  surface pressure.
* The EC covariance is the part of the w–θ covariance carried by frequencies between 0.1 and
  20 Hz: linear detrend, cospectrum of the full segment, summed over the band
  (`fluxpy.eddy_covariance.band_covariance`; frequency resolution ≈ 0.012 Hz).

## What cannot be reproduced from this repository

* The CARRA surface sensible heat flux at the segments is shipped in `along_track_inputs.nc`;
  the CARRA forecast file it was extracted from is not part of the repository. (The MODIS and
  ERA5 entries of that file are reproducible from the shipped gridded files.)
* The Polar 6 aircraft-level temperature and height per segment (Fig. 2a) are shipped in
  `polar6_level_at_segments.nc`; the Polar 6 nose-boom archive they come from is not part of the
  repository.

Raw data are available from the authors on request.

## Citation and license

Please cite the paper and the archived version of this repository: every release is archived on
Zenodo with its own DOI. `CITATION.cff` has the reference of the paper, `.zenodo.json` the metadata
of the archive.

The code is released under the [MIT License](LICENSE). The files in `data/` are not covered by it:
third-party products (ERA5, CARRA, MODIS, the merged MODIS–AMSR2 sea-ice concentration, NASA GIBS
imagery) remain under the licences of their providers, see `DATA.md`. The drawing on the website
(`docs/images/polar6_tbird.png`, generated with ChatGPT from a photograph by Anna-Maria Jörs) is not
covered by the MIT License either.

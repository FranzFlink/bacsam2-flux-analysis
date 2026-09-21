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
| `precompute` | Bulk fluxes for all forcings + the full sensitivity sweep | ~15 s |
| `check` | The key numbers equal the committed reference `expected_numbers.txt` | ~15 s |
| `tables` | LaTeX bodies of the data-derived tables → `tables_manuscript/` | ~5 s |
| `figures` | All manuscript figures → `figures_manuscript/output/` | ~3 min |

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
   Figures 1 and 4 download coastline data on first use, so an internet connection is needed once.

Single steps: `pixi run test`, `pixi run verify-ec`, `pixi run precompute`, `pixi run check`,
`pixi run numbers`, `pixi run tables`, `pixi run figures`. `pixi run all-from-raw` additionally
rewrites all EC products (segments, placement uncertainty, window-length experiment) from the
100 Hz data before everything else (~10 min).

Re-running rewrites the output files with a new time stamp; `git status` will then list
them as modified although the numbers are identical.

## What is where

| Path | Content |
|------|---------|
| `fluxpy/` | **All physics**: bulk flux (`bulk.py`), stability functions and transfer coefficients (`stability_lib.py`, `stability.py`), EC covariance and spectra (`flux_processing.py`, `ec.py`), spectral quality control (`qc.py`), segment aggregation (`aggregate.py`), sensitivity sweep and skill metrics (`sensitivity.py`) |
| `scripts/run_ec_aggregation.py` | 100 Hz data → 5 km EC segments |
| `scripts/ec_segment_placement.py`, `scripts/ec_window_length.py` | EC segment-placement uncertainty; EC window-length experiment (Fig. 3) |
| `scripts/verify_ec_reproduction.py` | Recomputes the EC segments and compares with the shipped segment file |
| `scripts/calc_bulk_and_ec_fluxes.py` | Bulk fluxes + sensitivity sweep → `data/model_output/` |
| `scripts/manuscript_numbers.py` | Prints the numbers quoted in the text and in the comparison table |
| `scripts/make_manuscript_tables.py` | Sensitivity tables and the uncertainty-budget table |
| `figures_manuscript/` | One script per manuscript figure (see its README) |
| `tests/` | Unit tests; `test_most_schemes.py` checks the physics against independent references (numerical integration of the published stability functions, log-law limit, sign convention, QC thresholds of Appendix A) |
| `preprocessing/` | How the shipped data files were produced from the raw archives (documentation; needs the raw data) |
| `book/` | A step-by-step, executable walk through the method (see below) |
| `DATA.md` | Description, units and origin of every data file |

## Want to understand the method rather than just run it?

Read the [book](book/README.md): five short notebooks that walk from the 100 Hz measurements to the
paper's conclusions, computing one segment by hand at each step. `pixi run book` re-executes them.

## Reference configuration

GLGS20 stability functions (Gryanik et al., 2020), scalar roughness ratio
α_T = z₀,T/z₀,m = 10⁻¹, surface-type-dependent momentum roughness, classic two-point bulk
Richardson number between the surface and the measurement level. The amplitude-matched
configuration uses α_T = 10⁻⁴. All fluxes are positive upward. 91 low-level 5 km segments over
sea ice; the three-criterion spectral quality control accepts 73, rates 18 tolerable and discards none.
Numbers are reported with one decimal, matching their precision.

Two implementation details worth knowing:

* The surface potential temperature uses the pressure measured at the T-Bird (≈ 24 m above
  the surface) because no surface pressure measurement exists.
* The EC covariance is the part of the w–θ covariance carried by frequencies between 0.1 and
  20 Hz: linear detrend, cospectrum of the full segment, summed over the band
  (`fluxpy.flux_processing.band_covariance`; frequency resolution ≈ 0.012 Hz).

## What cannot be reproduced from this repository

* Fig. 2a additionally shows the aircraft (Polar 6) level. These segment means are shipped in
  `polar6_level_at_segments.nc`; the aircraft nose-boom archive itself is not part of the repository.
* The CARRA surface sensible heat flux at the segments is shipped in `along_track_inputs.nc`;
  the CARRA forecast file it was extracted from is not part of the repository. (The MODIS and
  ERA5 entries of that file are reproducible from the shipped gridded files.)

Raw data are available from the authors on request.

## Citation

See `CITATION.cff`.

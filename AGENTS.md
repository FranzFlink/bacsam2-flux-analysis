# Notes for agents (and people) reviewing this repository against the paper

This repository is the reproducibility package of Müller et al., "Evaluation of reanalysis and
satellite products of sensible heat flux over Arctic sea ice against airborne eddy-covariance
measurements" (AMT, submitted). Its job is to show that every number, table and figure of the
paper follows from the published data. Read this file first, then `README.md`.

## Where the paper is

* The manuscript is not part of this repository. `README.md` ("Where is what in the paper?") maps
  every section, equation, table and figure to the code; every module and script names in its
  docstring the section it implements; the figure pages of the website (`docs/figures/`) show each
  figure with its caption from the paper (`docs/figures/captions.json`).
* If the paper and the code disagree, check `git log`: every change of a printed number is a
  commit of its own, with the numbers before and after in its message.

## How to check a claim of the paper

1. Find the section of the passage in `README.md` ("Where is what in the paper?"); it names the
   module or script, whose docstring quotes the section.
2. Numbers: every number printed in the paper is printed by one of
   `scripts/manuscript_numbers.py` (text, Table 3), `scripts/ec_quality.py` (Appendix A),
   `scripts/make_manuscript_tables.py` (bodies of Tables 2, 4, C1 in `tables_manuscript/`).
   Their combined output, plus the robustness checks (`scripts/robustness_checks.py`,
   `scripts/ec_band_sensitivity.py`), is committed as `expected_numbers.txt`.
   `pixi run check` recomputes everything and compares (tolerance 0.02).
3. Equations: each `fluxpy` function that implements an equation quotes it in its docstring
   (e.g. `fluxpy.util.potential_temperature`, `fluxpy.stability.bulk_richardson_2pt`,
   `fluxpy.eddy_covariance.band_covariance_random_error`). `tests/` checks them against
   independent references (numerical integration, published tables, Monte-Carlo).
4. Figures: `figures_manuscript/figNN_*.py` writes `figures_manuscript/output/figNN_*.{pdf,png}`
   at print size; the manuscript includes these files directly.

## Running things

Always through pixi (the environment is locked; calling `.pixi/envs/default/python.exe`
directly fails on Windows because scipy's DLLs are not on the path):

```
pixi run test         # unit and physics tests, 5 s
pixi run check        # every number against expected_numbers.txt, ~1 min
pixi run all          # PANGAEA (1.5 GB) and Zenodo (0.2 GB) downloads, once, then every output (segments ... figures) + check, ~15 min
pixi run fetch-pangaea  # only the PANGAEA download (measurements, master tracks), with checksums; pixi run inputs cuts the files
pixi run fetch-zenodo   # only the Zenodo data record (reanalysis, satellite, GIBS images, leg table), with checksums
pixi run numbers      # print the numbers of the text
pixi run robustness   # robustness of the comparison (dependence, baselines, MODIS subset, ...)
pixi run book         # re-execute the didactic notebooks in book/
pixi run api          # regenerate the API reference docs/api/ and the figure pages docs/figures/ (pixi run test checks them)
pixi run site         # build the website (docs/ + book/) into _build/html; site-serve for a live preview
pixi run site-deploy  # upload _build/html to Cloudflare behind a password (deploy/README.md; site-login once)
```

## Conventions that are decisions, not accidents

* All fluxes are positive **upward**; model-diagnosed fluxes are sign-flipped once, in
  `scripts/calc_bulk_and_ec_fluxes.py`. Variable names follow `fluxpy/naming.py`
  (`H_obs_ec`, `H_obs_bulk`, `H_rs_bulk`, `H_era_bulk`, `H_era_diag`, `H_car_diag`).
* One set of constants, in `fluxpy/util.py` (c_p = 1004 J kg⁻¹ K⁻¹ as the paper states).
* Every potential temperature uses the pressure at its own height: in situ, the surface pressure
  is reconstructed hydrostatically from the T-Bird pressure; for ERA5, level 136 uses its
  hybrid-coefficient pressure and the surface the surface pressure.
* EC flux = w–θ covariance between 0.1 and 20 Hz after linear detrending. No segment is
  discarded for being small (n = 91; the only exclusion is a constant flight level,
  `fluxpy.aggregate.SegmentConfig`); the integrity check is reported, not used to exclude
  (Table A1 repeats the comparison at several tolerances); every flux carries its random error
  (Appendix A).
* Amplitude skill = ordinary-least-squares slope of the estimate on the EC flux; variability
  skill = Pearson r. RMSE is reported but not used to rank (it rewards estimates near zero).
* Statistics (slopes and their intervals, correlations, biases, errors and uncertainties,
  sensitivities) are printed with two decimals; flux values, temperatures, Ri_b and flight
  facts with one; counts and percentages as integers (author decision 2026-09-24).

## Rules for changing anything

* A change that moves a printed number gets its own commit, with the numbers before and after
  in the message; then
  `pixi run python scripts/check_expected_numbers.py --update`.
* A change of a docstring, signature or `#:` comment in `fluxpy/`, of a script or figure script, or a
  change of a figure caption in the manuscript needs `pixi run api`: the API reference
  (`docs/api/`) and the figure pages (`docs/figures/`) are generated from them, and `pixi run test`
  fails otherwise.
* A refactor must leave `pixi run check` green. Converting float32 input to float64 is not
  neutral (it moved EC fluxes by 4·10⁻⁵ relative once).
* Do not edit files that contain backslashes (LaTeX, regexes, `"\n"`) with `sed` or shell
  heredocs on Windows; Git Bash turns `\n` into control characters.
* Stage files by name. Re-running the pipeline rewrites the output files with a new time stamp;
  `git checkout` the ones whose content did not change.
* The book chapters `book/*.ipynb` are generated by `book/build_book.py`; edit the text and code
  there and run `pixi run book`, never the notebooks themselves.
* A release carries one version in `pixi.toml`, `CITATION.cff` and `.zenodo.json`
  (`tests/test_release_metadata.py`); Zenodo archives every GitHub release and reads only
  `.zenodo.json`.

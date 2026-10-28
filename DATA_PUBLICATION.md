# Publishing the data of this repository

The data must not be published through the git repository. Measurements and the products derived
from them belong on PANGAEA; the subsets of third-party gridded products on Zenodo. The code
repository then keeps only code, the paper outputs and `data/manifest.tsv`, and downloads the data
from both archives. This file describes the steps. `data/manifest.tsv` lists every data file with
its size, SHA-256 checksum and target archive; `pixi run data-manifest` checks the files against it,
`pixi run data-manifest --write` recomputes it after a file has changed.

## 1. What goes where

**A. Already on PANGAEA, not submitted again.** The bundle *Observations from the BACSAM II aircraft
campaign with Polar 6* ([doi:10.1594/PANGAEA.995979](https://doi.org/10.1594/PANGAEA.995979),
CC BY 4.0) contains the master tracks
([971763](https://doi.org/10.1594/PANGAEA.971763)), the 100 Hz T-Bird observations
([995975](https://doi.org/10.1594/PANGAEA.995975)), the 100 Hz Polar 6 nose boom
([995974](https://doi.org/10.1594/PANGAEA.995974)), the 50 Hz KT-19 temperature
([995980](https://doi.org/10.1594/PANGAEA.995980)) and the 20 Hz irradiance
([996236](https://doi.org/10.1594/PANGAEA.996236)). The 13 master-track files in
`data/observations/raw/polar6/mastertracks/` are taken from there unchanged.

**B. New PANGAEA dataset, derived from the measurements** (archive `pangaea:new`, well under 1 MB):

| File in `data/observations/` | Content |
|---|---|
| `ar/analysis_ready_polar6_turbulent_fluxes_latest_5km.nc` | EC flux, random error, integrity check and segment means of the segments flown at one level (those below 50 m over sea ice are used in the paper) |
| `ar/analysis_ready_polar6_legs_metadata.csv` | start and end time of every straight leg |
| `ar/ec_segment_placement.nc` | segment-placement spread of the EC flux (Appendix A) |
| `ar/ec_leg_length_sensitivity/ec_sensitivity_summary.csv` | window-length experiment (Fig. 3) |

**C. Zenodo data record, third-party gridded products** (archive `zenodo:data`, about 197 MB): the ERA5
model-level and single-level subsets and the three CARRA files (`data/reanalysis/`), the MODIS ice surface
temperature and the merged MODIS–AMSR2 sea-ice concentration (`data/observations/processed/satellite_unified/`),
`data/observations/ar/along_track_inputs.nc` (the gridded values at the segments, written by `pixi run extract`)
and the 13 NASA GIBS images that form the background of Fig. 5 (`figures_manuscript/_gibs_cache/`).

**D. Not archived.** `pixi run inputs` (`scripts/prepare_inputs.py`) cuts the published T-Bird (995975),
nose-boom (995974) and KT-19 (995980) files, downloaded into `data/pangaea/<dataset>/`, to the windows around
the low-level legs (`data/observations/processed/`); `pixi run polar6-level` computes
`data/observations/ar/polar6_level_at_segments.nc` from the nose-boom file; `data/model_output/` is written by
`pixi run precompute`. The two `apr13_*_worldview.png` images in `figures_manuscript/_gibs_cache/` are not read
by any script.

## 2. Decisions

* **D1, T-Bird 100 Hz (decided 27 October 2026).** The analysis uses the published T-Bird data (995975);
  `scripts/prepare_inputs.py` cuts them to the leg windows. The earlier input, converted from the ASCII
  files of the raw-data host, differed from the published version (its EC fluxes on 23 April most of all)
  and is no longer used.
* **D2, KT-19 and Polar 6 nose boom (decided 27 October 2026).** Both come from the published datasets
  (995980, 995974), cut by the same script. The radiation file is no longer used (its irradiances are
  published in 996236).
* **D3, licence of the Zenodo record (decided 27 October 2026).** CC BY 4.0 for the compilation, with the
  terms of each provider stated in the description: ERA5 and CARRA under the Copernicus licence
  (redistribution allowed with the attribution "Contains modified Copernicus Climate Change Service
  information 2024"; check the current wording on the Climate Data Store), MODIS (NSIDC) and GIBS (NASA)
  without restriction but with citation, the merged sea-ice concentration under the terms of the
  University of Bremen. Check each before uploading.
* **D4, authors (open).** Who is listed as author of dataset B and of the Zenodo record. A natural choice
  for dataset B is the author who derived the products, together with the authors of the source datasets
  it builds on (L. Köhler, J. Hartmann), if they agree.

## 3. PANGAEA: dataset B

1. Log in at [pangaea.de/submit](https://www.pangaea.de/submit/) (account or ORCID) and fill in the
   submission form; the editors then work with you in the PANGAEA issue tracker, where metadata and
   files are exchanged.
2. Provide: title, authors with ORCID, an abstract, the method (Sects. 2–3 and Appendix A of the paper;
   the code repository and its DOI), the parameter list with units and long names (`DATA.md` and the
   netCDF attributes), the flights as events (`P6-247_BACSAM2_2024_…`), the licence (CC BY 4.0, as the
   bundle), and the relations: source datasets 995975, 995980 and 995974, bundle 995979, the paper.
3. Files: the netCDF and CSV files as listed above, so that the code can read them unchanged, together
   with their checksums from `data/manifest.tsv`. PANGAEA archives netCDF as files if they are fully
   described ([Binary objects](https://wiki.pangaea.de/wiki/Binary_objects)) and prefers tabular data as
   tab-delimited text ([Format](https://wiki.pangaea.de/wiki/Format)); ask the editors whether the
   segment table should in addition be imported as a table. Attachments above 100 MB need an upload link.
4. Ask whether the dataset can join the bundle 995979 or should reference it.
5. If the journal needs the data during review, ask for password-protected access for the editor.
   PANGAEA assigns the DOI after archiving; processing can take several weeks. Projects with
   publication funds are asked for a voluntary contribution (500 € net per submission).
6. Proof-read the final version; then the DOI is registered.

## 4. Zenodo: data record C

1. [zenodo.org](https://zenodo.org) → *New upload*. Under *Digital Object Identifier* answer "No" and
   click *Get a DOI now!*: the DOI is reserved and can be written into the files and the paper before
   publication (deleting the draft loses it).
2. Upload one zip archive that keeps the folder structure of the repository (`data/reanalysis/…`,
   `data/observations/…`, `figures_manuscript/_gibs_cache/…`), so that it unpacks at the repository
   root, together with a short README (content, origin, regridding: `preprocessing/`, licences).
3. Metadata: resource type *Dataset*; title; creators (D4); description; licence (D3); keywords;
   related identifiers: the paper, the code record, dataset B, the source products as cited in the paper;
   funding: Deutsche Forschungsgemeinschaft, custom award 268020496 (TRR 172, (AC)³).
4. Publish. Files of a published record cannot be replaced; a correction is a new version with its own DOI.

## 5. The repository after both publications

1. A script `pixi run fetch-data` downloads dataset B, the master tracks and record C into the
   paths of `data/manifest.tsv` and checks every checksum. (Master-track files downloaded again from
   PANGAEA can differ in their header from the shipped ones; compare them by their data lines.)
   The published T-Bird, nose-boom and KT-19 files already work this way: `pixi run fetch-pangaea`
   (`scripts/fetch_pangaea.py`) downloads them into `data/pangaea/` and checks their checksums, and
   `pixi run all` starts with it and with `pixi run inputs`; the cut files are not in git.
2. `data/` (except `manifest.tsv`) and `figures_manuscript/_gibs_cache/` leave git and go into
   `.gitignore`. `DATA.md`, `README.md`, `docs/reproduce.md`, `.zenodo.json` (the code record no longer
   contains data) and `.github/workflows/reproduce.yml` (fetch the data first) are updated.
3. The data are in all earlier commits, including the commits of pull request #1. Rewriting the history
   and force-pushing does not remove them from GitHub: commits referenced by a pull request stay
   reachable. Recommended: rename this repository (it stays private as an archive), create a new public
   repository under the old name, and push a history from which `data/` and the GIBS cache are removed
   (`git filter-repo`). The commit messages, which record every change of a printed number, are kept.
4. Code release on Zenodo (GitHub release), once the data DOIs exist.
5. Paper: the data availability section cites the bundle 995979, dataset B and record C; the code
   availability section the code record.

## 6. Order

1. Decision D4 with the co-authors concerned.
2. `pixi run all` on the final state and `pixi run data-manifest --write`; commit.
3. Zenodo record C: reserve the DOI, upload, publish (a day).
4. PANGAEA dataset B: submit (several weeks until the DOI).
5. Meanwhile: `fetch-data`, removal of the data from git, the new public repository.
6. Code release and the availability statements of the paper.

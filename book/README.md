# Method chapters

Five notebooks that follow the structure of the paper and recompute its central numbers from the shipped data with
the same `fluxpy` functions as the pipeline. They can be read on GitHub (outputs are included) or run locally.

| Chapter | Paper | Content |
|---|---|---|
| [1 · Data and flux estimates](01_overview.ipynb) | Sects. 1–2, Table 1, Fig. 1 | the six flux estimates, the 91 segments, the magnitude of the fluxes |
| [2 · Eddy-covariance reference flux](02_eddy_covariance.ipynb) | Sect. 3.1, Fig. 3, Appendix A | one segment from the 100 Hz signals to W m⁻², the 0.1–20 Hz band, integrity check, random error, window length, EC uncertainty |
| [3 · Bulk flux estimates](03_bulk_method.ipynb) | Sect. 3.2, Appendix B | one bulk flux by hand (Δθ, Ri_b, roughness, C_h), stability functions, the role of α_T |
| [4 · Comparison with the eddy-covariance flux](04_comparison.ipynb) | Sect. 4.1, Table 3, Fig. 6 | skill metrics, the comparison, the effect of the satellite surface temperature |
| [5 · Sensitivity and uncertainty](05_sensitivity_and_uncertainty.ipynb) | Sect. 4.2, Tables 2–3, Fig. 7 | scalar roughness ratio α_T, stability scheme and momentum roughness, input sensitivity, uncertainty budget |

The notebooks are generated and executed by [`build_book.py`](build_book.py):

```
pixi run book
```

Together with the pages in `docs/` and the API reference in `docs/api/`, the chapters form the website of the
repository: `pixi run site` builds it into `_build/html` (`pixi run site-serve` for a live preview); on GitHub it is
published by `.github/workflows/site.yml`.

Every hand calculation is checked against the pipeline output inside the notebook (e.g. the EC flux computed in
chapter 2 equals the value in the data file; the bulk flux computed in chapter 3 equals that of the pipeline).

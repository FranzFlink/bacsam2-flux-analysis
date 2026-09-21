# The method, step by step

Five short notebooks that follow the structure of the paper and recompute its central numbers from the shipped
data with the same `fluxpy` functions the pipeline uses. They are meant for readers who want to *understand* the
method — open them directly on GitHub (outputs are included) or run them yourself.

| Chapter | Paper section | What you see computed |
|---|---|---|
| [1 · What is compared, and on which data](01_overview.ipynb) | Introduction, data | the six flux estimates, the 91 segments, why the fluxes are small |
| [2 · The reference: eddy covariance](02_eddy_covariance.ipynb) | EC method, Fig. 3, Appendix A | one segment from the 100 Hz signals to W m⁻², the 0.1–20 Hz band, the three QC checks, window length, EC uncertainty |
| [3 · The bulk method](03_bulk_method.ipynb) | bulk method, Appendix B | one bulk flux by hand (Δθ, Ri_b, roughness, C_h), stability functions, the role of α_T |
| [4 · How well do the estimates reproduce the measured flux?](04_comparison.ipynb) | results, Fig. 6, comparison table | why slope and correlation instead of RMSE, the comparison, where the satellite surface temperature helps |
| [5 · What does the result depend on — and how uncertain is it?](05_sensitivity_and_uncertainty.ipynb) | sensitivity, Fig. 7, uncertainty table | α_T changes the amplitude but not the correlation, everything else is secondary, uncertainty budget |

The notebooks are generated and executed by [`build_book.py`](build_book.py):

```
pixi run book
```

Every hand calculation is checked against the pipeline output inside the notebook (e.g. the EC flux computed in
chapter 2 equals the value in the data file; the bulk flux computed in chapter 3 equals the pipeline's).

---
title: Symbols and terms
short_title: Glossary
---

## The six flux estimates

All fluxes are in W m⁻² and **positive upward** (from the surface to the atmosphere). The names are also
the variable names in `data/model_output/sensitivity_figure_data.nc` (`fluxpy/naming.py`).

| Symbol | Variable | Definition |
|--------|----------|------------|
| $H_\mathrm{S,obs,EC}$ | `H_obs_ec` | eddy covariance from the 100 Hz T-Bird data (**reference**) |
| $H_\mathrm{S,obs,bulk}$ | `H_obs_bulk` | bulk formula with measured inputs: KT-19 surface temperature, T-Bird air temperature, wind and pressure |
| $H_\mathrm{S,RS,bulk}$ | `H_rs_bulk` | bulk formula with the MODIS surface temperature and the ERA5 atmosphere ("ERA5–MODIS forcing") |
| $H_\mathrm{S,ERA,bulk}$ | `H_era_bulk` | bulk formula with ERA5 inputs only ("reanalysis-only forcing") |
| $H_\mathrm{S,ERA,diag}$ | `H_era_diag` | surface flux diagnosed by ERA5 (model output) |
| $H_\mathrm{S,CAR,diag}$ | `H_car_diag` | surface flux diagnosed by CARRA (model output; not available on 13 April) |

## Symbols

| Symbol | Meaning |
|--------|---------|
| $w'$, $\theta'$ | fluctuations of vertical wind and potential temperature about their linear trend over a segment |
| $\theta$ | potential temperature, $\theta = T (p_0/p)^{R_d/c_p}$ with $p_0$ = 1000 hPa; each level with the pressure at its own height |
| $T_\mathrm{s}$, $\theta_\mathrm{s}$ | surface skin temperature and its potential temperature |
| $\Delta\theta$ | $\theta_\mathrm{air} - \theta_\mathrm{s}$; negative is unstable (surface warmer than air), positive stable |
| $U$ | horizontal wind speed at the measurement level |
| $C_\mathrm{h}$, $C_\mathrm{h,n}$ | transfer coefficient of heat, and its neutral value |
| $f_h$ | stability factor, $C_\mathrm{h} = C_\mathrm{h,n}\, f_h$ |
| $z_{0,m}$, $z_{0,T}$ | roughness lengths for momentum and heat |
| $\alpha_T$ | scalar roughness ratio $z_{0,T}/z_{0,m}$; $10^{-1}$ in the reference configuration, $10^{-2}$ in the amplitude-matched one |
| $Ri_\mathrm{b}$ | two-point bulk Richardson number between the surface and the measurement level (Eq. B3); positive is stable |
| $\sigma_\mathrm{rand}$ | random (sampling) error of an EC flux (Finkelstein and Sims, 2001), 1σ |
| $b$ | slope of the ordinary-least-squares regression of an estimate on the EC flux: the **amplitude** it recovers (ideal 1) |
| $r$ | Pearson correlation with the EC flux: agreement in variability (ideal 1) |

## Terms

| Term | Meaning in this work |
|------|---------------------|
| leg | a complete constant-altitude flight stretch (tens of km) |
| segment | a 5 km piece of a leg; all estimates are compared on the same 91 segments |
| forcing | the source of the inputs of a bulk estimate (in situ, ERA5–MODIS, reanalysis only) |
| configuration | the parameter settings of the bulk algorithm (stability scheme, $\alpha_T$, roughness) |
| reference configuration | GLGS20 stability functions, $\alpha_T = 10^{-1}$, surface-type-dependent roughness |
| amplitude-matched configuration | as the reference, with $\alpha_T = 10^{-2}$, where the in-situ estimate has slope 1 |
| integrity check | whether the vertical-wind spectrum of a segment falls off like $f^{-5/3}$ (resolved turbulence), $\lvert s_w + 5/3 \rvert \le 0.4$; reported for every segment, no segment is excluded by it |
| model-diagnosed | a flux computed inside the reanalysis model, as opposed to a bulk flux recomputed offline from its output |

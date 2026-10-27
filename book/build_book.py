#!/usr/bin/env python3
"""Build and execute the notebooks of the didactic book.

The notebooks follow the structure of the manuscript and recompute its central
numbers from the shipped data, using the same ``fluxpy`` functions
as the pipeline. They are generated from the cell lists below and executed, so
every figure and number in them is produced on your machine.

Run:  pixi run book        (about 2 minutes; run `pixi run precompute` first)
"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent

SETUP = r'''
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
REPO = Path("..").resolve(); sys.path.insert(0, str(REPO))
plt.rcParams.update({"figure.dpi": 110, "figure.constrained_layout.use": True, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

AR  = REPO / "data/observations/ar"
FIG = xr.open_dataset(REPO / "data/model_output/sensitivity_figure_data.nc")      # all flux estimates on the segments
SEG = xr.open_dataset(AR / "analysis_ready_polar6_turbulent_fluxes_latest_5km.nc").sel(start_time=FIG.start_time.values)
EC  = FIG["H_obs_ec"].values
'''

# ═════════════════════════════════════════════════════════════════════════════
NB1 = [("md", r'''
# 1 · Data and flux estimates

Over Arctic sea ice, the turbulent sensible heat flux $H_\mathrm{S}$ is commonly estimated with a bulk formula
driven by reanalysis data. Assessing such estimates requires a measured reference. During BACSAM-II (Svalbard,
April 2024), the **T-Bird**, a sensor package towed about 23 m above the ice below the research aircraft Polar 6,
sampled wind and temperature at 100 Hz, from which the flux is obtained by **eddy covariance (EC)**.

All estimates are evaluated on the same 5 km flight segments:

| Symbol | Method | Inputs |
|---|---|---|
| $H_\mathrm{S,obs,EC}$ | eddy covariance; the reference against which all other estimates are evaluated | 100 Hz $w$, $\theta$ from the T-Bird |
| $H_\mathrm{S,obs,bulk}$ | bulk formula | measured: KT-19 surface temperature, T-Bird air temperature, wind and pressure |
| $H_\mathrm{S,RS,bulk}$ | bulk formula | MODIS surface temperature (ERA5 skin temperature under cloud) and ERA5 air temperature, wind and pressure at the model level closest to the T-Bird |
| $H_\mathrm{S,ERA,bulk}$ | bulk formula | ERA5 only |
| $H_\mathrm{S,ERA,diag}$, $H_\mathrm{S,CAR,diag}$ | model output | surface flux diagnosed by ERA5 and CARRA |

From top to bottom, measured inputs are successively replaced by products with pan-Arctic coverage, so that a
loss of skill can be attributed to the replaced input.

The chapters follow the paper's sections: the EC reference (2), the bulk method (3), the comparison (4), and sensitivity and
uncertainty (5).

> [!NOTE]
> All numbers are computed from the files in `data/` with the `fluxpy` functions that produce the paper, and hand
> calculations are checked against the pipeline output. Symbols are defined in the [glossary](../docs/glossary.md).
> The notebooks can be run in Jupyter from the `book/` folder (`pixi run jupyter lab`).
>
> **Paper:** Sects. 1–2, Table 1, Fig. 1. **Code:** `scripts/calc_bulk_and_ec_fluxes.py` evaluates all six
> estimates on the segments and writes `data/model_output/sensitivity_figure_data.nc`, which this chapter reads.
'''), ("code", SETUP), ("md", r'''
## Segments

The analysis uses the constant-altitude legs flown over sea ice with the T-Bird below 50 m; within a segment the
T-Bird height varies by less than 30 m (paper Sect. 2.1 and Appendix A1). The low-level legs of 25 April were flown
over open water and are excluded.
'''), ("code", r'''
t = pd.to_datetime(SEG.start_time.values)
tab = pd.DataFrame({"segments": 1, "T-Bird height (m)": SEG.altitude.values,
                    "EC flux (W m-2)": EC, "KT-19 surface temp. (degC)": SEG.t_kt19.values}, index=t)
g = tab.groupby(tab.index.strftime("%d %B"))
out = g.agg({"segments": "sum", "T-Bird height (m)": "mean", "EC flux (W m-2)": "median", "KT-19 surface temp. (degC)": "mean"}).round(1)
print(f"{len(tab)} segments of 5 km = {5 * len(tab)} km of low-level flight\n"); out
'''), ("md", r'''
The fluxes are small: over compact sea ice in spring, surface and air temperatures are nearly equal. With fluxes of
a few W m⁻², an estimate of zero everywhere already has a small error. The estimates are therefore assessed by the
*amplitude* they recover (regression slope) and by their *correlation* with the reference, not by the RMSE (Chapter 4).
'''), ("code", r'''
fig, ax = plt.subplots(figsize=(6, 2.8))
ax.hist(EC[np.isfinite(EC)], bins=np.arange(-8, 15, 1), color="0.35")
ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("EC sensible heat flux (W m$^{-2}$, positive upward)"); ax.set_ylabel("segments")
print(f"median {np.nanmedian(EC):.1f} W m-2, {np.mean(np.abs(EC[np.isfinite(EC)]) < 2) * 100:.0f} % of the segments within +-2 W m-2")
'''), ("md", r'''
**Figure 1.1.** Distribution of the EC sensible heat flux over all segments (bins of 1 W m⁻², positive upward).
'''), ("md", r'''
Flight tracks (Fig. 1):

![Fig. 1](../figures_manuscript/output/fig01_campaign_overview.png)
''')]

# ═════════════════════════════════════════════════════════════════════════════
NB2 = [("md", r'''
# 2 · Eddy-covariance reference flux

Turbulent eddies transport heat upward when rising air ($w' > 0$) is on average warmer ($\theta' > 0$) than
sinking air. The sensible heat flux is this covariance multiplied by the heat capacity of air:

$$H_\mathrm{S} = \rho\, c_p\, \overline{w'\theta'}$$

This chapter derives the flux of one 5 km segment from the 100 Hz data and reproduces the value stored for it in the
segment file `data/observations/ar/analysis_ready_polar6_turbulent_fluxes_latest_5km.nc`. That file holds one row per
5 km segment (EC flux, random error, integrity check and segment means) and is written by
`scripts/run_ec_aggregation.py`; its variables are listed under [Data files](../docs/data.md).

> [!NOTE]
> **Paper:** Sect. 3.1, Eq. (1), Fig. 3, Appendix A. **Code:** `fluxpy/eddy_covariance.py`
> (`band_covariance`, `band_covariance_random_error`, `segment_flux`), `fluxpy/qc.py` (integrity check),
> `scripts/ec_window_length.py` (Fig. 3), `scripts/ec_quality.py` (every number of Appendix A).
'''), ("code", SETUP + r'''
from fluxpy.eddy_covariance import band_covariance, segment_flux
from fluxpy.util import CP, RD, add_theta

TB = xr.open_dataset(REPO / "data/observations/processed/t-bird_unified/noseboom_unified.nc").sortby("time")
i = int(np.nanargmax(EC))                                   # the segment with the largest upward flux
t0, t1 = SEG.start_time.values[i], SEG.end_time.values[i]
seg = add_theta(TB.sel(time=slice(t0, t1))).dropna("time", how="any")
print(f"segment {pd.Timestamp(t0):%d %b %Y %H:%M:%S} UTC: {seg.time.size} samples = {seg.time.size / 100:.0f} s at 100 Hz, "
      f"height {float(seg.h.mean()):.0f} m, airspeed {float(seg.tas.mean()):.0f} m/s")
'''), ("md", r'''
## Detrended time series

Vertical wind $w$ and potential temperature $\theta$ are measured at 100 Hz. The platform motion has been removed from the wind
with the inertial navigation data; the potential rather than the virtual potential temperature is used because the
air is cold and dry (paper Sect. 3.1). Removing a linear trend from both series yields the fluctuations $w'$ and
$\theta'$.
'''), ("code", r'''
from scipy.signal import detrend
w, th = detrend(seg.w.values), detrend(seg.theta.values)
x_km = np.arange(w.size) / 100 * float(seg.tas.mean()) / 1000
fig, axs = plt.subplots(3, 1, figsize=(8, 5), sharex=True)
axs[0].plot(x_km, w, lw=0.4, color="C0"); axs[0].set_ylabel("w' (m s$^{-1}$)")
axs[1].plot(x_km, th, lw=0.4, color="C3"); axs[1].set_ylabel(r"$\theta$' (K)")
axs[2].plot(x_km, np.cumsum(w * th) / w.size, color="k"); axs[2].set_ylabel(r"running sum of w'$\theta$'"); axs[2].set_xlabel("distance along the segment (km)")
print(f"plain covariance of the detrended series: {np.mean(w * th):.4f} K m/s")
'''), ("md", r'''
**Figure 2.1.** Detrended vertical wind $w'$ (top) and potential temperature $\theta'$ (middle) along the segment, and
the running sum of $w'\theta'$ divided by the number of samples (bottom), which ends at the covariance of the segment.
'''), ("md", r'''
## Cospectrum and flux band

The cospectrum distributes the covariance over frequency. At the mean airspeed of 62 m s⁻¹, a frequency $f$
corresponds to an eddy length of $62\,\mathrm{m\,s^{-1}}/f$. The flux is the cospectrum integrated between
**0.1 and 20 Hz** (eddies of about 3 to 620 m): slower variations are attributed to mesoscale changes along the
track, faster ones to instrument noise. Only the shaded band enters the flux.
'''), ("code", r'''
n, fs = w.size, 100.0
f = np.fft.rfftfreq(n, 1 / fs)
co = 2 * np.real(np.fft.rfft(w) * np.conj(np.fft.rfft(th))) / n**2       # one-sided cospectrum; sums to the covariance
edges = np.logspace(np.log10(f[1]), np.log10(50), 40); mid = np.sqrt(edges[:-1] * edges[1:])
binned = np.array([co[(f >= a) & (f < b)].sum() for a, b in zip(edges[:-1], edges[1:])])

fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
axs[0].bar(mid, binned, width=np.diff(edges), color="0.4"); axs[0].set_xscale("log")
axs[0].set_ylabel("covariance per frequency bin (K m s$^{-1}$)"); axs[0].set_title("cospectrum")
axs[1].plot(f[1:], np.cumsum(co[1:][::-1])[::-1], color="k"); axs[1].set_xscale("log")
axs[1].set_ylabel("covariance from f upwards (K m s$^{-1}$)"); axs[1].set_title("cumulative from high to low frequency")
for ax in axs:
    ax.axvspan(0.1, 20, color="C1", alpha=0.15); ax.set_xlabel("frequency (Hz)")
in_band = band_covariance(w, th, fs, 0.1, 20.0)
print(f"covariance inside 0.1-20 Hz: {in_band:.4f} K m/s = {in_band / np.mean(w * th) * 100:.0f} % of the total covariance of this segment")
'''), ("md", r'''
**Figure 2.2.** Left: cospectrum of $w'$ and $\theta'$, summed in logarithmically spaced frequency bins. Right:
covariance contained in all frequencies above $f$, accumulated from high to low frequency. The shaded band
(0.1–20 Hz) is the part that enters the flux.
'''), ("md", r'''
## Conversion to W m⁻²
'''), ("code", r'''
rho = float((seg.p * 100 / (RD * (seg["T"] + 273.15))).mean())        # dry-air density from p (hPa) and T (degC)
H_by_hand = rho * CP * in_band
H_fluxpy = segment_flux(seg).flux
print(f"rho = {rho:.2f} kg m-3, c_p = {CP:.0f} J kg-1 K-1")
print(f"H by hand            : {H_by_hand:6.2f} W m-2")
print(f"H from fluxpy        : {float(H_fluxpy):6.2f} W m-2")
print(f"H in the segment file: {float(SEG.sensible_heat_flux[i]):6.2f} W m-2")
'''), ("md", r'''
## Integrity check and random error (Appendix A)

Two properties are assessed separately.

**Integrity.** Resolved turbulence shows the inertial-subrange slope in the power spectrum of $w$,
$|s_w + 5/3| \le 0.4$. The slope is reported for every segment; no segment is excluded by it. Table A1 of the paper
repeats the comparison on the segments within tolerances of 0.3 to 0.6.

**Precision.** A covariance from about 81 s of data carries a random error, because only a finite number of eddies is
sampled. It is estimated from the data (Finkelstein and Sims, 2001). For a true flux of zero, the estimate exceeds
1, 2 and 3 $\sigma_\mathrm{rand}$ by chance in 32 %, 5 % and 0.3 % of cases.

Fluxes that do not exceed their random error are **retained**. Over compact sea ice a flux indistinguishable from zero
is the typical and physically valid result; discarding such segments would bias the reference towards large fluxes. For
the same reason, criteria formulated as ratios to the flux (such as signal-to-noise ratios) are not applied: they reject
small fluxes and accept large ones irrespective of the quality of the measurement.
'''), ("code", r'''
from fluxpy.qc import segment_spectra, turbulence_resolved
from fluxpy.eddy_covariance import band_covariance_random_error
slope = segment_spectra(w=seg.w.values, theta=seg.theta.values, u=seg.u.values, v=seg.v.values, fs=100.0)["slope_w"]
sigma = rho * CP * band_covariance_random_error(w, th, fs)
print(f"this segment: spectral slope of w = {slope:.2f} -> turbulence resolved: {turbulence_resolved(slope)}")
print(f"              flux {float(H_fluxpy):.1f} W m-2, random error {sigma:.2f} W m-2 -> {abs(float(H_fluxpy)) / sigma:.0f} sigma")
ratio = FIG.ec_flux_over_sigma.values
print(f"\nall {ratio.size} segments: {int(SEG.turbulence_resolved.sum())} satisfy the integrity check (none is excluded); "
      f"median random error {np.median(FIG.sigma_random_ec.values):.2f} W m-2")
days = pd.to_datetime(FIG.start_time.values).strftime("%d %b")
pd.DataFrame({f"|H| > {k} sigma": {**{d: f"{(ratio[days == d] > k).sum()} / {(days == d).sum()}" for d in pd.unique(days)},
                                   "all": f"{(ratio > k).sum()} / {ratio.size}"} for k in (1, 2, 3)})
'''), ("md", r'''
On 15 April, the day with open leads and the largest fluxes, all fluxes clearly exceed their random error, and on
23 April most do; on 13 and 17 April most fluxes are statistically indistinguishable from zero. This is a property of
the surface, not a defect of the data, and it limits the correlation attainable by any estimate (Chapter 4).

## Segment length (Fig. 3)

Short windows under-sample the larger eddies; long windows average over leads and ice. Each of the six low-level legs
is cut into sub-windows of different length, and their mean flux is compared with the flux of the complete leg
(`scripts/ec_window_length.py`).
'''), ("code", r'''
wl = pd.read_csv(AR / "ec_leg_length_sensitivity/ec_sensitivity_summary.csv").query("length_km >= 1 and label != 'full leg'")
fig, axs = plt.subplots(1, 2, figsize=(9, 3))
axs[0].errorbar(wl.length_km, wl.ratio_mean, wl.ratio_std, fmt="o-", color="C1", ecolor="0.5", capsize=2)
axs[0].axhline(1, color="k", ls="--"); axs[0].set_ylabel("sub-window flux / full-leg flux")
axs[1].plot(wl.length_km, wl.random_error_median, "s-", color="C4"); axs[1].set_ylabel("random error (W m$^{-2}$)")
for ax in axs:
    ax.set_xscale("log"); ax.axvline(5, color="goldenrod", lw=3, ls="--"); ax.set_xlabel("window length (km)")
r5 = wl[wl.length_km == 5].iloc[0]
print(f"at 5 km: flux ratio {r5.ratio_mean:.2f}, random error {r5.random_error_median:.2f} W m-2, fluxes above 2 sigma {r5.fraction_above_2sigma * 100:.0f} %")
'''), ("md", r'''
**Figure 2.3.** Left: mean flux of the sub-windows of a leg relative to the flux of the complete leg, as a function of
the window length (mean and standard deviation over the legs). Right: median random error of the sub-window fluxes. The
dashed line marks the 5 km segments of the analysis.
'''), ("md", r'''
## Uncertainty of the EC flux

The total uncertainty combines, in quadrature, the **random** error (above) and the **instrument** error (5 %, from the
flow calibration of the five-hole probe). The **placement** spread, obtained by shifting the 5 km window along the leg in
0.05 km steps (`scripts/ec_segment_placement.py`), is an independent, empirical estimate of the random error. It is
reported as a cross-check and not added.
'''), ("code", r'''
from fluxpy.uncertainty import uncertainty_budget
U = uncertainty_budget(FIG)
pd.DataFrame({k.replace("ec_", ""): [np.nanmedian(U[k]), np.nanpercentile(U[k], 25), np.nanpercentile(U[k], 75)]
              for k in ("ec_random", "ec_instrument", "ec_total", "ec_placement")},
             index=["median (W m-2)", "25th percentile", "75th percentile"]).round(2)
'''), ("md", r'''
Appendix A of the paper contains two further checks, printed by `pixi run ec-quality`: the comparison is insensitive to
the screening of the fluxes (Table A1, eight alternative screenings), and the pitch resonance of the towed T-Bird near
1 Hz reduces the larger fluxes by less than a tenth (median), with practically no effect on the comparison.

**Summary.** The reference flux is a band-limited covariance that can be recomputed from the 100 Hz data in a few lines.
Its uncertainty is dominated by the finite number of eddies in 5 km, as confirmed by the placement spread, not by the
instrument.
''')]

# ═════════════════════════════════════════════════════════════════════════════
NB3 = [("md", r'''
# 3 · Bulk flux estimates

Without turbulence measurements, the flux is estimated from mean quantities:

$$H_\mathrm{S} = \rho\, c_p\, C_\mathrm{h}\, U\, (\theta_\mathrm{s} - \theta_\mathrm{air})$$

the wind speed $U$ times the surface–air temperature difference, scaled by the **transfer coefficient**
$C_\mathrm{h}$, which describes the efficiency of turbulent mixing. $C_\mathrm{h}$ depends on

* the **roughness** of the surface: $z_{0,m}$ for momentum and $z_{0,T} = \alpha_T\, z_{0,m}$ for heat;
* the **stability**, expressed by the bulk Richardson number $Ri_\mathrm{b}$: stable stratification
  ($Ri_\mathrm{b} > 0$, surface colder than the air) suppresses turbulence, unstable stratification enhances it.

This chapter computes $H_\mathrm{S,obs,bulk}$ for one segment and shows the dependence of $C_\mathrm{h}$ on stability
and on $\alpha_T$.

> [!NOTE]
> **Paper:** Sect. 3.2, Eq. (2), Appendix B, Table B1. **Code:** `fluxpy/bulk.py` (steps B1–B3),
> `fluxpy/stability.py` (Eq. B3), `fluxpy/stability_lib.py` (the eight stability schemes), `fluxpy/util.py`
> (constants, potential temperature, pressures).
'''), ("code", SETUP + r'''
from fluxpy.util import potential_temperature, air_density, CP, G
from fluxpy.stability import bulk_richardson_2pt, Cd_Ch_noniterative
from fluxpy.bulk import reference_config
cfg = reference_config()
i = int(np.nanargmax(EC)); s = SEG.isel(start_time=i)
print("reference configuration:", cfg.scheme_name, "| alpha_T =", cfg.zt_ratio, "| z0 (open/thin/thick) =", cfg.z0_ocean, cfg.z0_thin_ice, cfg.z0_thick_ice, "m")
'''), ("md", r'''
## Temperature difference and wind speed

The surface temperature is taken from the KT-19 infrared pyrometer, air temperature and wind from the T-Bird, all
averaged over the 5 km segment. Both temperatures are converted to potential temperature with the pressure at their
own height. Pressure is measured at the T-Bird, about 23 m above the ice, not at the surface; the surface pressure
follows from the hypsometric equation,

$$p_\mathrm{s} = p_z \exp\!\left(\frac{g\,z}{R_d\,\bar T}\right),$$

about 3 hPa above the T-Bird value. Using $p_z$ at the surface would make $\theta_\mathrm{s}$ too warm by
$g z / c_p \approx 0.23$ K, which is not negligible against typical temperature differences of one to two kelvin. The
reanalysis-driven estimates apply the same rule: the ERA5 surface pressure for $\theta_\mathrm{s}$, and the pressure of
model level 136 (about 31 m; `fluxpy.util.era5_level_pressure`) for the air temperature of that level.
'''), ("code", r'''
from fluxpy.util import surface_pressure_hypsometric
T_s, T_a, p, z = float(s.t_kt19), float(s.T_mean), float(s.p_mean), float(s.altitude)
u, v = float(s.u_mean), float(s.v_mean); U = np.hypot(u, v)
p_s = float(surface_pressure_hypsometric(p, z, T_mean=0.5 * (T_s + T_a)))
th_s, th_a = float(potential_temperature(T_s, p_s)), float(potential_temperature(T_a, p))
print(f"pressure {p:.1f} hPa at the T-Bird ({z:.0f} m)  ->  {p_s:.1f} hPa at the surface")
print(f"surface {T_s:.1f} degC, air {T_a:.1f} degC  ->  theta_s - theta_air = {th_s - th_a:+.2f} K "
      f"(with the T-Bird pressure at the surface it would be {float(potential_temperature(T_s, p)) - th_a:+.2f} K);  wind {U:.1f} m/s")
'''), ("md", r'''
## Bulk Richardson number

$$Ri_\mathrm{b} = \frac{g}{\bar\theta}\,\frac{\Delta\theta\,\Delta z}{\Delta u^2 + \Delta v^2}$$

between the surface (zero wind) and the measurement height; negative values indicate unstable stratification.
'''), ("code", r'''
Ri = float(bulk_richardson_2pt(dtheta=th_a - th_s, dz=z, du=u, dv=v, theta_ref=0.5 * (th_s + th_a), eps_wind2=cfg.ri_eps_wind2))
print(f"Ri_b = {Ri:+.2f}  ->", "unstable (surface warmer than air)" if Ri < 0 else "stable")
'''), ("md", r'''
## Roughness lengths

Within 5 km the surface consists of open water, thin ice and snow-covered ice. Each KT-19 sample is classified by its
temperature (above −1.8 °C: open water; at or below −20 °C: snow-covered ice; in between: thin ice). The effective
roughness length is the logarithmic average over the three fractions.
'''), ("code", r'''
fr = np.array([float(s.frac_open), float(s.frac_thin_ice), float(s.frac_thick_ice)])
z0 = float(np.exp(fr @ np.log([cfg.z0_ocean, cfg.z0_thin_ice, cfg.z0_thick_ice]))); zT = cfg.zt_ratio * z0
print(f"fractions open / thin / snow-covered: {fr.round(2)}  ->  z0m = {z0:.1e} m,  z0T = alpha_T * z0m = {zT:.1e} m")
'''), ("md", r'''
## Transfer coefficient and flux
'''), ("code", r'''
Cd, Ch = Cd_Ch_noniterative(Ri, z, z0, zT, cfg.scheme_name)
_, Ch_neutral = Cd_Ch_noniterative(0.0, z, z0, zT, cfg.scheme_name)
rho = float(air_density(T_a, p))
H = rho * CP * Ch * U * (th_s - th_a)
print(f"C_h = {Ch:.2e}  (neutral value {Ch_neutral:.2e}: stability changes it by a factor {Ch / Ch_neutral:.2f})")
print(f"H bulk by hand      : {H:6.1f} W m-2")
print(f"H bulk from pipeline: {float(FIG.H_obs_bulk[i]):6.1f} W m-2")
print(f"H measured (EC)     : {EC[i]:6.1f} W m-2")
'''), ("md", r'''
## Dependence of $C_\mathrm{h}$ on stability

The stability dependence follows from Monin–Obukhov similarity theory, for which several formulations exist. The paper
uses Gryanik et al. (2020, GLGS20) as reference and tests seven alternatives. The shading marks the range of
$Ri_\mathrm{b}$ sampled during the campaign (central 90 % of the segments). All schemes share the unstable branch; on
the stable branch they differ, but there $C_\mathrm{h}$ and the flux are small in every scheme. The effect on the
comparison is quantified in Chapter 5.
'''), ("code", r'''
lo, hi = np.nanpercentile(FIG.Ri_obs.values, [5, 95])                 # central 90 % of the sampled Ri_b
ri = np.linspace(min(-1.0, lo), max(0.5, hi), 300)
fig, ax = plt.subplots(figsize=(6.5, 3.4))
for name in ["GLGS20", "BD", "HB88", "BH91", "CB05", "GAFGP07"]:
    ax.plot(ri, [Cd_Ch_noniterative(r, 24.0, 3e-4, 3e-5, name)[1] * 1e3 for r in ri], label=name, lw=2 if name == "GLGS20" else 1)
ax.axvspan(lo, hi, color="0.85")
ax.axvline(0, color="k", lw=0.6); ax.set_xlabel("bulk Richardson number"); ax.set_ylabel("$C_h$ × 10$^3$"); ax.legend(ncol=2, fontsize=8)
ax.set_title("z = 24 m, z$_{0m}$ = 3·10$^{-4}$ m, α$_T$ = 0.1");
'''), ("md", r'''
**Figure 3.1.** Transfer coefficient $C_\mathrm{h}$ as a function of the bulk Richardson number for six of the eight
stability schemes, at the height and roughness given in the title. The grey band is the range of $Ri_\mathrm{b}$ sampled
during the campaign.
'''), ("md", r'''
## Dependence of $C_\mathrm{h}$ on $\alpha_T$

The scalar roughness ratio $\alpha_T = z_{0,T}/z_{0,m}$ is poorly constrained over sea ice: models use values between 1
(ERA5) and 0.1 (large-eddy simulations; the reference of the paper), whereas theory and earlier aircraft measurements
indicate much smaller values. $C_\mathrm{h}$ depends on it logarithmically:
'''), ("code", r'''
a = np.logspace(-7, 0, 50)
ch = np.array([Cd_Ch_noniterative(0.0, 24.0, 3e-4, 3e-4 * x, "GLGS20")[1] for x in a])
fig, ax = plt.subplots(figsize=(6, 3)); ax.semilogx(a, ch / ch[np.argmin(abs(a - 0.1))], color="k")
ax.axvline(0.1, color="C2", ls="--"); ax.axvline(1e-2, color="C1", ls="--")
ax.set_xlabel(r"$\alpha_T$"); ax.set_ylabel(r"$C_h$ relative to $\alpha_T$ = 0.1")
print(f"alpha_T from 1e-1 to 1e-4 (three orders of magnitude) changes C_h by a factor {ch[np.argmin(abs(a - 1e-4))] / ch[np.argmin(abs(a - 0.1))]:.2f}")
'''), ("md", r'''
**Figure 3.2.** $C_\mathrm{h}$ under neutral stratification as a function of $\alpha_T$, relative to its value at
$\alpha_T = 0.1$. Dashed lines: reference configuration ($\alpha_T = 0.1$, green) and amplitude-matched configuration
($\alpha_T = 10^{-2}$, orange).
'''), ("md", r'''
**Summary.** The bulk flux is the product of wind speed, temperature difference and $C_\mathrm{h}$, which contains
roughness and stability. Roughness scales the flux and thus changes its *amplitude*, but not *where* along the track the
flux is large or small; that is determined by the temperature difference and the wind. This distinction structures the
results of Chapters 4 and 5.
''')]

# ═════════════════════════════════════════════════════════════════════════════
NB4 = [("md", r'''
# 4 · Comparison with the eddy-covariance flux

## Skill metrics

Most measured fluxes lie within a few W m⁻² of zero (Chapter 1), so an estimate close to zero everywhere has a small
error without containing information. The table therefore includes the trivial estimate $H = 0$ on every segment; its
mean absolute error (MAE) and RMSE can be compared with those of the actual estimates.

The paper uses two separate measures:

* **slope $b$** of the regression of the estimate on the EC flux: the fraction of the observed *amplitude* recovered (ideal: 1);
* **correlation $r$**: agreement of estimate and reference in their variability (ideal: 1).

The slope equals $r$ times the ratio of the standard deviations of estimate and reference. A slope near 1 can therefore
result from the correct amplitude with good correlation or from excess variance with poor correlation; $b$ and $r$ are
interpreted together.

> [!NOTE]
> **Paper:** Sect. 4.1, Table 4, Figs. 4–6. **Code:** `fluxpy/sensitivity.py` (`skill_metrics`),
> `scripts/manuscript_numbers.py` (every number of Sect. 4.1), `figures_manuscript/fig06_flux_scatter.py`.
'''), ("code", SETUP + r'''
from fluxpy.sensitivity import skill_metrics
import logging; logging.disable(logging.INFO)
NAMES = {"H_obs_bulk": "H_S,obs,bulk  (measured inputs)", "H_rs_bulk": "H_S,RS,bulk   (MODIS + ERA5)",
         "H_era_bulk": "H_S,ERA,bulk  (ERA5 only)", "H_era_diag": "H_S,ERA,diag  (ERA5's own flux)", "H_car_diag": "H_S,CAR,diag  (CARRA's own flux)"}
T = pd.to_datetime(FIG.start_time.values)
LEG = np.cumsum(np.r_[0, np.diff(T.values) / np.timedelta64(1, "m")] > 10)          # a new leg after a gap of > 10 min

def r_leg(h):
    """correlation after removing the mean of every leg: only the variability along the legs counts"""
    d = pd.DataFrame({"ec": EC, "h": h, "leg": LEG}).dropna()
    a = d.groupby("leg")[["ec", "h"]].transform(lambda s: s - s.mean())
    return np.corrcoef(a.ec, a.h)[0, 1]

rows = {}
for key, name in NAMES.items():
    h = FIG[key].values; ok = np.isfinite(h)
    m = skill_metrics(EC, h, key)
    rows[name] = {"slope b": m["slope"], "r": m["r"], "r_leg": r_leg(h), "sd ratio": m["std_ratio"], "bias (W m-2)": m["bias"],
                  "MAE (W m-2)": np.mean(np.abs(h[ok] - EC[ok])), "RMSE (W m-2)": m["rmse"], "n": m["n"]}
rows["zero flux  (H = 0 everywhere)"] = {"bias (W m-2)": -EC.mean(), "MAE (W m-2)": np.mean(np.abs(EC)),
                                        "RMSE (W m-2)": np.sqrt(np.mean(EC ** 2)), "n": EC.size}
table = pd.DataFrame(rows).T; table["n"] = table["n"].astype(int)
stats = table.columns.drop("n")                          # two decimals, also in the columns with a gap
table[stats] = table[stats].apply(lambda col: col.map(lambda v: "–" if pd.isna(v) else f"{v:.2f}"))
table
'''), ("md", r'''
(CARRA is not available on 13 April, hence the smaller *n*.) The zero estimate has a smaller MAE and RMSE than every
estimate in the table except the one with measured inputs: in this weak-flux regime, these error measures reward
conservative estimates, not skill.

$r_\mathrm{leg}$ is the correlation after removing the mean of each leg from both series (Table 4 of the paper). It
measures whether an estimate follows the flux from segment to segment *along* a leg, rather than only the differences
between flight days. Only the estimate with measured inputs does so clearly.

The model-diagnosed fluxes of ERA5 and CARRA overestimate the amplitude (slope above 1) and are biased low. They follow
the observed variability better than the offline ERA5-only estimate, which the paper attributes to their internal skin
temperature and much larger scalar roughness (Sect. 2.2 and Discussion), but follow it only weakly along the legs.

## Bulk estimates against the reference (Fig. 6)
'''), ("code", r'''
fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), sharex=True, sharey=True)
for ax, key, c in zip(axs, ["H_obs_bulk", "H_rs_bulk", "H_era_bulk"], ["C2", "C1", "C0"]):
    h = FIG[key].values; ok = np.isfinite(EC) & np.isfinite(h); b, a = np.polyfit(EC[ok], h[ok], 1)
    ax.scatter(EC[ok], h[ok], s=12, color=c, alpha=0.7); xx = np.array([-8, 14])
    ax.plot(xx, xx, "k--", lw=0.8); ax.plot(xx, a + b * xx, color=c, lw=2)
    ax.set_title(f"{NAMES[key].split('(')[1][:-1]}\nslope {b:.2f}, r {np.corrcoef(EC[ok], h[ok])[0, 1]:.2f}", fontsize=9)
    ax.set_xlabel("EC flux (W m$^{-2}$)"); ax.set_xlim(-8, 14); ax.set_ylim(-30, 30)
axs[0].set_ylabel("bulk flux (W m$^{-2}$)");
'''), ("md", r'''
**Figure 4.1.** The three offline bulk estimates against the EC flux, one point per segment, with the 1:1 line (dashed)
and the regression line (solid). Left to right: measured inputs, MODIS surface temperature with the ERA5 atmosphere,
ERA5 only. Figure 6 of the paper shows the same comparison together with the model-diagnosed fluxes.
'''), ("md", r'''
From left to right, the panels show the main result of the paper (numbers in the panel titles):

* **Measured inputs:** the highest correlation, but the amplitude is overestimated (slope above 1). With measured
  inputs, the conventional transfer coefficient is too large (Chapter 5).
* **MODIS surface temperature:** with the ERA5 atmosphere unchanged, amplitude and correlation both increase.
* **ERA5 only:** the regression line is nearly flat; a small part of the amplitude is recovered and the correlation is weak.

## Effect of the satellite surface temperature

If the benefit of MODIS arises from resolving leads and thin ice that ERA5 does not represent, it should be greatest for
segments with a *heterogeneous* surface. The segments are split at the median of the within-segment standard deviation
of the KT-19 surface temperature.
'''), ("code", r'''
sd = SEG.T_sfc_std.values; rs, era = FIG.H_rs_bulk.values, FIG.H_era_bulk.values
ok = np.isfinite(EC) & np.isfinite(rs) & np.isfinite(era) & np.isfinite(sd); med = np.median(sd[ok])
rows = {}
for label, m in (("heterogeneous half", ok & (sd > med)), ("homogeneous half", ok & (sd <= med))):
    rows[label] = {"n": int(m.sum()), "r  ERA5 only": np.corrcoef(EC[m], era[m])[0, 1], "r  with MODIS": np.corrcoef(EC[m], rs[m])[0, 1],
                   "slope  ERA5 only": np.polyfit(EC[m], era[m], 1)[0], "slope  with MODIS": np.polyfit(EC[m], rs[m], 1)[0]}
print(f"median within-segment standard deviation of the surface temperature: {med:.1f} K"); pd.DataFrame(rows).T.round(2)
'''), ("md", r'''
**Summary.** The surface temperature determines whether a bulk estimate follows the observed variability. The ERA5
surface temperature does not represent leads and thin ice within a grid cell; on heterogeneous segments the ERA5-only
flux is anti-correlated with the measurement, and replacing the surface temperature by MODIS has the largest effect there.
''')]

# ═════════════════════════════════════════════════════════════════════════════
NB5 = [("md", r'''
# 5 · Sensitivity and uncertainty

With measured inputs, the bulk method overestimates the flux amplitude (Chapter 4). To identify the cause, the paper
varies one choice within the transfer coefficient at a time (scalar roughness ratio $\alpha_T$, stability scheme,
momentum roughness) and perturbs the two inputs, temperature difference and wind speed.

> [!NOTE]
> **Paper:** Sect. 4.2, Table 3, Fig. 7, Appendix C (Table C1), Table 2. **Code:** `fluxpy/sensitivity.py`
> (the configurations), `scripts/calc_bulk_and_ec_fluxes.py` (stages 3 and 4), `fluxpy/uncertainty.py` (Table 2),
> `scripts/make_manuscript_tables.py`.
'''), ("code", SETUP + r'''
import logging; logging.disable(logging.INFO)
S = pd.read_csv(REPO / "data/model_output/sensitivity_summary_table.csv").set_index("label")
S = S[~S.index.str.contains("era5only")]
'''), ("md", r'''
## Scalar roughness ratio $\alpha_T$
'''), ("code", r'''
a = S[S.index.str.contains("ztr") | (S.index == "GLGS20_ref")].sort_values("zt_ratio")
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True)
for ax, col, ideal, lab in ((axs[0], "slope", 1, "slope b (amplitude)"), (axs[1], "r", None, "correlation r")):
    ax.semilogx(a.zt_ratio, a["obs_" + col], "o-", color="C2", label="measured inputs"); ax.semilogx(a.zt_ratio, a["rs_" + col], "s-", color="C1", label="MODIS + ERA5")
    if ideal: ax.axhline(ideal, color="k", ls="--", lw=0.8)
    ax.set_xlabel(r"$\alpha_T$"); ax.set_ylabel(lab)
axs[1].set_ylim(0, 1); axs[0].legend(fontsize=8); axs[0].set_xlim(2e-8, 0.3)
tab = a.drop_duplicates("zt_ratio")[["zt_ratio", "obs_slope", "obs_r", "rs_slope", "rs_r"]]
tab = tab.assign(zt_ratio=tab.zt_ratio.map("{:.0e}".format)).round(2)        # format alpha_T before rounding the statistics
tab.rename(columns={"zt_ratio": "alpha_T", "obs_slope": "slope (measured inputs)", "obs_r": "r (measured inputs)",
                    "rs_slope": "slope (MODIS + ERA5)", "rs_r": "r (MODIS + ERA5)"}).set_index("alpha_T")
'''), ("md", r'''
**Figure 5.1.** Slope $b$ (left) and correlation $r$ (right) of the bulk estimates with measured inputs (green) and with
MODIS + ERA5 inputs (orange) as functions of $\alpha_T$; the dashed line marks the ideal slope of 1. The table lists the
same values.
'''), ("md", r'''
With measured inputs, the amplitude is matched (slope 1) near $\alpha_T = 10^{-2}$, the *amplitude-matched
configuration* of the paper, one order of magnitude below the conventional value. The correlation is unaffected, since
roughness only rescales the flux (Chapter 3).

This is not interpreted as a measurement of $z_{0,T}$: any error that scales the flux, for instance in the surface
temperature, is absorbed by the same parameter. The value of $\alpha_T$ that gives the in-situ estimate the correct
amplitude further reduces the amplitude of the ERA5–MODIS estimate, which is already too small; a single $\alpha_T$
cannot correct both.

## Stability scheme and momentum roughness
'''), ("code", r'''
groups = {"stability scheme (8 schemes)": S.index.str.endswith("_ref"),
          "alpha_T (1e-4 ... 1e-1)": (S.index.str.contains("ztr") & S.zt_ratio.between(1e-4, 1e-1)) | (S.index == "GLGS20_ref"),
          "momentum roughness set": S.index.str.contains("mosaic|sheba|nschnier") | (S.index == "GLGS20_ref"),
          "constant momentum roughness": S.index.str.contains("z0c")}
rng = lambda x: f"{x.min():.2f} - {x.max():.2f}"
pd.DataFrame({g: {"slope, measured inputs": rng(S[m].obs_slope), "r, measured inputs": rng(S[m].obs_r),
                  "slope, MODIS + ERA5": rng(S[m].rs_slope), "r, MODIS + ERA5": rng(S[m].rs_r)} for g, m in groups.items()}).T
'''), ("md", r'''
No choice within the transfer coefficient changes the correlation, and only the roughness lengths change the amplitude
appreciably. The eight stability schemes are practically indistinguishable: they share the functions for unstable
stratification and differ only under stable stratification, where the flux is small (Chapter 3).

## Sensitivity to the inputs
'''), ("code", r'''
from fluxpy.uncertainty import central_difference, uncertainty_budget
keep = np.isfinite(EC); q = lambda x: f"{np.nanmedian(x):+.2f}  [{np.nanpercentile(x, 25):+.2f}, {np.nanpercentile(x, 75):+.2f}]"
pd.DataFrame({name: {"per +1 K in (theta_air - theta_s)   (W m-2)": q(central_difference(FIG, f"H_{k}_dTair", "dT")[keep]),
                     "per +1 m/s in wind speed   (W m-2)": q(central_difference(FIG, f"H_{k}_dU", "dU")[keep])}
              for k, name in (("obs", "measured inputs"), ("rs", "MODIS + ERA5"))}).T
'''), ("md", r'''
The table gives the change of the bulk flux per unit change of one input: the median over the segments and, in
brackets, the 25th and 75th percentiles. A change of 1 K in the temperature difference has a much larger effect than
1 m s⁻¹ in wind speed, and the typical air–surface temperature difference of one to two kelvin is comparable
to the uncertainty of a satellite surface temperature. The surface temperature product is therefore decisive (Chapter 4).

## Uncertainty budget (Table 2)

For the bulk fluxes, the assumed input uncertainties are propagated through the sensitivities above (first-order
Gaussian error propagation).
'''), ("code", r'''
from fluxpy.uncertainty import INPUT_UNCERTAINTY
U = uncertainty_budget(FIG)
print("assumed 1-sigma input uncertainties (T_s [K], T_air [K], U [m/s]):", INPUT_UNCERTAINTY)
pd.DataFrame({name: [np.nanmedian(U[k]), np.nanpercentile(U[k], 25), np.nanpercentile(U[k], 75)]
              for k, name in (("ec_total", "EC reference"), ("obs_bulk", "bulk, measured inputs"), ("rs_bulk", "bulk, MODIS + ERA5"))},
             index=["median (W m-2)", "25th percentile", "75th percentile"]).round(2).T
'''), ("md", r'''
**Summary.** The measured reference is known to better than 1 W m⁻². A bulk estimate driven by satellite and reanalysis
inputs is uncertain by several W m⁻², more than the typical flux; surface temperature, air temperature and the large
ERA5 wind uncertainty contribute comparably. Whether an estimate follows the observed variability is determined by the
surface temperature (Chapter 4), its amplitude by the transfer coefficient; improving pan-Arctic flux estimates requires
both.
''')]

BOOK = {
    "01_overview.ipynb": NB1,
    "02_eddy_covariance.ipynb": NB2,
    "03_bulk_method.ipynb": NB3,
    "04_comparison.ipynb": NB4,
    "05_sensitivity_and_uncertainty.ipynb": NB5,
}


HIDE = "# (code collapsed on the website)\n"


def make_cell(kind: str, src: str):
    """Markdown or code cell. Code cells that start with the common set-up or with the HIDE line are
    collapsed on the website (MyST tag ``hide-input``); on GitHub and in Jupyter they look as before."""
    if kind == "md":
        return new_markdown_cell(src.strip())
    hide = src.strip().startswith(SETUP.strip()[:40]) or src.lstrip().startswith(HIDE.strip())
    return new_code_cell(src.strip(), metadata={"tags": ["hide-input"]} if hide else {})


if __name__ == "__main__":
    import sys
    only = sys.argv[1:]                                   # e.g. `python book/build_book.py 05` rebuilds chapter 5 only
    for fname, cells in BOOK.items():
        if only and not any(fname.startswith(o) for o in only):
            continue
        nb = new_notebook(metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
        nb.cells = [make_cell(kind, src) for kind, src in cells]
        NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
        nbformat.write(nb, HERE / fname)
        print("built", fname)

#!/usr/bin/env python3
"""Build and execute the notebooks of the didactic book.

The notebooks follow the structure of the manuscript and recompute its central
numbers step by step from the shipped data, using the same ``fluxpy`` functions
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
FIG = xr.open_dataset(REPO / "data/model_output/sensitivity_figure_data.nc")      # all flux estimates, 91 segments
SEG = xr.open_dataset(AR / "analysis_ready_polar6_turbulent_fluxes_latest_5km.nc").sel(start_time=FIG.start_time.values)
EC  = FIG["H_obs_ec"].values
'''

# ═════════════════════════════════════════════════════════════════════════════
NB1 = [("md", r'''
# 1 · What is compared, and on which data

**Question of the paper.** Over Arctic sea ice the turbulent sensible heat flux $H_\mathrm{S}$ is usually
*estimated* with a bulk formula from reanalysis data. How good are such estimates? To find out we need a
*measured* reference. BACSAM-II (Svalbard, April 2024) provides one: a sensor package (the **T-Bird**) towed
about 24 m above the ice below the research aircraft Polar 6, sampling wind and temperature at 100 Hz.
From these data the flux can be measured directly with the **eddy-covariance (EC)** method.

**The logic of the comparison.** All estimates are brought onto the same 5 km flight segments:

| Symbol | How it is obtained | Inputs |
|---|---|---|
| $H_\mathrm{S,obs,EC}$ | eddy covariance — **the reference** | 100 Hz $w$, $\theta$ from the T-Bird |
| $H_\mathrm{S,obs,bulk}$ | bulk formula | measured: KT-19 surface temperature, T-Bird air temperature and wind |
| $H_\mathrm{S,RS,bulk}$ | bulk formula | MODIS surface temperature + ERA5 air temperature and wind |
| $H_\mathrm{S,ERA,bulk}$ | bulk formula | ERA5 only |
| $H_\mathrm{S,ERA,diag}$, $H_\mathrm{S,CAR,diag}$ | taken from the model output | the flux ERA5 / CARRA diagnose themselves |

Going down the table, measured inputs are replaced step by step by products that are available everywhere.
Whatever skill is lost on the way can then be attributed to the input that was replaced.

The chapters follow the paper: **2** how the EC reference is measured · **3** how the bulk method works ·
**4** how the estimates compare · **5** what the result depends on, and how uncertain it is ·
**6** how robust all of this is with 91 segments from four flight days.

> [!NOTE]
> **How to read this book.** Every chapter computes its numbers live from the files in `data/`, with the same
> `fluxpy` functions that produce the paper, and checks its hand calculations against the pipeline output.
> Symbols are explained in the glossary of the website (`docs/glossary.md`). The code cells can be run in
> Jupyter from the `book/` folder (`pixi run jupyter lab`).
>
> **In the paper:** Sects. 1–2, Table 1, Fig. 1. **In the code:** `scripts/calc_bulk_and_ec_fluxes.py` puts all six
> estimates on the segments and writes `data/model_output/sensitivity_figure_data.nc`, which this chapter reads.
'''), ("code", SETUP), ("md", r'''
## The segments

The analysis uses the low-level legs (T-Bird below 50 m) of the four flight days over sea ice.
'''), ("code", r'''
t = pd.to_datetime(SEG.start_time.values)
tab = pd.DataFrame({"segments": 1, "T-Bird height (m)": SEG.altitude.values,
                    "EC flux (W m-2)": EC, "KT-19 surface temp. (degC)": SEG.t_kt19.values}, index=t)
g = tab.groupby(tab.index.strftime("%d %B"))
out = g.agg({"segments": "sum", "T-Bird height (m)": "mean", "EC flux (W m-2)": "median", "KT-19 surface temp. (degC)": "mean"}).round(1)
print(f"{len(tab)} segments of 5 km = {5 * len(tab)} km of low-level flight\n"); out
'''), ("md", r'''
The fluxes are **small**: over compact sea ice in spring the air and the surface have almost the same temperature.
This has a consequence for the whole analysis — with fluxes of a few W m⁻², an estimate that simply predicts
"zero everywhere" already has a small error. Chapter 4 shows why the paper therefore judges the estimates by the
*amplitude* they recover (regression slope) and by their *correlation* with the reference, not by the RMSE.
'''), ("code", r'''
fig, ax = plt.subplots(figsize=(6, 2.8))
ax.hist(EC[np.isfinite(EC)], bins=np.arange(-8, 15, 1), color="0.35")
ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("EC sensible heat flux (W m$^{-2}$, positive upward)"); ax.set_ylabel("segments")
print(f"median {np.nanmedian(EC):.1f} W m-2, {np.mean(np.abs(EC[np.isfinite(EC)]) < 2) * 100:.0f} % of the segments within +-2 W m-2")
'''), ("md", r'''
Flight tracks (Fig. 1 of the paper):

![Fig. 1](../figures_manuscript/output/fig01_campaign_overview.png)
''')]

# ═════════════════════════════════════════════════════════════════════════════
NB2 = [("md", r'''
# 2 · The reference: measuring the flux by eddy covariance

Turbulent eddies carry heat upward when rising air ($w' > 0$) is on average warmer ($\theta' > 0$) than sinking
air. The sensible heat flux is this covariance, scaled by the heat capacity of air:

$$H_\mathrm{S} = \rho\, c_p\, \overline{w'\theta'}$$

This chapter computes it for **one 5 km segment, step by step**, and arrives at the value stored in the data file.

> [!NOTE]
> **In the paper:** Sect. 3.1, Eq. (1), Fig. 3, Appendix A. **In the code:** `fluxpy/eddy_covariance.py`
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
## Step 1 — the raw signals

Vertical wind $w$ and potential temperature $\theta$. A linear trend is removed from both; what remains are the
turbulent fluctuations $w'$ and $\theta'$.
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
## Step 2 — which eddy sizes carry the flux?

The cospectrum splits the covariance by frequency. At 61 m s⁻¹ airspeed a frequency $f$ corresponds to an eddy of
length $61\,\mathrm{m\,s^{-1}}/f$. The paper integrates the cospectrum between **0.1 and 20 Hz** (eddies of about 3 to 610 m):
slower variations are mesoscale changes along the track rather than turbulence, faster ones are instrument noise.

The shaded band below is what enters the flux.
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
## Step 3 — from covariance to W m⁻²
'''), ("code", r'''
rho = float((seg.p * 100 / (RD * (seg["T"] + 273.15))).mean())        # dry-air density from p (hPa) and T (degC)
H_by_hand = rho * CP * in_band
H_fluxpy = segment_flux(seg).flux
print(f"rho = {rho:.2f} kg m-3, c_p = {CP:.0f} J kg-1 K-1")
print(f"H by hand         : {H_by_hand:6.2f} W m-2")
print(f"H from fluxpy     : {float(H_fluxpy):6.2f} W m-2")
print(f"H in the data file: {EC[i]:6.2f} W m-2")
'''), ("md", r'''
## Step 4 — how good is this number? (Appendix A)

Two questions have to be kept apart.

**Is the measurement sound?** Turbulence must be resolved: the power spectrum of $w$ has to fall off with a slope near
$-5/3$ (inertial subrange), $|s_w + 5/3| \le 0.4$. This is a pass/fail check of the *instrument and the processing*.

**How precisely is the flux known?** A covariance from 85 s of data has a random error, because only a limited number of
eddies were sampled. It is estimated from the data themselves (Finkelstein and Sims, 2001). The larger a flux is compared with
its random error $\sigma_\mathrm{rand}$, the more certain it is that it differs from zero: by chance alone a true zero flux exceeds
1, 2 and 3 $\sigma_\mathrm{rand}$ in 32 %, 5 % and 0.3 % of all cases.

Fluxes that do not stand out from their random error are **kept**. Over compact sea ice "no measurable flux" is the typical, physically correct
result; throwing these segments away would leave only the large fluxes and bias the reference. For the same reason the paper
does not use criteria that are ratios to the flux (such as signal-to-noise ratios): they fail for small fluxes and pass for
large ones, whatever the quality of the measurement.
'''), ("code", r'''
from fluxpy.qc import segment_spectra, turbulence_resolved
from fluxpy.eddy_covariance import band_covariance_random_error
slope = segment_spectra(w=seg.w.values, theta=seg.theta.values, u=seg.u.values, v=seg.v.values, fs=100.0)["slope_w"]
sigma = rho * CP * band_covariance_random_error(w, th, fs)
print(f"this segment: spectral slope of w = {slope:.2f} -> turbulence resolved: {turbulence_resolved(slope)}")
print(f"              flux {float(H_fluxpy):.1f} +- {sigma:.1f} W m-2 (random error) -> {abs(float(H_fluxpy)) / sigma:.0f} sigma")
ratio = FIG.ec_flux_over_sigma.values
print(f"\nall {ratio.size} segments: {int(SEG.turbulence_resolved.sum())} pass the integrity check; "
      f"median random error {np.median(FIG.sigma_random_ec.values):.1f} W m-2")
days = pd.to_datetime(FIG.start_time.values).strftime("%d %b")
pd.DataFrame({f"|H| > {k} sigma": {**{d: f"{(ratio[days == d] > k).sum()} / {(days == d).sum()}" for d in pd.unique(days)},
                                   "all": f"{(ratio > k).sum()} / {ratio.size}"} for k in (1, 2, 3)})
'''), ("md", r'''
On 15 April, the day with open leads and the largest fluxes, every flux stands out clearly; on the days over compact ice or with weak
temperature contrast most fluxes are statistically zero. That is a property of the Arctic surface, not a defect of the data —
and it is the reason why the correlations in Chapter 4 cannot be high for *any* estimate.

## Is 5 km long enough? (Fig. 3)

A short window might miss the larger eddies; a long one blurs leads and ice together. The experiment cuts each of the six
low-level legs into sub-windows of different length and compares their mean flux with the flux of the complete leg
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
print(f"at 5 km: flux ratio {r5.ratio_mean:.1f}, random error {r5.random_error_median:.1f} W m-2, fluxes above 2 sigma {r5.fraction_above_2sigma * 100:.0f} %")
'''), ("md", r'''
## How uncertain is one EC flux?

Two contributions, added in quadrature: the **random** error from Step 4 and the **instrument** error (5 % from the flow
calibration of the five-hole probe). An independent, purely empirical check of the random error is the **placement** spread —
what happens to the flux when the 5 km window is shifted along the leg in 0.05 km steps (`scripts/ec_segment_placement.py`).
It is not added, because it measures the same thing; it is there to show that the random error has the right size.
'''), ("code", r'''
from fluxpy.uncertainty import uncertainty_budget
U = uncertainty_budget(FIG)
pd.DataFrame({k.replace("ec_", ""): [np.nanmedian(U[k]), np.nanpercentile(U[k], 25), np.nanpercentile(U[k], 75)]
              for k in ("ec_random", "ec_instrument", "ec_total", "ec_placement")},
             index=["median (W m-2)", "25th percentile", "75th percentile"]).round(1)
'''), ("md", r'''
**Take-away.** The reference flux is a band-limited covariance that can be recomputed from the 100 Hz data in a few lines.
Its uncertainty is dominated by the limited number of eddies in 5 km — confirmed by shifting the window along the track — not by the instrument.
''')]

# ═════════════════════════════════════════════════════════════════════════════
NB3 = [("md", r'''
# 3 · The bulk method

Where no turbulence measurement exists, the flux is estimated from mean quantities:

$$H_\mathrm{S} = \rho\, c_p\, C_\mathrm{h}\, U\, (\theta_\mathrm{s} - \theta_\mathrm{air})$$

wind speed $U$ times the temperature difference between surface and air, times a **transfer coefficient**
$C_\mathrm{h}$ that says how efficiently turbulence mixes. Everything difficult is hidden in $C_\mathrm{h}$. It depends on

* the **roughness** of the surface: $z_{0,m}$ for momentum and $z_{0,T} = \alpha_T\, z_{0,m}$ for heat,
* the **stability** of the air, measured by the bulk Richardson number $Ri_\mathrm{b}$: stable stratification
  ($Ri_\mathrm{b} > 0$, cold surface) suppresses turbulence, unstable stratification enhances it.

This chapter computes $H_\mathrm{S,obs,bulk}$ for one segment by hand and then shows how $C_\mathrm{h}$ reacts to stability and to $\alpha_T$.

> [!NOTE]
> **In the paper:** Sect. 3.2, Eq. (2), Appendix B, Table B1. **In the code:** `fluxpy/bulk.py` (the steps B1–B3),
> `fluxpy/stability.py` (Eq. B2), `fluxpy/stability_lib.py` (the eight stability schemes), `fluxpy/util.py`
> (constants, potential temperature, pressures).
'''), ("code", SETUP + r'''
from fluxpy.util import potential_temperature, air_density, CP, G
from fluxpy.stability import bulk_richardson_2pt, Cd_Ch_noniterative
from fluxpy.bulk import reference_config
cfg = reference_config()
i = int(np.nanargmax(EC)); s = SEG.isel(start_time=i)
print("reference configuration:", cfg.scheme_name, "| alpha_T =", cfg.zt_ratio, "| z0 (open/thin/thick) =", cfg.z0_ocean, cfg.z0_thin_ice, cfg.z0_thick_ice, "m")
'''), ("md", r'''
## Step 1 — temperature difference and wind

Surface temperature from the KT-19 infrared pyrometer, air temperature and wind from the T-Bird — all averaged over
the 5 km segment. Both temperatures are converted to potential temperature, each with the pressure *at its own height*.
The T-Bird measures the pressure about 24 m above the ice; nobody measures it at the surface. The surface pressure follows
from the hydrostatic (hypsometric) equation,

$$p_\mathrm{s} = p_z \exp\!\left(\frac{g\,z}{R_d\,\bar T}\right),$$

about 3 hPa more than at the T-Bird. It looks negligible, but it is not: using $p_z$ at the surface would make
$\theta_\mathrm{s}$ too warm by $g z / c_p \approx 0.23$ K, and the typical temperature difference is only about 1 K.
The reanalysis-driven estimates follow the same rule the other way round: the surface pressure is known, and the
air temperature of ERA5 model level 136 (about 31 m) gets the pressure of that level (`fluxpy.util.era5_level_pressure`).
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
## Step 2 — stability: the bulk Richardson number

$$Ri_\mathrm{b} = \frac{g}{\bar\theta}\,\frac{\Delta\theta\,\Delta z}{\Delta u^2 + \Delta v^2}$$

between the surface (where the wind is zero) and the measurement height. Negative = unstable.
'''), ("code", r'''
Ri = float(bulk_richardson_2pt(dtheta=th_a - th_s, dz=z, du=u, dv=v, theta_ref=0.5 * (th_s + th_a), eps_wind2=cfg.ri_eps_wind2))
print(f"Ri_b = {Ri:+.2f}  ->", "unstable (surface warmer than air)" if Ri < 0 else "stable")
'''), ("md", r'''
## Step 3 — roughness

Within 5 km the surface is a mixture of open water, thin ice and snow-covered ice. The KT-19 classifies every
sample by its temperature (warmer than −1.8 °C: open water; colder than −20 °C: snow-covered ice; in between: thin ice).
The effective roughness length is the logarithmic average over the three fractions.
'''), ("code", r'''
fr = np.array([float(s.frac_open), float(s.frac_thin_ice), float(s.frac_thick_ice)])
z0 = float(np.exp(fr @ np.log([cfg.z0_ocean, cfg.z0_thin_ice, cfg.z0_thick_ice]))); zT = cfg.zt_ratio * z0
print(f"fractions open / thin / snow-covered: {fr.round(2)}  ->  z0m = {z0:.1e} m,  z0T = alpha_T * z0m = {zT:.1e} m")
'''), ("md", r'''
## Step 4 — transfer coefficient and flux
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
## How $C_\mathrm{h}$ reacts to stability

The stability dependence comes from Monin–Obukhov similarity theory; several published formulations exist.
The paper uses Gryanik et al. (2020, "GLGS20") as reference and tests seven others. The shading marks the range of
$Ri_\mathrm{b}$ sampled during the campaign (central 90 % of the segments). On the unstable side all schemes share the same
function. On the stable side they differ — but there $C_\mathrm{h}$, and with it the flux, is small in every scheme.
Chapter 5 shows what this means for the comparison: the choice of scheme hardly matters.
'''), ("code", r'''
ri = np.linspace(-1, 0.5, 200)
fig, ax = plt.subplots(figsize=(6.5, 3.4))
for name in ["GLGS20", "BD", "HB88", "BH91", "CB05", "GAFGP07"]:
    ax.plot(ri, [Cd_Ch_noniterative(r, 24.0, 3e-4, 3e-5, name)[1] * 1e3 for r in ri], label=name, lw=2 if name == "GLGS20" else 1)
lo, hi = np.nanpercentile(FIG.Ri_obs.values, [5, 95]); ax.axvspan(lo, hi, color="0.85")
ax.axvline(0, color="k", lw=0.6); ax.set_xlabel("bulk Richardson number"); ax.set_ylabel("$C_h$ × 10$^3$"); ax.legend(ncol=2, fontsize=8)
ax.set_title("z = 24 m, z$_{0m}$ = 3·10$^{-4}$ m, α$_T$ = 0.1");
'''), ("md", r'''
## How $C_\mathrm{h}$ reacts to $\alpha_T$

$\alpha_T = z_{0,T}/z_{0,m}$ is poorly known over sea ice: models use values between 1 (ERA5) and 0.1 (large-eddy
simulations, the paper's reference), theory and earlier aircraft measurements point to much smaller values.
$C_\mathrm{h}$ depends on it only *logarithmically*:
'''), ("code", r'''
a = np.logspace(-7, 0, 50)
ch = np.array([Cd_Ch_noniterative(0.0, 24.0, 3e-4, 3e-4 * x, "GLGS20")[1] for x in a])
fig, ax = plt.subplots(figsize=(6, 3)); ax.semilogx(a, ch / ch[np.argmin(abs(a - 0.1))], color="k")
ax.axvline(0.1, color="C2", ls="--"); ax.axvline(1e-4, color="C1", ls="--")
ax.set_xlabel(r"$\alpha_T$"); ax.set_ylabel(r"$C_h$ relative to $\alpha_T$ = 0.1")
print(f"alpha_T from 1e-1 to 1e-4 (three orders of magnitude) changes C_h by a factor {ch[np.argmin(abs(a - 1e-4))] / ch[np.argmin(abs(a - 0.1))]:.1f}")
'''), ("md", r'''
**Take-away.** The bulk flux is a product of four things: wind, temperature difference, and — inside $C_\mathrm{h}$ —
roughness and stability. Roughness acts as a constant scaling factor on the flux (it changes the *amplitude*); it cannot
change *where* along the track the flux is large or small. That is decided by the temperature difference and the wind.
This distinction organises the results in the next two chapters.
''')]

# ═════════════════════════════════════════════════════════════════════════════
NB4 = [("md", r'''
# 4 · How well do the estimates reproduce the measured flux?

## Why not simply the RMSE?

Most measured fluxes are within a few W m⁻² of zero (Chapter 1). An estimate that is *always close to zero* then has a
small error although it contains no information. The table below makes the point with a forecast that contains no information at
all: $H = 0$ on every segment. Compare its mean absolute error (MAE) and RMSE with those of the real estimates.

The paper therefore uses two separate measures:

* **slope $b$** of the regression of the estimate on the EC flux — the fraction of the observed *amplitude* that is recovered (ideal: 1),
* **correlation $r$** — whether the estimate is large where the measurement is large (ideal: 1).

The two are linked: the slope equals $r$ times the ratio of the standard deviations of estimate and reference. A slope near 1
can therefore mean "right amplitude and good correlation" or "too much variance and poor correlation"; read $b$ and $r$ together.

> [!NOTE]
> **In the paper:** Sect. 4.1, Table 4, Figs. 4–6. **In the code:** `fluxpy/sensitivity.py` (`skill_metrics`),
> `scripts/manuscript_numbers.py` (every number of Sect. 4.1), `figures_manuscript/fig06_flux_scatter.py`.
'''), ("code", SETUP + r'''
from fluxpy.sensitivity import skill_metrics
import logging; logging.disable(logging.INFO)
NAMES = {"H_obs_bulk": "H_S,obs,bulk  (measured inputs)", "H_rs_bulk": "H_S,RS,bulk   (MODIS + ERA5)",
         "H_era_bulk": "H_S,ERA,bulk  (ERA5 only)", "H_era_diag": "H_S,ERA,diag  (ERA5's own flux)", "H_car_diag": "H_S,CAR,diag  (CARRA's own flux)"}
rows = {}
for key, name in NAMES.items():
    h = FIG[key].values; ok = np.isfinite(h)
    m = skill_metrics(EC, h, key)
    rows[name] = {"slope b": m["slope"], "r": m["r"], "sd ratio": m["std_ratio"], "bias (W m-2)": m["bias"],
                  "MAE (W m-2)": np.mean(np.abs(h[ok] - EC[ok])), "RMSE (W m-2)": m["rmse"], "n": m["n"]}
rows["zero flux  (H = 0 everywhere)"] = {"bias (W m-2)": -EC.mean(), "MAE (W m-2)": np.mean(np.abs(EC)),
                                        "RMSE (W m-2)": np.sqrt(np.mean(EC ** 2)), "n": EC.size}
table = pd.DataFrame(rows).T.round(1); table["n"] = table["n"].astype(int)
table.fillna("–")
'''), ("md", r'''
(CARRA is not available on 13 April, hence the smaller *n*.) The forecast "no flux anywhere" has a smaller MAE and RMSE than
every estimate in the table: in this weak-flux regime the error measures reward caution, not skill.

## The three bulk estimates against the reference (Fig. 6)
'''), ("code", r'''
fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), sharex=True, sharey=True)
for ax, key, c in zip(axs, ["H_obs_bulk", "H_rs_bulk", "H_era_bulk"], ["C2", "C1", "C0"]):
    h = FIG[key].values; ok = np.isfinite(EC) & np.isfinite(h); b, a = np.polyfit(EC[ok], h[ok], 1)
    ax.scatter(EC[ok], h[ok], s=12, color=c, alpha=0.7); xx = np.array([-8, 14])
    ax.plot(xx, xx, "k--", lw=0.8); ax.plot(xx, a + b * xx, color=c, lw=2)
    ax.set_title(f"{NAMES[key].split('(')[1][:-1]}\nslope {b:.1f}, r {np.corrcoef(EC[ok], h[ok])[0, 1]:.1f}", fontsize=9)
    ax.set_xlabel("EC flux (W m$^{-2}$)"); ax.set_xlim(-8, 14); ax.set_ylim(-30, 30)
axs[0].set_ylabel("bulk flux (W m$^{-2}$)");
'''), ("md", r'''
Reading the three panels from right to left is reading the paper's main result (the numbers are in the panel titles):

* **ERA5 only** — the points lie on a nearly flat line: only a small part of the amplitude is recovered and the correlation is weak.
* **MODIS surface temperature instead of ERA5's** — the atmosphere is still ERA5, yet amplitude and correlation both rise.
* **measured inputs** — the best correlation, but now the amplitude is *over*estimated (slope well above 1).
  With correct inputs the conventional transfer coefficient is too large; Chapter 5 follows this up.

## Where does the satellite surface temperature help?

If the benefit of MODIS comes from resolving leads and thin ice that ERA5 cannot see, it should be concentrated on
segments with a *heterogeneous* surface. The segments are split at the median of the within-segment standard
deviation of the KT-19 surface temperature.
'''), ("code", r'''
sd = SEG.T_sfc_std.values; rs, era = FIG.H_rs_bulk.values, FIG.H_era_bulk.values
ok = np.isfinite(EC) & np.isfinite(rs) & np.isfinite(era) & np.isfinite(sd); med = np.median(sd[ok])
rows = {}
for label, m in (("heterogeneous half", ok & (sd > med)), ("homogeneous half", ok & (sd <= med))):
    rows[label] = {"n": int(m.sum()), "r  ERA5 only": np.corrcoef(EC[m], era[m])[0, 1], "r  with MODIS": np.corrcoef(EC[m], rs[m])[0, 1],
                   "slope  ERA5 only": np.polyfit(EC[m], era[m], 1)[0], "slope  with MODIS": np.polyfit(EC[m], rs[m], 1)[0]}
print(f"median within-segment standard deviation of the surface temperature: {med:.1f} K"); pd.DataFrame(rows).T.round(1)
'''), ("md", r'''
**Take-away.** The surface temperature decides whether a bulk estimate follows the observed variability.
ERA5's surface temperature cannot represent leads and thin ice inside a grid cell; on heterogeneous segments its flux
is even anti-correlated with the measurement, and this is where replacing it by the MODIS surface temperature changes the most.
How firmly this rests on 91 segments that come from a handful of legs is the subject of Chapter 6.
''')]

# ═════════════════════════════════════════════════════════════════════════════
NB5 = [("md", r'''
# 5 · What does the result depend on — and how uncertain is it?

Chapter 4 ended with a puzzle: with *measured* inputs the bulk method overestimates the flux amplitude.
Which of the choices hidden in the transfer coefficient is responsible? The paper changes one at a time:
the stability scheme, the scalar roughness ratio $\alpha_T$, the momentum roughness — and perturbs the two inputs,
temperature difference and wind.

> [!NOTE]
> **In the paper:** Sect. 4.2, Table 3, Fig. 7, Appendix C (Table C1), Table 2. **In the code:** `fluxpy/sensitivity.py`
> (the configurations), `scripts/calc_bulk_and_ec_fluxes.py` (stages 3 and 4), `fluxpy/uncertainty.py` (Table 2),
> `scripts/make_manuscript_tables.py`.
'''), ("code", SETUP + r'''
import logging; logging.disable(logging.INFO)
S = pd.read_csv(REPO / "data/model_output/sensitivity_summary_table.csv").set_index("label")
S = S[~S.index.str.contains("era5only")]
'''), ("md", r'''
## $\alpha_T$ changes the amplitude — and nothing else
'''), ("code", r'''
a = S[S.index.str.contains("ztr") | (S.index == "GLGS20_ref")].sort_values("zt_ratio")
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True)
for ax, col, ideal, lab in ((axs[0], "slope", 1, "slope b (amplitude)"), (axs[1], "r", None, "correlation r")):
    ax.semilogx(a.zt_ratio, a["obs_" + col], "o-", color="C2", label="measured inputs"); ax.semilogx(a.zt_ratio, a["rs_" + col], "s-", color="C1", label="MODIS + ERA5")
    if ideal: ax.axhline(ideal, color="k", ls="--", lw=0.8)
    ax.set_xlabel(r"$\alpha_T$"); ax.set_ylabel(lab)
axs[1].set_ylim(0, 1); axs[0].legend(fontsize=8); axs[0].set_xlim(2e-8, 0.3)
tab = a.drop_duplicates("zt_ratio")[["zt_ratio", "obs_slope", "obs_r", "rs_slope", "rs_r"]].round(1)
tab["zt_ratio"] = tab["zt_ratio"].map("{:.0e}".format)
tab.rename(columns={"zt_ratio": "alpha_T", "obs_slope": "slope (measured inputs)", "obs_r": "r (measured inputs)",
                    "rs_slope": "slope (MODIS + ERA5)", "rs_r": "r (MODIS + ERA5)"}).set_index("alpha_T")
'''), ("md", r'''
With measured inputs the amplitude is matched (slope 1) near $\alpha_T = 10^{-4}$ — the *amplitude-matched configuration*
of the paper, three orders of magnitude below the conventional value and close to what earlier aircraft measurements
over sea ice found. The correlation does not move at all: as explained in Chapter 3, roughness only rescales the flux.

The paper does **not** read this as a measurement of $z_{0,T}$: any error that scales the flux — for instance in the
surface temperature — would be absorbed by the same parameter. The figure also shows the price of tuning: the $\alpha_T$ that
gives the in-situ estimate the right amplitude makes the ERA5–MODIS estimate, which already recovers too little amplitude,
recover even less. A single $\alpha_T$ cannot correct both.

## Everything else is secondary
'''), ("code", r'''
groups = {"stability scheme (8 schemes)": S.index.str.endswith("_ref"),
          "alpha_T (1e-4 ... 1e-1)": (S.index.str.contains("ztr") & S.zt_ratio.between(1e-4, 1e-1)) | (S.index == "GLGS20_ref"),
          "momentum roughness set": S.index.str.contains("mosaic|sheba|nschnier") | (S.index == "GLGS20_ref"),
          "constant momentum roughness": S.index.str.contains("z0c")}
rng = lambda x: f"{x.min():.1f} - {x.max():.1f}"
pd.DataFrame({g: {"slope, measured inputs": rng(S[m].obs_slope), "r, measured inputs": rng(S[m].obs_r),
                  "slope, MODIS + ERA5": rng(S[m].rs_slope), "r, MODIS + ERA5": rng(S[m].rs_r)} for g, m in groups.items()}).T
'''), ("md", r'''
No choice in the transfer coefficient changes the correlation. Among them only the roughness lengths change the amplitude
appreciably; the eight stability schemes are practically indistinguishable, because the sampled conditions are close to neutral (Chapter 3).

## Sensitivity to the inputs
'''), ("code", r'''
from fluxpy.uncertainty import central_difference, uncertainty_budget
keep = np.isfinite(EC); q = lambda x: f"{np.nanmedian(x):+.1f}  [{np.nanpercentile(x, 25):+.1f}, {np.nanpercentile(x, 75):+.1f}]"
pd.DataFrame({name: {"per +1 K in (theta_air - theta_s)   (W m-2)": q(central_difference(FIG, f"H_{k}_dTair", "dT")[keep]),
                     "per +1 m/s in wind speed   (W m-2)": q(central_difference(FIG, f"H_{k}_dU", "dU")[keep])}
              for k, name in (("obs", "measured inputs"), ("rs", "MODIS + ERA5"))}).T
'''), ("md", r'''
(median and inter-quartile range over the segments.) One kelvin in the temperature difference matters far more than one
metre per second in wind — and the typical temperature *difference* between air and surface is itself only about one
kelvin, the same size as the uncertainty of a satellite surface temperature. This is why the surface temperature product is decisive (Chapter 4).

## Uncertainty budget (Table 2 of the paper)

For the bulk fluxes, the assumed uncertainties of the inputs are propagated through the sensitivities above
(first-order Gaussian error propagation).
'''), ("code", r'''
from fluxpy.uncertainty import INPUT_UNCERTAINTY
U = uncertainty_budget(FIG)
print("assumed 1-sigma input uncertainties (T_s [K], T_air [K], U [m/s]):", INPUT_UNCERTAINTY)
pd.DataFrame({name: [np.nanmedian(U[k]), np.nanpercentile(U[k], 25), np.nanpercentile(U[k], 75)]
              for k, name in (("ec_total", "EC reference"), ("obs_bulk", "bulk, measured inputs"), ("rs_bulk", "bulk, MODIS + ERA5"))},
             index=["median (W m-2)", "25th percentile", "75th percentile"]).round(1).T
'''), ("md", r'''
**Take-away.** The measured reference is known to about 1 W m⁻². A bulk estimate driven by satellite and reanalysis
inputs is uncertain by several W m⁻² — more than the typical flux itself — and most of that comes from the
temperature difference. Improving pan-Arctic flux estimates is therefore first of all a question of better surface
temperatures, and only then of better transfer coefficients.
''')]

# ═════════════════════════════════════════════════════════════════════════════
NB6 = [("md", r'''
# 6 · How robust is the result?

91 segments, four flight days, one instrument: every conclusion of the previous chapters rests on a small sample.
This chapter takes the objections a critical reader raises, one at a time, and checks each against the data.
Some conclusions come out unchanged; others are weaker than a naive count of 91 segments suggests.

> [!NOTE]
> **In the paper:** these checks answer review comments and are not (yet) part of the text. **In the code:**
> `scripts/robustness_checks.py` (`pixi run robustness`) and `scripts/ec_band_sensitivity.py` (`pixi run ec-bands`)
> print every number of this chapter, and `pixi run check` keeps them under version control.
'''), ("code", SETUP + r'''
sys.path.insert(0, str(REPO / "scripts"))
import logging; logging.disable(logging.WARNING)
T = pd.to_datetime(FIG.start_time.values); DAY = np.asarray(T.strftime("%d %b"))
LEG = np.cumsum(np.r_[0, np.diff(T.values) / np.timedelta64(1, "m")] > 10)     # a new leg after a gap of > 10 min
EST = {"H_obs_bulk": ("measured inputs", "C2"), "H_rs_bulk": ("MODIS + ERA5", "C1"), "H_era_bulk": ("ERA5 only", "C0")}

def slope_r(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return np.polyfit(x[ok], y[ok], 1)[0], np.corrcoef(x[ok], y[ok])[0, 1]
'''), ("md", r'''
## Neighbouring segments are not independent

A leg crosses the same ice regime, the same few ERA5 grid cells and the same satellite scene for tens of kilometres,
so neighbouring 5 km segments resemble each other. Counting them as 91 independent samples overstates how well the
numbers are known. How strong is the resemblance?
'''), ("code", r'''
fig, ax = plt.subplots(figsize=(8, 2.8))
for leg in np.unique(LEG):
    m = LEG == leg
    ax.plot(np.where(m)[0], EC[m], "o-", ms=3, lw=1)
ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("segment number (one colour per leg)"); ax.set_ylabel("EC flux (W m$^{-2}$)")
same = LEG[1:] == LEG[:-1]
rho = np.corrcoef(EC[:-1][same], EC[1:][same])[0, 1]
print(f"{np.unique(LEG).size} legs; correlation of each EC flux with its neighbour on the same leg: {rho:.1f}")
print(f"-> effective number of independent samples n (1 - rho) / (1 + rho) = {EC.size * (1 - rho) / (1 + rho):.0f}, not {EC.size}")
'''), ("md", r'''
The standard remedy is to resample *blocks* instead of single segments when a confidence interval is bootstrapped:
blocks of three neighbouring segments (15 km), or whole legs. The intervals widen. Where an interval still excludes
the value of interest, the conclusion does not depend on counting neighbours twice.
'''), ("code", r'''
# (code collapsed on the website)
rng = np.random.default_rng(7)
blocks =[np.where(LEG == l)[0][i:i + 3] for l in np.unique(LEG) for i in range(max((LEG == l).sum() - 2, 0))]
def draw(unit):
    if unit == "segments":
        return rng.integers(0, EC.size, EC.size)
    if unit == "blocks of 3":
        return np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), EC.size // 3 + 1)])[:EC.size]
    legs = np.unique(LEG)
    return np.concatenate([np.where(LEG == l)[0] for l in rng.choice(legs, legs.size)])

units, interval = ["segments", "blocks of 3", "legs"], {}
fig, ax = plt.subplots(figsize=(7, 3))
for k, (key, (name, c)) in enumerate(EST.items()):
    h = FIG[key].values
    for j, unit in enumerate(units):
        b = np.array([slope_r(EC[i], h[i])[0] for i in (draw(unit) for _ in range(1000))])
        interval[key, unit] = np.nanpercentile(b, [2.5, 97.5])
        ax.plot(interval[key, unit], [k + 0.22 * (j - 1)] * 2, color=c, lw=4 - 1.3 * j, solid_capstyle="butt")
    ax.plot(slope_r(EC, h)[0], k, "k|", ms=14)
ax.axvline(1, color="k", ls="--", lw=0.8); ax.axvline(0, color="0.6", lw=0.6)
ax.set_yticks(range(3)); ax.set_yticklabels([v[0] for v in EST.values()]); ax.set_xlabel("95 % interval of the slope b")
ax.set_title("resampled: single segments (thick), blocks of 3 (medium), whole legs (thin)", fontsize=9)
lo, hi = interval["H_obs_bulk", "legs"]
print(f"measured inputs, whole legs resampled: slope in [{lo:.1f}, {hi:.1f}] -> the overestimation of the amplitude "
      + ("holds" if lo > 1 else "is not established") + " with legs as the unit")
for key in ("H_rs_bulk", "H_era_bulk"):
    lo, hi = interval[key, "legs"]
    print(f"{EST[key][0]}, whole legs resampled: slope in [{lo:.1f}, {hi:.1f}]" + (" (includes zero)" if lo <= 0 else ""))
'''), ("md", r'''
The same question for the satellite benefit of Chapter 4: on the heterogeneous half of the segments, the correlation of
the ERA5–MODIS estimate exceeds that of the ERA5-only estimate by a wide margin. How often does a resample show no gain?
'''), ("code", r'''
# (code collapsed on the website)
sd = SEG.T_sfc_std.values; hetero = sd > np.nanmedian(sd)
rs, era = FIG.H_rs_bulk.values, FIG.H_era_bulk.values
def gain(i):
    i = i[hetero[i]]
    return np.corrcoef(EC[i], rs[i])[0, 1] - np.corrcoef(EC[i], era[i])[0, 1]
print(f"gain in correlation on the heterogeneous half: {gain(np.arange(EC.size)):+.1f}  "
      f"(heterogeneous segments per day: " + ", ".join(f"{d} {int((hetero & (DAY == d)).sum())}" for d in pd.unique(DAY)) + ")")
for unit in units:
    g = np.array([gain(draw(unit)) for _ in range(1000)]); g = g[np.isfinite(g)]
    print(f"  resampling {unit:<12}: 95 % interval [{np.percentile(g, 2.5):+.1f}, {np.percentile(g, 97.5):+.1f}], "
          f"resamples without gain: {np.mean(g <= 0) * 100:.1f} %")
'''), ("md", r'''
Resampling segments or short blocks, a zero gain practically never occurs. With whole legs as the unit it does in a few
per cent of the resamples: the heterogeneous segments come from only a few legs, and most of them from two days.
The gain is large and consistent in sign, but "highly significant" is a statement about segments, not about legs.

## Does one flight day carry the result?
'''), ("code", r'''
rows = {}
for day in pd.unique(DAY):
    keep = DAY != day
    rows[f"without {day} (n = {keep.sum()})"] = {f"{name}: b / r": "{:.1f} / {:.1f}".format(*slope_r(EC[keep], FIG[key].values[keep]))
                                                 for key, (name, _) in EST.items()}
rows[f"all days (n = {EC.size})"] = {f"{name}: b / r": "{:.1f} / {:.1f}".format(*slope_r(EC, FIG[key].values))
                                     for key, (name, _) in EST.items()}
pd.DataFrame(rows).T
'''), ("md", r'''
The measured-input estimate keeps the highest correlation whichever day is left out. The two reanalysis-driven estimates
depend much more on the day: without 15 April, the day with open leads and the largest fluxes, their correlations drop.

## Where MODIS actually sees the surface

MODIS measures only through clear sky. Where a segment is cloud-covered, the ERA5-MODIS forcing falls back on the ERA5 skin
temperature, and the two reanalysis-driven estimates become identical. How much of the satellite forcing is really satellite?
'''), ("code", r'''
# (code collapsed on the website)
cache = xr.open_dataset(AR / "along_track_inputs.nc")
genuine = np.isfinite(cache.modis_IST.values)
print("segments with a cloud-free MODIS value: " + ", ".join(f"{d} {int(genuine[DAY == d].sum())}/{int((DAY == d).sum())}"
                                                           for d in pd.unique(DAY)) + f", all {genuine.sum()}/{genuine.size}")
kt19 = FIG.T_sfc_C.values
fig, axs = plt.subplots(1, 2, figsize=(8, 3.4), sharex=True, sharey=True)
for ax, (var, name, c) in zip(axs, [("T_sfc_rs_C", "MODIS", "C1"), ("T_sfc_era_C", "ERA5", "C0")]):
    ts = FIG[var].values
    ax.scatter(kt19[genuine], ts[genuine], s=12, color=c)
    ax.plot([-30, 2], [-30, 2], "k--", lw=0.8); ax.set_xlabel("KT-19 surface temperature (°C)")
    d = ts[genuine] - kt19[genuine]
    ax.set_title(f"{name}: bias {d.mean():+.1f} K, RMSE {np.sqrt(np.mean(d ** 2)):.1f} K, r {np.corrcoef(kt19[genuine], ts[genuine])[0, 1]:.1f}", fontsize=9)
axs[0].set_ylabel("product surface temperature (°C)")
for label, m in (("cloud-free", genuine), ("gap-filled", ~genuine)):
    print(f"{label} segments (n = {m.sum()}): " + "   ".join(f"{name} b {b:.1f}, r {r:.1f}" for key, (name, _) in EST.items()
                                                          for b, r in [slope_r(EC[m], FIG[key].values[m])]))
'''), ("md", r'''
On the segments MODIS sees, its surface temperature follows the KT-19 much more closely than ERA5's (left vs right) — the
mechanism behind the satellite benefit. Split by cloud cover, the two reanalysis-driven estimates correlate less well in each
subset than over all segments together: part of their overall correlation comes from the contrast *between* days (large fluxes on the cloud-free 15 April, small ones on
the cloudy 23 April), not from the variability within them. On 23 April no segment has a MODIS value, so the ERA5–MODIS and
ERA5-only estimates coincide there.

## Which eddies count? The frequency band of the EC flux

The EC flux is the covariance carried by 0.1 to 20 Hz (Chapter 2). Slower variations are left out as mesoscale changes
along the track rather than turbulence. But part of them may be turbulent flux after all, and the choice of the cut-off then
changes the reference that everything is compared with. Here the flux of every segment is recomputed with other lower
cut-offs.
'''), ("code", r'''
# (code collapsed on the website)
from scipy.signal import detrend
from fluxpy.eddy_covariance import band_covariance
from fluxpy.util import CP, add_theta, air_density
TB =xr.open_dataset(REPO / "data/observations/processed/t-bird_unified/noseboom_unified.nc").sortby("time")
cutoffs = [0.0, 0.05, 0.1, 0.2, 0.5]                                   # lower edge of the band [Hz]; upper edge 20 Hz
H_band = np.zeros((len(cutoffs), EC.size))
for j, (t0, t1) in enumerate(zip(SEG.start_time.values, SEG.end_time.values)):
    seg = add_theta(TB.sel(time=slice(t0, t1))).dropna("time", how="any")
    w, th = detrend(seg.w.values.astype(float)), detrend(seg.theta.values)
    k = float(np.mean(air_density(seg["T"].values, seg.p.values))) * CP
    H_band[:, j] = [k * band_covariance(w, th, 100.0, f0, 20.0) for f0 in cutoffs]
large = np.abs(H_band[2]) > 2
fig, axs = plt.subplots(1, 2, figsize=(9, 3))
axs[0].plot(range(len(cutoffs)), [np.median(H_band[i][large] / H_band[2][large]) for i in range(len(cutoffs))], "ko-")
axs[0].set_ylabel("flux / flux(0.1-20 Hz)\n(median, |H| > 2 W m$^{-2}$)")
for key, (name, c) in EST.items():
    axs[1].plot(range(len(cutoffs)), [slope_r(H_band[i], FIG[key].values)[0] for i in range(len(cutoffs))], "o-", color=c, label=name)
axs[1].axhline(1, color="k", ls="--", lw=0.8); axs[1].set_ylabel("slope b against this EC flux"); axs[1].legend(fontsize=8)
for ax in axs:
    ax.set_xticks(range(len(cutoffs))); ax.set_xticklabels(["all < 20", "0.05", "0.1\n(paper)", "0.2", "0.5"])
    ax.set_xlabel("lower edge of the band (Hz)")
'''), ("md", r'''
Moving the lower edge changes the flux amplitude, and the regression slopes follow. The correlations hardly change
(`pixi run ec-bands` prints them). The slope of the measured-input estimate — the "overestimation of the amplitude" of
Chapter 4 — shrinks when slower motions are counted as flux. It is therefore a statement about the flux carried by eddies
of up to about 600 m, the scale that the band selects; whether the slower part is turbulence cannot be decided from a
5 km segment of about 85 s.

## How strong is the stability correction?

Chapter 3 showed that the stability schemes agree on the unstable side and differ on the stable side. How large is the
correction $f_h = C_\mathrm{h} / C_\mathrm{h,n}$ on the sampled segments, and how much do the eight schemes differ per segment?
'''), ("code", r'''
# (code collapsed on the website)
from calc_bulk_and_ec_fluxes import load_obs
from fluxpy.bulk import compute_bulk_shf_obs, reference_config, transfer_coefficients
obs = load_obs(); run = compute_bulk_shf_obs(obs, reference_config())
Ri, z0, zt, Ch = (run[v].values for v in ("Ri_obs", "z0_obs", "zt_obs", "Ch_obs"))
_, Chn = transfer_coefficients(np.zeros_like(Ri), obs.altitude.values, z0, zt, "GLGS20")
fh = Ch / Chn
fig, ax = plt.subplots(figsize=(6, 2.8))
ax.hist(fh[Ri > 0], bins=np.linspace(0, 2, 21), color="C0", alpha=0.7, label=f"stable (n = {(Ri > 0).sum()})")
ax.hist(fh[Ri <= 0], bins=np.linspace(0, 2, 21), color="C3", alpha=0.7, label=f"unstable (n = {(Ri <= 0).sum()})")
ax.axvline(1, color="k", ls="--", lw=0.8); ax.set_xlabel("$f_h = C_h / C_{h,n}$ (in situ, reference configuration)"); ax.legend(fontsize=8)
runs = [str(r) for r in FIG.run_obs.values]
schemes = FIG.H_obs_bulk_all.values[[i for i, r in enumerate(runs) if r.endswith("_ref")]]
spread = np.nanmax(schemes, axis=0) - np.nanmin(schemes, axis=0)
print(f"per-segment spread of the in-situ bulk flux across the eight schemes: median {np.median(spread):.1f}, maximum {spread.max():.1f} W m-2")
'''), ("md", r'''
The correction itself is not small: on the median stable segment it roughly halves the transfer coefficient, and on the
median unstable segment it raises it by about 40 %. What makes the choice of scheme immaterial is that the schemes *agree* where the flux is
large: all use the same unstable functions, and they differ only on the stable side, where the flux is small anyway.

## A segment that is not at one level

The segments are meant to be flown at constant height. The 100 Hz height record shows whether that holds.
'''), ("code", r'''
# (code collapsed on the website)
height_range = []
for t0, t1 in zip(SEG.start_time.values, SEG.end_time.values):
    h = pd.Series(TB.h.sel(time=slice(t0, t1)).values.astype(float)).rolling(1000, center=True).mean()
    height_range.append(h.max() - h.min())
height_range = np.asarray(height_range)
worst, typical = int(np.argmax(height_range)), int(np.argsort(height_range)[EC.size // 2])
fig, ax = plt.subplots(figsize=(7, 2.8))
for j, label in ((worst, "largest height change"), (typical, "typical segment")):
    h = TB.h.sel(time=slice(SEG.start_time.values[j], SEG.end_time.values[j]))
    ax.plot((h.time - h.time[0]) / np.timedelta64(1, "s"), h, label=f"{label}: {pd.Timestamp(SEG.start_time.values[j]):%d %b %H:%M}")
ax.set_xlabel("time in the segment (s)"); ax.set_ylabel("T-Bird height (m)"); ax.legend(fontsize=8)
level = height_range <= 20
print(f"segments whose height changes by more than 20 m: {(~level).sum()}")
print("without them: " + "   ".join(f"{name} b {b:.1f}, r {r:.1f}" for key, (name, _) in EST.items()
                                   for b, r in [slope_r(EC[level], FIG[key].values[level])]))
'''), ("md", r'''
One segment was flown while the T-Bird was being winched up; it also sets the maximum segment height of the data set.
Leaving it out changes none of the results.

**Take-away.** The measured-input estimate keeps the best correlation under every test, and its amplitude overestimation
survives resampling whole legs, although its size depends on which eddies are counted as flux. The satellite benefit
is large and has the same sign in every resample, but it rests on few legs and mostly on two days, so its significance is
best stated with legs, not segments, as the unit. The stability correction is substantial, but it is the same in every scheme
where it matters.
''')]

BOOK = {
    "01_overview.ipynb": NB1,
    "02_eddy_covariance.ipynb": NB2,
    "03_bulk_method.ipynb": NB3,
    "04_comparison.ipynb": NB4,
    "05_sensitivity_and_uncertainty.ipynb": NB5,
    "06_robustness.ipynb": NB6,
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
    only = sys.argv[1:]                                   # e.g. `python book/build_book.py 06` rebuilds chapter 6 only
    for fname, cells in BOOK.items():
        if only and not any(fname.startswith(o) for o in only):
            continue
        nb = new_notebook(metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
        nb.cells = [make_cell(kind, src) for kind, src in cells]
        NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
        nbformat.write(nb, HERE / fname)
        print("built", fname)

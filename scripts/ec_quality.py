#!/usr/bin/env python3
"""Quality of the EC reference: every number of manuscript Appendix A.

Sections (numbered as printed)
    1  Why ratio-type quality criteria are not used to discard segments: they respond to
       the size of the flux, not to the quality of the measurement (Appendix A3).
    2  Robustness: slope and correlation of four estimates under eight alternative
       screenings (Table A1, written to tables_manuscript/table_qc_robustness.tex).
    3  Random error: Finkelstein-Sims against surrogate data (which contain only the
       first term of the sum and are expected to be lower; Rannik et al., 2016) and
       against the segment-placement spread; sensitivity to the lag range; the factor
       missing from the shortcut form of the Lenschow et al. (1994) error (Appendix A2).
    4  Physical consistency: the direction of the measured flux against the measured
       temperature difference, without any transfer coefficient (Appendix A4).
    5  The pitch resonance of the towed T-Bird near 1 Hz: an upper bound of its effect
       on the fluxes and on the comparison (Appendix A4).

Run:  pixi run ec-quality      (about 1 minute; needs pixi run precompute)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import detrend, welch
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from fluxpy.eddy_covariance import band_covariance, band_covariance_random_error, band_pass  # noqa: E402
from fluxpy.qc import SLOPE_TOLERANCES_TABLE_A1, segment_spectra, turbulence_resolved  # noqa: E402
from fluxpy.sensitivity import slope_bootstrap_ci  # noqa: E402
from fluxpy.util import CP, add_theta, air_density  # noqa: E402

warnings.filterwarnings("ignore")
FS = 100.0
RESONANCE_HZ = (0.8, 1.2)                            # pitch resonance of the towed body
RESONANCE_EDGES = np.array([0.65, 0.8, 1.2, 1.5])    # resonance bin and its two neighbours
ESTIMATES = ["H_obs_bulk", "H_rs_bulk", "H_era_bulk", "H_era_diag"]   # columns of Table A1
RULE = "=" * 110

FIG = xr.open_dataset(REPO / "data/model_output/sensitivity_figure_data.nc")
SEG = xr.open_dataset(REPO / "data/observations/ar/analysis_ready_polar6_turbulent_fluxes_latest_5km.nc"
                      ).sel(start_time=FIG["start_time"].values)
TBIRD = xr.open_dataset(REPO / "data/observations/processed/t-bird_unified/noseboom_unified.nc").sortby("time")
PLACEMENT = xr.open_dataset(REPO / "data/observations/ar/ec_segment_placement.nc")
rng = np.random.default_rng(1)


# ── per-segment diagnostics ──────────────────────────────────────────────────

def spectral_peak(x, f_lo=RESONANCE_HZ[0], f_hi=RESONANCE_HZ[1]):
    """Power in the resonance band relative to the geometric mean of the two neighbouring bands."""
    f, P = welch(detrend(x), fs=FS, nperseg=2048)
    neighbours = ((f >= 0.45) & (f < 0.65)) | ((f > 1.5) & (f <= 1.9))
    return P[(f >= f_lo) & (f <= f_hi)].mean() / np.exp(np.log(P[neighbours]).mean())


def diagnostics(segment: xr.Dataset) -> dict:
    """Everything Appendix A needs from the 100 Hz series of one segment."""
    w, th = detrend(segment["w"].values), detrend(segment["theta"].values)
    n = w.size
    k = float(np.mean(air_density(segment["T"].values, segment["p"].values))) * CP     # rho * c_p: covariance -> W m-2
    H = k * band_covariance(w, th, FS)
    H_total = k * np.mean(w * th)                                                       # covariance of ALL frequencies
    wb, tb = band_pass(w, FS, 0.1, 20.0), band_pass(th, FS, 0.1, 20.0)

    # alternative screening metrics (Appendix A3): all of them ratios to the flux or built on the total covariance
    spectra = segment_spectra(w=segment["w"].values, theta=segment["theta"].values,
                              u=segment["u"].values, v=segment["v"].values, fs=FS)
    fifths = [np.mean(detrend(a) * detrend(b)) for a, b in zip(np.array_split(w, 5), np.array_split(th, 5))]
    stationarity = abs((k * np.mean(fifths) - H_total) / H_total)                      # Foken and Wichura (1996)
    H_half_1, H_half_2 = (k * band_covariance(detrend(a), detrend(b), FS)
                          for a, b in zip(np.array_split(w, 2), np.array_split(th, 2)))

    # random error three ways (Appendix A2)
    sigma_fs = k * band_covariance_random_error(w, th, FS)
    sigma_5s, sigma_30s = (k * band_covariance_random_error(w, th, FS, max_lag_s=m) for m in (5.0, 30.0))
    shifts = rng.integers(int(15 * FS), n - int(15 * FS), 300)                         # surrogates: theta shifted by > 15 s
    sigma_surrogate = np.std([k * np.mean(wb * np.roll(tb, int(s))) for s in shifts])
    r_wtheta = abs(np.corrcoef(wb, tb)[0, 1])

    # cospectrum around the pitch resonance (Appendix A4): replace the resonance bin by the
    # interpolation of its neighbours (in log-frequency density)
    f = np.fft.rfftfreq(n, 1 / FS)
    co = 2 * k * np.real(np.fft.rfft(w) * np.conj(np.fft.rfft(th))) / n ** 2
    band_sums = np.array([co[(f >= a) & (f < b)].sum() for a, b in zip(RESONANCE_EDGES[:-1], RESONANCE_EDGES[1:])])
    density = band_sums / np.log(RESONANCE_EDGES[1:] / RESONANCE_EDGES[:-1])
    H_interpolated = H - band_sums[1] + 0.5 * (density[0] + density[2]) * np.log(RESONANCE_EDGES[2] / RESONANCE_EDGES[1])

    return dict(H=H, H_total=H_total, R2=spectra["r2_cumulative_covariance"], fw_rel=stationarity,
                half_diff=abs(H_half_1 - H_half_2), sigma_fs=sigma_fs, sigma_m5=sigma_5s, sigma_m30=sigma_30s,
                sigma_sur=sigma_surrogate, r_wt=r_wtheta, H_interp=H_interpolated,
                dens_res=density[1], dens_nb=0.5 * (density[0] + density[2]),
                peak_w=spectral_peak(segment["w"].values), peak_theta=spectral_peak(segment["theta"].values),
                peak_pitch=spectral_peak(segment["pitch"].values))


rows = []
for t0, t1 in zip(SEG["start_time"].values, SEG["end_time"].values):
    rows.append(diagnostics(add_theta(TBIRD.sel(time=slice(t0, t1))).dropna("time", how="any")))
df = pd.DataFrame(rows, index=pd.to_datetime(SEG["start_time"].values))
df["sigma_place"] = PLACEMENT["sigma_placement"].sel(start_time=FIG["start_time"].values).values
for v in ESTIMATES:
    df[v] = FIG[v].values
abs_H = df["H"].abs()
dtheta_surface_minus_air = -FIG["dtheta_obs"].values
above_2sigma = (abs_H > 2.0 * df["sigma_fs"]).values


# ── 1  ratio-type criteria (Appendix A3) ─────────────────────────────────────
print(RULE)
print("1. RATIO-TYPE CRITERIA RESPOND TO THE SIZE OF THE FLUX")
print(RULE)
print(f"rank correlation with |H|:  R2 of the cumulative covariance {spearmanr(df.R2, abs_H)[0]:+.2f},  "
      f"relative stationarity test (Foken and Wichura, 1996) {spearmanr(df.fw_rel, abs_H)[0]:+.2f}")
low_r2 = df.R2 < 0.4
print(f"segments with R2 < 0.4: {low_r2.sum()};  median |H| {abs_H[low_r2].median():.1f} W m-2 (others: {abs_H[~low_r2].median():.1f});  "
      f"below 2 sigma_rand: {(~above_2sigma[low_r2]).mean() * 100:.0f} %;  "
      f"median |first half - second half| {df.half_diff[low_r2].median():.1f} W m-2 (others: {df.half_diff[~low_r2].median():.1f})")


# ── 2  robustness to the screening (Table A1) ────────────────────────────────
print("\n" + RULE)
print("2. ROBUSTNESS OF THE COMPARISON TO THE SCREENING  (slope b / correlation r against the EC reference)")
print(RULE)
SCREENINGS = [
    ("none (this study)", np.ones(len(df), bool)),
    (r"$R^2 \geq 0.4$ of the cumulative covariance", (df.R2 >= 0.4).values),
    (r"$R^2 \geq 0.6$", (df.R2 >= 0.6).values),
    (r"relative stationarity test $< 100\,\%$", (df.fw_rel <= 1.0).values),
    (r"relative stationarity test $< 30\,\%$", (df.fw_rel < 0.3).values),
    (r"segment halves agree within $3\,\sigma_\mathrm{rand}$", (df.half_diff <= 3 * np.sqrt(2) * df.sigma_fs).values),
    (r"only fluxes above $2\,\sigma_\mathrm{rand}$", above_2sigma),
    # integrity check (Appendix A1) as a screening at several tolerances of |s_w + 5/3|
    *[(rf"integrity check, $|s_w + 5/3| \leq {tol:.1f}$",
       np.array([turbulence_resolved(s, tol) for s in SEG["qc_slope_w"].values]))
      for tol in SLOPE_TOLERANCES_TABLE_A1],
    (r"pitch-resonance bin interpolated", None),          # no segment removed; the EC flux itself is changed
]


def slope_and_r(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return np.polyfit(x[ok], y[ok], 1)[0], np.corrcoef(x[ok], y[ok])[0, 1]


tex, table_a1 = [], []
for name, keep in SCREENINGS:
    x = df["H_interp"].values if keep is None else df["H"].values
    keep = np.ones(len(df), bool) if keep is None else keep
    cells = [slope_and_r(x[keep], df[c].values[keep]) for c in ESTIMATES]
    table_a1.append(np.ravel(cells))
    plain = (name.replace("$", "").replace("\\,", " ").replace("\\geq", ">=").replace("\\leq", "<=")
             .replace("\\%", "%").replace("\\sigma_\\mathrm{rand}", "sigma_rand"))
    print(f"  {plain:<48s} n = {keep.sum():2d}   " + "   ".join(f"{b:.2f} / {r:.2f}" for b, r in cells))
    tex.append(f"{name} & {keep.sum()} & " + " & ".join(f"{b:.2f} & {r:.2f}" for b, r in cells) + r" \\")
(REPO / "tables_manuscript" / "table_qc_robustness.tex").write_text("\n".join(tex) + "\n", encoding="utf-8", newline="\n")

lo, hi = slope_bootstrap_ci(df["H"].values, df["H_obs_bulk"].values)    # the interval of Table 4 and the Conclusions
print(f"  for scale: 95 % bootstrap interval of the T-Bird bulk slope without screening: [{lo:.2f}, {hi:.2f}]")


# ── 3  random error (Appendix A2) ────────────────────────────────────────────
print("\n" + RULE)
print("3. RANDOM ERROR: INDEPENDENT ESTIMATES")
print(RULE)
print(f"median random error (W m-2): Finkelstein-Sims {df.sigma_fs.median():.2f} | surrogate data {df.sigma_sur.median():.2f} | "
      f"segment-placement spread {df.sigma_place.median():.2f}")
print(f"rank correlation Finkelstein-Sims vs surrogates {spearmanr(df.sigma_fs, df.sigma_sur)[0]:.2f}, "
      f"vs placement spread {spearmanr(df.sigma_fs, df.sigma_place, nan_policy='omit')[0]:.2f}")
print(f"sensitivity to the lag range m (10 s used): median sigma_rand(m = 5 s) / sigma_rand(10 s) = {(df.sigma_m5 / df.sigma_fs).median():.2f}, "
      f"sigma_rand(30 s) / sigma_rand(10 s) = {(df.sigma_m30 / df.sigma_fs).median():.2f}")
print(f"median |correlation of w and theta| in the flux band: {df.r_wt.median():.2f}  "
      f"-> factor sqrt((1+r^2)/r^2) of Lenschow et al. (1994): {np.sqrt((1 + df.r_wt**2) / df.r_wt**2).median():.0f}")


# ── 4  direction of the flux vs measured temperature difference (Appendix A4) ──
print("\n" + RULE)
print("4. PHYSICAL CONSISTENCY: DIRECTION OF THE FLUX vs MEASURED TEMPERATURE DIFFERENCE")
print(RULE)
for k in (1, 2, 3):
    above = (abs_H > k * df["sigma_fs"]).values
    agree = np.sign(df["H"].values[above]) == np.sign(dtheta_surface_minus_air[above])
    small = (np.abs(df["H"].values[above][~agree]) < 0.5).sum()
    print(f"fluxes above {k} sigma_rand: {above.sum():2d}; direction agrees with sign(theta_s - theta_air) for {agree.sum():2d} "
          f"({agree.mean() * 100:.0f} %); of the {(~agree).sum()} others, {small} are smaller than 0.5 W m-2")


# ── 5  pitch resonance (Appendix A4) ─────────────────────────────────────────
print("\n" + RULE)
print("5. PITCH RESONANCE OF THE T-BIRD (0.8-1.2 Hz)")
print(RULE)
print(f"spectral peak relative to neighbouring bands, median: pitch {df.peak_pitch.median():.0f}x, "
      f"w {df.peak_w.median():.1f}x, theta {df.peak_theta.median():.1f}x")
large = (abs_H > 2).values
print(f"cospectral density in the resonance bin relative to its neighbours (mean over |H| > 2 W m-2, n = {large.sum()}): "
      f"{df.dens_res[large].mean() / df.dens_nb[large].mean():.1f}x")
reduction = (df["H"] - df["H_interp"])[large] / df["H"][large]
print(f"flux reduction if the resonance bin is replaced by the interpolation of its neighbours: "
      f"median {np.median(reduction) * 100:.0f} % (quartiles {np.percentile(reduction, 25) * 100:.0f} to {np.percentile(reduction, 75) * 100:.0f} %); "
      f"effect on the comparison: last row of the table above")
print(f"largest change of a slope or correlation of the comparison (last row against the first): "
      f"{np.max(np.abs(table_a1[-1] - table_a1[0])):.2f}")
print()

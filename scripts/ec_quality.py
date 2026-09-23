#!/usr/bin/env python3
"""Quality of the EC reference (manuscript Appendix A): every number quoted there.

1. Why ratio-type quality criteria are not used to discard segments: they respond
   to the size of the flux, not to the quality of the measurement.
2. Robustness: the comparison statistics under eight alternative screenings.
3. Consistency of the random error: Finkelstein-Sims vs surrogate data (which
   contain only the first term of the sum and are expected to be lower; Rannik et
   al., 2016) vs the segment-placement spread; sensitivity to the lag range; the
   factor missing in the shortcut form of the Lenschow et al. (1994) error.
4. Physical consistency: the direction of the measured flux against the measured
   temperature difference (no transfer coefficient involved).
5. The pitch resonance of the T-Bird near 1 Hz: upper bound of its effect.

Output: tables_manuscript/table_qc_robustness.tex and the printed numbers.
Run:    pixi run ec-quality      (about 1 minute)
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
from fluxpy.eddy_covariance import C_P, R, band_covariance, band_covariance_random_error, band_pass  # noqa: E402
from fluxpy.meteorology import add_theta  # noqa: E402
from fluxpy.qc import segment_spectra  # noqa: E402

warnings.filterwarnings("ignore")
FS = 100.0
RES_BAND = (0.8, 1.2)                        # Hz, pitch resonance of the towed body
RES_EDGES = np.array([0.65, 0.8, 1.2, 1.5])  # resonance bin and its two neighbours

FIG = xr.open_dataset(REPO / "data/model_output/sensitivity_figure_data.nc")
SEG = xr.open_dataset(
    REPO / "data/observations/ar/analysis_ready_polar6_turbulent_fluxes_latest_5km.nc"
).sel(start_time=FIG["start_time"].values)
TB = xr.open_dataset(REPO / "data/observations/processed/t-bird_unified/noseboom_unified.nc").sortby("time")
rng = np.random.default_rng(1)


def _peak(x, f_lo=RES_BAND[0], f_hi=RES_BAND[1]):
    """Spectral power in the resonance band relative to the geometric mean of the neighbouring bands."""
    f, P = welch(detrend(x), fs=FS, nperseg=2048)
    nb = ((f >= 0.45) & (f < 0.65)) | ((f > 1.5) & (f <= 1.9))
    return P[(f >= f_lo) & (f <= f_hi)].mean() / np.exp(np.log(P[nb]).mean())


rows = []
for t0, t1 in zip(SEG["start_time"].values, SEG["end_time"].values):
    s = add_theta(TB.sel(time=slice(t0, t1))).dropna("time", how="any")
    w, th = detrend(s["w"].values), detrend(s["theta"].values)
    n = w.size
    k = float((s["p"] * 100.0 / (R * (s["T"] + 273.15))).mean()) * C_P
    H, H_total = k * band_covariance(w, th, FS), k * np.mean(w * th)
    wb, tb = band_pass(w, FS, 0.1, 20.0), band_pass(th, FS, 0.1, 20.0)

    # alternative screening metrics (all of them ratios to the flux, or built on the total covariance)
    qc = segment_spectra(w=s["w"].values, theta=s["theta"].values, u=s["u"].values, v=s["v"].values, fs=FS)
    sub = [np.mean(detrend(a) * detrend(b)) for a, b in zip(np.array_split(w, 5), np.array_split(th, 5))]
    h1, h2 = (k * band_covariance(detrend(a), detrend(b), FS) for a, b in zip(np.array_split(w, 2), np.array_split(th, 2)))

    # random error three ways
    sigma_fs = k * band_covariance_random_error(w, th, FS)
    sigma_m = [k * band_covariance_random_error(w, th, FS, max_lag_s=m) for m in (5.0, 30.0)]   # sensitivity to the lag range
    sigma_sur = np.std([k * np.mean(wb * np.roll(tb, int(sh))) for sh in rng.integers(int(15 * FS), n - int(15 * FS), 300)])
    r_wt = np.corrcoef(wb, tb)[0, 1]

    # cospectrum around the pitch resonance: replace the resonance bin by the interpolation of its neighbours
    f = np.fft.rfftfreq(n, 1 / FS)
    co = 2 * k * np.real(np.fft.rfft(w) * np.conj(np.fft.rfft(th))) / n**2
    b = np.array([co[(f >= a) & (f < c)].sum() for a, c in zip(RES_EDGES[:-1], RES_EDGES[1:])])
    dens = b / np.log(RES_EDGES[1:] / RES_EDGES[:-1])
    H_interp = H - b[1] + 0.5 * (dens[0] + dens[2]) * np.log(RES_EDGES[2] / RES_EDGES[1])

    rows.append(dict(H=H, H_total=H_total, R2=qc["r2_cumulative_covariance"], fw_rel=abs((k * np.mean(sub) - H_total) / H_total),
                     half_diff=abs(h1 - h2), sigma_fs=sigma_fs, sigma_m5=sigma_m[0], sigma_m30=sigma_m[1], sigma_sur=sigma_sur, r_wt=abs(r_wt),
                     H_interp=H_interp, dens_res=dens[1], dens_nb=0.5 * (dens[0] + dens[2]),
                     peak_w=_peak(s["w"].values), peak_theta=_peak(s["theta"].values), peak_pitch=_peak(s["pitch"].values)))
df = pd.DataFrame(rows, index=pd.to_datetime(SEG["start_time"].values))
aH = df["H"].abs()
df["sigma_place"] = xr.open_dataset(REPO / "data/observations/ar/ec_segment_placement.nc")["sigma_placement"].sel(
    start_time=FIG["start_time"].values).values
for v in ("H_obs_bulk", "H_rs_bulk", "H_era_bulk", "H_era_diag"):
    df[v] = FIG[v].values
dtheta = -FIG["dtheta_obs"].values                     # theta_s - theta_air
det = (aH > 2.0 * df["sigma_fs"]).values            # "above 2 sigma"; 1 and 3 sigma are reported as well

print("=" * 110)
print("1. RATIO-TYPE CRITERIA RESPOND TO THE SIZE OF THE FLUX")
print("=" * 110)
print(f"rank correlation with |H|:  R2 of the cumulative covariance {spearmanr(df.R2, aH)[0]:+.1f},  "
      f"relative stationarity test (Foken and Wichura, 1996) {spearmanr(df.fw_rel, aH)[0]:+.1f}")
low = df.R2 < 0.4
print(f"segments with R2 < 0.4: {low.sum()};  median |H| {aH[low].median():.1f} W m-2 (others: {aH[~low].median():.1f});  "
      f"below 2 sigma_rand: {(~det[low]).mean() * 100:.0f} %;  "
      f"median |first half - second half| {df.half_diff[low].median():.1f} W m-2 (others: {df.half_diff[~low].median():.1f})")

print("\n" + "=" * 110)
print("2. ROBUSTNESS OF THE COMPARISON TO THE SCREENING  (slope b / correlation r against the EC reference)")
print("=" * 110)
SCREENINGS = [
    ("none (this study)", np.ones(len(df), bool)),
    (r"$R^2 \geq 0.4$ of the cumulative covariance", (df.R2 >= 0.4).values),
    (r"$R^2 \geq 0.6$", (df.R2 >= 0.6).values),
    (r"relative stationarity test $< 100\,\%$", (df.fw_rel <= 1.0).values),
    (r"relative stationarity test $< 30\,\%$", (df.fw_rel < 0.3).values),
    (r"segment halves agree within $3\,\sigma_\mathrm{rand}$", (df.half_diff <= 3 * np.sqrt(2) * df.sigma_fs).values),
    (r"only fluxes above $2\,\sigma_\mathrm{rand}$", det),
    (r"pitch-resonance bin interpolated", None),
]
COLS = ["H_obs_bulk", "H_rs_bulk", "H_era_bulk", "H_era_diag"]


def stats(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return np.polyfit(x[ok], y[ok], 1)[0], np.corrcoef(x[ok], y[ok])[0, 1]


tex = []
for name, m in SCREENINGS:
    x = df["H_interp"].values if m is None else df["H"].values
    m = np.ones(len(df), bool) if m is None else m
    cells = [stats(x[m], df[c].values[m]) for c in COLS]
    plain = name.replace("$", "").replace("\\,", " ").replace("\\geq", ">=").replace("\\%", "%").replace("\\sigma_\\mathrm{rand}", "sigma_rand")
    print(f"  {plain:<48s} n = {m.sum():2d}   " + "   ".join(f"{b:.1f} / {r:.1f}" for b, r in cells))
    tex.append(f"{name} & {m.sum()} & " + " & ".join(f"{b:.1f} & {r:.1f}" for b, r in cells) + r" \\")
out = REPO / "tables_manuscript" / "table_qc_robustness.tex"
out.write_text("\n".join(tex) + "\n", encoding="utf-8", newline="\n")
xb, yb = df["H"].values, df["H_obs_bulk"].values
_rb = np.random.default_rng(0)
boot = [np.polyfit(xb[i], yb[i], 1)[0] for i in (_rb.integers(0, len(df), len(df)) for _ in range(3000))]
print(f"  for scale: 95 % bootstrap interval of the T-Bird bulk slope without screening: "
      f"[{np.percentile(boot, 2.5):.1f}, {np.percentile(boot, 97.5):.1f}]")

print("\n" + "=" * 110)
print("3. RANDOM ERROR: INDEPENDENT ESTIMATES")
print("=" * 110)
print(f"median random error (W m-2): Finkelstein-Sims {df.sigma_fs.median():.1f} | surrogate data {df.sigma_sur.median():.1f} | "
      f"segment-placement spread {df.sigma_place.median():.1f}")
print(f"rank correlation Finkelstein-Sims vs surrogates {spearmanr(df.sigma_fs, df.sigma_sur)[0]:.2f}, "
      f"vs placement spread {spearmanr(df.sigma_fs, df.sigma_place, nan_policy='omit')[0]:.1f}")
print(f"sensitivity to the lag range m (10 s used): median sigma_rand(m = 5 s) / sigma_rand(10 s) = {(df.sigma_m5 / df.sigma_fs).median():.2f}, "
      f"sigma_rand(30 s) / sigma_rand(10 s) = {(df.sigma_m30 / df.sigma_fs).median():.2f}")
print(f"median |correlation of w and theta| in the flux band: {df.r_wt.median():.2f}  "
      f"-> factor sqrt((1+r^2)/r^2) of Lenschow et al. (1994): {np.sqrt((1 + df.r_wt**2) / df.r_wt**2).median():.0f}")

print("\n" + "=" * 110)
print("4. PHYSICAL CONSISTENCY: DIRECTION OF THE FLUX vs MEASURED TEMPERATURE DIFFERENCE")
print("=" * 110)
for k in (1, 2, 3):
    d = (aH > k * df["sigma_fs"]).values
    agree = np.sign(df["H"].values[d]) == np.sign(dtheta[d])
    small = (np.abs(df["H"].values[d][~agree]) < 0.5).sum()
    print(f"fluxes above {k} sigma_rand: {d.sum():2d}; direction agrees with sign(theta_s - theta_air) for {agree.sum():2d} "
          f"({agree.mean() * 100:.0f} %); of the {(~agree).sum()} others, {small} are smaller than 0.5 W m-2")

print("\n" + "=" * 110)
print("5. PITCH RESONANCE OF THE T-BIRD (0.8-1.2 Hz)")
print("=" * 110)
print(f"spectral peak relative to neighbouring bands, median: pitch {df.peak_pitch.median():.0f}x, "
      f"w {df.peak_w.median():.1f}x, theta {df.peak_theta.median():.1f}x")
big = (aH > 2).values
print(f"cospectral density in the resonance bin relative to its neighbours (mean over |H| > 2 W m-2, n = {big.sum()}): "
      f"{df.dens_res[big].mean() / df.dens_nb[big].mean():.1f}x")
rel = (df["H"] - df["H_interp"])[big] / df["H"][big]
print(f"flux reduction if the resonance bin is replaced by the interpolation of its neighbours: "
      f"median {np.median(rel) * 100:.0f} % (quartiles {np.percentile(rel, 25) * 100:.0f} to {np.percentile(rel, 75) * 100:.0f} %); "
      f"effect on the comparison: last row of the table above")
print()

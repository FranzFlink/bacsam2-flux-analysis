"""Authoritative number set for the manuscript (slope/correlation framework).

Single source of truth for every statistic printed in the paper. Run after any
pipeline change and copy the numbers from here -- never from an older PDF, a
notebook, or git history.

Framework (author decision 2026-07-27): the reference configuration is the
CONVENTIONAL alpha_T = 1e-1; amplitude skill is reported as a regression slope
and variability skill as a correlation, because RMSE in this low-flux regime
rewards estimates that shrink towards zero (see the ERA5-only row: best centred
RMSE, worst skill). The amplitude-matched alpha_T = 1e-4 is reported alongside.

Run:  pixi run python scripts/manuscript_numbers.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from fluxpy.sensitivity import amplitude_metrics, skill_metrics, slope_bootstrap_ci  # noqa: E402
FIG = xr.open_dataset(REPO / "data/model_output/sensitivity_figure_data.nc")
AR = xr.open_dataset(
    REPO / "data/observations/ar/analysis_ready_polar6_turbulent_fluxes_latest_5km.nc"
).pipe(lambda d: d.where(d.altitude <= 50, drop=True))

EC = np.asarray(FIG["H_obs_ec"])
T = pd.to_datetime(FIG["start_time"].values)
# leg identifier: a break whenever the along-track gap exceeds 10 minutes
LEG = np.cumsum(np.r_[0, np.diff(T.values) / np.timedelta64(1, "m")] > 10)

RUNS_O = [str(r) for r in FIG["run_obs"].values]
RUNS_R = [str(r) for r in FIG["run"].values]
HO_ALL = np.asarray(FIG["H_obs_bulk_all"])
HR_ALL = np.asarray(FIG["H_rs_bulk_all"])


def within_leg_r(h):
    m = np.isfinite(h) & np.isfinite(EC)
    d = pd.DataFrame({"ec": EC[m], "h": h[m], "leg": LEG[m]})
    a = d.groupby("leg").transform(lambda s: s - s.mean())
    return float(np.corrcoef(a["ec"], a["h"])[0, 1])


def line(name, h, ci=False):
    m = skill_metrics(EC, h, name)
    out = (f"{name:<26} {m['slope']:5.1f} {m['r']:6.1f} "
           f"{within_leg_r(h):8.1f} {m['bias']:+7.1f} {m['rmse']:6.1f} "
           f"{m['crmse']:6.1f} {m['n']:4d}")
    if ci:
        lo, hi = slope_bootstrap_ci(EC, h)
        out += f"   OLS 95% CI [{lo:.1f}, {hi:.1f}]"
    return out


print("\n" + "=" * 118)
ecv = EC[np.isfinite(EC)]
print(f"EC REFERENCE ({ecv.size} of {EC.size} low-level 5 km segments pass the integrity check)")
print("=" * 118)
_seg = AR.sel(start_time=FIG["start_time"].values)
print(f"integrity check (inertial-subrange slope of w): {int(_seg['turbulence_resolved'].sum())} of {EC.size} pass; "
      f"slope median {float(_seg['qc_slope_w'].median()):.1f}, range {float(_seg['qc_slope_w'].min()):.1f} to {float(_seg['qc_slope_w'].max()):.1f}")
_sig, _ratio = np.asarray(FIG["sigma_random_ec"]), np.asarray(FIG["ec_flux_over_sigma"])
print(f"random error: median {np.median(_sig):.1f} W m-2 [{np.percentile(_sig, 25):.1f}-{np.percentile(_sig, 75):.1f}]")
_days = pd.to_datetime(FIG["start_time"].values).strftime("%d %b")
from scipy.stats import norm  # noqa: E402
for _k in (1, 2, 3):
    _d = _ratio > _k
    print(f"   |H| > {_k} sigma_rand: {_d.sum():2d} of {EC.size}  (expected by chance if every true flux were zero: "
          f"{2 * (1 - norm.cdf(_k)) * EC.size:.1f});  per day: "
          + ", ".join(f"{d}: {_d[_days == d].sum()}/{(_days == d).sum()}" for d in pd.unique(_days)))
print(f"median {np.median(ecv):.1f}   mean {ecv.mean():.1f}   max {ecv.max():.1f}   "
      f"min {ecv.min():.1f}   |H|<2 W/m2: {np.mean(np.abs(ecv) < 2) * 100:.0f} % of segments")

print("\n" + "=" * 118)
print("REFERENCE CONFIGURATION  alpha_T = 1e-1 (conventional)")
print("=" * 118)
print(f"{'estimate':<26} {'slope':>5} {'r':>6} {'r_within':>8} "
      f"{'bias':>7} {'RMSE':>6} {'cRMSE':>6} {'n':>4}")
for key, nm in [("H_obs_bulk", "H_S,obs,bulk"), ("H_rs_bulk", "H_S,RS,bulk"),
                ("H_era_bulk", "H_S,ERA,bulk"), ("H_era_diag", "H_S,ERA,diag"),
                ("H_car_diag", "H_S,CAR,diag")]:
    print(line(nm, np.asarray(FIG[key]), ci=(key in ("H_obs_bulk", "H_rs_bulk"))))

print("\n" + "=" * 118)
print("AMPLITUDE-MATCHED CONFIGURATION  alpha_T = 1e-4  (Schroeder et al., 2003)")
print("=" * 118)
print(line("H_S,obs,bulk", HO_ALL[RUNS_O.index("GLGS20_ztr1e-04")], ci=True))
print(line("H_S,RS,bulk", HR_ALL[RUNS_R.index("GLGS20_ztr1e-04")]))
print(line("H_S,ERA,bulk", HR_ALL[RUNS_R.index("GLGS20_ztr1e-04_era5only")]))

print("\n" + "=" * 118)
print("alpha_T SWEEP  (slope / r / bias / RMSE)   -- note r is INVARIANT to alpha_T")
print("=" * 118)
print(f"{'alpha_T':>8} | {'T-Bird slope':>12} {'r':>5} {'bias':>6} {'RMSE':>5} |"
      f" {'ERA5-MODIS slope':>16} {'r':>5} | {'ERA5-only slope':>15} {'r':>5}")
for a in ["1e-01", "1e-02", "1e-03", "1e-04", "1e-05", "1e-06", "1e-07"]:
    try:
        o = skill_metrics(EC, HO_ALL[RUNS_O.index(f"GLGS20_ztr{a}")], "")
        r_ = skill_metrics(EC, HR_ALL[RUNS_R.index(f"GLGS20_ztr{a}")], "")
        e_ = skill_metrics(EC, HR_ALL[RUNS_R.index(f"GLGS20_ztr{a}_era5only")], "")
    except ValueError:
        continue
    print(f"{a:>8} | {o['slope']:12.1f} {o['r']:5.1f} {o['bias']:+6.1f} {o['rmse']:5.1f} |"
          f" {r_['slope']:16.1f} {r_['r']:5.1f} | {e_['slope']:15.1f} {e_['r']:5.1f}")

print("\n" + "=" * 118)
print("SATELLITE T_s BENEFIT vs WITHIN-SEGMENT T_s VARIABILITY  (reference configuration)")
print("=" * 118)
tss = pd.Series(np.asarray(AR["T_sfc_std"]),
                index=pd.to_datetime(AR["start_time"].values))
v = tss.reindex(T, method="nearest", tolerance=pd.Timedelta("60s")).values
rs = np.asarray(FIG["H_rs_bulk"])
era = np.asarray(FIG["H_era_bulk"])
ok = np.isfinite(EC) & np.isfinite(rs) & np.isfinite(era) & np.isfinite(v)
e, r_, a_, vv = EC[ok], rs[ok], era[ok], v[ok]
med = np.median(vv)


def rmse(x, y):
    return float(np.sqrt(np.mean((x - y) ** 2)))


print(f"median within-segment sigma(T_s) = {med:.1f} K")
for nm, m in [("above median", vv > med), ("below median", vv <= med)]:
    print(f"  {nm} (n={m.sum():2d}): "
          f"slope {np.polyfit(e[m], a_[m], 1)[0]:+.1f} -> {np.polyfit(e[m], r_[m], 1)[0]:+.1f}   "
          f"r {np.corrcoef(e[m], a_[m])[0, 1]:.1f} -> {np.corrcoef(e[m], r_[m])[0, 1]:.1f}   "
          f"RMSE {rmse(a_[m], e[m]):.1f} -> {rmse(r_[m], e[m]):.1f}")

rng = np.random.default_rng(42)
hi = vv > med
gain = rmse(a_[hi], e[hi]) - rmse(r_[hi], e[hi])
bs = [rmse(a_[hi][i], e[hi][i]) - rmse(r_[hi][i], e[hi][i])
      for i in (rng.integers(0, hi.sum(), hi.sum()) for _ in range(5000))]
obs = gain - (rmse(a_[~hi], e[~hi]) - rmse(r_[~hi], e[~hi]))
perm = []
for _ in range(5000):
    p = rng.permutation(len(vv)) < hi.sum()
    perm.append((rmse(a_[p], e[p]) - rmse(r_[p], e[p]))
                - (rmse(a_[~p], e[~p]) - rmse(r_[~p], e[~p])))
print(f"  RMSE gain above median: {gain:.1f} W/m2, bootstrap 95 % CI "
      f"[{np.percentile(bs, 2.5):.1f}, {np.percentile(bs, 97.5):.1f}], "
      f"permutation p = {np.mean(np.abs(perm) >= abs(obs)):.2f}")

print("\n" + "=" * 118)
print("ALPHA_T IS A WEAK, LOGARITHMIC KNOB")
print("=" * 118)
s01 = amplitude_metrics(EC[np.isfinite(EC)],
                        HO_ALL[RUNS_O.index("GLGS20_ztr1e-01")][np.isfinite(EC)])["slope"]
s04 = amplitude_metrics(EC[np.isfinite(EC)],
                        HO_ALL[RUNS_O.index("GLGS20_ztr1e-04")][np.isfinite(EC)])["slope"]
print(f"3 decades of alpha_T (1e-1 -> 1e-4) change the flux amplitude by a factor "
      f"{s04 / s01:.1f} (slope {s01:.1f} -> {s04:.1f}); the correlation does not change at all.")
print()

print("=" * 118)
print("EC WINDOW-LENGTH EXPERIMENT (Fig. 2)  -- scripts/ec_window_length.py")
print("=" * 118)
WL = pd.read_csv(REPO / "data/observations/ar/ec_leg_length_sensitivity/ec_sensitivity_summary.csv")
print(f"{int(WL.loc[0, 'n_legs'])} low-level legs, median length {WL.loc[0, 'length_km']:.1f} km")
for L in (1.0, 5.0, 10.0):
    w = WL[WL["length_km"] == L].iloc[0]
    print(f"  {L:4.0f} km windows: H_sub/H_full = {w['ratio_mean']:.1f} +- {w['ratio_std']:.1f}   "
          f"random error {w['random_error_median']:.1f} W m-2   "
          f"fluxes above 1 / 2 / 3 sigma_rand: {w['fraction_above_1sigma'] * 100:.0f} / "
          f"{w['fraction_above_2sigma'] * 100:.0f} / {w['fraction_above_3sigma'] * 100:.0f} %")

print("\n" + "=" * 118)
print("UNCERTAINTY BUDGET (Table tab:bulk_unc): median [p25-p75] in W m-2")
print("=" * 118)
from fluxpy.uncertainty import uncertainty_budget  # noqa: E402

for key, sig in uncertainty_budget(FIG).items():
    s = sig[np.isfinite(sig)]
    print(f"  {key:<14} {np.median(s):4.1f} [{np.percentile(s, 25):.1f}-{np.percentile(s, 75):.1f}]")
print()
print("=" * 118)
print("FURTHER NUMBERS QUOTED IN THE TEXT")
print("=" * 118)
from fluxpy.uncertainty import _central  # noqa: E402

keep = np.isfinite(EC)


def _q(x):
    return f"{np.nanmedian(x):+.1f} [{np.nanpercentile(x, 25):+.1f}, {np.nanpercentile(x, 75):+.1f}]"


print("input sensitivities, median [IQR] over segments (central difference about the unperturbed state):")
for k, nm in (("obs", "T-Bird bulk"), ("rs", "ERA5-MODIS bulk")):
    print(f"  {nm:<16} dH/d(dtheta) = {_q(_central(FIG, f'H_{k}_dTair', 'dT')[keep])} W m-2 K-1    "
          f"dH/dU = {_q(_central(FIG, f'H_{k}_dU', 'dU')[keep])} W m-2 (m/s)-1")

car, erad = np.asarray(FIG["H_car_diag"], float), np.asarray(FIG["H_era_diag"], float)
m = keep & np.isfinite(car)
print(f"model-diagnosed fluxes on the common CARRA subset (n = {m.sum()}):")
for nm, h in (("H_S,ERA,diag", erad), ("H_S,CAR,diag", car)):
    print(f"  {nm}: slope {np.polyfit(EC[m], h[m], 1)[0]:.1f}   r {np.corrcoef(EC[m], h[m])[0, 1]:.1f}   "
          f"bias {(h[m] - EC[m]).mean():+.1f}")

# gain in correlation from the satellite T_s in the heterogeneous half: paired bootstrap
rng2 = np.random.default_rng(42)
eh, rh, ah = e[hi], r_[hi], a_[hi]
dr = np.corrcoef(eh, rh)[0, 1] - np.corrcoef(eh, ah)[0, 1]
boot = []
for _ in range(5000):
    i = rng2.integers(0, eh.size, eh.size)
    boot.append(np.corrcoef(eh[i], rh[i])[0, 1] - np.corrcoef(eh[i], ah[i])[0, 1])
boot = np.asarray(boot)
print(f"heterogeneous half: r(ERA5-MODIS) - r(ERA5-only) = {dr:+.1f}, paired-bootstrap 95 % CI "
      f"[{np.percentile(boot, 2.5):+.1f}, {np.percentile(boot, 97.5):+.1f}], "
      f"fraction of resamples <= 0: {np.mean(boot <= 0):.4f}")
print()

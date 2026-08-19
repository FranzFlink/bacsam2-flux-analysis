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
    out = (f"{name:<26} {m['slope']:5.2f} {m['slope_deming']:7.2f} {m['r']:6.2f} "
           f"{within_leg_r(h):8.2f} {m['bias']:+7.2f} {m['rmse']:6.2f} "
           f"{m['crmse']:6.2f} {m['n']:4d}")
    if ci:
        lo, hi = slope_bootstrap_ci(EC, h)
        dlo, dhi = slope_bootstrap_ci(EC, h, key="slope_deming")
        out += f"   OLS 95% CI [{lo:.2f}, {hi:.2f}]  Deming [{dlo:.2f}, {dhi:.2f}]"
    return out


print("\n" + "=" * 118)
print("EC REFERENCE (81 QC-retained 5 km segments)")
print("=" * 118)
ecv = EC[np.isfinite(EC)]
print(f"median {np.median(ecv):.2f}   mean {ecv.mean():.2f}   max {ecv.max():.1f}   "
      f"min {ecv.min():.1f}   |H|<2 W/m2: {np.mean(np.abs(ecv) < 2) * 100:.0f} % of segments")

print("\n" + "=" * 118)
print("REFERENCE CONFIGURATION  alpha_T = 1e-1 (conventional)")
print("=" * 118)
print(f"{'estimate':<26} {'slope':>5} {'Deming':>7} {'r':>6} {'r_within':>8} "
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
    print(f"{a:>8} | {o['slope']:12.2f} {o['r']:5.2f} {o['bias']:+6.2f} {o['rmse']:5.2f} |"
          f" {r_['slope']:16.2f} {r_['r']:5.2f} | {e_['slope']:15.2f} {e_['r']:5.2f}")

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


print(f"median within-segment sigma(T_s) = {med:.2f} K")
for nm, m in [("above median", vv > med), ("below median", vv <= med)]:
    print(f"  {nm} (n={m.sum():2d}): "
          f"slope {np.polyfit(e[m], a_[m], 1)[0]:+.2f} -> {np.polyfit(e[m], r_[m], 1)[0]:+.2f}   "
          f"r {np.corrcoef(e[m], a_[m])[0, 1]:.2f} -> {np.corrcoef(e[m], r_[m])[0, 1]:.2f}   "
          f"RMSE {rmse(a_[m], e[m]):.2f} -> {rmse(r_[m], e[m]):.2f}")

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
print(f"  RMSE gain above median: {gain:.2f} W/m2, bootstrap 95 % CI "
      f"[{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}], "
      f"permutation p = {np.mean(np.abs(perm) >= abs(obs)):.4f}")

print("\n" + "=" * 118)
print("ALPHA_T IS A WEAK, LOGARITHMIC KNOB")
print("=" * 118)
s01 = amplitude_metrics(EC[np.isfinite(EC)],
                        HO_ALL[RUNS_O.index("GLGS20_ztr1e-01")][np.isfinite(EC)])["slope"]
s04 = amplitude_metrics(EC[np.isfinite(EC)],
                        HO_ALL[RUNS_O.index("GLGS20_ztr1e-04")][np.isfinite(EC)])["slope"]
print(f"3 decades of alpha_T (1e-1 -> 1e-4) change the flux amplitude by a factor "
      f"{s04 / s01:.2f} (slope {s01:.2f} -> {s04:.2f}); the correlation does not change at all.")
print()

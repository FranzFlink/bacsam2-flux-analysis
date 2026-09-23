"""Every number quoted in the manuscript text and in Table 4, printed from the pipeline output.

Single source of truth for the printed statistics. Run after any pipeline change
and copy the numbers from here, never from an older PDF, a notebook or git
history. ``pixi run check`` compares this output with ``expected_numbers.txt``.

Blocks (in the order printed) and where they appear in the manuscript
    EC reference ........................ Sect. 3.1, Appendix A2 (n, integrity check, random error, 1/2/3 sigma)
    reference configuration ............. Table 4, Sect. 4.1 (slope, r, r2, within-leg r, bias, RMSE, CI)
    amplitude-matched configuration ..... Table 4, Sect. 4.2 (alpha_T = 1e-4)
    alpha_T sweep ....................... Sect. 4.2, Fig. 6, Table C1
    satellite T_s benefit ............... Sect. 4.1, Discussion (heterogeneous vs homogeneous half)
    alpha_T is a weak knob .............. Discussion
    window-length experiment ............ Sect. 3.1, Fig. 2
    uncertainty budget .................. Table 2
    further numbers ..................... input sensitivities (Sect. 4.2), CARRA subset (Sect. 4.1), paired bootstrap

Framework (Sect. 4.1): amplitude skill is a regression slope and variability skill
a correlation, because RMSE in this low-flux regime rewards estimates that shrink
towards zero. The reference configuration uses the conventional alpha_T = 1e-1;
the amplitude-matched alpha_T = 1e-4 is reported alongside.

Run:  pixi run numbers
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from fluxpy.sensitivity import amplitude_metrics, skill_metrics, slope_bootstrap_ci  # noqa: E402
from fluxpy.uncertainty import central_difference, uncertainty_budget  # noqa: E402

RULE = "=" * 118

# ── pipeline output ──────────────────────────────────────────────────────────
FIG = xr.open_dataset(REPO / "data/model_output/sensitivity_figure_data.nc")      # 91 segments, all estimates
SEG = xr.open_dataset(REPO / "data/observations/ar/analysis_ready_polar6_turbulent_fluxes_latest_5km.nc")
SEG = SEG.where(SEG.altitude <= 50, drop=True)                                   # low-level segments (all days)

EC = np.asarray(FIG["H_obs_ec"])
T = pd.to_datetime(FIG["start_time"].values)
LEG = np.cumsum(np.r_[0, np.diff(T.values) / np.timedelta64(1, "m")] > 10)      # a new leg after a gap > 10 min
RUNS_OBS = [str(r) for r in FIG["run_obs"].values]
RUNS_RS = [str(r) for r in FIG["run"].values]
H_OBS_ALL = np.asarray(FIG["H_obs_bulk_all"])                                    # (run_obs, leg)
H_RS_ALL = np.asarray(FIG["H_rs_bulk_all"])                                      # (run, leg)


# ── helpers ──────────────────────────────────────────────────────────────────

def within_leg_r(h):
    """Correlation with the EC flux after removing each leg's mean (variability along the track only)."""
    m = np.isfinite(h) & np.isfinite(EC)
    d = pd.DataFrame({"ec": EC[m], "h": h[m], "leg": LEG[m]})
    a = d.groupby("leg").transform(lambda s: s - s.mean())
    return float(np.corrcoef(a["ec"], a["h"])[0, 1])


def line(name, h, ci=False):
    """One row of Table 4: slope, r, r2, within-leg r, bias, RMSE, centred RMSE, n (+ bootstrap CI of the slope)."""
    m = skill_metrics(EC, h, name)
    out = (f"{name:<26} {m['slope']:5.1f} {m['r']:6.1f} {m['r'] ** 2:5.1f} "
           f"{within_leg_r(h):8.1f} {m['bias']:+7.1f} {m['rmse']:6.1f} "
           f"{m['crmse']:6.1f} {m['n']:4d}")
    if ci:
        lo, hi = slope_bootstrap_ci(EC, h)
        out += f"   OLS 95% CI [{lo:.1f}, {hi:.1f}]"
    return out


def rmse(x, y):
    return float(np.sqrt(np.mean((x - y) ** 2)))


def quartiles(x):
    return f"{np.nanmedian(x):+.1f} [{np.nanpercentile(x, 25):+.1f}, {np.nanpercentile(x, 75):+.1f}]"


# ── blocks ───────────────────────────────────────────────────────────────────

def ec_reference():
    """Sect. 3.1 and Appendix A2: sample size, integrity check, random error, fluxes above 1/2/3 sigma."""
    ecv = EC[np.isfinite(EC)]
    print("\n" + RULE)
    print(f"EC REFERENCE ({ecv.size} of {EC.size} low-level 5 km segments pass the integrity check)")
    print(RULE)
    seg = SEG.sel(start_time=FIG["start_time"].values)
    print(f"integrity check (inertial-subrange slope of w): {int(seg['turbulence_resolved'].sum())} of {EC.size} pass; "
          f"slope median {float(seg['qc_slope_w'].median()):.1f}, range {float(seg['qc_slope_w'].min()):.1f} to {float(seg['qc_slope_w'].max()):.1f}")
    sigma, ratio = np.asarray(FIG["sigma_random_ec"]), np.asarray(FIG["ec_flux_over_sigma"])
    print(f"random error: median {np.median(sigma):.1f} W m-2 [{np.percentile(sigma, 25):.1f}-{np.percentile(sigma, 75):.1f}]")
    days = T.strftime("%d %b")
    for k in (1, 2, 3):
        above = ratio > k
        print(f"   |H| > {k} sigma_rand: {above.sum():2d} of {EC.size}  (expected by chance if every true flux were zero: "
              f"{2 * (1 - norm.cdf(k)) * EC.size:.1f});  per day: "
              + ", ".join(f"{d}: {above[days == d].sum()}/{(days == d).sum()}" for d in pd.unique(days)))
    print(f"median {np.median(ecv):.1f}   mean {ecv.mean():.1f}   max {ecv.max():.1f}   "
          f"min {ecv.min():.1f}   |H|<2 W/m2: {np.mean(np.abs(ecv) < 2) * 100:.0f} % of segments")


def reference_configuration():
    """Table 4 (Sect. 4.1): the five estimates against the EC reference, alpha_T = 1e-1."""
    print("\n" + RULE)
    print("REFERENCE CONFIGURATION  alpha_T = 1e-1 (conventional)")
    print(RULE)
    print(f"{'estimate':<26} {'slope':>5} {'r':>6} {'r2':>5} {'r_within':>8} "
          f"{'bias':>7} {'RMSE':>6} {'cRMSE':>6} {'n':>4}")
    for key, name in [("H_obs_bulk", "H_S,obs,bulk"), ("H_rs_bulk", "H_S,RS,bulk"),
                      ("H_era_bulk", "H_S,ERA,bulk"), ("H_era_diag", "H_S,ERA,diag"),
                      ("H_car_diag", "H_S,CAR,diag")]:
        print(line(name, np.asarray(FIG[key]), ci=(key in ("H_obs_bulk", "H_rs_bulk"))))


def amplitude_matched_configuration():
    """Table 4, lower part (Sect. 4.2): alpha_T = 1e-4."""
    print("\n" + RULE)
    print("AMPLITUDE-MATCHED CONFIGURATION  alpha_T = 1e-4  (Schroeder et al., 2003)")
    print(RULE)
    print(line("H_S,obs,bulk", H_OBS_ALL[RUNS_OBS.index("GLGS20_ztr1e-04")], ci=True))
    print(line("H_S,RS,bulk", H_RS_ALL[RUNS_RS.index("GLGS20_ztr1e-04")]))
    print(line("H_S,ERA,bulk", H_RS_ALL[RUNS_RS.index("GLGS20_ztr1e-04_era5only")]))


def alpha_sweep():
    """Sect. 4.2, Fig. 6 and Table C1: slope and r over alpha_T; r does not depend on alpha_T."""
    print("\n" + RULE)
    print("alpha_T SWEEP  (slope / r / bias / RMSE)   -- note r is INVARIANT to alpha_T")
    print(RULE)
    print(f"{'alpha_T':>8} | {'T-Bird slope':>12} {'r':>5} {'bias':>6} {'RMSE':>5} |"
          f" {'ERA5-MODIS slope':>16} {'r':>5} | {'ERA5-only slope':>15} {'r':>5}")
    for a in ["1e-01", "1e-02", "1e-03", "1e-04", "1e-05", "1e-06", "1e-07"]:
        try:
            o = skill_metrics(EC, H_OBS_ALL[RUNS_OBS.index(f"GLGS20_ztr{a}")], "")
            r_ = skill_metrics(EC, H_RS_ALL[RUNS_RS.index(f"GLGS20_ztr{a}")], "")
            e_ = skill_metrics(EC, H_RS_ALL[RUNS_RS.index(f"GLGS20_ztr{a}_era5only")], "")
        except ValueError:
            continue
        print(f"{a:>8} | {o['slope']:12.1f} {o['r']:5.1f} {o['bias']:+6.1f} {o['rmse']:5.1f} |"
              f" {r_['slope']:16.1f} {r_['r']:5.1f} | {e_['slope']:15.1f} {e_['r']:5.1f}")


def satellite_benefit():
    """Sect. 4.1 and Discussion: where the MODIS surface temperature helps. The segments are
    split at the median within-segment standard deviation of the KT-19 surface temperature.
    Returns the split (EC, ERA5-MODIS, ERA5-only fluxes and the heterogeneous mask) for the
    paired bootstrap in further_numbers()."""
    print("\n" + RULE)
    print("SATELLITE T_s BENEFIT vs WITHIN-SEGMENT T_s VARIABILITY  (reference configuration)")
    print(RULE)
    T_sfc_std = pd.Series(np.asarray(SEG["T_sfc_std"]), index=pd.to_datetime(SEG["start_time"].values))
    v = T_sfc_std.reindex(T, method="nearest", tolerance=pd.Timedelta("60s")).values
    rs, era = np.asarray(FIG["H_rs_bulk"]), np.asarray(FIG["H_era_bulk"])
    ok = np.isfinite(EC) & np.isfinite(rs) & np.isfinite(era) & np.isfinite(v)
    e, r_, a_, vv = EC[ok], rs[ok], era[ok], v[ok]
    med = np.median(vv)
    print(f"median within-segment sigma(T_s) = {med:.1f} K")
    for name, m in [("above median", vv > med), ("below median", vv <= med)]:
        print(f"  {name} (n={m.sum():2d}): "
              f"slope {np.polyfit(e[m], a_[m], 1)[0]:+.1f} -> {np.polyfit(e[m], r_[m], 1)[0]:+.1f}   "
              f"r {np.corrcoef(e[m], a_[m])[0, 1]:.1f} -> {np.corrcoef(e[m], r_[m])[0, 1]:.1f}   "
              f"RMSE {rmse(a_[m], e[m]):.1f} -> {rmse(r_[m], e[m]):.1f}")

    rng = np.random.default_rng(42)
    hi = vv > med
    gain = rmse(a_[hi], e[hi]) - rmse(r_[hi], e[hi])
    boot = [rmse(a_[hi][i], e[hi][i]) - rmse(r_[hi][i], e[hi][i])
            for i in (rng.integers(0, hi.sum(), hi.sum()) for _ in range(5000))]
    observed = gain - (rmse(a_[~hi], e[~hi]) - rmse(r_[~hi], e[~hi]))
    permuted = []
    for _ in range(5000):
        p = rng.permutation(len(vv)) < hi.sum()
        permuted.append((rmse(a_[p], e[p]) - rmse(r_[p], e[p]))
                        - (rmse(a_[~p], e[~p]) - rmse(r_[~p], e[~p])))
    print(f"  RMSE gain above median: {gain:.1f} W/m2, bootstrap 95 % CI "
          f"[{np.percentile(boot, 2.5):.1f}, {np.percentile(boot, 97.5):.1f}], "
          f"permutation p = {np.mean(np.abs(permuted) >= abs(observed)):.2f}")
    return e, r_, a_, hi


def alpha_is_a_weak_knob():
    """Discussion: three decades of alpha_T change the amplitude by a factor of about 1.6 only."""
    print("\n" + RULE)
    print("ALPHA_T IS A WEAK, LOGARITHMIC KNOB")
    print(RULE)
    ok = np.isfinite(EC)
    s01 = amplitude_metrics(EC[ok], H_OBS_ALL[RUNS_OBS.index("GLGS20_ztr1e-01")][ok])["slope"]
    s04 = amplitude_metrics(EC[ok], H_OBS_ALL[RUNS_OBS.index("GLGS20_ztr1e-04")][ok])["slope"]
    print(f"3 decades of alpha_T (1e-1 -> 1e-4) change the flux amplitude by a factor "
          f"{s04 / s01:.1f} (slope {s01:.1f} -> {s04:.1f}); the correlation does not change at all.")
    print()


def window_length_experiment():
    """Sect. 3.1 and Fig. 2: sub-window flux, random error and fluxes above 1/2/3 sigma vs window length."""
    print(RULE)
    print("EC WINDOW-LENGTH EXPERIMENT (Fig. 2)  -- scripts/ec_window_length.py")
    print(RULE)
    wl = pd.read_csv(REPO / "data/observations/ar/ec_leg_length_sensitivity/ec_sensitivity_summary.csv")
    print(f"{int(wl.loc[0, 'n_legs'])} low-level legs, median length {wl.loc[0, 'length_km']:.1f} km")
    for L in (1.0, 5.0, 10.0):
        w = wl[wl["length_km"] == L].iloc[0]
        print(f"  {L:4.0f} km windows: H_sub/H_full = {w['ratio_mean']:.1f} +- {w['ratio_std']:.1f}   "
              f"random error {w['random_error_median']:.1f} W m-2   "
              f"fluxes above 1 / 2 / 3 sigma_rand: {w['fraction_above_1sigma'] * 100:.0f} / "
              f"{w['fraction_above_2sigma'] * 100:.0f} / {w['fraction_above_3sigma'] * 100:.0f} %")


def uncertainty_budget_table():
    """Table 2: median and quartiles of the 1-sigma uncertainties."""
    print("\n" + RULE)
    print("UNCERTAINTY BUDGET (Table tab:bulk_unc): median [p25-p75] in W m-2")
    print(RULE)
    for key, sigma in uncertainty_budget(FIG).items():
        s = sigma[np.isfinite(sigma)]
        print(f"  {key:<14} {np.median(s):4.1f} [{np.percentile(s, 25):.1f}-{np.percentile(s, 75):.1f}]")
    print()


def further_numbers(split):
    """Input sensitivities (Sect. 4.2), the CARRA subset (Sect. 4.1) and the paired bootstrap of the
    correlation gain in the heterogeneous half (Discussion)."""
    print(RULE)
    print("FURTHER NUMBERS QUOTED IN THE TEXT")
    print(RULE)
    keep = np.isfinite(EC)
    print("input sensitivities, median [IQR] over segments (central difference about the unperturbed state):")
    for k, name in (("obs", "T-Bird bulk"), ("rs", "ERA5-MODIS bulk")):
        print(f"  {name:<16} dH/d(dtheta) = {quartiles(central_difference(FIG, f'H_{k}_dTair', 'dT')[keep])} W m-2 K-1    "
              f"dH/dU = {quartiles(central_difference(FIG, f'H_{k}_dU', 'dU')[keep])} W m-2 (m/s)-1")

    car, erad = np.asarray(FIG["H_car_diag"], float), np.asarray(FIG["H_era_diag"], float)
    m = keep & np.isfinite(car)
    print(f"model-diagnosed fluxes on the common CARRA subset (n = {m.sum()}):")
    for name, h in (("H_S,ERA,diag", erad), ("H_S,CAR,diag", car)):
        print(f"  {name}: slope {np.polyfit(EC[m], h[m], 1)[0]:.1f}   r {np.corrcoef(EC[m], h[m])[0, 1]:.1f}   "
              f"bias {(h[m] - EC[m]).mean():+.1f}")

    e, r_, a_, hi = split
    rng = np.random.default_rng(42)
    eh, rh, ah = e[hi], r_[hi], a_[hi]
    dr = np.corrcoef(eh, rh)[0, 1] - np.corrcoef(eh, ah)[0, 1]
    boot = []
    for _ in range(5000):
        i = rng.integers(0, eh.size, eh.size)
        boot.append(np.corrcoef(eh[i], rh[i])[0, 1] - np.corrcoef(eh[i], ah[i])[0, 1])
    boot = np.asarray(boot)
    print(f"heterogeneous half: r(ERA5-MODIS) - r(ERA5-only) = {dr:+.1f}, paired-bootstrap 95 % CI "
          f"[{np.percentile(boot, 2.5):+.1f}, {np.percentile(boot, 97.5):+.1f}], "
          f"fraction of resamples <= 0: {np.mean(boot <= 0):.4f}")
    print()


if __name__ == "__main__":
    ec_reference()
    reference_configuration()
    amplitude_matched_configuration()
    alpha_sweep()
    split = satellite_benefit()
    alpha_is_a_weak_knob()
    window_length_experiment()
    uncertainty_budget_table()
    further_numbers(split)

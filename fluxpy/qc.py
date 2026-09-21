r"""
fluxpy.qc
=========
Spectral quality control of the eddy-covariance segments.

Each 5 km segment is screened with four complementary diagnostics
(manuscript Appendix A) that test whether its cospectrum is well resolved,
converged, stationary, and clearly above the noise:

1. **Inertial-subrange slope.** The vertical-wind power spectrum is fitted
   in double-logarithmic space over the flux band; a resolved turbulent
   cascade requires

   $$ \left| s_w + \tfrac{5}{3} \right| \le 0.4, $$

   following the $f^{-5/3}$ prediction for locally isotropic turbulence
   (Kolmogorov, 1941; Obukhov, 1941).

2. **Ogive convergence.** The normalised cumulative cospectrum (ogive),
   integrated from high towards low frequencies, must approach unity below
   the 0.1 Hz cut-off,

   $$ \varepsilon_\mathrm{plateau} =
      \left| 1 - \left\langle \tilde O_{w\theta}(f)
      \right\rangle_{f < 0.1\,\mathrm{Hz}} \right| < 10^{-3}, $$

   confirming that the flux-carrying scales are contained in the segment.

3. **Along-track stationarity.** The normalised cumulative covariance
   $S(j)$ must grow approximately linearly with fractional segment length;
   a linear fit requires $R^2 \ge 0.4$.

4. **Cospectral signal-to-noise ratio.** The peak of the
   frequency-weighted cospectrum $|f\,C_{w\theta}(f)|$ within the
   turbulent band must exceed twice the background level estimated
   outside it ($\mathrm{SNR} > 2$).

Segments passing all four criteria are *accepted*, failing exactly one
*tolerable*, and failing two or more *discarded*; accepted and tolerable
segments enter the flux comparison.

Public API
----------
highpass(x, fs, fc, order)             – 4th-order Butterworth high-pass filter
cospectrum(w, x, fs, ...)              – cross-power spectral density (real part)
powerspec(x, fs, ...)                  – one-sided power spectral density (Welch)
ogive_from_cospec(f, Co)               – cumulative ogive (high → low frequency)
qc_leg(w, theta, u, v, fs, x_dist)     – per-segment spectral metrics
classify_leg_soft(qc, leg_length_km)   – graded verdict
classify_leg_simple(qc, leg_length_km) – manuscript four-criterion verdict
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, detrend, csd, welch, get_window

__all__ = [
    "highpass",
    "cospectrum",
    "powerspec",
    "ogive_from_cospec",
    "qc_leg",
    "classify_leg_soft",
    "classify_leg_simple",
]


# ---------------------------------------------------------------------------
# Signal processing primitives
# ---------------------------------------------------------------------------

def highpass(x: np.ndarray, fs: float, fc: float = 0.1, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth high-pass filter.

    Parameters
    ----------
    x     : 1-D array of signal values.
    fs    : Sampling frequency [Hz].
    fc    : Cut-off frequency [Hz].  Default 0.1 Hz (Tetzlaff 2015).
    order : Filter order.  Default 4.
    """
    b, a = butter(order, fc / (fs / 2.0), btype="highpass")
    return filtfilt(b, a, x)


def cospectrum(
    w: np.ndarray,
    x: np.ndarray,
    fs: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Co-spectral density of (w, x) using Welch / CSD.

    Returns
    -------
    f   : Frequency array [Hz].
    Co  : Real part of cross-PSD.
    """
    if nperseg is None:
        nperseg = int(8 * fs)          # ~8-s segments
    if noverlap is None:
        noverlap = nperseg // 2
    f, Pwx = csd(
        w, x,
        fs=fs,
        window=get_window("hann", nperseg),
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="linear",
    )
    return f, np.real(Pwx)


def powerspec(
    x: np.ndarray,
    fs: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided power spectral density via Welch method.

    Returns
    -------
    f   : Frequency array [Hz].
    Pxx : Power spectral density.
    """
    if nperseg is None:
        nperseg = int(8 * fs)
    if noverlap is None:
        noverlap = nperseg // 2
    f, Pxx = welch(
        x,
        fs=fs,
        window=get_window("hann", nperseg),
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="linear",
    )
    return f, Pxx


def ogive_from_cospec(f: np.ndarray, Co: np.ndarray) -> np.ndarray:
    """Cumulative ogive integral from high to low frequency.

    Og(f₀) = ∫_{f₀}^{∞} Co(f) df   (approximated by cumulative sum)
    """
    df = np.diff(f, prepend=f[0])
    og = np.flip(np.cumsum(np.flip(Co * df)))
    return og


# ---------------------------------------------------------------------------
# Private inertial-subrange helpers
# ---------------------------------------------------------------------------

def _band_mask(f: np.ndarray, a: float, b: float) -> np.ndarray:
    return (f >= a) & (f <= b)


def _find_inertial_band(f: np.ndarray, P: np.ndarray, fs: float) -> np.ndarray:
    """Adaptive inertial-subrange mask.

    Uses the peak of f·P(f) as lower edge, caps upper edge
    conservatively at min(20 Hz, Nyquist/2). Falls back to
    generic 0.2–10 Hz band if fewer than 8 points are found.
    """
    fp = f[np.argmax(f * P)]
    f1 = max(0.12, fp)
    f2 = min(20.0, 0.25 * fs)
    m = _band_mask(f, f1, f2) & np.isfinite(P) & (P > 0)
    if m.sum() < 8:
        m = _band_mask(f, 0.2, min(10.0, 0.25 * fs)) & (P > 0)
    return m


def _log_slope(f: np.ndarray, P: np.ndarray, mask: np.ndarray) -> float:
    """Log-log slope of P(f) in the masked frequency range."""
    if mask.sum() < 6:
        return np.nan
    return float(np.polyfit(np.log10(f[mask]), np.log10(P[mask]), 1)[0])


def _integrated_flux(
    f: np.ndarray,
    Co: np.ndarray,
    m: np.ndarray | None = None,
) -> float:
    """Trapezoidal integral of Co(f) over masked region."""
    if m is None:
        m = np.isfinite(Co)
    if not np.any(m):
        return np.nan
    return float(np.trapezoid(Co[m], f[m]))


def _f_percent_from_high(
    f: np.ndarray,
    Co: np.ndarray,
    frac: float = 0.95,
) -> float:
    """Frequency above which *frac* of total flux is contained (from high→low).

    Returns the frequency threshold [Hz] or ``nan`` if the ogive is degenerate.
    """
    m = np.isfinite(Co)
    if not np.any(m):
        return np.nan
    total = _integrated_flux(f, Co, m)
    if np.isclose(total, 0.0):
        return np.nan
    order = np.argsort(f[m])[::-1]
    ff = f[m][order]
    cc = Co[m][order]
    df = np.abs(np.diff(ff, append=ff[-1]))
    cum = np.cumsum(cc * df)
    target = frac * np.abs(total)
    idx = np.argmax(np.abs(cum) >= target)
    return float(ff[idx])


def _cospec_peak_snr(
    f: np.ndarray,
    fwC: np.ndarray,
    peak_band: tuple | None = None,
    noise_bands: tuple | None = None,
    peak: tuple | None = None,
    noise: tuple | None = None,
) -> float:
    """Cospectral SNR: max |f·Co| in peak band / median |f·Co| in noise bands.

    Accepts both ``(peak_band, noise_bands)`` and ``(peak, noise)`` naming
    for backward compatibility.
    """
    if peak is None:
        peak = peak_band if peak_band is not None else (0.1, 2.0)
    if noise is None:
        noise = noise_bands if noise_bands is not None else ((0.01, 0.05), (5.0, 20.0))

    m_peak = _band_mask(f, *peak)
    if not np.any(m_peak):
        return np.nan
    peak_val = float(np.nanmax(np.abs(fwC[m_peak])))

    noise_vals = []
    for a, b in noise:
        m = _band_mask(f, a, b)
        if np.any(m):
            noise_vals.append(float(np.median(np.abs(fwC[m]))))
    if not noise_vals:
        return np.nan
    noise_med = float(np.median(noise_vals))
    if not np.isfinite(noise_med) or noise_med == 0.0:
        return np.nan
    return peak_val / noise_med


# ---------------------------------------------------------------------------
# Per-leg spectral QC
# ---------------------------------------------------------------------------

def qc_leg(
    w: np.ndarray,
    theta: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    fs: float,
    x_dist: np.ndarray | None = None,
) -> dict:
    """Compute spectral QC metrics for a single flight leg.

    Parameters
    ----------
    w, theta, u, v : 1-D arrays of detrended time series (raw – detrending
                     is applied internally).
    fs             : Sampling frequency [Hz].
    x_dist         : Along-track distance array [km].  If None,
                     fractional distance [0, 1] is used.

    Returns
    -------
    dict with keys:
    ``f``, ``Pth``, ``Pw``, ``Pu``, ``Pv``,
    ``Cwth``, ``fwCwth``,
    ``og_th_norm``, ``cum_frac``, ``cum_cov_norm``,
    ``slope_th``, ``slope_w``, ``tv_og``, ``R2``, ``max_dev``
    """
    # 1. Detrend + high-pass (Tetzlaff 2015)
    w  = highpass(detrend(w,     type="linear"), fs)
    th = highpass(detrend(theta, type="linear"), fs)
    uu = highpass(detrend(u,     type="linear"), fs)
    vv = highpass(detrend(v,     type="linear"), fs)

    # 2. Power spectra
    f, Pth = powerspec(th, fs)
    f, Pw  = powerspec(w,  fs)
    f, Pu  = powerspec(uu, fs)
    f, Pv  = powerspec(vv, fs)

    # 3. Welch co-spectrum and normalised ogive (used for SNR / fwC diagnostics)
    f, Cwth    = cospectrum(w, th, fs)
    og_th      = ogive_from_cospec(f, Cwth)
    denom      = np.mean((w - w.mean()) * (th - th.mean()))
    og_th_norm = og_th / denom if denom != 0 else og_th * np.nan
    # Self-normalised ogive: og_th / og_th[0]  (Welch integral as denominator).
    # og_self[0] = 1 by construction, shape-only, unaffected by flux magnitude.
    og_th_self = og_th / og_th[0] if abs(og_th[0]) > 1e-12 else og_th * np.nan

    # 3b. Single-realisation FFT ogive (resolves down to 1/T_leg ≈ 0.02 Hz).
    # Zero-pad to next power-of-2 for speed; use a Hann window to suppress
    # spectral leakage.  The plateau check uses only f_fft < fc (= 0.1 Hz,
    # the pre-filter cutoff) where the ogive should be flat.
    N_fft   = len(w)
    N_pad   = int(2 ** np.ceil(np.log2(N_fft)))   # next power-of-2
    hann    = np.hanning(N_fft)
    # Factor to normalise Hann window (preserves RMS amplitude)
    hann_nf = np.sqrt(np.mean(hann**2))
    Ww  = np.fft.rfft(w  * hann / hann_nf, n=N_pad)
    Wth = np.fft.rfft(th * hann / hann_nf, n=N_pad)
    f_fft   = np.fft.rfftfreq(N_pad, d=1.0/fs)
    df_fft  = f_fft[1]                             # frequency resolution [Hz]
    Co_fft  = np.real(Ww * np.conj(Wth)) / (N_pad * fs)  # two-sided → one-sided below
    Co_fft[1:-1] *= 2.0                            # double non-DC/Nyquist bins
    og_fft  = ogive_from_cospec(f_fft, Co_fft)
    # Self-normalise by the FFT total flux estimate (og_fft[0])
    og_fft_norm = og_fft / og_fft[0] if abs(og_fft[0]) > 1e-12 else og_fft * np.nan
    # Plateau metrics: mean and spread in the sub-0.1 Hz band
    m_plat  = (f_fft > 0.0) & (f_fft < 0.1)       # skip DC bin
    if m_plat.sum() >= 2:
        og_fft_plateau     = float(np.nanmean(og_fft_norm[m_plat]))
        og_fft_plateau_err = float(abs(1.0 - og_fft_plateau))
        og_fft_spread      = float(np.nanmax(og_fft_norm[m_plat]) -
                                   np.nanmin(og_fft_norm[m_plat]))
    else:                                          # segment too short to resolve plateau
        og_fft_plateau_err = np.nan
        og_fft_spread      = np.nan

    # 4. Frequency-weighted co-spectrum
    fwCwth = f * Cwth

    # 5. Cumulative co-variance vs along-track distance (stationarity check)
    if x_dist is None:
        x_dist = np.linspace(0.0, 1.0, len(w))
    w_p  = w  - w.mean()
    th_p = th - th.mean()
    cum_cov      = np.cumsum(w_p * th_p)
    cum_cov_norm = cum_cov / cum_cov[-1] if cum_cov[-1] != 0 else cum_cov * np.nan
    frac = (x_dist - x_dist.min()) / (x_dist.max() - x_dist.min())

    # 6. Inertial-subrange spectral slopes (log-log, 0.1–20 Hz)
    def _slope_in_band(P, f1=0.1, f2=20.0):
        m = (f >= f1) & (f <= f2) & (P > 0)
        if m.sum() < 5:
            return np.nan
        return float(np.polyfit(np.log10(f[m]), np.log10(P[m]), 1)[0])

    slope_th = _slope_in_band(Pth)
    slope_w  = _slope_in_band(Pw)

    # 7. Ogive total variation in 0.1–5 Hz (smoothness).
    # Computed on the self-normalised ogive so it is flux-magnitude independent.
    m_og      = (f >= 0.1) & (f <= 5.0)
    tv_og     = float(np.abs(np.diff(og_th_norm[m_og])).sum())   # legacy (time-domain denom)
    tv_og_self = float(np.abs(np.diff(og_th_self[m_og])).sum())  # Welch-self-normalised

    # 8. Homogeneity: linear fit R² and max |residual|
    coef  = np.polyfit(frac, cum_cov_norm, 1)
    fit   = np.polyval(coef, frac)
    resid = cum_cov_norm - fit
    SS_res = float(np.sum(resid ** 2))
    SS_tot = float(np.sum((cum_cov_norm - cum_cov_norm.mean()) ** 2))
    R2      = (1.0 - SS_res / SS_tot) if SS_tot > 0 else np.nan
    max_dev = float(np.max(np.abs(resid)))

    return dict(
        f=f, Pth=Pth, Pw=Pw, Pu=Pu, Pv=Pv,
        Cwth=Cwth, fwCwth=fwCwth,
        og_th_norm=og_th_norm, og_th_self=og_th_self,
        f_fft=f_fft, og_fft_norm=og_fft_norm,
        og_fft_plateau_err=og_fft_plateau_err, og_fft_spread=og_fft_spread,
        cum_frac=frac, cum_cov_norm=cum_cov_norm,
        slope_th=slope_th, slope_w=slope_w,
        tv_og=tv_og, tv_og_self=tv_og_self, R2=R2, max_dev=max_dev,
    )


# ---------------------------------------------------------------------------
# Soft QC classification
# ---------------------------------------------------------------------------

def classify_leg_soft(
    qc: dict,
    leg_length_km: float,
    kind: str = "heat",
) -> dict:
    """Soft QC verdict for a flight leg.

    Aggregates multiple spectral and stationarity metrics into a single
    weighted score and assigns one of three verdicts:

    * ``'accepted'``   – score ≥ 10
    * ``'tolerable'``  – score ≥ 6
    * ``'discarded'``  – score < 6

    Parameters
    ----------
    qc            : Dict returned by :func:`qc_leg`.
    leg_length_km : Actual length of the bin [km].
    kind          : ``'heat'`` (default) or ``'momentum'`` – controls the
                    minimum-length thresholds.

    Returns
    -------
    dict with keys:
    ``verdict``      str
    ``score``        float
    ``flux_int_abs`` float   – |∫Co(f)df|, guard for near-zero-flux legs
    ``metrics``      dict    – named diagnostics (R2, slopes, f95, …)
    ``reasons``      list    – list of (name, status, value) tuples
    """
    f       = qc["f"]
    Pth     = qc["Pth"]
    Pw      = qc["Pw"]
    Cwth    = qc["Cwth"]
    fwCwth  = qc["fwCwth"]
    og_norm = qc["og_th_norm"]
    R2      = float(qc["R2"])
    max_dev = float(qc["max_dev"])
    tv_og   = float(qc["tv_og"])

    fs = float(np.nanmax(f) * 2.0)   # approximate Nyquist

    # Integrated flux magnitude (guard for tiny-flux legs)
    flux_int = abs(_integrated_flux(f, Cwth))

    # Adaptive inertial-subrange slopes
    mth = _find_inertial_band(f, Pth, fs)
    mw  = _find_inertial_band(f, Pw,  fs)
    slope_th = _log_slope(f, Pth, mth)
    slope_w  = _log_slope(f, Pw,  mw)

    # Flux-convergence frequency (95 %)
    f95 = _f_percent_from_high(f, Cwth, frac=0.95)

    # Ogive plateau (lowest 10 % in frequency = FIRST elements of og_norm,
    # since og_norm[0] is at the lowest frequency and should plateau at 1)
    n    = len(og_norm)
    n_tail = max(1, int(0.1 * n))
    tail = og_norm[:n_tail] if n > 10 else og_norm
    og_plateau     = float(np.nanmean(tail))
    og_plateau_err = abs(1.0 - og_plateau)
    og_spread      = float(np.nanmax(tail) - np.nanmin(tail))

    # Cospectral SNR
    snr = _cospec_peak_snr(f, fwCwth)

    # Minimum-length thresholds (km)
    min_len_acc = 2.5 if kind == "heat" else 4.0
    min_len_tol = 1.5 if kind == "heat" else 2.5

    # ── Thresholds ──────────────────────────────────────────────────────────
    TH = dict(
        R2_acc=0.90,      R2_tol=0.80,
        maxdev_acc=0.10,  maxdev_tol=0.20,
        tv_acc=0.50,      tv_tol=0.80,
        f95_acc=0.15,     f95_tol=0.25,       # Hz
        ogerr_acc=0.10,   ogerr_tol=0.15,
        ogspr_acc=0.10,   ogspr_tol=0.15,
        slope_mu=-1.6667, slope_acc=0.50, slope_tol=0.80,
        snr_acc=2.5,      snr_tol=1.5,
        len_acc=min_len_acc, len_tol=min_len_tol,
        small_flux=0.01,  # K m s⁻¹  (∫Cwθ df) guard
    )

    # ── Scoring ──────────────────────────────────────────────────────────────
    score   = 0.0
    reasons = []

    def add_score(val, acc, tol, greater=True, w=1.0, name=None):
        nonlocal score
        if not np.isfinite(val):
            reasons.append((name, "nan", 0))
            return
        ok    = (val >= acc) if greater else (val <= acc)
        tolok = (val >= tol) if greater else (val <= tol)
        if ok:
            score += 2.0 * w
            reasons.append((name, "acc", val))
        elif tolok:
            score += 1.0 * w
            reasons.append((name, "tol", val))
        else:
            reasons.append((name, "fail", val))

    # Stationarity / homogeneity
    add_score(R2,      TH["R2_acc"],      TH["R2_tol"],      greater=True,  w=2.0, name="R2")
    add_score(max_dev, TH["maxdev_acc"],  TH["maxdev_tol"],  greater=False, w=2.0, name="max_dev")
    # Ogive smoothness
    add_score(tv_og, TH["tv_acc"], TH["tv_tol"], greater=False, w=1.5, name="tv_og")
    # Flux convergence (95 %)
    add_score(f95, TH["f95_acc"], TH["f95_tol"], greater=False, w=1.5, name="f95_Hz")

    # Plateau metrics (skip if flux is near-zero)
    if flux_int >= TH["small_flux"]:
        add_score(og_plateau_err, TH["ogerr_acc"], TH["ogerr_tol"], greater=False, w=1.0, name="og_plateau_err")
        add_score(og_spread,      TH["ogspr_acc"], TH["ogspr_tol"], greater=False, w=1.0, name="og_spread")
    else:
        reasons.append(("og_plateau_err", "skipped_small_flux", og_plateau_err))
        reasons.append(("og_spread",      "skipped_small_flux", og_spread))

    # Inertial slopes (θ and w)
    for s, nm in [(slope_th, "slope_theta"), (slope_w, "slope_w")]:
        if np.isfinite(s):
            dev = abs(s - TH["slope_mu"])
            add_score(dev, TH["slope_acc"], TH["slope_tol"], greater=False, w=1.0, name=nm)
        else:
            reasons.append((nm, "nan", s))

    # Cospectral SNR
    add_score(snr, TH["snr_acc"], TH["snr_tol"], greater=True, w=1.0, name="snr_fwC")

    # Leg length
    if leg_length_km >= TH["len_acc"]:
        score += 2.0
        reasons.append(("length_km", "acc", leg_length_km))
    elif leg_length_km >= TH["len_tol"]:
        score += 1.0
        reasons.append(("length_km", "tol", leg_length_km))
    else:
        reasons.append(("length_km", "fail", leg_length_km))

    # Verdict (max ≈ 14 points)
    if score >= 10:
        verdict = "accepted"
    elif score >= 6:
        verdict = "tolerable"
    else:
        verdict = "discarded"

    return {
        "verdict":      verdict,
        "score":        score,
        "flux_int_abs": flux_int,
        "metrics": dict(
            R2=R2, max_dev=max_dev, tv_og=tv_og,
            f95_Hz=f95,
            og_plateau_err=og_plateau_err, og_spread=og_spread,
            slope_theta=slope_th, slope_w=slope_w,
            snr_fwC=snr,
            leg_length_km=leg_length_km,
        ),
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Simple / manuscript QC classification
# ---------------------------------------------------------------------------

def classify_leg_simple(
    qc: dict,
    leg_length_km: float,
) -> dict:
    """Manuscript-described QC verdict for a flight leg.

    Evaluates the four criteria of manuscript Appendix A (thresholds are
    pinned by ``tests/test_most_schemes.py::test_qc_thresholds_match_manuscript``):

    1. Spectral slope of *w*:   |s_w + 5/3| ≤ 0.4
    2. FFT ogive plateau error: og_fft_plateau_err < 0.001  (normalised ogive at f < 0.1 Hz)
    3. Stationarity (R²):       R² ≥ 0.4  (cum-covariance linear fit)
    4. SNR:                     SNR > 2

    No length gate is applied: all segments are 5 km by construction.

    The ogive plateau is computed from a single-realisation FFT (not the Welch
    estimate), which resolves frequencies down to 1/T_leg ≈ 0.02 Hz — below
    the 0.1 Hz pre-filter cutoff where the normalised ogive converges to 1.
    The R² criterion is the fraction of the cumulative co-covariance curve
    explained by a linear trend (stationary = linear accumulation).

    Verdict rules
    -------------
    * ``'accepted'``  – all criteria passed
    * ``'tolerable'`` – exactly one criterion violated
    * ``'discarded'`` – two or more criteria violated

    Parameters
    ----------
    qc            : Dict returned by :func:`qc_leg`.
    leg_length_km : Actual length of the bin [km].

    Returns
    -------
    dict with keys:
    ``verdict``   str
    ``n_failed``  int   – number of violated criteria (excludes length gate)
    ``metrics``   dict  – the numeric values of each diagnostic
    ``reasons``   list  – list of (name, status, value) tuples
    """
    f                  = qc["f"]
    fwCwth             = qc["fwCwth"]
    og_fft_plateau_err = float(qc["og_fft_plateau_err"])  # FFT ogive plateau check
    R2                 = float(qc["R2"])                  # stationarity

    # ── Slope of w spectrum in inertial subrange (0.1–20 Hz log-log) ──────
    slope_w = float(qc["slope_w"])   # pre-computed in qc_leg

    # ── Cospectral SNR ─────────────────────────────────────────────────────
    snr = _cospec_peak_snr(f, fwCwth)

    # ── Thresholds ────────────────────────────────────────────────────────
    SLOPE_TOL  = 0.4        # |s_w + 5/3| ≤ this
    SLOPE_MU   = -5.0 / 3.0
    OGERR_LIM  = 0.001       # FFT ogive plateau error
    R2_LIM     = 0.4        # minimum R² for stationarity
    SNR_LIM    = 2.0

    # ── Evaluate each criterion ────────────────────────────────────────────
    reasons = []
    n_failed = 0

    def _check(name, val, passed):
        nonlocal n_failed
        status = "acc" if passed else "fail"
        if not passed:
            n_failed += 1
        reasons.append((name, status, val))

    slope_ok = np.isfinite(slope_w)          and abs(slope_w - SLOPE_MU) <= SLOPE_TOL
    _check("slope_w",           slope_w,           slope_ok)
    og_ok    = np.isfinite(og_fft_plateau_err) and og_fft_plateau_err < OGERR_LIM
    _check("og_fft_plateau_err", og_fft_plateau_err, og_ok)
    r2_ok    = np.isfinite(R2)                and R2 >= R2_LIM
    _check("R2",                R2,                r2_ok)
    snr_ok   = np.isfinite(snr)               and snr > SNR_LIM
    _check("snr_fwC",           snr,               snr_ok)

    # ── Verdict ────────────────────────────────────────────────────────────
    if n_failed == 0:
        verdict = "accepted"
        reasons.append(("length_km", "acc", leg_length_km))
    elif n_failed == 1:
        verdict = "tolerable"
        reasons.append(("length_km", "acc", leg_length_km))
    else:
        verdict = "discarded"
        reasons.append(("length_km", "acc", leg_length_km))

    return {
        "verdict":  verdict,
        "n_failed": n_failed,
        "metrics": dict(
            slope_w=slope_w,
            og_fft_plateau_err=og_fft_plateau_err,
            R2=R2,
            snr_fwC=snr,
            leg_length_km=leg_length_km,
        ),
        "reasons": reasons,
    }

r"""
fluxpy.qc
=========
Spectral quality control of the eddy-covariance segments.

Quality of the eddy-covariance segments (manuscript Appendix A).

A quality control of a flux *reference* must test the measurement, not the
measurand. Over compact sea ice most fluxes are close to zero, and every
criterion that is a ratio to the flux (signal-to-noise ratios, relative
stationarity tests, the R2 of the cumulative covariance) then fails for small
fluxes and passes for large ones: on the 91 segments of the paper such criteria
correlate with |H| at |rho| = 0.6-0.7 and would censor the near-zero fluxes.
They are therefore not used to discard data. Instead:

* **Integrity check (pass/fail).** The turbulence must be resolved: the power
  spectrum of the vertical wind has to follow the inertial-subrange slope,
  $|s_w + 5/3| \le 0.4$ between 0.1 and 20 Hz (Kolmogorov, 1941). Data
  completeness and constant flight level are enforced when the segments are cut
  (``fluxpy.aggregate``).
* **Random error and detection limit.** Every flux carries its own random
  error (``flux_processing.band_covariance_random_error``; Finkelstein and
  Sims, 2001). A flux is *detectable* when it exceeds 1.96 times this error.
  Fluxes below the detection limit are valid measurements of a small flux and
  are kept (Langford et al., 2015).

``qc_leg`` still returns the spectra, the cospectrum, the ogive and the
cumulative covariance, which are shown in Fig. A1 and used by
``scripts/ec_quality.py`` for the robustness table.

Public API
----------
highpass(x, fs, fc, order)             – 4th-order Butterworth high-pass filter
cospectrum(w, x, fs, ...)              – cross-power spectral density (real part)
powerspec(x, fs, ...)                  – one-sided power spectral density (Welch)
ogive_from_cospec(f, Co)               – cumulative ogive (high → low frequency)
qc_leg(w, theta, u, v, fs, x_dist)     – per-segment spectral metrics
turbulence_resolved(slope_w)           – inertial-subrange integrity check
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
    "turbulence_resolved",
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
    # 1. Linear detrend only (no high-pass; manuscript Appendix A)
    w  = detrend(w,     type="linear")
    th = detrend(theta, type="linear")
    uu = detrend(u,     type="linear")
    vv = detrend(v,     type="linear")

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
    # spectral leakage.  Shown in Fig. A1e; not used as a QC criterion.
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
        cum_frac=frac, cum_cov_norm=cum_cov_norm,
        slope_th=slope_th, slope_w=slope_w,
        tv_og=tv_og, tv_og_self=tv_og_self, R2=R2, max_dev=max_dev,
    )


# ---------------------------------------------------------------------------
# Integrity check
# ---------------------------------------------------------------------------

SLOPE_TOL = 0.4          # |s_w + 5/3| <= SLOPE_TOL


def turbulence_resolved(slope_w: float) -> bool:
    """True when the w power spectrum follows the inertial-subrange slope,
    |s_w + 5/3| <= 0.4 (the only pass/fail criterion; see the module docstring)."""
    return bool(np.isfinite(slope_w) and abs(slope_w + 5.0 / 3.0) <= SLOPE_TOL)

r"""
fluxpy.qc
=========
Quality of the eddy-covariance reference (manuscript Appendix A).

The paper separates two things:

* **A1, integrity check (pass / fail).** The turbulence must be resolved: the
  power spectrum of the vertical wind has to follow the inertial-subrange slope,
  ``|s_w + 5/3| <= 0.4`` between 0.1 and 20 Hz (Kolmogorov, 1941).
  -> :func:`spectral_slope`, :func:`turbulence_resolved`.
  Data completeness and a constant flight level are enforced when the segments
  are cut (``fluxpy.aggregate``).
* **A2, random error of every flux.** Reported, never used to discard data
  (``fluxpy.eddy_covariance.band_covariance_random_error``).

No criterion that is a ratio to the flux is applied (A3): over compact sea ice
most fluxes are close to zero, and such criteria fail for small fluxes and pass
for large ones (on the 90 segments they correlate with |H| at |rho| = 0.6).
``scripts/ec_quality.py`` evaluates them anyway to show that the results do not
depend on them (Table A1).

:func:`segment_spectra` returns the spectra, cospectrum and ogive of Fig. A1
together with the slope of the integrity check.

All diagnostics are calculated from the 100 Hz series after linear detrending.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import csd, detrend, get_window, welch

__all__ = ["INERTIAL_SLOPE", "SLOPE_TOLERANCE", "SLOPE_BAND_HZ",
           "spectral_slope", "turbulence_resolved", "segment_spectra"]

#: Kolmogorov inertial-subrange slope of the power spectrum
INERTIAL_SLOPE = -5.0 / 3.0
#: a segment passes the integrity check when |s_w - INERTIAL_SLOPE| <= SLOPE_TOLERANCE
SLOPE_TOLERANCE = 0.4
#: frequency interval of the slope fit [Hz] (the flux band of Sect. 3.1)
SLOPE_BAND_HZ = (0.1, 20.0)


# ---------------------------------------------------------------------------
# A1  Integrity check of the resolved turbulence
# ---------------------------------------------------------------------------

def spectral_slope(f: np.ndarray, P: np.ndarray,
                   fmin: float = SLOPE_BAND_HZ[0], fmax: float = SLOPE_BAND_HZ[1]) -> float:
    """Log-log slope ``s`` of a power spectrum, ``log10 P = a + s log10 f``,
    by ordinary least squares over ``fmin <= f <= fmax`` (Appendix A1)."""
    use = (f >= fmin) & (f <= fmax) & (P > 0)
    if use.sum() < 5:
        return np.nan
    return float(np.polyfit(np.log10(f[use]), np.log10(P[use]), 1)[0])


def turbulence_resolved(slope_w: float) -> bool:
    """Integrity check: True when the w power spectrum follows the inertial-subrange
    slope, ``|s_w + 5/3| <= 0.4``. The only pass/fail criterion of the paper."""
    return bool(np.isfinite(slope_w) and abs(slope_w - INERTIAL_SLOPE) <= SLOPE_TOLERANCE)


# ---------------------------------------------------------------------------
# Spectral diagnostics of one segment (Fig. A1, Table A1)
# ---------------------------------------------------------------------------

def _welch_spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided power spectral density, Welch method, 8 s Hann windows, 50 % overlap."""
    nperseg = int(8 * fs)
    return welch(x, fs=fs, window=get_window("hann", nperseg), nperseg=nperseg,
                 noverlap=nperseg // 2, detrend="linear")


def _welch_cospectrum(x: np.ndarray, y: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Real part of the cross-spectral density (same Welch settings as the spectra)."""
    nperseg = int(8 * fs)
    f, Pxy = csd(x, y, fs=fs, window=get_window("hann", nperseg), nperseg=nperseg,
                 noverlap=nperseg // 2, detrend="linear")
    return f, np.real(Pxy)


def _ogive(f: np.ndarray, cospectrum: np.ndarray) -> np.ndarray:
    """Cumulative cospectrum from high to low frequency, Og(f0) = sum_{f >= f0} Co(f) df."""
    df = np.diff(f, prepend=f[0])
    return np.flip(np.cumsum(np.flip(cospectrum * df)))


def segment_spectra(w: np.ndarray, theta: np.ndarray, u: np.ndarray, v: np.ndarray,
                    fs: float) -> dict:
    """Spectra of one segment as shown in Fig. A1, plus the integrity-check slope.

    Parameters
    ----------
    w, theta, u, v : 100 Hz series of the segment (raw; detrended here).
    fs             : sampling frequency [Hz].

    Returns
    -------
    dict with
    ``f``                  Welch frequency axis [Hz]
    ``P_theta``, ``P_w``, ``P_u``, ``P_v``   one-sided power spectra (Fig. A1a-d)
    ``f_cospectrum``       f * Co_w,theta, the frequency-weighted cospectrum (Fig. A1e)
    ``f_fft``, ``ogive_norm``   frequency axis and normalised ogive of the full-record
                           FFT cospectrum (Fig. A1f); resolves down to 1/T_segment
    ``slope_w``            inertial-subrange slope of P_w (Appendix A1)
    ``r2_cumulative_covariance``   R^2 of a straight line through the cumulative
                           covariance along the segment; one of the relative
                           criteria evaluated, and not applied, in Appendix A3
    """
    w = detrend(np.asarray(w, dtype=float), type="linear")
    theta = detrend(np.asarray(theta, dtype=float), type="linear")
    u = detrend(np.asarray(u, dtype=float), type="linear")
    v = detrend(np.asarray(v, dtype=float), type="linear")

    # power spectra and Welch cospectrum (Fig. A1a-e)
    f, P_theta = _welch_spectrum(theta, fs)
    _, P_w = _welch_spectrum(w, fs)
    _, P_u = _welch_spectrum(u, fs)
    _, P_v = _welch_spectrum(v, fs)
    _, cospectrum = _welch_cospectrum(w, theta, fs)

    # full-record FFT cospectrum and its ogive (Fig. A1f): Hann taper, zero-padded
    # to the next power of two, normalised so that Og(0) = 1
    n = w.size
    n_fft = int(2 ** np.ceil(np.log2(n)))
    taper = np.hanning(n)
    taper /= np.sqrt(np.mean(taper ** 2))                    # keep the RMS amplitude
    W = np.fft.rfft(w * taper, n=n_fft)
    TH = np.fft.rfft(theta * taper, n=n_fft)
    f_fft = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    co_fft = np.real(W * np.conj(TH)) / (n_fft * fs)
    co_fft[1:-1] *= 2.0                                      # two-sided -> one-sided
    ogive = _ogive(f_fft, co_fft)
    ogive_norm = ogive / ogive[0] if abs(ogive[0]) > 1e-12 else ogive * np.nan

    # cumulative covariance along the segment vs a straight line (Appendix A3)
    cum_cov = np.cumsum((w - w.mean()) * (theta - theta.mean()))
    cum_cov_norm = cum_cov / cum_cov[-1] if cum_cov[-1] != 0 else cum_cov * np.nan
    fraction = np.linspace(0.0, 1.0, n)
    residual = cum_cov_norm - np.polyval(np.polyfit(fraction, cum_cov_norm, 1), fraction)
    ss_tot = float(np.sum((cum_cov_norm - cum_cov_norm.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual ** 2)) / ss_tot if ss_tot > 0 else np.nan

    return dict(
        f=f, P_theta=P_theta, P_w=P_w, P_u=P_u, P_v=P_v,
        f_cospectrum=f * cospectrum,
        f_fft=f_fft, ogive_norm=ogive_norm,
        slope_w=spectral_slope(f, P_w),
        r2_cumulative_covariance=r2,
    )

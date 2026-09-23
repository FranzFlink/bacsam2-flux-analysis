r"""
fluxpy.eddy_covariance
======================
The eddy-covariance (EC) reference flux of the paper and its random error.

Manuscript: Sect. 3.1 "Eddy-covariance method" (Eq. 1) and Appendix A2
"Random error".

    H_S,obs,EC = rho * c_p * cov_band(w', theta')        [W m-2, positive upward]

* ``w'`` and ``theta'`` are the linearly detrended 100 Hz vertical wind and
  potential temperature of one 5 km segment (about 82 s of flight).
* ``cov_band`` is the part of their covariance carried by frequencies between
  0.1 and 20 Hz (``band_covariance``). At 61 m/s true airspeed this band
  corresponds to eddies of about 3 to 610 m along the track.
* ``rho`` is the segment-mean dry-air density, ``c_p`` the specific heat of dry
  air (``fluxpy.util``).
* Every flux carries a random (sampling) error, the standard deviation the
  covariance would have if the same turbulence were sampled again
  (``band_covariance_random_error``, Finkelstein and Sims, 2001).

The one-stop function for a segment is :func:`segment_flux`. The functions
below are deliberately small so that a reader can check each step against the
equations in the paper; ``book/02_eddy_covariance.ipynb`` walks through them
for one segment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.signal import detrend

from .util import RD, air_density

__all__ = [
    "FLUX_BAND_HZ", "SAMPLING_HZ", "C_P",
    "band_covariance", "band_pass", "band_covariance_random_error",
    "SegmentFlux", "segment_flux",
]

#: frequency band of the EC flux [Hz] (manuscript Sect. 3.1; Tetzlaff et al., 2015)
FLUX_BAND_HZ = (0.1, 20.0)
#: sampling frequency of the T-Bird nose boom [Hz]
SAMPLING_HZ = 100.0
#: lag range of the random-error estimator [s] (Appendix A2: m corresponds to 10 s)
RANDOM_ERROR_MAX_LAG_S = 10.0

# Specific heat of dry air used on the EC side [J kg-1 K-1].
# NOTE: differs from fluxpy.util.CP (1004.0, used by the bulk fluxes and stated
# in the manuscript); the two are unified in the next commit so that this
# commit changes no number.
C_P = 1004.67
R = RD


# ---------------------------------------------------------------------------
# 1. Band-limited covariance = the kinematic flux  w'theta'  (Eq. 1)
# ---------------------------------------------------------------------------

def band_covariance(x: np.ndarray, y: np.ndarray, fs: float,
                    fmin: float = FLUX_BAND_HZ[0], fmax: float = FLUX_BAND_HZ[1]) -> float:
    """Covariance of ``x`` and ``y`` carried by frequencies ``fmin <= f <= fmax``.

    The one-sided cospectrum of the full record (no segmenting, no taper) is
    summed over the band. By Parseval's theorem the sum over ALL frequencies
    equals the ordinary covariance ``mean(x' y')``; the result is therefore
    exactly the in-band part of the covariance. The frequency resolution is
    ``fs / n`` (about 0.012 Hz for a 5 km segment), so the band edges are met
    to within one such bin. ``x`` and ``y`` must be detrended 1-D arrays of
    equal length.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    X = np.fft.rfft(x - x.mean())
    Y = np.fft.rfft(y - y.mean())
    cospectrum = np.real(X * np.conj(Y)) / float(n) ** 2     # two-sided, per bin
    one_sided = np.full(cospectrum.size, 2.0)                # fold negative frequencies ...
    one_sided[0] = 1.0                                       # ... except the mean
    if n % 2 == 0:
        one_sided[-1] = 1.0                                  # ... and the Nyquist bin
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    in_band = (f >= fmin) & (f <= fmax)
    return float(np.dot(one_sided[in_band], cospectrum[in_band]))


def band_pass(x: np.ndarray, fs: float,
              fmin: float = FLUX_BAND_HZ[0], fmax: float = FLUX_BAND_HZ[1]) -> np.ndarray:
    """Ideal (FFT) band-pass of a detrended 1-D series: keeps ``fmin <= f <= fmax``."""
    x = np.asarray(x, dtype=float)
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(x.size, d=1.0 / fs)
    X[(f < fmin) | (f > fmax)] = 0.0
    return np.fft.irfft(X, n=x.size)


# ---------------------------------------------------------------------------
# 2. Random error of the covariance (Appendix A2; Finkelstein and Sims, 2001)
# ---------------------------------------------------------------------------

def band_covariance_random_error(x: np.ndarray, y: np.ndarray, fs: float,
                                 fmin: float = FLUX_BAND_HZ[0], fmax: float = FLUX_BAND_HZ[1],
                                 max_lag_s: float = RANDOM_ERROR_MAX_LAG_S) -> float:
    """Random (sampling) error of the band-limited covariance, 1 sigma.

    Finkelstein and Sims (2001, J. Geophys. Res. 106, 3503-3509), Eq. (8)-(10):
    the variance of a covariance estimated from ``n`` samples is

        var = (1/n) * sum_{p=-m..m} [ g_xx(p) g_yy(p) + g_xy(p) g_yx(p) ]

    with the auto- and cross-covariance functions ``g`` at lag ``p``. They are
    evaluated on the series band-passed to ``[fmin, fmax]`` so that the error
    belongs to the same quantity as :func:`band_covariance`. ``m`` covers
    ``max_lag_s`` seconds, several times the integral time scale; the result
    changes by a few per cent for 5 to 30 s (Rannik et al., 2016, found the
    estimator robust and insensitive to this choice).

    Note: the often quoted shortcut ``sqrt(2 L / X)`` (Lenschow et al., 1994)
    lacks the factor ``sqrt((1 + r^2) / r^2)`` of the correlation ``r`` between
    ``x`` and ``y``; for the weakly correlated w and theta over sea ice
    (|r| ~ 0.08) it understates the error by an order of magnitude.
    """
    xb = band_pass(x, fs, fmin, fmax)
    yb = band_pass(y, fs, fmin, fmax)
    n = xb.size

    def covariance_function(a, b):
        """Circular (FFT) covariance function g_ab(p), normalised by n."""
        return np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n=n) / n

    m = int(max_lag_s * fs)
    lags = np.r_[0:m + 1, n - m:n]                          # p = 0..m and p = -m..-1
    var = (covariance_function(xb, xb)[lags] * covariance_function(yb, yb)[lags]
           + covariance_function(xb, yb)[lags] * covariance_function(yb, xb)[lags]).sum() / n
    return float(np.sqrt(max(var, 0.0)))


# ---------------------------------------------------------------------------
# 3. One segment: from the 100 Hz series to W m-2
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentFlux:
    """EC result of one segment (all in SI; flux positive upward)."""
    flux: float             #: H_S,obs,EC [W m-2]
    random_error: float     #: sigma_rand of the flux, 1 sigma [W m-2]
    covariance: float       #: band-limited kinematic flux w'theta' [K m s-1]
    air_density: float      #: segment-mean dry-air density [kg m-3]
    n_samples: int          #: number of 100 Hz samples

    @property
    def flux_over_random_error(self) -> float:
        """|H| / sigma_rand; the paper counts the fluxes above 1, 2 and 3."""
        return abs(self.flux) / self.random_error if self.random_error > 0 else np.nan


def segment_flux(segment: xr.Dataset, fs: float = SAMPLING_HZ,
                 band: tuple[float, float] = FLUX_BAND_HZ,
                 max_lag_s: float = RANDOM_ERROR_MAX_LAG_S,
                 w: str = "w", theta: str = "theta", T: str = "T", p: str = "p") -> SegmentFlux:
    """EC sensible heat flux and its random error for one segment.

    ``segment`` holds the 100 Hz series of the segment: vertical wind ``w``
    [m s-1], potential temperature ``theta`` [K], air temperature ``T`` [degC]
    and static pressure ``p`` [hPa] (the T-Bird file as shipped, with ``theta``
    added by ``fluxpy.meteorology.add_theta``). Steps, as in Sect. 3.1:

    1. linear detrend of ``w`` and ``theta``          ->  w', theta'
    2. covariance carried by 0.1-20 Hz                 ->  :func:`band_covariance`
    3. H = rho * c_p * covariance, rho from T and p   ->  Eq. (1)
    4. random error of step 2, scaled like step 3      ->  :func:`band_covariance_random_error`
    """
    w_prime = detrend(np.asarray(segment[w], dtype=float), type="linear")
    theta_prime = detrend(np.asarray(segment[theta], dtype=float), type="linear")
    rho = float(np.mean(air_density(np.asarray(segment[T], dtype=float),
                                    np.asarray(segment[p], dtype=float))))
    cov = band_covariance(w_prime, theta_prime, fs, *band)
    sigma_cov = band_covariance_random_error(w_prime, theta_prime, fs, *band, max_lag_s=max_lag_s)
    return SegmentFlux(flux=rho * C_P * cov, random_error=rho * C_P * sigma_cov,
                       covariance=cov, air_density=rho, n_samples=int(w_prime.size))

"""The EC segment flux (manuscript Sect. 3.1, Eq. 1) assembled from its parts."""
import numpy as np
import pytest
import xarray as xr
from scipy.signal import detrend

from fluxpy.eddy_covariance import (FLUX_BAND_HZ, band_covariance,
                                    band_covariance_random_error, segment_flux)
from fluxpy.util import CP as C_P, air_density


def _synthetic_segment(n=8200, fs=100.0, seed=7):
    """82 s of correlated w and theta plus a slow trend and a mesoscale wave."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    common = rng.standard_normal(n)
    w = 0.3 * common + 0.2 * rng.standard_normal(n) + 0.5 * np.sin(2 * np.pi * 0.02 * t)
    theta = 0.1 * common + 0.05 * rng.standard_normal(n) + 0.002 * t + 255.0
    return xr.Dataset({"w": ("time", w), "theta": ("time", theta),
                       "T": ("time", np.full(n, -18.0)), "p": ("time", np.full(n, 1005.0))},
                      coords={"time": t})


def test_segment_flux_equals_rho_cp_times_band_covariance():
    seg = _synthetic_segment()
    ec = segment_flux(seg)
    w, th = detrend(seg["w"].values), detrend(seg["theta"].values)
    rho = float(np.mean(air_density(seg["T"].values, seg["p"].values)))
    assert ec.air_density == pytest.approx(rho)
    assert ec.covariance == pytest.approx(band_covariance(w, th, 100.0, *FLUX_BAND_HZ))
    assert ec.flux == pytest.approx(rho * C_P * ec.covariance)
    assert ec.random_error == pytest.approx(rho * C_P * band_covariance_random_error(w, th, 100.0))
    assert ec.flux_over_random_error == pytest.approx(abs(ec.flux) / ec.random_error)
    assert ec.n_samples == 8200


def test_segment_flux_is_positive_for_upward_transport():
    """Warm updraughts (w' and theta' correlated) give a positive, upward flux."""
    assert segment_flux(_synthetic_segment()).flux > 0.0


def test_mesoscale_wave_outside_the_band_does_not_enter_the_flux():
    seg = _synthetic_segment()
    with_wave = segment_flux(seg).flux
    seg2 = seg.assign(w=seg["w"] - 0.5 * np.sin(2 * np.pi * 0.02 * seg["time"].values))
    without_wave = segment_flux(seg2).flux
    assert with_wave == pytest.approx(without_wave, rel=1e-2)

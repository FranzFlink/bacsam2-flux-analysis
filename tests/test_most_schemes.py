"""Physics tests against INDEPENDENT references (no re-implementation of the code).

* psi functions vs numerical quadrature of the published phi functions,
  psi(zeta) = int_0^zeta (phi(0) - phi(t)) / t dt
* neutral limit of the transfer coefficients vs the analytic log law
* sign convention and monotonic behaviour of the bulk flux
* integrity threshold of Appendix A; random error of the flux vs a Monte-Carlo experiment
"""
import math

import numpy as np
import pytest
from scipy.integrate import quad

from fluxpy.stability_lib import SCHEMES, Cd_Ch_iterative, Cd_Ch_noniterative
from fluxpy.util import KAPPA

ZETAS = (0.01, 0.1, 1.0, 5.0, 20.0)


def _psi_from_phi(phi, zeta):
    return quad(lambda t: (phi(0.0) - phi(t)) / t, 0.0, zeta)[0]


# Published stable-branch phi functions -------------------------------------
def _gafgp07():
    # Grachev et al. (2007): a_m = 5, b_m = a_m/6.5, a_h = b_h = 5, c_h = 3
    am, bm, ah, bh, ch = 5.0, 5.0 / 6.5, 5.0, 5.0, 3.0
    phi_m = lambda z: 1.0 + am * z * (1.0 + z) ** (1.0 / 3.0) / (1.0 + bm * z)
    phi_h = lambda z: 1.0 + (ah * z + bh * z * z) / (1.0 + ch * z + z * z)
    return phi_m, phi_h


def _glgs20():
    # Gryanik et al. (2020): a_m = a_h = 5, b_m = 0.3, b_h = 0.4, Pr0 = 0.98
    phi_m = lambda z: 1.0 + 5.0 * z / (1.0 + 0.3 * z) ** (2.0 / 3.0)
    phi_h = lambda z: 0.98 * (1.0 + 5.0 * z / (1.0 + 0.4 * z))
    return phi_m, phi_h


def _bd():
    # Businger-Dyer, stable: phi = 1 + 5 zeta
    phi = lambda z: 1.0 + 5.0 * z
    return phi, phi


@pytest.mark.parametrize("name, phis", [("GAFGP07", _gafgp07()), ("GLGS20", _glgs20()), ("BD", _bd())])
@pytest.mark.parametrize("zeta", ZETAS)
def test_psi_equals_integral_of_published_phi(name, phis, zeta):
    phi_m, phi_h = phis
    sc = SCHEMES[name]
    assert sc.psi_m(zeta) == pytest.approx(_psi_from_phi(phi_m, zeta), rel=1e-6, abs=1e-8)
    assert sc.psi_h(zeta) == pytest.approx(_psi_from_phi(phi_h, zeta), rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("name", sorted(SCHEMES))
def test_psi_vanishes_at_neutral(name):
    sc = SCHEMES[name]
    assert sc.psi_m(0.0) == pytest.approx(0.0, abs=1e-12)
    assert sc.psi_h(0.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("fn", [Cd_Ch_noniterative, Cd_Ch_iterative])
def test_neutral_limit_is_log_law(fn):
    z, z0, zt = 24.0, 3e-4, 3e-5
    Cd, Ch = fn(0.0, z, z0, zt, "GLGS20")
    Pr0 = SCHEMES["GLGS20"].Pr0
    assert Cd == pytest.approx(KAPPA**2 / math.log(z / z0) ** 2, rel=1e-6)
    assert Ch == pytest.approx(KAPPA**2 / (Pr0 * math.log(z / z0) * math.log(z / zt)), rel=1e-6)


def test_transfer_coefficient_decreases_with_stability():
    z, z0, zt = 24.0, 3e-4, 3e-5
    ch = [Cd_Ch_noniterative(ri, z, z0, zt, "GLGS20")[1] for ri in (-0.5, 0.0, 0.05, 0.2)]
    assert all(a > b for a, b in zip(ch, ch[1:]))


def test_bulk_flux_sign_convention_positive_upward():
    """Warm surface under cold air -> upward (positive) flux, and vice versa."""
    import xarray as xr
    from fluxpy.bulk import compute_bulk_shf_obs, reference_config

    n = 2
    ds = xr.Dataset({
        "t_kt19":   ("start_time", np.array([-5.0, -25.0])),   # degC
        "T_mean":   ("start_time", np.array([-15.0, -15.0])),  # degC
        "p_mean":   ("start_time", np.full(n, 1010.0)),        # hPa
        "u_mean":   ("start_time", np.full(n, 5.0)),
        "v_mean":   ("start_time", np.full(n, 0.0)),
        "altitude": ("start_time", np.full(n, 24.0)),
    }, coords={"start_time": np.arange(n)})
    out = compute_bulk_shf_obs(ds, reference_config())
    H = np.asarray(out["H_obs_bulk"])
    assert H[0] > 0.0 and H[1] < 0.0


def test_integrity_check_threshold():
    """Appendix A: turbulence is resolved when |s_w + 5/3| <= 0.4."""
    from fluxpy.qc import turbulence_resolved

    assert turbulence_resolved(-5 / 3)
    assert turbulence_resolved(-5 / 3 + 0.39) and turbulence_resolved(-5 / 3 - 0.39)
    assert not turbulence_resolved(-5 / 3 + 0.41)
    assert not turbulence_resolved(float("nan"))


def test_random_error_matches_monte_carlo():
    """Finkelstein-Sims random error of the band covariance vs the spread of the
    covariance over many independent realisations of the same process."""
    from fluxpy.eddy_covariance import band_covariance, band_covariance_random_error

    rng = np.random.default_rng(3)
    n, fs = 8000, 100.0
    cov, err = [], []
    for _ in range(300):
        x = rng.standard_normal(n)
        y = 0.1 * x + rng.standard_normal(n)              # weakly correlated, like w and theta
        cov.append(band_covariance(x, y, fs)); err.append(band_covariance_random_error(x, y, fs))
    assert np.mean(err) == pytest.approx(np.std(cov), rel=0.15)


def test_band_covariance_parseval_and_band_selection():
    """The band-limited covariance (EC flux estimator) must (i) sum to the ordinary
    covariance over the full band and (ii) keep only the in-band part of a two-tone signal."""
    from fluxpy.eddy_covariance import band_covariance

    rng = np.random.default_rng(0)
    n, fs = 8200, 100.0
    x = rng.standard_normal(n)
    y = 0.5 * x + rng.standard_normal(n)
    full = band_covariance(x, y, fs, 0.0, fs / 2)
    assert full == pytest.approx(np.mean((x - x.mean()) * (y - y.mean())), rel=1e-10)

    t = np.arange(n) / fs
    f_out, f_in = 5 / (n / fs), 82 / (n / fs)          # 0.061 Hz (outside), 1.0 Hz (inside)
    s = 2.0 * np.sin(2 * np.pi * f_out * t) + 1.0 * np.sin(2 * np.pi * f_in * t)
    assert band_covariance(s, s, fs, 0.1, 20.0) == pytest.approx(0.5, rel=1e-6)   # amplitude 1 -> variance 1/2

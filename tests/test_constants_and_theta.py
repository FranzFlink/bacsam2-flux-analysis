"""The constants and the potential temperature are those of the manuscript (Appendix B)."""
import numpy as np
import pytest
import xarray as xr

from fluxpy import util


def test_constants_match_the_manuscript():
    assert util.CP == 1004.0           # Appendix B1
    assert util.KAPPA == 0.40          # Appendix B3
    assert util.RD == pytest.approx(287.05)
    assert util.P0 == 100000.0         # p0 = 1000 hPa


def test_potential_temperature_poisson_equation():
    # theta = T (p0/p)^(Rd/cp); degC/hPa and K/Pa inputs give the same result
    theta = util.potential_temperature(-20.0, 950.0)
    assert theta == pytest.approx(253.15 * (1000.0 / 950.0) ** (287.05 / 1004.0))
    assert util.potential_temperature(253.15, 95000.0) == pytest.approx(theta)


def test_add_theta_adds_float64_theta_with_units():
    ds = xr.Dataset({"T": ("time", np.array([-20.0, -19.5], dtype=np.float32)),
                     "p": ("time", np.array([1000.0, 999.0], dtype=np.float32))})
    out = util.add_theta(ds)
    assert out["theta"].dtype == np.float64
    assert out["theta"].attrs["units"] == "K"
    assert float(out["theta"][0]) == pytest.approx(253.15)     # at p0, theta equals T
    assert "theta" not in ds                                    # input untouched


def test_air_density_ideal_gas():
    assert float(util.air_density(-20.0, 1000.0)) == pytest.approx(100000.0 / (287.05 * 253.15))


def test_surface_pressure_integrates_the_hydrostatic_equation():
    # independent reference: integrate dp/dz = -p g / (Rd T) downward from 24 m in 1 cm steps (isothermal)
    p, T, dz = 1000.0, 250.0, 0.01
    for _ in range(2400):
        p += p * util.G / (util.RD * T) * dz
    assert float(util.surface_pressure_hypsometric(1000.0, 24.0, T)) == pytest.approx(p, rel=1e-6)
    assert float(util.surface_pressure_hypsometric(1000.0, 24.0, -23.15)) == pytest.approx(p, rel=1e-6)   # degC input


def test_hydrostatic_surface_theta_offset_is_g_z_over_cp():
    # using the pressure 24 m up at the surface overestimates theta_s by about g z / c_p (Appendix B1)
    T_s, z, p_z = -20.0, 24.0, 1000.0
    p_s = util.surface_pressure_hypsometric(p_z, z, T_s)
    offset = float(util.potential_temperature(T_s, p_z) - util.potential_temperature(T_s, p_s))
    assert offset == pytest.approx(util.G * z / util.CP, rel=0.01)

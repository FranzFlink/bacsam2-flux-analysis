"""The bulk method of Appendix B and the skill metrics of Sect. 4.1."""
import numpy as np
import pytest
import xarray as xr

from fluxpy.bulk import (BulkConfig, bulk_richardson_in_situ, bulk_richardson_remote,
                         compute_bulk_shf_obs, effective_roughness, reference_config)
from fluxpy.naming import PLOT_LABEL
from fluxpy.sensitivity import sensitivity_matrix, skill_metrics
from fluxpy.stability import bulk_richardson_2pt, effective_wind_speed
from fluxpy.util import G


# ── Eq. (B3): the two-point bulk Richardson number ───────────────────────────

def test_bulk_richardson_2pt_matches_manuscript_equation():
    dtheta, dz, du, dv, th_bar = 2.5, 31.0, 3.0, -1.5, 258.0
    expected = (G / th_bar) * dtheta * dz / (du ** 2 + dv ** 2)
    assert bulk_richardson_2pt(dtheta=dtheta, dz=dz, du=du, dv=dv, theta_ref=th_bar,
                               eps_wind2=0.0) == pytest.approx(expected, rel=1e-12)


def test_bulk_richardson_sign_convention():
    stable = bulk_richardson_2pt(dtheta=1.0, dz=30.0, du=3.0, dv=0.0, theta_ref=260.0)
    unstable = bulk_richardson_2pt(dtheta=-1.0, dz=30.0, du=3.0, dv=0.0, theta_ref=260.0)
    assert stable > 0 > unstable


def test_bulk_richardson_component_and_speed_modes_agree():
    ri_c = bulk_richardson_2pt(dtheta=1.0, dz=30.0, du=3.0, dv=4.0, theta_ref=260.0)
    ri_s = bulk_richardson_2pt(dtheta=1.0, dz=30.0, theta_ref=260.0, wspd=5.0)
    assert np.isclose(ri_c, ri_s, rtol=1e-10)


def test_in_situ_ri_uses_segment_means_with_epsilon():
    """In-situ pathway: surface (no-slip) to T-Bird level, measured wind, eps = 1e-4."""
    cfg = reference_config()
    theta_sfc = np.array([255.0, 262.0, 258.0])
    theta_air = np.array([258.0, 259.0, 258.5])
    u, v, z = np.array([3.0, 0.5, -2.0]), np.array([1.0, 0.0, 2.0]), np.array([30.0, 32.0, 28.0])
    ri = bulk_richardson_in_situ(theta_sfc, theta_air, u, v, z, cfg)
    th_bar = 0.5 * (theta_sfc + theta_air)
    expected = (G / th_bar) * (theta_air - theta_sfc) * z / (u ** 2 + v ** 2 + cfg.ri_eps_wind2 * z ** 2)
    assert np.allclose(ri, expected, rtol=1e-10)


def test_ri_clip_and_epsilon_are_configurable():
    theta_sfc, theta_air = np.array([250.0]), np.array([256.0])     # strongly stable
    u, v, z = np.array([0.05]), np.array([0.0]), np.array([31.0])   # near calm
    raw = bulk_richardson_in_situ(theta_sfc, theta_air, u, v, z, BulkConfig(ri_eps_wind2=0.0))
    assert raw[0] > 10.0                                            # diverges without eps or bound
    clipped = bulk_richardson_in_situ(theta_sfc, theta_air, u, v, z, BulkConfig(ri_clip=2.0))
    assert clipped[0] == 2.0
    regularised = bulk_richardson_in_situ(theta_sfc, theta_air, u, v, z, BulkConfig(ri_eps_wind2=1e-2))
    assert regularised[0] < raw[0]


def test_effective_wind_speed_bounds_both_branches():
    S = effective_wind_speed(np.array([0.0, 5.0]), np.array([3.0, 3.0]),
                             np.array([31.0, 31.0]), np.array([258.0, 258.0]))
    assert np.isclose(S[0], 0.5)                       # Jordan et al. (1999): floor of 0.5 m/s
    assert 5.0 < S[1] < 5.01                           # and S -> U in moderate wind
    S_u = effective_wind_speed(np.array([0.0]), np.array([-5.0]), np.array([31.0]), np.array([258.0]))
    assert np.isfinite(S_u[0]) and S_u[0] > 0.3        # Fairall et al. (1996): gustiness
    S_n = effective_wind_speed(np.array([2.0]), np.array([np.nan]), np.array([31.0]), np.array([258.0]))
    assert S_n[0] == 2.0                               # NaN input passes the raw wind through


def test_remote_ri_is_finite_in_dead_calm_without_a_bound():
    cfg = reference_config()
    assert np.isinf(cfg.ri_clip)
    ri = bulk_richardson_remote(np.array([250.0, 262.0]), np.array([256.0, 254.0]),
                                np.zeros(2), np.array([31.0, 31.0]), cfg)
    assert np.all(np.isfinite(ri)) and ri[0] > 0.0 > ri[1]


# ── Eq. (B4): effective roughness ────────────────────────────────────────────

def test_effective_roughness_is_the_fraction_weighted_geometric_mean():
    cfg = reference_config()
    z0, zt = effective_roughness(np.array([1.0, 0.0, 0.5]), np.array([0.0, 1.0, 0.0]),
                                 np.array([0.0, 0.0, 0.5]), cfg)
    assert z0[0] == pytest.approx(cfg.z0_ocean) and z0[1] == pytest.approx(cfg.z0_thin_ice)
    assert z0[2] == pytest.approx(np.sqrt(cfg.z0_ocean * cfg.z0_thick_ice))
    assert np.allclose(zt, z0 * cfg.zt_ratio)


def test_bulk_flux_does_not_depend_on_the_roughness_for_ri():
    """Ri_b enters through the temperature and wind only; the roughness changes C_h, not Ri_b."""
    n = 2
    ds = xr.Dataset({"t_kt19": ("start_time", np.array([-5.0, -25.0])),
                     "T_mean": ("start_time", np.array([-15.0, -15.0])),
                     "p_mean": ("start_time", np.full(n, 1010.0)),
                     "u_mean": ("start_time", np.full(n, 5.0)), "v_mean": ("start_time", np.zeros(n)),
                     "altitude": ("start_time", np.full(n, 24.0))}, coords={"start_time": np.arange(n)})
    a = compute_bulk_shf_obs(ds, reference_config())
    b = compute_bulk_shf_obs(ds, BulkConfig(z0_constant=5e-3, label="rough"))
    assert np.allclose(a["Ri_obs"], b["Ri_obs"])
    assert not np.allclose(a["Ch_obs"], b["Ch_obs"])


# ── the sensitivity matrix and the skill metrics ─────────────────────────────

def test_sensitivity_matrix_contains_the_runs_quoted_in_the_paper():
    labels = [c.label for c in sensitivity_matrix()]
    assert labels[0] == "GLGS20_ref"
    for needed in ("GLGS20_ztr1e-04", "BD_ref", "GAFGP07_ref", "GLGS20_mosaic", "GLGS20_z0c4e-04"):
        assert needed in labels
    assert len(labels) == len(set(labels))


def test_skill_metrics_basic():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    m = skill_metrics(ref, 2.0 * ref + 1.0)
    assert m["n"] == 4 and np.isclose(m["slope"], 2.0) and np.isclose(m["r"], 1.0)
    assert np.isclose(m["bias"], np.mean(ref + 1.0))


def test_plot_labels_cover_all_headline_series():
    for key in ["H_obs_ec", "H_obs_bulk", "H_rs_bulk", "H_era_bulk", "H_era_diag", "H_car_diag"]:
        assert key in PLOT_LABEL

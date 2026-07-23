"""
fluxpy
======
Turbulent surface flux library for BACSAM-II.

Modules
-------
util          Physical constants and unit helpers.
stability     MOST schemes; classic 2-pt bulk Ri_b (the only Richardson
              number in the code base — manuscript Eq. rib).
ec            Eddy-covariance flux wrappers (spectral, scalar output).
bulk          MOST bulk flux; BulkConfig; classic 2-pt Ri_b throughout.
qc            Spectral QC: per-leg diagnostics and soft verdict classification.
aggregate     Parameterised aggregation pipeline and xarray export.
sensitivity   Sensitivity sweep, skill metrics.

Quick start
-----------
    import fluxpy
    from fluxpy import stability, ec, bulk, qc, aggregate

    # Classic 2-pt bulk Richardson number (manuscript Eq. rib):
    # surface (no-slip) -> measurement level, layer-mean theta_ref
    Ri_b = stability.bulk_richardson_2pt(dtheta=th1 - th0, dz=z1,
                                         du=u1, dv=v1,
                                         theta_ref=0.5 * (th0 + th1))

    # Bulk SHF with T-Bird as intermediate level
    cfg = bulk.reference_config()
    ds_bulk = bulk.compute_bulk_shf_obs(ds_obs, cfg, ds_tbird=ds_tbird)

    # EC flux from a raw leg dataset
    H_obs_ec = ec.sensible_heat_flux(leg_ds)

    # Full aggregation pipeline (parameterised bin length etc.)
    acfg = aggregate.AggConfig(bin_length_km=5.0, min_samples=5000)
    df   = aggregate.process_legs(legs, df_meta, ds_rad, ds_tbird, ds_obs,
                                  ds_sic, ds_track, ds_ar, cfg=acfg)
    ds   = aggregate.to_xarray(df)

    # Full sensitivity sweep
    stacked, summary = sensitivity.run_sensitivity(ds_obs, ds_rs, ds_modis)
"""
from . import util, stability, ec, bulk, qc, aggregate, sensitivity   # noqa: F401

from .util       import potential_temperature, air_density                 # noqa: F401
from .stability  import bulk_richardson_2pt, bulk_richardson  # noqa: F401
from .bulk       import BulkConfig, reference_config                       # noqa: F401
from .qc         import qc_leg, classify_leg_soft                          # noqa: F401
from .aggregate  import AggConfig, process_legs, to_xarray                 # noqa: F401
from .sensitivity import skill_metrics                                     # noqa: F401

__version__ = "0.2.0"

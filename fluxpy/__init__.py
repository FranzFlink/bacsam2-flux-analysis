"""
fluxpy
======
All physics of Müller et al., "Evaluation of reanalysis and satellite products
of sensible heat flux over Arctic sea ice against airborne eddy-covariance
measurements" (AMT). Every module names the manuscript section it implements.

Module map (paper section -> module)
-----------------------------------
Sect. 3.1  Eddy-covariance method, Eq. (1) ...... eddy_covariance   band-limited covariance,
                                                                    random error, segment_flux
Sect. 2.1 / 3.1  5 km segments .................. aggregate         100 Hz legs -> one row per segment
Appendix A  Quality of the EC reference ......... qc                integrity check, Fig. A1 spectra
Sect. 3.2 / Appendix B  Bulk method, Eq. (2) .... bulk              H = rho c_p C_h U (theta_s - theta_air)
Appendix B1  Bulk Richardson number, Eq. (B3) ... stability        2-point Ri_b, effective wind
Appendix B3  Transfer coefficient, f_h .......... stability_lib     MOST schemes of Gryanik and Lüpkes (2023)
Sect. 4.2 / Appendix C  Sensitivity ............. sensitivity       configuration matrix, skill metrics
Table 2  Uncertainty budget ..................... uncertainty
constants, theta, rho, surface classes .......... util
figure style and variable names ................. figstyle, naming

Quick start
-----------
    from fluxpy.eddy_covariance import segment_flux          # one 5 km segment -> H, sigma_rand
    from fluxpy.bulk import reference_config, compute_bulk_shf_obs
    from fluxpy.sensitivity import skill_metrics              # slope, r, bias, RMSE against the EC flux
"""
from . import util, stability, bulk, qc, aggregate, sensitivity, eddy_covariance   # noqa: F401

from .util import potential_temperature, air_density                          # noqa: F401
from .stability import bulk_richardson_2pt                                    # noqa: F401
from .bulk import BulkConfig, reference_config, amplitude_matched_config     # noqa: F401
from .qc import segment_spectra, turbulence_resolved                          # noqa: F401
from .eddy_covariance import segment_flux                                     # noqa: F401
from .aggregate import SegmentConfig, process_legs, to_xarray                 # noqa: F401
from .sensitivity import skill_metrics, amplitude_metrics, slope_bootstrap_ci  # noqa: F401

__version__ = "1.5.0"

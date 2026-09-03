"""
Mathematical helpers for TPMS, surface textures, and lattice modeling.
"""

from graphite.math.textures import (
    bump_field,
    knurl_field,
    microgroove_field,
    spinodal_spectral_field,
    triplanar_map,
    triplanar_weights,
)
from graphite.math.tpms import (
    calculate_integrated_phase,
    evaluate_tpms,
    evaluate_tpms_phase,
    gyroid,
    lidinoid,
    neovius,
    schwarz_d,
    schwarz_diamond,
    schwarz_p,
    split_p,
)

__all__ = [
    "bump_field",
    "calculate_integrated_phase",
    "evaluate_tpms",
    "evaluate_tpms_phase",
    "gyroid",
    "knurl_field",
    "lidinoid",
    "microgroove_field",
    "neovius",
    "schwarz_d",
    "schwarz_diamond",
    "schwarz_p",
    "spinodal_spectral_field",
    "split_p",
    "triplanar_map",
    "triplanar_weights",
]

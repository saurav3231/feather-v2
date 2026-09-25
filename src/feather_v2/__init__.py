"""Feather v2 — The People's LLM Engine.

CPU-native, open-source, maximum output / minimum resource.
13 mathematics + 7 components.
"""

from __future__ import annotations

from .base import BaseComponent
from .hardware import detect_cpu_features, get_best_kernel, summary
from .model import FeatherV2Model
from .utils import (
    cos_sim,
    hybrid_adaptive_tokenizer,
    hybrid_wht,
    normalize,
)
from .utils import (
    adaptive_clifford_product as clifford_product,
)
from .utils import (
    adaptive_equilibrium_update as equilibrium_update,
)
from .utils import (
    adaptive_fractional_weights as fractional_weights,
)
from .utils import (
    adaptive_jacobi_decode as jacobi_decode,
)
from .utils import (
    adaptive_p_adic_chunk_retrieve as p_adic_chunk_retrieve,
)
from .utils import (
    adaptive_p_adic_distance as p_adic_distance,
)
from .utils import (
    adaptive_rough_path_signature as rough_path_signature,
)
from .utils import (
    adaptive_sheaf_consistency as sheaf_consistency,
)
from .utils import (
    adaptive_sinkhorn as sinkhorn,
)
from .utils import (
    adaptive_tropical_matmul as tropical_matmul,
)
from .utils import (
    adaptive_tropical_min as tropical_min,
)
from .utils import (
    adaptive_tt_compress as tt_compress,
)
from .utils import kan_activation

__all__ = [
    "BaseComponent",
    "FeatherV2Model",
    "detect_cpu_features",
    "get_best_kernel",
    "summary",
    "cos_sim",
    "hybrid_adaptive_tokenizer",
    "hybrid_wht",
    "normalize",
    "clifford_product",
    "equilibrium_update",
    "fractional_weights",
    "jacobi_decode",
    "p_adic_chunk_retrieve",
    "p_adic_distance",
    "rough_path_signature",
    "sheaf_consistency",
    "sinkhorn",
    "tropical_matmul",
    "tropical_min",
    "tt_compress",
    "kan_activation",
]

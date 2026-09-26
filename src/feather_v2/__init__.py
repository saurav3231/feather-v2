"""Feather v2 — The People's LLM Engine.

CPU-native, open-source, maximum output / minimum resource.
13 mathematics + 7 components.

The trainable model lives in :mod:`feather_v2.model` and requires PyTorch. The
NumPy reference implementations in :mod:`feather_v2.utils` and
:mod:`feather_v2.models` remain available for inference-time cross-checks.
"""

from __future__ import annotations

from .base import BaseComponent
from .hardware import detect_cpu_features, get_best_kernel, summary
from .model import DEFAULT_CONFIG, CognitiveBlock, FeatherV2Model, load_config
from .nn_math import (
    KANLinear,
    alpha_dropout,
    clifford_gate,
    divisibility_profile,
    equilibrium_update,
    fwht,
    godel_log_code,
    ifwht,
    p_adic_weights,
    rough_path_signature,
    sheaf_project,
    sinkhorn,
    tt_compress,
    tt_factors,
    tt_matmul,
)
from .utils import (
    build_param_from_p,
    cos_sim,
    create_param_from_p,
    hybrid_adaptive_tokenizer,
    hybrid_wht,
    normalize,
    normalize_L2,
    tt_clifford_reduce,
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
    "CognitiveBlock",
    "DEFAULT_CONFIG",
    "FeatherV2Model",
    "KANLinear",
    "alpha_dropout",
    "build_param_from_p",
    "clifford_gate",
    "cos_sim",
    "create_param_from_p",
    "detect_cpu_features",
    "divisibility_profile",
    "equilibrium_update",
    "fwht",
    "get_best_kernel",
    "godel_log_code",
    "hybrid_adaptive_tokenizer",
    "hybrid_wht",
    "ifwht",
    "load_config",
    "normalize",
    "normalize_L2",
    "p_adic_weights",
    "rough_path_signature",
    "sheaf_project",
    "sinkhorn",
    "summary",
    "tt_clifford_reduce",
    "tt_compress",
    "tt_factors",
    "tt_matmul",
    "clifford_product",
    "fractional_weights",
    "jacobi_decode",
    "p_adic_chunk_retrieve",
    "p_adic_distance",
    "kan_activation",
]

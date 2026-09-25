"""Feather v2 — configuration loader."""

from __future__ import annotations

from typing import Any

from .hardware import get_best_kernel


class FeatherV2Config:
    """Feather v2 model configuration."""

    def __init__(self, **kwargs: Any) -> None:
        self.dim = int(kwargs.get("dim", 512))
        self.hv_dim = int(kwargs.get("hv_dim", 8192))
        self.seq_len = int(kwargs.get("seq_len", 512))
        self.chunk = int(kwargs.get("chunk", 32))
        self.num_chunks = int(kwargs.get("num_chunks", 16))
        self.tt_rank = int(kwargs.get("tt_rank", 6))
        self.moe_experts = int(kwargs.get("moe_experts", 96))
        self.moe_top_k = int(kwargs.get("moe_top_k", 1))
        self.threads = int(kwargs.get("threads", 2))
        self.precision = str(kwargs.get("precision", "int8"))
        self.vocab = int(kwargs.get("vocab", 8256))
        self.alpha_fractional = float(kwargs.get("alpha_fractional", 0.7))
        self.K_frac_recent = int(kwargs.get("K_frac_recent", 32))
        self.tau_tropical = float(kwargs.get("tau_tropical", 0.1))
        self.p_adic_p = int(kwargs.get("p_adic_p", 2))
        self.seed = int(kwargs.get("seed", 42))
        self.batch_size = int(kwargs.get("batch_size", 1))
        self.params = str(kwargs.get("params", "40M"))
        self.ram = str(kwargs.get("ram", "0.9GB"))
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    @classmethod
    def from_file(cls, path: str) -> "FeatherV2Config":
        import json

        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        fv = data.get("feather_v2_config", data)
        return cls(**fv)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

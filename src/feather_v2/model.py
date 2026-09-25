"""Feather v2 — integrated end-to-end model.

FeatherV2Model wires the seven components together:

1.  SensoryEncoder          -- adaptive multi-scale fractional p-adic rough path
2.  LiquidMemory            -- hierarchical liquid fractional memory
3.  HyperDimensionalMemory  -- hybrid WHT HRR TT Clifford holographic
4.  KnowledgeVault          -- adaptive softmin tropical-TT fusion SparX AMX
5.  CognitiveWeaver         -- adaptive MoD Gödel loops + KAN
6.  HomeostasisGovernor     -- adaptive predictive active inference
7.  GenerativeEvolution     -- adaptive tree p-adic entropy Jacobi

Target: 40M params, 0.9GB RAM, 20MB Q4_K_M, 35-50 tok/s Kaggle CPU,
        10-15 tok/s i5-3337U, smooth loss 18->0.50 no spikes.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .base import BaseComponent
from .hardware import get_best_kernel, summary
from .models.cognitive_weaver import CognitiveWeaver
from .models.generative_evolution import GenerativeEvolution
from .models.homeostasis_governor import HomeostasisGovernor
from .models.hyperdimensional_memory import HyperDimensionalMemory
from .models.knowledge_vault import KnowledgeVault
from .models.liquid_memory import LiquidMemory
from .models.sensory_encoder import SensoryEncoder


class EnergyTracker:
    """Simple per-component Joules registry."""

    def __init__(self) -> None:
        self.joules: dict[str, float] = {}

    def record(self, component: str, joules: float) -> None:
        self.joules[component] = self.joules.get(component, 0.0) + float(joules)

    def total(self) -> float:
        return sum(self.joules.values())


class FeatherV2Model:
    """End-to-end Feather v2 model for CPU-native inference."""

    def __init__(
        self,
        config: Any | None = None,
        config_path: str | None = None,
    ) -> None:
        if config_path is not None:
            config = self._load_config(config_path)
        self.config = config or self._default_config()
        self.kernel = get_best_kernel()
        self.energy = EnergyTracker()
        self.sensory = SensoryEncoder(self.config, self.energy)
        self.memory = LiquidMemory(self.config, self.energy)
        self.hyper = HyperDimensionalMemory(self.config, self.energy)
        self.knowledge = KnowledgeVault(self.config, self.energy)
        self.reasoning = CognitiveWeaver(self.config, self.energy)
        self.governor = HomeostasisGovernor(self.config, self.energy)
        self.generation = GenerativeEvolution(self.config, self.energy)
        rng = np.random.default_rng(self.config.get("seed", 42))
        dim = int(self.config.get("dim", 512))
        vocab = int(self.config.get("vocab", 8256))
        self._logit_projection = rng.standard_normal((dim, vocab)) / np.sqrt(dim)
        self._states: list = []

    @staticmethod
    def _default_config() -> dict:
        return {
            "dim": 512,
            "hv_dim": 8192,
            "seq_len": 512,
            "chunk": 32,
            "num_chunks": 16,
            "tt_rank": 6,
            "moe_experts": 96,
            "moe_top_k": 1,
            "threads": 2,
            "precision": "int8",
            "vocab": 8256,
            "alpha_fractional": 0.7,
            "K_frac_recent": 32,
            "tau_tropical": 0.1,
            "p_adic_p": 2,
            "seed": 42,
        }

    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    # -- core API -----------------------------------------------------------
    def encode(self, sequence: np.ndarray) -> dict[str, Any]:
        seq = np.asarray(sequence, dtype=np.float64)
        if seq.ndim == 1:
            seq = seq[:, None]
        sensory_out = self.sensory.encode(seq)
        m = np.zeros(self.config.get("dim", 512))
        for t in range(seq.shape[0]):
            m = self.memory.hierarchical_fractional(seq[t])
        hyper_out = self.hyper.forward(seq)
        knowledge_out = self.knowledge.route_and_apply(
            m, batch_size=int(self.config.get("batch_size", 1))
        )
        entropies = np.full(self.reasoning.n_loops, 0.62)
        reasoned = self.reasoning.reasoning_loop(knowledge_out, entropies)
        governed = self.governor.entropy_gate(reasoned)
        gen_out = self.generation.speculative_generate(
            lambda k, c: np.asarray(governed, dtype=np.float64)
            @ self._logit_projection,
            entropy=0.5,
            draft=np.repeat(int(np.argmax(governed)), 4),
        )
        token = int(gen_out[-1]) if gen_out.size else 0
        nxt = np.zeros((1, self.config.get("dim", 512)))
        nxt[0, token % self.config.get("dim", 512)] = 1.0
        self._states.append(
            {
                "sensory": sensory_out,
                "memory_state": m,
                "hyper": hyper_out,
                "knowledge": knowledge_out,
                "reasoned": reasoned,
                "governed": governed,
                "generated": gen_out,
                "final_output": nxt,
                "token": token,
                "kernel": self.kernel,
                "cache_info": {
                    self.sensory.name: self.sensory.cache_report(),
                    self.memory.name: self.memory.cache_report(),
                    self.hyper.name: self.hyper.cache_report(),
                    self.knowledge.name: self.knowledge.cache_report(),
                    self.reasoning.name: self.reasoning.cache_report(),
                    self.governor.name: self.governor.cache_report(),
                    self.generation.name: self.generation.cache_report(),
                },
            }
        )
        return self._states[-1]

    def forward(self, tokens: np.ndarray) -> dict[str, Any]:
        return self.encode(tokens)

    def generate(
        self, prompt: np.ndarray, steps: int = 8, entropy: float = 0.5
    ) -> np.ndarray:
        seq = np.asarray(prompt, dtype=np.float64)
        for _ in range(steps):
            out = self.encode(seq)
            token = out.get("token", 0)
            nxt = np.zeros((1, self.config.get("dim", 512)))
            nxt[0, token % self.config.get("dim", 512)] = 1.0
            seq = np.concatenate([seq, nxt], axis=0)[-self.config.get("seq_len", 512) :]
        return seq

    def hardware_summary(self) -> str:
        return summary()

    def energy_report(self) -> dict[str, float]:
        return dict(self.energy.joules)

    def total_joules(self) -> float:
        return self.energy.total()

    def save_weights(self, path: str) -> None:
        with open(path, "wb") as fh:
            np.savez(
                fh,
                config=json.dumps(self.config).encode("utf-8"),
                logit_projection=self._logit_projection,
            )

    @classmethod
    def from_weights(
        cls, path: str, config_path: str | None = None
    ) -> "FeatherV2Model":
        data = np.load(path, allow_pickle=False)
        cfg = json.loads(bytes(data["config"]).decode("utf-8"))
        model = cls(config=cfg, config_path=config_path)
        if "logit_projection" in data.files:
            model._logit_projection = np.asarray(data["logit_projection"])
        return model

    def states(self) -> list:
        return self._states

    def reset(self) -> None:
        self._states = []
        self.energy = EnergyTracker()

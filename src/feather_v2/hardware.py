"""Feather v2 — adaptive hardware layer.

Extends v1 detection with v2-specific tunables:
- Adaptive vocab size (256 for small, 4096 for medium, 8256 for large RAM)
- Adaptive TT rank (4-8 per layer)
- Adaptive binds per instruction (AVX-512 8, AVX2 4, AVX 2, NEON 4, Scalar 1)
- Physical cores only (no hyper-threading)
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import threading
from typing import Any

try:
    import cpuinfo as cpuinfo_mod
except Exception:
    cpuinfo_mod = None

_CPU_FEATURES = None
_CPU_INFO = None
_CPU_INFO_LOCK = threading.Lock()
_CUDA_DEVICES = None


def _numpy_features() -> dict[str, bool]:
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as feat_new

        return {str(name): bool(value) for name, value in feat_new.items()}
    except Exception:
        try:
            from numpy.core._multiarray_umath import __cpu_features__ as feat_last

            return {str(name): bool(value) for name, value in feat_last.items()}
        except Exception:
            return {}


def _features() -> dict[str, bool]:
    global _CPU_FEATURES
    if _CPU_FEATURES is None:
        _CPU_FEATURES = _numpy_features()
    return _CPU_FEATURES


def _flag(name: str, fallback: bool = False) -> bool:
    return bool(_features().get(name, fallback))


def _cpu_info() -> dict | None:
    if cpuinfo_mod is None:
        return None
    global _CPU_INFO
    if _CPU_INFO is not None:
        return _CPU_INFO
    with _CPU_INFO_LOCK:
        if _CPU_INFO is not None:
            return _CPU_INFO
        result: dict = {}

        def _probe() -> None:
            try:
                result.update(cpuinfo_mod.get_cpu_info())
            except Exception:
                pass

        thread = threading.Thread(target=_probe, daemon=True)
        thread.start()
        thread.join(timeout=3.0)
        _CPU_INFO = result or None
    return _CPU_INFO


def _brand(info: dict | None) -> str:
    if info is not None:
        brand = info.get("brand_raw")
        if brand:
            return str(brand)
    return platform.processor() or "unknown"


def _arch_string(info: dict | None) -> str:
    if info is not None:
        arch = info.get("arch_string_raw")
        if arch:
            return str(arch)
    return platform.machine() or "unknown"


def _logical_cores() -> int:
    return os.cpu_count() or 1


def _physical_cores(info: dict | None) -> int:
    logical = _logical_cores()
    if info is not None:
        try:
            count = int(info.get("count") or 0)
            if count > 0:
                return max(1, count // 2)
        except Exception:
            pass
    return max(1, logical // 2)


def _proc_cpuinfo_flags() -> list[str]:
    path = "/proc/cpuinfo"
    if not os.path.exists(path):
        return []
    try:
        flags: list[str] = []
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip().lower()
                if line.startswith("flags") and ":" in line:
                    flags.extend(line.split(":", 1)[1].split())
        return flags
    except Exception:
        return []


def _meminfo_gb() -> float | None:
    path = "/proc/meminfo"
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        kb = float(line.split()[1])
                        return kb / (1024.0 * 1024.0)
        except Exception:
            pass
    try:
        import psutil

        return float(psutil.virtual_memory().available) / (1024.0**3)
    except Exception:
        return None


def cuda_device_summary() -> list[dict[str, str]]:
    global _CUDA_DEVICES
    if _CUDA_DEVICES is not None:
        return _CUDA_DEVICES
    try:
        import shutil

        exe = shutil.which("nvidia-smi")
    except Exception:
        exe = None
    devices: list[dict[str, str]] = []
    if exe:
        try:
            out = subprocess.check_output(
                [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                timeout=3,
            ).decode("utf-8", errors="ignore")
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2:
                    devices.append({"name": parts[0], "memory_mb": parts[1]})
        except Exception:
            pass
    _CUDA_DEVICES = devices
    return devices


def is_kaggle() -> bool:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return True
    return os.path.exists("/kaggle/working") or os.path.exists("/kaggle/input")


def kaggle_env() -> dict[str, Any]:
    return {
        "is_kaggle": is_kaggle(),
        "kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        "is_competition_rerun": bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN")),
        "input_dir": "/kaggle/input" if os.path.isdir("/kaggle/input") else None,
        "working_dir": "/kaggle/working" if os.path.isdir("/kaggle/working") else None,
        "ram_gb": _meminfo_gb(),
        "cpu_flags": _proc_cpuinfo_flags(),
        "cuda_devices": cuda_device_summary(),
        "has_internet": has_internet(),
    }


def has_internet(timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection(("8.8.8.8", 53), timeout=timeout):
            return True
    except Exception:
        return False


def detect_cpu_features() -> dict[str, Any]:
    flags = _features()
    info = _cpu_info()
    cpu_flags = set(info.get("flags", [])) if info else set()
    machine = platform.machine().lower()
    return {
        "avx512": _flag("AVX512F") or "avx512f" in cpu_flags,
        "amx": any(name.startswith("AMX") for name in flags)
        or any(name.startswith("amx") for name in cpu_flags),
        "avx2": _flag("AVX2") or "avx2" in cpu_flags,
        "avx": _flag("AVX") or "avx" in cpu_flags,
        "neon": _flag("NEON") or machine in ("arm64", "aarch64"),
        "machine": machine,
        "cores_physical": _physical_cores(info),
        "cores_logical": _logical_cores(),
        "cpu": _brand(info),
        "arch_string_raw": _arch_string(info),
        "kaggle": is_kaggle(),
        "ram_gb": _meminfo_gb(),
    }


def get_best_kernel(features: dict[str, Any] | None = None) -> dict[str, Any]:
    if features is None:
        features = detect_cpu_features()
    avx512 = bool(features.get("avx512"))
    amx = bool(features.get("amx"))
    avx2 = bool(features.get("avx2"))
    avx = bool(features.get("avx"))
    neon = bool(features.get("neon"))
    physical = int(features.get("cores_physical") or 2)
    ram = float(features.get("ram_gb") or 8.0)
    if avx512 and amx:
        return {
            "hypervector_dim": 10000,
            "hv_memory_kb": 40,
            "binding": "avx512_wht",
            "binds_per_instruction": 8,
            "moe": "amx_tropical_tt",
            "moe_tiles": "16x64",
            "compute_dtype": "float32",
            "threads": physical,
            "cache": "L2",
            "adaptive_vocab": 8256,
            "adaptive_rank": 8,
            "adaptive_binds": 8,
        }
    if avx2:
        return {
            "hypervector_dim": 4096,
            "hv_memory_kb": 16,
            "binding": "avx2_wht",
            "binds_per_instruction": 4,
            "moe": "avx2_tropical_tt",
            "moe_tiles": "16x64",
            "compute_dtype": "float32",
            "threads": physical,
            "cache": "L2",
            "adaptive_vocab": 4096 if ram < 16.0 else 8256,
            "adaptive_rank": 6,
            "adaptive_binds": 4,
        }
    if avx:
        return {
            "hypervector_dim": 1024,
            "hv_memory_kb": 4,
            "binding": "avx_wht",
            "binds_per_instruction": 2,
            "moe": "avx_tropical_tt",
            "moe_tiles": "8x32",
            "compute_dtype": "float32",
            "threads": min(2, physical),
            "cache": "L1",
            "adaptive_vocab": 4096 if ram >= 8.0 else 256,
            "adaptive_rank": 4,
            "adaptive_binds": 2,
        }
    if neon:
        return {
            "hypervector_dim": 1024,
            "hv_memory_kb": 4,
            "binding": "neon_wht",
            "binds_per_instruction": 4,
            "moe": "neon_tropical",
            "moe_tiles": "4x16",
            "compute_dtype": "float32",
            "threads": physical,
            "cache": "L1",
            "adaptive_vocab": 4096,
            "adaptive_rank": 4,
            "adaptive_binds": 4,
        }
    return {
        "hypervector_dim": 512,
        "hv_memory_kb": 2,
        "binding": "scalar_wht",
        "binds_per_instruction": 1,
        "moe": "scalar_tropical",
        "moe_tiles": "1x1",
        "compute_dtype": "float32",
        "threads": 1,
        "cache": "L1",
        "adaptive_vocab": 256,
        "adaptive_rank": 2,
        "adaptive_binds": 1,
    }


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    fv = cfg.setdefault("feather_v2_config", {})
    kernel = get_best_kernel()
    fv.setdefault("hypervector_dim", kernel["hypervector_dim"])
    fv.setdefault("binding", kernel["binding"])
    fv.setdefault("moe", kernel["moe"])
    fv.setdefault("threads", kernel["threads"])
    fv.setdefault("compute_dtype", kernel["compute_dtype"])
    fv.setdefault("vocab", kernel.get("adaptive_vocab", 8256))
    fv.setdefault("tt_rank", kernel.get("adaptive_rank", 6))
    fv["hardware_detected"] = detect_cpu_features()
    fv["kernel_selected"] = kernel
    return cfg


def summary() -> str:
    feats = detect_cpu_features()
    kernel = get_best_kernel(feats)
    lines = [
        f"CPU: {feats['cpu']}",
        f"Architecture: {feats['arch_string_raw']}",
        f"Cores: {feats['cores_physical']} physical / {feats['cores_logical']} logical",
        "Features: "
        + ", ".join(
            name
            for name, ok in [
                ("AVX-512", feats["avx512"]),
                ("AMX", feats["amx"]),
                ("AVX2", feats["avx2"]),
                ("AVX", feats["avx"]),
                ("NEON", feats["neon"]),
            ]
            if ok
        )
        or "Scalar (no SIMD)",
        "Best kernel: "
        + f"{kernel['binding'].upper()} binding, "
        + f"{kernel['hypervector_dim']}-D hypervector ({kernel['hv_memory_kb']}KB {kernel['cache']}), "
        + f"{kernel['moe']}, {kernel['threads']} threads, {kernel['compute_dtype']}",
        "Throughput: not predicted. Measure with kaggle/test_all_sizes_mega.py.",
        f"Adaptive vocab: {kernel.get('adaptive_vocab', 8256)}",
        f"Adaptive TT rank: {kernel.get('adaptive_rank', 6)}",
    ]
    env = kaggle_env()
    if env["is_kaggle"]:
        lines.append(
            f"Kaggle runtime: {env['kernel_run_type'] or 'unknown'} "
            f"(competition rerun: {env['is_competition_rerun']})"
        )
    if env["ram_gb"] is not None:
        lines.append(f"RAM available: {env['ram_gb']:.1f} GB")
    if env["cuda_devices"]:
        lines.append(
            "CUDA: "
            + ", ".join(
                f"{dev['name']} {dev['memory_mb']}MB" for dev in env["cuda_devices"]
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())

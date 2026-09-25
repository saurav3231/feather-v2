"""Feather v2 — hardware detection tests."""

from __future__ import annotations

import pytest

from feather_v2.hardware import detect_cpu_features, get_best_kernel, summary


def test_detect_cpu_features_returns_dict():
    feats = detect_cpu_features()
    assert isinstance(feats, dict)
    assert "cores_physical" in feats
    assert "cores_logical" in feats
    assert feats["cores_physical"] >= 1


def test_get_best_kernel_never_fails():
    kernel = get_best_kernel()
    assert isinstance(kernel, dict)
    assert "hypervector_dim" in kernel
    assert "binding" in kernel
    assert "threads" in kernel
    assert kernel["threads"] >= 1


def test_summary_is_string():
    s = summary()
    assert isinstance(s, str)
    assert len(s) > 0


def test_kernel_threads_are_physical_cores():
    feats = detect_cpu_features()
    kernel = get_best_kernel(feats)
    assert kernel["threads"] <= feats["cores_physical"]


def test_adaptive_vocab_in_kernel():
    kernel = get_best_kernel()
    assert "adaptive_vocab" in kernel
    assert kernel["adaptive_vocab"] >= 256


def test_adaptive_rank_in_kernel():
    kernel = get_best_kernel()
    assert "adaptive_rank" in kernel
    assert kernel["adaptive_rank"] >= 1

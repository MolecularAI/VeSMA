"""Scaling benchmarks for the paper match-only and match-plus-split tables.

These run under ``pytest-benchmark``: the matching call is timed via
``benchmark.pedantic`` (rounds + warmup folded in) and every non-timing quantity is
attached to ``benchmark.extra_info``. pytest-benchmark writes one JSON file per
invocation via ``--benchmark-json``; the Makefile runs one backend per invocation
across the configured molecule sizes.

Knobs (all from the environment, set by the Makefile):
- ``BENCH_SCALING_SIZES``  molecule counts to sweep (default ``1 10 100 1000 2881``)
- ``BENCH_ROUNDS``         timed rounds per case (default 5)
- ``BENCH_MATCH_ONLY_BACKENDS`` / ``BENCH_SPLIT_BACKENDS``  which backend(s) to run
"""

from __future__ import annotations

import gc
import os
import sys
from collections.abc import Mapping

import pytest

from tests.vesma.bench_matcher import (
    MATCH_ONLY_BUILDERS,
    SPLIT_BUILDERS,
    load_match_corpus,
    load_positive_pairs,
)

pytestmark = [pytest.mark.perf, pytest.mark.paper_benchmark]

DEFAULT_SIZES = (1, 10, 100, 1000, 2881)
DEFAULT_MATCH_ONLY_BACKENDS = (
    "rdkit-filtercatalog-rules",
    "rdkit-substructlibrary-frontier",
    "cpu-serial",
    "cpu-parallel",
)
DEFAULT_SPLIT_BACKENDS = ("rdkit-serial", "cpu-serial", "cpu-parallel")


def _cuda_oom_errors() -> tuple[type[BaseException], ...]:
    """CUDA out-of-memory exception types, resolved lazily.

    Built on demand so hosts without CuPy (running only CPU/RDKit backends) never import
    it. An empty tuple (nothing resolvable) simply means the ``except`` catches nothing.
    """
    errors: list[type[BaseException]] = []
    try:
        from vesma.cuda_prune import CudaOutputCapacityError

        errors.append(CudaOutputCapacityError)
    except Exception:
        pass
    try:
        import cupy as cp

        errors.append(cp.cuda.memory.OutOfMemoryError)
    except Exception:
        pass
    return tuple(errors)


def test_match_only_scaling(benchmark, match_only_backend: str, size: int) -> None:
    corpus = load_match_corpus(size)
    target, finalize = MATCH_ONLY_BUILDERS[match_only_backend](corpus)
    result = benchmark.pedantic(
        target, rounds=rounds(), warmup_rounds=warmup_rounds(), iterations=1
    )
    benchmark.extra_info.update(
        finalize(result)
        | {
            "experiment": "match-only",
            "backend": match_only_backend,
            "size": size,
            "frontier": "n1-d2-f0",
            "template_source": "uspto42k-compat",
        }
    )
    cleanup_benchmark_iteration()


def test_match_plus_split_scaling(benchmark, split_backend: str, size: int) -> None:
    corpus = load_match_corpus(size)
    positive_pairs = load_positive_pairs(corpus)
    target, finalize = SPLIT_BUILDERS[split_backend](corpus, positive_pairs)
    try:
        result = benchmark.pedantic(
            target, rounds=rounds(), warmup_rounds=warmup_rounds(), iterations=1
        )
    except _cuda_oom_errors() as exc:
        # cuda-device keeps the whole precursor frontier resident in VRAM; the largest
        # sizes can exceed a smaller GPU. Skip (no JSON row -> renders as missing/OOM in
        # the paper) instead of failing the run. No silent fallback to host mode.
        pytest.skip(f"{split_backend} out of GPU memory at size {size}: {exc}")
    benchmark.extra_info.update(
        finalize(result)
        | {
            "experiment": "match-plus-split",
            "backend": split_backend,
            "size": size,
            "positive_pair_source": "paroutes-n1-d2-match-uspto42k",
            "frontier": "n1-d2-f0",
            "template_source": "uspto42k-compat",
        }
    )
    cleanup_benchmark_iteration()


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "size" in metafunc.fixturenames:
        metafunc.parametrize("size", scaling_sizes())
    if "match_only_backend" in metafunc.fixturenames:
        metafunc.parametrize(
            "match_only_backend",
            selected_backends(
                "BENCH_MATCH_ONLY_BACKENDS",
                DEFAULT_MATCH_ONLY_BACKENDS,
                MATCH_ONLY_BUILDERS,
            ),
        )
    if "split_backend" in metafunc.fixturenames:
        metafunc.parametrize(
            "split_backend",
            selected_backends(
                "BENCH_SPLIT_BACKENDS",
                DEFAULT_SPLIT_BACKENDS,
                SPLIT_BUILDERS,
            ),
        )


def scaling_sizes() -> tuple[int, ...]:
    raw = os.environ.get("BENCH_SCALING_SIZES")
    if not raw:
        return DEFAULT_SIZES
    return tuple(int(value) for value in raw.replace(",", " ").split() if value.strip())


def rounds() -> int:
    value = int(os.environ.get("BENCH_ROUNDS", "5"))
    if value <= 0:
        raise ValueError("BENCH_ROUNDS must be positive.")
    return value


def warmup_rounds() -> int:
    value = int(os.environ.get("BENCH_WARMUP_ROUNDS", "1"))
    if value < 0:
        raise ValueError("BENCH_WARMUP_ROUNDS must be non-negative.")
    return value


def selected_backends(
    env_name: str,
    default: tuple[str, ...],
    available: Mapping[str, object],
) -> tuple[str, ...]:
    raw = os.environ.get(env_name)
    backends = (
        default
        if not raw
        else tuple(value.strip() for value in raw.replace(",", " ").split())
    )
    unsupported = [backend for backend in backends if backend not in available]
    if unsupported:
        raise ValueError(f"{env_name} contains unsupported backends: {unsupported}")
    return tuple(backend for backend in backends if backend)


def cleanup_benchmark_iteration() -> None:
    gc.collect()
    cupy = sys.modules.get("cupy")
    if cupy is not None:
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()

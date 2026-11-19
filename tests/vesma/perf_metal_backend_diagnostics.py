from __future__ import annotations

import cProfile
import io
import pstats
import time
from statistics import median
from typing import Any

import numpy as np
import pytest

from tests.datasets import load_retro_benchmark_corpus
from tests.vesma.perf_metal_retro_benchmark import _load_benchmark_data
from vesma.gmol import GMol
from vesma.metal_matcher import GpuUnavailableError, MetalMatcher, probe_metal

pytestmark = [pytest.mark.perf, pytest.mark.gpu, pytest.mark.metal]

THREAD_VALUES = (32, 64, 128, 256, 512)
PRUNE_SWEEP_VALUES = (2, 4, 8, 16, 32, 64)
LEAF_SWEEP_VALUES = (1, 8, 32, 128, 256, 1024, 65536)
DEFAULT_PRUNE_ROUNDS = 64
DEFAULT_LEAF_CHECKS = 65536


def skip_if_no_metal() -> None:
    try:
        probe_metal()
    except (GpuUnavailableError, RuntimeError) as exc:
        pytest.skip(f"Metal backend not available: {exc}")


def pair_indices(pattern_count: int, molecule_count: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(pattern_count) for j in range(molecule_count)]


def rdkit_cross_product(rd_queries: list[Any], rd_molecules: list[Any]) -> np.ndarray:
    return np.asarray(
        [
            [molecule.HasSubstructMatch(query) for molecule in rd_molecules]
            for query in rd_queries
        ],
        dtype=bool,
    )


def run_status(
    patterns: list[GMol],
    molecules: list[GMol],
    *,
    max_prunings: int = DEFAULT_PRUNE_ROUNDS,
    max_leaf_checks: int = DEFAULT_LEAF_CHECKS,
    threads_per_group: int = 256,
) -> np.ndarray:
    matcher = MetalMatcher(
        max_prunings=max_prunings,
        max_leaf_checks=max_leaf_checks,
        threads_per_group=threads_per_group,
    )
    status, _ = matcher._run_batch_raw(
        patterns,
        molecules,
        pair_indices(pattern_count=len(patterns), molecule_count=len(molecules)),
        max_assignments=1,
        collect_assignments=False,
    )
    return status


def measure_status_ms(
    patterns: list[GMol],
    molecules: list[GMol],
    *,
    max_prunings: int = DEFAULT_PRUNE_ROUNDS,
    max_leaf_checks: int = DEFAULT_LEAF_CHECKS,
    threads_per_group: int = 256,
    repeat: int = 5,
) -> tuple[np.ndarray, float]:
    status = run_status(
        patterns,
        molecules,
        max_prunings=max_prunings,
        max_leaf_checks=max_leaf_checks,
        threads_per_group=threads_per_group,
    )
    timings_ms: list[float] = []
    for _ in range(repeat):
        started = time.perf_counter_ns()
        status = run_status(
            patterns,
            molecules,
            max_prunings=max_prunings,
            max_leaf_checks=max_leaf_checks,
            threads_per_group=threads_per_group,
        )
        timings_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    return status, median(timings_ms)


def status_counts(codes: np.ndarray) -> dict[str, int]:
    return {
        "no": int(np.count_nonzero(codes == 0)),
        "yes": int(np.count_nonzero(codes == 1)),
        "unknown": int(np.count_nonzero(codes == 2)),
    }


def integer_percentiles(values: np.ndarray) -> dict[str, int]:
    if values.size == 0:
        return {"p50": 0, "p90": 0, "p99": 0, "max": 0}
    return {
        "p50": int(np.percentile(values, 50)),
        "p90": int(np.percentile(values, 90)),
        "p99": int(np.percentile(values, 99)),
        "max": int(np.max(values)),
    }


@pytest.mark.perf
def test_perf_metal_retro_budget_sweep() -> None:
    skip_if_no_metal()
    chem = pytest.importorskip("rdkit.Chem")
    corpus = load_retro_benchmark_corpus()

    patterns = [entry.query for entry in corpus.patterns]
    molecules = [entry.molecule for entry in corpus.molecules]
    rd_queries = [chem.MolFromSmarts(entry.lhs_smarts) for entry in corpus.patterns]
    rd_molecules = [chem.MolFromSmiles(entry.smiles) for entry in corpus.molecules]
    assert all(query is not None for query in rd_queries)
    assert all(molecule is not None for molecule in rd_molecules)

    rdkit_matrix = rdkit_cross_product(rd_queries, rd_molecules)
    default_status, default_ms = measure_status_ms(patterns, molecules)
    default_codes = default_status[:, 1]
    default_positive = default_codes == 1
    pair_count = len(patterns) * len(molecules)

    print(
        {
            "pairs": pair_count,
            "default_ms": round(default_ms, 3),
            "default_pairs_per_sec": round(pair_count / (default_ms / 1000), 2),
            "default_status_counts": status_counts(default_codes),
            "default_pruning_rounds": integer_percentiles(default_status[:, 2]),
            "default_leaf_checks": integer_percentiles(default_status[:, 3]),
            "default_vs_rdkit_agreement": round(
                float(np.mean(default_positive == rdkit_matrix.reshape(-1))),
                3,
            ),
        }
    )

    print("budget_sweep:")
    for max_prunings in PRUNE_SWEEP_VALUES:
        for max_leaf_checks in LEAF_SWEEP_VALUES:
            status, elapsed_ms = measure_status_ms(
                patterns,
                molecules,
                max_prunings=max_prunings,
                max_leaf_checks=max_leaf_checks,
                repeat=3,
            )
            codes = status[:, 1]
            print(
                {
                    "max_prunings": max_prunings,
                    "max_leaf_checks": max_leaf_checks,
                    "metal_ms": round(elapsed_ms, 3),
                    "pairs_per_sec": round(pair_count / (elapsed_ms / 1000), 2),
                    "status_counts": status_counts(codes),
                    "same_as_default": round(
                        float(np.mean((codes == 1) == default_positive)),
                        3,
                    ),
                    "unknown_rate": round(
                        float(np.count_nonzero(codes == 2) / pair_count),
                        3,
                    ),
                }
            )


@pytest.mark.perf
def test_perf_metal_threadgroup_sweep_repeated_retro_subset() -> None:
    skip_if_no_metal()
    corpus = load_retro_benchmark_corpus()

    patterns = [entry.query for entry in corpus.patterns]
    molecules = [entry.molecule for entry in corpus.molecules] * 16
    pair_count = len(patterns) * len(molecules)
    baseline = run_status(patterns, molecules, threads_per_group=256)[:, 1] == 1

    print("threadgroup_sweep:")
    for threads_per_group in THREAD_VALUES:
        status, elapsed_ms = measure_status_ms(
            patterns,
            molecules,
            threads_per_group=threads_per_group,
            repeat=5,
        )
        codes = status[:, 1]
        print(
            {
                "threads_per_group": threads_per_group,
                "metal_ms": round(elapsed_ms, 3),
                "pairs_per_sec": round(pair_count / (elapsed_ms / 1000), 2),
                "status_counts": status_counts(codes),
                "same_as_256": round(float(np.mean((codes == 1) == baseline)), 3),
            }
        )


@pytest.mark.perf
def test_perf_metal_profile_uspto50k_subset() -> None:
    skip_if_no_metal()

    pattern_rows, molecule_rows = _load_benchmark_data()
    patterns = [entry[0] for entry in pattern_rows]
    molecules = [entry[0] for entry in molecule_rows]
    matcher = MetalMatcher()

    profile = cProfile.Profile()
    profile.enable()
    matches = matcher.has_match_many(patterns, molecules)
    profile.disable()

    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(
        12
    )
    print(stream.getvalue())
    print(
        {
            "pairs": len(patterns) * len(molecules),
            "hits": int(np.count_nonzero(matches)),
        }
    )

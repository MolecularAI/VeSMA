"""Quick benchmark for the folded linear fingerprint bit budget.

Folds the path features (length <= 7) into bit vectors of several widths and
reports, per width:

- collision rate: how many distinct path features collapse onto a shared bit, and
- prune ratio / over-retention: how many (molecule, rule) pairs the folded
  query-subset test keeps versus the dense baseline, and how many extra it keeps
  relative to the exact counted-sparse prefilter that folding replaces.

Prints a human-readable table and records the per-bit-size metrics in
``benchmark.extra_info`` (pytest-benchmark), written out with ``--benchmark-json``.

Run: ``make bench-folded-fingerprint``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest

from tests.datasets import load_retro_benchmark_corpus
from tests.vertsy.paper_provenance import source_artifact
from vesma.pathfp import (
    SparseFingerprintConfig,
    fold_features,
    folded_query_subset,
    sparse_path_fingerprint,
)

pytestmark = [pytest.mark.perf, pytest.mark.paper_benchmark]

BIT_SIZES = (32, 64, 128, 256, 512, 1024, 2048, 4096)
PATH_LENGTH = 7
DEFAULT_CASES = ((60, 200),)


def sparse_subset(query, molecule) -> bool:
    """Exact counted-sparse prefilter: every query feature present with >= count."""
    return all(molecule.get(feature, 0) >= count for feature, count in query.items())


def test_bench_folded_fingerprint(
    benchmark: Any,
    readonly_datadir: Path,
    folded_case: tuple[int, int],
) -> None:
    n_molecules, n_patterns = folded_case
    dataset_dir = readonly_datadir / "external" / "remix" / "retro-benchmark"

    def target() -> dict:
        return run(n_molecules, n_patterns, dataset_dir)

    metrics = benchmark.pedantic(
        target, rounds=rounds(), warmup_rounds=warmup_rounds(), iterations=1
    )
    print_table(metrics)
    benchmark.extra_info.update(
        folded_extra_info(
            metrics,
            n_molecules=n_molecules,
            n_patterns=n_patterns,
            dataset_dir=dataset_dir,
        )
    )


def rounds() -> int:
    value = int(os.environ.get("BENCH_ROUNDS", "3"))
    if value <= 0:
        raise ValueError("BENCH_ROUNDS must be positive.")
    return value


def warmup_rounds() -> int:
    value = int(os.environ.get("BENCH_WARMUP_ROUNDS", "1"))
    if value < 0:
        raise ValueError("BENCH_WARMUP_ROUNDS must be non-negative.")
    return value


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "folded_case" in metafunc.fixturenames:
        metafunc.parametrize("folded_case", folded_cases())


def folded_cases() -> tuple[tuple[int, int], ...]:
    raw = os.environ.get("BENCH_FOLDED_FINGERPRINT_CASES")
    if not raw:
        return DEFAULT_CASES
    return tuple(parse_folded_case(value) for value in raw.replace(",", " ").split())


def parse_folded_case(value: str) -> tuple[int, int]:
    parts = value.lower().replace("x", ":").split(":")
    if len(parts) != 2:
        raise ValueError(
            "BENCH_FOLDED_FINGERPRINT_CASES entries must look like '60x200'."
        )
    n_molecules, n_patterns = (int(part) for part in parts)
    if n_molecules <= 0 or n_patterns <= 0:
        raise ValueError("Folded fingerprint benchmark sizes must be positive.")
    return n_molecules, n_patterns


def run(n_molecules: int, n_patterns: int, dataset_dir: Path) -> dict:
    corpus = load_retro_benchmark_corpus(dataset_dir)
    patterns = corpus.patterns[:n_patterns]
    config = SparseFingerprintConfig(radius=PATH_LENGTH)

    started = time.perf_counter()
    mol_sparse = [
        (m.atom_count, sparse_path_fingerprint(m.molecule, query=False, config=config))
        for m in corpus.molecules[:n_molecules]
    ]
    rule_sparse = [
        (p.lhs_atom_count, sparse_path_fingerprint(p.query, query=True, config=config))
        for p in patterns
    ]
    feature_seconds = time.perf_counter() - started

    distinct_features = {f for _, fp in mol_sparse + rule_sparse for f in fp}
    dense_pairs = len(mol_sparse) * len(rule_sparse)

    # Ground-truth (exact counted-sparse) survivors, atom-count gated like prefilter.py.
    sparse_survivors = sum(
        1
        for r_atoms, r_fp in rule_sparse
        for m_atoms, m_fp in mol_sparse
        if r_atoms <= m_atoms and sparse_subset(r_fp, m_fp)
    )

    rows: list[dict] = []
    for n_bits in BIT_SIZES:
        distinct_bits = {f % n_bits for f in distinct_features}
        collision = (
            0.0
            if not distinct_features
            else 1.0 - len(distinct_bits) / len(distinct_features)
        )
        mol_bits = [(atoms, fold_features(fp, n_bits)) for atoms, fp in mol_sparse]
        rule_bits = [(atoms, fold_features(fp, n_bits)) for atoms, fp in rule_sparse]
        survivors = sum(
            1
            for r_atoms, r_bits in rule_bits
            for m_atoms, m_bits in mol_bits
            if r_atoms <= m_atoms and folded_query_subset(r_bits, m_bits)
        )
        prune = 0.0 if dense_pairs == 0 else 1.0 - survivors / dense_pairs
        over_retain = (
            0.0 if dense_pairs == 0 else (survivors - sparse_survivors) / dense_pairs
        )
        rows.append(
            {
                "n_bits": n_bits,
                "collision_rate": round(collision, 4),
                "survivors": survivors,
                "prune_ratio": round(prune, 4),
                "over_retention": round(over_retain, 4),
            }
        )

    return {
        "path_length": PATH_LENGTH,
        "molecule_count": len(mol_sparse),
        "pattern_count": len(rule_sparse),
        "dense_pairs": dense_pairs,
        "distinct_features": len(distinct_features),
        "sparse_survivors": sparse_survivors,
        "feature_build_seconds": round(feature_seconds, 3),
        "bit_sizes": rows,
    }


def folded_extra_info(
    metrics: dict,
    *,
    n_molecules: int,
    n_patterns: int,
    dataset_dir: Path,
) -> dict[str, Any]:
    return {
        "experiment": "folded-fingerprint-bit-size",
        "backend": "cpu",
        "requested_molecules": n_molecules,
        "requested_patterns": n_patterns,
        "path_length": PATH_LENGTH,
        "bit_sizes": list(BIT_SIZES),
        "molecule_count": metrics["molecule_count"],
        "pattern_count": metrics["pattern_count"],
        "dense_pairs": metrics["dense_pairs"],
        "distinct_features": metrics["distinct_features"],
        "sparse_survivors": metrics["sparse_survivors"],
        "feature_build_seconds": metrics["feature_build_seconds"],
        "per_bit_size": metrics["bit_sizes"],
        "source_artifacts": {"corpus": source_artifact(dataset_dir)},
    }


def print_table(payload: dict) -> None:
    print(
        f"molecules={payload['molecule_count']} patterns={payload['pattern_count']} "
        f"dense_pairs={payload['dense_pairs']} "
        f"distinct_features={payload['distinct_features']} "
        f"sparse_survivors={payload['sparse_survivors']} "
        f"feature_build={payload['feature_build_seconds']}s"
    )
    print(
        f"{'n_bits':>7} {'collision%':>11} {'survivors':>10} "
        f"{'prune%':>8} {'over-retain%':>13}"
    )
    for row in payload["bit_sizes"]:
        print(
            f"{row['n_bits']:>7} {row['collision_rate'] * 100:>10.2f}% "
            f"{row['survivors']:>10} {row['prune_ratio'] * 100:>7.2f}% "
            f"{row['over_retention'] * 100:>12.3f}%"
        )

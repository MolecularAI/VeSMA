from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from tests.datasets import load_retro_benchmark_corpus
from vesma.cuda_prune import match_fused_cuda, prepare_cuda_prune_batch

THREAD_VALUES = (32,)
PRUNE_SWEEP_VALUES = (8, 16, 32, 64)
LEAF_SWEEP_VALUES = (32, 64, 128, 256)
DEFAULT_PRUNE_ROUNDS = 64
DEFAULT_LEAF_CHECKS = 256


def _skip_if_no_cuda() -> Any:
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CuPy cannot access CUDA on this machine.")
    return cp


def _rdkit_cross_product(
    rd_queries: list[Any],
    rd_molecules: list[Any],
) -> np.ndarray:
    return np.asarray(
        [
            [mol.HasSubstructMatch(query) for mol in rd_molecules]
            for query in rd_queries
        ],
        dtype=bool,
    )


def _measure_cpu_ms(fn, repeat: int = 3) -> float:
    started = time.perf_counter_ns()
    for _ in range(repeat):
        fn()
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return elapsed_ms / repeat


def _measure_cuda_ms(cp: Any, fn: Any, repeat: int = 10) -> float:
    for _ in range(3):
        fn()
    cp.cuda.Stream.null.synchronize()
    started = cp.cuda.Event()
    finished = cp.cuda.Event()
    started.record()
    for _ in range(repeat):
        fn()
    finished.record()
    finished.synchronize()
    return float(cp.cuda.get_elapsed_time(started, finished) / repeat)


def _status_counts(codes: np.ndarray) -> dict[str, int]:
    return {
        "no": int(np.count_nonzero(codes == 0)),
        "yes": int(np.count_nonzero(codes == 1)),
        "maybe": int(np.count_nonzero(codes == 2)),
    }


@pytest.mark.perf
@pytest.mark.gpu
@pytest.mark.cuda
def test_perf_cuda_match_vs_rdkit() -> None:
    cp = _skip_if_no_cuda()
    chem = pytest.importorskip("rdkit.Chem")
    corpus = load_retro_benchmark_corpus()

    rd_queries = []
    for pattern in corpus.patterns:
        rd_query = chem.MolFromSmarts(pattern.lhs_smarts)
        if rd_query is None:
            raise AssertionError(
                "RDKit could not parse a curated retro benchmark query."
            )
        rd_queries.append(rd_query)

    rd_molecules = []
    for molecule in corpus.molecules:
        rd_mol = chem.MolFromSmiles(molecule.smiles)
        if rd_mol is None:
            raise AssertionError(
                "RDKit could not parse a curated retro benchmark molecule."
            )
        rd_molecules.append(rd_mol)

    vesma_patterns = [pattern.query for pattern in corpus.patterns]
    vesma_molecules = [molecule.molecule for molecule in corpus.molecules]
    pair_count = len(vesma_patterns) * len(vesma_molecules)

    batch = prepare_cuda_prune_batch(vesma_patterns, vesma_molecules)
    rdkit_matrix = _rdkit_cross_product(rd_queries, rd_molecules)
    rdkit_ms = _measure_cpu_ms(lambda: _rdkit_cross_product(rd_queries, rd_molecules))

    thread_ms = {
        threads: _measure_cuda_ms(
            cp,
            lambda threads=threads: match_fused_cuda(
                batch,
                threads_per_block=threads,
                max_prune_rounds=DEFAULT_PRUNE_ROUNDS,
                max_leaf_checks=DEFAULT_LEAF_CHECKS,
            ),
        )
        for threads in THREAD_VALUES
    }

    result = match_fused_cuda(
        batch,
        threads_per_block=32,
        max_prune_rounds=DEFAULT_PRUNE_ROUNDS,
        max_leaf_checks=DEFAULT_LEAF_CHECKS,
    )
    codes = result.status_codes_host()
    strict_positive = codes == 1
    permissive_positive = codes != 0

    print(
        {
            "pairs": pair_count,
            "rdkit_ms": round(rdkit_ms, 3),
            "cuda_match_ms": {key: round(value, 3) for key, value in thread_ms.items()},
            "rdkit_pairs_per_sec": round(pair_count / (rdkit_ms / 1000), 2),
            "cuda_pairs_per_sec": {
                key: round(pair_count / (value / 1000), 2)
                for key, value in thread_ms.items()
            },
            "status_counts": _status_counts(codes),
            "pattern_classes": {
                "positive": sum(
                    pattern.hit_class == "positive" for pattern in corpus.patterns
                ),
                "negative": sum(
                    pattern.hit_class == "negative" for pattern in corpus.patterns
                ),
            },
            "strict_agreement": round(
                float(np.mean(strict_positive == rdkit_matrix)), 3
            ),
            "permissive_agreement": round(
                float(np.mean(permissive_positive == rdkit_matrix)),
                3,
            ),
        }
    )

    print("budget_sweep:")
    for max_prune_rounds in PRUNE_SWEEP_VALUES:
        for max_leaf_checks in LEAF_SWEEP_VALUES:
            sweep_ms = _measure_cuda_ms(
                cp,
                lambda max_prune_rounds=max_prune_rounds, max_leaf_checks=max_leaf_checks: (
                    match_fused_cuda(
                        batch,
                        threads_per_block=32,
                        max_prune_rounds=max_prune_rounds,
                        max_leaf_checks=max_leaf_checks,
                    )
                ),
                repeat=5,
            )
            sweep = match_fused_cuda(
                batch,
                threads_per_block=32,
                max_prune_rounds=max_prune_rounds,
                max_leaf_checks=max_leaf_checks,
            )
            sweep_codes = sweep.status_codes_host()
            print(
                {
                    "max_prune_rounds": max_prune_rounds,
                    "max_leaf_checks": max_leaf_checks,
                    "cuda_ms": round(sweep_ms, 3),
                    "pairs_per_sec": round(pair_count / (sweep_ms / 1000), 2),
                    "maybe_rate": round(
                        float(np.count_nonzero(sweep_codes == 2) / pair_count),
                        3,
                    ),
                    "yes_rate": round(
                        float(np.count_nonzero(sweep_codes == 1) / pair_count),
                        3,
                    ),
                    "strict_agreement": round(
                        float(np.mean((sweep_codes == 1) == rdkit_matrix)),
                        3,
                    ),
                    "permissive_agreement": round(
                        float(np.mean((sweep_codes != 0) == rdkit_matrix)),
                        3,
                    ),
                }
            )

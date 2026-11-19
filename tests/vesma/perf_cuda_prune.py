from __future__ import annotations

import time
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tests.datadirs import external_data_root
from vesma import SmartsError, cmol_from_smiles, gmol_from_cmol, gmol_from_smarts
from vesma.adjacency import bond_presence
from vesma.cpu_matcher import initialize_candidates, prune
from vesma.cuda_prune import (
    _launch_bitset_kernel,
    prepare_cuda_prune_batch,
    prune_bitset_cuda,
)
from vesma.gmol import GMol

SMILES_MEMBER = "v2.0/zinc_lead-like_2011-02-12_first100k.smi"
DATASET_DIR = external_data_root() / "external" / "zbh-smarts" / "raw"
PATTERN_SMARTS = [
    "[#7]-2-c:1:c:c:c:c:c:1-[#6](=[#7])-c:3:c-2:c:c:c:c:3",
    "[#6](-[#6]#[#7])(-[#6]#[#7])=[#6]-c:1:c:c:c:c:c:1",
    "[#6]-1(-[#6]#[#7])(-[#6]#[#7])-[#6](-[#1])(-[#6](=[#8])-[#6])-[#6]-1-[#1]",
]
MOLECULE_COUNT = 64


def pattern(smarts: str) -> GMol:
    return gmol_from_smarts(smarts)[0]


def molecule(smiles: str) -> GMol:
    return gmol_from_cmol(cmol_from_smiles(smiles)[0])


def _skip_if_no_cuda():
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CuPy cannot access CUDA on this machine.")
    return cp


def _iter_smiles(path: Path):
    with zipfile.ZipFile(path) as archive:
        payload = archive.read(SMILES_MEMBER).decode("utf-8", errors="ignore")
    for line in payload.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        yield stripped.split()[0]


def _sample_molecules(limit: int) -> list[GMol]:
    sampled: list[GMol] = []
    for smiles in _iter_smiles(DATASET_DIR / "smarts-benchmark-v2.0.zip"):
        try:
            target = molecule(smiles)
        except SmartsError:
            continue
        if target.atom_count == 0 or target.atom_count > 64:
            continue
        sampled.append(target)
        if len(sampled) >= limit:
            return sampled
    raise AssertionError("Not enough molecules for CUDA prune benchmark.")


def _expected_pairs(
    patterns: list[GMol],
    molecules: list[GMol],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    candidates_out: list[np.ndarray] = []
    prune_rounds = []
    pruning_complete = []
    for query in patterns:
        pattern_edges = bond_presence(query.bonds)
        for target in molecules:
            candidates = initialize_candidates(query, target)
            pruned, rounds, complete = prune(
                candidates,
                pattern_edges,
                bond_presence(target.bonds),
            )
            candidates_out.append(pruned)
            prune_rounds.append(rounds)
            pruning_complete.append(complete)
    return (
        candidates_out,
        np.asarray(prune_rounds, dtype=np.int32),
        np.asarray(pruning_complete, dtype=bool),
    )


def _prepared_cpu_pairs(patterns: list[GMol], molecules: list[GMol]):
    payload = []
    for query in patterns:
        pattern_edges = bond_presence(query.bonds)
        for target in molecules:
            payload.append(
                (
                    initialize_candidates(query, target),
                    pattern_edges,
                    bond_presence(target.bonds),
                )
            )
    return payload


def _measure_cpu_ms(payload, repeat: int = 5) -> float:
    started = time.perf_counter_ns()
    for _ in range(repeat):
        for candidates, pattern_edges, molecule_edges in payload:
            prune(candidates, pattern_edges, molecule_edges)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return elapsed_ms / repeat


def _measure_cuda_kernel_ms(cp, fn, repeat: int = 20) -> float:
    for _ in range(5):
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


def _measure_e2e_ms(cp, fn, repeat: int = 10) -> float:
    started = time.perf_counter_ns()
    for _ in range(repeat):
        fn()
        cp.cuda.Stream.null.synchronize()
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return elapsed_ms / repeat


@pytest.mark.perf
@pytest.mark.gpu
@pytest.mark.cuda
def test_perf_cuda_prune() -> None:
    cp = _skip_if_no_cuda()

    patterns = [pattern(smarts) for smarts in PATTERN_SMARTS]
    molecules = _sample_molecules(MOLECULE_COUNT)
    expected_candidates, expected_rounds, expected_complete = _expected_pairs(
        patterns,
        molecules,
    )

    batch = prepare_cuda_prune_batch(patterns, molecules)
    bitset = prune_bitset_cuda(batch, threads_per_block=32)

    for observed, expected in zip(
        bitset.candidates_host(),
        expected_candidates,
        strict=True,
    ):
        assert np.array_equal(observed, expected)
    assert np.array_equal(bitset.prune_rounds_host(), expected_rounds)
    assert np.array_equal(bitset.pruning_complete_host(), expected_complete)

    cpu_ms = _measure_cpu_ms(_prepared_cpu_pairs(patterns, molecules))
    bitset_kernel_ms = {
        threads: _measure_cuda_kernel_ms(
            cp,
            lambda threads=threads: _launch_bitset_kernel(
                batch,
                threads_per_block=threads,
            ),
        )
        for threads in (1, 32, 64)
    }
    bitset_e2e_ms = {
        threads: _measure_e2e_ms(
            cp,
            lambda threads=threads: prune_bitset_cuda(
                prepare_cuda_prune_batch(patterns, molecules),
                threads_per_block=threads,
            ),
        )
        for threads in (1, 32, 64)
    }

    print(
        {
            "pairs": len(patterns) * len(molecules),
            "cpu_prune_ms": round(cpu_ms, 3),
            "bitset_kernel_ms": {
                key: round(value, 3) for key, value in bitset_kernel_ms.items()
            },
            "bitset_e2e_ms": {
                key: round(value, 3) for key, value in bitset_e2e_ms.items()
            },
        }
    )

from __future__ import annotations

import time
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tests.datadirs import external_data_root
from vesma import SmartsError, cmol_from_smiles, gmol_from_cmol, gmol_from_smarts
from vesma.adjacency import bond_presence
from vesma.cpu_matcher import enumerate_assignments, initialize_candidates

from .prune_kernel_prototypes import (
    pack_rows_u32,
    pack_rows_u64,
    prune_bitset_dirty_u32,
    prune_bitset_dirty_u64,
    prune_bitset_hybrid_u32,
    prune_bitset_hybrid_u64,
    prune_bitset_popcount_block16_u64,
    prune_bitset_popcount_block32_u64,
    prune_bitset_popcount_block64_u64,
    prune_bitset_popcount_u64,
    prune_bitset_u32,
    prune_bitset_u64,
    prune_bitset_vectorized_u32,
    prune_bitset_vectorized_u64,
    prune_dense_gemm,
    prune_dense_reference,
    prune_signature_bitset_u32,
    prune_signature_bitset_u64,
    prune_signature_hybrid_u32,
    prune_signature_hybrid_u64,
    prune_signature_vectorized_u32,
    prune_signature_vectorized_u64,
)

SMILES_MEMBER = "v2.0/zinc_lead-like_2011-02-12_first100k.smi"
DATASET_DIR = external_data_root() / "external" / "zbh-smarts" / "raw"
REPRESENTATIVE_PATTERNS = [
    "[#7]-2-c:1:c:c:c:c:c:1-[#6](=[#7])-c:3:c-2:c:c:c:c:3",
    "[#6](-[#6]#[#7])(-[#6]#[#7])=[#6]-c:1:c:c:c:c:c:1",
    "[#6]-1(-[#6]#[#7])(-[#6]#[#7])-[#6](-[#1])(-[#6](=[#8])-[#6])-[#6]-1-[#1]",
]
MOLECULE_COUNT = 100
LOOPS = 300


def _iter_smiles(path: Path):
    with zipfile.ZipFile(path) as archive:
        payload = archive.read(SMILES_MEMBER).decode("utf-8", errors="ignore")
    for line in payload.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        yield stripped.split()[0]


def _representative_case(smarts: str):
    query = gmol_from_smarts(smarts)[0]
    pattern_edges = bond_presence(query.bonds)

    best = None
    best_true = -1
    examined = 0
    for smiles in _iter_smiles(DATASET_DIR / "smarts-benchmark-v2.0.zip"):
        try:
            molecule = gmol_from_cmol(cmol_from_smiles(smiles)[0])
        except SmartsError:
            continue
        if molecule.atom_count == 0 or molecule.atom_count > 40:
            continue
        examined += 1
        candidates = initialize_candidates(query, molecule)
        true_count = int(candidates.sum())
        if true_count > best_true:
            best = (smiles, molecule, candidates)
            best_true = true_count
        if best is not None and (
            best_true >= query.atom_count * 8 or examined >= MOLECULE_COUNT
        ):
            break

    assert best is not None
    smiles, molecule, candidates = best
    return {
        "smarts": smarts,
        "smiles": smiles,
        "query": query,
        "molecule": molecule,
        "candidates": candidates,
        "pattern_edges": pattern_edges,
        "molecule_edges": bond_presence(molecule.bonds),
    }


def _benchmark_pack(case, pack_fn) -> float:
    candidates = case["candidates"]
    pattern_edges = case["pattern_edges"]
    molecule_edges = case["molecule_edges"]

    for _ in range(10):
        pack_fn(candidates)
        pack_fn(pattern_edges)
        pack_fn(molecule_edges)

    started = time.perf_counter_ns()
    for _ in range(LOOPS):
        pack_fn(candidates)
        pack_fn(pattern_edges)
        pack_fn(molecule_edges)
    return (time.perf_counter_ns() - started) / 1_000_000


def _benchmark_prune(
    case, prune_fn, *, expect_exact: bool = True
) -> dict[str, float | int]:
    expected = prune_dense_reference(
        case["candidates"],
        case["pattern_edges"],
        case["molecule_edges"],
    )
    expected_status = enumerate_assignments(
        expected[0],
        case["query"].bonds,
        case["molecule"].bonds,
        pattern_aromaticity=case["query"].atom_masks.A,
        molecule_aromaticity=case["molecule"].atom_masks.A,
        return_first=True,
    ).status

    for _ in range(5):
        observed = prune_fn(case)
        if expect_exact:
            assert np.array_equal(observed[0], expected[0])
            assert observed[2] == expected[2]
        observed_status = enumerate_assignments(
            observed[0],
            case["query"].bonds,
            case["molecule"].bonds,
            pattern_aromaticity=case["query"].atom_masks.A,
            molecule_aromaticity=case["molecule"].atom_masks.A,
            return_first=True,
        ).status
        assert observed_status == expected_status

    started = time.perf_counter_ns()
    for _ in range(LOOPS):
        observed = prune_fn(case)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    observed_status = enumerate_assignments(
        observed[0],
        case["query"].bonds,
        case["molecule"].bonds,
        pattern_aromaticity=case["query"].atom_masks.A,
        molecule_aromaticity=case["molecule"].atom_masks.A,
        return_first=True,
    ).status

    return {
        "prune_ms": round(elapsed_ms, 3),
        "per_call_ms": round(elapsed_ms / LOOPS, 5),
        "per_round_ms": round(elapsed_ms / max(LOOPS * observed[1], 1), 5),
        "rounds": observed[1],
        "candidate_true_after": int(observed[0].sum()),
        "status": observed_status,
    }


@pytest.mark.perf
def test_perf_prune_kernel() -> None:
    print("prune_kernel_microbench:")
    for smarts in REPRESENTATIVE_PATTERNS:
        case = _representative_case(smarts)
        dense_metrics = _benchmark_prune(
            case,
            lambda payload: prune_dense_reference(
                payload["candidates"],
                payload["pattern_edges"],
                payload["molecule_edges"],
            ),
        )
        dense_gemm_metrics = _benchmark_prune(
            case,
            lambda payload: prune_dense_gemm(
                payload["candidates"],
                payload["pattern_edges"],
                payload["molecule_edges"],
            ),
        )
        report = {
            "smarts": smarts,
            "smiles": case["smiles"],
            "candidate_shape": tuple(case["candidates"].shape),
            "candidate_true_before": int(case["candidates"].sum()),
            "dense": dense_metrics,
            "dense_gemm": dense_gemm_metrics,
            "u64_pack_ms": round(_benchmark_pack(case, pack_rows_u64), 3),
            "u64": _benchmark_prune(
                case,
                lambda payload: prune_bitset_u64(
                    payload["candidates"],
                    payload["pattern_edges"],
                    payload["molecule_edges"],
                ),
            ),
            "u64_dirty": _benchmark_prune(
                case,
                lambda payload: prune_bitset_dirty_u64(
                    payload["candidates"],
                    payload["pattern_edges"],
                    payload["molecule_edges"],
                ),
            ),
            "u64_hybrid": _benchmark_prune(
                case,
                lambda payload: prune_bitset_hybrid_u64(
                    payload["candidates"],
                    payload["pattern_edges"],
                    payload["molecule_edges"],
                ),
            ),
            "u64_popcount": _benchmark_prune(
                case,
                lambda payload: prune_bitset_popcount_u64(
                    payload["candidates"],
                    payload["pattern_edges"],
                    payload["molecule_edges"],
                ),
            ),
            "u64_popcount_block16": _benchmark_prune(
                case,
                lambda payload: prune_bitset_popcount_block16_u64(
                    payload["candidates"],
                    payload["pattern_edges"],
                    payload["molecule_edges"],
                ),
            ),
            "u64_popcount_block32": _benchmark_prune(
                case,
                lambda payload: prune_bitset_popcount_block32_u64(
                    payload["candidates"],
                    payload["pattern_edges"],
                    payload["molecule_edges"],
                ),
            ),
            "u64_popcount_block64": _benchmark_prune(
                case,
                lambda payload: prune_bitset_popcount_block64_u64(
                    payload["candidates"],
                    payload["pattern_edges"],
                    payload["molecule_edges"],
                ),
            ),
            "u64_vectorized": _benchmark_prune(
                case,
                lambda payload: prune_bitset_vectorized_u64(
                    payload["candidates"],
                    payload["pattern_edges"],
                    payload["molecule_edges"],
                ),
            ),
            "u64_signature": _benchmark_prune(
                case,
                lambda payload: prune_signature_bitset_u64(
                    payload["query"],
                    payload["molecule"],
                    payload["candidates"],
                    payload["pattern_edges"],
                    payload["molecule_edges"],
                ),
                expect_exact=False,
            ),
            "u64_signature_hybrid": _benchmark_prune(
                case,
                lambda payload: prune_signature_hybrid_u64(
                    payload["query"],
                    payload["molecule"],
                    payload["candidates"],
                    payload["pattern_edges"],
                    payload["molecule_edges"],
                ),
                expect_exact=False,
            ),
            "u64_signature_vectorized": _benchmark_prune(
                case,
                lambda payload: prune_signature_vectorized_u64(
                    payload["query"],
                    payload["molecule"],
                    payload["candidates"],
                    payload["pattern_edges"],
                    payload["molecule_edges"],
                ),
                expect_exact=False,
            ),
        }
        if case["candidates"].shape[0] <= 32 and case["candidates"].shape[1] <= 32:
            report["u32_pack_ms"] = round(_benchmark_pack(case, pack_rows_u32), 3)
            report["u32"] = _benchmark_prune(
                case,
                lambda payload: prune_bitset_u32(
                    payload["candidates"],
                    payload["pattern_edges"],
                    payload["molecule_edges"],
                ),
            )
            report["u32_dirty"] = _benchmark_prune(
                case,
                lambda payload: prune_bitset_dirty_u32(
                    payload["candidates"],
                    payload["pattern_edges"],
                    payload["molecule_edges"],
                ),
            )
            report["u32_hybrid"] = _benchmark_prune(
                case,
                lambda payload: prune_bitset_hybrid_u32(
                    payload["candidates"],
                    payload["pattern_edges"],
                    payload["molecule_edges"],
                ),
            )
            report["u32_vectorized"] = _benchmark_prune(
                case,
                lambda payload: prune_bitset_vectorized_u32(
                    payload["candidates"],
                    payload["pattern_edges"],
                    payload["molecule_edges"],
                ),
            )
            report["u32_signature"] = _benchmark_prune(
                case,
                lambda payload: prune_signature_bitset_u32(
                    payload["query"],
                    payload["molecule"],
                    payload["candidates"],
                    payload["pattern_edges"],
                    payload["molecule_edges"],
                ),
                expect_exact=False,
            )
            report["u32_signature_hybrid"] = _benchmark_prune(
                case,
                lambda payload: prune_signature_hybrid_u32(
                    payload["query"],
                    payload["molecule"],
                    payload["candidates"],
                    payload["pattern_edges"],
                    payload["molecule_edges"],
                ),
                expect_exact=False,
            )
            report["u32_signature_vectorized"] = _benchmark_prune(
                case,
                lambda payload: prune_signature_vectorized_u32(
                    payload["query"],
                    payload["molecule"],
                    payload["candidates"],
                    payload["pattern_edges"],
                    payload["molecule_edges"],
                ),
                expect_exact=False,
            )
        print(report)

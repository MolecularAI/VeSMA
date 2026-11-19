from __future__ import annotations

import tarfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tests.datadirs import external_data_root
from vesma import (
    SmartsError,
    cmol_from_smiles,
    find_matches,
    gmol_from_cmol,
    gmol_from_smarts,
)
from vesma.adjacency import bond_presence
from vesma.cpu_matcher import enumerate_assignments, initialize_candidates

from .prune_kernel_prototypes import (
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
RNG_SEED = 20260317
PATTERN_COUNT = 100
MOLECULE_COUNT = 100
SLOW_NO_HIT_COUNT = 10
POSITIVE_HIT_COUNT = 10
CONTROL_REPEATS = 50


def _iter_pains_patterns(path: Path):
    with tarfile.open(path, "r:gz") as archive:
        member = archive.getmember("pains.smarts")
        payload = archive.extractfile(member)
        text = payload.read().decode("utf-8", errors="ignore") if payload else ""

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            line = line.split("\t", 1)[0].strip()
        elif " <" in line:
            line = line.split(" <", 1)[0].strip()
        if line:
            yield line


def _iter_smiles(path: Path):
    with zipfile.ZipFile(path) as archive:
        payload = archive.read(SMILES_MEMBER).decode("utf-8", errors="ignore")
    for line in payload.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        yield stripped.split()[0]


def _sample_corpus():
    rng = np.random.default_rng(RNG_SEED)

    patterns_all = []
    for smarts in _iter_pains_patterns(DATASET_DIR / "pains-smarts.tar.gz"):
        try:
            query = gmol_from_smarts(smarts)[0]
        except SmartsError:
            continue
        if query.atom_count == 0 or query.atom_count > 16:
            continue
        patterns_all.append((smarts, query))

    molecules_all = []
    for smiles in _iter_smiles(DATASET_DIR / "smarts-benchmark-v2.0.zip"):
        try:
            molecule = gmol_from_cmol(cmol_from_smiles(smiles)[0])
        except SmartsError:
            continue
        if molecule.atom_count == 0 or molecule.atom_count > 40:
            continue
        molecules_all.append((smiles, molecule))
        if len(molecules_all) >= 5_000:
            break

    pattern_idx = rng.choice(len(patterns_all), size=PATTERN_COUNT, replace=False)
    molecule_idx = rng.choice(len(molecules_all), size=MOLECULE_COUNT, replace=False)
    patterns = [patterns_all[int(i)] for i in pattern_idx]
    molecules = [molecules_all[int(i)] for i in molecule_idx]
    return patterns, molecules


def _rank_patterns(patterns, molecules):
    ranked = []
    for smarts, query in patterns:
        total_ns = 0
        hits = 0
        for _, molecule in molecules:
            started = time.perf_counter_ns()
            result = find_matches(query, molecule, return_first=True)
            total_ns += time.perf_counter_ns() - started
            hits += int(result.status == "match")
        ranked.append(
            {
                "smarts": smarts,
                "query": query,
                "vesma_ms": total_ns / 1_000_000,
                "hits": hits,
            }
        )

    slow_no_hit = sorted(
        [row for row in ranked if row["hits"] == 0],
        key=lambda row: row["vesma_ms"],
        reverse=True,
    )[:SLOW_NO_HIT_COUNT]
    positive_hit = sorted(
        [row for row in ranked if row["hits"] > 0],
        key=lambda row: (row["hits"], row["vesma_ms"]),
        reverse=True,
    )[:POSITIVE_HIT_COUNT]
    return slow_no_hit, positive_hit


def _prepared_pairs(pattern_rows, molecules):
    pairs = []
    for row in pattern_rows:
        query = row["query"]
        pattern_edges = bond_presence(query.bonds)
        for smiles, molecule in molecules:
            candidates = initialize_candidates(query, molecule)
            pairs.append(
                {
                    "smarts": row["smarts"],
                    "smiles": smiles,
                    "query": query,
                    "molecule": molecule,
                    "candidates": candidates,
                    "pattern_edges": pattern_edges,
                    "molecule_edges": bond_presence(molecule.bonds),
                }
            )
    return pairs


def _control_pairs():
    controls = []
    for smarts, smiles in [
        ("[C:1][C:2]", "CCC"),
        ("C(C)C", "CC(C)C"),
        ("C1CCCCC1", "C1CCCCC1"),
        ("C1CC1", "C1CCC1"),
    ]:
        query = gmol_from_smarts(smarts)[0]
        molecule = gmol_from_cmol(cmol_from_smiles(smiles)[0])
        controls.append(
            {
                "smarts": smarts,
                "smiles": smiles,
                "query": query,
                "molecule": molecule,
                "candidates": initialize_candidates(query, molecule),
                "pattern_edges": bond_presence(query.bonds),
                "molecule_edges": bond_presence(molecule.bonds),
            }
        )
    return controls * CONTROL_REPEATS


def _benchmark_group(name: str, pairs) -> None:
    prototype_specs = [
        (
            "dense_gemm",
            lambda pair: prune_dense_gemm(
                pair["candidates"],
                pair["pattern_edges"],
                pair["molecule_edges"],
            ),
            lambda pair: True,
        ),
        (
            "u64",
            lambda pair: prune_bitset_u64(
                pair["candidates"],
                pair["pattern_edges"],
                pair["molecule_edges"],
            ),
            lambda pair: (
                pair["query"].atom_count <= 64 and pair["molecule"].atom_count <= 64
            ),
        ),
        (
            "u64_dirty",
            lambda pair: prune_bitset_dirty_u64(
                pair["candidates"],
                pair["pattern_edges"],
                pair["molecule_edges"],
            ),
            lambda pair: (
                pair["query"].atom_count <= 64 and pair["molecule"].atom_count <= 64
            ),
        ),
        (
            "u64_hybrid",
            lambda pair: prune_bitset_hybrid_u64(
                pair["candidates"],
                pair["pattern_edges"],
                pair["molecule_edges"],
            ),
            lambda pair: (
                pair["query"].atom_count <= 64 and pair["molecule"].atom_count <= 64
            ),
        ),
        (
            "u64_popcount",
            lambda pair: prune_bitset_popcount_u64(
                pair["candidates"],
                pair["pattern_edges"],
                pair["molecule_edges"],
            ),
            lambda pair: (
                pair["query"].atom_count <= 64 and pair["molecule"].atom_count <= 64
            ),
        ),
        (
            "u64_popcount_block16",
            lambda pair: prune_bitset_popcount_block16_u64(
                pair["candidates"],
                pair["pattern_edges"],
                pair["molecule_edges"],
            ),
            lambda pair: (
                pair["query"].atom_count <= 64 and pair["molecule"].atom_count <= 64
            ),
        ),
        (
            "u64_popcount_block32",
            lambda pair: prune_bitset_popcount_block32_u64(
                pair["candidates"],
                pair["pattern_edges"],
                pair["molecule_edges"],
            ),
            lambda pair: (
                pair["query"].atom_count <= 64 and pair["molecule"].atom_count <= 64
            ),
        ),
        (
            "u64_popcount_block64",
            lambda pair: prune_bitset_popcount_block64_u64(
                pair["candidates"],
                pair["pattern_edges"],
                pair["molecule_edges"],
            ),
            lambda pair: (
                pair["query"].atom_count <= 64 and pair["molecule"].atom_count <= 64
            ),
        ),
        (
            "u64_vectorized",
            lambda pair: prune_bitset_vectorized_u64(
                pair["candidates"],
                pair["pattern_edges"],
                pair["molecule_edges"],
            ),
            lambda pair: (
                pair["query"].atom_count <= 64 and pair["molecule"].atom_count <= 64
            ),
        ),
        (
            "u64_signature",
            lambda pair: prune_signature_bitset_u64(
                pair["query"],
                pair["molecule"],
                pair["candidates"],
                pair["pattern_edges"],
                pair["molecule_edges"],
            ),
            lambda pair: (
                pair["query"].atom_count <= 64 and pair["molecule"].atom_count <= 64
            ),
        ),
        (
            "u64_signature_hybrid",
            lambda pair: prune_signature_hybrid_u64(
                pair["query"],
                pair["molecule"],
                pair["candidates"],
                pair["pattern_edges"],
                pair["molecule_edges"],
            ),
            lambda pair: (
                pair["query"].atom_count <= 64 and pair["molecule"].atom_count <= 64
            ),
        ),
        (
            "u64_signature_vectorized",
            lambda pair: prune_signature_vectorized_u64(
                pair["query"],
                pair["molecule"],
                pair["candidates"],
                pair["pattern_edges"],
                pair["molecule_edges"],
            ),
            lambda pair: (
                pair["query"].atom_count <= 64 and pair["molecule"].atom_count <= 64
            ),
        ),
        (
            "u32",
            lambda pair: prune_bitset_u32(
                pair["candidates"],
                pair["pattern_edges"],
                pair["molecule_edges"],
            ),
            lambda pair: (
                pair["query"].atom_count <= 32 and pair["molecule"].atom_count <= 32
            ),
        ),
        (
            "u32_dirty",
            lambda pair: prune_bitset_dirty_u32(
                pair["candidates"],
                pair["pattern_edges"],
                pair["molecule_edges"],
            ),
            lambda pair: (
                pair["query"].atom_count <= 32 and pair["molecule"].atom_count <= 32
            ),
        ),
        (
            "u32_hybrid",
            lambda pair: prune_bitset_hybrid_u32(
                pair["candidates"],
                pair["pattern_edges"],
                pair["molecule_edges"],
            ),
            lambda pair: (
                pair["query"].atom_count <= 32 and pair["molecule"].atom_count <= 32
            ),
        ),
        (
            "u32_vectorized",
            lambda pair: prune_bitset_vectorized_u32(
                pair["candidates"],
                pair["pattern_edges"],
                pair["molecule_edges"],
            ),
            lambda pair: (
                pair["query"].atom_count <= 32 and pair["molecule"].atom_count <= 32
            ),
        ),
        (
            "u32_signature",
            lambda pair: prune_signature_bitset_u32(
                pair["query"],
                pair["molecule"],
                pair["candidates"],
                pair["pattern_edges"],
                pair["molecule_edges"],
            ),
            lambda pair: (
                pair["query"].atom_count <= 32 and pair["molecule"].atom_count <= 32
            ),
        ),
        (
            "u32_signature_hybrid",
            lambda pair: prune_signature_hybrid_u32(
                pair["query"],
                pair["molecule"],
                pair["candidates"],
                pair["pattern_edges"],
                pair["molecule_edges"],
            ),
            lambda pair: (
                pair["query"].atom_count <= 32 and pair["molecule"].atom_count <= 32
            ),
        ),
        (
            "u32_signature_vectorized",
            lambda pair: prune_signature_vectorized_u32(
                pair["query"],
                pair["molecule"],
                pair["candidates"],
                pair["pattern_edges"],
                pair["molecule_edges"],
            ),
            lambda pair: (
                pair["query"].atom_count <= 32 and pair["molecule"].atom_count <= 32
            ),
        ),
    ]
    aggregates = {
        spec[0]: {
            "pairs": 0,
            "dense_ms": 0.0,
            "proto_ms": 0.0,
            "matrix_agreements": 0,
            "status_agreements": 0,
            "before_true": 0,
            "after_true": 0,
            "rounds": 0,
        }
        for spec in prototype_specs
    }

    for pair in pairs:
        started = time.perf_counter_ns()
        dense = prune_dense_reference(
            pair["candidates"],
            pair["pattern_edges"],
            pair["molecule_edges"],
        )
        dense_ms = (time.perf_counter_ns() - started) / 1_000_000
        dense_status = enumerate_assignments(
            dense[0],
            pair["query"].bonds,
            pair["molecule"].bonds,
            pattern_aromaticity=pair["query"].atom_masks.A,
            molecule_aromaticity=pair["molecule"].atom_masks.A,
            return_first=True,
        ).status
        before_true = int(pair["candidates"].sum())

        for proto_name, proto_fn, applicable in prototype_specs:
            if not applicable(pair):
                continue
            started = time.perf_counter_ns()
            observed = proto_fn(pair)
            proto_ms = (time.perf_counter_ns() - started) / 1_000_000
            observed_status = enumerate_assignments(
                observed[0],
                pair["query"].bonds,
                pair["molecule"].bonds,
                pattern_aromaticity=pair["query"].atom_masks.A,
                molecule_aromaticity=pair["molecule"].atom_masks.A,
                return_first=True,
            ).status

            aggregates[proto_name]["pairs"] += 1
            aggregates[proto_name]["dense_ms"] += dense_ms
            aggregates[proto_name]["proto_ms"] += proto_ms
            aggregates[proto_name]["matrix_agreements"] += int(
                observed[1:] == dense[1:] and np.array_equal(observed[0], dense[0])
            )
            aggregates[proto_name]["status_agreements"] += int(
                observed_status == dense_status
            )
            aggregates[proto_name]["before_true"] += before_true
            aggregates[proto_name]["after_true"] += int(observed[0].sum())
            aggregates[proto_name]["rounds"] += observed[1]

    print(f"group={name}")
    for proto_name, stats in aggregates.items():
        if stats["pairs"] == 0:
            continue
        print(
            {
                "prototype": proto_name,
                "pairs": stats["pairs"],
                "matrix_agreement": round(
                    stats["matrix_agreements"] / stats["pairs"], 3
                ),
                "status_agreement": round(
                    stats["status_agreements"] / stats["pairs"], 3
                ),
                "dense_ms": round(stats["dense_ms"], 3),
                "proto_ms": round(stats["proto_ms"], 3),
                "speedup_vs_dense": round(stats["dense_ms"] / stats["proto_ms"], 3),
                "avg_before_true": round(stats["before_true"] / stats["pairs"], 2),
                "avg_after_true": round(stats["after_true"] / stats["pairs"], 2),
                "avg_rounds": round(stats["rounds"] / stats["pairs"], 2),
            }
        )


@pytest.mark.perf
def test_perf_prune_kernels_compare() -> None:
    patterns, molecules = _sample_corpus()
    slow_no_hit, positive_hit = _rank_patterns(patterns, molecules)

    print("slow_no_hit_patterns:")
    for row in slow_no_hit:
        print(
            {
                "smarts": row["smarts"],
                "atoms": row["query"].atom_count,
                "vesma_ms": round(row["vesma_ms"], 3),
            }
        )

    print("positive_hit_patterns:")
    for row in positive_hit:
        print(
            {
                "smarts": row["smarts"],
                "atoms": row["query"].atom_count,
                "hits": row["hits"],
                "vesma_ms": round(row["vesma_ms"], 3),
            }
        )

    _benchmark_group("slow_no_hit", _prepared_pairs(slow_no_hit, molecules))
    _benchmark_group("positive_hit", _prepared_pairs(positive_hit, molecules))
    _benchmark_group("control", _control_pairs())

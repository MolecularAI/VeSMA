from __future__ import annotations

import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any

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
from vesma.cpu_matcher import enumerate_assignments, initialize_candidates, prune

SMILES_MEMBER = "v2.0/zinc_lead-like_2011-02-12_first100k.smi"
DATASET_DIR = external_data_root() / "external" / "zbh-smarts" / "raw"
RNG_SEED = 20260317
PATTERN_COUNT = 100
MOLECULE_COUNT = 100
TOP_PATTERN_COUNT = 10
STAGE_BREAKDOWN_COUNT = 5
SWEEP_PATTERN_COUNT = 10
PRUNE_SWEEP_VALUES = [1, 2, 4, 8, None]


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


def _pattern_stats(
    query,
    molecules: list[tuple[str, Any, Any]],
) -> dict[str, float | int]:
    pattern_edges = bond_presence(query.bonds)
    init_true_counts: list[int] = []
    pruned_true_counts: list[int] = []
    root_rounds: list[int] = []
    empty_after_root_prune = 0
    total_prunings = 0
    total_branches = 0
    total_leaves = 0

    for _, _, vesma_mol in molecules:
        molecule_edges = bond_presence(vesma_mol.bonds)
        candidates = initialize_candidates(query, vesma_mol)
        init_true_counts.append(int(candidates.sum()))
        pruned, prune_rounds, _ = prune(candidates, pattern_edges, molecule_edges)
        pruned_true_counts.append(int(pruned.sum()))
        root_rounds.append(prune_rounds)
        empty_after_root_prune += int((pruned.sum(axis=1) == 0).any())

        result = enumerate_assignments(
            pruned,
            query.bonds,
            vesma_mol.bonds,
            pattern_aromaticity=query.atom_masks.A,
            molecule_aromaticity=vesma_mol.atom_masks.A,
            return_first=True,
        )
        total_prunings += result.prunings + prune_rounds
        total_branches += result.branches_explored
        total_leaves += result.leaves_explored

    return {
        "avg_init_true": float(np.mean(init_true_counts)),
        "avg_pruned_true": float(np.mean(pruned_true_counts)),
        "empty_after_root_prune": empty_after_root_prune,
        "avg_root_prune_rounds": float(np.mean(root_rounds)),
        "max_root_prune_rounds": int(np.max(root_rounds)),
        "total_prunings": total_prunings,
        "total_branches": total_branches,
        "total_leaves": total_leaves,
    }


def _prune_sweep(
    patterns: list[dict[str, object]],
    molecules: list[tuple[str, Any, Any]],
) -> None:
    baseline: dict[tuple[str, str], str] = {}
    for row in patterns:
        query = row["vesma_query"]
        smarts = str(row["smarts"])
        for smiles, _, vesma_mol in molecules:
            result = find_matches(query, vesma_mol, return_first=True)
            baseline[(smarts, smiles)] = result.status

    print("prune_round_sweep:")
    for max_prune_rounds in PRUNE_SWEEP_VALUES:
        started = time.perf_counter_ns()
        statuses = {"match": 0, "no_match": 0, "unknown": 0}
        agreements = 0
        total_pairs = 0

        for row in patterns:
            query = row["vesma_query"]
            smarts = str(row["smarts"])
            for smiles, _, vesma_mol in molecules:
                result = find_matches(
                    query,
                    vesma_mol,
                    return_first=True,
                    max_prune_rounds=max_prune_rounds,
                )
                statuses[result.status] += 1
                agreements += int(result.status == baseline[(smarts, smiles)])
                total_pairs += 1

        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        label = "unbounded" if max_prune_rounds is None else str(max_prune_rounds)
        print(
            {
                "max_prune_rounds": label,
                "vesma_ms": round(elapsed_ms, 3),
                "statuses": statuses,
                "agreement_with_unbounded": round(agreements / total_pairs, 3),
            }
        )


@pytest.mark.perf
def test_perf_matcher_vs_rdkit() -> None:
    chem = pytest.importorskip("rdkit.Chem")
    rng = np.random.default_rng(RNG_SEED)

    patterns_all = []
    for smarts in _iter_pains_patterns(DATASET_DIR / "pains-smarts.tar.gz"):
        rd_query = chem.MolFromSmarts(smarts)
        if (
            rd_query is None
            or rd_query.GetNumAtoms() == 0
            or rd_query.GetNumAtoms() > 16
        ):
            continue
        try:
            vesma_query = gmol_from_smarts(smarts)[0]
        except SmartsError:
            continue
        patterns_all.append((smarts, rd_query, vesma_query))

    pattern_idx = rng.choice(len(patterns_all), size=PATTERN_COUNT, replace=False)
    patterns = [patterns_all[int(i)] for i in pattern_idx]

    molecules_all = []
    for smiles in _iter_smiles(DATASET_DIR / "smarts-benchmark-v2.0.zip"):
        rd_mol = chem.MolFromSmiles(smiles)
        if rd_mol is None or rd_mol.GetNumAtoms() == 0 or rd_mol.GetNumAtoms() > 40:
            continue
        try:
            vesma_mol = gmol_from_cmol(cmol_from_smiles(smiles)[0])
        except SmartsError:
            continue
        molecules_all.append((smiles, rd_mol, vesma_mol))
        if len(molecules_all) >= 5_000:
            break

    molecule_idx = rng.choice(len(molecules_all), size=MOLECULE_COUNT, replace=False)
    molecules = [molecules_all[int(i)] for i in molecule_idx]

    per_pattern = []
    for smarts, rd_query, vesma_query in patterns:
        rdkit_ns = 0
        vesma_ns = 0
        rdkit_hits = 0
        vesma_hits = 0

        for _, rd_mol, vesma_mol in molecules:
            started = time.perf_counter_ns()
            rd_hit = rd_mol.HasSubstructMatch(rd_query)
            rdkit_ns += time.perf_counter_ns() - started

            started = time.perf_counter_ns()
            vesma_hit = (
                find_matches(vesma_query, vesma_mol, return_first=True).status
                == "match"
            )
            vesma_ns += time.perf_counter_ns() - started

            rdkit_hits += int(rd_hit)
            vesma_hits += int(vesma_hit)

        per_pattern.append(
            {
                "smarts": smarts,
                "atoms": vesma_query.atom_count,
                "rdkit_ms": rdkit_ns / 1_000_000,
                "vesma_ms": vesma_ns / 1_000_000,
                "ratio": (vesma_ns / rdkit_ns) if rdkit_ns else float("inf"),
                "rdkit_hits": rdkit_hits,
                "vesma_hits": vesma_hits,
                "vesma_query": vesma_query,
            }
        )

    per_pattern.sort(key=lambda row: row["ratio"], reverse=True)
    total_rdkit_ms = sum(float(row["rdkit_ms"]) for row in per_pattern)
    total_vesma_ms = sum(float(row["vesma_ms"]) for row in per_pattern)
    total_pairs = PATTERN_COUNT * MOLECULE_COUNT

    print(f"pairs={total_pairs}")
    print(f"rdkit_total_ms={total_rdkit_ms:.3f}")
    print(f"vesma_total_ms={total_vesma_ms:.3f}")
    print(f"vesma_vs_rdkit_ratio={total_vesma_ms / total_rdkit_ms:.3f}x")
    print("top_patterns_vs_rdkit:")
    for row in per_pattern[:TOP_PATTERN_COUNT]:
        print(
            {
                "smarts": row["smarts"],
                "atoms": row["atoms"],
                "rdkit_ms": round(float(row["rdkit_ms"]), 3),
                "vesma_ms": round(float(row["vesma_ms"]), 3),
                "ratio": round(float(row["ratio"]), 3),
                "rdkit_hits": row["rdkit_hits"],
                "vesma_hits": row["vesma_hits"],
            }
        )

    print("stage_breakdown:")
    for row in per_pattern[:STAGE_BREAKDOWN_COUNT]:
        query = row["vesma_query"]
        pattern_edges = bond_presence(query.bonds)
        init_ns = 0
        prune_ns = 0
        search_ns = 0

        for _, _, vesma_mol in molecules:
            molecule_edges = bond_presence(vesma_mol.bonds)

            started = time.perf_counter_ns()
            candidates = initialize_candidates(query, vesma_mol)
            init_ns += time.perf_counter_ns() - started

            started = time.perf_counter_ns()
            pruned, _, _ = prune(candidates, pattern_edges, molecule_edges)
            prune_ns += time.perf_counter_ns() - started

            started = time.perf_counter_ns()
            enumerate_assignments(
                pruned,
                query.bonds,
                vesma_mol.bonds,
                pattern_aromaticity=query.atom_masks.A,
                molecule_aromaticity=vesma_mol.atom_masks.A,
                return_first=True,
            )
            search_ns += time.perf_counter_ns() - started

        print(
            {
                "smarts": row["smarts"],
                "init_ms": round(init_ns / 1_000_000, 3),
                "prune_ms": round(prune_ns / 1_000_000, 3),
                "search_ms": round(search_ns / 1_000_000, 3),
            }
        )

    print("search_shape:")
    for row in per_pattern[:STAGE_BREAKDOWN_COUNT]:
        stats = _pattern_stats(row["vesma_query"], molecules)
        print(
            {
                "smarts": row["smarts"],
                "avg_init_true": round(stats["avg_init_true"], 2),
                "avg_pruned_true": round(stats["avg_pruned_true"], 2),
                "empty_after_root_prune": stats["empty_after_root_prune"],
                "avg_root_prune_rounds": round(stats["avg_root_prune_rounds"], 2),
                "max_root_prune_rounds": stats["max_root_prune_rounds"],
                "total_prunings": stats["total_prunings"],
                "total_branches": stats["total_branches"],
                "total_leaves": stats["total_leaves"],
            }
        )

    _prune_sweep(per_pattern[:SWEEP_PATTERN_COUNT], molecules)

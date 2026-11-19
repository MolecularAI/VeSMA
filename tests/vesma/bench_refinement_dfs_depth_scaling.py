"""CUDA subgraph-isomorphism refinement DFS depth sweep.

Sweeps the fused matcher's ``max_prefix_depth`` (the depth of the Ullmann-style
refinement/DFS) and reports per-depth timing and prune quality. Named to distinguish it
from the molecule-count *scaling* benchmarks (match-only / match-plus-split).
"""

import gzip
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.datadirs import external_data_root
from tests.vertsy.paper_provenance import source_artifact


@pytest.mark.perf
@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.paper_benchmark
def test_bench_refinement_dfs_depth_scaling(benchmark: Any) -> None:
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CuPy cannot access CUDA on this machine.")

    from rdkit import Chem

    from tests.datasets import load_compatible_templates
    from vesma.cuda_matcher import CudaReactionMatcher, cuda_stack_mols
    from vesma.cuda_prune import (
        match_fused_cuda,
        prepare_cuda_prune_batch_from_device,
    )
    from vesma.interop import cmol_from_mol

    print("Loading templates...")
    template_limit = int(os.environ.get("BENCH_DEPTH_RULE_LIMIT", "0"))
    templates = load_compatible_templates()
    if template_limit > 0:
        templates = templates[:template_limit]
    rules = tuple(
        template.pattern
        for template in templates
        if template.pattern.lhs.atom_count <= 64
        and template.pattern.rhs.atom_count <= 64
    )
    if not rules:
        raise ValueError("Depth scaling benchmark has no compatible rules.")

    print("Building matcher...")
    # Initialize with depth 2 to generate precursors exactly as earlier tests might have
    matcher = CudaReactionMatcher(rules, max_leaf_checks=4096, max_prefix_depth=2)

    print("Loading targets...")
    target_limit = int(os.environ.get("BENCH_DEPTH_TARGET_LIMIT", "10"))
    targets_path = (
        external_data_root()
        / "external"
        / "paroutes"
        / "processed"
        / "n1-d2-targets.smi.gz"
    )
    with gzip.open(targets_path, "rt", encoding="utf-8") as handle:
        target_smiles = [line.strip().split()[0] for line in handle if line.strip()]
    if target_limit > 0:
        target_smiles = target_smiles[:target_limit]
    targets = [cmol_from_mol(Chem.MolFromSmiles(smiles)) for smiles in target_smiles]

    print("Running Layer 0 match to generate Layer 1 precursors...")
    precursors, hits = matcher.match(targets)

    print(f"Generated {len(precursors)} unique precursors (Layer 1 frontier).")

    # Combine targets and precursors
    # Note: precursors is a CudaMoleculeBatch. We can extract it back to host and combine
    # Actually, simpler to combine SMARTS strings? No, we don't have precursor SMARTS natively here.
    # We can just stack them as CudaMoleculeBatch
    from vesma.cuda_matcher import CudaMoleculeHostBatch, CudaMoleculeHostPart

    target_mols = cuda_stack_mols(targets)

    # Convert both to CudaMoleculeHostPart to combine them
    keys_targets = [(0, 0)] * len(target_mols)
    keys_precursors = [(0, 0)] * len(precursors)
    host_targets = CudaMoleculeHostPart.from_cuda(target_mols, keys_targets)
    host_precursors = CudaMoleculeHostPart.from_cuda(precursors, keys_precursors)

    combined_host_batch = CudaMoleculeHostBatch((host_targets, host_precursors))
    combined_cuda_batch = combined_host_batch.to_cuda()

    total_mols = len(combined_cuda_batch)
    print(
        f"Testing {total_mols} molecules against {len(rules)} templates -> {total_mols * len(rules):,} pairs."
    )

    chunk = matcher._rule_chunks[0]

    # We chunk by molecules to avoid OOM
    batch_size = 2000

    # Warmup
    print("Warming up kernel...")
    sub_indices = np.arange(10, dtype=np.int32)
    sub_mols = combined_host_batch.select_physical(sub_indices).to_cuda()
    pair_pattern_index = np.repeat(
        np.arange(chunk.pattern_count, dtype=np.int32), len(sub_mols)
    )
    pair_molecule_index = np.tile(
        np.arange(len(sub_mols), dtype=np.int32), chunk.pattern_count
    )
    warmup_batch = prepare_cuda_prune_batch_from_device(
        chunk.pattern_count,
        chunk.pattern_atom_counts_device,
        chunk.pattern_atoms_u32_device,
        chunk.pattern_rows_u64_device,
        chunk.pattern_bonds_u8_device,
        pattern_groups_u32_device=chunk.pattern_groups_u32_device,
        molecule_atom_counts_device=sub_mols.atom_counts_device,
        molecule_atoms_u32=sub_mols.atoms_device,
        molecule_rows_u64=sub_mols.rows_device,
        molecule_bonds_u8=sub_mols.bonds_device,
        pair_pattern_index=pair_pattern_index,
        pair_molecule_index=pair_molecule_index,
    )
    match_fused_cuda(
        warmup_batch,
        threads_per_block=32,
        max_prune_rounds=64,
        max_leaf_checks=4096,
        max_prefix_depth=4,
    )
    cp.cuda.Stream.null.synchronize()

    depths_to_test = [0, 2, 4, 8, 16]

    def run_depth_sweep() -> list[dict]:
        results: list[dict] = []
        for depth in depths_to_test:
            print(f"\n--- Testing max_prefix_depth = {depth} ---")

            total_time_ms = 0.0
            total_valid = 0
            total_unknowns = 0
            max_leaf = 0
            max_branch = 0

            for i in range(0, total_mols, batch_size):
                end_idx = min(i + batch_size, total_mols)
                sub_indices = np.arange(i, end_idx, dtype=np.int32)
                sub_mols = combined_host_batch.select_physical(sub_indices).to_cuda()
                sub_count = len(sub_mols)

                p_pattern = np.repeat(
                    np.arange(chunk.pattern_count, dtype=np.int32), sub_count
                )
                p_molecule = np.tile(
                    np.arange(sub_count, dtype=np.int32), chunk.pattern_count
                )

                c_batch = prepare_cuda_prune_batch_from_device(
                    chunk.pattern_count,
                    chunk.pattern_atom_counts_device,
                    chunk.pattern_atoms_u32_device,
                    chunk.pattern_rows_u64_device,
                    chunk.pattern_bonds_u8_device,
                    molecule_atom_counts_device=sub_mols.atom_counts_device,
                    molecule_atoms_u32=sub_mols.atoms_device,
                    molecule_rows_u64=sub_mols.rows_device,
                    molecule_bonds_u8=sub_mols.bonds_device,
                    pair_pattern_index=p_pattern,
                    pair_molecule_index=p_molecule,
                )

                start_event = cp.cuda.Event()
                end_event = cp.cuda.Event()

                start_event.record()
                result = match_fused_cuda(
                    c_batch,
                    threads_per_block=32,
                    max_prune_rounds=64,
                    max_leaf_checks=4096,
                    max_prefix_depth=depth,
                )
                end_event.record()
                end_event.synchronize()

                elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)
                total_time_ms += elapsed_ms

                status_arr = result.status_codes_device
                total_unknowns += int(cp.sum(status_arr == 2))
                total_valid += int(cp.sum(status_arr == 1))

                valid_mask = status_arr > 0
                if cp.any(valid_mask):
                    max_leaf = max(
                        max_leaf, int(cp.max(result.leaf_checks_device[valid_mask]))
                    )
                    max_branch = max(
                        max_branch,
                        int(cp.max(result.branches_explored_device[valid_mask])),
                    )

            print(f"Time: {total_time_ms:.2f} ms")
            print(f"Valid hits: {total_valid:,}")
            print(f"Unknowns (timeouts): {total_unknowns:,}")
            print(f"Max leaf checks per pair: {max_leaf}")
            print(f"Max branches explored per pair: {max_branch}")

            results.append(
                {
                    "max_prefix_depth": depth,
                    "total_pairs": total_mols * chunk.pattern_count,
                    "total_time_ms": total_time_ms,
                    "valid_hits": total_valid,
                    "unknown_timeouts": total_unknowns,
                    "max_leaf_checks": max_leaf,
                    "max_branches_explored": max_branch,
                }
            )
        return results

    results = benchmark.pedantic(
        run_depth_sweep, rounds=rounds(), warmup_rounds=warmup_rounds(), iterations=1
    )
    benchmark.extra_info.update(
        depth_extra_info(
            results,
            depths=depths_to_test,
            template_limit=template_limit,
            target_limit=target_limit,
            targets_loaded=len(target_smiles),
            rule_count=len(rules),
            total_mols=total_mols,
            pattern_count=chunk.pattern_count,
            batch_size=batch_size,
            targets_path=targets_path,
        )
    )


def depth_extra_info(
    results: list[dict],
    *,
    depths: list[int],
    template_limit: int,
    target_limit: int,
    targets_loaded: int,
    rule_count: int,
    total_mols: int,
    pattern_count: int,
    batch_size: int,
    targets_path: Path,
) -> dict[str, Any]:
    from tests.datasets import compatible_templates_path

    return {
        "experiment": "refinement-dfs-depth-scaling",
        "backend": "cuda",
        "template_source": "uspto42k-compat",
        "template_limit": template_limit,
        "target_limit": target_limit,
        "targets_loaded": targets_loaded,
        "rules": rule_count,
        "combined_molecules": total_mols,
        "pattern_count": pattern_count,
        "pairs": total_mols * pattern_count,
        "batch_size": batch_size,
        "depths": depths,
        "max_prune_rounds": 64,
        "max_leaf_checks": 4096,
        "threads_per_block": 32,
        "per_depth": results,
        "source_artifacts": {
            "targets": source_artifact(targets_path),
            "templates": source_artifact(compatible_templates_path()),
        },
    }


def rounds() -> int:
    value = int(os.environ.get("BENCH_ROUNDS", "1"))
    if value <= 0:
        raise ValueError("BENCH_ROUNDS must be positive.")
    return value


def warmup_rounds() -> int:
    value = int(os.environ.get("BENCH_WARMUP_ROUNDS", "0"))
    if value < 0:
        raise ValueError("BENCH_WARMUP_ROUNDS must be non-negative.")
    return value

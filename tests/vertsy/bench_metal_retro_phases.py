"""Replay identical Metal retro assignments through edits, components, and perception.

Run with ``make bench-metal-retro-phases METAL_PIPELINE_TARGETS_FILE=...``. A
count-only matcher pass finds positive pairs; a bounded assignment sample is
then captured once and replayed unchanged through each profiling mode.
"""

from __future__ import annotations

import ctypes
import os
from collections import Counter
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from vesma.adjacency import bond_kind_matrix
from vesma.metal_retro import RetroStatus, probe_retro_pipelines

if TYPE_CHECKING:
    from tests.vertsy.bench_metal_pipeline import ProfileSetup

pytest_plugins = ["tests.vertsy.bench_metal_pipeline"]

pytestmark = [
    pytest.mark.perf,
    pytest.mark.external_data,
    pytest.mark.gpu,
    pytest.mark.metal,
    pytest.mark.filterwarnings(
        "ignore:.*precursor component.* over .* atoms.*:UserWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:Setting the shape on a NumPy array has been deprecated.*:DeprecationWarning:joblib.numpy_pickle"
    ),
]


def status_rows(status: ctypes.Array[RetroStatus], count: int) -> np.ndarray:
    fields = ctypes.sizeof(RetroStatus) // 4
    return (
        np.frombuffer(status, dtype=np.uint32, count=count * fields)
        .reshape(count, fields)
        .copy()
    )


def assignment_precursor_hashes(
    produced: Any, count: int
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Each assignment's precursor identities, through the keyer the arm uses.

    This read `mols[index].hash_lo/.hash_hi` while the construct kernel filled hash fields on an
    interleaved `GpuMol`. It keys nothing now, so the identities come from `metal_wl_hash` over the
    arrays it wrote -- one dispatch for the whole batch rather than a field read per row.
    """
    from vesma.metal_retro import metal_wl_hash

    if not len(produced.atom_counts):
        return tuple(() for _ in produced.status[:count])
    keyed = metal_wl_hash(produced.atoms, produced.bonds, produced.atom_counts)
    tokens = list(zip(keyed.hash_lo.tolist(), keyed.hash_hi.tolist(), strict=True))
    return tuple(
        tuple(
            sorted(
                tokens[index]
                for index in range(
                    row.output_base, row.output_base + row.component_count
                )
            )
        )
        for row in produced.status[:count]
    )


def distribution(values: np.ndarray) -> dict[str, float]:
    if not values.size:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


@pytest.mark.parametrize("frontier_size", [1, 8, 64])
def test_bench_metal_retro_phases(
    benchmark: Any, profile_setup: ProfileSetup, frontier_size: int
) -> None:
    frontier = profile_setup.source[:frontier_size]
    # Handed over on the arm rather than recovered from `arm.match.__self__`, which only worked
    # while `match` was a bound method on the matcher instead of a composed function.
    wrapper = profile_setup.arm.matcher
    inner = wrapper._inner
    constructor = inner._constructor
    construct = constructor.construct_packed
    captured: list[
        tuple[dict[str, Any], np.ndarray, tuple[tuple[tuple[int, int], ...], ...]]
    ] = []
    baseline_gpu_seconds = 0.0
    candidates = wrapper._prefilter_pairs(frontier)
    match_pairs = np.ascontiguousarray(candidates[:, ::-1])
    m_atoms = frontier.atoms.reshape(-1)
    m_rows = frontier.adjacency_rows().reshape(-1)
    m_bonds = frontier.bonds.reshape(-1)
    m_counts = np.ascontiguousarray(frontier.atom_counts, dtype=np.uint32)
    positive_parts = []
    for start in range(0, len(match_pairs), 512):
        chunk = match_pairs[start : start + 512]
        statuses, _ = inner._matcher.dispatch_match_packed(
            p_atoms=inner._p_atoms,
            p_rows=inner._p_rows,
            p_bonds=inner._p_bonds,
            p_groups=inner._p_groups,
            p_preds=inner._p_preds,
            p_counts=inner._p_counts,
            m_atoms=m_atoms,
            m_rows=m_rows,
            m_bonds=m_bonds,
            m_counts=m_counts,
            pair_indices=chunk,
            max_assignments=1,
            collect_assignments=False,
            pattern_token=inner._reaction_token,
        )
        positive_parts.append(chunk[statuses[:, 0] > 0])
    positive = np.concatenate(positive_parts) if positive_parts else match_pairs[:0]
    pair_limit = 128 * frontier_size
    if len(positive) > pair_limit:
        selected = np.random.default_rng(0).choice(
            len(positive), pair_limit, replace=False
        )
        positive = positive[np.sort(selected)]

    for start in range(0, len(positive), 128):
        pairs = positive[start : start + 128]
        statuses, assignments = inner._matcher.dispatch_match_packed(
            p_atoms=inner._p_atoms,
            p_rows=inner._p_rows,
            p_bonds=inner._p_bonds,
            p_groups=inner._p_groups,
            p_preds=inner._p_preds,
            p_counts=inner._p_counts,
            m_atoms=m_atoms,
            m_rows=m_rows,
            m_bonds=m_bonds,
            m_counts=m_counts,
            pair_indices=pairs,
            max_assignments=inner._assignment_probe_cap,
            copy_valid_only=True,
            pattern_token=inner._reaction_token,
        )
        counts = statuses[:, 0]
        valid = np.arange(inner._assignment_probe_cap)[None, :] < counts[:, None]
        packed = {
            "reaction_token": inner._reaction_token,
            "products_token": 0,
            "product_atoms": m_atoms,
            "product_bonds": bond_kind_matrix(frontier.bonds).reshape(-1),
            "product_counts": m_counts,
            "unique_product_count": len(frontier),
            "reactions": inner._reaction_data,
            "assignments": assignments[valid].copy(),
            "pair_indices": np.repeat(pairs, counts, axis=0).copy(),
        }
        if not len(packed["assignments"]):
            continue
        produced = construct(**packed)
        count = len(packed["assignments"])
        captured.append(
            (
                packed,
                status_rows(produced.status, count),
                assignment_precursor_hashes(produced, count),
            )
        )
        construct(**packed)
        baseline_gpu_seconds += constructor.last_construct_seconds
    assert captured

    # Construct's modes 6..9 used to profile the WL stages, because one `profile_mode` argument
    # had to address two kernels fused into one entry point. The keyer is its own entry now, so the
    # WL stages are profiled by timing `metal_wl_hash` at `profile_mode` 1..4 over precursors
    # construct already produced -- which needs the construct kernel once rather than five times.
    profile_wl = os.environ.get("METAL_RETRO_PROFILE_WL") == "1"
    modes = (3,) if profile_wl else (1, 2, 5, 4, 3)
    rows_by_mode: dict[int, list[dict[str, float]]] = {mode: [] for mode in modes}
    full_rows: list[np.ndarray] = []

    def run() -> None:
        for mode in modes:
            started = perf_counter()
            gpu_seconds = 0.0
            statuses: list[np.ndarray] = []
            for kwargs, expected_status, expected_hashes in captured:
                produced = construct(**kwargs, profile_mode=mode)
                count = len(kwargs["assignments"])
                rows = status_rows(produced.status, count)
                statuses.append(rows)
                gpu_seconds += constructor.last_construct_seconds
                if mode == 3:
                    # Atomic output order can vary; the per-assignment status and
                    # multiset of precursor identities must still match normal mode.
                    assert np.array_equal(rows[:, :4], expected_status[:, :4])
                    assert np.array_equal(rows[:, 5:7], expected_status[:, 5:7])
                    assert (
                        assignment_precursor_hashes(produced, count) == expected_hashes
                    )
            rows_by_mode[mode].append(
                {
                    "gpu_seconds": gpu_seconds,
                    "wall_seconds": perf_counter() - started,
                }
            )
            if mode == 3 and not full_rows:
                full_rows.extend(statuses)

    benchmark.pedantic(
        run,
        rounds=int(os.environ.get("METAL_PIPELINE_ROUNDS", "3")),
        warmup_rounds=0,
        iterations=1,
    )

    rows = np.concatenate(full_rows)
    counts = Counter(rows[:, 5].tolist())
    stages = Counter(rows[:, 7].tolist())
    gpu = {
        mode: median(item["gpu_seconds"] for item in records)
        for mode, records in rows_by_mode.items()
    }
    scratch_bytes = probe_retro_pipelines()["scratch_bytes"]
    summary = {
        "frontier_size": frontier_size,
        "source_target_count": min(profile_setup.source_target_count, frontier_size),
        "assignments": len(rows),
        "dispatches": len(captured),
        "candidate_pairs": len(candidates),
        "sampled_pairs": len(positive),
        "emitted_components": int(rows[:, 0].sum()),
        "multi_component_assignments": int(np.count_nonzero(rows[:, 0] > 1)),
        "component_fanout": dict(Counter(rows[:, 0].tolist())),
        "cumulative_reserved_scratch_bytes": len(rows) * scratch_bytes,
        "peak_reserved_scratch_bytes": max(
            len(kwargs["assignments"]) for kwargs, _, _ in captured
        )
        * scratch_bytes,
        "edited_bond_square_bytes": int(np.square(rows[:, 8].astype(np.uint64)).sum()),
        "rejection_reasons": dict(counts),
        "terminal_stages": dict(stages),
        "edited_atoms": distribution(rows[:, 8]),
        "component_merges": distribution(rows[:, 9]),
        "kept_components": distribution(rows[:, 10]),
        "perceived_atoms": distribution(rows[:, 11]),
        "aromatic_atoms": distribution(rows[:, 12]),
        "cycles": distribution(rows[:, 13]),
        "gpu_seconds": gpu,
        "baseline_gpu_seconds": baseline_gpu_seconds,
        "profile_full_overhead_ratio": gpu[3] / baseline_gpu_seconds,
        "mode_runs": rows_by_mode,
    }
    if profile_wl:
        from vesma.metal_retro import metal_wl_hash

        # One construct pass for the precursors, then the keyer at each stage over the same arrays.
        # Stage numbers are the WL kernel's own: 1 bond identity, 2 initial labels, 3 after
        # refinement, 4 bond identity with the short-ring search omitted, 0 the whole thing.
        wl: dict[int, float] = {}
        for stage in (0, 1, 2, 3, 4):
            total = 0.0
            for kwargs, _status, _hashes in captured:
                produced = construct(**kwargs)
                if not len(produced.atom_counts):
                    continue
                total += metal_wl_hash(
                    produced.atoms,
                    produced.bonds,
                    produced.atom_counts,
                    profile_mode=stage,
                ).seconds
            wl[stage] = total
        summary.update(
            wl_gpu_seconds=wl,
            wl_identity_without_ring_seconds=wl[4],
            wl_short_ring_search_seconds_estimate=wl[1] - wl[4],
            wl_identity_bonds_seconds=wl[1],
            wl_initial_labels_seconds_estimate=wl[2] - wl[1],
            wl_refinement_seconds_estimate=wl[3] - wl[2],
            wl_sort_hash_seconds_estimate=wl[0] - wl[3],
        )
    else:
        summary.update(
            edit_seconds=gpu[1],
            component_seconds_estimate=gpu[2] - gpu[1],
            aromatic_cycle_search_seconds_estimate=gpu[5] - gpu[2],
            aromatic_postcycle_seconds_estimate=gpu[4] - gpu[5],
            property_and_emit_seconds_estimate=gpu[3] - gpu[4],
        )
    benchmark.extra_info.update(summary)
    print(
        f"Metal retro phases: {summary | {'mode_runs': 'see benchmark extra_info'}}",
        flush=True,
    )

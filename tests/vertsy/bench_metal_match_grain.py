"""Compare Metal matcher threadgroup sizes on fixed probe and retry pairs.

Run with ``make bench-metal-match-grain METAL_PIPELINE_TARGETS_FILE=...``.
The source fixture uses 64 stock molecules and used14k templates so one setup
serves every lane count. All variants see the same folded-prefilter pairs.
"""

from __future__ import annotations

import os
from statistics import median
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from tests.vertsy.bench_metal_pipeline import ProfileSetup

pytest_plugins = ["tests.vertsy.bench_metal_pipeline"]

pytestmark = [
    pytest.mark.perf,
    pytest.mark.external_data,
    pytest.mark.gpu,
    pytest.mark.metal,
]


@pytest.mark.parametrize("threads_per_group", [32, 64, 128, 256, 512, 1024])
def test_bench_metal_match_grain(
    benchmark: Any, profile_setup: ProfileSetup, threads_per_group: int
) -> None:
    frontier = profile_setup.source
    # The arm hands its matcher over; `arm.match` is a composed function, so it has no
    # `__self__` to reach through. Same fix as `bench_metal_retro_phases`.
    wrapper = profile_setup.arm.matcher
    inner = wrapper._inner
    matcher = inner._matcher
    candidates = wrapper._prefilter_pairs(frontier)
    pairs = np.ascontiguousarray(candidates[:, ::-1])
    m_atoms = frontier.atoms.reshape(-1)
    m_rows = frontier.adjacency_rows().reshape(-1)
    m_bonds = frontier.bonds.reshape(-1)
    m_counts = np.ascontiguousarray(frontier.atom_counts, dtype=np.uint32)
    probe_cap = inner._assignment_probe_cap

    def dispatch(selected: np.ndarray, cap: int) -> np.ndarray:
        status, _ = matcher.dispatch_match_packed(
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
            pair_indices=selected,
            max_assignments=cap,
            copy_valid_only=True,
            pattern_token=inner._reaction_token,
        )
        return status

    matcher.threads_per_group = 256
    expected_probe = dispatch(pairs, probe_cap)
    saturated = pairs[expected_probe[:, 0] == probe_cap]
    expected_retry = dispatch(saturated, inner._max_assignments)
    records: list[dict[str, float]] = []

    def run() -> None:
        matcher.threads_per_group = threads_per_group
        probe = dispatch(pairs, probe_cap)
        probe_profile = matcher.last_dispatch_profile
        retry = dispatch(saturated, inner._max_assignments)
        retry_profile = matcher.last_dispatch_profile
        assert np.array_equal(probe[:, :2], expected_probe[:, :2])
        assert np.array_equal(retry[:, :2], expected_retry[:, :2])
        records.append(
            {
                "probe_wall_seconds": probe_profile.total_seconds,
                "probe_kernel_seconds": probe_profile.kernel_seconds,
                "probe_copyback_seconds": probe_profile.copyback_seconds,
                "retry_wall_seconds": retry_profile.total_seconds,
                "retry_kernel_seconds": retry_profile.kernel_seconds,
                "retry_copyback_seconds": retry_profile.copyback_seconds,
                "probe_prunings": float(probe[:, 2].sum()),
                "probe_leaf_checks": float(probe[:, 3].sum()),
                "retry_prunings": float(retry[:, 2].sum()),
                "retry_leaf_checks": float(retry[:, 3].sum()),
            }
        )

    benchmark.pedantic(
        run,
        rounds=int(os.environ.get("METAL_PIPELINE_ROUNDS", "5")),
        warmup_rounds=0,
        iterations=1,
    )
    summary = {
        "threads_per_group": threads_per_group,
        "candidate_pairs": len(pairs),
        "positive_pairs": int(np.count_nonzero(expected_probe[:, 0])),
        "saturated_pairs": len(saturated),
        "medians": {name: median(row[name] for row in records) for name in records[0]},
        "runs": records,
    }
    benchmark.extra_info.update(summary)
    print(
        f"Metal match grain: {summary | {'runs': 'see benchmark extra_info'}}",
        flush=True,
    )

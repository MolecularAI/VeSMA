"""Profile one complete Metal expansion and stock heuristic at several frontier sizes.

The source frontier is a deduplicated precursor batch from one real target. This exercises
matching, construction, keying, AO* interning, fingerprinting, and stock scan without running a
multi-pass search. The host and Metal fingerprint measurements share the same match and scan,
so their summed complete-step times are comparable. Use ``METAL_PIPELINE_STOCK_LIMIT=0`` for the
full catalogue after its preparation cache is warm.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np
import pytest

from tests.vertsy.bench_host_aostar import (
    FINGERPRINT,
    STOCK_FILES,
    TEMPLATE_FILES,
    Arm,
    cached_stock,
    load_library,
    metal_arm,
    read_stock_smiles,
    read_targets,
    skip_unless_metal,
)
from vertsy.expansions import Expansions
from vertsy.host_similarity import NearestStock
from vertsy.interner import DictInterner
from vertsy.search import step
from vertsy.store.numpy_store import NumpyExpansionStore
from vesma.configurations import ConfigurationError
from vesma.metal_circularfp import compute_circular_fingerprints_metal
from vesma.metal_reaction_matcher import MetalMoleculeBatch, metal_short_ring_wl_key
from vesma.packed_circularfp import circular_fingerprints_packed
from vesma.reaction import reaction_pickle_cache_path

pytestmark = [
    pytest.mark.perf,
    pytest.mark.external_data,
    pytest.mark.gpu,
    pytest.mark.metal,
    pytest.mark.filterwarnings(
        "ignore:Setting the shape on a NumPy array has been deprecated.*:DeprecationWarning:joblib.numpy_pickle"
    ),
]


@dataclass(frozen=True)
class ProfileSetup:
    arm: Arm
    source: MetalMoleculeBatch
    stock_fingerprints: np.ndarray
    setup_seconds: dict[str, float]
    template_cache_hit: bool
    source_target_count: int


def measured(call: Callable[[], Any]) -> tuple[Any, float]:
    started = perf_counter()
    result = call()
    return result, perf_counter() - started


@pytest.fixture(scope="module")
def profile_setup() -> ProfileSetup:
    skip_unless_metal()
    targets_file = os.environ.get("METAL_PIPELINE_TARGETS_FILE", "")
    if not targets_file or not Path(targets_file).exists():
        pytest.skip("Set METAL_PIPELINE_TARGETS_FILE to a readable target SMILES file.")
    rule_set = os.environ.get("METAL_PIPELINE_RULE_SET", "raw68k")
    stock_name = os.environ.get("METAL_PIPELINE_STOCK", "enamine")
    template_file = TEMPLATE_FILES[rule_set]
    stock_file = STOCK_FILES[stock_name]
    if not template_file.exists() or not stock_file.exists():
        pytest.skip("Full template and stock datasets are not installed.")

    library, template_load_s = measured(lambda: load_library(rule_set, template_file))
    stock_limit = int(os.environ.get("METAL_PIPELINE_STOCK_LIMIT", "50000")) or None
    stock_smiles, stock_read_s = measured(
        lambda: read_stock_smiles(stock_file, stock_limit)
    )
    stock, stock_prepare_s = measured(lambda: cached_stock(stock_smiles, 1))

    cache_hit = reaction_pickle_cache_path(library.smarts).exists()
    arm, arm_setup_s = measured(
        lambda: metal_arm(
            library,
            stock,
            heuristic_kind="stock",
            scan_kind="metal",
            weight=1.0,
            n_jobs=1,
            assignment_probe_cap=int(
                os.environ.get("METAL_PIPELINE_ASSIGNMENT_PROBE_CAP", "8")
            ),
        )
    )
    source_mode = os.environ.get("METAL_PIPELINE_SOURCE")
    if source_mode in ("targets-stock", "stock-only"):
        sources = []
        target_smiles = (
            read_targets(Path(targets_file), 64, 0)
            if source_mode == "targets-stock"
            else ()
        )
        target_count = 0
        for index, smiles in enumerate(target_smiles + stock_smiles):
            if len(sources) >= 64:
                break
            try:
                sources.append(arm.mol_of(smiles))
                if index < len(target_smiles):
                    target_count += 1
            except ConfigurationError:
                continue
        if len(sources) < 64:
            pytest.skip("Need 64 distinct target and stock molecules.")
        # A batch is molecules; the keyer answers for it when the bench needs sids. This used to
        # `replace(...)` a `keys` field onto the batch, which no longer exists.
        source, seed_s = measured(lambda: arm.stack_mols(sources))
    else:
        target = read_targets(Path(targets_file), 1, 0)[0]
        generated, seed_s = measured(
            lambda: arm.match(arm.stack_mols([arm.mol_of(target)]))
        )
        seed_batch, seed_keys = generated.precursors, list(generated.keys)
        unique = list(dict.fromkeys(seed_keys))
        if len(unique) < 64:
            pytest.skip(
                f"seed target yielded only {len(unique)} distinct precursors; need 64"
            )
        first_index = {key: i for i, key in enumerate(seed_keys)}
        source = seed_batch[np.asarray([first_index[key] for key in unique[:64]])]
        target_count = 1
    setup_seconds = {
        "template_file": template_load_s,
        "stock_file": stock_read_s,
        "stock_prepare": stock_prepare_s,
        "arm_setup": arm_setup_s,
        "seed_match": seed_s,
    } | arm.setup_seconds
    print(
        f"Metal pipeline setup={setup_seconds} template_cache_hit={cache_hit}",
        flush=True,
    )
    return ProfileSetup(
        arm, source, stock.wl_fingerprints, setup_seconds, cache_hit, target_count
    )


@pytest.mark.parametrize("frontier_size", [1, 8, 64])
def test_bench_metal_pipeline(
    benchmark: Any, profile_setup: ProfileSetup, frontier_size: int
) -> None:
    arm = profile_setup.arm
    frontier = profile_setup.source[:frontier_size]
    heuristic = arm.heuristic
    assert isinstance(heuristic, NearestStock)
    records: list[dict[str, float]] = []

    def run() -> None:
        expansions = Expansions(NumpyExpansionStore(), DictInterner())
        # Keyed here rather than read off the batch: identity is the keyer's answer, not a field.
        frontier_sids = expansions.intern(metal_short_ring_wl_key(frontier))
        # `step` returns a `Layer`, not a tuple -- it has since `da3f1be` gave the estimate a
        # name so `search` would not have to skip a third field. This bench still unpacked two.
        layer, step_s = measured(
            lambda: step(
                frontier_sids,
                frontier,
                expansions,
                arm.stock,
                match=arm.match,
            )
        )
        fresh_sids, fresh = layer.sids, layer.mols
        assert len(fresh_sids) > 0
        diagnostics = arm.diagnostics()
        metal_words, metal_fp_s = measured(
            lambda: compute_circular_fingerprints_metal(
                fresh.atoms, fresh.bonds, fresh.atom_counts, FINGERPRINT
            )
        )
        host_words, host_fp_s = measured(
            lambda: circular_fingerprints_packed(
                fresh.atoms, fresh.bonds, fresh.atom_counts, FINGERPRINT
            )
        )
        assert np.array_equal(metal_words, host_words)
        _, scan_s = measured(
            lambda: heuristic.scan(
                metal_words, profile_setup.stock_fingerprints, weight=heuristic.weight
            )
        )
        records.append(
            {
                "step_seconds": step_s,
                "metal_fingerprint_seconds": metal_fp_s,
                "host_fingerprint_seconds": host_fp_s,
                "scan_seconds": scan_s,
                "metal_complete_seconds": step_s + metal_fp_s + scan_s,
                "host_fingerprint_complete_seconds": step_s + host_fp_s + scan_s,
                "frontier_count": float(len(frontier)),
                "novel_count": float(len(fresh)),
                "fingerprint_input_bytes": float(
                    fresh.atoms.nbytes + fresh.bonds.nbytes + fresh.atom_counts.nbytes
                ),
                "fingerprint_copyback_bytes": float(metal_words.nbytes),
            }
            | diagnostics
        )

    benchmark.pedantic(
        run,
        rounds=int(os.environ.get("METAL_PIPELINE_ROUNDS", "3")),
        warmup_rounds=0,
        iterations=1,
    )
    medians = {name: median(row[name] for row in records) for name in records[0]}
    benchmark.extra_info.update(
        {
            "frontier_size": frontier_size,
            "template_cache_hit": profile_setup.template_cache_hit,
            "source_target_count": min(
                profile_setup.source_target_count, frontier_size
            ),
            "setup_seconds": profile_setup.setup_seconds,
            "stage_medians": medians,
            "stage_runs": records,
        }
    )
    print(f"Metal pipeline frontier={frontier_size} median={medians}", flush=True)


@pytest.mark.parametrize("assignment_cap", [8, 32, 256])
def test_bench_metal_match_assignment_capacity(
    benchmark: Any, profile_setup: ProfileSetup, assignment_cap: int
) -> None:
    """Compare dense matcher capacities before choosing a compact output design."""
    frontier = profile_setup.source
    # The arm hands its matcher over; `arm.match` is a composed function, so it has no
    # `__self__` to reach through. Same fix as `bench_metal_retro_phases`.
    wrapper = profile_setup.arm.matcher
    inner = wrapper._inner
    candidates = wrapper._prefilter_pairs(frontier)
    pairs = np.ascontiguousarray(candidates[:, ::-1])
    chunk_size = max(1, (512 * 1024 * 1024) // (assignment_cap * 64 * 4))
    m_atoms = frontier.atoms.reshape(-1)
    m_rows = frontier.adjacency_rows().reshape(-1)
    m_bonds = frontier.bonds.reshape(-1)
    m_counts = np.ascontiguousarray(frontier.atom_counts, dtype=np.uint32)
    records: list[dict[str, float]] = []

    def run() -> None:
        saturated = assignments = copied_bytes = 0
        started = perf_counter()
        for start in range(0, len(pairs), chunk_size):
            status, _ = inner._matcher.dispatch_match_packed(
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
                pair_indices=pairs[start : start + chunk_size],
                max_assignments=assignment_cap,
                copy_valid_only=True,
                pattern_token=inner._reaction_token,
            )
            counts = status[:, 0]
            saturated += int(np.count_nonzero(counts == assignment_cap))
            assignments += int(counts.sum())
            copied_bytes += inner._matcher.last_dispatch_profile.output_copyback_bytes
        records.append(
            {
                "seconds": perf_counter() - started,
                "candidate_pairs": float(len(pairs)),
                "saturated_pairs": float(saturated),
                "stored_assignments": float(assignments),
                "reserved_bytes": float(len(pairs) * assignment_cap * 64 * 4),
                "copied_bytes": float(copied_bytes),
            }
        )

    benchmark.pedantic(
        run,
        rounds=int(os.environ.get("METAL_PIPELINE_ROUNDS", "3")),
        warmup_rounds=0,
        iterations=1,
    )
    medians = {name: median(row[name] for row in records) for name in records[0]}
    benchmark.extra_info.update(
        {"assignment_cap": assignment_cap, "stage_medians": medians}
    )
    print(f"Metal assignment cap={assignment_cap} median={medians}", flush=True)

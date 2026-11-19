"""Parity tests for `CudaReactionMatcher` (Phase 6 step 4).

The CUDA reaction matcher must reproduce the CPU backend exactly: same expansion
layers (`build_layer` output) and bit-exact short-ring WL tokens. Matching and keying
are separate stages — the matcher emits un-keyed precursor mols and
`cuda_short_ring_wl_key` hashes them — mirroring `CpuReactionMatcher` + the WL codec.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from vesma import cmol_from_smiles, gmol_from_cmol
from vesma.cpu_matcher import CpuReactionMatcher
from vesma.cuda_matcher import (
    CUDA_HOST_FRONTIER_MAX_PAIRS_PER_LAUNCH,
    CudaMoleculeHostBatch,
    CudaReactionMatcher,
    cuda_short_ring_wl_key,
)
from vesma.cuda_prune import (
    CudaOutputCapacityError,
    match_fused_assignments_cuda,
    prepare_cuda_prune_batch,
)
from vesma.cuda_retro import (
    CudaMoleculeBatch,
    construct_retro_cuda,
    prepare_cuda_reaction_batch,
)
from vesma.keyer import short_ring_wl_keyer
from vesma.reaction import (
    ReactionPattern,
    parse_reaction_smarts,
    reaction_patterns_from_smarts,
)


def build_layer(frontier_keys, hits, precursor_keys):
    """Local copy of the retired dict layer-builder, for CPU/CUDA matcher parity checks.

    Groups flat ``(target_idx, rule_id, app_idx, precursor_idx)`` hits into
    ``{product_key: frozenset[(rule_id, frozenset[precursor_key])]}``; keys are read
    through the hit's precursor index, so a deduplicated precursor batch works too.
    Every frontier key gets an entry so dead ends surface as empty step sets.
    """
    from collections import defaultdict

    layer: dict = {key: set() for key in frontier_keys}
    groups: dict = defaultdict(list)
    for target_idx, rule_id, app_idx, precursor_idx in hits:
        groups[(target_idx, rule_id, app_idx)].append(precursor_keys[precursor_idx])
    for (target_idx, rule_id, _), prec_keys in groups.items():
        layer[frontier_keys[target_idx]].add((rule_id, frozenset(prec_keys)))
    return {key: frozenset(steps) for key, steps in layer.items()}


def _skip_if_no_cuda() -> None:
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CuPy cannot access CUDA on this machine.")


RULES: dict[int, str] = {
    0: "[C:1][C:2]>>[C:1].[C:2]",
    1: "[C:1][O:2]>>[C:1].[O:2]",
    2: "[C:1][N:2]>>[C:1].[N:2]",
    3: "[C:1]=[O:2]>>[C:1].[O:2]",
}

TARGETS: tuple[str, ...] = (
    "CCO",
    "CCN",
    "CCCC",
    "CC(=O)O",
    "CCOC",
    "OCC(N)CO",
    "c1ccccc1C",
)


@pytest.fixture(scope="module")
def compiled_rules() -> Sequence[ReactionPattern]:
    _skip_if_no_cuda()
    return reaction_patterns_from_smarts(
        tuple(RULES.values()),
        n_jobs=1,
        cache=False,
    )


def _rules_from_smarts(smarts: tuple[str, ...]) -> Sequence[ReactionPattern]:
    return reaction_patterns_from_smarts(smarts, n_jobs=1, cache=False)


def _cpu_layer(
    target_smiles: str,
    rules: Sequence[ReactionPattern],
) -> dict:
    codec = short_ring_wl_keyer
    matcher = CpuReactionMatcher(rules)
    product = cmol_from_smiles(target_smiles)[0]
    frontier_keys = [codec.from_smiles(target_smiles)]
    precursors, hits = matcher.match([product])
    keys = [codec.encode(m) for m in precursors]
    return build_layer(frontier_keys, hits, keys)


def _cuda_layer(target_smiles: str, matcher: CudaReactionMatcher) -> dict:
    codec = short_ring_wl_keyer
    product = cmol_from_smiles(target_smiles)[0]
    frontier_keys = [codec.from_smiles(target_smiles)]
    precursors, hits = matcher.match([product])
    keys = cuda_short_ring_wl_key(precursors)
    return build_layer(frontier_keys, hits, keys)


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize("target", TARGETS)
def test_cuda_matcher_layer_matches_cpu(
    target: str,
    compiled_rules: Sequence[ReactionPattern],
) -> None:
    """One-layer expansion is byte-identical to the CPU backend (bit-exact WL tokens)."""
    matcher = CudaReactionMatcher(compiled_rules)
    assert _cuda_layer(target, matcher) == _cpu_layer(target, compiled_rules)


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_matcher_batched_targets_match_cpu(
    compiled_rules: Sequence[ReactionPattern],
) -> None:
    """A single multi-target match call reproduces the per-target CPU layers."""
    codec = short_ring_wl_keyer
    matcher = CudaReactionMatcher(compiled_rules)

    products = [cmol_from_smiles(smi)[0] for smi in TARGETS]
    frontier_keys = [codec.from_smiles(smi) for smi in TARGETS]
    precursors, hits = matcher.match(products)
    keys = cuda_short_ring_wl_key(precursors)
    cuda_layer = build_layer(frontier_keys, hits, keys)

    expected: dict = {}
    for smi in TARGETS:
        expected.update(_cpu_layer(smi, compiled_rules))
    assert cuda_layer == expected


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_keyed_host_match_matches_device_and_supports_selection(
    compiled_rules: Sequence[ReactionPattern],
) -> None:
    codec = short_ring_wl_keyer
    products = [cmol_from_smiles(smi)[0] for smi in TARGETS]
    frontier_keys = [codec.from_smiles(smi) for smi in TARGETS]

    device = CudaReactionMatcher(compiled_rules, frontier_mode="device")
    device_mols, device_keys, device_hits = device.match_keyed(products)
    host = CudaReactionMatcher(compiled_rules, frontier_mode="host")
    host_mols, host_keys, host_hits = host.match_keyed(products)

    assert isinstance(device_mols, CudaMoleculeBatch)
    assert len(device_mols) == len(device_keys)
    assert len(device_mols) == len(set(device_keys))
    assert isinstance(host_mols, CudaMoleculeHostBatch)
    assert len(host_mols) == len(host_keys)
    assert host_mols.physical_count == len(set(host_keys))
    assert build_layer(frontier_keys, host_hits, host_keys) == build_layer(
        frontier_keys,
        device_hits,
        device_keys,
    )
    selected = host_mols[np.arange(len(host_mols)) % 2 == 0]
    assert len(selected) == (len(host_mols) + 1) // 2
    assert selected.nbytes > 0
    assert device.last_diagnostics is not None
    assert device.last_diagnostics.frontier_mode == "device"
    assert device.last_diagnostics.unique_precursor_count == len(device_keys)
    assert (
        device.last_diagnostics.raw_precursor_count
        == device.last_diagnostics.unique_precursor_count
        + device.last_diagnostics.duplicate_precursor_count
    )
    assert device.last_diagnostics.retained_device_to_host_seconds == 0.0
    assert device.last_diagnostics.retained_molecule_count == len(set(device_keys))
    assert host.last_diagnostics is not None
    assert host.last_diagnostics.frontier_mode == "host"
    assert host.last_diagnostics.unique_precursor_count == len(host_keys)
    assert (
        host.last_diagnostics.raw_precursor_count
        == host.last_diagnostics.unique_precursor_count
        + host.last_diagnostics.duplicate_precursor_count
    )
    assert host.last_diagnostics.key_seconds == pytest.approx(
        host.last_diagnostics.wl_kernel_seconds
        + host.last_diagnostics.hash_device_to_host_seconds
        + host.last_diagnostics.key_materialize_seconds
    )
    assert (
        host.last_diagnostics.device_to_host_seconds
        >= host.last_diagnostics.retained_device_to_host_seconds
    )
    assert host.last_diagnostics.unclassified_seconds >= 0.0


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_keyed_device_frontier_capacity_overflow(
    compiled_rules: Sequence[ReactionPattern],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    products = [cmol_from_smiles(smi)[0] for smi in TARGETS]
    matcher = CudaReactionMatcher(compiled_rules, frontier_mode="device")
    monkeypatch.setattr(
        matcher,
        "_device_frontier_capacity",
        lambda *, input_molecule_count, pair_count: 1,
    )

    with pytest.raises(CudaOutputCapacityError, match="device frontier"):
        matcher.match_keyed(products)


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_host_frontier_defaults_to_bounded_pair_launches(
    compiled_rules: Sequence[ReactionPattern],
) -> None:
    device = CudaReactionMatcher(compiled_rules, frontier_mode="device")
    host = CudaReactionMatcher(compiled_rules, frontier_mode="host")
    explicit = CudaReactionMatcher(
        compiled_rules,
        frontier_mode="host",
        max_pairs_per_launch=1,
    )

    assert device.max_pairs_per_launch is None
    assert host.max_pairs_per_launch == CUDA_HOST_FRONTIER_MAX_PAIRS_PER_LAUNCH
    assert explicit.max_pairs_per_launch == 1


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_selective_restricts_to_requested_rules(
    compiled_rules: Sequence[ReactionPattern],
) -> None:
    """An explicit (mol, rule) pair set restricts the work; hits carry only that rule."""
    matcher = CudaReactionMatcher(compiled_rules)
    ethanol = cmol_from_smiles("CCO")[0]

    # Restrict to the C-N rule (id 2): ethanol has no C-N bond -> no precursors.
    precursors, hits = matcher.match([ethanol], [(0, 2)])
    assert len(hits) == 0
    assert precursors.count == 0

    # Restrict to the C-O rule (id 1): matches, and every hit carries rule id 1.
    precursors, hits = matcher.match([ethanol], [(0, 1)])
    assert len(hits)
    assert {rule_id for _, rule_id, _, _ in hits} == {1}
    assert all(mol_idx == 0 for mol_idx, _, _, _ in hits)


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_keyed_selective_matches_unkeyed_selective(
    compiled_rules: Sequence[ReactionPattern],
) -> None:
    _skip_if_no_cuda()
    products = [cmol_from_smiles("CCO")[0], cmol_from_smiles("CCN")[0]]
    pairs = [(0, 1), (1, 2)]

    unkeyed = CudaReactionMatcher(compiled_rules)
    precursors, hits = unkeyed.match(products, pairs)
    expected_keys = cuda_short_ring_wl_key(precursors)

    keyed = CudaReactionMatcher(compiled_rules, frontier_mode="host")
    output, observed_keys, observed_hits = keyed.match_keyed(products, pairs)
    codec = short_ring_wl_keyer
    frontier_keys = [codec.encode(product) for product in products]

    assert isinstance(output, CudaMoleculeHostBatch)
    assert build_layer(frontier_keys, observed_hits, observed_keys) == build_layer(
        frontier_keys,
        hits,
        expected_keys,
    )
    assert keyed.last_diagnostics is not None
    assert keyed.last_diagnostics.frontier_mode == "host"


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_match_diagnostics_surface_pair_counts(
    compiled_rules: Sequence[ReactionPattern],
) -> None:
    """Stage-2 diagnostics are surfaced on the matcher after each match call."""
    matcher = CudaReactionMatcher(compiled_rules)
    products = [cmol_from_smiles(smi)[0] for smi in ("CCO", "CCN", "CCCC")]

    matcher.match(products)
    diag = matcher.last_diagnostics
    assert diag is not None
    assert diag.pair_count == len(RULES) * len(products)
    assert diag.status_no + diag.status_yes + diag.status_maybe == diag.pair_count
    assert diag.status_yes >= 1  # at least one (mol, rule) matched
    assert diag.incomplete_pairs == 0  # small graphs: leaf enumeration is complete
    assert diag.candidate_assignments_total >= diag.assignments_total
    assert diag.candidate_assignments_max >= diag.assignments_max
    assert diag.candidate_overflow_pairs == 0
    assert diag.pair_launches == 1
    assert diag.resident_rule_bytes > 0


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize("target", ["CCCC", "CC(=O)O", "OCC(N)CO"])
def test_defused_keyer_matches_fused_wl(target: str) -> None:
    """De-fusing WL (compute_wl=False + separate keyer) yields identical tokens to the
    fused path (compute_wl=True), proving the de-fuse is value-preserving."""
    _skip_if_no_cuda()
    reactions = [parse_reaction_smarts(s) for s in RULES.values()]
    product = cmol_from_smiles(target)[0]

    batch = prepare_cuda_prune_batch(
        [r.lhs for r in reactions], [gmol_from_cmol(product)]
    )
    match = match_fused_assignments_cuda(batch, max_leaf_checks=4096)
    reaction_batch = prepare_cuda_reaction_batch(reactions)

    fused = construct_retro_cuda(batch, match, reaction_batch, compute_wl=True)
    defused = construct_retro_cuda(batch, match, reaction_batch, compute_wl=False)

    fused_lo = fused.precursor_hash_lo_host()
    fused_hi = fused.precursor_hash_hi_host()
    fused_assignment_ids = fused.precursor_assignment_ids_host()
    fused_by_assignment: dict[int, set[tuple[int, int]]] = {}
    for assignment_id, lo, hi in zip(
        fused_assignment_ids, fused_lo, fused_hi, strict=True
    ):
        fused_by_assignment.setdefault(int(assignment_id), set()).add(
            (int(lo), int(hi))
        )
    fused_tokens = {frozenset(token_set) for token_set in fused_by_assignment.values()}

    # The de-fused path leaves hashes zero; reproduce them via the standalone keyer.
    from vesma.cuda_retro import CudaMoleculeBatch

    precursors = CudaMoleculeBatch(
        tokens=(),
        atom_counts_device=defused.precursor_atom_counts_device,
        atoms_device=defused.precursor_atoms_device,
        bonds_device=defused.precursor_bonds_device,
        rows_device=defused.precursor_rows_device,
    )
    keys = cuda_short_ring_wl_key(precursors)

    # Regroup the flat keys back into per-assignment component sets.
    defused_tokens: set[frozenset[tuple[int, int]]] = set()
    by_assignment: dict[int, set[tuple[int, int]]] = {}
    for assignment_id, key in zip(
        defused.precursor_assignment_ids_host(), keys, strict=True
    ):
        by_assignment.setdefault(int(assignment_id), set()).add(key)
    for token_set in by_assignment.values():
        defused_tokens.add(frozenset(token_set))

    assert defused_tokens == fused_tokens


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_chunking_matches_single_batch(
    compiled_rules: Sequence[ReactionPattern],
) -> None:
    """Rule + molecule chunking (tiny chunk sizes) reproduces the single-batch match:
    same build_layer, hit rule ids, global mol indices, and precursor indices."""
    codec = short_ring_wl_keyer
    products = [cmol_from_smiles(smi)[0] for smi in TARGETS]
    frontier_keys = [codec.from_smiles(smi) for smi in TARGETS]

    single = CudaReactionMatcher(compiled_rules)
    chunked = CudaReactionMatcher(
        compiled_rules,
        max_pairs_per_launch=1,
    )

    p1, h1 = single.match(products)
    layer1 = build_layer(frontier_keys, h1, cuda_short_ring_wl_key(p1))
    p2, h2 = chunked.match(products)
    layer2 = build_layer(frontier_keys, h2, cuda_short_ring_wl_key(p2))

    assert layer2 == layer1
    assert len(h2) != 0
    assert len(h2) == len(h1)
    assert h1.shape[1] == h2.shape[1] == 4
    assert {r for _, r, _, _ in h2} == {r for _, r, _, _ in h1}
    # mol indices stay within range across molecule chunks
    assert {m for m, _, _, _ in h2} <= set(range(len(products)))
    np.testing.assert_array_equal(h1[:, 3], np.arange(p1.count))
    np.testing.assert_array_equal(h2[:, 3], np.arange(p2.count))
    diagnostics = chunked.last_diagnostics
    assert diagnostics is not None
    assert diagnostics.pair_count == len(RULES) * len(products)
    assert diagnostics.pair_launches == len(RULES) * len(products)


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_all_vs_all_retiles_resident_rule_chunk(
    monkeypatch,
    compiled_rules: Sequence[ReactionPattern],
) -> None:
    products = [cmol_from_smiles(smi)[0] for smi in TARGETS]
    matcher = CudaReactionMatcher(compiled_rules)
    monkeypatch.setattr(
        matcher,
        "_pair_capacity",
        lambda *, resident_rule_bytes: 1,
    )

    _, hits = matcher.match(products)

    diagnostics = matcher.last_diagnostics
    assert diagnostics is not None
    assert diagnostics.pair_count == len(RULES) * len(products)
    assert diagnostics.pair_launches == len(RULES) * len(products)
    assert {rule_id for _, rule_id, _, _ in hits} <= set(RULES)
    assert {molecule_id for molecule_id, _, _, _ in hits} <= set(range(len(products)))


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_selective_works_on_chunked_matcher(
    compiled_rules: Sequence[ReactionPattern],
) -> None:
    """Selective (pairs) on a chunked matcher packs only the referenced rules."""
    matcher = CudaReactionMatcher(
        compiled_rules,
        max_pairs_per_launch=1,
    )
    ethanol = cmol_from_smiles("CCO")[0]

    _, no_hits = matcher.match([ethanol], [(0, 2)])  # C-N rule: no match
    assert len(no_hits) == 0

    precursors, hits = matcher.match([ethanol], [(0, 1)])  # C-O rule
    assert len(hits)
    assert {rule_id for _, rule_id, _, _ in hits} == {1}
    assert precursors.count >= 1


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_diagnostics_report_compact_output_capacity() -> None:
    _skip_if_no_cuda()
    matcher = CudaReactionMatcher(
        _rules_from_smarts(("[C:1][C:2]>>[C:1].[C:2]",)),
        max_device_memory_gb=0.0002,
    )

    matcher.match([cmol_from_smiles("CCCC")[0]])

    diagnostics = matcher.last_diagnostics
    assert diagnostics is not None
    assert diagnostics.assignment_required > 0
    assert diagnostics.assignment_capacity >= diagnostics.assignment_required
    assert diagnostics.precursor_required > 0
    assert diagnostics.precursor_capacity >= diagnostics.precursor_required
    assert diagnostics.peak_estimated_workspace_bytes > 0


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_memory_cap_rejects_minimum_resident_working_set() -> None:
    _skip_if_no_cuda()
    with pytest.raises(ValueError, match="exceed|max_device_memory|cannot fit"):
        CudaReactionMatcher(
            _rules_from_smarts(("[C:1][C:2]>>[C:1].[C:2]",)),
            max_device_memory_gb=0.000001,
        )

from __future__ import annotations

import numpy as np
import pytest

from vesma import cmol_from_smiles, find_matches, gmol_from_cmol, gmol_from_smarts
from vesma.adjacency import bond_presence
from vesma.cpu_matcher import initialize_candidates, prune
from vesma.cuda_prune import (
    _ATOM_WORD_COUNT,
    _CUDA_BUCKET_SIZE,
    CudaUnavailableError,
    _initialize_candidate_rows_device,
    _launch_match_kernel,
    _pack_atom_masks_u32,
    match_fused_assignments_cuda,
    match_fused_cuda,
    prepare_cuda_prune_batch,
    prune_bitset_cuda,
)
from vesma.gmol import GMol
from vesma.gmol_atoms import atom_masks_from_props


def pattern(smarts: str) -> GMol:
    return gmol_from_smarts(smarts)[0]


def molecule(smiles: str) -> GMol:
    return gmol_from_cmol(cmol_from_smiles(smiles)[0])


def _skip_if_no_cuda() -> None:
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CuPy cannot access CUDA on this machine.")


def _expected_initial_candidates(
    patterns: list[GMol],
    molecules: list[GMol],
) -> list[np.ndarray]:
    candidates_out: list[np.ndarray] = []
    for query in patterns:
        for target in molecules:
            candidates_out.append(initialize_candidates(query, target))
    return candidates_out


def _expected_pruned_pairs(
    patterns: list[GMol],
    molecules: list[GMol],
    *,
    max_prune_rounds: int | None = None,
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
                max_prune_rounds=max_prune_rounds,
            )
            candidates_out.append(pruned)
            prune_rounds.append(rounds)
            pruning_complete.append(complete)
    return (
        candidates_out,
        np.asarray(prune_rounds, dtype=np.int32),
        np.asarray(pruning_complete, dtype=bool),
    )


def _expected_match_labels(
    patterns: list[GMol],
    molecules: list[GMol],
) -> np.ndarray:
    rows: list[list[str]] = []
    for query in patterns:
        row: list[str] = []
        for target in molecules:
            status = find_matches(query, target, return_first=True).status
            row.append("yes" if status == "match" else "no")
        rows.append(row)
    return np.asarray(rows)


def _assignment_set(values: np.ndarray) -> set[tuple[int, ...]]:
    return {tuple(int(item) for item in row) for row in values}


def _giant_graph(count: int) -> GMol:
    masks = atom_masks_from_props([{} for _ in range(count)])
    bonds = np.zeros((count, count), dtype=np.uint8)
    atom_map = np.zeros(count, dtype=np.int32)
    return GMol(
        atom_masks=masks,
        bonds=bonds,
        atom_map=atom_map,
        atom_group=np.zeros(count, dtype=np.int32),
        atom_predicate=np.full(count, -1, dtype=np.int32),
        predicates=(),
    )


def test_pack_atom_masks_u32_expands_all_a_to_cuda_overlap_sentinel() -> None:
    packed = _pack_atom_masks_u32(pattern("*")).reshape(
        _CUDA_BUCKET_SIZE,
        _ATOM_WORD_COUNT,
    )

    assert int(packed[0, 4]) == 0xFF


def test_prepare_cuda_prune_batch_raises_when_cupy_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vesma import cuda_prune

    def missing_cupy():
        raise CudaUnavailableError("CuPy is required for the CUDA prune experiment.")

    monkeypatch.setattr(cuda_prune, "_import_cupy", missing_cupy)

    with pytest.raises(CudaUnavailableError, match="CuPy"):
        prepare_cuda_prune_batch([], [])


def test_prepare_cuda_prune_batch_raises_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vesma import cuda_prune

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeCupy:
        cuda = FakeCuda()

    monkeypatch.setattr(cuda_prune, "_import_cupy", lambda: FakeCupy())

    with pytest.raises(CudaUnavailableError, match="CUDA is not available"):
        prepare_cuda_prune_batch([], [])


@pytest.mark.gpu
@pytest.mark.cuda
def test_prepare_cuda_prune_batch_rejects_graphs_larger_than_64() -> None:
    _skip_if_no_cuda()

    giant = _giant_graph(65)

    with pytest.raises(ValueError, match="up to 64 atoms"):
        prepare_cuda_prune_batch([giant], [giant])


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize("threads_per_block", [1, 32, 64])
def test_initialize_candidate_rows_cuda_matches_cpu_cross_product(
    threads_per_block: int,
) -> None:
    _skip_if_no_cuda()

    patterns = [pattern("CO"), pattern("N"), pattern("C1CC1")]
    molecules = [molecule("CCO"), molecule("CCC"), molecule("CCN"), molecule("C1CCC1")]
    expected_candidates = _expected_initial_candidates(patterns, molecules)

    batch = prepare_cuda_prune_batch(patterns, molecules)
    observed_padded = _initialize_candidate_rows_device(
        batch,
        threads_per_block=threads_per_block,
    ).get()

    observed_candidates = [
        observed_padded[idx, :pattern_atoms, :molecule_atoms].copy()
        for idx, (pattern_atoms, molecule_atoms) in enumerate(
            zip(
                batch.pair_pattern_atom_counts,
                batch.pair_molecule_atom_counts,
                strict=True,
            )
        )
    ]
    for observed, expected in zip(
        observed_candidates,
        expected_candidates,
        strict=True,
    ):
        assert np.array_equal(observed, expected)


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize("threads_per_block", [1, 32, 64])
@pytest.mark.parametrize("max_prune_rounds", [None, 1])
def test_prune_bitset_cuda_matches_cpu_cross_product(
    threads_per_block: int,
    max_prune_rounds: int | None,
) -> None:
    _skip_if_no_cuda()

    patterns = [pattern("CO"), pattern("N"), pattern("C1CC1")]
    molecules = [molecule("CCO"), molecule("CCC"), molecule("CCN"), molecule("C1CCC1")]
    expected_candidates, expected_rounds, expected_complete = _expected_pruned_pairs(
        patterns,
        molecules,
        max_prune_rounds=max_prune_rounds,
    )

    batch = prepare_cuda_prune_batch(patterns, molecules)
    result = prune_bitset_cuda(
        batch,
        threads_per_block=threads_per_block,
        max_prune_rounds=max_prune_rounds,
    )

    observed_candidates = result.candidates_host()
    assert len(observed_candidates) == len(expected_candidates)
    for observed, expected in zip(
        observed_candidates,
        expected_candidates,
        strict=True,
    ):
        assert np.array_equal(observed, expected)
    assert np.array_equal(result.prune_rounds_host(), expected_rounds)
    assert np.array_equal(result.pruning_complete_host(), expected_complete)


@pytest.mark.gpu
@pytest.mark.cuda
def test_prune_cuda_handles_empty_cross_product() -> None:
    _skip_if_no_cuda()

    batch = prepare_cuda_prune_batch([pattern("CO")], [])

    initialized = _initialize_candidate_rows_device(batch)
    bitset = prune_bitset_cuda(batch)

    assert initialized.get().shape == (0, 64, 64)
    assert bitset.candidates_host() == []
    assert bitset.prune_rounds_host().shape == (0,)
    assert bitset.pruning_complete_host().shape == (0,)


@pytest.mark.gpu
@pytest.mark.cuda
def test_match_fused_cuda_matches_cpu_cross_product() -> None:
    _skip_if_no_cuda()

    patterns = [pattern("CO"), pattern("N"), pattern("C1CC1"), pattern("[C:1][C:2]")]
    molecules = [
        molecule("CCO"),
        molecule("CCC"),
        molecule("CCN"),
        molecule("C1CCC1"),
        molecule("C"),
    ]
    expected = _expected_match_labels(patterns, molecules)

    batch = prepare_cuda_prune_batch(patterns, molecules)
    result = match_fused_cuda(
        batch,
        max_prune_rounds=64,
        max_leaf_checks=4096,
    )

    assert np.array_equal(result.status_labels_host(), expected)


@pytest.mark.gpu
@pytest.mark.cuda
def test_match_fused_cuda_reports_maybe_when_leaf_budget_exhausts() -> None:
    _skip_if_no_cuda()

    batch = prepare_cuda_prune_batch([pattern("C1CC1")], [molecule("C1CCC1")])
    result = match_fused_cuda(
        batch,
        threads_per_block=32,
        max_prune_rounds=64,
        max_leaf_checks=1,
        max_prefix_depth=0,
    )

    assert result.status_labels_host().tolist() == [["maybe"]]
    assert result.pruning_complete_host()[0, 0]
    assert not result.search_complete_host()[0, 0]
    assert result.leaf_checks_host()[0, 0] == 1


@pytest.mark.gpu
@pytest.mark.cuda
def test_match_fused_cuda_can_decide_with_incomplete_pruning() -> None:
    _skip_if_no_cuda()

    batch = prepare_cuda_prune_batch([pattern("C(C)C")], [molecule("CC(C)C")])
    result = match_fused_cuda(
        batch,
        threads_per_block=32,
        max_prune_rounds=1,
        max_leaf_checks=4096,
    )

    assert result.status_labels_host().tolist() == [["yes"]]
    assert not result.pruning_complete_host()[0, 0]


@pytest.mark.gpu
@pytest.mark.cuda
def test_match_fused_cuda_rejects_singleton_injectivity_collision() -> None:
    _skip_if_no_cuda()

    batch = prepare_cuda_prune_batch([pattern("OO")], [molecule("CO")])
    result = match_fused_cuda(
        batch,
        threads_per_block=32,
        max_prune_rounds=64,
        max_leaf_checks=64,
    )

    assert result.status_labels_host().tolist() == [["no"]]
    assert result.search_complete_host()[0, 0]
    assert result.leaf_checks_host()[0, 0] == 0


@pytest.mark.gpu
@pytest.mark.cuda
def test_match_fused_cuda_rejects_invalid_bounds() -> None:
    _skip_if_no_cuda()

    batch = prepare_cuda_prune_batch([pattern("CO")], [molecule("CCO")])

    with pytest.raises(ValueError, match="max_prune_rounds"):
        match_fused_cuda(batch, max_prune_rounds=0)

    with pytest.raises(ValueError, match="max_leaf_checks"):
        match_fused_cuda(batch, max_leaf_checks=0)

    with pytest.raises(ValueError, match="32-thread warp"):
        match_fused_cuda(batch, threads_per_block=64)


@pytest.mark.gpu
@pytest.mark.cuda
def test_match_fused_cuda_is_bond_type_aware() -> None:
    _skip_if_no_cuda()

    batch = prepare_cuda_prune_batch([pattern("C=O")], [molecule("CO")])

    assert match_fused_cuda(batch).status_labels_host().tolist() == [["no"]]


@pytest.mark.gpu
@pytest.mark.cuda
def test_match_only_launch_does_not_allocate_assignment_outputs() -> None:
    _skip_if_no_cuda()

    batch = prepare_cuda_prune_batch([pattern("CC")], [molecule("CCC")])
    outputs = _launch_match_kernel(
        batch,
        threads_per_block=32,
        max_prune_rounds=64,
        max_leaf_checks=4096,
        max_prefix_depth=16,
        collect_assignments=False,
        assignment_capacity=0,
    )

    for output in outputs[1:7]:
        assert output.size == 0


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("smarts", "smiles", "max_leaf_checks"),
    [
        ("CO", "CCO", 4096),
        ("C=O", "CO", 4096),
        ("OO", "CO", 4096),
        ("C1CC1", "C1CCC1", 1),
    ],
)
def test_match_only_and_assignment_modes_have_status_parity(
    smarts: str,
    smiles: str,
    max_leaf_checks: int,
) -> None:
    _skip_if_no_cuda()

    batch = prepare_cuda_prune_batch([pattern(smarts)], [molecule(smiles)])
    match_only = match_fused_cuda(batch, max_leaf_checks=max_leaf_checks)
    assignments = match_fused_assignments_cuda(
        batch,
        max_leaf_checks=max_leaf_checks,
    )

    assert np.array_equal(
        match_only.status_codes_host(),
        assignments.status_codes_host(),
    )


@pytest.mark.gpu
@pytest.mark.cuda
def test_match_only_and_assignment_modes_handle_empty_cross_product() -> None:
    _skip_if_no_cuda()

    batch = prepare_cuda_prune_batch([pattern("CO")], [])
    match_only = match_fused_cuda(batch)
    assignments = match_fused_assignments_cuda(batch)

    assert match_only.status_codes_host().shape == (1, 0)
    assert assignments.status_codes_host().shape == (1, 0)
    assert assignments.assignment_required == 0


@pytest.mark.gpu
@pytest.mark.cuda
def test_match_only_and_assignment_modes_handle_empty_pattern() -> None:
    _skip_if_no_cuda()

    batch = prepare_cuda_prune_batch([_giant_graph(0)], [molecule("C")])
    match_only = match_fused_cuda(batch)
    assignments = match_fused_assignments_cuda(batch)

    assert match_only.status_labels_host().tolist() == [["yes"]]
    assert assignments.status_labels_host().tolist() == [["yes"]]
    assert assignments.assignment_counts_host().tolist() == [[1]]


@pytest.mark.gpu
@pytest.mark.cuda
def test_match_fused_cuda_handles_empty_cross_product() -> None:
    _skip_if_no_cuda()

    batch = prepare_cuda_prune_batch([pattern("CO")], [])
    result = match_fused_cuda(batch)

    assert result.status_codes_host().shape == (1, 0)
    assert result.prune_rounds_host().shape == (1, 0)
    assert result.pruning_complete_host().shape == (1, 0)
    assert result.search_complete_host().shape == (1, 0)
    assert result.leaf_checks_host().shape == (1, 0)


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("smarts", "smiles"),
    [
        ("CC", "CCC"),
        ("CO", "CCO"),
        ("C(C)C", "CC(C)C"),
        ("C1CC1", "C1CCC1"),
    ],
)
def test_match_fused_assignments_cuda_matches_cpu_assignments(
    smarts: str,
    smiles: str,
) -> None:
    _skip_if_no_cuda()

    query = pattern(smarts)
    target = molecule(smiles)
    expected = find_matches(query, target).assignments
    batch = prepare_cuda_prune_batch([query], [target])

    result = match_fused_assignments_cuda(
        batch,
        max_prune_rounds=64,
        max_leaf_checks=4096,
    )

    count = int(result.assignment_counts_host()[0, 0])
    observed = result.assignments_host()[:count, : query.atom_count]

    expected_status = "yes" if expected.size else "no"
    assert result.status_labels_host().tolist() == [[expected_status]]
    assert _assignment_set(observed) == _assignment_set(expected)


@pytest.mark.gpu
@pytest.mark.cuda
def test_match_fused_assignments_cuda_retries_capacity_overflow() -> None:
    _skip_if_no_cuda()

    query = pattern("CC")
    target = molecule("CCC")
    batch = prepare_cuda_prune_batch([query], [target])

    result = match_fused_assignments_cuda(
        batch,
        assignment_capacity=1,
        max_leaf_checks=4096,
    )

    assert result.status_labels_host().tolist() == [["yes"]]
    assert result.assignment_counts_host().tolist() == [[4]]
    assert result.assignment_required == 4
    assert result.assignment_capacity == 4
    assert result.overflow_retries == 1
    assert result.search_complete_host()[0, 0]


@pytest.mark.gpu
@pytest.mark.cuda
def test_assignment_overflow_retry_preserves_original_pair_ids() -> None:
    _skip_if_no_cuda()

    batch = prepare_cuda_prune_batch(
        [pattern("CC"), pattern("N#N")],
        [molecule("CCC")],
    )
    result = match_fused_assignments_cuda(
        batch,
        assignment_capacity=1,
        max_leaf_checks=4096,
    )

    assert result.status_labels_host().tolist() == [["yes"], ["no"]]
    assert result.assignment_pair_ids_host().tolist() == [0, 0, 0, 0]


@pytest.mark.gpu
@pytest.mark.cuda
def test_match_fused_assignments_cuda_rejects_incompatible_bond_types() -> None:
    _skip_if_no_cuda()

    batch = prepare_cuda_prune_batch([pattern("C=O")], [molecule("CO")])
    result = match_fused_assignments_cuda(
        batch,
        max_leaf_checks=64,
    )

    assert result.status_labels_host().tolist() == [["no"]]
    assert result.assignment_counts_host().tolist() == [[0]]

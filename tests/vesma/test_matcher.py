import numpy as np
import pytest

import vesma
from tests.datasets import load_retro_benchmark_corpus
from vesma import (
    MatchResult,
    cmol_from_smiles,
    find_matches,
    gmol_from_cmol,
    gmol_from_smarts,
    has_match,
    has_match_many,
)
from vesma.adjacency import bond_presence
from vesma.cpu_matcher import (
    enumerate_assignments,
    initialize_candidates,
    prune,
)
from vesma.matcher import keyed_match, keyed_match_folding


def pattern(smarts: str):
    return gmol_from_smarts(smarts)[0]


def molecule(smiles: str):
    return gmol_from_cmol(cmol_from_smiles(smiles)[0])


def topology(query, target) -> tuple[np.ndarray, np.ndarray]:
    return bond_presence(query.bonds), bond_presence(target.bonds)


def prepared_candidates(query, target, *, max_prune_rounds: int | None = None):
    candidates = initialize_candidates(query, target)
    pattern_edges, molecule_edges = topology(query, target)
    return (
        prune(
            candidates,
            pattern_edges,
            molecule_edges,
            max_prune_rounds=max_prune_rounds,
        ),
        pattern_edges,
        molecule_edges,
    )


def test_has_match_simple_pattern() -> None:
    assert has_match(pattern("CO"), molecule("CCO"))


def test_has_match_rejects_wrong_atom_type() -> None:
    assert not has_match(pattern("N"), molecule("CCO"))


def test_has_match_branch_pattern() -> None:
    assert has_match(pattern("C(O)C"), molecule("CC(O)C"))


def test_default_query_tightening_rejects_some_bond_order_mismatches() -> None:
    assert not has_match(pattern("C=C"), molecule("CC"))


def test_final_assignment_rejects_double_bond_pattern_on_single_bond() -> None:
    assert not has_match(pattern("[*]=[*]"), molecule("CC"))
    assert has_match(pattern("[*]=[*]"), molecule("C=C"))


def test_final_assignment_rejects_triple_bond_pattern_on_double_bond() -> None:
    assert not has_match(pattern("[*]#[*]"), molecule("C=C"))
    assert has_match(pattern("[*]#[*]"), molecule("C#C"))


@pytest.mark.parametrize("smiles", ["CC", "C=C", "C#C"])
def test_final_assignment_accepts_single_double_or_triple_bond_pattern(
    smiles: str,
) -> None:
    assert has_match(pattern("*!:*"), molecule(smiles))


def test_final_assignment_enforces_ring_bond_topology() -> None:
    # @ requires a ring bond, !@ requires a chain bond.
    assert not has_match(pattern("*-@*"), molecule("CC"))
    assert has_match(pattern("*-@*"), molecule("C1CCCCC1"))
    assert has_match(pattern("*-!@*"), molecule("CC"))
    assert not has_match(pattern("*-!@*"), molecule("C1CCCCC1"))


def test_final_assignment_rejects_aromatic_bond_pattern_on_aliphatic_bond() -> None:
    assert not has_match(pattern("*:*"), molecule("CC"))


def test_final_assignment_accepts_aromatic_bond_pattern_in_kekulized_ring() -> None:
    assert has_match(pattern("*:*"), molecule("c1ccccc1"))
    assert has_match(pattern("c:c"), molecule("c1ccccc1"))


def test_aromatic_bond_pattern_rejects_nonaromatic_bridge_between_aromatic_rings() -> (
    None
):
    assert not has_match(
        pattern("c1ccccc1:c2ccccc2"),
        molecule("c1ccccc1-c2ccccc2"),
    )


def test_final_assignment_accepts_double_or_aromatic_bond_pattern() -> None:
    query = pattern("*=,:*")

    assert not has_match(query, molecule("CC"))
    assert has_match(query, molecule("C=C"))
    assert has_match(query, molecule("c1ccccc1"))


def test_explicit_single_query_rejects_aromatic_ring_bond() -> None:
    # A biaryl/Suzuki coupling template uses an explicit ``-`` for the new bond formed
    # between two aromatic rings. It must NOT match an internal aromatic ring bond
    # (e.g. naphthalene's ring-fusion bond) -- that is not a couplable single bond.
    biaryl = pattern("c-c")
    assert not has_match(biaryl, molecule("c1ccc2ccccc2c1"))  # naphthalene
    assert not has_match(biaryl, molecule("c1ccccc1"))  # benzene ring bonds
    # ...but it MUST match a genuine inter-ring single bond (biphenyl).
    assert has_match(biaryl, molecule("c1ccccc1-c1ccccc1"))


def test_default_query_still_matches_aromatic_bonds() -> None:
    # An omitted/default bond means "single or aromatic", so it must still match
    # aromatic targets (the Daylight semantics RDKit follows).
    assert has_match(pattern("cc"), molecule("c1ccc2ccccc2c1"))
    assert has_match(pattern("[#6][#6]"), molecule("c1ccccc1"))
    # And an explicit double query must reject a Kekule double aromatic ring bond.
    assert not has_match(pattern("*=*"), molecule("c1ccccc1"))


def test_retro_benchmark_previous_relaxed_bond_false_positive_is_rejected() -> None:
    chem = pytest.importorskip("rdkit.Chem")
    corpus = load_retro_benchmark_corpus()

    benchmark_pattern = next(
        entry for entry in corpus.patterns if entry.template_code == "6"
    )
    benchmark_molecule = next(
        entry for entry in corpus.molecules if entry.chembl_id == "403458"
    )
    rd_query = chem.MolFromSmarts(benchmark_pattern.lhs_smarts)
    rd_mol = chem.MolFromSmiles(benchmark_molecule.smiles)

    assert rd_query is not None
    assert rd_mol is not None
    assert not rd_mol.HasSubstructMatch(rd_query)
    assert not has_match(benchmark_pattern.query, benchmark_molecule.molecule)


@pytest.mark.parametrize(
    ("smarts", "smiles"),
    [
        (
            "[#7](-c:1:c:c:c:c:c:1)-c2[n+]c(cs2)-c:3:c:c:c:c:c:3",
            "Clc1ccc(cc1)C=2[N+](=C(SC2)Nc3c(cccc3C)C)C",
        ),
        (
            "[ND1H2]a",
            "Fc1ccc(cc1)C2=Nc3c(cccc3)C(=O)N2N",
        ),
        (
            "[cD3H0;r5,r6;R2](:[cD3H0;r5,r6;R2])(:[cD3H0;r5,r6;R2])",
            "O=C1N(N=Cc2ccccc2)C(=O)C=3Nc4c(cc(OC)cc4)C3N1",
        ),
        (
            "C~C~C~C",
            "S1C(=O)NC=2S[C@H](C([O-])=O)[C@H]3C(=O)Oc4c(cccc4)[C@H]3C12",
        ),
    ],
)
def test_has_match_handles_kekule_aromatic_regressions(
    smarts: str, smiles: str
) -> None:
    assert has_match(pattern(smarts), molecule(smiles))


def test_has_match_many_iterates_pairs() -> None:
    patterns = [pattern("CO"), pattern("N")]
    molecules = [molecule("CCO"), molecule("CCN")]
    matches = has_match_many(patterns, molecules)
    assert np.array_equal(matches, np.array([[True, False], [False, True]]))


def test_has_match_many_parallel_matches_serial() -> None:
    patterns = [pattern("CO"), pattern("N")]
    molecules = [molecule("CCO"), molecule("CCN")]

    serial = has_match_many(patterns, molecules, n_jobs=1)
    parallel = has_match_many(patterns, molecules, n_jobs=2)

    assert np.array_equal(parallel, serial)


def test_initialize_candidates_no_longer_degree_filters() -> None:
    query = pattern("C(C)C")
    target = molecule("CC(C)C")
    candidates = initialize_candidates(query, target)

    # initialize_candidates matches on element/charge/aromaticity but NOT degree, so
    # every pattern carbon is a candidate for every target carbon; degree is enforced
    # later by topology pruning and the injective leaf check, not by initialization.
    assert np.all(candidates)


def test_prune_stops_early_when_bounded() -> None:
    # OCCN vs NCCO needs two refinement rounds to reach a fixpoint (round 1 prunes the
    # O/N endpoint candidates, which only then lets round 2 prune their neighbours), so
    # bounding to a single round must stop before completion.
    query = pattern("OCCN")
    target = molecule("NCCO")
    candidates = initialize_candidates(query, target)
    pattern_edges, molecule_edges = topology(query, target)

    pruned, prunings, pruning_complete = prune(
        candidates,
        pattern_edges,
        molecule_edges,
        max_prune_rounds=1,
    )

    assert pruned.shape == candidates.shape
    assert prunings == 1
    assert not pruning_complete


def test_prune_reaches_fixpoint_by_default() -> None:
    query = pattern("C(C)C")
    target = molecule("CC(C)C")
    candidates = initialize_candidates(query, target)
    pattern_edges, molecule_edges = topology(query, target)

    pruned, prunings, pruning_complete = prune(
        candidates,
        pattern_edges,
        molecule_edges,
    )

    assert pruned.shape == candidates.shape
    assert prunings >= 1
    assert pruning_complete


def test_stage_three_enumerates_injective_assignments() -> None:
    query = pattern("[C:1][C:2]")
    target = molecule("CCC")
    (candidates, _, _), _, _ = prepared_candidates(query, target)

    result = enumerate_assignments(candidates, query.bonds, target.bonds)

    assert result.assignments.shape[1] == 2
    assert result.assignments.shape[0] >= 2


def test_find_matches_defaults_wrap_the_full_pipeline() -> None:
    query = pattern("[C:1][C:2]")
    target = molecule("CCC")

    result = find_matches(query, target)

    assert isinstance(result, MatchResult)
    assert result.status == "match"
    assert result.pruning_complete
    assert result.search_complete
    assert result.metadata["branches_explored"] >= 0
    assert result.leaves_explored >= result.assignments.shape[0]
    assert has_match(query, target)


def test_prefix_depth_variants_agree_on_positive_case() -> None:
    query = pattern("[C:1][C:2]")
    target = molecule("CCC")
    reference = find_matches(query, target)

    for prefix_depth in (0, 1, 2, None):
        result = find_matches(query, target, prefix_depth=prefix_depth)
        assert np.array_equal(result.assignments, reference.assignments)
        assert result.leaves_explored >= result.assignments.shape[0]


def test_prefix_depth_variants_agree_on_negative_case() -> None:
    query = pattern("C1CC1")
    target = molecule("C1CCC1")

    for prefix_depth in (0, 1, 2, None):
        result = find_matches(query, target, prefix_depth=prefix_depth)
        assert result.status == "no_match"
        assert result.search_complete


def test_pruning_bound_can_return_unknown() -> None:
    result = find_matches(
        pattern("C1CC1"),
        molecule("C1CCC1"),
        max_prune_rounds=1,
    )

    assert result.status == "unknown"
    assert not result.pruning_complete
    assert result.prunings >= 1


def test_leaf_bound_can_return_unknown() -> None:
    result = find_matches(
        pattern("C1CC1"),
        molecule("C1CCC1"),
        prefix_depth=0,
        max_leaf_checks=1,
    )

    assert result.status == "unknown"
    assert result.pruning_complete
    assert not result.search_complete
    assert result.leaves_explored == 1


def test_leaf_bound_is_global_across_dfs_singleton_hits() -> None:
    result = find_matches(
        pattern("[C:1][C:2]"),
        molecule("CCC"),
        prefix_depth=1,
        max_leaf_checks=1,
    )

    assert result.status == "match"
    assert result.pruning_complete
    assert not result.search_complete
    assert result.leaves_explored == 1
    assert np.array_equal(result.assignments, np.array([[0, 1]], dtype=np.int32))


def test_return_first_truncates_results() -> None:
    result = find_matches(
        pattern("[C:1][C:2]"),
        molecule("CCC"),
        prefix_depth=0,
        return_first=True,
    )

    assert result.status == "match"
    assert result.pruning_complete
    assert not result.search_complete
    assert np.array_equal(result.assignments, np.array([[0, 1]], dtype=np.int32))


def test_rejects_invalid_bounds() -> None:
    query = pattern("CC")
    target = molecule("CCC")
    candidates = initialize_candidates(query, target)

    with pytest.raises(ValueError, match="prefix_depth"):
        enumerate_assignments(candidates, query.bonds, target.bonds, prefix_depth=-1)

    with pytest.raises(ValueError, match="max_prune_rounds"):
        enumerate_assignments(
            candidates,
            query.bonds,
            target.bonds,
            max_prune_rounds=0,
        )

    with pytest.raises(ValueError, match="max_leaf_checks"):
        enumerate_assignments(
            candidates,
            query.bonds,
            target.bonds,
            max_leaf_checks=0,
        )


def test_prefix_depth_zero_uses_deterministic_root_order() -> None:
    query = pattern("[C:1][C:2]")
    target = molecule("CCC")
    (candidates, _, _), _, _ = prepared_candidates(query, target)

    result = enumerate_assignments(
        candidates,
        query.bonds,
        target.bonds,
        prefix_depth=0,
    )

    expected = np.array([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=np.int32)
    assert np.array_equal(result.assignments, expected)


def test_root_api_exports_only_minimal_matcher_surface() -> None:
    assert vesma.find_matches is find_matches
    assert vesma.has_match is has_match
    assert vesma.has_match_many is has_match_many
    assert vesma.MatchResult is MatchResult
    assert not hasattr(vesma, "initialize_candidates")
    assert not hasattr(vesma, "prune")
    assert not hasattr(vesma, "enumerate_assignments")


# --------------------------------------------------------------------------- #
# keyed_match_folding: identity folded in as batches arrive
# --------------------------------------------------------------------------- #
def _batched(*pieces: tuple[list[str], list[tuple[int, int, int, int]]]):
    """A `BatchedMatch` over hand-written pieces: molecules as letters, hits as rows."""

    def run(_mols: object):
        for molecules, hits in pieces:
            yield (
                np.array(molecules, dtype=object),
                np.array(hits, dtype=np.int64).reshape(-1, 4),
            )

    return run


def _letters(batch: np.ndarray) -> list[str]:
    return [str(value) for value in batch]


def _concat(parts):
    return np.concatenate(parts) if parts else np.array([], dtype=object)


def test_folding_keyed_match_collapses_duplicate_keys_within_a_batch() -> None:
    keyed = keyed_match_folding(
        _batched((["A", "B", "A"], [(0, 0, 0, 0), (0, 0, 0, 1), (0, 1, 1, 2)])),
        _letters,
        _concat,
    )
    batch, keys, hits = keyed([])

    # Three precursors, two molecules: the third row is the first molecule again.
    assert list(keys) == ["A", "B"]
    assert _letters(batch) == ["A", "B"]
    # Every hit still exists and names the row holding its molecule.
    assert hits.shape == (3, 4)
    assert [_letters(batch)[row] for row in hits[:, 3]] == ["A", "B", "A"]


def test_folding_keyed_match_collapses_across_batches() -> None:
    """A repeat in a later piece points back at the row an earlier piece minted."""
    keyed = keyed_match_folding(
        _batched(
            (["A", "B"], [(0, 0, 0, 0), (0, 0, 0, 1)]),
            (["C", "A"], [(1, 0, 2, 0), (1, 0, 2, 1)]),
        ),
        _letters,
        _concat,
    )
    batch, keys, hits = keyed([])

    assert list(keys) == ["A", "B", "C"]
    assert _letters(batch) == ["A", "B", "C"]
    # The second piece's "A" was row 1 locally and resolves to row 0 globally.
    assert hits[:, 3].tolist() == [0, 1, 2, 0]
    # Columns 0-2 are the backend's and are passed through untouched.
    assert hits[:, :3].tolist() == [[0, 0, 0], [0, 0, 0], [1, 0, 2], [1, 0, 2]]


def test_folding_keyed_match_leaves_no_orphan_rows() -> None:
    """The contract every matcher owes: each returned row is named by at least one hit."""
    keyed = keyed_match_folding(
        _batched(
            (["A", "B"], [(0, 0, 0, 0), (0, 0, 0, 1)]),
            (["C"], [(1, 1, 1, 0)]),
        ),
        _letters,
        _concat,
    )
    batch, _keys, hits = keyed([])

    assert set(hits[:, 3].tolist()) == set(range(len(batch)))


def test_folding_keyed_match_matches_keyed_match_on_a_single_piece() -> None:
    """One piece with distinct keys is the undeduplicated case, and must agree exactly."""
    molecules = ["A", "B", "C"]
    rows = [(0, 0, 0, 0), (0, 0, 0, 1), (0, 1, 1, 2)]

    folded = keyed_match_folding(_batched((molecules, rows)), _letters, _concat)
    plain = keyed_match(
        lambda _mols: (
            np.array(molecules, dtype=object),
            np.array(rows, dtype=np.int64),
        ),
        _letters,
    )

    folded_batch, folded_keys, folded_hits = folded([])
    plain_batch, plain_keys, plain_hits = plain([])
    assert _letters(folded_batch) == _letters(plain_batch)
    assert list(folded_keys) == list(plain_keys)
    assert np.array_equal(folded_hits, plain_hits)


def test_folding_keyed_match_handles_a_source_that_yields_nothing() -> None:
    batch, keys, hits = keyed_match_folding(_batched(), _letters, _concat)([])
    assert len(batch) == 0
    assert list(keys) == []
    assert hits.shape == (0, 4)


def test_folding_keyed_match_rejects_a_keyer_that_miscounts() -> None:
    keyed = keyed_match_folding(
        _batched((["A", "B"], [(0, 0, 0, 0)])),
        lambda _batch: ["only-one"],
        _concat,
    )
    with pytest.raises(ValueError, match="one key per precursor row"):
        keyed([])

import numpy as np
import pytest

from vesma import cmol_from_smiles, find_matches, gmol_from_cmol, gmol_from_smarts
from vesma.adjacency import bond_presence
from vesma.cpu_matcher import enumerate_assignments, initialize_candidates, prune

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
    signature_prefilter,
    unpack_rows_u32,
    unpack_rows_u64,
)


def pattern(smarts: str):
    return gmol_from_smarts(smarts)[0]


def molecule(smiles: str):
    return gmol_from_cmol(cmol_from_smiles(smiles)[0])


def prepared_case(smarts: str, smiles: str):
    query = pattern(smarts)
    target = molecule(smiles)
    candidates = initialize_candidates(query, target)
    pattern_edges = bond_presence(query.bonds)
    molecule_edges = bond_presence(target.bonds)
    return query, target, candidates, pattern_edges, molecule_edges


@pytest.mark.parametrize(
    ("smarts", "smiles"),
    [
        ("[C:1][C:2]", "CCC"),
        ("C(C)C", "CC(C)C"),
        ("C1CCCCC1", "C1CCCCC1"),
        ("C1CC1", "C1CCC1"),
        ("c1ccccc1N", "CCc1ccccc1N"),
    ],
)
def test_matcher_prune_matches_dense_gemm(smarts: str, smiles: str) -> None:
    _, _, candidates, pattern_edges, molecule_edges = prepared_case(smarts, smiles)

    expected = prune(candidates, pattern_edges, molecule_edges)
    observed_gemm = prune_dense_gemm(candidates, pattern_edges, molecule_edges)

    assert observed_gemm[1:] == expected[1:]
    assert np.array_equal(observed_gemm[0], expected[0])


def test_matcher_prune_matches_dense_gemm_on_bounded_prune_rounds() -> None:
    _, _, candidates, pattern_edges, molecule_edges = prepared_case("C(C)C", "CC(C)C")

    expected = prune(
        candidates,
        pattern_edges,
        molecule_edges,
        max_prune_rounds=1,
    )
    observed = prune_dense_gemm(
        candidates,
        pattern_edges,
        molecule_edges,
        max_prune_rounds=1,
    )

    assert observed[1:] == expected[1:]
    assert np.array_equal(observed[0], expected[0])


@pytest.mark.parametrize(
    ("smarts", "smiles"),
    [
        ("[C:1][C:2]", "CCC"),
        ("C(C)C", "CC(C)C"),
        ("C1CCCCC1", "C1CCCCC1"),
        ("C1CC1", "C1CCC1"),
        ("c1ccccc1N", "CCc1ccccc1N"),
    ],
)
def test_dense_gemm_matches_dense_reference(smarts: str, smiles: str) -> None:
    _, _, candidates, pattern_edges, molecule_edges = prepared_case(smarts, smiles)
    expected = prune_dense_reference(candidates, pattern_edges, molecule_edges)
    observed = prune_dense_gemm(candidates, pattern_edges, molecule_edges)

    assert observed[1:] == expected[1:]
    assert np.array_equal(observed[0], expected[0])


@pytest.mark.parametrize(
    ("smarts", "smiles"),
    [
        ("[C:1][C:2]", "CCC"),
        ("C(C)C", "CC(C)C"),
        ("C1CCCCC1", "C1CCCCC1"),
        ("C1CC1", "C1CCC1"),
        ("c1ccccc1N", "CCc1ccccc1N"),
    ],
)
def test_u32_and_u64_bitset_match_dense(smarts: str, smiles: str) -> None:
    _, _, candidates, pattern_edges, molecule_edges = prepared_case(smarts, smiles)
    expected = prune_dense_reference(candidates, pattern_edges, molecule_edges)

    observed_u32 = prune_bitset_u32(candidates, pattern_edges, molecule_edges)
    observed_u64 = prune_bitset_u64(candidates, pattern_edges, molecule_edges)
    observed_dirty_u32 = prune_bitset_dirty_u32(
        candidates, pattern_edges, molecule_edges
    )
    observed_dirty_u64 = prune_bitset_dirty_u64(
        candidates, pattern_edges, molecule_edges
    )
    observed_hybrid_u32 = prune_bitset_hybrid_u32(
        candidates,
        pattern_edges,
        molecule_edges,
    )
    observed_hybrid_u64 = prune_bitset_hybrid_u64(
        candidates,
        pattern_edges,
        molecule_edges,
    )
    observed_popcount_u64 = prune_bitset_popcount_u64(
        candidates,
        pattern_edges,
        molecule_edges,
    )
    observed_popcount_block16_u64 = prune_bitset_popcount_block16_u64(
        candidates,
        pattern_edges,
        molecule_edges,
    )
    observed_popcount_block32_u64 = prune_bitset_popcount_block32_u64(
        candidates,
        pattern_edges,
        molecule_edges,
    )
    observed_popcount_block64_u64 = prune_bitset_popcount_block64_u64(
        candidates,
        pattern_edges,
        molecule_edges,
    )
    observed_vectorized_u32 = prune_bitset_vectorized_u32(
        candidates,
        pattern_edges,
        molecule_edges,
    )
    observed_vectorized_u64 = prune_bitset_vectorized_u64(
        candidates,
        pattern_edges,
        molecule_edges,
    )

    for observed in (
        observed_u32,
        observed_u64,
        observed_dirty_u32,
        observed_dirty_u64,
        observed_hybrid_u32,
        observed_hybrid_u64,
        observed_popcount_u64,
        observed_popcount_block16_u64,
        observed_popcount_block32_u64,
        observed_popcount_block64_u64,
        observed_vectorized_u32,
        observed_vectorized_u64,
    ):
        assert observed[1:] == expected[1:]
        assert np.array_equal(observed[0], expected[0])


def test_u32_and_u64_match_on_bounded_prune_rounds() -> None:
    _, _, candidates, pattern_edges, molecule_edges = prepared_case("C(C)C", "CC(C)C")
    expected = prune_dense_reference(
        candidates,
        pattern_edges,
        molecule_edges,
        max_prune_rounds=1,
    )

    observed_u32 = prune_bitset_u32(
        candidates,
        pattern_edges,
        molecule_edges,
        max_prune_rounds=1,
    )
    observed_u64 = prune_bitset_u64(
        candidates,
        pattern_edges,
        molecule_edges,
        max_prune_rounds=1,
    )
    observed_dirty_u32 = prune_bitset_dirty_u32(
        candidates,
        pattern_edges,
        molecule_edges,
        max_prune_rounds=1,
    )
    observed_dirty_u64 = prune_bitset_dirty_u64(
        candidates,
        pattern_edges,
        molecule_edges,
        max_prune_rounds=1,
    )
    observed_hybrid_u32 = prune_bitset_hybrid_u32(
        candidates,
        pattern_edges,
        molecule_edges,
        max_prune_rounds=1,
    )
    observed_hybrid_u64 = prune_bitset_hybrid_u64(
        candidates,
        pattern_edges,
        molecule_edges,
        max_prune_rounds=1,
    )
    observed_popcount_u64 = prune_bitset_popcount_u64(
        candidates,
        pattern_edges,
        molecule_edges,
        max_prune_rounds=1,
    )
    observed_popcount_block16_u64 = prune_bitset_popcount_block16_u64(
        candidates,
        pattern_edges,
        molecule_edges,
        max_prune_rounds=1,
    )
    observed_popcount_block32_u64 = prune_bitset_popcount_block32_u64(
        candidates,
        pattern_edges,
        molecule_edges,
        max_prune_rounds=1,
    )
    observed_popcount_block64_u64 = prune_bitset_popcount_block64_u64(
        candidates,
        pattern_edges,
        molecule_edges,
        max_prune_rounds=1,
    )
    observed_vectorized_u32 = prune_bitset_vectorized_u32(
        candidates,
        pattern_edges,
        molecule_edges,
        max_prune_rounds=1,
    )
    observed_vectorized_u64 = prune_bitset_vectorized_u64(
        candidates,
        pattern_edges,
        molecule_edges,
        max_prune_rounds=1,
    )

    for observed in (
        observed_u32,
        observed_u64,
        observed_dirty_u32,
        observed_dirty_u64,
        observed_hybrid_u32,
        observed_hybrid_u64,
        observed_popcount_u64,
        observed_popcount_block16_u64,
        observed_popcount_block32_u64,
        observed_popcount_block64_u64,
        observed_vectorized_u32,
        observed_vectorized_u64,
    ):
        assert observed[1:] == expected[1:]
        assert np.array_equal(observed[0], expected[0])


@pytest.mark.parametrize(
    ("smarts", "smiles"),
    [
        ("C(C)C", "CC(C)O"),
        ("[C:1][C:2]", "CCC"),
        ("C1CC1", "C1CCC1"),
        ("c1ccccc1N", "CCc1ccccc1N"),
    ],
)
def test_signature_prefilter_preserves_match_status(smarts: str, smiles: str) -> None:
    query, target, candidates, pattern_edges, molecule_edges = prepared_case(
        smarts,
        smiles,
    )
    filtered = signature_prefilter(query, target, candidates)
    filtered_pruned = prune_dense_reference(filtered, pattern_edges, molecule_edges)
    filtered_result = enumerate_assignments(
        filtered_pruned[0],
        query.bonds,
        target.bonds,
        pattern_aromaticity=query.atom_masks.A,
        molecule_aromaticity=target.atom_masks.A,
        return_first=True,
    )
    baseline = find_matches(query, target, return_first=True)

    assert np.all(filtered <= candidates)
    assert filtered_result.status == baseline.status


@pytest.mark.parametrize(
    ("smarts", "smiles"),
    [
        ("C(C)C", "CC(C)O"),
        ("[C:1][C:2]", "CCC"),
        ("C1CC1", "C1CCC1"),
        ("c1ccccc1N", "CCc1ccccc1N"),
    ],
)
def test_signature_bitset_preserves_match_status(smarts: str, smiles: str) -> None:
    query, target, candidates, pattern_edges, molecule_edges = prepared_case(
        smarts,
        smiles,
    )
    baseline = find_matches(query, target, return_first=True)

    for prune_fn in (prune_signature_bitset_u32, prune_signature_bitset_u64):
        observed = prune_fn(
            query,
            target,
            candidates,
            pattern_edges,
            molecule_edges,
        )
        result = enumerate_assignments(
            observed[0],
            query.bonds,
            target.bonds,
            pattern_aromaticity=query.atom_masks.A,
            molecule_aromaticity=target.atom_masks.A,
            return_first=True,
        )
        assert result.status == baseline.status


@pytest.mark.parametrize(
    ("smarts", "smiles"),
    [
        ("C(C)C", "CC(C)O"),
        ("[C:1][C:2]", "CCC"),
        ("C1CC1", "C1CCC1"),
        ("c1ccccc1N", "CCc1ccccc1N"),
    ],
)
def test_signature_vectorized_preserves_match_status(smarts: str, smiles: str) -> None:
    query, target, candidates, pattern_edges, molecule_edges = prepared_case(
        smarts,
        smiles,
    )
    baseline = find_matches(query, target, return_first=True)

    for prune_fn in (prune_signature_vectorized_u32, prune_signature_vectorized_u64):
        observed = prune_fn(
            query,
            target,
            candidates,
            pattern_edges,
            molecule_edges,
        )
        result = enumerate_assignments(
            observed[0],
            query.bonds,
            target.bonds,
            pattern_aromaticity=query.atom_masks.A,
            molecule_aromaticity=target.atom_masks.A,
            return_first=True,
        )
        assert result.status == baseline.status


@pytest.mark.parametrize(
    ("smarts", "smiles"),
    [
        ("C(C)C", "CC(C)O"),
        ("[C:1][C:2]", "CCC"),
        ("C1CC1", "C1CCC1"),
        ("c1ccccc1N", "CCc1ccccc1N"),
    ],
)
def test_signature_hybrid_preserves_match_status(smarts: str, smiles: str) -> None:
    query, target, candidates, pattern_edges, molecule_edges = prepared_case(
        smarts,
        smiles,
    )
    baseline = find_matches(query, target, return_first=True)

    for prune_fn in (prune_signature_hybrid_u32, prune_signature_hybrid_u64):
        observed = prune_fn(
            query,
            target,
            candidates,
            pattern_edges,
            molecule_edges,
        )
        result = enumerate_assignments(
            observed[0],
            query.bonds,
            target.bonds,
            pattern_aromaticity=query.atom_masks.A,
            molecule_aromaticity=target.atom_masks.A,
            return_first=True,
        )
        assert result.status == baseline.status


def test_pack_rows_roundtrip_u32_and_u64() -> None:
    matrix = np.array(
        [
            [True, False, True, False, False, True],
            [False, True, False, True, True, False],
        ],
        dtype=bool,
    )

    packed_u32 = pack_rows_u32(matrix)
    packed_u64 = pack_rows_u64(matrix)

    assert np.array_equal(unpack_rows_u32(packed_u32, matrix.shape[1]), matrix)
    assert np.array_equal(unpack_rows_u64(packed_u64, matrix.shape[1]), matrix)


def test_u32_rejects_overwide_candidate_matrices() -> None:
    long_smarts = "C" * 33
    query, target, candidates, pattern_edges, molecule_edges = prepared_case(
        long_smarts,
        long_smarts,
    )

    assert query.atom_count == 33
    assert target.atom_count == 33
    with pytest.raises(ValueError, match="32 bits"):
        prune_bitset_u32(candidates, pattern_edges, molecule_edges)

    observed_u64 = prune_bitset_u64(candidates, pattern_edges, molecule_edges)
    expected = prune_dense_reference(candidates, pattern_edges, molecule_edges)
    assert observed_u64[1:] == expected[1:]
    assert np.array_equal(observed_u64[0], expected[0])

    observed_vectorized_u64 = prune_bitset_vectorized_u64(
        candidates,
        pattern_edges,
        molecule_edges,
    )
    assert observed_vectorized_u64[1:] == expected[1:]
    assert np.array_equal(observed_vectorized_u64[0], expected[0])

    observed_hybrid_u64 = prune_bitset_hybrid_u64(
        candidates,
        pattern_edges,
        molecule_edges,
    )
    assert observed_hybrid_u64[1:] == expected[1:]
    assert np.array_equal(observed_hybrid_u64[0], expected[0])

    observed_popcount_u64 = prune_bitset_popcount_u64(
        candidates,
        pattern_edges,
        molecule_edges,
    )
    assert observed_popcount_u64[1:] == expected[1:]
    assert np.array_equal(observed_popcount_u64[0], expected[0])

    observed_popcount_block16_u64 = prune_bitset_popcount_block16_u64(
        candidates,
        pattern_edges,
        molecule_edges,
    )
    assert observed_popcount_block16_u64[1:] == expected[1:]
    assert np.array_equal(observed_popcount_block16_u64[0], expected[0])

    observed_popcount_block32_u64 = prune_bitset_popcount_block32_u64(
        candidates,
        pattern_edges,
        molecule_edges,
    )
    assert observed_popcount_block32_u64[1:] == expected[1:]
    assert np.array_equal(observed_popcount_block32_u64[0], expected[0])

    observed_popcount_block64_u64 = prune_bitset_popcount_block64_u64(
        candidates,
        pattern_edges,
        molecule_edges,
    )
    assert observed_popcount_block64_u64[1:] == expected[1:]
    assert np.array_equal(observed_popcount_block64_u64[0], expected[0])

    with pytest.raises(ValueError, match="32 bits"):
        prune_bitset_vectorized_u32(candidates, pattern_edges, molecule_edges)
    with pytest.raises(ValueError, match="32 bits"):
        prune_bitset_hybrid_u32(candidates, pattern_edges, molecule_edges)


def test_u64_popcount_rejects_overwide_candidate_matrices() -> None:
    long_smarts = "C" * 65
    _, _, candidates, pattern_edges, molecule_edges = prepared_case(
        long_smarts,
        long_smarts,
    )

    for prune_fn in (
        prune_bitset_popcount_u64,
        prune_bitset_popcount_block16_u64,
        prune_bitset_popcount_block32_u64,
        prune_bitset_popcount_block64_u64,
    ):
        with pytest.raises(ValueError, match="64 bits"):
            prune_fn(candidates, pattern_edges, molecule_edges)

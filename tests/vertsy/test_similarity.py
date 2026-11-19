from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit.Chem")

from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors

from vertsy.heuristics import strongest
from vertsy.similarity import (
    FINGERPRINTS,
    atom_pair_generator,
    ecfp4_matrix,
    ecfp4_matrix_from_smiles,
    fingerprint_matrix,
    morgan_generator,
    stock_similarity,
)
from vesma.keyer import rdkit_mol_canonical_smiles, rdkit_token_from_smiles

STOCK = ("CCO", "c1ccccc1", "CC(=O)O", "CCN", "c1ccncc1")


def mols(smiles: tuple[str, ...] | list[str]) -> list[Chem.Mol]:
    """The heuristic scores molecules, so a test that starts from text has to parse first."""
    return [Chem.MolFromSmiles(smi) for smi in smiles]


def test_stock_must_be_keyed_the_way_precursors_are() -> None:
    """Stock membership is decided in token space, so raw file SMILES cannot be the key.

    A generated precursor is keyed with `rdkit_mol_canonical_smiles`, which drops stereochemistry. Stock
    read straight from a `.smi` file keeps it. Comparing the two silently hides every
    stereochemical stock entry from the search: it is never found buyable, so it is expanded
    forever and routes through it never solve. On PaRoutes n1 that is 9.5% of the stock.
    """
    stereo = "C[C@H](N)C(=O)O"
    stereo_token = rdkit_token_from_smiles(stereo)
    assert stereo_token != stereo, "fixture must carry stereochemistry"

    # What the search actually compares against stock: the key of a generated precursor.
    precursor_key = rdkit_mol_canonical_smiles(Chem.MolFromSmiles(stereo))

    raw_stock = frozenset({stereo})  # stock straight off disk
    keyed_stock = frozenset(
        {rdkit_token_from_smiles(stereo)}
    )  # stock through the keyer

    assert precursor_key not in raw_stock, "buyable molecule looks unbuyable"
    assert precursor_key in keyed_stock


def test_fingerprint_rows_keep_their_shape() -> None:
    matrix = ecfp4_matrix(mols(STOCK), n_bits=2048)
    assert matrix.shape == (len(STOCK), 2048)
    assert matrix.dtype == np.float32
    assert set(np.unique(matrix)) <= {0.0, 1.0}


def test_unparseable_smiles_raises_instead_of_corrupting_the_batch() -> None:
    """Fail fast and say which molecule, rather than yield a wrong-shaped row.

    The previous implementation wrote a zero row of the wrong length for anything RDKit
    could not parse, which surfaced as a shape error from `vstack` several frames away.
    """
    with pytest.raises(ValueError, match="cannot parse SMILES at index 1"):
        ecfp4_matrix_from_smiles(["CCO", "not_a_molecule", "CCN"])


def test_a_none_molecule_raises_instead_of_corrupting_the_batch() -> None:
    """The molecule-space equivalent: a caller that hands over a failed parse hears about it."""
    with pytest.raises(ValueError, match="molecule at index 1 is None"):
        ecfp4_matrix([Chem.MolFromSmiles("CCO"), None, Chem.MolFromSmiles("CCN")])


def test_similarity_matches_rdkit_tanimoto_exactly() -> None:
    """The matmul is an optimisation, not an approximation."""
    queries = ["CCOC", "c1ccc(O)cc1"]
    scored = stock_similarity(STOCK, weight=1.0)(mols(queries))

    stock_fps = [
        rdMolDescriptors.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(s), 2, nBits=2048
        )
        for s in STOCK
    ]
    for index, smi in enumerate(queries):
        query_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(smi), 2, nBits=2048
        )
        expected = 1.0 - max(DataStructs.BulkTanimotoSimilarity(query_fp, stock_fps))
        assert scored[index] == pytest.approx(expected, abs=1e-6)


def test_similarity_vanishes_at_the_stock() -> None:
    """The property that makes this a heuristic at all: zero distance once buyable."""
    scored = stock_similarity(STOCK, weight=1.0)(mols(STOCK))
    assert scored == pytest.approx(np.zeros(len(STOCK)), abs=1e-6)


def test_similarity_weight_scales_the_estimate() -> None:
    """`weight` is the admissibility knob, and at 1.0 the estimate never exceeds a reaction."""
    query = ["CCCCCCCCCCCCCCCCCC(=O)N1CCC(c2ccccc2)CC1"]
    plain = stock_similarity(STOCK, weight=1.0)(mols(query))
    scaled = stock_similarity(STOCK, weight=10.0)(mols(query))

    assert 0.0 <= plain[0] <= 1.0
    assert scaled[0] == pytest.approx(10.0 * plain[0], rel=1e-6)


def test_similarity_rejects_an_empty_stock() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        stock_similarity([], weight=1.0)


def test_similarity_scores_an_empty_query_batch() -> None:
    assert stock_similarity(STOCK, weight=1.0)([]).shape == (0,)


@pytest.mark.parametrize("kind", sorted(FINGERPRINTS))
def test_every_fingerprint_vanishes_at_the_stock(kind: str) -> None:
    """The one property a heuristic cannot trade away, checked per view rather than assumed."""
    scored = stock_similarity(STOCK, weight=1.0, fingerprint=kind)(mols(STOCK))
    assert scored == pytest.approx(np.zeros(len(STOCK)), abs=1e-6)
    # And a bit matrix that is neither empty nor saturated, or the distance would be degenerate.
    matrix = fingerprint_matrix(mols(STOCK), kind)
    assert 0 < matrix.sum() < matrix.size


def test_an_unknown_fingerprint_is_named_in_the_error() -> None:
    with pytest.raises(ValueError, match="unknown fingerprint"):
        fingerprint_matrix(mols(STOCK), "circular-ish")


def test_strongest_takes_the_tightest_bound_and_keeps_both_properties() -> None:
    """`strongest` must dominate each part, vanish at the stock, and never sum.

    The domination is what makes it the right combinator: a maximum of admissible estimates is
    still admissible, and larger is better informed for a lower bound. Summing would be neither.
    """
    query = ["CCCCCCCCCCCCCCCCCC(=O)N1CCC(c2ccccc2)CC1", "CCOC"]
    parts = [
        stock_similarity(STOCK, weight=1.0, fingerprint=kind)
        for kind in ("ecfp4", "maccs", "path")
    ]
    scored = [part(mols(query)) for part in parts]
    combined = strongest(*parts)(mols(query))

    assert combined == pytest.approx(np.maximum.reduce(scored))
    for one in scored:
        assert np.all(combined >= one - 1e-6)  # dominates every part
        assert np.all(combined <= np.sum(scored, axis=0) + 1e-6)  # and is not their sum
    # Still zero on the stock, so the contract survives the combination.
    assert strongest(*parts)(mols(STOCK)) == pytest.approx(
        np.zeros(len(STOCK)), abs=1e-6
    )
    assert strongest(*parts)([]).shape == (0,)


def test_strongest_needs_something_to_combine() -> None:
    with pytest.raises(ValueError, match="at least one"):
        strongest()


def test_one_neighbour_is_the_incumbent_and_more_is_an_average() -> None:
    """`neighbours=1` must be the old behaviour exactly, and k>1 the mean of the k nearest."""
    query = ["CCOC", "CCCCCCCCCCCCCCCCCC(=O)N1CCC(c2ccccc2)CC1"]
    nearest = stock_similarity(STOCK, weight=1.0)(mols(query))
    explicit = stock_similarity(STOCK, weight=1.0, neighbours=1)(mols(query))
    assert nearest == pytest.approx(explicit)

    # Averaging over more of the catalogue can only move the estimate away from the best match,
    # and the best match is the smallest distance, so the mean is never smaller.
    for k in (2, 3, len(STOCK)):
        averaged = stock_similarity(STOCK, weight=1.0, neighbours=k)(mols(query))
        assert np.all(averaged >= nearest - 1e-6)
    # Asking for more neighbours than the stock holds is clamped, not an error.
    assert stock_similarity(STOCK, weight=1.0, neighbours=999)(mols(query)).shape == (
        2,
    )
    with pytest.raises(ValueError, match="at least one"):
        stock_similarity(STOCK, weight=1.0, neighbours=0)


def test_idf_reweights_bits_without_breaking_the_contract() -> None:
    """Rarity weighting changes the estimate but not what a heuristic has to be."""
    plain = stock_similarity(STOCK, weight=1.0)
    weighted = stock_similarity(STOCK, weight=1.0, idf=True)
    query = ["CCOC(=O)c1ccccc1", "CCCCCCCCCCCCCCCCCC(=O)N1CCC(c2ccccc2)CC1"]

    assert not np.allclose(plain(mols(query)), weighted(mols(query)))
    # Still zero on the stock, and still inside [0, weight].
    assert weighted(mols(STOCK)) == pytest.approx(np.zeros(len(STOCK)), abs=1e-6)
    scored = weighted(mols(query))
    assert np.all(scored >= -1e-6) and np.all(scored <= 1.0 + 1e-6)


def test_the_generators_reproduce_the_deprecated_calls_they_replaced() -> None:
    """Switching to RDKit's fingerprint generators must not move any estimate.

    :data:`FINGERPRINTS` used to call the ``rdMolDescriptors`` helpers directly; those are
    deprecated in favour of generator objects, and a generator is also reusable, which matters
    when the caller fingerprints a million-row catalogue. But every heuristic weight, every
    admissibility bound and every benchmark number downstream is a function of these bits, so the
    replacement is only legitimate if it is bit-for-bit the same fingerprint.
    """
    probes = mols(
        [
            "CCOC(=O)c1ccccc1",
            "Cn1nc(C2CCC(=O)NC2=O)c2ccc(N3CCNCC3)cc21",
            "O=S(=O)(N)c1ccc(Cl)cc1",
            "C1CNCCN1",
        ]
    )
    n_bits = 2048
    for mol in probes:
        assert list(FINGERPRINTS["ecfp4"](mol, n_bits).GetOnBits()) == list(
            rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, 2, nBits=n_bits
            ).GetOnBits()
        )
        assert FINGERPRINTS["ecfp4-count"](mol, n_bits).GetNonzeroElements() == (
            rdMolDescriptors.GetHashedMorganFingerprint(
                mol, 2, nBits=n_bits
            ).GetNonzeroElements()
        )
        assert list(FINGERPRINTS["atompair"](mol, n_bits).GetOnBits()) == list(
            rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
                mol, nBits=n_bits
            ).GetOnBits()
        )
        assert FINGERPRINTS["atompair-count"](mol, n_bits).GetNonzeroElements() == (
            rdMolDescriptors.GetHashedAtomPairFingerprint(
                mol, nBits=n_bits
            ).GetNonzeroElements()
        )


def test_a_generator_is_built_once_per_width_and_reused() -> None:
    """The reason for the cache: one generator per width, not one per molecule."""
    assert morgan_generator(2048) is morgan_generator(2048)
    assert morgan_generator(2048) is not morgan_generator(1024)
    assert atom_pair_generator(2048) is atom_pair_generator(2048)

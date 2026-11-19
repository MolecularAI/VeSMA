from __future__ import annotations

import numpy as np

from vesma import cmol_from_smiles, gmol_from_cmol, gmol_from_smarts
from vesma.bonds import BondKind
from vesma.invariants import (
    FLEX_BOND,
    bond_feature_classes,
)
from vesma.pathfp import (
    FoldedFingerprintConfig,
    fold_features,
    fold_features_words,
    folded_path_fingerprint,
    folded_path_fingerprint_words,
    folded_query_subset,
    folded_query_subset_words,
    sparse_path_fingerprint,
)
from vesma.reaction import parse_reaction_smarts


def packed_words(bits: np.ndarray) -> np.ndarray:
    packed = np.packbits(bits.astype(np.uint8, copy=False), bitorder="little")
    return np.ascontiguousarray(packed).view(np.uint64)


def test_folded_word_fingerprint_matches_bit_fingerprint() -> None:
    config = FoldedFingerprintConfig(n_bits=2048, max_length=7)
    molecule = gmol_from_cmol(cmol_from_smiles("OCC(N)CO")[0])
    graph = gmol_from_smarts("C[N,O]")[0]

    molecule_bits = folded_path_fingerprint(
        molecule,
        query=False,
        config=config,
    )
    molecule_words = folded_path_fingerprint_words(
        molecule,
        query=False,
        config=config,
    )
    query_bits = folded_path_fingerprint(graph, query=True, config=config)
    query_words = folded_path_fingerprint_words(graph, query=True, config=config)

    assert np.array_equal(molecule_words, packed_words(molecule_bits))
    assert np.array_equal(query_words, packed_words(query_bits))
    assert folded_query_subset(query_bits, molecule_bits) == folded_query_subset_words(
        query_words,
        molecule_words,
    )


def test_fold_features_words_matches_fold_features() -> None:
    features = sparse_path_fingerprint(gmol_from_smarts("C1CCCCC1")[0], query=True)

    assert np.array_equal(
        fold_features_words(features, 2048),
        packed_words(fold_features(features, 2048)),
    )


def test_bond_features_are_query_requirements_and_molecule_capabilities() -> None:
    assert bond_feature_classes(
        int(BondKind.SINGLE),
        query=True,
    ) == (FLEX_BOND,)
    assert bond_feature_classes(
        int(BondKind.DOUBLE),
        query=True,
    ) == (FLEX_BOND,)
    assert bond_feature_classes(
        int(BondKind.TRIPLE),
        query=True,
    ) == (FLEX_BOND,)
    assert bond_feature_classes(
        int(BondKind.DELOCALIZED),
        query=True,
    ) == (FLEX_BOND,)

    assert bond_feature_classes(
        int(BondKind.SINGLE),
        query=False,
    ) == (FLEX_BOND,)
    assert bond_feature_classes(
        int(BondKind.DELOCALIZED),
        query=False,
    ) == (FLEX_BOND,)


def test_folded_prefilter_keeps_single_bond_match_inside_short_ring() -> None:
    product = gmol_from_cmol(
        cmol_from_smiles("CN1CCC(Oc2ccc3[nH]nc(S(=O)(=O)c4cccc5ccccc45)c3c2)C1")[0]
    )
    rule = parse_reaction_smarts(
        "[C:2]-[CH2;D2;+0:1]-[C:3]>>I-[CH;D3;+0:1](-[C:2])-[C:3]"
    )

    query_words = folded_path_fingerprint_words(rule.lhs, query=True)
    product_words = folded_path_fingerprint_words(product, query=False)

    assert folded_query_subset_words(query_words, product_words)

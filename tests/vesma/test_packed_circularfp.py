"""The packed batch fingerprint is the CPU reference, batched.

:func:`vesma.circularfp.circular_fingerprint_words` is the specification and
:func:`vesma.packed_circularfp.circular_fingerprints_packed` reads a packed batch instead of a
``GMol``. Nothing downstream survives them disagreeing: a search fingerprints its stock through
the reference and its frontier through the packed path, so a one-bit divergence would make the
heuristic measure distance to a catalogue that is not the one solvedness checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from vesma import cmol_from_smiles, gmol_from_cmol
from vesma.circularfp import CircularFingerprintConfig, circular_fingerprint_words
from vesma.gmol import GMol
from vesma.metal_matcher import _pack_atom_masks, _pack_bond_matrix
from vesma.packed_circularfp import (
    PACKED_BUCKET,
    circular_fingerprints_packed,
    nearest_stock_distance,
)

MOLECULES = (
    "C",
    "CC",
    "CCCCCCCC",
    "c1ccccc1",
    "CC(=O)OC",
    "COC(=O)CC(=O)OC",
    "O=C(O)c1ccc(N)cc1",
    "C1CNCCN1",
    "Cn1nc(C2CCC(=O)NC2=O)c2ccc(N3CCNCC3)cc21",
    "CC(C)(C)OC(=O)N1CCOCC(O)(c2ccc(Br)cc2)C1",
    "[Na+].[Cl-]",
    "c1ccc2[nH]ccc2c1",
    "FC(F)(F)c1ccncc1",
    "OS(=O)(=O)c1ccccc1",
)


def graphs(smiles: tuple[str, ...]) -> list[GMol]:
    return [gmol_from_cmol(cmol_from_smiles(smi)[0]) for smi in smiles]


def packed_batch(gmols: list[GMol]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three arrays a Metal frontier carries, built from concrete molecules."""
    return (
        np.stack([_pack_atom_masks(gmol, PACKED_BUCKET) for gmol in gmols]),
        np.stack([_pack_bond_matrix(gmol, PACKED_BUCKET) for gmol in gmols]),
        np.array([gmol.atom_count for gmol in gmols], dtype=np.uint32),
    )


@pytest.mark.parametrize("radius", [0, 1, 2, 3])
def test_packed_fingerprints_match_the_cpu_reference(radius: int) -> None:
    config = CircularFingerprintConfig(n_bits=2048, radius=radius)
    gmols = graphs(MOLECULES)
    atoms, bonds, counts = packed_batch(gmols)
    reference = np.stack([circular_fingerprint_words(gmol, config) for gmol in gmols])
    assert np.array_equal(
        circular_fingerprints_packed(atoms, bonds, counts, config), reference
    )


@pytest.mark.parametrize("n_bits", [64, 256, 2048])
def test_packed_fingerprints_match_the_reference_at_every_width(n_bits: int) -> None:
    config = CircularFingerprintConfig(n_bits=n_bits, radius=2)
    gmols = graphs(MOLECULES)
    atoms, bonds, counts = packed_batch(gmols)
    reference = np.stack([circular_fingerprint_words(gmol, config) for gmol in gmols])
    assert np.array_equal(
        circular_fingerprints_packed(atoms, bonds, counts, config), reference
    )


def test_a_molecule_is_scored_the_same_alone_as_in_a_batch() -> None:
    """The padding slots must not contribute, which is the one thing batching can break."""
    config = CircularFingerprintConfig(n_bits=2048, radius=2)
    gmols = graphs(MOLECULES)
    atoms, bonds, counts = packed_batch(gmols)
    batched = circular_fingerprints_packed(atoms, bonds, counts, config)
    for index in range(len(gmols)):
        alone = circular_fingerprints_packed(
            atoms[index : index + 1],
            bonds[index : index + 1],
            counts[index : index + 1],
            config,
        )
        assert np.array_equal(alone[0], batched[index])


def test_an_empty_batch_returns_an_empty_matrix_of_the_right_width() -> None:
    config = CircularFingerprintConfig(n_bits=2048, radius=2)
    words = circular_fingerprints_packed(
        np.zeros((0, PACKED_BUCKET * 14), dtype=np.uint32),
        np.zeros((0, PACKED_BUCKET * PACKED_BUCKET), dtype=np.uint8),
        np.zeros(0, dtype=np.uint32),
        config,
    )
    assert words.shape == (0, config.n_bits // 64)


def dense_nearest_distance(
    queries: np.ndarray, stock: np.ndarray, weight: float
) -> np.ndarray:
    """Unpacked Jaccard reference: the definition, written the slow obvious way."""
    query_bits = np.unpackbits(queries.view(np.uint8), axis=1).astype(np.float64)
    stock_bits = np.unpackbits(stock.view(np.uint8), axis=1).astype(np.float64)
    intersection = query_bits @ stock_bits.T
    union = (
        query_bits.sum(axis=1)[:, None] + stock_bits.sum(axis=1)[None, :] - intersection
    )
    return weight * (1.0 - (intersection / np.maximum(union, 1.0)).max(axis=1))


def test_nearest_stock_distance_matches_the_unpacked_definition() -> None:
    config = CircularFingerprintConfig(n_bits=2048, radius=2)
    gmols = graphs(MOLECULES)
    atoms, bonds, counts = packed_batch(gmols)
    fingerprints = circular_fingerprints_packed(atoms, bonds, counts, config)
    stock, queries = fingerprints[:6], fingerprints[4:]
    distances = nearest_stock_distance(queries, stock, weight=1.0)
    assert distances == pytest.approx(
        dense_nearest_distance(queries, stock, 1.0), abs=1e-6
    )


def test_a_stock_molecule_is_at_distance_zero_from_the_stock() -> None:
    """The property :class:`vertsy.heuristics.Heuristic` requires: the estimate vanishes there."""
    config = CircularFingerprintConfig(n_bits=2048, radius=2)
    gmols = graphs(MOLECULES)
    atoms, bonds, counts = packed_batch(gmols)
    fingerprints = circular_fingerprints_packed(atoms, bonds, counts, config)
    assert nearest_stock_distance(
        fingerprints, fingerprints, weight=3.0
    ) == pytest.approx(np.zeros(len(fingerprints)))


@pytest.mark.parametrize("budget_bytes", [1, 256, 4096, 64 << 20])
def test_the_budget_is_a_memory_knob_and_not_an_answer(budget_bytes: int) -> None:
    config = CircularFingerprintConfig(n_bits=2048, radius=2)
    gmols = graphs(MOLECULES)
    atoms, bonds, counts = packed_batch(gmols)
    fingerprints = circular_fingerprints_packed(atoms, bonds, counts, config)
    stock, queries = fingerprints[:8], fingerprints[6:]
    assert nearest_stock_distance(
        queries, stock, weight=1.0, budget_bytes=budget_bytes
    ) == pytest.approx(nearest_stock_distance(queries, stock, weight=1.0))


def test_the_budget_bounds_the_working_set_whatever_the_batch_size() -> None:
    """The property a stock-row knob could not give, and the reason this one replaced it.

    An AO* pass scores every precursor it just discovered, which is thousands of molecules, so a
    knob that bounds only the stock block leaves the intermediate growing linearly in the batch —
    at 19,000 queries against a 4,096-row block that was a 20 GB allocation on a 17 GB machine.
    Tiling both sides means the budget is the budget.
    """
    words = 32
    stock = np.ones((4096, words), dtype=np.uint64)
    queries = np.ones((2048, words), dtype=np.uint64)
    budget = 1 << 16
    tile = budget // (words * 8)
    block = budget // (tile * words * 8)
    assert tile * block * words * 8 <= budget
    assert tile < len(
        queries
    )  # the batch really is larger than the budget allows at once
    assert nearest_stock_distance(
        queries, stock, weight=1.0, budget_bytes=budget
    ) == pytest.approx(np.zeros(len(queries)))


def test_a_non_positive_budget_is_refused() -> None:
    stock = np.ones((4, 32), dtype=np.uint64)
    with pytest.raises(ValueError, match="budget_bytes must be positive"):
        nearest_stock_distance(stock, stock, weight=1.0, budget_bytes=0)


def test_an_empty_query_batch_scores_nothing() -> None:
    stock = np.ones((4, 32), dtype=np.uint64)
    assert nearest_stock_distance(
        np.zeros((0, 32), dtype=np.uint64), stock, weight=1.0
    ).shape == (0,)


def test_mismatched_fingerprint_widths_raise() -> None:
    with pytest.raises(ValueError, match="fingerprint widths differ"):
        nearest_stock_distance(
            np.zeros((2, 16), dtype=np.uint64),
            np.zeros((4, 32), dtype=np.uint64),
            weight=1.0,
        )

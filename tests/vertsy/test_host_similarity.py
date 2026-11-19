"""The host heuristics must be admissible estimates of the same distance, on either representation.

:class:`vertsy.heuristics.Heuristic` asks for two properties and everything downstream depends on
them: the estimate vanishes at the stock, so a solved graph is never charged for finishing, and it
is bounded by the cheapest step, so AO* cannot be talked out of an optimal route. Neither is
visible from a single call, so they are tested as properties over a small molecule set.

The other thing pinned here is that :class:`vertsy.host_similarity.NearestStock` agrees
with :func:`vertsy.similarity.stock_similarity` — it exists only to trade that function's float
matrix for packed bits, so answering differently would make it a different heuristic rather than a
cheaper one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from rdkit import Chem

from vertsy.host_similarity import (
    NearestStock,
    ecfp4_stock,
    pack_bit_vectors,
    packed_ecfp4_words,
    packed_stock,
)
from vertsy.search import object_mol_array
from vertsy.similarity import stock_similarity
from vesma import cmol_from_smiles, gmol_from_cmol
from vesma.circularfp import CircularFingerprintConfig, circular_fingerprint_words
from vesma.metal_matcher import _pack_atom_masks, _pack_bond_matrix
from vesma.packed_circularfp import PACKED_ATOM_WORD_COUNT, PACKED_BUCKET

STOCK = (
    "c1ccccc1",
    "CC(=O)O",
    "C1CNCCN1",
    "O=C(O)c1ccc(N)cc1",
    "CN1N=CC2=CC=CC=C12",
    "O=C1CCC(N)C(=O)N1",
)
QUERIES = (
    "c1ccccc1",  # in stock, so distance zero
    "Cn1nc(C2CCC(=O)NC2=O)c2ccc(N3CCNCC3)cc21",
    "CC(C)(C)OC(=O)N1CCNCC1",
    "FC(F)(F)c1ccncc1",
)
CONFIG = CircularFingerprintConfig(n_bits=2048, radius=2)


class PackedFrontier:
    """The three arrays a Metal frontier exposes, packed from concrete molecules."""

    def __init__(self, smiles: tuple[str, ...]) -> None:
        gmols = [gmol_from_cmol(cmol_from_smiles(smi)[0]) for smi in smiles]
        self.atoms = np.zeros(
            (len(gmols), PACKED_BUCKET * PACKED_ATOM_WORD_COUNT), dtype=np.uint32
        )
        self.bonds = np.zeros(
            (len(gmols), PACKED_BUCKET * PACKED_BUCKET), dtype=np.uint8
        )
        for index, gmol in enumerate(gmols):
            self.atoms[index] = _pack_atom_masks(gmol, PACKED_BUCKET)
            self.bonds[index] = _pack_bond_matrix(gmol, PACKED_BUCKET)
        self.atom_counts = np.array([g.atom_count for g in gmols], dtype=np.uint32)

    def __len__(self) -> int:
        return int(self.atom_counts.shape[0])


def vesma_stock_words(smiles: tuple[str, ...]) -> np.ndarray:
    return np.stack(
        [
            circular_fingerprint_words(gmol_from_cmol(cmol_from_smiles(smi)[0]), CONFIG)
            for smi in smiles
        ]
    )


def rdkit_batch(smiles: tuple[str, ...]) -> Any:
    """An ``object`` ndarray of ``Mol``, which is what `object_mol_array` builds for the search."""
    return object_mol_array([Chem.MolFromSmiles(smi) for smi in smiles])


def packed_heuristic(weight: float = 1.0) -> NearestStock:
    return packed_stock(vesma_stock_words(STOCK), weight=weight, config=CONFIG)


def ecfp4_heuristic(weight: float = 1.0) -> NearestStock:
    mols = [Chem.MolFromSmiles(smi) for smi in STOCK]
    return ecfp4_stock(packed_ecfp4_words(mols), weight=weight)


# --------------------------------------------------------------------------- heuristic contract


def test_the_packed_estimate_vanishes_on_every_stock_molecule() -> None:
    scores = packed_heuristic(weight=5.0)(PackedFrontier(STOCK))
    assert scores == pytest.approx(np.zeros(len(STOCK)))


def test_the_ecfp4_estimate_vanishes_on_every_stock_molecule() -> None:
    scores = ecfp4_heuristic(weight=5.0)(rdkit_batch(STOCK))
    assert scores == pytest.approx(np.zeros(len(STOCK)))


@pytest.mark.parametrize("weight", [0.0, 1.0, 3.5])
def test_the_estimate_stays_within_the_weight(weight: float) -> None:
    """Bounded above by ``weight``, which is what makes the weight the admissibility knob."""
    for scores in (
        packed_heuristic(weight)(PackedFrontier(QUERIES)),
        ecfp4_heuristic(weight)(rdkit_batch(QUERIES)),
    ):
        assert np.all(scores >= 0.0)
        assert np.all(scores <= weight + 1e-6)


def test_the_weight_scales_the_estimate_linearly() -> None:
    unit = packed_heuristic(1.0)(PackedFrontier(QUERIES))
    assert packed_heuristic(3.0)(PackedFrontier(QUERIES)) == pytest.approx(
        3.0 * unit, abs=1e-5
    )


def test_an_off_stock_molecule_is_estimated_strictly_positive() -> None:
    """Otherwise the estimate would be zero everywhere and no better than `zero_heuristic`."""
    off_stock = QUERIES[1:]
    assert np.all(packed_heuristic()(PackedFrontier(off_stock)) > 0.0)
    assert np.all(ecfp4_heuristic()(rdkit_batch(off_stock)) > 0.0)


def test_an_empty_frontier_is_scored_without_touching_the_stock() -> None:
    """AO* scores `fresh_mols[novel]`, which is empty on a pass that discovered nothing new."""
    assert packed_heuristic()(PackedFrontier(())).shape == (0,)
    assert ecfp4_heuristic()(rdkit_batch(())).shape == (0,)


@pytest.mark.parametrize("dtype", [np.float32])
def test_scores_come_back_as_float32_one_per_molecule(dtype: type) -> None:
    for scores in (
        packed_heuristic()(PackedFrontier(QUERIES)),
        ecfp4_heuristic()(rdkit_batch(QUERIES)),
    ):
        assert scores.dtype == dtype
        assert scores.shape == (len(QUERIES),)


# --------------------------------------------------------------------------- agreement


def test_the_packed_ecfp4_heuristic_agrees_with_the_float_matrix_one() -> None:
    """The reason this class exists is memory, so it must not also change the answer."""
    reference = stock_similarity(STOCK, weight=1.0)
    assert ecfp4_heuristic()(rdkit_batch(QUERIES)) == pytest.approx(
        reference(rdkit_batch(QUERIES)), abs=1e-5
    )


def test_the_two_representations_rank_the_same_molecule_set_the_same_way() -> None:
    """Different fingerprints, so different distances — but the ordering is the shared claim."""
    packed = packed_heuristic()(PackedFrontier(QUERIES))
    ecfp4 = ecfp4_heuristic()(rdkit_batch(QUERIES))
    assert np.array_equal(np.argsort(packed, kind="stable")[:1], np.argsort(ecfp4)[:1])


def test_packing_round_trips_an_rdkit_bit_vector() -> None:
    from rdkit.Chem import rdMolDescriptors

    mol = Chem.MolFromSmiles(QUERIES[1])
    vector = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    words = pack_bit_vectors([vector], 2048)
    bits = np.nonzero(np.unpackbits(words.view(np.uint8), bitorder="little"))[0]
    assert sorted(bits.tolist()) == sorted(vector.GetOnBits())


# --------------------------------------------------------------------------- the scan seam


class RecordingScan:
    """A scan that answers zero and remembers what it was asked, standing in for a device one."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...], float]] = []

    def __call__(
        self, queries: np.ndarray, stock: np.ndarray, *, weight: float
    ) -> np.ndarray:
        self.calls.append((queries.shape, stock.shape, weight))
        return np.zeros(len(queries), dtype=np.float32)


@pytest.mark.parametrize("representation", ["packed", "ecfp4"])
def test_the_scan_receives_the_fingerprinted_frontier_and_the_whole_stock(
    representation: str,
) -> None:
    """The seam's contract: the heuristic fingerprints, the scan does every popcount.

    Pinned because a device scan is only substitutable if the split is exactly here — a heuristic
    that scanned some rows itself would make the backends disagree by however many it kept.
    """
    scan = RecordingScan()
    if representation == "packed":
        stock = vesma_stock_words(STOCK)
        scores = packed_stock(stock, weight=2.0, config=CONFIG, scan=scan)(
            PackedFrontier(QUERIES)
        )
    else:
        stock = packed_ecfp4_words([Chem.MolFromSmiles(smi) for smi in STOCK])
        scores = ecfp4_stock(stock, weight=2.0, scan=scan)(rdkit_batch(QUERIES))
    assert scan.calls == [((len(QUERIES), 32), (len(STOCK), 32), 2.0)]
    assert scores == pytest.approx(np.zeros(len(QUERIES)))


def test_the_default_scan_is_the_numpy_one() -> None:
    """So a caller who asks for nothing gets the portable path rather than an accident."""
    from vertsy.host_similarity import HostNearestScan

    assert isinstance(
        packed_stock(vesma_stock_words(STOCK), weight=1.0, config=CONFIG).scan,
        HostNearestScan,
    )
    assert isinstance(ecfp4_heuristic().scan, HostNearestScan)


def test_an_empty_frontier_does_not_reach_the_scan_at_all() -> None:
    scan = RecordingScan()
    packed_stock(vesma_stock_words(STOCK), weight=1.0, config=CONFIG, scan=scan)(
        PackedFrontier(())
    )
    assert scan.calls == []


# --------------------------------------------------------------------------- refusals


def test_a_stock_of_the_wrong_fingerprint_width_is_refused() -> None:
    with pytest.raises(ValueError, match="fingerprint width must be"):
        packed_stock(np.zeros((4, 8), dtype=np.uint64), weight=1.0, config=CONFIG)


def test_an_empty_stock_is_refused() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        packed_stock(np.zeros((0, 32), dtype=np.uint64), weight=1.0, config=CONFIG)


def test_a_negative_weight_is_refused() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        packed_stock(vesma_stock_words(STOCK), weight=-1.0, config=CONFIG)
    with pytest.raises(ValueError, match="non-negative"):
        ecfp4_stock(np.zeros((4, 32), dtype=np.uint64), weight=-1.0)


def test_a_float_stock_matrix_is_refused() -> None:
    """Packed bits are the whole point; a float array would silently be popcounted as garbage."""
    with pytest.raises(ValueError, match="dtype uint64"):
        packed_stock(np.zeros((4, 32), dtype=np.float32), weight=1.0, config=CONFIG)
    with pytest.raises(ValueError, match="dtype uint64"):
        ecfp4_stock(np.zeros((4, 32), dtype=np.float32), weight=1.0)

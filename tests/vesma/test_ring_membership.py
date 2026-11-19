"""`R` and `r` from the relevant cycles: exact, and independent of atom order.

The reference is every simple cycle put through the same size-tiered GF(2) test -- exponential,
so it serves only these small cages, which are where a perception that keeps one shortest path
per atom goes wrong.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from tests.vesma.ring_membership_prototypes import brute_relevant_ring_info
from vesma.rings import ring_info_from_presence

Chem = pytest.importorskip("rdkit.Chem")

CAGES = {
    "K3,3": "C12C3C4C1C3C24",  # all nine 4-cycles relevant: six through every atom
    "cubane": "C12C3C4C1C5C2C3C45",
    "prismane": "C12C3C1C4C2C34",
    "adamantane": "C1C2CC3CC1CC(C2)C3",
    "twistane": "C1CC2CC3CC1CC2C3",
    "norbornane": "C1CC2CCC1C2",
    "bicyclo[2.2.2]octane": "C1CC2CCC1CC2",
    "spiro": "C1CCC2(CC1)CCCCC2",
    "pyrene": "c1cc2ccc3cccc4ccc(c1)c2c34",
    "steroid": "C1CCC2C(C1)CCC1C2CCC2CCCC21",
}


def adjacency(smiles: str) -> np.ndarray:
    return Chem.GetAdjacencyMatrix(Chem.MolFromSmiles(smiles)).astype(bool)


@pytest.mark.parametrize("name", list(CAGES))
def test_ring_membership_is_the_relevant_cycles(name: str) -> None:
    matrix = adjacency(CAGES[name])
    info = ring_info_from_presence(matrix)
    exact = brute_relevant_ring_info(matrix)
    assert info.membership_count.tolist() == [min(c, 7) for c in exact.count]
    assert info.smallest_ring_size.tolist() == [min(c, 7) for c in exact.smallest]


@pytest.mark.parametrize("name", list(CAGES))
def test_ring_membership_does_not_depend_on_atom_order(name: str) -> None:
    matrix = adjacency(CAGES[name])
    info = ring_info_from_presence(matrix)
    rng = random.Random(name)
    for _ in range(5):
        order = list(range(len(matrix)))
        rng.shuffle(order)
        shuffled = ring_info_from_presence(matrix[np.ix_(order, order)])
        assert (
            shuffled.membership_count.tolist() == info.membership_count[order].tolist()
        )
        assert (
            shuffled.smallest_ring_size.tolist()
            == info.smallest_ring_size[order].tolist()
        )

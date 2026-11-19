from __future__ import annotations

import numpy as np
import pytest

from vesma import cmol_from_smiles, gmol_from_smarts
from vesma.adjacency import (
    bond_delocalization,
    bond_order_matrix,
    bond_presence,
    bonds_with_unknown_topology,
)
from vesma.bonds import BOND_CONCRETE_KINDS, BOND_TOPOLOGY_ANY, BondKind
from vesma.rings import ring_info_from_bonds


def _gmol(smarts: str):
    return gmol_from_smarts(smarts)[0]


def _molecule(smiles: str):
    return cmol_from_smiles(smiles)[0]


def test_parse_smiles_stores_rich_bond_types() -> None:
    graph = _molecule("C=CC#N")
    assert graph.bonds[0, 1] == BondKind.DOUBLE
    assert graph.bonds[2, 3] == BondKind.TRIPLE
    assert np.array_equal(bond_presence(graph.bonds), graph.bonds != 0)


def test_parse_smarts_stores_bond_types() -> None:
    graph = _gmol("[#6]=[#8]")
    assert graph.bonds.shape == (2, 2)
    assert graph.bonds[0, 1] == (int(BondKind.DOUBLE) | BOND_TOPOLOGY_ANY)


def test_aromatic_ring_stores_canonical_pure_aromatic_bonds() -> None:
    aromatic = _molecule("c1ccccc1")
    localized = aromatic.bonds & np.uint8(BOND_CONCRETE_KINDS)
    # Canonical aromatic bonds carry only the aromatic bit, no Kekule order.
    assert int(localized.sum()) == 0
    assert int(bond_delocalization(aromatic.bonds).sum()) == 12
    # The order lookup maps a pure aromatic bond to order 0.
    assert int(bond_order_matrix(aromatic.bonds).sum()) == 0


def test_ring_info_detects_cycle_membership() -> None:
    graph = _molecule("C1CCCCC1")
    ring_info = ring_info_from_bonds(graph.bonds)
    assert np.all(ring_info.membership_count >= 1)
    assert np.all(ring_info.smallest_ring_size >= 3)


def test_exact_bond_helpers_reject_query_topology_masks() -> None:
    smarts = "C@C"
    graph = _gmol(smarts)
    with pytest.raises(ValueError, match="one bond kind per edge"):
        bond_order_matrix(graph.bonds)


def test_unknown_topology_boundary_keeps_only_class_and_topology_bits() -> None:
    bonds = np.asarray(
        [
            [0x00, 0x81],
            [0x81, 0x80],
        ],
        dtype=np.uint8,
    )

    assert np.array_equal(
        bonds_with_unknown_topology(bonds),
        np.asarray(
            [
                [0x00, int(BondKind.SINGLE) | BOND_TOPOLOGY_ANY],
                [int(BondKind.SINGLE) | BOND_TOPOLOGY_ANY, 0x00],
            ],
            dtype=np.uint8,
        ),
    )

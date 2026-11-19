from __future__ import annotations

import numpy as np

from vesma import cmol_from_smiles, gmol_from_cmol
from vesma.adjacency import bond_kind_matrix, bond_presence
from vesma.bonds import (
    BOND_TOPOLOGY_CHAIN,
    BOND_TOPOLOGY_MASK,
    BOND_TOPOLOGY_RING,
)
from vesma.rings import annotate_bond_topology


def annotated_bonds(smiles: str) -> tuple[np.ndarray, np.ndarray]:
    molecule = gmol_from_cmol(cmol_from_smiles(smiles)[0])
    return molecule.bonds, annotate_bond_topology(molecule.bonds)


def test_annotate_bond_topology_assigns_one_exact_state_per_edge() -> None:
    original, annotated = annotated_bonds("C1CCCCC1.CCC")
    present = bond_presence(annotated)
    topology = annotated & np.uint8(BOND_TOPOLOGY_MASK)

    assert np.all(topology[present] != 0)
    assert np.all(topology[present] != BOND_TOPOLOGY_MASK)
    assert np.all(topology[~present] == 0)
    assert np.array_equal(bond_kind_matrix(annotated), bond_kind_matrix(original))


def test_annotate_bond_topology_marks_ring_chain_and_bridge_edges() -> None:
    _, ring = annotated_bonds("C1CCCCC1")
    _, chain = annotated_bonds("CCC")
    _, biphenyl = annotated_bonds("c1ccccc1-c2ccccc2")

    assert np.all(
        (ring[bond_presence(ring)] & BOND_TOPOLOGY_MASK) == BOND_TOPOLOGY_RING
    )
    assert np.all(
        (chain[bond_presence(chain)] & BOND_TOPOLOGY_MASK) == BOND_TOPOLOGY_CHAIN
    )

    topology = biphenyl & np.uint8(BOND_TOPOLOGY_MASK)
    assert np.count_nonzero(topology == BOND_TOPOLOGY_RING) == 24
    assert np.count_nonzero(topology == BOND_TOPOLOGY_CHAIN) == 2

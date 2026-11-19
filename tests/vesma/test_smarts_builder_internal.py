from __future__ import annotations

import pytest

from vesma.bonds import (
    BOND_KIND_MASK,
    BOND_TOPOLOGY_ANY,
    BOND_TOPOLOGY_CHAIN,
    BOND_TOPOLOGY_RING,
    BondKind,
)
from vesma.parsing.builder import build_smarts_mol, merge_bond_flags
from vesma.parsing.errors import InvalidSmartsSyntaxError
from vesma.parsing.grammar_custom import parse_graph_steps
from vesma.parsing.ir import SmartsAtom


def test_build_smarts_mol_preserves_atoms_and_bonds() -> None:
    smarts_mol = build_smarts_mol(parse_graph_steps("C(=O)N"))
    assert smarts_mol.atom_count == 3
    assert isinstance(smarts_mol.atoms[0], SmartsAtom)
    assert smarts_mol.bonds[0, 1] == (int(BondKind.DOUBLE) | BOND_TOPOLOGY_ANY)
    assert smarts_mol.bonds[0, 2] != 0


def test_build_smarts_mol_keeps_dot_disconnected_components() -> None:
    smarts_mol = build_smarts_mol(parse_graph_steps("C.C"))
    assert smarts_mol.atom_count == 2
    assert smarts_mol.bonds[0, 1] == 0


def test_merge_bond_flags_intersects_class_and_topology_sets() -> None:
    assert merge_bond_flags(
        int(BondKind.SINGLE) | BOND_TOPOLOGY_ANY,
        BOND_KIND_MASK | BOND_TOPOLOGY_RING,
    ) == (int(BondKind.SINGLE) | BOND_TOPOLOGY_RING)


def test_merge_bond_flags_rejects_disjoint_topology_sets() -> None:
    with pytest.raises(InvalidSmartsSyntaxError):
        merge_bond_flags(
            int(BondKind.SINGLE) | BOND_TOPOLOGY_RING,
            int(BondKind.SINGLE) | BOND_TOPOLOGY_CHAIN,
        )

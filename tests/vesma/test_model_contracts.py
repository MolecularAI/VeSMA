from __future__ import annotations

import numpy as np
import pytest

from vesma.bonds import BOND_TOPOLOGY_RING, BondKind
from vesma.cmol import CMol
from vesma.gmol import GMol
from vesma.gmol_atoms import ALIPHATIC, AtomMasks, atom_masks_from_props


def valid_atom_masks() -> AtomMasks:
    return atom_masks_from_props([{}])


def valid_gmol() -> GMol:
    return GMol(
        atom_masks=valid_atom_masks(),
        bonds=np.zeros((1, 1), dtype=np.uint8),
        atom_map=np.zeros(1, dtype=np.int32),
        atom_group=np.zeros(1, dtype=np.int32),
        atom_predicate=np.full(1, -1, dtype=np.int32),
        predicates=(),
    )


def valid_cmol() -> CMol:
    return CMol(
        Z=np.array([6], dtype=np.int16),
        M=np.array([0], dtype=np.int16),
        A=np.array([ALIPHATIC], dtype=np.uint8),
        C=np.array([0], dtype=np.int8),
        D=np.array([0], dtype=np.int16),
        H=np.array([0], dtype=np.int16),
        h=np.array([0], dtype=np.int16),
        v=np.array([0], dtype=np.int16),
        X=np.array([0], dtype=np.int16),
        R=np.array([0], dtype=np.int16),
        r=np.array([0], dtype=np.int16),
        x=np.array([0], dtype=np.int16),
        bonds=np.zeros((1, 1), dtype=np.uint8),
        atom_map=np.zeros(1, dtype=np.int32),
    )


def test_atom_masks_rejects_wrong_dtype() -> None:
    masks = valid_atom_masks()
    with pytest.raises(ValueError, match="Z_lo dtype must be uint64"):
        AtomMasks(
            Z_lo=masks.Z_lo.astype(np.int64),
            Z_hi=masks.Z_hi,
            A=masks.A,
            C=masks.C,
            D=masks.D,
            H=masks.H,
            h=masks.h,
            v=masks.v,
            X=masks.X,
            R=masks.R,
            r=masks.r,
            x=masks.x,
            M=masks.M,
        )


def test_atom_masks_rejects_mismatched_shapes() -> None:
    masks = valid_atom_masks()
    with pytest.raises(ValueError, match="x shape must match atom count"):
        AtomMasks(
            Z_lo=masks.Z_lo,
            Z_hi=masks.Z_hi,
            A=masks.A,
            C=masks.C,
            D=masks.D,
            H=masks.H,
            h=masks.h,
            v=masks.v,
            X=masks.X,
            R=masks.R,
            r=masks.r,
            x=np.zeros(2, dtype=np.uint8),
            M=np.zeros(2, dtype=np.int16),
        )


def test_gmol_rejects_wrong_atom_map_dtype() -> None:
    graph = valid_gmol()
    with pytest.raises(ValueError, match="atom_map dtype must be int32"):
        GMol(
            atom_masks=graph.atom_masks,
            bonds=graph.bonds,
            atom_map=graph.atom_map.astype(np.int64),
            atom_group=graph.atom_group,
            atom_predicate=graph.atom_predicate,
            predicates=graph.predicates,
        )


def test_cmol_rejects_wrong_bonds_dtype() -> None:
    molecule = valid_cmol()
    with pytest.raises(ValueError, match="bonds dtype must be uint8"):
        CMol(
            Z=molecule.Z,
            A=molecule.A,
            C=molecule.C,
            D=molecule.D,
            H=molecule.H,
            h=molecule.h,
            v=molecule.v,
            X=molecule.X,
            R=molecule.R,
            r=molecule.r,
            x=molecule.x,
            M=molecule.M,
            bonds=molecule.bonds.astype(np.int16),
            atom_map=molecule.atom_map,
        )


@pytest.mark.parametrize(
    "invalid_bonds",
    [
        np.array([[int(BondKind.SINGLE) | int(BondKind.DOUBLE)]], dtype=np.uint8),
        np.array(
            [[int(BondKind.SINGLE) | BOND_TOPOLOGY_RING]],
            dtype=np.uint8,
        ),
    ],
)
def test_cmol_rejects_non_exact_bond_masks(invalid_bonds: np.ndarray) -> None:
    molecule = valid_cmol()
    with pytest.raises(ValueError, match="exact bond types only"):
        CMol(
            Z=molecule.Z,
            A=molecule.A,
            C=molecule.C,
            D=molecule.D,
            H=molecule.H,
            h=molecule.h,
            v=molecule.v,
            X=molecule.X,
            R=molecule.R,
            r=molecule.r,
            x=molecule.x,
            M=molecule.M,
            bonds=invalid_bonds,
            atom_map=molecule.atom_map,
        )

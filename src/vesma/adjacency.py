from __future__ import annotations

import numpy as np

from .bonds import (
    BOND_KIND_MASK,
    BOND_ORDER_BY_KIND,
    BOND_TOPOLOGY_ANY,
    BOND_TOPOLOGY_MASK,
    EXACT_BOND_KINDS,
    BondKind,
    validate_bond_flags,
)

BondMatrix = np.ndarray[tuple[int, int], np.dtype[np.uint8]]
BoolBondMatrix = np.ndarray[tuple[int, int], np.dtype[np.bool_]]


def bond_kind_matrix(bonds: BondMatrix) -> BondMatrix:
    return bonds & np.uint8(BOND_KIND_MASK)


def bonds_with_unknown_topology(bonds: BondMatrix) -> BondMatrix:
    """Return matcher-facing kind bonds whose topology may be ring or chain.

    Only `gmol_from_cmol` needs this: a `CMol` stores exact bonds with no topology
    field, and the CPU matcher resolves the placeholder to an exact ring-or-chain state
    with `rings.annotate_bond_topology` just before it searches. Nothing else should
    hand a molecule to a matcher in this state -- a bond that claims both states passes
    every `@` and every `!@`, which is an answer to neither question.
    """
    kinds = bond_kind_matrix(bonds)
    return np.where(
        kinds != int(BondKind.NONE),
        kinds | np.uint8(BOND_TOPOLOGY_ANY),
        np.uint8(BondKind.NONE),
    ).astype(np.uint8)


def bond_presence(bond_types: BondMatrix) -> BoolBondMatrix:
    return bond_kind_matrix(bond_types) != BondKind.NONE


def bond_delocalization(bond_types: BondMatrix) -> BoolBondMatrix:
    return (bond_types & np.uint8(BondKind.DELOCALIZED)) != 0


def bond_order_matrix(bond_types: BondMatrix) -> BondMatrix:
    """Concrete bond orders. Delocalized edges carry no order and contribute 0."""
    require_exact_bond_matrix(
        bond_types,
        error_message="Exact bond helpers require one bond kind per edge.",
    )
    return BOND_ORDER_BY_KIND[bond_types]


def bond_type_compatibility_matrix(
    pattern_bonds: BondMatrix,
    molecule_bonds: BondMatrix,
) -> BoolBondMatrix:
    pattern_absent = pattern_bonds == 0
    kind_ok = (bond_kind_matrix(pattern_bonds) & bond_kind_matrix(molecule_bonds)) != 0
    topology_ok = (
        (pattern_bonds & np.uint8(BOND_TOPOLOGY_MASK))
        & (molecule_bonds & np.uint8(BOND_TOPOLOGY_MASK))
    ) != 0
    return pattern_absent | (kind_ok & topology_ok)


def validate_bond_flag_matrix(bonds: BondMatrix) -> None:
    for value in np.unique(bonds):
        validate_bond_flags(int(value))


def require_exact_bond_matrix(bonds: BondMatrix, *, error_message: str) -> None:
    """Every present edge carries exactly one `BondKind` and no topology bits.

    This is the invariant that keeps an exact molecule out of query space: a value
    combining kinds, or carrying topology, is a query bond and never a molecule bond.
    """
    legal = bonds == int(BondKind.NONE)
    for kind in EXACT_BOND_KINDS:
        legal |= bonds == int(kind)
    if not np.all(legal):
        raise ValueError(error_message)

from __future__ import annotations

import numpy as np
import pytest

from vesma.adjacency import bond_presence, require_exact_bond_matrix
from vesma.bonds import (
    BOND_KIND_MASK,
    BOND_TOPOLOGY_ANY,
    BOND_TOPOLOGY_CHAIN,
    BOND_TOPOLOGY_RING,
    QBOND_DEFAULT,
    BondKind,
    bond_flags_from_smarts_token,
    bond_kind_compatible,
    bond_order,
    exact_bond_kind,
    validate_bond_flags,
)

# Concrete molecule bond classes used in the truth table below. Under Option C a
# concrete bond is a single class bit; aromatic is the pure delocalized class 0x08
# (no Kekule order bit), so 0x09/0x0a never occur in a concrete molecule.
_MOL_SINGLE = int(BondKind.SINGLE) | BOND_TOPOLOGY_CHAIN
_MOL_DOUBLE = int(BondKind.DOUBLE) | BOND_TOPOLOGY_CHAIN
_MOL_TRIPLE = int(BondKind.TRIPLE) | BOND_TOPOLOGY_CHAIN
_MOL_AROM_PURE = int(BondKind.DELOCALIZED) | BOND_TOPOLOGY_CHAIN


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("-", int(BondKind.SINGLE) | BOND_TOPOLOGY_ANY),
        ("/", int(BondKind.SINGLE) | BOND_TOPOLOGY_ANY),
        ("\\", int(BondKind.SINGLE) | BOND_TOPOLOGY_ANY),
        ("/?", int(BondKind.SINGLE) | BOND_TOPOLOGY_ANY),
        ("\\?", int(BondKind.SINGLE) | BOND_TOPOLOGY_ANY),
        ("=", int(BondKind.DOUBLE) | BOND_TOPOLOGY_ANY),
        ("#", int(BondKind.TRIPLE) | BOND_TOPOLOGY_ANY),
        (":", int(BondKind.DELOCALIZED) | BOND_TOPOLOGY_ANY),
        ("@", BOND_KIND_MASK | BOND_TOPOLOGY_RING),
        ("~", BOND_KIND_MASK | BOND_TOPOLOGY_ANY),
        ("=@", int(BondKind.DOUBLE) | BOND_TOPOLOGY_RING),
        ("-!@", int(BondKind.SINGLE) | BOND_TOPOLOGY_CHAIN),
        ("-;!@", int(BondKind.SINGLE) | BOND_TOPOLOGY_CHAIN),
        (".", int(BondKind.NONE)),
    ],
)
def test_bond_flags_from_smarts_token_maps_supported_tokens(
    token: str, expected: int
) -> None:
    assert bond_flags_from_smarts_token(token) == expected


@pytest.mark.parametrize("token", ["!", "=&@", "-&!@"])
def test_bond_flags_from_smarts_token_rejects_unsupported_tokens(token: str) -> None:
    with pytest.raises(ValueError, match="Unsupported SMARTS bond token"):
        bond_flags_from_smarts_token(token)


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (int(BondKind.SINGLE), BondKind.SINGLE),
        (int(BondKind.DOUBLE), BondKind.DOUBLE),
        (int(BondKind.TRIPLE), BondKind.TRIPLE),
        (int(BondKind.DELOCALIZED), BondKind.DELOCALIZED),
        (int(BondKind.SINGLE) | BOND_TOPOLOGY_ANY, BondKind.SINGLE),
        (BOND_KIND_MASK | BOND_TOPOLOGY_ANY, None),
        (int(BondKind.SINGLE) | BOND_TOPOLOGY_RING, None),
    ],
)
def test_bond_type_from_flags_decodes_only_exact_concrete_bonds(
    flags: int, expected: BondKind | None
) -> None:
    assert exact_bond_kind(flags) == expected


def test_bond_type_bits_coerce_cleanly_into_uint8_arrays() -> None:
    values = np.asarray(
        [
            BondKind.SINGLE,
            BondKind.DOUBLE,
            BondKind.TRIPLE,
            BondKind.DELOCALIZED,
        ],
        dtype=np.uint8,
    )
    assert values.dtype == np.uint8
    assert tuple(int(value) for value in values) == (1, 2, 4, 8)


def test_concrete_bond_matrix_accepts_pure_aromatic_bond() -> None:
    # A pure aromatic bond (byte 8, no Kekule order) is the canonical concrete
    # form for a delocalized bond, so it is accepted rather than rejected.
    bonds = np.asarray([[0, 8], [8, 0]], dtype=np.uint8)

    require_exact_bond_matrix(
        bonds,
        error_message="Concrete bonds must be localized.",
    )


@pytest.mark.parametrize(
    ("pattern_flags", "molecule_flags", "expected"),
    [
        # explicit single (-): only the single class
        (int(BondKind.SINGLE) | BOND_TOPOLOGY_ANY, _MOL_SINGLE, True),
        (int(BondKind.SINGLE) | BOND_TOPOLOGY_ANY, _MOL_DOUBLE, False),
        (int(BondKind.SINGLE) | BOND_TOPOLOGY_ANY, _MOL_AROM_PURE, False),
        # explicit double (=): only the double class
        (int(BondKind.DOUBLE) | BOND_TOPOLOGY_ANY, _MOL_DOUBLE, True),
        (int(BondKind.DOUBLE) | BOND_TOPOLOGY_ANY, _MOL_SINGLE, False),
        (int(BondKind.DOUBLE) | BOND_TOPOLOGY_ANY, _MOL_AROM_PURE, False),
        # explicit triple (#)
        (int(BondKind.TRIPLE) | BOND_TOPOLOGY_ANY, _MOL_TRIPLE, True),
        (int(BondKind.TRIPLE) | BOND_TOPOLOGY_ANY, _MOL_AROM_PURE, False),
        # explicit aromatic (:): only the aromatic class
        (int(BondKind.DELOCALIZED) | BOND_TOPOLOGY_ANY, _MOL_SINGLE, False),
        (int(BondKind.DELOCALIZED) | BOND_TOPOLOGY_ANY, _MOL_AROM_PURE, True),
        # default (single or aromatic)
        (QBOND_DEFAULT, _MOL_SINGLE, True),
        (QBOND_DEFAULT, _MOL_DOUBLE, False),
        (QBOND_DEFAULT, _MOL_AROM_PURE, True),
        # any (~): matches every class
        (BOND_KIND_MASK | BOND_TOPOLOGY_ANY, _MOL_SINGLE, True),
        (BOND_KIND_MASK | BOND_TOPOLOGY_ANY, _MOL_DOUBLE, True),
        (BOND_KIND_MASK | BOND_TOPOLOGY_ANY, _MOL_AROM_PURE, True),
        (
            int(BondKind.SINGLE) | BOND_TOPOLOGY_RING,
            _MOL_SINGLE,
            False,
        ),
        (
            int(BondKind.SINGLE) | BOND_TOPOLOGY_RING,
            int(BondKind.SINGLE) | BOND_TOPOLOGY_RING,
            True,
        ),
        (
            int(BondKind.SINGLE) | BOND_TOPOLOGY_CHAIN,
            _MOL_SINGLE,
            True,
        ),
        (
            int(BondKind.SINGLE) | BOND_TOPOLOGY_CHAIN,
            int(BondKind.SINGLE) | BOND_TOPOLOGY_RING,
            False,
        ),
        (
            int(BondKind.SINGLE) | BOND_TOPOLOGY_RING,
            int(BondKind.SINGLE) | BOND_TOPOLOGY_ANY,
            True,
        ),
        (
            int(BondKind.SINGLE) | BOND_TOPOLOGY_CHAIN,
            int(BondKind.SINGLE) | BOND_TOPOLOGY_ANY,
            True,
        ),
        (
            int(BondKind.DELOCALIZED) | BOND_TOPOLOGY_RING,
            int(BondKind.SINGLE) | BOND_TOPOLOGY_RING,
            False,
        ),
        # 0x00 means "no bond constraint" (the matcher tests the full pattern matrix,
        # including atom pairs with no query bond), so it accepts anything. A present
        # query edge always carries a type bit, so 0x00 never arises as a real bond.
        (0, _MOL_SINGLE, True),
        (0, _MOL_AROM_PURE, True),
    ],
)
def test_bond_type_compatible_strict_aromatic_class(
    pattern_flags: int,
    molecule_flags: int,
    expected: bool,
) -> None:
    assert bond_kind_compatible(pattern_flags, molecule_flags) is expected


@pytest.mark.parametrize(
    "flags",
    [
        int(BondKind.SINGLE),
        BOND_TOPOLOGY_ANY,
    ],
)
def test_validate_bond_flags_rejects_incomplete_present_bond(flags: int) -> None:
    with pytest.raises(ValueError):
        validate_bond_flags(flags)


def test_bond_presence_reads_only_class_bits() -> None:
    annotations_only = np.array([[0, BOND_TOPOLOGY_ANY]], dtype=np.uint8)
    assert not np.any(bond_presence(annotations_only))


@pytest.mark.parametrize(
    "flags, expected",
    [
        (int(BondKind.SINGLE), BondKind.SINGLE),
        (int(BondKind.DELOCALIZED), BondKind.DELOCALIZED),
        (int(BondKind.SINGLE) | BOND_TOPOLOGY_ANY, BondKind.SINGLE),
        (int(BondKind.NONE), None),
        # A delocalized bond is categorical, so it never carries an order alongside.
        (int(BondKind.SINGLE | BondKind.DELOCALIZED), None),
        (int(BondKind.DOUBLE | BondKind.DELOCALIZED), None),
        (int(BondKind.SINGLE) | BOND_TOPOLOGY_RING, None),
    ],
)
def test_exact_bond_kind_admits_one_kind_only(
    flags: int, expected: BondKind | None
) -> None:
    assert exact_bond_kind(flags) is expected


def test_require_exact_bond_matrix_rejects_a_combined_kind() -> None:
    mixed = int(BondKind.DOUBLE | BondKind.DELOCALIZED)
    bonds = np.array([[0, mixed], [mixed, 0]], dtype=np.uint8)

    with pytest.raises(ValueError, match="one bond kind per edge"):
        require_exact_bond_matrix(bonds, error_message="one bond kind per edge")


def test_delocalized_bonds_carry_no_order() -> None:
    assert bond_order(BondKind.DELOCALIZED) == 0
    assert bond_order(BondKind.NONE) == 0
    assert [
        bond_order(k) for k in (BondKind.SINGLE, BondKind.DOUBLE, BondKind.TRIPLE)
    ] == [1, 2, 3]

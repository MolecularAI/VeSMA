"""Bond encoding shared by query and matcher-facing molecule graphs.

Two different things are encoded in one byte, and keeping them apart is the point of
this module:

- an **exact** bond, carried by ``CMol.bonds`` and reaction scratch matrices, is exactly
  one `BondKind`: single, double, triple, or delocalized;
- a **query** bond, carried by ``GMol.bonds``, is a *set* of accepted kinds in the low
  nibble plus a set of accepted topologies in the next two bits.

Kind values are single bits so a query test is one masked AND, and the exclusivity of an
exact bond is an invariant (`exact_bond_kind`, `require_exact_bond_matrix`) rather than
something the encoding permits. There is no combined order-and-delocalized value:
delocalization is categorical, so a delocalized edge has no bond order to carry.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class BondKind(IntEnum):
    """The exact bond on a present edge. A present edge carries exactly one."""

    NONE = 0
    SINGLE = 0b0000_0001
    DOUBLE = 0b0000_0010
    TRIPLE = 0b0000_0100
    DELOCALIZED = 0b0000_1000


# Matcher-facing topology is an allowed/possible-state mask:
# 00 = empty (only valid for an absent edge), 01 = ring, 10 = chain, 11 = either.
BOND_TOPOLOGY_RING = 0b0001_0000
BOND_TOPOLOGY_CHAIN = 0b0010_0000
BOND_TOPOLOGY_ANY = BOND_TOPOLOGY_RING | BOND_TOPOLOGY_CHAIN
BOND_TOPOLOGY_MASK = BOND_TOPOLOGY_ANY

# The concrete (order-bearing) kinds, as a query mask. Delocalized is deliberately not a
# member: it is a category, not a larger or smaller order.
BOND_CONCRETE_KINDS = int(BondKind.SINGLE | BondKind.DOUBLE | BondKind.TRIPLE)
BOND_KIND_MASK = BOND_CONCRETE_KINDS | int(BondKind.DELOCALIZED)
BOND_KNOWN_MASK = BOND_KIND_MASK | BOND_TOPOLOGY_MASK

# Query/pattern bonds are "accepted target-kind" masks. An omitted/default SMARTS bond
# means "single or delocalized" (Daylight semantics), distinct from an explicit ``-``.
QBOND_DEFAULT = int(BondKind.SINGLE | BondKind.DELOCALIZED) | BOND_TOPOLOGY_ANY

# Indexed by the kind bits, so any low-nibble value is a safe index. A delocalized edge
# contributes no bond order; it contributes to sigma degree instead, which is edge
# presence and is counted separately.
BOND_ORDER_BY_KIND = np.zeros(BOND_KIND_MASK + 1, dtype=np.uint8)
BOND_ORDER_BY_KIND[int(BondKind.SINGLE)] = 1
BOND_ORDER_BY_KIND[int(BondKind.DOUBLE)] = 2
BOND_ORDER_BY_KIND[int(BondKind.TRIPLE)] = 3

EXACT_BOND_KINDS: tuple[BondKind, ...] = (
    BondKind.SINGLE,
    BondKind.DOUBLE,
    BondKind.TRIPLE,
    BondKind.DELOCALIZED,
)

_DIRECT_SMARTS_BOND_TOKEN_TO_FLAGS: dict[str, int] = {
    "-!:": int(BondKind.SINGLE) | BOND_TOPOLOGY_ANY,
    "=,:": int(BondKind.DOUBLE) | int(BondKind.DELOCALIZED) | BOND_TOPOLOGY_ANY,
    "!:": BOND_CONCRETE_KINDS | BOND_TOPOLOGY_ANY,
    "!@": BOND_KIND_MASK | BOND_TOPOLOGY_CHAIN,
    "-": int(BondKind.SINGLE) | BOND_TOPOLOGY_ANY,
    "/": int(BondKind.SINGLE) | BOND_TOPOLOGY_ANY,
    "\\": int(BondKind.SINGLE) | BOND_TOPOLOGY_ANY,
    "/?": int(BondKind.SINGLE) | BOND_TOPOLOGY_ANY,
    "\\?": int(BondKind.SINGLE) | BOND_TOPOLOGY_ANY,
    "=": int(BondKind.DOUBLE) | BOND_TOPOLOGY_ANY,
    "#": int(BondKind.TRIPLE) | BOND_TOPOLOGY_ANY,
    ":": int(BondKind.DELOCALIZED) | BOND_TOPOLOGY_ANY,
    "@": BOND_KIND_MASK | BOND_TOPOLOGY_RING,
    "~": BOND_KIND_MASK | BOND_TOPOLOGY_ANY,
    ".": int(BondKind.NONE),
}

_ORDERED_BOND_PREFIX_TO_FLAGS: dict[str, int] = {
    "-": int(BondKind.SINGLE),
    "=": int(BondKind.DOUBLE),
    "#": int(BondKind.TRIPLE),
    ":": int(BondKind.DELOCALIZED),
}

SMARTS_BOND_TOKENS: tuple[str, ...] = (
    "-;!@",
    *(f"{prefix}@" for prefix in _ORDERED_BOND_PREFIX_TO_FLAGS),
    *(f"{prefix}!@" for prefix in _ORDERED_BOND_PREFIX_TO_FLAGS),
    *_DIRECT_SMARTS_BOND_TOKEN_TO_FLAGS,
)


def bond_flags_from_smarts_token(token: str) -> int:
    if token == "-;!@":
        token = "-!@"

    direct_flags = _DIRECT_SMARTS_BOND_TOKEN_TO_FLAGS.get(token)
    if direct_flags is not None:
        return direct_flags

    if token.endswith("@") and token[:-1] in _ORDERED_BOND_PREFIX_TO_FLAGS:
        return _ORDERED_BOND_PREFIX_TO_FLAGS[token[:-1]] | BOND_TOPOLOGY_RING

    if token.endswith("!@") and token[:-2] in _ORDERED_BOND_PREFIX_TO_FLAGS:
        return _ORDERED_BOND_PREFIX_TO_FLAGS[token[:-2]] | BOND_TOPOLOGY_CHAIN

    raise ValueError(f"Unsupported SMARTS bond token: {token}")


def bond_kind_flags(flags: int) -> int:
    """The accepted-kind mask of a query bond, or the kind bit of an exact bond."""
    return flags & BOND_KIND_MASK


def exact_bond_kind(flags: int) -> BondKind | None:
    """The single `BondKind` these flags denote, or None if they denote a set.

    Returns None for a mask accepting several kinds and for any leftover topology bits,
    so callers that require exactness cannot silently accept a query bond.
    """
    if flags & BOND_TOPOLOGY_MASK not in {0, BOND_TOPOLOGY_ANY}:
        return None
    kind = bond_kind_flags(flags)
    if kind == 0 or kind & (kind - 1):
        return None
    return BondKind(kind)


def bond_order(kind: BondKind | int) -> int:
    """Concrete bond order. A delocalized or absent edge has none, and yields 0."""
    return int(BOND_ORDER_BY_KIND[int(kind)])


def bond_is_delocalized(flags: int) -> bool:
    return bool(flags & int(BondKind.DELOCALIZED))


def resolve_default_bond_flags(
    flags: int, *, left_aromatic: bool, right_aromatic: bool
) -> int:
    """Resolve an omitted/default query bond mask to an exact seed bond.

    A default SMARTS bond is the accepted-kind mask ``QBOND_DEFAULT`` ("single or
    delocalized"). Converting a pattern graph into a concrete molecule must pick a
    definite kind: delocalized when both endpoint *atoms* are aromatic, otherwise
    single. Non-default flags pass through unchanged.
    """
    if flags == QBOND_DEFAULT:
        if left_aromatic and right_aromatic:
            return int(BondKind.DELOCALIZED)
        return int(BondKind.SINGLE)
    return flags


def bond_kind_compatible(pattern_flags: int, molecule_flags: int) -> bool:
    """Does a query bond accept an exact molecule bond?

    The molecule bond is one kind bit and the query is an accepted-kind mask, so the
    kind test is one masked AND. Delocalized is its own kind: an explicit ``-``/``=``/
    ``#`` never accepts it, ``:`` accepts only it, and ``~``/default carry its bit.
    Flags of zero mean "no bond constraint here", because the dense matcher also visits
    absent query pairs. Every present query edge carries nonempty kind and topology masks.
    """
    if pattern_flags == 0:
        return True
    topology_compatible = bool(
        (pattern_flags & BOND_TOPOLOGY_MASK) & (molecule_flags & BOND_TOPOLOGY_MASK)
    )
    return (
        bool(bond_kind_flags(pattern_flags) & bond_kind_flags(molecule_flags))
        and topology_compatible
    )


def validate_bond_flags(flags: int) -> None:
    """Validate query-bond flags: a present edge needs a kind and a topology."""
    if flags & ~BOND_KNOWN_MASK:
        raise ValueError("Bond flags contain unknown bits.")
    if flags == 0:
        return
    if flags & BOND_KIND_MASK == 0:
        raise ValueError("Bond flags require a bond kind when nonzero.")
    if flags & BOND_TOPOLOGY_MASK == 0:
        raise ValueError("GMol bond flags require a topology state when nonzero.")


__all__ = [
    "BOND_CONCRETE_KINDS",
    "BOND_KIND_MASK",
    "BOND_KNOWN_MASK",
    "BOND_ORDER_BY_KIND",
    "BOND_TOPOLOGY_ANY",
    "BOND_TOPOLOGY_CHAIN",
    "BOND_TOPOLOGY_MASK",
    "BOND_TOPOLOGY_RING",
    "EXACT_BOND_KINDS",
    "QBOND_DEFAULT",
    "SMARTS_BOND_TOKENS",
    "BondKind",
    "bond_flags_from_smarts_token",
    "bond_is_delocalized",
    "bond_kind_compatible",
    "bond_kind_flags",
    "bond_order",
    "exact_bond_kind",
    "resolve_default_bond_flags",
    "validate_bond_flags",
]

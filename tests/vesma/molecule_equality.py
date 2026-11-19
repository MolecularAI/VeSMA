"""Comparing two CMols as molecules rather than as spellings."""

from __future__ import annotations

from vesma import CMol, short_ring_wl_molecule_id
from vesma.canonicalization import canonicalish_atom_order, reorder_cmol

# Every field of a CMol except the two that are not chemistry. `h` is spelling -- SMILES
# has no notation for an *implicit* hydrogen on a bracketed atom, so writing `[OH:3]` as
# OpenSMILES 3.6 requires necessarily moves that hydrogen from implicit to explicit -- and
# `atom_map` is provenance carried through an edit, not a property of the molecule.
CHEMISTRY_FIELDS = ("Z", "M", "A", "C", "D", "H", "v", "X", "R", "r", "x")

__all__ = ["CHEMISTRY_FIELDS", "assert_same_molecule"]


def assert_same_molecule(left: CMol, right: CMol) -> None:
    """Two independent comparisons, so neither can carry a test on its own.

    A WL token is a 128-bit hash of a perceived graph, so it could in principle merge two
    molecules that differ; the field comparison is exact but leans on
    ``canonicalish_atom_order`` to line the atoms up, so it could in principle separate
    two that do not. They fail in different ways, and a round trip has to satisfy both.
    """
    assert short_ring_wl_molecule_id(left) == short_ring_wl_molecule_id(right)

    a = reorder_cmol(left, canonicalish_atom_order(left))
    b = reorder_cmol(right, canonicalish_atom_order(right))
    for field in CHEMISTRY_FIELDS:
        assert list(getattr(a, field)) == list(getattr(b, field)), field
    assert (a.bonds == b.bonds).all()

"""Shared graph-invariant vocabulary for the WL and path engines.

These are the small, pure building blocks that the WL hash identity
(``wl.py``), the path fingerprints (``pathfp.py``), and the WL molecule
IDs (``identity.py``) all draw from: the 64-bit hash mixers, exact atomic-number
label extraction, atom/path feature hashers, and the two bond-classing schemes
(the small-ring key and the full-aromaticity key). The refinement *loops* live
with their consumers; only the vocabulary is shared here.
"""

from __future__ import annotations

import numpy as np

from .adjacency import BondMatrix
from .bonds import BondKind, bond_kind_flags
from .gmol_atoms import AtomMasks
from .rings import short_ring_edges

LabelArray = np.ndarray[tuple[int, ...], np.dtype[np.uint64]]
LabelVector = np.ndarray[tuple[int], np.dtype[np.uint64]]

UINT64_MASK = (1 << 64) - 1

# Magic constants kept as Python ints so the numpy and scalar mixers share one source.
WL_COMBINE_CONSTANT_INT = 0x517CC1B727220A95
SPLITMIX64_GAMMA_INT = 0x9E3779B97F4A7C15
SPLITMIX64_MUL1_INT = 0xBF58476D1CE4E5B9
SPLITMIX64_MUL2_INT = 0x94D049BB133111EB

WL_COMBINE_CONSTANT = np.uint64(WL_COMBINE_CONSTANT_INT)
SPLITMIX64_GAMMA = np.uint64(SPLITMIX64_GAMMA_INT)
SPLITMIX64_MUL1 = np.uint64(SPLITMIX64_MUL1_INT)
SPLITMIX64_MUL2 = np.uint64(SPLITMIX64_MUL2_INT)

# Feature-hash namespaces for the path (and future WL) fingerprints.
ATOM_FEATURE = 0xA70F_4A7D_9E37_0001
PATH_FEATURE = 0xA70F_4A7D_9E37_0002

# Linear path bond classes.
SINGLE_BOND = 1
DOUBLE_BOND = 2
TRIPLE_BOND = 3
FLEX_BOND = 4


__all__ = [
    "ATOM_FEATURE",
    "DOUBLE_BOND",
    "FLEX_BOND",
    "PATH_FEATURE",
    "SINGLE_BOND",
    "TRIPLE_BOND",
    "UINT64_MASK",
    "WL_COMBINE_CONSTANT",
    "WL_COMBINE_CONSTANT_INT",
    "aromatic_bond_classes",
    "atom_feature",
    "bond_feature_classes",
    "combine_uint64",
    "combine_uint64_scalar",
    "exact_atomic_number_from_words",
    "exact_atomic_number_labels",
    "exact_atomic_number_labels_from_packed_atoms",
    "identity_bond_classes",
    "mix_uint64",
    "mix_uint64_scalar",
    "path_hash",
    "short_ring_bond_classes",
    "signed_values_as_uint64",
]


# --------------------------------------------------------------------------- #
# 64-bit hash mixers (vector + scalar twins).
# --------------------------------------------------------------------------- #


def signed_values_as_uint64(values: np.ndarray) -> LabelVector:
    return values.astype(np.int64, copy=False).view(np.uint64)


def mix_uint64(values: LabelArray) -> LabelArray:
    mixed = values + SPLITMIX64_GAMMA
    mixed = (mixed ^ (mixed >> np.uint64(30))) * SPLITMIX64_MUL1
    mixed = (mixed ^ (mixed >> np.uint64(27))) * SPLITMIX64_MUL2
    return mixed ^ (mixed >> np.uint64(31))


def combine_uint64(left: LabelArray, right: LabelArray) -> LabelArray:
    combined = left ^ (right + WL_COMBINE_CONSTANT + (left << 6) + (left >> 2))
    return mix_uint64(combined.astype(np.uint64, copy=False))


def mix_uint64_scalar(value: int) -> int:
    """Scalar (Python-int) twin of :func:`mix_uint64` for per-element hashing."""
    mixed = (value + SPLITMIX64_GAMMA_INT) & UINT64_MASK
    mixed = ((mixed ^ (mixed >> 30)) * SPLITMIX64_MUL1_INT) & UINT64_MASK
    mixed = ((mixed ^ (mixed >> 27)) * SPLITMIX64_MUL2_INT) & UINT64_MASK
    return (mixed ^ (mixed >> 31)) & UINT64_MASK


def combine_uint64_scalar(left: int, right: int) -> int:
    """Scalar (Python-int) twin of :func:`combine_uint64`."""
    combined = (
        left
        ^ (right + WL_COMBINE_CONSTANT_INT + ((left << 6) & UINT64_MASK) + (left >> 2))
    ) & UINT64_MASK
    return mix_uint64_scalar(combined)


# --------------------------------------------------------------------------- #
# Exact atomic-number labels (single-bit Z masks decode to an element; otherwise
# None, meaning the atom is not an exact element).
# --------------------------------------------------------------------------- #


def exact_atomic_number_labels(atom_masks: AtomMasks) -> list[int | None]:
    return [
        exact_atomic_number_from_words(int(lo), int(hi))
        for lo, hi in zip(atom_masks.Z_lo, atom_masks.Z_hi, strict=True)
    ]


def exact_atomic_number_labels_from_packed_atoms(
    rows: np.ndarray,
) -> list[int | None]:
    labels: list[int | None] = []
    for row in rows:
        z_lo = int(row[0]) | (int(row[1]) << 32)
        z_hi = int(row[2]) | (int(row[3]) << 32)
        labels.append(exact_atomic_number_from_words(z_lo, z_hi))
    return labels


def exact_atomic_number_from_words(z_lo: int, z_hi: int) -> int | None:
    if z_lo and not z_hi and z_lo & (z_lo - 1) == 0:
        return z_lo.bit_length() - 1
    if z_hi and not z_lo and z_hi & (z_hi - 1) == 0:
        return 64 + z_hi.bit_length() - 1
    return None


# --------------------------------------------------------------------------- #
# Atom / path feature hashers (shared by the path fingerprints).
# --------------------------------------------------------------------------- #


def atom_feature(atomic_number: int) -> int:
    return combine_uint64_scalar(ATOM_FEATURE, atomic_number)


def path_hash(
    atom_labels: list[int | None],
    edge_classes: tuple[int, ...],
) -> int:
    value = PATH_FEATURE
    for label in atom_labels:
        if label is None:
            raise ValueError("path hash requires exact atom labels.")
        value = combine_uint64_scalar(value, label)
    for edge_class in edge_classes:
        value = combine_uint64_scalar(value, edge_class)
    return value


def bond_feature_classes(
    flags: int,
    *,
    query: bool,
) -> tuple[int, ...]:
    """The bond classes a path feature records for one edge: presence, and nothing else.

    `query` is accepted and ignored, which is deliberate enough to write down because the
    machinery around it suggests otherwise. The idea was that a *query* bond should expand into
    every concrete class it could match -- `-!@` into single, `~` into all four -- so a query
    fingerprint would be the union of its readings while a molecule's named one class exactly.
    `SINGLE_BOND`, `DOUBLE_BOND` and `TRIPLE_BOND` exist for that, the device ports carry an
    `options[7][4]` table to hold the expansion, and none of it is reached: every present bond,
    on both sides, is `FLEX_BOND`.

    So the folded-path prefilter is weaker than designed and **subset-sound**, which is the
    property that matters. Bond order is not in the feature at all, so a query bond can never
    fail to find its counterpart; and an atom that is not one concrete element labels as `None`,
    which drops its paths rather than emitting a wrong one. Both make a query's fingerprint a
    subset of what the molecule's contains, so the prefilter cannot prune a pair the matcher
    would match. It only lets through more pairs than a sharper one would.

    Making the query side expand for real would sharpen it, and getting that wrong would prune a
    true candidate -- chemistry lost silently, with no refusal and no counter to show it.
    `tests/vesma/test_prefilter_soundness.py` is what makes attempting it safe: over the shipped
    rule library it asserts that every pair the dense matcher matches survives the screen, on the
    host and on Metal. Checked against an injected fault, so it does bite.

    The two device ports -- `metal_pathfp.mm` and `cuda_pathfp.py` -- thread the same inert flag
    through the same dead expansion. That is faithfulness to this function, not three
    independent oversights, and it is why this docstring is the one place it is explained.
    """
    if bond_kind_flags(flags) == 0:
        return ()
    return (FLEX_BOND,)


# --------------------------------------------------------------------------- #
# Bond-classing schemes for the WL identity: the small-ring key and the
# full-aromaticity key. Both reduce a concrete bond matrix to a single class per
# edge so the WL hash is invariant to the bond representation of the input.
# --------------------------------------------------------------------------- #


def identity_bond_classes(bonds: BondMatrix) -> BondMatrix:
    orders = bonds & np.uint8(BondKind.SINGLE | BondKind.DOUBLE | BondKind.TRIPLE)
    aromatic = (bonds & np.uint8(BondKind.DELOCALIZED)) != 0
    return np.where(
        (orders == 0) & aromatic,
        BondKind.DELOCALIZED,
        orders,
    ).astype(np.uint8)


def short_ring_bond_classes(bonds: BondMatrix) -> BondMatrix:
    ring_edges = short_ring_edges(bonds)
    classes = identity_bond_classes(bonds)
    triples = (bonds & np.uint8(BondKind.TRIPLE)) != 0
    classes[ring_edges & ~triples] = np.uint8(BondKind.DELOCALIZED)
    return classes


def aromatic_bond_classes(bonds: BondMatrix) -> BondMatrix:
    aromatic = (bonds & np.uint8(BondKind.DELOCALIZED)) != 0
    classes = identity_bond_classes(bonds)
    classes[aromatic] = np.uint8(BondKind.DELOCALIZED)
    return classes

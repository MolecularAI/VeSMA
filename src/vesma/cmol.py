from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .adjacency import BondMatrix, require_exact_bond_matrix
from .bonds import BondKind
from .gmol_atoms import (
    ALIPHATIC,
    AROMATIC,
    BUCKET_MAX,
    CHARGE_MAX,
    CHARGE_MIN,
    MAX_ISOTOPE,
    Aromaticity,
    AtomVector,
    Charge,
)

AtomMapVector = np.ndarray[tuple[int], np.dtype[np.int32]]


@dataclass(frozen=True)
class CMol:
    """Concrete molecule graph with exact bonds in the shared `bonds` encoding."""

    Z: AtomVector
    M: AtomVector
    A: Aromaticity
    C: Charge
    D: AtomVector
    H: AtomVector
    h: AtomVector
    v: AtomVector
    X: AtomVector
    R: AtomVector
    r: AtomVector
    x: AtomVector
    bonds: BondMatrix
    atom_map: AtomMapVector

    def __post_init__(self) -> None:
        expected_vectors = (
            ("Z", np.int16),
            ("M", np.int16),
            ("A", np.uint8),
            ("C", np.int8),
            ("D", np.int16),
            ("H", np.int16),
            ("h", np.int16),
            ("v", np.int16),
            ("X", np.int16),
            ("R", np.int16),
            ("r", np.int16),
            ("x", np.int16),
            ("atom_map", np.int32),
        )
        for name, dtype in expected_vectors:
            values = getattr(self, name)
            if values.dtype != dtype:
                raise ValueError(f"{name} dtype must be {np.dtype(dtype)}.")
            if values.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional.")

        if self.bonds.dtype != np.uint8:
            raise ValueError("bonds dtype must be uint8.")
        if self.bonds.ndim != 2:
            raise ValueError("Bond matrix must be two-dimensional.")

        atom_count = self.atom_count
        for name in ("M", "A", "C", "D", "H", "h", "v", "X", "R", "r", "x", "atom_map"):
            if getattr(self, name).shape != (atom_count,):
                raise ValueError(f"{name} shape must match atom count.")
        if self.bonds.shape != (atom_count, atom_count):
            raise ValueError("Bond matrix shape must match atom count.")
        require_exact_bond_matrix(
            self.bonds,
            error_message="CMol bonds must use exact bond types only.",
        )
        if np.any((self.A != ALIPHATIC) & (self.A != AROMATIC)):
            raise ValueError("CMol aromaticity values must be ALIPHATIC or AROMATIC.")
        aromatic_bonds = (self.bonds & np.uint8(BondKind.DELOCALIZED)) != 0
        aromatic_atoms = self.A == AROMATIC
        if np.any(
            aromatic_bonds & ~(aromatic_atoms[:, None] & aromatic_atoms[None, :])
        ):
            raise ValueError("CMol aromatic bonds must connect aromatic atoms.")
        if np.any(aromatic_atoms & ~np.any(aromatic_bonds, axis=1)):
            raise ValueError("CMol aromatic atoms must belong to aromatic bonds.")
        for name in ("D", "H", "h", "v", "X", "R", "r", "x"):
            if np.any(getattr(self, name) < -1):
                raise ValueError(f"CMol requires valid exact {name} values (>= -1).")
        if np.any((self.H >= 0) & (self.h >= 0) & (self.H < self.h)):
            raise ValueError("CMol requires H >= h for every atom.")
        # `-1` means "not determined", which is a legitimate state for either field on its own.
        # Knowing the total while not knowing the implicit part is not: the split is `H = explicit
        # + h`, so a known `H` with an unknown `h` leaves the explicit count unknowable, and a
        # consumer computing `H - h` -- `identity.py` for the WL seed, `smiles.py` for the writer
        # -- silently reads one hydrogen too many. That combination is how a nitrile nitrogen
        # reached a keyer as `H=0, h=-1` and serialised as `[NH]`, and the one check guarding this
        # invariant skipped it because it required both fields to be known first.
        if np.any((self.H >= 0) & (self.h < 0)):
            raise ValueError(
                "CMol with a determined H requires a determined h: the explicit count is "
                "H - h, so a known total with an unknown implicit part is not a molecule."
            )
        if np.any(self.M < 0) or np.any(self.M > MAX_ISOTOPE):
            raise ValueError(
                "CMol isotope must be a mass number of at most "
                f"{MAX_ISOTOPE}, or MASS_UNLABELED for natural abundance; "
                "a molecule never carries a query wildcard."
            )
        # The same contract the SMILES reading enforces on text, enforced here on the
        # model, so a molecule can never hold a property the writer would emit and the
        # reader would then refuse: `C=+5` serializes as `[CH4+5]` and `H=8` as `[CH8]`,
        # and neither reads back. The masks saturate at these bounds, so a value outside
        # them is one the encoding could only keep by rounding it. Counts may still be
        # -1, the "not computed" sentinel a narrowed RingDescriptors leaves behind.
        if np.any((self.C < CHARGE_MIN) | (self.C > CHARGE_MAX)):
            raise ValueError(
                f"CMol charge must lie in {CHARGE_MIN:+d} to {CHARGE_MAX:+d}, "
                "the range the atom masks hold exactly."
            )
        if np.any(self.H > BUCKET_MAX):
            raise ValueError(
                f"CMol hydrogen count must be at most {BUCKET_MAX}, "
                "the largest the count buckets hold exactly."
            )

    @property
    def atom_count(self) -> int:
        return int(self.Z.shape[0])


def cmol_union(parts: Sequence[CMol]) -> CMol:
    """One molecule holding every part, side by side and unbonded.

    What a *bimolecular* reaction pattern needs something to match against. A forward template for a
    coupling has a two-component left-hand side, and a subgraph match is a match into one graph, so
    the two starting materials have to be one graph first. Retro application never needed this --
    it starts from a single product and *produces* components -- which is why it does not exist
    already.

    Every per-atom property is unchanged by standing two molecules next to each other, so each
    vector is a concatenation and the bond matrix is block diagonal with no bonds across the blocks.
    Ring membership included: a disjoint union creates no ring, so ``R``, ``r`` and ``x`` carry over
    untouched rather than needing perception again.

    **Atom maps are cleared.** They number atoms *within* one template application, so two parts
    that each carry a map 1 would give one molecule with two atoms claiming the same number -- a
    provenance statement that is simply false. A combined molecule exists to be matched and its
    products to be keyed, and neither reads a map, so dropping them loses nothing and asserts
    nothing untrue.
    """
    if not parts:
        raise ValueError("a union needs at least one molecule.")
    if len(parts) == 1:
        return parts[0]
    sizes = [part.atom_count for part in parts]
    total = sum(sizes)
    bonds = np.zeros((total, total), dtype=np.uint8)
    at = 0
    for part, size in zip(parts, sizes):
        bonds[at : at + size, at : at + size] = part.bonds
        at += size
    joined = {
        name: np.concatenate([getattr(part, name) for part in parts])
        for name in ("Z", "M", "A", "C", "D", "H", "h", "v", "X", "R", "r", "x")
    }
    return CMol(
        **joined,
        bonds=bonds,
        atom_map=np.zeros(total, dtype=np.int32),
    )


__all__ = [
    "CMol",
    "cmol_union",
]

"""Molecular composition and exact mass, for reading a report rather than matching a graph.

Nothing in matching or precursor construction needs either: the engine counts atoms by `Z` and
hydrogens by the configuration it chose, and a formula is a rendering of what it already decided.
What wants them is a *report* -- what a chemist should expect to see in the pot, and where on a
mass spectrum to look for it -- which is a different consumer with a different requirement.

Hydrogen is included here and deliberately. :func:`vesma.reaction.heavy_atom_formula` omits it,
because it names a component the model decided was not a precursor and the question there is only
"what was it"; the hydrogen count of a fragment the engine refused is not worth claiming. For an
accepted outcome the hydrogens are pinned and the count is exact, and without them a by-product
formula is wrong in the way that matters: an esterification sheds `H2O` at 18.011, not `O` at
15.995.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

from .cmol import CMol
from .symbols import Z_TO_ELEMENT

__all__ = [
    "ELECTRON_MASS",
    "MONOISOTOPIC_MASS",
    "Coincidence",
    "composition",
    "difference",
    "exact_mass",
    "heavy_only",
    "hill_formula",
    "mass_coincidence",
    "net_charge",
    "nominal_mass",
]

ELECTRON_MASS = 0.00054858

# Monoisotopic masses of the most abundant isotope, in u, for the elements this engine declares.
# Monoisotopic rather than average, because a mass spectrum resolves isotopes and the peak a reader
# is looking for is the lightest one. CODATA/AME values rounded to five decimals, which is finer
# than any instrument this would be read against.
MONOISOTOPIC_MASS: dict[int, float] = {
    1: 1.00783,
    5: 11.00931,
    6: 12.00000,
    7: 14.00307,
    8: 15.99491,
    9: 18.99840,
    11: 22.98977,
    12: 23.98504,
    13: 26.98154,
    14: 27.97693,
    15: 30.97376,
    16: 31.97207,
    17: 34.96885,
    19: 38.96371,
    25: 54.93805,
    29: 62.92960,
    30: 63.92914,
    32: 73.92118,
    33: 74.92160,
    34: 79.91652,
    35: 78.91834,
    50: 119.90220,
    51: 120.90381,
    52: 129.90622,
    53: 126.90447,
    76: 191.96148,
}


def composition(molecule: CMol) -> Counter[int]:
    """Atoms by proton number, hydrogen included.

    The hydrogen count is the molecule's own pinned total per atom, so it is whatever the engine
    concluded rather than a valence re-derivation: an outcome the retro path accepted carries a
    hydrogen count it already had to commit to, and a second opinion here could only disagree with
    the molecule that was actually emitted.
    """
    counts = Counter(int(z) for z in molecule.Z.tolist())
    hydrogens = sum(int(count) for count in molecule.H.tolist())
    if hydrogens:
        counts[1] += hydrogens
    return +counts


def net_charge(molecule: CMol) -> int:
    """The molecule's formal charge, summed over its atoms."""
    return sum(int(charge) for charge in molecule.C.tolist())


HYDROGEN = 1


def heavy_only(counts: Counter[int]) -> Counter[int]:
    """The same composition with hydrogen dropped.

    Hydrogen is an attached atom here, never a node: `[H][H]` occupies no configuration, so a
    reaction that formally consumes or releases H2 cannot name the reagent that supplied it. A
    hydride reduction gains two hydrogens from a borohydride and an aromatisation loses two to the
    air, and in both the hydrogen *is* the reagent's contribution rather than a gap in the account.

    So a redox step is balanced on heavy atoms and the hydrogen difference is the reagent's share,
    which is a statement worth making instead of an imbalance worth tolerating. Used for exactly
    that: see `tests/vertsy/test_forward_templates.py::REDOX_HYDROGEN`, where the *direction* of
    the difference is also required to agree with the row's condition tag.
    """
    return Counter({z: n for z, n in counts.items() if z != HYDROGEN})


def difference(
    before: Counter[int], after: Counter[int]
) -> tuple[Counter[int], Counter[int]]:
    """``(lost, gained)`` between two compositions, each a count of what moved one way.

    Two values rather than a signed one, because they mean different things about a template. What
    is *lost* going forward is the by-product: the leaving group, the water, the hydrogen halide --
    the extra peak. What is *gained* is an atom the rule invented, which can only have come from a
    reagent the rule did not name, so a non-empty ``gained`` says the balance is open and the
    by-product derived from ``lost`` alone is incomplete.
    """
    lost = Counter(
        {z: n - after.get(z, 0) for z, n in before.items() if n > after.get(z, 0)}
    )
    gained = Counter(
        {z: n - before.get(z, 0) for z, n in after.items() if n > before.get(z, 0)}
    )
    return lost, gained


def hill_formula(counts: Counter[int]) -> str:
    """Hill order -- carbon, then hydrogen, then the rest alphabetically -- as one string."""
    present = {z: n for z, n in counts.items() if n}
    ordered = sorted(
        present,
        key=lambda z: (
            {"C": 0, "H": 1}.get(Z_TO_ELEMENT.get(z, ""), 2),
            Z_TO_ELEMENT.get(z, ""),
        ),
    )
    return "".join(
        f"{Z_TO_ELEMENT.get(z, f'Z{z}')}{present[z] if present[z] > 1 else ''}"
        for z in ordered
    )


def exact_mass(counts: Counter[int], *, charge: int = 0) -> float:
    """Monoisotopic mass of a composition, less the mass of any electrons it gained.

    ``charge`` is the species' net charge, and an anion is *heavier* than its neutral formula by
    one electron per negative charge. 0.5 mu is below any instrument this would be read against,
    but RDKit's `ExactMolWt` models it and agreeing with an outside implementation is worth more
    than an argument about whether it matters: `[AlH4-]` is 31.01339 and not 31.01284.

    Raises on an element with no mass on file rather than returning a number that is quietly short:
    a mass spectrum is read against this, and a missing contribution looks exactly like a different
    compound.
    """
    missing = sorted(z for z in counts if counts[z] and z not in MONOISOTOPIC_MASS)
    if missing:
        names = ", ".join(Z_TO_ELEMENT.get(z, f"Z{z}") for z in missing)
        raise ValueError(f"no monoisotopic mass on file for {names}.")
    return (
        sum(MONOISOTOPIC_MASS[z] * count for z, count in counts.items())
        - charge * ELECTRON_MASS
    )


Coincidence = Literal["same formula", "same nominal mass", "resolved"]
"""How badly two compositions collide on a mass spectrum, worst first.

Three tiers and not one tolerance, because they are three different instruments rather than three
points on one scale.

``same formula`` is two compositions that are the same multiset of atoms -- isomers, or the very
same molecule reached twice. Their exact masses are *identical*, so **no mass spectrometer at any
resolution separates them**, ever. Only chromatography or a different kind of measurement can.

``same nominal mass`` is two different formulas whose integer masses agree: `C3H8` at 44.0626
against `C2H4O` at 44.0262, 36 mDa apart. A unit-resolution instrument reports one peak; a
high-resolution one reports two. So this is an answer about the kit on the bench rather than about
the chemistry.

``resolved`` is everything else, which any instrument separates.

A single ppm or mDa window would conflate the first two, and the difference between them is the
difference between "buy a better instrument" and "you cannot tell these apart by mass at all".
"""


def nominal_mass(counts: Counter[int]) -> int:
    """The integer mass: each element's most abundant isotope, rounded, summed.

    Rounding :data:`MONOISOTOPIC_MASS` rather than carrying a second table, which is exact for
    every element on file -- chlorine's 34.96885 rounds to 35 and bromine's 78.91834 to 79 -- and
    cannot drift away from the masses :func:`exact_mass` reports.

    The charge is ignored: an electron is 0.00055 u and no integer mass notices it.
    """
    missing = sorted(z for z in counts if z not in MONOISOTOPIC_MASS)
    if missing:
        names = ", ".join(Z_TO_ELEMENT.get(z, f"Z{z}") for z in missing)
        raise ValueError(f"no monoisotopic mass on file for {names}.")
    return sum(round(MONOISOTOPIC_MASS[z]) * count for z, count in counts.items())


def mass_coincidence(left: Counter[int], right: Counter[int]) -> Coincidence:
    """Which tier of :data:`Coincidence` these two compositions fall into.

    Compared as compositions and not as masses, so ``same formula`` is decided by the atoms rather
    than by a float equality that happens to hold.
    """
    if left == right:
        return "same formula"
    if nominal_mass(left) == nominal_mass(right):
        return "same nominal mass"
    return "resolved"

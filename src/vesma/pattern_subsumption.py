"""When is one rule redundant because another already covers it?

Identity finds two spellings of one rule. This finds the asymmetric case, which is where the defects
turn out to be: rule P is **subsumed** by rule Q when every match P has, Q also has, and both do the
same thing to it. P then adds nothing a search could use -- and, more usefully, the pair is worth
reading, because a rule broader than its near-twin is broader *somewhere* and that somewhere may be
a mistake. `iSCB 70` was `MC 75` minus one `-!@`, and the breadth it bought was opening lactones.

**The direction is containment, where matching is overlap.** `vesma.cpu_matcher` asks whether a
pattern mask and a molecule value have any bit in common, because a molecule holds one value per
property and the pattern offers a set. Here both sides are *sets*, so the question is whether one
set lies inside the other: `(narrow & broad) == narrow`, field by field. Bond flags are bitsets too
-- kind in the low nibble, topology above it -- so one `&` covers both.

**Declining is the safe answer and this declines readily.** A false "subsumed" would invite deleting
a rule that does different chemistry; a false "not subsumed" only keeps a rule that costs search
time. So every condition below is required rather than inferred, and anything this cannot decide
comes back `False`.

Four conditions beyond the embedding, and the last two are the ones that get forgotten.

*The edits must agree* under the embedding -- same atoms deleted, same correspondence, same
assertions written.

*And the same bonds must be built.* `MC 64a` and `MC 65a` match the same thioether, delete nothing
and each invent one chlorine; they differ only in which bond they break, aryl-S against S-alkyl.
Every per-atom condition passes on that pair, so without an equality over the right-hand bonds this
reported two opposite disconnections as one rule -- and reported `P 4`, which reduces a phosphoryl
to `P-O`, as covering `P 2b`, which keeps `P=O`.

*Every atom of P outside the embedding must be inert*: carried through untouched. An atom P matches
and Q does not is extra *specificity*, which is fine, but only if P does nothing with it. The
subtlety is that "does nothing" cannot mean "asserts nothing": `[CH3:3]-[C:1]-[Cl]>>[CH3:3]-[C:1]-[OH]`
re-asserts the methyl exactly as the left-hand side already forced it, and that is precisely the
redundancy worth finding.

*And its bonds must be rebuilt with the kind they had.* `removed_lhs_edges` holds **every** left-hand
edge, so the edit clears all of them and re-adds what the right-hand side spells. A left-hand bond
mask admitting several kinds is therefore a *rewrite*: `[CH3:3]~[C:1]-[Cl]>>[CH3:3]-[C:1]-[OH]`
normalises a double bond to single, so it is not subsumed by the `-` version even though its mask is
wider.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import permutations

import numpy as np

from .bonds import BOND_TOPOLOGY_MASK, bond_kind_flags
from .gmol import GMol
from .gmol_atoms import MASS_ANY
from .reaction import ReactionPattern

__all__ = [
    "BITSET_FIELDS",
    "RhsProgram",
    "added_states",
    "atom_masks_contain",
    "bond_flags_contain",
    "lhs_embeddings",
    "rhs_bonds_agree",
    "rhs_edges",
    "rhs_node",
    "rhs_program",
    "subsumes",
]

BITSET_FIELDS = ("Z_lo", "Z_hi", "A", "C", "D", "H", "h", "v", "X", "R", "r", "x")
"""The mask fields that are sets of allowed values, so containment is `(narrow & broad) == narrow`.

`M` is absent on purpose: it is an exact-or-any isotope *value*, not a set, and its containment rule
is the one `cpu_matcher.mass_compatibility` already uses -- `MASS_ANY` accepts anything, and any
other value must match exactly.
"""


@dataclass(frozen=True)
class RhsProgram:
    """What a rule writes onto one surviving atom, and nothing about how it was spelled.

    Read from the edit rather than from the right-hand masks, for the reason
    :func:`vesma.pattern_identity.rhs_atom_key` does: `RHS_ANNOTATIONS` are recomputed afterwards as
    mismatches and never enforced, so two rules differing only there are one rule.
    """

    z: int
    m: int
    c: int
    c_is_explicit: bool
    h_total: int


def atom_masks_contain(broad: GMol, wide: int, narrow: GMol, tight: int) -> bool:
    """Does every atom `narrow`'s predicate admits, `broad`'s predicate admit too?"""
    left, right = broad.atom_masks, narrow.atom_masks
    if not predicate_contains(broad, wide, narrow, tight):
        return False
    for field in BITSET_FIELDS:
        allowed = int(getattr(left, field)[wide])
        wanted = int(getattr(right, field)[tight])
        if wanted & allowed != wanted:
            return False
    mass = int(left.M[wide])
    return mass == MASS_ANY or mass == int(right.M[tight])


def predicate_contains(broad: GMol, wide: int, narrow: GMol, tight: int) -> bool:
    """Conservative: a recursive predicate on the broad atom is only known to admit what the
    narrow atom does when the narrow atom states the very same one."""
    index = int(broad.atom_predicate[wide])
    if index < 0:
        return True
    other = int(narrow.atom_predicate[tight])
    return other >= 0 and broad.predicates[index].key == narrow.predicates[other].key


def bond_flags_contain(allowed: int, wanted: int) -> bool:
    """Does a broad bond constraint admit everything a narrow one does?

    Zero on the broad side is "no constraint here" and admits anything, including the absence of a
    bond. Zero on the narrow side where the broad side asks for something is the failing case: the
    narrow rule matches molecules with no such bond, which the broad rule would refuse.
    """
    if allowed == 0:
        return True
    if wanted == 0:
        return False
    return bool(
        bond_kind_flags(wanted) & bond_kind_flags(allowed) == bond_kind_flags(wanted)
        and (wanted & BOND_TOPOLOGY_MASK) & (allowed & BOND_TOPOLOGY_MASK)
        == (wanted & BOND_TOPOLOGY_MASK)
    )


def lhs_embeddings(
    broad: GMol, narrow: GMol, *, max_leaves: int = 20000
) -> Iterator[tuple[int, ...]]:
    """Every injective placement of `broad`'s atoms onto `narrow`'s that widens every constraint.

    A plain backtracking search, where the matcher would bring arc consistency and three backends:
    the graphs here are patterns, a few atoms each, and there are a few hundred of them. The budget
    exists so a pathological pair declines rather than running away -- every other search in this
    repository carries one.
    """
    if broad.atom_count > narrow.atom_count:
        return
    candidates = [
        [
            tight
            for tight in range(narrow.atom_count)
            if atom_masks_contain(broad, wide, narrow, tight)
        ]
        for wide in range(broad.atom_count)
    ]
    if any(not row for row in candidates):
        return
    # Most-constrained first, which is what makes a naive search tolerable here.
    order = sorted(range(broad.atom_count), key=lambda wide: len(candidates[wide]))
    placement = [-1] * broad.atom_count
    used = [False] * narrow.atom_count
    leaves = 0

    def walk(depth: int) -> Iterator[tuple[int, ...]]:
        nonlocal leaves
        if depth == len(order):
            leaves += 1
            yield tuple(placement)
            return
        wide = order[depth]
        for tight in candidates[wide]:
            if used[tight] or leaves >= max_leaves:
                continue
            if not all(
                bond_flags_contain(
                    int(broad.bonds[wide, other]),
                    int(narrow.bonds[tight, placement[other]]),
                )
                for other in order[:depth]
            ):
                continue
            placement[wide], used[tight] = tight, True
            yield from walk(depth + 1)
            placement[wide], used[tight] = -1, False

    yield from walk(0)


def rhs_program(pattern: ReactionPattern) -> dict[int, RhsProgram]:
    """Per surviving left-hand atom, what the rule writes onto it."""
    edit = pattern.edit
    return {
        int(lhs): RhsProgram(
            z=int(edit.mapped_rhs_z[at]),
            m=int(edit.mapped_rhs_m[at]),
            c=int(edit.mapped_rhs_c[at]),
            c_is_explicit=bool(edit.mapped_rhs_c_is_explicit[at]),
            h_total=int(edit.mapped_rhs_h_total[at]),
        )
        for at, lhs in enumerate(np.asarray(edit.mapped_lhs_indices).tolist())
    }


def forced_singleton(graph: GMol, atom: int, field: str) -> int | None:
    """The one value a mask admits, or ``None`` when it admits several."""
    mask = int(getattr(graph.atom_masks, field)[atom])
    return mask.bit_length() - 1 if mask and not mask & (mask - 1) else None


def is_inert(pattern: ReactionPattern, atom: int) -> bool:
    """Is this left-hand atom carried through with nothing done to it?

    Not "asserts nothing" -- a rule may restate what its own left-hand side already forced, and that
    restatement is exactly the redundancy worth finding. So each assertion is checked against the
    mask: writing `[CH3:3]` over an atom the query already pinned to one carbon with three hydrogens
    changes nothing.

    And every incident bond must be rebuilt with the kind it had. The edit clears every left-hand
    edge and re-adds what the right-hand side spells, so a left-hand mask admitting several kinds
    makes the rule a *rewrite* of that bond rather than a query on it.
    """
    program = rhs_program(pattern).get(atom)
    if program is None:
        return False  # deleted, which is not nothing
    if program.z != -1 and forced_singleton(pattern.lhs, atom, "Z_lo") != program.z:
        return False
    if program.m != MASS_ANY and int(pattern.lhs.atom_masks.M[atom]) != program.m:
        return False
    if program.c_is_explicit:
        forced = forced_singleton(pattern.lhs, atom, "C")
        if forced is None or forced - 3 != program.c:
            return False
    if (
        program.h_total != -1
        and forced_singleton(pattern.lhs, atom, "H") != program.h_total
    ):
        return False
    # Every bond this atom carries must come back as the single kind the query already pinned.
    rhs_of = dict(
        zip(
            np.asarray(pattern.edit.mapped_lhs_indices).tolist(),
            np.asarray(pattern.edit.mapped_rhs_indices).tolist(),
            strict=True,
        )
    )
    written = {
        tuple(sorted((int(left), int(right)))): int(kind)
        for (left, right), kind in zip(
            np.asarray(pattern.edit.added_rhs_edges).reshape(-1, 2).tolist(),
            np.asarray(pattern.edit.added_rhs_bond_types).tolist(),
            strict=True,
        )
    }
    for other in range(pattern.lhs.atom_count):
        flag = int(pattern.lhs.bonds[atom, other])
        if not flag:
            continue
        kinds = bond_kind_flags(flag)
        if kinds & (kinds - 1):
            return False  # a multi-kind query bond is a rewrite, not a carry-through
        here, there = rhs_of.get(atom), rhs_of.get(other)
        if here is None or there is None:
            return False
        if written.get(tuple(sorted((int(here), int(there))))) != kinds:
            return False
    return True


def rhs_node(pattern: ReactionPattern) -> dict[int, tuple[str, int]]:
    """Each right-hand atom named by *what it is* rather than by where it sits.

    A carried-through atom is named by the left-hand atom it came from, so the name survives the
    embedding; an invented one is named by its slot in the added arrays, which is the only handle it
    has. Right-hand indices themselves are numbering and must not reach a comparison.
    """
    edit = pattern.edit
    names: dict[int, tuple[str, int]] = {
        int(right): ("lhs", int(left))
        for left, right in zip(
            np.asarray(edit.mapped_lhs_indices).tolist(),
            np.asarray(edit.mapped_rhs_indices).tolist(),
            strict=True,
        )
    }
    for slot, right in enumerate(np.asarray(edit.added_rhs_indices).tolist()):
        names[int(right)] = ("add", slot)
    return names


def rhs_edges(pattern: ReactionPattern) -> dict[tuple[tuple[str, int], ...], int]:
    """Every bond the rule writes, keyed by those names and valued by the kind written.

    `removed_lhs_edges` clears all of the matched edges, so `added_rhs_edges` is not a delta -- it is
    the whole right-hand bond structure, which is what makes an equality the right comparison.
    """
    names = rhs_node(pattern)
    edit = pattern.edit
    return {
        tuple(sorted((names[int(left)], names[int(right)]))): int(kind)
        for (left, right), kind in zip(
            np.asarray(edit.added_rhs_edges).reshape(-1, 2).tolist(),
            np.asarray(edit.added_rhs_bond_types).tolist(),
            strict=True,
        )
    }


def added_states(pattern: ReactionPattern) -> list[tuple[int, int, int, int]]:
    """What each invented atom is, by slot. Position is the slot, not a numbering of the molecule."""
    edit = pattern.edit
    return [
        (
            int(edit.added_rhs_z[at]),
            int(edit.added_rhs_m[at]),
            int(edit.added_rhs_c[at]),
            int(edit.added_rhs_h_total[at]),
        )
        for at in range(len(np.asarray(edit.added_rhs_indices)))
    ]


def rhs_bonds_agree(
    broad: ReactionPattern, narrow: ReactionPattern, placement: Sequence[int]
) -> bool:
    """Do the two rules build the same molecule out of the atoms they share?

    The condition this file was missing, and the omission was not subtle: `MC 64a` and `MC 65a` match
    the same thioether, delete nothing, and each invent one chlorine, so every other condition
    passed -- while one breaks the aryl-S bond and the other the S-alkyl bond. Equal atoms and equal
    edits are not equal chemistry; *which atoms end up bonded* is the disconnection.

    Compared as an **equality** over the shared part. `narrow`'s bonds to atoms outside the embedding
    are excluded, because `broad` never clears those and :func:`is_inert` has already required
    `narrow` to rebuild each of them as it found it.

    Invented atoms carry no identity across the two rules, so a bijection between them is searched
    rather than assumed: slot order is an artefact of writing order, and two rules adding `Cl` and
    `Br` could pair them either way round. There are rarely more than two.
    """
    embedded = set(placement)

    def shared(node: tuple[str, int]) -> bool:
        return node[0] == "add" or node[1] in embedded

    narrow_edges = {
        key: kind
        for key, kind in rhs_edges(narrow).items()
        if all(shared(node) for node in key)
    }
    broad_edges = rhs_edges(broad)
    broad_added, narrow_added = added_states(broad), added_states(narrow)
    for pairing in permutations(range(len(narrow_added))):
        if any(
            broad_added[slot] != narrow_added[other]
            for slot, other in enumerate(pairing)
        ):
            continue

        def under(
            node: tuple[str, int], pairing: Sequence[int] = pairing
        ) -> tuple[str, int]:
            """One of `broad`'s right-hand atoms named the way `narrow` would name it."""
            kind_of, index = node
            if kind_of == "add":
                return "add", pairing[index]
            return "lhs", placement[index]

        translated = {
            tuple(sorted(under(node) for node in key)): kind
            for key, kind in broad_edges.items()
        }
        if translated == narrow_edges:
            return True
    return False


def edits_agree(
    broad: ReactionPattern, narrow: ReactionPattern, placement: Sequence[int]
) -> bool:
    """Do the two rules do the same thing to the atoms they share?

    Same atoms deleted, same correspondence, same assertions, same bonds built. Compared through the
    embedding, never positionally -- the mapped-index arrays are ordered by map number, so their
    positions encode a numbering that is none of this function's business.
    """
    broad_deleted = {
        int(i) for i in np.asarray(broad.edit.deleted_lhs_indices).tolist()
    }
    narrow_deleted = {
        int(i) for i in np.asarray(narrow.edit.deleted_lhs_indices).tolist()
    }
    if {placement[i] for i in broad_deleted} != narrow_deleted:
        return False
    broad_program, narrow_program = rhs_program(broad), rhs_program(narrow)
    for wide, tight in enumerate(placement):
        if (wide in broad_program) != (tight in narrow_program):
            return False
        if wide in broad_program and broad_program[wide] != narrow_program[tight]:
            return False
    # The same atoms invented, in the same state, before the bonds between them are matched up.
    if sorted(added_states(broad)) != sorted(added_states(narrow)):
        return False
    return rhs_bonds_agree(broad, narrow, placement)


def subsumes(
    broad: ReactionPattern, narrow: ReactionPattern, *, max_leaves: int = 20000
) -> bool:
    """Does every match of `narrow`, doing the same thing, also match `broad`?

    True means `narrow` adds nothing `broad` does not already offer -- so the pair is worth reading,
    and `narrow` is a candidate for removal once a chemist agrees the breadth was intended.

    Grouping is not handled and its presence declines the pair. Zero-level grouping is an equality
    between two relations rather than a constraint that widens, so it needs an implication test of
    its own; until that exists, a rule stating its molecularity is never reported as redundant.
    """
    for graph in (broad.lhs, narrow.lhs, broad.rhs, narrow.rhs):
        if np.any(graph.atom_group):
            return False
    for placement in lhs_embeddings(broad.lhs, narrow.lhs, max_leaves=max_leaves):
        if not edits_agree(broad, narrow, placement):
            continue
        outside = set(range(narrow.lhs.atom_count)) - set(placement)
        if all(is_inert(narrow, atom) for atom in outside):
            return True
    return False

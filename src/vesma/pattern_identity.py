"""When are two reaction templates the same rule, written differently?

A text comparison answers "are these the same string", which is not the question. `iSCB 63a` is
`Hartenfeller 29a` with different atom-map numbers, and the two differ in no way a matcher could
ever observe. Someone noticed that one and hand-rewrote `63` with `29`'s numbering so the text check
would fire -- which worked until zero-level grouping was applied to one side and not the other, at
which point four rules silently began compiling as distinct and nothing failed.

So the answer has to come from the graph. This builds a **certificate**: a canonical form of the
pattern that is identical for two spellings of one rule and different for two rules, computed
exactly rather than hashed.

**Three things are bookkeeping and must not reach it.** Atom numbering, because that is what a
canonical order exists to remove. Atom-*map* numbers, because `[C:1]...[N:2]` and `[C:7]...[N:3]`
state the same correspondence -- so the correspondence travels as an *edge* between partners rather
than as a number on each. And zero-level group *ids*, for exactly the same reason: `(A).(B)` numbers
its groups 1 and 2 arbitrarily, so what travels is the relation `same group` or `other group`.

**Everything a matcher reads must reach it.** The atom predicates are the compiled mask words, which
is what makes this a *semantic* key rather than a syntactic one: `[#6]` and `[C,c]` compile to
identical masks, as do `[#6;X4]` and `[CX4,cX4]`, so the certificate cannot tell those spellings
apart -- correctly, since neither can the matcher. Bond flags travel whole, topology bits included,
because `-!@` and `-` are different rules.

Both sides live in **one** graph. A reaction is not two patterns; it is two patterns plus the
correspondence between them, and splitting them would lose exactly the part that says which bond
moves.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .canonicalization import (
    RankingState,
    canonical_component,
    connected_components_from_neighbors,
)
from .gmol import GMol
from .reaction import ReactionPattern

__all__ = [
    "EDGE_MAP",
    "EDGE_OTHER_GROUP",
    "EDGE_SAME_GROUP",
    "PATTERN_CERTIFICATE_SCHEMA",
    "PatternCertificate",
    "pattern_certificate",
    "pattern_ranking_state",
]

# Edge labels for the three relations that are not bonds. A bond flag occupies six bits -- four of
# kind and two of topology -- so every value below is beyond any flag and the two namespaces cannot
# collide. Asserted where the matrix is built rather than trusted.
BOND_FLAG_CEILING = 64
EDGE_MAP = 64
"""This left-hand atom becomes that right-hand atom. Carries the correspondence, not its number."""
EDGE_SAME_GROUP = 65
"""Both atoms are in one zero-level group, so they must match within one molecule."""
EDGE_OTHER_GROUP = 66
"""The atoms are in different zero-level groups, so they must match in different molecules."""
EDGE_RHS_BOND = 128
"""Added to a *resolved* right-hand bond kind, so the two sides' bond labels cannot be confused.

The right-hand side is a rewrite program, not a query, so what matters there is the concrete kind
the edit will write and not the flags the author spelled. `>>[C:1][O:2]` and `>>[C:1]-[O:2]` carry
different mask flags -- 57 against 49 -- and resolve to the same single bond, so a certificate
reading `rhs.bonds` would call one rule two."""

# Namespaces the certificate the way `SHORT_RING_WL_SCHEMA` and `AROMATIC_WL_SCHEMA` namespace the
# molecule tokens: a change to what the certificate reads must not look like a change of rule.
PATTERN_CERTIFICATE_SCHEMA = 1

MASK_FIELDS = ("Z_lo", "Z_hi", "A", "C", "D", "H", "h", "v", "X", "R", "r", "x")
"""Every bitset field of `AtomMasks`, in a fixed order. `M` is read separately: it is an exact-or-any
isotope *value* rather than a set, so it cannot be compared the same way."""

PatternCertificate = tuple[object, ...]


def lhs_atom_key(graph: GMol, atom: int) -> tuple[int, ...]:
    """A left-hand atom's whole predicate: every mask word, verbatim.

    All of it, because the left-hand side *is* a query and the matcher reads every field. That is
    also what makes the key semantic: identical masks mean identical predicates, so `[#6]` and
    `[C,c]` arrive here as one atom.
    """
    masks = graph.atom_masks
    return (
        0,
        *(int(getattr(masks, field)[atom]) for field in MASK_FIELDS),
        int(masks.M[atom]),
        predicate_key_number(graph, atom),
    )


def predicate_key_number(graph: GMol, atom: int) -> int:
    """The atom's recursive predicate as a stable integer; ``0`` for none.

    Part of the key because two rules differing only in a ``$(...)`` have identical masks, and
    must still not be one rule. Hashed rather than carried as text so keys stay integer tuples.
    """
    index = int(graph.atom_predicate[atom])
    if index < 0:
        return 0
    digest = hashlib.sha256(graph.predicates[index].key.encode()).digest()
    return 1 + int.from_bytes(digest[:8], "little")


def rhs_atom_key(pattern: ReactionPattern, atom: int) -> tuple[int, ...]:
    """A right-hand atom, read from the **edit** rather than from its masks.

    The right-hand side is a rewrite program and most of its query spelling is not obeyed:
    `RHS_ANNOTATIONS = ("A", "D", "X", "v", "R", "r", "x")` are read and recomputed afterwards as
    annotation *mismatches*, never enforced (`reaction.py:175`). Measured: `>>[C:1]-[OH]` and
    `>>[CX4:1]-[OH]` produce identical precursors, so they are one rule -- and a certificate built
    from `rhs.atom_masks` calls them two.

    What is obeyed is exactly what the edit extracted: the element, the isotope, the charge when the
    mask was single-bit enough to assert one, and the hydrogen total when written. The role --
    carried through from the left, or created here -- is part of it too, since an atom the rule
    *makes* is not an atom it *keeps*.
    """
    edit = pattern.edit
    for role, indices, zs, ms, cs, explicits, hs in (
        (
            1,
            edit.mapped_rhs_indices,
            edit.mapped_rhs_z,
            edit.mapped_rhs_m,
            edit.mapped_rhs_c,
            edit.mapped_rhs_c_is_explicit,
            edit.mapped_rhs_h_total,
        ),
        (
            2,
            edit.added_rhs_indices,
            edit.added_rhs_z,
            edit.added_rhs_m,
            edit.added_rhs_c,
            None,
            edit.added_rhs_h_total,
        ),
    ):
        positions = np.flatnonzero(np.asarray(indices) == atom)
        if not positions.size:
            continue
        at = int(positions[0])
        return (
            role,
            int(zs[at]),
            int(ms[at]),
            int(cs[at]),
            # An added atom always asserts its charge; a mapped one only when the mask said so,
            # and `exact_c` yields 0 both for "assert neutral" and "mask merely allows neutral".
            1 if explicits is None else int(bool(explicits[at])),
            int(hs[at]),
        )
    raise ValueError(
        f"right-hand atom {atom} is neither mapped nor added, which the edit cannot produce."
    )


def pattern_ranking_state(
    pattern: ReactionPattern, *, grouping: bool = True
) -> RankingState:
    """Both sides of the reaction as one labelled graph, ready to canonicalise.

    Pass ``grouping=False`` to leave the zero-level group relation out, which asks a different and
    useful question: *the same rule up to what it claims about its own molecularity*. That is what a
    half-finished grouping migration looks like, and it is the one comparison where the grouping is
    the thing being varied rather than part of the rule.

    The returned state's `bonds` matrix holds **edge labels**, of which bond flags are a subset --
    see `EDGE_MAP` and friends. It is the same shape and dtype a bond matrix has, and
    :func:`vesma.canonicalization.refined_ranks` only ever reads it as an integer label, which is
    why the whole canonicaliser can be reused unchanged on a graph it was not written for.
    """
    lhs, rhs = pattern.lhs, pattern.rhs
    offset = lhs.atom_count
    size = offset + rhs.atom_count
    labels = np.zeros((size, size), dtype=np.uint8)

    lhs_flags = lhs.bonds
    if lhs_flags.size and int(lhs_flags.max()) >= BOND_FLAG_CEILING:
        raise ValueError(
            "a bond flag reached the edge-label namespace; the certificate's relation "
            "labels would collide with it."
        )
    labels[:offset, :offset] = lhs_flags

    # The right-hand side's bonds come from the edit's *resolved* kinds, for the reason
    # `rhs_atom_key` reads the edit: the flags the author spelled are not what gets written.
    for (left, right), kind in zip(
        np.asarray(pattern.edit.added_rhs_edges).reshape(-1, 2).tolist(),
        np.asarray(pattern.edit.added_rhs_bond_types).tolist(),
        strict=True,
    ):
        labels[offset + left, offset + right] = EDGE_RHS_BOND + int(kind)
        labels[offset + right, offset + left] = EDGE_RHS_BOND + int(kind)

    for graph, base in ((lhs, 0), (rhs, offset)) if grouping else ():
        group = graph.atom_group
        for left in range(graph.atom_count):
            for right in range(left + 1, graph.atom_count):
                if not (group[left] and group[right]):
                    continue
                relation = (
                    EDGE_SAME_GROUP if group[left] == group[right] else EDGE_OTHER_GROUP
                )
                # A group relation never shares a pair with a bond: nothing bonds across a group
                # boundary, and two atoms of one group are related whether or not they are bonded.
                # Where both would apply the bond is the stronger statement and stays.
                if not labels[base + left, base + right]:
                    labels[base + left, base + right] = relation
                    labels[base + right, base + left] = relation

    # The correspondence, taken from the compiled edit rather than re-read from `atom_map`: the two
    # arrays are positionally aligned, so partners are a zip and the numbers never appear.
    for left, right in zip(
        np.asarray(pattern.edit.mapped_lhs_indices).tolist(),
        np.asarray(pattern.edit.mapped_rhs_indices).tolist(),
        strict=True,
    ):
        labels[left, offset + right] = EDGE_MAP
        labels[offset + right, left] = EDGE_MAP

    neighbors = tuple(
        tuple(int(other) for other in np.flatnonzero(labels[atom]))
        for atom in range(size)
    )
    keys = tuple(
        (
            lhs_atom_key(lhs, atom)
            if atom < offset
            else rhs_atom_key(pattern, atom - offset)
        )
        for atom in range(size)
    )
    return RankingState(
        neighbors=neighbors,
        atom_initial_keys=keys,
        atom_identity_keys=keys,
        bonds=labels,
    )


def pattern_certificate(
    pattern: ReactionPattern, *, grouping: bool = True
) -> PatternCertificate:
    """A canonical form of the whole reaction: equal exactly when the rules are the same.

    Per connected component, because the graph need not be connected -- an ungrouped `A.B` side
    whose halves carry separate map numbers is two components -- and then sorted, so which component
    the author wrote first cannot change the answer. The same shape
    :func:`vesma.canonicalization.canonicalish_atom_order` uses for a molecule's fragments.

    Exact, not hashed. A Weisfeiler-Lehman token would be cheaper and is the right tool at corpus
    scale, but it is a hash-identity rather than a proof: WL cannot separate some non-isomorphic
    graphs, so merging on it needs a verifier anyway. At a few hundred templates the branch-and-bound
    canonicaliser is affordable and there is nothing left to verify.

    ``grouping=False`` forgets what the rule says about its own molecularity, so `(A).(B)`, `(A.B)`
    and a bare `A.B` of one chemistry all land on one certificate. Deliberately *not* the default:
    those are three different rules, and merging them was the first draft's error.
    """
    state = pattern_ranking_state(pattern, grouping=grouping)
    components = connected_components_from_neighbors(state.neighbors)
    certificates = sorted(
        canonical_component(state, component)[0] for component in components
    )
    return (
        PATTERN_CERTIFICATE_SCHEMA,
        pattern.lhs.atom_count,
        pattern.rhs.atom_count,
        tuple(certificates),
    )

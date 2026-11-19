"""When two reaction templates are the same rule written differently.

The certificate has to forget three things and remember everything else, and both halves are
testable. It must forget atom numbering, atom-*map* numbering and zero-level group *ids*, because
all three are bookkeeping. It must remember every predicate a matcher reads -- including the ones
that look like spelling and are not, such as a bond's topology bit.
"""

from __future__ import annotations

import pytest

from vesma import parse_reaction_smarts
from vesma.pattern_identity import (
    EDGE_MAP,
    EDGE_OTHER_GROUP,
    EDGE_SAME_GROUP,
    pattern_certificate,
    pattern_ranking_state,
)

WILLIAMSON = "[CH2:4]-[O:3]-[CX4:2]>>([C:4]-[Cl]).([O:3]-[C:2])"


def certificate(smarts: str):
    return pattern_certificate(parse_reaction_smarts(smarts))


def test_atom_map_numbering_does_not_reach_the_certificate() -> None:
    """The failure this exists for. `iSCB 63a` is `Hartenfeller 29a` renumbered.

    A text check calls those two different rules, and they differ in nothing a matcher could
    observe. Somebody had to notice and rewrite one by hand so the text check would fire; the
    certificate needs nobody's help.
    """
    renumbered = "[CH2:1]-[O:2]-[CX4:3]>>([C:1]-[Cl]).([O:2]-[C:3])"
    sparse = "[CH2:9]-[O:7]-[CX4:5]>>([C:9]-[Cl]).([O:7]-[C:5])"

    assert certificate(WILLIAMSON) == certificate(renumbered)
    assert certificate(WILLIAMSON) == certificate(sparse)
    # And the texts really are different, so the text check would not have found them.
    assert WILLIAMSON != renumbered != sparse


def test_group_numbering_and_order_do_not_reach_it_either() -> None:
    """Group ids are as arbitrary as map numbers, so the *relation* travels and not the number."""
    swapped = "[CH2:4]-[O:3]-[CX4:2]>>([O:3]-[C:2]).([C:4]-[Cl])"

    assert certificate(WILLIAMSON) == certificate(swapped)


def test_a_predicate_spelled_two_ways_is_one_predicate() -> None:
    """Semantic, not syntactic: the certificate carries the compiled masks.

    `[#6]` and `[C,c]` compile to identical masks, so they are one predicate -- and they must be,
    because the matcher cannot tell them apart either. A key that separated them would report two
    rules where a search sees one.
    """
    spelled = "[CH2:4]-[#8;A:3]-[C;X4:2]>>([C:4]-[Cl]).([O:3]-[C:2])"

    assert certificate(WILLIAMSON) == certificate(spelled)


@pytest.mark.parametrize(
    ("other", "why"),
    [
        (
            "[CH2:4]-[O:3]-[CX4:2]>>([C:4]-[Cl].[O:3]-[C:2])",
            "one molecule in two fragments is a different rule from two molecules",
        ),
        (
            "[CH2:4]-[O:3]-[CX4:2]>>[C:4]-[Cl].[O:3]-[C:2]",
            "stating nothing is a different rule from stating two",
        ),
        (
            "[CH2:4]-@[O:3]-[CX4:2]>>([C:4]-[Cl]).([O:3]-[C:2])",
            "a ring-only bond is a different rule",
        ),
        (
            "[CH2:4]-[O:3]-[CX3:2]>>([C:4]-[Cl]).([O:3]-[C:2])",
            "a different connectivity is a different rule",
        ),
        (
            "[CH2:4]-[O:3]-[CX4:2]>>([C:4]-[Br]).([O:3]-[C:2])",
            "a different halide is a different rule",
        ),
    ],
)
def test_what_must_stay_different(other: str, why: str) -> None:
    assert certificate(WILLIAMSON) != certificate(other), why


def test_the_grouping_is_what_the_first_draft_forgot() -> None:
    """Recorded because it was a real error and an easy one.

    Leaving `atom_group` out of the colouring merged every parent with its intramolecular sibling
    -- `Hartenfeller 29a` with `MC 100a`, `Hartenfeller 47` with `MC 102` -- which are deliberately
    different rules whose whole difference is the grouping.
    """
    parent = "[C:1](=O)-[N:2]-[#6:3]>>([C:1](=O)[Cl]).([N:2]-[#6:3])"
    sibling = "[C:1](=O)-[N:2]-[#6:3]>>([C:1](=O)[Cl].[N:2]-[#6:3])"

    assert certificate(parent) != certificate(sibling)


def test_both_sides_live_in_one_graph_joined_by_the_correspondence() -> None:
    """A reaction is two patterns *plus* which atom becomes which, and that is an edge.

    Splitting the sides and keying each would lose exactly the part that says which bond moves: two
    rules can share both sides and differ only in the correspondence between them.
    """
    state = pattern_ranking_state(parse_reaction_smarts(WILLIAMSON))
    labels = {int(value) for value in state.bonds.ravel()} - {0}

    assert EDGE_MAP in labels, "the correspondence is in the graph"
    assert EDGE_OTHER_GROUP in labels, "and so is the grouping relation"
    # One graph, so the state spans both sides at once.
    pattern = parse_reaction_smarts(WILLIAMSON)
    assert len(state.neighbors) == pattern.lhs.atom_count + pattern.rhs.atom_count


def test_a_group_relation_never_overwrites_a_bond() -> None:
    """Two atoms of one group are related whether or not they are bonded, and the bond says more."""
    state = pattern_ranking_state(
        parse_reaction_smarts("[C:1]-[N:2]>>([C:1]-[O].[N:2]-[O])")
    )
    labels = {int(value) for value in state.bonds.ravel()}

    assert EDGE_SAME_GROUP in labels, "the fragments share one group"
    # The bonds inside each fragment survive as bonds rather than being relabelled.
    assert any(0 < label < EDGE_MAP for label in labels)


def test_the_right_hand_side_is_read_from_the_edit_and_not_from_its_masks() -> None:
    """Because most of a right-hand side's query spelling is not obeyed.

    `RHS_ANNOTATIONS = ("A", "D", "X", "v", "R", "r", "x")` are read and then recomputed as
    annotation *mismatches*, never enforced (`reaction.py:175`). Measured: `>>[C:1]-[OH]` and
    `>>[CX4:1]-[OH]` give identical precursors from both an acyclic and a cyclic chloride, so they
    are one rule. A certificate built from `rhs.atom_masks` called them two.
    """
    plain = "[C:1]-[Cl]>>[C:1]-[OH]"

    assert certificate(plain) == certificate("[C:1]-[Cl]>>[CX4:1]-[OH]")
    assert certificate(plain) == certificate("[C:1]-[Cl]>>[C;R1:1]-[OH]")
    # The *left*-hand side is a query, so there the same annotation is part of the rule.
    assert certificate(plain) != certificate("[CX4:1]-[Cl]>>[C:1]-[OH]")


def test_a_right_hand_bond_is_its_resolved_kind_and_not_its_spelling() -> None:
    """An omitted RHS bond and an explicit single bond carry different flags -- 57 against 49 --
    and resolve to the same single bond, because a right-hand side is a rewrite and not a query.
    What the edit will write is the only thing that matters."""
    assert certificate("[C:1]-[Cl]>>[C:1][O:2][C:3]") == certificate(
        "[C:1]-[Cl]>>[C:1]-[O:2]-[C:3]"
    )
    # A kind that really differs is still a different rule.
    assert certificate("[C:1]-[Cl]>>[C:1]-[O:2]-[C:3]") != certificate(
        "[C:1]-[Cl]>>[C:1]=[O:2]"
    )


def test_an_atom_the_rule_makes_is_not_an_atom_it_keeps() -> None:
    """The role is part of the key, so carrying an atom through differs from creating it."""
    carried = "[C:1]-[N:2]>>[C:1]-[Cl].[N:2]"
    created = "[C:1]-[N:2]>>[C:1]-[Cl].[NH3]"

    assert certificate(carried) != certificate(created)

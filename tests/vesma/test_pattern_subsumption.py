"""When one rule is redundant because another already covers it.

Subsumption is the asymmetric relation, so each case is asserted in **both** directions: the whole
value of the relation is that it points at which of a pair is the broader one. Declining is the safe
answer, and the two regression cases at the bottom are rules this file once wrongly merged.
"""

from __future__ import annotations

import pytest

from vesma import parse_reaction_smarts
from vesma.pattern_subsumption import subsumes

# Broad enough to be the covering rule in most cases below: any carbon bearing a chloride.
CHLORIDE = "[C:1]-[Cl]>>[C:1]-[OH]"


def covers(broad: str, narrow: str) -> bool:
    return subsumes(parse_reaction_smarts(broad), parse_reaction_smarts(narrow))


def test_a_narrower_atom_predicate_is_subsumed() -> None:
    """Mask containment is the whole idea: `[CX4]` admits a subset of what `[C]` admits."""
    assert covers(CHLORIDE, "[CX4:1]-[Cl]>>[CX4:1]-[OH]")
    # And not the other way round: the broad rule is not subsumed by the narrow one.
    assert not covers("[CX4:1]-[Cl]>>[CX4:1]-[OH]", CHLORIDE)


def test_a_narrower_bond_is_subsumed() -> None:
    """Bond flags are bitsets too, topology included: `-!@` admits a subset of what `-` admits."""
    acyclic = "[C:1](=[O:2])-!@[O:3]-[#6:4]>>[C:1](=[O:2])-[OH].[OH:3]-[#6:4]"
    either = "[C:1](=[O:2])-[O:3]-[#6:4]>>[C:1](=[O:2])-[OH].[OH:3]-[#6:4]"

    # The ester rule that opens lactones covers the one that does not, and not the reverse.
    assert covers(either, acyclic)
    assert not covers(acyclic, either)


def test_an_extra_atom_carried_through_untouched_is_specificity_and_not_chemistry() -> (
    None
):
    """The subtle condition. `is_inert` cannot mean "asserts nothing".

    The narrow rule re-asserts a methyl its own left-hand side already pinned, which changes
    nothing -- and that restatement is exactly the redundancy worth finding.
    """
    assert covers(CHLORIDE, "[CH3:3]-[C:1]-[Cl]>>[CH3:3]-[C:1]-[OH]")


def test_a_multi_kind_query_bond_is_a_rewrite_and_not_a_carry_through() -> None:
    """`removed_lhs_edges` clears every matched edge, so the right-hand side rebuilds all of them.

    A left-hand mask admitting several kinds therefore *normalises* that bond -- here a double bond
    to single -- which is a thing done to the molecule and not a carry-through.
    """
    assert not covers(CHLORIDE, "[CH3:3]~[C:1]-[Cl]>>[CH3:3]-[C:1]-[OH]")


def test_a_different_product_is_not_subsumed() -> None:
    assert not covers(CHLORIDE, "[CX4:1]-[Cl]>>[CX4:1]-[SH]")


def test_a_rule_subsumes_itself() -> None:
    assert covers(CHLORIDE, CHLORIDE)


def test_which_bond_breaks_is_the_disconnection() -> None:
    """`MC 64a` and `MC 65a`, and the reason `rhs_bonds_agree` exists.

    Both match the same aryl alkyl thioether under the same embedding, both delete nothing, both
    invent one chlorine and both write one `S-H`. Every per-atom condition passes. They are
    opposite disconnections: one makes an aryl halide plus an alkyl thiol, the other a thiophenol
    plus an alkyl halide. Without an equality over the bonds the rules *build*, this reported them
    as one rule -- which would have invited deleting a genuine disconnection.
    """
    aryl_halide = "[c:1]-[S;D2:2]-[C:3]>>[c:1]-[Cl].[SH:2]-[C:3]"
    alkyl_halide = "[c:1]-[S;D2:2]-[CX4:3]>>[c:1]-[SH:2].[CX4:3]-[Cl]"

    assert not covers(aryl_halide, alkyl_halide)
    assert not covers(alkyl_halide, aryl_halide)


def test_a_bond_kind_written_differently_is_a_different_product() -> None:
    """`P 4` and `P 2b`: the same phosphonate disconnection, except that one of them reduces.

    `P 4` writes `[P:2](-[O:3])`, turning a phosphoryl into a single bond; `P 2b` keeps `=[O:3]`.
    `P 4` is broader on its left-hand side -- `[#6:1]` against `[c:1]` -- so every left-hand
    condition passes and only the right-hand bond kinds separate them.
    """
    reduces = (
        "[#6:1]-[P;D4:2](=[O:3])(-[O:4])-[O:5]>>[#6:1]-Br.[P:2](-[O:3])(-[O:4])-[O:5]"
    )
    keeps = "[c:1]-[P;D4:2](=[O:3])(-[O:4])-[O:5]>>[c:1]-Br.[P:2](=[O:3])(-[O:4])-[O:5]"

    assert not covers(reduces, keeps)


def test_an_invented_atom_is_paired_rather_than_matched_by_slot() -> None:
    """Slot order is writing order, so two rules adding the same atoms may list them either way."""
    one = "[C:1]-[N:2]>>[C:1]-[Cl].[N:2]-[Br]"
    other = "[C:1]-[N:2]>>[N:2]-[Br].[C:1]-[Cl]"

    # The same rule, its added atoms written in the other order.
    assert covers(one, other)


@pytest.mark.parametrize(
    "grouped",
    [
        "[C:1]-[Cl]>>([C:1]-[OH])",
        "[CX4:1]-[Cl]>>([CX4:1]-[OH])",
    ],
)
def test_a_rule_stating_its_molecularity_declines(grouped: str) -> None:
    """Zero-level grouping is an equality between two relations, not a constraint that widens.

    It needs an implication test of its own, so until that exists a rule stating its molecularity
    is never reported as redundant -- in either role.
    """
    assert not covers(CHLORIDE, grouped)
    assert not covers(grouped, CHLORIDE)

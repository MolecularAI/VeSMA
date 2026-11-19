"""What a template changes, and whether two templates want the same atom."""

from __future__ import annotations

from dataclasses import replace

import pytest

from vertsy.selectivity import (
    CentreAtom,
    different_site,
    reaction_centre,
    same_site,
)
from vesma import reaction_patterns_from_smarts

AMIDE = "[C:1](=[O:2])-[OH].[N:3]-[H]>>[C:1](=[O:2])-[N:3]"
SULFONYL = "[S:1](=[O:2])(=[O:4])-[Cl].[N:3]-[H]>>[S:1](=[O:2])(=[O:4])-[N:3]"
ESTER = "[C:1](=[O:2])-[OH].[O:3]-[H]>>[C:1](=[O:2])-[O:3]"
ALKYL = "[N:1]-[H].[C:2]-[Cl]>>[N:1]-[C:2]"
SUZUKI = "[c:1]-[Br].[c:2]-[B]([OH])[OH]>>[c:1]-[c:2]"


def centre(smarts: str) -> tuple[CentreAtom, ...]:
    return reaction_centre(reaction_patterns_from_smarts((smarts,))[0])


def roles(smarts: str) -> set[tuple[tuple[int, ...], bool, bool, bool]]:
    return {
        (atom.elements, atom.loses_heavy_bond, atom.gains_bond, atom.loses_hydrogen)
        for atom in centre(smarts)
    }


def test_the_centre_is_what_the_edit_changes_not_what_the_author_numbered() -> None:
    """A map number is bookkeeping; changed bonding is chemistry.

    Templates routinely map spectator atoms to carry them through -- this repository's own
    star-mapped reagents are exactly that -- so "mapped by one rule and not the other" says
    nothing. An amide coupling changes two atoms: the acyl carbon, which loses its hydroxyl and
    gains the nitrogen, and the nitrogen, which loses a hydrogen and gains the carbon. Its carbonyl
    oxygen is mapped throughout and is not in the centre.
    """
    assert roles(AMIDE) == {
        ((6,), True, True, False),  # the acyl carbon
        ((7,), False, True, True),  # the amine nitrogen
    }


def test_a_bond_removed_and_put_back_is_not_a_bond_broken() -> None:
    """The edit rebuilds the matched region rather than editing it in place.

    So `removed_lhs_edges` holds the amide's own carbonyl, which the addition restores. Reading
    the removals alone would name every matched atom a reaction centre -- here, the carbonyl
    oxygen along with the rest.
    """
    oxygens = [atom for atom in centre(AMIDE) if atom.elements == (8,)]

    assert oxygens == [], "the carbonyl oxygen is carried through, not changed"


def test_two_reactions_wanting_the_same_group_compete_at_the_same_site() -> None:
    """The split the channel needs, and the reason it is the useful one.

    A competitor attacking a *different* site is answered by protection -- a bounded, buyable,
    three-step detour the selector can purchase. One attacking the **same** atom cannot be
    protected against, because protecting the site stops both reactions, so its yield split is set
    by a rate ratio and nothing in the step repertoire removes it.
    """
    patterns = reaction_patterns_from_smarts((AMIDE, SULFONYL, ESTER, SUZUKI))
    # Each toy retro rule's centre is its own forward rule's centre, at the same position. Stated
    # rather than assumed: in a real run a retro id is a position in whichever rule sets were
    # compiled and has nothing to do with a forward row's position, which is the whole reason the
    # intended reaction's centre is looked up rather than indexed.
    competes = same_site(
        patterns, {rule: reaction_centre(patterns[rule]) for rule in range(4)}
    )
    amide, sulfonyl, ester, suzuki = range(4)

    # Both acylate the amine nitrogen, so they want the same material.
    assert competes(amide, sulfonyl)
    # Both consume the acid's carbonyl carbon.
    assert competes(amide, ester)
    # An acyl carbon is not an aryl carbon, and neither touches the other's partner.
    assert not competes(amide, suzuki)
    assert not competes(sulfonyl, suzuki)


def test_aromaticity_is_what_keeps_an_acyl_carbon_from_an_aryl_one() -> None:
    """Element and role alone lump every electrophilic carbon together.

    Without the aromatic mask an amide coupling reads as competing with a Suzuki: both have a
    carbon that breaks a heavy bond and makes one. They are not the same atom and not the same
    chemistry.
    """
    carbons = {
        name: next(
            a for a in centre(smarts) if a.elements == (6,) and a.loses_heavy_bond
        )
        for name, smarts in (("amide", AMIDE), ("suzuki", SUZUKI))
    }

    assert carbons["amide"].aromaticity == 1
    assert carbons["suzuki"].aromaticity == 2
    assert not carbons["amide"].aromaticity & carbons["suzuki"].aromaticity


def test_the_rule_level_tier_cannot_tell_an_acyl_carbon_from_an_alkyl_one() -> None:
    """The approximation this tier makes, pinned so that it is a known limit and not a surprise.

    Two templates are compared, never two matches, so nothing here knows which *atom* of this
    molecule each would land on. An esterification's acyl carbon and an alkylation's alkyl carbon
    are both aliphatic carbons that break a heavy bond and make one, so they read as the same
    site. The match-level tier -- the competitor's matched atoms overlapping the intended
    reaction's -- is what removes this, and it needs the matcher to return assignments.
    """
    patterns = reaction_patterns_from_smarts((ESTER, ALKYL))
    competes = same_site(patterns, {0: reaction_centre(patterns[0])})

    assert competes(0, 1), "a known false positive, not a claim about chemistry"


def test_a_rule_the_split_cannot_judge_is_in_neither_half() -> None:
    """The replacement for an assertion that locked in a bug, and the argument that was wrong.

    What stood here said "a predicate that cannot judge must not silently clear: the default is the
    crude reading", and returned ``True`` -- same-site -- for anything out of range. That reasoning
    is what made the split unmeasurable: ``intended`` is a *retro* rule id indexed into a list of
    *forward* patterns, so with 204 patterns against 459 compiled rules, 255 rules fell through the
    guard and were labelled same-site unconditionally. The default was not crude, it was one half
    of a two-way split, and it swallowed the majority of the measurement.

    A predicate that cannot judge abstains from **both** halves. Nothing is waved through by that,
    because the crude reading is a channel that already exists: the unsplit `competitors` measures
    every one of these applications. The split is a diagnostic refining that channel, and a
    diagnostic with no evidence should say nothing rather than pick a side.
    """
    patterns = reaction_patterns_from_smarts((AMIDE,))
    centres = {0: reaction_centre(patterns[0])}
    same = same_site(patterns, centres)
    different = different_site(patterns, centres)

    # A candidate outside the forward table: no centre to compare, so neither half claims it.
    assert not same(0, 99) and not different(0, 99)
    # An intended reaction no forward row describes: likewise, and this is the majority case.
    assert not same(1, 0) and not different(1, 0)
    assert not same(-1, 0) and not different(-1, 0)


@pytest.mark.parametrize("smarts", [AMIDE, SULFONYL, ESTER, ALKYL, SUZUKI])
def test_every_centre_atom_is_changed_by_its_template(smarts: str) -> None:
    """A centre atom that neither gains nor loses anything is a bookkeeping error."""
    for atom in centre(smarts):
        assert atom.loses_heavy_bond or atom.gains_bond or atom.loses_hydrogen
        assert atom.elements and 1 not in atom.elements


def test_the_two_halves_partition_the_candidates() -> None:
    """One probe pass each, and between them every candidate is asked exactly once.

    That is what makes splitting affordable: the halves are complementary, so running both costs
    what running the unsplit channel costs, and nothing is counted twice under two prices.
    """
    from tests.vertsy.bench_host_aostar import ForwardLibrary, _competes_for

    forward = ForwardLibrary(
        smarts=(AMIDE, SULFONYL, ESTER, SUZUKI),
        codes=("a", "s", "e", "z"),
        arities=(2, 2, 2, 2),
        declared=(2, 2, 2, 2),
        # No conditions declared, so the conditions predicate admits everything and the two halves
        # are still each other's complement -- which is what this test is about.
        requires=(frozenset(),) * 4,
        # All in the default band: the severity channel is not what is partitioned here.
        severity=("",) * 4,
        efficiency=(0.0,) * 4,
        conditions={},
        index={},
        describes={rule: (rule,) for rule in range(4)},
        intended_centres={
            rule: reaction_centre(pattern)
            for rule, pattern in enumerate(
                reaction_patterns_from_smarts((AMIDE, SULFONYL, ESTER, SUZUKI))
            )
        },
        yields={},
    )
    same = _competes_for("competitors:same-site", forward)
    different = _competes_for("competitors:different-site", forward)

    assert same is not None and different is not None
    for intended in range(4):
        for candidate in range(4):
            assert same(intended, candidate) != different(intended, candidate)
    # The crude reading keeps only the conditions predicate, which is a filter and not a split.
    assert _competes_for("competitors", forward) is not None
    with pytest.raises(ValueError, match="unknown competitor channel"):
        _competes_for("competitors:badly-named", forward)


def test_the_halves_partition_the_candidates_the_conditions_admit() -> None:
    """And not every candidate, which is what adding a conditions filter changed.

    A candidate the intended reaction's medium cannot turn over is in **neither** half: it is not a
    competition at all, so there is no site question to ask about it. The two halves still partition
    what is left, which is the property that makes running both cost what running one costs.
    """
    from tests.vertsy.bench_host_aostar import ForwardLibrary, _competes_for

    forward = ForwardLibrary(
        smarts=(AMIDE, SULFONYL, ESTER, SUZUKI),
        codes=("a", "s", "e", "z"),
        arities=(2, 2, 2, 2),
        declared=(2, 2, 2, 2),
        # The Suzuki needs palladium; nothing supplies it, so it is excluded everywhere.
        requires=(frozenset(), frozenset(), frozenset(), frozenset({"palladium"})),
        severity=("",) * 4,
        efficiency=(0.0,) * 4,
        conditions={0: frozenset(), 1: frozenset(), 2: frozenset(), 3: frozenset()},
        index={},
        describes={rule: (rule,) for rule in range(4)},
        intended_centres={
            rule: reaction_centre(pattern)
            for rule, pattern in enumerate(
                reaction_patterns_from_smarts((AMIDE, SULFONYL, ESTER, SUZUKI))
            )
        },
        yields={},
    )
    same = _competes_for("competitors:same-site", forward)
    different = _competes_for("competitors:different-site", forward)

    assert same is not None and different is not None
    for intended in range(4):
        assert not same(intended, 3) and not different(intended, 3), "no palladium here"
        for candidate in range(3):
            assert same(intended, candidate) != different(intended, candidate)

    # And it is admitted where the medium does supply it.
    supplied = replace(forward, conditions={0: frozenset({"palladium"})})
    assert _competes_for("competitors", supplied)(0, 3)


def test_a_right_hand_index_is_translated_and_not_used_as_a_position() -> None:
    """The bug this fixes, on the rule that showed it.

    `mapped_lhs_indices` and `mapped_rhs_indices` are positionally aligned and ordered by *map
    number*, so pairing them is the translation and indexing either one by an atom index is not.
    Reading `mapped_lhs_indices[rhs_atom]` happens to work where the mapped right-hand atoms sit at
    right-hand positions 0..k-1 in that order -- 142 of the 480 shipped rules -- and silently
    mistranslates the other 338.

    `Hartenfeller 33a` breaks an aryl-methyl bond and halogenates both ends. Its new c-Cl bond sits
    at right-hand indices (0, 1), which read positionally give left-hand (1, 0): the aryl-methyl
    bond itself. That phantom *addition* cancelled the genuine removal, and the rule came back with
    **no reaction centre at all**.
    """
    found = centre("[c:2]-[CH3:1]>>[c:2]-[Cl].[C:1]-[Br]")

    assert len(found) == 2, "both ends of the broken bond are in the centre"
    assert all(atom.loses_heavy_bond for atom in found)
    # One aromatic carbon and one aliphatic carbon, which is the whole point of carrying `A`.
    assert sorted(atom.aromaticity for atom in found) == [1, 2]


def test_the_centre_is_blind_to_a_bond_that_only_changes_order() -> None:
    """A known shape of the abstraction, recorded so it is not mistaken for the bug above.

    `broken` and `made` are differences of edge *presence*, so a bond that stays put and changes
    kind appears in both and cancels -- and `CentreAtom` has no slot for "order changed" beside
    `loses_heavy_bond` / `gains_bond` / `loses_hydrogen`. Seven shipped rules are affected, all
    reductions or oxidations: `MC 5`, `MC 6`, `MC 38`, `MC 39`, `MC 79`, `MC 80`, `MC 87`.

    It matters because `same_site` reads the centre, and an empty centre is never same-site, so
    those rules fall wholly into the different-site half. Three of them are bimolecular and so can
    be competitors at all; the rest are arity 1 and never enter the channel.
    """
    # A ketone reduction: the C-O bond goes from double to single and nothing is broken or made.
    assert (
        centre("[CH1:1](-[#6:3])(-[#6:4])-[OH:2]>>[C:1](-[#6:3])(-[#6:4])=[O:2]") == ()
    )
    # Where a bond really is broken, the centre is found.
    assert centre("[C:1]-[O:2]-[C:3]>>[C:1]-[O:2].[C:3]-[Cl]")


def test_severity_is_a_worst_case_and_not_a_count() -> None:
    """The property the channel exists for: multiplicity cannot move the number.

    A pot can make several weak side products and still deliver the one you wanted; one reaction
    that consumes the substrate delivers nothing. So three mild competitors and one mild competitor
    are the same measurement, and a single destructive one is a different measurement from either.
    A count cannot say that, which is why this reduction exists beside `distinct_outcomes`.
    """
    import numpy as np

    from vertsy.selectivity import severity_outcomes

    # Forward rules 0 and 1 are ordinary -- nothing to take -- and rule 2 consumes its substrate.
    reduce = severity_outcomes((0.0, 0.0, 0.90), lambda rule: 0.80, (0.56, 0.49, 0.41))
    keys = ["p0", "p1", "p2"]
    rules = [7]

    one_mild = np.array([[0, 0, 0, 0]], dtype=np.int64)
    three_mild = np.array([[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 2, 2]], dtype=np.int64)
    one_destructive = np.array([[0, 2, 0, 0]], dtype=np.int64)

    assert reduce(keys, one_mild, 1, None, rules) == [1], "mild competition is free"
    assert reduce(keys, three_mild, 1, None, rules) == [1], "three are still free"
    # And one destruction is not free.
    assert reduce(keys, one_destructive, 1, None, rules) != [1]
    # Nothing fired at all reads the same as mild, which is what "band 1 is free" means.
    assert reduce(keys, np.empty((0, 4), dtype=np.int64), 1, None, rules) == [1]


def test_the_severity_band_is_relative_to_the_intended_reaction() -> None:
    """The chemistry the absolute form could not express, and the reason the band is computed here.

    Two reactions competing for one substrate split it by their rates, so what survives for the
    intended reaction is `y / (y + e)` and not `1 - e`. The second is a *mass balance* -- it assumes
    the degradation runs first and takes its share -- and these are parallel competitors.

    The consequence is the test: the same degradation costs a **robust** step less than a **hard**
    one, because a fast intended reaction wins more of the substrate. A `Factor` cannot express that
    at all, since it sees one integer and cannot ask what the step was for; computing the band where
    both reactions are known is the way across.
    """
    import numpy as np

    from vertsy.selectivity import severity_outcomes

    EDGES = (0.56, 0.49, 0.41)
    fired = np.array([[0, 0, 0, 0]], dtype=np.int64)
    keys, rules = ["p"], [3]

    def band(intended: float, efficiency: float) -> int:
        reduce = severity_outcomes((efficiency,), lambda rule: intended, EDGES)
        return reduce(keys, fired, 1, None, rules)[0]

    # One degradation, three strengths of intended reaction: the stronger it is, the milder the band.
    robust = band(0.90, 0.90)
    reliable = band(0.80, 0.90)
    hard = band(0.50, 0.90)
    assert robust < reliable < hard

    # And one intended reaction against degradations of differing completeness.
    assert band(0.80, 0.50) < band(0.80, 0.90)

    # Every band is above clean, because something destructive did fire in all of them.
    assert min(robust, reliable, hard) >= 2


def test_a_severity_reading_charges_where_a_count_reading_cannot() -> None:
    """Why the scale starts at one, stated as the arithmetic rather than as a preference.

    `vertsy.route_selection.clean_log_cost` takes `charge(rule, (1,) * channels)` as the price's
    free reference, so whatever a channel stores at `1` costs nothing by construction. On a count
    channel `1` is *one competitor* -- the very event a severity price has to charge -- and on this
    one `1` is "nothing to worry about". Shifting the measurement is what makes the band priceable
    without bending a curve or changing `drop_intended`.
    """
    import numpy as np

    from vertsy.costs import banded, cost_of_factor
    from vertsy.selectivity import severity_outcomes

    reduce = severity_outcomes((0.90,), lambda rule: 0.80, (0.56, 0.49, 0.41))
    keys, rules = ["p"], [3]
    clean = reduce(keys, np.empty((0, 4), dtype=np.int64), 1, None, rules)[0]
    destroyed = reduce(keys, np.array([[0, 0, 0, 0]], dtype=np.int64), 1, None, rules)[
        0
    ]

    curve = banded((1.0, 0.60, 0.52, 0.44, 0.37))
    # The clean reading is the multiplicative identity, so it adds nothing to any step's cost.
    assert clean == 1
    assert curve(clean) == 1.0
    assert cost_of_factor(curve(clean)) == 0.0
    # And the destroyed reading does cost, which is the whole point.
    assert destroyed > clean
    assert curve(destroyed) < 1.0
    assert cost_of_factor(curve(destroyed)) > 0.0

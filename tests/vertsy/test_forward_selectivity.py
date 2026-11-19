"""The forward probe, on real chemistry, with the case that motivated it.

Everything here goes through `vesma`'s ordinary CPU reaction matcher. A forward template is not a
new kind of object -- the machinery matches a pattern's left-hand side and applies its edit, so what
a retro pass calls precursors a forward pass calls products, and no new chemistry code exists.
"""

from __future__ import annotations

import numpy as np
import pytest

from vertsy.costs import UNPROBED
from vertsy.search import object_mol_array
from vertsy.selectivity import (
    ProbeReport,
    competitor_probe,
    conditions_allow,
    declared_molecularity,
    distinct_outcomes,
    object_combine,
    site_probe,
    template_arities,
)
from vesma import cmol_from_smiles, parse_reaction_smarts
from vesma.cpu_matcher import CpuReactionMatcher
from vesma.keyer import cmol_smiles_keyer


# The *sites* channel asks what the forward chemistry does to an application's precursors, which is
# a question about one side, so it never reads the frontier the application came from. Passing
# something that raises on contact rather than passing the precursors twice: if it ever starts
# reading the other side, these tests say so instead of quietly measuring the wrong batch.
#
# The *competitor* channel does read it, and only for one thing: an outcome equal to the molecule
# the application was disconnecting is not a competitor, because a product that *is* the intended
# product cannot compete with itself. That makes the question a comparison between the two sides
# rather than a count on one, which is why those tests pass a real frontier below.
class _Unread:
    def __len__(self) -> int:
        raise AssertionError(
            "a selectivity channel read the frontier; it should not need it."
        )

    def __getitem__(self, selector: object) -> object:
        raise AssertionError(
            "a selectivity channel read the frontier; it should not need it."
        )


UNREAD_PRODUCTS = _Unread()


# Xylitol, and the disconnection a chemist objects to. Five hydroxyls, and the retro rule proposes
# chlorinating exactly one of them.
XYLITOL = "OCC(O)C(O)C(O)CO"
# One primary alcohol and nothing else a chlorinating agent can touch.
BUTANOL = "CCCCO"

# The forward counterparts of `FGI 1a` and `FGI 4a` in `templates-medchem.csv`, written flat. Both
# are what a chlorinating agent does: the reagent does not know which hydroxyl was intended, which
# is why one retro rule is probed by *both* of them.
PRIMARY_CHLORINATION = "[CH2:1](-[CX4:2])-[OH]>>[CH2:1](-[CX4:2])-[Cl]"
SECONDARY_CHLORINATION = (
    "[CH1:1](-[CX4:2])(-[CX4:3])-[OH]>>[CH1:1](-[CX4:2])(-[CX4:3])-[Cl]"
)

# The retro rule id the graph would carry for `FGI 1a`. Arbitrary, and the point is that it is the
# *retro* id the index is keyed by.
FGI_1A = 137


def molecule(smiles: str):
    built, _info = cmol_from_smiles(smiles)
    return built


def forward_matcher():
    return CpuReactionMatcher(
        (
            parse_reaction_smarts(PRIMARY_CHLORINATION),
            parse_reaction_smarts(SECONDARY_CHLORINATION),
        )
    )


def keys(molecules):
    """Name a whole returned batch, which is the axis the probe hands over."""
    return [cmol_smiles_keyer.encode(each) for each in molecules]


def probe_of(index, **kwargs):
    reports: list[ProbeReport] = []
    measure = site_probe(
        forward_matcher(), index, key=keys, report=reports.append, **kwargs
    )
    return measure, reports


def one_application(smiles: str, rule: int = FGI_1A):
    """A single unimolecular disconnection: one product, one template, one precursor.

    The precursors arrive as a *batch*, because that is what the probe is handed -- it selects rows
    with an index array rather than holding molecules, which is what lets the same probe run over a
    packed device batch.
    """
    return object_mol_array([molecule(smiles)]), np.array(
        [(0, rule, 0, 0)], dtype=np.int64
    )


# --------------------------------------------------------------- the motivating measurement


def test_a_pentitol_reads_three_products_and_not_five_sites() -> None:
    """The whole reason the count is over *products* keyed by identity, not over match sites.

    Xylitol has five hydroxyls, and a chlorinating agent matches all five. It makes **three**
    distinct monochlorides, because the molecule is meso: its two primary positions are equivalent
    and give one product, and of its three secondary positions two are equivalent and give a second,
    leaving the central one to give a third. Counting sites would say five and would be counting the
    molecule's symmetry as if it were a selectivity problem.

    Three is also the number that matters: under `steps_equivalent` a three-product step is worth
    four steps, which is the first value that clears a three-step protection detour.
    """
    measure, reports = probe_of({FGI_1A: (0, 1)})
    precursors, hits = one_application(XYLITOL)

    assert np.asarray(measure(UNREAD_PRODUCTS, precursors, hits)).tolist() == [3]
    assert reports[-1].probed == 1 and reports[-1].coverage == 1.0


def test_a_substrate_with_one_reactive_site_reads_one() -> None:
    """The clean case, and the one that must cost nothing.

    1-butanol has a single hydroxyl and no secondary one, so the pair of forward rules between them
    make exactly one product. A measurement of ``1`` is what "this step is selective" looks like,
    and `steps_equivalent` prices it at exactly one step.
    """
    measure, _ = probe_of({FGI_1A: (0, 1)})
    precursors, hits = one_application(BUTANOL)

    assert np.asarray(measure(UNREAD_PRODUCTS, precursors, hits)).tolist() == [1]


def test_only_the_forward_rules_paired_with_a_retro_rule_are_run() -> None:
    """Which is what makes the probe affordable, and also what makes it mean something.

    Probing xylitol with only the *primary* chlorination reads 1: that rule alone makes one product,
    because the two primary hydroxyls are equivalent. The three-way answer needs the secondary rule
    too, because a chlorinating agent does not know which hydroxyl was intended. Pairing is
    therefore a statement about a reagent's scope, not a lookup of one template's mirror image.
    """
    primary_only, _ = probe_of({FGI_1A: (0,)})
    both, _ = probe_of({FGI_1A: (0, 1)})
    precursors, hits = one_application(XYLITOL)

    assert np.asarray(primary_only(UNREAD_PRODUCTS, precursors, hits)).tolist() == [1]
    assert np.asarray(both(UNREAD_PRODUCTS, precursors, hits)).tolist() == [3]


# --------------------------------------------------------------- unprobed is not clean


def test_a_rule_with_no_forward_counterpart_is_unprobed_and_charged_nothing() -> None:
    """The distinction this module most needs to keep.

    A library nobody has written forward rules for must cost exactly what it costs today. Reading an
    unpaired rule as ``1`` would declare it perfectly selective, which is the one failure mode worth
    designing against -- raw68k is tens of thousands of rules with no forward counterparts, and a
    run over it must not come out looking clean.

    It reads `UNPROBED` and not ``0``, because ``0`` is a *result*: the forward chemistry ran and
    made nothing, which says the pairing is broken and is charged as such. A probe that could not
    tell those apart is what let a Stille whose forward rule wanted bromide beat a Suzuki that was
    honestly measured as a two-product mixture.
    """
    measure, reports = probe_of({FGI_1A: (0, 1)})
    precursors, hits = one_application(XYLITOL, rule=999)

    assert np.asarray(measure(UNREAD_PRODUCTS, precursors, hits)).tolist() == [UNPROBED]
    assert reports[-1].unpaired == 1 and reports[-1].probed == 0
    assert reports[-1].coverage == 0.0


def test_a_multi_precursor_application_is_unprobed_when_nothing_can_combine_it() -> (
    None
):
    """A bimolecular forward pattern has to match across both starting materials at once.

    Without a way to build that one molecule there is nothing to match against, so the application
    is unprobed rather than half-probed -- reading one precursor alone would measure a different
    reaction from the one the step performs. Unprobed and not silent: nothing was examined, so there
    is no verdict to charge for.
    """
    measure, reports = probe_of({FGI_1A: (0, 1)})
    precursors = object_mol_array([molecule(XYLITOL), molecule(BUTANOL)])
    hits = np.array([(0, FGI_1A, 0, 0), (0, FGI_1A, 0, 1)], dtype=np.int64)

    assert np.asarray(measure(UNREAD_PRODUCTS, precursors, hits)).tolist() == [
        UNPROBED,
        UNPROBED,
    ]
    assert reports[-1].unbuildable == 1 and reports[-1].applications == 1


def test_a_forward_rule_that_fires_on_nothing_is_reported_rather_than_read_as_clean() -> (
    None
):
    """A curation defect, not a chemistry finding, so it is counted and charged nothing.

    A forward rule that cannot reproduce its own partner's step from its own partner's precursors is
    wrong. The round-trip test over the forward table is where that gets caught; here it only has to
    not be mistaken for selectivity.
    """
    measure, reports = probe_of({FGI_1A: (0, 1)})
    # Acetone has no hydroxyl at all, so neither chlorination rule matches.
    precursors, hits = one_application("CC(C)=O")

    assert np.asarray(measure(UNREAD_PRODUCTS, precursors, hits)).tolist() == [0]
    # Examined, and found nothing. On this channel that reads as a defect in the rule.
    assert reports[-1].probed == 1 and reports[-1].silent == 1


# --------------------------------------------------------------- the per-hit contract


def test_every_row_of_one_application_carries_the_same_measurement() -> None:
    """The invariant `layer_tables` checks, produced here rather than assumed.

    A measurement is a property of ``(rule, precursors)``, so every hit row of one application is
    one reaction and they cannot disagree. The column is per hit row because that is the axis a
    matcher returns; it is redundant by the application's arity, and this is that redundancy.
    """
    measure, _ = probe_of(
        {FGI_1A: (0, 1)}, combine=object_combine(lambda parts: parts[0])
    )
    precursors = object_mol_array([molecule(XYLITOL), molecule(XYLITOL)])
    hits = np.array([(0, FGI_1A, 0, 0), (0, FGI_1A, 0, 1)], dtype=np.int64)

    assert np.asarray(measure(UNREAD_PRODUCTS, precursors, hits)).tolist() == [3, 3]


def test_two_applications_of_one_product_are_measured_separately() -> None:
    """Each disconnection gets its own answer, keyed by the rule that made it."""
    measure, reports = probe_of({FGI_1A: (0,), 555: (1,)})
    precursors = object_mol_array([molecule(XYLITOL)])
    hits = np.array([(0, FGI_1A, 0, 0), (0, 555, 1, 0)], dtype=np.int64)

    # Primary alone makes one product; secondary alone makes two.
    assert np.asarray(measure(UNREAD_PRODUCTS, precursors, hits)).tolist() == [1, 2]
    assert reports[-1].applications == 2 and reports[-1].probed == 2


def test_an_empty_hit_array_measures_nothing() -> None:
    measure, reports = probe_of({FGI_1A: (0, 1)})
    assert (
        np.asarray(
            measure(
                UNREAD_PRODUCTS,
                object_mol_array([]),
                np.zeros((0, 4), dtype=np.int64),
            )
        ).tolist()
        == []
    )
    assert reports[-1] == ProbeReport("sites", 0, 0, 0, 0, 0)


@pytest.mark.parametrize(
    "components,expected",
    [
        ([("a",), ("a",)], 1),  # one outcome reached twice
        ([("a",), ("b",)], 2),
        ([("a", "water"), ("water", "a")], 1),  # order is not identity
    ],
)
def test_an_outcome_is_the_whole_component_multiset(components, expected) -> None:
    """A forward reaction may emit several molecules; an acetalisation releases water.

    So an outcome is the multiset, sorted, and two spellings of one outcome are one outcome --
    otherwise every condensation would read as a mixture of its product and its by-product.
    """
    flat = [part for outcome in components for part in outcome]
    hits = np.array(
        [
            (0, 0, app, offset + position)
            for app, (outcome, offset) in enumerate(
                zip(components, np.cumsum([0, *(len(o) for o in components[:-1])]))
            )
            for position in range(len(outcome))
        ],
        dtype=np.int64,
    )
    assert distinct_outcomes(flat, hits, 1) == [expected]


# --------------------------------------------------------------- bimolecular, and what it found

BENZALDEHYDE = "O=Cc1ccccc1"
# The supplemental template from the hard-22 campaign, written forward: a 1,3-diol and an aldehyde
# close to a six-membered benzylidene acetal. Its left-hand side has **two components**, which is
# the whole reason a union exists.
BENZYLIDENE = (
    "[OH:1]-[C:5]-[C:6]-[C:7]-[OH:4].[CH:2](=[O])-[c:3]"
    ">>[O:1]1-[C:5]-[C:6]-[C:7]-[O:4]-[CH:2]1-[c:3]"
)
ACETALISE = 4100


def acetal_probe(combine):
    matcher = CpuReactionMatcher((parse_reaction_smarts(BENZYLIDENE),))
    reports: list[ProbeReport] = []
    return (
        site_probe(
            matcher,
            {ACETALISE: (0,)},
            key=keys,
            combine=combine,
            report=reports.append,
        ),
        reports,
    )


def two_precursor_application(left: str, right: str, rule: int = ACETALISE):
    return (
        object_mol_array([molecule(left), molecule(right)]),
        np.array([(0, rule, 0, 0), (0, rule, 0, 1)], dtype=np.int64),
    )


def test_a_two_component_pattern_matches_the_union_and_neither_part_alone() -> None:
    """The union is what makes a bimolecular forward rule matchable at all.

    A subgraph match is a match into one graph, so two starting materials have to be one graph
    first. Retro application never needed that -- it starts from a single product and produces
    components -- which is why `cmol_union` did not exist until now.
    """
    from vesma.cmol import cmol_union

    measure, _ = acetal_probe(object_combine(cmol_union))
    precursors, hits = two_precursor_application("OCCCO", BENZALDEHYDE)

    assert np.asarray(measure(UNREAD_PRODUCTS, precursors, hits)).tolist() == [1, 1]

    # Neither half alone is the reaction: a diol with no aldehyde makes nothing.
    alone, reports = acetal_probe(object_combine(cmol_union))
    assert np.asarray(
        alone(
            UNREAD_PRODUCTS,
            [molecule("OCCCO")],
            np.array([(0, ACETALISE, 0, 0)]),
        )
    ).tolist() == [0]
    assert reports[-1].silent == 1


def test_symmetry_collapses_a_bimolecular_count_too() -> None:
    """Two reactive partners, one product, because the partners' sites are equivalent."""
    from vesma.cmol import cmol_union

    amide = CpuReactionMatcher(
        (
            parse_reaction_smarts(
                "[C:1](=[O:2])-[OH].[N;H2:3]-[C:4]>>[C:1](=[O:2])-[N;H1:3]-[C:4]"
            ),
        )
    )
    measure = site_probe(amide, {0: (0,)}, key=keys, combine=object_combine(cmol_union))

    symmetric, hits = two_precursor_application("CC(=O)O", "NCCN", rule=0)
    asymmetric, _ = two_precursor_application("CC(=O)O", "NCC(C)N", rule=0)

    # 1,2-diaminoethane's two amines are equivalent; 1,2-diaminopropane's are not.
    assert np.asarray(measure(UNREAD_PRODUCTS, symmetric, hits)).tolist() == [1, 1]
    assert np.asarray(measure(UNREAD_PRODUCTS, asymmetric, hits)).tolist() == [2, 2]


def test_the_measured_xylitol_routes_are_both_mixtures_and_the_detour_is_the_worse_one() -> (
    None
):
    """What the probe actually says about the route that motivated it. It is not what was expected.

    The reviewer's objection was that ``xylitol -> chlorinate -> acetalise`` should have acetalised
    first, to block two of the five hydroxyls. Measured, **every step of both routes is a mixture**:

    ==================================  ========  ==============
    step                                products  worth (n + 1)
    ==================================  ========  ==============
    chlorinate free xylitol             3         4 steps
    ----------------------------------  --------  --------------
    acetalise xylitol                   2         3 steps
    chlorinate the acetal               2 or 3    3 or 4 steps
    deprotect                           1         1 step
    ==================================  ========  ==============

    So the one-step route is worth 4 and the three-step detour is worth 7 or 8, and the penalty makes
    the detour *less* attractive rather than more.

    That is a correct measurement and an honest limitation, not a bug. A 1,3-benzylidene acetal on a
    pentitol is not regioselective either, and it blocks two of five hydroxyls, leaving three for the
    chlorination that follows. The literature route works because one acetal is thermodynamically
    favoured and crystallises -- information about *which* product dominates, which a count of
    possible products cannot contain by construction. This is precisely the gap
    `docs/paper/paper/benchmarks.qmd` names: enumerating alternatives establishes that competitors
    exist, not that the intended one loses.

    What the probe does deliver on this target is the honest verdict that **neither route is clean**,
    which is more useful than a confident wrong ordering. Making the detour win needs a ratio or a
    curated note that a protection is thermodynamically controlled, and neither is a product count.
    """
    from vesma.cmol import cmol_union

    chlorinate, _ = probe_of({FGI_1A: (0, 1)}, combine=object_combine(cmol_union))
    acetalise, _ = acetal_probe(object_combine(cmol_union))

    free, free_hits = one_application(XYLITOL)
    assert np.asarray(chlorinate(UNREAD_PRODUCTS, free, free_hits)).tolist() == [3]

    protect, protect_hits = two_precursor_application(XYLITOL, BENZALDEHYDE)
    assert np.asarray(acetalise(UNREAD_PRODUCTS, protect, protect_hits)).tolist() == [
        2,
        2,
    ]


# --------------------------------------------------------------- the competitor channel

# A forward amide coupling and a forward esterification. Both need **two** starting materials, which
# is what makes either a candidate competitor at all.
FORWARD_AMIDE = "[C:1](=[O:2])-[OH].[N;H2:3]-[C:4]>>[C:1](=[O:2])-[N;H1:3]-[C:4]"
FORWARD_ESTER = "[C:1](=[O:2])-[OH].[O;H1:3]-[C:4]>>[C:1](=[O:2])-[O:3]-[C:4]"
# A unimolecular one, for the case the channel must *not* count.
FORWARD_OXIDATION = "[CH1:1](-[CX4:2])(-[CX4:3])-[OH]>>[C:1](-[CX4:2])(-[CX4:3])=O"

COUPLE = 7000


def competitor_measure(smarts, *, index=None, competes=None, admit_unimolecular=False):
    from vesma.cmol import cmol_union

    reports: list[ProbeReport] = []
    matcher = CpuReactionMatcher(tuple(parse_reaction_smarts(t) for t in smarts))
    probe = competitor_probe(
        matcher,
        template_arities(smarts),
        key=keys,
        combine=object_combine(cmol_union),
        index=index,
        competes=competes,
        admit_unimolecular=admit_unimolecular,
        report=reports.append,
    )
    return probe, reports


def two_partners(left: str, right: str, rule: int = COUPLE):
    return (
        object_mol_array([molecule(left), molecule(right)]),
        np.array([(0, rule, 0, 0), (0, rule, 0, 1)], dtype=np.int64),
    )


def frontier(smiles: str):
    """A one-molecule frontier, so `product_idx` 0 names what the step was disconnecting."""
    return object_mol_array([molecule(smiles)])


# Neither of the two products the amide/ester pot can give, so both still count. A step whose own
# product is one of them is the separate case, tested on its own below.
OTHER_TARGET = frontier("c1ccccc1")


def test_a_competitor_is_a_reaction_the_same_partners_could_do_instead() -> None:
    """An amino alcohol and an acid can couple through either the amine or the hydroxyl.

    Both partners are in the pot by construction -- the probe matches the application's own
    precursors -- so the esterification is a genuine competitor to the amide coupling and not a
    hypothetical one.
    """
    measure, reports = competitor_measure([FORWARD_AMIDE, FORWARD_ESTER])
    precursors, hits = two_partners("CC(=O)O", "NCCO")

    # Amide and ester are different products, so two ways this pot can react.
    assert np.asarray(measure(OTHER_TARGET, precursors, hits)).tolist() == [2, 2]
    assert reports[-1].channel == "competitors"
    assert reports[-1].probed == 1


def test_the_intended_reactions_own_rules_are_not_counted_as_competing_with_it() -> (
    None
):
    """They belong to the sites channel, and counting them twice would price one fact twice."""
    paired = {COUPLE: (0,)}  # the amide coupling *is* the intended reaction here
    measure, _ = competitor_measure([FORWARD_AMIDE, FORWARD_ESTER], index=paired)
    precursors, hits = two_partners("CC(=O)O", "NCCO")

    # Only the esterification is left, so one competing reaction rather than two ways to react.
    assert np.asarray(measure(OTHER_TARGET, precursors, hits)).tolist() == [1, 1]


def test_a_unimolecular_rule_is_never_a_competitor() -> None:
    """The arity filter, and the whole physical argument for this channel.

    The oxidation matches the amino alcohol's secondary hydroxyl, so a probe that ran every forward
    rule would report it. It is not a competition: an oxidant is not in the pot, and it would need
    its own catalyst and conditions. Counting it would make every polyfunctional substrate read as
    a mixture and measure molecular complexity rather than a competing reaction.
    """
    substrate, hits = two_partners("CC(=O)O", "NCC(C)O")
    with_unimolecular, _ = competitor_measure(
        [FORWARD_AMIDE, FORWARD_ESTER, FORWARD_OXIDATION]
    )
    without, _ = competitor_measure([FORWARD_AMIDE, FORWARD_ESTER])

    assert np.asarray(with_unimolecular(OTHER_TARGET, substrate, hits)).tolist() == (
        np.asarray(without(OTHER_TARGET, substrate, hits)).tolist()
    )
    # And asserted directly: the oxidation is not among the rules a competitor probe would run.
    assert template_arities([FORWARD_OXIDATION]) == (1,)


def test_a_single_partner_application_has_no_competitors_to_find() -> None:
    """Nothing multi-component can fire on one molecule, so the channel is silent rather than clean.

    Which is the right reading: "no competing *bimolecular* reaction" is not "this step is clean",
    and a unimolecular step's competitors are a different question this channel deliberately does
    not answer.
    """
    measure, reports = competitor_measure([FORWARD_AMIDE, FORWARD_ESTER])
    precursors = object_mol_array([molecule("NCCO")])
    hits = np.array([(0, COUPLE, 0, 0)], dtype=np.int64)

    assert np.asarray(measure(OTHER_TARGET, precursors, hits)).tolist() == [0]
    # Examined, and found nothing. On *this* channel that is the good answer.
    assert reports[-1].probed == 1 and reports[-1].silent == 1


def test_competes_is_where_thermodynamics_and_an_allow_list_would_enter() -> None:
    """One predicate on ``(intended, candidate)``, and today's default is the crudest honest reading.

    A favourability table would refuse a candidate far less favourable than the intended reaction; a
    conditions table, or an explicit allow-list of pairs, would refuse one that cannot run under the
    intended reaction's conditions. Both are this signature, and neither changes anything else. Here
    a stand-in allow-list says the esterification is compatible with the coupling -- different
    catalyst, so it never runs alongside -- and the count falls to the amide alone.
    """

    def allowed_together(intended: int, candidate: int) -> bool:
        return candidate != 1  # rule 1 is the esterification

    measure, _ = competitor_measure(
        [FORWARD_AMIDE, FORWARD_ESTER], competes=allowed_together
    )
    precursors, hits = two_partners("CC(=O)O", "NCCO")

    assert np.asarray(measure(OTHER_TARGET, precursors, hits)).tolist() == [1, 1]


def test_template_arities_reads_the_left_hand_side() -> None:
    """Components on the left, because that is what has to be present for a rule to fire."""
    assert template_arities([FORWARD_AMIDE, FORWARD_OXIDATION, BENZYLIDENE]) == (
        2,
        1,
        2,
    )


# --------------------------------------------------------------- the same answer on the device


@pytest.mark.metal
def test_the_packed_device_probe_and_the_host_probe_agree() -> None:
    """The parity claim the device probe has to earn: same graph in, same counts out.

    The two do genuinely different work. The host probe pulls molecules out of an object array, hands
    them to `cmol_union`, and re-packs whatever the matcher wants. The device probe writes each
    group's atom and bond blocks straight into the packed layout the Metal matcher already consumes,
    so **no molecule is ever built** -- and naming the products is one more dispatch over those
    same arrays rather than a trip through the host. Two implementations of one measurement, and
    the only reason to trust the faster one is that it answers identically.
    """
    from vertsy.selectivity import object_combine
    from vesma.cmol import cmol_union
    from vesma.keyer import short_ring_wl_keyer
    from vesma.metal_matcher import GpuUnavailableError, probe_metal
    from vesma.metal_reaction_matcher import (
        MetalReactionMatcher,
        metal_combine,
        metal_short_ring_wl_key,
        metal_stack_mols,
    )

    try:
        probe_metal()
    except GpuUnavailableError as exc:
        pytest.skip(str(exc))

    smarts = (FORWARD_AMIDE, FORWARD_ESTER, BENZYLIDENE)
    patterns = tuple(parse_reaction_smarts(text) for text in smarts)
    partners = [molecule("CC(=O)O"), molecule("NCCO"), molecule("OCCCO")]
    # Three applications: a bimolecular one, another with the partners swapped, and a unimolecular
    # one -- so the gather path and the combine path are both exercised.
    hits = np.array(
        [
            (0, COUPLE, 0, 0),
            (0, COUPLE, 0, 1),
            (0, COUPLE, 1, 1),
            (0, COUPLE, 1, 2),
            (1, COUPLE, 0, 2),
        ],
        dtype=np.int64,
    )
    index = {COUPLE: (0, 1, 2)}

    host = site_probe(
        CpuReactionMatcher(patterns),
        index,
        key=lambda mols: [short_ring_wl_keyer.encode(m) for m in mols],
        combine=object_combine(cmol_union),
    )
    device = site_probe(
        MetalReactionMatcher(patterns),
        index,
        # The retro kernel WL-hashes every molecule it builds, so these are already computed.
        key=metal_short_ring_wl_key,
        combine=metal_combine,
    )

    on_host = np.asarray(host(OTHER_TARGET, object_mol_array(partners), hits)).tolist()
    on_device = np.asarray(
        device(OTHER_TARGET, metal_stack_mols(partners), hits)
    ).tolist()

    assert on_host == on_device
    assert any(count > 0 for count in on_host)  # and it measured something, not nothing


@pytest.mark.metal
def test_the_device_combination_refuses_a_group_over_the_atom_cap() -> None:
    """A truncated molecule is a different molecule, so it is refused rather than shortened.

    ``GPU_MATCH_ATOM_CAP`` is 64, and a combined precursor set can exceed it where neither part
    does. The row is written empty rather than truncated -- a truncated one would be measured, and
    measured wrongly -- and the **mask** is what says so. An empty row matches nothing, so a caller
    that could not tell it from a real row would read `SILENT` and charge a device atom budget as a
    broken pairing, which is exactly the confusion `UNPROBED` exists to prevent.
    """
    from vesma.metal_reaction_matcher import metal_combine, metal_stack_mols

    batch = metal_stack_mols([molecule("C" * 40)])
    fits, fits_mask = metal_combine(
        batch, np.array([0], dtype=np.int64), np.array([0, 1], dtype=np.int64)
    )
    over, over_mask = metal_combine(
        batch, np.array([0, 0], dtype=np.int64), np.array([0, 2], dtype=np.int64)
    )

    assert int(fits.atom_counts[0]) == 40 and bool(fits_mask[0])
    assert int(over.atom_counts[0]) == 0  # 80 atoms wanted, none written
    assert not bool(over_mask[0])


def test_a_group_the_backend_cannot_build_stays_unprobed_rather_than_silent() -> None:
    """The distinction end to end, through the probe rather than through the combiner alone.

    A combiner with an atom budget hands back a placeholder row for a group it cannot build. That
    row matches nothing, so a probe that ran it anyway would record zero outcomes -- `SILENT`, which
    `doubted` charges as a probably-broken pairing. A device limit is not a broken pairing, and the
    two must not be spelled the same way.
    """

    def refuse_everything(precursors, members, indptr):
        """A combiner with no room: a placeholder row per group, and a mask that says so."""
        groups = len(indptr) - 1
        return object_mol_array([molecule("C")] * groups), np.zeros(groups, dtype=bool)

    measure, reports = probe_of({FGI_1A: (0, 1)}, combine=refuse_everything)
    precursors = object_mol_array([molecule(XYLITOL), molecule(BUTANOL)])
    hits = np.array([(0, FGI_1A, 0, 0), (0, FGI_1A, 0, 1)], dtype=np.int64)

    assert np.asarray(measure(OTHER_TARGET, precursors, hits)).tolist() == [
        UNPROBED,
        UNPROBED,
    ]
    assert reports[-1].unbuildable == 1
    assert reports[-1].probed == 0 and reports[-1].silent == 0


def test_a_rule_firing_several_times_in_one_pot_is_one_application() -> None:
    """The unit every `ProbeReport` field counts is the application, and attribution is no exception.

    A forward application id is minted per *match*, so a pattern with three matches in one pot
    carries three hit rows with three different ids. Counting those separately makes a rule's fire
    count exceed the number of applications it was offered, which is how this was caught.
    """
    from vertsy.selectivity import _tally_by_rule

    hits = np.array(
        [
            (0, 7, 0, 0),  # rule 7, group 0, three matches
            (0, 7, 1, 1),
            (0, 7, 2, 0),
            (0, 9, 0, 0),  # rule 9, group 0
            (1, 7, 0, 0),  # rule 7, group 1
        ],
        dtype=np.int64,
    )

    assert _tally_by_rule(hits) == {7: 2, 9: 1}
    assert _tally_by_rule(np.zeros((0, 4), dtype=np.int64)) == {}


def test_no_rule_can_fire_on_more_applications_than_it_was_offered() -> None:
    """The invariant that makes the two tallies comparable, on a real probe pass.

    Xylitol has three distinct monochlorination products across two rules, so both fire and the
    primary rule matches in more than one place -- which is exactly the shape that broke the count.
    """
    measure, reports = probe_of({FGI_1A: (0, 1)})
    precursors, hits = one_application(XYLITOL)
    measure(OTHER_TARGET, precursors, hits)

    (report,) = reports
    assert report.fired_by_rule, "neither chlorination fired, so this asserts nothing"
    assert all(
        report.fired_by_rule.get(rule, 0) <= offered
        for rule, offered in report.offered_by_rule.items()
    )
    # One application, so no rule can be credited with more than one however often it matched.
    assert set(report.fired_by_rule.values()) == {1}
    assert report.offered_by_rule == {0: 1, 1: 1}


def test_an_unstated_rule_does_not_compete() -> None:
    """Three answers, and the third is why this is not `template_arities`.

    A bare dot states nothing: the pattern matches across two molecules and inside one
    indifferently, so reading it as "two separate starting materials" is the assumption that put
    cyclisations into the competitor channel. `declared_molecularity` answers `0` there, and the
    channel's `>= 2` filter drops it without having to know the sentinel exists.

    A partially grouped side is unstated too, and Daylight says why: in `(C).C` the ungrouped atom
    is unconstrained and may match the same component, so the side has not claimed two molecules.
    """
    assert declared_molecularity(
        [
            "[C:1].[N:2]>>[C:1][N:2]",
            "([C:1]).([N:2])>>[C:1][N:2]",
            "([C:1].[N:2])>>[C:1][N:2]",
            "([C:1]).[N:2]>>[C:1]",
            "[C:1](-[N:2])>>[C:1]",
            "([C:1]).([N:2]).([O:3])>>[C:1]",
        ]
    ) == (0, 2, 1, 0, 1, 3)
    # The literal piece count is a different question and still answers it.
    assert template_arities(["[C:1].[N:2]>>[C:1][N:2]"]) == (2,)


def test_a_competitor_that_remakes_the_intended_product_is_not_one() -> None:
    """The step's own product cannot compete with itself, whatever rule reached it.

    Two shapes of this were measured and both were being counted. One reaction under two codes --
    `MC 75` and `iSCB 70` are the same esterification, so each appeared as the other's competitor
    -- and one bond by two mechanisms, `Hartenfeller 51g` Buchwald against `FWD H 54r` SNAr on an
    azine that admits both. 39 of 590 competitor products over 25 PaRoutes targets, 6.6%.

    Here the pot is an acid and an amino alcohol, which can couple through either the amine or the
    hydroxyl. Against a frontier holding neither product, both count. Against the amide, the amide
    coupling drops out and only the esterification is left -- without any pairing being declared,
    which is the point: the filter catches a synonym nobody wrote down.
    """
    measure, _ = competitor_measure([FORWARD_AMIDE, FORWARD_ESTER])
    precursors, hits = two_partners("CC(=O)O", "NCCO")

    assert np.asarray(measure(OTHER_TARGET, precursors, hits)).tolist() == [2, 2]
    amide = np.asarray(measure(frontier("OCCNC(C)=O"), precursors, hits)).tolist()
    ester = np.asarray(measure(frontier("CC(=O)OCCN"), precursors, hits)).tolist()

    assert amide == [1, 1], "the amide coupling is the step, not a competitor of it"
    assert ester == [1, 1], "and symmetrically for the ester"


def test_the_sites_channel_keeps_the_intended_product() -> None:
    """The asymmetry between the two channels, which is the whole reason this is a flag.

    `sites` counts where the intended reagent attacked, and the intended product is the first of
    those places. Dropping it would turn a perfectly selective step from one into zero and a
    two-site step into one, which is the threshold the price is set against. So the sites probe
    still never reads the frontier, and `UNREAD_PRODUCTS` proves it on every one of its tests.
    """
    measure, _ = probe_of({FGI_1A: (0, 1)})
    precursors, hits = one_application(BUTANOL)

    assert np.asarray(measure(UNREAD_PRODUCTS, precursors, hits)).tolist() == [1]


def test_conditions_allow_is_a_subset_and_not_a_formula() -> None:
    """Every condition the candidate needs must be one the medium supplies. That is all of it.

    A reaction needs *all* of its conditions, so the test is a conjunction over the candidate's
    own requirements, which a subset expresses exactly: no disjunction, no grouping, no expression
    language. A rule requiring nothing competes everywhere, which is right -- a base-mediated
    alkylation or a condensation runs in most pots -- and a rule requiring palladium competes only
    where the intended reaction brought palladium with it.
    """
    allowed = conditions_allow(
        provides={
            0: frozenset({"palladium"}),
            1: frozenset(),
            2: frozenset({"palladium", "strong-acid"}),
        },
        requires=(
            frozenset(),  # 0: needs nothing
            frozenset({"palladium"}),  # 1: a coupling
            frozenset({"carbanion"}),  # 2: a Grignard
            frozenset({"palladium", "strong-acid"}),  # 3: both
        ),
    )

    # Rule 0 needs nothing, so it runs anywhere.
    assert allowed(0, 0) and allowed(1, 0) and allowed(2, 0)
    assert allowed(0, 1) and not allowed(1, 1), "palladium only where palladium is"
    # No medium here is a carbanion medium, so rule 2 never competes.
    assert not any(allowed(intended, 2) for intended in (0, 1, 2))
    assert allowed(2, 3) and not allowed(0, 3), "both conditions, or neither"
    # An intended rule nobody tagged supplies nothing, so it admits only the undemanding.
    assert allowed(99, 0) and not allowed(99, 1)


def test_an_outcome_that_makes_the_intended_product_beside_something_else_is_not_competing() -> (
    None
):
    """Contains, not equals -- and this is the test that has to hold before by-products are added.

    `FWD MC 85a` already emits two components: an ester reduction gives the alcohol from the acyl
    half and the alcohol from the alkoxy half. Reproducing the intended product beside another
    molecule still reproduces the intended product, so it is not a competition, and an equality
    test missed exactly that. Writing by-products onto the forward table would make every row
    that shape, which is why this cannot wait for them.
    """
    one = cmol_smiles_keyer.encode(molecule("CCO"))
    other = cmol_smiles_keyer.encode(molecule("CCN"))
    hits = np.array([(0, 0, 0, 0), (0, 0, 0, 1), (0, 1, 1, 0)], dtype=np.int64)

    # Application (0, 0) emitted both; application (1, 1) emitted only the intended product.
    assert distinct_outcomes([one, other], hits, 1, intended=[one]) == [0]
    # And an outcome that does *not* contain it is still counted.
    assert distinct_outcomes([one, other], hits, 1, intended=[other]) == [1]


def test_admit_unimolecular_relaxes_the_filter_the_conditions_gate_replaces() -> None:
    """The arity filter is a proxy for "its reagent is not in the pot"; `competes` says that itself.

    `FORWARD_OXIDATION` is the unimolecular rule this channel exists to exclude -- an alcohol
    elsewhere on a precursor is not a competition, because the oxidant is not in the pot. With
    `admit_unimolecular` it is counted, and then it is `competes` that has to do the excluding,
    which is the point: one filter stating the physical argument instead of two, one of them a
    stand-in.

    Default off, so this asserts the default as much as the flag.
    """
    smarts = [FORWARD_ESTER, FORWARD_OXIDATION]
    pot = two_partners("CC(=O)O", "CC(C)O", rule=0)

    strict, _ = competitor_measure(smarts)
    relaxed, _ = competitor_measure(smarts, admit_unimolecular=True)

    # The esterification remakes the intended product, so it is not a competitor either way --
    # which leaves the oxidation as the only thing the flag can change.
    assert strict(frontier("CC(=O)OC(C)C"), *pot)[0] == 0
    assert relaxed(frontier("CC(=O)OC(C)C"), *pot)[0] == 1, "and is one when admitted"

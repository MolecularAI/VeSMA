"""The hand-written forward rule set: it parses, it names real partners, and it round-trips.

These rows are the other half of a retro template -- what the reaction *does*, rather than what it
came from -- and they exist because a retro SMARTS cannot be reversed mechanically. A left-hand side
is a match pattern and may contain alternation; a right-hand side is a rewrite program, where
alternation means nothing; and ``parse_reaction_smarts`` validates the right-hand side for concrete
elements and bond types, which a query-laden product side is not.

Nothing upstream will catch a regression in a row written here, so what is locked below is what was
actually hard to get right: the arity has to match the retro rule's precursor side, the probes have
to name codes that exist, and every row has to reproduce its own partner's product from its own
partner's starting materials.
"""

from __future__ import annotations

import csv
from functools import cache

import pytest

from tests.datadirs import external_data_root
from vertsy.selectivity import forward_index
from vertsy.template_lint import (
    duplicates,
    top_level_pieces,
    ungrouped_components,
)
from vesma import cmol_from_smiles, generate_precursors, parse_reaction_smarts
from vesma.cmol import cmol_union
from vesma.keyer import cmol_smiles_keyer

TEMPLATES = external_data_root() / "templates"
FORWARD = TEMPLATES / "templates-forward.csv"
# Every table a `probes` entry may name. Hartenfeller is vendored; the rest are written here.
PARTNER_SETS = (
    TEMPLATES / "templates-medchem.csv",
    TEMPLATES / "templates-protecting-groups.csv",
    TEMPLATES / "templates-selenium.csv",
    TEMPLATES / "templates-carbohydrate.csv",
    TEMPLATES / "templates-phosphorus.csv",
    TEMPLATES / "templates-hartenfeller.csv",
)


@cache
def forward_rows() -> tuple[dict[str, str], ...]:
    with FORWARD.open() as handle:
        return tuple(csv.DictReader(handle))


@cache
def partner_codes() -> frozenset[str]:
    codes: set[str] = set()
    for path in PARTNER_SETS:
        with path.open() as handle:
            codes.update(row["template_code"] for row in csv.DictReader(handle))
    return frozenset(codes)


@cache
def partner_templates() -> dict[str, str]:
    """``template_code -> retro_template`` over every table a ``probes`` entry may name."""
    found: dict[str, str] = {}
    for path in PARTNER_SETS:
        with path.open() as handle:
            for row in csv.DictReader(handle):
                found.setdefault(row["template_code"], row["retro_template"])
    return found


def pairings() -> tuple[tuple[str, str], tuple[str, ...]]:
    """Every ``(forward code, retro code)`` the table asserts, flattened, with ids to match."""
    pairs = tuple(
        (row["forward_code"], code.strip())
        for row in forward_rows()
        for code in row["probes"].split(";")
        if code.strip() and not code.strip().startswith("class:")
    )
    return pairs, tuple(f"{forward}->{retro}" for forward, retro in pairs)


def molecule(smiles: str):
    built, _info = cmol_from_smiles(smiles)
    return built


def ids() -> list[str]:
    return [row["forward_code"] for row in forward_rows()]


@pytest.mark.parametrize("row", forward_rows(), ids=ids())
def test_every_forward_row_parses_under_vesma(row) -> None:
    """Recursive SMARTS is refused on purpose, so a row carrying one is a row nobody can run.

    ``vesma.parsing`` rejects ``$(`` deliberately, and the vendored Hartenfeller *retro* table was
    flattened by hand for exactly this reason -- its file name says ``nonrecursive``. A forward row
    gets the same treatment rather than an exception to the rule.
    """
    assert "$(" not in row["forward_template"]
    parse_reaction_smarts(row["forward_template"])


@pytest.mark.parametrize("row", forward_rows(), ids=ids())
def test_every_forward_row_probes_a_retro_rule_that_exists(row) -> None:
    """Otherwise the pairing drifts silently and the probe measures nothing while looking busy.

    **An empty ``probes`` is a category, not a drift.** A few rows describe a reaction nobody would
    ever want -- an arylboronic acid destroyed by NBS -- so no retro rule is its inverse, because
    nobody disconnects an arene *into* a boronic acid on purpose. Such a row is never a ``sites``
    probe, and that is right: it is not anybody's intended reaction. It is a *competitor*, which is
    exactly what that channel runs -- every multi-component rule the intended rule is not paired
    with -- so the row runs after all, and the reason once given here for forbidding this ("a
    forward rule that probes nothing would never run") was false.

    What still has to hold is that a code written down resolves, so a typo cannot masquerade as a
    degradation row.
    """
    # Semicolon-separated: a template code contains a space ("FGI 1a").
    probes = {code for code in row["probes"].split(";") if code}
    if not probes:
        assert len(row["forward_template"].split(">>")[0].split(".")) >= 2, (
            f"{row['forward_code']} pairs with nothing, so it can only ever run as a competitor "
            "-- and that channel runs multi-component rules only, so a unimolecular row with no "
            "partner would never run at all."
        )
        return
    assert probes <= partner_codes()


@pytest.mark.parametrize("row", forward_rows(), ids=ids())
def test_every_forward_row_reproduces_its_partners_product(row) -> None:
    """The lint that replaces mass balance, and the only one that means anything here.

    A retro library is linted for atoms going *missing*, because a retro step that loses a fragment
    is always a defect. A forward rule legitimately loses atoms -- a condensation expels water -- and
    legitimately gains them -- an alkylation adds a whole group -- so mass balance says nothing about
    it and is deliberately not applied.

    What does mean something is the round trip: run the forward rule on its partner's starting
    materials and the partner's product has to come out. A rule that cannot do that is measuring a
    different reaction from the one it is paired with, and every count it produces is about the wrong
    chemistry.
    """
    parts = [molecule(part) for part in row["example_precursors"].split(".")]
    outcomes = generate_precursors(
        product=cmol_union(parts),
        reaction=parse_reaction_smarts(row["forward_template"]),
    )
    produced = {
        cmol_smiles_keyer.encode(component)
        for outcome in outcomes
        for component in outcome.precursors
    }
    assert cmol_smiles_keyer.encode(molecule(row["example_product"])) in produced


@pytest.mark.parametrize("row", forward_rows(), ids=ids())
def test_no_forward_row_needs_sssr_ring_perception(row) -> None:
    """The Metal and CUDA pipelines switch ring perception off, so a forward rule cannot ask for it.

    Bond-level ``@`` and ``!@`` are fine -- those need only per-edge cycle membership, which both
    device paths request. Ring *count* primitives are what is unavailable.
    """
    left = row["forward_template"].split(">>")[0]
    assert ";R2" not in left and ";r" not in left


def test_forward_templates_are_distinct() -> None:
    """Two rows with one SMARTS would double-count every outcome they both produce."""
    assert not list(duplicates([row["forward_template"] for row in forward_rows()]))


def test_forward_codes_are_distinct() -> None:
    codes = [row["forward_code"] for row in forward_rows()]
    assert len(set(codes)) == len(codes)


def test_a_forward_rule_has_the_arity_of_its_partners_precursor_side() -> None:
    """The constraint that decides how most forward rules are written, and why.

    A retro template may drop a bulk reagent: ``PG 9`` removes a benzylidene acetal and its
    precursor side is the bare diol, with the benzylidene carbon unmapped and deleted. The
    application the probe is asked about therefore has **one** precursor, so a two-component forward
    rule would match nothing and read as silent. The forward rule invents the reagent on its
    right-hand side instead, mirroring the convention its partner already uses.

    Where the retro rule *does* name both partners -- ``Hartenfeller 47`` gives an acid chloride and
    an amine -- the forward rule is bimolecular and the union has something to do.

    Arity is the number of zero-level *groups*, not of dots. A dot inside a group is two fragments
    of one molecule -- the intramolecular form, `FWD MC 102`'s amino acid chloride closing to a
    lactam -- and counting it would ask that row for two example precursors when the reaction has
    one. The same correction `vertsy.selectivity.template_arities` needed, and for the same reason.
    """
    arity = {
        row["forward_code"]: len(
            top_level_pieces(row["forward_template"].split(">>")[0])
        )
        for row in forward_rows()
    }
    # PG 9's precursor side was the diol alone until the benzaldehyde it condenses with was named;
    # `FWD MC 102` below is the shape that is still genuinely one molecule.
    assert arity["FWD PG 9"] == 2
    assert arity["FWD H 47"] == 2  # Hartenfeller 47 names both partners
    assert arity["FWD MC 102"] == 1  # the same reaction, intramolecular
    assert all(
        count == len(row["example_precursors"].split("."))
        for row, count in zip(forward_rows(), arity.values())
    )


# --------------------------------------------------------------------- resolving codes to ids


def test_pairings_resolve_to_the_ids_a_graph_actually_stores() -> None:
    """Codes are how a chemist writes a pairing; ids are how a graph stores one.

    A rule id is a position in whichever rule set a run compiled, so the names live where the rule
    sets are loaded and `vertsy.selectivity` only ever sees ints. One retro rule collects every
    forward rule that probes it -- which is the reagent-scope statement: a chlorinating agent does
    not know which hydroxyl was intended, so both the primary and the secondary chlorination probe
    the same disconnection.
    """
    probes = {row["forward_code"]: row["probes"].split(";") for row in forward_rows()}
    retro_ids = {"FGI 1a": 10, "FGI 4a": 11, "PG 9": 12}
    forward_ids = {"FWD FGI 1": 0, "FWD FGI 4": 1, "FWD PG 9": 2}

    index = forward_index(probes, retro_ids, forward_ids)

    assert index[10] == (
        0,
        1,
    )  # both chlorinations probe the primary-alcohol disconnection
    assert index[11] == (0, 1)  # and the secondary one
    assert index[12] == (2,)


def test_a_pairing_for_a_rule_set_this_run_did_not_load_is_simply_dropped() -> None:
    """A run may load a subset of the rule sets, and a pairing with nothing to do is not an error.

    The error worth raising over is a code that exists *nowhere*, and that is caught by the table
    test above, before any run starts.
    """
    index = forward_index({"FWD FGI 1": ["FGI 1a"]}, {}, {"FWD FGI 1": 0})
    assert index == {}

    absent_forward = forward_index({"FWD FGI 1": ["FGI 1a"]}, {"FGI 1a": 10}, {})
    assert absent_forward == {}


def test_the_bench_resolves_the_table_against_a_rule_set_it_compiled() -> None:
    """Codes become ids only where the rule sets are loaded, so that seam gets its own test.

    What this pins is the *shape*, not the size, and the shape has three tiers. Most paired rules
    carry one forward row -- their own inverse. Nine carry two. Six are the halide FGIs, where
    three halogens of the primary-alcohol disconnection and three of the secondary are each probed
    by both the chlorination and the bromination, because a halogenating agent does not know which
    hydroxyl was intended. The other three are the aluminium-hydride reductions `MC 85a`, `MC 85b`
    and `MC 86`, each seeing its own inverse plus the other carbonyl that reagent reaches.

    That tier held three rows apiece until the two ester rows became one. They were split
    methyl-only and ethyl-only not for chemistry but because a flat right-hand side has to delete
    the *entire* leaving group: widening `-[O][CH3]` to `-[O]-[CX4]` severs the rest of the alkyl
    chain, and the engine refuses the application rather than inventing a fragment. Carrying the
    leaving group through as `[*:4]` and emitting the alcohol it becomes says the same chemistry
    once, for any ester. The borohydride pair `MC 5` and `MC 6` sit in the one-row tier, because a
    borohydride reaches only an aldehyde and a ketone -- the asymmetry is the point and is
    described in the templates README.

    The totals move whenever a row or a scope entry is added, which is why they are asserted once
    here rather than described twice. Everything else in the set is unprobed, which is the honest
    reading and what the coverage line reports.
    """
    from tests.vertsy.bench_host_aostar import (
        TEMPLATE_FILES,
        load_forward_library,
        load_template_library,
    )

    _count, _smarts, sources = load_template_library(
        "medchem", TEMPLATE_FILES["medchem"]
    )
    forward = load_forward_library(sources)

    assert len(forward.smarts) == len(forward_rows())
    # 88 before the nine intramolecular siblings `MC 99`-`MC 103` were written and paired,
    # 97 before `MC 104` added lactonisation beside the esterification that now refuses it, and 98
    # before `MC 105a`/`MC 105b` gave `FWD iSCB 68o` the ester-from-acyl-chloride disconnection it
    # had been pointing at an amide rule for.
    # 100 before the sixteen rows that closed the recorded-route gaps -- six amine N-alkylations,
    # the dehydration, three POBr3 azine halogenations and six gamma rows for the 1,2-diazines.
    assert forward.paired_rules == 116
    # Two forward rows apiece for the halide FGIs -- `FWD FGI 1` and `FWD FGI 4` are one
    # chemistry written against two retro spellings -- and one apiece for everything since. The
    # old blanket "exactly two" was an accident of the table holding only the first kind.
    #
    # The tail shortened when the hydride reductions started naming their reagent. `FWD MC 5`
    # and `FWD MC 6` carried a `scope` pairing apiece listing the stronger hydrides, because the
    # pattern could not say "either hydride does this" while the reagent was implicit. It can now
    # -- `[BH4-,AlH4-]` on the left-hand side -- so those pairings are gone and the rules they
    # named lost a forward partner each.
    counts = sorted(len(found) for found in forward.index.values())

    # One rule now carries three forward rows: `MC 73a` became the single sulfonate-ester
    # disconnection, with the sulfonyl group riding a mapped star, so the mesylate, tosylate and
    # triflate forward rules all probe it. The two rows they used to probe are gone, which is
    # also why the paired total fell by two.
    # Five more singletons: the directed-metalation rules `MC 97a` to `MC 97c`, each probed by
    # the one forward row that describes its own halogen source, and the two gamma POCl3 rules.
    # Nine more singletons: the intramolecular siblings `MC 99` to `MC 103`, each probed by the
    # one forward row that describes the ring closure. Three of them share a row apiece --
    # `FWD MC 100` probes `MC 100a/b/c` -- because one pattern covers all three halogens, which
    # is how their bimolecular parents are written too.
    # And `MC 75` picked up a second row: `FWD MC 75` and `FWD iSCB 70` now both probe
    # `MC 75;iSCB 70`, because those two retro rules became one and whichever survives the loader's
    # dedup has to stay paired. Without that, the dropped code's forward row would lose its pairing
    # and join the competitor pool -- competing against the surviving twin's own step, which is the
    # measured gap where `MC 75` and `iSCB 70` each appeared as the other's competitor.
    # 89 in the one-row tier before the sixteen rows that closed the recorded-route gaps; every
    # one of them landed there, which is what a row written as its partner's own inverse looks like.
    assert counts == [1] * 105 + [2] * 10 + [3] * 1, counts
    assert forward.paired_rules < len(sources)  # and most of the set is unprobed


def test_the_cpu_arm_runs_every_forward_row_and_the_rdkit_arm_runs_the_unimolecular_ones() -> (
    None
):
    """The split that decides where the table is usable, asserted rather than left in prose.

    RDKit's `RunReactants` takes one molecule per reactant *template component*, and
    `vesma.matcher.ReactionMatcher` hands it one molecule per `(molecule, rule)` pair -- so a
    two-component forward rule has no way to receive its second starting material. Vesma's matcher
    matches a disconnected pattern into the disconnected graph `cmol_union` builds, so it has no
    such limit.

    Thirty-four of the forty-four rows need two starting materials, which means the bulk of this
    table and the whole competitor channel live on the vesma-native arm.
    """
    from tests.vertsy.bench_host_aostar import (
        TEMPLATE_FILES,
        load_forward_library,
        load_template_library,
    )

    _count, _smarts, sources = load_template_library(
        "hartenfeller", TEMPLATE_FILES["hartenfeller"]
    )
    forward = load_forward_library(sources)

    unimolecular = [rule for rule, arity in enumerate(forward.arities) if arity == 1]
    multi = [rule for rule, arity in enumerate(forward.arities) if arity >= 2]

    # Far more multi-component rows than unimolecular ones, and the ratio is not an accident:
    # every one of ChemoDOTS' 62 rules joins two things, because that is what a combinatorial
    # library enumerates. What is left unimolecular is a functional-group interconversion or a
    # protection whose reagent the retro partner does not name -- so this count *falls* as the
    # retro table names more of them, and fifteen rows moved across when it did.
    #
    # Naming the reagent is what lets a rule be a competitor at all: the arity filter asks
    # "could this fire in this pot", and a rule needing NBS can only answer yes where NBS is a
    # precursor. A formylation or a benzylic bromination had no way to say that while it was
    # unimolecular, so it was excluded from the channel by construction rather than by chemistry.
    #
    # Five of the unimolecular rows are unimolecular for a different reason, and they are why this
    # count is read by groups rather than by dots: `FWD MC 99` to `FWD MC 103` describe the
    # *intramolecular* half of a reaction whose bimolecular half is already here, so their one
    # starting material carries both halves and their left-hand side is `(A.B)`.
    # Six before the intramolecular siblings, and `FWD MC 104` is the seventh of that kind: its
    # left-hand side is `(A.B)`, one hydroxy acid closing on itself.
    # 129 multi before `FWD MC 105b`, the aryl half of the acyl-chloride esterification; 61/130
    # before the nine halogenations named their reagent -- SOCl2, CBr4/PPh3, I2/PPh3 -- and 52/139
    # before the fifteen protections, acetals and sulfonylations named theirs, and 37/154 before the
    # two ester hydrolyses named their water, and 35/156 before the azide displacements, the Boc and
    # acetate removals and the Suzuki boron named theirs, and 30/161 before the azine chlorinations
    # and the deprotections named POCl3, hydrazine, fluoride and their waters. Each named reagent moves its
    # row from unimolecular to multi-component, which is the whole point: the pot really does hold
    # two starting materials.
    #
    # 16/175 before the thirteen rows that closed the gaps the recorded PaRoutes routes exposed:
    # amine N-alkylation, the POBr3 azine halogenations and their 1,2-diazine ring shapes, the
    # benzothiazole from an acid. Twelve of those name a second starting material; the thirteenth,
    # `FWD MC 107`, is the tertiary-benzylic dehydration, and it is unimolecular for the first
    # reason in this comment rather than the second -- the acid is the medium, not a precursor.
    # 17/187 before `FWD MC 38` and `FWD MC 39` named their hypochlorite, which is this comment's
    # first paragraph happening again: the count falls as the table names more of its reagents.
    # 15/189 before the thirteen rows that had been crossing a heavy atom named theirs -- hydrazine,
    # ammonia, hydroxylamine, magnesium, water, a boric acid -- and the six reductive aminations
    # named their hydride. `FWD MC 79`, alkene hydrogenation, is one of the two left unimolecular
    # by hydrogen gas rather than by anybody's choice.
    # 13/191 before the four destructive rows -- N-halogenation by a halosuccinimide, the
    # hypochlorite N-chlorination, and the two amide dehydrations -- which is the count rising for
    # the first time in this comment's history rather than falling. Each names its reagent as a
    # left-hand component, which is what a destructive row has to do: the whole claim is that this
    # reagent is in the pot and eats a bystander group, so a row that left it implicit would fire
    # on substrate alone and say nothing about the medium.
    # 13/195 before `FWD DEG 7`, the N-iodination by molecular iodine, which the first severity
    # measurement asked for: `MC 97c` halogenates with `II` rather than an N-halosuccinimide, so
    # `FWD DEG 3` could not see it and the channel read a clean 0 on a substrate carrying a free
    # aniline.
    assert len(unimolecular) == 13 and len(multi) == 199
    # And the pairing this buys against a real rule set, which is the number a run prints. 168 and
    # not 183 since the fifteen duplicate `iSCB` rows were deleted; each was already dropped at load,
    # so this count fell without any pairing losing a rule that compiles. 169 once `Hartenfeller 4b`
    # gave the benzothiazole the carboxylic-acid route its benzoxazole twin already had.
    assert forward.paired_rules == 169
    assert forward.paired_rules / len(sources) > 0.95


def test_the_competitor_channel_has_the_multi_component_rules_to_work_with() -> None:
    """A channel that ran only unimolecular rules would have nothing to say, by construction."""
    from tests.vertsy.bench_host_aostar import (
        TEMPLATE_FILES,
        load_forward_library,
        load_template_library,
    )
    from vertsy.selectivity import competitor_probe

    _count, _smarts, sources = load_template_library(
        "medchem", TEMPLATE_FILES["medchem"]
    )
    forward = load_forward_library(sources)
    assert sum(1 for arity in forward.arities if arity >= 2) == 199

    # The probe refuses to be built without a union, because every rule it runs needs one.
    with pytest.raises(TypeError):
        competitor_probe(object(), forward.arities, key=lambda mols: [])  # type: ignore[call-arg]


# ------------------------------------------------------------ the pairing composes both ways


#: Pairings that are right about the *reaction* and cannot compose, because the vendored retro
#: rule's own precursors do not react. `Hartenfeller 57`/`58` write the urea disconnection as
#: ``[N:3]-[C:1](=O)-[N:2]>>[N:3]-[C:1]=O.[N:2]``, and the carbonyl carbon then gains an implicit
#: hydrogen: the precursor is an N-substituted *formamide*, not the isocyanate the author meant,
#: and a formamide does not make a urea with an amine. The vendored table is the source of truth
#: and is not edited here, so the pairing stays and the probe reports **silent** -- which is the
#: correct verdict, and is charged as low confidence rather than treated as clean.
#:
#: Asserted to still not compose, so this list cannot rot into a way of muting a real regression.
RETRO_PRECURSORS_DO_NOT_REACT = frozenset({"Hartenfeller 57", "Hartenfeller 58"})


def forward_outcomes(template: str, parts) -> set[frozenset]:
    """Every distinct component multiset the forward rule makes from one pot."""
    return {
        frozenset(cmol_smiles_keyer.encode(part) for part in outcome.precursors)
        for outcome in generate_precursors(
            product=cmol_union(list(parts)),
            reaction=parse_reaction_smarts(template),
        )
    }


@pytest.mark.parametrize("pairing", pairings()[0], ids=list(pairings()[1]))
def test_a_pairing_that_can_be_composed_does_compose(pairing: tuple[str, str]) -> None:
    """Split the forward row's product with the retro partner, then rebuild it with the forward rule.

    The other round trip runs a forward rule on the precursors *its own row* names, which the author
    chose to satisfy it, and a retro rule on nothing at all. Neither can catch the failure that
    actually happened: a retro rule emitting an aryl chloride paired with a forward rule that wants
    bromide or iodide passes both one-sided checks, then reports **silent** on every real
    application -- and silence costs nothing, so the broken half of the pairing wins the route.
    Composing the two is the only check that sees it.

    Skipped, not failed, when the retro partner does not split this product: one forward rule pairs
    with a family of retro variants specialised by substitution pattern, and a single example
    product can only exercise one of them. That is a limit of having one example per row rather
    than a defect in the pairing, and failing on it would make the suite demand examples nobody
    needs.
    """
    forward_code, retro_code = pairing
    row = next(each for each in forward_rows() if each["forward_code"] == forward_code)
    retro = partner_templates()[retro_code]
    product = molecule(row["example_product"])
    wanted = cmol_smiles_keyer.encode(product)

    splits = generate_precursors(product=product, reaction=parse_reaction_smarts(retro))
    if not splits:
        pytest.skip(f"{retro_code} does not split this forward row's example product")
    if retro_code in RETRO_PRECURSORS_DO_NOT_REACT:
        assert not any(
            wanted in outcome
            for split in splits
            for outcome in forward_outcomes(row["forward_template"], split.precursors)
        ), f"{retro_code} now composes -- drop it from RETRO_PRECURSORS_DO_NOT_REACT"
        return

    rebuilt = [
        forward_outcomes(row["forward_template"], split.precursors) for split in splits
    ]
    assert any(wanted in outcome for outcomes in rebuilt for outcome in outcomes), (
        f"{forward_code} cannot rebuild {row['example_product']} from the precursors "
        f"{retro_code} makes of it -- the two halves disagree about scope"
    )


def test_no_forward_rule_leaves_its_left_hand_side_unstated() -> None:
    """The guard that replaced `require_span`, and the reason that subtraction could retire.

    The competitor channel runs the multi-component forward rules and asks what two starting
    materials do to each other. A bare dot does not ask that: a query dot is no constraint, so a
    two-piece pattern matched against a pot of two precursors also matches both pieces *inside
    one* of them, and the cyclisation that results was counted as a competition. `require_span`
    worked around it by probing each precursor alone and subtracting; grouping states it, and a
    grouped pattern cannot match inside one molecule at all -- so the subtraction became provably
    empty and came out.

    Provably empty only while this holds. One ungrouped bimolecular row would bring the whole
    defect back, silently, so the condition is a test rather than a note.
    """
    unstated = [
        forward_rows()[index]["forward_code"]
        for index, left, _right in ungrouped_components(
            [row["forward_template"] for row in forward_rows()]
        )
        if left
    ]

    assert not unstated, unstated


# What a forward row gains, where its pairing names the reagent and the row declares what leaves.
# The gain *is* the reagent, which is the whole account: these rules invent their reagent on the
# right-hand side by the table's convention, so moving it onto the left would change their arity
# and admit them to the competitor channel. Declaring the by-products instead makes the
# stoichiometry complete and checkable without that.
DECLARED_REAGENTS = {
    "FWD PG 1": "C10H18O5",  # di-tert-butyl dicarbonate
    "FWD PG 11": "C3H6O",  # acetone
    "FWD PG 12": "C3H6O",
    "FWD PG 13": "C4H6O3",  # acetic anhydride
    "FWD PG 14": "C4H6O3",
    "FWD PG 15": "C7H7Br",  # benzyl bromide
    # Zinc metal, declarable once `Zn` gained the zero-coordinate entry magnesium always had.
    "FWD H 33": "Zn",
    "FWD H 33d": "Zn",
    "FWD H 33g": "Zn",
    # 2-cyanoethyl N,N-diisopropylchlorophosphoramidite, the phosphitylating reagent.
    "FWD P 5": "C9H18ClN2OP",
    "FWD PG 17": "C4F6O3",  # trifluoroacetic anhydride
    # The aluminium-hydride reductions, which used to sit in `IMPLICIT_REAGENTS` crossing a bare
    # `Al`. The reagent that closed each is where the oxygen on the aluminium came from: the amide
    # supplies its own, because the reduction breaks that C-O bond, so the hydride was already
    # enough; an ester keeps both oxygens on carbon and a nitrile has none, so for those it is the
    # workup water. All three now emit `[O-][AlH2]` and balance on heavy atoms, hydrogen and
    # charge alike.
    "FWD MC 85a": "H2O",
    "FWD MC 86": "H4Al",
    "FWD MC 87": "H2O",
    # The alcohol oxidations, out of `REDOX_HYDROGEN` for the same reason the hydride reductions
    # came out of `IMPLICIT_REAGENTS`: the reagent was nameable all along. Hypochlorite takes both
    # hydrogens as the water it becomes and leaves chloride, so the rows balance on heavy atoms,
    # hydrogen and charge. Named as the anion, the way `[AlH4-]` and `[BH4-]` are, because bleach
    # on a right-hand side fragments into its ions.
    "FWD MC 38": "ClO",
    "FWD MC 39": "ClO",
    # The condensations, each closed by naming the reagent that supplies the ring atoms it was
    # inventing. None of these needed anything the engine could not spell; they had simply never
    # been asked. `H 16`/`H 16e` gained two ring nitrogens from nowhere, which is hydrazine;
    # `H 22` two more, which is the two ammonia Debus-Radziszewski has always used; `H 28` one,
    # which is the hydroxylamine that makes the amidoxime.
    "FWD H 16": "H4N2",
    "FWD H 16e": "H4N2",
    "FWD H 22": "H3N",
    "FWD H 28": "H3NO",
    # The Grignards, closed by the metal and the water that hydrolyses the adduct. `[Mg]` is named
    # the way `[Zn]` is for the Negishi rows -- the metal really is what goes in the flask.
    "FWD H 44": "Mg",
    "FWD H 45": "Mg",
    # And the row whose reagent was water all along: the Staudinger's hydrolysis.
    "FWD MC 45": "H2O",
    # The degradation rows, whose reagent is the thing that does the destroying. Each used to name
    # water, which no pot holding an N-halosuccinimide also holds -- so both boron rows declared a
    # third component that never appeared and neither had ever matched anything. The reagent was
    # the halosuccinimide all along; the boron leaves as metaboric acid or its halo ester instead.
    "FWD DEG 1": "C4H4BrNO2",
    "FWD DEG 2": "C4H4BrNO2",
    # Narrowed to the aliphatic amines, which is where N-halogenation is irreversible: a
    # primary/secondary N-haloamine eliminates to the imine or oxidises onward, while an
    # N-haloamide is a halogen *reservoir* -- NBS is one -- and an N-haloanilide rearranges back to
    # ring halogenation. So an amide, sulfonamide, lactam or aniline N-H is no longer called
    # destroyed, and the two nitrogen classes are separate rows because the dialect has no
    # recursive SMARTS to exclude an amide neighbour with.
    "FWD DEG 3a": "C4H4BrNO2",
    "FWD DEG 3b": "C4H4BrNO2",
    # Hypochlorite named as the anion, for the reason `FWD MC 38` names it that way.
    "FWD DEG 4a": "ClO",
    "FWD DEG 4b": "ClO",
    "FWD DEG 5": "Cl2OS",  # thionyl chloride
    "FWD DEG 6": "Cl3OP",  # phosphoryl chloride
    # Molecular iodine, which is not an N-halosuccinimide and so was invisible to `FWD DEG 3`.
    # Found by the measurement: `MC 97c` iodinates with `II`, and on a substrate carrying a free
    # aniline the destructive channel read a clean 0 while the chemistry is anything but.
    "FWD DEG 7a": "I2",
    "FWD DEG 7b": "I2",
    # The reductive aminations, closed the way `FWD MC 5` and `FWD MC 6` already were: the hydride
    # named and mapped through a wildcard, so the residue is `[BH3-]O` or `[OH][AlH3-]` according
    # to which hydride the pot held, and the water that protonates the alkoxide named beside it.
    # These sat in `REDOX_HYDROGEN` as though a hydride were hydrogen gas. It is not -- a hydride
    # is a molecule, and the hydride reductions one row above had already proved it.
    "FWD H 30": "H4B",
    "FWD H 30c": "H4B",
    "FWD H 30k": "H4B",
    "FWD H 30ck": "H4B",
    "FWD MC 99": "H4B",
    "FWD MC 103": "H4B",
    # The aerobic oxidations, closed by naming the dioxygen and the peroxide it becomes. The
    # two-electron path is what makes this 1:1 and so writable at all; the four-electron path to
    # water would need one O2 per two substrates, which one template cannot say.
    "FWD H 3": "O2",
    "FWD H 4": "O2",
    "FWD H 5": "O2",
    "FWD H 49": "O2",
}


@pytest.mark.parametrize(
    ("code", "reagent"),
    sorted(DECLARED_REAGENTS.items()),
    ids=sorted(DECLARED_REAGENTS),
)
def test_what_a_declaring_row_gains_is_exactly_its_reagent(
    code: str, reagent: str
) -> None:
    """These rows now name their reagent, so the invariant is the stronger one: nothing moves.

    It used to be "what this row gains must be exactly the reagent its retro partner names". That
    was the best available account while the reagent sat off the left-hand side: more would mean the
    declaration invented something, less that a by-product was missing. The reagent is now a
    left-hand component, so there is nothing left to infer -- the row gains nothing and loses
    nothing on its own example, and the formula below records which reagent closed it.

    Kept as a parametrised list rather than folded into a table-wide balance check, because these
    are the rows where the choice of reagent was a judgement -- Boc2O over Boc-Cl, Ac2O over acetyl
    chloride -- and the formula is where that judgement is written down.
    """
    from collections import Counter

    from vesma.cmol import cmol_union
    from vesma.formula import composition, difference, hill_formula
    from vesma.keyer import cmol_rdkit_smiles_keyer

    row = next(one for one in forward_rows() if one["forward_code"] == code)
    pot = cmol_union([molecule(part) for part in row["example_precursors"].split(".")])
    wanted = cmol_rdkit_smiles_keyer.encode(molecule(row["example_product"]))
    outcome = next(
        one
        for one in generate_precursors(
            pot, parse_reaction_smarts(row["forward_template"])
        )
        if any(
            cmol_rdkit_smiles_keyer.encode(part) == wanted for part in one.precursors
        )
    )
    emitted: Counter[int] = Counter()
    for part in outcome.precursors:
        emitted += composition(part)
    lost, gained = difference(composition(pot), emitted)

    # The reagent is named on the left now, so nothing is gained and nothing is lost.
    assert not gained and not lost
    # And it really is that reagent: its formula is one of the row's own starting materials.
    #
    # Read from `example_precursors` rather than from the template's left-hand side. The earlier
    # version parsed each left-hand component as a molecule and skipped whatever would not parse,
    # on the reasoning that a substrate side is a query and only a reagent literal could answer.
    # That made the check available only to rows that spell their reagent *unmapped* -- and so it
    # argued for the weaker notation, since a mapped reagent is a query too. Mapping is the better
    # statement where the reagent's atoms go somewhere nameable: hypochlorite's oxygen becomes the
    # water's and its chlorine becomes the chloride, which `([O-;H0:4][Cl:5])` says and
    # `([O-]Cl)` only balances by count. The example precursors are a statement of fact about the
    # row either way, so the check belongs there and the notation is free.
    named = []
    for piece in row["example_precursors"].split("."):
        named.append(hill_formula(composition(molecule(piece))))
    assert reagent in named


def test_no_forward_row_leaves_any_side_unstated() -> None:
    """The forward table is fully migrated, both sides, and this is what keeps it that way.

    The left-hand side matters because an unstated multi-component pattern matches inside one
    molecule as readily as across two, which is what put cyclisations into the competitor channel.
    The right-hand side matters now that rows declare their by-products: a product and its water
    come apart, and saying so is a postcondition the engine checks rather than a comment.

    The retro table is not there yet -- `ungrouped_components` reports the rules still to go --
    which is why this is asserted here and not over everything.
    """
    unstated = [
        forward_rows()[index]["forward_code"]
        for index, left, right in ungrouped_components(
            [row["forward_template"] for row in forward_rows()]
        )
        if left or right
    ]

    assert not unstated, unstated


def test_every_condition_class_is_in_the_shared_vocabulary() -> None:
    """`requires` is a controlled list, not free text.

    The vocabulary file carries the definition of each class, which is the only place the reasoning
    for a tag lives -- why `oxidant` is one class and not one per reagent, why `base` is not a class
    at all. A typo would read as a condition nothing supplies, which silently drops every candidate
    needing it, so the spelling is checked rather than trusted.
    """
    with (TEMPLATES / "template-tag-vocabulary.csv").open(newline="") as handle:
        vocabulary = {row["tag"] for row in csv.DictReader(handle)}

    used = {
        tag.strip()
        for row in forward_rows()
        for tag in row["requires"].split(";")
        if tag.strip()
    }

    assert used <= vocabulary, f"not in the vocabulary: {sorted(used - vocabulary)}"
    # And every class is actually used, so a definition cannot outlive the rows it was written for.
    assert vocabulary <= used, f"defined and unused: {sorted(vocabulary - used)}"


def test_a_reaction_supplies_what_its_own_forward_rule_needs() -> None:
    """The invariant that makes the conditions gate self-consistent.

    `conditions_allow` admits a candidate when `requires[candidate] <= provides[intended]`. A
    reaction must therefore pass its *own* test: the pot of a retro rule contains the reagents of
    the forward reaction that realises it, so `provides` is seeded from the `requires` of every
    forward row naming it.

    Without this a reaction could fail to be its own competitor -- `FWD MC 5` reducing an aldehyde
    would be excluded from the pot of the very reduction it describes -- and the sites channel,
    which probes a rule's own forward partners, would go silent wherever a condition was declared.
    """
    from tests.vertsy.bench_host_aostar import (
        TEMPLATE_FILES,
        load_forward_library,
        load_merged_library,
    )

    merged = load_merged_library(
        tuple(
            (name, TEMPLATE_FILES[name])
            for name in ("hartenfeller", "medchem", "protecting-groups")
        )
    )
    forward = load_forward_library(merged.sources)
    by_code = {
        source.columns["template_code"]: rule_id
        for rule_id, source in enumerate(merged.sources)
        if "template_code" in source.columns
    }

    unmet = []
    for forward_id, row in enumerate(forward_rows()):
        needs = forward.requires[forward_id]
        for code in row["probes"].split(";"):
            rule_id = by_code.get(code.strip())
            if rule_id is not None and not needs <= forward.conditions[rule_id]:
                unmet.append((row["forward_code"], code.strip()))

    assert unmet == [], f"a reaction that cannot run in its own pot: {unmet}"


def test_a_coupling_agent_reaction_cannot_shed_water() -> None:
    """The chemistry rule that caught a mechanical pass getting four rows wrong.

    Declaring by-products from the atom balance alone is safe for a condensation -- a benzimidazole
    really does expel water -- and wrong wherever a reagent carries the oxygen away. That is exactly
    what `coupling-agent` names: a carbodiimide becomes the urea, a Mitsunobu's phosphine becomes
    the oxide. Taking up the oxygen *is* the agent's job, so a row that both declares the agent and
    sheds water is describing two mechanisms at once.

    Four rows did. `FWD Cpl 1` claimed a peak at 18 for an amide coupling whose real by-product is
    dicyclohexylurea at 224, and `FWD H 34`/`35`/`36` the same for the Mitsunobu. Both now name
    their reagents and balance exactly, which a composition check alone would never have asked for.
    """
    offenders = [
        row["forward_code"]
        for row in forward_rows()
        if "coupling-agent" in row["requires"]
        and "[OH2]" in row["forward_template"].partition(">>")[2]
    ]

    # A coupling agent takes the oxygen, so a row declaring one cannot also shed water.
    assert offenders == []


def test_step_mixture_actually_names_peaks() -> None:
    """Nothing covered this, and that is why a shadowed variable emptied every report.

    `step_mixture` built its three groups in a loop whose variable was called `key` -- the same
    name as the keyer parameter it then handed to `expected_mixture`. So every step that classified
    even one candidate passed the string "same_site" where a callable belonged, `expected_mixture`
    raised `TypeError`, and a bare `except Exception` turned that into an empty mixture. An empty
    block reads as "not derived", which was indistinguishable from the truth.

    Asserted as "names at least the product and its by-product", not as an exact list: the peaks
    depend on the table and must be free to change. What must not change is that the function
    answers at all.
    """
    from tests.vertsy.bench_host_aostar import (
        TEMPLATE_FILES,
        MixtureReport,
        load_forward_library,
        load_merged_library,
        step_mixture,
    )
    from vesma.keyer import cmol_rdkit_smiles_keyer, short_ring_wl_keyer

    merged = load_merged_library(
        tuple(
            (name, TEMPLATE_FILES[name])
            for name in ("hartenfeller", "medchem", "protecting-groups")
        )
    )
    forward = load_forward_library(merged.sources)
    # The report list is the new contract and is asserted below rather than discarded: a mixture
    # block derived from no rules at all would otherwise pass this test by naming nothing.
    derived: list[MixtureReport] = []
    mixture = step_mixture(
        forward,
        short_ring_wl_keyer.encode,
        cmol_rdkit_smiles_keyer.encode,
        derived.append,
    )
    # An amide coupling: a rule with forward partners, so the paired group is not empty either.
    rule = next(
        rule_id
        for rule_id, source in enumerate(merged.sources)
        if source.columns.get("template_code") == "Cpl 1"
    )

    peaks = mixture("CC(=O)Nc1ccccc1", ("CC(=O)O", "Nc1ccccc1"), rule)

    assert peaks, "the mixture must name the peaks, not swallow the attempt"
    assert len(derived) == 1, derived
    assert derived[0].rules > 0, "a block derived from no forward rules proves nothing"
    kinds = {peak.kind for peak in peaks}
    assert "product" in kinds
    # And every peak carries a mass, which is the whole point of the report.
    assert all(peak.exact_mass > 0 for peak in peaks)


# Rows whose reagent is deliberately *not* a starting material, and the exact composition that
# therefore crosses the row: `(gained, lost)` on the row's own example. Every entry is a boundary
# of the chemistry model rather than a curation gap, and each is here so that it stays the boundary
# it is -- a row drifting to a different residual is a changed rule, not a tolerated imbalance.
IMPLICIT_REAGENTS: dict[str, tuple[str, str]] = {
    # **Empty, and meant to stay that way.** Every forward row now accounts for every atom it
    # touches, bar the three in `REDOX_HYDROGEN` whose reagent is hydrogen gas. An entry here is
    # a row that crosses a heavy atom, written as the `(gained, lost)` composition it crosses, and
    # `test_the_table_accounts_for_every_atom` compares this whole mapping against what the table
    # actually does -- so adding a row that cannot balance means writing its crossing down, and
    # closing one means deleting the entry.
    #
    # The list once held four rows and then fourteen, each with a reason that turned out to be
    # wrong. Nothing was ever unspellable: hydrazine, ammonia, hydroxylamine, magnesium metal,
    # boric acid, dinitrogen, peroxide, hypochlorite, the aluminates and borates, dioxygen and
    # SnCl3(OH) all parse, and always did.
    #
    # `FWD MC 3` was the last of them and the most instructive, because the first two answers were
    # both wrong. A nitro group is a six-electron reduction, so it needs three equivalents of a
    # two-electron reductant -- and written as `3 SnCl2 + 6 HCl -> 3 SnCl4 + 2 H2O` it balanced
    # and cost **32 seconds for one application**: six interchangeable HCl and three
    # interchangeable SnCl2 are 34,560 distinct embeddings, which is 3! x 6! permutations times
    # the two chlorine orderings inside each SnCl2. The matcher is not at fault there -- those are
    # genuinely distinct injective maps, and injectivity is not symmetry -- but only 4.4s of the
    # 32 was matching; the rest was building precursors for all 34,560 before the dedup threw them
    # away. (`max_mappings` would not have helped: it slices after enumeration, and only
    # `max_mappings=1` short-circuits.)
    #
    # The fix was chemistry, not plumbing. Tin(IV) in aqueous HCl is as much SnCl3(OH) as SnCl4,
    # and `3 SnCl2 + 3 HCl + H2O -> 3 SnCl3(OH)` needs half the hydrogen chloride: 288 embeddings,
    # 0.23s, balanced on heavy atoms, hydrogen and charge. And that cost is only paid where the
    # pattern matches -- measured at 0.0001s on a pot with no tin -- so it lands once per nitro
    # reduction rather than on every pot the probe visits.
}


# Redox rows, and which way their hydrogen moves. Hydrogen is an attached atom in this engine,
# never a node -- `[H][H]` occupies no configuration -- so a reduction cannot name the hydride that
# supplied its two hydrogens and an aromatisation cannot name the air that took them. A borohydride
# is no better: `[BH4-]` parses and the `[BH3-]` it would become does not.
#
# These are therefore balanced on *heavy atoms*, with the hydrogen difference being the reagent's
# own contribution rather than a hole in the account. "reductant" means the row gains hydrogen and
# "oxidant" that it loses it, and the test below requires that to agree with the row's `requires`
# column -- a mis-tagged redox row is then a failure rather than a silent inversion.
REDOX_HYDROGEN = {
    # Three rows, and one reason: **hydrogen gas is the reagent, and a lone hydrogen is not a
    # molecule here by design.** Hydrogen is an attached property of an atom in this engine, never
    # a node -- `[H][H]` has no declared configuration, checked -- and that is deliberate rather
    # than incidental. Allowing a bare hydrogen would mean deciding how to re-attach it during
    # construction and parsing, and canonicalisation would stop working. So an alkene
    # hydrogenation and the two hydrogenolyses cannot name what reduces them, and nothing here
    # should try.
    #
    # The four aerobic oxidations used to sit here too, on the claim that air could not be named.
    # `O=O` parses and is in `stock-common-reagents.smi`; what was actually missing was the
    # *path*. One O2 oxidising two substrates is a 2:1 ratio a one-substrate template cannot
    # state, but the two-electron reduction of O2 gives **peroxide**, and that is 1:1 -- so
    # `H 3`, `H 4`, `H 5` and `H 49` name the dioxygen and emit the H2O2 it becomes. See
    # `DECLARED_REAGENTS`.
    "FWD MC 79": "reductant",
    "FWD PG 15r": "reductant",
    "FWD PG 2r": "reductant",
}


@pytest.mark.parametrize(
    ("code", "direction"),
    sorted(REDOX_HYDROGEN.items()),
    ids=sorted(REDOX_HYDROGEN),
)
def test_a_redox_row_balances_on_heavy_atoms_and_moves_hydrogen_the_right_way(
    code: str, direction: str
) -> None:
    """The reagent's hydrogen is the reaction, so its absence is not an imbalance.

    Two things are asserted and the second is the one with teeth. Heavy atoms must balance exactly,
    which says the row accounts for every atom it could name. And the hydrogen must move in the
    direction the row's own condition tag claims: a `reductant` row gains it, an `oxidant` row
    loses it. Tag and template then cannot disagree about which way the reaction runs.
    """
    from collections import Counter

    from vesma.cmol import cmol_union
    from vesma.formula import composition, difference, heavy_only, hill_formula
    from vesma.keyer import cmol_rdkit_smiles_keyer

    row = next(one for one in forward_rows() if one["forward_code"] == code)
    # The tag is what the direction is checked against, so it has to be there.
    assert direction in row["requires"]
    pot = cmol_union([molecule(part) for part in row["example_precursors"].split(".")])
    wanted = cmol_rdkit_smiles_keyer.encode(molecule(row["example_product"]))
    outcome = next(
        one
        for one in generate_precursors(
            pot, parse_reaction_smarts(row["forward_template"])
        )
        if any(
            cmol_rdkit_smiles_keyer.encode(part) == wanted for part in one.precursors
        )
    )
    emitted: Counter[int] = Counter()
    for part in outcome.precursors:
        emitted += composition(part)

    heavy_lost, heavy_gained = difference(
        heavy_only(composition(pot)), heavy_only(emitted)
    )
    assert not hill_formula(heavy_lost) and not hill_formula(heavy_gained)

    lost, gained = difference(composition(pot), emitted)
    if direction == "reductant":
        assert hill_formula(gained) and not hill_formula(lost)
    else:
        assert hill_formula(lost) and not hill_formula(gained)


def test_the_table_accounts_for_every_atom() -> None:
    """Every row balances, or crosses hydrogen alone, or is listed -- and the lists are closed.

    The two registries used to be read-only pins: a row named in one was checked, and a row named
    in neither was checked by nothing. That is how `FWD PG 5r` kept a charge that did not balance
    while sitting *inside* a registry whose entry described only its composition, and how a dozen
    rows crossed heavy atoms with no test watching at all.

    So the partition is asserted both ways. A row crossing nothing needs no entry; a row crossing
    hydrogen only must declare its direction in `REDOX_HYDROGEN`; a row crossing any heavy atom
    must appear in `IMPLICIT_REAGENTS`. An entry with nothing left to excuse is a failure too,
    which is what makes closing one of these gaps show up here instead of going unnoticed.

    Charge has no exception list. A reagent ion has to leave as an ion, and after the hydride and
    fluoride rows named their water there is no row in the table for which that is not true.
    """
    from collections import Counter

    from vesma.cmol import cmol_union
    from vesma.formula import composition, difference, hill_formula, net_charge
    from vesma.keyer import cmol_rdkit_smiles_keyer

    balanced: set[str] = set()
    hydrogen_only: set[str] = set()
    crossings: dict[str, tuple[str, str]] = {}
    for row in forward_rows():
        code = row["forward_code"]
        pot = cmol_union(
            [molecule(part) for part in row["example_precursors"].split(".")]
        )
        wanted = cmol_rdkit_smiles_keyer.encode(molecule(row["example_product"]))
        outcome = next(
            one
            for one in generate_precursors(
                pot, parse_reaction_smarts(row["forward_template"])
            )
            if any(
                cmol_rdkit_smiles_keyer.encode(part) == wanted
                for part in one.precursors
            )
        )
        emitted: Counter[int] = Counter()
        charge = 0
        for part in outcome.precursors:
            emitted += composition(part)
            charge += net_charge(part)
        # A reagent ion leaves as an ion, in every row, with no exceptions listed.
        assert charge == net_charge(pot), code

        lost, gained = difference(composition(pot), emitted)
        moved = dict(list(lost.items()) + list(gained.items()))
        if any(z != 1 for z in moved):
            crossings[code] = (hill_formula(gained), hill_formula(lost))
        elif moved:
            hydrogen_only.add(code)
        else:
            balanced.add(code)

    assert crossings == IMPLICIT_REAGENTS
    assert hydrogen_only == set(REDOX_HYDROGEN)
    # The great majority of the table needs no excuse at all, which is the point of the exercise.
    assert len(balanced) > 2 * (len(crossings) + len(hydrogen_only))


# The four bands `typical_yield` may take, documented in `tests/data/templates/README.md`.
YIELD_BANDS = frozenset({0.90, 0.80, 0.65, 0.50})

# The two bands `severity` may take, documented in `tests/data/templates/README.md`.
SEVERITY_BANDS = frozenset({"", "destructive"})


def test_every_forward_row_carries_a_severity_from_the_two_band_vocabulary() -> None:
    """A controlled vocabulary, so a typo fails instead of reading as the default band.

    `destuctive` or `Destructive` would otherwise be silently demoted to "competing", which is the
    worst available failure: the row would keep describing a reaction that destroys the substrate
    while being counted among the ones that merely cost a separation, and nothing would say so.
    """
    bands = [(row["forward_code"], row["severity"]) for row in forward_rows()]

    unknown = [(code, band) for code, band in bands if band not in SEVERITY_BANDS]
    assert not unknown, f"not a severity band: {unknown}"
    # Blank is the default and most of the table takes it; the named band is the exception.
    destructive = [code for code, band in bands if band == "destructive"]
    assert len(destructive) < len(bands) // 10, (
        "a severity that fires on a tenth of the table is not a severity, it is a tax -- "
        f"{len(destructive)} of {len(bands)}"
    )


def test_a_destructive_row_states_the_reagent_that_destroys_the_group() -> None:
    """Two implications that hold, where the biconditional against `probes` does not.

    A destructive row claims "this reagent is in the pot and it eats a bystander group". So the
    reagent has to be a left-hand component -- a row matching on substrate alone would say nothing
    about the medium -- and the row has to name a `requires` class, because one requiring nothing
    competes in every pot. That is the failure the derived-conditions experiment measured, where
    162 of 190 rows required nothing and the gate kept 99.2% of the mixture it was asked to filter.
    """
    rows = [row for row in forward_rows() if row["severity"] == "destructive"]
    assert rows, "the vocabulary is pinned by a table that uses it"

    # A destructive row that is unimolecular names no reagent; one requiring nothing competes in
    # every pot. Both codes are reported together so one run names every row to fix.
    unimolecular = [
        row["forward_code"]
        for row in rows
        if len(top_level_pieces(row["forward_template"].split(">>")[0])) < 2
    ]
    unconditioned = [row["forward_code"] for row in rows if not row["requires"].strip()]
    assert not unimolecular and not unconditioned, (unimolecular, unconditioned)


def test_severity_is_not_derived_from_the_probes_column() -> None:
    """The independence is a fact about the table, so it is pinned rather than merely unasserted.

    Today every destructive row happens to have an empty `probes`, which makes a derivation look
    correct and would make an assertion look harmless. Both would be wrong, because the two columns
    make different claims: `probes` says "no retro rule is my inverse", `severity` says "this
    consumes the substrate". `FWD MC 88` is the counterexample already in the table -- it dehydrates
    a primary amide to a nitrile as somebody's *intended* reaction, with a partner, which is the
    same bond change `FWD DEG 5` and `FWD DEG 6` describe as a degradation. The same reaction is a
    synthesis when you asked for it and a degradation when SOCl2 does it to your amide on the way to
    an acid chloride.

    So neither direction is asserted. `destructive => probes == ""` would forbid a route that uses a
    degradation on purpose from ever pairing one; `probes == "" => destructive` would forbid an
    unpaired competitor that merely costs a separation.
    """
    rows = {row["forward_code"]: row for row in forward_rows()}

    intended = rows["FWD MC 88"]
    assert intended["probes"].strip(), "FWD MC 88 is paired"
    assert intended["severity"] == "", "FWD MC 88 is somebody's intended reaction"
    destructive = rows["FWD DEG 5"]
    assert destructive["severity"] == "destructive"
    # The same transformation, both ways round: amide to nitrile, intended and destructive.
    assert "nitrile" in intended["classification"]
    assert "nitrile" in destructive["classification"]


def test_a_destructive_rows_yield_is_read_as_its_degradation_efficiency() -> None:
    """Same four bands, read upward: how completely the reagent destroys the group.

    The ordering is the claim, as it is for the intended reactions. N-halogenation of a free amine
    by an N-halosuccinimide is the fast, complete one; protodeboronation needs the right pot and is
    not. Nothing reads these numbers today -- `yields` is keyed by retro rule id and built from
    `probes`, so an unpaired row contributes to no base -- which is exactly why the reading has to
    be written down before something starts reading it.
    """
    yielded = {
        row["forward_code"]: float(row["typical_yield"])
        for row in forward_rows()
        if row["severity"] == "destructive"
    }

    # One vocabulary, two readings.
    assert set(yielded.values()) <= YIELD_BANDS
    # N-halogenation of a free aliphatic amine by an N-halosuccinimide outruns protodeboronation.
    assert yielded["FWD DEG 3a"] > yielded["FWD DEG 1"]
    # And a primary amine is no easier to halogenate than a secondary one.
    assert yielded["FWD DEG 3a"] == yielded["FWD DEG 3b"]
    # SOCl2 and POCl3 dehydrate an amide equally well.
    assert yielded["FWD DEG 5"] == yielded["FWD DEG 6"]
    # Iodine is the weakest of the three halogens here: N-I is the weakest bond and the
    # N-iodoamine the least stable, so it destroys less completely than NBS or NCS.
    assert yielded["FWD DEG 7a"] < yielded["FWD DEG 3a"]


def test_every_forward_row_carries_a_yield_from_one_of_four_bands() -> None:
    """A controlled vocabulary, for the same reason `requires` has one.

    Bands rather than a number per row because a yield depends on the substrate far more than on
    the reaction class, so a curated `0.83` for "Suzuki" would claim resolution nobody has. What
    the bands carry is an *ordering*, and that is the signal a price can use. A row drifting to
    `0.83` is then a decision somebody has to make on purpose rather than a typo that reads as
    curation.
    """
    values = [row["typical_yield"] for row in forward_rows()]

    # Every row carries one; the catch-all is for retro rules nobody paired.
    assert all(values)
    assert {float(one) for one in values} <= YIELD_BANDS
    # Strictly inside (0, 1): `cost_of_factor` refuses 0 and a yield of 1 is a free step.
    assert all(0.0 < float(one) < 1.0 for one in values)


def test_the_bands_order_the_chemistry_the_way_a_chemist_would() -> None:
    """The ordering is the claim; the absolute values are priors and are not asserted here.

    Spot-checked at both ends, because a banding that put a Fischer indole above a Boc protection
    would be worse than no banding at all -- it would price confidently in the wrong direction.
    """
    yielded = {
        row["forward_code"]: float(row["typical_yield"]) for row in forward_rows()
    }

    # Proverbially clean, and the two that are quantitative in practice.
    assert yielded["FWD MC 72"] == 0.90, "CuAAC"
    assert yielded["FWD PG 1"] == 0.90, "Boc protection"
    # Robust couplings above substrate-sensitive ones.
    assert yielded["FWD H 31"] > yielded["FWD H 51"], "Suzuki over Buchwald"
    assert yielded["FWD H 31"] > yielded["FWD H 33"], "Suzuki over Negishi"
    # And the multi-bond condensations at the bottom.
    assert yielded["FWD H 23"] == 0.50, "Fischer indole"
    assert yielded["FWD H 8"] == 0.50, "Niementowski"
    assert yielded["FWD PG 1"] > yielded["FWD H 23"]


def test_a_rules_yield_is_the_best_of_its_forward_partners() -> None:
    """The max, not the mean: a rule's best known realisation is what it can do.

    `Hartenfeller 47` is probed by two amide formations, from a primary and a secondary amine. Both
    are the robust band, so the max is unambiguous there -- what the max is *for* is a rule one
    forward row runs cleanly and another runs badly, where averaging would invent a worse reaction
    than anybody would run.
    """
    from tests.vertsy.bench_host_aostar import (
        TEMPLATE_FILES,
        load_forward_library,
        load_merged_library,
    )

    merged = load_merged_library(
        tuple(
            (name, TEMPLATE_FILES[name])
            for name in ("hartenfeller", "medchem", "protecting-groups")
        )
    )
    forward = load_forward_library(merged.sources)
    by_code = {
        source.columns["template_code"]: rule_id
        for rule_id, source in enumerate(merged.sources)
        if "template_code" in source.columns
    }
    row_yield = {
        row["forward_code"]: float(row["typical_yield"]) for row in forward_rows()
    }

    for code, forward_codes in (
        ("Hartenfeller 47", ("FWD H 47", "FWD H 47b")),
        ("MC 75", ("FWD MC 75",)),
    ):
        rule = by_code[code]
        assert forward.yields[rule] == max(row_yield[one] for one in forward_codes)

    # Every curated value is one of the bands, and a rule nobody paired is simply absent.
    assert set(forward.yields.values()) <= YIELD_BANDS
    # Most rules have no forward partner at all.
    assert len(forward.yields) < len(merged.sources)


def test_the_two_price_readings_are_selected_from_one_place() -> None:
    """So the search and the report cannot drift into two readings of one measurement.

    `route_price` and the search both take their shape from `substrate_of`, which is the whole
    reason it exists as a function rather than as two `selective` calls. `steps` has to stay
    selectable because every number in `docs/note-forward-selectivity.md` was measured under it.
    """
    from tests.vertsy.bench_host_aostar import factor_of, factors_of, penalties_of
    from vertsy.costs import Factor, Multiplier

    yields = factors_of("shared:0.85:0.5", "none")
    assert yields is not None and set(yields) == {"sites"}
    # A factor is free when clean and never rises; that is the law its price depends on.
    curve: Factor = yields["sites"]
    assert curve(1) == 1.0 and curve(2) == pytest.approx(0.85)
    assert curve(99) == 0.5, "the floor holds"

    steps = penalties_of("steps", "none")
    assert steps is not None and set(steps) == {"sites"}
    multiplier: Multiplier = steps["sites"]
    assert multiplier(1) == 1.0 and multiplier(2) == 3.0

    assert factors_of("none", "none") is None
    with pytest.raises(ValueError, match="unknown yield shape"):
        factor_of("graded:0.1")


def test_a_rule_nobody_paired_falls_to_the_catch_all() -> None:
    """And the catch-all must not be kinder than the curated median, which is the measured trap.

    `shape_of` takes the minimum over an application's templates, so a default cheaper than the
    curated values makes every uncurated template systematically win -- a bias decided by which
    rules happen to be curated rather than by which chemistry is right.
    """
    from tests.vertsy.bench_host_aostar import YIELD_DEFAULT, yield_base
    from vertsy.costs import cost_of_factor

    base = yield_base({7: 0.90})

    assert base(7) == cost_of_factor(0.90)
    assert base(8) == cost_of_factor(YIELD_DEFAULT), "unpaired rules take the catch-all"

    # Costlier, not cheaper, than a curated rule at the median -- and the median is *measured*
    # from the table rather than written in as 0.80. Comparing against a literal left the trap
    # able to reopen in silence: were the bands to shift so that the median became 0.65, the
    # catch-all would no longer be at or below it and this test would have passed anyway.
    import statistics

    curated = sorted(float(row["typical_yield"]) for row in forward_rows())
    median = statistics.median(curated)
    assert YIELD_DEFAULT <= median
    assert base(8) >= cost_of_factor(median)

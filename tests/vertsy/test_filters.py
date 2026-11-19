"""A veto drops whole disconnections before the graph exists, and must drop nothing else."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from tests.datadirs import external_data_root
from vertsy.filters import (
    CLEAR,
    FORCES,
    UNEXAMINED,
    DeficitReport,
    Veto,
    VetoReport,
    host_screen,
    read_tags,
    read_vetoes,
    refused,
    unaccounted,
    validated,
    vetoed,
    vetoes_in_scope,
)
from vertsy.search import expansion, measured, object_mol_array, scored_with
from vertsy.selectivity import object_combine
from vesma import cmol_from_smiles
from vesma.cmol import cmol_union


class Fake:
    """A molecule that is only its heavy-atom count, which is all the stub verdict reads."""

    def __init__(self, size: int) -> None:
        self.size = size


def shrinking(products, precursors, hits) -> np.ndarray:
    """A stub `ReactionScore` condemning an application whose precursors weigh less.

    A driver for the grouping tests below and nothing more. It is deliberately **not** the
    mass veto that used to live in this module: that one is deleted, because 31% of the
    hand-written templates lose heavy atoms legitimately -- a reagent supplies them and
    nobody writes a reagent as a precursor. What the tests here need is any verdict that
    distinguishes one application from another, so they can check that `refused` groups,
    conjoins and gathers correctly; the arithmetic is incidental.
    """
    rows = np.asarray(hits, dtype=np.int64).reshape(-1, 4)
    sizes = np.array([mol.size for mol in precursors], dtype=np.int64)
    products_sizes = np.array([mol.size for mol in products], dtype=np.int64)
    groups, inverse = np.unique(rows[:, :3], axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    supplied = np.bincount(
        inverse, weights=sizes[rows[:, 3]], minlength=groups.shape[0]
    )
    short = products_sizes[groups[:, 0]] - supplied.astype(np.int64) > 0
    return np.where(short, 1, CLEAR).astype(np.int64)[inverse]


def matcher(batch, keys, hits, **scores):
    """A match double, lifted through `expansion` exactly as a real backend is.

    ``scores`` become per-hit-row reaction columns, which is what lets a test assert that a
    wrapper narrowing the hits narrows them too.
    """

    def match(mols):
        return (
            object_mol_array(batch),
            keys,
            np.array(hits, dtype=np.int64).reshape(-1, 4),
        )

    lifted = expansion(match)
    if not scores:
        return lifted

    def with_scores(mols):
        return scored_with(
            lifted(mols),
            reactions={
                name: np.asarray(column, dtype=np.int64)
                for name, column in scores.items()
            },
        )

    return with_scores


def judged(batch, keys, hits, **scores):
    """The canonical composition: measure a verdict, then refuse by it."""
    base = matcher(batch, keys, hits, **scores)
    return refused(
        measured(base, reactions={"vetoed": shrinking}),
        applications=("vetoed",),
    )


def test_a_mass_losing_disconnection_is_dropped_whole() -> None:
    """The measured failure: a big product from one small precursor, and the rest deleted.

    Two applications of one product. The first breaks a 10-atom molecule into 6 + 5, which gains an
    atom as a real retro step does. The second returns a single 2-atom precursor and loses eight --
    the ester-hydrolysis template that discards its alcohol half. Only the first survives, and it
    survives *entire*: half a conjunction would claim a molecule is made from some of its pieces.
    """
    product = Fake(10)
    precursors = [Fake(6), Fake(5), Fake(2)]
    hits = [
        (0, 100, 0, 0),  # app 0 -> 6 atoms
        (0, 100, 0, 1),  # app 0 -> 5 atoms, so 11 >= 10, kept
        (0, 200, 1, 2),  # app 1 -> 2 atoms alone, so 2 < 10, dropped
    ]
    keys = ["a", "b", "c"]
    found = judged(precursors, keys, hits)(object_mol_array([product]))

    assert found.hits.tolist() == [[0, 100, 0, 0], [0, 100, 0, 1]]
    # the batch is left whole; an unnamed precursor is never interned
    assert found.keys == keys


def test_a_verdict_that_condemns_nothing_changes_nothing() -> None:
    """Transparency is the property a wrapper has to earn before it is allowed to be a wrapper."""
    hits = [(0, 1, 0, 0), (0, 1, 0, 1), (1, 2, 1, 0)]
    base = matcher([Fake(3), Fake(4)], ["x", "y"], hits, verdict=[CLEAR, CLEAR, CLEAR])
    kept = refused(base, applications=("verdict",))

    batch = object_mol_array([Fake(5), Fake(5)])
    assert kept(batch).hits.tolist() == base(batch).hits.tolist()


def test_refusing_by_nothing_at_all_is_the_identity() -> None:
    """The empty conjunction, so a veto-free search is the unwrapped search by construction."""
    hits = [(0, 1, 0, 0), (1, 2, 1, 1)]
    base = matcher([Fake(3), Fake(4)], ["x", "y"], hits, outcomes=[5, 6])
    batch = object_mol_array([Fake(5), Fake(5)])

    before, after = base(batch), refused(base)(batch)
    assert after.hits.tolist() == before.hits.tolist()
    assert after.reaction_scores["outcomes"].tolist() == [5, 6]


def test_an_unexamined_row_is_not_a_refused_one() -> None:
    """Claiming a verdict nobody reached would be inventing one.

    ``UNEXAMINED`` exists so a silently skipped veto cannot read as chemistry that passed it, and
    that cuts both ways: it must not read as chemistry that *failed* either, or an arm unable to
    evaluate a pattern would quietly delete the applications it could not judge.
    """
    hits = [(0, 1, 0, 0), (0, 2, 1, 1)]
    base = matcher([Fake(3), Fake(4)], ["x", "y"], hits, verdict=[UNEXAMINED, 1])
    found = refused(base, applications=("verdict",))(object_mol_array([Fake(5)]))

    assert found.hits.tolist() == [[0, 1, 0, 0]]


def test_the_axes_combine_by_disjunction_with_no_winner_to_pick() -> None:
    """A refusal is a refusal, so a molecule verdict and a reaction verdict need no tie-break.

    App 0 is condemned on the application axis, app 1 on the molecule axis through the precursor it
    names, and app 2 by both at once. All three go, and nothing has to decide which objection was
    the more specific.
    """
    hits = [(0, 1, 0, 0), (0, 2, 1, 1), (0, 3, 2, 2), (0, 4, 3, 3)]
    base = matcher(
        [Fake(9), Fake(9), Fake(9), Fake(9)],
        ["a", "b", "c", "d"],
        hits,
        reaction_verdict=[1, CLEAR, 1, CLEAR],
    )

    def molecule_verdict(mols):
        return np.array([CLEAR, 1, 1, CLEAR], dtype=np.int64)

    kept = refused(
        measured(base, molecules={"molecule_verdict": molecule_verdict}),
        applications=("reaction_verdict",),
        molecules=("molecule_verdict",),
    )
    found = kept(object_mol_array([Fake(5)]))

    assert found.hits.tolist() == [[0, 4, 3, 3]]


def test_enforcing_a_verdict_nobody_measured_raises() -> None:
    """Absence is not consent: an unwired veto must not look like chemistry that passed."""
    base = matcher([Fake(3)], ["x"], [(0, 1, 0, 0)], outcomes=[2])
    kept = refused(base, applications=("deficit",))
    with pytest.raises(KeyError, match="no reaction column named 'deficit'"):
        kept(object_mol_array([Fake(5)]))

    molecular = refused(base, molecules=("too_strained",))
    with pytest.raises(KeyError, match="no molecule column named 'too_strained'"):
        molecular(object_mol_array([Fake(5)]))


def test_an_empty_match_stays_empty() -> None:
    found = judged([], [], [])(object_mol_array([Fake(4)]))
    assert found.hits.reshape(-1, 4).shape[0] == 0 and found.keys == []


def test_application_ids_are_only_unique_within_a_rule() -> None:
    """The bug that made this wrapper a silent no-op on real data.

    A raw application id is minted per rule, so id 0 recurs under every rule that matched. Grouping
    by ``(product, application)`` merges them, pools one rule's precursors with another's, and a
    mass-losing disconnection hides behind a well-behaved one sharing its id. On a real molecule that
    dropped the count of mass-losing applications from one to zero and the veto changed nothing at
    all.

    Here rule 100 splits the 10-atom product into 6 + 5 and rule 200 returns a single 2-atom
    precursor -- both under application id 0. Only the first may survive.
    """
    product = Fake(10)
    hits = [
        (0, 100, 0, 0),
        (0, 100, 0, 1),
        (0, 200, 0, 2),  # same app id, different rule, and it loses eight atoms
    ]
    found = judged([Fake(6), Fake(5), Fake(2)], ["a", "b", "c"], hits)(
        object_mol_array([product])
    )

    assert found.hits.tolist() == [[0, 100, 0, 0], [0, 100, 0, 1]]


def test_a_reaction_score_is_narrowed_with_the_hits_it_parallels() -> None:
    """Why measurements are carried per hit row rather than per application.

    Refusal reorders and drops hit rows, and a column on that axis follows the same index with no
    wrapper needing to know what an application id is. Here app 1 is dropped for losing mass, and its
    measurement has to go with it -- a column left at its original length would silently reassociate
    every surviving row's measurement with a different reaction.
    """
    product = Fake(10)
    hits = [(0, 100, 0, 0), (0, 100, 0, 1), (0, 200, 1, 2)]
    found = judged(
        [Fake(6), Fake(5), Fake(2)], ["a", "b", "c"], hits, outcomes=[3, 3, 7]
    )(object_mol_array([product]))

    assert found.hits.tolist() == [[0, 100, 0, 0], [0, 100, 0, 1]]
    assert found.reaction_scores["outcomes"].tolist() == [3, 3]  # 7 left with app 1
    assert found.reaction_scores["vetoed"].tolist() == [CLEAR, CLEAR]


def test_two_wrappers_measuring_one_name_refuse_rather_than_pick_a_winner() -> None:
    """Which of them wins is not a question with a good answer, so it is not answered."""
    base = matcher([Fake(5)], ["a"], [(0, 1, 0, 0)], outcomes=[2])
    with pytest.raises(ValueError, match="already measured"):
        scored_with(
            base(object_mol_array([Fake(5)])),
            reactions={"outcomes": np.array([9], dtype=np.int64)},
        )


# ------------------------------------------------------------------------- the veto table


VETOES = external_data_root() / "templates" / "templates-vetoes.csv"

# An ester reduction, a benzylic bromination and an amide reduction, in that order, so a rule index
# means something a scope can name.
RULES = ("MC 85a", "MC 84a", "MC 86", "Hartenfeller 99")
TAGS = external_data_root() / "templates" / "template-tags.csv"
TAG_VOCABULARY = external_data_root() / "templates" / "template-tag-vocabulary.csv"


def molecule(smiles: str):
    return cmol_from_smiles(smiles)[0]


def row(
    code: str,
    scope: str,
    on: str,
    sense: str,
    pattern: str,
    cost: str = "pattern",
    force: str = "refuse",
):
    return Veto(
        code=code,
        cost=cost,  # type: ignore[arg-type]
        scope=frozenset(entry for entry in scope.split(";") if entry),
        on=on,  # type: ignore[arg-type]
        force=force,  # type: ignore[arg-type]
        sense=sense,  # type: ignore[arg-type]
        pattern=pattern,
        rationale="a test row",
    )


def verdict(vetoes, hits, products, precursors, *, combine=object_combine(cmol_union)):
    """The veto column for one hand-built layer, on the host arm."""
    score = vetoed(
        vetoes,
        vetoes_in_scope(vetoes, RULES),
        screen=host_screen(vetoes),
        combine=combine,
    )
    return np.asarray(
        score(
            object_mol_array(products),
            object_mol_array(precursors),
            np.array(hits, dtype=np.int64).reshape(-1, 4),
        )
    ).tolist()


# --------------------------------------------------------------------------- the table


def test_the_shipped_table_loads_and_every_pattern_compiles() -> None:
    """A veto table is read eagerly so an unparsable row fails on load, not on a layer.

    Also the one place the SMARTS ceiling is enforced in practice: a recursive ``$()`` row would be
    refused by :mod:`vesma.parsing` right here, which is why no published alert set can be dropped in.
    """
    from vesma.gmol_builder import gmol_from_smarts

    vetoes = read_vetoes(VETOES)
    assert vetoes, "the shipped table should not be empty"
    assert len({veto.code for veto in vetoes}) == len(vetoes)
    for veto in vetoes:
        pattern, _ = gmol_from_smarts(veto.pattern)
        assert pattern.atom_count >= 1, veto.code
        assert veto.rationale


def test_a_recursive_pattern_is_refused_rather_than_silently_narrowed() -> None:
    """The ceiling, stated as a test so nobody imports PAINS and wonders why nothing fires."""
    from vesma.parsing.errors import UnsupportedSmartsFeatureError

    with pytest.raises(UnsupportedSmartsFeatureError, match=r"Recursive SMARTS"):
        host_screen([row("free-amine", "", "product", "forbid", "[NX3;!$(N-C=O)]")])


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("sense", "forbidden", "expected one of"),
        ("on", "precursor", "expected one of"),
        ("cost", "cheapish", "expected one of"),
    ],
)
def test_a_field_outside_its_vocabulary_is_refused(
    tmp_path, field: str, value: str, message: str
) -> None:
    """A veto misread does not degrade a score, it removes chemistry nobody asked it to remove.

    ``sense`` is the sharp one: a typo there inverts the row, so the table would refuse every
    application the row was written to permit.
    """
    columns = {
        "veto_code": "B 1",
        "cost": "pattern",
        "scope": "",
        "on": "product",
        "force": "refuse",
        "sense": "forbid",
        "pattern": "[B]",
        "rationale": "why",
    }
    columns[field] = value
    path = tmp_path / "vetoes.csv"
    path.write_text(
        ",".join(columns) + "\n" + ",".join(columns.values()) + "\n", newline=""
    )
    with pytest.raises(ValueError, match=message):
        read_vetoes(path)


def test_a_row_without_a_rationale_or_a_duplicate_code_is_refused(tmp_path) -> None:
    header = "veto_code,cost,scope,on,force,sense,pattern,rationale\n"
    nameless = tmp_path / "nameless.csv"
    nameless.write_text(
        header + "B 1,pattern,,product,refuse,forbid,[B],\n", newline=""
    )
    with pytest.raises(ValueError, match="no rationale"):
        read_vetoes(nameless)

    twice = tmp_path / "twice.csv"
    twice.write_text(
        header
        + "B 1,pattern,,product,refuse,forbid,[B],why\n"
        + "B 1,pattern,,product,refuse,forbid,[F],why\n",
        newline="",
    )
    with pytest.raises(ValueError, match="repeats veto_code"):
        read_vetoes(twice)


def test_scope_resolves_once_and_an_unscoped_row_applies_everywhere() -> None:
    """Three scopes, three behaviours: everything, one code, and every code a tag names.

    A ``tag:`` entry used to match nothing, on the reasoning that resolving it would mean inventing
    a vocabulary here. There is a curated vocabulary now, so it resolves -- and an *undefined* tag
    raises rather than matching nothing, which is the failure a veto table cannot afford.
    """
    vetoes = (
        row("B universal", "", "product", "forbid", "[B]"),
        row("B scoped", "MC 86", "product", "forbid", "[F]"),
        row("B tagged", "tag:hydride", "product", "forbid", "[I]"),
    )
    tags = {"hydride": frozenset({"MC 85a", "MC 86"})}
    in_scope = vetoes_in_scope(vetoes, RULES, tags)

    assert in_scope[RULES.index("MC 85a")] == (0, 2)
    assert in_scope[RULES.index("MC 86")] == (0, 1, 2)
    assert in_scope[RULES.index("Hartenfeller 99")] == (0,)


# ------------------------------------------------------------------- measuring a verdict


def test_a_veto_fires_on_the_group_its_scope_names_and_on_no_other() -> None:
    """The whole point of scoping, and the cheapest thing to get wrong.

    Two applications of one product, both yielding a precursor that carries an ester. Rule ``MC 86``
    is an amide reduction and B 2 says the hydride would reduce that ester too; rule
    ``Hartenfeller 99`` is in nobody's scope and must come back clear even though the very same
    molecule is in front of the very same pattern.
    """
    ester_amide = molecule("COC(=O)CCC(=O)NC")
    vetoes = (row("B 2", "MC 86", "precursors", "forbid", "[CX3](=[OX1])-[OX2]-[CX4]"),)
    hits = [
        (0, RULES.index("MC 86"), 0, 0),
        (0, RULES.index("Hartenfeller 99"), 0, 0),
    ]

    assert verdict(vetoes, hits, [molecule("CNCCCC(=O)OC")], [ester_amide]) == [
        1,
        CLEAR,
    ]


def test_a_precursor_without_the_forbidden_group_is_clear() -> None:
    """Clear and not merely unrefused: the veto ran, and had no objection."""
    vetoes = (row("B 2", "MC 86", "precursors", "forbid", "[CX3](=[OX1])-[OX2]-[CX4]"),)
    hits = [(0, RULES.index("MC 86"), 0, 0)]
    assert verdict(vetoes, hits, [molecule("CNCCC")], [molecule("CCCC(=O)NC")]) == [
        CLEAR
    ]


def test_require_is_the_dual_and_fires_on_absence() -> None:
    """One comparison, which is why the two senses are one field rather than two tables."""
    needs_boron = (row("B r", "MC 84a", "precursors", "require", "[B]"),)
    hits = [(0, RULES.index("MC 84a"), 0, 0)]

    assert verdict(needs_boron, hits, [molecule("CCC")], [molecule("Cc1ccccc1")]) == [1]
    assert verdict(
        needs_boron, hits, [molecule("CCC")], [molecule("Cc1ccc(B(O)O)cc1")]
    ) == [CLEAR]


def test_the_product_side_sees_what_survived_and_the_precursor_side_what_went_in() -> (
    None
):
    """Why ``on`` exists, and the rule of thumb for choosing it.

    A veto on the precursor union cannot tell the template's own substrate from an interferent --
    it has no atom mapping -- so a group the step *consumes* must not be named there. Read on the
    product instead: a leaving group still present in the product is one the step did not use.

    Here the benzylic bromide is consumed by the forward alkylation, so it is absent from the product
    and present in the precursors, and the two sides disagree. That disagreement is the mechanism.
    """
    benzylic = row("B 5", "MC 84a", "product", "forbid", "[c]-[CH2]-[Cl,Br,I]")
    on_precursors = row("B 5p", "MC 84a", "precursors", "forbid", "[c]-[CH2]-[Cl,Br,I]")
    hits = [(0, RULES.index("MC 84a"), 0, 0)]
    product, precursor = molecule("Cc1ccccc1"), molecule("BrCc1ccccc1")

    assert verdict((benzylic,), hits, [product], [precursor]) == [CLEAR]
    assert verdict((on_precursors,), hits, [product], [precursor]) == [1]


def test_a_multi_precursor_application_is_judged_on_the_union() -> None:
    """How "these two reagents react with each other instead" is said.

    Neither precursor carries an acyl chloride *and* an amine, so a pattern needing both is only
    found across the union -- which is what the pot contains. The single-component form of the same
    veto finds the acyl chloride in one of them, which is the form B 6 actually ships because the
    amine is present by construction.
    """
    both = row(
        "B pair", "MC 86", "precursors", "forbid", "[NH2;D1]-[CX4].[CX3](=[OX1])-[Cl]"
    )
    hits = [(0, RULES.index("MC 86"), 0, 0), (0, RULES.index("MC 86"), 0, 1)]
    precursors = [molecule("CCCN"), molecule("CCC(=O)Cl")]

    assert verdict((both,), hits, [molecule("CCC")], precursors) == [1, 1]
    # Each half alone, as two separate one-precursor applications, finds nothing.
    apart = [(0, RULES.index("MC 86"), 0, 0), (0, RULES.index("MC 86"), 1, 1)]
    assert verdict((both,), apart, [molecule("CCC")], precursors) == [CLEAR, CLEAR]


def test_a_multi_precursor_group_with_no_combiner_is_unexamined() -> None:
    """Not clear: an arm that cannot build a pot must not read as one that inspected it.

    The honest answer for a backend with no union operation, and the same distinction
    :data:`vertsy.costs.UNPROBED` draws. A single-precursor group needs no union, so it is still
    judged, which is why the two rows here differ.
    """
    vetoes = (row("B 6", "MC 86", "precursors", "forbid", "[CX3](=[OX1])-[Cl]"),)
    hits = [
        (0, RULES.index("MC 86"), 0, 0),  # two precursors, needs a union
        (0, RULES.index("MC 86"), 0, 1),
        (0, RULES.index("MC 86"), 1, 1),  # one precursor, a gather
    ]
    precursors = [molecule("CCCN"), molecule("CCC(=O)Cl")]

    assert verdict(vetoes, hits, [molecule("CCC")], precursors, combine=None) == [
        UNEXAMINED,
        UNEXAMINED,
        1,
    ]


def test_a_row_declaring_an_expensive_cost_does_not_run_in_the_inner_loop() -> None:
    """The declaration is what keeps an expensive veto out of expansion, not care.

    A geometry veto embeds a structure and is far too slow per application per layer. It is
    evaluated at replay, where molecules exist again; a layer pass asks only for `pattern` rows and
    so never reaches it. That the prohibition is a *declaration* is the whole safety property --
    nothing has to remember to leave it out.
    """
    later = (
        row(
            "B late",
            "MC 86",
            "precursors",
            "forbid",
            "[CX3](=[OX1])-[OX2]-[CX4]",
            cost="geometry",
        ),
    )
    hits = [(0, RULES.index("MC 86"), 0, 0)]
    assert verdict(later, hits, [molecule("CCC")], [molecule("COC(=O)CCC(=O)NC")]) == [
        CLEAR
    ]


def test_an_empty_layer_measures_nothing() -> None:
    vetoes = read_vetoes(VETOES)
    score = vetoed(
        vetoes,
        vetoes_in_scope(vetoes, RULES, read_tags(TAGS)),
        screen=host_screen(vetoes),
        combine=object_combine(cmol_union),
    )
    column = score(
        object_mol_array([]), object_mol_array([]), np.zeros((0, 4), dtype=np.int64)
    )
    assert np.asarray(column).shape[0] == 0


# ------------------------------------------------------------------- enforcing a verdict


def test_measuring_then_refusing_drops_the_vetoed_disconnection_whole() -> None:
    """The composition the design is for, end to end, with nothing mocked but the hit rows.

    App 0 reduces an amide on a molecule that also carries an ester, which B 2 forbids; app 1 is the
    same molecule under a rule nobody scoped. The first goes entire -- a disconnection is a
    conjunction -- and the second is untouched.
    """
    ester_amide = molecule("COC(=O)CCC(=O)NC")
    spare = molecule("CO")
    vetoes = (row("B 2", "MC 86", "precursors", "forbid", "[CX3](=[OX1])-[OX2]-[CX4]"),)

    def match(mols):
        return (
            object_mol_array([ester_amide, spare]),
            ["amide", "methanol"],
            np.array(
                [
                    (0, RULES.index("MC 86"), 0, 0),
                    (0, RULES.index("MC 86"), 0, 1),
                    (0, RULES.index("Hartenfeller 99"), 0, 0),
                ],
                dtype=np.int64,
            ),
        )

    pipeline = refused(
        measured(
            expansion(match),
            reactions={
                "vetoed": vetoed(
                    vetoes,
                    vetoes_in_scope(vetoes, RULES),
                    screen=host_screen(vetoes),
                    combine=object_combine(cmol_union),
                )
            },
        ),
        applications=("vetoed",),
    )
    found = pipeline(object_mol_array([molecule("CNCCCC(=O)OC")]))

    assert found.hits.tolist() == [[0, RULES.index("Hartenfeller 99"), 0, 0]]
    assert found.reaction_scores["vetoed"].tolist() == [CLEAR]


def test_an_end_to_end_layer_from_a_real_matcher_is_vetoed() -> None:
    """Nothing hand-built: a real retro matcher, its real hit rows, and a real veto.

    ``MC 86`` reduces an amide to an amine, so applied to this product it proposes the amide as the
    precursor -- and that amide carries a methyl ester, which the same hydride would reduce. The
    disconnection is real, the chemistry is wrong, and B 2 is what says so.
    """
    from vesma.cpu_matcher import CpuReactionMatcher
    from vesma.reaction import reaction_patterns_from_smarts

    amide_reduction = "[#6:3]-[CH2:1]-[N:2]>>[#6:3]-[C:1](=[O])-[N:2]"
    matcher = CpuReactionMatcher(
        reaction_patterns_from_smarts((amide_reduction,), cache=True)
    )
    target = molecule("COC(=O)CCCNC")
    precursors, hits = matcher.match([target], n_jobs=1)
    assert len(hits), "the amide reduction should find a disconnection here"

    vetoes = (row("B 2", "", "precursors", "forbid", "[CX3](=[OX1])-[OX2]-[CX4]"),)
    score = vetoed(
        vetoes,
        vetoes_in_scope(vetoes, ("MC 86",)),
        screen=host_screen(vetoes),
        combine=object_combine(cmol_union),
    )
    column = np.asarray(score(object_mol_array([target]), precursors, np.asarray(hits)))

    assert column.shape[0] == np.asarray(hits).reshape(-1, 4).shape[0]
    assert bool((column > CLEAR).any()), "every precursor here still carries the ester"


def test_a_table_with_no_rows_at_this_cost_is_the_identity() -> None:
    """A veto-free layer must be the unvetoed layer, measured or not."""
    hits = [(0, RULES.index("MC 86"), 0, 0)]
    assert verdict((), hits, [molecule("CCC")], [molecule("COC(=O)CC")]) == [CLEAR]


def test_scored_then_refused_leaves_other_columns_aligned() -> None:
    """A dropped row takes its measurements with it, on every channel at once."""
    ester_amide = molecule("COC(=O)CCC(=O)NC")
    vetoes = (row("B 2", "MC 86", "precursors", "forbid", "[CX3](=[OX1])-[OX2]-[CX4]"),)
    rows = np.array(
        [
            (0, RULES.index("MC 86"), 0, 0),
            (0, RULES.index("Hartenfeller 99"), 0, 1),
        ],
        dtype=np.int64,
    )

    def match(mols):
        return object_mol_array([ester_amide, molecule("CO")]), ["a", "b"], rows

    def with_sites(mols):
        return scored_with(
            expansion(match)(mols),
            reactions={"sites": np.array([7, 9], dtype=np.int64)},
        )

    pipeline = refused(
        measured(
            with_sites,
            reactions={
                "vetoed": vetoed(
                    vetoes,
                    vetoes_in_scope(vetoes, RULES),
                    screen=host_screen(vetoes),
                    combine=object_combine(cmol_union),
                )
            },
        ),
        applications=("vetoed",),
    )
    found = pipeline(object_mol_array([molecule("CNCCCC(=O)OC")]))

    assert found.hits.tolist() == [[0, RULES.index("Hartenfeller 99"), 0, 1]]
    assert found.reaction_scores["sites"].tolist() == [9]


# ------------------------------------------------------------------- measuring the table


def reported(vetoes, hits, products, precursors, *, combine=object_combine(cmol_union)):
    """The reports one hand-built layer emits, which is how a table is read before it refuses."""
    reports: list[VetoReport] = []
    score = vetoed(
        vetoes,
        vetoes_in_scope(vetoes, RULES),
        screen=host_screen(vetoes),
        combine=combine,
        report=reports.append,
    )
    score(
        object_mol_array(products),
        object_mol_array(precursors),
        np.array(hits, dtype=np.int64).reshape(-1, 4),
    )
    return reports


def test_a_report_counts_the_applications_each_row_condemned() -> None:
    """Per row, because a row is the unit of the decision a table faces."""
    vetoes = (
        row("B 2", "MC 86", "precursors", "forbid", "[CX3](=[OX1])-[OX2]-[CX4]"),
        row("B 1", "MC 84a", "precursors", "forbid", "[B]"),
    )
    hits = [
        (0, RULES.index("MC 86"), 0, 0),
        (1, RULES.index("MC 84a"), 0, 1),
        (2, RULES.index("Hartenfeller 99"), 0, 2),
    ]
    (report,) = reported(
        vetoes,
        hits,
        [molecule("CCC"), molecule("CCCC"), molecule("CCCCC")],
        [molecule("COC(=O)CC"), molecule("OB(O)c1ccccc1"), molecule("CCO")],
    )
    assert report.applications == 3
    assert report.condemned == 2
    assert report.unexamined == 0
    assert report.examined == 3
    assert report.fired == {"B 1": 1, "B 2": 1}


def test_two_rows_condemning_one_application_are_both_credited() -> None:
    """The overlap a table most wants to see, and the reason the counts do not sum.

    A row is deleted as redundant only when something else catches what it catches, so a tally that
    attributed each application to one row would hide exactly the fact the decision needs.
    """
    vetoes = (
        row("B 2", "", "precursors", "forbid", "[CX3](=[OX1])-[OX2]-[CX4]"),
        row("B 4", "", "precursors", "forbid", "[CX3](=[OX1])-[NX3]"),
    )
    (report,) = reported(
        vetoes,
        [(0, RULES.index("MC 86"), 0, 0)],
        [molecule("CCC")],
        [molecule("COC(=O)CCC(=O)NC")],
    )
    assert report.condemned == 1
    assert report.fired == {"B 2": 1, "B 4": 1}
    assert sum(report.fired.values()) > report.condemned


def test_an_unexamined_application_is_reported_as_neither_clear_nor_condemned() -> None:
    """The sentinel's whole purpose, carried into the report rather than rounded off."""
    (report,) = reported(
        (row("B 2", "", "precursors", "forbid", "[CX3](=[OX1])-[OX2]-[CX4]"),),
        [(0, RULES.index("MC 86"), 0, 0), (0, RULES.index("MC 86"), 0, 1)],
        [molecule("CCC")],
        [molecule("CCO"), molecule("CCN")],
        combine=None,
    )
    assert (report.applications, report.examined, report.unexamined) == (1, 0, 1)
    assert report.condemned == 0
    assert report.fired == {}


def test_a_pass_that_did_not_run_reports_nothing() -> None:
    """A row of zeroes would describe a pass that never happened."""
    assert reported((), [(0, 0, 0, 0)], [molecule("CCC")], [molecule("CCO")]) == []
    assert (
        reported(
            (row("B 2", "", "precursors", "forbid", "[B]"),),
            [],
            [molecule("CCC")],
            [molecule("CCO")],
        )
        == []
    )


def test_what_a_measuring_pass_condemns_is_what_a_refusing_pass_removes() -> None:
    """The claim that makes `measure` mode worth running: it predicts `refuse` mode exactly."""
    vetoes = (row("B 1", "", "precursors", "forbid", "[B]"),)
    products = [molecule("CCC")]
    precursors = [molecule("OB(O)c1ccccc1"), molecule("CCO")]
    hits = np.array(
        [
            (0, RULES.index("MC 84a"), 0, 0),
            (0, RULES.index("MC 84a"), 1, 1),
        ],
        dtype=np.int64,
    )
    reports: list[VetoReport] = []
    score = vetoed(
        vetoes,
        vetoes_in_scope(vetoes, RULES),
        screen=host_screen(vetoes),
        combine=object_combine(cmol_union),
        report=reports.append,
    )

    def match(mols):
        return object_mol_array(precursors), ["a", "b"], hits

    pipeline = refused(
        measured(expansion(match), reactions={"vetoed": score}),
        applications=("vetoed",),
    )
    found = pipeline(object_mol_array(products))

    (report,) = reports
    assert report.applications == 2
    assert report.condemned == 1
    assert len(found.hits) == len(hits) - report.condemned
    assert found.hits[:, 2].tolist() == [1]


# ----------------------------------------------------------- no mass loss, retro only


def deficits(hits, products, precursors, *, slack=0):
    reports: list[DeficitReport] = []
    score = unaccounted(cmol_sizes, slack=slack, report=reports.append)
    column = np.asarray(
        score(
            object_mol_array(products),
            object_mol_array(precursors),
            np.array(hits, dtype=np.int64).reshape(-1, 4),
        )
    ).tolist()
    return column, reports


def cmol_sizes(molecules):
    return np.array([molecule.atom_count for molecule in molecules], dtype=np.int64)


def test_precursors_that_do_not_account_for_the_product_are_condemned() -> None:
    """The defect: a step that makes atoms out of nothing, which a search hunts for.

    Chlorobenzene from benzene, with no chlorine source named. Six carbons in, seven heavy atoms
    out, so one atom came from nowhere -- and the precursor is smaller and nearer the catalogue than
    an honest one, which is exactly what a stock-distance heuristic rewards.
    """
    column, (report,) = deficits(
        [(0, 3, 0, 0)], [molecule("Clc1ccccc1")], [molecule("c1ccccc1")]
    )

    assert column == [1]
    assert (report.applications, report.unaccounted, report.atoms) == (1, 1, 1)
    assert report.by_rule == {3: 1}


def test_naming_the_reagent_clears_the_same_disconnection() -> None:
    """The repair this check exists to verify, and the whole reason it is worth running.

    The same chlorination with NCS named as a precursor. The precursors now weigh *more* than the
    product, which is not a defect: the succinimide leaves again as a by-product nobody writes down.
    """
    column, (report,) = deficits(
        [(0, 3, 0, 0), (0, 3, 0, 1)],
        [molecule("Clc1ccccc1")],
        [molecule("c1ccccc1"), molecule("ClN1C(=O)CCC1=O")],
    )

    assert column == [CLEAR, CLEAR]
    assert report.unaccounted == 0 and report.by_rule == {}


def test_an_application_is_judged_on_all_its_precursors_together() -> None:
    """An application is a conjunction, so what it supplies is the total of what it names."""
    column, _ = deficits(
        [(0, 1, 0, 0), (0, 1, 0, 1)],
        [molecule("CCCCCC")],
        [molecule("CCC"), molecule("CCC")],
    )

    assert column == [CLEAR, CLEAR]


def test_slack_tolerates_a_named_deficit_and_zero_is_the_default() -> None:
    """A template short by one has left a halogen implicit, which is the commonest form of this."""
    assert deficits([(0, 3, 0, 0)], [molecule("Clc1ccccc1")], [molecule("c1ccccc1")])[
        0
    ] == [1]
    assert deficits(
        [(0, 3, 0, 0)],
        [molecule("Clc1ccccc1")],
        [molecule("c1ccccc1")],
        slack=1,
    )[0] == [CLEAR]


def test_it_composes_with_the_pattern_table_and_neither_knows_the_other() -> None:
    """Two vetoes, two named columns, one `refused` -- and no wrapper mentions another veto.

    `refused` combines named columns by disjunction with no winner to pick, so adding a second kind
    of veto is naming it. That is the whole of the composition, and the reason the verdict lives on
    a score axis rather than in a wrapper that has to be nested in some order.
    """
    vetoes = (row("B 1", "", "precursors", "forbid", "[B]"),)
    boron = molecule("OB(O)c1ccccc1")
    invented = molecule("c1ccccc1")
    rows = np.array(
        [(0, RULES.index("MC 84a"), 0, 0), (1, RULES.index("MC 86"), 0, 1)],
        dtype=np.int64,
    )

    def match(mols):
        return object_mol_array([boron, invented]), ["a", "b"], rows

    pipeline = refused(
        measured(
            expansion(match),
            reactions={
                "vetoed": vetoed(
                    vetoes,
                    vetoes_in_scope(vetoes, RULES),
                    screen=host_screen(vetoes),
                    combine=object_combine(cmol_union),
                ),
                "unaccounted": unaccounted(cmol_sizes),
            },
        ),
        applications=("vetoed", "unaccounted"),
    )
    found = pipeline(object_mol_array([molecule("Clc1ccccc1"), molecule("Clc1ccccc1")]))

    # The first application is condemned by the pattern (boron), the second by the deficit
    # (benzene cannot account for chlorobenzene) -- so nothing survives, by two independent routes.
    assert not len(found.hits)


# ------------------------------------------------------- condition tags and what they scope


def test_every_tag_used_is_a_tag_the_vocabulary_defines() -> None:
    """A curated list is only curated if something checks it, so this is the check.

    The tags name *condition classes* -- "turned over by a palladium catalyst" -- and deliberately
    not functional groups, which are what a veto's `pattern` is for and what SMARTS-RX and the ZBH
    collections already name. This is the other axis, and the same per-rule attribute the
    `competes` predicate wants as its `requires` column.
    """
    used = set(read_tags(TAGS))
    with TAG_VOCABULARY.open(newline="") as handle:
        defined = {
            row["tag"].strip(): row["definition"] for row in csv.DictReader(handle)
        }

    assert used <= set(defined), f"undefined tags: {sorted(used - set(defined))}"
    assert all(defined.values()), "a tag without a definition is a tag nobody can audit"


def test_a_tag_scope_resolves_to_every_rule_the_tag_names() -> None:
    """One pattern, many rules -- the whole reason the tag exists."""
    vetoes = read_vetoes(VETOES)
    tags = read_tags(TAGS)
    codes = sorted(tags["palladium"])
    in_scope = vetoes_in_scope(vetoes, codes, tags)

    poisoning = {
        veto.code for veto in vetoes if veto.scope == frozenset({"tag:palladium"})
    }
    assert poisoning == {"B 7", "B 8"}
    # Every palladium rule, and the same two rows on each. 83 and not 89 since the six `iSCB 69`
    # rows were deleted as duplicates of `Hartenfeller 41a-f`, which carry the tag themselves.
    assert len(codes) == 83
    assert all(
        {vetoes[index].code for index in indices} >= poisoning for indices in in_scope
    )


def test_a_scope_naming_an_undefined_tag_raises_rather_than_matching_nothing() -> None:
    """The one failure a veto table cannot afford: looking curated and removing nothing."""
    vetoes = (row("B 99", "tag:nonesuch", "precursors", "forbid", "[SX2H1]"),)

    with pytest.raises(ValueError, match="no tag table defines"):
        vetoes_in_scope(vetoes, ("MC 1",), {"palladium": frozenset({"MC 1"})})


def test_a_thiol_poisons_palladium_and_a_thioether_does_not() -> None:
    """Both selectivity channels count products, and a poisoned catalyst makes none.

    So this is invisible to them by construction rather than for want of curation, which is the
    cleanest argument for the veto layer existing at all.
    """
    vetoes = read_vetoes(VETOES)
    screen = host_screen(vetoes)
    index = {veto.code: position for position, veto in enumerate(vetoes)}
    cases = [
        ("B 7", "Sc1ccccc1", True),  # a free thiol
        ("B 7", "CSCc1ccccc1", False),  # a thioether is tolerated
        ("B 8", "c1ccc(P(c2ccccc2)c2ccccc2)cc1", True),  # a free phosphine
        ("B 8", "CCOP(=O)(OCC)C", False),  # a phosphonate is a spectator
    ]
    molecules = [molecule(smiles) for _code, smiles, _want in cases]
    pairs = [(index[code], position) for position, (code, _s, _w) in enumerate(cases)]

    got = np.asarray(screen(molecules, pairs), dtype=bool).tolist()
    assert got == [want for _code, _smiles, want in cases]


def test_an_azine_is_too_electron_poor_for_an_electrophile() -> None:
    """Read on the product, because only that side says where the electrophile went.

    Three rows per electrophile because an aromatic-bond path has no variable length without
    recursive SMARTS, which no arm here allows. Pyrrole-type nitrogen is excluded for free: it
    activates rather than deactivates, and `[n]` in an indole is not ring-adjacent to the site in
    the way these patterns require.

    **Two shapes, for a chemical reason.** The ortho rows stay paths, because a carbon alpha to a
    pyridine-type nitrogen is electron-poor in a five-membered ring as much as in a six. The meta
    and para rows are six-membered *rings*, because those relations mean nothing outside one, and
    a path of aromatic bonds crosses a ring fusion into a neighbouring, often electron-rich ring.
    Every row's nitrogen is divalent, which is what separates an azine from its oxo tautomer.
    """
    vetoes = read_vetoes(VETOES)
    screen = host_screen(vetoes)
    azine = {veto.code for veto in vetoes if ":[nX2]" in veto.pattern}
    assert azine == {f"B {n}" for n in range(9, 18)}

    cases = [
        ("O=[N+]([O-])c1ccccc1", False),  # nitrobenzene is made this way
        ("O=[N+]([O-])c1cccnc1", True),  # 3-nitropyridine is not
        ("O=[N+]([O-])c1cnccn1", True),  # nor a nitropyrazine
        ("Clc1ccncc1", True),  # nor a chloroazine by halogenation
        ("O=Cc1ccccn1", True),  # nor a pyridine aldehyde by Vilsmeier
        ("O=[N+]([O-])c1cccc2ccccc12", False),  # naphthalene nitrates fine
        ("O=[N+]([O-])c1ccc2[nH]ccc2c1", False),  # pyrrole-type N activates
        ("O=[N+]([O-])c1ccccc1CCc1ccccn1", False),  # a distant pyridine is irrelevant
        # A pyridone is not an azine: its nitrogen carries a hydrogen and donates, so the ring is
        # activated like a phenol and 2-pyridone really does brominate at C-3 with NBS.
        ("Cc1cc(Cl)c(CN)c(=O)[nH]1", False),
        ("O=C(Nc1ccc2[nH]c(=O)c(Cl)c(Cl)c2c1)NC1CC1", False),
        ("O=[N+]([O-])c1cccc(=O)[nH]1", False),
        # Both chlorines here are on the *other* ring, which is a genuine azine, so this one does
        # fire -- the pyridone's donation is not where the electrophile has to land.
        ("O=c1ccc2c(Cl)c(Cl)ncc2[nH]1", True),
        # The halogen is on the electron-rich thiophene of a thienopyridine, three bonds from the
        # azine nitrogen. The path form condemned it; a ring cannot reach across the fusion.
        ("Cc1ccc2c(Cl)c(Br)sc2n1", False),
        # C-2 of a thiazole is as electron-poor as C-2 of a pyridine, which is why ortho is a path.
        ("Brc1nccs1", True),
        # C-4 of a pyrazole is not, and meta has nothing to say about a five-ring.
        ("Ic1cn[nH]c1", False),
        # The positions a diazine genuinely does halogenate at: 5-bromopyrimidine and 5-bromouracil
        # are made this way, and a second ring nitrogen is what spares them.
        ("Brc1cncnc1", False),
        ("O=c1[nH]cc(Br)c(=O)[nH]1", False),
        # While every diazine position alpha to a nitrogen is still caught.
        ("Clc1ncccn1", True),
        ("Clc1cccnn1", True),
    ]
    positions = {veto.code: index for index, veto in enumerate(vetoes)}
    molecules = [molecule(smiles) for smiles, _want in cases]
    pairs = [
        (positions[code], row) for row in range(len(cases)) for code in sorted(azine)
    ]
    hits = np.asarray(screen(molecules, pairs), dtype=bool).reshape(
        len(cases), len(azine)
    )

    assert hits.any(axis=1).tolist() == [want for _smiles, want in cases]


def test_a_group_the_medium_destroys_is_vetoed_not_measured() -> None:
    """The three rows added for faults the selectivity channels cannot reach, and why they cannot.

    Each channel counts distinct *products*. A Boc cleaved by the acid, an azide reduced by the
    phosphine, a Grignard quenched by a proton: none of these yields a product to count, and in
    every case the destroying species is the **medium or the catalyst and not a named precursor**,
    so there is nothing in the pot for either channel to see. That is the test for whether a fault
    belongs here rather than in a channel -- and it is why hydrogenation reducing a nitro group
    does *not* belong here, H2 being the reagent and so a `sites` question.
    """
    vetoes = read_vetoes(VETOES)
    screen = host_screen(vetoes)
    at = {veto.code: index for index, veto in enumerate(vetoes)}
    cases = [
        # Boc and tert-butyl ester go, a methyl ester and a Cbz stay.
        ("B 18", "CC(C)(C)OC(=O)N1CCCCC1", True),
        ("B 18", "CC(C)(C)OC(=O)CCc1ccccc1", True),
        ("B 18", "COC(=O)CCc1ccccc1", False),
        ("B 18", "O=C(OCc1ccccc1)N1CCCCC1", False),
        ("B 18", "CC(C)(C)OCc1ccccc1", False),
        # An organic azide is Staudingered; the azide anion is a reagent and a triazole is spent.
        ("B 19", "[N-]=[N+]=NCc1ccccc1", True),
        ("B 19", "[N-]=[N+]=[N-]", False),
        ("B 19", "c1cn(Cc2ccccc2)nn1", False),
        # Protic hydrogens quench; ethers, tertiary amines and ketones do not.
        ("B 20", "OC(=O)CCc1ccccc1", True),
        ("B 20", "COCCc1ccccc1", False),
        ("B 21", "NCCc1ccccc1", True),
        ("B 21", "CN(C)CCc1ccccc1", False),
        ("B 22", "SCCc1ccccc1", True),
        ("B 22", "CC(=O)CCc1ccccc1", False),
    ]
    molecules = [molecule(smiles) for _code, smiles, _want in cases]
    pairs = [(at[code], row) for row, (code, _s, _w) in enumerate(cases)]

    got = np.asarray(screen(molecules, pairs), dtype=bool).tolist()
    assert got == [want for _code, _smiles, want in cases]


def test_the_protic_disjunction_has_to_be_split_and_the_table_says_so() -> None:
    """Why B 20 to B 22 are three rows: the one-row form is refused, informatively.

    Worth pinning, because the obvious reading of the table is that three near-identical rows are
    an oversight. The one-row form ties element to connectivity, which vesma compiles to an atom
    predicate -- and the veto table refuses predicates until every arm evaluates them.
    """
    from vesma.gmol_builder import gmol_from_smarts
    from vesma.parsing.errors import UnsupportedSmartsFeatureError

    assert gmol_from_smarts("[OX2H1,SX2H1,NX3;H1,H2]")[0].predicates
    with pytest.raises(UnsupportedSmartsFeatureError, match="tied together"):
        host_screen([row("B 20", "", "product", "forbid", "[OX2H1,SX2H1,NX3;H1,H2]")])

    # Each half is fine on its own.
    for pattern in ("[OX2H1]", "[NX3;H1,H2]", "[SX2H1]"):
        assert gmol_from_smarts(pattern)[0] is not None


def test_force_splits_one_table_into_a_wall_column_and_a_price_column() -> None:
    """A fired count cannot mean "remove this" and "charge for this" at once.

    So the force selects rows exactly as the cost does, and the table yields two columns. Neither
    needs new machinery to be consumed: `refused` reads the first, and `vertsy.costs.selective`
    reads the second as a channel like any other, its `penalties` being keyed by channel name.
    """
    vetoes = (
        row("B wall", "", "precursors", "forbid", "[B]"),
        row("B cost", "", "precursors", "forbid", "[OX2H1]", force="price"),
    )
    in_scope = vetoes_in_scope(vetoes, RULES)
    hits = np.array([(0, 0, 0, 0), (1, 0, 0, 1)], dtype=np.int64)
    products = object_mol_array([molecule("CCC"), molecule("CCCC")])
    # A pinacol boronate rather than a boronic acid: the acid carries two free hydroxyls, so it
    # would legitimately trip the price row as well and the test would prove nothing about force.
    precursors = object_mol_array(
        [molecule("CC1(C)OB(OC1(C)C)c1ccccc1"), molecule("OCCc1ccccc1")]
    )

    columns = {
        force: np.asarray(
            vetoed(
                vetoes,
                in_scope,
                screen=host_screen(vetoes),
                force=force,
                combine=object_combine(cmol_union),
            )(products, precursors, hits)
        ).tolist()
        for force in ("refuse", "price")
    }

    # The boron application is condemned by the wall column only, the alcohol by the price column
    # only, and neither row leaks into the other's answer.
    assert columns["refuse"] == [1, CLEAR]
    assert columns["price"] == [CLEAR, 1]


def test_the_shipped_table_declares_a_force_for_every_row() -> None:
    """And the ones declared `price` are the ones whose rationale says so.

    `B 20` to `B 22` forbid a protic hydrogen in a Grignard, which costs an equivalent of reagent
    rather than stopping the reaction -- excess overcomes it. A table that could only say "wall"
    would have to either forbid chemistry that works or stay silent about chemistry that is worse
    than the alternative.
    """
    vetoes = read_vetoes(VETOES)

    assert {veto.code for veto in vetoes if veto.force == "price"} == {
        "B 20",
        "B 21",
        "B 22",
    }
    assert all(veto.force in FORCES for veto in vetoes)


# --------------------------------------------------- validating at replay, where molecules are


def test_validating_a_replay_drops_the_condemned_application_and_counts_it() -> None:
    """The same predicate as `refused`, at the other end of the pipeline.

    Replay is where an expensive veto belongs, because it is the only place molecules exist again.
    Only the shapes differ: a layer match answers an `Expansion`, a replay match answers
    `(precursors, keys, hits)`, and the verdict is the same `ReactionScore` either way.

    Dropping the row is how a route is refused -- replay cannot reproduce a step that is not there,
    so it reports a mismatch and the caller takes the next candidate.
    """
    vetoes = (row("B 1", "", "precursors", "forbid", "[B]"),)
    verdict = vetoed(
        vetoes,
        vetoes_in_scope(vetoes, RULES),
        screen=host_screen(vetoes),
        combine=object_combine(cmol_union),
    )
    precursors = [molecule("OB(O)c1ccccc1"), molecule("CCO")]
    hits = np.array([(0, 0, 0, 0), (0, 0, 1, 1)], dtype=np.int64)

    def match(molecules, pairs):
        del pairs
        return object_mol_array(precursors), ["a", "b"], hits

    counted: list[dict] = []
    wrapped = validated(match, verdict=verdict, report=counted.append)
    _precursors, _keys, kept = wrapped(object_mol_array([molecule("CCC")]), None)

    # The boron application is gone, the other survives, and the refusal was counted.
    assert np.asarray(kept).tolist() == [[0, 0, 1, 1]]
    assert counted == [{"refused": 1}]


def test_validating_a_replay_with_nothing_to_refuse_is_the_identity() -> None:
    """An observer that changes nothing must change nothing, including the hit array it returns."""
    vetoes = (row("B 1", "", "precursors", "forbid", "[B]"),)
    verdict = vetoed(
        vetoes,
        vetoes_in_scope(vetoes, RULES),
        screen=host_screen(vetoes),
        combine=object_combine(cmol_union),
    )
    hits = np.array([(0, 0, 0, 0)], dtype=np.int64)

    def match(molecules, pairs):
        del pairs
        return object_mol_array([molecule("CCO")]), ["a"], hits

    counted: list[dict] = []
    wrapped = validated(match, verdict=verdict, report=counted.append)
    _precursors, keys, kept = wrapped(object_mol_array([molecule("CCC")]), None)

    assert np.asarray(kept).tolist() == hits.tolist()
    assert keys == ["a"]
    assert counted == []


def test_a_species_that_does_not_exist_as_drawn_is_refused_everywhere() -> None:
    """A different kind of fault from every scoped row: the molecule, not the reaction.

    So these are **unscoped** -- no reaction may hand such a species on -- and both selectivity
    channels are blind to them for the usual reason: each counts distinct products, and this *is* a
    product, just not of the structure claimed.

    Each pattern's decoy is the stable relative that must survive, and in three cases that decoy is
    what the one-row form of the pattern got wrong.
    """
    vetoes = read_vetoes(VETOES)
    screen = host_screen(vetoes)
    at = {veto.code: index for index, veto in enumerate(vetoes)}
    cases = [
        # A diazene with an N-H goes to arene + N2; an azo dye has none and must survive.
        ("B 23", "N=N", True),
        ("B 23", "N=Nc1ccccc1", True),
        ("B 23", "c1ccc(N=Nc2ccccc2)cc1", False),
        ("B 23", "CC(=O)N=Nc1ccccc1", False),
        # Carbonic and carbamic monoacids decarboxylate; the di-ester and the carbamate do not.
        ("B 24", "CCOC(=O)O", True),
        ("B 24", "CCOC(=O)OCC", False),
        ("B 25", "NC(=O)O", True),
        ("B 25", "CC(C)(C)OC(=O)NC", False),
        # A beta-halo amine closes to the aziridinium. A nitrogen mustard is tertiary, and a
        # beta-halo *amide* is stable -- which is why both neighbours must be sp3 carbon.
        ("B 26", "NCCCl", True),
        ("B 26", "NC(=O)CCCl", False),
        ("B 27", "CNCCCl", True),
        ("B 27", "CC(=O)NCCCl", False),
        ("B 27", "ClCCN(CCCl)c1ccccc1", False),
        # A gamma- or delta-hydroxy aldehyde is the lactol. Glucose drawn cyclic already is one.
        ("B 28", "O=CCCCO", True),
        ("B 28", "OC1OC(CO)C(O)C(O)C1O", False),
        ("B 29", "O=CCCCCO", True),
        # Seven-membered and larger are left alone: the equilibrium no longer decides.
        ("B 29", "O=CCCCCCO", False),
        # An unstabilised enol is the keto compound's tautomer, not a second catalogue entry.
        # Every enol the catalogue actually sells is stabilised from one side or the other, and
        # each decoy below is one the measured pattern had to be narrowed to spare.
        ("B 32", "C=C(O)CC1CO1", True),
        ("B 33", "CC(O)=CC1CO1", True),
        # A phenol is aromatic, so no enol row can reach it.
        ("B 32", "Oc1ccccc1", False),
        ("B 33", "Oc1ccccc1", False),
        # Stabilised by a carbonyl on the far carbon: a fluorinated 1,3-dione and a cyclic one.
        ("B 33", "O=C(/C=C(\\O)C(F)(F)F)C(F)(F)F", False),
        ("B 33", "CC1CCC(O)=CC1=O", False),
        # And by a nitrile, which stabilises exactly as a carbonyl does.
        ("B 33", "N#CC=C(O)C(F)F", False),
        # An oxirene is a matrix species; cyclopropene, the all-carbon analogue, is a compound.
        ("B 34", "CC(O)CC1=CO1", True),
        ("B 34", "CC(O)CC1CO1", False),
        ("B 34", "C1=CC1", False),
    ]
    molecules = [molecule(smiles) for _code, smiles, _want in cases]
    pairs = [(at[code], row) for row, (code, _s, _w) in enumerate(cases)]

    got = np.asarray(screen(molecules, pairs), dtype=bool).tolist()
    assert got == [want for _code, _smiles, want in cases]


def test_the_non_isolable_rows_are_unscoped_so_they_reach_every_rule() -> None:
    """The fault is a property of the molecule, so no template is exempt from it."""
    vetoes = read_vetoes(VETOES)
    family = {f"B {n}" for n in (*range(23, 30), 32, 33, 34)}

    assert all(not veto.scope for veto in vetoes if veto.code in family)
    # A rule no scoped row names still carries every one of them.
    reached = {
        vetoes[index].code
        for index in vetoes_in_scope(vetoes, ("MC 5",), read_tags(TAGS))[0]
    }
    assert reached == family


def test_a_forward_rule_too_permissive_is_corrected_on_the_product_side() -> None:
    """The two rows found by watching a raised price pick the *wrong* route.

    A too-permissive forward rule does not merely fail to catch a bad step -- it reports one clean
    product, the cheapest verdict a step can get, and so actively recommends it. Raising the
    selectivity price amplifies that, because the correct alternative is charged for the mixture it
    honestly makes. Narrowing the forward pattern instead would need "an alkyl halide bearing no
    electrophile", which is recursive and unavailable on every arm, so the correction is a veto read
    on the product where the surviving group is unambiguous.
    """
    vetoes = read_vetoes(VETOES)
    screen = host_screen(vetoes)
    at = {veto.code: index for index, veto in enumerate(vetoes)}
    cases = [
        # No organometallic forms on a halide carrying an epoxide -- the metal opens the ring.
        # Oxetane and THF are not matched: three-membered is what makes it win.
        ("B 30", "CC(O)CC1CO1", True),
        ("B 30", "ClCC1CO1", True),
        ("B 30", "C1COC1", False),
        ("B 30", "C1CCOC1", False),
        # An enolisable aldehyde cannot be in the flask when an alkoxide is made. Benzaldehyde has
        # no alpha hydrogen, which is why the alpha carbon is named and not the aldehyde alone.
        ("B 31", "C#CCCOCCCCCC=O", True),
        ("B 31", "O=CCCCCCO", True),
        ("B 31", "O=Cc1ccccc1", False),
        ("B 31", "CC(=O)C", False),
    ]
    molecules = [molecule(smiles) for _code, smiles, _want in cases]
    pairs = [(at[code], row) for row, (code, _s, _w) in enumerate(cases)]

    got = np.asarray(screen(molecules, pairs), dtype=bool).tolist()
    assert got == [want for _code, _smiles, want in cases]


def test_both_precision_rows_read_the_product_and_are_scoped_to_one_family() -> None:
    """Read on the product because the precursor union cannot tell substrate from interferent.

    `B 30`'s scope is the Grignard family and `B 31`'s the Williamson family -- these are claims
    about what a particular reaction's conditions destroy, not about a molecule, so unlike the
    non-isolable rows they must be scoped.
    """
    vetoes = {veto.code: veto for veto in read_vetoes(VETOES)}

    for code in ("B 30", "B 31"):
        assert vetoes[code].on == "product"
        assert vetoes[code].force == "refuse"
        assert vetoes[code].scope, f"{code} must be scoped to the reaction it is about"
    assert all("Hartenfeller 4" in entry for entry in vetoes["B 30"].scope)
    # Six and not twelve: the `iSCB 63a-f` half of the scope was deleted as duplicates of
    # `Hartenfeller 29a-f`, which the row still names.
    assert len(vetoes["B 31"].scope) == 6


def test_the_protic_quench_rows_reach_an_aryllithium_through_its_tag() -> None:
    """The second rule family to need one pattern is what earns a tag.

    B 20 to B 22 named six Grignard codes explicitly. Directed metalation needs exactly the same
    three patterns -- any protic hydrogen quenches an aryllithium, harder than it quenches a
    Grignard -- so the membership moved beside the reactions, where a third family can join it
    without touching the veto table.
    """
    vetoes = read_vetoes(VETOES)
    tags = read_tags(TAGS)
    protic = {"B 20", "B 21", "B 22"}

    assert {veto.scope for veto in vetoes if veto.code in protic} == {
        frozenset({"tag:carbanion"})
    }
    for code in ("MC 97a", "MC 97b", "MC 97c", "Hartenfeller 45a"):
        reached = {
            vetoes[index].code for index in vetoes_in_scope(vetoes, (code,), tags)[0]
        }
        assert protic <= reached, code
    # And a reaction with no carbanion in it is still spared them.
    untagged = {
        vetoes[index].code for index in vetoes_in_scope(vetoes, ("MC 5",), tags)[0]
    }
    assert not (protic & untagged)

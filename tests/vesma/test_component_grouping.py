"""Zero-level grouping: how a pattern says which of its parts are separate molecules.

A dot cannot say it. In a query a dot is *no constraint at all* -- `bonds.bond_kind_compatible`
treats flag `0` as "no opinion" -- so `CCC.NNN` matches `NNNXXXCCC`, one molecule, and the halves
may even be adjacent. Ring closures also bond straight across a dot: `C1.C12.C2` is a spelling of
`CCC`. And dots are wanted exactly as they are, for the intramolecular rules that use one to wire
two ends of a single molecule together.

So the statement has to be written, and these are the three forms:

===============  ======================================  ===============================
written          means                                   groups
===============  ======================================  ===============================
`A.B`            unspecified -- today's behaviour         all zero
`(A).(B)`        two molecules                            different, non-zero
`(A.B)`          one molecule, two fragments of it        the same, non-zero
===============  ======================================  ===============================

The middle column is the point: a group both **separates and binds**, so `(A.B)` is a statement
rather than a no-op, and an intramolecular rule can finally make it.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from tests.datadirs import external_data_root
from vertsy.selectivity import template_arities
from vesma import (
    cmol_from_smiles,
    gmol_from_cmol,
    gmol_from_smarts,
    gmol_from_smiles,
    reaction_patterns_from_smarts,
)
from vesma.connected_components import grouping_agrees_with_components
from vesma.cpu_matcher import find_matches, generate_precursors, has_match
from vesma.keyer import cmol_rdkit_smiles_keyer as rdkit_key
from vesma.parsing import parse_smarts, parse_smiles
from vesma.parsing.errors import SmartsError
from vesma.parsing.grammar_custom import parse_graph_steps
from vesma.parsing.ir import (
    AtomStep,
    ComponentCloseStep,
    ComponentOpenStep,
    LParenStep,
    RParenStep,
)
from vesma.reaction import precursors_from_mapping_result
from vesma.rings import RingDescriptors


def groups(text: str) -> list[int]:
    return [int(group) for group in parse_smarts(text).groups]


def bonded(text: str) -> list[tuple[int, int]]:
    mol = parse_smarts(text)
    return [
        (left, right)
        for left in range(mol.atom_count)
        for right in range(left + 1, mol.atom_count)
        if mol.bonds[left, right]
    ]


def test_a_parenthesis_after_an_atom_is_a_branch_and_one_before_is_a_group() -> None:
    """Only position distinguishes them, and the parser is the only place that can see it.

    A dot reaches the builder as `AtomStep.leading_bond == 0` and nothing else, so `C(N)` and
    `C.(N)` would arrive indistinguishable if the decision were left any later.
    """
    assert [type(step) for step in parse_graph_steps("C(N)")] == [
        AtomStep,
        LParenStep,
        AtomStep,
        RParenStep,
    ]
    assert [type(step) for step in parse_graph_steps("C.(N)")] == [
        AtomStep,
        ComponentOpenStep,
        AtomStep,
        ComponentCloseStep,
    ]


def test_the_three_forms_say_three_different_things() -> None:
    """Separate, same, and unspecified -- the whole feature in one assertion."""
    assert groups("(C).(N)") == [1, 2], "two molecules"
    assert groups("(C.N)") == [1, 1], "one molecule, two fragments"
    assert groups("C.N") == [0, 0], "no statement"


def test_an_ungrouped_pattern_is_untouched() -> None:
    """The reason 471 retro and 184 forward rules keep working without being edited.

    Zero is not a group; it constrains nothing. A library that never writes a parenthesis gets
    exactly the behaviour it has today.
    """
    for text in ("CCC", "C.N", "C(N)O", "c1ccccc1", "[C:1]-[O:2]-[C:3]"):
        assert set(groups(text)) == {0}, text


def test_a_ring_closure_still_bonds_across_a_dot() -> None:
    """`C1.C12.C2` is a spelling of `CCC`, which is why a dot cannot imply separate molecules.

    The guard on the whole premise: if this ever stopped being one molecule, reading a dot as a
    separation would have become tempting again.
    """
    assert groups("C1.C12.C2") == [0, 0, 0]
    assert bonded("C1.C12.C2") == [(0, 1), (1, 2)]


def test_nothing_bonds_across_a_group_boundary() -> None:
    """A group starts a fragment of its own, so `(A).(B)` cannot quietly join A to B.

    Without that, the first atom of the second group would bond to the last atom of the first the
    way `A(B)` does, and the pattern would mean the opposite of what it says.
    """
    assert bonded("(CC).(N)") == [(0, 1)], "inside the first group only"
    assert bonded("(C).(N)") == []
    assert bonded("(C.N)") == [], "a dot inside a group is still no bond"


def test_a_group_survives_the_hydrogen_fold() -> None:
    """Explicit hydrogens are folded onto their neighbour, which reindexes every per-atom array.

    This is why the groups live on the *molecule* and are a required field: `reading.py` rebuilds
    atoms twice and reindexes them once, and a defaulted field on the atom would be dropped there
    without a word. Three written atoms become two, and the groups must follow.
    """
    mol = parse_smiles("(C([H])).(N)")

    assert mol.atom_count == 2
    assert [int(group) for group in mol.groups] == [1, 2]


def test_an_atom_map_and_a_group_coexist() -> None:
    """They are different annotations of the same atom and neither displaces the other."""
    mol = parse_smarts("([CH3:1]).([NH2:2])")

    assert [int(group) for group in mol.groups] == [1, 2]
    assert [atom.atom_map for atom in mol.atoms] == [1, 2]


def test_a_group_inside_a_group_is_refused() -> None:
    """One statement per part. Nesting would be a second, and the model has no reading for it."""
    with pytest.raises(SmartsError):
        parse_smarts("((C))")


def test_an_unclosed_group_is_refused() -> None:
    with pytest.raises(SmartsError):
        parse_smarts("(C.N")


def test_the_group_reaches_the_compiled_pattern() -> None:
    """Beside `atom_map`, which is the model: a per-atom annotation that is not a mask.

    It cannot be a mask. `atom_masks_overlap` ANDs every word from the fourth upward against the
    molecule's own value for that atom, and a group id has no molecule counterpart -- it would be
    ANDed against zero and no atom would ever match again.
    """
    from vesma import gmol_from_smarts

    pattern, _info = gmol_from_smarts("([C:1]-[Cl]).([N:2])")

    assert [int(group) for group in pattern.atom_group] == [1, 1, 2]
    assert [int(number) for number in pattern.atom_map] == [1, 0, 2]


def test_a_molecule_compiled_for_matching_carries_no_grouping() -> None:
    """Grouping is what a query says about the molecules its parts come from.

    The target is the thing being matched against, not the thing making the claim -- so a molecule
    read from SMILES has no groups even when it is disconnected.
    """
    from vesma import cmol_from_smiles, gmol_from_cmol

    pattern = gmol_from_cmol(cmol_from_smiles("CC.N")[0])

    assert {int(group) for group in pattern.atom_group} == {0}


def test_a_reaction_pattern_keeps_its_groups_on_both_sides() -> None:
    """A retro rule states its intent on the right: `(B).(C)` separate, `(B.C)` one molecule."""
    from vesma import reaction_patterns_from_smarts

    separate, together = reaction_patterns_from_smarts(
        (
            "[C:1]-[N:2]>>([C:1]-[Cl]).([N:2])",
            "[C:1]-[N:2]>>([C:1]-[Cl].[N:2])",
        )
    )

    assert [int(g) for g in separate.rhs.atom_group] == [1, 1, 2]
    assert [int(g) for g in together.rhs.atom_group] == [1, 1, 1]
    # The left-hand side of a retro rule is one molecule and says nothing.
    assert {int(g) for g in separate.lhs.atom_group} == {0}


def test_a_group_is_enforced_at_match_time() -> None:
    """The point of the whole exercise: `(A).(B)` does not match one molecule.

    `CCC.NNN` matching `NNNOCCC` is not a bug to be fixed -- a query dot is no constraint, and
    rules rely on it. The fix is that there is now a way to say otherwise.
    """
    for pattern, target, expected in (
        ("CCC.NNN", "NNNCCC", True),
        ("CCC.NNN", "NNNOCCC", True),
        ("(CCC).(NNN)", "NNNCCC", False),
        ("(CCC).(NNN)", "NNNOCCC", False),
        ("(CCC).(NNN)", "CCC.NNN", True),
        ("(CCC.NNN)", "NNNCCC", True),
        ("(CCC.NNN)", "CCC.NNN", False),
        ("C1.C12.C2", "CCC", True),
    ):
        query, _info = gmol_from_smarts(pattern)
        molecule, _info = gmol_from_smiles(target)
        assert has_match(query, molecule) is expected, f"{pattern} vs {target}"


def test_an_ungrouped_atom_beside_a_group_constrains_nothing() -> None:
    """The permissive reading of a half-written pattern, recorded so it is a choice and not an
    accident. `C.(N)` says where the N must come from and nothing about the C."""
    query, _info = gmol_from_smarts("C.(N)")
    molecule, _info = gmol_from_smiles("CN")

    assert has_match(query, molecule)


def test_grouping_agrees_with_components_reads_both_relations() -> None:
    """Same group, same molecule; different groups, different molecules -- and `0` neither."""
    assert grouping_agrees_with_components(
        np.array([1, 1, 2], dtype=np.int32), np.array([0, 0, 1])
    )
    assert not grouping_agrees_with_components(
        np.array([1, 1, 2], dtype=np.int32), np.array([0, 0, 0])
    ), "two groups, one molecule"
    assert not grouping_agrees_with_components(
        np.array([1, 1], dtype=np.int32), np.array([0, 1])
    ), "one group, two molecules"
    assert grouping_agrees_with_components(
        np.array([0, 0], dtype=np.int32), np.array([0, 1])
    ), "ungrouped atoms say nothing"


def test_the_rhs_grouping_is_a_postcondition_on_the_product() -> None:
    """A retro rule can finally say whether it cuts a bond in two or opens a ring.

    The same disconnection on the same two targets, written three ways. Ungrouped does both,
    which is today's behaviour and the ambiguity being removed; `(B).(C)` keeps only the genuine
    bimolecular cut; `(B.C)` keeps only the ring opening. A product cannot be *required* to come
    apart -- that depends on the bonds -- so the grouping is checked after the edit and a
    disagreement refuses the application.
    """
    for smarts, expected in (
        ("[C:1]-[N:2]>>[C:1]-[Cl].[N:2]", {"C1CCNCC1": 1, "CCNC": 2}),
        ("[C:1]-[N:2]>>([C:1]-[Cl]).([N:2])", {"C1CCNCC1": 0, "CCNC": 2}),
        ("[C:1]-[N:2]>>([C:1]-[Cl].[N:2])", {"C1CCNCC1": 1, "CCNC": 0}),
    ):
        (pattern,) = reaction_patterns_from_smarts((smarts,))
        for target, precursor_count in expected.items():
            product, _info = cmol_from_smiles(target)
            outcomes = generate_precursors(product, pattern)
            counts = {len(outcome.precursors) for outcome in outcomes}
            wanted = {precursor_count} if precursor_count else set()
            # An empty set is the refusal: no application survived the postcondition.
            assert counts == wanted, smarts


def test_a_refused_grouping_is_counted() -> None:
    """Countable like its sibling postconditions, in the diagnostics rather than as a rejection
    reason: the rejection counters are mirrored into both device kernels, where an appended
    member would shift every count recorded before it."""
    (pattern,) = reaction_patterns_from_smarts(("[C:1]-[N:2]>>([C:1]-[Cl]).([N:2])",))
    product, _info = cmol_from_smiles("C1CCNCC1")

    result = precursors_from_mapping_result(
        product=product,
        reaction=pattern,
        mapping=find_matches(pattern.lhs, gmol_from_cmol(product)).assignments[0],
        ring_descriptors=RingDescriptors.ALL,
        max_precursor_atoms=None,
    )

    assert result.precursors == ()
    assert result.diagnostics.grouping_violations == 1


def test_arity_counts_groups_so_an_intramolecular_rule_leaves_the_channel() -> None:
    """Where the measured forward defect dies.

    `template_arities` decides which rules may be competitors, and it counted dots -- so a rule
    describing an intramolecular reaction with a dotted left-hand side was admitted as
    bimolecular and then matched both halves inside one molecule. A zero-level dot still counts;
    a dot inside a group does not.
    """
    assert template_arities(
        [
            "[C:1].[N:2]>>[C:1][N:2]",
            "([C:1]).([N:2])>>[C:1][N:2]",
            "([C:1].[N:2])>>[C:1][N:2]",
            "[C:1](-[N:2])-[O:3]>>[C:1]",
        ]
    ) == (2, 2, 1, 1)


def test_arity_agrees_with_rdkit_across_the_shipped_library() -> None:
    """The one place RDKit *is* an oracle for zero-level grouping.

    `Chem.MolFromSmarts` refuses `(C).(N)` outright -- verified against 2025.09.3 -- so there is
    no RDKit answer to compare a *match* against, and the matching semantics above are tested
    against what they claim rather than against another implementation. But reaction SMARTS is
    different: `ReactionFromSmarts` reads zero-level groups exactly as the template list, so
    `([C:1].[N:2])>>` is one reactant and `([C:1]).([N:2])>>` is two. That is the same question
    `template_arities` answers, and it is the number that decides which rules may compete, so it
    is worth holding to an outside implementation across every shipped row rather than to a
    handful of written cases.
    """
    reactions = pytest.importorskip("rdkit.Chem.rdChemReactions")
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    checked = 0
    for path in sorted((external_data_root() / "templates").glob("templates-*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                texts = [
                    text
                    for key, text in row.items()
                    if key and "template" in key and text and ">>" in text
                ]
                for text in texts:
                    parsed = reactions.ReactionFromSmarts(text)
                    assert parsed is not None, text
                    checked += 1
                    # Both sides: a forward rule states its arity on the left, a retro rule
                    # states how many precursors it claims on the right.
                    assert template_arities([text]) == (
                        parsed.GetNumReactantTemplates(),
                    ), text
                    assert template_arities([text.split(">>")[1] + ">>"]) == (
                        parsed.GetNumProductTemplates(),
                    ), text

    assert checked > 600, checked


@pytest.mark.parametrize(
    ("side", "expected"),
    [
        ("[C:1].[N:2]", 2),
        ("([C:1]).([N:2])", 2),
        ("([C:1].[N:2])", 1),
        ("([C:1].[N:2]).[O:3]", 2),
        ("[C:1](-[N:2])-[O:3]", 1),
    ],
)
def test_the_grouped_forms_agree_with_rdkit_on_arity(side: str, expected: int) -> None:
    """Including the two forms no shipped template writes yet, which is the point of checking."""
    reactions = pytest.importorskip("rdkit.Chem.rdChemReactions")
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    parsed = reactions.ReactionFromSmarts(f"{side}>>[C:1]")

    assert parsed.GetNumReactantTemplates() == expected
    assert template_arities([f"{side}>>[C:1]"]) == (expected,)


def test_a_bond_across_a_group_boundary_is_refused() -> None:
    """`(C1CC).(C1CC)` asks for one molecule and two at once, so it can never match.

    Only a ring closure can write such a bond -- a group starts its own fragment, so no chain
    bond crosses the boundary -- and reusing a ring index across two groups is the easy mistake.
    Refused rather than left to match nothing, because a pattern that silently matches nothing is
    the one failure a library of several hundred rules does not get read for.
    """
    for text in ("(C1CC).(C1CC)", "(C1CC).(C1)", "(CC1).(CC1)"):
        with pytest.raises(SmartsError):
            parse_smarts(text)


def test_a_bond_from_an_ungrouped_atom_into_a_group_is_allowed() -> None:
    """It contradicts nothing: the group constrains its own atoms and the bond brings one more in.

    The permissive half of the same reading that lets `C.(N)` match a single molecule.
    """
    assert groups("C1.(N1)") == [0, 1]
    assert bonded("C1.(N1)") == [(0, 1)]


def test_the_rhs_grouping_agrees_with_rdkit_on_its_own_open_issue() -> None:
    """[rdkit#6136](https://github.com/rdkit/rdkit/issues/6136), closed as expected behaviour.

    A bimolecular disconnection applied to tetrahydropyran: RDKit returns two propanols where
    the chemistry gives one diol, and the maintainer's answer is "you have asked for two
    products, so you get two products", with parentheses as the remedy. That remedy is this
    feature, and on the parenthesised spelling the two engines agree exactly -- which is a
    stronger statement than the parity the dotted spelling has, where they do not.

    `(B).(C)` is where they part, and upstream says why: the RDKit Book lists "component level
    grouping requiring matches in different components" among the SMARTS features it does not
    support, so RDKit reads those parentheses as the product-template split a dot already gives.
    vesma reads them as an assertion and refuses the ring.
    """
    chem = pytest.importorskip("rdkit.Chem")
    reactions = pytest.importorskip("rdkit.Chem.rdChemReactions")
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    sides = "[C:2]-[C:1]-[O:4]-[C:3]-[C:5]>>"
    together = f"{sides}(O-[C:1]-[C:2].[C:5]-[C:3]-O)"
    separate = f"{sides}(O-[C:1]-[C:2]).([C:5]-[C:3]-O)"
    dotted = f"{sides}O-[C:1]-[C:2].[C:5]-[C:3]-O"

    def rdkit_products(smarts: str) -> set[str]:
        reaction = reactions.ReactionFromSmarts(smarts)
        return {
            ".".join(sorted(chem.MolToSmiles(mol) for mol in outcome))
            for outcome in reaction.RunReactants((chem.MolFromSmiles("C1CCOCC1"),))
        }

    def vesma_products(smarts: str) -> set[str]:
        (pattern,) = reaction_patterns_from_smarts((smarts,))
        product, _info = cmol_from_smiles("C1CCOCC1")
        return {
            ".".join(sorted(rdkit_key.encode(one) for one in outcome.precursors))
            for outcome in generate_precursors(product, pattern)
        }

    assert rdkit_products(together) == {"OCCCCCO"}
    assert vesma_products(together) == {"OCCCCCO"}, "agreement on the fixed spelling"
    assert rdkit_products(dotted) == {"CCCO.CCCO"}, "the issue as reported"
    assert vesma_products(dotted) == {"OCCCCCO"}, "vesma edits the graph instead"
    assert rdkit_products(separate) == {"CCCO.CCCO"}, "not a constraint RDKit has"
    assert vesma_products(separate) == set(), "vesma refuses the ring"


def test_the_lhs_grouping_agrees_with_rdkits_own_intramolecular_example() -> None:
    """The RDKit Book's ring-closing metathesis, `([C:1]=[C;H2].[C:2]=[C;H2])>>[*:1]=[*:2]`.

    Its "Intramolecular reactions" section introduces zero-level grouping for exactly this, and
    both engines close `C=CCCCCC=C` to cyclohexene. The enforcement differs: `RunReactants` takes
    one molecule per reactant template, so RDKit gets the molecularity from the call, while vesma
    matches into one combined pot and gets it from the parentheses. Which is why the *dotted*
    spelling diverges in the other direction -- two templates to RDKit, so it cannot be asked
    about a single molecule at all, while vesma matches it inside the diene.
    """
    chem = pytest.importorskip("rdkit.Chem")
    reactions = pytest.importorskip("rdkit.Chem.rdChemReactions")
    from rdkit import RDLogger

    from vesma.cmol import cmol_union

    RDLogger.DisableLog("rdApp.*")

    grouped = "([C:1]=[C;H2].[C:2]=[C;H2])>>[*:1]=[*:2]"
    reaction = reactions.ReactionFromSmarts(grouped)

    assert reaction.GetNumReactantTemplates() == 1
    assert {
        chem.MolToSmiles(outcome[0])
        for outcome in reaction.RunReactants((chem.MolFromSmiles("C=CCCCCC=C"),))
    } == {"C1=CCCCC1"}

    (pattern,) = reaction_patterns_from_smarts((grouped,))
    diene, _info = cmol_from_smiles("C=CCCCCC=C")
    assert {
        rdkit_key.encode(one)
        for outcome in generate_precursors(diene, pattern)
        for one in outcome.precursors
    } == {"C1=CCCCC1"}

    # And the grouping is what refuses the pot the reaction was never about.
    pot = cmol_union([cmol_from_smiles(text)[0] for text in ("C=CCC", "C=CCC")])
    assert not generate_precursors(pot, pattern)


# Daylight's theory manual, section 4.5, verbatim: the six rows of its match-behaviour table.
# This is the specification, so it is the authority here rather than cases chosen locally.
DAYLIGHT_TABLE = (
    ("C.C", "CCCC", True, "no component level grouping specified"),
    ("(C.C)", "CCCC", True, "both carbons in the query match the same component"),
    (
        "(C).(C)",
        "CCCC",
        False,
        "the query must match carbons in two different components",
    ),
    ("(C).(C)", "CCCC.CCCC", True, "it does match carbons in two different components"),
    ("(C).C", "CCCC", True, "both carbons in the query match the same component"),
    ("(C).(C).C", "CCCC.CCCC", True, "the third matches a carbon anywhere"),
)


@pytest.mark.parametrize(
    ("pattern", "target", "expected", "why"),
    DAYLIGHT_TABLE,
    ids=[f"{row[0]}-vs-{row[1]}" for row in DAYLIGHT_TABLE],
)
def test_daylight_section_4_5(
    pattern: str, target: str, expected: bool, why: str
) -> None:
    """Both halves of the semantics, because the table states both.

    The prose above the table says only that atoms inside one set of parentheses must match
    within a single component -- the binding half. The table adds the separating half outright:
    `(C).(C)` against `CCCC` is "no, the query must match carbons in two different components".
    So the behaviour here is conformance, not an extension, and the mixed row `(C).C` is where
    the permissive reading of an ungrouped atom comes from rather than from a choice made here.
    """
    query, _info = gmol_from_smarts(pattern)
    molecule, _info = gmol_from_smiles(target)

    assert has_match(query, molecule) is expected, why


def test_daylight_distinguishes_esterification_from_lactonisation() -> None:
    """Why the operators exist at all, in Daylight's own words and its own example.

    Section 4.5 closes: the grouping operators "were added specifically for reaction processing.
    Without this construct, it is impossible to distinguish inter- versus intramolecular reaction
    queries" -- and gives these three left-hand sides. That is the ambiguity measured in this
    repository from both ends: a forward rule counting a cyclisation as a competition between two
    starting materials, and a retro rule claiming two precursors while opening a ring.
    """
    from vesma import gmol_from_cmol
    from vesma.cmol import cmol_union

    pot = cmol_union([cmol_from_smiles(text)[0] for text in ("CC(=O)O", "CCO")])
    hydroxy_acid, _info = cmol_from_smiles("OCCCCC(=O)O")

    def matches(side: str) -> tuple[bool, bool]:
        query, _info = gmol_from_smarts(side)
        return (
            has_match(query, gmol_from_cmol(pot)),
            has_match(query, gmol_from_cmol(hydroxy_acid)),
        )

    assert matches("C(=O)O.OCC") == (True, True), "esterifications, either way"
    assert matches("(C(=O)O).(OCC)") == (True, False), "intermolecular esterifications"
    assert matches("(C(=O)O.OCC)") == (False, True), "lactonisations"


def test_a_refused_grouping_refuses_the_application_and_not_the_rule() -> None:
    """`>>(B).(C)` is how a template asks for two precursors, and it gets them.

    Easy to misread the other way from a single-target comparison. The postcondition is about
    what *this* edit produced, so one rule answers differently on two substrates: the acyclic
    ether is cut in two, and only the ring -- where the halves stay joined through the rest of
    the cycle, so the edit produced one molecule -- is dropped. `(B.C)` is the exact mirror, and
    between them the two spellings partition what the dotted form answers for both.
    """
    sides = "[C:2]-[C:1]-[O:4]-[C:3]-[C:5]>>"
    separate = f"{sides}(O-[C:1]-[C:2]).([C:5]-[C:3]-O)"
    together = f"{sides}(O-[C:1]-[C:2].[C:5]-[C:3]-O)"
    dotted = f"{sides}O-[C:1]-[C:2].[C:5]-[C:3]-O"

    def outcomes(smarts: str, target: str) -> set[str]:
        (pattern,) = reaction_patterns_from_smarts((smarts,))
        product, _info = cmol_from_smiles(target)
        return {
            ".".join(sorted(rdkit_key.encode(one) for one in outcome.precursors))
            for outcome in generate_precursors(product, pattern)
        }

    assert outcomes(separate, "CCCOCCC") == {"CCCO.CCCO"}, "two precursors, as asked"
    # The ring gave one molecule, so the claim is false and the application is dropped.
    assert outcomes(separate, "C1CCOCC1") == set()
    assert outcomes(together, "C1CCOCC1") == {"OCCCCCO"}, "the mirror"
    assert outcomes(together, "CCCOCCC") == set()
    # The dotted form answers for both, which is the ambiguity the two spellings divide.
    assert outcomes(dotted, "CCCOCCC") == {"CCCO.CCCO"}
    assert outcomes(dotted, "C1CCOCC1") == {"OCCCCCO"}

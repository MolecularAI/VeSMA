"""The linter finds atoms a template drops, and must not invent any."""

from __future__ import annotations

import csv
from functools import cache

import pytest

from vertsy.template_lint import (
    duplicates,
    half_migrated_duplicates,
    identical_patterns,
    lint,
    lost_atom_maps,
    subsumed_patterns,
    unbalanced,
    ungrouped_components,
)


@cache
def shipped_retro_rules() -> dict[str, str]:
    """Every retro rule of our own curated tables, by template code, first spelling wins.

    First-wins because that is what `load_merged_library` does, so a lint reading this sees the
    same table a search compiles.
    """
    from tests.datadirs import external_data_root

    rules: dict[str, str] = {}
    for path in sorted((external_data_root() / "templates").glob("templates-*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                text = row.get("retro_template")
                if text and ">>" in text:
                    rules.setdefault(row["template_code"], text)
    return rules


# The measured offender: an ester hydrolysis that maps the alcohol carbon and never returns it, so
# RDKit deletes it and everything attached. On a 45-heavy-atom ester it discarded 33 atoms.
BROKEN = "[C:4]-[O;H0;D2;+0:3]-[C:2]=[O;D1;H0:1]>>[O;D1;H0:1]=[C:2]-[OH;D1;+0:3]"


def test_it_names_the_atom_that_disappears() -> None:
    assert lost_atom_maps(BROKEN) == frozenset({4})
    assert unbalanced(BROKEN)


def test_a_template_that_accounts_for_every_atom_is_clean() -> None:
    assert lost_atom_maps("[C:1]-[C:2]>>[C:1].[C:2]") == frozenset()
    assert not unbalanced("[C:1]-[C:2]>>[C:1].[C:2]")
    # Precursors may carry MORE than the product -- a leaving group is the normal case.
    assert not unbalanced("[C:1]-[C:2]>>[C:1]-[Br:9].[C:2]")


def test_aromatic_bond_colons_are_not_atom_maps() -> None:
    """The mistake that made an earlier count read 10% instead of 0.1%.

    `[c:1]:[n:2]` contains three colons and two map numbers. Counting colons swallows the bond and
    inflates the product side of every aromatic template, which is most of the library.
    """
    aromatic = "[c:1]1:[n:2]:[c:3]:[cH]:[cH]:1>>[c:1].[n:2].[c:3]"
    assert lost_atom_maps(aromatic) == frozenset()
    assert aromatic.count(":") > 6  # plenty of colons, none of them lost atoms


def test_agents_count_as_accounted_for() -> None:
    """A three-part `a>b>c` template keeps its middle section: an agent's atoms are not deleted."""
    assert lost_atom_maps("[C:1]-[C:2]>[Pd:3]>[C:1].[C:2]") == frozenset()


def test_lint_reports_indices_into_the_caller_s_own_numbering() -> None:
    """Indices, not a filtered library: three rule-id schemes already exist in this repo."""
    templates = ["[C:1]>>[C:1]", BROKEN, "[C:2]-[C:3]>>[C:2].[C:3]", BROKEN]
    assert [index for index, _, _ in lint(templates)] == [1, 3]
    assert [sorted(lost) for _, lost, _ in lint(templates)] == [[4], [4]]
    # A tolerance counts only the egregious cases, and a negative one is a caller error.
    assert list(lint(templates, tolerance=1)) == []
    with pytest.raises(ValueError, match="non-negative"):
        unbalanced(BROKEN, tolerance=-1)


def test_duplicates_keeps_the_first_occurrence_and_reports_the_rest() -> None:
    """Indices of the repeats, so dropping them preserves source order."""
    templates = ["[C:1]>>[C:1]", BROKEN, "[C:1]>>[C:1]", BROKEN, "[N:1]>>[N:1]", BROKEN]
    assert list(duplicates(templates)) == [2, 3, 5]
    assert list(duplicates([])) == []
    assert list(duplicates(["[C:1]>>[C:1]"])) == []


def test_duplicates_compares_text_and_not_chemistry() -> None:
    """Two spellings of one transformation are not this function's business.

    Deciding that differently-mapped SMARTS are the same reaction is what `template_hash`
    does; a linter making that call from a string comparison would be guessing. What this
    catches is the cheap case a hand-curated table accumulates -- the identical line twice.
    """
    same_reaction_other_numbering = [
        "[C:1]-[O:2]>>[C:1].[O:2]",
        "[C:3]-[O:4]>>[C:3].[O:4]",
    ]
    assert list(duplicates(same_reaction_other_numbering)) == []
    assert list(duplicates(["[C:1]>>[C:1] ", "[C:1]>>[C:1]"])) == []


def test_ungrouped_components_is_the_migration_worklist() -> None:
    """Which side a rule leaves unstated, and nothing about which reading it meant.

    A bare dot is no constraint at all, so a multi-component side spelled with one admits both
    readings: `MC 91a` meant one molecule and `Hartenfeller 47` meant two, and the two were
    written identically. Deciding which is a judgement about the reaction, so this reports and
    refuses nothing.
    """
    assert list(ungrouped_components(["[C:1].[N:2]>>[C:1][N:2]"])) == [(0, 2, 0)]
    assert list(ungrouped_components(["[C:1][N:2]>>[C:1].[N:2]"])) == [(0, 0, 2)]
    assert not list(ungrouped_components(["([C:1]).([N:2])>>[C:1][N:2]"])), "separate"
    assert not list(ungrouped_components(["([C:1].[N:2])>>[C:1][N:2]"])), "one molecule"
    assert not list(ungrouped_components(["[C:1][N:2]>>([C:1]).([N:2])"]))
    # One piece per side states nothing, so there is nothing to report.
    assert not list(ungrouped_components(["[C:1][N:2]>>[C:1]"]))


def test_a_branch_is_not_a_group() -> None:
    """`(` after an atom opens a branch and `(` before one opens a group.

    The only thing that distinguishes them is position, so a side whose pieces merely *begin*
    and *end* with a parenthesis is not necessarily migrated: `(N)C(O)` is one piece with two
    branches, and reading it as grouped would mark an unstated rule as done.
    """
    assert list(ungrouped_components(["[C:1](-[N:2]).[O:3]>>[C:1]"])) == [(0, 2, 0)]
    assert not list(ungrouped_components(["([C:1](-[N:2])).([O:3])>>[C:1]"]))


def test_half_migrated_duplicates_catches_a_broken_deduplication() -> None:
    """The fault this exists for was made in this repository and measured.

    `iSCB 63a` was character-for-character `Hartenfeller 29a`, so `load_merged_library`
    deduplicated it first-wins and only one ever compiled. Grouping the Hartenfeller row made the
    texts differ, the dedup stopped firing, and four rules describing the same chemistry compiled
    under two different claims about their own molecularity -- which is worse than either claim.
    """
    pairs = sorted(
        set(
            half_migrated_duplicates(
                [
                    "[C:1]-[N:2]>>([C:1]-[Cl]).([N:2])",  # 0: migrated
                    "[C:1]-[N:2]>>[C:1]-[Cl].[N:2]",  # 1: its forgotten twin
                    "[C:1]-[O:2]>>[C:1]-[Cl].[O:2]",  # 2: unrelated, and unstated
                ]
            )
        )
    )

    assert pairs == [(0, 1)]


def test_a_parent_and_its_intramolecular_sibling_are_not_reported() -> None:
    """They differ only in parentheses too, and that is the whole point of them.

    `Hartenfeller 47` says two precursors and `MC 102` says one molecule in two fragments. Both
    have stated something, so neither is a migration left half done -- which is why "stated" here
    cannot be a count of top-level pieces: a lone `(A.B)` is one piece and a statement.
    """
    assert not list(
        half_migrated_duplicates(
            [
                "[C:1]-[N:2]>>([C:1]-[Cl]).([N:2])",
                "[C:1]-[N:2]>>([C:1]-[Cl].[N:2])",
            ]
        )
    )


def test_the_shipped_retro_tables_have_no_half_migrated_pair() -> None:
    """A standing check, because the regression was invisible: the gate stayed green through it.

    Nothing failed when four rules stopped deduplicating. The only visible trace was a rule count
    that rose by four, in a run whose rule count was already rising for a good reason.
    """
    from tests.datadirs import external_data_root

    rules: dict[str, str] = {}
    for path in sorted((external_data_root() / "templates").glob("templates-*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                text = row.get("retro_template")
                if text and ">>" in text:
                    rules[row["template_code"]] = text

    codes = list(rules)
    pairs = sorted(set(half_migrated_duplicates(list(rules.values()))))

    assert not pairs, [(codes[left], codes[right]) for left, right in pairs]


def test_identical_patterns_sees_a_renumbering_the_text_check_cannot() -> None:
    """The whole reason it exists. One rule, three numberings, no two texts alike.

    `duplicates` finds none of these pairs; the certificate finds all of them, because atom-map
    numbers never reach it. That is the class that silently un-deduplicated four shipped rules when
    grouping was applied to one side of a hand-matched pair and not the other.
    """
    numberings = [
        "[CH2:4]-[O:3]-[CX4:2]>>([C:4]-[Cl]).([O:3]-[C:2])",
        "[CH2:1]-[O:2]-[CX4:3]>>([C:1]-[Cl]).([O:2]-[C:3])",
        "[CH2:9]-[O:7]-[CX4:5]>>([C:9]-[Cl]).([O:7]-[C:5])",
    ]

    assert not list(duplicates(numberings)), "no two texts are the same"
    assert sorted(set(identical_patterns(numberings))) == [(0, 1), (0, 2), (1, 2)]


def test_identical_patterns_is_semantic_about_predicates() -> None:
    """`[#6]` and `[C,c]` compile to the same masks, so they are one predicate.

    A key that separated them would report two rules where a search sees one -- the matcher reads
    the masks, not the text.
    """
    spellings = [
        "[CH2:4]-[O:3]-[CX4:2]>>([C:4]-[Cl]).([O:3]-[C:2])",
        "[CH2:4]-[#8;A:3]-[C;X4:2]>>([C:4]-[Cl]).([O:3]-[C:2])",
    ]

    assert sorted(set(identical_patterns(spellings))) == [(0, 1)]


def test_identical_patterns_keeps_a_parent_and_its_sibling_apart() -> None:
    """Their whole difference is the grouping, and the grouping is part of the rule."""
    pair = [
        "[C:1](=O)-[N:2]-[#6:3]>>([C:1](=O)[Cl]).([N:2]-[#6:3])",
        "[C:1](=O)-[N:2]-[#6:3]>>([C:1](=O)[Cl].[N:2]-[#6:3])",
    ]

    assert not list(identical_patterns(pair))


def test_a_template_that_will_not_compile_is_skipped() -> None:
    """It cannot be a duplicate of anything, and the parse failure is another lint's finding."""
    rows = [
        "[CH2:4]-[O:3]-[CX4:2]>>([C:4]-[Cl]).([O:3]-[C:2])",
        "this is not a reaction",
        "[CH2:1]-[O:2]-[CX4:3]>>([C:1]-[Cl]).([O:2]-[C:3])",
    ]

    assert sorted(set(identical_patterns(rows))) == [(0, 2)]


def test_the_shipped_tables_carry_no_duplicate_rule_at_all() -> None:
    """Both relations are now silent on the table, which is the state the deletions bought.

    They used to find fifteen pairs, every one a byte-identical `iSCB` copy of a `Hartenfeller` or
    `MC` row -- and every one *already dropped at load*, so they looked like rules and were not.
    Deleting them changed nothing a search compiles and removed the class entirely.

    Asserted as emptiness rather than as a count, so adding a genuinely new rule cannot fail this,
    and re-introducing a duplicate cannot pass it.
    """
    rules = shipped_retro_rules()
    texts = list(rules.values())
    by_text: dict[str, int] = {}
    text_pairs = set()
    for index, text in enumerate(texts):
        if text in by_text:
            text_pairs.add((by_text[text], index))
        else:
            by_text[text] = index

    assert text_pairs == set(), "no row repeats another's text"
    # And the graph relation agrees, which is the stronger statement: no two rows are the same rule
    # under any numbering or spelling, not merely under the same characters.
    assert list(identical_patterns(texts)) == []


def test_subsumed_patterns_reports_the_broader_rule_first() -> None:
    """The order in the pair is the finding: it names which of the two is the broader one.

    Taken from the shipped table. `Hartenfeller 14a` spells out a substituent on the triazole
    carbon; `Hartenfeller 13a` leaves that position to the molecule, so it fires on everything `14a`
    fires on and produces the same internal alkyne. Confirmed by outcome on `Cn1nnc(C)c1C`.
    """
    broad = "[c:1]1:[c:2]:[n]:[n]:[n]1-[C:3]>>[C:1]#[C:2].[C:3]-[Cl]"
    narrow = "[c:1]1:[c:2](-[C:4]):[n]:[n]:[n]1-[C:3]>>[C:1]#[C:2]-[C:4].[C:3]-[Cl]"

    assert list(subsumed_patterns([broad, narrow])) == [(0, 1)]
    # The index order follows the input, so the pair still names the broad rule first.
    assert list(subsumed_patterns([narrow, broad])) == [(1, 0)]


def test_subsumption_is_mutual_exactly_where_the_certificate_agrees() -> None:
    """Two relations written independently, and on an identical pair they must both fire.

    Mutual subsumption is identity, so a pair the certificate calls one rule must be reported in
    both directions -- and a pair it calls two must not be. On the shipped table they agree
    exactly.
    """
    numberings = [
        "[CH2:4]-[O:3]-[CX4:2]>>[C:4]-[Cl].[O:3]-[C:2]",
        "[CH2:9]-[O:7]-[CX4:5]>>[C:9]-[Cl].[O:7]-[C:5]",
    ]

    assert sorted(subsumed_patterns(numberings)) == [(0, 1), (1, 0)]
    assert sorted(set(identical_patterns(numberings))) == [(0, 1)]


def test_subsumed_patterns_does_not_merge_opposite_disconnections() -> None:
    """`MC 64a` and `MC 65a` break different bonds of one thioether; see the detail in
    `tests/vesma/test_pattern_subsumption.py`. Locked here too, because this is the generator a
    worklist would be read from and a false row here is a row inviting a deletion."""
    rows = [
        "[c:1]-[S;D2:2]-[C:3]>>[c:1]-[Cl].[SH:2]-[C:3]",
        "[c:1]-[S;D2:2]-[CX4:3]>>[c:1]-[SH:2].[CX4:3]-[Cl]",
    ]

    assert list(subsumed_patterns(rows)) == []


def test_half_migration_is_found_through_a_renumbering() -> None:
    """What the old text comparison could not do, and the reason it was rebuilt on the certificate.

    Stripping parentheses and comparing strings needs the two rows to agree character for character
    everywhere else -- including their atom-map numbers. These two do not, and they are the same
    rule with one side's molecularity stated and the other's not.
    """
    stated = "[CH2:4]-[O:3]-[CX4:2]>>([C:4]-[Cl]).([O:3]-[C:2])"
    unstated_and_renumbered = "[CH2:1]-[O:2]-[CX4:3]>>[C:1]-[Cl].[O:2]-[C:3]"

    assert list(half_migrated_duplicates([stated, unstated_and_renumbered])) == [(0, 1)]
    assert not list(duplicates([stated, unstated_and_renumbered])), "the texts differ"
    # And they are not the same rule: one of them states a molecularity and the other does not.
    assert not list(identical_patterns([stated, unstated_and_renumbered]))


def test_a_parent_and_its_intramolecular_sibling_are_not_a_half_migration() -> None:
    """Both state a molecularity. Different ones, on purpose -- `Hartenfeller 47` and `MC 102`."""
    parent = "[C:1](=O)-[N:2]-[#6:3]>>([C:1](=O)[Cl]).([N:2]-[#6:3])"
    sibling = "[C:1](=O)-[N:2]-[#6:3]>>([C:1](=O)[Cl].[N:2]-[#6:3])"

    assert list(half_migrated_duplicates([parent, sibling])) == []


def test_what_the_two_relations_find_on_the_shipped_table() -> None:
    """Locked, because this is the worklist and a change to it is a change to the table.

    Three findings, and the second is the live worklist.

    **No pair is mutual any more**, because mutual subsumption is identity and the duplicates are
    deleted. Kept as an assertion rather than dropped: it is the cheap cross-check that the two
    relations -- one canonicalising, one searching for an embedding -- still agree, since a mutual
    pair the certificate calls two rules would mean one of them is wrong.

    **No strict pair either**, which is the state the narrowing bought. There were nine, all of one
    shape: the narrow rule spelled out a substituent the broad rule left to the molecule --
    `Hartenfeller 14a-c` ⊂ `13a-c` on the triazole carbon, `42a-f` ⊂ `41a-f` on the Heck alkene.
    Deleting the narrow rows would have been wrong, because the broad rule was the inverse of
    *neither* forward rule but the union of both: `FWD H 13` needs a terminal alkyne and `FWD H 14`
    a disubstituted one, so a step taken through the broad rule could not be reproduced by the
    forward rule meant to verify it. The fix was on the broad side -- `[cH1:2]` and `[CH1:2]`, the
    hydrogen each rule's own forward partner implies. Each substrate now has exactly one rule and
    gets the same answer as before.

    **Nothing is half migrated**, and that must keep holding now that the lockstep pairs the check
    was written for are gone -- the next grouping migration has no twin to fall out of step with,
    which is a reason the check could rot unnoticed.
    """
    texts = list(shipped_retro_rules().values())

    pairs = set(subsumed_patterns(texts))
    mutual = {pair for pair in pairs if (pair[1], pair[0]) in pairs}
    strict = pairs - mutual

    assert mutual == set(), "no two rules are identical any more"
    assert strict == set(), "no rule is a broader twin of another"
    assert pairs == set(), "so the relation is silent on the shipped table"
    # Guard against the degenerate pass: the relation must still be *able* to fire. A rule always
    # subsumes itself, and `subsumed_patterns` skips that pair, so prove it on a crafted one.
    assert list(
        subsumed_patterns(
            [
                "[C:1]-[Cl]>>[C:1]-[OH]",
                "[CH3:3]-[CX4:1]-[Cl]>>[CH3:3]-[CX4:1]-[OH]",
            ]
        )
    ) == [(0, 1)]
    certified = set(identical_patterns(texts))
    assert {tuple(sorted(pair)) for pair in mutual} <= certified
    assert certified == set(), "and the certificate agrees there is no duplicate left"
    assert list(half_migrated_duplicates(texts)) == []

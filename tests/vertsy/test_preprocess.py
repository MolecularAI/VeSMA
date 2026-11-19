"""Target preparation finds duplicates and runs probes, and leaves targets alone otherwise."""

from __future__ import annotations

import pytest

from vertsy.preprocess import (
    UNCHANGED,
    canonical_smiles,
    neutralized,
    prepare_target,
    prepare_targets,
)
from vesma import cmol_from_smiles

# A protonated ring nitrogen whose hydrogen is written as its own atom node -- the shape that
# created this module, because the matcher refused it. It no longer does: the fold lives in
# `vesma.parsing.reading`, and the assertion below is what keeps that claim honest here.
EXPLICIT_H = (
    "O[C@H]1[C@H](O)C[N@+](C[C@H]([C@@H]2O[C@H](C3=C([N+]([O-])=O)C=CC=C3)OC[C@H]2O)O)"
    "([H])[C@@H]1CO"
)


def test_the_case_that_created_this_module_is_now_the_parsers_job() -> None:
    """No preprocessing needed to read it, so `prepare_target` must not claim credit."""

    cmol_from_smiles(EXPLICIT_H)

    assert prepare_target(EXPLICIT_H).note == UNCHANGED
    assert not prepare_target(EXPLICIT_H).changed


def test_a_target_is_not_rewritten_without_being_asked() -> None:
    """A target differing from its canonical form is not grounds for rewriting it.

    Rewriting for cosmetic reasons would put a molecule in the report nobody asked about.
    """

    prepared = prepare_target("C(O)C")

    assert prepared.searched == "C(O)C"
    assert prepared.note == UNCHANGED


def test_an_unparseable_target_is_named_here() -> None:
    """Rather than surfacing as an empty result after a long search."""

    with pytest.raises(ValueError, match="not parseable"):
        prepare_target("this is not a molecule")


def test_duplicates_are_compared_as_molecules_not_as_strings() -> None:
    """`C(O)C` and `CCO` are one molecule written two ways, and the second is the duplicate."""

    prepared = prepare_targets(("C(O)C", "CCO", "OCC"))

    assert [target.duplicate_of for target in prepared] == [None, 1, 1]
    assert "duplicate of target 1" in prepared[1].note


def test_duplicates_are_kept_so_indices_still_line_up() -> None:
    """Report filenames and `targets.jsonl` line numbers are positional."""

    prepared = prepare_targets(("CCO", "CCO", "CCC"))

    assert len(prepared) == 3
    assert [target.original for target in prepared] == ["CCO", "CCO", "CCC"]
    assert prepared[2].duplicate_of is None


def test_neutralization_is_off_by_default_and_reported_when_used() -> None:
    """It changes the molecule, so it is a probe the caller asks for, not a default."""

    assert prepare_target("C[NH3+]").searched == "C[NH3+]"

    probe = prepare_target("C[NH3+]", neutralize=True)

    assert probe.searched == "CN"
    assert "free-base probe" in probe.note


def test_neutralization_cannot_rescue_every_charged_target() -> None:
    """A quaternary ammonium has no neutral form, so the probe is not a general fix."""

    assert neutralized("C[N+](C)(C)C") == "C[N+](C)(C)C"
    assert prepare_target("C[N+](C)(C)C", neutralize=True).note == UNCHANGED


def test_a_repeated_target_is_reported_as_a_duplicate_of_the_first() -> None:
    """Duplicates are derived from the molecules, not from a hardcoded index.

    The campaign file that motivated this held one target twice, and the original archive
    builder special-cased its position. This asserts the general behaviour instead, so the
    property survives without the repository needing that file.
    """

    prepared = prepare_targets(
        (
            "CCO",
            EXPLICIT_H,
            "OCC",  # the same molecule as the first, written differently
        )
    )

    assert len(prepared) == 3
    assert prepared[0].duplicate_of is None
    assert prepared[2].duplicate_of == 1

    # and none of the three was rewritten to get there
    for target in prepared:
        cmol_from_smiles(target.searched)
        assert not target.changed


@pytest.mark.external_data
def test_canonical_smiles_keeps_stereochemistry() -> None:
    """`vesma.keyer.rdkit_token_from_smiles` drops it; a target must not.

    Dropping stereochemistry here would search for a different molecule.
    """

    assert canonical_smiles("C[C@H](N)O") != canonical_smiles("C[C@@H](N)O")

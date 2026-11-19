"""Isotopes on Metal: matched exactly, and carried through retro construction as on the CPU.

Word 13 of the packed layout holds `x` in its low half and the isotope in its high half; these
tests hold the device to the CPU reference wherever that half is read or written.
"""

from __future__ import annotations

import numpy as np
import pytest

from vesma import cmol_from_smiles, gmol_from_cmol, gmol_from_smarts
from vesma.cpu_matcher import CpuReactionMatcher, has_match_many
from vesma.metal_matcher import (
    ISOTOPE_SHIFT,
    ISOTOPE_WORD,
    GpuUnavailableError,
    MetalMatcher,
    probe_metal,
)
from vesma.reaction import parse_reaction_smarts

pytestmark = [pytest.mark.gpu, pytest.mark.metal]

PATTERNS = [
    "[13C]",
    "[12C]",  # a written 12 is a label: natural-abundance carbon is unlabelled, not 12
    "[13CH3]O",
    "[!13C]",
    "[13C,N]",
    "[C;$([13C]=O)]",
    "[C;!$([13C])]",
    "[18O]",
    "[O;!18O]",
    # `x` shares word 13 with the isotope: a predicate naming x must not name a mass.
    "[Cx2,Nx0]",
    "C",
]
MOLECULES = [
    "C",
    "[13CH4]",
    "[13CH3]O",
    "CO",
    "[13CH](=O)O",
    "C(=O)O",
    "N",
    "[13CH3][18OH]",
    "[12CH4]",
    "[13CH2]1[13CH2][13CH2]1",
]


@pytest.fixture(scope="module", autouse=True)
def metal_available() -> None:
    try:
        probe_metal()
    except (GpuUnavailableError, RuntimeError) as exc:
        pytest.skip(f"Metal backend not available: {exc}")


def test_metal_matches_isotopes_like_the_cpu() -> None:
    patterns = [gmol_from_smarts(s)[0] for s in PATTERNS]
    molecules = [gmol_from_cmol(cmol_from_smiles(s)[0]) for s in MOLECULES]
    expected = has_match_many(patterns, molecules, n_jobs=1)
    assert expected.any() and not expected.all()
    np.testing.assert_array_equal(
        MetalMatcher().has_match_many(patterns, molecules), expected
    )


def generated(rule: str, smiles: str):
    from vesma.metal_reaction_matcher import MetalReactionMatcher, metal_stack_mols

    matcher = MetalReactionMatcher([parse_reaction_smarts(rule)])
    batch, hits = matcher.match(metal_stack_mols([cmol_from_smiles(smiles)[0]]))
    return batch, hits, matcher


def masses(batch, index: int, atom_count: int) -> list[int]:
    words = batch.atoms[index].reshape(64, 14)[:atom_count]
    return (words[:, ISOTOPE_WORD] >> ISOTOPE_SHIFT).tolist()


def test_a_mapped_atom_keeps_its_isotope() -> None:
    molecule = cmol_from_smiles("[13CH3][18OH]")[0]
    batch, _, _ = generated("[*:1]>>[*:1]", "[13CH3][18OH]")
    assert len(batch) > 0
    atom_count = molecule.Z.size
    for index in range(len(batch)):
        assert masses(batch, index, atom_count) == molecule.M.tolist()


def test_an_added_atom_takes_the_templates_isotope() -> None:
    batch, _, _ = generated("[O:1]>>[O:1][13CH3]", "CO")
    found = sorted(
        mass
        for index in range(len(batch))
        for mass in masses(batch, index, int(batch.atom_counts[index]))
    )
    assert 13 in found
    assert found.count(13) == len(batch)


def test_an_added_atom_with_no_isotope_is_unlabelled() -> None:
    batch, _, _ = generated("[O:1]>>[O:1]C", "CO")
    assert len(batch) > 0
    for index in range(len(batch)):
        assert set(masses(batch, index, int(batch.atom_counts[index]))) == {0}


def test_a_mapped_atom_cannot_change_isotope() -> None:
    rule = "[C:1]>>[13C:1]"
    cpu_precursors, cpu_hits = CpuReactionMatcher([parse_reaction_smarts(rule)]).match(
        [cmol_from_smiles("CO")[0]]
    )
    batch, hits, _ = generated(rule, "CO")
    assert len(cpu_hits) == len(hits) == 0
    assert len(cpu_precursors) == len(batch) == 0

"""Reading text as a molecule, where SMILES and SMARTS disagree.

The specifications quoted here are OpenSMILES 1.0 and Daylight's SMARTS theory manual.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.vesma.molecule_equality import assert_same_molecule
from vesma import (
    cmol_from_smiles,
    gmol_from_cmol,
    gmol_from_smarts,
    gmol_from_smiles,
    smiles_from_cmol,
)
from vesma.configurations import ChemistryRejection
from vesma.cpu_matcher import has_match
from vesma.gmol_atoms import BUCKET_MAX, CHARGE_MAX, CHARGE_MIN, MAX_ISOTOPE
from vesma.parsing.errors import NotMolecularSmilesError


def molecule(smiles: str):
    return cmol_from_smiles(smiles)[0]


def total_hydrogens(smiles: str) -> list[int]:
    return [int(count) for count in molecule(smiles).H]


# --------------------------------------------------------------------------- #
# OpenSMILES 3.1: "If no `Hn` is specified, it is identical to `H0`. If `H` is
# specified without a number, it is identical to `H1`."
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("written", "hydrogens"),
    [
        ("[CH4]", 4),
        ("[NH3]", 3),
        ("[OH2]", 2),
        ("[NH](C)C", 1),
        ("[NH1](C)C", 1),
        ("[OH-]", 1),
    ],
)
def test_written_hydrogen_count_is_taken_exactly(written: str, hydrogens: int) -> None:
    assert total_hydrogens(written)[0] == hydrogens


def test_bare_h_inside_brackets_means_one() -> None:
    assert total_hydrogens("[NH](C)C") == total_hydrogens("[NH1](C)C")


def test_unbracketed_atom_still_fills_to_its_lowest_normal_valence() -> None:
    """OpenSMILES 3.1, Organic Subset. The bracket is the whole signal."""
    assert total_hydrogens("C") == [4]
    assert total_hydrogens("N") == [3]
    assert total_hydrogens("O") == [2]


@pytest.mark.parametrize("smiles", ["[C]", "[N]", "[O]", "[S]", "[CH0]", "[NH0]"])
def test_bracket_atom_without_hydrogens_is_open_shell_and_refused(smiles: str) -> None:
    """`[N]` is atomic nitrogen, not ammonia, and this model does not invent one."""
    with pytest.raises(ChemistryRejection):
        cmol_from_smiles(smiles)


# --------------------------------------------------------------------------- #
# The chemistry the old reading got wrong
# --------------------------------------------------------------------------- #


def test_tempo_is_a_radical_and_is_refused_rather_than_hydrogenated() -> None:
    """TEMPO's `[O]` carries one bond and no hydrogen: a nitroxide radical.

    Read leniently it gained a hydrogen and came back as TEMPO-H, the hydroxylamine --
    a different, closed-shell molecule, returned with no refusal and no telemetry. A
    refusal that names its cause is the point of the model; a silently wrong molecule is
    what it exists to prevent.
    """
    with pytest.raises(ChemistryRejection):
        cmol_from_smiles("CC1(C)CCCC(C)(C)N1[O]")

    # The hydroxylamine really is a different molecule, and it still builds.
    assert total_hydrogens("CC1(C)CCCC(C)(C)N1O")[-1] == 1


def test_cyclopentadienyl_carbon_bearing_a_substituent_carries_no_hydrogen() -> None:
    """A ligand spelling from the building-block corpus: `[C]` bears the phosphine."""
    with pytest.raises(ChemistryRejection):
        cmol_from_smiles("CC(C)P([C]1[CH][CH][CH][CH]1)C(C)C")


# --------------------------------------------------------------------------- #
# A query is not a molecule
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "smarts", ["[C;!H0]", "[C;H1,H2]", "[C;R]", "[C;D2]", "[C,N]", "[#6]", "[*]"]
)
def test_a_query_handed_to_the_molecule_reading_is_refused(smarts: str) -> None:
    """Concretization cannot catch these on its own.

    ``AtomMasks.exact_h_total`` answers -1 for a mask that is not a single bit and
    ``exact_c`` answers 0 for any mask that merely includes neutral, so each of these
    reached the builder and became methane.
    """
    with pytest.raises(NotMolecularSmilesError):
        cmol_from_smiles(smarts)


# --------------------------------------------------------------------------- #
# Properties the masks cannot hold exactly
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("smiles", ["[C+5]", "[S-4]", "[Fe+7]"])
def test_a_charge_outside_the_representable_range_is_refused(smiles: str) -> None:
    """The bucket saturates at -3 and +4, so `[C+5]` would silently become a +4 carbon."""
    with pytest.raises(NotMolecularSmilesError):
        cmol_from_smiles(smiles)


@pytest.mark.parametrize("smiles", ["[CH8]", "[PH9]"])
def test_a_hydrogen_count_outside_the_representable_range_is_refused(
    smiles: str,
) -> None:
    """OpenSMILES permits up to nine; the count bucket saturates at seven."""
    with pytest.raises(NotMolecularSmilesError):
        cmol_from_smiles(smiles)


@pytest.mark.parametrize("smiles", ["[C+4]", "[NH4+]", "[SH2]", "C[O-]"])
def test_the_edges_of_the_representable_range_still_build(smiles: str) -> None:
    assert molecule(smiles).atom_count > 0


def test_a_query_may_still_use_the_saturating_buckets() -> None:
    """The limit binds the molecule reading only. `[C+4]` as a pattern means "+4 or more"."""
    assert gmol_from_smarts("[C+5]")[0].atom_masks.size == 1


# --------------------------------------------------------------------------- #
# Isotopes
# --------------------------------------------------------------------------- #


def test_isotope_zero_is_refused_rather_than_read_as_natural_abundance() -> None:
    """A documented divergence from OpenSMILES, and a loud one.

    The specification keeps `[0S]` distinct from `[S]` so a writer can round-trip its own
    output. There is no nuclide of mass zero, so this refuses the spelling instead of
    keying it as natural abundance and silently agreeing with `[S]`.
    """
    with pytest.raises(NotMolecularSmilesError):
        cmol_from_smiles("[0S]C")


def test_isotope_outside_the_supported_range_is_refused() -> None:
    with pytest.raises(NotMolecularSmilesError):
        cmol_from_smiles("[1000S]C")


def test_the_writer_cannot_emit_an_isotope_the_reader_would_refuse() -> None:
    """One contract, enforced on the model, so a round trip cannot fall out of it."""
    built = molecule("C")
    with pytest.raises(ValueError, match="at most 999"):
        replace(built, M=np.array([MAX_ISOTOPE + 1], dtype=np.int16))

    labelled = replace(built, M=np.array([MAX_ISOTOPE], dtype=np.int16))
    assert_same_molecule(labelled, molecule(smiles_from_cmol(labelled)))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("C", CHARGE_MAX + 1, "charge must lie"),
        ("C", CHARGE_MIN - 1, "charge must lie"),
        ("H", BUCKET_MAX + 1, "hydrogen count must be at most"),
    ],
)
def test_a_cmol_cannot_hold_what_the_reader_would_refuse(
    field: str, value: int, message: str
) -> None:
    """The bound belongs on the model, not only on the text.

    Otherwise a validly built molecule serializes to `[CH4+5]` or `[CH8]` and fails to
    read back -- the writer emitting what the reader rejects, which is the asymmetry the
    isotope bound already closed.
    """
    built = molecule("C")
    dtype = np.int8 if field == "C" else np.int16
    changes = {field: np.array([value], dtype=dtype)}
    if field == "H":
        changes["h"] = np.array([value], dtype=np.int16)
    with pytest.raises(ValueError, match=message):
        replace(built, **changes)


# --------------------------------------------------------------------------- #
# Spellings that must not have moved
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("left", "right"),
    [("C", "[CH4]"), ("N", "[NH3]"), ("O", "[OH2]"), ("c1ccccc1", "C1=CC=CC=C1")],
)
def test_one_molecule_however_it_was_spelled(left: str, right: str) -> None:
    assert_same_molecule(molecule(left), molecule(right))


@pytest.mark.parametrize(
    "smiles",
    [
        "C[Si](C)(C)C",
        "CCCC[Sn](CCCC)(CCCC)C=C",
        "[nH]1cccc1",
        "[N+](C)(C)(C)C",
        "C[O-]",
        "[C@H](N)(O)C",
        "O=[N+]([O-])c1ccccc1",
        "[13CH3]CO",
    ],
)
def test_spellings_the_reading_must_leave_alone(smiles: str) -> None:
    """Charged, stereo, aromatic-heteroatom and saturated heavy-element brackets.

    Charged atoms already read as zero-hydrogen through the charge rule, and a bracket
    carrying only chirality strips to an element whose bonds already fill its valence.
    """
    assert molecule(smiles).atom_count > 0


# --------------------------------------------------------------------------- #
# What the SMILES graph is, and is not
# --------------------------------------------------------------------------- #


def test_the_smiles_graph_states_only_what_the_text_stated() -> None:
    """`gmol_from_smiles` is the unperceived seed, not a molecule expressed as a query.

    A bare `C` writes no hydrogen count, so the graph leaves H, degree and charge open and
    matches any aliphatic carbon. Only after perception does the graph pin the molecule.
    """
    seed, _ = gmol_from_smiles("C")
    assert int(seed.atom_masks.H[0]) == 0xFF
    assert has_match(seed, gmol_from_cmol(molecule("CC")))

    exact = gmol_from_cmol(molecule("C"))
    assert not has_match(exact, gmol_from_cmol(molecule("CC")))


# --------------------------------------------------------------------------- #
# The writer and the reader have to agree
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "smiles",
    [
        "C",
        "CCO",
        "CC(=O)O",
        "c1ccccc1",
        "[nH]1cccc1",
        "C[O-]",
        "[N+](C)(C)(C)C",
        "[13CH3]CO",
        "O=[N+]([O-])c1ccccc1",
        "CN1CCC(Oc2ccc3[nH]nc(S(=O)(=O)c4cccc5ccccc45)c3c2)C1",
    ],
)
def test_written_smiles_reads_back_as_the_same_molecule(smiles: str) -> None:
    """A stricter reader is only correct if the writer is strict too.

    OpenSMILES 3.6: "Any atom that is specified with square brackets must have its
    attached hydrogens explicitly represented." A bracket opened for a charge, a map or
    an isotope that then omitted the count denoted a different molecule -- and used to
    survive only because the reader was equally lenient.
    """
    built = molecule(smiles)
    assert_same_molecule(built, molecule(smiles_from_cmol(built)))


# --------------------------------------------------------------------------- #
# Hydrogens written as their own atoms
# --------------------------------------------------------------------------- #

# A protonated ring nitrogen whose hydrogen is a separate atom node. Valid
# SMILES for a real target, and the reason this fold exists. The `@` / `@@` difference between
# the two spellings is what the *string* needs once the neighbour is dropped; the matcher does
# not model chirality, so nothing here has to derive it.
EXPLICIT_H_TARGET = (
    "O[C@H]1[C@H](O)C[N@+](C[C@H]([C@@H]2O[C@H](C3=C([N+]([O-])=O)C=CC=C3)OC[C@H]2O)O)"
    "([H])[C@@H]1CO"
)
FOLDED_TARGET = (
    "O[C@H]1[C@H](O)C[N@@H+](C[C@H]([C@@H]2O[C@H](C3=C([N+]([O-])=O)C=CC=C3)OC[C@H]2O)O)"
    "[C@@H]1CO"
)


@pytest.mark.parametrize(
    "written,same_as,why",
    [
        (
            "C[H]",
            "C",
            "unbracketed neighbour: the hydrogen law refills the vacated bond",
        ),
        ("[CH3][H]", "C", "bracketed neighbour: its written count is incremented"),
        ("[H]O[H]", "O", "water with both hydrogens written out"),
        ("[NH3+][H]", "[NH4+]", "a charged neighbour still just gains a hydrogen"),
        ("C[C@@H]([H])O", "CCO", "a bare H in the bracket means one, not zero"),
    ],
)
def test_an_explicit_hydrogen_atom_folds_into_its_neighbour(
    written: str, same_as: str, why: str
) -> None:
    assert_same_molecule(cmol_from_smiles(written)[0], cmol_from_smiles(same_as)[0])


def test_the_target_that_forced_this_parses_as_supplied() -> None:
    """It needed a hand-made normalised copy outside the repository until this fold existed."""

    assert_same_molecule(
        cmol_from_smiles(EXPLICIT_H_TARGET)[0], cmol_from_smiles(FOLDED_TARGET)[0]
    )


@pytest.mark.parametrize(
    "written,why",
    [
        ("[2H]C", "deuterium is a different nuclide, not a hydrogen count"),
        ("[H+]", "a proton is a species, not a substituent"),
        ("[H][H]", "dihydrogen has no heavy atom to carry the count"),
        ("[2H]O[2H]", "heavy water keeps both nuclides"),
    ],
)
def test_what_must_not_be_folded_is_still_refused(written: str, why: str) -> None:
    """Each of these would be silently relabelled as something else if it folded."""

    with pytest.raises(Exception, match="no declared configuration"):
        cmol_from_smiles(written)


def test_a_query_keeps_its_explicit_hydrogen_atom() -> None:
    """The fold is part of the molecule reading only, so pattern semantics do not move.

    `parse_smarts` never calls `smiles_reading`, so `[H]` in a pattern still means an explicit
    hydrogen atom to match. Neither shipped rule set writes one, but the separation is what
    makes that a fact about the data rather than a thing to be careful about.
    """

    from vesma.parsing import parse_smarts, parse_smiles

    assert parse_smarts("C[H]").atom_count == 2
    assert parse_smiles("C[H]").atom_count == 1

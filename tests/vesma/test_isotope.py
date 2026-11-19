"""Isotope is exact molecular state: matched, conserved, serialized -- but not identity."""

from __future__ import annotations

import pytest

from vesma import cmol_from_smiles
from vesma.cmol_builder import gmol_from_cmol
from vesma.cpu_matcher import CpuMatcher, generate_precursors
from vesma.gmol_atoms import MASS_ANY, MASS_UNLABELED
from vesma.gmol_builder import gmol_from_smarts
from vesma.identity import short_ring_wl_molecule_id
from vesma.reaction import UnsupportedReactionError, parse_reaction_smarts
from vesma.smiles import smiles_from_cmol

Chem = pytest.importorskip("rdkit.Chem")


def matches(pattern_smarts: str, molecule_smiles: str) -> bool:
    molecule = gmol_from_cmol(cmol_from_smiles(molecule_smiles)[0])
    return CpuMatcher().has_match(gmol_from_smarts(pattern_smarts)[0], molecule)


@pytest.mark.parametrize(
    "pattern, molecule, expected",
    [
        ("[13C]", "[13CH3]CO", True),
        ("[12C]", "[13CH3]CO", False),
        ("C", "[13CH3]CO", True),  # a silent query accepts any nucleus
        ("[13C]", "CCO", False),  # an asserted mass rejects the unlabelled atom
    ],
)
def test_isotope_query_is_exact_or_any(
    pattern: str, molecule: str, expected: bool
) -> None:
    assert matches(pattern, molecule) is expected


def test_unlabelled_atoms_carry_the_natural_abundance_sentinel() -> None:
    molecule = cmol_from_smiles("CCO")[0]
    assert set(molecule.M.tolist()) == {MASS_UNLABELED}
    assert int(gmol_from_smarts("C")[0].atom_masks.M[0]) == MASS_ANY


def test_isotope_round_trips_through_rdkit_and_smiles() -> None:
    from vesma.interop import cmol_from_mol, mol_from_cmol

    molecule = cmol_from_mol(Chem.MolFromSmiles("[13CH3]CO"))

    assert molecule.M.tolist() == [13, 0, 0]
    assert "[13CH3]" in smiles_from_cmol(molecule)
    assert Chem.MolToSmiles(mol_from_cmol(molecule)) == "[13CH3]CO"


def test_a_mapped_atom_keeps_its_nucleus_through_a_reaction() -> None:
    product = cmol_from_smiles("[13CH3]CO")[0]
    reaction = parse_reaction_smarts("[C:1]-[C:2]>>[C:1].[C:2]")

    outcomes = generate_precursors(product, reaction)

    assert outcomes
    assert 13 in {
        int(mass)
        for outcome in outcomes
        for precursor in outcome.precursors
        for mass in precursor.M
    }


def test_an_rhs_nucleus_assertion_that_disagrees_is_rejected() -> None:
    product = cmol_from_smiles("[13CH3]CO")[0]
    reaction = parse_reaction_smarts("[C:1]-[C:2]>>[12C:1].[C:2]")

    with pytest.raises(UnsupportedReactionError, match="isotope"):
        generate_precursors(product, reaction)


def test_isotopologues_share_identity() -> None:
    """Identity is deliberately nucleus-blind, so a label never splits a key."""
    labelled = short_ring_wl_molecule_id(cmol_from_smiles("[13CH3]CO")[0])
    unlabelled = short_ring_wl_molecule_id(cmol_from_smiles("CCO")[0])

    assert labelled == unlabelled

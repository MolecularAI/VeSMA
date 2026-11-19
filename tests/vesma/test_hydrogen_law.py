"""The hydrogen law: set, copy, or complete -- and reject when none of the three applies."""

from __future__ import annotations

import pytest

from vesma import cmol_from_smiles
from vesma.configurations import select_configuration
from vesma.cpu_matcher import generate_precursors
from vesma.gmol_atoms import AROMATIC
from vesma.hydrogens import (
    HydrogenFailure,
    complete_hydrogens,
    hydrogens_after_edit,
)
from vesma.reaction import UnsupportedReactionError, parse_reaction_smarts
from vesma.symbols import ELEMENT_TO_Z


def after_edit(**kwargs):
    base = {
        "z": ELEMENT_TO_Z["C"],
        "charge_before": 0,
        "charge_after": 0,
        "sigma_degree_before": 1,
        "bond_order_sum_before": 1,
        "total_h_before": 3,
        "delocalized_before": False,
        "sigma_degree_after": 1,
        "bond_order_sum_after": 1,
        "delocalized_after": False,
        "bonding_changed": False,
    }
    return hydrogens_after_edit(**(base | kwargs))


def test_an_untouched_atom_keeps_its_hydrogen_exactly() -> None:
    assert after_edit(total_h_before=3) == 3


def test_losing_a_bond_completes_from_the_matched_configuration() -> None:
    """N goes from three bonds to two, so the freed valence becomes a hydrogen."""
    assert (
        after_edit(
            z=ELEMENT_TO_Z["N"],
            sigma_degree_before=3,
            bond_order_sum_before=3,
            total_h_before=0,
            sigma_degree_after=2,
            bond_order_sum_after=2,
            bonding_changed=True,
        )
        == 1
    )


def test_a_carbon_losing_a_bond_gains_a_hydrogen() -> None:
    assert (
        after_edit(
            sigma_degree_before=3,
            bond_order_sum_before=3,
            total_h_before=1,
            sigma_degree_after=2,
            bond_order_sum_after=2,
            bonding_changed=True,
        )
        == 2
    )


def test_a_charge_change_with_hydrogen_omitted_is_rejected() -> None:
    assert (
        after_edit(charge_after=1, bonding_changed=True)
        is HydrogenFailure.CHARGE_CHANGED_WITH_H_OMITTED
    )


def test_gaining_more_bonds_than_the_configuration_allows_is_negative() -> None:
    assert (
        after_edit(
            sigma_degree_before=1,
            bond_order_sum_before=1,
            total_h_before=3,
            sigma_degree_after=5,
            bond_order_sum_after=5,
            bonding_changed=True,
        )
        is HydrogenFailure.NEGATIVE
    )


def test_a_delocalized_atom_is_completed_from_its_sigma_capacity() -> None:
    """An aromatic carbon that gains a substituent loses its hydrogen to it."""
    aromatic_carbon = select_configuration(
        z=ELEMENT_TO_Z["C"],
        charge=0,
        sigma_degree=2,
        bond_order_sum=0,
        total_h=1,
        delocalized=True,
    )

    completed = complete_hydrogens(
        aromatic_carbon, bond_order_sum=0, sigma_degree=3, delocalized=True
    )

    assert completed == 0


def test_a_metal_completes_to_no_hydrogen() -> None:
    magnesium = select_configuration(
        z=ELEMENT_TO_Z["Mg"],
        charge=0,
        sigma_degree=2,
        bond_order_sum=2,
        total_h=0,
        delocalized=False,
    )

    assert (
        complete_hydrogens(
            magnesium, bond_order_sum=1, sigma_degree=1, delocalized=False
        )
        == 0
    )


def test_context_hydrogen_survives_a_disconnection() -> None:
    """Splitting into components is not a bond change, so context comes through intact."""
    product = cmol_from_smiles("CCCC(=O)NCc1ccccc1")[0]
    reaction = parse_reaction_smarts(
        "[C:1](=[O:2])-[NH1:3]-[C:4]>>[C:1](=[O:2])-[OH].[NH2:3]-[C:4]"
    )

    outcomes = generate_precursors(product, reaction)

    assert outcomes
    benzyl = next(
        precursor for precursor in outcomes[0].precursors if AROMATIC in precursor.A
    )
    ring_hydrogens = [int(h) for h, a in zip(benzyl.H, benzyl.A) if a == AROMATIC]
    assert sorted(ring_hydrogens) == [0, 1, 1, 1, 1, 1]  # the five CH are untouched


def test_an_edit_the_configuration_cannot_absorb_is_rejected() -> None:
    product = cmol_from_smiles("[NH4+]")[0]
    reaction = parse_reaction_smarts("[N+:1]>>[N;+0:1]")

    with pytest.raises(UnsupportedReactionError, match="charge_changed_with_h_omitted"):
        generate_precursors(product, reaction, max_mappings=1)

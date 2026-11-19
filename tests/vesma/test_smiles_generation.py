from __future__ import annotations

import numpy as np
import pytest

from tests.vesma.molecule_equality import assert_same_molecule
from vesma import (
    cmol_from_gmol,
    cmol_from_smiles,
    gmol_from_cmol,
    gmol_from_smarts,
    smiles_from_cmol,
)
from vesma.bonds import BondKind
from vesma.cpu_matcher import generate_precursors, has_match
from vesma.gmol import GMol
from vesma.gmol_atoms import (
    ALIPHATIC,
    atom_masks_from_props,
    bucket_mask_exact,
    mask_for_charge,
    z_mask_for_numbers,
)
from vesma.reaction import parse_reaction_smarts
from vesma.smiles import _bond_symbol


def _gmol(smarts: str):
    return gmol_from_smarts(smarts)[0]


def _molecule(smiles: str):
    return cmol_from_smiles(smiles)[0]


@pytest.mark.parametrize(
    "smiles",
    [
        "CCO",
        "C1CC1",
        "CC(=O)OC",
        "c1ccccc1",
        "[NH4+]",
        "[CH3:7][OH:8]",
        "CC.CN",
    ],
)
def test_generate_smiles_round_trip_matches_original(smiles: str) -> None:
    original = _molecule(smiles)
    generated = smiles_from_cmol(original)
    rebuilt = _molecule(generated)

    assert generated
    assert has_match(gmol_from_cmol(original), gmol_from_cmol(rebuilt))
    assert has_match(gmol_from_cmol(rebuilt), gmol_from_cmol(original))


def test_generate_smiles_from_retro_precursors_is_parseable_and_equivalent() -> None:
    product = _molecule("CC(=O)OC")
    reaction = parse_reaction_smarts(
        "[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])-[O:3].[C:4]"
    )

    outcomes = generate_precursors(product, reaction)
    assert outcomes

    precursor_smiles = [
        smiles_from_cmol(precursor) for precursor in outcomes[0].precursors
    ]
    assert len(precursor_smiles) == 2

    for precursor, smiles in zip(
        outcomes[0].precursors, precursor_smiles, strict=False
    ):
        rebuilt = _molecule(smiles)
        assert_same_molecule(precursor, rebuilt)


def test_generate_smiles_from_single_bit_masks_without_concrete_arrays() -> None:
    atom_data = [
        {
            "Z": z_mask_for_numbers([6]),
            "A": ALIPHATIC,
            "C": mask_for_charge(0),
            "D": bucket_mask_exact(0),
            "H": bucket_mask_exact(0),
            "h": bucket_mask_exact(0),
            "v": bucket_mask_exact(0),
            "X": bucket_mask_exact(0),
            "R": bucket_mask_exact(0),
            "r": bucket_mask_exact(0),
            "x": bucket_mask_exact(0),
        }
    ]
    graph = GMol(
        atom_masks=atom_masks_from_props(atom_data),
        bonds=np.zeros((1, 1), dtype=np.uint8),
        atom_map=np.zeros(1, dtype=np.int32),
        atom_group=np.zeros(1, dtype=np.int32),
        atom_predicate=np.full(1, -1, dtype=np.int32),
        predicates=(),
    )
    # The masks declare H0, so this is atomic carbon and `[C]` is exactly right.
    assert smiles_from_cmol(cmol_from_gmol(graph)) == "[C]"


def test_generate_smiles_joins_disconnected_components_with_dot() -> None:
    generated = smiles_from_cmol(_molecule("CC.CN"))
    assert "." in generated
    assert len(generated.split(".")) == 2


def test_generate_smiles_rejects_ambiguous_atom_masks() -> None:
    pattern = _gmol("[#6,#7]")
    with pytest.raises(ValueError, match="single-bit atomic number mask"):
        cmol_from_gmol(pattern)


def test_generate_smiles_rejects_any_bond_type() -> None:
    pattern = _gmol("[C]~[C]")
    with pytest.raises(ValueError, match="exact bond types"):
        cmol_from_gmol(pattern)


def test_bond_symbol_rejects_bond_none() -> None:
    with pytest.raises(ValueError, match="BOND_NONE"):
        _bond_symbol(int(BondKind.NONE), both_aromatic=False)


def test_bond_symbol_emits_aromatic_forms() -> None:
    aromatic = int(BondKind.DELOCALIZED)
    assert _bond_symbol(aromatic, both_aromatic=True) == ""
    assert _bond_symbol(aromatic, both_aromatic=False) == ":"
    assert _bond_symbol(int(BondKind.SINGLE), both_aromatic=True) == "-"

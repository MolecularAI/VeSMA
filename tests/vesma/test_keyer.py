from __future__ import annotations

import pytest

from vesma import cmol_from_smiles
from vesma.keyer import (
    aromatic_wl_keyer,
    cmol_rdkit_smiles_keyer,
    cmol_smiles_keyer,
    rdkit_inchi_keyer,
    rdkit_inchikey_keyer,
    rdkit_smiles_keyer,
    short_ring_wl_keyer,
)


def test_short_ring_wl_codec_identifies_three_to_eight_schema() -> None:
    codec = short_ring_wl_keyer

    assert codec.name == "short-ring-wl3-8-zchd-v6-r8"
    assert ":short-ring-wl3-8-zchd-v6:rounds=8" in codec.version()


def test_aromatic_wl_codec_name_and_version() -> None:
    codec = aromatic_wl_keyer

    assert codec.name == "aromatic-wl-zchd-v5-r8"
    assert ":aromatic-wl-zchd-v5:rounds=8" in codec.version()


def test_keyer_call_matches_encode() -> None:
    molecule = cmol_from_smiles("CCO")[0]
    codec = short_ring_wl_keyer

    assert codec(molecule) == codec.encode(molecule)


@pytest.mark.parametrize(
    "keyer",
    [
        short_ring_wl_keyer,
        aromatic_wl_keyer,
        cmol_smiles_keyer,
        cmol_rdkit_smiles_keyer,
    ],
)
def test_cmol_keyer_molecule_and_smiles_paths_agree(keyer) -> None:
    smiles = "CC(O)Cl"
    molecule = cmol_from_smiles(smiles)[0]

    assert keyer.encode(molecule) == keyer.from_smiles(smiles)


@pytest.mark.parametrize(
    "keyer",
    [
        rdkit_smiles_keyer,
        rdkit_inchi_keyer,
        rdkit_inchikey_keyer,
    ],
)
def test_rdkit_keyer_molecule_and_smiles_paths_agree(keyer) -> None:
    Chem = pytest.importorskip("rdkit.Chem")
    smiles = "CC(O)Cl"
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None

    assert keyer.encode(molecule) == keyer.from_smiles(smiles)


@pytest.mark.parametrize(
    "keyer",
    [
        rdkit_smiles_keyer,
        rdkit_inchi_keyer,
        rdkit_inchikey_keyer,
    ],
)
def test_rdkit_keyers_do_not_mutate_input(keyer) -> None:
    Chem = pytest.importorskip("rdkit.Chem")
    molecule = Chem.MolFromSmiles("[CH3:7][C@H:8]([OH:9])[Cl:10]")
    assert molecule is not None
    before = Chem.MolToSmiles(molecule, canonical=False, isomericSmiles=True)

    keyer.encode(molecule)

    after = Chem.MolToSmiles(molecule, canonical=False, isomericSmiles=True)
    assert after == before


def test_cmol_and_rdkit_canonical_smiles_keyers_agree() -> None:
    Chem = pytest.importorskip("rdkit.Chem")
    smiles = "[CH3:7][C@H:8]([OH:9])[Cl:10]"
    rdkit_molecule = Chem.MolFromSmiles(smiles)
    assert rdkit_molecule is not None
    cmol = cmol_from_smiles(smiles)[0]

    assert cmol_rdkit_smiles_keyer.encode(cmol) == rdkit_smiles_keyer.encode(
        rdkit_molecule
    )

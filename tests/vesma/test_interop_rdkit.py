from __future__ import annotations

import csv

import numpy as np
import pytest

from tests.datadirs import external_data_root
from vesma import (
    CMol,
    cmol_from_gmol,
    cmol_from_mol,
    cmol_from_smiles,
    gmol_from_cmol,
    gmol_from_mol,
    gmol_from_smarts,
    mol_from_cmol,
)
from vesma.aromaticity import OrphanDelocalizedBondError
from vesma.bonds import (
    BOND_CONCRETE_KINDS,
    BOND_TOPOLOGY_ANY,
    BOND_TOPOLOGY_MASK,
    QBOND_DEFAULT,
    BondKind,
)
from vesma.gmol_atoms import ALIPHATIC, AROMATIC

Chem = pytest.importorskip("rdkit.Chem")

CORPUS_PATH = (
    external_data_root() / "external" / "remix" / "retro-benchmark" / "molecules.tsv"
)


def load_corpus_smiles(limit: int) -> list[str]:
    with CORPUS_PATH.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [row["smiles"] for _, row in zip(range(limit), reader, strict=False)]


def assert_cmol_equal(left: CMol, right: CMol, *, implicit_h: bool = True) -> None:
    names = ("Z", "A", "C", "D", "H", "h", "v", "X", "R", "r", "x", "atom_map")
    if not implicit_h:
        names = tuple(name for name in names if name != "h")
    for name in names:
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
    np.testing.assert_array_equal(left.bonds, right.bonds)


def assert_gmol_equal(left, right) -> None:
    for name in ("Z_lo", "Z_hi", "A", "C", "D", "H", "h", "v", "X", "R", "r", "x"):
        np.testing.assert_array_equal(
            getattr(left.atom_masks, name),
            getattr(right.atom_masks, name),
        )
    np.testing.assert_array_equal(left.bonds, right.bonds)
    np.testing.assert_array_equal(left.atom_map, right.atom_map)


def test_cmol_from_mol_copies_aromatic_and_charge_properties() -> None:
    phenol = cmol_from_mol(Chem.MolFromSmiles("c1ccccc1O"))
    assert np.all(phenol.A[:6] == AROMATIC)
    assert int(phenol.A[6]) == int(ALIPHATIC)
    assert np.count_nonzero(phenol.bonds & int(BondKind.DELOCALIZED)) == 12
    # Aromatic bonds are the pure delocalized class (no Kekule order bit), so the
    # only nonzero order is the single c-O bond.
    assert set(np.unique(phenol.bonds & np.uint8(BOND_CONCRETE_KINDS))) == {0, 1}
    assert set(np.unique(phenol.bonds)) == {
        0,
        int(BondKind.SINGLE),
        int(BondKind.DELOCALIZED),
    }
    assert int(phenol.H[6]) == 1
    assert int(phenol.h[6]) == 1

    salt = cmol_from_mol(Chem.MolFromSmiles("[NH4+].[Cl-]"))
    assert tuple(int(value) for value in salt.C) == (1, -1)
    assert tuple(int(value) for value in salt.H) == (4, 0)
    assert np.count_nonzero(salt.bonds) == 0


@pytest.mark.parametrize(
    "smiles",
    ["CCO", "[NH4+].[Cl-]", "c1ccccc1O", "c1[nH]ccc1", "C[N+](C)(C)C"],
)
def test_mol_from_cmol_roundtrips_through_cmol_from_mol(smiles: str) -> None:
    """Molecular state survives the round trip; the implicit/explicit split does not.

    ``h`` records how the input happened to be written, not what the molecule is, and the
    export deliberately does not carry it: it writes every hydrogen explicitly so that
    vesma's total is what RDKit reads, instead of handing part of the count back to
    RDKit's valence model. ``H`` is the state, and it is preserved.
    """
    mol = Chem.MolFromSmiles(smiles)
    original = cmol_from_mol(mol)
    rebuilt = cmol_from_mol(mol_from_cmol(original))
    assert_cmol_equal(rebuilt, original, implicit_h=False)
    assert np.array_equal(rebuilt.H, original.H)


def test_export_states_vesmas_hydrogen_rather_than_deferring_to_rdkit() -> None:
    """A precursor whose hydrogen count RDKit's valence model would not reproduce.

    Vesma gives this phosphorus two hydrogens at graph valence five. Exporting only the
    hydrogens vesma called explicit, with implicit hydrogen left enabled, wrote ``O=PO``
    -- which reads back as a phosphorus with none. Input molecules never showed this,
    because their ``h`` comes from RDKit in the first place.
    """
    phosphinic, _ = cmol_from_smiles("O=[PH2]O")
    rebuilt = cmol_from_mol(mol_from_cmol(phosphinic))

    assert np.array_equal(rebuilt.H, phosphinic.H)
    assert Chem.MolToSmiles(mol_from_cmol(phosphinic)) == "O=[PH2]O"


@pytest.mark.parametrize(
    "smiles",
    [
        "c1ccccc1",
        "c1ccncc1",
        "CCN(CC)CCN/C(C)=C1\\C(=O)NC(=O)N(c2ccccc2OC)C1=O",
        "CCCCCC/C(C)=C1/SC(=S)NC1=O",
    ],
)
def test_cmol_from_gmol_exact_roundtrip_preserves_concrete_molecule(
    smiles: str,
) -> None:
    original = cmol_from_mol(Chem.MolFromSmiles(smiles))
    query = gmol_from_cmol(original)
    present = query.bonds != 0
    assert np.all((query.bonds[present] & BOND_TOPOLOGY_MASK) == BOND_TOPOLOGY_ANY)
    rebuilt = cmol_from_gmol(query)

    assert_cmol_equal(rebuilt, original)
    assert Chem.MolToSmiles(mol_from_cmol(rebuilt))


def test_cmol_from_mol_rejects_query_mol() -> None:
    with pytest.raises(ValueError, match="concrete RDKit Mol"):
        cmol_from_mol(Chem.MolFromSmarts("C=O"))


@pytest.mark.parametrize("smarts", ["[C,N;H1]", "C=O", "C~O", "C/O"])
def test_gmol_from_mol_matches_explicit_query_smarts(smarts: str) -> None:
    mol = Chem.MolFromSmarts(smarts)
    graph, info = gmol_from_mol(mol)
    expected, expected_info = gmol_from_smarts(smarts)
    assert_gmol_equal(graph, expected)
    assert info.messages == expected_info.messages
    assert info.smarts_string == Chem.MolToSmarts(mol)


def test_gmol_from_mol_reports_mass_conversion_info() -> None:
    mol = Chem.MolFromSmarts("[13C]")
    graph, info = gmol_from_mol(mol)
    expected, expected_info = gmol_from_smarts("[13C]")
    assert_gmol_equal(graph, expected)
    assert info.messages == expected_info.messages
    assert info.smarts_string == Chem.MolToSmarts(mol)


@pytest.mark.parametrize(
    "smarts",
    [
        "C@C",
        "C!@C",
        "C-@C",
        "C=@C",
        "C#@C",
        "C:@C",
        "C-!@C",
        "C=!@C",
        "C#!@C",
        "C:!@C",
        "C=,:C",
        "C-;!@C",
    ],
)
def test_gmol_from_mol_preserves_supported_query_bonds(smarts: str) -> None:
    mol = Chem.MolFromSmarts(smarts)
    graph, _ = gmol_from_mol(mol)
    expected, _ = gmol_from_smarts(smarts)
    assert_gmol_equal(graph, expected)


def test_gmol_from_mol_copies_aromatic_query_bonds_from_rdkit() -> None:
    mol = Chem.MolFromSmarts("c1ccccc1O")
    graph, info = gmol_from_mol(mol)
    # All bonds here are omitted/default SMARTS bonds (the 6 aromatic ring bonds and the
    # c-O bond), so they parse to the QBOND_DEFAULT accepted-class mask (single|aromatic).
    # 7 undirected bonds -> 14 nonzero entries in the symmetric matrix.
    assert np.count_nonzero(graph.bonds == QBOND_DEFAULT) == 14
    assert np.count_nonzero(graph.bonds) == 14
    assert info.smarts_string == Chem.MolToSmarts(mol)


@pytest.mark.parametrize(
    ("smiles", "expected_v", "expected_h", "expected_h_implicit"),
    [
        ("c1ccccc1", [4, 4, 4, 4, 4, 4], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]),
        ("c1ccsc1", [4, 4, 4, 2, 4], [1, 1, 1, 0, 1], [1, 1, 1, 0, 1]),
        ("c1cc[nH]c1", [4, 4, 4, 3, 4], [1, 1, 1, 1, 1], [1, 1, 1, 0, 1]),
    ],
)
def test_cmol_from_smarts_assigns_consistent_aromatic_valence(
    smiles: str,
    expected_v: list[int],
    expected_h: list[int],
    expected_h_implicit: list[int],
) -> None:
    molecule = cmol_from_smiles(smiles)[0]

    assert molecule.v.tolist() == expected_v
    assert molecule.H.tolist() == expected_h
    assert molecule.h.tolist() == expected_h_implicit


@pytest.mark.parametrize(
    ("smiles", "expected_v", "expected_h", "expected_h_implicit"),
    [
        ("c1ccccc1", [4, 4, 4, 4, 4, 4], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]),
        ("c1ccsc1", [4, 4, 4, 2, 4], [1, 1, 1, 0, 1], [1, 1, 1, 0, 1]),
        ("c1cc[nH]c1", [4, 4, 4, 3, 4], [1, 1, 1, 1, 1], [1, 1, 1, 0, 1]),
    ],
)
def test_cmol_from_mol_uses_rdkit_aromatic_valence_semantics(
    smiles: str,
    expected_v: list[int],
    expected_h: list[int],
    expected_h_implicit: list[int],
) -> None:
    molecule = cmol_from_mol(Chem.MolFromSmiles(smiles))

    assert molecule.v.tolist() == expected_v
    assert molecule.H.tolist() == expected_h
    assert molecule.h.tolist() == expected_h_implicit


def test_cmol_from_smarts_rejects_an_open_delocalized_fragment() -> None:
    """A delocalized bond with no circuit to carry it asserts something false.

    It is not quietly resolved to a single bond: that would invent a bond order the
    input never gave and hand back a different molecule than the one asked for.
    """
    with pytest.raises(OrphanDelocalizedBondError):
        cmol_from_smiles("cc")


@pytest.mark.parametrize(
    "smiles",
    [
        "Clc1ccc(cc1)C=2[N+](=C(SC2)Nc3c(cccc3C)C)C",
        "Fc1ccc(cc1)C2=Nc3c(cccc3)C(=O)N2N",
        "O=C1N(N=Cc2ccccc2)C(=O)C=3Nc4c(cc(OC)cc4)C3N1",
        "S1C(=O)NC=2S[C@H](C([O-])=O)[C@H]3C(=O)Oc4c(cccc4)[C@H]3C12",
        "[O-]C(=O)[C@H](c1cc2c(Oc3ncccc3C2)cc1)C",
    ],
)
def test_cmol_from_smarts_perceives_kekule_aromaticity_like_rdkit(
    smiles: str,
) -> None:
    from_smarts = cmol_from_smiles(smiles)[0]
    from_rdkit = cmol_from_mol(Chem.MolFromSmiles(smiles))

    np.testing.assert_array_equal(from_smarts.A, from_rdkit.A)


@pytest.mark.parametrize("smiles", load_corpus_smiles(limit=100))
def test_exact_roundtrip_canonicalizes_corpus_sample(smiles: str) -> None:
    original = cmol_from_mol(Chem.MolFromSmiles(smiles))
    rebuilt = cmol_from_gmol(gmol_from_cmol(original))

    assert_cmol_equal(rebuilt, original)
    assert Chem.MolToSmiles(mol_from_cmol(rebuilt))

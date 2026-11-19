from __future__ import annotations

import csv
from collections import defaultdict

import pytest

from tests.datadirs import external_data_root
from vesma import (
    cmol_canonicalish_smiles,
    cmol_from_smiles,
    rdkit_canonical_smiles,
    rdkit_inchi,
    rdkit_inchi_key,
)
from vesma.canonicalization import reorder_cmol

CORPUS_PATH = (
    external_data_root() / "external" / "remix" / "retro-benchmark" / "molecules.tsv"
)


def molecule(smiles: str):
    return cmol_from_smiles(smiles)[0]


@pytest.mark.parametrize(
    "smiles",
    [
        "CCO",
        "CC(C)O",
        "C1CC1",
        "c1ccccc1O",
        "[CH3:7][OH:8]",
        "CC.CN",
    ],
)
def test_cmol_canonicalish_smiles_is_stable_under_atom_reordering(
    smiles: str,
) -> None:
    original = molecule(smiles)
    reordered = reorder_cmol(
        original,
        tuple(reversed(range(original.atom_count))),
    )

    assert cmol_canonicalish_smiles(reordered) == cmol_canonicalish_smiles(original)


@pytest.mark.parametrize(
    "smiles",
    [
        "CCO",
        "CC(C)O",
        "C1CC1",
        "c1ccccc1O",
        "[NH4+]",
        "[CH3:7][OH:8]",
        "CC.CN",
    ],
)
def test_cmol_canonicalish_smiles_round_trips_through_parser(smiles: str) -> None:
    token = cmol_canonicalish_smiles(molecule(smiles))
    rebuilt = molecule(token)

    assert cmol_canonicalish_smiles(rebuilt) == token


def test_cmol_canonicalish_smiles_is_stable_for_symmetric_corpus_molecule() -> None:
    original = molecule("c1ccc(NCc2cc(CNc3ccccc3)cc(CNc3ccccc3)c2)cc1")
    expected = cmol_canonicalish_smiles(original)
    permutations = [
        tuple(reversed(range(original.atom_count))),
        tuple(range(0, original.atom_count, 2))
        + tuple(range(1, original.atom_count, 2)),
        tuple((idx * 7) % original.atom_count for idx in range(original.atom_count)),
    ]

    for permutation in permutations:
        reordered = reorder_cmol(original, permutation)
        assert cmol_canonicalish_smiles(reordered) == expected


def test_rdkit_canonical_smiles_strips_maps_and_stereo() -> None:
    pytest.importorskip("rdkit.Chem")

    mapped = molecule("[CH3:7][C@H:8]([OH:9])[Cl:10]")
    plain = molecule("CC(O)Cl")

    assert rdkit_canonical_smiles(mapped) == rdkit_canonical_smiles(plain)


def test_rdkit_inchi_tokens_strip_maps_and_stereo() -> None:
    pytest.importorskip("rdkit.Chem.rdinchi")

    mapped = molecule("[CH3:7][C@H:8]([OH:9])[Cl:10]")
    plain = molecule("CC(O)Cl")

    assert rdkit_inchi(mapped) == rdkit_inchi(plain)
    assert rdkit_inchi_key(mapped) == rdkit_inchi_key(plain)


@pytest.mark.slow
def test_cmol_canonicalish_smiles_matches_rdkit_identity_groups_on_corpus() -> None:
    Chem = pytest.importorskip("rdkit.Chem")
    from vesma import cmol_from_mol

    rdkit_to_cmol: dict[str, set[str]] = defaultdict(set)
    cmol_to_rdkit: dict[str, set[str]] = defaultdict(set)

    with CORPUS_PATH.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rd_mol = Chem.MolFromSmiles(row["smiles"])
            if rd_mol is None:
                continue
            molecule_from_rdkit = cmol_from_mol(rd_mol)
            rdkit_token = rdkit_canonical_smiles(molecule_from_rdkit)
            cmol_token = cmol_canonicalish_smiles(molecule_from_rdkit)
            rdkit_to_cmol[rdkit_token].add(cmol_token)
            cmol_to_rdkit[cmol_token].add(rdkit_token)

    splits = sum(len(values) - 1 for values in rdkit_to_cmol.values())
    clashes = sum(len(values) - 1 for values in cmol_to_rdkit.values())
    checked = len(rdkit_to_cmol)

    assert checked > 0
    assert splits + clashes <= max(1, checked // 1000)

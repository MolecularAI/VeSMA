from __future__ import annotations

import gzip

import numpy as np
import pytest

from vesma import (
    aromatic_wl_molecule_id,
    cmol_from_smiles,
    short_ring_wl_molecule_id,
    short_ring_wl_molecule_ids,
)
from vesma.aromaticity import perceive_aromaticity
from vesma.bonds import BondKind
from vesma.canonicalization import reorder_cmol
from vesma.identity import (
    SHORT_RING_WL_SCHEMA,
    aromatic_wl_token_from_smiles,
    identity_seed_from_cmol,
    identity_seed_from_smiles,
    short_ring_bond_classes,
    short_ring_wl_token_from_smiles,
    total_hydrogens_for_identity,
)


def _vesma_rejects_an_input_aromatic_ring(aromatic_smiles: str) -> bool:
    """True when perception drops an aromatic bond present in the input.

    A localization-free codec cannot recover Kekule orders from an aromatic
    SMILES, so a ring that the input flags aromatic but vesma perceives as
    non-aromatic leaves "orphaned" aromatic edges. The aromatic WL codec is only
    input-invariant outside this set (see the corpus test below).
    """
    seed = identity_seed_from_smiles(aromatic_smiles)
    aromatic_bit = np.uint8(BondKind.DELOCALIZED)
    source_aromatic = (seed.bonds & aromatic_bit) != 0
    _, perceived, _ = perceive_aromaticity(
        Z=seed.Z,
        C=seed.C,
        explicit_h=seed.explicit_h,
        implicit_h=seed.implicit_h,
        bonds=seed.bonds,
    )
    perceived_aromatic = (perceived & aromatic_bit) != 0
    return bool(np.any(source_aromatic & ~perceived_aromatic))


def molecule(smiles: str):
    return cmol_from_smiles(smiles)[0]


@pytest.mark.parametrize(
    "smiles",
    ["CCO", "CC(C)O", "C1CC1", "c1ccccc1O", "CC.CN"],
)
def test_short_ring_wl_is_stable_under_atom_reordering(smiles: str) -> None:
    original = molecule(smiles)
    reordered = reorder_cmol(
        original,
        tuple(reversed(range(original.atom_count))),
    )

    assert short_ring_wl_molecule_id(reordered) == short_ring_wl_molecule_id(original)


def test_short_ring_wl_ignores_component_order_and_atom_maps() -> None:
    original = molecule("[CH3:7][OH:8].N")
    reordered = reorder_cmol(original, (2, 0, 1))

    assert short_ring_wl_molecule_id(reordered) == short_ring_wl_molecule_id(
        molecule("N.CO")
    )


def test_short_ring_wl_distinguishes_hydrogen_charge_and_chain_bond_order() -> None:
    assert short_ring_wl_molecule_id(molecule("N")) != short_ring_wl_molecule_id(
        molecule("[NH4+]")
    )
    assert short_ring_wl_molecule_id(molecule("CO")) != short_ring_wl_molecule_id(
        molecule("C=O")
    )


@pytest.mark.parametrize(
    ("aromatic", "kekule"),
    [
        ("c1ccccc1", "C1=CC=CC=C1"),
        ("c1ncccc1", "C1=NC=CC=C1"),
        ("c1ncc[nH]1", "C1=NC=C[NH]1"),
    ],
)
def test_wl_codecs_match_aromatic_and_kekule_forms(
    aromatic: str,
    kekule: str,
) -> None:
    assert short_ring_wl_token_from_smiles(aromatic) == short_ring_wl_token_from_smiles(
        kekule
    )
    assert aromatic_wl_token_from_smiles(aromatic) == aromatic_wl_token_from_smiles(
        kekule
    )


@pytest.mark.parametrize("ring_size", [3, 4, 5, 6, 7, 8])
def test_short_ring_wl_collapses_single_and_double_ring_bonds(
    ring_size: int,
) -> None:
    smiles = "C1=" + "C" * (ring_size - 2) + "C1"
    classes = short_ring_bond_classes(molecule(smiles).bonds)
    edge_classes = classes[np.triu(classes != 0, k=1)]

    assert set(edge_classes) == {int(BondKind.DELOCALIZED)}


@pytest.mark.parametrize("ring_size", [9])
def test_short_ring_wl_keeps_bond_orders_outside_short_rings(
    ring_size: int,
) -> None:
    smiles = "C1=" + "C" * (ring_size - 2) + "C1"
    classes = short_ring_bond_classes(molecule(smiles).bonds)
    edge_classes = set(classes[np.triu(classes != 0, k=1)])

    assert edge_classes == {
        int(BondKind.SINGLE),
        int(BondKind.DOUBLE),
    }


@pytest.mark.parametrize("smiles", ["C1#CC1", "C1#CCC1", "C1#CCCC1"])
def test_short_ring_wl_keeps_triple_bonds_exact_in_short_rings(
    smiles: str,
) -> None:
    classes = short_ring_bond_classes(molecule(smiles).bonds)
    edge_classes = set(classes[np.triu(classes != 0, k=1)])

    assert edge_classes == {
        int(BondKind.TRIPLE),
        int(BondKind.DELOCALIZED),
    }


@pytest.mark.parametrize(
    ("aromatic", "kekule"),
    [
        ("c1c[cH+]1", "C1=C[CH+]1"),
        ("CCOc1c(Nc2ccncc2)c(=O)c1=O", "CCOC1=C(NC2=CC=NC=C2)C(=O)C1=O"),
        (
            "CCOc1c(Nc2cccc3[nH]c(=O)oc23)c(=O)c1=O",
            "CCOC1=C(NC2=CC=CC3=C2OC(=O)N3)C(=O)C1=O",
        ),
        ("CCOc1c(OCC)c(=O)c1=O", "CCOC1=C(OCC)C(=O)C1=O"),
    ],
)
def test_short_ring_wl_matches_small_aromatic_kekule_forms(
    aromatic: str,
    kekule: str,
) -> None:
    assert short_ring_wl_token_from_smiles(aromatic) == short_ring_wl_token_from_smiles(
        kekule
    )


@pytest.mark.parametrize(
    "smiles",
    [
        "[CH-]1C=CC=C1",  # cyclopentadienyl anion (Kekule)
        "[cH-]1cccc1",  # cyclopentadienyl anion (aromatic)
        "[cH+]1cccccc1",  # tropylium
        "c1c[cH+]1",  # cyclopropenyl cation
    ],
)
def test_identity_h_matches_perception_on_charged_aromatic_rings(smiles: str) -> None:
    """WL/identity and perception share one hydrogen model, so a charged aromatic ring
    carbon is sp2 (three-atom sigma frame) and carries the degree-derived hydrogen
    (3 - degree = 1 here), not the zero the old charge-guarded identity model produced.
    """
    cmol = molecule(smiles)
    seed = identity_seed_from_cmol(cmol)
    _, _, perception_h = perceive_aromaticity(
        Z=seed.Z,
        C=seed.C,
        explicit_h=seed.explicit_h,
        implicit_h=seed.implicit_h,
        bonds=seed.bonds,
    )
    identity_h = total_hydrogens_for_identity(seed)

    assert np.array_equal(identity_h, perception_h)
    heavy_degree = (cmol.bonds != 0).sum(axis=1)
    charged_ring_carbon = (cmol.C != 0) & (cmol.Z == 6)
    assert charged_ring_carbon.any()
    assert np.all(
        identity_h[charged_ring_carbon] == 3 - heavy_degree[charged_ring_carbon]
    )
    assert np.all(identity_h[charged_ring_carbon] == 1)


@pytest.mark.parametrize(
    ("saturated", "unsaturated"),
    [
        ("C1CC1", "C1=CC1"),
        ("C1CCC1", "C1=CCC1"),
    ],
)
def test_short_ring_wl_distinguishes_saturated_small_rings(
    saturated: str,
    unsaturated: str,
) -> None:
    assert short_ring_wl_token_from_smiles(
        saturated
    ) != short_ring_wl_token_from_smiles(unsaturated)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("FC1=C(Cl)C1Br", "FC1C(Cl)=C1Br"),
        ("FC1=C(Cl)C(Br)C1I", "FC1C(Cl)=C(Br)C1I"),
    ],
)
def test_short_ring_wl_retains_substituted_small_ring_bond_placement(
    left: str,
    right: str,
) -> None:
    assert short_ring_wl_token_from_smiles(left) != short_ring_wl_token_from_smiles(
        right
    )


@pytest.mark.parametrize(
    "smiles",
    [
        "C",
        "[CH4]",
        "N",
        # A bare `[O-]` is a radical anion, so it is not a molecule this model builds;
        # methoxide is the closed-shell way to exercise the same charged-oxygen encoding.
        "C[O-]",
        "[OH-]",
        "O=[N+]([O-])c1ccccc1",
        "c1ccccc1",
        "C1=CC=CC=C1",
        "Clc1cnc2nc(Cl)c(Cl)nc2c1",
        "Cn1cc(-c2cnc3nnn(CC4CCCN(C(=O)O)C4)c3n2)cn1",
        "CC.CN",
    ],
)
def test_minimal_smiles_path_matches_full_cmol_encoding(smiles: str) -> None:
    assert short_ring_wl_token_from_smiles(smiles) == short_ring_wl_molecule_id(
        molecule(smiles)
    )


def test_short_ring_wl_distinguishes_explicit_hydrogen_on_charged_atoms() -> None:
    assert short_ring_wl_token_from_smiles("[O-]") != short_ring_wl_token_from_smiles(
        "[OH-]"
    )


def test_wl_schemas_are_explicitly_distinct() -> None:
    value = molecule("CCO")

    assert SHORT_RING_WL_SCHEMA == 6
    assert short_ring_wl_molecule_id(value) != aromatic_wl_molecule_id(value)


def test_short_ring_wl_many_matches_scalar_tokens() -> None:
    molecules = [molecule("CC"), molecule("CO"), molecule("C=O")]

    assert short_ring_wl_molecule_ids(molecules) == [
        short_ring_wl_molecule_id(value) for value in molecules
    ]


def test_short_ring_wl_accepts_large_acyclic_molecule() -> None:
    token = short_ring_wl_molecule_id(molecule("C" * 65))

    assert isinstance(token[0], int)
    assert isinstance(token[1], int)


def test_short_ring_wl_handles_large_cyclic_molecule() -> None:
    # The cycle enumerator has no 64-atom limit (arbitrary-precision masks), so a
    # >64-atom molecule is classified, not rejected. A 65-membered macrocycle has no
    # ring of size <= 8, so no bond is reclassified as a short (aromatic-class) ring.
    atom_count = 65
    bonds = np.zeros((atom_count, atom_count), dtype=np.uint8)
    for left in range(atom_count):
        right = (left + 1) % atom_count
        bonds[left, right] = np.uint8(BondKind.SINGLE)
        bonds[right, left] = np.uint8(BondKind.SINGLE)

    classes = short_ring_bond_classes(bonds)

    assert not np.any((classes & np.uint8(BondKind.DELOCALIZED)) != 0)


@pytest.mark.slow
def test_wl_aromatic_kekule_splits_on_500_rdkit_stock_pairs() -> None:
    Chem = pytest.importorskip("rdkit.Chem")
    from tests.datadirs import external_data_root

    stock_path = (
        external_data_root() / "external" / "paroutes" / "processed" / "n1-stock.smi.gz"
    )
    checked = 0
    short_ring_splits: list[str] = []
    aromatic_splits: list[str] = []

    with gzip.open(stock_path, "rt") as handle:
        for line in handle:
            molecule = Chem.MolFromSmiles(line.split()[0])
            if molecule is None or not any(
                atom.GetIsAromatic() for atom in molecule.GetAtoms()
            ):
                continue
            aromatic = Chem.MolToSmiles(
                molecule,
                canonical=True,
                isomericSmiles=False,
            )
            kekule = Chem.MolToSmiles(
                molecule,
                canonical=True,
                isomericSmiles=False,
                kekuleSmiles=True,
            )
            if short_ring_wl_token_from_smiles(
                aromatic
            ) != short_ring_wl_token_from_smiles(kekule):
                short_ring_splits.append(aromatic)
            if aromatic_wl_token_from_smiles(aromatic) != aromatic_wl_token_from_smiles(
                kekule
            ) and not _vesma_rejects_an_input_aromatic_ring(aromatic):
                aromatic_splits.append(aromatic)
            checked += 1
            if checked == 500:
                break

    assert checked == 500
    # The default short-ring codec is aromaticity-independent, so it is bit-exact
    # across Kekule and aromatic encodings of every molecule.
    assert short_ring_splits == []
    # The opt-in aromatic codec is bit-exact wherever vesma's perception accepts
    # the input's aromatic rings; it may legitimately differ only when vesma
    # rejects a ring the input flagged aromatic (lossy, no localizer).
    assert aromatic_splits == []

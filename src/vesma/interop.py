from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pyparsing as pp

from .bonds import (
    QBOND_DEFAULT,
    BondKind,
    bond_flags_from_smarts_token,
)
from .cmol import CMol
from .cmol_builder import cmol_from_seed_data
from .gmol import GMol
from .gmol_atoms import AROMATIC
from .gmol_builder import SmartsConversionInfo, gmol_from_normalized_smarts
from .parsing.errors import SmartsError
from .parsing.grammar_custom import parse_smarts_atom
from .parsing.ir import NormalizedSmartsAtom, NormalizedSmartsMol
from .parsing.normalizer import normalize_smarts_atom

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Atom as RdAtom
    from rdkit.Chem.rdchem import Bond as RdBond
    from rdkit.Chem.rdchem import Mol as RdMol
else:
    RdAtom = Any
    RdBond = Any
    RdMol = Any


def cmol_from_mol(mol: RdMol) -> CMol:
    """Import an RDKit molecule through vesma's single concrete-molecule builder, so a
    molecule read from RDKit is represented identically to one parsed from SMILES:
    interop only supplies atoms (Z, charge, the explicit/implicit hydrogen split) and
    bond classes; aromaticity, degree, valence, ring info and total H are all
    re-perceived by vesma (one consistent model everywhere)."""
    _require_concrete_rdkit_mol(mol)
    atoms = list(mol.GetAtoms())
    bonds = _concrete_bonds_from_rdkit_mol(mol)

    Z = np.array([atom.GetAtomicNum() for atom in atoms], dtype=np.int16)
    M = np.array([atom.GetIsotope() for atom in atoms], dtype=np.int16)
    C = np.array([atom.GetFormalCharge() for atom in atoms], dtype=np.int8)
    total_h = np.array([atom.GetTotalNumHs() for atom in atoms], dtype=np.int16)
    implicit_h = np.array([atom.GetNumImplicitHs() for atom in atoms], dtype=np.int16)
    explicit_h = np.maximum(total_h - implicit_h, 0).astype(np.int16)
    atom_map = np.array([atom.GetAtomMapNum() for atom in atoms], dtype=np.int32)

    return cmol_from_seed_data(
        Z=Z,
        M=M,
        C=C,
        explicit_h=explicit_h,
        implicit_h=implicit_h,
        atom_map=atom_map,
        bonds=bonds,
    )


def mol_from_cmol(molecule: CMol) -> RdMol:
    from rdkit import Chem

    if molecule.atom_count == 0:
        return Chem.Mol()

    rw_mol = Chem.RWMol()
    for atom_idx in range(molecule.atom_count):
        rd_atom = Chem.Atom(int(molecule.Z[atom_idx]))
        rd_atom.SetIsotope(int(molecule.M[atom_idx]))
        rd_atom.SetFormalCharge(int(molecule.C[atom_idx]))
        rd_atom.SetIsAromatic(bool(molecule.A[atom_idx] == AROMATIC))
        # Vesma's total hydrogen is authoritative, so state it rather than delegate part
        # of it back to RDKit's valence model. Exporting only ``H - h`` and leaving
        # implicit hydrogen enabled means RDKit re-derives the rest under its own rules,
        # and silently wins wherever the two models disagree: a precursor whose added
        # phosphorus vesma gives two hydrogens is written `O=PO`, which reads back as
        # none. Input molecules never showed it -- their `h` comes from RDKit in the
        # first place -- so it surfaced only on reaction-built precursors.
        rd_atom.SetNumExplicitHs(int(molecule.H[atom_idx]))
        rd_atom.SetNoImplicit(True)
        atom_map = int(molecule.atom_map[atom_idx])
        if atom_map > 0:
            rd_atom.SetAtomMapNum(atom_map)
        rw_mol.AddAtom(rd_atom)

    for left in range(molecule.atom_count):
        for right in range(left + 1, molecule.atom_count):
            bond_type = int(molecule.bonds[left, right])
            if bond_type == int(BondKind.NONE):
                continue
            rd_bond_type, aromatic = _rdkit_bond_spec_from_cmol(
                left=left,
                right=right,
                molecule=molecule,
            )
            rw_mol.AddBond(left, right, rd_bond_type)
            rd_bond = rw_mol.GetBondBetweenAtoms(left, right)
            rd_bond.SetIsAromatic(aromatic)

    mol = rw_mol.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def gmol_from_mol(mol: RdMol) -> tuple[GMol, SmartsConversionInfo]:
    from rdkit import Chem

    bonds = _query_bonds_from_rdkit_mol(mol)
    atoms = [_normalized_smarts_atom_from_rdkit_atom(atom) for atom in mol.GetAtoms()]
    smarts_mol = NormalizedSmartsMol(
        atoms=atoms,
        bonds=bonds,
        # A molecule read from RDKit carries no zero-level grouping: grouping is a statement a
        # *query* makes about which molecules its parts come from, and this is already one molecule.
        groups=np.zeros(len(atoms), dtype=np.int32),
        smarts_string=Chem.MolToSmarts(mol),
    )
    return gmol_from_normalized_smarts(smarts_mol)


def _require_concrete_rdkit_mol(mol: RdMol) -> None:
    if any(atom.HasQuery() for atom in mol.GetAtoms()):
        raise ValueError(
            "CMol conversion requires a concrete RDKit Mol; query atoms are unsupported."
        )
    if any(bond.HasQuery() for bond in mol.GetBonds()):
        raise ValueError(
            "CMol conversion requires a concrete RDKit Mol; query bonds are unsupported."
        )


def _normalized_smarts_atom_from_rdkit_atom(atom: RdAtom) -> NormalizedSmartsAtom:
    atom_smarts = atom.GetSmarts()
    try:
        return normalize_smarts_atom(parse_smarts_atom(atom_smarts))
    except pp.ParseBaseException as exc:
        raise SmartsError(f"Unsupported RDKit atom SMARTS: {atom_smarts}") from exc


def _concrete_bonds_from_rdkit_mol(
    mol: RdMol,
) -> np.ndarray[tuple[int, int], np.dtype[np.uint8]]:
    """Bond classes for a concrete molecule: an aromatic bond is the pure delocalized
    ``BOND_AROMATIC`` class (no Kekule order), every other bond carries its order.
    Vesma re-perceives aromaticity from these classes, so no kekulization is needed."""
    atom_count = int(mol.GetNumAtoms())
    bonds = np.zeros((atom_count, atom_count), dtype=np.uint8)
    for bond in mol.GetBonds():
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()
        flags = int(_exact_bond_type_from_rdkit_bond(bond))
        bonds[left, right] = np.uint8(flags)
        bonds[right, left] = np.uint8(flags)
    return bonds


def _query_bonds_from_rdkit_mol(
    mol: RdMol,
) -> np.ndarray[tuple[int, int], np.dtype[np.uint8]]:
    atom_count = int(mol.GetNumAtoms())
    bonds = np.zeros((atom_count, atom_count), dtype=np.uint8)
    for bond in mol.GetBonds():
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()
        flags = _bond_flags_from_rdkit_bond(bond)
        bonds[left, right] = np.uint8(flags)
        bonds[right, left] = np.uint8(flags)
    return bonds


def _bond_flags_from_rdkit_bond(bond: RdBond) -> int:
    bond_smarts = cast(Any, bond).GetSmarts()
    if bond_smarts:
        normalized_bond_smarts = bond_smarts.replace("&", "")
        try:
            return bond_flags_from_smarts_token(normalized_bond_smarts)
        except ValueError as exc:
            raise SmartsError(f"Unsupported RDKit bond SMARTS: {bond_smarts}") from exc
    # An omitted/default query bond means "single or aromatic" (Daylight semantics),
    # distinct from an explicit ``-``. cmol_from_gmol resolves it to a concrete seed
    # bond when the query graph is converted into a molecule.
    return QBOND_DEFAULT


def _exact_bond_type_from_rdkit_bond(bond: RdBond) -> BondKind:
    from rdkit import Chem

    rd_bond_type = bond.GetBondType()
    if bond.GetIsAromatic() or rd_bond_type == Chem.BondType.AROMATIC:
        return BondKind.DELOCALIZED
    if rd_bond_type == Chem.BondType.SINGLE:
        return BondKind.SINGLE
    if rd_bond_type == Chem.BondType.DOUBLE:
        return BondKind.DOUBLE
    if rd_bond_type == Chem.BondType.TRIPLE:
        return BondKind.TRIPLE

    raise SmartsError(f"Unsupported RDKit bond type: {rd_bond_type}")


def _rdkit_bond_spec_from_cmol(
    *, left: int, right: int, molecule: CMol
) -> tuple[Any, bool]:
    from rdkit import Chem

    bond_flags = int(molecule.bonds[left, right])
    aromatic = bool(bond_flags & int(BondKind.DELOCALIZED))
    bond_type = bond_flags & 0b0000_0111
    if aromatic:
        left_aromatic = int(molecule.A[left]) == int(AROMATIC)
        right_aromatic = int(molecule.A[right]) == int(AROMATIC)
        if not (left_aromatic and right_aromatic):
            raise ValueError(
                "RDKit Mol conversion requires aromatic bonds to connect aromatic atoms."
            )
        if bond_type == int(BondKind.NONE):
            # Canonical pure-aromatic bond (no Kekule order): hand RDKit an
            # aromatic bond and let SanitizeMol kekulize it.
            return Chem.BondType.AROMATIC, True
    if bond_type == int(BondKind.SINGLE):
        return Chem.BondType.SINGLE, aromatic
    if bond_type == int(BondKind.DOUBLE):
        return Chem.BondType.DOUBLE, aromatic
    if bond_type == int(BondKind.TRIPLE):
        return Chem.BondType.TRIPLE, False
    raise ValueError("RDKit Mol conversion requires exact bond types.")


__all__ = ["cmol_from_mol", "gmol_from_mol", "mol_from_cmol"]

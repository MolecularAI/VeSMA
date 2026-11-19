from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .adjacency import BondMatrix, bond_presence
from .aromaticity import perceive_aromaticity
from .bonds import (
    BOND_CONCRETE_KINDS,
    BOND_ORDER_BY_KIND,
    QBOND_DEFAULT,
    BondKind,
    exact_bond_kind,
    resolve_default_bond_flags,
)
from .cmol import CMol
from .gmol_atoms import AtomVector, Charge
from .hydrogens import total_hydrogens
from .invariants import (
    aromatic_bond_classes,
    identity_bond_classes,
    short_ring_bond_classes,
)
from .parsing import parse_ir
from .parsing.ir import PropVal, SmartsAtom
from .parsing.reading import smiles_reading
from .symbols import ELEMENT_TO_Z
from .wl import (
    WL_DEFAULT_ROUNDS,
    WlMol,
    WlToken,
    wl_molecule_id,
)

# Aromatic B and C derive their hydrogen from the ring; every other aromatic element
# carries it explicitly, so an unwritten count on them means none.
AROMATIC_HYDROGEN_BEARING_Z = frozenset({ELEMENT_TO_Z["B"], ELEMENT_TO_Z["C"]})

SHORT_RING_WL_SCHEMA = 6
AROMATIC_WL_SCHEMA = 5


__all__ = [
    "AROMATIC_WL_SCHEMA",
    "IdentitySeed",
    "SHORT_RING_WL_SCHEMA",
    "WlToken",
    "aromatic_bond_classes",
    "aromatic_wl_mol",
    "aromatic_wl_mol_from_seed",
    "aromatic_wl_mol_from_smiles",
    "aromatic_wl_molecule_id",
    "aromatic_wl_molecule_ids",
    "aromatic_wl_token_from_smiles",
    "identity_bond_classes",
    "identity_seed_from_cmol",
    "identity_seed_from_smiles",
    "short_ring_bond_classes",
    "short_ring_wl_mol",
    "short_ring_wl_mol_from_seed",
    "short_ring_wl_mol_from_smiles",
    "short_ring_wl_molecule_id",
    "short_ring_wl_molecule_ids",
    "short_ring_wl_token_from_smiles",
    "total_hydrogens_for_identity",
]


@dataclass(frozen=True)
class IdentitySeed:
    Z: AtomVector
    C: Charge
    explicit_h: AtomVector
    implicit_h: AtomVector
    bonds: BondMatrix


def identity_seed_from_smiles(smiles: str) -> IdentitySeed:
    """The seed for a WL token, read as SMILES.

    This path answers what a molecule *is*, not whether the model can hold it: it selects
    no configuration, so it keys species the molecule path refuses. `[CH0]` and `[NH0]`
    have always keyed here while refusing there, and `[N]` now joins them by behaving
    exactly like `[NH0]` on both -- which is agreement between the two readings, not a
    new gap between the two paths. Stock loading skips whatever refuses, so an
    open-shell row keys to something the search can never build and simply never matches.
    """
    # The same SMILES reading the CMol path uses, not a second copy of the rule: this
    # path never normalizes, so the reading has to live on the IR to serve both.
    molecule = smiles_reading(parse_ir(smiles.strip(), language="SMILES"))
    atom_count = molecule.atom_count
    Z = np.zeros(atom_count, dtype=np.int16)
    C = np.zeros(atom_count, dtype=np.int8)
    explicit_h = np.zeros(atom_count, dtype=np.int16)
    implicit_h = np.full(atom_count, -1, dtype=np.int16)

    for atom_index, atom in enumerate(molecule.atoms):
        atomic_number, charge, hydrogen = exact_atom_identity(atom)
        Z[atom_index] = np.int16(atomic_number)
        C[atom_index] = np.int8(charge)
        if hydrogen is not None:
            explicit_h[atom_index] = np.int16(hydrogen)
            implicit_h[atom_index] = 0
        # Only unbracketed atoms reach here: the SMILES reading writes an exact count
        # on every bracketed one, so `hydrogen is not None` above catches them all. A
        # charge cannot be written outside brackets, which is why the old `charge != 0`
        # arm is now unreachable and has been dropped. What remains is the
        # aromatic-heteroatom convention -- a lowercase `n` with two ring neighbours is
        # pyridine-like and carries no hydrogen.
        elif atom.is_aromatic and atomic_number not in AROMATIC_HYDROGEN_BEARING_Z:
            implicit_h[atom_index] = 0

    atom_aromatic = np.array(
        [bool(atom.is_aromatic) for atom in molecule.atoms], dtype=bool
    )
    bonds = concrete_bonds_from_smarts_bonds(molecule.bonds, atom_aromatic)
    return IdentitySeed(
        Z=Z,
        C=C,
        explicit_h=explicit_h,
        implicit_h=implicit_h,
        bonds=bonds,
    )


def identity_seed_from_cmol(molecule: CMol) -> IdentitySeed:
    if np.any(molecule.H < 0) or np.any(molecule.h < 0):
        raise ValueError("WL identity requires exact CMol hydrogen counts.")
    # The molecule already carries its exact hydrogen counts, so the seed writes them
    # rather than reconstructing a hint for perception to re-derive from. The split is
    # only storage -- explicit plus implicit is H on every atom -- and re-perceiving the
    # seed therefore reproduces H exactly instead of recomputing it from the sigma frame.
    explicit_h = (molecule.H - molecule.h).astype(np.int16)
    return IdentitySeed(
        Z=molecule.Z,
        C=molecule.C,
        explicit_h=explicit_h,
        implicit_h=molecule.h.astype(np.int16),
        bonds=molecule.bonds,
    )


def exact_atom_identity(atom: SmartsAtom) -> tuple[int, int, int | None]:
    if len(atom.formula) != 1 or len(atom.formula[0]) != 1:
        raise ValueError("WL stock identity requires exact atom expressions.")

    literals = atom.formula[0][0]
    if any(literal.negated for literal in literals):
        raise ValueError("WL stock identity does not support negated atoms.")

    symbols = [literal for literal in literals if literal.prop == "symbol"]
    if len(symbols) != 1:
        raise ValueError("WL stock identity requires one exact element.")
    symbol = symbols[0].value
    if not isinstance(symbol, str) or symbol in {"*", "A", "a"}:
        raise ValueError("WL stock identity requires one exact element.")
    element = symbol.capitalize()
    if element not in ELEMENT_TO_Z:
        raise ValueError(f"Unsupported stock element: {symbol}")

    allowed = {"symbol", "M", "C", "H"}
    unsupported = {literal.prop for literal in literals} - allowed
    if unsupported:
        joined = ", ".join(sorted(unsupported))
        raise ValueError(f"Unsupported stock atom properties: {joined}")

    charge = one_literal_value(literals, "C", default=0)
    hydrogen_literal = one_literal(literals, "H")
    if hydrogen_literal is None:
        hydrogen = None
    elif hydrogen_literal.value is None:
        hydrogen = 1
    elif isinstance(hydrogen_literal.value, int):
        hydrogen = hydrogen_literal.value
    else:
        raise ValueError("WL stock identity requires an integer H value.")
    return ELEMENT_TO_Z[element], charge, hydrogen


def one_literal(
    literals: list[PropVal],
    prop: str,
) -> PropVal | None:
    values = [literal for literal in literals if literal.prop == prop]
    if len(values) > 1:
        raise ValueError(f"WL stock identity requires at most one {prop} value.")
    return values[0] if values else None


def one_literal_value(
    literals: list[PropVal],
    prop: str,
    *,
    default: int,
) -> int:
    literal = one_literal(literals, prop)
    if literal is None:
        return default
    if not isinstance(literal.value, int):
        raise ValueError(f"WL stock identity requires an integer {prop} value.")
    return literal.value


def concrete_bonds_from_smarts_bonds(
    bonds: BondMatrix, atom_aromatic: np.ndarray
) -> BondMatrix:
    result = np.zeros_like(bonds)
    for flags in np.unique(bonds):
        value = int(flags)
        if value == 0:
            continue
        if value == QBOND_DEFAULT:
            # Omitted/default bond ("single or aromatic"): resolve per edge using the
            # endpoints' aromatic intent, matching cmol_from_gmol concretization.
            rows, cols = np.nonzero(bonds == flags)
            for r, c in zip(rows.tolist(), cols.tolist()):
                concrete = resolve_default_bond_flags(
                    value,
                    left_aromatic=bool(atom_aromatic[r]),
                    right_aromatic=bool(atom_aromatic[c]),
                )
                result[r, c] = np.uint8(concrete)
            continue
        concrete = exact_bond_kind(value)
        if concrete is None:
            raise ValueError("WL stock identity requires exact bond types.")
        result[bonds == flags] = np.uint8(concrete)
    return result


def short_ring_wl_mol_from_smiles(smiles: str) -> WlMol:
    return short_ring_wl_mol_from_seed(identity_seed_from_smiles(smiles))


def aromatic_wl_mol_from_smiles(smiles: str) -> WlMol:
    return aromatic_wl_mol_from_seed(identity_seed_from_smiles(smiles))


def short_ring_wl_mol_from_seed(seed: IdentitySeed) -> WlMol:
    H = total_hydrogens_for_identity(seed)
    return WlMol(
        Z=seed.Z,
        C=seed.C,
        H=H,
        bonds=short_ring_bond_classes(seed.bonds),
    )


def aromatic_wl_mol_from_seed(seed: IdentitySeed) -> WlMol:
    _, perceived_bonds, _ = perceive_aromaticity(
        Z=seed.Z,
        C=seed.C,
        explicit_h=seed.explicit_h,
        implicit_h=seed.implicit_h,
        bonds=seed.bonds,
    )
    H = total_hydrogens_for_identity(seed)
    return WlMol(
        Z=seed.Z,
        C=seed.C,
        H=H,
        bonds=aromatic_bond_classes(perceived_bonds),
    )


def short_ring_wl_mol(molecule: CMol) -> WlMol:
    return short_ring_wl_mol_from_seed(identity_seed_from_cmol(molecule))


def aromatic_wl_mol(molecule: CMol) -> WlMol:
    return aromatic_wl_mol_from_seed(identity_seed_from_cmol(molecule))


def total_hydrogens_for_identity(seed: IdentitySeed) -> AtomVector:
    # Identity H uses the one shared model (vesma/hydrogens.total_hydrogens), so it
    # cannot drift from perception: source-aromatic B/C get degree-derived H
    # (3 - degree, even when charged, e.g. cyclopentadienyl / tropylium), aromatic
    # heteroatoms take their preset ([nH]-style) H, and everything else closes
    # valence from concrete bond orders. Kekule and aromatic encodings agree
    # atom-for-atom, which is what makes identity input-invariant. Inputs come from
    # the seed's own bonds, never from perceived/canonicalized bonds.
    bonds = seed.bonds
    heavy_degrees = bond_presence(bonds).sum(axis=1).astype(np.int16)
    order_sums = (
        BOND_ORDER_BY_KIND[bonds & np.uint8(BOND_CONCRETE_KINDS)]
        .sum(axis=1)
        .astype(np.int16)
    )
    source_aromatic = ((bonds & np.uint8(BondKind.DELOCALIZED)) != 0).any(axis=1)
    return total_hydrogens(
        Z=seed.Z,
        C=seed.C,
        explicit_h=seed.explicit_h,
        implicit_h=seed.implicit_h,
        order_sums=order_sums,
        heavy_degrees=heavy_degrees,
        bond_orders_omitted=source_aromatic,
    )


def short_ring_wl_molecule_id(
    molecule: CMol,
    *,
    rounds: int = WL_DEFAULT_ROUNDS,
) -> WlToken:
    return wl_molecule_id(
        short_ring_wl_mol(molecule),
        schema=SHORT_RING_WL_SCHEMA,
        rounds=rounds,
    )


def short_ring_wl_molecule_ids(
    molecules: Sequence[CMol],
    *,
    rounds: int = WL_DEFAULT_ROUNDS,
) -> list[WlToken]:
    return [
        short_ring_wl_molecule_id(molecule, rounds=rounds) for molecule in molecules
    ]


def aromatic_wl_molecule_id(
    molecule: CMol,
    *,
    rounds: int = WL_DEFAULT_ROUNDS,
) -> WlToken:
    return wl_molecule_id(
        aromatic_wl_mol(molecule),
        schema=AROMATIC_WL_SCHEMA,
        rounds=rounds,
    )


def aromatic_wl_molecule_ids(
    molecules: Sequence[CMol],
    *,
    rounds: int = WL_DEFAULT_ROUNDS,
) -> list[WlToken]:
    return [aromatic_wl_molecule_id(molecule, rounds=rounds) for molecule in molecules]


def short_ring_wl_token_from_smiles(
    smiles: str,
    *,
    rounds: int = WL_DEFAULT_ROUNDS,
) -> WlToken:
    return wl_molecule_id(
        short_ring_wl_mol_from_smiles(smiles),
        schema=SHORT_RING_WL_SCHEMA,
        rounds=rounds,
    )


def aromatic_wl_token_from_smiles(
    smiles: str,
    *,
    rounds: int = WL_DEFAULT_ROUNDS,
) -> WlToken:
    return wl_molecule_id(
        aromatic_wl_mol_from_smiles(smiles),
        schema=AROMATIC_WL_SCHEMA,
        rounds=rounds,
    )

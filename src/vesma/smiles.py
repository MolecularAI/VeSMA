from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .adjacency import bond_presence
from .bonds import BondKind
from .cmol import CMol
from .gmol_atoms import AROMATIC, MASS_UNLABELED
from .symbols import AROMATIC_UNBRACKET, UNBRACKET_SYMBOLS, Z_TO_ELEMENT


def smiles_from_cmol(molecule: CMol) -> str:
    n_atoms = molecule.bonds.shape[0]
    if n_atoms == 0:
        return ""

    atom_aromatic = molecule.A == AROMATIC
    atom_explicit_h = molecule.H - molecule.h
    adjacency = bond_presence(molecule.bonds)
    parent = np.full(n_atoms, -1, dtype=np.int32)
    visited = np.zeros(n_atoms, dtype=bool)
    children: list[list[int]] = [[] for _ in range(n_atoms)]
    tree_edges: set[tuple[int, int]] = set()

    for root in range(n_atoms):
        if visited[root]:
            continue
        _dfs_tree_visit(root, adjacency, visited, parent, children, tree_edges)

    ring_annotations = _build_ring_annotations(
        molecule.bonds,
        adjacency,
        tree_edges,
        atom_aromatic,
    )
    atom_tokens = _build_atom_tokens(
        atom_z=molecule.Z,
        atom_aromatic=atom_aromatic,
        atom_charge=molecule.C,
        atom_explicit_h=atom_explicit_h,
        atom_total_h=molecule.H,
        atom_degree=adjacency.sum(axis=1),
        atom_map=molecule.atom_map,
        atom_mass=molecule.M,
    )
    roots = [idx for idx in range(n_atoms) if int(parent[idx]) < 0]
    component_smiles = [
        _emit_tree_smiles(
            root=root,
            children=children,
            bond_types=molecule.bonds,
            ring_annotations=ring_annotations,
            atom_tokens=atom_tokens,
            atom_aromatic=atom_aromatic,
        )
        for root in roots
    ]
    return ".".join(component_smiles)


def _dfs_tree_visit(
    node: int,
    adjacency: np.ndarray,
    visited: np.ndarray,
    parent: np.ndarray,
    children: list[list[int]],
    tree_edges: set[tuple[int, int]],
) -> None:
    visited[node] = True
    for neighbor in np.flatnonzero(adjacency[node]):
        n = int(neighbor)
        if visited[n]:
            continue
        parent[n] = node
        children[node].append(n)
        tree_edges.add(_edge_key(node, n))
        _dfs_tree_visit(n, adjacency, visited, parent, children, tree_edges)


def _build_ring_annotations(
    bond_types: np.ndarray,
    adjacency: np.ndarray,
    tree_edges: set[tuple[int, int]],
    atom_aromatic: np.ndarray,
) -> list[list[str]]:
    n_atoms = adjacency.shape[0]
    annotations: list[list[tuple[int, str]]] = [[] for _ in range(n_atoms)]

    all_edges = [
        _edge_key(int(i), int(j))
        for i, j in zip(*np.where(np.triu(adjacency, k=1)), strict=False)
    ]
    ring_edges = sorted(edge for edge in all_edges if edge not in tree_edges)

    for ring_id, (left, right) in enumerate(ring_edges, start=1):
        if ring_id > 99:
            raise ValueError(
                "SMILES generation currently supports at most 99 ring-closure edges."
            )
        both_aromatic = bool(atom_aromatic[left]) and bool(atom_aromatic[right])
        token = (
            f"{_bond_symbol(int(bond_types[left, right]), both_aromatic=both_aromatic)}"
            f"{_ring_id_token(ring_id)}"
        )
        annotations[left].append((ring_id, token))
        annotations[right].append((ring_id, token))

    ordered: list[list[str]] = []
    for atom_tokens in annotations:
        atom_tokens.sort(key=lambda item: item[0])
        ordered.append([token for _, token in atom_tokens])
    return ordered


def _build_atom_tokens(
    *,
    atom_z: np.ndarray,
    atom_aromatic: np.ndarray,
    atom_charge: np.ndarray,
    atom_explicit_h: np.ndarray,
    atom_total_h: np.ndarray,
    atom_degree: np.ndarray,
    atom_map: np.ndarray,
    atom_mass: np.ndarray,
) -> list[str]:
    tokens: list[str] = []
    unbracket_symbols = set(UNBRACKET_SYMBOLS.split())
    for idx in range(atom_z.shape[0]):
        z = int(atom_z[idx])
        aromatic = bool(atom_aromatic[idx])
        charge = int(atom_charge[idx])
        explicit_h = int(atom_explicit_h[idx])
        total_h = int(atom_total_h[idx])
        degree = int(atom_degree[idx])
        mapped = int(atom_map[idx])
        mass = int(atom_mass[idx])

        symbol = Z_TO_ELEMENT.get(z, "*")
        aromatic_symbol = (
            symbol.lower()
            if aromatic and symbol in AROMATIC_UNBRACKET.values()
            else symbol
        )

        if (
            aromatic_symbol in unbracket_symbols
            and charge == 0
            and explicit_h == 0
            and degree > 0
            and mapped == 0
            and mass == MASS_UNLABELED
        ):
            tokens.append(aromatic_symbol)
            continue

        # Whether to bracket is decided by the *explicit* count, so an atom whose
        # hydrogens were implicit stays compact. What a bracket then *states* is the
        # total, because OpenSMILES section 3.6 requires it: "Any atom that is specified
        # with square brackets must have its attached hydrogens explicitly represented."
        # A bracket opened for a charge, a map or an isotope would otherwise say H0 and
        # denote a different molecule -- methane written `[C]`, and an acid's `[OH:3]`
        # written `[O:3]` and read back as an acyl radical.
        parts: list[str] = [aromatic_symbol]
        if mass != MASS_UNLABELED:
            parts.insert(0, str(mass))
        if total_h > 0:
            parts.append("H" if total_h == 1 else f"H{total_h}")
        if charge != 0:
            parts.append(_charge_token(charge))
        if mapped > 0:
            parts.append(f":{mapped}")
        tokens.append(f"[{''.join(parts)}]")

    return tokens


def _emit_tree_smiles(
    root: int,
    children: Sequence[Sequence[int]],
    bond_types: np.ndarray,
    ring_annotations: Sequence[Sequence[str]],
    atom_tokens: Sequence[str],
    atom_aromatic: np.ndarray,
) -> str:
    text = atom_tokens[root] + "".join(ring_annotations[root])
    child_list = list(children[root])
    if not child_list:
        return text

    first = child_list[0]
    for child in child_list[1:]:
        branch = _bond_symbol(
            int(bond_types[root, child]),
            both_aromatic=bool(atom_aromatic[root]) and bool(atom_aromatic[child]),
        )
        branch += _emit_tree_smiles(
            root=child,
            children=children,
            bond_types=bond_types,
            ring_annotations=ring_annotations,
            atom_tokens=atom_tokens,
            atom_aromatic=atom_aromatic,
        )
        text += f"({branch})"

    text += _bond_symbol(
        int(bond_types[root, first]),
        both_aromatic=bool(atom_aromatic[root]) and bool(atom_aromatic[first]),
    )
    text += _emit_tree_smiles(
        root=first,
        children=children,
        bond_types=bond_types,
        ring_annotations=ring_annotations,
        atom_tokens=atom_tokens,
        atom_aromatic=atom_aromatic,
    )

    return text


def _bond_symbol(bond_type: int, *, both_aromatic: bool) -> str:
    order = bond_type & 0b0000_0111
    if order == int(BondKind.SINGLE):
        # A single bond between two aromatic atoms (e.g. a biphenyl bridge) must
        # be explicit so it is not read as the implicit aromatic bond.
        return "-"
    if order == int(BondKind.DOUBLE):
        return "="
    if order == int(BondKind.TRIPLE):
        return "#"
    if bond_type & int(BondKind.DELOCALIZED):
        # A pure aromatic bond is implicit between two aromatic atoms; otherwise
        # it needs an explicit aromatic-bond symbol.
        return "" if both_aromatic else ":"
    if bond_type == 0:
        raise ValueError(
            "BOND_NONE is not a serializable bond; disconnected components use '.'."
        )
    raise ValueError("SMILES generation requires exact bond types.")


def _ring_id_token(ring_id: int) -> str:
    if ring_id < 10:
        return str(ring_id)
    return f"%{ring_id:02d}"


def _charge_token(charge: int) -> str:
    sign = "+" if charge > 0 else "-"
    magnitude = abs(charge)
    if magnitude == 1:
        return sign
    return f"{sign}{magnitude}"


def _edge_key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .adjacency import BondMatrix, bond_presence
from .cmol import CMol
from .gmol_atoms import AROMATIC

RankMap = Mapping[int, int]
# A key is anything the refinement can sort and compare, and the two shapes in this file are not
# the only ones: `vesma.pattern_identity` canonicalises a *reaction pattern* with the same search,
# and its atoms carry a side marker and twelve mask words instead of the five concrete properties a
# `CMol` atom has. The algorithm never reads a field -- `refined_ranks` only compresses and compares
# -- so the arity is the caller's business. See `atom_initial_key` and `atom_identity_key` below for
# what a molecule puts here.
AtomInitialKey = tuple[int | tuple[int, ...], ...]
# Plain ints, and that is load-bearing rather than incidental: identity keys end up inside a
# `Certificate`, and certificates are *ordered* against each other to pick a canonical component.
# A key mixing ints with tuples would make that comparison raise on some pairs and not others, so
# the nested part of an initial key has no place here. Both shapes oblige -- a molecule's five
# properties and a pattern atom's side plus its mask words are all integers.
AtomIdentityKey = tuple[int, ...]
Certificate = tuple[tuple[AtomIdentityKey, ...], tuple[tuple[int, ...], ...]]


@dataclass(frozen=True)
class RankingState:
    neighbors: tuple[tuple[int, ...], ...]
    atom_initial_keys: tuple[AtomInitialKey, ...]
    atom_identity_keys: tuple[AtomIdentityKey, ...]
    bonds: BondMatrix


__all__ = [
    "AtomIdentityKey",
    "AtomInitialKey",
    "Certificate",
    "RankMap",
    "RankingState",
    "canonicalish_atom_order",
    "reorder_cmol",
]


def canonicalish_atom_order(molecule: CMol) -> tuple[int, ...]:
    state = ranking_state_from_cmol(molecule)
    components = connected_components_from_neighbors(state.neighbors)
    ordered_components = [
        canonical_component(state, component) for component in components
    ]
    ordered_components.sort(key=lambda item: item[0])
    return tuple(
        atom for _, component_order in ordered_components for atom in component_order
    )


def reorder_cmol(molecule: CMol, order: tuple[int, ...]) -> CMol:
    idx = np.asarray(order, dtype=np.int64)
    return CMol(
        Z=molecule.Z[idx].copy(),
        M=molecule.M[idx].copy(),
        A=molecule.A[idx].copy(),
        C=molecule.C[idx].copy(),
        D=molecule.D[idx].copy(),
        H=molecule.H[idx].copy(),
        h=molecule.h[idx].copy(),
        v=molecule.v[idx].copy(),
        X=molecule.X[idx].copy(),
        R=molecule.R[idx].copy(),
        r=molecule.r[idx].copy(),
        x=molecule.x[idx].copy(),
        bonds=molecule.bonds[np.ix_(idx, idx)].copy(),
        atom_map=np.zeros(len(order), dtype=np.int32),
    )


def canonical_component(
    state: RankingState,
    atoms: tuple[int, ...],
) -> tuple[Certificate, tuple[int, ...]]:
    return canonical_component_search(state, atoms, {})


def canonical_component_search(
    state: RankingState,
    atoms: tuple[int, ...],
    individualized: dict[int, int],
) -> tuple[Certificate, tuple[int, ...]]:
    ranks = refined_ranks(state, atoms, individualized)
    groups = rank_groups(atoms, ranks)
    tied_groups = [group for group in groups if len(group) > 1]
    if not tied_groups:
        order = tuple(sorted(atoms, key=lambda atom: ranks[atom]))
        return component_certificate(state, order), order

    group = min(tied_groups, key=lambda values: (len(values), ranks[values[0]]))
    best: tuple[Certificate, tuple[int, ...]] | None = None
    next_rank = len(individualized)
    for atom in group:
        candidate = canonical_component_search(
            state,
            atoms,
            individualized | {atom: next_rank},
        )
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        raise ValueError("Could not canonicalize molecule component.")
    return best


def refined_ranks(
    state: RankingState,
    atoms: tuple[int, ...],
    individualized: RankMap,
) -> dict[int, int]:
    ranks = compressed_ranks(
        {
            atom: (
                state.atom_initial_keys[atom],
                individualized.get(atom, len(atoms)),
            )
            for atom in atoms
        }
    )
    atom_set = set(atoms)

    for _ in atoms:
        next_ranks = compressed_ranks(
            {
                atom: (
                    ranks[atom],
                    tuple(
                        sorted(
                            (
                                int(state.bonds[atom, neighbor]),
                                ranks[neighbor],
                            )
                            for neighbor in state.neighbors[atom]
                            if neighbor in atom_set
                        )
                    ),
                )
                for atom in atoms
            }
        )
        if next_ranks == ranks:
            return ranks
        ranks = next_ranks

    return ranks


def compressed_ranks(values: Mapping[int, object]) -> dict[int, int]:
    keys = {key: rank for rank, key in enumerate(sorted(set(values.values())))}
    return {atom: keys[value] for atom, value in values.items()}


def rank_groups(atoms: tuple[int, ...], ranks: RankMap) -> list[tuple[int, ...]]:
    grouped: dict[int, list[int]] = {}
    for atom in atoms:
        grouped.setdefault(ranks[atom], []).append(atom)
    return [tuple(grouped[rank]) for rank in sorted(grouped)]


def ranking_state_from_cmol(molecule: CMol) -> RankingState:
    adjacency = bond_presence(molecule.bonds)
    neighbors = tuple(
        tuple(int(neighbor) for neighbor in np.flatnonzero(adjacency[atom]))
        for atom in range(molecule.atom_count)
    )
    atom_identity_keys = tuple(
        atom_identity_key(molecule, atom) for atom in range(molecule.atom_count)
    )
    atom_initial_keys = tuple(
        atom_initial_key(molecule, atom, neighbors[atom])
        for atom in range(molecule.atom_count)
    )
    return RankingState(
        neighbors=neighbors,
        atom_initial_keys=atom_initial_keys,
        atom_identity_keys=atom_identity_keys,
        bonds=molecule.bonds,
    )


def atom_initial_key(
    molecule: CMol,
    atom: int,
    neighbors: tuple[int, ...],
) -> AtomInitialKey:
    incident_bonds = tuple(
        sorted(int(molecule.bonds[atom, neighbor]) for neighbor in neighbors)
    )
    return (
        int(molecule.Z[atom]),
        int(molecule.M[atom]),
        int(molecule.A[atom] == AROMATIC),
        int(molecule.C[atom]),
        int(molecule.H[atom]),
        len(neighbors),
        incident_bonds,
    )


def component_certificate(
    state: RankingState,
    order: tuple[int, ...],
) -> Certificate:
    atom_keys = tuple(state.atom_identity_keys[atom] for atom in order)
    bond_rows = tuple(
        tuple(int(state.bonds[left, right]) for right in order[idx + 1 :])
        for idx, left in enumerate(order)
    )
    return atom_keys, bond_rows


def atom_identity_key(molecule: CMol, atom: int) -> AtomIdentityKey:
    return (
        int(molecule.Z[atom]),
        int(molecule.M[atom]),
        int(molecule.A[atom] == AROMATIC),
        int(molecule.C[atom]),
        int(molecule.H[atom]),
    )


def connected_components_from_neighbors(
    neighbors: tuple[tuple[int, ...], ...],
) -> list[tuple[int, ...]]:
    visited = [False for _ in neighbors]
    components: list[tuple[int, ...]] = []

    for root in range(len(neighbors)):
        if visited[root]:
            continue
        stack = [root]
        nodes: list[int] = []
        visited[root] = True
        while stack:
            atom = stack.pop()
            nodes.append(atom)
            for neighbor in neighbors[atom]:
                if visited[neighbor]:
                    continue
                visited[neighbor] = True
                stack.append(neighbor)
        components.append(tuple(nodes))

    return components

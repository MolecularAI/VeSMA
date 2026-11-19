from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntFlag, auto

import numpy as np

from .adjacency import BondMatrix, BoolBondMatrix, bond_presence
from .bonds import (
    BOND_TOPOLOGY_CHAIN,
    BOND_TOPOLOGY_MASK,
    BOND_TOPOLOGY_RING,
)
from .gmol_atoms import AtomIndices

AtomCounts = np.ndarray[
    tuple[int], np.dtype[np.uint8]
]  # per-atom ring count/descriptor
CycleAtoms = np.ndarray[tuple[int], np.dtype[np.int16]]  # atom indices of one cycle
ParentVector = np.ndarray[tuple[int], np.dtype[np.int_]]  # union-find parent pointers


# One bounded simple-cycle enumerator serves both the short-ring WL identity edge
# mask and aromaticity perception. Both flex/evaluate rings of size 3-8.
SHORT_CYCLE_MAX_SIZE = 8
AROMATIC_CYCLE_MIN_SIZE = 3
AROMATIC_CYCLE_MAX_SIZE = 8

__all__ = [
    "AROMATIC_CYCLE_MAX_SIZE",
    "AROMATIC_CYCLE_MIN_SIZE",
    "RingDescriptors",
    "RingInfo",
    "SHORT_CYCLE_MAX_SIZE",
    "connected_components_from_adjacency",
    "is_acyclic",
    "annotate_bond_topology",
    "ring_edge_mask",
    "ring_info_from_bonds",
    "ring_info_from_presence",
    "short_cycle_edge_mask",
    "short_ring_edges",
    "simple_cycles",
    "walk_simple_cycles",
]


class RingDescriptors(IntFlag):
    """Which per-atom ring descriptors a caller wants; the rest keep the -1 sentinel.

    CONNECTIVITY (`x`) counts an atom's incident ring bonds, and a ring bond is any
    non-bridge. MEMBERSHIP (`R` and `r`) counts and measures the relevant cycles, see
    :func:`ring_info_from_presence`. Both are independent of atom order; Metal perceives
    both on the device, CUDA still leaves membership unknown.
    """

    NONE = 0
    CONNECTIVITY = auto()
    MEMBERSHIP = auto()
    ALL = CONNECTIVITY | MEMBERSHIP


@dataclass(frozen=True)
class RingInfo:
    membership_count: AtomCounts
    smallest_ring_size: AtomCounts
    ring_connectivity: AtomCounts


def walk_simple_cycles(
    adjacency: BoolBondMatrix,
    *,
    max_size: int,
    on_cycle: Callable[[tuple[int, ...]], None],
) -> None:
    """Enumerate every simple cycle of size 3..max_size exactly once.

    Bounded DFS from each root, where the root is the lowest atom index in the
    cycle and only the direction with ``path[1] < path[-1]`` is accepted, so each
    cycle is reported once. Both the short-ring identity edge mask and aromaticity
    perception share this enumerator; callers filter by minimum size. ``on_cycle``
    receives the cycle as a tuple of atom indices (length equal to cycle size).
    """
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Cycle enumeration requires a square adjacency matrix.")
    atom_count = adjacency.shape[0]
    # The per-atom adjacency masks below are arbitrary-precision Python ints, so the
    # enumerator has no intrinsic atom-count limit. Large connected components do arise
    # in route replay (a heavily protected retro intermediate before the split into
    # connected components), so we enumerate them rather than rejecting them. The GPU
    # backends keep their own 64-bit-mask caps in limits.py.

    neighbor_masks = tuple(
        sum(1 << int(neighbor) for neighbor in np.flatnonzero(adjacency[atom]))
        for atom in range(atom_count)
    )
    for root in range(atom_count):
        extend_simple_cycle(
            root=root,
            current=root,
            path=(root,),
            visited=1 << root,
            neighbor_masks=neighbor_masks,
            max_size=max_size,
            on_cycle=on_cycle,
        )


def extend_simple_cycle(
    *,
    root: int,
    current: int,
    path: tuple[int, ...],
    visited: int,
    neighbor_masks: tuple[int, ...],
    max_size: int,
    on_cycle: Callable[[tuple[int, ...]], None],
) -> None:
    neighbors = neighbor_masks[current]
    while neighbors:
        neighbor_bit = neighbors & -neighbors
        neighbors ^= neighbor_bit
        neighbor = neighbor_bit.bit_length() - 1

        if neighbor == root:
            if len(path) >= 3 and path[1] < path[-1]:
                on_cycle(path)
            continue
        if len(path) == max_size or neighbor < root or visited & neighbor_bit:
            continue

        extend_simple_cycle(
            root=root,
            current=neighbor,
            path=(*path, neighbor),
            visited=visited | neighbor_bit,
            neighbor_masks=neighbor_masks,
            max_size=max_size,
            on_cycle=on_cycle,
        )


def simple_cycles(
    adjacency: BoolBondMatrix,
    *,
    min_size: int,
    max_size: int,
) -> list[CycleAtoms]:
    """Simple cycles with ``min_size <= size <= max_size`` as atom-index arrays,
    each canonicalized to start at its lowest atom index."""
    cycles: list[CycleAtoms] = []

    def collect(path: tuple[int, ...]) -> None:
        if len(path) >= min_size:
            cycles.append(np.asarray(path, dtype=np.int16))

    walk_simple_cycles(adjacency, max_size=max_size, on_cycle=collect)
    return cycles


def short_cycle_edge_mask(adjacency: BoolBondMatrix) -> BoolBondMatrix:
    """Return edges belonging to simple cycles of size 3 through 8."""
    if adjacency.shape[0] > 64 and is_acyclic(adjacency):
        return np.zeros_like(adjacency)
    result = np.zeros_like(adjacency)

    def mark(path: tuple[int, ...]) -> None:
        closed = (*path, path[0])
        for left, right in zip(closed[:-1], closed[1:], strict=True):
            result[left, right] = True
            result[right, left] = True

    walk_simple_cycles(adjacency, max_size=SHORT_CYCLE_MAX_SIZE, on_cycle=mark)
    return result


def short_ring_edges(bond_types: BondMatrix) -> BoolBondMatrix:
    """Edges lying in a simple ring of size <= 8, taken straight from a bond
    matrix. The single short-ring detector the WL and path codecs share
    to flex/reclass small-ring bonds."""
    return short_cycle_edge_mask(bond_presence(bond_types))


def is_acyclic(adjacency: BoolBondMatrix) -> bool:
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Cycle enumeration requires a square adjacency matrix.")
    active = np.any(adjacency, axis=1)
    component_count = len(connected_components_from_adjacency(adjacency)) + int(
        np.count_nonzero(~active)
    )
    edge_count = int(np.count_nonzero(np.triu(adjacency, k=1)))
    return edge_count == adjacency.shape[0] - component_count


def connected_components_from_adjacency(adjacency: BoolBondMatrix) -> list[AtomIndices]:
    visited = np.zeros(adjacency.shape[0], dtype=bool)
    components: list[AtomIndices] = []
    for start in range(adjacency.shape[0]):
        if visited[start] or not np.any(adjacency[start]):
            continue
        stack = [start]
        visited[start] = True
        nodes: list[int] = []
        while stack:
            node = stack.pop()
            nodes.append(node)
            for neighbor in np.flatnonzero(adjacency[node]):
                if visited[neighbor]:
                    continue
                visited[neighbor] = True
                stack.append(int(neighbor))
        components.append(np.asarray(nodes, dtype=np.int64))
    return components


def ring_info_from_presence(adjacency: BoolBondMatrix) -> RingInfo:
    """Per-atom ring descriptors: ``R`` and ``r`` from the relevant cycles, ``x`` from ring bonds.

    ``R`` counts the relevant cycles through an atom and ``r`` is the smallest, both capped at the
    bucket maximum of 7. Relevant cycles (Vismara) are the cycles not in the GF(2) span of strictly
    smaller ones: the union of every minimum cycle basis, so unlike one spanning forest's
    fundamental cycles they do not depend on atom order. On the stock, n1-d2 and cage corpora they
    agree with RDKit's ``R``/``r`` on every atom (``tests/vesma/perf_ring_membership.py``).
    """
    n_atoms = adjacency.shape[0]
    ring = ring_edge_mask(adjacency)
    rows = tuple(
        sum(1 << int(neighbor) for neighbor in np.flatnonzero(ring[atom]))
        for atom in range(n_atoms)
    )
    count, smallest = relevant_ring_membership(rows)
    return RingInfo(
        membership_count=np.minimum(count, 7).astype(np.uint8),
        smallest_ring_size=np.minimum(smallest, 7).astype(np.uint8),
        ring_connectivity=np.minimum(ring.sum(axis=1), 7).astype(np.uint8),
    )


def bit_indices(mask: int) -> list[int]:
    found = []
    while mask:
        low = mask & -mask
        found.append(low.bit_length() - 1)
        mask ^= low
    return found


def relevant_ring_membership(rows: tuple[int, ...]) -> tuple[list[int], list[int]]:
    """``(R, r)`` per atom from ring-bond adjacency rows (one int bitset per atom, bridges gone).

    A relevant cycle is isometric -- were two of its atoms closer through the graph than along
    it, it would be the sum of two strictly shorter cycles -- so from any root on it, it is two
    shortest paths and the edge joining their ends. Every such cycle is therefore among the
    candidates built from *all* pairs of shortest paths: one canonical path per atom would miss
    those that differ only where shortest paths tie, which is what makes a cage's counts depend
    on atom order. Candidates are built size by size (a root's paths to the ends of edge
    ``(x, y)`` close a cycle of ``d(x) + d(y) + 1``), each size tested by GF(2) against the basis
    of the smaller ones, and the sizes stop once the basis reaches the cycle rank. Each cycle is
    collected only from its lowest atom, which isometry allows: paths never step below the root,
    though distances are the whole graph's. The same algorithm as the Metal retro kernel's
    ``ring_membership``.
    """
    n_atoms = len(rows)
    count = [0] * n_atoms
    smallest = [0] * n_atoms
    edges = [
        (left, right)
        for left, row in enumerate(rows)
        for right in bit_indices(row)
        if left < right
    ]
    if not edges:
        return count, smallest
    edge_index = {edge: position for position, edge in enumerate(edges)}
    active = [atom for atom, row in enumerate(rows) if row]
    # A ring system with as many bonds as atoms is one lone ring: R = 1, r = its size, no search.
    # Only fused, bridged and spiro systems go on, and the rank is theirs alone.
    rank = 0
    complex_atoms: list[int] = []
    for system in row_systems(rows, active):
        bonds = sum(rows[atom].bit_count() for atom in system) // 2
        if bonds == len(system):
            for atom in system:
                count[atom] = 1
                smallest[atom] = len(system)
        else:
            rank += bonds - len(system) + 1
            complex_atoms.extend(system)
    if rank == 0:
        return count, smallest
    trees = {
        root: shortest_path_trees(rows, root, edge_index)
        for root in sorted(complex_atoms)
    }

    basis: dict[int, int] = {}

    def reduce(vector: int) -> int:
        while vector:
            pivot = basis.get(vector.bit_length() - 1)
            if pivot is None:
                return vector
            vector ^= pivot
        return 0

    for size in range(3, n_atoms + 1):
        if len(basis) >= rank:
            break
        tier: set[int] = set()
        for root, (distance, paths) in trees.items():
            root_bit = 1 << root
            below = root_bit - 1  # a cycle is collected from its lowest atom only
            for (left, right), position in edge_index.items():
                if left < root:
                    continue
                if distance.get(left, -1) + distance.get(right, -1) + 1 != size:
                    continue
                if left not in distance or right not in distance:
                    continue
                for left_atoms, left_edges in paths[left]:
                    if left_atoms & below:
                        continue
                    for right_atoms, right_edges in paths[right]:
                        if (
                            left_atoms & right_atoms == root_bit
                            and not right_atoms & below
                        ):
                            tier.add(left_edges | right_edges | 1 << position)
        # Relevance against the basis frozen at the smaller sizes, then grow it.
        relevant = [mask for mask in sorted(tier) if reduce(mask)]
        for mask in relevant:
            for atom in {a for position in bit_indices(mask) for a in edges[position]}:
                count[atom] += 1
                smallest[atom] = size if smallest[atom] == 0 else smallest[atom]
            residue = reduce(mask)
            if residue:
                basis[residue.bit_length() - 1] = residue
    return count, smallest


def shortest_path_trees(
    rows: tuple[int, ...], root: int, edge_index: dict[tuple[int, int], int]
) -> tuple[dict[int, int], dict[int, list[tuple[int, int]]]]:
    """Distances from ``root``, and every shortest path to each atom as (atom bits, edge bits)."""
    distance = {root: 0}
    paths: dict[int, list[tuple[int, int]]] = {root: [(1 << root, 0)]}
    frontier = [root]
    while frontier:
        discovered: set[int] = set()
        for atom in frontier:
            for neighbor in bit_indices(rows[atom]):
                if neighbor not in distance:
                    discovered.add(neighbor)
        level = distance[frontier[0]] + 1
        for atom in discovered:
            distance[atom] = level
        for atom in sorted(discovered):
            paths[atom] = [
                (
                    atoms | 1 << atom,
                    path_edges | 1 << edge_index[min(p, atom), max(p, atom)],
                )
                for p in bit_indices(rows[atom])
                if distance.get(p) == level - 1
                for atoms, path_edges in paths[p]
            ]
        frontier = sorted(discovered)
    return distance, paths


def row_systems(rows: tuple[int, ...], active: list[int]) -> list[list[int]]:
    """The connected components of the atoms with any bond, each as its atom list."""
    seen = 0
    systems = []
    for start in active:
        if seen >> start & 1:
            continue
        component = frontier = 1 << start
        seen |= frontier
        while frontier:
            reach = 0
            for atom in bit_indices(frontier):
                reach |= rows[atom]
            frontier = reach & ~seen
            seen |= frontier
            component |= frontier
        systems.append(bit_indices(component))
    return systems


def ring_edge_mask(adjacency: BoolBondMatrix) -> BoolBondMatrix:
    """Boolean matrix of edges lying in a ring of any size (a bond is in a ring iff it
    is not a bridge). Built from a spanning forest: every non-tree (back) edge closes a
    cycle, and that cycle is the back edge plus the tree path between its endpoints.
    Marking the back edge and those tree edges covers exactly the non-bridge edges."""
    n = adjacency.shape[0]
    parent = np.full(n, -1, dtype=int)
    depth = np.full(n, -1, dtype=int)
    for root in range(n):
        if depth[root] != -1:
            continue
        depth[root] = 0
        parent[root] = root
        stack = [root]
        while stack:
            node = stack.pop()
            for neighbor in np.flatnonzero(adjacency[node]):
                if depth[neighbor] == -1:
                    depth[neighbor] = depth[node] + 1
                    parent[neighbor] = node
                    stack.append(int(neighbor))

    ring = np.zeros_like(adjacency)
    for left in range(n):
        for right in np.flatnonzero(adjacency[left]):
            right = int(right)
            if left >= right:
                continue
            if parent[left] == right or parent[right] == left:
                continue  # tree edge; only back edges seed cycles
            ring[left, right] = ring[right, left] = True
            a, b = left, right
            while a != b:
                if depth[a] >= depth[b]:
                    ring[a, parent[a]] = ring[parent[a], a] = True
                    a = int(parent[a])
                else:
                    ring[b, parent[b]] = ring[parent[b], b] = True
                    b = int(parent[b])
    return ring


def annotate_bond_topology(bonds: BondMatrix) -> BondMatrix:
    """Return matcher-facing bonds with exact ring-or-chain topology per edge."""
    present = bond_presence(bonds)
    ring = ring_edge_mask(present)
    result = bonds.copy()
    result[present] &= np.uint8(~BOND_TOPOLOGY_MASK & 0xFF)
    result[present & ~ring] |= np.uint8(BOND_TOPOLOGY_CHAIN)
    result[ring] |= np.uint8(BOND_TOPOLOGY_RING)
    return result


def ring_info_from_bonds(bond_types: BondMatrix) -> RingInfo:
    return ring_info_from_presence(bond_presence(bond_types))


def _path_to_root(node: int, parent: ParentVector) -> list[int]:
    path = [node]
    while parent[node] != node and parent[node] != -1:
        node = int(parent[node])
        path.append(node)
    return path

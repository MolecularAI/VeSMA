"""Prototype ring-membership (`R`, `r`) perceptions, measured against the short-ring check.

Measurement scaffolding for the proposal to compute SMARTS `R`/`r` on the GPU from Horton
candidates and size-tiered GF(2) elimination. Nothing in `src/` calls this; it lives beside the
perf tests until the numbers say it should move.

Every function works on adjacency *rows*: one Python int per atom whose set bits are its
neighbours, the same representation the GPU kernels keep in `ulong rows[64]`. Each also counts
the inner-loop word operations it performs (`ops`), a hardware-neutral proxy for kernel work that
the timings in Python cannot give.

* `short_ring_edge_rows` -- the current device check: per edge, a bitset BFS of depth <= 7 with the
  edge removed (Metal/CUDA `bond_in_short_cycle_rows`).
* `smallest_cycle_r` -- the same BFS, unbounded, returning the distance: exact, order-independent
  smallest ring through each atom.
* `horton_relevant_ring_info` -- the proposal: Horton candidates (one canonical shortest-path tree
  per root), tiered GF(2) against the basis frozen at the previous size, per-atom `R` and `r`.
* `brute_relevant_ring_info` -- the same tiered elimination over *every* simple cycle: the exact
  relevant-cycle set, the reference for the prototype.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from vesma.adjacency import BoolBondMatrix
from vesma.rings import ring_edge_mask, walk_simple_cycles

__all__ = [
    "RingMembership",
    "adjacency_rows",
    "brute_relevant_ring_info",
    "cycle_rank",
    "horton_relevant_ring_info",
    "ring_rows",
    "short_ring_edge_rows",
    "smallest_cycle_r",
]

Rows = tuple[int, ...]
Edge = tuple[int, int]


@dataclass(frozen=True)
class RingMembership:
    """Per-atom `R` (rings through the atom) and `r` (smallest such ring, 0 if none)."""

    count: tuple[int, ...]
    smallest: tuple[int, ...]
    candidates: int
    relevant: int
    ops: int


def adjacency_rows(adjacency: BoolBondMatrix) -> Rows:
    return tuple(
        sum(1 << int(neighbor) for neighbor in np.flatnonzero(adjacency[atom]))
        for atom in range(adjacency.shape[0])
    )


def ring_rows(adjacency: BoolBondMatrix) -> Rows:
    """Rows restricted to ring (non-bridge) bonds: chains carry no cycles."""
    return adjacency_rows(ring_edge_mask(adjacency))


def edges_of(rows: Rows) -> list[Edge]:
    return [
        (left, right)
        for left, row in enumerate(rows)
        for right in bits(row)
        if left < right
    ]


def bits(mask: int) -> list[int]:
    found = []
    while mask:
        low = mask & -mask
        found.append(low.bit_length() - 1)
        mask ^= low
    return found


def cycle_rank(rows: Rows) -> int:
    """mu = E - V + C over the atoms that have any bond."""
    active = [atom for atom, row in enumerate(rows) if row]
    seen = 0
    components = 0
    for start in active:
        if seen >> start & 1:
            continue
        components += 1
        frontier = 1 << start
        seen |= frontier
        while frontier:
            reach = 0
            for atom in bits(frontier):
                reach |= rows[atom]
            frontier = reach & ~seen
            seen |= frontier
    return len(edges_of(rows)) - len(active) + components


def alternate_path_length(
    rows: Rows, left: int, right: int, limit: int
) -> tuple[int, int]:
    """Shortest left->right path avoiding the edge itself, up to `limit` edges (0 if none),
    and the word operations spent. One frontier step costs one OR per frontier atom."""
    target = 1 << right
    visited = 1 << left
    frontier = visited
    ops = 0
    for distance in range(1, limit + 1):
        reach = 0
        for atom in bits(frontier):
            row = rows[atom]
            if atom == left:
                row &= ~target
            reach |= row
            ops += 1
        frontier = reach & ~visited
        if distance >= 2 and frontier & target:
            return distance, ops
        if not frontier:
            return 0, ops
        visited |= frontier
    return 0, ops


def short_ring_edge_rows(rows: Rows) -> tuple[frozenset[Edge], int]:
    """Edges on a 3-8 ring, by the device's per-edge depth-7 BFS."""
    hits = []
    ops = 0
    for left, right in edges_of(rows):
        length, spent = alternate_path_length(rows, left, right, 7)
        ops += spent
        if length:
            hits.append((left, right))
    return frozenset(hits), ops


def smallest_cycle_r(rows: Rows) -> tuple[tuple[int, ...], int]:
    """Exact smallest ring through each atom: the shortest ring through an incident edge."""
    smallest = [0] * len(rows)
    ops = 0
    limit = len(rows)
    for left, right in edges_of(rows):
        length, spent = alternate_path_length(rows, left, right, limit)
        ops += spent
        if not length:
            continue
        size = length + 1
        for atom in (left, right):
            if smallest[atom] == 0 or size < smallest[atom]:
                smallest[atom] = size
    return tuple(smallest), ops


def horton_candidates(
    rows: Rows, edge_index: dict[Edge, int]
) -> tuple[dict[int, int], int]:
    """Horton's set as edge masks -> size: for every root and edge (x, y), the canonical
    shortest paths root->x and root->y plus the edge, kept when the paths meet only at root.
    """
    candidates: dict[int, int] = {}
    ops = 0
    edges = list(edge_index)
    for root in range(len(rows)):
        if not rows[root]:
            continue
        path_atoms = {root: 1 << root}
        path_edges = {root: 0}
        frontier = [root]
        while frontier:
            discovered: list[int] = []
            for atom in frontier:  # ascending, so the lowest-index parent wins
                for neighbor in bits(rows[atom]):
                    ops += 1
                    if neighbor in path_atoms:
                        continue
                    edge = (atom, neighbor) if atom < neighbor else (neighbor, atom)
                    path_atoms[neighbor] = path_atoms[atom] | 1 << neighbor
                    path_edges[neighbor] = path_edges[atom] | 1 << edge_index[edge]
                    discovered.append(neighbor)
            frontier = sorted(discovered)
        for left, right in edges:
            ops += 1
            if left not in path_atoms or right not in path_atoms:
                continue
            if path_atoms[left] & path_atoms[right] != 1 << root:
                continue
            edge_bit = 1 << edge_index[(left, right)]
            if (path_edges[left] | path_edges[right]) & edge_bit:
                continue  # a tree edge closes nothing
            mask = path_edges[left] ^ path_edges[right] | edge_bit
            candidates.setdefault(mask, mask.bit_count())
    return candidates, ops


def tiered_relevant(
    candidates: dict[int, int], mu: int, stop_at_full_rank: bool
) -> tuple[list[int], int]:
    """Cycles not in the span of strictly smaller ones (Vismara's relevance test), by GF(2)
    elimination tier by tier: each size is tested against the basis frozen at the sizes below.
    """
    basis: dict[int, int] = {}
    relevant: list[int] = []
    ops = 0

    def reduce(vector: int) -> int:
        nonlocal ops
        while vector:
            pivot = basis.get(vector.bit_length() - 1)
            if pivot is None:
                return vector
            vector ^= pivot
            ops += 1
        return 0

    by_size: dict[int, list[int]] = {}
    for mask, size in candidates.items():
        by_size.setdefault(size, []).append(mask)
    for size in sorted(by_size):
        tier = [mask for mask in sorted(by_size[size]) if reduce(mask)]
        relevant.extend(tier)
        for mask in tier:
            residue = reduce(mask)
            if residue:
                basis[residue.bit_length() - 1] = residue
        if stop_at_full_rank and len(basis) >= mu:
            break
    return relevant, ops


def membership_from_cycles(
    cycles: Sequence[int], edges: Sequence[Edge], atom_count: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    count = [0] * atom_count
    smallest = [0] * atom_count
    for mask in cycles:
        size = mask.bit_count()
        atoms = {atom for position in bits(mask) for atom in edges[position]}
        for atom in atoms:
            count[atom] += 1
            if smallest[atom] == 0 or size < smallest[atom]:
                smallest[atom] = size
    return tuple(count), tuple(smallest)


def horton_relevant_ring_info(rows: Rows) -> RingMembership:
    edges = edges_of(rows)
    edge_index = {edge: position for position, edge in enumerate(edges)}
    mu = cycle_rank(rows)
    candidates, candidate_ops = horton_candidates(rows, edge_index)
    relevant, gf2_ops = tiered_relevant(candidates, mu, stop_at_full_rank=True)
    count, smallest = membership_from_cycles(relevant, edges, len(rows))
    return RingMembership(
        count=count,
        smallest=smallest,
        candidates=len(candidates),
        relevant=len(relevant),
        ops=candidate_ops + gf2_ops,
    )


def brute_relevant_ring_info(adjacency: BoolBondMatrix) -> RingMembership:
    """Exact relevant cycles: every simple cycle, tiered GF(2). Exponential; small graphs only."""
    rows = ring_rows(adjacency)
    edges = edges_of(rows)
    edge_index = {edge: position for position, edge in enumerate(edges)}
    candidates: dict[int, int] = {}

    def collect(path: tuple[int, ...]) -> None:
        closed = (*path, path[0])
        mask = 0
        for left, right in zip(closed[:-1], closed[1:], strict=True):
            mask |= 1 << edge_index[(min(left, right), max(left, right))]
        candidates[mask] = len(path)

    present = np.zeros_like(adjacency)
    for left, right in edges:
        present[left, right] = present[right, left] = True
    walk_simple_cycles(present, max_size=adjacency.shape[0], on_cycle=collect)
    relevant, ops = tiered_relevant(
        candidates, cycle_rank(rows), stop_at_full_rank=False
    )
    count, smallest = membership_from_cycles(relevant, edges, len(rows))
    return RingMembership(
        count=count,
        smallest=smallest,
        candidates=len(candidates),
        relevant=len(relevant),
        ops=ops,
    )

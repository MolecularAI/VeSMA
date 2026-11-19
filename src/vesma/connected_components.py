from __future__ import annotations

import numpy as np

from .adjacency import BondMatrix, BoolBondMatrix, bond_presence
from .gmol import AtomMapVector
from .gmol_atoms import AtomIndices


def connected_components_from_bonds(bond_types: BondMatrix) -> list[AtomIndices]:
    return _connected_components(bond_presence(bond_types))


def grouping_agrees_with_components(
    groups: AtomMapVector,
    labels: AtomIndices,
) -> bool:
    """Does the component structure say what the zero-level grouping claimed?

    A group both separates and binds: atoms sharing a non-zero id must sit in one component,
    atoms with different non-zero ids in different ones. So the test is an equality between two
    relations rather than a count -- "same group" must agree with "same component" everywhere --
    which is what makes `(A.B)` a statement and not a no-op.

    Group ``0`` is not a group. It is dropped before the comparison, so an atom written outside
    any parenthesis constrains nothing and is constrained by nothing.

    Both callers reduce to this one question. The matcher asks it of a candidate assignment, with
    `labels` the target's components gathered through it; the retro path asks it of the edited
    product, with `labels` the components the edit actually produced. The first is a
    precondition on a match, the second a postcondition on a rewrite, and they share a predicate
    because they share a meaning.
    """
    grouped = groups != 0
    present = groups[grouped]
    components = labels[grouped]
    same_group = present[:, None] == present[None, :]
    same_component = components[:, None] == components[None, :]
    return bool(np.all(same_group == same_component))


def component_labels_from_bonds(bond_types: BondMatrix) -> AtomIndices:
    """Per atom, which connected component it belongs to.

    The same traversal as :func:`connected_components_from_bonds`, returned as a label per atom
    rather than a list of members, because the question a matcher asks is "are these two atoms in
    the same molecule" and a label answers it with one comparison.
    """
    labels = np.zeros(bond_types.shape[0], dtype=np.int64)
    for label, members in enumerate(connected_components_from_bonds(bond_types)):
        labels[members] = label
    return labels


def _connected_components(adjacency: BoolBondMatrix) -> list[AtomIndices]:
    n_atoms = adjacency.shape[0]
    visited = np.zeros(n_atoms, dtype=bool)
    components: list[AtomIndices] = []

    for root in range(n_atoms):
        if visited[root]:
            continue
        stack = [root]
        nodes: list[int] = []
        visited[root] = True
        while stack:
            node = stack.pop()
            nodes.append(node)
            for neighbor in np.flatnonzero(adjacency[node]):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))
        components.append(np.sort(np.array(nodes, dtype=np.int64)))

    return components

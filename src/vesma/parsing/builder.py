from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bonds import (
    BOND_KIND_MASK,
    BOND_TOPOLOGY_MASK,
    QBOND_DEFAULT,
)
from .errors import InvalidSmartsSyntaxError, UnsupportedSmartsFeatureError
from .ir import (
    AtomStep,
    ComponentCloseStep,
    ComponentOpenStep,
    LParenStep,
    RingStep,
    SmartsAtom,
    SmartsMol,
    Step,
)


@dataclass
class BuildState:
    atoms: list[SmartsAtom]
    bonds: list[dict[int, int]]
    current_atom: int | None
    branch_stack: list[int]
    open_rings: dict[str, tuple[int, int | None]]
    groups: list[int]
    """Which zero-level group each atom was written in, parallel to ``atoms``; ``0`` for none."""
    group: int = 0
    """The group being written now, or ``0`` outside one."""
    groups_seen: int = 0
    """How many groups have opened, so each gets its own id rather than being counted at the end."""


def build_smarts_mol(steps: tuple[Step, ...]) -> SmartsMol:
    state = BuildState(
        atoms=[],
        bonds=[],
        current_atom=None,
        branch_stack=[],
        open_rings={},
        groups=[],
    )

    for step in steps:
        if isinstance(step, AtomStep):
            apply_atom_step(state, step)
            continue
        if isinstance(step, RingStep):
            apply_ring_step(state, step)
            continue
        if isinstance(step, ComponentOpenStep):
            if state.group:
                raise UnsupportedSmartsFeatureError(
                    "Nested zero-level grouping is not supported; write each group once."
                )
            state.groups_seen += 1
            state.group = state.groups_seen
            # A group starts a fragment of its own: nothing bonds across its boundary, which is
            # what keeps `(A).(B)` from quietly joining A to B the way `A(B)` would.
            state.current_atom = None
            continue
        if isinstance(step, ComponentCloseStep):
            state.group = 0
            state.current_atom = None
            continue
        if isinstance(step, LParenStep):
            if state.current_atom is None:
                raise InvalidSmartsSyntaxError(
                    "Opening parenthesis requires a preceding atom."
                )
            state.branch_stack.append(state.current_atom)
            continue
        if not state.branch_stack:
            raise InvalidSmartsSyntaxError(
                "Closing parenthesis without matching opening parenthesis."
            )
        state.current_atom = state.branch_stack.pop()

    if state.branch_stack:
        raise InvalidSmartsSyntaxError("Unclosed parenthesis in SMARTS.")
    if state.open_rings:
        raise InvalidSmartsSyntaxError("Unclosed ring index in SMARTS.")
    if not state.atoms:
        raise InvalidSmartsSyntaxError("SMARTS must contain at least one atom.")

    # A bond between two different groups contradicts the groups: the bond says one molecule and
    # the groups say two, so nothing can ever match. Only a ring closure can write it -- a group
    # starts its own fragment, so no chain bond crosses the boundary -- and `(C1CC).(C1CC)` is the
    # shape, where the author almost certainly meant two separate rings and reused the index.
    # Refused rather than left to match nothing, because a pattern that silently matches nothing is
    # the one failure a library of 388 rules cannot be read for.
    #
    # A bond from an *ungrouped* atom into a group is allowed and means what it says: the group
    # constrains its own atoms, the bond puts the ungrouped one in with them.
    for left, neighbours in enumerate(state.bonds):
        for right in neighbours:
            left_group = state.groups[left]
            right_group = state.groups[right]
            if left_group and right_group and left_group != right_group:
                raise UnsupportedSmartsFeatureError(
                    "A bond crosses a zero-level group boundary, so the pattern asks for one "
                    "molecule and two at once; reuse of a ring index across groups is the usual "
                    "cause."
                )

    return SmartsMol(
        atoms=state.atoms.copy(),
        bonds=build_bonds(state.bonds),
        groups=np.array(state.groups, dtype=np.int32),
    )


def apply_atom_step(state: BuildState, step: AtomStep) -> None:
    atom_index = len(state.atoms)
    state.atoms.append(step.atom)
    state.bonds.append({})
    state.groups.append(state.group)

    if state.current_atom is not None:
        if step.leading_bond == 0:
            pass
        else:
            bond_flags = (
                step.leading_bond
                if step.leading_bond is not None
                else implicit_bond_flags(state.atoms[state.current_atom], step.atom)
            )
            set_bond(state.bonds, state.current_atom, atom_index, bond_flags)
    state.current_atom = atom_index


def apply_ring_step(state: BuildState, step: RingStep) -> None:
    if state.current_atom is None:
        raise InvalidSmartsSyntaxError("Ring index requires a current atom.")

    if step.ring_id not in state.open_rings:
        state.open_rings[step.ring_id] = (state.current_atom, step.leading_bond)
        return

    other_atom, opening_bond = state.open_rings.pop(step.ring_id)
    bond_flags = resolve_ring_bond(
        state.atoms[other_atom],
        state.atoms[state.current_atom],
        opening_bond,
        step.leading_bond,
    )
    if bond_flags == 0:
        return
    set_bond(state.bonds, state.current_atom, other_atom, bond_flags)


def resolve_ring_bond(
    opening_atom: SmartsAtom,
    closing_atom: SmartsAtom,
    opening: int | None,
    closing: int | None,
) -> int:
    if opening is not None and closing is not None:
        return merge_bond_flags(opening, closing)
    if opening is not None:
        return opening
    if closing is not None:
        return closing
    return implicit_bond_flags(opening_atom, closing_atom)


def implicit_bond_flags(left: SmartsAtom, right: SmartsAtom) -> int:
    # An omitted/default SMARTS bond means "single or aromatic" (Daylight semantics),
    # encoded as the accepted-class mask QBOND_DEFAULT so it is distinct from an
    # explicit single bond ``-``. When the graph is concretized into a molecule,
    # cmol_from_gmol resolves this back to a single or aromatic seed bond using the
    # atoms' aromatic intent.
    return QBOND_DEFAULT


def set_bond(
    bonds: list[dict[int, int]], left: int, right: int, bond_flags: int
) -> None:
    if left == right:
        raise InvalidSmartsSyntaxError("Self-bonds are unsupported.")

    current = bonds[left].get(right)
    if current is None:
        bonds[left][right] = bond_flags
        bonds[right][left] = bond_flags
        return
    merged = merge_bond_flags(current, bond_flags)
    if merged == current:
        return
    bonds[left][right] = merged
    bonds[right][left] = merged


def merge_bond_flags(left: int, right: int) -> int:
    if left == 0 or right == 0:
        if left == right:
            return 0
        raise InvalidSmartsSyntaxError("Conflicting ring closure bond types.")

    bond_classes = (left & BOND_KIND_MASK) & (right & BOND_KIND_MASK)
    topology = (left & BOND_TOPOLOGY_MASK) & (right & BOND_TOPOLOGY_MASK)
    if bond_classes == 0 or topology == 0:
        raise InvalidSmartsSyntaxError(
            "Conflicting bond assignments for the same atom pair."
        )
    return int(bond_classes | topology)


def build_bonds(
    bonds: list[dict[int, int]],
) -> np.ndarray[tuple[int, int], np.dtype[np.uint8]]:
    n_atoms = len(bonds)
    matrix = np.zeros((n_atoms, n_atoms), dtype=np.uint8)
    for left, neighbors in enumerate(bonds):
        for right, bond_flags in neighbors.items():
            matrix[left, right] = bond_flags
    return matrix

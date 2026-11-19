from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .adjacency import BondMatrix, validate_bond_flag_matrix
from .gmol_atoms import AtomMasks
from .parsing.ir import Expression, PropVal

AtomMapVector = np.ndarray[tuple[int], np.dtype[np.int32]]


def literal_text(literal: PropVal) -> str:
    sign = "!" if literal.negated else ""
    if literal.prop == "$":
        return f"{sign}$({literal.value})"
    return f"{sign}{literal.prop}={literal.value}"


def expression_text(expression: Expression) -> str:
    """A canonical spelling of an expression: equal expressions, equal text."""
    return ";".join(
        sorted(
            ",".join(
                sorted("&".join(sorted(map(literal_text, term))) for term in clause)
            )
            for clause in expression
        )
    )


@dataclass(frozen=True)
class AtomPredicate:
    """What one query atom asks that its independent masks cannot hold: the clauses of its
    expression naming a recursive ``$(...)``, or tying properties together, like ``[C,n]``.

    ``expression`` is an AND of ORs of conjunctions. A ``$`` literal holds where
    ``subpatterns[body]`` embeds rooted at its first atom; every other literal is a primitive
    tested on the atom itself. Clauses the masks hold stay there and are not repeated here.
    """

    expression: Expression
    subpatterns: tuple[tuple[str, GMol], ...]
    """Each recursive body this CNF names, by its text, compiled. A body may carry predicates
    of its own, which is how nesting is represented: a finite tree, not a loop."""

    @property
    def key(self) -> str:
        return expression_text(self.expression)

    @property
    def level(self) -> int:
        """1 + the deepest body it names; a level-k predicate needs only levels below k."""
        return max((subpattern_level(body) for _, body in self.subpatterns), default=0)


def subpattern_level(graph: GMol) -> int:
    return 1 + max((predicate.level for predicate in graph.predicates), default=0)


@dataclass(frozen=True)
class GMol:
    """Query molecule graph with SMARTS bond masks in `bonds`."""

    atom_masks: AtomMasks
    bonds: BondMatrix
    atom_map: AtomMapVector
    atom_group: AtomMapVector
    """Per atom, which zero-level group it was written in; ``0`` for none.

    Beside :attr:`atom_map` and not inside :class:`AtomMasks`, because it is not a mask. Every
    entry of `atom_masks` is ANDed against the molecule's own value for that atom, and the device
    does exactly that over words 4 upward -- a group id has no molecule counterpart to be ANDed
    with, so it would be ANDed against zero and no atom would ever match. A group is a statement
    about *pairs* of atoms: same non-zero id, same molecule; different non-zero ids, different
    molecules. Required rather than defaulted so that no construction site can forget it.
    """
    atom_predicate: AtomMapVector
    """Per atom, an index into :attr:`predicates`, or ``-1`` for an atom whose masks say it all.

    Beside the masks for the reason :attr:`atom_group` is: a predicate is a statement about the
    atom's neighbourhood, so it is evaluated against the molecule ahead of matching and enters
    as a restriction of that atom's candidates, never as a mask word.
    """
    predicates: tuple[AtomPredicate, ...]
    metal_cache: dict[str, np.ndarray] = field(
        default_factory=dict, repr=False, hash=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.bonds.dtype != np.uint8:
            raise ValueError("bonds dtype must be uint8.")
        if self.bonds.ndim != 2:
            raise ValueError("Bond matrix must be two-dimensional.")
        if self.bonds.shape != (self.atom_masks.size, self.atom_masks.size):
            raise ValueError("Bond matrix shape must match atom count.")
        validate_bond_flag_matrix(self.bonds)
        if self.atom_map.dtype != np.int32:
            raise ValueError("atom_map dtype must be int32.")
        if self.atom_map.ndim != 1:
            raise ValueError("Atom map must be one-dimensional.")
        if self.atom_map.shape != (self.atom_masks.size,):
            raise ValueError("Atom map shape must match atom count.")
        if self.atom_group.dtype != np.int32:
            raise ValueError("atom_group dtype must be int32.")
        if self.atom_group.ndim != 1:
            raise ValueError("Atom group must be one-dimensional.")
        if self.atom_group.shape != (self.atom_masks.size,):
            raise ValueError("Atom group shape must match atom count.")
        if self.atom_predicate.dtype != np.int32:
            raise ValueError("atom_predicate dtype must be int32.")
        if self.atom_predicate.shape != (self.atom_masks.size,):
            raise ValueError("Atom predicate shape must match atom count.")
        if np.any(self.atom_predicate < -1) or np.any(
            self.atom_predicate >= len(self.predicates)
        ):
            raise ValueError("Atom predicate index out of range.")

    @property
    def atom_count(self) -> int:
        return self.atom_masks.size

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BondFlagMatrix = np.ndarray[tuple[int, int], np.dtype[np.uint8]]
GroupVector = np.ndarray[tuple[int], np.dtype[np.int32]]


@dataclass(frozen=True)
class PropVal:
    prop: str
    value: str | int | None
    negated: bool = False

    def negate(self) -> PropVal:
        return PropVal(self.prop, self.value, negated=not self.negated)


type HighAndClause = list[PropVal]
type OrClause = list[HighAndClause]
type AtomFormula = list[OrClause]


@dataclass(frozen=True)
class SmartsAtom:
    # ``bracketed`` records syntax, not meaning: whether the author wrote the atom inside
    # ``[]``. It is the one fact that separates a SMILES molecule from a SMARTS query --
    # SMILES says an omitted hydrogen count inside brackets is H0, SMARTS says an omitted
    # property is simply not part of the pattern -- and the parser is the only place that
    # can still see it. See ``vesma.parsing.reading``.
    formula: AtomFormula
    is_aromatic: bool = False
    atom_map: int | None = None
    bracketed: bool = False


@dataclass(frozen=True)
class AtomStep:
    atom: SmartsAtom
    leading_bond: int | None = None


@dataclass(frozen=True)
class RingStep:
    ring_id: str
    leading_bond: int | None = None


@dataclass(frozen=True)
class LParenStep:
    pass


@dataclass(frozen=True)
class RParenStep:
    pass


@dataclass(frozen=True)
class ComponentOpenStep:
    """The `(` of a zero-level group -- `(A).(B)` -- as distinct from a branch's `(`.

    Both are written the same and only position tells them apart, so the parser decides and says
    which it found. The builder could not: a dot reaches it only as ``AtomStep.leading_bond == 0``,
    which makes `C(N)` and `C.(N)` arrive identical.
    """


@dataclass(frozen=True)
class ComponentCloseStep:
    pass


type Step = (
    AtomStep
    | RingStep
    | LParenStep
    | RParenStep
    | ComponentOpenStep
    | ComponentCloseStep
)

type Clause = frozenset[PropVal]
type Cnf = frozenset[Clause]

type Term = frozenset[PropVal]
"""A conjunction of literals. Its primitives together are one **box** -- a mask per property, as
one packed query row holds -- and its ``$`` literals are recursive roots it also requires. The
empty term is true."""
type Expression = frozenset[frozenset[Term]]
"""An AND of ORs of :data:`Term`: a CNF whose literals are conjunctions.

What a query atom asks, in a form that stays the size of what was written: ``[C,n]`` is one
clause of two boxes, where a CNF of single literals needs four clauses and grows exponentially
with every alternative. A CNF of single literals embeds as one-literal terms, which is how every
expression that independent masks can hold is still written. ``frozenset()`` is true and
``frozenset({frozenset()})`` false."""


@dataclass(frozen=True)
class SmartsMol:
    atoms: list[SmartsAtom]
    bonds: BondFlagMatrix
    groups: GroupVector
    """Per atom, which zero-level group it was written in; ``0`` for none.

    On the molecule rather than on :class:`SmartsAtom`, and required rather than defaulted, so
    that every rewrite has to say what happens to it. `reading.py` rebuilds atoms twice and
    reindexes them once; a defaulted field on the atom would be silently dropped there, which is
    the one failure this placement rules out.

    A group both **separates and binds**: atoms in different non-zero groups are different
    molecules, atoms sharing a non-zero group are the same molecule. So `(A.B)` is a statement,
    not a no-op -- it is how an intramolecular rule says so.
    """

    @property
    def atom_count(self) -> int:
        return len(self.atoms)


@dataclass(frozen=True)
class NormalizedSmartsAtom:
    expression: Expression
    atom_map: int | None = None
    bracketed: bool = False


@dataclass(frozen=True)
class NormalizedSmartsMol:
    atoms: list[NormalizedSmartsAtom]
    bonds: BondFlagMatrix
    groups: GroupVector
    """Carried through normalisation unchanged; see :class:`SmartsMol`."""
    smarts_string: str = ""

    @property
    def atom_count(self) -> int:
        return len(self.atoms)

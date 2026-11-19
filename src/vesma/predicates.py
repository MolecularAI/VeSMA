"""Atom predicates -- the recursive-SMARTS part of a query atom -- as sets of molecule atoms.

A predicate's answer for one molecule is the set of molecule atoms it holds at: a boolean vector
here, one ``u64`` on the device (a molecule has at most 64 atoms). Everything in this module is
the backend-neutral algebra on those sets; *finding* where a recursive body embeds is the
matcher's job, so the body roots come in as an argument.

``$(X)`` holds at atom ``a`` when some embedding of X maps X's first atom to ``a``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

from .gmol import AtomPredicate, GMol, subpattern_level
from .gmol_atoms import AtomMask
from .gmol_builder import literal_mask
from .parsing.errors import UnsupportedSmartsFeatureError
from .parsing.ir import PropVal

__all__ = [
    "PredicateTable",
    "atom_predicate_ids",
    "compile_predicate_table",
    "literal_atoms",
    "predicate_atoms",
    "require_no_predicates",
]


def require_no_predicates(graph: GMol, *, backend: str) -> None:
    """Refuse a pattern with atom predicates on a backend that cannot evaluate them yet.

    Silently packing only the masks would match more than the pattern says.
    """
    if graph.predicates:
        raise UnsupportedSmartsFeatureError(
            "Recursive SMARTS $(...), and properties tied together like [C,n], need atom "
            f"predicates, which the {backend} backend does not evaluate yet; use the CPU matcher."
        )


def literal_atoms(literal: PropVal, molecule: GMol) -> AtomMask:
    """The molecule atoms one primitive literal holds at, by the matcher's own mask overlap."""
    masks = molecule.atom_masks
    if literal.prop == "M":
        equal = masks.M == cast(int, literal.value)
        return ~equal if literal.negated else equal
    mask = literal_mask(literal)
    if literal.prop == "Z":
        lo = np.uint64(mask & ((1 << 64) - 1))
        hi = np.uint64(mask >> 64)
        return ((masks.Z_lo & lo) != 0) | ((masks.Z_hi & hi) != 0)
    return (getattr(masks, literal.prop) & mask) != 0


def predicate_atoms(
    predicate: AtomPredicate, molecule: GMol, roots: Mapping[str, AtomMask]
) -> AtomMask:
    """Evaluate a predicate's expression over the molecule's atoms.

    ``roots`` maps each recursive body the predicate names to the atoms it embeds at.
    """
    held = np.ones(molecule.atom_count, dtype=bool)
    for clause in predicate.expression:
        any_held = np.zeros(molecule.atom_count, dtype=bool)
        for term in clause:
            term_held = np.ones(molecule.atom_count, dtype=bool)
            for literal in term:
                term_held &= term_atoms(literal, molecule, roots)
            any_held |= term_held
        held &= any_held
    return held


def term_atoms(
    literal: PropVal, molecule: GMol, roots: Mapping[str, AtomMask]
) -> AtomMask:
    if literal.prop != "$":
        return literal_atoms(literal, molecule)
    embedded = roots[cast(str, literal.value)]
    return ~embedded if literal.negated else embedded


@dataclass(frozen=True)
class PredicateTable:
    """Every predicate a rule set uses, flat and deduplicated, for a device to evaluate by level.

    Two kinds of entry, each numbered globally:

    - **residuals** -- an atom's predicate CNF -- are what a pattern atom points at. Their roots are
      what the matcher ANDs into that atom's candidate row.
    - **subpatterns** -- recursive bodies -- are what a residual's ``$`` literals name. A body is
      matched like any pattern, its own atoms pointing at lower-level residuals.

    Levels are innermost first. A level-``k`` subpattern needs residuals below ``k`` only, and a
    level-``k`` residual needs subpatterns up to ``k``, so evaluation runs residuals at level 0
    (bodies that are pure primitives), then for each ``k``: subpatterns, then residuals. Nesting
    depth is data -- ``depth`` -- and never shapes code.
    """

    residuals: tuple[AtomPredicate, ...]
    residual_levels: tuple[int, ...]
    subpatterns: tuple[GMol, ...]
    subpattern_levels: tuple[int, ...]
    residual_ids: Mapping[str, int]
    subpattern_ids: Mapping[str, int]

    @property
    def depth(self) -> int:
        return max((*self.residual_levels, *self.subpattern_levels), default=0)

    def residuals_at(self, level: int) -> list[int]:
        return [i for i, k in enumerate(self.residual_levels) if k == level]

    def subpatterns_at(self, level: int) -> list[int]:
        return [i for i, k in enumerate(self.subpattern_levels) if k == level]

    def level_starts(self, levels: Sequence[int]) -> np.ndarray:
        """Where each level 0..depth begins in an id range sorted by level, plus the end."""
        return np.searchsorted(
            np.asarray(levels, dtype=np.int64), np.arange(self.depth + 2)
        ).astype(np.uint32)


def compile_predicate_table(patterns: Iterable[GMol]) -> PredicateTable:
    """Collect every residual and body reachable from ``patterns``, each once."""
    residuals: dict[str, AtomPredicate] = {}
    subpatterns: dict[str, GMol] = {}

    def visit(graph: GMol) -> None:
        for predicate in graph.predicates:
            if predicate.key in residuals:
                continue
            residuals[predicate.key] = predicate
            for body, subpattern in predicate.subpatterns:
                if body not in subpatterns:
                    subpatterns[body] = subpattern
                    visit(subpattern)

    for pattern in patterns:
        visit(pattern)
    # Numbered by level, so each level is one contiguous id range a device can dispatch over.
    by_level = sorted(residuals.items(), key=lambda item: item[1].level)
    bodies = sorted(subpatterns.items(), key=lambda item: subpattern_level(item[1]))
    return PredicateTable(
        residuals=tuple(p for _, p in by_level),
        residual_levels=tuple(p.level for _, p in by_level),
        subpatterns=tuple(g for _, g in bodies),
        subpattern_levels=tuple(subpattern_level(g) for _, g in bodies),
        residual_ids={key: i for i, (key, _) in enumerate(by_level)},
        subpattern_ids={body: i for i, (body, _) in enumerate(bodies)},
    )


def atom_predicate_ids(table: PredicateTable, graph: GMol) -> np.ndarray:
    """Per atom of ``graph``, its residual's global id in ``table``, or ``-1``."""
    local_to_global = np.array(
        [table.residual_ids[p.key] for p in graph.predicates] + [-1], dtype=np.int32
    )
    # Local -1 indexes the trailing -1.
    return local_to_global[graph.atom_predicate]

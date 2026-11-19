"""Backend-neutral compilation of atom predicates into GPU offset arrays."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

from .gmol import GMol
from .gmol_builder import literal_mask
from .gpu_atom_masks import ISOTOPE_ANY, ISOTOPE_SHIFT, ISOTOPE_WORD
from .limits import ATOM_MASK_WORD_COUNT
from .parsing.errors import UnsupportedSmartsFeatureError
from .parsing.ir import PropVal
from .predicates import PredicateTable, atom_predicate_ids


class UnresolvedPredicateError(RuntimeError):
    """A recursive body search exhausted its budget; its roots remain unknown."""


def packed(rows: list[np.ndarray], dtype: type) -> np.ndarray:
    return np.concatenate(rows) if rows else np.zeros(0, dtype=dtype)


NO_PREDICATE = np.uint32(0xFFFFFFFF)

PACKED_PROPS = ("C", "D", "H", "h", "v", "X", "R", "r", "x")


def pack_atom_predicates(
    table: PredicateTable, gmol: GMol, bucket_size: int
) -> np.ndarray:
    """A pattern's predicate ids in ``table``, one per atom slot, ``NO_PREDICATE`` elsewhere.

    Public and required for the reason :func:`pack_atom_groups` is. Not cached on the graph:
    the ids are global to one table, and the same graph may sit in several.
    """
    preds = np.full(bucket_size, NO_PREDICATE, dtype=np.uint32)
    preds[: gmol.atom_count] = atom_predicate_ids(table, gmol).astype(np.uint32)
    return preds


def pack_collection_atom_predicates(
    table: PredicateTable, gmols: Sequence[GMol], bucket_size: int
) -> np.ndarray | None:
    """:func:`pack_atom_predicates` for every graph, concatenated; ``None`` for a table without
    predicates -- every rule set that never writes ``$(`` -- which then uploads nothing.
    """
    if not table.residuals:
        return None
    return np.concatenate([pack_atom_predicates(table, g, bucket_size) for g in gmols])


TERM_NEGATED = 0x80000000
TERM_SUBPATTERN = 0x40000000


def literal_box(literal: PropVal) -> np.ndarray:
    """A packed query row admitting exactly what one primitive literal does: a one-property box.

    Every other word is all-ones, so the row is tested with the matcher's own mask overlap.
    """
    box = np.full(ATOM_MASK_WORD_COUNT, 0xFFFFFFFF, dtype=np.uint32)
    if literal.prop == "M":
        # Exact-or-any has no complement box; a negated mass is a negated factor instead.
        if literal.negated:
            raise UnsupportedSmartsFeatureError("A negated isotope has no box form.")
        box[ISOTOPE_WORD] = (cast(int, literal.value) << ISOTOPE_SHIFT) | 0xFFFF
        return box
    mask = literal_mask(literal)
    if literal.prop == "Z":
        box[:4] = [(mask >> shift) & 0xFFFFFFFF for shift in (0, 32, 64, 96)]
    elif literal.prop == "A":
        box[4] = mask
    else:
        word = 5 + PACKED_PROPS.index(literal.prop)
        # `x` shares word 13 with the isotope: its box says nothing about the mass.
        isotope = (ISOTOPE_ANY << ISOTOPE_SHIFT) if word == ISOTOPE_WORD else 0
        box[word] = isotope | mask
    return box


@dataclass(frozen=True)
class GpuPredicateProgram:
    """A :class:`PredicateTable` packed for the device, once per rule set.

    Bodies are packed exactly like patterns, in table (level) order. Residuals are an AND of ORs
    of conjunctions over offset arrays: residual ``r`` owns clauses
    ``clause_start[r]:clause_start[r+1]``, clause ``c`` owns terms ``term_start[c]:term_start[c+1]``,
    term ``t`` owns factors ``factor_start[t]:factor_start[t+1]``. A factor is a ``u32`` -- a
    probe index, or ``TERM_SUBPATTERN`` with a body index, maybe ``TERM_NEGATED``. A probe is a
    packed query row, the box of one term's primitives, one per distinct box.
    """

    residual_count: int
    body_atoms: np.ndarray
    body_rows: np.ndarray
    body_bonds: np.ndarray
    body_groups: np.ndarray
    body_preds: np.ndarray
    body_counts: np.ndarray
    probes: np.ndarray
    clause_start: np.ndarray
    term_start: np.ndarray
    factor_start: np.ndarray
    factors: np.ndarray
    residual_level_start: np.ndarray
    body_level_start: np.ndarray
    depth: int


def compile_predicate_program(
    table: PredicateTable,
    *,
    pack_atoms: Callable[[GMol], np.ndarray],
    pack_rows: Callable[[GMol], np.ndarray],
    pack_bonds: Callable[[GMol], np.ndarray],
    pack_groups: Callable[[GMol], np.ndarray],
) -> GpuPredicateProgram:
    bucket_size = 64
    bodies = table.subpatterns
    probe_ids: dict[frozenset[PropVal], int] = {}
    clause_start = [0]
    term_start = [0]
    factor_start = [0]
    factors: list[int] = []
    for residual in table.residuals:
        for clause in sorted(residual.expression, key=sorted_terms_key):
            for term in sorted(clause, key=sorted_literals_key):
                # A term's primitives -- negated ones included, as complement masks -- are one
                # box; each of its roots is a factor of its own.
                box = frozenset(
                    literal
                    for literal in term
                    if literal.prop != "$"
                    and not (literal.prop == "M" and literal.negated)
                )
                if box:
                    factors.append(probe_ids.setdefault(box, len(probe_ids)))
                for literal in sorted(term - box, key=literal_sort_key):
                    if literal.prop == "M":  # "not this mass": the mass's box, negated
                        positive = frozenset({literal.negate()})
                        index = probe_ids.setdefault(positive, len(probe_ids))
                        factors.append(index | TERM_NEGATED)
                for literal in sorted(term - box, key=literal_sort_key):
                    if literal.prop == "M":
                        continue
                    index = table.subpattern_ids[str(literal.value)] | TERM_SUBPATTERN
                    factors.append(index | (TERM_NEGATED if literal.negated else 0))
                factor_start.append(len(factors))
            term_start.append(len(factor_start) - 1)
        clause_start.append(len(term_start) - 1)

    return GpuPredicateProgram(
        residual_count=len(table.residuals),
        body_atoms=packed([pack_atoms(g) for g in bodies], np.uint32),
        body_rows=packed([pack_rows(g) for g in bodies], np.uint64),
        body_bonds=packed([pack_bonds(g) for g in bodies], np.uint8),
        body_groups=packed([pack_groups(g) for g in bodies], np.uint32),
        body_preds=packed(
            [pack_atom_predicates(table, g, bucket_size) for g in bodies], np.uint32
        ),
        body_counts=np.array([g.atom_count for g in bodies], dtype=np.uint32),
        probes=np.ascontiguousarray(
            np.stack([term_box_row(box) for box in probe_ids])
            if probe_ids
            else np.zeros((0, ATOM_MASK_WORD_COUNT), dtype=np.uint32)
        ).reshape(-1),
        clause_start=np.array(clause_start, dtype=np.uint32),
        term_start=np.array(term_start, dtype=np.uint32),
        factor_start=np.array(factor_start, dtype=np.uint32),
        factors=np.array(factors, dtype=np.uint32),
        residual_level_start=table.level_starts(table.residual_levels),
        body_level_start=table.level_starts(table.subpattern_levels),
        depth=table.depth,
    )


def literal_sort_key(literal: PropVal) -> tuple[str, str, bool]:
    return (literal.prop, repr(literal.value), literal.negated)


def sorted_literals_key(term: frozenset[PropVal]) -> list[tuple[str, str, bool]]:
    return sorted(literal_sort_key(literal) for literal in term)


def sorted_terms_key(
    clause: frozenset[frozenset[PropVal]],
) -> list[list[tuple[str, str, bool]]]:
    return sorted(sorted_literals_key(term) for term in clause)


def term_box_row(box: frozenset[PropVal]) -> np.ndarray:
    """The packed query row of a conjunction of primitives: each literal's row, word-wise ANDed."""
    row = np.full(ATOM_MASK_WORD_COUNT, 0xFFFFFFFF, dtype=np.uint32)
    for literal in box:
        row &= literal_box(literal)
    return row

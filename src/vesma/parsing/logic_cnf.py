from __future__ import annotations

import functools

from .ir import Clause, Cnf, PropVal

TRUE_CNF: Cnf = frozenset()
FALSE_CNF: Cnf = frozenset({frozenset()})


def and_all(parts: list[Cnf]) -> Cnf:
    result = TRUE_CNF
    for part in parts:
        if result == FALSE_CNF or part == FALSE_CNF:
            return FALSE_CNF
        if result == TRUE_CNF:
            result = part
            continue
        if part == TRUE_CNF:
            continue
        result = result | part
    return result


def or_all(parts: list[Cnf]) -> Cnf:
    if not parts:
        return FALSE_CNF

    result = parts[0]
    for part in parts[1:]:
        if result == TRUE_CNF or part == TRUE_CNF:
            return TRUE_CNF
        if result == FALSE_CNF:
            result = part
            continue
        if part == FALSE_CNF:
            continue
        result = frozenset(
            merge_clauses(clause_left, clause_right)
            for clause_left in result
            for clause_right in part
        )
    return result


def merge_clauses(left: Clause, right: Clause) -> Clause:
    return left | right


def clause_subsumes(left: Clause, right: Clause) -> bool:
    return left <= right


def unit_cnf(literal: PropVal) -> Cnf:
    return frozenset({frozenset({literal})})


@functools.lru_cache(maxsize=16384)
def simplify_cnf(cnf: Cnf) -> Cnf:
    if cnf == TRUE_CNF or cnf == FALSE_CNF:
        return cnf

    current = cnf
    while True:
        simplified = drop_tautological_clauses(current)
        if has_empty_clause(simplified):
            return FALSE_CNF

        simplified = apply_clause_subsumption(simplified)
        if has_complementary_units(simplified):
            return FALSE_CNF

        simplified = propagate_units(simplified)
        if simplified == FALSE_CNF:
            return FALSE_CNF
        if simplified == current:
            return simplified
        current = simplified


def drop_tautological_clauses(cnf: Cnf) -> Cnf:
    if cnf == TRUE_CNF or cnf == FALSE_CNF:
        return cnf
    return frozenset(clause for clause in cnf if not clause_is_tautology(clause))


def clause_is_tautology(clause: Clause) -> bool:
    return any(literal.negate() in clause for literal in clause)


def has_empty_clause(cnf: Cnf) -> bool:
    return any(not clause for clause in cnf)


def apply_clause_subsumption(cnf: Cnf) -> Cnf:
    if cnf == TRUE_CNF or cnf == FALSE_CNF:
        return cnf
    return frozenset(
        clause
        for clause in cnf
        if not any(other != clause and clause_subsumes(other, clause) for other in cnf)
    )


def has_complementary_units(cnf: Cnf) -> bool:
    units = unit_literals(cnf)
    return any(literal.negate() in units for literal in units)


def unit_literals(cnf: Cnf) -> frozenset[PropVal]:
    return frozenset(next(iter(clause)) for clause in cnf if len(clause) == 1)


def propagate_units(cnf: Cnf) -> Cnf:
    if cnf == TRUE_CNF or cnf == FALSE_CNF:
        return cnf

    units = unit_literals(cnf)
    if not units:
        return cnf

    propagated = {clause for clause in cnf if len(clause) == 1}
    negated_units = frozenset(literal.negate() for literal in units)

    for clause in cnf:
        if len(clause) == 1:
            continue
        if clause & units:
            continue
        reduced = frozenset(
            literal for literal in clause if literal not in negated_units
        )
        if not reduced:
            return FALSE_CNF
        propagated.add(reduced)

    return frozenset(propagated)

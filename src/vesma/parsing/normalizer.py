from __future__ import annotations

import functools
from typing import cast

from ..symbols import ELEMENT_TO_Z
from .builder import build_smarts_mol
from .errors import UnsupportedSmartsFeatureError
from .grammar_custom import parse_graph_steps
from .ir import (
    AtomFormula,
    Cnf,
    Expression,
    HighAndClause,
    NormalizedSmartsAtom,
    NormalizedSmartsMol,
    OrClause,
    PropVal,
    SmartsAtom,
    SmartsMol,
    Term,
)
from .logic_cnf import FALSE_CNF, TRUE_CNF, and_all, or_all, simplify_cnf, unit_cnf

EXACTLY_ONE_BUCKETS = frozenset({"D", "H", "v", "X"})
AT_LEAST_ONE_BUCKETS = frozenset({"h", "R", "r", "x"})
ELEMENT_TO_Z_MAP = ELEMENT_TO_Z


def normalize_smarts_mol(
    smarts_mol: SmartsMol, *, smarts_string: str = ""
) -> NormalizedSmartsMol:
    return NormalizedSmartsMol(
        atoms=[normalize_smarts_atom(atom) for atom in smarts_mol.atoms],
        bonds=smarts_mol.bonds,
        groups=smarts_mol.groups,
        smarts_string=smarts_string,
    )


def normalize_smarts_atom(atom: SmartsAtom) -> NormalizedSmartsAtom:
    return NormalizedSmartsAtom(
        expression=atom_expression(atom.formula),
        atom_map=atom.atom_map,
        bracketed=atom.bracketed,
    )


#: Terms one OR may expand to before normalization refuses. The box form is linear in what was
#: written; only an OR of multi-clause parts, or a negated multi-clause body, is expanded.
EXPRESSION_TERM_LIMIT = 4096

TRUE_EXPRESSION: Expression = frozenset()
FALSE_EXPRESSION: Expression = frozenset({frozenset()})


class ExpressionTooLargeError(UnsupportedSmartsFeatureError):
    pass


def atom_expression(formula: AtomFormula) -> Expression:
    """What an atom asks, as an :data:`Expression`.

    An expression the independent per-property masks can hold takes the single-literal CNF it
    always took, embedded term by term -- so every pattern that compiled before compiles to the
    same masks. Anything else -- a ``$(...)``, or alternatives that tie two properties together
    like ``[C,n]`` -- is read into box form instead.
    """
    if not formula_names_recursion(formula):
        cnf = normalize_atom_formula(formula)
        if all(single_mask_clause(clause) for clause in cnf):
            return literal_expression(cnf)
    return box_expression(formula)


def formula_names_recursion(formula: AtomFormula) -> bool:
    return any(literal.prop == "$" for oc in formula for hc in oc for literal in hc)


def single_mask_clause(clause: frozenset[PropVal]) -> bool:
    """One property's mask: what an independent mask holds, alone among CNF clauses."""
    props = {literal.prop for literal in clause}
    if props == {"M"}:
        return len(clause) == 1 and not next(iter(clause)).negated
    return len(props) == 1 and "$" not in props


def literal_expression(cnf: Cnf) -> Expression:
    return frozenset(
        frozenset(frozenset({literal}) for literal in clause) for clause in cnf
    )


def box_expression(formula: AtomFormula) -> Expression:
    return and_expressions(
        [
            or_expressions(
                [
                    and_expressions([literal_box_expression(lit) for lit in hc])
                    for hc in oc
                ]
            )
            for oc in formula
        ]
    )


def literal_box_expression(literal: PropVal) -> Expression:
    if literal.prop == "$":
        return recursive_expression(literal)
    return literal_expression(normalize_syntax_literal(literal))


def recursive_expression(literal: PropVal) -> Expression:
    """A ``$(...)``: inlined when its body is one atom, else a root term.

    A one-atom body says nothing about neighbours, so ``$([X])`` holds exactly where ``[X]``
    does. It stays a root -- matched, not evaluated -- only when inlining would outgrow
    :data:`EXPRESSION_TERM_LIMIT`.
    """
    atom = single_atom_body(cast(str, literal.value))
    if atom is not None:
        try:
            inner = atom_expression(atom.formula)
            return negate_expression(inner) if literal.negated else inner
        except ExpressionTooLargeError:
            pass
    return frozenset({frozenset({frozenset({literal})})})


def make_term(literals: frozenset[PropVal]) -> Term | None:
    """The conjunction, or None where it states a literal and its negation."""
    if any(literal.negate() in literals for literal in literals):
        return None
    return literals


def and_expressions(parts: list[Expression]) -> Expression:
    clauses: set[frozenset[Term]] = set()
    for part in parts:
        clauses |= part
    if frozenset() in clauses:
        return FALSE_EXPRESSION
    # A clause holding the true term is true; one-term clauses are one conjunction.
    clauses = {clause for clause in clauses if frozenset() not in clause}
    units = [next(iter(clause)) for clause in clauses if len(clause) == 1]
    if len(units) > 1:
        merged = make_term(frozenset().union(*units))
        if merged is None:
            return FALSE_EXPRESSION
        clauses = {clause for clause in clauses if len(clause) != 1}
        clauses.add(frozenset({merged}))
    return frozenset(clauses)


def or_expressions(parts: list[Expression]) -> Expression:
    """An OR of expressions as ONE clause: each part spread into its terms, then pooled.

    A part that is already one clause -- every alternative written as a conjunction -- spreads
    into exactly its terms, so a written ``[a,b,c]`` costs nothing extra.
    """
    if len(parts) == 1:
        return parts[0]
    if any(part == TRUE_EXPRESSION for part in parts):
        return TRUE_EXPRESSION
    terms: set[Term] = set()
    for part in parts:
        terms |= disjunctive_terms(part)
    if frozenset() in terms:
        return TRUE_EXPRESSION
    return frozenset({frozenset(absorbed(terms))})


def disjunctive_terms(expression: Expression) -> set[Term]:
    """The expression as an OR of conjunctions: one term per choice of one term per clause."""
    terms: set[Term] = {frozenset()}
    for clause in expression:
        grown: set[Term] = set()
        for term in terms:
            for choice in clause:
                merged = make_term(term | choice)
                if merged is not None:
                    grown.add(merged)
        if len(grown) > EXPRESSION_TERM_LIMIT:
            raise ExpressionTooLargeError(
                f"SMARTS atom expression expands past {EXPRESSION_TERM_LIMIT} alternatives."
            )
        terms = grown
    return terms


def absorbed(terms: set[Term]) -> set[Term]:
    """Drop every term that implies another: in an OR, the weaker one already covers it."""
    if len(terms) > 512:
        return terms
    return {t for t in terms if not any(other < t for other in terms)}


def negate_expression(expression: Expression) -> Expression:
    """``!E`` by De Morgan: an OR over clauses of the AND of each term's negation."""
    return or_expressions(
        [
            and_expressions(
                [
                    frozenset({frozenset(frozenset({lit.negate()}) for lit in term)})
                    for term in clause
                ]
            )
            for clause in expression
        ]
    )


def normalize_atom_formula(formula: AtomFormula) -> Cnf:
    return simplify_cnf(
        and_all([normalize_or_clause(or_clause) for or_clause in formula])
    )


def normalize_or_clause(or_clause: OrClause) -> Cnf:
    return or_all(
        [normalize_high_and_clause(high_and_clause) for high_and_clause in or_clause]
    )


def normalize_high_and_clause(high_and_clause: HighAndClause) -> Cnf:
    return and_all([normalize_syntax_literal(propval) for propval in high_and_clause])


@functools.lru_cache(maxsize=4096)
def single_atom_body(body: str) -> SmartsAtom | None:
    """The one atom of a recursive body, or None when the body has more than one."""
    mol = build_smarts_mol(parse_graph_steps(body))
    return mol.atoms[0] if mol.atom_count == 1 else None


def normalize_syntax_literal(propval: PropVal) -> Cnf:
    if propval.prop == "$":
        return unit_cnf(propval)

    if propval.prop == "symbol":
        return normalize_symbol_literal(
            cast(str, propval.value), negated=propval.negated
        )

    if propval.prop in {"Z", "M", "C"}:
        return unit_cnf(propval)

    if propval.prop in EXACTLY_ONE_BUCKETS or propval.prop in AT_LEAST_ONE_BUCKETS:
        return normalize_bucket_literal(propval)

    raise UnsupportedSmartsFeatureError(f"Unsupported SMARTS primitive: {propval.prop}")


def normalize_symbol_literal(symbol: str, *, negated: bool) -> Cnf:
    literals = symbol_positive_literals(symbol)
    if not literals:
        return FALSE_CNF if negated else TRUE_CNF
    if not negated:
        return and_all([unit_cnf(literal) for literal in literals])
    return or_all([unit_cnf(literal.negate()) for literal in literals])


def symbol_positive_literals(symbol: str) -> list[PropVal]:
    if symbol == "*":
        return []
    if symbol == "A":
        return [PropVal("A", 1)]
    if symbol == "a":
        return [PropVal("A", 1, negated=True)]
    if symbol.islower():
        element = symbol.capitalize()
        if element not in ELEMENT_TO_Z:
            raise UnsupportedSmartsFeatureError(
                f"Unsupported SMARTS primitive: {symbol}"
            )
        return [
            PropVal("Z", ELEMENT_TO_Z_MAP[element]),
            PropVal("A", 1, negated=True),
        ]
    if symbol not in ELEMENT_TO_Z:
        raise UnsupportedSmartsFeatureError(f"Unsupported SMARTS primitive: {symbol}")
    return [PropVal("Z", ELEMENT_TO_Z_MAP[symbol]), PropVal("A", 1)]


def normalize_bucket_literal(propval: PropVal) -> Cnf:
    if propval.value is None and propval.prop in EXACTLY_ONE_BUCKETS:
        literal = PropVal(propval.prop, 1, negated=propval.negated)
    elif propval.value is None and propval.prop in AT_LEAST_ONE_BUCKETS:
        literal = PropVal(propval.prop, 0, negated=not propval.negated)
    else:
        literal = PropVal(
            propval.prop,
            cast(int, propval.value),
            negated=propval.negated,
        )
    return unit_cnf(literal)

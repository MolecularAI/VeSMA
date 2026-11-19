from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np

from .gmol import AtomPredicate, GMol
from .gmol_atoms import (
    ALIPHATIC,
    BUCKET_DEFAULT_MASKS,
    MASS_ANY,
    AtomData,
    atom_masks_from_props,
    bucket_mask_exact,
    mask_for_charge,
    z_mask_for_numbers,
)
from .parsing.errors import (
    InconsistentSmartsAtomExpressionError,
    UnsupportedSmartsFeatureError,
)
from .parsing.ir import (
    Cnf,
    Expression,
    NormalizedSmartsAtom,
    NormalizedSmartsMol,
    PropVal,
    Term,
)
from .parsing.logic_cnf import FALSE_CNF
from .parsing.normalizer import single_mask_clause

SPLIT_HINT = "Expression is not representable as independent bucket masks; split it into multiple SMARTS patterns."
BUCKET_PRIMITIVES = frozenset({"D", "H", "h", "v", "X", "R", "r", "x"})


@dataclass(frozen=True)
class SmartsConversionInfo:
    smarts_string: str
    messages: list[str] = field(default_factory=list)


def gmol_from_smarts(smarts: str) -> tuple[GMol, SmartsConversionInfo]:
    """A query graph under the SMARTS reading: an unwritten property is unconstrained."""
    from .parsing import parse_smarts

    return gmol_from_normalized_smarts(parse_smarts(smarts))


def gmol_from_smiles(smiles: str) -> tuple[GMol, SmartsConversionInfo]:
    """The graph of the SMILES reading: what the text stated, and nothing more.

    Exact wherever the text was explicit, and wildcard wherever the hydrogen law has yet
    to supply a value. ``[CH4]`` fixes H; a bare ``C`` leaves H, degree and charge at
    ``0xFF``, so this graph matches any aliphatic carbon rather than methane alone. It is
    the unperceived intermediate ``cmol_from_smiles`` consumes, not a molecule expressed
    as a query.

    For a graph that matches one molecule and nothing else, perceive first::

        gmol_from_cmol(cmol_from_smiles(text)[0])
    """
    from .parsing import parse_smiles

    return gmol_from_normalized_smarts(parse_smiles(smiles))


def gmol_from_normalized_smarts(
    smarts_mol: NormalizedSmartsMol,
) -> tuple[GMol, SmartsConversionInfo]:
    messages: list[str] = []
    encoded = [
        encode_atom_expression(atom.expression, messages=messages)
        for atom in smarts_mol.atoms
    ]
    atom_data = [
        {**masks, **({"AtomMap": atom.atom_map} if atom.atom_map is not None else {})}
        for (masks, _), atom in zip(encoded, smarts_mol.atoms, strict=True)
    ]
    atom_map = np.array(
        [int(atom.get("AtomMap", 0)) for atom in atom_data],
        dtype=np.int32,
    )
    residuals = [residual for _, residual in encoded]
    keys = list(dict.fromkeys(residual for residual in residuals if residual))
    gmol = GMol(
        atom_masks=atom_masks_from_props(atom_data),
        bonds=smarts_mol.bonds,
        atom_map=atom_map,
        atom_group=smarts_mol.groups,
        atom_predicate=np.array(
            [keys.index(residual) if residual else -1 for residual in residuals],
            dtype=np.int32,
        ),
        predicates=tuple(atom_predicate(residual) for residual in keys),
    )
    return gmol, SmartsConversionInfo(
        smarts_string=smarts_mol.smarts_string,
        messages=list(messages),
    )


def atom_predicate(expression: Expression) -> AtomPredicate:
    bodies = sorted(
        {
            str(literal.value)
            for clause in expression
            for term in clause
            for literal in term
            if literal.prop == "$"
        }
    )
    return AtomPredicate(
        expression=expression,
        subpatterns=tuple((body, subpattern_gmol(body)) for body in bodies),
    )


@functools.lru_cache(maxsize=4096)
def subpattern_gmol(body: str) -> GMol:
    """A recursive body compiled as a query; shared, so equal bodies are one object."""
    return gmol_from_smarts(body)[0]


def encode_smarts_atom(
    atom: NormalizedSmartsAtom, *, messages: list[str] | None = None
) -> AtomData:
    """The atom's masks; whatever they cannot hold is its predicate, see
    :func:`encode_atom_expression`."""
    compiled = encode_atom_expression(atom.expression, messages=messages)[0].copy()
    if atom.atom_map is not None:
        compiled["AtomMap"] = atom.atom_map
    return compiled


def encode_atom_expression(
    expression: Expression, *, messages: list[str] | None = None
) -> tuple[AtomData, Expression]:
    """Split an atom's expression into independent masks and the residual predicate.

    A single-literal CNF of one-property clauses -- every expression masks alone can hold, as
    written before box form existed -- goes through :func:`encode_atom_cnf` unchanged. Otherwise
    each clause is a union of boxes: one that merges into a single box is masks; one that does
    not, or names a ``$(...)``, joins the residual, and its hull -- the smallest box holding it
    -- still narrows the masks, which is sound and lets every prefilter see it.
    """
    cnf = single_literal_cnf(expression)
    if cnf is not None:
        return encode_atom_cnf(cnf, messages=messages), frozenset()

    props: AtomData = {}
    residual: set[frozenset[Term]] = set()
    for clause in sorted(expression, key=_clause_terms_key):
        live = {
            term: box
            for term in clause
            if (box := term_box(term, messages=messages)) is not None
        }
        if not live:
            raise InconsistentSmartsAtomExpressionError(
                "SMARTS atom expression is unsatisfiable."
            )
        if frozenset() in live:
            continue
        pure = all(
            not box.negated_mass and not any(lit.prop == "$" for lit in term)
            for term, box in live.items()
        )
        merged = merge_boxes(list(live.values())) if pure else None
        if merged is not None:
            and_box_into(props, merged)
            continue
        residual.add(frozenset(live))
        and_box_into(props, box_hull(list(live.values())))
    return props, frozenset(residual)


def single_literal_cnf(expression: Expression) -> Cnf | None:
    """The expression as a mask-only single-literal CNF, or None if it is not one."""
    if any(len(term) != 1 for clause in expression for term in clause):
        return None
    cnf = frozenset(frozenset().union(*clause) for clause in expression)
    if not all(single_mask_clause(clause) for clause in cnf):
        return None
    return cnf


@dataclass(frozen=True)
class Box:
    """A term's primitives as one packed query row: a mask per property, full where silent.

    ``masks`` follows :data:`BOX_PROPS`. Isotope is exact-or-any, so ``mass`` is a value or
    ``MASS_ANY``; a negated isotope has no box form and is only flagged.
    """

    masks: tuple[int, ...]
    mass: int = MASS_ANY
    negated_mass: bool = False


BOX_PROPS = ("Z", "A", "C", "D", "H", "h", "v", "X", "R", "r", "x")
FULL_BOX_MASKS = tuple(BUCKET_DEFAULT_MASKS[prop] for prop in BOX_PROPS)


def term_box(term: Term, *, messages: list[str] | None = None) -> Box | None:
    """The box of a term's primitives, or None where they cannot all hold at one atom."""
    masks = list(FULL_BOX_MASKS)
    mass = MASS_ANY
    negated_mass = False
    for literal in sorted(term, key=_literal_key):
        if literal.prop == "$":
            continue
        if literal.prop == "M":
            if literal.negated:
                negated_mass = True
                continue
            value = int(cast(int, literal.value))
            if mass not in (MASS_ANY, value):
                return None
            mass = value
            continue
        index = BOX_PROPS.index(literal.prop)
        masks[index] &= literal_mask(literal, messages=messages)
        if masks[index] == 0:
            return None
    return Box(tuple(masks), mass, negated_mass)


def box_contains(outer: Box, inner: Box) -> bool:
    return all(i & ~o == 0 for o, i in zip(outer.masks, inner.masks, strict=True)) and (
        outer.mass in (MASS_ANY, inner.mass)
    )


def merge_boxes(boxes: list[Box]) -> Box | None:
    """The union as one box when it is one, by absorbing contained boxes and joining pairs
    that differ in a single property; None when what remains is still more than one box.
    """
    pool = list(dict.fromkeys(boxes))
    changed = True
    while changed and len(pool) > 1:
        changed = False
        # Deduplicated, so a box another one contains is strictly inside it.
        pool = [
            box
            for box in pool
            if not any(other != box and box_contains(other, box) for other in pool)
        ]
        for i, left in enumerate(pool):
            for right in pool[i + 1 :]:
                if left.mass != right.mass:
                    continue
                differ = [
                    k
                    for k, (a, b) in enumerate(
                        zip(left.masks, right.masks, strict=True)
                    )
                    if a != b
                ]
                if len(differ) == 1:
                    masks = list(left.masks)
                    masks[differ[0]] |= right.masks[differ[0]]
                    pool = [box for box in pool if box not in (left, right)]
                    pool.append(Box(tuple(masks), left.mass))
                    changed = True
                    break
            if changed:
                break
    return pool[0] if len(pool) == 1 else None


def box_hull(boxes: list[Box]) -> Box:
    masks = [0] * len(BOX_PROPS)
    for box in boxes:
        masks = [m | b for m, b in zip(masks, box.masks, strict=True)]
    masses = {box.mass for box in boxes}
    return Box(tuple(masks), masses.pop() if len(masses) == 1 else MASS_ANY)


def and_box_into(props: AtomData, box: Box) -> None:
    for prop, mask, full in zip(BOX_PROPS, box.masks, FULL_BOX_MASKS, strict=True):
        if mask == full:
            continue
        props[prop] = props.get(prop, full) & mask
        if props[prop] == 0:
            raise InconsistentSmartsAtomExpressionError(
                "SMARTS atom expression is unsatisfiable."
            )
    if box.mass != MASS_ANY and props.setdefault("M", box.mass) != box.mass:
        raise InconsistentSmartsAtomExpressionError(
            "SMARTS atom expression is unsatisfiable."
        )


def _clause_terms_key(clause: frozenset[Term]) -> list[list[tuple[str, str, bool]]]:
    return sorted(sorted(_literal_key(lit) for lit in term) for term in clause)


def encode_atom_cnf(cnf: Cnf, *, messages: list[str] | None = None) -> AtomData:
    if cnf == FALSE_CNF:
        raise InconsistentSmartsAtomExpressionError(
            "SMARTS atom expression is unsatisfiable."
        )

    props: AtomData = {}
    for clause in _sorted_clauses(cnf):
        if not clause:
            raise InconsistentSmartsAtomExpressionError(
                "SMARTS atom expression is unsatisfiable."
            )

        clause_props = frozenset(literal.prop for literal in clause)
        if clause_props == {"M"}:
            mass = exact_mass_from_literals(_sorted_literals(clause))
            if props.setdefault("M", mass) != mass:
                raise InconsistentSmartsAtomExpressionError(
                    "SMARTS atom expression is unsatisfiable."
                )
            continue
        if len(clause_props) > 1:
            joined = ", ".join(sorted(clause_props))
            raise InconsistentSmartsAtomExpressionError(
                f"SMARTS clause spans multiple properties ({joined}) and cannot be represented as independent masks. "
                f"{SPLIT_HINT}"
            )

        prop = cast(
            Literal["Z", "A", "C", "D", "H", "h", "v", "X", "R", "r", "x"],
            next(iter(clause_props)),
        )
        clause_mask = 0
        for literal in _sorted_literals(clause):
            clause_mask |= literal_mask(literal, messages=messages)

        if prop in props:
            props[prop] &= clause_mask
            if props[prop] == 0:
                raise InconsistentSmartsAtomExpressionError(
                    "SMARTS atom expression is unsatisfiable."
                )
            continue

        props[prop] = clause_mask

    return props


def exact_mass_from_literals(literals: Sequence[PropVal]) -> int:
    """Decode an isotope clause, which asserts exactly one mass.

    Unlike the bucket primitives, isotope is not a set: there is no mask to widen, so a
    negated mass or a clause offering alternatives has no representation and is rejected
    rather than approximated.
    """
    if len(literals) != 1 or literals[0].negated:
        raise UnsupportedSmartsFeatureError(
            "Isotope constraints must assert exactly one mass."
        )
    return int(cast(int, literals[0].value))


def literal_mask(literal: PropVal, *, messages: list[str] | None = None) -> int:
    if literal.prop == "Z":
        value = cast(int, literal.value)
        if not 0 <= value < 128:
            raise UnsupportedSmartsFeatureError(
                "Atomic number must be in range 0..127 for GMol encoding."
            )
        base_mask = z_mask_for_numbers([value])
    elif literal.prop == "A":
        value = cast(int, literal.value)
        if value != 1:
            raise UnsupportedSmartsFeatureError(
                "Unsupported normalized aromaticity literal; expected A=1 or !A=1."
            )
        base_mask = ALIPHATIC
    elif literal.prop == "C":
        value = cast(int, literal.value)
        if value < -3:
            _append_message(
                messages,
                f"Charge {value:+d} was lumped into the -3 bucket when converting SMARTS to GMol.",
            )
        elif value > 4:
            _append_message(
                messages,
                f"Charge {value:+d} was lumped into the +4 bucket when converting SMARTS to GMol.",
            )
        base_mask = mask_for_charge(value)
    elif literal.prop in BUCKET_PRIMITIVES:
        value = cast(int, literal.value)
        if value > 7:
            _append_message(
                messages,
                f"Bucket {literal.prop}={value} was lumped into the 7+ bucket when converting SMARTS to GMol.",
            )
        base_mask = bucket_mask_exact(value)
    else:
        raise UnsupportedSmartsFeatureError(
            f"Unsupported SMARTS primitive: {literal.prop}"
        )

    if literal.negated:
        return BUCKET_DEFAULT_MASKS[literal.prop] ^ base_mask
    return base_mask


def _append_message(messages: list[str] | None, message: str) -> None:
    if messages is not None:
        messages.append(message)


def _sorted_clauses(cnf: Cnf) -> list[frozenset[PropVal]]:
    return sorted(cnf, key=_clause_key)


def _sorted_literals(clause: frozenset[PropVal]) -> list[PropVal]:
    return sorted(clause, key=_literal_key)


def _clause_key(clause: frozenset[PropVal]) -> tuple[tuple[str, str, bool], ...]:
    return tuple(_literal_key(literal) for literal in _sorted_literals(clause))


def _literal_key(literal: PropVal) -> tuple[str, str, bool]:
    return (literal.prop, repr(literal.value), literal.negated)


def literal_text(literal: PropVal) -> str:
    prefix = "!" if literal.negated else ""
    return f"{prefix}{literal.prop}={literal.value}"


__all__ = [
    "SmartsConversionInfo",
    "encode_atom_cnf",
    "encode_smarts_atom",
    "gmol_from_normalized_smarts",
    "gmol_from_smarts",
    "gmol_from_smiles",
    "literal_mask",
]

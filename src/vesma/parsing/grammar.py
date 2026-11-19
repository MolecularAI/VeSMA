from __future__ import annotations

import re

import pyparsing as pp

from ..bonds import SMARTS_BOND_TOKENS, bond_flags_from_smarts_token
from ..symbols import ALL_ATOM_SYMBOLS
from .ir import (
    AtomFormula,
    AtomStep,
    LParenStep,
    PropVal,
    RingStep,
    RParenStep,
    SmartsAtom,
    Step,
)

pp.ParserElement.set_default_whitespace_chars("")

_sorted_syms = sorted(
    ALL_ATOM_SYMBOLS.split(), key=lambda symbol: len(symbol), reverse=True
)
_syms_pattern = "|".join(re.escape(s) for s in _sorted_syms)

CHIRALITY_PATTERN = re.compile(rf"^((?:\d+)?(?:#\d+|{_syms_pattern}))(@@?\??)(.*)$")


def is_aromatic_formula(formula: AtomFormula) -> bool:
    for or_clause in formula:
        for and_clause in or_clause:
            for literal in and_clause:
                if (
                    literal.prop == "symbol"
                    and isinstance(literal.value, str)
                    and (literal.value == "a" or literal.value.islower())
                    and not literal.negated
                ):
                    return True
    return False


def parse_bracket_atom_spec(tokens: pp.ParseResults) -> SmartsAtom:
    formula = tokens[0]
    atom_map = tokens.get("atom_map")
    # Ensure we don't return an empty ParseResults or string
    if not isinstance(atom_map, int):
        atom_map = None

    return SmartsAtom(
        formula=formula,
        is_aromatic=is_aromatic_formula(formula),
        atom_map=atom_map,
    )


def parse_unbracket_atom_spec(tokens: pp.ParseResults) -> SmartsAtom:
    sym = tokens.sym
    formula: AtomFormula = [[[PropVal("symbol", sym)]]]
    is_aromatic = sym == "a" or sym.islower()
    return SmartsAtom(formula=formula, is_aromatic=is_aromatic, atom_map=None)


def parse_signed_literal(tokens: pp.ParseResults) -> PropVal:
    literal = tokens.lit
    if tokens.neg is not None:
        return literal.negate()
    return literal


def parse_base_literal(tokens: pp.ParseResults) -> PropVal:
    if tokens.atomic_num:
        return PropVal("Z", int(tokens.atomic_num[1:]))
    if tokens.bucket:
        val = int(tokens.bucket_val) if tokens.bucket_val else None
        return PropVal(tokens.bucket, val)
    if tokens.charge:
        s = tokens.charge
        if s[-1].isdigit():
            val = int(s)
        else:
            val = len(s) if s[0] == "+" else -len(s)
        return PropVal("C", val)
    if tokens.sym:
        return PropVal("symbol", tokens.sym)
    return PropVal("M", int(tokens.mass))


def strip_ignored_chirality(text: str) -> str:
    def repl(m: re.Match) -> str:
        content = m.group(1)
        match = CHIRALITY_PATTERN.match(content)
        if match:
            return f"[{match.group(1)}{match.group(3)}]"
        return f"[{content}]"

    return re.sub(r"\[([^\]]+)\]", repl, text)


integer = pp.Regex(r"\d+").set_parse_action(lambda t: int(t[0]))

lbr, rbr = pp.Suppress("["), pp.Suppress("]")
high_and_op = pp.Suppress("&")
or_op = pp.Suppress(",")
low_and_op = pp.Suppress(";")
not_op = pp.Literal("!")

ATOMIC_NUM = r"(?P<atomic_num>#\d+)"
BUCKET = r"(?P<bucket>[DHhRrvXx])(?P<bucket_val>\d+)?"
CHARGE = r"(?P<charge>[+-]\d+|[+]+|[-]+)"
SYMBOL = rf"(?P<sym>{_syms_pattern})"
MASS = r"(?P<mass>\d+)"

BASE_LITERAL_PATTERN = f"{ATOMIC_NUM}|{BUCKET}|{CHARGE}|{SYMBOL}|{MASS}"

base_literal = pp.Regex(BASE_LITERAL_PATTERN).set_parse_action(parse_base_literal)

signed_literal = pp.Optional(not_op, default=None)("neg") + base_literal("lit")
signed_literal = signed_literal.set_parse_action(parse_signed_literal)

high_and_clause = signed_literal + pp.ZeroOrMore(
    pp.Optional(high_and_op) + signed_literal
)
high_and_clause = pp.Group(high_and_clause, aslist=True)

or_clause = high_and_clause + pp.ZeroOrMore(or_op + high_and_clause)
or_clause = pp.Group(or_clause, aslist=True)

atom_props_expr = pp.Group(
    or_clause + pp.ZeroOrMore(low_and_op + or_clause), aslist=True
)

atom_mapping = pp.Suppress(":") + integer("atom_map")
bracket_atom = lbr + atom_props_expr("expr") + pp.Optional(atom_mapping) + rbr
bracket_atom = bracket_atom.set_parse_action(parse_bracket_atom_spec)

unbracket_atom = pp.Regex(r"(?P<sym>Cl|Br|[BCNOPSFIbcnopsa\*])").set_parse_action(
    parse_unbracket_atom_spec
)

atom = bracket_atom | unbracket_atom

bond_token = pp.one_of(list(SMARTS_BOND_TOKENS))
bond_token = bond_token.set_parse_action(
    lambda tokens: bond_flags_from_smarts_token(tokens[0])
)

ring_id = pp.Regex(r"\d|%\d{2}").set_parse_action(
    lambda t: t[0][1:] if t[0].startswith("%") else t[0]
)

first_atom_step = atom.copy().set_parse_action(
    lambda tokens: AtomStep(atom=tokens[0], leading_bond=None)
)

lparen_step = pp.Suppress("(")
lparen_step = lparen_step.set_parse_action(pp.replace_with(LParenStep()))

rparen_step = pp.Suppress(")")
rparen_step = rparen_step.set_parse_action(pp.replace_with(RParenStep()))

# Factor out bond_token to avoid backtracking
non_first_step = pp.Optional(bond_token, default=None)("leading_bond") + (
    atom("atom") | ring_id("ring_id")
)


def parse_non_first_step(tokens: pp.ParseResults) -> Step:
    if "atom" in tokens:
        return AtomStep(atom=tokens.atom[0], leading_bond=tokens.leading_bond)
    return RingStep(ring_id=tokens.ring_id[0], leading_bond=tokens.leading_bond)


non_first_step = non_first_step.set_parse_action(parse_non_first_step)

graph_step = non_first_step | lparen_step | rparen_step
graph = first_atom_step + pp.ZeroOrMore(graph_step)


def parse_atom_formula(text: str) -> AtomFormula:
    return atom_props_expr.parse_string(text, parse_all=True)[0]


def parse_smarts_atom(text: str) -> SmartsAtom:
    return atom.parse_string(strip_ignored_chirality(text), parse_all=True)[0]


def parse_graph_steps(text: str) -> tuple[Step, ...]:
    parsed = graph.parse_string(strip_ignored_chirality(text), parse_all=True)
    return tuple(parsed)

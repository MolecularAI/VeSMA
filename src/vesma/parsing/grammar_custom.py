from __future__ import annotations

import re

from ..bonds import SMARTS_BOND_TOKENS, bond_flags_from_smarts_token
from ..symbols import ALL_ATOM_SYMBOLS
from .errors import SmartsError
from .ir import (
    AtomFormula,
    AtomStep,
    ComponentCloseStep,
    ComponentOpenStep,
    LParenStep,
    PropVal,
    RingStep,
    RParenStep,
    SmartsAtom,
    Step,
)

_sorted_syms = sorted(ALL_ATOM_SYMBOLS.split(), key=lambda s: len(s), reverse=True)
_syms_pattern = "|".join(re.escape(s) for s in _sorted_syms)

CHIRALITY_PATTERN = re.compile(rf"^((?:\d+)?(?:#\d+|{_syms_pattern}))(@@?\??)(.*)$")

# Two-letter element symbols are tried before the single-letter bucket primitives,
# because `re` alternation is ordered and `D H R X` are both primitives and the first
# letter of an element. Without this, `[Hg]` fails to parse and `[Rn]`, `[Rb]`, `[Rh]`
# and `[Ho]` silently become a primitive plus a *different element* -- `[Rn]` read as
# ring-membership plus aromatic nitrogen, which is ammonia rather than radon.
#
# Preferring the element matches RDKit, which reads every one of these as the element in
# both languages. The collisions are real -- `[Hs]` could be hassium or "one hydrogen and
# an aromatic sulfur" -- and RDKit resolves each to the element; `[Hn]` and `[Hc]`, whose
# two letters name no element, still read as a count plus an aromatic atom.
_two_letter_syms = sorted(s for s in ALL_ATOM_SYMBOLS.split() if len(s) == 2)
_two_letter_pattern = "|".join(re.escape(s) for s in _two_letter_syms)

TWO_LETTER_SYMBOL = rf"(?P<two_letter_sym>{_two_letter_pattern})"
ATOMIC_NUM = r"(?P<atomic_num>#\d+)"
BUCKET = r"(?P<bucket>[DHhRrvXx])(?P<bucket_val>\d+)?"
CHARGE = r"(?P<charge>[+-]\d+|[+]+|[-]+)"
SYMBOL = rf"(?P<sym>{_syms_pattern})"
MASS = r"(?P<mass>\d+)"

BASE_LITERAL_PATTERN = (
    f"{ATOMIC_NUM}|{TWO_LETTER_SYMBOL}|{BUCKET}|{CHARGE}|{SYMBOL}|{MASS}"
)
LITERAL_RE = re.compile(BASE_LITERAL_PATTERN)
UNBRACKET_RE = re.compile(r"(Cl|Br|[BCNOPSFIbcnopsa\*])")


def strip_ignored_chirality(text: str) -> str:
    def repl(m: re.Match) -> str:
        content = m.group(1)
        match = CHIRALITY_PATTERN.match(content)
        if match:
            return f"[{match.group(1)}{match.group(3)}]"
        return f"[{content}]"

    # Innermost brackets only: a recursive ``[$([C@H](N)O)]`` nests one bracket atom inside
    # another, and the inner one is the one carrying the chirality.
    return re.sub(r"\[([^\[\]]+)\]", repl, text)


def recursive_body_end(text: str, start: int) -> int:
    """Index of the ``)`` closing a ``$(`` whose body begins at ``start``.

    Only parentheses are counted. A bracket atom never contains a bare one: the only ``(``
    a bracket can hold opens a nested ``$(``, which is itself balanced.
    """
    depth = 1
    for pos in range(start, len(text)):
        if text[pos] == "(":
            depth += 1
        elif text[pos] == ")":
            depth -= 1
            if depth == 0:
                return pos
    raise SmartsError("Unclosed recursive SMARTS $(")


def elemental_hydrogen(formula: AtomFormula) -> AtomFormula:
    """Read a bracket whose only content is a bare ``H`` as the hydrogen *atom*.

    ``H`` is both an element and the total-hydrogen-count primitive, and which one a
    bracket means depends on what else is in it. Daylight's SMARTS table carries the
    footnote "Semantics of [H] changed in v4.5" for exactly this, and RDKit implements the
    post-4.5 reading in both languages: ``[H]``, ``[2H]`` and ``[H+]`` are hydrogen atoms,
    while the ``H`` of ``[CH3]`` or ``[nH]`` is a count, and so is the one in ``[H,Cl]``
    -- where the alternation makes the bracket an expression rather than an atom.

    So the rewrite is deliberately narrow: a single conjunction, naming no element, whose
    one ``H`` carries no number. ``[H1]`` stays a count and names no element, as it does
    for RDKit. ``[HH]`` is left alone too; OpenSMILES calls it illegal, and it refuses
    here rather than being guessed at.
    """
    if len(formula) != 1 or len(formula[0]) != 1:
        return formula
    literals = formula[0][0]
    # ``#7`` names an element as surely as ``N`` does: ``[#7H]`` is an NH, not N plus H.
    if any(literal.prop in ("symbol", "Z") for literal in literals):
        return formula
    hydrogens = [literal for literal in literals if literal.prop == "H"]
    if len(hydrogens) != 1:
        return formula
    written = hydrogens[0]
    if written.value is not None or written.negated:
        return formula
    return [
        [
            [
                PropVal("symbol", "H") if literal is written else literal
                for literal in literals
            ]
        ]
    ]


class CustomParser:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.length = len(text)

    def peek(self) -> str:
        if self.pos < self.length:
            return self.text[self.pos]
        return ""

    def consume(self, s: str) -> bool:
        if self.peek() == s:
            self.pos += 1
            return True
        return False

    def parse_signed_literal(self) -> PropVal:
        negated = self.consume("!")
        if self.text.startswith("$(", self.pos):
            return self.parse_recursive_literal(negated=negated)
        match = LITERAL_RE.match(self.text, self.pos)
        if not match:
            raise SmartsError(f"Expected literal at {self.pos}")
        self.pos = match.end()
        d = match.groupdict()

        if d["atomic_num"]:
            prop = "Z"
            val = int(d["atomic_num"][1:])
        elif d["two_letter_sym"]:
            prop = "symbol"
            val = d["two_letter_sym"]
        elif d["bucket"]:
            prop = d["bucket"]
            v = d["bucket_val"]
            val = int(v) if v else None
        elif d["charge"]:
            s = d["charge"]
            if s[-1].isdigit():
                val = int(s)
            else:
                val = len(s) if s[0] == "+" else -len(s)
            prop = "C"
        elif d["sym"]:
            prop = "symbol"
            val = d["sym"]
        elif d["mass"]:
            prop = "M"
            val = int(d["mass"])
        else:
            raise SmartsError("Invalid literal")

        p = PropVal(prop, val)
        if negated:
            return p.negate()
        return p

    def parse_recursive_literal(self, *, negated: bool) -> PropVal:
        """A ``$(...)`` primitive, kept as the body's text until it is compiled.

        The text is what makes the literal hashable, so it can sit in a CNF clause like any
        other primitive, and it is the key equal recursive bodies are shared by.
        """
        start = self.pos + 2
        end = recursive_body_end(self.text, start)
        body = self.text[start:end]
        if not body:
            raise SmartsError("Empty recursive SMARTS $()")
        self.pos = end + 1
        literal = PropVal("$", body)
        return literal.negate() if negated else literal

    def parse_high_and_clause(self) -> list[PropVal]:
        clause = [self.parse_signed_literal()]
        while self.pos < self.length:
            p = self.peek()
            if p == "&":
                self.pos += 1
                clause.append(self.parse_signed_literal())
            elif p in (",", ";", "]", ":"):
                break
            else:
                # implicit high and
                if (
                    p == "!"
                    or self.text.startswith("$(", self.pos)
                    or LITERAL_RE.match(self.text, self.pos)
                ):
                    clause.append(self.parse_signed_literal())
                else:
                    break
        return clause

    def parse_or_clause(self) -> list[list[PropVal]]:
        clauses = [self.parse_high_and_clause()]
        while self.consume(","):
            clauses.append(self.parse_high_and_clause())
        return clauses

    def parse_atom_props_expr(self) -> AtomFormula:
        expr = [self.parse_or_clause()]
        while self.consume(";"):
            expr.append(self.parse_or_clause())
        return expr

    def parse_atom_mapping(self) -> int | None:
        if self.consume(":"):
            m = re.match(r"\d+", self.text[self.pos :])
            if m:
                self.pos = m.end() + self.pos
                return int(m.group(0))
        return None

    def parse_bracket_atom(self) -> SmartsAtom:
        if not self.consume("["):
            raise SmartsError("Expected [")
        formula = self.parse_atom_props_expr()
        atom_map = self.parse_atom_mapping()
        if not self.consume("]"):
            raise SmartsError(f"Expected ] at {self.pos}")

        is_arom = False
        for or_clause in formula:
            for and_clause in or_clause:
                for literal in and_clause:
                    if (
                        literal.prop == "symbol"
                        and isinstance(literal.value, str)
                        and (literal.value == "a" or literal.value.islower())
                        and not literal.negated
                    ):
                        is_arom = True

        return SmartsAtom(
            formula=elemental_hydrogen(formula),
            is_aromatic=is_arom,
            atom_map=atom_map,
            bracketed=True,
        )

    def parse_unbracket_atom(self) -> SmartsAtom:
        m = UNBRACKET_RE.match(self.text, self.pos)
        if m:
            sym = m.group(1)
            self.pos = m.end()
            formula: AtomFormula = [[[PropVal("symbol", sym)]]]
            is_arom = sym == "a" or sym.islower()
            return SmartsAtom(
                formula=formula,
                is_aromatic=is_arom,
                atom_map=None,
                bracketed=False,
            )
        raise SmartsError(f"Expected unbracketed atom at {self.pos}")

    def parse_atom(self) -> SmartsAtom:
        if self.peek() == "[":
            return self.parse_bracket_atom()
        return self.parse_unbracket_atom()

    def parse_bond(self) -> int | None:
        flags = 0
        parsed_any = False

        # Sort tokens by length descending to match longest first
        sorted_tokens = sorted(SMARTS_BOND_TOKENS, key=lambda t: len(t), reverse=True)

        while self.pos < self.length:
            matched = False
            for token in sorted_tokens:
                if self.text.startswith(token, self.pos):
                    self.pos += len(token)
                    flags |= bond_flags_from_smarts_token(token)
                    parsed_any = True
                    matched = True
                    break

            if matched:
                if self.peek() == ",":
                    self.pos += 1
                    continue
                else:
                    break
            else:
                break
        return flags if parsed_any else None

    def parse_ring_id(self) -> str | None:
        if self.peek() == "%":
            if (
                self.pos + 2 < self.length
                and self.text[self.pos + 1 : self.pos + 3].isdigit()
            ):
                val = self.text[self.pos + 1 : self.pos + 3]
                self.pos += 3
                return val
        elif self.peek().isdigit():
            val = self.text[self.pos]
            self.pos += 1
            return val
        return None


def parse_atom_formula(text: str) -> AtomFormula:
    parser = CustomParser(text)
    res = parser.parse_atom_props_expr()
    if parser.pos != parser.length:
        raise SmartsError("Trailing characters")
    return res


def parse_smarts_atom(text: str) -> SmartsAtom:
    parser = CustomParser(strip_ignored_chirality(text))
    res = parser.parse_atom()
    if parser.pos != parser.length:
        raise SmartsError("Trailing characters")
    return res


def parse_graph_steps(text: str) -> tuple[Step, ...]:
    """Tokenise a graph into steps, telling a **component group** from a **branch**.

    Both are written `(` and only position distinguishes them: a parenthesis standing where an
    *atom* is expected -- at the start of the pattern, or just after a top-level dot -- opens a
    zero-level group, the `(A).(B)` of the SMARTS specification; one standing after an atom opens a
    branch. The decision has to be made here, because a dot reaches the builder only as
    ``AtomStep.leading_bond == 0`` and `C(N)` and `C.(N)` would arrive indistinguishable.

    Nothing about *depth* is inferred: a group may only open at the top level, so the two parsed
    before this -- a leading `(` ("Failed to parse initial atom") and a `.(` ("Dangling bond") --
    are the only two refusals this lifts.
    """
    parser = CustomParser(strip_ignored_chirality(text))
    steps: list[Step] = []
    # One flag per open parenthesis: True where it opened a component group, False a branch. The
    # closing parenthesis has no way of its own to tell which it ends.
    open_groups: list[bool] = []
    started = False

    while parser.pos < parser.length:
        char = parser.peek()
        if char == ")":
            if not open_groups:
                raise SmartsError(
                    "Closing parenthesis without matching opening parenthesis"
                )
            parser.pos += 1
            steps.append(ComponentCloseStep() if open_groups.pop() else RParenStep())
            continue
        if char == "(":
            parser.pos += 1
            # A group only at the top level and only where an atom was due.
            component = not open_groups and not started
            open_groups.append(component)
            steps.append(ComponentOpenStep() if component else LParenStep())
            continue
        if not started:
            try:
                atom = parser.parse_atom()
            except SmartsError as e:
                raise SmartsError(f"Failed to parse initial atom: {e}") from e
            steps.append(AtomStep(atom=atom, leading_bond=None))
            started = True
            continue

        bond = parser.parse_bond()
        ring = parser.parse_ring_id()
        if ring is not None:
            steps.append(RingStep(ring_id=ring, leading_bond=bond))
            continue
        if bond == 0 and not open_groups and parser.peek() == "(":
            # A top-level dot then a parenthesis: the dot is what the grouping already says, so it
            # is spent here rather than carried onto the group's first atom.
            parser.pos += 1
            open_groups.append(True)
            steps.append(ComponentOpenStep())
            continue
        try:
            atom = parser.parse_atom()
            steps.append(AtomStep(atom=atom, leading_bond=bond))
        except SmartsError as e:
            if bond is not None:
                raise SmartsError("Dangling bond") from e
            break

    if open_groups:
        raise SmartsError("Unclosed parenthesis")
    if parser.pos != parser.length:
        raise SmartsError(f"Unparsed trailing characters: {parser.text[parser.pos :]}")

    return tuple(steps)

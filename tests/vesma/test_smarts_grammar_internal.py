from __future__ import annotations

import pytest

from vesma.bonds import bond_flags_from_smarts_token
from vesma.gmol_atoms import bucket_mask_exact
from vesma.gmol_builder import encode_smarts_atom
from vesma.parsing.errors import SmartsError
from vesma.parsing.grammar_custom import (
    parse_atom_formula,
    parse_graph_steps,
    parse_smarts_atom,
)
from vesma.parsing.ir import (
    AtomStep,
    LParenStep,
    PropVal,
    RingStep,
    RParenStep,
    SmartsAtom,
)
from vesma.parsing.normalizer import normalize_smarts_atom
from vesma.symbols import ELEMENT_TO_Z


def test_parse_atom_formula_preserves_adjacent_high_and_literals() -> None:
    formula = parse_atom_formula("CH2")
    assert len(formula) == 1
    assert len(formula[0]) == 1
    assert formula[0][0][0] == PropVal("symbol", "C")
    assert formula[0][0][1] == PropVal("H", 2)


def test_parse_atom_formula_accepts_optional_ampersand_in_high_and_clause() -> None:
    assert parse_atom_formula("C13") == parse_atom_formula("C&13")


def test_parse_atom_formula_keeps_multi_digit_bucket_values() -> None:
    formula_h = parse_atom_formula("H12")
    formula_r = parse_atom_formula("R10")
    assert formula_h[0][0][0] == PropVal("H", 12)
    assert formula_r[0][0][0] == PropVal("R", 10)


def test_parse_atom_formula_emits_canonical_numeric_prop_names() -> None:
    atomic_number = parse_atom_formula("#6")
    mass = parse_atom_formula("13")
    assert atomic_number[0][0][0] == PropVal("Z", 6)
    assert mass[0][0][0] == PropVal("M", 13)


def test_parse_atom_formula_preserves_operator_precedence() -> None:
    formula = parse_atom_formula("c,n;H1")
    assert len(formula) == 2
    assert formula[0][0][0] == PropVal("symbol", "c")
    assert formula[0][1][0] == PropVal("symbol", "n")
    assert formula[1][0][0] == PropVal("H", 1)


def test_smarts_grammar_atom_formula_is_three_levels_of_lists() -> None:
    parsed = parse_atom_formula("C,N;H1")
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert isinstance(parsed[0], list)
    assert len(parsed[0]) == 2
    assert isinstance(parsed[0][0], list)
    assert len(parsed[0][0]) == 1
    assert isinstance(parsed[0][0][0], PropVal)
    assert parsed[0][0][0].negated is False


@pytest.mark.parametrize("expr", ["+9", "-9", "++", "--", "+10", "-25"])
def test_parse_atom_formula_parses_charge_tokens(expr: str) -> None:
    formula = parse_atom_formula(expr)
    assert formula[0][0][0].prop == "C"


def test_parse_atom_formula_parses_charge_values() -> None:
    assert parse_atom_formula("+1")[0][0][0].value == 1
    assert parse_atom_formula("-1")[0][0][0].value == -1
    assert parse_atom_formula("++")[0][0][0].value == 2
    assert parse_atom_formula("--")[0][0][0].value == -2
    assert parse_atom_formula("+")[0][0][0].value == 1
    assert parse_atom_formula("-")[0][0][0].value == -1


def test_parse_atom_formula_reads_recursive_literal() -> None:
    formula = parse_atom_formula("C;!$(N(=O)[C;$(C=O)])")
    assert formula[1][0][0] == PropVal("$", "N(=O)[C;$(C=O)]", negated=True)


def test_parse_atom_formula_rejects_unclosed_recursion() -> None:
    with pytest.raises(SmartsError):
        parse_atom_formula("C;$(N")


def test_parse_atom_formula_rejects_repeated_negation_markers() -> None:
    with pytest.raises(SmartsError):
        parse_atom_formula("!!#1")


def test_parse_smarts_atom_returns_formula_and_map() -> None:
    atom = parse_smarts_atom("[CH3:7]")
    assert isinstance(atom, SmartsAtom)
    assert atom.formula[0][0][0] == PropVal("symbol", "C")
    assert atom.formula[0][0][1] == PropVal("H", 3)
    assert atom.atom_map == 7

    compiled = encode_smarts_atom(normalize_smarts_atom(atom))
    assert compiled["Z"] == (1 << ELEMENT_TO_Z["C"])
    assert compiled["H"] == bucket_mask_exact(3)
    assert compiled["AtomMap"] == 7


def test_parse_smarts_atom_accepts_unbracketed_atom_smarts() -> None:
    atom = parse_smarts_atom("c")
    assert atom == SmartsAtom(
        formula=[[[PropVal("symbol", "c")]]], is_aromatic=True, atom_map=None
    )


def test_parse_smarts_atom_accepts_bracketed_aromatic_selenium() -> None:
    atom = parse_smarts_atom("[se]")
    assert atom == SmartsAtom(
        formula=[[[PropVal("symbol", "se")]]],
        is_aromatic=True,
        atom_map=None,
        bracketed=True,
    )


@pytest.mark.parametrize("text", ["[C:-1]"])
def test_parse_smarts_atom_rejects_negative_map(text: str) -> None:
    with pytest.raises(SmartsError):
        parse_smarts_atom(text)


def test_parse_graph_steps_emits_typed_steps() -> None:
    steps = parse_graph_steps("C(=O)N")
    assert isinstance(steps[0], AtomStep)
    assert isinstance(steps[1], LParenStep)
    assert isinstance(steps[2], AtomStep)
    assert isinstance(steps[3], RParenStep)
    assert isinstance(steps[4], AtomStep)
    assert steps[2].leading_bond == bond_flags_from_smarts_token("=")


def test_parse_graph_steps_allows_dot_prefixed_ring_step() -> None:
    steps = parse_graph_steps("C1.C.1")
    assert isinstance(steps[1], RingStep)
    assert isinstance(steps[2], AtomStep)
    assert isinstance(steps[3], RingStep)
    assert steps[2].leading_bond == 0
    assert steps[3].leading_bond == 0

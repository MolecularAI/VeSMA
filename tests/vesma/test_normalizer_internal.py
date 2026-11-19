from __future__ import annotations

import inspect

from vesma.parsing import logic_cnf
from vesma.parsing.builder import build_smarts_mol
from vesma.parsing.grammar_custom import parse_atom_formula, parse_graph_steps
from vesma.parsing.ir import NormalizedSmartsMol, PropVal
from vesma.parsing.logic_cnf import FALSE_CNF, TRUE_CNF
from vesma.parsing.normalizer import normalize_atom_formula, normalize_smarts_mol
from vesma.symbols import ELEMENT_TO_Z


def _normalize(expr: str):
    return normalize_atom_formula(parse_atom_formula(expr))


def test_logic_cnf_module_stays_domain_free() -> None:
    source = inspect.getsource(logic_cnf)
    assert "BUCKET_DEFAULT_MASKS" not in source
    assert ".atom_props" not in source


def test_normalize_atom_formula_canonicalizes_element_and_aromaticity() -> None:
    aliphatic_c = _normalize("C")
    aromatic_c = _normalize("c")
    assert aliphatic_c == frozenset(
        {
            frozenset({PropVal("Z", ELEMENT_TO_Z["C"])}),
            frozenset({PropVal("A", 1)}),
        }
    )
    assert aromatic_c == frozenset(
        {
            frozenset({PropVal("Z", ELEMENT_TO_Z["C"])}),
            frozenset({PropVal("A", 1, negated=True)}),
        }
    )


def test_normalize_atom_formula_canonicalizes_two_letter_aromatic_element() -> None:
    assert _normalize("se") == frozenset(
        {
            frozenset({PropVal("Z", ELEMENT_TO_Z["Se"])}),
            frozenset({PropVal("A", 1, negated=True)}),
        }
    )


def test_normalize_atom_formula_wildcard_is_tautology() -> None:
    assert _normalize("*") == TRUE_CNF


def test_normalize_atom_formula_drops_tautological_aromaticity() -> None:
    assert _normalize("C,c") == frozenset(
        {frozenset({PropVal("Z", ELEMENT_TO_Z["C"])})}
    )


def test_normalize_atom_formula_propagates_units_to_fixed_point() -> None:
    assert _normalize("#6;!#6,#7") == frozenset(
        {
            frozenset({PropVal("Z", 6)}),
            frozenset({PropVal("Z", 7)}),
        }
    )


def test_normalize_atom_formula_detects_complementary_aromaticity_units() -> None:
    assert _normalize("A;a") == FALSE_CNF


def test_normalize_atom_formula_preserves_unbounded_mass_and_unclamped_charge_literals() -> (
    None
):
    assert _normalize("13") == frozenset({frozenset({PropVal("M", 13)})})
    assert _normalize("999") == frozenset({frozenset({PropVal("M", 999)})})
    assert _normalize("+123") == frozenset({frozenset({PropVal("C", 123)})})


def test_normalize_atom_formula_rewrites_bucket_defaults() -> None:
    assert _normalize("R") == frozenset({frozenset({PropVal("R", 0, negated=True)})})
    assert _normalize("!R") == frozenset({frozenset({PropVal("R", 0)})})


def test_normalize_atom_formula_merges_shared_constraints_across_or_branches() -> None:
    cnf = _normalize("CH1,NH1")
    assert cnf == frozenset(
        {
            frozenset({PropVal("Z", 6), PropVal("Z", 7)}),
            frozenset({PropVal("A", 1)}),
            frozenset({PropVal("H", 1)}),
        }
    )
    assert all(len({literal.prop for literal in clause}) == 1 for clause in cnf)


def test_normalize_atom_formula_matches_equivalent_low_and_form() -> None:
    assert _normalize("CH1,NH1") == _normalize("C,N;H1")


def test_normalize_atom_formula_keeps_cross_property_or_when_not_encodable() -> None:
    cnf = _normalize("CH1,NH2")
    assert any(len({literal.prop for literal in clause}) > 1 for clause in cnf)


def test_normalize_smarts_mol_builds_normalized_ir() -> None:
    normalized = normalize_smarts_mol(build_smarts_mol(parse_graph_steps("[CH3:7]O")))
    assert isinstance(normalized, NormalizedSmartsMol)
    assert normalized.atom_count == 2
    assert normalized.atoms[0].atom_map == 7
    assert normalized.bonds.shape == (2, 2)

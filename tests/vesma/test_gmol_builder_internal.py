from __future__ import annotations

import pytest

from vesma.gmol import GMol
from vesma.gmol_atoms import (
    ALIPHATIC,
    AROMATIC,
    bucket_mask_at_least,
    bucket_mask_exact,
)
from vesma.gmol_builder import (
    encode_atom_cnf,
    encode_smarts_atom,
    gmol_from_normalized_smarts,
    gmol_from_smarts,
)
from vesma.parsing.builder import build_smarts_mol
from vesma.parsing.errors import SmartsError
from vesma.parsing.grammar_custom import (
    parse_atom_formula,
    parse_graph_steps,
    parse_smarts_atom,
)
from vesma.parsing.ir import PropVal
from vesma.parsing.normalizer import (
    normalize_atom_formula,
    normalize_smarts_atom,
    normalize_smarts_mol,
)
from vesma.symbols import ELEMENT_TO_Z


def _encode(expr: str, *, messages: list[str] | None = None):
    return encode_atom_cnf(
        normalize_atom_formula(parse_atom_formula(expr)),
        messages=messages,
    )


def test_encode_atom_cnf_element_and_aromaticity_masks() -> None:
    aliphatic_c = _encode("C")
    aromatic_c = _encode("c")
    assert aliphatic_c["Z"] & (1 << ELEMENT_TO_Z["C"])
    assert aromatic_c["Z"] & (1 << ELEMENT_TO_Z["C"])
    assert aliphatic_c["A"] == ALIPHATIC
    assert aromatic_c["A"] == AROMATIC


def test_encode_atom_cnf_or_logic() -> None:
    union_mask = _encode("C,N")
    assert union_mask["Z"] & (1 << ELEMENT_TO_Z["C"])
    assert union_mask["Z"] & (1 << ELEMENT_TO_Z["N"])
    assert union_mask["A"] == ALIPHATIC


def test_encode_atom_cnf_or_with_cross_bucket_dependency_raises() -> None:
    with pytest.raises(SmartsError, match="split it into multiple SMARTS patterns"):
        _encode("C,n")


def test_encode_atom_cnf_negation_flips_atomic_number_bit() -> None:
    mask = _encode("!#1")
    assert not (mask["Z"] & (1 << 1))
    assert mask["Z"] & (1 << ELEMENT_TO_Z["C"])


def test_encode_atom_cnf_rejects_out_of_range_atomic_number() -> None:
    with pytest.raises(SmartsError, match="Atomic number must be in range"):
        encode_atom_cnf(frozenset({frozenset({PropVal("Z", 128)})}))


def test_encode_atom_cnf_default_ring_bucket_is_at_least_one() -> None:
    mask = _encode("R")
    assert mask["R"] == bucket_mask_at_least(1)


def test_encode_atom_cnf_negated_ring_bucket_is_zero() -> None:
    mask = _encode("!R")
    assert mask["R"] == bucket_mask_exact(0)


def test_encode_atom_cnf_charge_is_clamped_by_mask_encoding() -> None:
    plus_two = _encode("++")
    plus_nine = _encode("+9")
    plus_large = _encode("+123")
    minus_nine = _encode("-9")
    minus_large = _encode("-123")
    assert plus_two["C"] == (1 << 5)
    assert plus_nine["C"] == (1 << 7)
    assert plus_large["C"] == (1 << 7)
    assert minus_nine["C"] == (1 << 0)
    assert minus_large["C"] == (1 << 0)


def test_encode_atom_cnf_props_keep_only_touched_buckets() -> None:
    props = _encode("C;H2")
    assert set(props) == {"Z", "A", "H"}


def test_encode_atom_cnf_records_the_asserted_mass() -> None:
    messages: list[str] = []
    props = _encode("C13", messages=messages)
    assert props["Z"] == (1 << ELEMENT_TO_Z["C"])
    assert props["A"] == ALIPHATIC
    assert props["M"] == 13
    assert messages == []


def test_encode_atom_cnf_reports_charge_lumping() -> None:
    messages: list[str] = []
    _encode("+9", messages=messages)
    assert messages == [
        "Charge +9 was lumped into the +4 bucket when converting SMARTS to GMol."
    ]


def test_encode_atom_cnf_reports_bucket_lumping() -> None:
    messages: list[str] = []
    _encode("R8", messages=messages)
    assert messages == [
        "Bucket R=8 was lumped into the 7+ bucket when converting SMARTS to GMol."
    ]


def test_encode_smarts_atom_adds_atom_map() -> None:
    atom = parse_smarts_atom("[CH3:7]")
    encoded = encode_smarts_atom(normalize_smarts_atom(atom))
    assert encoded["Z"] == (1 << ELEMENT_TO_Z["C"])
    assert encoded["H"] == bucket_mask_exact(3)
    assert encoded["AtomMap"] == 7


def test_gmol_from_normalized_smarts_builds_gmol() -> None:
    normalized = normalize_smarts_mol(
        build_smarts_mol(parse_graph_steps("[CH3:7]O")),
        smarts_string="[CH3:7]O",
    )
    gmol, info = gmol_from_normalized_smarts(normalized)
    assert isinstance(gmol, GMol)
    assert gmol.atom_count == 2
    assert int(gmol.atom_map[0]) == 7
    assert info.smarts_string == "[CH3:7]O"
    assert info.messages == []


def test_gmol_from_smarts_returns_gmol_and_info() -> None:
    gmol, info = gmol_from_smarts("[C13]")
    assert isinstance(gmol, GMol)
    assert int(gmol.atom_masks.M[0]) == 13
    assert info.smarts_string == "[C13]"
    assert info.messages == []

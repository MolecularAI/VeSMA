"""Which element a bracket names, when the symbol collides with a SMARTS primitive.

`D H h R r v X x` are query primitives *and* the first letters of elements, and `re`
alternation is ordered, so the tokenizer has to prefer one reading. RDKit is the
comparison instrument throughout: it reads every case below as the element, in both
languages, and these tests pin the same answers.
"""

from __future__ import annotations

import pytest

from vesma import cmol_from_smiles, smiles_from_cmol
from vesma.parsing.grammar_custom import parse_smarts_atom
from vesma.symbols import (
    AROMATIC_UNBRACKET,
    ELEMENT_TO_Z,
    PERIODIC_SYMBOLS,
    Z_TO_ELEMENT,
)


def only_symbol(written: str) -> str | None:
    """The single element symbol a bracket names, or None if it names none."""
    atom = parse_smarts_atom(written)
    symbols = [
        literal.value
        for or_clause in atom.formula
        for and_clause in or_clause
        for literal in and_clause
        if literal.prop == "symbol"
    ]
    return symbols[0] if len(symbols) == 1 else None


# --------------------------------------------------------------------------- #
# The table itself
# --------------------------------------------------------------------------- #


def test_the_periodic_table_is_dense_and_indexed_by_atomic_number() -> None:
    symbols = PERIODIC_SYMBOLS.split()
    assert symbols[0] == "*"
    assert len(symbols) == 119, "wildcard plus elements 1 through 118"
    for z, symbol in [
        (1, "H"),
        (6, "C"),
        (34, "Se"),
        (86, "Rn"),
        (92, "U"),
        (118, "Og"),
    ]:
        assert Z_TO_ELEMENT[z] == symbol
        assert ELEMENT_TO_Z[symbol] == z


@pytest.mark.parametrize("symbol", sorted(set(PERIODIC_SYMBOLS.split()) - {"*"}))
def test_every_element_symbol_parses_as_that_element(symbol: str) -> None:
    """The whole table, not a sample: a symbol that fails to parse becomes another atom."""
    assert only_symbol(f"[{symbol}]") == symbol


# --------------------------------------------------------------------------- #
# The collisions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "written", ["[Hg]", "[He]", "[Hf]", "[Xe]", "[Re]", "[Ru]", "[Dy]", "[Ra]", "[Rf]"]
)
def test_a_two_letter_element_beats_the_primitive_that_shares_its_first_letter(
    written: str,
) -> None:
    """These used to raise: the bucket primitive ate the first letter."""
    assert only_symbol(written) == written[1:-1]


@pytest.mark.parametrize(
    ("written", "was_read_as"),
    [("[Rn]", "N"), ("[Rb]", "B"), ("[Ho]", "O"), ("[Rh]", None)],
)
def test_the_silent_mis_parses_now_name_the_right_element(
    written: str, was_read_as: str | None
) -> None:
    """The dangerous half: these parsed, and named a *different* element or none at all.

    `[Rn]` read as ring-membership plus aromatic nitrogen, so radon built as ammonia.
    """
    element = written[1:-1]
    assert only_symbol(written) == element
    assert ELEMENT_TO_Z[element] != ELEMENT_TO_Z.get(was_read_as, -1)


@pytest.mark.parametrize("written", ["[Hn]", "[Hc]"])
def test_a_pair_that_names_no_element_stays_a_primitive(written: str) -> None:
    """`Hn` and `Hc` are not elements, so these keep the count-plus-aromatic reading."""
    assert only_symbol(written) == written[2]


# --------------------------------------------------------------------------- #
# Hydrogen, which is both an element and a count
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("written", ["[H]", "[2H]", "[H+]"])
def test_a_bracket_whose_only_content_is_h_names_hydrogen(written: str) -> None:
    """Daylight: "Semantics of [H] changed in v4.5". RDKit reads all three as the atom."""
    assert only_symbol(written) == "H"


@pytest.mark.parametrize("written", ["[H1]", "[CH3]", "[nH]", "[NH4+]", "[H,Cl]"])
def test_h_alongside_anything_else_stays_a_hydrogen_count(written: str) -> None:
    """`[H,Cl]` is the one to watch: the alternation makes it an expression, not an atom."""
    assert only_symbol(written) != "H"


@pytest.mark.parametrize("written", ["[H]", "[2H]", "[H+]", "[H][H]"])
def test_hydrogen_as_an_atom_parses_and_then_the_chemistry_declines(
    written: str,
) -> None:
    """The refusal moved from the parser to the model, and is still a refusal.

    Hydrogen occupies no declared configuration, because the model carries it as a count
    on its neighbour rather than as a node. Supporting it is a chemistry change.
    """
    with pytest.raises(Exception, match="no declared configuration"):
        cmol_from_smiles(written)


# --------------------------------------------------------------------------- #
# Counts and spellings that must not have moved
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("written", "hydrogens"), [("[CH3]C", 3), ("[NH4+]", 4), ("[13CH3]C", 3)]
)
def test_a_written_hydrogen_count_is_still_a_count(
    written: str, hydrogens: int
) -> None:
    assert int(cmol_from_smiles(written)[0].H[0]) == hydrogens


def test_the_aromatic_symbol_map_names_real_elements() -> None:
    """`upper()` gave `SE`, which is no element, so the writer never matched selenium."""
    assert all(value in ELEMENT_TO_Z for value in AROMATIC_UNBRACKET.values())
    assert AROMATIC_UNBRACKET["se"] == "Se"
    assert AROMATIC_UNBRACKET["as"] == "As"


def test_aromatic_selenium_is_written_lowercase() -> None:
    """It was emitted as `[Se]` inside a lowercase ring -- an aliphatic atom in an
    aromatic one, which only survived because both readers re-perceive."""
    molecule, _ = cmol_from_smiles("c1cc[se]c1")
    assert smiles_from_cmol(molecule) == "c1cc[se]c1"

from __future__ import annotations

import pytest

from vesma import gmol_from_smarts
from vesma.parsing import SmartsError, parse_smarts, parse_smiles
from vesma.parsing.errors import (
    InconsistentSmartsAtomExpressionError,
    InvalidSmartsSyntaxError,
    UnsupportedSmartsFeatureError,
)


def test_parse_smarts_raises_smarts_error_on_invalid_graph() -> None:
    with pytest.raises(SmartsError):
        parse_smarts("[C")


def test_gmol_from_smarts_raises_smarts_error_on_contradictory_box() -> None:
    with pytest.raises(SmartsError, match="unsatisfiable"):
        gmol_from_smarts("[C&n]")


def test_parse_smarts_uses_invalid_syntax_subtype_for_invalid_graph() -> None:
    with pytest.raises(InvalidSmartsSyntaxError):
        parse_smarts("[C")


def test_parse_smiles_refuses_recursion() -> None:
    with pytest.raises(SmartsError):
        parse_smiles("[$(CC)]")


def test_gmol_from_smarts_uses_expression_subtype_for_contradictory_box() -> None:
    with pytest.raises(InconsistentSmartsAtomExpressionError):
        gmol_from_smarts("[C&n]")


def test_an_expression_past_the_term_limit_is_unsupported() -> None:
    # Thirteen two-box clauses with independent literals, ORed with one more alternative: the
    # OR must expand them into 2**13 conjunctions, past the limit.
    bodies = [f"$([CH{k},NX{k}])" for k in range(7)] + [
        f"$([OD{k},Sv{k}])" for k in range(1, 7)
    ]
    with pytest.raises(UnsupportedSmartsFeatureError, match="alternatives"):
        gmol_from_smarts("[" + "&".join(bodies) + ",F]")

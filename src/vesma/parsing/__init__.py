from __future__ import annotations

import pyparsing as pp

from .builder import build_smarts_mol
from .errors import (
    InvalidSmartsSyntaxError,
    NotMolecularSmilesError,
    SmartsError,
    UnsupportedSmartsFeatureError,
)
from .grammar_custom import parse_graph_steps
from .ir import NormalizedSmartsMol, SmartsMol
from .normalizer import normalize_smarts_mol
from .reading import smiles_reading, validate_smiles_ir

__all__ = [
    "NotMolecularSmilesError",
    "SmartsError",
    "parse_ir",
    "parse_smarts",
    "parse_smiles",
    "smiles_reading",
    "validate_smiles_ir",
]


def parse_ir(text: str, *, language: str) -> SmartsMol:
    """Parse to the raw IR, which records what was written and means nothing yet.

    Both languages share this: SMILES is a syntactic subset of SMARTS, so one grammar
    covers them and the IR is language-neutral. Which language it *is* becomes a question
    only when the IR is read -- see ``vesma.parsing.reading``.
    """
    if not text:
        raise InvalidSmartsSyntaxError(f"Empty {language} string.")
    if ">" in text:
        raise UnsupportedSmartsFeatureError(
            "Reaction SMARTS is not valid atom/query SMARTS. Use parse_reaction_smarts()."
        )
    try:
        return build_smarts_mol(parse_graph_steps(text))
    except pp.ParseBaseException as exc:
        raise InvalidSmartsSyntaxError(
            f"Invalid {language} near column {exc.column}: {exc.line}"
        ) from exc
    except SmartsError as exc:
        raise InvalidSmartsSyntaxError(f"Invalid {language}: {exc}") from exc


def parse_smarts(smarts: str) -> NormalizedSmartsMol:
    """The query reading: an unwritten property is not part of the pattern."""
    text = smarts.strip()
    return normalize_smarts_mol(parse_ir(text, language="SMARTS"), smarts_string=text)


def parse_smiles(smiles: str) -> NormalizedSmartsMol:
    """The molecule reading: an unwritten hydrogen count inside brackets is zero.

    ``smiles_reading`` validates before it rewrites, so a query handed to this function
    is refused rather than quietly read as a molecule.
    """
    text = smiles.strip()
    return normalize_smarts_mol(
        smiles_reading(parse_ir(text, language="SMILES")),
        smarts_string=text,
    )

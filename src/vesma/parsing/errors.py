from __future__ import annotations


class SmartsError(ValueError):
    pass


class InvalidSmartsSyntaxError(SmartsError):
    pass


class UnsupportedSmartsFeatureError(SmartsError):
    pass


class InconsistentSmartsAtomExpressionError(SmartsError):
    pass


class NotMolecularSmilesError(SmartsError):
    """The text parses as SMARTS but does not denote a molecule.

    Raised only on the SMILES reading, so a query keeps its own, wider vocabulary.
    """


__all__ = ["SmartsError"]

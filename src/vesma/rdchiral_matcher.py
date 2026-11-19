"""Reaction matcher over RDChiral, used as a comparison backend.

RDChiral applies retro templates through RDKit's ``RunReactants`` and then post-processes
each outcome: components carrying duplicated atom maps are stitched back into one
molecule, chirality is restored, the result is canonicalized to SMILES, and *distinct*
outcomes are collected in a set. Two consequences shape this module:

- the returned unit is a distinct outcome, not one atom assignment (see ``app_idx`` on
  `RdchiralReactionMatcher.match`), and
- precursors come back as SMILES strings, RDChiral's native output type.

Compare against `vesma.rdkit_matcher.RdkitReactionMatcher`, which applies the same
templates through bare RDKit and keeps one outcome per assignment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .matcher import Hit, HitArray, Pair, hit_array

__all__ = [
    "RdchiralDiagnostics",
    "RdchiralReactionMatcher",
    "rdchiral_reactants_from_smiles",
    "rdchiral_reactions_from_smarts",
    "rdchiral_version",
]


@dataclass(frozen=True)
class RdchiralDiagnostics:
    """Non-timing properties of one `RdchiralReactionMatcher.match` call.

    ``outcome_semantics`` and ``precursor_representation`` are recorded rather than
    implied because they are exactly where this backend departs from the others: its
    precursor count is outcome-deduplicated, not a raw application multiset.
    """

    rules_total: int
    pairs_evaluated: int
    pairs_failed: int
    # Which pairs failed, not just how many. A pair RDChiral could not evaluate produced
    # no outcomes for a reason that is not chemistry, so an audit has to keep it in the
    # denominator under its own status rather than score it as "RDChiral found nothing".
    failed_pairs: tuple[Pair, ...]
    outcomes: int
    combine_enantiomers: bool
    keep_mapnums: bool
    outcome_semantics: str
    precursor_representation: str


class RdchiralReactionMatcher:
    """Reaction matcher over `rdchiralReactants` molecules (M = rdchiralReactants).

    Precursors are component SMILES strings; pair with `vesma.keyer.rdkit_token_from_smiles`
    (K=str), which puts them in the same key space as `rdkit_smiles_keyer` (RDKit mols) and
    `cmol_rdkit_smiles_keyer` (vesma CMols).

    ``app_idx`` is NOT comparable to the RDKit, CPU and GPU backends'. There, one app_idx is
    one atom assignment. RDChiral collects outcomes in a ``set``, so identical outcomes from
    different assignments collapse before they are returned; here one app_idx is one
    *distinct* outcome and the count is a lower bound on the number of applications. The
    deduplication cannot be disabled. Component multiplicity *within* one outcome survives.

    Serial only, with no ``n_jobs``: `rdchiralReaction` and `rdchiralReactants` hold live
    RDKit atom references and do not pickle, and `rdchiralRun` mutates the template's atom
    map numbers in place (relying on its own ``reset()``), so it is not thread-safe either.
    The comparison target, `RdkitReactionMatcher` at ``n_jobs=1``, is serial regardless.
    """

    def __init__(
        self,
        reactions: Sequence[Any],
        *,
        combine_enantiomers: bool = True,
    ) -> None:
        self.reactions = tuple(reactions)
        self.combine_enantiomers = combine_enantiomers
        self.last_diagnostics: RdchiralDiagnostics | None = None

    def match(
        self,
        mols: Sequence[Any],
        pairs: Sequence[Pair] | None = None,
    ) -> tuple[np.ndarray, HitArray]:
        from rdchiral.main import rdchiralRun

        if pairs is None:
            pairs = [
                (mol_idx, rule_id)
                for mol_idx in range(len(mols))
                for rule_id in range(len(self.reactions))
            ]

        precursors: list[str] = []
        hits: list[Hit] = []
        outcomes_seen = 0
        failed_pairs: list[Pair] = []

        for mol_idx, rule_id in pairs:
            try:
                outcomes = rdchiralRun(
                    self.reactions[rule_id],
                    mols[mol_idx],
                    keep_mapnums=False,
                    combine_enantiomers=self.combine_enantiomers,
                )
            except Exception:
                # RDChiral 1.1.0 predates the current RDKit stereochemistry API, and its
                # own cis/trans restoration can trip an RDKit pre-condition ("stereo atoms
                # should be specified before specifying CIS/TRANS bond stereochemistry").
                # The failure is per pair and upstream, so the pair is dropped and counted
                # rather than losing a whole sweep -- the same treatment
                # `rdkit_matcher.split_outcome` gives an outcome RDKit cannot sanitize.
                failed_pairs.append((mol_idx, rule_id))
                continue
            # rdchiralRun returns a list built from a set, so its order varies between
            # runs and hash seeds. Sorting makes app_idx reproducible, the same reason
            # RdkitFilterCatalogReactionMatcher sorts its candidate pairs.
            for app_idx, outcome in enumerate(sorted(outcomes)):
                outcomes_seen += 1
                for component in outcome.split("."):
                    hits.append((mol_idx, rule_id, app_idx, len(precursors)))
                    precursors.append(component)

        self.last_diagnostics = RdchiralDiagnostics(
            rules_total=len(self.reactions),
            pairs_evaluated=len(pairs),
            pairs_failed=len(failed_pairs),
            failed_pairs=tuple(failed_pairs),
            outcomes=outcomes_seen,
            combine_enantiomers=self.combine_enantiomers,
            keep_mapnums=False,
            outcome_semantics="distinct-deduplicated",
            precursor_representation="smiles",
        )
        return np.array(precursors, dtype=object), hit_array(hits)


def rdchiral_reactions_from_smarts(smarts: Sequence[str]) -> tuple[Any, ...]:
    """Compile retro reaction SMARTS into the reusable `rdchiralReaction` objects.

    `rdchiralReaction` runs RDKit's reaction validator and raises on a template it
    rejects; that failure escalates, as it does in `rdkit_reactions_from_smarts`. All
    42,554 templates of the benchmark's ``uspto42k-compat`` collection compile.

    Compiling that collection costs roughly three minutes, so callers that sweep several
    workload sizes should compile once and reuse. Reuse is safe: `rdchiralRun` resets the
    template's atom map numbers before each application.
    """
    from rdchiral.initialization import rdchiralReaction

    return tuple(rdchiralReaction(text) for text in smarts)


def rdchiral_reactants_from_smiles(smiles: Sequence[str]) -> tuple[Any, ...]:
    """Build the reusable `rdchiralReactants` objects, one per molecule.

    `rdchiralReactants` parses each SMILES into a chiral and an achiral copy and indexes
    them by atom map number. It is RDChiral's analogue of `Chem.MolFromSmiles`, which the
    other backends also receive prebuilt, so benchmarks should build these outside the
    timed region and report the cost separately. One object is reusable across templates.
    """
    from rdchiral.initialization import rdchiralReactants

    return tuple(rdchiralReactants(text) for text in smiles)


def rdchiral_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("rdchiral")
    except PackageNotFoundError:
        return "unknown"

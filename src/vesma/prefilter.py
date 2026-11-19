from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, cast

import numpy as np

from .cmol import CMol
from .gmol import GMol
from .matcher import HitArray, Pair, R, ReactionMatcher
from .pathfp import (
    SparseFingerprint,
    SparseFingerprintConfig,
    sparse_path_fingerprint,
    sparse_path_fingerprint_from_cmol,
    sparse_path_fingerprints_from_cuda_batch,
)
from .reaction import ReactionPattern

FeatureProvider = Callable[[Any], list[SparseFingerprint]]


@dataclass(frozen=True)
class PrefilterDiagnostics:
    dense_pair_count: int
    candidate_pair_count: int
    pruned_pair_count: int
    prune_ratio: float
    feature_seconds: float
    pair_seconds: float
    inner_match_seconds: float


@dataclass(frozen=True)
class SparseRuleFingerprintIndex:
    rule_ids: tuple[R, ...]
    rule_features: Mapping[R, SparseFingerprint]
    rule_atom_counts: Mapping[R, int]
    inverted: Mapping[int, tuple[R, ...]]
    featureful_rule_ids: frozenset[R]
    featureless_rule_ids: tuple[R, ...]

    def candidate_pairs(
        self,
        molecule_features: Sequence[SparseFingerprint],
        molecule_atom_counts: Sequence[int],
        *,
        pairs: Sequence[Pair] | None = None,
    ) -> list[Pair]:
        if pairs is not None:
            return [
                (mol_idx, rule_id)
                for mol_idx, rule_id in pairs
                if self.pair_survives(
                    molecule_features[mol_idx],
                    molecule_atom_counts[mol_idx],
                    rule_id,
                )
            ]

        result: list[Pair] = []
        for mol_idx, fingerprint in enumerate(molecule_features):
            atom_count = molecule_atom_counts[mol_idx]
            result.extend(
                (mol_idx, rule_id)
                for rule_id in self.candidate_rules(fingerprint, atom_count)
            )
        return result

    def candidate_rules(
        self,
        molecule_features: SparseFingerprint,
        molecule_atom_count: int,
    ) -> list[R]:
        matched_feature_counts: Counter[R] = Counter()
        for feature, molecule_count in molecule_features.items():
            for rule_id in self.inverted.get(feature, ()):
                if molecule_count >= self.rule_features[rule_id][feature]:
                    matched_feature_counts[rule_id] += 1

        candidates = [
            rule_id
            for rule_id in self.rule_ids
            if self.rule_atom_counts[rule_id] <= molecule_atom_count
            and (
                rule_id in self.featureless_rule_ids
                or matched_feature_counts[rule_id] == len(self.rule_features[rule_id])
            )
        ]
        return candidates

    def pair_survives(
        self,
        molecule_features: SparseFingerprint,
        molecule_atom_count: int,
        rule_id: R,
    ) -> bool:
        if self.rule_atom_counts[rule_id] > molecule_atom_count:
            return False
        rule_features = self.rule_features[rule_id]
        return all(
            molecule_features.get(feature, 0) >= count
            for feature, count in rule_features.items()
        )


def sparse_rule_fingerprint_index(
    rules: Sequence[ReactionPattern],
    *,
    config: SparseFingerprintConfig = SparseFingerprintConfig(),
) -> SparseRuleFingerprintIndex:
    rule_items = tuple(enumerate(rules))

    features = {
        rule_id: sparse_path_fingerprint(
            pattern.lhs,
            query=True,
            config=config,
        )
        for rule_id, pattern in rule_items
    }
    atom_counts = {rule_id: pattern.lhs.atom_count for rule_id, pattern in rule_items}
    inverted_lists: dict[int, list[R]] = {}
    for rule_id, fingerprint in features.items():
        for feature in fingerprint:
            inverted_lists.setdefault(feature, []).append(rule_id)

    featureful = frozenset(
        rule_id for rule_id, fingerprint in features.items() if fingerprint
    )
    return SparseRuleFingerprintIndex(
        rule_ids=tuple(rule_id for rule_id, _ in rule_items),
        rule_features=features,
        rule_atom_counts=atom_counts,
        inverted={
            feature: tuple(rule_ids) for feature, rule_ids in inverted_lists.items()
        },
        featureful_rule_ids=featureful,
        featureless_rule_ids=tuple(
            rule_id for rule_id, fingerprint in features.items() if not fingerprint
        ),
    )


class PrefilteredReactionMatcher:
    """ReactionMatcher wrapper that precomputes sparse candidate pair indices."""

    def __init__(
        self,
        matcher: ReactionMatcher[Any],
        rules: Sequence[ReactionPattern],
        *,
        feature_provider: FeatureProvider,
        config: SparseFingerprintConfig = SparseFingerprintConfig(),
    ) -> None:
        self.matcher = matcher
        self.feature_provider = feature_provider
        self.index = sparse_rule_fingerprint_index(rules, config=config)
        self.last_diagnostics: PrefilterDiagnostics | None = None

    def match(
        self,
        mols: Any,
        pairs: Sequence[Pair] | None = None,
    ) -> tuple[Any, HitArray]:
        feature_started = perf_counter()
        molecule_features = self.feature_provider(mols)
        feature_seconds = perf_counter() - feature_started

        molecule_atom_counts = atom_counts(mols)
        dense_pair_count = (
            len(pairs)
            if pairs is not None
            else len(molecule_features) * len(self.index.rule_ids)
        )

        pair_started = perf_counter()
        candidate_pairs = self.index.candidate_pairs(
            molecule_features,
            molecule_atom_counts,
            pairs=pairs,
        )
        pair_seconds = perf_counter() - pair_started

        inner_started = perf_counter()
        precursors, hits = self.matcher.match(mols, candidate_pairs)
        inner_match_seconds = perf_counter() - inner_started
        candidate_pair_count = len(candidate_pairs)
        pruned_pair_count = dense_pair_count - candidate_pair_count
        self.last_diagnostics = PrefilterDiagnostics(
            dense_pair_count=dense_pair_count,
            candidate_pair_count=candidate_pair_count,
            pruned_pair_count=pruned_pair_count,
            prune_ratio=(
                0.0 if dense_pair_count == 0 else pruned_pair_count / dense_pair_count
            ),
            feature_seconds=feature_seconds,
            pair_seconds=pair_seconds,
            inner_match_seconds=inner_match_seconds,
        )
        return precursors, hits


def cpu_sparse_path_feature_provider(
    mols: Sequence[CMol] | Sequence[GMol],
    *,
    config: SparseFingerprintConfig = SparseFingerprintConfig(),
) -> list[SparseFingerprint]:
    return [
        (
            sparse_path_fingerprint_from_cmol(mol, config=config)
            if isinstance(mol, CMol)
            else sparse_path_fingerprint(cast(GMol, mol), query=False, config=config)
        )
        for mol in mols
    ]


def cuda_host_sparse_path_feature_provider(
    mols: Any,
    *,
    config: SparseFingerprintConfig = SparseFingerprintConfig(),
) -> list[SparseFingerprint]:
    if hasattr(mols, "atom_counts_device"):
        return sparse_path_fingerprints_from_cuda_batch(mols, config=config)
    return cpu_sparse_path_feature_provider(cast(Sequence[CMol], mols), config=config)


def atom_counts(mols: Any) -> list[int]:
    if hasattr(mols, "atom_counts_device"):
        values = mols.atom_counts_device
        if hasattr(values, "get"):
            values = values.get()
        return [int(value) for value in np.asarray(values)]
    return [int(mol.atom_count) for mol in mols]


__all__ = [
    "FeatureProvider",
    "PrefilterDiagnostics",
    "PrefilteredReactionMatcher",
    "SparseRuleFingerprintIndex",
    "cpu_sparse_path_feature_provider",
    "cuda_host_sparse_path_feature_provider",
    "sparse_rule_fingerprint_index",
]

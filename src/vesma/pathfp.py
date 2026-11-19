"""Path-based molecule fingerprints (linear-path enumeration).

Two flavors share one path-enumeration core:

- a *sparse* counted fingerprint (``Counter[int]``) of raw hashed atom/path
  features for the subgraph-monotone reaction prefilter, and
- a *folded* fixed-width bit fingerprint (RDKit-style) for a cheap bitwise
  query-subset prefilter at a chosen bit budget.

Full Morgan/WL neighborhood labels are not substructure-monotone, so this emits
raw hashed atom/path features up to a path length instead: every emitted query
feature is necessary for a subgraph match, so ``query_features <=
molecule_features`` (and, folded, ``query_bits & target_bits == query_bits``) is
a safe prefilter. The shared vocabulary (hash mixers, exact-Z labels, atom/path
feature hashers, bond classing) lives in ``invariants.py``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from .adjacency import bond_presence
from .cmol import CMol
from .cmol_builder import gmol_from_cmol
from .gmol import GMol
from .invariants import (
    atom_feature,
    bond_feature_classes,
    exact_atomic_number_labels,
    exact_atomic_number_labels_from_packed_atoms,
    path_hash,
)

SparseFingerprint = Counter[int]
FoldedFingerprint = np.ndarray[tuple[int], np.dtype[np.uint8]]
FoldedFingerprintWords = np.ndarray[tuple[int], np.dtype[np.uint64]]

FOLDED_DEFAULT_BITS = 2048
FOLDED_DEFAULT_PATH_LENGTH = 7


@dataclass(frozen=True)
class SparseFingerprintConfig:
    radius: int = 2

    def __post_init__(self) -> None:
        if self.radius < 0:
            raise ValueError("radius must be non-negative.")


@dataclass(frozen=True)
class FoldedFingerprintConfig:
    n_bits: int = FOLDED_DEFAULT_BITS
    max_length: int = FOLDED_DEFAULT_PATH_LENGTH

    def __post_init__(self) -> None:
        if self.n_bits <= 0:
            raise ValueError("n_bits must be positive.")
        if self.max_length < 0:
            raise ValueError("max_length must be non-negative.")

    @property
    def sparse_config(self) -> SparseFingerprintConfig:
        return SparseFingerprintConfig(
            radius=self.max_length,
        )


def sparse_path_fingerprint(
    graph: GMol,
    *,
    query: bool,
    config: SparseFingerprintConfig = SparseFingerprintConfig(),
) -> SparseFingerprint:
    """Raw sparse path features for substructure prefiltering."""

    labels = exact_atomic_number_labels(graph.atom_masks)
    features: SparseFingerprint = Counter()
    for label in labels:
        if label is not None:
            features[atom_feature(label)] += 1

    adjacency = bond_presence(graph.bonds)
    for length in range(1, config.radius + 1):
        for path in simple_paths(adjacency, length):
            path_features = path_feature_variants(
                graph.bonds,
                labels,
                path,
                query=query,
            )
            features.update(path_features)
    return features


def sparse_path_fingerprint_from_cmol(
    molecule: CMol,
    *,
    config: SparseFingerprintConfig = SparseFingerprintConfig(),
) -> SparseFingerprint:
    return sparse_path_fingerprint(
        gmol_from_cmol(molecule),
        query=False,
        config=config,
    )


def sparse_path_fingerprints_from_cuda_batch(
    batch: Any,
    *,
    config: SparseFingerprintConfig = SparseFingerprintConfig(),
) -> list[SparseFingerprint]:
    """Host-side sparse features from a ``CudaMoleculeBatch``.

    The current CUDA WL kernel returns final identity tokens, not raw path
    features. This provider keeps the wrapper usable for CUDA now by copying the
    packed batch arrays to host and applying the same sparse feature semantics.
    """

    atom_counts = host_array(batch.atom_counts_device).astype(np.int32, copy=False)
    atoms = host_array(batch.atoms_device).astype(np.uint32, copy=False)
    bonds = host_array(batch.bonds_device).astype(np.uint8, copy=False)
    result: list[SparseFingerprint] = []
    for index, atom_count in enumerate(atom_counts):
        n = int(atom_count)
        labels = exact_atomic_number_labels_from_packed_atoms(atoms[index, :n])
        molecule_bonds = bonds[index, :n, :n]
        result.append(
            sparse_fingerprint_from_labels_and_bonds(
                labels,
                molecule_bonds,
                query=False,
                radius=config.radius,
            )
        )
    return result


def sparse_fingerprint_from_labels_and_bonds(
    labels: list[int | None],
    bonds: np.ndarray,
    *,
    query: bool,
    radius: int,
) -> SparseFingerprint:
    features: SparseFingerprint = Counter()
    for label in labels:
        if label is not None:
            features[atom_feature(label)] += 1
    adjacency = bond_presence(bonds)
    for length in range(1, radius + 1):
        for path in simple_paths(adjacency, length):
            features.update(
                path_feature_variants(
                    bonds,
                    labels,
                    path,
                    query=query,
                )
            )
    return features


# --------------------------------------------------------------------------- #
# Folded (fixed-width bit) fingerprint: the sparse features modulo-folded into an
# ``n_bits`` presence vector for a cheap bitwise query-subset prefilter.
# --------------------------------------------------------------------------- #


def folded_path_fingerprint(
    graph: GMol,
    *,
    query: bool,
    config: FoldedFingerprintConfig = FoldedFingerprintConfig(),
) -> FoldedFingerprint:
    sparse = sparse_path_fingerprint(graph, query=query, config=config.sparse_config)
    return fold_features(sparse, config.n_bits)


def folded_path_fingerprint_words(
    graph: GMol,
    *,
    query: bool,
    config: FoldedFingerprintConfig = FoldedFingerprintConfig(),
) -> FoldedFingerprintWords:
    sparse = sparse_path_fingerprint(graph, query=query, config=config.sparse_config)
    return fold_features_words(sparse, config.n_bits)


def folded_path_fingerprint_from_cmol(
    molecule: CMol,
    *,
    config: FoldedFingerprintConfig = FoldedFingerprintConfig(),
) -> FoldedFingerprint:
    return folded_path_fingerprint(
        gmol_from_cmol(molecule),
        query=False,
        config=config,
    )


def folded_path_fingerprint_words_from_cmol(
    molecule: CMol,
    *,
    config: FoldedFingerprintConfig = FoldedFingerprintConfig(),
) -> FoldedFingerprintWords:
    return folded_path_fingerprint_words(
        gmol_from_cmol(molecule),
        query=False,
        config=config,
    )


def fold_features(features: SparseFingerprint, n_bits: int) -> FoldedFingerprint:
    """Fold sparse feature ids into an ``n_bits`` presence vector. Counts collapse
    to presence; the query-subset test stays sound because every necessary query
    feature still maps to a set bit that the target must also set."""
    bits = np.zeros(n_bits, dtype=np.uint8)
    for feature in features:
        bits[feature % n_bits] = np.uint8(1)
    return bits


def fold_features_words(
    features: SparseFingerprint,
    n_bits: int,
) -> FoldedFingerprintWords:
    if n_bits % 64 != 0:
        raise ValueError("n_bits must be a multiple of 64 for word fingerprints.")
    words = np.zeros(n_bits // 64, dtype=np.uint64)
    one = np.uint64(1)
    for feature in features:
        bit = feature % n_bits
        words[bit // 64] |= one << np.uint64(bit % 64)
    return words


def folded_query_subset(
    query_bits: FoldedFingerprint,
    target_bits: FoldedFingerprint,
) -> bool:
    """True when every query bit is also set in the target (necessary condition
    for a substructure match)."""
    return bool(np.array_equal(query_bits & target_bits, query_bits))


def folded_query_subset_words(
    query_words: FoldedFingerprintWords,
    target_words: FoldedFingerprintWords,
) -> bool:
    if query_words.shape != target_words.shape:
        raise ValueError("query_words and target_words must have the same shape.")
    return bool(np.array_equal(query_words & target_words, query_words))


# --------------------------------------------------------------------------- #
# Shared path enumeration.
# --------------------------------------------------------------------------- #


def simple_paths(adjacency: np.ndarray, length: int) -> list[tuple[int, ...]]:
    paths: list[tuple[int, ...]] = []
    for root in range(adjacency.shape[0]):
        extend_paths(adjacency, (root,), length, paths)
    unique = {canonical_path(path) for path in paths}
    return sorted(unique)


def extend_paths(
    adjacency: np.ndarray,
    prefix: tuple[int, ...],
    target_length: int,
    paths: list[tuple[int, ...]],
) -> None:
    if len(prefix) == target_length + 1:
        paths.append(prefix)
        return
    current = prefix[-1]
    used = set(prefix)
    for neighbor in np.flatnonzero(adjacency[current]):
        next_atom = int(neighbor)
        if next_atom in used:
            continue
        extend_paths(adjacency, (*prefix, next_atom), target_length, paths)


def canonical_path(path: tuple[int, ...]) -> tuple[int, ...]:
    reverse = tuple(reversed(path))
    return min(path, reverse)


def path_feature_variants(
    bonds: np.ndarray,
    labels: list[int | None],
    path: tuple[int, ...],
    *,
    query: bool,
) -> SparseFingerprint:
    atom_labels = [labels[index] for index in path]
    if any(label is None for label in atom_labels):
        return Counter()
    edge_options = [
        bond_feature_classes(
            int(bonds[left, right]),
            query=query,
        )
        for left, right in zip(path, path[1:])
    ]
    if any(not options for options in edge_options):
        return Counter()

    features: SparseFingerprint = Counter()
    for classes in product(*edge_options):
        forward = path_hash(atom_labels, classes)
        reverse = path_hash(list(reversed(atom_labels)), tuple(reversed(classes)))
        features[int(min(forward, reverse))] += 1
    return features


def host_array(array: Any) -> np.ndarray:
    if hasattr(array, "get"):
        return array.get()
    return np.asarray(array)


__all__ = [
    "FoldedFingerprint",
    "FoldedFingerprintConfig",
    "FoldedFingerprintWords",
    "SparseFingerprint",
    "SparseFingerprintConfig",
    "fold_features",
    "fold_features_words",
    "folded_path_fingerprint",
    "folded_path_fingerprint_from_cmol",
    "folded_path_fingerprint_words",
    "folded_path_fingerprint_words_from_cmol",
    "folded_query_subset",
    "folded_query_subset_words",
    "sparse_path_fingerprint",
    "sparse_path_fingerprint_from_cmol",
    "sparse_path_fingerprints_from_cuda_batch",
]

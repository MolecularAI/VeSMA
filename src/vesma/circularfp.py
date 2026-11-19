"""Small WL-like circular fingerprints for concrete molecular graphs.

This is intentionally an ECFP4 analogue rather than an RDKit-compatible Morgan
implementation.  Atom environments at radii zero through two are refined with the
same 64-bit mixers used by VeSMA's WL identity and folded into a binary bit vector.
The CPU implementation is the readable specification for the CUDA kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gmol import GMol
from .invariants import combine_uint64, mix_uint64

__all__ = [
    "CircularFingerprintConfig",
    "circular_fingerprint_words",
]

CIRCULAR_ATOM_FEATURE = np.uint64(0xA70F_4A7D_9E37_0003)
BOND_KIND_MASK = np.uint8(0x0F)


@dataclass(frozen=True)
class CircularFingerprintConfig:
    """Shape of the binary circular fingerprint."""

    n_bits: int = 2048
    radius: int = 2

    def __post_init__(self) -> None:
        if self.n_bits <= 0 or self.n_bits % 64:
            raise ValueError("n_bits must be a positive multiple of 64.")
        if self.radius < 0:
            raise ValueError("radius must be non-negative.")


def atom_labels(graph: GMol) -> np.ndarray:
    """Initial labels from concrete atom state available in packed CUDA batches."""
    masks = graph.atom_masks
    degree = np.count_nonzero(graph.bonds, axis=1).astype(np.uint64, copy=False)
    with np.errstate(over="ignore"):
        labels = np.full(graph.atom_count, CIRCULAR_ATOM_FEATURE, dtype=np.uint64)
        for values in (
            masks.Z_lo,
            masks.Z_hi,
            masks.A,
            masks.C,
            masks.H,
            degree,
        ):
            labels = combine_uint64(labels, values.astype(np.uint64, copy=False))
    return labels


def refined_labels(
    labels: np.ndarray,
    graph: GMol,
) -> np.ndarray:
    """One permutation-invariant WL refinement round."""
    adjacency = graph.bonds != 0
    degree = adjacency.sum(axis=1, dtype=np.uint64)
    bond_classes = (graph.bonds & BOND_KIND_MASK).astype(np.uint64, copy=False)
    with np.errstate(over="ignore"):
        messages = combine_uint64(
            np.broadcast_to(labels[None, :], adjacency.shape),
            bond_classes,
        )
        aggregate = np.where(adjacency, messages, np.uint64(0)).sum(
            axis=1,
            dtype=np.uint64,
        )
        return mix_uint64(combine_uint64(combine_uint64(labels, degree), aggregate))


def set_label_bits(words: np.ndarray, labels: np.ndarray, n_bits: int) -> None:
    bits = labels % np.uint64(n_bits)
    for bit in bits.tolist():
        words[int(bit) // 64] |= np.uint64(1) << np.uint64(int(bit) % 64)


def circular_fingerprint_words(
    graph: GMol,
    config: CircularFingerprintConfig = CircularFingerprintConfig(),
) -> np.ndarray:
    """Packed binary circular fingerprint, one uint64 per 64 bits."""
    words = np.zeros(config.n_bits // 64, dtype=np.uint64)
    labels = atom_labels(graph)
    set_label_bits(words, labels, config.n_bits)
    for _ in range(config.radius):
        labels = refined_labels(labels, graph)
        set_label_bits(words, labels, config.n_bits)
    return words

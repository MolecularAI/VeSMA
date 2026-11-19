"""Host circular fingerprints over packed molecule batches, and exact nearest-stock distance.

The numpy twin of :mod:`vesma.cuda_circularfp`. It reads the same packed layout the Metal and
CUDA batches carry — ``(N, 64 * 14)`` ``uint32`` atom mask words and ``(N, 64 * 64)`` ``uint8``
bond flags — so a backend whose precursors come back as a packed *host* batch can be scored
without a device kernel and without rebuilding one molecule at a time.

:func:`vesma.circularfp.circular_fingerprint_words` remains the readable specification; this
module is its batched form and must agree with it bit for bit, which
``tests/vesma/test_packed_circularfp.py`` pins.

Cost, so nobody is surprised by it: :func:`nearest_stock_distance` is exact against the whole
catalogue and its work is ``queries * stock * n_bits / 64`` popcounts. Measured on an Apple M5
against a 988k-molecule stock at 2048 bits that is **~30 ms per query**, which suits a catalogue
of tens of thousands and not one of a million — for that, score on the GPU. A batch does not
amortise: the work is one popcount per (query, stock) pair however the pairs are grouped, so a
frontier of 20,000 molecules against a million-row catalogue is ten CPU-minutes and no arrangement
of the loops changes that.
"""

from __future__ import annotations

import numpy as np

from .circularfp import BOND_KIND_MASK, CIRCULAR_ATOM_FEATURE, CircularFingerprintConfig
from .invariants import combine_uint64, mix_uint64
from .limits import ATOM_MASK_WORD_COUNT, GPU_MATCH_ATOM_CAP

__all__ = [
    "NEAREST_BUDGET_BYTES",
    "PACKED_ATOM_WORD_COUNT",
    "PACKED_BUCKET",
    "circular_fingerprints_packed",
    "nearest_stock_distance",
    "packed_atom_labels",
    "refined_packed_labels",
]

#: This module's names for the two packed-layout dimensions; see `vesma.limits`.
PACKED_BUCKET = GPU_MATCH_ATOM_CAP
PACKED_ATOM_WORD_COUNT = ATOM_MASK_WORD_COUNT

# 64 MB for :func:`nearest_stock_distance`'s working set. Throughput is flat from 4 MB to 256 MB
# on an M5 -- the function is bound by the popcount, not by how much of the stock is in flight --
# so the default buys residency rather than speed.
NEAREST_BUDGET_BYTES = 64 << 20

# Which of the 14 packed mask words the fingerprint reads, in the order the CPU reference
# combines them: Z_lo and Z_hi arrive as two uint32 halves each, then A, C and H. Word 6 is `D`
# and is skipped, because the reference takes degree from the bond matrix instead — the two are
# the same number for a concrete molecule, and the bond matrix is the one the refinement rounds
# already walk.
PACKED_Z_LO_WORDS = (0, 1)
PACKED_Z_HI_WORDS = (2, 3)
PACKED_SCALAR_WORDS = (4, 5, 7)


def joined_halves(packed: np.ndarray, words: tuple[int, int]) -> np.ndarray:
    """One ``uint64`` mask from the two ``uint32`` halves it was packed into."""
    low, high = words
    return packed[..., low].astype(np.uint64) | (
        packed[..., high].astype(np.uint64) << np.uint64(32)
    )


def packed_atom_labels(packed: np.ndarray, degree: np.ndarray) -> np.ndarray:
    """Radius-zero labels for a packed batch: ``(N, 64)`` ``uint64``.

    ``degree`` is passed in rather than derived here because the refinement rounds need the same
    array, and recomputing a popcount per round is the one avoidable cost in this function.
    """
    with np.errstate(over="ignore"):
        labels = np.full(degree.shape, CIRCULAR_ATOM_FEATURE, dtype=np.uint64)
        for values in (
            joined_halves(packed, PACKED_Z_LO_WORDS),
            joined_halves(packed, PACKED_Z_HI_WORDS),
            *(packed[..., word].astype(np.uint64) for word in PACKED_SCALAR_WORDS),
            degree,
        ):
            labels = combine_uint64(labels, values)
    return labels


def refined_packed_labels(
    labels: np.ndarray,
    adjacency: np.ndarray,
    bond_classes: np.ndarray,
    degree: np.ndarray,
) -> np.ndarray:
    """One permutation-invariant WL round over the whole batch at once.

    ``messages[m, i, j]`` is atom ``j``'s label combined with the ``i``--``j`` bond class, which is
    the neighbour's contribution to atom ``i``; summing the masked rows is the invariant aggregate.
    """
    with np.errstate(over="ignore"):
        messages = combine_uint64(
            np.broadcast_to(labels[:, None, :], adjacency.shape),
            bond_classes,
        )
        aggregate = np.where(adjacency, messages, np.uint64(0)).sum(
            axis=2,
            dtype=np.uint64,
        )
        return mix_uint64(combine_uint64(combine_uint64(labels, degree), aggregate))


def set_packed_label_bits(
    words: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    n_bits: int,
) -> None:
    """Fold every real atom's label into its molecule's bit vector, in place.

    ``valid`` is what keeps the padding out: a batch is a fixed 64 slots wide and the tail slots
    hold zeros, whose label is a perfectly ordinary hash that would otherwise set a bit in every
    short molecule and make them all look alike.
    """
    molecule, _ = np.nonzero(valid)
    bits = labels[valid] % np.uint64(n_bits)
    width = words.shape[1]
    target = molecule.astype(np.intp) * width + (bits >> np.uint64(6)).astype(np.intp)
    np.bitwise_or.at(
        words.reshape(-1),
        target,
        np.uint64(1) << (bits & np.uint64(63)),
    )


def circular_fingerprints_packed(
    atoms: np.ndarray,
    bonds: np.ndarray,
    atom_counts: np.ndarray,
    config: CircularFingerprintConfig = CircularFingerprintConfig(),
) -> np.ndarray:
    """Packed binary circular fingerprints for a packed host batch: ``(N, n_bits // 64)``.

    Takes the three arrays rather than a batch object so it serves every packed backend — the
    Metal and CUDA host batches name their fields differently and neither should have to be
    imported here.
    """
    count = int(np.asarray(atom_counts).shape[0])
    words = np.zeros((count, config.n_bits // 64), dtype=np.uint64)
    if not count:
        return words

    packed = atoms.reshape(count, PACKED_BUCKET, PACKED_ATOM_WORD_COUNT)
    matrix = bonds.reshape(count, PACKED_BUCKET, PACKED_BUCKET)
    valid = np.arange(PACKED_BUCKET)[None, :] < np.asarray(atom_counts).reshape(-1, 1)
    # Padding slots are zero in `bonds` already; masking both axes says so explicitly and keeps
    # this correct for a backend that leaves its tail dirty.
    adjacency = (matrix != 0) & valid[:, None, :] & valid[:, :, None]
    degree = adjacency.sum(axis=2, dtype=np.uint64)

    labels = packed_atom_labels(packed, degree)
    set_packed_label_bits(words, labels, valid, config.n_bits)
    bond_classes = (matrix & BOND_KIND_MASK).astype(np.uint64)
    for _ in range(config.radius):
        labels = refined_packed_labels(labels, adjacency, bond_classes, degree)
        set_packed_label_bits(words, labels, valid, config.n_bits)
    return words


def nearest_stock_distance(
    queries: np.ndarray,
    stock: np.ndarray,
    *,
    weight: float,
    budget_bytes: int = NEAREST_BUDGET_BYTES,
) -> np.ndarray:
    """Exact nearest-stock binary Tanimoto distance, one ``float32`` per query.

    The host twin of :func:`vesma.cuda_circularfp.nearest_stock_distance_cuda`, and the same
    answer: ``weight * (1 - max_s |q & s| / |q | s|)``, with an all-zero pair scoring similarity
    ``0`` rather than a nan.

    ``budget_bytes`` caps the one ``(query tile, stock block, words)`` intermediate this allocates,
    and both tiles are derived from it, so **peak memory does not depend on how many queries
    arrive**. That is the whole reason it is a byte budget rather than a row count: a knob that
    bounds only the stock block still scales the intermediate linearly in the batch, and an AO*
    pass hands this the several thousand precursors it just discovered — 19,000 queries against a
    4,096-row block is a 20 GB allocation, which is not a slow answer but no answer at all.
    Answers are identical at every budget; throughput is flat from 4 MB up, so the default is
    chosen to be small enough to stay resident rather than large enough to be fast.

    The union comes from the two popcount vectors via ``|q | s| = |q| + |s| - |q & s|`` instead of
    a second popcount over the pairs. Same identity the counts branch of
    :func:`vertsy.similarity.stock_similarity` uses, and worth stating because it halves the work:
    the per-pair popcount is this function's entire cost, and one of the two was redundant.
    """
    if weight < 0.0:
        raise ValueError(f"weight must be non-negative, got {weight!r}.")
    if queries.ndim != 2 or stock.ndim != 2:
        raise ValueError("fingerprint arrays must be two-dimensional.")
    if queries.shape[1] != stock.shape[1]:
        raise ValueError(
            f"fingerprint widths differ: {queries.shape[1]} query words against "
            f"{stock.shape[1]} stock words."
        )
    if budget_bytes <= 0:
        raise ValueError(f"budget_bytes must be positive, got {budget_bytes!r}.")
    if not len(queries):
        return np.zeros(0, dtype=np.float32)
    if not len(stock):
        raise ValueError("stock fingerprints must not be empty.")

    row_bytes = queries.shape[1] * queries.itemsize
    tile = max(1, min(len(queries), budget_bytes // row_bytes))
    block = max(1, budget_bytes // (tile * row_bytes))
    query_bits = np.bitwise_count(queries).sum(axis=1, dtype=np.int64)
    stock_bits = np.bitwise_count(stock).sum(axis=1, dtype=np.int64)

    best = np.zeros(len(queries), dtype=np.float32)
    for start in range(0, len(queries), tile):
        stop = start + tile
        pairs = queries[start:stop, None, :]
        counts = query_bits[start:stop, None]
        nearest = best[start:stop]
        for offset in range(0, len(stock), block):
            candidates = stock[offset : offset + block]
            intersection = np.bitwise_count(pairs & candidates[None, :, :]).sum(axis=2)
            union = counts + stock_bits[None, offset : offset + block] - intersection
            similarity = intersection / np.maximum(union, 1)
            np.maximum(nearest, similarity.max(axis=1).astype(np.float32), out=nearest)
    return (weight * (1.0 - best)).astype(np.float32)

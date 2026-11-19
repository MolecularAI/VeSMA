"""Exact nearest-stock Tanimoto distance on Apple Metal.

The device counterpart of :func:`vesma.packed_circularfp.nearest_stock_distance`, computing the same
number by the same definition and differing only in where the popcounts run. The host path is the
specification and ``tests/vesma/test_metal_circularfp.py`` pins the two together; the comparison is
exact rather than approximate because the kernel picks its nearest neighbour by integer
cross-multiplication rather than by dividing, so both paths agree on *which* stock row is nearest
before either of them rounds anything.

Why this exists at all, in one number: the estimate is priced per (query, catalogue row) pair, and
one AO* pass at width 64 over a 67k-rule library discovers ~20,000 novel precursors. Against a 988k
Enamine catalogue that is 6e11 popcounts per pass, which the host path pays at ~30 ms per query --
ten CPU-minutes for one pass, hours for a 64x16 search. It is the search's dominant cost by two
orders of magnitude, and it is also embarrassingly parallel and arithmetic-dense, which is the
profile a GPU exists for.

**The stock stays on the device between calls**, keyed by ``stock_token``. A catalogue is 253 MB and
a search scores against it once per pass, so re-uploading would cost more than scanning. The token
is an identity the caller mints (see :func:`next_stock_token`) rather than anything derived from the
array, because a freed array can be replaced by another of the same shape at the same address and
that would be a wrong answer rather than a slow one. Pass ``0`` to force a fresh upload.

One catalogue is resident at a time. Alternating two large stocks re-uploads on every call, which is
correct and slow; a search holds one catalogue for its whole life, which is the case this serves.

**Note on binned bounded k-NN, which this deliberately does not do.** A natural alternative is to
reserve one bin per attainable distance — 2048 for Hamming, or ~100 for Jaccard bucketed at 0.01 —
and have each lane write a found/not-found flag into its bin as it scans, then read the lowest
occupied bin. Its appeal is that the writes are idempotent, so collisions need no arbitration and the
scan needs no reduction at all. It buys nothing *here*: this asks for one scalar per query, which the
atomic max over an IEEE bit pattern already answers in one word with the same collision-freedom and
without quantising the answer. Where the bins earn their keep is when the *identities* are wanted
rather than the value — nearest-neighbour indices, or ``k > 1`` neighbours — because then a single
running max cannot hold the answer, a full sort is far more work than the question needs, and a bin
sweep bounds the candidate set to the few buckets nearest the target before anything is ranked
exactly. Reach for them when a caller wants "which stock molecules", not "how far".
"""

from __future__ import annotations

import ctypes
import itertools
from collections.abc import Callable
from functools import cache
from pathlib import Path

import numpy as np

from .circularfp import CircularFingerprintConfig
from .metal_runtime import f32_ptr, load_metal_dylib, u32_ptr, u64_ptr

__all__ = [
    "METAL_MIN_STOCK_TILE_ROWS",
    "METAL_QUERY_TILE_BYTES",
    "METAL_TARGET_THREADGROUPS",
    "compute_circular_fingerprints_metal",
    "even_words",
    "metal_circularfp_available",
    "metal_tiling",
    "nearest_stock_distance_metal",
    "nearest_stock_similarity_metal",
    "next_stock_token",
    "packed_bit_counts",
]

METAL_CIRCULARFP_SOURCE_PATH = Path(__file__).with_name("metal_circularfp.mm")

# A threadgroup holds one tile of queries in threadgroup memory and streams the stock past it, so
# this budget is what sets the tile: 16 KB of the device's 32 KB leaves room for two threadgroups
# per core, which matters more than a wider tile does. At 2048 bits that is 64 queries, and the
# catalogue is then read `queries / 64` times rather than once per query -- the difference between
# 76 GB of traffic per pass and 4.9 TB.
METAL_QUERY_TILE_BYTES = 16 << 10

# Enough threadgroups to fill an 8-core GPU several times over. The query tiles alone supply this
# for a full frontier (19k queries is 302 tiles); splitting the stock is what keeps the *initial*
# frontier of one molecule from running on a single threadgroup.
METAL_TARGET_THREADGROUPS = 1024

# Below this a threadgroup spends more time loading its query tile than scanning, so the stock is
# split no further even when that leaves the GPU under-filled.
METAL_MIN_STOCK_TILE_ROWS = 512

STOCK_TOKENS = itertools.count(1)


def next_stock_token() -> int:
    """A fresh identity for a catalogue, for use as ``stock_token``.

    Monotonic and process-local, which is exactly the guarantee needed: two live catalogues can
    never share a token, so a cache hit can never be a stale one. Mint one where the stock is built
    and keep it for as long as the stock lives.
    """
    return next(STOCK_TOKENS)


@cache
def metal_circularfp_library() -> ctypes.CDLL:
    """The compiled dylib, built on first use and rebuilt when the source outgrows it."""
    lib = load_metal_dylib(METAL_CIRCULARFP_SOURCE_PATH)

    lib.vesma_metal_circularfp_last_error.restype = ctypes.c_char_p
    lib.vesma_metal_circularfp_max_threadgroup_memory.restype = ctypes.c_uint32
    lib.vesma_metal_circularfp_max_threads_per_group.restype = ctypes.c_uint32

    lib.vesma_metal_circular_fingerprints.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),  # packed atoms
        ctypes.POINTER(ctypes.c_uint8),  # packed bonds
        ctypes.POINTER(ctypes.c_uint32),  # atom counts
        ctypes.c_uint32,  # molecule count
        ctypes.c_uint32,  # radius
        ctypes.c_uint32,  # bit count
        ctypes.POINTER(ctypes.c_uint64),  # fingerprints
        ctypes.POINTER(ctypes.c_double),  # kernel seconds, device clock
    ]
    lib.vesma_metal_circular_fingerprints.restype = ctypes.c_int

    lib.vesma_metal_nearest_stock_similarity.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),  # stock
        ctypes.POINTER(ctypes.c_uint32),  # stock_popcounts
        ctypes.c_uint32,  # stock_count
        ctypes.POINTER(ctypes.c_uint64),  # queries
        ctypes.POINTER(ctypes.c_uint32),  # query_popcounts
        ctypes.c_uint32,  # query_count
        ctypes.c_uint32,  # word_count
        ctypes.c_uint64,  # stock_token
        ctypes.c_uint32,  # query_tile
        ctypes.c_uint32,  # stock_tile_rows
        ctypes.POINTER(ctypes.c_float),  # out
        ctypes.POINTER(ctypes.c_double),  # kernel seconds, device clock
    ]
    lib.vesma_metal_nearest_stock_similarity.restype = ctypes.c_int
    return lib


def metal_circularfp_available() -> bool:
    """Whether a Metal device is reachable and the shader compiles, without raising if not.

    For test collection and backend selection. A sandbox that denies GPU access fails here rather
    than in the middle of a search.
    """
    try:
        return bool(
            metal_circularfp_library().vesma_metal_circularfp_max_threads_per_group()
        )
    except Exception:
        return False


def even_words(fingerprints: np.ndarray) -> np.ndarray:
    """The same fingerprints in a whole number of ``uint4``, padding one zero word if need be.

    The kernel reads four 32-bit words per iteration, so a row must be an even number of ``uint64``
    words. Padding with a zero word changes neither the intersection nor the union, so this is a
    layout change and not an approximation — and no width anyone uses is odd, so the copy it costs is
    paid only by a 64-bit fingerprint in a test.
    """
    if fingerprints.shape[1] % 2 == 0:
        return np.ascontiguousarray(fingerprints)
    return np.pad(fingerprints, ((0, 0), (0, 1)))


def packed_bit_counts(fingerprints: np.ndarray) -> np.ndarray:
    """Set bits per row, as ``uint32``, for the union identity the kernel uses.

    Precomputed on the host and uploaded with the stock because in the kernel it would be recomputed
    once per (query tile, row) rather than once per row -- the whole saving is that it is *not* in
    the inner loop.

    **These counts also bound the similarity, and that bound is worthless here.** Tanimoto is at
    most ``min(|A|, |B|) / max(|A|, |B|)``, so a row whose ceiling cannot beat the best found so
    far could be skipped before the word loop runs at all -- the classic length filter. Measured
    against the real catalogue and real frontier queries it prunes **1.2%**, and that is the
    optimistic figure, taken against each query's *final* best rather than the running one.

    The distributions say why, and they are a property of this chemistry rather than of the idea:
    query popcounts run 26 to 51 (median 39) and stock popcounts 24 to 43 (median 33), because
    drug-like molecules all have similar ECFP4 density, while the best similarities actually found
    run 0.29 to 0.54 (median 0.37). A ratio of two numbers that close is almost always above 0.7,
    and the filter only starts paying at a best of 0.7 -- where it would prune 35%, which this
    catalogue never reaches. Recorded so the next reader does not implement it to find out.
    """
    return np.bitwise_count(fingerprints).sum(axis=1, dtype=np.uint32)


def metal_tiling(
    query_count: int, stock_count: int, word_count: int
) -> tuple[int, int]:
    """``(queries per threadgroup, stock rows per threadgroup)`` for one scan.

    The query tile follows from :data:`METAL_QUERY_TILE_BYTES` and is a multiple of the 32-wide
    simdgroup so no lanes are idle. The stock is then split only as far as it takes to reach
    :data:`METAL_TARGET_THREADGROUPS`, because splitting it re-loads the query tile and re-runs the
    final atomic: a full frontier needs no split at all, and a one-molecule frontier needs a large
    one or it runs on a single threadgroup.

    **Swept once the kernel had a device clock, and the choice does not matter.** Over 20
    combinations at the shape the heuristic actually uses -- 440 queries against 920,700 rows at 32
    words -- every query tile in (32, 64, 128, 256) crossed with every stock tile from 4,096 rows
    to the whole catalogue landed between 88.5 ms and 90.6 ms, a 2.4% spread, with what this
    function chooses within 2% of the best of them. So the derivation above arrives somewhere flat
    rather than somewhere lucky, and there is no retuning available: the scan is bound by the work
    itself, 440 x 920,700 x 32 words of popcount, 98% of it on the GPU.
    """
    row_bytes = word_count * 8
    tile = max(32, min(256, (METAL_QUERY_TILE_BYTES // row_bytes) // 32 * 32))
    query_groups = -(-query_count // tile)
    stock_groups = min(
        max(1, -(-METAL_TARGET_THREADGROUPS // query_groups)),
        max(1, -(-stock_count // METAL_MIN_STOCK_TILE_ROWS)),
    )
    return tile, -(-stock_count // stock_groups)


def compute_circular_fingerprints_metal(
    atoms: np.ndarray,
    bonds: np.ndarray,
    atom_counts: np.ndarray,
    config: CircularFingerprintConfig = CircularFingerprintConfig(),
    record: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Fingerprint one packed host batch on Metal and return packed ``uint64`` words.

    `record`, if given, is handed the kernel's seconds on the **device's own clock** --
    `GPUEndTime - GPUStartTime`, the same measurement `metal_retro.mm` reports. A sink rather than
    a return value, on the same reasoning `report=` is a sink for the forward probe's coverage:
    callers in a hot path should not have to unpack a tuple to ignore a number, and the timing
    belongs to the call that produced it rather than to an attribute that outlives it.
    """
    count = len(atom_counts)
    out = np.zeros((count, config.n_bits // 64), dtype=np.uint64)
    if not count:
        return out
    packed_atoms = np.ascontiguousarray(atoms)
    packed_bonds = np.ascontiguousarray(bonds)
    packed_counts = np.ascontiguousarray(atom_counts)
    lib = metal_circularfp_library()
    kernel_seconds = ctypes.c_double(0.0)
    ok = lib.vesma_metal_circular_fingerprints(
        packed_atoms.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        packed_bonds.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        packed_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        count,
        config.radius,
        config.n_bits,
        u64_ptr(out),
        ctypes.byref(kernel_seconds),
    )
    if ok != 1:
        message = lib.vesma_metal_circularfp_last_error()
        raise RuntimeError(
            message.decode("utf-8") if message else "Unknown Metal error"
        )
    if record is not None:
        record(float(kernel_seconds.value))
    return out


def nearest_stock_similarity_metal(
    queries: np.ndarray,
    stock: np.ndarray,
    *,
    stock_token: int,
    stock_bit_counts: np.ndarray | None = None,
    record: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Largest Tanimoto similarity between each query and any stock row: one ``float32`` each.

    `record`, if given, is handed the scan kernel's seconds on the device's own clock. This is the
    measurement that was missing: timed from the host, a call is ~104 ms and nothing could say how
    much of that was the scan.

    The similarity rather than the distance, so that the ``weight * (1 - s)`` which turns it into a
    heuristic estimate happens in exactly one place across both backends —
    :func:`nearest_stock_distance_metal` here, numpy on the host — and a parity test is therefore a
    test of the kernel and not of that arithmetic.

    ``stock_bit_counts`` is :func:`packed_bit_counts` of ``stock`` and may be passed in when the
    caller already holds it, which a search does: recomputing it is a numpy pass over the whole
    253 MB catalogue on every call, for a quantity that never changes.
    """
    if queries.ndim != 2 or stock.ndim != 2:
        raise ValueError("fingerprint arrays must be two-dimensional.")
    if queries.dtype != np.uint64 or stock.dtype != np.uint64:
        raise ValueError("fingerprint arrays must have dtype uint64.")
    if queries.shape[1] != stock.shape[1]:
        raise ValueError(
            f"fingerprint widths differ: {queries.shape[1]} query words against "
            f"{stock.shape[1]} stock words."
        )
    if not len(stock):
        raise ValueError("stock fingerprints must not be empty.")
    if not len(queries):
        return np.zeros(0, dtype=np.float32)

    stock_rows = even_words(stock)
    query_rows = even_words(queries)
    word_count = int(stock_rows.shape[1])
    counts = (
        packed_bit_counts(stock_rows)
        if stock_bit_counts is None
        else np.ascontiguousarray(stock_bit_counts, dtype=np.uint32)
    )
    if counts.shape != (len(stock_rows),):
        raise ValueError(
            f"stock_bit_counts must have one entry per stock row, got {counts.shape} "
            f"for {len(stock_rows)} rows."
        )
    query_counts = packed_bit_counts(query_rows)
    tile, stock_tile_rows = metal_tiling(len(query_rows), len(stock_rows), word_count)

    out = np.zeros(len(query_rows), dtype=np.float32)
    lib = metal_circularfp_library()
    kernel_seconds = ctypes.c_double(0.0)
    ok = lib.vesma_metal_nearest_stock_similarity(
        u64_ptr(stock_rows.reshape(-1)),
        u32_ptr(counts),
        len(stock_rows),
        u64_ptr(query_rows.reshape(-1)),
        u32_ptr(query_counts),
        len(query_rows),
        word_count,
        stock_token,
        tile,
        stock_tile_rows,
        f32_ptr(out),
        ctypes.byref(kernel_seconds),
    )
    if ok != 1:
        message = lib.vesma_metal_circularfp_last_error()
        raise RuntimeError(
            message.decode("utf-8") if message else "Unknown Metal error"
        )
    if record is not None:
        record(float(kernel_seconds.value))
    return out


def nearest_stock_distance_metal(
    queries: np.ndarray,
    stock: np.ndarray,
    *,
    weight: float,
    stock_token: int,
    stock_bit_counts: np.ndarray | None = None,
) -> np.ndarray:
    """Exact nearest-stock binary Tanimoto distance, one ``float32`` per query.

    Signature-compatible with :func:`vesma.packed_circularfp.nearest_stock_distance` apart from
    ``stock_token``, and answering to the same definition: ``weight * (1 - max_s |q & s| / |q | s|)``
    with an all-zero pair scoring similarity ``0`` rather than a nan.
    """
    if weight < 0.0:
        raise ValueError(f"weight must be non-negative, got {weight!r}.")
    best = nearest_stock_similarity_metal(
        queries, stock, stock_token=stock_token, stock_bit_counts=stock_bit_counts
    )
    return (weight * (1.0 - best)).astype(np.float32)

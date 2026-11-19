"""Archived prune-kernel prototypes used only by tests and perf benchmarks.

`prune_dense_reference` preserves the pre-GEMM matcher prune for comparison.
`prune_dense_gemm` mirrors the current live matcher kernel.
The popcount variants keep a CPU-side packed-intersection reference that would
typically be computed on the fly in GPU registers or shared state.
All other functions are prototype-only alternatives and are not production code.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from vesma.adjacency import bond_presence
from vesma.gmol import GMol
from vesma.gmol_atoms import ALIPHATIC, AROMATIC, bucket_mask_exact

BoolMatrix = np.ndarray[tuple[int, int], np.dtype[np.bool_]]
U32Rows = np.ndarray[tuple[int], np.dtype[np.uint32]]
U64Rows = np.ndarray[tuple[int], np.dtype[np.uint64]]
PruneResult = tuple[BoolMatrix, int, bool]

__all__ = [
    "pack_rows_u32",
    "pack_rows_u64",
    "unpack_rows_u32",
    "unpack_rows_u64",
    "prune_dense_reference",
    "prune_dense_gemm",
    "prune_bitset_hybrid_u32",
    "prune_bitset_hybrid_u64",
    "prune_bitset_u32",
    "prune_bitset_u64",
    "prune_bitset_dirty_u32",
    "prune_bitset_dirty_u64",
    "prune_bitset_vectorized_u32",
    "prune_bitset_vectorized_u64",
    "prune_bitset_popcount_u64",
    "prune_bitset_popcount_block16_u64",
    "prune_bitset_popcount_block32_u64",
    "prune_bitset_popcount_block64_u64",
    "signature_prefilter",
    "prune_signature_bitset_u32",
    "prune_signature_bitset_u64",
    "prune_signature_hybrid_u32",
    "prune_signature_hybrid_u64",
    "prune_signature_vectorized_u32",
    "prune_signature_vectorized_u64",
]


def pack_rows_u32(matrix: BoolMatrix) -> U32Rows:
    return pack_rows(matrix, np.uint32, 32)


def pack_rows_u64(matrix: BoolMatrix) -> U64Rows:
    return pack_rows(matrix, np.uint64, 64)


def unpack_rows_u32(rows: U32Rows, width: int) -> BoolMatrix:
    return unpack_rows(rows, width, 32)


def unpack_rows_u64(rows: U64Rows, width: int) -> BoolMatrix:
    return unpack_rows(rows, width, 64)


def pack_rows(
    matrix: BoolMatrix,
    dtype: type[np.uint32] | type[np.uint64],
    word_bits: Literal[32, 64],
) -> U32Rows | U64Rows:
    if matrix.shape[1] > word_bits:
        raise ValueError(f"Packed row width exceeds {word_bits} bits.")
    if matrix.shape[1] == 0:
        return np.zeros(matrix.shape[0], dtype=dtype)

    padded = np.pad(
        matrix.astype(np.uint8, copy=False),
        ((0, 0), (0, word_bits - matrix.shape[1])),
    )
    packed_bytes = np.ascontiguousarray(np.packbits(padded, axis=1, bitorder="little"))
    return packed_bytes.view(dtype).reshape(matrix.shape[0])


def unpack_rows(
    rows: U32Rows | U64Rows,
    width: int,
    word_bits: Literal[32, 64],
) -> BoolMatrix:
    if width > word_bits:
        raise ValueError(f"Unpacked row width exceeds {word_bits} bits.")
    if width == 0:
        return np.zeros((rows.shape[0], 0), dtype=bool)

    packed_bytes = (
        np.ascontiguousarray(rows)
        .view(np.uint8)
        .reshape(
            rows.shape[0],
            word_bits // 8,
        )
    )
    return np.unpackbits(packed_bytes, axis=1, bitorder="little")[:, :width].astype(
        bool, copy=False
    )


def prune_dense_reference(
    candidates: BoolMatrix,
    pattern_edges: BoolMatrix,
    molecule_edges: BoolMatrix,
    *,
    max_prune_rounds: int | None = None,
) -> PruneResult:
    if candidates.size == 0:
        return candidates.copy(), 0, True

    current = candidates.copy()
    prune_rounds = 0

    while True:
        if max_prune_rounds is not None and prune_rounds >= max_prune_rounds:
            return current, prune_rounds, False

        support = np.einsum(
            "jl,kl->kj",
            molecule_edges.astype(np.int8, copy=False),
            current.astype(np.int8, copy=False),
            optimize=True,
        )
        refined = current & np.all(
            (~pattern_edges)[:, :, None] | (support > 0)[None, :, :],
            axis=1,
        )
        prune_rounds += 1
        if np.array_equal(refined, current):
            return refined, prune_rounds, True
        current = refined
        if not np.all(current.any(axis=1)):
            return current, prune_rounds, True


def prune_dense_gemm(
    candidates: BoolMatrix,
    pattern_edges: BoolMatrix,
    molecule_edges: BoolMatrix,
    *,
    max_prune_rounds: int | None = None,
) -> PruneResult:
    if candidates.size == 0:
        return candidates.copy(), 0, True

    current = candidates.copy()
    prune_rounds = 0
    pattern_edges_i16 = pattern_edges.astype(np.int16, copy=False)
    molecule_edges_i16 = molecule_edges.astype(np.int16, copy=False)
    pattern_degree = pattern_edges_i16.sum(axis=1, keepdims=True, dtype=np.int16)

    while True:
        if max_prune_rounds is not None and prune_rounds >= max_prune_rounds:
            return current, prune_rounds, False

        support = (current.astype(np.int16, copy=False) @ molecule_edges_i16.T) > 0
        satisfied_neighbors = pattern_edges_i16 @ support.astype(np.int16, copy=False)
        refined = current & (satisfied_neighbors == pattern_degree)
        prune_rounds += 1
        if np.array_equal(refined, current):
            return refined, prune_rounds, True
        current = refined
        if not np.all(current.any(axis=1)):
            return current, prune_rounds, True


def prune_bitset_u32(
    candidates: BoolMatrix,
    pattern_edges: BoolMatrix,
    molecule_edges: BoolMatrix,
    *,
    max_prune_rounds: int | None = None,
) -> PruneResult:
    return prune_bitset_single_word(
        candidates,
        pattern_edges,
        molecule_edges,
        word_bits=32,
        pack_rows_fn=pack_rows_u32,
        unpack_rows_fn=unpack_rows_u32,
        max_prune_rounds=max_prune_rounds,
    )


def prune_bitset_u64(
    candidates: BoolMatrix,
    pattern_edges: BoolMatrix,
    molecule_edges: BoolMatrix,
    *,
    max_prune_rounds: int | None = None,
) -> PruneResult:
    return prune_bitset_single_word(
        candidates,
        pattern_edges,
        molecule_edges,
        word_bits=64,
        pack_rows_fn=pack_rows_u64,
        unpack_rows_fn=unpack_rows_u64,
        max_prune_rounds=max_prune_rounds,
    )


def prune_bitset_dirty_u32(
    candidates: BoolMatrix,
    pattern_edges: BoolMatrix,
    molecule_edges: BoolMatrix,
    *,
    max_prune_rounds: int | None = None,
) -> PruneResult:
    return prune_bitset_dirty_single_word(
        candidates,
        pattern_edges,
        molecule_edges,
        word_bits=32,
        pack_rows_fn=pack_rows_u32,
        unpack_rows_fn=unpack_rows_u32,
        max_prune_rounds=max_prune_rounds,
    )


def prune_bitset_dirty_u64(
    candidates: BoolMatrix,
    pattern_edges: BoolMatrix,
    molecule_edges: BoolMatrix,
    *,
    max_prune_rounds: int | None = None,
) -> PruneResult:
    return prune_bitset_dirty_single_word(
        candidates,
        pattern_edges,
        molecule_edges,
        word_bits=64,
        pack_rows_fn=pack_rows_u64,
        unpack_rows_fn=unpack_rows_u64,
        max_prune_rounds=max_prune_rounds,
    )


def prune_bitset_hybrid_u32(
    candidates: BoolMatrix,
    pattern_edges: BoolMatrix,
    molecule_edges: BoolMatrix,
    *,
    max_prune_rounds: int | None = None,
) -> PruneResult:
    return prune_bitset_hybrid_single_word(
        candidates,
        pattern_edges,
        molecule_edges,
        word_bits=32,
        pack_rows_fn=pack_rows_u32,
        max_prune_rounds=max_prune_rounds,
    )


def prune_bitset_hybrid_u64(
    candidates: BoolMatrix,
    pattern_edges: BoolMatrix,
    molecule_edges: BoolMatrix,
    *,
    max_prune_rounds: int | None = None,
) -> PruneResult:
    return prune_bitset_hybrid_single_word(
        candidates,
        pattern_edges,
        molecule_edges,
        word_bits=64,
        pack_rows_fn=pack_rows_u64,
        max_prune_rounds=max_prune_rounds,
    )


def prune_bitset_vectorized_u32(
    candidates: BoolMatrix,
    pattern_edges: BoolMatrix,
    molecule_edges: BoolMatrix,
    *,
    max_prune_rounds: int | None = None,
) -> PruneResult:
    return prune_bitset_vectorized_single_word(
        candidates,
        pattern_edges,
        molecule_edges,
        word_bits=32,
        pack_rows_fn=pack_rows_u32,
        unpack_rows_fn=unpack_rows_u32,
        dtype=np.uint32,
        max_prune_rounds=max_prune_rounds,
    )


def prune_bitset_vectorized_u64(
    candidates: BoolMatrix,
    pattern_edges: BoolMatrix,
    molecule_edges: BoolMatrix,
    *,
    max_prune_rounds: int | None = None,
) -> PruneResult:
    return prune_bitset_vectorized_single_word(
        candidates,
        pattern_edges,
        molecule_edges,
        word_bits=64,
        pack_rows_fn=pack_rows_u64,
        unpack_rows_fn=unpack_rows_u64,
        dtype=np.uint64,
        max_prune_rounds=max_prune_rounds,
    )


def prune_bitset_popcount_u64(
    candidates: BoolMatrix,
    pattern_edges: BoolMatrix,
    molecule_edges: BoolMatrix,
    *,
    max_prune_rounds: int | None = None,
) -> PruneResult:
    return prune_bitset_popcount_u64_internal(
        candidates,
        pattern_edges,
        molecule_edges,
        block_size=None,
        max_prune_rounds=max_prune_rounds,
    )


def prune_bitset_popcount_block16_u64(
    candidates: BoolMatrix,
    pattern_edges: BoolMatrix,
    molecule_edges: BoolMatrix,
    *,
    max_prune_rounds: int | None = None,
) -> PruneResult:
    return prune_bitset_popcount_u64_internal(
        candidates,
        pattern_edges,
        molecule_edges,
        block_size=16,
        max_prune_rounds=max_prune_rounds,
    )


def prune_bitset_popcount_block32_u64(
    candidates: BoolMatrix,
    pattern_edges: BoolMatrix,
    molecule_edges: BoolMatrix,
    *,
    max_prune_rounds: int | None = None,
) -> PruneResult:
    return prune_bitset_popcount_u64_internal(
        candidates,
        pattern_edges,
        molecule_edges,
        block_size=32,
        max_prune_rounds=max_prune_rounds,
    )


def prune_bitset_popcount_block64_u64(
    candidates: BoolMatrix,
    pattern_edges: BoolMatrix,
    molecule_edges: BoolMatrix,
    *,
    max_prune_rounds: int | None = None,
) -> PruneResult:
    return prune_bitset_popcount_u64_internal(
        candidates,
        pattern_edges,
        molecule_edges,
        block_size=64,
        max_prune_rounds=max_prune_rounds,
    )


def signature_prefilter(
    pattern: GMol,
    molecule: GMol,
    candidates: BoolMatrix,
) -> BoolMatrix:
    if candidates.size == 0:
        return candidates.copy()

    pattern_edges = bond_presence(pattern.bonds)
    molecule_edges = bond_presence(molecule.bonds)

    pattern_degree = pattern_edges.sum(axis=1, dtype=np.int16)
    molecule_degree = molecule_edges.sum(axis=1, dtype=np.int16)

    pattern_aromatic = pattern.atom_masks.A == AROMATIC
    pattern_aliphatic = pattern.atom_masks.A == ALIPHATIC
    molecule_aromatic = molecule.atom_masks.A == AROMATIC
    molecule_aliphatic = molecule.atom_masks.A == ALIPHATIC

    zero_ring_mask = bucket_mask_exact(0)
    pattern_requires_ring = (pattern.atom_masks.R & zero_ring_mask) == 0
    molecule_in_ring = (molecule.atom_masks.R & zero_ring_mask) == 0

    pattern_exact_carbon = (pattern.atom_masks.Z_lo == np.uint64(1 << 6)) & (
        pattern.atom_masks.Z_hi == 0
    )
    molecule_exact_carbon = (molecule.atom_masks.Z_lo == np.uint64(1 << 6)) & (
        molecule.atom_masks.Z_hi == 0
    )

    aromatic_neighbor_req = pattern_edges.astype(
        np.int16, copy=False
    ) @ pattern_aromatic.astype(np.int16, copy=False)
    carbon_neighbor_req = pattern_edges.astype(
        np.int16, copy=False
    ) @ pattern_exact_carbon.astype(np.int16, copy=False)
    ring_neighbor_req = pattern_edges.astype(
        np.int16, copy=False
    ) @ pattern_requires_ring.astype(np.int16, copy=False)

    aromatic_neighbor_count = molecule_edges.astype(
        np.int16, copy=False
    ) @ molecule_aromatic.astype(np.int16, copy=False)
    carbon_neighbor_count = molecule_edges.astype(
        np.int16, copy=False
    ) @ molecule_exact_carbon.astype(np.int16, copy=False)
    ring_neighbor_count = molecule_edges.astype(
        np.int16, copy=False
    ) @ molecule_in_ring.astype(np.int16, copy=False)

    return (
        candidates
        & (molecule_degree[None, :] >= pattern_degree[:, None])
        & (~pattern_aromatic[:, None] | molecule_aromatic[None, :])
        & (~pattern_aliphatic[:, None] | molecule_aliphatic[None, :])
        & (~pattern_requires_ring[:, None] | molecule_in_ring[None, :])
        & (~pattern_exact_carbon[:, None] | molecule_exact_carbon[None, :])
        & (aromatic_neighbor_count[None, :] >= aromatic_neighbor_req[:, None])
        & (carbon_neighbor_count[None, :] >= carbon_neighbor_req[:, None])
        & (ring_neighbor_count[None, :] >= ring_neighbor_req[:, None])
    )


def prune_signature_bitset_u32(
    pattern: GMol,
    molecule: GMol,
    candidates: BoolMatrix,
    pattern_edges: BoolMatrix,
    molecule_edges: BoolMatrix,
    *,
    max_prune_rounds: int | None = None,
) -> PruneResult:
    filtered = signature_prefilter(pattern, molecule, candidates)
    return prune_bitset_u32(
        filtered,
        pattern_edges,
        molecule_edges,
        max_prune_rounds=max_prune_rounds,
    )


def prune_signature_bitset_u64(
    pattern: GMol,
    molecule: GMol,
    candidates: BoolMatrix,
    pattern_edges: BoolMatrix,
    molecule_edges: BoolMatrix,
    *,
    max_prune_rounds: int | None = None,
) -> PruneResult:
    filtered = signature_prefilter(pattern, molecule, candidates)
    return prune_bitset_u64(
        filtered,
        pattern_edges,
        molecule_edges,
        max_prune_rounds=max_prune_rounds,
    )


def prune_signature_hybrid_u32(
    pattern: GMol,
    molecule: GMol,
    candidates: BoolMatrix,
    pattern_edges: BoolMatrix,
    molecule_edges: BoolMatrix,
    *,
    max_prune_rounds: int | None = None,
) -> PruneResult:
    filtered = signature_prefilter(pattern, molecule, candidates)
    return prune_bitset_hybrid_u32(
        filtered,
        pattern_edges,
        molecule_edges,
        max_prune_rounds=max_prune_rounds,
    )


def prune_signature_hybrid_u64(
    pattern: GMol,
    molecule: GMol,
    candidates: BoolMatrix,
    pattern_edges: BoolMatrix,
    molecule_edges: BoolMatrix,
    *,
    max_prune_rounds: int | None = None,
) -> PruneResult:
    filtered = signature_prefilter(pattern, molecule, candidates)
    return prune_bitset_hybrid_u64(
        filtered,
        pattern_edges,
        molecule_edges,
        max_prune_rounds=max_prune_rounds,
    )


def prune_signature_vectorized_u32(
    pattern: GMol,
    molecule: GMol,
    candidates: BoolMatrix,
    pattern_edges: BoolMatrix,
    molecule_edges: BoolMatrix,
    *,
    max_prune_rounds: int | None = None,
) -> PruneResult:
    filtered = signature_prefilter(pattern, molecule, candidates)
    return prune_bitset_vectorized_u32(
        filtered,
        pattern_edges,
        molecule_edges,
        max_prune_rounds=max_prune_rounds,
    )


def prune_signature_vectorized_u64(
    pattern: GMol,
    molecule: GMol,
    candidates: BoolMatrix,
    pattern_edges: BoolMatrix,
    molecule_edges: BoolMatrix,
    *,
    max_prune_rounds: int | None = None,
) -> PruneResult:
    filtered = signature_prefilter(pattern, molecule, candidates)
    return prune_bitset_vectorized_u64(
        filtered,
        pattern_edges,
        molecule_edges,
        max_prune_rounds=max_prune_rounds,
    )


def prune_bitset_single_word(
    candidates: BoolMatrix,
    pattern_edges: BoolMatrix,
    molecule_edges: BoolMatrix,
    *,
    word_bits: Literal[32, 64],
    pack_rows_fn,
    unpack_rows_fn,
    max_prune_rounds: int | None = None,
) -> PruneResult:
    if candidates.size == 0:
        return candidates.copy(), 0, True

    rows, pattern_rows, molecule_rows, column_bits, row_bits = single_word_payload(
        candidates,
        pattern_edges,
        molecule_edges,
        word_bits=word_bits,
        pack_rows_fn=pack_rows_fn,
    )
    pruned_rows, prune_rounds, pruning_complete = prune_rows_full_scan(
        rows,
        pattern_rows,
        molecule_rows,
        column_bits,
        row_bits,
        max_prune_rounds=max_prune_rounds,
    )
    return (
        unpack_rows_fn(
            np.array(pruned_rows, dtype=np.uint32 if word_bits == 32 else np.uint64),
            candidates.shape[1],
        ),
        prune_rounds,
        pruning_complete,
    )


def prune_bitset_dirty_single_word(
    candidates: BoolMatrix,
    pattern_edges: BoolMatrix,
    molecule_edges: BoolMatrix,
    *,
    word_bits: Literal[32, 64],
    pack_rows_fn,
    unpack_rows_fn,
    max_prune_rounds: int | None = None,
) -> PruneResult:
    if candidates.size == 0:
        return candidates.copy(), 0, True

    rows, pattern_rows, molecule_rows, column_bits, _ = single_word_payload(
        candidates,
        pattern_edges,
        molecule_edges,
        word_bits=word_bits,
        pack_rows_fn=pack_rows_fn,
    )
    pruned_rows, prune_rounds, pruning_complete = prune_rows_dirty(
        rows,
        pattern_rows,
        molecule_rows,
        column_bits,
        max_prune_rounds=max_prune_rounds,
    )
    return (
        unpack_rows_fn(
            np.array(pruned_rows, dtype=np.uint32 if word_bits == 32 else np.uint64),
            candidates.shape[1],
        ),
        prune_rounds,
        pruning_complete,
    )


def prune_bitset_hybrid_single_word(
    candidates: BoolMatrix,
    pattern_edges: BoolMatrix,
    molecule_edges: BoolMatrix,
    *,
    word_bits: Literal[32, 64],
    pack_rows_fn,
    max_prune_rounds: int | None = None,
) -> PruneResult:
    if candidates.size == 0:
        return candidates.copy(), 0, True
    if candidates.shape[0] > word_bits:
        raise ValueError(f"Pattern atom count exceeds {word_bits} bits.")
    if candidates.shape[1] > word_bits:
        raise ValueError(f"Molecule atom count exceeds {word_bits} bits.")

    current = candidates.copy()
    current_packed = pack_rows_fn(current)
    molecule_rows = pack_rows_fn(molecule_edges)
    pattern_edges_i16 = pattern_edges.astype(np.int16, copy=False)
    pattern_degree = pattern_edges_i16.sum(axis=1, keepdims=True, dtype=np.int16)
    prune_rounds = 0

    while True:
        if max_prune_rounds is not None and prune_rounds >= max_prune_rounds:
            return current, prune_rounds, False

        support = (current_packed[:, None] & molecule_rows[None, :]) != 0
        satisfied_neighbors = pattern_edges_i16 @ support.astype(np.int16, copy=False)
        refined = current & (satisfied_neighbors == pattern_degree)
        prune_rounds += 1
        if np.array_equal(refined, current):
            return refined, prune_rounds, True
        current = refined
        if not np.all(current.any(axis=1)):
            return current, prune_rounds, True
        current_packed = pack_rows_fn(current)


def prune_bitset_vectorized_single_word(
    candidates: BoolMatrix,
    pattern_edges: BoolMatrix,
    molecule_edges: BoolMatrix,
    *,
    word_bits: Literal[32, 64],
    pack_rows_fn,
    unpack_rows_fn,
    dtype: type[np.uint32] | type[np.uint64],
    max_prune_rounds: int | None = None,
) -> PruneResult:
    if candidates.size == 0:
        return candidates.copy(), 0, True

    rows, pattern_rows, molecule_rows, _, _ = single_word_payload(
        candidates,
        pattern_edges,
        molecule_edges,
        word_bits=word_bits,
        pack_rows_fn=pack_rows_fn,
    )
    current = np.array(rows, dtype=dtype)
    pattern_rows_array = np.array(pattern_rows, dtype=dtype)
    molecule_rows_array = np.array(molecule_rows, dtype=dtype)
    prune_rounds = 0

    while True:
        if max_prune_rounds is not None and prune_rounds >= max_prune_rounds:
            return (
                unpack_rows_fn(current, candidates.shape[1]),
                prune_rounds,
                False,
            )

        support_bool = (current[None, :] & molecule_rows_array[:, None]) != 0
        supported_rows = pack_rows_fn(support_bool)
        allowed_columns = (pattern_rows_array[:, None] & ~supported_rows[None, :]) == 0
        allowed_rows = pack_rows_fn(allowed_columns)
        refined = current & allowed_rows
        prune_rounds += 1
        if np.array_equal(refined, current):
            return (
                unpack_rows_fn(refined, candidates.shape[1]),
                prune_rounds,
                True,
            )
        current = refined
        if not np.all(current != dtype(0)):
            return (
                unpack_rows_fn(current, candidates.shape[1]),
                prune_rounds,
                True,
            )


def prune_bitset_popcount_u64_internal(
    candidates: BoolMatrix,
    pattern_edges: BoolMatrix,
    molecule_edges: BoolMatrix,
    *,
    block_size: Literal[16, 32, 64] | None,
    max_prune_rounds: int | None = None,
) -> PruneResult:
    if candidates.size == 0:
        return candidates.copy(), 0, True
    if candidates.shape[0] > 64:
        raise ValueError("Pattern atom count exceeds 64 bits.")
    if candidates.shape[1] > 64:
        raise ValueError("Molecule atom count exceeds 64 bits.")

    current = candidates.copy()
    molecule_rows = pack_rows_u64(molecule_edges)
    pattern_edges_i16 = pattern_edges.astype(np.int16, copy=False)
    pattern_degree = pattern_edges_i16.sum(axis=1, keepdims=True, dtype=np.int16)
    prune_rounds = 0

    while True:
        if max_prune_rounds is not None and prune_rounds >= max_prune_rounds:
            return current, prune_rounds, False

        support = popcount_support_u64(
            pack_rows_u64(current),
            molecule_rows,
            block_size=block_size,
        )
        satisfied_neighbors = pattern_edges_i16 @ support.astype(np.int16, copy=False)
        refined = current & (satisfied_neighbors == pattern_degree)
        prune_rounds += 1
        if np.array_equal(refined, current):
            return refined, prune_rounds, True
        current = refined
        if not np.all(current.any(axis=1)):
            return current, prune_rounds, True


def popcount_support_u64(
    current_rows: U64Rows,
    molecule_rows: U64Rows,
    *,
    block_size: Literal[16, 32, 64] | None,
) -> BoolMatrix:
    if block_size is None:
        intersection = current_rows[:, None, None] & molecule_rows[None, :, None]
        popcounts = np.bitwise_count(intersection)
        return popcounts.sum(axis=-1, dtype=np.uint16) > 0

    support = np.zeros((current_rows.shape[0], molecule_rows.shape[0]), dtype=bool)
    for start in range(0, molecule_rows.shape[0], block_size):
        block = molecule_rows[start : start + block_size]
        intersection = current_rows[:, None, None] & block[None, :, None]
        popcounts = np.bitwise_count(intersection)
        support[:, start : start + block.shape[0]] = (
            popcounts.sum(axis=-1, dtype=np.uint16) > 0
        )
    return support


def single_word_payload(
    candidates: BoolMatrix,
    pattern_edges: BoolMatrix,
    molecule_edges: BoolMatrix,
    *,
    word_bits: Literal[32, 64],
    pack_rows_fn,
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    if candidates.shape[0] > word_bits:
        raise ValueError(f"Pattern atom count exceeds {word_bits} bits.")
    if candidates.shape[1] > word_bits:
        raise ValueError(f"Molecule atom count exceeds {word_bits} bits.")

    return (
        [int(value) for value in pack_rows_fn(candidates)],
        [int(value) for value in pack_rows_fn(pattern_edges)],
        [int(value) for value in pack_rows_fn(molecule_edges)],
        [1 << idx for idx in range(candidates.shape[1])],
        [1 << idx for idx in range(candidates.shape[0])],
    )


def prune_rows_full_scan(
    rows: list[int],
    pattern_rows: list[int],
    molecule_rows: list[int],
    column_bits: list[int],
    row_bits: list[int],
    *,
    max_prune_rounds: int | None,
) -> tuple[list[int], int, bool]:
    current = list(rows)
    prune_rounds = 0

    while True:
        if max_prune_rounds is not None and prune_rounds >= max_prune_rounds:
            return current, prune_rounds, False

        supported_rows = column_supported_rows(current, molecule_rows, row_bits)
        refined = refined_rows(current, pattern_rows, supported_rows, column_bits)
        prune_rounds += 1
        if refined == current:
            return refined, prune_rounds, True
        current = refined
        if not has_row_support(current):
            return current, prune_rounds, True


def prune_rows_dirty(
    rows: list[int],
    pattern_rows: list[int],
    molecule_rows: list[int],
    column_bits: list[int],
    *,
    max_prune_rounds: int | None,
) -> tuple[list[int], int, bool]:
    current = list(rows)
    dirty_rows = (1 << len(rows)) - 1
    prune_rounds = 0

    while True:
        if max_prune_rounds is not None and prune_rounds >= max_prune_rounds:
            return current, prune_rounds, False

        refined = list(current)
        changed_rows = 0
        bits = dirty_rows
        while bits:
            bit = bits & -bits
            row_idx = bit.bit_length() - 1
            refined_row = refine_row(
                current,
                pattern_rows[row_idx],
                molecule_rows,
                column_bits,
                current[row_idx],
            )
            refined[row_idx] = refined_row
            if refined_row != current[row_idx]:
                changed_rows |= bit
            bits ^= bit

        prune_rounds += 1
        if not has_row_support(refined):
            return refined, prune_rounds, True
        if changed_rows == 0:
            return refined, prune_rounds, True

        current = refined
        dirty_rows = next_dirty_rows(changed_rows, pattern_rows)
        if dirty_rows == 0:
            return current, prune_rounds, True


def column_supported_rows(
    rows: list[int],
    molecule_rows: list[int],
    row_bits: list[int],
) -> list[int]:
    supported_rows: list[int] = []
    for molecule_row in molecule_rows:
        supported = 0
        for row_word, row_bit in zip(rows, row_bits, strict=True):
            if row_word & molecule_row:
                supported |= row_bit
        supported_rows.append(supported)
    return supported_rows


def refined_rows(
    rows: list[int],
    pattern_rows: list[int],
    supported_rows: list[int],
    column_bits: list[int],
) -> list[int]:
    refined: list[int] = []
    for row_word, pattern_row in zip(rows, pattern_rows, strict=True):
        allowed_columns = 0
        for column_bit, supported in zip(column_bits, supported_rows, strict=True):
            if (pattern_row & ~supported) == 0:
                allowed_columns |= column_bit
        refined.append(row_word & allowed_columns)
    return refined


def refine_row(
    rows: list[int],
    pattern_row: int,
    molecule_rows: list[int],
    column_bits: list[int],
    row_word: int,
) -> int:
    refined = 0
    for column_bit, molecule_row in zip(column_bits, molecule_rows, strict=True):
        if (row_word & column_bit) == 0:
            continue
        neighbors = pattern_row
        supported = True
        while neighbors:
            neighbor_bit = neighbors & -neighbors
            neighbor_row = neighbor_bit.bit_length() - 1
            if (rows[neighbor_row] & molecule_row) == 0:
                supported = False
                break
            neighbors ^= neighbor_bit
        if supported:
            refined |= column_bit
    return refined


def next_dirty_rows(changed_rows: int, pattern_rows: list[int]) -> int:
    dirty_rows = 0
    bits = changed_rows
    while bits:
        bit = bits & -bits
        row_idx = bit.bit_length() - 1
        dirty_rows |= pattern_rows[row_idx]
        bits ^= bit
    return dirty_rows


def has_row_support(rows: list[int]) -> bool:
    return all(row != 0 for row in rows)

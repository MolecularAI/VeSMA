"""CUDA circular fingerprints and exact nearest-stock Tanimoto distance."""

from __future__ import annotations

from typing import Any

import numpy as np

from .circularfp import CircularFingerprintConfig
from .cuda_prune import _require_cupy
from .cuda_retro import CudaMoleculeBatch

__all__ = [
    "compute_circular_fingerprints_cuda",
    "nearest_stock_distance_cuda",
]

CUDA_FINGERPRINT_THREADS = 128
CUDA_SIMILARITY_THREADS = 256

CUDA_CIRCULAR_SOURCE = r"""
#define CUDA_BUCKET_SIZE 64
#define ATOM_WORD_COUNT 14
#define BOND_KIND_MASK 0x0f
#define CIRCULAR_ATOM_FEATURE 0xA70F4A7D9E370003ull
#define WL_COMBINE_CONSTANT 0x517CC1B727220A95ull
#define SPLITMIX64_GAMMA 0x9E3779B97F4A7C15ull
#define SPLITMIX64_MUL1 0xBF58476D1CE4E5B9ull
#define SPLITMIX64_MUL2 0x94D049BB133111EBull

extern "C" {

__device__ __forceinline__ unsigned long long mix_uint64_circular(
    unsigned long long value
) {
    value += SPLITMIX64_GAMMA;
    value = (value ^ (value >> 30)) * SPLITMIX64_MUL1;
    value = (value ^ (value >> 27)) * SPLITMIX64_MUL2;
    return value ^ (value >> 31);
}

__device__ __forceinline__ unsigned long long combine_uint64_circular(
    unsigned long long left,
    unsigned long long right
) {
    return mix_uint64_circular(
        left ^ (right + WL_COMBINE_CONSTANT + (left << 6) + (left >> 2))
    );
}

__device__ __forceinline__ void set_circular_bit(
    unsigned long long* words,
    int n_bits,
    unsigned long long label
) {
    const unsigned long long bit = label % (unsigned long long)n_bits;
    atomicOr(words + (bit >> 6), 1ull << (bit & 63ull));
}

__global__ void circular_fingerprint_kernel(
    const unsigned int* atoms_global,
    const unsigned char* bonds_global,
    const unsigned long long* rows_global,
    const int* atom_counts,
    int molecule_count,
    int radius,
    int n_bits,
    unsigned long long* out
) {
    const int molecule = blockIdx.x;
    if (molecule >= molecule_count) return;
    const int atom = threadIdx.x;
    const int atom_count = atom_counts[molecule];
    const int word_count = n_bits / 64;
    const unsigned int* atoms = atoms_global
        + (size_t)molecule * CUDA_BUCKET_SIZE * ATOM_WORD_COUNT;
    const unsigned char* bonds = bonds_global
        + (size_t)molecule * CUDA_BUCKET_SIZE * CUDA_BUCKET_SIZE;
    const unsigned long long* rows = rows_global
        + (size_t)molecule * CUDA_BUCKET_SIZE;
    unsigned long long* words = out + (size_t)molecule * word_count;

    __shared__ unsigned long long previous[CUDA_BUCKET_SIZE];
    __shared__ unsigned long long next[CUDA_BUCKET_SIZE];

    if (atom < atom_count) {
        const unsigned int* packed = atoms + atom * ATOM_WORD_COUNT;
        unsigned long long label = CIRCULAR_ATOM_FEATURE;
        const unsigned long long z_lo =
            (unsigned long long)packed[0] | ((unsigned long long)packed[1] << 32);
        const unsigned long long z_hi =
            (unsigned long long)packed[2] | ((unsigned long long)packed[3] << 32);
        label = combine_uint64_circular(label, z_lo);
        label = combine_uint64_circular(label, z_hi);
        label = combine_uint64_circular(label, (unsigned long long)packed[4]);
        label = combine_uint64_circular(label, (unsigned long long)packed[5]);
        label = combine_uint64_circular(label, (unsigned long long)packed[7]);
        label = combine_uint64_circular(label, (unsigned long long)__popcll(rows[atom]));
        previous[atom] = label;
        set_circular_bit(words, n_bits, label);
    }
    __syncthreads();

    for (int round = 0; round < radius; ++round) {
        if (atom < atom_count) {
            unsigned long long aggregate = 0ull;
            unsigned long long neighbors = rows[atom];
            while (neighbors) {
                const int neighbor = __ffsll((long long)neighbors) - 1;
                neighbors &= neighbors - 1;
                const unsigned long long bond = (unsigned long long)(
                    bonds[atom * CUDA_BUCKET_SIZE + neighbor] & BOND_KIND_MASK
                );
                aggregate += combine_uint64_circular(previous[neighbor], bond);
            }
            const unsigned long long degree = (unsigned long long)__popcll(rows[atom]);
            next[atom] = mix_uint64_circular(
                combine_uint64_circular(
                    combine_uint64_circular(previous[atom], degree),
                    aggregate
                )
            );
        }
        __syncthreads();
        if (atom < atom_count) {
            previous[atom] = next[atom];
            set_circular_bit(words, n_bits, previous[atom]);
        }
        __syncthreads();
    }
}

__global__ void nearest_stock_distance_kernel(
    const unsigned long long* queries,
    const unsigned long long* stock,
    int query_count,
    int stock_count,
    int word_count,
    float weight,
    float* out
) {
    const int query = blockIdx.x;
    if (query >= query_count) return;
    const int lane = threadIdx.x;
    const unsigned long long* query_words = queries + (size_t)query * word_count;
    float best = 0.0f;
    for (int candidate = lane; candidate < stock_count; candidate += blockDim.x) {
        const unsigned long long* stock_words = stock + (size_t)candidate * word_count;
        unsigned int intersection = 0u;
        unsigned int union_count = 0u;
        for (int word = 0; word < word_count; ++word) {
            const unsigned long long left = query_words[word];
            const unsigned long long right = stock_words[word];
            intersection += (unsigned int)__popcll(left & right);
            union_count += (unsigned int)__popcll(left | right);
        }
        const float similarity = union_count
            ? (float)intersection / (float)union_count
            : 0.0f;
        best = fmaxf(best, similarity);
    }

    __shared__ float maxima[256];
    maxima[lane] = best;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (lane < stride) maxima[lane] = fmaxf(maxima[lane], maxima[lane + stride]);
        __syncthreads();
    }
    if (lane == 0) out[query] = weight * (1.0f - maxima[0]);
}

}
"""

CUDA_CIRCULAR_MODULE: Any | None = None


def circular_module(cp: Any) -> Any:
    global CUDA_CIRCULAR_MODULE
    if CUDA_CIRCULAR_MODULE is None:
        CUDA_CIRCULAR_MODULE = cp.RawModule(
            code=CUDA_CIRCULAR_SOURCE,
            options=("-std=c++11",),
        )
    return CUDA_CIRCULAR_MODULE


def compute_circular_fingerprints_cuda(
    batch: CudaMoleculeBatch,
    config: CircularFingerprintConfig = CircularFingerprintConfig(),
) -> Any:
    """Packed binary fingerprints for an already device-resident molecule batch."""
    cp = _require_cupy()
    count = len(batch)
    words = cp.zeros((count, config.n_bits // 64), dtype=cp.uint64)
    if not count:
        return words
    kernel = circular_module(cp).get_function("circular_fingerprint_kernel")
    kernel(
        (count,),
        (CUDA_FINGERPRINT_THREADS,),
        (
            cp.ascontiguousarray(batch.atoms_device).reshape(-1),
            cp.ascontiguousarray(batch.bonds_device).reshape(-1),
            cp.ascontiguousarray(batch.rows_device).reshape(-1),
            batch.atom_counts_device.astype(cp.int32, copy=False),
            np.int32(count),
            np.int32(config.radius),
            np.int32(config.n_bits),
            words.reshape(-1),
        ),
    )
    return words


def nearest_stock_distance_cuda(
    query_fingerprints_device: Any,
    stock_fingerprints_device: Any,
    *,
    weight: float,
) -> Any:
    """Exact nearest-stock binary Tanimoto distance, one float per query."""
    cp = _require_cupy()
    if weight < 0.0:
        raise ValueError(f"weight must be non-negative, got {weight!r}.")
    if query_fingerprints_device.ndim != 2 or stock_fingerprints_device.ndim != 2:
        raise ValueError("fingerprint arrays must be two-dimensional.")
    if (
        query_fingerprints_device.dtype != cp.uint64
        or stock_fingerprints_device.dtype != cp.uint64
    ):
        raise ValueError("fingerprint arrays must have dtype uint64.")
    if query_fingerprints_device.shape[1] != stock_fingerprints_device.shape[1]:
        raise ValueError("query and stock fingerprints must have equal widths.")
    if not stock_fingerprints_device.shape[0]:
        raise ValueError("stock fingerprints must not be empty.")
    query_count = int(query_fingerprints_device.shape[0])
    out = cp.empty(query_count, dtype=cp.float32)
    if not query_count:
        return out
    kernel = circular_module(cp).get_function("nearest_stock_distance_kernel")
    kernel(
        (query_count,),
        (CUDA_SIMILARITY_THREADS,),
        (
            cp.ascontiguousarray(query_fingerprints_device).reshape(-1),
            cp.ascontiguousarray(stock_fingerprints_device).reshape(-1),
            np.int32(query_count),
            np.int32(stock_fingerprints_device.shape[0]),
            np.int32(query_fingerprints_device.shape[1]),
            np.float32(weight),
            out,
        ),
    )
    return out

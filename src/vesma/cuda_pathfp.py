from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .cuda_prune import _require_cupy
from .pathfp import FoldedFingerprintConfig

__all__ = [
    "CudaPrefilterResult",
    "compact_folded_prefilter_pairs_cuda",
    "compute_folded_path_fingerprints_cuda",
]

FP_WORD_BITS = 64
CUDA_LINEAR_THREADS = 128
CUDA_PREFILTER_THREADS = 256

_CUDA_LINEAR_MODULE: Any | None = None
_CUDA_FINGERPRINT_KERNEL: Any | None = None
_CUDA_PREFILTER_KERNEL: Any | None = None

_CUDA_LINEAR_SOURCE = r"""
#define CUDA_BUCKET_SIZE 64
#define ATOM_WORD_COUNT 14
#define BOND_NONE 0
#define BOND_KIND_MASK 0x0f
#define BOND_SINGLE 1
#define BOND_DOUBLE 2
#define BOND_TRIPLE 4
#define BOND_AROMATIC 8
#define SINGLE_BOND_CLASS 1ull
#define DOUBLE_BOND_CLASS 2ull
#define TRIPLE_BOND_CLASS 3ull
#define FLEX_BOND_CLASS 4ull
#define ATOM_FEATURE 0xA70F4A7D9E370001ull
#define PATH_FEATURE 0xA70F4A7D9E370002ull
#define WL_COMBINE_CONSTANT 0x517CC1B727220A95ull
#define SPLITMIX64_GAMMA 0x9E3779B97F4A7C15ull
#define SPLITMIX64_MUL1 0xBF58476D1CE4E5B9ull
#define SPLITMIX64_MUL2 0x94D049BB133111EBull
#define FP_WORD_COUNT 32
#define FP_TILE_SIZE 64

extern "C" {

__device__ __forceinline__ unsigned long long mix_uint64_linear(
    unsigned long long value
) {
    value += SPLITMIX64_GAMMA;
    value = (value ^ (value >> 30)) * SPLITMIX64_MUL1;
    value = (value ^ (value >> 27)) * SPLITMIX64_MUL2;
    return value ^ (value >> 31);
}

__device__ __forceinline__ unsigned long long combine_uint64_linear(
    unsigned long long left,
    unsigned long long right
) {
    const unsigned long long combined =
        left ^ (right + WL_COMBINE_CONSTANT + (left << 6) + (left >> 2));
    return mix_uint64_linear(combined);
}

__device__ __forceinline__ int popcount_atom_z_words(
    const unsigned int atom[ATOM_WORD_COUNT]
) {
    return __popc(atom[0]) + __popc(atom[1]) + __popc(atom[2]) + __popc(atom[3]);
}

__device__ __forceinline__ int ctz_u32_linear(unsigned int value) {
    return __ffs((int)value) - 1;
}

__device__ int exact_single_z_from_atom(const unsigned int atom[ATOM_WORD_COUNT]) {
    if (popcount_atom_z_words(atom) != 1) return -1;
    if (atom[0] != 0u) return ctz_u32_linear(atom[0]);
    if (atom[1] != 0u) return ctz_u32_linear(atom[1]) + 32;
    if (atom[2] != 0u) return ctz_u32_linear(atom[2]) + 64;
    return ctz_u32_linear(atom[3]) + 96;
}

__device__ __forceinline__ unsigned long long atom_feature_linear(int z) {
    return combine_uint64_linear(ATOM_FEATURE, (unsigned long long)z);
}

__device__ unsigned long long path_hash_linear(
    const int labels[CUDA_BUCKET_SIZE],
    const int path[8],
    const unsigned long long edge_classes[7],
    int length,
    bool reverse
) {
    unsigned long long value = PATH_FEATURE;
    if (reverse) {
        for (int i = length; i >= 0; --i) {
            value = combine_uint64_linear(value, (unsigned long long)labels[path[i]]);
        }
        for (int i = length - 1; i >= 0; --i) {
            value = combine_uint64_linear(value, edge_classes[i]);
        }
    } else {
        for (int i = 0; i <= length; ++i) {
            value = combine_uint64_linear(value, (unsigned long long)labels[path[i]]);
        }
        for (int i = 0; i < length; ++i) {
            value = combine_uint64_linear(value, edge_classes[i]);
        }
    }
    return value;
}

__device__ __forceinline__ void set_fingerprint_bit(
    unsigned long long* out,
    int word_count,
    unsigned long long feature
) {
    const unsigned long long bit = feature % (unsigned long long)(word_count * 64);
    atomicOr(out + (bit >> 6), 1ull << (bit & 63ull));
}

// `query` is accepted and ignored, and SINGLE_/DOUBLE_/TRIPLE_BOND_CLASS are unreached.
// Faithful to `invariants.bond_feature_classes`, whose docstring explains the choice and why the
// prefilter stays subset-sound; do not "fix" it on one backend alone.
__device__ int bond_feature_classes_linear(
    unsigned char flags,
    bool query,
    unsigned long long classes[4]
) {
    if ((flags & BOND_KIND_MASK) == BOND_NONE) return 0;
    classes[0] = FLEX_BOND_CLASS;
    return 1;
}

__device__ void emit_path_feature_variants(
    unsigned long long* out,
    int word_count,
    unsigned char bonds[CUDA_BUCKET_SIZE][CUDA_BUCKET_SIZE],
    const int labels[CUDA_BUCKET_SIZE],
    const int path[8],
    int length,
    bool query
) {
    unsigned long long options[7][4];
    int option_counts[7];
    for (int i = 0; i <= length; ++i) {
        if (labels[path[i]] < 0) return;
    }
    for (int i = 0; i < length; ++i) {
        const int left = path[i];
        const int right = path[i + 1];
        option_counts[i] = bond_feature_classes_linear(
            bonds[left][right], query, options[i]
        );
        if (option_counts[i] == 0) return;
    }

    int choice[7];
    for (int i = 0; i < length; ++i) choice[i] = 0;
    while (true) {
        unsigned long long edge_classes[7];
        for (int i = 0; i < length; ++i) {
            edge_classes[i] = options[i][choice[i]];
        }
        const unsigned long long forward = path_hash_linear(
            labels, path, edge_classes, length, false
        );
        const unsigned long long reverse = path_hash_linear(
            labels, path, edge_classes, length, true
        );
        set_fingerprint_bit(out, word_count, forward < reverse ? forward : reverse);

        int pos = length - 1;
        while (pos >= 0) {
            choice[pos] += 1;
            if (choice[pos] < option_counts[pos]) break;
            choice[pos] = 0;
            pos -= 1;
        }
        if (pos < 0) break;
    }
}

__device__ void enumerate_paths_from_root(
    unsigned long long* out,
    int word_count,
    unsigned long long rows[CUDA_BUCKET_SIZE],
    unsigned char bonds[CUDA_BUCKET_SIZE][CUDA_BUCKET_SIZE],
    const int labels[CUDA_BUCKET_SIZE],
    int atom_count,
    int root,
    int max_length,
    bool query
) {
    int path[8];
    int next_neighbor[8];
    int depth = 0;
    unsigned long long visited = 1ull << root;
    path[0] = root;
    next_neighbor[0] = 0;

    while (depth >= 0) {
        if (depth == max_length || next_neighbor[depth] >= atom_count) {
            visited &= ~(1ull << path[depth]);
            depth -= 1;
            continue;
        }
        const int neighbor = next_neighbor[depth];
        next_neighbor[depth] += 1;
        if (((rows[path[depth]] >> neighbor) & 1ull) == 0ull) continue;
        if ((visited & (1ull << neighbor)) != 0ull) continue;

        depth += 1;
        path[depth] = neighbor;
        next_neighbor[depth] = 0;
        visited |= 1ull << neighbor;
        if (path[0] < path[depth]) {
            emit_path_feature_variants(
                out, word_count, bonds, labels, path, depth, query
            );
        }
    }
}

__global__ void folded_path_fingerprint_kernel(
    const unsigned int* atoms_global,
    const unsigned char* bonds_global,
    const unsigned long long* rows_global,
    const int* atom_counts,
    int molecule_count,
    int query,
    int max_length,
    int word_count,
    unsigned long long* out
) {
    const int mol_idx = blockIdx.x;
    if (mol_idx >= molecule_count) return;
    const int tid = threadIdx.x;
    const int atom_count = atom_counts[mol_idx];
    const unsigned int* atoms = atoms_global + (size_t)mol_idx * CUDA_BUCKET_SIZE * ATOM_WORD_COUNT;
    const unsigned char* bonds_flat = bonds_global + (size_t)mol_idx * CUDA_BUCKET_SIZE * CUDA_BUCKET_SIZE;
    const unsigned long long* rows_flat = rows_global + (size_t)mol_idx * CUDA_BUCKET_SIZE;
    unsigned long long* mol_out = out + (size_t)mol_idx * word_count;

    __shared__ unsigned int atom_words[CUDA_BUCKET_SIZE][ATOM_WORD_COUNT];
    __shared__ unsigned char bonds[CUDA_BUCKET_SIZE][CUDA_BUCKET_SIZE];
    __shared__ unsigned long long rows[CUDA_BUCKET_SIZE];
    __shared__ int labels[CUDA_BUCKET_SIZE];

    for (int idx = tid; idx < CUDA_BUCKET_SIZE * ATOM_WORD_COUNT; idx += blockDim.x) {
        atom_words[idx / ATOM_WORD_COUNT][idx % ATOM_WORD_COUNT] = atoms[idx];
    }
    for (int idx = tid; idx < CUDA_BUCKET_SIZE * CUDA_BUCKET_SIZE; idx += blockDim.x) {
        bonds[idx / CUDA_BUCKET_SIZE][idx % CUDA_BUCKET_SIZE] = bonds_flat[idx];
    }
    for (int row = tid; row < CUDA_BUCKET_SIZE; row += blockDim.x) {
        rows[row] = rows_flat[row];
    }
    __syncthreads();

    for (int row = tid; row < CUDA_BUCKET_SIZE; row += blockDim.x) {
        labels[row] = row < atom_count ? exact_single_z_from_atom(atom_words[row]) : -1;
    }
    __syncthreads();

    for (int atom = tid; atom < atom_count; atom += blockDim.x) {
        if (labels[atom] >= 0) {
            set_fingerprint_bit(mol_out, word_count, atom_feature_linear(labels[atom]));
        }
    }
    __syncthreads();

    const int bounded_max_length = max_length > 7 ? 7 : max_length;
    for (int root = tid; root < atom_count; root += blockDim.x) {
        enumerate_paths_from_root(
            mol_out, word_count, rows, bonds, labels, atom_count,
            root, bounded_max_length, query != 0
        );
    }
}

__global__ void compact_folded_prefilter_pairs_kernel(
    const unsigned long long* molecule_fps,
    const unsigned long long* pattern_fps,
    const int* molecule_atom_counts,
    const int* pattern_atom_counts,
    int molecule_count,
    int pattern_count,
    int word_count,
    int pattern_index_offset,
    int* pair_pattern_index,
    int* pair_molecule_index,
    unsigned long long* survivor_count
) {
    __shared__ unsigned long long s_molecules[FP_TILE_SIZE][FP_WORD_COUNT];
    __shared__ unsigned long long s_patterns[FP_TILE_SIZE][FP_WORD_COUNT];
    __shared__ int s_molecule_atom_counts[FP_TILE_SIZE];
    __shared__ int s_pattern_atom_counts[FP_TILE_SIZE];

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;

    const int molecule_base = blockIdx.x * FP_TILE_SIZE;
    const int pattern_base = blockIdx.y * FP_TILE_SIZE;

    for (int idx = threadIdx.x; idx < FP_TILE_SIZE * FP_WORD_COUNT; idx += blockDim.x) {
        const int slot = idx / FP_WORD_COUNT;
        const int word = idx % FP_WORD_COUNT;
        const int molecule_idx = molecule_base + slot;
        const int pattern_idx = pattern_base + slot;
        s_molecules[slot][word] = molecule_idx < molecule_count
            ? molecule_fps[(size_t)molecule_idx * word_count + word]
            : 0ull;
        s_patterns[slot][word] = pattern_idx < pattern_count
            ? pattern_fps[(size_t)pattern_idx * word_count + word]
            : 0ull;
    }
    for (int slot = threadIdx.x; slot < FP_TILE_SIZE; slot += blockDim.x) {
        const int molecule_idx = molecule_base + slot;
        const int pattern_idx = pattern_base + slot;
        s_molecule_atom_counts[slot] = molecule_idx < molecule_count
            ? molecule_atom_counts[molecule_idx]
            : -1;
        s_pattern_atom_counts[slot] = pattern_idx < pattern_count
            ? pattern_atom_counts[pattern_idx]
            : 2147483647;
    }
    __syncthreads();

    for (
        int pair_idx = warp_in_block;
        pair_idx < FP_TILE_SIZE * FP_TILE_SIZE;
        pair_idx += warps_per_block
    ) {
        const int molecule_slot = pair_idx % FP_TILE_SIZE;
        const int pattern_slot = pair_idx / FP_TILE_SIZE;
        const int molecule_idx = molecule_base + molecule_slot;
        const int pattern_idx = pattern_base + pattern_slot;
        bool local_match =
            molecule_idx < molecule_count &&
            pattern_idx < pattern_count &&
            s_pattern_atom_counts[pattern_slot] <= s_molecule_atom_counts[molecule_slot];
        if (local_match) {
            const unsigned long long pattern_word = s_patterns[pattern_slot][lane];
            const unsigned long long molecule_word = s_molecules[molecule_slot][lane];
            local_match = (molecule_word & pattern_word) == pattern_word;
        }
        const unsigned int vote = __ballot_sync(0xffffffffu, local_match);
        if (lane == 0 && vote == 0xffffffffu) {
            const unsigned long long out_idx = atomicAdd(survivor_count, 1ull);
            pair_pattern_index[out_idx] = pattern_index_offset + pattern_idx;
            pair_molecule_index[out_idx] = molecule_idx;
        }
    }
}

}
"""


@dataclass(frozen=True)
class CudaPrefilterResult:
    pair_pattern_index_device: Any
    pair_molecule_index_device: Any
    pair_count: int
    dense_pair_count: int


def cuda_linear_module(cp: Any) -> Any:
    global _CUDA_LINEAR_MODULE
    if _CUDA_LINEAR_MODULE is None:
        _CUDA_LINEAR_MODULE = cp.RawModule(
            code=_CUDA_LINEAR_SOURCE,
            options=("-std=c++11",),
        )
    return _CUDA_LINEAR_MODULE


def fingerprint_kernel(cp: Any) -> Any:
    global _CUDA_FINGERPRINT_KERNEL
    if _CUDA_FINGERPRINT_KERNEL is None:
        _CUDA_FINGERPRINT_KERNEL = cuda_linear_module(cp).get_function(
            "folded_path_fingerprint_kernel"
        )
    return _CUDA_FINGERPRINT_KERNEL


def prefilter_kernel(cp: Any) -> Any:
    global _CUDA_PREFILTER_KERNEL
    if _CUDA_PREFILTER_KERNEL is None:
        _CUDA_PREFILTER_KERNEL = cuda_linear_module(cp).get_function(
            "compact_folded_prefilter_pairs_kernel"
        )
    return _CUDA_PREFILTER_KERNEL


def compute_folded_path_fingerprints_cuda(
    batch: Any,
    *,
    query: bool,
    config: FoldedFingerprintConfig = FoldedFingerprintConfig(),
) -> Any:
    cp = _require_cupy()
    if config.n_bits % FP_WORD_BITS != 0:
        raise ValueError(
            "config.n_bits must be a multiple of 64 for CUDA fingerprints."
        )
    if config.max_length > 7:
        raise ValueError("CUDA folded path fingerprints support max_length <= 7.")
    count = int(batch.atom_counts_device.shape[0])
    word_count = config.n_bits // FP_WORD_BITS
    out = cp.zeros((count, word_count), dtype=cp.uint64)
    if count == 0:
        return out
    fingerprint_kernel(cp)(
        (count,),
        (CUDA_LINEAR_THREADS,),
        (
            cp.ascontiguousarray(batch.atoms_device).reshape(-1),
            cp.ascontiguousarray(batch.bonds_device).reshape(-1),
            cp.ascontiguousarray(batch.rows_device).reshape(-1),
            batch.atom_counts_device.astype(cp.int32, copy=False),
            np.int32(count),
            np.int32(1 if query else 0),
            np.int32(config.max_length),
            np.int32(word_count),
            out.reshape(-1),
        ),
    )
    return out


def compact_folded_prefilter_pairs_cuda(
    *,
    molecule_fingerprints_device: Any,
    pattern_fingerprints_device: Any,
    molecule_atom_counts_device: Any,
    pattern_atom_counts_device: Any,
    pattern_index_offset: int = 0,
) -> CudaPrefilterResult:
    cp = _require_cupy()
    molecule_count = int(molecule_fingerprints_device.shape[0])
    pattern_count = int(pattern_fingerprints_device.shape[0])
    if molecule_count == 0 or pattern_count == 0:
        return CudaPrefilterResult(
            pair_pattern_index_device=cp.empty(0, dtype=cp.int32),
            pair_molecule_index_device=cp.empty(0, dtype=cp.int32),
            pair_count=0,
            dense_pair_count=molecule_count * pattern_count,
        )
    if molecule_fingerprints_device.shape[1] != pattern_fingerprints_device.shape[1]:
        raise ValueError(
            "molecule and pattern fingerprints must have equal word counts."
        )

    dense_pair_count = molecule_count * pattern_count
    pair_pattern_index = cp.empty(dense_pair_count, dtype=cp.int32)
    pair_molecule_index = cp.empty(dense_pair_count, dtype=cp.int32)
    survivor_count_device = cp.zeros(1, dtype=cp.uint64)
    word_count = int(molecule_fingerprints_device.shape[1])
    if word_count != 32:
        raise ValueError("CUDA folded prefilter expects 2048-bit fingerprints.")
    warps_per_block = CUDA_PREFILTER_THREADS // 32
    del warps_per_block
    tile_size = 64
    block_count = (
        (molecule_count + tile_size - 1) // tile_size,
        (pattern_count + tile_size - 1) // tile_size,
    )
    prefilter_kernel(cp)(
        block_count,
        (CUDA_PREFILTER_THREADS,),
        (
            cp.ascontiguousarray(molecule_fingerprints_device).reshape(-1),
            cp.ascontiguousarray(pattern_fingerprints_device).reshape(-1),
            molecule_atom_counts_device.astype(cp.int32, copy=False),
            pattern_atom_counts_device.astype(cp.int32, copy=False),
            np.int32(molecule_count),
            np.int32(pattern_count),
            np.int32(word_count),
            np.int32(pattern_index_offset),
            pair_pattern_index,
            pair_molecule_index,
            survivor_count_device,
        ),
    )
    pair_count = int(survivor_count_device.get()[0])
    return CudaPrefilterResult(
        pair_pattern_index_device=pair_pattern_index[:pair_count],
        pair_molecule_index_device=pair_molecule_index[:pair_count],
        pair_count=pair_count,
        dense_pair_count=dense_pair_count,
    )

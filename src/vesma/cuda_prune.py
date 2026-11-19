from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .adjacency import bond_presence
from .gmol import GMol
from .gmol_atoms import AtomMask
from .gpu_atom_masks import pack_atom_masks
from .gpu_predicates import (
    GpuPredicateProgram,
    UnresolvedPredicateError,
    compile_predicate_program,
    pack_collection_atom_predicates,
)
from .limits import ATOM_MASK_WORD_COUNT
from .matcher import CandidateMatrix
from .predicates import compile_predicate_table
from .rings import annotate_bond_topology

PairVector = np.ndarray[
    tuple[int], np.dtype[np.int32]
]  # per-candidate-pair int32 array
StatusMatrix = np.ndarray[tuple[int, int], np.dtype[np.uint8]]  # per-pair status codes

__all__ = [
    "CudaMatchResult",
    "CudaPruneBatch",
    "CudaPruneResult",
    "CudaAssignmentResult",
    "CudaOutputCapacityError",
    "CudaPredicateProgram",
    "CudaUnavailableError",
    "UnresolvedPredicateError",
    "apply_cuda_predicates",
    "evaluate_cuda_predicate_roots",
    "match_fused_cuda",
    "match_fused_assignments_cuda",
    "prepare_cuda_prune_batch_from_device",
    "prepare_cuda_prune_batch_from_device_pairs",
    "prepare_cuda_prune_batch",
    "prepare_cuda_predicates",
    "prune_bitset_cuda",
]


_CUDA_BUCKET_SIZE = 64
_ATOM_WORD_COUNT = ATOM_MASK_WORD_COUNT
_MATCH_STATUS_NAMES = np.asarray(["no", "yes", "maybe"])
_CUDA_MODULE: Any | None = None
_BITSET_KERNEL: Any | None = None
_INITIALIZE_KERNEL: Any | None = None
_MATCH_KERNEL: Any | None = None
_CUDA_KERNEL_SOURCE = r"""
#define CUDA_BUCKET_SIZE 64
#define ATOM_WORD_COUNT 14
#define BOND_KIND_MASK 0x0fu
#define BOND_TOPOLOGY_MASK 0x30u

extern "C" {

__device__ __forceinline__ unsigned char bond_class(unsigned char flags) {
    return flags & BOND_KIND_MASK;
}

__device__ __forceinline__ bool atom_masks_overlap(
    const unsigned int* pattern_row,
    const unsigned int* molecule_row
) {
    const bool z_match =
        ((pattern_row[0] & molecule_row[0]) != 0u) ||
        ((pattern_row[1] & molecule_row[1]) != 0u) ||
        ((pattern_row[2] & molecule_row[2]) != 0u) ||
        ((pattern_row[3] & molecule_row[3]) != 0u);
    if (!z_match) {
        return false;
    }
    #pragma unroll
    for (int offset = 4; offset < ATOM_WORD_COUNT; ++offset) {
        const unsigned int p = pattern_row[offset];
        const unsigned int m = molecule_row[offset];
        if (offset == 13) {
            if ((p & m & 0xFFFFu) == 0u) return false;
            const unsigned int mass = p >> 16;
            if (mass != 0xFFFFu && mass != (m >> 16)) return false;
        } else if ((p & m) == 0u) {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ void build_candidate_rows(
    unsigned long long current_rows[CUDA_BUCKET_SIZE],
    const unsigned int pattern_atoms[CUDA_BUCKET_SIZE][ATOM_WORD_COUNT],
    const unsigned int molecule_atoms[CUDA_BUCKET_SIZE][ATOM_WORD_COUNT],
    const int pattern_atom_count,
    const int molecule_atom_count,
    const unsigned int* pattern_predicates,
    const unsigned long long* predicate_roots,
    const int predicate_stride,
    const int root_atom,
    const int tid,
    const int block_threads
) {
    for (int row = tid; row < CUDA_BUCKET_SIZE; row += block_threads) {
        unsigned long long row_bits = 0ull;
        if (row < pattern_atom_count) {
            for (int col = 0; col < molecule_atom_count; ++col) {
                if (atom_masks_overlap(pattern_atoms[row], molecule_atoms[col])) {
                    row_bits |= (1ull << col);
                }
            }
        }
        if (row < pattern_atom_count && predicate_stride != 0) {
            const unsigned int predicate = pattern_predicates[row];
            if (predicate != 0xFFFFFFFFu) row_bits &= predicate_roots[predicate];
        }
        if (row == 0 && root_atom >= 0) row_bits &= 1ull << root_atom;
        current_rows[row] = row_bits;
    }
}

__global__ void initialize_candidate_rows_pairs(
    const unsigned int* pattern_atoms_global,
    const unsigned int* molecule_atoms_global,
    const int* pattern_atom_counts,
    const int* molecule_atom_counts,
    const int* pair_pattern_index,
    const int* pair_molecule_index,
    const unsigned int* pattern_predicates_global,
    const unsigned long long* predicate_roots_global,
    const int predicate_stride,
    const int* pair_root_atoms,
    const int use_root_atoms,
    unsigned long long* output_rows
) {
    const int pair_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int pattern_idx = pair_pattern_index[pair_idx];
    const int molecule_idx = pair_molecule_index[pair_idx];
    const int pattern_atom_count = pattern_atom_counts[pattern_idx];
    const int molecule_atom_count = molecule_atom_counts[molecule_idx];
    const int pattern_atom_base = pattern_idx * CUDA_BUCKET_SIZE * ATOM_WORD_COUNT;
    const int molecule_atom_base = molecule_idx * CUDA_BUCKET_SIZE * ATOM_WORD_COUNT;
    const int output_base = pair_idx * CUDA_BUCKET_SIZE;

    __shared__ unsigned int pattern_atoms[CUDA_BUCKET_SIZE][ATOM_WORD_COUNT];
    __shared__ unsigned int molecule_atoms[CUDA_BUCKET_SIZE][ATOM_WORD_COUNT];
    __shared__ unsigned long long current_rows[CUDA_BUCKET_SIZE];

    for (int idx = tid; idx < CUDA_BUCKET_SIZE * ATOM_WORD_COUNT; idx += blockDim.x) {
        pattern_atoms[idx / ATOM_WORD_COUNT][idx % ATOM_WORD_COUNT] =
            pattern_atoms_global[pattern_atom_base + idx];
        molecule_atoms[idx / ATOM_WORD_COUNT][idx % ATOM_WORD_COUNT] =
            molecule_atoms_global[molecule_atom_base + idx];
    }
    __syncthreads();

    if (pattern_atom_count == 0 || molecule_atom_count == 0) {
        for (int row = tid; row < CUDA_BUCKET_SIZE; row += blockDim.x) {
            output_rows[output_base + row] = 0ull;
        }
        return;
    }

    build_candidate_rows(
        current_rows,
        pattern_atoms,
        molecule_atoms,
        pattern_atom_count,
        molecule_atom_count,
        predicate_stride != 0 ? pattern_predicates_global + pattern_idx * CUDA_BUCKET_SIZE : pattern_predicates_global,
        predicate_stride != 0 ? predicate_roots_global + molecule_idx * predicate_stride : predicate_roots_global,
        predicate_stride,
        use_root_atoms != 0 ? pair_root_atoms[pair_idx] : -1,
        tid,
        blockDim.x
    );
    __syncthreads();

    for (int row = tid; row < CUDA_BUCKET_SIZE; row += blockDim.x) {
        output_rows[output_base + row] = current_rows[row];
    }
}

__global__ void prune_bitset_pairs(
    const unsigned int* pattern_atoms_global,
    const unsigned int* molecule_atoms_global,
    const unsigned long long* pattern_rows_global,
    const unsigned long long* molecule_rows_global,
    const int* pattern_atom_counts,
    const int* molecule_atom_counts,
    const int* pair_pattern_index,
    const int* pair_molecule_index,
    const unsigned int* pattern_predicates_global,
    const unsigned long long* predicate_roots_global,
    const int predicate_stride,
    const int* pair_root_atoms,
    const int use_root_atoms,
    const int max_prune_rounds,
    unsigned long long* output_rows,
    int* prune_rounds_out,
    unsigned char* pruning_complete_out
) {
    const int pair_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int pattern_idx = pair_pattern_index[pair_idx];
    const int molecule_idx = pair_molecule_index[pair_idx];
    const int pattern_atom_count = pattern_atom_counts[pattern_idx];
    const int molecule_atom_count = molecule_atom_counts[molecule_idx];
    const int pattern_atom_base = pattern_idx * CUDA_BUCKET_SIZE * ATOM_WORD_COUNT;
    const int molecule_atom_base = molecule_idx * CUDA_BUCKET_SIZE * ATOM_WORD_COUNT;
    const int pattern_row_base = pattern_idx * CUDA_BUCKET_SIZE;
    const int molecule_row_base = molecule_idx * CUDA_BUCKET_SIZE;
    const int output_base = pair_idx * CUDA_BUCKET_SIZE;

    __shared__ unsigned int pattern_atoms[CUDA_BUCKET_SIZE][ATOM_WORD_COUNT];
    __shared__ unsigned int molecule_atoms[CUDA_BUCKET_SIZE][ATOM_WORD_COUNT];
    __shared__ unsigned long long current_rows[CUDA_BUCKET_SIZE];
    __shared__ unsigned long long next_rows[CUDA_BUCKET_SIZE];
    __shared__ unsigned long long pattern_rows[CUDA_BUCKET_SIZE];
    __shared__ unsigned long long molecule_rows[CUDA_BUCKET_SIZE];
    __shared__ int rounds;
    __shared__ int pruning_complete;
    __shared__ int stop;
    __shared__ int changed;
    __shared__ int unsupported;

    for (int idx = tid; idx < CUDA_BUCKET_SIZE * ATOM_WORD_COUNT; idx += blockDim.x) {
        pattern_atoms[idx / ATOM_WORD_COUNT][idx % ATOM_WORD_COUNT] =
            pattern_atoms_global[pattern_atom_base + idx];
        molecule_atoms[idx / ATOM_WORD_COUNT][idx % ATOM_WORD_COUNT] =
            molecule_atoms_global[molecule_atom_base + idx];
    }
    for (int row = tid; row < CUDA_BUCKET_SIZE; row += blockDim.x) {
        pattern_rows[row] = pattern_rows_global[pattern_row_base + row];
        molecule_rows[row] = molecule_rows_global[molecule_row_base + row];
        next_rows[row] = 0ull;
    }
    if (tid == 0) {
        rounds = 0;
        pruning_complete = 1;
        stop = 0;
        changed = 0;
        unsupported = 0;
    }
    __syncthreads();

    if (pattern_atom_count == 0 || molecule_atom_count == 0) {
        for (int row = tid; row < CUDA_BUCKET_SIZE; row += blockDim.x) {
            output_rows[output_base + row] = 0ull;
        }
        if (tid == 0) {
            prune_rounds_out[pair_idx] = 0;
            pruning_complete_out[pair_idx] = 1;
        }
        return;
    }

    build_candidate_rows(
        current_rows,
        pattern_atoms,
        molecule_atoms,
        pattern_atom_count,
        molecule_atom_count,
        predicate_stride != 0 ? pattern_predicates_global + pattern_idx * CUDA_BUCKET_SIZE : pattern_predicates_global,
        predicate_stride != 0 ? predicate_roots_global + molecule_idx * predicate_stride : predicate_roots_global,
        predicate_stride,
        use_root_atoms != 0 ? pair_root_atoms[pair_idx] : -1,
        tid,
        blockDim.x
    );
    __syncthreads();

    while (true) {
        if (tid == 0) {
            if (max_prune_rounds >= 0 && rounds >= max_prune_rounds) {
                pruning_complete = 0;
                stop = 1;
            } else {
                stop = 0;
            }
            changed = 0;
            unsupported = 0;
        }
        __syncthreads();
        if (stop != 0) {
            break;
        }

        for (int row = tid; row < pattern_atom_count; row += blockDim.x) {
            const unsigned long long row_mask = current_rows[row];
            unsigned long long refined = 0ull;
            unsigned long long bits = row_mask;

            while (bits != 0ull) {
                const unsigned long long bit = bits & (0ull - bits);
                const int col = __ffsll((long long)bit) - 1;

                if (col >= molecule_atom_count) {
                    bits ^= bit;
                    continue;
                }

                unsigned long long neighbors = pattern_rows[row];
                int supported = 1;
                while (neighbors != 0ull) {
                    const unsigned long long neighbor_bit = neighbors & (0ull - neighbors);
                    const int neighbor = __ffsll((long long)neighbor_bit) - 1;
                    if ((current_rows[neighbor] & molecule_rows[col]) == 0ull) {
                        supported = 0;
                        break;
                    }
                    neighbors ^= neighbor_bit;
                }
                if (supported != 0) {
                    refined |= bit;
                }
                bits ^= bit;
            }

            next_rows[row] = refined;
            if (refined != row_mask) {
                atomicOr(&changed, 1);
            }
            if (refined == 0ull) {
                atomicOr(&unsupported, 1);
            }
        }
        for (int row = pattern_atom_count + tid; row < CUDA_BUCKET_SIZE; row += blockDim.x) {
            next_rows[row] = 0ull;
        }
        __syncthreads();

        if (tid == 0) {
            rounds += 1;
            if (unsupported != 0 || changed == 0) {
                stop = 1;
            }
        }
        __syncthreads();

        for (int row = tid; row < CUDA_BUCKET_SIZE; row += blockDim.x) {
            current_rows[row] = next_rows[row];
        }
        __syncthreads();

        if (stop != 0) {
            break;
        }
    }

    for (int row = tid; row < CUDA_BUCKET_SIZE; row += blockDim.x) {
        output_rows[output_base + row] = current_rows[row];
    }
    if (tid == 0) {
        prune_rounds_out[pair_idx] = rounds;
        pruning_complete_out[pair_idx] = pruning_complete;
    }
}


__device__ __forceinline__ int nth_set_bit_index(
    unsigned long long mask,
    int ordinal
) {
    int seen = 0;
    while (mask != 0ull) {
        const unsigned long long bit = mask & (0ull - mask);
        if (seen == ordinal) {
            return __ffsll((long long)bit) - 1;
        }
        mask ^= bit;
        seen += 1;
    }
    return -1;
}

__device__ __forceinline__ bool assignment_valid_with_bonds(
    const int assignment[CUDA_BUCKET_SIZE],
    const unsigned long long pattern_rows[CUDA_BUCKET_SIZE],
    const unsigned long long molecule_rows[CUDA_BUCKET_SIZE],
    const unsigned char* pattern_bonds_global,
    const unsigned char* molecule_bonds_global,
    const unsigned int* pattern_groups,
    const unsigned long long comp_mask[CUDA_BUCKET_SIZE],
    const int has_groups,
    const int pattern_bond_base,
    const int molecule_bond_base,
    const int pattern_atom_count
) {
    unsigned long long used_bits = 0ull;
    for (int row = 0; row < pattern_atom_count; ++row) {
        const int col = assignment[row];
        if (col < 0) {
            return false;
        }
        const unsigned long long bit = 1ull << col;
        if ((used_bits & bit) != 0ull) {
            return false;
        }
        used_bits |= bit;
    }

    for (int row = 0; row < pattern_atom_count; ++row) {
        const int col = assignment[row];
        unsigned long long neighbors = pattern_rows[row];
        if (row < 63) {
            neighbors &= ~((1ull << (row + 1)) - 1ull);
        } else {
            neighbors = 0ull;
        }
        while (neighbors != 0ull) {
            const unsigned long long neighbor_bit = neighbors & (0ull - neighbors);
            const int neighbor = __ffsll((long long)neighbor_bit) - 1;
            const int neighbor_col = assignment[neighbor];
            if (neighbor_col < 0) {
                return false;
            }
            if ((molecule_rows[col] & (1ull << neighbor_col)) == 0ull) {
                return false;
            }

            const unsigned char pattern_flags =
                pattern_bonds_global[
                    pattern_bond_base + row * CUDA_BUCKET_SIZE + neighbor
                ];
            const unsigned char molecule_flags =
                molecule_bonds_global[
                    molecule_bond_base + col * CUDA_BUCKET_SIZE + neighbor_col
                ];
            // Query and molecule topology use the same allowed-state field:
            // 01 ring, 10 chain, 11 either. Compatibility requires both the
            // bond-class sets and topology-state sets to intersect. Zero query
            // flags mean that this dense matrix position has no bond constraint.
            bool compatible =
                pattern_flags == 0u
                || ((bond_class(pattern_flags) & bond_class(molecule_flags)) != 0u
                    && ((((unsigned int)pattern_flags) & BOND_TOPOLOGY_MASK)
                        & (((unsigned int)molecule_flags) & BOND_TOPOLOGY_MASK)) != 0u);
            if (!compatible) {
                return false;
            }
            neighbors ^= neighbor_bit;
        }
    }

    // Zero-level grouping: what the pattern's parentheses said about separate molecules.
    // Same non-zero group id, same component of the target; different ids, different
    // components. Group 0 says nothing and is skipped.
    if (has_groups != 0) {
        for (int row = 0; row < pattern_atom_count; ++row) {
            const unsigned int group = pattern_groups[row];
            if (group == 0u) continue;
            for (int other = row + 1; other < pattern_atom_count; ++other) {
                const unsigned int other_group = pattern_groups[other];
                if (other_group == 0u) continue;
                const bool together =
                    (comp_mask[assignment[row]] & (1ull << assignment[other])) != 0ull;
                if ((group == other_group) != together) {
                    return false;
                }
            }
        }
    }
    return true;
}

__global__ void match_fused_pairs(
    const unsigned int* pattern_atoms_global,
    const unsigned int* molecule_atoms_global,
    const unsigned long long* pattern_rows_global,
    const unsigned long long* molecule_rows_global,
    const unsigned char* pattern_bonds_global,
    const unsigned char* molecule_bonds_global,
    // Pattern side only. The molecule's components are a closure over the adjacency rows the
    // kernel already loads, so there is nothing to upload for them.
    const unsigned int* pattern_groups_global,
    const int* pattern_atom_counts,
    const int* molecule_atom_counts,
    const int* pair_pattern_index,
    const int* pair_molecule_index,
    const unsigned int* pattern_predicates_global,
    const unsigned long long* predicate_roots_global,
    const int predicate_stride,
    const int* pair_root_atoms,
    const int use_root_atoms,
    const int max_prune_rounds,
    const int max_leaf_checks,
    const int max_prefix_depth,
    const int collect_assignments,
    const unsigned long long assignment_capacity,
    unsigned char* status_out,
    int* assignment_counts_out,
    unsigned long long* assignment_required_out,
    int* assignment_pair_ids_out,
    int* assignments_out,
    unsigned long long* candidate_assignments_out,
    unsigned char* candidate_overflow_out,
    int* prune_rounds_out,
    unsigned char* pruning_complete_out,
    unsigned char* search_complete_out,
    int* leaf_checks_out,
    int* branches_explored_out
) {
    const int pair_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int pattern_idx = pair_pattern_index[pair_idx];
    const int molecule_idx = pair_molecule_index[pair_idx];
    const int pattern_atom_count = pattern_atom_counts[pattern_idx];
    const int molecule_atom_count = molecule_atom_counts[molecule_idx];
    const int pattern_atom_base = pattern_idx * CUDA_BUCKET_SIZE * ATOM_WORD_COUNT;
    const int molecule_atom_base = molecule_idx * CUDA_BUCKET_SIZE * ATOM_WORD_COUNT;
    const int pattern_row_base = pattern_idx * CUDA_BUCKET_SIZE;
    const int molecule_row_base = molecule_idx * CUDA_BUCKET_SIZE;
    const int pattern_bond_base = pattern_idx * CUDA_BUCKET_SIZE * CUDA_BUCKET_SIZE;
    const int molecule_bond_base = molecule_idx * CUDA_BUCKET_SIZE * CUDA_BUCKET_SIZE;
    __shared__ unsigned int pattern_atoms[CUDA_BUCKET_SIZE][ATOM_WORD_COUNT];
    __shared__ unsigned int molecule_atoms[CUDA_BUCKET_SIZE][ATOM_WORD_COUNT];
    __shared__ unsigned long long current_rows[CUDA_BUCKET_SIZE];
    __shared__ unsigned long long next_rows[CUDA_BUCKET_SIZE];
    __shared__ unsigned long long pattern_rows[CUDA_BUCKET_SIZE];
    __shared__ unsigned long long molecule_rows[CUDA_BUCKET_SIZE];
    __shared__ int row_counts[CUDA_BUCKET_SIZE];
    __shared__ int ambiguous_rows[CUDA_BUCKET_SIZE];
    __shared__ int fixed_assignment[CUDA_BUCKET_SIZE];
    __shared__ int rounds;
    __shared__ int pruning_complete;
    __shared__ int stop;
    __shared__ int changed;
    __shared__ int unsupported;
    __shared__ int ambiguous_count;
    __shared__ int singleton_conflict;
    __shared__ unsigned long long singleton_bits;
    __shared__ unsigned long long total_leaf_count;
    __shared__ unsigned long long candidate_assignments;
    __shared__ unsigned long long leaf_limit;
    __shared__ int assignment_count;
    __shared__ int search_complete;
    __shared__ int leaf_checks;
    __shared__ unsigned int status_code;
    __shared__ int candidate_overflow;
    __shared__ unsigned long long chunk_output_base;
    // Which atoms of the target share a molecule with each atom, as a bitset per atom.
    __shared__ unsigned long long comp_mask[CUDA_BUCKET_SIZE];
    __shared__ int has_groups;
    const unsigned int* pattern_groups = pattern_groups_global + pattern_row_base;

    for (int idx = tid; idx < CUDA_BUCKET_SIZE * ATOM_WORD_COUNT; idx += blockDim.x) {
        pattern_atoms[idx / ATOM_WORD_COUNT][idx % ATOM_WORD_COUNT] =
            pattern_atoms_global[pattern_atom_base + idx];
        molecule_atoms[idx / ATOM_WORD_COUNT][idx % ATOM_WORD_COUNT] =
            molecule_atoms_global[molecule_atom_base + idx];
    }
    for (int row = tid; row < CUDA_BUCKET_SIZE; row += blockDim.x) {
        pattern_rows[row] = pattern_rows_global[pattern_row_base + row];
        molecule_rows[row] = molecule_rows_global[molecule_row_base + row];
        next_rows[row] = 0ull;
        row_counts[row] = 0;
        ambiguous_rows[row] = -1;
        fixed_assignment[row] = -1;
    }
    if (tid == 0) {
        rounds = 0;
        pruning_complete = 1;
        stop = 0;
        changed = 0;
        unsupported = 0;
        ambiguous_count = 0;
        singleton_conflict = 0;
        singleton_bits = 0ull;
        total_leaf_count = 0ull;
        candidate_assignments = 0ull;
        leaf_limit = 0ull;
        assignment_count = 0;
        search_complete = 1;
        leaf_checks = 0;
        status_code = 2u;
        candidate_overflow = 0;
        chunk_output_base = 0ull;
    }
    __syncthreads();

    // Components of the target, but only if the pattern asks about them. One thread, because
    // the closure reads the array it is growing; a pattern with no group skips it entirely.
    if (tid == 0) {
        has_groups = 0;
        for (int row = 0; row < pattern_atom_count; ++row) {
            if (pattern_groups[row] != 0u) { has_groups = 1; break; }
        }
        if (has_groups != 0) {
            for (int atom = 0; atom < molecule_atom_count; ++atom) {
                comp_mask[atom] = molecule_rows[atom] | (1ull << atom);
            }
            bool grew = true;
            while (grew) {
                grew = false;
                for (int atom = 0; atom < molecule_atom_count; ++atom) {
                    unsigned long long reached = comp_mask[atom];
                    unsigned long long grown = reached;
                    while (reached != 0ull) {
                        const unsigned long long bit = reached & (0ull - reached);
                        grown |= comp_mask[__ffsll((long long)bit) - 1];
                        reached ^= bit;
                    }
                    if (grown != comp_mask[atom]) { comp_mask[atom] = grown; grew = true; }
                }
            }
        }
    }
    __syncthreads();

    if (pattern_atom_count == 0) {
        if (tid == 0) {
            if (collect_assignments != 0) {
                const unsigned long long output_index =
                    atomicAdd(assignment_required_out, 1ull);
                if (output_index < assignment_capacity) {
                    assignment_pair_ids_out[output_index] = pair_idx;
                    for (int row = 0; row < CUDA_BUCKET_SIZE; ++row) {
                        assignments_out[output_index * CUDA_BUCKET_SIZE + row] = -1;
                    }
                }
                assignment_counts_out[pair_idx] = 1;
                candidate_assignments_out[pair_idx] = 1ull;
                candidate_overflow_out[pair_idx] = 0u;
            }
            status_out[pair_idx] = 1u;
            prune_rounds_out[pair_idx] = 0;
            pruning_complete_out[pair_idx] = 1u;
            search_complete_out[pair_idx] = 1u;
            leaf_checks_out[pair_idx] = 1;
            branches_explored_out[pair_idx] = 0;
        }
        return;
    }

    if (molecule_atom_count == 0 || pattern_atom_count > molecule_atom_count) {
        if (tid == 0) {
            if (collect_assignments != 0) {
                assignment_counts_out[pair_idx] = 0;
                candidate_assignments_out[pair_idx] = 0ull;
                candidate_overflow_out[pair_idx] = 0u;
            }
            status_out[pair_idx] = 0u;
            prune_rounds_out[pair_idx] = 0;
            pruning_complete_out[pair_idx] = 1u;
            search_complete_out[pair_idx] = 1u;
            leaf_checks_out[pair_idx] = 0;
            branches_explored_out[pair_idx] = 0;
        }
        return;
    }

    build_candidate_rows(
        current_rows,
        pattern_atoms,
        molecule_atoms,
        pattern_atom_count,
        molecule_atom_count,
        predicate_stride != 0 ? pattern_predicates_global + pattern_idx * CUDA_BUCKET_SIZE : pattern_predicates_global,
        predicate_stride != 0 ? predicate_roots_global + molecule_idx * predicate_stride : predicate_roots_global,
        predicate_stride,
        use_root_atoms != 0 ? pair_root_atoms[pair_idx] : -1,
        tid,
        blockDim.x
    );
    __syncthreads();

    __shared__ unsigned long long stack_rows[16][CUDA_BUCKET_SIZE];
    __shared__ int stack_branch_row[16];
    __shared__ unsigned long long stack_branch_cols[16];
    __shared__ int prefix_depth;
    __shared__ int branches_explored;
    __shared__ int dfs_stop;
    __shared__ int save_state;
    __shared__ int do_leaf_eval;

    if (tid == 0) {
        prefix_depth = 0;
        branches_explored = 0;
        dfs_stop = 0;
        save_state = 0;
        status_code = 2u;
        for (int i = 0; i < 16; ++i) {
            stack_branch_cols[i] = 0ull;
        }
    }
    __syncthreads();

    while (true) {
        if (tid == 0) {
            rounds = 0;
        }
        __syncthreads();

        while (true) {
            if (tid == 0) {
                if (max_prune_rounds >= 0 && rounds >= max_prune_rounds) {
                    pruning_complete = 0;
                    stop = 1;
                } else {
                    stop = 0;
                }
                changed = 0;
                unsupported = 0;
            }
            __syncthreads();
            if (stop != 0) break;

            for (int row = tid; row < pattern_atom_count; row += blockDim.x) {
                const unsigned long long row_mask = current_rows[row];
                unsigned long long refined = 0ull;
                unsigned long long bits = row_mask;

                while (bits != 0ull) {
                    const unsigned long long bit = bits & (0ull - bits);
                    const int col = __ffsll((long long)bit) - 1;
                    if (col >= molecule_atom_count) {
                        bits ^= bit;
                        continue;
                    }
                    unsigned long long neighbors = pattern_rows[row];
                    int supported = 1;
                    while (neighbors != 0ull) {
                        const unsigned long long neighbor_bit = neighbors & (0ull - neighbors);
                        const int neighbor = __ffsll((long long)neighbor_bit) - 1;
                        if ((current_rows[neighbor] & molecule_rows[col]) == 0ull) {
                            supported = 0;
                            break;
                        }
                        neighbors ^= neighbor_bit;
                    }
                    if (supported != 0) {
                        refined |= bit;
                    }
                    bits ^= bit;
                }
                next_rows[row] = refined;
                if (refined != row_mask) atomicOr(&changed, 1);
                if (refined == 0ull) atomicOr(&unsupported, 1);
            }
            for (int row = pattern_atom_count + tid; row < CUDA_BUCKET_SIZE; row += blockDim.x) {
                next_rows[row] = 0ull;
            }
            __syncthreads();

            if (tid == 0) {
                rounds += 1;
                if (unsupported != 0 || changed == 0) {
                    stop = 1;
                }
            }
            __syncthreads();

            for (int row = tid; row < CUDA_BUCKET_SIZE; row += blockDim.x) {
                current_rows[row] = next_rows[row];
            }
            __syncthreads();
            if (stop != 0) break;
        }

        if (tid == 0) {
            save_state = 0;
            do_leaf_eval = 0;
            if (unsupported != 0) {
                status_code = 0u;
            } else {
                total_leaf_count = 1ull;
                candidate_assignments = 1ull;
                candidate_overflow = 0;
                singleton_bits = 0ull;
                singleton_conflict = 0;
                ambiguous_count = 0;
                for (int row = 0; row < pattern_atom_count; ++row) {
                    const int count = __popcll(current_rows[row]);
                    row_counts[row] = count;
                    if (count == 0) {
                        status_code = 0u;
                        candidate_assignments = 0ull;
                        break;
                    }
                    if (candidate_assignments > (~0ull / (unsigned long long)count)) {
                        candidate_assignments = ~0ull;
                        candidate_overflow = 1;
                    } else {
                        candidate_assignments *= (unsigned long long)count;
                    }
                    if (count == 1) {
                        const int col = __ffsll((long long)current_rows[row]) - 1;
                        const unsigned long long bit = 1ull << col;
                        fixed_assignment[row] = col;
                        if ((singleton_bits & bit) != 0ull) {
                            singleton_conflict = 1;
                        }
                        singleton_bits |= bit;
                    } else {
                        fixed_assignment[row] = -1;
                        ambiguous_rows[ambiguous_count] = row;
                        ambiguous_count += 1;
                        if (total_leaf_count <= ((unsigned long long)max_leaf_checks / (unsigned long long)count)) {
                            total_leaf_count *= (unsigned long long)count;
                        } else {
                            total_leaf_count = (unsigned long long)max_leaf_checks + 1ull;
                        }
                    }
                }
                if (status_code != 0u && singleton_conflict != 0) {
                    status_code = 0u;
                }
                if (status_code != 0u && ambiguous_count == 0) {
                    leaf_checks = 1;
                    int assignment[CUDA_BUCKET_SIZE];
                    for (int row = 0; row < pattern_atom_count; ++row) {
                        assignment[row] = fixed_assignment[row];
                    }
                    if (assignment_valid_with_bonds(
                        assignment, pattern_rows, molecule_rows,
                        pattern_bonds_global, molecule_bonds_global,
                        pattern_groups, comp_mask, has_groups,
                        pattern_bond_base, molecule_bond_base, pattern_atom_count
                    )) {
                        if (tid == 0) {
                            status_code = 1u;
                            assignment_count += 1;
                            if (collect_assignments != 0) {
                                const unsigned long long output_index = atomicAdd(assignment_required_out, 1ull);
                                if (output_index < assignment_capacity) {
                                    assignment_pair_ids_out[output_index] = pair_idx;
                                    for (int row = 0; row < CUDA_BUCKET_SIZE; ++row) {
                                        assignments_out[output_index * CUDA_BUCKET_SIZE + row] = row < pattern_atom_count ? assignment[row] : -1;
                                    }
                                }
                                status_code = 0u; // FORCE BACKTRACK
                            }
                        }
                    } else {
                        if (tid == 0) {
                            status_code = 0u;
                        }
                    }
                } else if (status_code != 0u) {
                    if (prefix_depth < max_prefix_depth && prefix_depth < 16) {
                        int branch_row = ambiguous_rows[0];
                        int min_count = row_counts[branch_row];
                        for (int i = 1; i < ambiguous_count; ++i) {
                            int r = ambiguous_rows[i];
                            if (row_counts[r] < min_count) {
                                min_count = row_counts[r];
                                branch_row = r;
                            }
                        }
                        stack_branch_row[prefix_depth] = branch_row;
                        stack_branch_cols[prefix_depth] = current_rows[branch_row];
                        save_state = 1;
                        status_code = 2u;
                    } else {
                        leaf_limit = total_leaf_count;
                        do_leaf_eval = 1;
                    }
                }
            }
            if (status_code == 0u) {
                if (prefix_depth == 0) {
                    dfs_stop = 1;
                }
            } else if (status_code == 1u) {
                dfs_stop = 1;
            }
        }
        __syncthreads();

        if (dfs_stop != 0) break;

        if (save_state != 0) {
            for (int row = tid; row < CUDA_BUCKET_SIZE; row += blockDim.x) {
                stack_rows[prefix_depth][row] = current_rows[row];
            }
        }
        __syncthreads();

        if (do_leaf_eval != 0) {
            unsigned long long remaining_leaves = ((unsigned long long)max_leaf_checks > (unsigned long long)leaf_checks) ? (unsigned long long)max_leaf_checks - (unsigned long long)leaf_checks : 0ull;
            unsigned long long chunk_limit = leaf_limit;
            if (chunk_limit > remaining_leaves) {
                chunk_limit = remaining_leaves;
            }
            if (tid == 0 && chunk_limit < leaf_limit) {
                search_complete = 0;
            }

            for (unsigned long long chunk_start = 0ull; chunk_start < chunk_limit; chunk_start += (unsigned long long)blockDim.x) {
                const unsigned long long leaf_index = chunk_start + (unsigned long long)tid;
                int assignment[CUDA_BUCKET_SIZE];
                bool valid = leaf_index < chunk_limit;
                if (valid) {
                    for (int row = 0; row < pattern_atom_count; ++row) {
                        assignment[row] = fixed_assignment[row];
                    }
                    unsigned long long used_bits = singleton_bits;
                    unsigned long long remainder = leaf_index;
                    for (int depth = ambiguous_count - 1; depth >= 0; --depth) {
                        const int row = ambiguous_rows[depth];
                        const int count = row_counts[row];
                        const int ordinal = (int)(remainder % (unsigned long long)count);
                        remainder /= (unsigned long long)count;
                        const int col = nth_set_bit_index(current_rows[row], ordinal);
                        if (col < 0) {
                            valid = false;
                            break;
                        }
                        const unsigned long long bit = 1ull << col;
                        if ((used_bits & bit) != 0ull) {
                            valid = false;
                            break;
                        }
                        used_bits |= bit;
                        assignment[row] = col;
                    }
                    valid = valid && assignment_valid_with_bonds(
                            assignment, pattern_rows, molecule_rows,
                            pattern_bonds_global, molecule_bonds_global,
                            pattern_groups, comp_mask, has_groups,
                            pattern_bond_base, molecule_bond_base, pattern_atom_count);
                }

                const unsigned int valid_mask = __ballot_sync(0xffffffffu, valid);
                const int valid_count = __popc(valid_mask);
                const int lane = tid & 31;
                if (lane == 0 && valid_count > 0 && collect_assignments != 0) {
                    chunk_output_base = atomicAdd(assignment_required_out, (unsigned long long)valid_count);
                    atomicAdd(&assignment_count, valid_count);
                }
                __syncthreads();

                if (collect_assignments != 0 && valid_count > 0 && valid) {
                    if (chunk_output_base < assignment_capacity) {
                        const unsigned int lower_lanes = lane == 0 ? 0u : ((1u << lane) - 1u);
                        const unsigned long long local_index = __popc(valid_mask & lower_lanes);
                        const unsigned long long output_index = chunk_output_base + local_index;
                        if (output_index < assignment_capacity) {
                            assignment_pair_ids_out[output_index] = pair_idx;
                            for (int row = 0; row < CUDA_BUCKET_SIZE; ++row) {
                                assignments_out[output_index * CUDA_BUCKET_SIZE + row] =
                                    (row < pattern_atom_count) ? assignment[row] : -1;
                            }
                        }
                    }
                }
                __syncthreads();

                if (tid == 0) {
                    if (collect_assignments == 0 && valid_count > 0) {
                        assignment_count = 1;
                        status_code = 1u;
                        search_complete = 0;
                    }
                }
                __syncthreads();
                if (collect_assignments == 0 && assignment_count > 0) break;
            }

            if (tid == 0) {
                unsigned long long chunk_end = chunk_limit;
                leaf_checks += (int)chunk_end;

                if (assignment_count > 0) {
                    status_code = 1u;
                } else if (chunk_limit < leaf_limit) {
                    status_code = 2u;
                } else {
                    status_code = 0u; // triggers backtrack
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            if (status_code == 1u && collect_assignments == 0) {
                dfs_stop = 1;
            } else {
                while (prefix_depth >= 0 && stack_branch_cols[prefix_depth] == 0ull) {
                    prefix_depth -= 1;
                }
                if (prefix_depth < 0) {
                    status_code = (search_complete == 0 && assignment_count == 0) ? 2u : 0u;
                    dfs_stop = 1;
                } else {
                    branches_explored += 1;
                    status_code = 2u;
                }
            }
        }
        __syncthreads();

        if (dfs_stop != 0) break;

        for (int row = tid; row < CUDA_BUCKET_SIZE; row += blockDim.x) {
            current_rows[row] = stack_rows[prefix_depth][row];
        }
        __syncthreads();

        if (tid == 0) {
            int branch_row = stack_branch_row[prefix_depth];
            unsigned long long bit = stack_branch_cols[prefix_depth] & (~stack_branch_cols[prefix_depth] + 1ull);
            current_rows[branch_row] = bit;
            stack_branch_cols[prefix_depth] ^= bit;
            prefix_depth += 1;
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (assignment_count > 0) {
            status_code = 1u;
        } else if (search_complete == 0) {
            status_code = 2u;
        }
        status_out[pair_idx] = (unsigned char)status_code;
        if (collect_assignments != 0) {
            assignment_counts_out[pair_idx] = assignment_count;
            unsigned long long reported_candidate_assignments = candidate_assignments;
            if (reported_candidate_assignments < (unsigned long long)assignment_count) {
                reported_candidate_assignments = (unsigned long long)assignment_count;
            }
            candidate_assignments_out[pair_idx] = reported_candidate_assignments;
            candidate_overflow_out[pair_idx] = (unsigned char)candidate_overflow;
        }
        prune_rounds_out[pair_idx] = rounds;
        pruning_complete_out[pair_idx] = (unsigned char)pruning_complete;
        search_complete_out[pair_idx] = (unsigned char)search_complete;
        leaf_checks_out[pair_idx] = leaf_checks;
        branches_explored_out[pair_idx] = branches_explored;
    }
}


__global__ void predicate_probe_bits(
    const unsigned int* probes, const unsigned int* molecule_atoms,
    const int* molecule_counts, unsigned long long* output,
    const int molecule_count, const int probe_count
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= molecule_count * probe_count) return;
    const int molecule = i / probe_count;
    const int probe = i % probe_count;
    unsigned long long bits = 0ull;
    for (int atom = 0; atom < molecule_counts[molecule]; ++atom) {
        if (atom_masks_overlap(probes + probe * ATOM_WORD_COUNT,
                molecule_atoms + (molecule * CUDA_BUCKET_SIZE + atom) * ATOM_WORD_COUNT)) {
            bits |= 1ull << atom;
        }
    }
    output[i] = bits;
}

__global__ void predicate_eval_residuals(
    const unsigned int* clauses, const unsigned int* terms,
    const unsigned int* factor_start, const unsigned int* factors,
    const unsigned long long* probes, const unsigned long long* sub_roots,
    unsigned long long* roots, const int* molecule_counts,
    const int molecule_count, const int first, const int count,
    const int residual_count, const int body_count, const int probe_count
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= molecule_count * count) return;
    const int molecule = i / count;
    const int residual = first + i % count;
    const int atoms = molecule_counts[molecule];
    const unsigned long long valid = atoms >= 64 ? ~0ull : ((1ull << atoms) - 1ull);
    unsigned long long held = valid;
    for (unsigned int c = clauses[residual]; c < clauses[residual + 1]; ++c) {
        unsigned long long any = 0ull;
        for (unsigned int t = terms[c]; t < terms[c + 1]; ++t) {
            unsigned long long all = valid;
            for (unsigned int f = factor_start[t]; f < factor_start[t + 1]; ++f) {
                const unsigned int factor = factors[f];
                const unsigned int index = factor & 0x3FFFFFFFu;
                unsigned long long value = (factor & 0x40000000u) != 0u
                    ? sub_roots[molecule * body_count + index]
                    : probes[molecule * probe_count + index];
                if ((factor & 0x80000000u) != 0u) value = ~value & valid;
                all &= value;
            }
            any |= all;
        }
        held &= any;
    }
    roots[molecule * residual_count + residual] = held;
}

__global__ void predicate_fold_roots(
    const unsigned char* status, const unsigned char* search_complete,
    const int* bodies, const int* molecules, const int* root_atoms,
    unsigned long long* roots, unsigned int* unresolved,
    const int count, const int body_count
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (status[i] == 1u) {
        atomicOr(roots + molecules[i] * body_count + bodies[i], 1ull << root_atoms[i]);
    } else if (status[i] == 2u || search_complete[i] == 0u) {
        atomicAdd(unresolved, 1u);
    }
}

}
"""


class CudaUnavailableError(ValueError):
    pass


class CudaOutputCapacityError(ValueError):
    def __init__(self, stage: str, required: int, capacity_limit: int) -> None:
        self.stage = stage
        self.required = required
        self.capacity_limit = capacity_limit
        super().__init__(
            f"CUDA {stage} output requires {required} rows, exceeding the "
            f"configured capacity limit of {capacity_limit}."
        )


@dataclass(frozen=True)
class CudaPruneBatch:
    pattern_count: int
    molecule_count: int
    pair_count: int
    pair_pattern_atom_counts: PairVector
    pair_molecule_atom_counts: PairVector
    pattern_atom_counts_device: Any
    molecule_atom_counts_device: Any
    pair_pattern_index_device: Any
    pair_molecule_index_device: Any
    pattern_atoms_u32: Any
    molecule_atoms_u32: Any
    pattern_rows_u64: Any
    molecule_rows_u64: Any
    pattern_bonds_u8: Any
    molecule_bonds_u8: Any
    # Pattern side only: a target's components are derived in the kernel from the adjacency
    # rows it already loads, so there is nothing to upload for the molecule side.
    pattern_groups_u32: Any
    pattern_predicates_u32: Any | None = None
    predicate_roots_u64: Any | None = None
    predicate_count: int = 0
    pair_root_atoms_i32: Any | None = None


@dataclass(frozen=True)
class CudaPruneResult:
    pair_pattern_atom_counts: PairVector
    pair_molecule_atom_counts: PairVector
    candidates_device: Any
    prune_rounds_device: Any
    pruning_complete_device: Any

    def candidates_host(self) -> list[CandidateMatrix]:
        padded = _to_numpy(self.candidates_device)
        return [
            padded[idx, :pattern_atoms, :molecule_atoms].copy()
            for idx, (pattern_atoms, molecule_atoms) in enumerate(
                zip(
                    self.pair_pattern_atom_counts,
                    self.pair_molecule_atom_counts,
                    strict=True,
                )
            )
        ]

    def prune_rounds_host(self) -> PairVector:
        return _to_numpy(self.prune_rounds_device).astype(np.int32, copy=False)

    def pruning_complete_host(self) -> AtomMask:
        return _to_numpy(self.pruning_complete_device).astype(bool, copy=False)


@dataclass(frozen=True)
class CudaMatchResult:
    pattern_count: int
    molecule_count: int
    status_codes_device: Any
    prune_rounds_device: Any
    pruning_complete_device: Any
    search_complete_device: Any
    leaf_checks_device: Any
    branches_explored_device: Any

    def status_codes_host(self) -> StatusMatrix:
        return _reshape_pair_host(
            self.status_codes_device,
            self.pattern_count,
            self.molecule_count,
            dtype=np.uint8,
        )

    def status_labels_host(self) -> np.ndarray:
        return _MATCH_STATUS_NAMES[self.status_codes_host()]

    def prune_rounds_host(self) -> np.ndarray:
        return _reshape_pair_host(
            self.prune_rounds_device,
            self.pattern_count,
            self.molecule_count,
            dtype=np.int32,
        )

    def pruning_complete_host(self) -> np.ndarray:
        return _reshape_pair_host(
            self.pruning_complete_device,
            self.pattern_count,
            self.molecule_count,
            dtype=bool,
        )

    def search_complete_host(self) -> np.ndarray:
        return _reshape_pair_host(
            self.search_complete_device,
            self.pattern_count,
            self.molecule_count,
            dtype=bool,
        )

    def leaf_checks_host(self) -> np.ndarray:
        return _reshape_pair_host(
            self.leaf_checks_device,
            self.pattern_count,
            self.molecule_count,
            dtype=np.int32,
        )

    def branches_explored_host(self) -> np.ndarray:
        return _reshape_pair_host(
            self.branches_explored_device,
            self.pattern_count,
            self.molecule_count,
            dtype=np.int32,
        )


@dataclass(frozen=True)
class CudaAssignmentResult:
    pattern_count: int
    molecule_count: int
    assignment_capacity: int
    assignment_required: int
    overflow_retries: int
    status_codes_device: Any
    assignment_counts_device: Any
    assignment_pair_ids_device: Any
    assignments_device: Any
    candidate_assignments_device: Any
    candidate_overflow_device: Any
    prune_rounds_device: Any
    pruning_complete_device: Any
    search_complete_device: Any
    leaf_checks_device: Any
    branches_explored_device: Any

    def status_codes_host(self) -> StatusMatrix:
        return _reshape_pair_host(
            self.status_codes_device,
            self.pattern_count,
            self.molecule_count,
            dtype=np.uint8,
        )

    def status_labels_host(self) -> np.ndarray:
        return _MATCH_STATUS_NAMES[self.status_codes_host()]

    def assignment_counts_host(self) -> np.ndarray:
        return _reshape_pair_host(
            self.assignment_counts_device,
            self.pattern_count,
            self.molecule_count,
            dtype=np.int32,
        )

    def assignments_host(self) -> np.ndarray:
        return _to_numpy(self.assignments_device).astype(np.int32, copy=False)

    def assignment_pair_ids_host(self) -> np.ndarray:
        return _to_numpy(self.assignment_pair_ids_device).astype(
            np.int32,
            copy=False,
        )

    def candidate_assignments_host(self) -> np.ndarray:
        return _reshape_pair_host(
            self.candidate_assignments_device,
            self.pattern_count,
            self.molecule_count,
            dtype=np.uint64,
        )

    def candidate_overflow_host(self) -> np.ndarray:
        return _reshape_pair_host(
            self.candidate_overflow_device,
            self.pattern_count,
            self.molecule_count,
            dtype=bool,
        )

    def prune_rounds_host(self) -> np.ndarray:
        return _reshape_pair_host(
            self.prune_rounds_device,
            self.pattern_count,
            self.molecule_count,
            dtype=np.int32,
        )

    def pruning_complete_host(self) -> np.ndarray:
        return _reshape_pair_host(
            self.pruning_complete_device,
            self.pattern_count,
            self.molecule_count,
            dtype=bool,
        )

    def search_complete_host(self) -> np.ndarray:
        return _reshape_pair_host(
            self.search_complete_device,
            self.pattern_count,
            self.molecule_count,
            dtype=bool,
        )

    def leaf_checks_host(self) -> np.ndarray:
        return _reshape_pair_host(
            self.leaf_checks_device,
            self.pattern_count,
            self.molecule_count,
            dtype=np.int32,
        )

    def branches_explored_host(self) -> np.ndarray:
        return _reshape_pair_host(
            self.branches_explored_device,
            self.pattern_count,
            self.molecule_count,
            dtype=np.int32,
        )


def prepare_cuda_prune_batch(
    patterns: list[GMol],
    molecules: list[GMol],
    *,
    max_prune_rounds: int = 64,
    max_leaf_checks: int = 4096,
    max_prefix_depth: int = 16,
) -> CudaPruneBatch:
    cp = _require_cupy()
    _validate_graph_sizes(patterns, molecules)

    pattern_count = len(patterns)
    molecule_count = len(molecules)
    pair_pattern_index, pair_molecule_index = _pair_indices(
        pattern_count,
        molecule_count,
    )
    pair_pattern_atom_counts, pair_molecule_atom_counts = _pair_atom_counts(
        patterns,
        molecules,
        pair_pattern_index,
        pair_molecule_index,
    )

    pattern_atom_counts = np.asarray(
        [pattern.atom_count for pattern in patterns],
        dtype=np.int32,
    )
    molecule_atom_counts = np.asarray(
        [molecule.atom_count for molecule in molecules],
        dtype=np.int32,
    )
    pattern_atoms_u32 = _pack_collection_atom_masks(patterns)
    molecule_atoms_u32 = _pack_collection_atom_masks(molecules)
    pattern_rows_u64 = _pack_collection_adjacency_rows(gmols=patterns)
    molecule_rows_u64 = _pack_collection_adjacency_rows(gmols=molecules)
    pattern_bonds_u8 = _pack_collection_bond_matrices(gmols=patterns)
    pattern_groups_u32 = pack_collection_atom_groups(gmols=patterns)
    molecule_bonds_u8 = _pack_collection_bond_matrices(
        gmols=molecules, annotate_topology=True
    )

    batch = CudaPruneBatch(
        pattern_count=pattern_count,
        molecule_count=molecule_count,
        pair_count=pair_pattern_index.size,
        pair_pattern_atom_counts=pair_pattern_atom_counts,
        pair_molecule_atom_counts=pair_molecule_atom_counts,
        pattern_atom_counts_device=cp.asarray(pattern_atom_counts, dtype=cp.int32),
        molecule_atom_counts_device=cp.asarray(molecule_atom_counts, dtype=cp.int32),
        pair_pattern_index_device=cp.asarray(pair_pattern_index, dtype=cp.int32),
        pair_molecule_index_device=cp.asarray(pair_molecule_index, dtype=cp.int32),
        pattern_atoms_u32=cp.asarray(pattern_atoms_u32, dtype=cp.uint32),
        molecule_atoms_u32=cp.asarray(molecule_atoms_u32, dtype=cp.uint32),
        pattern_rows_u64=cp.asarray(pattern_rows_u64, dtype=cp.uint64),
        molecule_rows_u64=cp.asarray(molecule_rows_u64, dtype=cp.uint64),
        pattern_bonds_u8=cp.asarray(pattern_bonds_u8, dtype=cp.uint8),
        molecule_bonds_u8=cp.asarray(molecule_bonds_u8, dtype=cp.uint8),
        pattern_groups_u32=cp.asarray(pattern_groups_u32, dtype=cp.uint32),
    )

    return apply_cuda_predicates(
        batch,
        prepare_cuda_predicates(patterns),
        max_prune_rounds=max_prune_rounds,
        max_leaf_checks=max_leaf_checks,
        max_prefix_depth=max_prefix_depth,
    )


def prepare_cuda_prune_batch_from_device(
    pattern_count: int,
    pattern_atom_counts_device: Any,
    pattern_atoms_u32_device: Any,
    pattern_rows_u64_device: Any,
    pattern_bonds_u8_device: Any,
    *,
    pattern_groups_u32_device: Any,
    molecule_atom_counts_device: Any,
    molecule_atoms_u32: Any,
    molecule_rows_u64: Any,
    molecule_bonds_u8: Any,
    pair_pattern_index: Any = None,
    pair_molecule_index: Any = None,
) -> CudaPruneBatch:
    cp = _require_cupy()

    molecule_count = int(molecule_atom_counts_device.shape[0])
    molecule_counts_host = _to_numpy(molecule_atom_counts_device).astype(
        np.int32,
        copy=False,
    )
    if np.any(molecule_counts_host > _CUDA_BUCKET_SIZE):
        raise ValueError("CUDA prune experiment supports graphs up to 64 atoms.")

    if (pair_pattern_index is None) != (pair_molecule_index is None):
        raise ValueError(
            "pair_pattern_index and pair_molecule_index must be supplied together."
        )
    if pair_pattern_index is None:
        # Default: the full pattern x molecule cross product (all-vs-all).
        pair_pattern_index, pair_molecule_index = _pair_indices(
            pattern_count,
            molecule_count,
        )
    else:
        # Explicit (rule, molecule) pair set (selective / some-vs-some matching).
        pair_pattern_index = np.asarray(pair_pattern_index, dtype=np.int32)
        pair_molecule_index = np.asarray(pair_molecule_index, dtype=np.int32)
        if pair_pattern_index.shape != pair_molecule_index.shape:
            raise ValueError("pair index arrays must have matching shape.")
    pattern_atom_counts = _to_numpy(pattern_atom_counts_device).astype(
        np.int32, copy=False
    )
    pair_pattern_atom_counts = (
        pattern_atom_counts[pair_pattern_index]
        if pair_pattern_index.size
        else np.zeros(0, dtype=np.int32)
    )
    pair_molecule_atom_counts = (
        molecule_counts_host[pair_molecule_index]
        if pair_molecule_index.size
        else np.zeros(0, dtype=np.int32)
    )

    return CudaPruneBatch(
        pattern_count=pattern_count,
        molecule_count=molecule_count,
        pair_count=pair_pattern_index.size,
        pair_pattern_atom_counts=pair_pattern_atom_counts,
        pair_molecule_atom_counts=pair_molecule_atom_counts,
        pattern_atom_counts_device=pattern_atom_counts_device,
        molecule_atom_counts_device=molecule_atom_counts_device.astype(
            cp.int32,
            copy=False,
        ),
        pair_pattern_index_device=cp.asarray(pair_pattern_index, dtype=cp.int32),
        pair_molecule_index_device=cp.asarray(pair_molecule_index, dtype=cp.int32),
        pattern_atoms_u32=pattern_atoms_u32_device,
        molecule_atoms_u32=cp.ascontiguousarray(molecule_atoms_u32).reshape(-1),
        pattern_rows_u64=pattern_rows_u64_device,
        molecule_rows_u64=cp.ascontiguousarray(molecule_rows_u64).reshape(-1),
        pattern_bonds_u8=pattern_bonds_u8_device,
        molecule_bonds_u8=cp.ascontiguousarray(molecule_bonds_u8).reshape(-1),
        pattern_groups_u32=pattern_groups_u32_device,
    )


def prepare_cuda_prune_batch_from_device_pairs(
    pattern_count: int,
    pattern_atom_counts_device: Any,
    pattern_atoms_u32_device: Any,
    pattern_rows_u64_device: Any,
    pattern_bonds_u8_device: Any,
    *,
    pattern_groups_u32_device: Any,
    molecule_atom_counts_device: Any,
    molecule_atoms_u32: Any,
    molecule_rows_u64: Any,
    molecule_bonds_u8: Any,
    pair_pattern_index_device: Any,
    pair_molecule_index_device: Any,
    pair_count: int,
) -> CudaPruneBatch:
    cp = _require_cupy()
    if pair_pattern_index_device.shape != pair_molecule_index_device.shape:
        raise ValueError("pair index device arrays must have matching shape.")
    if int(pair_pattern_index_device.shape[0]) != pair_count:
        raise ValueError("pair_count must match the device pair index length.")
    molecule_count = int(molecule_atom_counts_device.shape[0])
    return CudaPruneBatch(
        pattern_count=pattern_count,
        molecule_count=molecule_count,
        pair_count=pair_count,
        pair_pattern_atom_counts=np.zeros(0, dtype=np.int32),
        pair_molecule_atom_counts=np.zeros(0, dtype=np.int32),
        pattern_atom_counts_device=pattern_atom_counts_device,
        molecule_atom_counts_device=molecule_atom_counts_device.astype(
            cp.int32,
            copy=False,
        ),
        pair_pattern_index_device=pair_pattern_index_device.astype(
            cp.int32,
            copy=False,
        ),
        pair_molecule_index_device=pair_molecule_index_device.astype(
            cp.int32,
            copy=False,
        ),
        pattern_atoms_u32=pattern_atoms_u32_device,
        molecule_atoms_u32=cp.ascontiguousarray(molecule_atoms_u32).reshape(-1),
        pattern_rows_u64=pattern_rows_u64_device,
        molecule_rows_u64=cp.ascontiguousarray(molecule_rows_u64).reshape(-1),
        pattern_bonds_u8=pattern_bonds_u8_device,
        molecule_bonds_u8=cp.ascontiguousarray(molecule_bonds_u8).reshape(-1),
        pattern_groups_u32=pattern_groups_u32_device,
    )


def prune_bitset_cuda(
    batch: CudaPruneBatch,
    *,
    threads_per_block: int = 32,
    max_prune_rounds: int | None = None,
) -> CudaPruneResult:
    rows, prune_rounds, pruning_complete = _launch_bitset_kernel(
        batch,
        threads_per_block=threads_per_block,
        max_prune_rounds=max_prune_rounds,
    )
    candidates = _unpack_rows_u64_device(rows)
    return CudaPruneResult(
        pair_pattern_atom_counts=batch.pair_pattern_atom_counts,
        pair_molecule_atom_counts=batch.pair_molecule_atom_counts,
        candidates_device=candidates,
        prune_rounds_device=prune_rounds,
        pruning_complete_device=pruning_complete,
    )


def match_fused_cuda(
    batch: CudaPruneBatch,
    *,
    threads_per_block: int = 32,
    max_prune_rounds: int = 64,
    max_leaf_checks: int = 4096,
    max_prefix_depth: int = 16,
) -> CudaMatchResult:
    (
        status_codes,
        _,
        _,
        _,
        _,
        _,
        _,
        prune_rounds,
        pruning_complete,
        search_complete,
        leaf_checks,
        branches_explored,
    ) = _launch_match_kernel(
        batch,
        threads_per_block=threads_per_block,
        max_prune_rounds=max_prune_rounds,
        max_leaf_checks=max_leaf_checks,
        max_prefix_depth=max_prefix_depth,
        collect_assignments=False,
        assignment_capacity=0,
    )
    return CudaMatchResult(
        pattern_count=batch.pattern_count,
        molecule_count=batch.molecule_count,
        status_codes_device=status_codes,
        prune_rounds_device=prune_rounds,
        pruning_complete_device=pruning_complete,
        search_complete_device=search_complete,
        leaf_checks_device=leaf_checks,
        branches_explored_device=branches_explored,
    )


def match_fused_assignments_cuda(
    batch: CudaPruneBatch,
    *,
    threads_per_block: int = 32,
    max_prune_rounds: int = 64,
    max_leaf_checks: int = 4096,
    max_prefix_depth: int = 16,
    assignment_capacity: int | None = None,
    assignment_capacity_limit: int | None = None,
) -> CudaAssignmentResult:
    if threads_per_block != 32:
        raise ValueError(
            "Sparse CUDA assignment enumeration requires one 32-thread warp per pair."
        )
    initial_capacity = (
        max(int(np.ceil(batch.pair_count * 1.1)), 1)
        if assignment_capacity is None
        else assignment_capacity
    )
    if assignment_capacity_limit is not None:
        _validate_positive_bound(
            assignment_capacity_limit,
            name="assignment_capacity_limit",
        )
        initial_capacity = min(initial_capacity, assignment_capacity_limit)
    _validate_positive_bound(initial_capacity, name="assignment_capacity")
    (
        status_codes,
        assignment_counts,
        assignment_required,
        assignment_pair_ids,
        assignments,
        candidate_assignments,
        candidate_overflow,
        prune_rounds,
        pruning_complete,
        search_complete,
        leaf_checks,
        branches_explored,
    ) = _launch_match_kernel(
        batch,
        threads_per_block=threads_per_block,
        max_prune_rounds=max_prune_rounds,
        max_leaf_checks=max_leaf_checks,
        max_prefix_depth=max_prefix_depth,
        collect_assignments=True,
        assignment_capacity=initial_capacity,
    )
    required = int(_to_numpy(assignment_required).reshape(-1)[0])
    overflow_retries = 0
    final_capacity = initial_capacity
    if required > initial_capacity:
        if (
            assignment_capacity_limit is not None
            and required > assignment_capacity_limit
        ):
            raise CudaOutputCapacityError(
                "assignment",
                required,
                assignment_capacity_limit,
            )
        overflow_retries = 1
        final_capacity = required
        positive_pair_ids = np.flatnonzero(
            _to_numpy(status_codes).astype(np.uint8, copy=False) == 1
        ).astype(np.int32)
        pair_pattern_index = _to_numpy(batch.pair_pattern_index_device).astype(
            np.int32,
            copy=False,
        )
        pair_molecule_index = _to_numpy(batch.pair_molecule_index_device).astype(
            np.int32,
            copy=False,
        )
        retry_batch = prepare_cuda_prune_batch_from_device(
            batch.pattern_count,
            batch.pattern_atom_counts_device,
            batch.pattern_atoms_u32,
            batch.pattern_rows_u64,
            batch.pattern_bonds_u8,
            pattern_groups_u32_device=batch.pattern_groups_u32,
            molecule_atom_counts_device=batch.molecule_atom_counts_device,
            molecule_atoms_u32=batch.molecule_atoms_u32,
            molecule_rows_u64=batch.molecule_rows_u64,
            molecule_bonds_u8=batch.molecule_bonds_u8,
            pair_pattern_index=pair_pattern_index[positive_pair_ids],
            pair_molecule_index=pair_molecule_index[positive_pair_ids],
        )
        retry_batch = replace(
            retry_batch,
            pattern_predicates_u32=batch.pattern_predicates_u32,
            predicate_roots_u64=batch.predicate_roots_u64,
            predicate_count=batch.predicate_count,
            pair_root_atoms_i32=(
                None
                if batch.pair_root_atoms_i32 is None
                else batch.pair_root_atoms_i32[positive_pair_ids]
            ),
        )
        (
            retry_status_codes,
            retry_assignment_counts,
            assignment_required,
            assignment_pair_ids,
            assignments,
            retry_candidate_assignments,
            retry_candidate_overflow,
            retry_prune_rounds,
            retry_pruning_complete,
            retry_search_complete,
            retry_leaf_checks,
            retry_branches_explored,
        ) = _launch_match_kernel(
            retry_batch,
            threads_per_block=threads_per_block,
            max_prune_rounds=max_prune_rounds,
            max_leaf_checks=max_leaf_checks,
            max_prefix_depth=max_prefix_depth,
            collect_assignments=True,
            assignment_capacity=required,
        )
        del (
            retry_status_codes,
            retry_assignment_counts,
            retry_candidate_assignments,
            retry_candidate_overflow,
            retry_prune_rounds,
            retry_pruning_complete,
            retry_search_complete,
            retry_leaf_checks,
        )
        second_required = int(_to_numpy(assignment_required).reshape(-1)[0])
        if second_required != required:
            raise RuntimeError(
                "CUDA assignment count changed between capacity retry passes: "
                f"{required} != {second_required}"
            )
        cp = _require_cupy()
        assignment_pair_ids = cp.asarray(
            positive_pair_ids,
            dtype=cp.int32,
        )[assignment_pair_ids]
    return CudaAssignmentResult(
        pattern_count=batch.pattern_count,
        molecule_count=batch.molecule_count,
        assignment_capacity=final_capacity,
        assignment_required=required,
        overflow_retries=overflow_retries,
        status_codes_device=status_codes,
        assignment_counts_device=assignment_counts,
        assignment_pair_ids_device=assignment_pair_ids,
        assignments_device=assignments,
        candidate_assignments_device=candidate_assignments,
        candidate_overflow_device=candidate_overflow,
        prune_rounds_device=prune_rounds,
        pruning_complete_device=pruning_complete,
        search_complete_device=search_complete,
        leaf_checks_device=leaf_checks,
        branches_explored_device=branches_explored,
    )

    def branches_explored_host(self) -> np.ndarray:
        return _reshape_pair_host(
            self.branches_explored_device,
            self.pattern_count,
            self.molecule_count,
            dtype=np.int32,
        )


def _initialize_candidate_rows_device(
    batch: CudaPruneBatch,
    *,
    threads_per_block: int = 32,
) -> Any:
    rows = _launch_initialize_candidate_rows_kernel(
        batch,
        threads_per_block=threads_per_block,
    )
    return _unpack_rows_u64_device(rows)


def _launch_initialize_candidate_rows_kernel(
    batch: CudaPruneBatch,
    *,
    threads_per_block: int,
) -> Any:
    cp = _require_cupy()
    _validate_threads_per_block(threads_per_block)
    rows = cp.zeros((batch.pair_count, _CUDA_BUCKET_SIZE), dtype=cp.uint64)
    if batch.pair_count == 0:
        return rows

    kernel = _get_initialize_kernel(cp)
    kernel(
        (batch.pair_count,),
        (threads_per_block,),
        (
            batch.pattern_atoms_u32,
            batch.molecule_atoms_u32,
            batch.pattern_atom_counts_device,
            batch.molecule_atom_counts_device,
            batch.pair_pattern_index_device,
            batch.pair_molecule_index_device,
            *predicate_arguments(batch),
            rows,
        ),
    )
    return rows


def _launch_bitset_kernel(
    batch: CudaPruneBatch,
    *,
    threads_per_block: int,
    max_prune_rounds: int | None = None,
) -> tuple[Any, Any, Any]:
    cp = _require_cupy()
    _validate_threads_per_block(threads_per_block)
    _validate_max_prune_rounds(max_prune_rounds)

    rows = cp.zeros((batch.pair_count, _CUDA_BUCKET_SIZE), dtype=cp.uint64)
    prune_rounds = cp.zeros(batch.pair_count, dtype=cp.int32)
    pruning_complete = cp.zeros(batch.pair_count, dtype=cp.uint8)
    if batch.pair_count == 0:
        return rows, prune_rounds, pruning_complete.astype(cp.bool_, copy=False)

    kernel = _get_bitset_kernel(cp)
    kernel(
        (batch.pair_count,),
        (threads_per_block,),
        (
            batch.pattern_atoms_u32,
            batch.molecule_atoms_u32,
            batch.pattern_rows_u64,
            batch.molecule_rows_u64,
            batch.pattern_atom_counts_device,
            batch.molecule_atom_counts_device,
            batch.pair_pattern_index_device,
            batch.pair_molecule_index_device,
            *predicate_arguments(batch),
            np.int32(-1 if max_prune_rounds is None else max_prune_rounds),
            rows,
            prune_rounds,
            pruning_complete,
        ),
    )
    return rows, prune_rounds, pruning_complete.astype(cp.bool_, copy=False)


def _launch_match_kernel(
    batch: CudaPruneBatch,
    *,
    threads_per_block: int,
    max_prune_rounds: int,
    max_leaf_checks: int,
    max_prefix_depth: int,
    collect_assignments: bool,
    assignment_capacity: int,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    cp = _require_cupy()
    if threads_per_block != 32:
        raise ValueError("Fused CUDA matching requires one 32-thread warp per pair.")
    _validate_positive_bound(max_prune_rounds, name="max_prune_rounds")
    _validate_positive_bound(max_leaf_checks, name="max_leaf_checks")
    if collect_assignments:
        _validate_positive_bound(assignment_capacity, name="assignment_capacity")
    elif assignment_capacity != 0:
        raise ValueError("Match-only CUDA launches require zero assignment capacity.")

    status_codes = cp.zeros(batch.pair_count, dtype=cp.uint8)
    assignment_counts = cp.zeros(
        batch.pair_count if collect_assignments else 0,
        dtype=cp.int32,
    )
    assignment_required = cp.zeros(1 if collect_assignments else 0, dtype=cp.uint64)
    assignment_pair_ids = cp.full(assignment_capacity, -1, dtype=cp.int32)
    assignments = cp.full((assignment_capacity, _CUDA_BUCKET_SIZE), -1, dtype=cp.int32)
    candidate_assignments = cp.zeros(
        batch.pair_count if collect_assignments else 0,
        dtype=cp.uint64,
    )
    candidate_overflow = cp.zeros(
        batch.pair_count if collect_assignments else 0,
        dtype=cp.uint8,
    )
    prune_rounds = cp.zeros(batch.pair_count, dtype=cp.int32)
    pruning_complete = cp.zeros(batch.pair_count, dtype=cp.uint8)
    search_complete = cp.zeros(batch.pair_count, dtype=cp.uint8)
    leaf_checks = cp.zeros(batch.pair_count, dtype=cp.int32)
    branches_explored = cp.zeros(batch.pair_count, dtype=cp.int32)
    if batch.pair_count == 0:
        return (
            status_codes,
            assignment_counts,
            assignment_required,
            assignment_pair_ids,
            assignments,
            candidate_assignments,
            candidate_overflow.astype(cp.bool_, copy=False),
            prune_rounds,
            pruning_complete.astype(cp.bool_, copy=False),
            search_complete.astype(cp.bool_, copy=False),
            leaf_checks,
            branches_explored,
        )

    kernel = _get_match_kernel(cp)
    kernel(
        (batch.pair_count,),
        (threads_per_block,),
        (
            batch.pattern_atoms_u32,
            batch.molecule_atoms_u32,
            batch.pattern_rows_u64,
            batch.molecule_rows_u64,
            batch.pattern_bonds_u8,
            batch.molecule_bonds_u8,
            batch.pattern_groups_u32,
            batch.pattern_atom_counts_device,
            batch.molecule_atom_counts_device,
            batch.pair_pattern_index_device,
            batch.pair_molecule_index_device,
            *predicate_arguments(batch),
            np.int32(max_prune_rounds),
            np.int32(max_leaf_checks),
            np.int32(max_prefix_depth),
            np.int32(collect_assignments),
            np.uint64(assignment_capacity),
            status_codes,
            assignment_counts,
            assignment_required,
            assignment_pair_ids,
            assignments,
            candidate_assignments,
            candidate_overflow,
            prune_rounds,
            pruning_complete,
            search_complete,
            leaf_checks,
            branches_explored,
        ),
    )
    return (
        status_codes,
        assignment_counts,
        assignment_required,
        assignment_pair_ids,
        assignments,
        candidate_assignments,
        candidate_overflow.astype(cp.bool_, copy=False),
        prune_rounds,
        pruning_complete.astype(cp.bool_, copy=False),
        search_complete.astype(cp.bool_, copy=False),
        leaf_checks,
        branches_explored,
    )


def _import_cupy() -> Any:
    try:
        return importlib.import_module("cupy")
    except ModuleNotFoundError as exc:
        raise CudaUnavailableError(
            "CuPy is required for the CUDA prune experiment."
        ) from exc


def _require_cupy() -> Any:
    cp = _import_cupy()
    if not cp.cuda.is_available():
        raise CudaUnavailableError("CUDA is not available to CuPy.")
    return cp


def _get_cuda_module(cp: Any) -> Any:
    global _CUDA_MODULE
    if _CUDA_MODULE is None:
        _CUDA_MODULE = cp.RawModule(
            code=_CUDA_KERNEL_SOURCE,
            options=("-std=c++11",),
        )
    return _CUDA_MODULE


def _get_initialize_kernel(cp: Any) -> Any:
    global _INITIALIZE_KERNEL
    if _INITIALIZE_KERNEL is None:
        _INITIALIZE_KERNEL = _get_cuda_module(cp).get_function(
            "initialize_candidate_rows_pairs"
        )
    return _INITIALIZE_KERNEL


def _get_bitset_kernel(cp: Any) -> Any:
    global _BITSET_KERNEL
    if _BITSET_KERNEL is None:
        _BITSET_KERNEL = _get_cuda_module(cp).get_function("prune_bitset_pairs")
    return _BITSET_KERNEL


def _get_match_kernel(cp: Any) -> Any:
    global _MATCH_KERNEL
    if _MATCH_KERNEL is None:
        _MATCH_KERNEL = _get_cuda_module(cp).get_function("match_fused_pairs")
    return _MATCH_KERNEL


def _pair_indices(
    pattern_count: int,
    molecule_count: int,
) -> tuple[PairVector, PairVector]:
    return (
        np.repeat(np.arange(pattern_count, dtype=np.int32), molecule_count),
        np.tile(np.arange(molecule_count, dtype=np.int32), pattern_count),
    )


def _pair_atom_counts(
    patterns: list[GMol],
    molecules: list[GMol],
    pair_pattern_index: PairVector,
    pair_molecule_index: PairVector,
) -> tuple[PairVector, PairVector]:
    pattern_counts = np.asarray(
        [pattern.atom_count for pattern in patterns],
        dtype=np.int32,
    )
    molecule_counts = np.asarray(
        [molecule.atom_count for molecule in molecules],
        dtype=np.int32,
    )
    return (
        (
            pattern_counts[pair_pattern_index]
            if pair_pattern_index.size
            else np.zeros(0, dtype=np.int32)
        ),
        (
            molecule_counts[pair_molecule_index]
            if pair_molecule_index.size
            else np.zeros(0, dtype=np.int32)
        ),
    )


def _pack_collection_atom_masks(gmols: list[GMol]) -> np.ndarray:
    if not gmols:
        return np.zeros((0, _CUDA_BUCKET_SIZE * _ATOM_WORD_COUNT), dtype=np.uint32)
    return np.stack([_pack_atom_masks_u32(gmol) for gmol in gmols]).reshape(-1)


def _pack_atom_masks_u32(gmol: GMol) -> np.ndarray:
    return pack_atom_masks(gmol, _CUDA_BUCKET_SIZE, aromaticity_any=0xFF)


def _pack_collection_adjacency_rows(*, gmols: list[GMol]) -> np.ndarray:
    if not gmols:
        return np.zeros((0, _CUDA_BUCKET_SIZE), dtype=np.uint64)
    return np.stack([_pack_adjacency_rows_u64(gmol) for gmol in gmols]).reshape(-1)


def pack_collection_atom_groups(*, gmols: list[GMol]) -> np.ndarray:
    """Zero-level group ids per atom slot, one row per pattern, flattened.

    Public where its siblings are not, because every construction path for a
    :class:`CudaPruneBatch` has to supply it and none may leave it out: a pattern whose
    parentheses were dropped on the way to the device matches more than it says, silently.
    """
    if not gmols:
        return np.zeros((0, _CUDA_BUCKET_SIZE), dtype=np.uint32)
    return np.stack([pack_atom_groups(graph) for graph in gmols]).reshape(-1)


def pack_atom_groups(graph: GMol) -> np.ndarray:
    groups = np.zeros(_CUDA_BUCKET_SIZE, dtype=np.uint32)
    groups[: graph.atom_count] = graph.atom_group
    return groups


def _pack_adjacency_rows_u64(gmol: GMol) -> np.ndarray:
    return _pack_rows_u64(_pad_bool_matrix(bond_presence(gmol.bonds)))


def _pack_collection_bond_matrices(
    *, gmols: list[GMol], annotate_topology: bool = False
) -> np.ndarray:
    if not gmols:
        return np.zeros((0, _CUDA_BUCKET_SIZE, _CUDA_BUCKET_SIZE), dtype=np.uint8)
    return np.stack(
        [
            _pack_bond_matrix_u8(gmol, annotate_topology=annotate_topology)
            for gmol in gmols
        ]
    ).reshape(-1)


def _pack_bond_matrix_u8(gmol: GMol, *, annotate_topology: bool = False) -> np.ndarray:
    bonds = annotate_bond_topology(gmol.bonds) if annotate_topology else gmol.bonds
    padded = np.zeros((_CUDA_BUCKET_SIZE, _CUDA_BUCKET_SIZE), dtype=np.uint8)
    count = gmol.atom_count
    padded[:count, :count] = bonds
    return padded


def _validate_graph_sizes(patterns: list[GMol], molecules: list[GMol]) -> None:
    for graph in [*patterns, *molecules]:
        if graph.atom_count > _CUDA_BUCKET_SIZE:
            raise ValueError("CUDA prune experiment supports graphs up to 64 atoms.")


def _validate_threads_per_block(threads_per_block: int) -> None:
    if threads_per_block <= 0 or threads_per_block > _CUDA_BUCKET_SIZE:
        raise ValueError("threads_per_block must be between 1 and 64.")


def _validate_max_prune_rounds(max_prune_rounds: int | None) -> None:
    if max_prune_rounds is not None and max_prune_rounds <= 0:
        raise ValueError("max_prune_rounds must be positive when provided.")


def _validate_positive_bound(value: int, *, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _pad_bool_matrix(matrix: np.ndarray) -> CandidateMatrix:
    padded = np.zeros((_CUDA_BUCKET_SIZE, _CUDA_BUCKET_SIZE), dtype=bool)
    rows, cols = matrix.shape
    padded[:rows, :cols] = matrix.astype(bool, copy=False)
    return padded


def _pack_rows_u64(
    matrix: CandidateMatrix,
) -> np.ndarray[tuple[int], np.dtype[np.uint64]]:
    packed = np.packbits(
        matrix.astype(np.uint8, copy=False),
        axis=-1,
        bitorder="little",
    )
    return np.ascontiguousarray(packed).view(np.uint64).reshape(matrix.shape[0])


def _unpack_rows_u64_device(rows: Any) -> Any:
    cp = _require_cupy()
    bit_offsets = cp.arange(_CUDA_BUCKET_SIZE, dtype=cp.uint64)
    return ((rows[:, :, None] >> bit_offsets[None, None, :]) & cp.uint64(1)).astype(
        cp.bool_,
        copy=False,
    )


def _to_numpy(array: Any) -> np.ndarray:
    if hasattr(array, "get"):
        return array.get()
    return np.asarray(array)


def _reshape_pair_host(
    array: Any,
    pattern_count: int,
    molecule_count: int,
    *,
    dtype: Any,
) -> np.ndarray:
    return (
        _to_numpy(array)
        .astype(dtype, copy=False)
        .reshape(
            pattern_count,
            molecule_count,
        )
    )


CUDA_PREDICATE_PAIRS_PER_LAUNCH = 32_768
GPU_PROGRAM_ARRAYS = (
    "body_atoms",
    "body_rows",
    "body_bonds",
    "body_groups",
    "body_preds",
    "body_counts",
    "probes",
    "clause_start",
    "term_start",
    "factor_start",
    "factors",
)


@dataclass(frozen=True)
class CudaPredicateProgram:
    """A shared predicate program and its rule-owned CUDA buffers."""

    program: GpuPredicateProgram
    pattern_predicates: Any
    device: Mapping[str, Any]


def prepare_cuda_predicates(patterns: list[GMol]) -> CudaPredicateProgram | None:
    table = compile_predicate_table(patterns)
    if not table.residuals:
        return None
    cp = _require_cupy()
    program = compile_predicate_program(
        table,
        pack_atoms=_pack_atom_masks_u32,
        pack_rows=_pack_adjacency_rows_u64,
        pack_bonds=_pack_bond_matrix_u8,
        pack_groups=pack_atom_groups,
    )
    device = {name: cp.asarray(getattr(program, name)) for name in GPU_PROGRAM_ARRAYS}
    device["body_counts"] = device["body_counts"].astype(cp.int32)
    return CudaPredicateProgram(
        program=program,
        pattern_predicates=cp.asarray(
            pack_collection_atom_predicates(table, patterns, _CUDA_BUCKET_SIZE)
        ),
        device=device,
    )


def predicate_arguments(
    batch: CudaPruneBatch,
) -> tuple[Any, Any, np.int32, Any, np.int32]:
    """Optional query metadata, represented by null pointers when unused."""
    predicates = batch.pattern_predicates_u32
    roots = batch.predicate_roots_u64
    pins = batch.pair_root_atoms_i32
    return (
        np.uint64(0) if predicates is None else predicates,
        np.uint64(0) if roots is None else roots,
        np.int32(batch.predicate_count),
        np.uint64(0) if pins is None else pins,
        np.int32(pins is not None),
    )


def apply_cuda_predicates(
    batch: CudaPruneBatch,
    program: CudaPredicateProgram | None,
    *,
    max_prune_rounds: int = 64,
    max_leaf_checks: int = 4096,
    max_prefix_depth: int = 16,
) -> CudaPruneBatch:
    """Narrow the query using roots evaluated from the resident molecule arrays."""
    if program is None:
        return batch
    return replace(
        batch,
        pattern_predicates_u32=program.pattern_predicates,
        predicate_roots_u64=evaluate_cuda_predicate_roots(
            program,
            batch,
            max_prune_rounds=max_prune_rounds,
            max_leaf_checks=max_leaf_checks,
            max_prefix_depth=max_prefix_depth,
        ),
        predicate_count=program.program.residual_count,
    )


def evaluate_cuda_predicate_roots(
    compiled: CudaPredicateProgram,
    batch: CudaPruneBatch,
    *,
    max_prune_rounds: int = 64,
    max_leaf_checks: int = 4096,
    max_prefix_depth: int = 16,
) -> Any:
    """Evaluate innermost predicates first, keeping molecule data and roots on CUDA.

    Recursive bodies are existential searches pinned at every admissible first atom, not
    capped assignment enumeration. Scratch is bounded by the launch size. Molecule arrays
    and root bitsets stay on device; an unresolved body is reported before roots are returned.
    """
    cp = _require_cupy()
    program = compiled.program
    device = compiled.device
    molecule_count = batch.molecule_count
    residual_count = program.residual_count
    body_count = len(program.body_counts)
    probe_count = len(program.probes) // _ATOM_WORD_COUNT
    roots = cp.zeros((molecule_count, residual_count), dtype=cp.uint64)
    if molecule_count == 0:
        return roots
    probes = cp.zeros((molecule_count, probe_count), dtype=cp.uint64)
    sub_roots = cp.zeros((molecule_count, body_count), dtype=cp.uint64)
    unresolved = cp.zeros(1, dtype=cp.uint32)
    module = _get_cuda_module(cp)
    if probe_count:
        module.get_function("predicate_probe_bits")(
            ((molecule_count * probe_count + 255) // 256,),
            (256,),
            (
                device["probes"],
                batch.molecule_atoms_u32,
                batch.molecule_atom_counts_device,
                probes,
                np.int32(molecule_count),
                np.int32(probe_count),
            ),
        )
    atom_bits = cp.left_shift(cp.uint64(1), cp.arange(64, dtype=cp.uint64))
    for level in range(program.depth + 1):
        first_body = int(program.body_level_start[level])
        end_body = int(program.body_level_start[level + 1])
        for start in range(
            0,
            (end_body - first_body) * molecule_count,
            CUDA_PREDICATE_PAIRS_PER_LAUNCH,
        ):
            indices = cp.arange(
                start,
                min(
                    start + CUDA_PREDICATE_PAIRS_PER_LAUNCH,
                    (end_body - first_body) * molecule_count,
                ),
                dtype=cp.int32,
            )
            bodies = first_body + indices // molecule_count
            molecules = indices % molecule_count
            body_batch = replace(
                batch,
                pattern_count=body_count,
                pair_count=len(indices),
                pattern_atom_counts_device=device["body_counts"],
                pattern_atoms_u32=device["body_atoms"],
                pattern_rows_u64=device["body_rows"],
                pattern_bonds_u8=device["body_bonds"],
                pattern_groups_u32=device["body_groups"],
                pattern_predicates_u32=device["body_preds"],
                predicate_roots_u64=roots,
                predicate_count=residual_count,
                pair_pattern_index_device=bodies,
                pair_molecule_index_device=molecules,
                pair_root_atoms_i32=None,
            )
            rows = _launch_initialize_candidate_rows_kernel(
                body_batch, threads_per_block=32
            )[:, 0]
            selected, atoms = cp.nonzero((rows[:, None] & atom_bits[None, :]) != 0)
            for offset in range(0, len(selected), CUDA_PREDICATE_PAIRS_PER_LAUNCH):
                positions = selected[offset : offset + CUDA_PREDICATE_PAIRS_PER_LAUNCH]
                pinned = replace(
                    body_batch,
                    pair_count=len(positions),
                    pair_pattern_index_device=bodies[positions],
                    pair_molecule_index_device=molecules[positions],
                    pair_root_atoms_i32=atoms[offset : offset + len(positions)].astype(
                        cp.int32
                    ),
                )
                result = match_fused_cuda(
                    pinned,
                    max_prune_rounds=max_prune_rounds,
                    max_leaf_checks=max_leaf_checks,
                    max_prefix_depth=max_prefix_depth,
                )
                module.get_function("predicate_fold_roots")(
                    ((len(positions) + 255) // 256,),
                    (256,),
                    (
                        result.status_codes_device,
                        result.search_complete_device,
                        pinned.pair_pattern_index_device,
                        pinned.pair_molecule_index_device,
                        pinned.pair_root_atoms_i32,
                        sub_roots,
                        unresolved,
                        np.int32(len(positions)),
                        np.int32(body_count),
                    ),
                )
        first = int(program.residual_level_start[level])
        count = int(program.residual_level_start[level + 1]) - first
        if count:
            module.get_function("predicate_eval_residuals")(
                ((molecule_count * count + 255) // 256,),
                (256,),
                (
                    device["clause_start"],
                    device["term_start"],
                    device["factor_start"],
                    device["factors"],
                    probes,
                    sub_roots,
                    roots,
                    batch.molecule_atom_counts_device,
                    np.int32(molecule_count),
                    np.int32(first),
                    np.int32(count),
                    np.int32(residual_count),
                    np.int32(body_count),
                    np.int32(probe_count),
                ),
            )
    unknown = int(unresolved.get()[0])
    if unknown:
        raise UnresolvedPredicateError(
            f"CUDA left {unknown} recursive body searches unresolved; predicate roots are unknown."
        )
    return roots

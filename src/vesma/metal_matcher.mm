#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char g_last_error[1024] = {0};
static id<MTLBuffer> s_pattern_atoms = nil;
static id<MTLBuffer> s_pattern_rows = nil;
static id<MTLBuffer> s_pattern_bonds = nil;
static id<MTLBuffer> s_pattern_counts = nil;
static id<MTLBuffer> s_pattern_groups = nil;
static id<MTLBuffer> s_pattern_preds = nil;
static uint64_t s_pattern_token = 0;
static uint32_t s_pattern_count = 0;

struct Params {
    uint32_t bucket_size;
    uint32_t atom_word_count;
    uint32_t max_prunings;
    uint32_t max_leaf_checks;
    uint32_t max_assignments;
    uint32_t write_assignments;
    uint32_t pair_count;
    uint32_t max_prefix_depth;
    uint32_t pred_stride;
    uint32_t pair_base;
    uint32_t list_mode;
};

struct PredParams {
    uint32_t n_mols;
    uint32_t probe_count;
    uint32_t first;
    uint32_t count;
    uint32_t pred_stride;
    uint32_t sub_stride;
    uint32_t atom_word_count;
    uint32_t bucket_size;
    uint32_t level;
    uint32_t max_pairs;
};

static const char* kShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

typedef ulong bitset_t;

constant constexpr uint MAX_DFS_DEPTH = 16u;
constant constexpr uchar BOND_CLASS_MASK = 0x0fu;
constant constexpr uchar BOND_TOPOLOGY_MASK = 0x30u;

inline uchar bond_class(uchar flags) {
    return flags & BOND_CLASS_MASK;
}

struct Params {
    uint bucket_size;
    uint atom_word_count;
    uint max_prunings;
    uint max_leaf_checks;
    uint max_assignments;
    uint write_assignments;
    uint pair_count;
    uint max_prefix_depth;
    uint pred_stride;
    // Added to the threadgroup index, so one level's pairs can be split across dispatches.
    uint pair_base;
    // Nonzero: pair i is pair_list[i] = (pattern, (molecule << 6) | pinned root), a list the
    // device built itself -- the predicate levels, whose pair count no host ever sees.
    uint list_mode;
};

struct PredParams {
    uint n_mols;
    uint probe_count;
    uint first;
    uint count;
    uint pred_stride;
    uint sub_stride;
    uint atom_word_count;
    uint bucket_size;
    uint level;
    uint max_pairs;
};

constant constexpr uint NO_PREDICATE = 0xFFFFFFFFu;
// Word 13 holds two fields: the ring-connectivity (`x`) bucket mask in its low half, and the
// isotope in its high half -- the mass, 0 for unlabelled, or ISOTOPE_ANY in a query that names
// none. Isotope is exact-or-any, not a set, so that half is compared, never ANDed.
constant constexpr uint ISOTOPE_WORD = 13u;
constant constexpr uint ISOTOPE_SHIFT = 16u;
constant constexpr uint ISOTOPE_ANY = 0xFFFFu;
constant constexpr uint LOW_FIELD = 0xFFFFu;
constant constexpr uint TERM_NEGATED = 0x80000000u;
constant constexpr uint TERM_SUBPATTERN = 0x40000000u;
constant constexpr uint TERM_INDEX = 0x3FFFFFFFu;

template<typename T1, typename T2>
inline bool atom_masks_overlap(
    T1 pattern_atoms,
    T2 molecule_atoms,
    uint p_off,
    uint m_off,
    uint word_count) {
    // Basic atom mask overlap check (Z_lo, Z_hi, and other properties).
    bool z_match = ((pattern_atoms[p_off + 0] & molecule_atoms[m_off + 0]) != 0) ||
                   ((pattern_atoms[p_off + 1] & molecule_atoms[m_off + 1]) != 0) ||
                   ((pattern_atoms[p_off + 2] & molecule_atoms[m_off + 2]) != 0) ||
                   ((pattern_atoms[p_off + 3] & molecule_atoms[m_off + 3]) != 0);
    if (!z_match) return false;
    for (uint i = 4; i < word_count; ++i) {
        const uint p = pattern_atoms[p_off + i];
        const uint m = molecule_atoms[m_off + i];
        if (i == ISOTOPE_WORD) {
            if ((p & m & LOW_FIELD) == 0) return false;
            const uint mass = p >> ISOTOPE_SHIFT;
            if (mass != ISOTOPE_ANY && mass != (m >> ISOTOPE_SHIFT)) return false;
        } else if ((p & m) == 0) {
            return false;
        }
    }
    return true;
}

// Index of the n-th (0-based) set bit in a mask, or -1 if fewer than n+1 bits set.
inline int nth_set_bit_index(bitset_t mask, int ordinal) {
    int seen = 0;
    while (mask != 0) {
        uint b = ctz(mask);
        if (seen == ordinal) return (int)b;
        mask &= (mask - 1);
        seen += 1;
    }
    return -1;
}

// Injectivity + edge-preservation + bond-type compatibility + zero-level grouping, for a
// complete assignment.
inline bool final_assignment_valid(
    thread const int* assignment,
    threadgroup const bitset_t* pattern_adj,
    threadgroup const bitset_t* molecule_adj,
    device const uchar* p_bonds,
    device const uchar* m_bonds,
    device const uint* p_groups,
    threadgroup const bitset_t* comp_mask,
    int has_groups,
    uint pattern_count) {

    // 1. Uniqueness (injectivity) using a bitmask.
    bitset_t assigned_mask = 0;
    for (uint i = 0; i < pattern_count; ++i) {
        int m_i = assignment[i];
        if (m_i < 0) return false;
        if ((assigned_mask & (1UL << m_i)) != 0) return false;
        assigned_mask |= (1UL << m_i);
    }

    // 2. Edge preservation + bond type compatibility (each undirected edge once, j > i).
    for (uint i = 0; i < pattern_count; ++i) {
        bitset_t p_neighbors = pattern_adj[i];
        if (i < 63) {
            p_neighbors &= ~((1UL << (i + 1)) - 1);
        } else {
            p_neighbors = 0;
        }

        while (p_neighbors != 0) {
            uint j = ctz(p_neighbors);
            uint m_i = assignment[i];
            uint m_j = assignment[j];

            // Edge preservation.
            if ((molecule_adj[m_i] & (1UL << m_j)) == 0) return false;

            // Query and molecule topology use the same allowed-state field:
            // 01 ring, 10 chain, 11 either. Compatibility requires both the
            // bond-class sets and topology-state sets to intersect. Zero query
            // flags mean that this dense matrix position has no bond constraint.
            uchar p_flags = p_bonds[i * 64 + j];
            uchar m_flags = m_bonds[m_i * 64 + m_j];
            bool ok = p_flags == 0u
                   || ((bond_class(p_flags) & bond_class(m_flags)) != 0u
                       && ((p_flags & BOND_TOPOLOGY_MASK)
                           & (m_flags & BOND_TOPOLOGY_MASK)) != 0u);
            if (!ok) return false;

            p_neighbors ^= (1UL << j);
        }
    }

    // 3. Zero-level grouping: what the pattern's parentheses said about separate molecules.
    // Same non-zero group id, same component of the target; different ids, different
    // components. Group 0 says nothing and is skipped, which is what leaves an
    // unparenthesised pattern -- nearly the whole library -- on exactly its old path.
    if (has_groups != 0) {
        for (uint i = 0; i < pattern_count; ++i) {
            uint g_i = p_groups[i];
            if (g_i == 0u) continue;
            for (uint j = i + 1; j < pattern_count; ++j) {
                uint g_j = p_groups[j];
                if (g_j == 0u) continue;
                bool together = (comp_mask[assignment[i]] & (1UL << assignment[j])) != 0;
                if ((g_i == g_j) != together) return false;
            }
        }
    }
    return true;
}

kernel void match_fused(
    device const uint* pattern_atoms_all [[buffer(0)]],
    device const uint* molecule_atoms_all [[buffer(1)]],
    device const ulong* pattern_rows_all [[buffer(2)]],
    device const ulong* molecule_rows_all [[buffer(3)]],
    device const uint* pair_pattern_indices [[buffer(4)]],
    device const uint* pair_molecule_indices [[buffer(5)]],
    device const uint* pattern_atom_counts [[buffer(6)]],
    device const uint* molecule_atom_counts [[buffer(7)]],
    device int* assignments_out [[buffer(8)]], // [pair_count * max_assignments * 64]
    device int* status_out [[buffer(9)]],      // [pair_count * 4] -> [count, status, prunings, leaf_checks]
    constant Params& params [[buffer(10)]],
    device const uchar* pattern_bonds_all [[buffer(11)]],
    device const uchar* molecule_bonds_all [[buffer(12)]],
    device const uint* pattern_groups_all [[buffer(13)]],
    // Per pattern atom slot, its recursive predicate's id in the roots table, or NO_PREDICATE.
    device const uint* pattern_preds_all [[buffer(14)]],
    // Per (molecule, predicate), the molecule atoms the predicate holds at: [m * pred_stride + p].
    device const ulong* roots_all [[buffer(15)]],
    // List mode only: see Params::list_mode.
    device const uint2* pair_list [[buffer(17)]],
    uint group_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]) {

    if (group_idx >= params.pair_count) return;
    uint pair_idx = params.pair_base + group_idx;

    uint p_idx;
    uint m_idx;
    bitset_t pin = ~0UL;
    if (params.list_mode != 0u) {
        uint2 entry = pair_list[pair_idx];
        p_idx = entry.x;
        m_idx = entry.y >> 6;
        pin = 1UL << (entry.y & 63u);
    } else {
        p_idx = pair_pattern_indices[pair_idx];
        m_idx = pair_molecule_indices[pair_idx];
    }
    uint p_count = pattern_atom_counts[p_idx];
    uint m_count = molecule_atom_counts[m_idx];

    uint max_prefix_depth = min(params.max_prefix_depth, MAX_DFS_DEPTH);

    // --- threadgroup state ---
    threadgroup uint p_atoms_cache[64 * 14];
    threadgroup uint m_atoms_cache[64 * 14];
    threadgroup bitset_t pattern_adj[64];
    threadgroup bitset_t molecule_adj[64];
    // Which atoms of the target share a molecule with each atom, as a bitset per atom. Derived
    // here rather than uploaded: the molecule's adjacency is already resident, and a component is
    // a closure over it.
    threadgroup bitset_t comp_mask[64];
    threadgroup bitset_t current_rows[64];
    threadgroup bitset_t u_cols[64];
    threadgroup bitset_t next_rows[64];
    threadgroup bitset_t stack_rows[MAX_DFS_DEPTH][64];
    threadgroup int stack_branch_row[MAX_DFS_DEPTH];
    threadgroup bitset_t stack_branch_cols[MAX_DFS_DEPTH];
    threadgroup int row_counts[64];
    threadgroup int ambiguous_rows[64];
    threadgroup int fixed_assignment[64];

    threadgroup int prefix_depth;
    threadgroup int rounds;
    threadgroup int prune_rounds;
    threadgroup int stop;
    threadgroup atomic_uint changed;
    threadgroup atomic_uint unsupported;
    threadgroup int ambiguous_count;
    threadgroup int singleton_conflict;
    threadgroup bitset_t singleton_bits;
    threadgroup ulong total_leaf_count;
    threadgroup ulong leaf_limit;
    threadgroup atomic_uint assignment_count;
    threadgroup int search_complete;
    threadgroup int leaf_checks;
    threadgroup uint status_code;
    threadgroup int dfs_stop;
    threadgroup int save_state;
    threadgroup int do_leaf_eval;
    threadgroup int branches_explored;
    threadgroup int has_groups;

    device const uint* p_atoms = pattern_atoms_all + (p_idx * params.bucket_size * params.atom_word_count);
    device const uint* m_atoms = molecule_atoms_all + (m_idx * params.bucket_size * params.atom_word_count);
    device const ulong* p_rows = pattern_rows_all + (p_idx * params.bucket_size);
    device const ulong* m_rows = molecule_rows_all + (m_idx * params.bucket_size);
    device const uchar* p_bonds = pattern_bonds_all + (p_idx * params.bucket_size * params.bucket_size);
    device const uchar* m_bonds = molecule_bonds_all + (m_idx * params.bucket_size * params.bucket_size);
    device const uint* p_groups = pattern_groups_all + (p_idx * params.bucket_size);
    device const uint* p_preds = pattern_preds_all + (p_idx * params.bucket_size);

    // Candidate construction uses atom masks; only final validation reads bond
    // flags. Keep the latter in device buffers instead of copying two padded
    // 64x64 matrices into every candidate pair's threadgroup.
    for (uint i = tid; i < 64 * 14; i += threads_per_group) {
        p_atoms_cache[i] = (i < p_count * params.atom_word_count) ? p_atoms[i] : 0;
        m_atoms_cache[i] = (i < m_count * params.atom_word_count) ? m_atoms[i] : 0;
    }
    for (uint i = tid; i < 64; i += threads_per_group) {
        pattern_adj[i] = (i < p_count) ? p_rows[i] : 0;
        molecule_adj[i] = (i < m_count) ? m_rows[i] : 0;
        current_rows[i] = 0;
        next_rows[i] = 0;
        row_counts[i] = 0;
        ambiguous_rows[i] = -1;
        fixed_assignment[i] = -1;
    }
    if (tid == 0) {
        prefix_depth = 0;
        prune_rounds = 0;
        search_complete = 1;
        leaf_checks = 0;
        status_code = 2u;
        dfs_stop = 0;
        save_state = 0;
        branches_explored = 0;
        singleton_bits = 0;
        total_leaf_count = 0;
        leaf_limit = 0;
        atomic_store_explicit(&assignment_count, 0u, memory_order_relaxed);
        for (uint i = 0; i < MAX_DFS_DEPTH; ++i) stack_branch_cols[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Trivial early-outs.
    if (p_count == 0) {
        if (tid == 0) {
            status_out[pair_idx * 4 + 0] = 0;
            status_out[pair_idx * 4 + 1] = 1; // empty pattern matches anything
            status_out[pair_idx * 4 + 2] = 0;
            status_out[pair_idx * 4 + 3] = 0;
        }
        return;
    }
    if (m_count == 0 || p_count > m_count) {
        if (tid == 0) {
            status_out[pair_idx * 4 + 0] = 0;
            status_out[pair_idx * 4 + 1] = 0; // no_match
            status_out[pair_idx * 4 + 2] = 0;
            status_out[pair_idx * 4 + 3] = 0;
        }
        return;
    }

    // Components of the target, but only if the pattern asks about them. One thread, because the
    // closure reads the array it is growing; a chain of 64 atoms is a few thousand word-ops
    // against a DFS that is about to do far more, and a pattern with no group skips it entirely.
    if (tid == 0) {
        has_groups = 0;
        for (uint i = 0; i < p_count; ++i) {
            if (p_groups[i] != 0u) { has_groups = 1; break; }
        }
        if (has_groups != 0) {
            for (uint i = 0; i < m_count; ++i) comp_mask[i] = molecule_adj[i] | (1UL << i);
            bool grew = true;
            while (grew) {
                grew = false;
                for (uint i = 0; i < m_count; ++i) {
                    bitset_t reached = comp_mask[i];
                    bitset_t grown = reached;
                    while (reached != 0) {
                        uint j = ctz(reached);
                        grown |= comp_mask[j];
                        reached ^= (1UL << j);
                    }
                    if (grown != comp_mask[i]) { comp_mask[i] = grown; grew = true; }
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Build initial candidate rows (one pattern row per thread).
    for (uint row = tid; row < 64; row += threads_per_group) {
        bitset_t bits = 0;
        if (row < p_count) {
            for (uint col = 0; col < m_count; ++col) {
                if (atom_masks_overlap(
                        p_atoms_cache, m_atoms_cache,
                        row * params.atom_word_count, col * params.atom_word_count,
                        params.atom_word_count)) {
                    bits |= (1UL << col);
                }
            }
            // A recursive predicate narrows the row to where it holds, evaluated ahead of time.
            // No roots means no pattern names one, and the predicate ids were never uploaded.
            if (params.pred_stride != 0u) {
                uint pred = p_preds[row];
                if (pred != NO_PREDICATE) bits &= roots_all[m_idx * params.pred_stride + pred];
            }
            if (row == 0) bits &= pin;
        }
        current_rows[row] = bits;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- DFS with re-refinement at each node ---
    while (true) {
        if (tid == 0) rounds = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Bit-parallel Ullmann refinement to a fixpoint (double-buffered).
        while (true) {
            if (tid == 0) {
                stop = (rounds >= (int)params.max_prunings) ? 1 : 0;
                atomic_store_explicit(&changed, 0u, memory_order_relaxed);
                atomic_store_explicit(&unsupported, 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (stop != 0) break;

            // Clamp to the threadgroup array bound (u_cols/molecule_adj are [64]).
            // The host rejects atom counts > 64, but guard here so a violated
            // invariant degrades gracefully instead of corrupting adjacent state.
            uint refine_cols = min(m_count, 64u);
            for (uint col = tid; col < refine_cols; col += threads_per_group) {
                bitset_t m_adj = molecule_adj[col];
                bitset_t u_col = 0;
                for (uint r = 0; r < p_count; ++r) {
                    if ((current_rows[r] & m_adj) != 0) {
                        u_col |= (1UL << r);
                    }
                }
                u_cols[col] = u_col;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint row = tid; row < p_count; row += threads_per_group) {
                bitset_t row_mask = current_rows[row];
                bitset_t refined = 0;
                bitset_t bits = row_mask;
                bitset_t p_neighbors = pattern_adj[row];
                while (bits != 0) {
                    uint col = ctz(bits);
                    if ((u_cols[col] & p_neighbors) == p_neighbors) {
                        refined |= (1UL << col);
                    }
                    bits &= (bits - 1);
                }
                next_rows[row] = refined;
                if (refined != row_mask) atomic_fetch_or_explicit(&changed, 1u, memory_order_relaxed);
                if (refined == 0) atomic_fetch_or_explicit(&unsupported, 1u, memory_order_relaxed);
            }
            for (uint row = p_count + tid; row < 64; row += threads_per_group) {
                next_rows[row] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0) {
                rounds += 1;
                prune_rounds += 1;
                if (atomic_load_explicit(&unsupported, memory_order_relaxed) != 0 ||
                    atomic_load_explicit(&changed, memory_order_relaxed) == 0) {
                    stop = 1;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint row = tid; row < 64; row += threads_per_group) {
                current_rows[row] = next_rows[row];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (stop != 0) break;
        }

        // Classify the refined node (serial on tid 0).
        if (tid == 0) {
            save_state = 0;
            do_leaf_eval = 0;
            if (atomic_load_explicit(&unsupported, memory_order_relaxed) != 0) {
                status_code = 0u;
            } else {
                total_leaf_count = 1;
                singleton_bits = 0;
                singleton_conflict = 0;
                ambiguous_count = 0;
                bool zero = false;
                for (uint row = 0; row < p_count; ++row) {
                    int count = popcount(current_rows[row]);
                    row_counts[row] = count;
                    if (count == 0) { status_code = 0u; zero = true; break; }
                    if (count == 1) {
                        uint col = ctz(current_rows[row]);
                        bitset_t bit = 1UL << col;
                        fixed_assignment[row] = (int)col;
                        if ((singleton_bits & bit) != 0) singleton_conflict = 1;
                        singleton_bits |= bit;
                    } else {
                        fixed_assignment[row] = -1;
                        ambiguous_rows[ambiguous_count] = (int)row;
                        ambiguous_count += 1;
                        if (total_leaf_count <= (ulong)params.max_leaf_checks / (ulong)count) {
                            total_leaf_count *= (ulong)count;
                        } else {
                            total_leaf_count = (ulong)params.max_leaf_checks + 1;
                        }
                    }
                }
                if (!zero) {
                    if (singleton_conflict != 0) {
                        status_code = 0u;
                    } else if (ambiguous_count == 0) {
                        // Fully determined assignment -> validate against bond types.
                        int assignment[64];
                        for (uint row = 0; row < p_count; ++row) assignment[row] = fixed_assignment[row];
                        leaf_checks += 1;
                        if (final_assignment_valid(assignment, pattern_adj, molecule_adj,
                                                   p_bonds, m_bonds, p_groups, comp_mask,
                                                   has_groups, p_count)) {
                            uint slot = atomic_fetch_add_explicit(&assignment_count, 1u, memory_order_relaxed);
                            if (slot < params.max_assignments && params.write_assignments != 0) {
                                device int* out = assignments_out + (pair_idx * params.max_assignments * 64) + (slot * 64);
                                for (uint row = 0; row < p_count; ++row) out[row] = assignment[row];
                            }
                            if (slot + 1 >= params.max_assignments) {
                                status_code = 1u; // collected enough
                            } else {
                                status_code = 0u; // force backtrack to find more
                            }
                        } else {
                            status_code = 0u;
                        }
                    } else if (prefix_depth < (int)max_prefix_depth) {
                        // Branch on the ambiguous row with fewest candidates.
                        int branch_row = ambiguous_rows[0];
                        int min_count = row_counts[branch_row];
                        for (int i = 1; i < ambiguous_count; ++i) {
                            int r = ambiguous_rows[i];
                            if (row_counts[r] < min_count) { min_count = row_counts[r]; branch_row = r; }
                        }
                        stack_branch_row[prefix_depth] = branch_row;
                        stack_branch_cols[prefix_depth] = current_rows[branch_row];
                        save_state = 1;
                        status_code = 2u;
                    } else {
                        // Depth budget reached: enumerate the remaining product space.
                        leaf_limit = total_leaf_count;
                        do_leaf_eval = 1;
                    }
                }
            }

            if (status_code == 0u && prefix_depth == 0) {
                dfs_stop = 1;
            } else if (status_code == 1u) {
                dfs_stop = 1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (dfs_stop != 0) break;

        // Save full candidate state for this depth before descending.
        if (save_state != 0) {
            for (uint row = tid; row < 64; row += threads_per_group) {
                stack_rows[prefix_depth][row] = current_rows[row];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Parallel leaf enumeration (depth budget exhausted but rows still ambiguous).
        if (do_leaf_eval != 0) {
            ulong remaining = ((ulong)params.max_leaf_checks > (ulong)leaf_checks)
                ? (ulong)params.max_leaf_checks - (ulong)leaf_checks : 0;
            ulong chunk_limit = (leaf_limit < remaining) ? leaf_limit : remaining;
            if (tid == 0 && chunk_limit < leaf_limit) search_complete = 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (ulong chunk_start = 0; chunk_start < chunk_limit; chunk_start += (ulong)threads_per_group) {
                ulong leaf_index = chunk_start + (ulong)tid;
                bool valid = leaf_index < chunk_limit;
                int assignment[64];
                if (valid && atomic_load_explicit(&assignment_count, memory_order_relaxed) < params.max_assignments) {
                    for (uint row = 0; row < p_count; ++row) assignment[row] = fixed_assignment[row];
                    bitset_t used = singleton_bits;
                    ulong remainder = leaf_index;
                    for (int depth = ambiguous_count - 1; depth >= 0; --depth) {
                        int row = ambiguous_rows[depth];
                        int count = row_counts[row];
                        int ordinal = (int)(remainder % (ulong)count);
                        remainder /= (ulong)count;
                        int col = nth_set_bit_index(current_rows[row], ordinal);
                        if (col < 0) { valid = false; break; }
                        bitset_t bit = 1UL << col;
                        if ((used & bit) != 0) { valid = false; break; }
                        used |= bit;
                        assignment[row] = col;
                    }
                    if (valid) {
                        valid = final_assignment_valid(assignment, pattern_adj, molecule_adj,
                                                       p_bonds, m_bonds, p_groups, comp_mask,
                                                       has_groups, p_count);
                    }
                } else {
                    valid = false;
                }

                if (valid) {
                    uint slot = atomic_fetch_add_explicit(&assignment_count, 1u, memory_order_relaxed);
                    if (slot < params.max_assignments && params.write_assignments != 0) {
                        device int* out = assignments_out + (pair_idx * params.max_assignments * 64) + (slot * 64);
                        for (uint row = 0; row < p_count; ++row) out[row] = assignment[row];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (atomic_load_explicit(&assignment_count, memory_order_relaxed) >= params.max_assignments) break;
            }

            if (tid == 0) {
                leaf_checks += (int)chunk_limit;
                uint ac = atomic_load_explicit(&assignment_count, memory_order_relaxed);
                if (ac >= params.max_assignments) {
                    status_code = 1u;          // collected enough
                } else if (chunk_limit < leaf_limit) {
                    status_code = 2u;          // truncated: result unknown for this subtree
                } else {
                    status_code = 0u;          // subtree fully enumerated -> backtrack
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Backtrack / advance to next sibling branch (serial on tid 0).
        if (tid == 0) {
            if (status_code == 1u) {
                dfs_stop = 1;
            } else {
                while (prefix_depth >= 0 && stack_branch_cols[prefix_depth] == 0) {
                    prefix_depth -= 1;
                }
                if (prefix_depth < 0) {
                    uint ac = atomic_load_explicit(&assignment_count, memory_order_relaxed);
                    status_code = (search_complete == 0 && ac == 0) ? 2u : 0u;
                    dfs_stop = 1;
                } else {
                    branches_explored += 1;
                    status_code = 2u;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (dfs_stop != 0) break;

        // Restore the saved state at the resumed depth and fix the next candidate bit.
        for (uint row = tid; row < 64; row += threads_per_group) {
            current_rows[row] = stack_rows[prefix_depth][row];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            int branch_row = stack_branch_row[prefix_depth];
            bitset_t bit = stack_branch_cols[prefix_depth] & (~stack_branch_cols[prefix_depth] + 1);
            current_rows[branch_row] = bit;
            stack_branch_cols[prefix_depth] ^= bit;
            prefix_depth += 1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        uint final_count = atomic_load_explicit(&assignment_count, memory_order_relaxed);
        uint stored_count = min(final_count, params.max_assignments);
        int out_status;
        if (final_count > 0) {
            out_status = 1; // match
        } else if (search_complete != 0) {
            out_status = 0; // no_match (search fully explored)
        } else {
            out_status = 2; // unknown (budget exhausted)
        }
        status_out[pair_idx * 4 + 0] = (int)stored_count;
        status_out[pair_idx * 4 + 1] = out_status;
        status_out[pair_idx * 4 + 2] = prune_rounds;
        status_out[pair_idx * 4 + 3] = leaf_checks;
    }
}

// Per (probe, molecule): the atoms one packed query row admits. A probe is a box -- one mask per
// property, all-ones where it says nothing -- so a term of any width is one row.
kernel void probe_bits(
    device const uint* probes [[buffer(0)]],
    device const uint* molecule_atoms_all [[buffer(1)]],
    device const uint* molecule_atom_counts [[buffer(2)]],
    device ulong* probe_bits_out [[buffer(3)]],
    constant PredParams& pp [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    uint p = gid % pp.probe_count;
    uint m = gid / pp.probe_count;
    if (m >= pp.n_mols) return;
    uint stride = pp.bucket_size * pp.atom_word_count;
    bitset_t bits = 0;
    uint n = min(molecule_atom_counts[m], 64u);
    for (uint a = 0; a < n; ++a) {
        if (atom_masks_overlap(probes + p * pp.atom_word_count, molecule_atoms_all + m * stride,
                               0u, a * pp.atom_word_count, pp.atom_word_count)) {
            bits |= (1UL << a);
        }
    }
    probe_bits_out[m * pp.probe_count + p] = bits;
}

// Per (residual, molecule): an AND of ORs of conjunctions, every value a 64-atom bitset, so
// evaluation is a few word operations however wide the expression. Three offset arrays:
// residual r owns clauses [clause_start[r], clause_start[r+1]), clause c owns terms
// [term_start[c], term_start[c+1]), term t owns factors [factor_start[t], factor_start[t+1]).
// A factor is a probe (a box of primitives) or a recursive body's roots, the latter maybe negated.
kernel void eval_residuals(
    device const uint* clause_start [[buffer(0)]],
    device const uint* term_start [[buffer(1)]],
    device const uint* factors [[buffer(2)]],
    device const ulong* probe_bits_all [[buffer(3)]],
    device const ulong* sub_roots [[buffer(4)]],
    device ulong* roots_all [[buffer(5)]],
    device const uint* molecule_atom_counts [[buffer(6)]],
    constant PredParams& pp [[buffer(7)]],
    device const uint* factor_start [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    uint r = pp.first + gid % pp.count;
    uint m = gid / pp.count;
    if (m >= pp.n_mols) return;
    uint n = molecule_atom_counts[m];
    bitset_t valid = n >= 64u ? ~0UL : ((1UL << n) - 1UL);
    bitset_t held = valid;
    for (uint c = clause_start[r]; c < clause_start[r + 1]; ++c) {
        bitset_t any = 0;
        for (uint t = term_start[c]; t < term_start[c + 1]; ++t) {
            bitset_t all = valid;
            for (uint f = factor_start[t]; f < factor_start[t + 1]; ++f) {
                uint factor = factors[f];
                uint index = factor & TERM_INDEX;
                bitset_t x = (factor & TERM_SUBPATTERN) != 0u
                    ? sub_roots[m * pp.sub_stride + index]
                    : probe_bits_all[m * pp.probe_count + index];
                if ((factor & TERM_NEGATED) != 0u) x = ~x & valid;
                all &= x;
            }
            any |= all;
        }
        held &= any;
    }
    roots_all[m * pp.pred_stride + r] = held;
}

// Per (body of the level, molecule): the roots its first atom can take -- its masks and its own
// predicate -- appended to the level's pair list. Only these become threadgroups.
kernel void level_candidates(
    device const uint* body_atoms [[buffer(0)]],
    device const uint* body_preds [[buffer(1)]],
    device const uint* molecule_atoms_all [[buffer(2)]],
    device const uint* molecule_atom_counts [[buffer(3)]],
    device const ulong* roots_all [[buffer(4)]],
    device atomic_uint* level_counts [[buffer(5)]],
    device uint2* pair_list [[buffer(6)]],
    constant PredParams& pp [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    uint j = gid % pp.count;
    uint m = gid / pp.count;
    if (m >= pp.n_mols) return;
    uint body = pp.first + j;
    uint stride = pp.bucket_size * pp.atom_word_count;
    uint n = min(molecule_atom_counts[m], 64u);
    bitset_t viable = 0;
    for (uint a = 0; a < n; ++a) {
        if (atom_masks_overlap(body_atoms + body * stride, molecule_atoms_all + m * stride,
                               0u, a * pp.atom_word_count, pp.atom_word_count)) {
            viable |= (1UL << a);
        }
    }
    uint pred0 = body_preds[body * pp.bucket_size];
    if (pred0 != NO_PREDICATE) viable &= roots_all[m * pp.pred_stride + pred0];
    uint found = popcount(viable);
    if (found == 0u) return;
    uint slot = atomic_fetch_add_explicit(&level_counts[pp.level], found, memory_order_relaxed);
    while (viable != 0) {
        uint a = ctz(viable);
        pair_list[slot++] = uint2(body, (m << 6) | a);
        viable &= viable - 1UL;
    }
}

// One thread per dispatch slice: how many of the level's listed pairs the slice runs.
kernel void level_dispatch_args(
    device const uint* level_counts [[buffer(0)]],
    device uint* args [[buffer(1)]],
    constant PredParams& pp [[buffer(2)]],
    uint slice [[thread_position_in_grid]]) {
    uint listed = level_counts[pp.level];
    uint base = slice * pp.max_pairs;
    args[slice * 3 + 0] = listed > base ? min(listed - base, pp.max_pairs) : 0u;
    args[slice * 3 + 1] = 1u;
    args[slice * 3 + 2] = 1u;
}

// Per listed pair: a match sets its root's bit in the body's roots, through the two 32-bit
// halves since there is no 64-bit atomic OR. A search that ran out of budget (status 2) is
// neither: it is counted, and the host refuses the roots rather than read it as absent --
// a negated body would turn that absence into a match.
kernel void fold_roots(
    device const int* status [[buffer(0)]],
    device const uint2* pair_list [[buffer(1)]],
    device const uint* level_counts [[buffer(2)]],
    device atomic_uint* sub_roots_words [[buffer(3)]],
    constant PredParams& pp [[buffer(4)]],
    device atomic_uint* unresolved [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= level_counts[pp.level]) return;
    const int outcome = status[i * 4 + 1];
    if (outcome == 2) {
        atomic_fetch_add_explicit(unresolved, 1u, memory_order_relaxed);
        return;
    }
    if (outcome != 1) return;
    uint2 entry = pair_list[i];
    uint m = entry.y >> 6;
    uint a = entry.y & 63u;
    uint word = (m * pp.sub_stride + entry.x) * 2u + (a >> 5);
    atomic_fetch_or_explicit(&sub_roots_words[word], 1u << (a & 31u), memory_order_relaxed);
}
)";

extern "C" int vesma_metal_probe(char* name_out, int name_cap, uint32_t* max_threads_out) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            snprintf(g_last_error, sizeof(g_last_error), "No Metal device found");
            return 0;
        }
        if (name_out) snprintf(name_out, name_cap, "%s", [[device name] UTF8String]);
        if (max_threads_out) *max_threads_out = (uint32_t)[device maxThreadsPerThreadgroup].width;
        return 1;
    }
}

static id<MTLDevice> s_device = nil;
static id<MTLCommandQueue> s_queue = nil;
static id<MTLComputePipelineState> s_pipeline = nil;
static id<MTLComputePipelineState> s_probe_pipeline = nil;
static id<MTLComputePipelineState> s_residual_pipeline = nil;
static id<MTLComputePipelineState> s_candidates_pipeline = nil;
static id<MTLComputePipelineState> s_args_pipeline = nil;
static id<MTLComputePipelineState> s_fold_pipeline = nil;
static bool s_init_failed = false;

static id<MTLComputePipelineState> make_pipeline(id<MTLLibrary> library, NSString* name) {
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) {
        snprintf(g_last_error, sizeof(g_last_error), "Function %s not found", [name UTF8String]);
        return nil;
    }
    id<MTLComputePipelineState> pipeline = [s_device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        snprintf(g_last_error, sizeof(g_last_error), "Pipeline creation failed: %s", [[error localizedDescription] UTF8String]);
    }
    return pipeline;
}

// One device, queue and library for every entry point here, built once.
static bool ensure_pipelines(void) {
    static dispatch_once_t s_once;
    dispatch_once(&s_once, ^{
        @autoreleasepool {
            s_device = MTLCreateSystemDefaultDevice();
            if (!s_device) {
                snprintf(g_last_error, sizeof(g_last_error), "No Metal device found");
                s_init_failed = true;
                return;
            }
            NSError* error = nil;
            id<MTLLibrary> library = [s_device newLibraryWithSource:[NSString stringWithUTF8String:kShaderSource] options:nil error:&error];
            if (!library) {
                snprintf(g_last_error, sizeof(g_last_error), "Shader compilation failed: %s", [[error localizedDescription] UTF8String]);
                s_init_failed = true;
                return;
            }
            s_pipeline = make_pipeline(library, @"match_fused");
            s_probe_pipeline = make_pipeline(library, @"probe_bits");
            s_residual_pipeline = make_pipeline(library, @"eval_residuals");
            s_candidates_pipeline = make_pipeline(library, @"level_candidates");
            s_args_pipeline = make_pipeline(library, @"level_dispatch_args");
            s_fold_pipeline = make_pipeline(library, @"fold_roots");
            s_queue = [s_device newCommandQueue];
            if (!s_pipeline || !s_probe_pipeline || !s_residual_pipeline || !s_candidates_pipeline
                || !s_args_pipeline || !s_fold_pipeline || !s_queue) {
                if (!s_queue) snprintf(g_last_error, sizeof(g_last_error), "Command queue creation failed");
                s_init_failed = true;
            }
        }
    });
    return !s_init_failed;
}

extern "C" int vesma_metal_match(
    const uint32_t* pattern_atoms,
    const uint32_t* molecule_atoms,
    const uint64_t* pattern_rows,
    const uint64_t* molecule_rows,
    const uint32_t* pair_pattern_indices,
    const uint32_t* pair_molecule_indices,
    const uint32_t* pattern_atom_counts,
    const uint32_t* molecule_atom_counts,
    const uint8_t* pattern_bonds,
    const uint8_t* molecule_bonds,
    // Pattern side only. The molecule's components are a closure over adjacency the kernel
    // already holds, so there is nothing to upload for them.
    const uint32_t* pattern_groups,
    // NULL when no pattern names a predicate; then pred_stride must be 0.
    const uint32_t* pattern_preds,
    // [molecule_count * pred_stride]; may be NULL when pred_stride is 0.
    const uint64_t* roots,
    int pred_stride,
    int bucket_size,
    int atom_word_count,
    int pair_count,
    int max_prunings,
    int max_leaf_checks,
    int max_assignments,
    int write_assignments,
    int copy_valid_only,
    int threads_per_group,
    int max_prefix_depth,
    int pattern_count,
    uint64_t pattern_token,
    int32_t* assignments_out,
    int32_t* status_out,
    double* timings_out,
    uint64_t* bytes_out) {

    if (!ensure_pipelines()) return 0;

    @autoreleasepool {
        id<MTLDevice> device = s_device;
        id<MTLComputePipelineState> pipeline = s_pipeline;
        id<MTLCommandQueue> queue = s_queue;

        if (threads_per_group <= 0) {
            snprintf(g_last_error, sizeof(g_last_error), "threads_per_group must be positive");
            return 0;
        }
        if ((NSUInteger)threads_per_group > pipeline.maxTotalThreadsPerThreadgroup) {
            snprintf(g_last_error, sizeof(g_last_error), "threads_per_group exceeds Metal device limit");
            return 0;
        }

        uint32_t max_p_idx = 0;
        uint32_t max_m_idx = 0;
        for (int i = 0; i < pair_count; ++i) {
            if (pair_pattern_indices[i] > max_p_idx) max_p_idx = pair_pattern_indices[i];
            if (pair_molecule_indices[i] > max_m_idx) max_m_idx = pair_molecule_indices[i];
        }
        uint32_t unique_p_count = max_p_idx + 1;
        uint32_t unique_m_count = max_m_idx + 1;
        if (pattern_token != 0) {
            if (pattern_count < (int)unique_p_count) {
                snprintf(g_last_error, sizeof(g_last_error), "pattern_count is smaller than the largest pair index");
                return 0;
            }
            unique_p_count = (uint32_t)pattern_count;
        }

        NSUInteger p_atoms_size = (NSUInteger)(unique_p_count * bucket_size * atom_word_count * sizeof(uint32_t));
        NSUInteger m_atoms_size = (NSUInteger)(unique_m_count * bucket_size * atom_word_count * sizeof(uint32_t));
        NSUInteger p_rows_size = (NSUInteger)(unique_p_count * bucket_size * sizeof(uint64_t));
        NSUInteger m_rows_size = (NSUInteger)(unique_m_count * bucket_size * sizeof(uint64_t));
        NSUInteger p_bonds_size = (NSUInteger)(unique_p_count * bucket_size * bucket_size * sizeof(uint8_t));
        NSUInteger m_bonds_size = (NSUInteger)(unique_m_count * bucket_size * bucket_size * sizeof(uint8_t));
        NSUInteger p_groups_size = (NSUInteger)(unique_p_count * bucket_size * sizeof(uint32_t));

        const auto buffer_started = std::chrono::steady_clock::now();

        id<MTLBuffer> buf_p_atoms = nil;
        id<MTLBuffer> buf_p_rows = nil;
        id<MTLBuffer> buf_p_bonds = nil;
        id<MTLBuffer> buf_p_counts = nil;
        id<MTLBuffer> buf_p_groups = nil;
        id<MTLBuffer> buf_p_preds = nil;
        bool uploaded_patterns = false;
        // Narrow deliberately: see the lock-width note in `metal_retro.mm`'s construct entry.
        // Cached pattern buffers are immutable once uploaded and a concurrent replacement makes a
        // new buffer, so releasing before the dispatch is safe here. It would not be for a buffer
        // the kernel writes.
        @synchronized (device) {
            if (pattern_token != 0 && pattern_token == s_pattern_token) {
                if (s_pattern_count != unique_p_count) {
                    snprintf(g_last_error, sizeof(g_last_error), "pattern token reused with a different pattern count");
                    return 0;
                }
                buf_p_atoms = s_pattern_atoms;
                buf_p_rows = s_pattern_rows;
                buf_p_bonds = s_pattern_bonds;
                buf_p_counts = s_pattern_counts;
                buf_p_groups = s_pattern_groups;
                buf_p_preds = s_pattern_preds;
            } else {
                buf_p_atoms = [device newBufferWithBytes:pattern_atoms length:p_atoms_size options:MTLResourceStorageModeShared];
                buf_p_rows = [device newBufferWithBytes:pattern_rows length:p_rows_size options:MTLResourceStorageModeShared];
                buf_p_bonds = [device newBufferWithBytes:pattern_bonds length:p_bonds_size options:MTLResourceStorageModeShared];
                buf_p_counts = [device newBufferWithBytes:pattern_atom_counts length:(NSUInteger)unique_p_count * sizeof(uint32_t) options:MTLResourceStorageModeShared];
                buf_p_groups = [device newBufferWithBytes:pattern_groups length:p_groups_size options:MTLResourceStorageModeShared];
                buf_p_preds = pattern_preds
                    ? [device newBufferWithBytes:pattern_preds length:p_groups_size options:MTLResourceStorageModeShared]
                    : [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
                if (!buf_p_atoms || !buf_p_rows || !buf_p_bonds || !buf_p_counts || !buf_p_groups || !buf_p_preds) {
                    snprintf(g_last_error, sizeof(g_last_error), "Could not allocate Metal pattern buffers");
                    return 0;
                }
                uploaded_patterns = true;
                if (pattern_token != 0) {
                    s_pattern_atoms = buf_p_atoms;
                    s_pattern_rows = buf_p_rows;
                    s_pattern_bonds = buf_p_bonds;
                    s_pattern_counts = buf_p_counts;
                    s_pattern_groups = buf_p_groups;
                    s_pattern_preds = buf_p_preds;
                    s_pattern_count = unique_p_count;
                    s_pattern_token = pattern_token;
                }
            }
        }
        id<MTLBuffer> buf_m_atoms = [device newBufferWithBytes:molecule_atoms length:m_atoms_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_m_rows = [device newBufferWithBytes:molecule_rows length:m_rows_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_m_bonds = [device newBufferWithBytes:molecule_bonds length:m_bonds_size options:MTLResourceStorageModeShared];

        id<MTLBuffer> buf_pair_p_idx = [device newBufferWithBytes:pair_pattern_indices length:pair_count * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_pair_m_idx = [device newBufferWithBytes:pair_molecule_indices length:pair_count * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_m_counts = [device newBufferWithBytes:molecule_atom_counts length:unique_m_count * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        // Per call, never cached: the roots belong to this molecule batch.
        // A one-word placeholder stands in for an unused buffer; the kernel never reads it.
        uint64_t unused_word = 0;
        NSUInteger roots_size = (NSUInteger)unique_m_count * (NSUInteger)pred_stride * sizeof(uint64_t);
        id<MTLBuffer> buf_roots = (pred_stride > 0 && roots)
            ? [device newBufferWithBytes:roots length:roots_size options:MTLResourceStorageModeShared]
            : [device newBufferWithBytes:&unused_word length:sizeof(uint64_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_unused = [device newBufferWithBytes:&unused_word length:sizeof(uint64_t) options:MTLResourceStorageModeShared];

        NSUInteger assignments_size = write_assignments
            ? (NSUInteger)pair_count * max_assignments * 64 * sizeof(int32_t)
            : sizeof(int32_t);
        id<MTLBuffer> buf_assignments = [device newBufferWithLength:assignments_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_status = [device newBufferWithLength:(NSUInteger)pair_count * 4 * sizeof(int32_t) options:MTLResourceStorageModeShared];

        Params params = {
            (uint32_t)bucket_size, (uint32_t)atom_word_count, (uint32_t)max_prunings,
            (uint32_t)max_leaf_checks, (uint32_t)max_assignments,
            (uint32_t)write_assignments, (uint32_t)pair_count, (uint32_t)max_prefix_depth,
            (uint32_t)pred_stride, 0u, 0u
        };
        id<MTLBuffer> buf_params = [device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];

        const auto buffers_ready = std::chrono::steady_clock::now();

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        commandBuffer.label = @"vesma.match_fused";
        encoder.label = @"match_fused";
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buf_p_atoms offset:0 atIndex:0];
        [encoder setBuffer:buf_m_atoms offset:0 atIndex:1];
        [encoder setBuffer:buf_p_rows offset:0 atIndex:2];
        [encoder setBuffer:buf_m_rows offset:0 atIndex:3];
        [encoder setBuffer:buf_pair_p_idx offset:0 atIndex:4];
        [encoder setBuffer:buf_pair_m_idx offset:0 atIndex:5];
        [encoder setBuffer:buf_p_counts offset:0 atIndex:6];
        [encoder setBuffer:buf_m_counts offset:0 atIndex:7];
        [encoder setBuffer:buf_assignments offset:0 atIndex:8];
        [encoder setBuffer:buf_status offset:0 atIndex:9];
        [encoder setBuffer:buf_params offset:0 atIndex:10];
        [encoder setBuffer:buf_p_bonds offset:0 atIndex:11];
        [encoder setBuffer:buf_m_bonds offset:0 atIndex:12];
        [encoder setBuffer:buf_p_groups offset:0 atIndex:13];
        [encoder setBuffer:buf_p_preds offset:0 atIndex:14];
        [encoder setBuffer:buf_roots offset:0 atIndex:15];
        [encoder setBuffer:buf_unused offset:0 atIndex:17];  // pair list: unread outside list mode

        NSUInteger threadsPerGroup = (NSUInteger)threads_per_group;

        [encoder dispatchThreadgroups:MTLSizeMake(pair_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        const auto kernel_done = std::chrono::steady_clock::now();

        if ([commandBuffer status] == MTLCommandBufferStatusError) {
            snprintf(g_last_error, sizeof(g_last_error), "Command buffer error: %s", [[[commandBuffer error] localizedDescription] UTF8String]);
            return 0;
        }

        memcpy(status_out, [buf_status contents], (NSUInteger)pair_count * 4 * sizeof(int32_t));
        NSUInteger assignment_bytes_copied = 0;
        if (write_assignments) {
            if (copy_valid_only) {
                const int32_t* source = (const int32_t*)[buf_assignments contents];
                for (int pair = 0; pair < pair_count; ++pair) {
                    const int32_t count = status_out[pair * 4];
                    if (count <= 0) continue;
                    const NSUInteger bytes = (NSUInteger)count * 64u * sizeof(int32_t);
                    const NSUInteger offset = (NSUInteger)pair * max_assignments * 64u;
                    memcpy(assignments_out + offset, source + offset, bytes);
                    assignment_bytes_copied += bytes;
                }
            } else {
                assignment_bytes_copied = assignments_size;
                memcpy(assignments_out, [buf_assignments contents], assignments_size);
            }
        }

        if (timings_out) {
            const auto copy_done = std::chrono::steady_clock::now();
            timings_out[0] = std::chrono::duration<double>(buffers_ready - buffer_started).count();
            timings_out[1] = std::chrono::duration<double>(kernel_done - buffers_ready).count();
            timings_out[2] = std::chrono::duration<double>(copy_done - kernel_done).count();
            timings_out[3] = commandBuffer.GPUEndTime - commandBuffer.GPUStartTime;
        }
        if (bytes_out) {
            bytes_out[0] = uploaded_patterns ? p_atoms_size + p_rows_size + p_bonds_size + (pattern_preds ? 2u : 1u) * p_groups_size + (NSUInteger)unique_p_count * sizeof(uint32_t) : 0u;
            bytes_out[1] = m_atoms_size + m_rows_size + m_bonds_size + (NSUInteger)unique_m_count * sizeof(uint32_t) + (pred_stride > 0 && roots ? roots_size : 0u);
            bytes_out[2] = (NSUInteger)pair_count * 2u * sizeof(uint32_t);
            bytes_out[3] = (NSUInteger)pair_count * 4u * sizeof(int32_t) + assignment_bytes_copied;
        }

        return 1;
    }
}

static id<MTLBuffer> shared_buffer(id<MTLDevice> device, const void* bytes, NSUInteger length) {
    // Metal refuses a zero-length buffer; an empty input still needs something to bind.
    static const uint64_t zero = 0;
    if (length == 0 || bytes == NULL) {
        return [device newBufferWithBytes:&zero length:sizeof(zero) options:MTLResourceStorageModeShared];
    }
    return [device newBufferWithBytes:bytes length:length options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> scratch_buffer(id<MTLDevice> device, NSUInteger length) {
    return [device newBufferWithLength:(length ? length : sizeof(uint64_t)) options:MTLResourceStorageModePrivate];
}

static void dispatch_threads(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline, NSUInteger count) {
    NSUInteger width = MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
}

// Every recursive-SMARTS predicate of a table over one packed molecule batch, in ONE command
// buffer: probe bits, then per level the pinned body matches, their fold into root bitsets, and
// that level's residual expressions. The host reads back only the final (molecule, residual)
// roots. Levels are ranges: residuals of level k are [residual_level_start[k],
// residual_level_start[k+1]) for k in 0..depth, bodies likewise for k in 1..depth.
extern "C" int vesma_metal_predicate_roots(
    const uint32_t* molecule_atoms,
    const uint64_t* molecule_rows,
    const uint8_t* molecule_bonds,
    const uint32_t* molecule_atom_counts,
    int n_mols,
    const uint32_t* body_atoms,
    const uint64_t* body_rows,
    const uint8_t* body_bonds,
    const uint32_t* body_groups,
    const uint32_t* body_preds,
    const uint32_t* body_atom_counts,
    int n_bodies,
    const uint32_t* probes,
    int n_probes,
    const uint32_t* clause_start,
    const uint32_t* term_start,
    const uint32_t* factor_start,
    const uint32_t* factors,
    int n_residuals,
    int n_clauses,
    int n_terms,
    int n_factors,
    const uint32_t* residual_level_start,
    const uint32_t* body_level_start,
    int depth,
    int bucket_size,
    int atom_word_count,
    int max_prunings,
    int max_leaf_checks,
    int threads_per_group,
    int max_prefix_depth,
    int max_pairs_per_dispatch,
    uint64_t* roots_out,
    uint32_t* unresolved_out,
    double* timings_out) {

    *unresolved_out = 0;
    if (!ensure_pipelines()) return 0;
    if (n_mols <= 0 || n_residuals <= 0) return 1;
    if ((NSUInteger)threads_per_group > s_pipeline.maxTotalThreadsPerThreadgroup || threads_per_group <= 0) {
        snprintf(g_last_error, sizeof(g_last_error), "threads_per_group exceeds Metal device limit");
        return 0;
    }

    @autoreleasepool {
        id<MTLDevice> device = s_device;
        const auto started = std::chrono::steady_clock::now();

        // A level lists at most every body at every atom; the real count stays on the device.
        uint64_t total_atoms = 0;
        for (int m = 0; m < n_mols; ++m) total_atoms += molecule_atom_counts[m];
        uint64_t max_level_pairs = 0;
        for (int k = 1; k <= depth; ++k) {
            uint64_t pairs = (uint64_t)(body_level_start[k + 1] - body_level_start[k]) * total_atoms;
            if (pairs > max_level_pairs) max_level_pairs = pairs;
        }
        uint64_t max_slices = (max_level_pairs + (uint64_t)max_pairs_per_dispatch - 1) / (uint64_t)max_pairs_per_dispatch;

        NSUInteger mol_stride = (NSUInteger)bucket_size * atom_word_count * sizeof(uint32_t);
        id<MTLBuffer> buf_m_atoms = shared_buffer(device, molecule_atoms, (NSUInteger)n_mols * mol_stride);
        id<MTLBuffer> buf_m_rows = shared_buffer(device, molecule_rows, (NSUInteger)n_mols * bucket_size * sizeof(uint64_t));
        id<MTLBuffer> buf_m_bonds = shared_buffer(device, molecule_bonds, (NSUInteger)n_mols * bucket_size * bucket_size);
        id<MTLBuffer> buf_m_counts = shared_buffer(device, molecule_atom_counts, (NSUInteger)n_mols * sizeof(uint32_t));
        id<MTLBuffer> buf_b_atoms = shared_buffer(device, body_atoms, (NSUInteger)n_bodies * mol_stride);
        id<MTLBuffer> buf_b_rows = shared_buffer(device, body_rows, (NSUInteger)n_bodies * bucket_size * sizeof(uint64_t));
        id<MTLBuffer> buf_b_bonds = shared_buffer(device, body_bonds, (NSUInteger)n_bodies * bucket_size * bucket_size);
        id<MTLBuffer> buf_b_groups = shared_buffer(device, body_groups, (NSUInteger)n_bodies * bucket_size * sizeof(uint32_t));
        id<MTLBuffer> buf_b_preds = shared_buffer(device, body_preds, (NSUInteger)n_bodies * bucket_size * sizeof(uint32_t));
        id<MTLBuffer> buf_b_counts = shared_buffer(device, body_atom_counts, (NSUInteger)n_bodies * sizeof(uint32_t));
        id<MTLBuffer> buf_probes = shared_buffer(device, probes, (NSUInteger)n_probes * atom_word_count * sizeof(uint32_t));
        id<MTLBuffer> buf_clause_start = shared_buffer(device, clause_start, (NSUInteger)(n_residuals + 1) * sizeof(uint32_t));
        id<MTLBuffer> buf_term_start = shared_buffer(device, term_start, (NSUInteger)(n_clauses + 1) * sizeof(uint32_t));
        id<MTLBuffer> buf_factor_start = shared_buffer(device, factor_start, (NSUInteger)(n_terms + 1) * sizeof(uint32_t));
        id<MTLBuffer> buf_factors = shared_buffer(device, factors, (NSUInteger)n_factors * sizeof(uint32_t));
        id<MTLBuffer> buf_pair_list = scratch_buffer(device, (NSUInteger)max_level_pairs * 2 * sizeof(uint32_t));
        id<MTLBuffer> buf_level_counts = scratch_buffer(device, ((NSUInteger)depth + 1) * sizeof(uint32_t));
        id<MTLBuffer> buf_args = scratch_buffer(device, (NSUInteger)max_slices * 3 * sizeof(uint32_t));
        id<MTLBuffer> buf_probe_bits = scratch_buffer(device, (NSUInteger)n_mols * n_probes * sizeof(uint64_t));
        id<MTLBuffer> buf_sub_roots = scratch_buffer(device, (NSUInteger)n_mols * n_bodies * sizeof(uint64_t));
        id<MTLBuffer> buf_status = scratch_buffer(device, (NSUInteger)max_level_pairs * 4 * sizeof(int32_t));
        id<MTLBuffer> buf_roots = [device newBufferWithLength:(NSUInteger)n_mols * n_residuals * sizeof(uint64_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_unused = shared_buffer(device, NULL, 0);
        id<MTLBuffer> buf_unresolved = shared_buffer(device, NULL, 0);  // one zeroed word
        if (!buf_m_atoms || !buf_b_atoms || !buf_probe_bits || !buf_sub_roots || !buf_status || !buf_roots) {
            snprintf(g_last_error, sizeof(g_last_error), "Could not allocate Metal predicate buffers");
            return 0;
        }

        PredParams base = {
            (uint32_t)n_mols, (uint32_t)n_probes, 0u, 0u, (uint32_t)n_residuals,
            (uint32_t)n_bodies, (uint32_t)atom_word_count, (uint32_t)bucket_size, 0u,
            (uint32_t)max_pairs_per_dispatch
        };

        const auto buffers_ready = std::chrono::steady_clock::now();
        id<MTLCommandBuffer> commandBuffer = [s_queue commandBuffer];
        commandBuffer.label = @"vesma.predicate_roots";
        {
            id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
            [blit fillBuffer:buf_sub_roots range:NSMakeRange(0, buf_sub_roots.length) value:0];
            [blit fillBuffer:buf_level_counts range:NSMakeRange(0, buf_level_counts.length) value:0];
            [blit endEncoding];
        }

        if (n_probes > 0) {
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            encoder.label = @"probe_bits";
            [encoder setComputePipelineState:s_probe_pipeline];
            [encoder setBuffer:buf_probes offset:0 atIndex:0];
            [encoder setBuffer:buf_m_atoms offset:0 atIndex:1];
            [encoder setBuffer:buf_m_counts offset:0 atIndex:2];
            [encoder setBuffer:buf_probe_bits offset:0 atIndex:3];
            [encoder setBytes:&base length:sizeof(base) atIndex:4];
            dispatch_threads(encoder, s_probe_pipeline, (NSUInteger)n_probes * n_mols);
            [encoder endEncoding];
        }

        for (int k = 0; k <= depth; ++k) {
            uint32_t b_first = k >= 1 ? body_level_start[k] : 0u;
            uint32_t b_count = k >= 1 ? body_level_start[k + 1] - body_level_start[k] : 0u;
            uint64_t level_capacity = (uint64_t)b_count * total_atoms;
            if (level_capacity > 0) {
                PredParams pp = base;
                pp.first = b_first;
                pp.count = b_count;
                pp.level = (uint32_t)k;
                uint64_t slices = (level_capacity + (uint64_t)max_pairs_per_dispatch - 1) / (uint64_t)max_pairs_per_dispatch;

                id<MTLComputeCommandEncoder> listing = [commandBuffer computeCommandEncoder];
                listing.label = @"level_candidates";
                [listing setComputePipelineState:s_candidates_pipeline];
                [listing setBuffer:buf_b_atoms offset:0 atIndex:0];
                [listing setBuffer:buf_b_preds offset:0 atIndex:1];
                [listing setBuffer:buf_m_atoms offset:0 atIndex:2];
                [listing setBuffer:buf_m_counts offset:0 atIndex:3];
                [listing setBuffer:buf_roots offset:0 atIndex:4];
                [listing setBuffer:buf_level_counts offset:0 atIndex:5];
                [listing setBuffer:buf_pair_list offset:0 atIndex:6];
                [listing setBytes:&pp length:sizeof(pp) atIndex:7];
                dispatch_threads(listing, s_candidates_pipeline, (NSUInteger)b_count * n_mols);
                [listing endEncoding];

                id<MTLComputeCommandEncoder> sizing = [commandBuffer computeCommandEncoder];
                sizing.label = @"level_dispatch_args";
                [sizing setComputePipelineState:s_args_pipeline];
                [sizing setBuffer:buf_level_counts offset:0 atIndex:0];
                [sizing setBuffer:buf_args offset:0 atIndex:1];
                [sizing setBytes:&pp length:sizeof(pp) atIndex:2];
                dispatch_threads(sizing, s_args_pipeline, (NSUInteger)slices);
                [sizing endEncoding];

                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                encoder.label = @"match_level";
                [encoder setComputePipelineState:s_pipeline];
                [encoder setBuffer:buf_b_atoms offset:0 atIndex:0];
                [encoder setBuffer:buf_m_atoms offset:0 atIndex:1];
                [encoder setBuffer:buf_b_rows offset:0 atIndex:2];
                [encoder setBuffer:buf_m_rows offset:0 atIndex:3];
                [encoder setBuffer:buf_unused offset:0 atIndex:4];
                [encoder setBuffer:buf_unused offset:0 atIndex:5];
                [encoder setBuffer:buf_b_counts offset:0 atIndex:6];
                [encoder setBuffer:buf_m_counts offset:0 atIndex:7];
                [encoder setBuffer:buf_unused offset:0 atIndex:8];
                [encoder setBuffer:buf_status offset:0 atIndex:9];
                [encoder setBuffer:buf_b_bonds offset:0 atIndex:11];
                [encoder setBuffer:buf_m_bonds offset:0 atIndex:12];
                [encoder setBuffer:buf_b_groups offset:0 atIndex:13];
                [encoder setBuffer:buf_b_preds offset:0 atIndex:14];
                [encoder setBuffer:buf_roots offset:0 atIndex:15];
                [encoder setBuffer:buf_pair_list offset:0 atIndex:17];
                // One grid caps at max_pairs_per_dispatch threadgroups (larger grids drop work);
                // each slice's real size is what level_dispatch_args wrote.
                for (uint64_t slice = 0; slice < slices; ++slice) {
                    Params params = {
                        (uint32_t)bucket_size, (uint32_t)atom_word_count, (uint32_t)max_prunings,
                        (uint32_t)max_leaf_checks, 1u, 0u, (uint32_t)max_pairs_per_dispatch,
                        (uint32_t)max_prefix_depth, (uint32_t)n_residuals,
                        (uint32_t)(slice * (uint64_t)max_pairs_per_dispatch), 1u
                    };
                    [encoder setBytes:&params length:sizeof(params) atIndex:10];
                    [encoder dispatchThreadgroupsWithIndirectBuffer:buf_args
                                               indirectBufferOffset:(NSUInteger)(slice * 3 * sizeof(uint32_t))
                                              threadsPerThreadgroup:MTLSizeMake((NSUInteger)threads_per_group, 1, 1)];
                }
                [encoder endEncoding];

                id<MTLComputeCommandEncoder> fold = [commandBuffer computeCommandEncoder];
                fold.label = @"fold_roots";
                [fold setComputePipelineState:s_fold_pipeline];
                [fold setBuffer:buf_status offset:0 atIndex:0];
                [fold setBuffer:buf_pair_list offset:0 atIndex:1];
                [fold setBuffer:buf_level_counts offset:0 atIndex:2];
                [fold setBuffer:buf_sub_roots offset:0 atIndex:3];
                [fold setBytes:&pp length:sizeof(pp) atIndex:4];
                [fold setBuffer:buf_unresolved offset:0 atIndex:5];
                dispatch_threads(fold, s_fold_pipeline, (NSUInteger)level_capacity);
                [fold endEncoding];
            }

            uint32_t r_first = residual_level_start[k];
            uint32_t r_count = residual_level_start[k + 1] - residual_level_start[k];
            if (r_count > 0) {
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                encoder.label = @"eval_residuals";
                PredParams pp = base;
                pp.first = r_first;
                pp.count = r_count;
                [encoder setComputePipelineState:s_residual_pipeline];
                [encoder setBuffer:buf_clause_start offset:0 atIndex:0];
                [encoder setBuffer:buf_term_start offset:0 atIndex:1];
                [encoder setBuffer:buf_factors offset:0 atIndex:2];
                [encoder setBuffer:buf_probe_bits offset:0 atIndex:3];
                [encoder setBuffer:buf_sub_roots offset:0 atIndex:4];
                [encoder setBuffer:buf_roots offset:0 atIndex:5];
                [encoder setBuffer:buf_m_counts offset:0 atIndex:6];
                [encoder setBytes:&pp length:sizeof(pp) atIndex:7];
                [encoder setBuffer:buf_factor_start offset:0 atIndex:8];
                dispatch_threads(encoder, s_residual_pipeline, (NSUInteger)r_count * n_mols);
                [encoder endEncoding];
            }
        }

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if ([commandBuffer status] == MTLCommandBufferStatusError) {
            snprintf(g_last_error, sizeof(g_last_error), "Command buffer error: %s", [[[commandBuffer error] localizedDescription] UTF8String]);
            return 0;
        }
        memcpy(roots_out, [buf_roots contents], (NSUInteger)n_mols * n_residuals * sizeof(uint64_t));
        *unresolved_out = *(const uint32_t*)[buf_unresolved contents];
        if (timings_out) {
            const auto done = std::chrono::steady_clock::now();
            timings_out[0] = std::chrono::duration<double>(buffers_ready - started).count();
            timings_out[1] = std::chrono::duration<double>(done - buffers_ready).count();
            timings_out[2] = commandBuffer.GPUEndTime - commandBuffer.GPUStartTime;
        }
        return 1;
    }
}

extern "C" const char* vesma_metal_last_error(void) { return g_last_error; }

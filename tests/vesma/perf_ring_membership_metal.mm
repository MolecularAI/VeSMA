// Ring-membership microbenchmark kernels: the short-ring check against `R`/`r` perception.
//
// Measurement scaffolding for `perf_ring_membership.py`, not production code. Every kernel takes
// the same input -- per molecule, 64 `ulong` ring-bond adjacency rows (chains already removed, as
// `ring_bond_rows` leaves them in metal_retro.mm) and an atom count -- and runs one thread per
// molecule, so the GPU times compare like with like:
//
//   0  short   -- per ring edge, bitset BFS of depth <= 7 with the edge removed. The check the WL
//                 identity runs today (`bond_in_short_cycle_rows`). Writes the 3-8 ring edge rows.
//   1  bfs_r   -- the same BFS without the depth cap, returning the distance: exact smallest ring
//                 through each atom (`r`).
//   2  horton  -- the proposal: per root, a canonical shortest-path tree carried as atom and
//                 128-bit edge bitmasks; Horton candidates from every edge whose two tree paths
//                 meet only at the root; then size-tiered GF(2) elimination against the basis
//                 frozen at the smaller sizes, stopping after the tier that reaches rank mu.
//                 Writes `R` (relevant rings through the atom) and `r`.
//
// Edges are indexed without a table: for u < v, index = prefix[u] + popcount(upper[u] & below(v)),
// where upper[u] is row u above u and prefix[u] the running popcount of the earlier upper rows.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static char g_last_error[1024] = {0};

static const char* kShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

#define CAND_CAP 128
#define BASIS_CAP 64
#define STATUS_EDGE_OVERFLOW 1u
#define STATUS_CANDIDATE_OVERFLOW 2u
#define STATUS_BASIS_OVERFLOW 4u

inline int alternate_path_length(thread const ulong* rows, int left, int right, int limit) {
    const ulong target = 1ul << right;
    ulong visited = 1ul << left;
    ulong frontier = visited;
    for (int distance = 1; distance <= limit; ++distance) {
        ulong reach = 0ul;
        ulong active = frontier;
        while (active != 0ul) {
            const int atom = ctz(active);
            ulong row = rows[atom];
            if (atom == left) row &= ~target;
            reach |= row;
            active &= active - 1ul;
        }
        frontier = reach & ~visited;
        if (distance >= 2 && (frontier & target) != 0ul) return distance;
        if (frontier == 0ul) return 0;
        visited |= frontier;
    }
    return 0;
}

inline void load_rows(device const ulong* rows_in, uint mol, int n, thread ulong* rows) {
    for (int a = 0; a < 64; ++a) rows[a] = a < n ? rows_in[mol * 64 + a] : 0ul;
}

kernel void short_ring(device const ulong* rows_in [[buffer(0)]],
                       device const uint* counts [[buffer(1)]],
                       device ulong* out_rows [[buffer(2)]],
                       constant uint& mol_count [[buffer(3)]],
                       uint mol [[thread_position_in_grid]]) {
    if (mol >= mol_count) return;
    const int n = (int)counts[mol];
    ulong rows[64];
    load_rows(rows_in, mol, n, rows);
    ulong hit[64];
    for (int a = 0; a < 64; ++a) hit[a] = 0ul;
    for (int u = 0; u < n; ++u) {
        ulong upper = rows[u] & ~((2ul << u) - 1ul);
        while (upper != 0ul) {
            const int v = ctz(upper);
            upper &= upper - 1ul;
            if (alternate_path_length(rows, u, v, 7) != 0) {
                hit[u] |= 1ul << v;
                hit[v] |= 1ul << u;
            }
        }
    }
    for (int a = 0; a < 64; ++a) out_rows[mol * 64 + a] = hit[a];
}

kernel void bfs_r(device const ulong* rows_in [[buffer(0)]],
                  device const uint* counts [[buffer(1)]],
                  device uchar* out_r [[buffer(2)]],
                  constant uint& mol_count [[buffer(3)]],
                  uint mol [[thread_position_in_grid]]) {
    if (mol >= mol_count) return;
    const int n = (int)counts[mol];
    ulong rows[64];
    load_rows(rows_in, mol, n, rows);
    uchar smallest[64];
    for (int a = 0; a < 64; ++a) smallest[a] = 0;
    for (int u = 0; u < n; ++u) {
        ulong upper = rows[u] & ~((2ul << u) - 1ul);
        while (upper != 0ul) {
            const int v = ctz(upper);
            upper &= upper - 1ul;
            const int length = alternate_path_length(rows, u, v, n);
            if (length == 0) continue;
            const uchar size = (uchar)(length + 1);
            if (smallest[u] == 0 || size < smallest[u]) smallest[u] = size;
            if (smallest[v] == 0 || size < smallest[v]) smallest[v] = size;
        }
    }
    for (int a = 0; a < 64; ++a) out_r[mol * 64 + a] = smallest[a];
}

struct Mask128 { ulong lo; ulong hi; };

inline Mask128 edge_bit(int index) {
    Mask128 m;
    m.lo = index < 64 ? 1ul << index : 0ul;
    m.hi = index >= 64 ? 1ul << (index - 64) : 0ul;
    return m;
}

inline int popcount128(Mask128 m) { return (int)(popcount(m.lo) + popcount(m.hi)); }

inline int lead128(Mask128 m) {
    if (m.hi != 0ul) return 127 - (int)clz(m.hi);
    if (m.lo != 0ul) return 63 - (int)clz(m.lo);
    return -1;
}

kernel void horton(device const ulong* rows_in [[buffer(0)]],
                   device const uint* counts [[buffer(1)]],
                   device uchar* out_count [[buffer(2)]],
                   device uchar* out_r [[buffer(3)]],
                   device uint* out_status [[buffer(4)]],
                   constant uint& mol_count [[buffer(5)]],
                   uint mol [[thread_position_in_grid]]) {
    if (mol >= mol_count) return;
    const int n = (int)counts[mol];
    ulong rows[64];
    load_rows(rows_in, mol, n, rows);

    uchar ring_count[64];
    uchar smallest[64];
    for (int a = 0; a < 64; ++a) { ring_count[a] = 0; smallest[a] = 0; }
    uint status = 0u;

    // Edge index: prefix popcounts of the upper rows, and the endpoints of each index.
    ulong upper[64];
    int prefix[64];
    uchar eu[128];
    uchar ev[128];
    int edge_count = 0;
    for (int u = 0; u < 64; ++u) {
        upper[u] = rows[u] & ~((2ul << u) - 1ul);
        prefix[u] = edge_count;
        ulong walk = upper[u];
        while (walk != 0ul) {
            const int v = ctz(walk);
            walk &= walk - 1ul;
            if (edge_count < 128) { eu[edge_count] = (uchar)u; ev[edge_count] = (uchar)v; }
            ++edge_count;
        }
    }
    if (edge_count > 128) {
        out_status[mol] = STATUS_EDGE_OVERFLOW;
        return;
    }

    // mu = E - V + C over atoms with a ring bond.
    ulong active = 0ul;
    for (int a = 0; a < n; ++a) if (rows[a] != 0ul) active |= 1ul << a;
    int components = 0;
    ulong seen = 0ul;
    ulong pending = active;
    while (pending != 0ul) {
        ulong frontier = 1ul << ctz(pending);
        seen |= frontier;
        ++components;
        while (frontier != 0ul) {
            ulong reach = 0ul;
            ulong walk = frontier;
            while (walk != 0ul) { reach |= rows[ctz(walk)]; walk &= walk - 1ul; }
            frontier = reach & ~seen;
            seen |= frontier;
        }
        pending = active & ~seen;
    }
    const int mu = edge_count - (int)popcount(active) + components;
    if (mu == 0) {
        for (int a = 0; a < 64; ++a) { out_count[mol * 64 + a] = 0; out_r[mol * 64 + a] = 0; }
        out_status[mol] = 0u;
        return;
    }

    // Horton candidates, deduplicated on insertion.
    Mask128 cand[CAND_CAP];
    uchar cand_size[CAND_CAP];
    int cand_count = 0;
    ulong path_atoms[64];
    Mask128 path_edges[64];
    ulong roots = active;
    while (roots != 0ul) {
        const int root = ctz(roots);
        roots &= roots - 1ul;
        const ulong root_bit = 1ul << root;
        path_atoms[root] = root_bit;
        path_edges[root].lo = 0ul;
        path_edges[root].hi = 0ul;
        ulong visited = root_bit;
        ulong frontier = root_bit;
        while (frontier != 0ul) {
            ulong next = 0ul;
            ulong walk = frontier;
            while (walk != 0ul) {  // ascending, so the lowest-index parent wins
                const int atom = ctz(walk);
                walk &= walk - 1ul;
                ulong fresh = rows[atom] & ~visited & ~next;
                next |= fresh;
                while (fresh != 0ul) {
                    const int child = ctz(fresh);
                    fresh &= fresh - 1ul;
                    const int lo = min(atom, child);
                    const int hi = max(atom, child);
                    const int index = prefix[lo] + (int)popcount(upper[lo] & ((1ul << hi) - 1ul));
                    const Mask128 bit = edge_bit(index);
                    path_atoms[child] = path_atoms[atom] | (1ul << child);
                    path_edges[child].lo = path_edges[atom].lo | bit.lo;
                    path_edges[child].hi = path_edges[atom].hi | bit.hi;
                }
            }
            visited |= next;
            frontier = next;
        }
        for (int index = 0; index < edge_count; ++index) {
            const int x = eu[index];
            const int y = ev[index];
            if (((visited >> x) & 1ul) == 0ul || ((visited >> y) & 1ul) == 0ul) continue;
            if ((path_atoms[x] & path_atoms[y]) != root_bit) continue;
            const Mask128 bit = edge_bit(index);
            if (((path_edges[x].lo | path_edges[y].lo) & bit.lo) != 0ul ||
                ((path_edges[x].hi | path_edges[y].hi) & bit.hi) != 0ul) continue;
            Mask128 mask;
            mask.lo = (path_edges[x].lo ^ path_edges[y].lo) | bit.lo;
            mask.hi = (path_edges[x].hi ^ path_edges[y].hi) | bit.hi;
            bool duplicate = false;
            for (int c = 0; c < cand_count; ++c) {
                if (cand[c].lo == mask.lo && cand[c].hi == mask.hi) { duplicate = true; break; }
            }
            if (duplicate) continue;
            if (cand_count == CAND_CAP) { status |= STATUS_CANDIDATE_OVERFLOW; continue; }
            cand[cand_count] = mask;
            cand_size[cand_count] = (uchar)popcount128(mask);
            ++cand_count;
        }
    }

    // Size-tiered GF(2): a tier is tested against the basis frozen below it, then committed.
    Mask128 basis[BASIS_CAP];
    char lead_slot[128];
    for (int i = 0; i < 128; ++i) lead_slot[i] = -1;
    int rank = 0;
    for (int size = 3; size <= 64 && rank < mu; ++size) {
        const int frozen = rank;
        for (int c = 0; c < cand_count; ++c) {
            if (cand_size[c] != size) continue;
            Mask128 v = cand[c];
            for (;;) {
                const int lead = lead128(v);
                if (lead < 0) break;
                const int slot = lead_slot[lead];
                if (slot < 0 || slot >= frozen) break;
                v.lo ^= basis[slot].lo;
                v.hi ^= basis[slot].hi;
            }
            if (lead128(v) < 0) continue;  // a sum of smaller rings: not relevant
            // Relevant: credit its atoms.
            ulong atoms = 0ul;
            ulong walk = cand[c].lo;
            while (walk != 0ul) { const int e = ctz(walk); walk &= walk - 1ul; atoms |= (1ul << eu[e]) | (1ul << ev[e]); }
            walk = cand[c].hi;
            while (walk != 0ul) { const int e = 64 + ctz(walk); walk &= walk - 1ul; atoms |= (1ul << eu[e]) | (1ul << ev[e]); }
            while (atoms != 0ul) {
                const int a = ctz(atoms);
                atoms &= atoms - 1ul;
                ring_count[a] += 1;
                if (smallest[a] == 0) smallest[a] = (uchar)size;
            }
            // Commit against the full current basis.
            for (;;) {
                const int lead = lead128(v);
                if (lead < 0) break;
                const int slot = lead_slot[lead];
                if (slot < 0) {
                    if (rank == BASIS_CAP) { status |= STATUS_BASIS_OVERFLOW; break; }
                    basis[rank] = v;
                    lead_slot[lead] = (char)rank;
                    ++rank;
                    break;
                }
                v.lo ^= basis[slot].lo;
                v.hi ^= basis[slot].hi;
            }
        }
    }
    for (int a = 0; a < 64; ++a) {
        out_count[mol * 64 + a] = ring_count[a];
        out_r[mol * 64 + a] = smallest[a];
    }
    out_status[mol] = status;
}

// ---- Cooperative variants: one 32-lane SIMD group per molecule. ----------------------------
//
// The edge index is built with a SIMD prefix sum (lane owns atoms lane and lane+32). The short
// check stripes edges across lanes. Horton stripes roots across lanes (each lane builds its
// roots' shortest-path trees and appends candidates), then each size tier tests its candidates
// in parallel against the frozen basis, and lane 0 commits the tier serially (dedupe, credit
// atoms, insert pivots) -- the commit is the only serial part, and it touches only the few
// candidates that survived.

#define COOP_LANES 32
#define COOP_CAND_CAP 256

inline int coop_edge_index(threadgroup const ulong* rows_tg, threadgroup ulong* upper,
                           threadgroup int* prefix, threadgroup uchar* eu, threadgroup uchar* ev,
                           uint lane) {
    const uint a0 = lane;
    const uint a1 = lane + 32;
    upper[a0] = rows_tg[a0] & ~((2ul << a0) - 1ul);
    upper[a1] = rows_tg[a1] & ~((2ul << a1) - 1ul);
    const int c0 = (int)popcount(upper[a0]);
    const int c1 = (int)popcount(upper[a1]);
    const int total0 = simd_sum(c0);
    const int p0 = simd_prefix_exclusive_sum(c0);
    const int p1 = total0 + simd_prefix_exclusive_sum(c1);
    const int edge_count = total0 + simd_sum(c1);
    prefix[a0] = p0;
    prefix[a1] = p1;
    if (edge_count <= 128) {
        int slot = p0;
        ulong walk = upper[a0];
        while (walk != 0ul) { eu[slot] = (uchar)a0; ev[slot] = (uchar)ctz(walk); ++slot; walk &= walk - 1ul; }
        slot = p1;
        walk = upper[a1];
        while (walk != 0ul) { eu[slot] = (uchar)a1; ev[slot] = (uchar)ctz(walk); ++slot; walk &= walk - 1ul; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return edge_count;
}

inline int alternate_path_length_tg(threadgroup const ulong* rows, int left, int right, int limit) {
    const ulong target = 1ul << right;
    ulong visited = 1ul << left;
    ulong frontier = visited;
    for (int distance = 1; distance <= limit; ++distance) {
        ulong reach = 0ul;
        ulong active = frontier;
        while (active != 0ul) {
            const int atom = ctz(active);
            ulong row = rows[atom];
            if (atom == left) row &= ~target;
            reach |= row;
            active &= active - 1ul;
        }
        frontier = reach & ~visited;
        if (distance >= 2 && (frontier & target) != 0ul) return distance;
        if (frontier == 0ul) return 0;
        visited |= frontier;
    }
    return 0;
}

kernel void short_ring_coop(device const ulong* rows_in [[buffer(0)]],
                            device const uint* counts [[buffer(1)]],
                            device ulong* out_rows [[buffer(2)]],
                            constant uint& mol_count [[buffer(3)]],
                            uint mol [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_threadgroup]]) {
    if (mol >= mol_count) return;
    threadgroup ulong rows[64];
    threadgroup ulong upper[64];
    threadgroup int prefix[64];
    threadgroup uchar eu[128];
    threadgroup uchar ev[128];
    threadgroup uchar hit[128];
    const int n = (int)counts[mol];
    rows[lane] = (int)lane < n ? rows_in[mol * 64 + lane] : 0ul;
    rows[lane + 32] = (int)lane + 32 < n ? rows_in[mol * 64 + lane + 32] : 0ul;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int edge_count = min(coop_edge_index(rows, upper, prefix, eu, ev, lane), 128);
    for (int e = (int)lane; e < edge_count; e += COOP_LANES) {
        hit[e] = alternate_path_length_tg(rows, eu[e], ev[e], 7) != 0 ? 1 : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint a = lane; a < 64; a += COOP_LANES) {
        ulong out = 0ul;
        for (int e = 0; e < edge_count; ++e) {
            if (!hit[e]) continue;
            if (eu[e] == a) out |= 1ul << ev[e];
            if (ev[e] == a) out |= 1ul << eu[e];
        }
        out_rows[mol * 64 + a] = out;
    }
}

kernel void horton_coop(device const ulong* rows_in [[buffer(0)]],
                        device const uint* counts [[buffer(1)]],
                        device uchar* out_count [[buffer(2)]],
                        device uchar* out_r [[buffer(3)]],
                        device uint* out_status [[buffer(4)]],
                        constant uint& mol_count [[buffer(5)]],
                        uint mol [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_threadgroup]]) {
    if (mol >= mol_count) return;
    threadgroup ulong rows[64];
    threadgroup ulong upper[64];
    threadgroup int prefix[64];
    threadgroup uchar eu[128];
    threadgroup uchar ev[128];
    threadgroup Mask128 cand[COOP_CAND_CAP];
    threadgroup Mask128 residue[COOP_CAND_CAP];
    threadgroup uchar cand_size[COOP_CAND_CAP];
    threadgroup uchar keep[COOP_CAND_CAP];
    threadgroup Mask128 basis[BASIS_CAP];
    threadgroup char lead_slot[128];
    threadgroup uchar ring_count[64];
    threadgroup uchar smallest[64];
    threadgroup atomic_uint cand_counter;
    threadgroup atomic_uint max_size;
    threadgroup int shared_rank;
    threadgroup int shared_mu;
    threadgroup uint status;

    const int n = (int)counts[mol];
    rows[lane] = (int)lane < n ? rows_in[mol * 64 + lane] : 0ul;
    rows[lane + 32] = (int)lane + 32 < n ? rows_in[mol * 64 + lane + 32] : 0ul;
    ring_count[lane] = 0; ring_count[lane + 32] = 0;
    smallest[lane] = 0; smallest[lane + 32] = 0;
    for (uint i = lane; i < 128; i += COOP_LANES) lead_slot[i] = -1;
    if (lane == 0) {
        atomic_store_explicit(&cand_counter, 0u, memory_order_relaxed);
        atomic_store_explicit(&max_size, 0u, memory_order_relaxed);
        shared_rank = 0;
        status = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int edge_count = coop_edge_index(rows, upper, prefix, eu, ev, lane);
    if (edge_count > 128) {
        if (lane == 0) out_status[mol] = STATUS_EDGE_OVERFLOW;
        return;
    }

    // mu = E - V + C; components by lane 0 (a handful of bitset sweeps).
    // simd_or has no 64-bit overload: reduce the two halves as uints.
    const uint active_lo = simd_or(rows[lane] != 0ul ? 1u << lane : 0u);
    const uint active_hi = simd_or(rows[lane + 32] != 0ul ? 1u << lane : 0u);
    const ulong active = (ulong)active_lo | ((ulong)active_hi << 32);
    if (lane == 0) {
        int components = 0;
        ulong seen = 0ul;
        ulong pending = active;
        while (pending != 0ul) {
            ulong frontier = 1ul << ctz(pending);
            seen |= frontier;
            ++components;
            while (frontier != 0ul) {
                ulong reach = 0ul;
                ulong walk = frontier;
                while (walk != 0ul) { reach |= rows[ctz(walk)]; walk &= walk - 1ul; }
                frontier = reach & ~seen;
                seen |= frontier;
            }
            pending = active & ~seen;
        }
        shared_mu = edge_count - (int)popcount(active) + components;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int mu = shared_mu;

    if (mu > 0) {
        // Candidates: lanes own roots.
        ulong path_atoms[64];
        Mask128 path_edges[64];
        ulong roots = active;
        while (roots != 0ul) {
            const int root = ctz(roots);
            roots &= roots - 1ul;
            if ((root & (COOP_LANES - 1)) != (int)lane) continue;
            const ulong root_bit = 1ul << root;
            path_atoms[root] = root_bit;
            path_edges[root].lo = 0ul;
            path_edges[root].hi = 0ul;
            ulong visited = root_bit;
            ulong frontier = root_bit;
            while (frontier != 0ul) {
                ulong next = 0ul;
                ulong walk = frontier;
                while (walk != 0ul) {
                    const int atom = ctz(walk);
                    walk &= walk - 1ul;
                    ulong fresh = rows[atom] & ~visited & ~next;
                    next |= fresh;
                    while (fresh != 0ul) {
                        const int child = ctz(fresh);
                        fresh &= fresh - 1ul;
                        const int lo = min(atom, child);
                        const int hi = max(atom, child);
                        const int index = prefix[lo] + (int)popcount(upper[lo] & ((1ul << hi) - 1ul));
                        const Mask128 bit = edge_bit(index);
                        path_atoms[child] = path_atoms[atom] | (1ul << child);
                        path_edges[child].lo = path_edges[atom].lo | bit.lo;
                        path_edges[child].hi = path_edges[atom].hi | bit.hi;
                    }
                }
                visited |= next;
                frontier = next;
            }
            for (int index = 0; index < edge_count; ++index) {
                const int x = eu[index];
                const int y = ev[index];
                if (((visited >> x) & 1ul) == 0ul || ((visited >> y) & 1ul) == 0ul) continue;
                if ((path_atoms[x] & path_atoms[y]) != root_bit) continue;
                const Mask128 bit = edge_bit(index);
                if (((path_edges[x].lo | path_edges[y].lo) & bit.lo) != 0ul ||
                    ((path_edges[x].hi | path_edges[y].hi) & bit.hi) != 0ul) continue;
                Mask128 mask;
                mask.lo = (path_edges[x].lo ^ path_edges[y].lo) | bit.lo;
                mask.hi = (path_edges[x].hi ^ path_edges[y].hi) | bit.hi;
                const uint slot = atomic_fetch_add_explicit(&cand_counter, 1u, memory_order_relaxed);
                if (slot >= COOP_CAND_CAP) continue;
                const uint size = (uint)popcount128(mask);
                cand[slot] = mask;
                cand_size[slot] = (uchar)size;
                atomic_fetch_max_explicit(&max_size, size, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint raw = atomic_load_explicit(&cand_counter, memory_order_relaxed);
        if (raw > COOP_CAND_CAP && lane == 0) status |= STATUS_CANDIDATE_OVERFLOW;
        const int count = (int)min(raw, (uint)COOP_CAND_CAP);
        const int top = (int)atomic_load_explicit(&max_size, memory_order_relaxed);

        for (int size = 3; size <= top; ++size) {
            if (shared_rank >= mu) break;  // uniform: written by lane 0 before the barrier
            const int frozen = shared_rank;
            // Parallel test against the frozen basis.
            for (int c = (int)lane; c < count; c += COOP_LANES) {
                keep[c] = 0;
                if (cand_size[c] != size) continue;
                Mask128 v = cand[c];
                for (;;) {
                    const int lead = lead128(v);
                    if (lead < 0) break;
                    const int slot = lead_slot[lead];
                    if (slot < 0 || slot >= frozen) break;
                    v.lo ^= basis[slot].lo;
                    v.hi ^= basis[slot].hi;
                }
                if (lead128(v) >= 0) { keep[c] = 1; residue[c] = v; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Serial commit of the survivors.
            if (lane == 0) {
                for (int c = 0; c < count; ++c) {
                    if (!keep[c]) continue;
                    bool duplicate = false;
                    for (int d = 0; d < c; ++d) {
                        if (keep[d] && cand[d].lo == cand[c].lo && cand[d].hi == cand[c].hi) { duplicate = true; break; }
                    }
                    if (duplicate) { keep[c] = 0; continue; }
                    ulong atoms = 0ul;
                    ulong walk = cand[c].lo;
                    while (walk != 0ul) { const int e = ctz(walk); walk &= walk - 1ul; atoms |= (1ul << eu[e]) | (1ul << ev[e]); }
                    walk = cand[c].hi;
                    while (walk != 0ul) { const int e = 64 + ctz(walk); walk &= walk - 1ul; atoms |= (1ul << eu[e]) | (1ul << ev[e]); }
                    while (atoms != 0ul) {
                        const int a = ctz(atoms);
                        atoms &= atoms - 1ul;
                        ring_count[a] += 1;
                        if (smallest[a] == 0) smallest[a] = (uchar)size;
                    }
                    Mask128 v = residue[c];
                    for (;;) {
                        const int lead = lead128(v);
                        if (lead < 0) break;
                        const int slot = lead_slot[lead];
                        if (slot < 0) {
                            if (shared_rank == BASIS_CAP) { status |= STATUS_BASIS_OVERFLOW; break; }
                            basis[shared_rank] = v;
                            lead_slot[lead] = (char)shared_rank;
                            ++shared_rank;
                            break;
                        }
                        v.lo ^= basis[slot].lo;
                        v.hi ^= basis[slot].hi;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    out_count[mol * 64 + lane] = ring_count[lane];
    out_count[mol * 64 + lane + 32] = ring_count[lane + 32];
    out_r[mol * 64 + lane] = smallest[lane];
    out_r[mol * 64 + lane + 32] = smallest[lane + 32];
    if (lane == 0) out_status[mol] = status;
}

// ---- Lean one-thread Horton: the same algorithm with about half the private memory. ----------
//
// `horton` keeps, per atom, the root path's atom mask (8 B) and 128-bit edge mask (16 B), plus a
// 128-entry candidate list and a 64-entry basis -- ~6.5 KB of per-thread arrays that spill to
// memory. Here the tree is one parent byte and one parent-edge byte per atom; a candidate's two
// paths are walked up to the root only for the few non-tree edges that close a cycle, and the
// walk stops at the first shared atom. Upper rows are recomputed from `rows` instead of stored,
// prefix sums fit a byte, and the candidate and basis caps drop to what 64 atoms need in practice
// (overflow is flagged, never silent).

#define LEAN_CAND_CAP 64
#define LEAN_BASIS_CAP 32
#define NO_PARENT 255

inline ulong upper_row(thread const ulong* rows, int u) { return rows[u] & ~((2ul << u) - 1ul); }

kernel void horton_lean(device const ulong* rows_in [[buffer(0)]],
                        device const uint* counts [[buffer(1)]],
                        device uchar* out_count [[buffer(2)]],
                        device uchar* out_r [[buffer(3)]],
                        device uint* out_status [[buffer(4)]],
                        constant uint& mol_count [[buffer(5)]],
                        uint mol [[thread_position_in_grid]]) {
    if (mol >= mol_count) return;
    const int n = (int)counts[mol];
    ulong rows[64];
    load_rows(rows_in, mol, n, rows);
    uint status = 0u;

    uchar prefix[64];
    uchar eu[128];
    uchar ev[128];
    int edge_count = 0;
    ulong active = 0ul;
    for (int u = 0; u < n; ++u) {
        if (rows[u] != 0ul) active |= 1ul << u;
        prefix[u] = (uchar)min(edge_count, 255);
        ulong walk = upper_row(rows, u);
        while (walk != 0ul) {
            const int v = ctz(walk);
            walk &= walk - 1ul;
            if (edge_count < 128) { eu[edge_count] = (uchar)u; ev[edge_count] = (uchar)v; }
            ++edge_count;
        }
    }
    device uchar* count_out = out_count + mol * 64;
    device uchar* r_out = out_r + mol * 64;
    for (int a = 0; a < 64; ++a) { count_out[a] = 0; r_out[a] = 0; }
    if (edge_count > 128) { out_status[mol] = STATUS_EDGE_OVERFLOW; return; }

    // Per ring system (connected component of ring bonds): one with as many bonds as atoms is a
    // lone ring, whose atoms get R = 1 and r = its size with no search. Only the fused, bridged
    // and spiro systems (mu_c >= 2) go on to Horton; mu is summed over those alone.
    int mu = 0;
    ulong complex_atoms = 0ul;
    ulong seen = 0ul;
    ulong pending = active;
    while (pending != 0ul) {
        ulong component = 1ul << ctz(pending);
        ulong frontier = component;
        seen |= frontier;
        while (frontier != 0ul) {
            ulong reach = 0ul;
            ulong walk = frontier;
            while (walk != 0ul) { reach |= rows[ctz(walk)]; walk &= walk - 1ul; }
            frontier = reach & ~seen;
            seen |= frontier;
            component |= frontier;
        }
        pending = active & ~seen;
        int degree_sum = 0;
        ulong walk = component;
        while (walk != 0ul) { degree_sum += (int)popcount(rows[ctz(walk)]); walk &= walk - 1ul; }
        const int atoms_c = (int)popcount(component);
        const int mu_c = degree_sum / 2 - atoms_c + 1;
        if (mu_c == 1) {
            walk = component;
            while (walk != 0ul) {
                const int a = ctz(walk);
                walk &= walk - 1ul;
                count_out[a] = 1;
                r_out[a] = (uchar)atoms_c;
            }
        } else {
            mu += mu_c;
            complex_atoms |= component;
        }
    }
    if (mu == 0) { out_status[mol] = 0u; return; }

    Mask128 cand[LEAN_CAND_CAP];
    uchar cand_size[LEAN_CAND_CAP];
    int cand_count = 0;
    int top = 0;
    uchar parent[64];
    uchar parent_edge[64];
    ulong roots = complex_atoms;
    while (roots != 0ul) {
        const int root = ctz(roots);
        roots &= roots - 1ul;
        const ulong root_bit = 1ul << root;
        parent[root] = NO_PARENT;
        ulong visited = root_bit;
        ulong frontier = root_bit;
        while (frontier != 0ul) {
            ulong next = 0ul;
            ulong walk = frontier;
            while (walk != 0ul) {  // ascending, so the lowest-index parent wins
                const int atom = ctz(walk);
                walk &= walk - 1ul;
                ulong fresh = rows[atom] & ~visited & ~next;
                next |= fresh;
                while (fresh != 0ul) {
                    const int child = ctz(fresh);
                    fresh &= fresh - 1ul;
                    const int lo = min(atom, child);
                    const int hi = max(atom, child);
                    parent[child] = (uchar)atom;
                    parent_edge[child] =
                        (uchar)(prefix[lo] + (int)popcount(upper_row(rows, lo) & ((1ul << hi) - 1ul)));
                }
            }
            visited |= next;
            frontier = next;
        }
        for (int index = 0; index < edge_count; ++index) {
            const int x = eu[index];
            const int y = ev[index];
            if (((visited >> x) & 1ul) == 0ul || ((visited >> y) & 1ul) == 0ul) continue;
            if (parent[y] == x || parent[x] == y) continue;  // a tree edge closes nothing
            Mask128 mask = edge_bit(index);
            ulong path_x = 0ul;
            for (int a = x; a != root; a = parent[a]) {
                path_x |= 1ul << a;
                const Mask128 bit = edge_bit(parent_edge[a]);
                mask.lo |= bit.lo;
                mask.hi |= bit.hi;
            }
            bool simple = true;
            for (int a = y; a != root; a = parent[a]) {
                if ((path_x >> a) & 1ul) { simple = false; break; }
                const Mask128 bit = edge_bit(parent_edge[a]);
                mask.lo |= bit.lo;
                mask.hi |= bit.hi;
            }
            if (!simple) continue;
            bool duplicate = false;
            for (int c = 0; c < cand_count; ++c) {
                if (cand[c].lo == mask.lo && cand[c].hi == mask.hi) { duplicate = true; break; }
            }
            if (duplicate) continue;
            if (cand_count == LEAN_CAND_CAP) { status |= STATUS_CANDIDATE_OVERFLOW; continue; }
            const int size = popcount128(mask);
            cand[cand_count] = mask;
            cand_size[cand_count] = (uchar)size;
            top = max(top, size);
            ++cand_count;
        }
    }

    Mask128 basis[LEAN_BASIS_CAP];
    char lead_slot[128];
    for (int i = 0; i < 128; ++i) lead_slot[i] = -1;
    int rank = 0;
    for (int size = 3; size <= top && rank < mu; ++size) {
        const int frozen = rank;
        for (int c = 0; c < cand_count; ++c) {
            if (cand_size[c] != size) continue;
            Mask128 v = cand[c];
            for (;;) {
                const int lead = lead128(v);
                if (lead < 0) break;
                const int slot = lead_slot[lead];
                if (slot < 0 || slot >= frozen) break;
                v.lo ^= basis[slot].lo;
                v.hi ^= basis[slot].hi;
            }
            if (lead128(v) < 0) continue;
            ulong atoms = 0ul;
            ulong walk = cand[c].lo;
            while (walk != 0ul) { const int e = ctz(walk); walk &= walk - 1ul; atoms |= (1ul << eu[e]) | (1ul << ev[e]); }
            walk = cand[c].hi;
            while (walk != 0ul) { const int e = 64 + ctz(walk); walk &= walk - 1ul; atoms |= (1ul << eu[e]) | (1ul << ev[e]); }
            while (atoms != 0ul) {
                const int a = ctz(atoms);
                atoms &= atoms - 1ul;
                count_out[a] += 1;
                if (r_out[a] == 0) r_out[a] = (uchar)size;
            }
            for (;;) {
                const int lead = lead128(v);
                if (lead < 0) break;
                const int slot = lead_slot[lead];
                if (slot < 0) {
                    if (rank == LEAN_BASIS_CAP) { status |= STATUS_BASIS_OVERFLOW; break; }
                    basis[rank] = v;
                    lead_slot[lead] = (char)rank;
                    ++rank;
                    break;
                }
                v.lo ^= basis[slot].lo;
                v.hi ^= basis[slot].hi;
            }
        }
    }
    out_status[mol] = status;
}
)";

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLComputePipelineState> g_pipelines[6] = {nil, nil, nil, nil, nil, nil};

static int ensure_pipelines(void) {
    if (g_device != nil) return 0;
    g_device = MTLCreateSystemDefaultDevice();
    if (g_device == nil) {
        snprintf(g_last_error, sizeof(g_last_error), "No Metal device.");
        return -1;
    }
    g_queue = [g_device newCommandQueue];
    NSError* error = nil;
    id<MTLLibrary> library = [g_device newLibraryWithSource:[NSString stringWithUTF8String:kShaderSource]
                                                    options:nil
                                                      error:&error];
    if (library == nil) {
        snprintf(g_last_error, sizeof(g_last_error), "%s", [[error localizedDescription] UTF8String]);
        g_device = nil;
        return -1;
    }
    const char* names[6] = {"short_ring", "bfs_r", "horton", "short_ring_coop", "horton_coop", "horton_lean"};
    for (int k = 0; k < 6; ++k) {
        id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:names[k]]];
        g_pipelines[k] = [g_device newComputePipelineStateWithFunction:function error:&error];
        if (g_pipelines[k] == nil) {
            snprintf(g_last_error, sizeof(g_last_error), "%s", [[error localizedDescription] UTF8String]);
            g_device = nil;
            return -1;
        }
    }
    return 0;
}

extern "C" const char* ring_bench_last_error(void) { return g_last_error; }

// Runs `kernel` (0 short, 1 bfs_r, 2 horton) over `mol_count` molecules `repeats` times and
// writes each run's GPU time to `gpu_seconds`. Outputs hold the last run.
extern "C" int ring_bench_run(int kernel,
                              const uint64_t* rows,
                              const uint32_t* counts,
                              uint32_t mol_count,
                              uint32_t threads_per_group,
                              uint32_t repeats,
                              uint64_t* out_rows,
                              uint8_t* out_count,
                              uint8_t* out_r,
                              uint32_t* out_status,
                              double* gpu_seconds) {
    @autoreleasepool {
        if (ensure_pipelines() != 0) return -1;
        const size_t row_bytes = (size_t)mol_count * 64 * sizeof(uint64_t);
        const size_t atom_bytes = (size_t)mol_count * 64;
        id<MTLBuffer> buf_rows = [g_device newBufferWithBytes:rows length:row_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_counts = [g_device newBufferWithBytes:counts length:(size_t)mol_count * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_a = [g_device newBufferWithLength:((kernel == 0 || kernel == 3) ? row_bytes : atom_bytes) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_b = [g_device newBufferWithLength:atom_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_status = [g_device newBufferWithLength:(size_t)mol_count * 4 options:MTLResourceStorageModeShared];
        id<MTLComputePipelineState> pipeline = g_pipelines[kernel];
        const NSUInteger width = MIN((NSUInteger)threads_per_group, pipeline.maxTotalThreadsPerThreadgroup);
        for (uint32_t run = 0; run < repeats; ++run) {
            id<MTLCommandBuffer> commandBuffer = [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:buf_rows offset:0 atIndex:0];
            [encoder setBuffer:buf_counts offset:0 atIndex:1];
            [encoder setBuffer:buf_a offset:0 atIndex:2];
            if (kernel == 2 || kernel == 4 || kernel == 5) {
                [encoder setBuffer:buf_b offset:0 atIndex:3];
                [encoder setBuffer:buf_status offset:0 atIndex:4];
                [encoder setBytes:&mol_count length:sizeof(mol_count) atIndex:5];
            } else {
                [encoder setBytes:&mol_count length:sizeof(mol_count) atIndex:3];
            }
            if (kernel == 3 || kernel == 4) {
                [encoder dispatchThreadgroups:MTLSizeMake(mol_count, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            } else {
                [encoder dispatchThreads:MTLSizeMake(mol_count, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
            }
            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            if ([commandBuffer status] == MTLCommandBufferStatusError) {
                snprintf(g_last_error, sizeof(g_last_error), "%s",
                         [[[commandBuffer error] localizedDescription] UTF8String]);
                return -1;
            }
            gpu_seconds[run] = commandBuffer.GPUEndTime - commandBuffer.GPUStartTime;
        }
        if (kernel == 0 || kernel == 3) memcpy(out_rows, [buf_a contents], row_bytes);
        if (kernel == 1) memcpy(out_r, [buf_a contents], atom_bytes);
        if (kernel == 2 || kernel == 4 || kernel == 5) {
            memcpy(out_count, [buf_a contents], atom_bytes);
            memcpy(out_r, [buf_b contents], atom_bytes);
            memcpy(out_status, [buf_status contents], (size_t)mol_count * 4);
        }
        return 0;
    }
}

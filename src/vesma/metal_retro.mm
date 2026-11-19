#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>

static char g_last_error[1024] = {0};

// Buffers that outlive a single dispatch.
//
// The reaction table is why this exists. At ~25 KB per rule a 42k-rule library is over a
// gigabyte, and it was being re-uploaded for every chunk of assignments -- which cost more
// than every kernel in the pipeline put together, and got worse as the chunk got smaller.
// The rules do not change between chunks, so the caller stamps the table with a token and
// the upload happens once per token. A token of zero means "do not cache", which is what a
// caller that cannot promise the contents are stable should pass.
//
// The working buffers are grow-only: a dispatch that fits the last allocation reuses it.
static id<MTLBuffer> s_rxn_buffer = nil;
static uint64_t s_rxn_token = 0;
static size_t s_rxn_size = 0;
static id<MTLBuffer> s_p_atoms_buffer = nil;
static id<MTLBuffer> s_p_bonds_buffer = nil;
static id<MTLBuffer> s_p_counts_buffer = nil;
static uint64_t s_products_token = 0;
static size_t s_p_atoms_size = 0;
static id<MTLBuffer> s_scratch_buffer = nil;
static size_t s_scratch_capacity = 0;
static id<MTLBuffer> s_out_atoms_buffer = nil;
static size_t s_out_atoms_capacity = 0;
static id<MTLBuffer> s_out_bonds_buffer = nil;
static size_t s_out_bonds_capacity = 0;
static id<MTLBuffer> s_out_counts_buffer = nil;
static size_t s_out_counts_capacity = 0;
static id<MTLBuffer> s_out_status_buffer = nil;
static size_t s_out_status_capacity = 0;
static id<MTLBuffer> s_counter_buffer = nil;

// The closed-shell configuration table, uploaded once per process.
//
// Unlike the reaction table it needs no token: `configurations.configuration_table` is a pure
// function of the declared chemistry and takes no arguments, so one process has exactly one
// table. The sizes are kept and checked anyway, cheaply, so a caller that believes otherwise
// gets an error instead of a silently stale upload. CUDA caches it the same way, in
// `cuda_configuration_table`.
static id<MTLBuffer> s_config_index_buffer = nil;
static id<MTLBuffer> s_config_rows_buffer = nil;
static uint32_t s_config_index_count = 0;
static uint32_t s_config_row_count = 0;

// The host's side of four constants the shader also declares. The shader source is a raw
// string below and cannot `#include` anything, so these are genuinely separate declarations of
// one number -- and `vesma.limits` holds a third. `constants_probe` reports both sides of every
// one of them so a test can compare all three rather than trusting these comments.
static const size_t MAX_COMPONENTS = 6;
static const size_t MAX_REACTION_EDGES = 512;
static const size_t EDITED_BUCKET = 256;
// Lanes the WL kernel's threadgroup must carry: its reduction sums `partial_edges[0..WL_THREADS)`,
// so a narrower dispatch leaves slots no lane wrote. This was the bare literal 32 at the dispatch.
static const size_t WL_THREADS_PER_GROUP = 32;

struct ReactionData {
    uint32_t lhs_atom_count;
    uint32_t rhs_atom_count;

    uint32_t deleted_lhs_count;
    int32_t  deleted_lhs_indices[64];

    uint32_t common_count;
    int32_t  common_lhs_indices[64];
    int32_t  common_rhs_indices[64];
    int32_t  common_maps[64];

    uint32_t lhs_edge_count;
    int32_t  lhs_edge_left[512];
    int32_t  lhs_edge_right[512];

    uint32_t rhs_edge_count;
    int32_t  rhs_edge_left[512];
    int32_t  rhs_edge_right[512];
    uint8_t  rhs_edge_bond_types[512];

    uint32_t rhs_atoms[14 * 256];
    int32_t  rhs_atom_maps[256];
    // Which zero-level group each RHS atom was written in; 0 for none. The postcondition
    // below is the only reader.
    int32_t  rhs_atom_groups[256];
    int32_t  common_charge[64];
    int32_t  common_charge_explicit[64];
    uint32_t padding1;
    uint32_t padding2;
};


struct RetroStatus {
    uint32_t component_count;
    uint32_t overflow;  // non-zero if >MAX_COMPONENTS components: kept heaviest, dropped rest
    uint32_t atom_overflow;  // count of >64 atom components dropped
    uint32_t edited_atom_overflow;  // non-zero if edited graph exceeded 256 atoms
    uint32_t output_base;  // start index of this assignment's components in the output arrays
    // RejectionReason for a refused application, or REJECT_NONE. A refusal costs
    // only its own application: the batch keeps every other one.
    uint32_t rejection_reason;
    // Components dropped as debris: a template that orphans is under-specified,
    // and the count is how it says so.
    uint32_t undescribed_components;
    // The subset the *edit* severed, which is what refuses the application. A component
    // the template never touched is an unreacted reagent in the pot.
    uint32_t orphaned_components;
    // Applications whose product disagreed with the RHS's zero-level grouping: a rule written
    // `>>(B).(C)` that handed back one molecule, or a `>>(B.C)` that fell apart. Refuses the
    // application, like the orphan above and for the same reason -- it presents an impossible
    // disconnection as a complete one.
    uint32_t grouping_violations;
    // Why the kernel gave up on an assignment without naming a reason.
    //
    // Three places refuse structurally rather than chemically: a mapped LHS atom whose assignment
    // lands outside the product (1), a mapped product atom with no edited slot (2), and an RHS
    // edge whose endpoint never got one (3). They are not `RejectionReason`s -- those are
    // chemistry -- so they used to return leaving `rejection_reason == REJECT_NONE`, which made
    // the assignment neither a success nor a counted refusal. Measured first rather than named
    // blind: zero over the parity corpus, 110,449 applications of 42,554 rules, so the branches
    // are unreachable in practice. They now also set `REJECT_UNSUPPORTED_TEMPLATE`, which is not
    // a chemistry claim but "this engine cannot act on this template", so every assignment is a
    // success or a counted refusal and there is no third state. This code says which branch,
    // which the reason alone cannot.
    uint32_t inconsistent_assignment;
    // Populated only by explicit constructor profiling calls.
    uint32_t profile_stage;
    uint32_t profile_edited_atoms;
    uint32_t profile_component_merges;
    uint32_t profile_kept_components;
    uint32_t profile_perceived_atoms;
    uint32_t profile_aromatic_atoms;
    uint32_t profile_cycles;
};

// The per-assignment device scratch pool is sized by the shader's own sizeof(Scratch),
// queried once via the scratch_size_probe kernel (see s_scratch_bytes below), so the
// host stride can never disagree with the kernel's scratch_pool[pair_idx] indexing.

// The Metal retro kernel is a faithful serial port of the CUDA pipeline in
// vesma/cuda_retro.py (retro_construct_assignments -> normalize_aromaticity_and_hydrogens
// -> component_token). Aromaticity is re-perceived from scratch on the edited graph with
// the delocalized-pi model (perceive_aromaticity_device), H/valence atom invariants are
// refreshed, and the short-ring Weisfeiler-Leman identity hash is computed so Metal
// precursor keys land in the same space as the CPU short_ring_wl_hash_codec. Everything
// runs on thread 0 of each threadgroup; the grid is one threadgroup per assignment.
static const char* kShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

#define BUCKET 64
#define EDITED_BUCKET 256
#define ATOM_WORD_COUNT 14
#define MAX_REACTION_EDGES 512
// Output component slots reserved per assignment. Kept in sync with
// MAX_COMPONENTS_PER_ASSIGNMENT in metal_retro.py.
#define MAX_COMPONENTS 6
#define AROM_MAX_CYCLES 48
// Lanes per threadgroup in the standalone WL-hash kernel (one threadgroup/precursor).
// One SIMD group on Apple GPUs; the host must dispatch this many threads per group.
#define WL_THREADS 32
#define MATCH_ATOM_CAP 64

// RejectionReason, mirrored from vesma/rejection.py. The values are the enum's own, so a
// device refusal lands in the same counter slot as a CPU one; REJECT_NONE is a sentinel
// rather than a reason, because zero is itself a declared reason.
#define REJECT_NONE 0xFFFFFFFFu
#define REJECT_UNDECLARED_CONFIGURATION 1u
#define REJECT_NUCLEAR_CONSERVATION 2u
#define REJECT_CONFIGURATION_INCOMPATIBLE 3u
#define REJECT_CONFIGURATION_AMBIGUOUS 4u
#define REJECT_HYDROGEN_COMPLETION_NEGATIVE 5u
#define REJECT_HYDROGEN_COMPLETION_INCOMPATIBLE 6u
#define REJECT_HYDROGEN_COMPLETION_AMBIGUOUS 7u
#define REJECT_HYDROGEN_OMITTED_WITH_CHARGE_CHANGE 8u
#define REJECT_DELOCALIZED_EDGE_NOT_CYCLIC 9u
#define REJECT_DELOCALIZED_SYSTEM_INVALID 10u
#define REJECT_REQUIRES_DELOCALIZED_ANNOTATION 11u
#define REJECT_PRECURSOR_CAPACITY 13u
// Not a chemistry claim but "this engine cannot act on this template", which is what the three
// structural give-ups mean. See `RetroStatus.inconsistent_assignment`.
#define REJECT_UNSUPPORTED_TEMPLATE 14u

// The GF(2) cycle-space basis. Only edges that appear in an examined cycle need a basis
// position -- restricting the basis to them does not change those cycles' rank -- so the
// width is bounded by the cycles the enumeration can hold plus the asserted circuits, each
// of which spans at most a 64-atom ring. 1024 positions cannot be filled by that, which is
// what lets the builder refuse rather than silently drop an edge.
#define GF2_MAX_EDGES 1024
#define GF2_WORDS 16
#define GF2_MAX_VECTORS (AROM_MAX_CYCLES + 8)

#define BOND_NONE 0
#define BOND_SINGLE 1
#define BOND_DOUBLE 2
#define BOND_TRIPLE 4
#define BOND_AROMATIC 8
#define BOND_CLASS_MASK 15
// Matcher topology field: 01 ring, 10 chain, 11 either; 00 only for no edge.
#define BOND_TOPOLOGY_RING 0x10
#define BOND_TOPOLOGY_CHAIN 0x20

#define ALIPHATIC 1u
#define AROMATIC 2u
#define WILDCARD 0xFFFFFFFFu
// Word 13: `x` in the low half, the isotope in the high half (see metal_matcher.mm).
#define ISOTOPE_WORD 13
#define ISOTOPE_SHIFT 16u
#define ISOTOPE_ANY 0xFFFFu
#define ISOTOPE_FIELD 0xFFFF0000u
// Relevant-cycle perception (ring_membership) bounds. A component with more ring bonds,
// candidates or basis vectors than these keeps R and r unknown (WILDCARD) rather than guess.
#define HORTON_EDGE_CAP 128
#define HORTON_CAND_CAP 128
#define HORTON_BASIS_CAP 64
#define HORTON_FAR 255
// Steps the shortest-path enumerators may take per component before giving R and r up.
#define HORTON_PATH_BUDGET (1 << 20)

inline uchar bond_class(uchar flags) {
    return flags & BOND_CLASS_MASK;
}

#define GPU_WL_HASH_VERSION 6ul
#define GPU_WL_ROUNDS 8

#define WL_COMBINE_CONSTANT 0x517CC1B727220A95ul
#define WL_FINAL_LOW_SEED 0x4D6F7267616E574Cul
#define WL_FINAL_HIGH_SEED 0x7665736D615F776Cul
#define SPLITMIX64_GAMMA 0x9E3779B97F4A7C15ul
#define SPLITMIX64_MUL1 0xBF58476D1CE4E5B9ul
#define SPLITMIX64_MUL2 0x94D049BB133111EBul

#define PRIME64_1 11400714785074694791ul
#define PRIME64_2 14029467366897019727ul
#define PRIME64_3 1609587929392839161ul
#define PRIME64_4 9650029242287828579ul
#define PRIME64_5 2870177450012600261ul

struct ReactionData {
    uint lhs_atom_count;
    uint rhs_atom_count;
    uint deleted_lhs_count;
    int  deleted_lhs_indices[64];
    uint common_count;
    int  common_lhs_indices[64];
    int  common_rhs_indices[64];
    int  common_maps[64];
    uint lhs_edge_count;
    int  lhs_edge_left[MAX_REACTION_EDGES];
    int  lhs_edge_right[MAX_REACTION_EDGES];
    uint rhs_edge_count;
    int  rhs_edge_left[MAX_REACTION_EDGES];
    int  rhs_edge_right[MAX_REACTION_EDGES];
    uchar rhs_edge_bond_types[MAX_REACTION_EDGES];
    uint rhs_atoms[14 * EDITED_BUCKET];
    int  rhs_atom_maps[EDITED_BUCKET];
    int  rhs_atom_groups[EDITED_BUCKET];
    // What charge the RHS wrote for each mapped atom, and whether it wrote one at all.
    // The packed mask cannot answer the second question -- an unconstrained charge and an
    // explicit +0 are different instructions -- so the flag rides alongside it.
    int  common_charge[MATCH_ATOM_CAP];
    int  common_charge_explicit[MATCH_ATOM_CAP];
    uint padding1;
    uint padding2;
};


struct RetroStatus {
    uint component_count;
    uint overflow;  // non-zero if >MAX_COMPONENTS components: kept heaviest, dropped rest
    uint atom_overflow;  // count of >64 atom components dropped
    uint edited_atom_overflow;  // non-zero if edited graph exceeded 256 atoms
    uint output_base;  // start index of this assignment's components in the output arrays
    // RejectionReason for a refused application, or REJECT_NONE. A refusal costs
    // only its own application: the batch keeps every other one.
    uint rejection_reason;
    // Components dropped as debris: a template that orphans is under-specified,
    // and the count is how it says so.
    uint undescribed_components;
    uint orphaned_components;
    // Applications whose product disagreed with the RHS's zero-level grouping: a rule written
    // `>>(B).(C)` that handed back one molecule, or a `>>(B.C)` that fell apart. Refuses the
    // application, like the orphan above and for the same reason -- it presents an impossible
    // disconnection as a complete one.
    uint grouping_violations;
    // See the host declaration of this field for what the three codes mean.
    uint inconsistent_assignment;
    uint profile_stage;
    uint profile_edited_atoms;
    uint profile_component_merges;
    uint profile_kept_components;
    uint profile_perceived_atoms;
    uint profile_aromatic_atoms;
    uint profile_cycles;
};

// All per-pair working state, kept in threadgroup memory and driven by thread 0.
struct Scratch {
    // Whole edited molecule produced by bond edits.
    uint  edited_atoms[EDITED_BUCKET][ATOM_WORD_COUNT];
    uchar edited_bonds[EDITED_BUCKET][EDITED_BUCKET];
    int   edited_map[EDITED_BUCKET];
    int   explicit_h[EDITED_BUCKET];
    int   implicit_h[EDITED_BUCKET];
    uchar keep_product[BUCKET];
    int   product_to_edited[BUCKET];
    int   rhs_to_edited[EDITED_BUCKET];
    int   assignment[BUCKET];
    int   comp_labels[EDITED_BUCKET];
    int   comp_atoms[EDITED_BUCKET];
    int   root_size[EDITED_BUCKET];   // heavy-atom count per component root (overflow handling)
    uchar keep_comp[EDITED_BUCKET];   // 1 if this root's component is emitted

    // Contiguous working copy of the component currently being emitted.
    uint  comp_words[BUCKET][ATOM_WORD_COUNT];
    uchar comp_bonds[BUCKET][BUCKET];
    int   comp_eh[BUCKET];
    int   comp_ih[BUCKET];
    int   z[BUCKET];
    int   charge[BUCKET];

    // perceive_aromaticity_device scratch + outputs.
    int   heavy_degree[BUCKET];
    int   total_h[BUCKET];
    ulong adjacency[BUCKET];
    ulong edge_rows[BUCKET];
    ulong atom_mask;
    uchar cycle_path[AROM_MAX_CYCLES][8];
    int   cycle_len[AROM_MAX_CYCLES];
    ulong cycle_mask[AROM_MAX_CYCLES];
    int   clabel[AROM_MAX_CYCLES];

    // ring_bond_rows working state and output: the non-bridge edge set of the component.
    ulong ring_rows[BUCKET];
    ulong ring_path[BUCKET];
    ulong ring_children[BUCKET];
    ulong ring_parent_bit[BUCKET];

    // ring_membership working state and output: R and r per atom from the relevant cycles.
    uchar ring_count[BUCKET];
    uchar ring_smallest[BUCKET];
    uchar horton_prefix[BUCKET];
    uchar horton_dist[BUCKET];  // from the current root; HORTON_FAR if unreached
    uchar horton_eu[HORTON_EDGE_CAP];
    uchar horton_ev[HORTON_EDGE_CAP];
    char  horton_lead_slot[HORTON_EDGE_CAP];
    // Two shortest-path enumerators (one per end of an edge), each a DFS stack down the
    // shortest-path DAG from its target to the root: the node, the predecessors still to try,
    // and the atom and edge bits of the path so far.
    uchar path_node[2][BUCKET];
    ulong path_choice[2][BUCKET];
    ulong path_atoms[2][BUCKET];
    ulong path_lo[2][BUCKET];
    ulong path_hi[2][BUCKET];
    int   path_level[2];
    int   path_fresh[2];
    int   path_budget;
    ulong horton_cand_lo[HORTON_CAND_CAP];
    ulong horton_cand_hi[HORTON_CAND_CAP];
    ulong horton_basis_lo[HORTON_BASIS_CAP];
    ulong horton_basis_hi[HORTON_BASIS_CAP];

    // Total hydrogen decided by the hydrogen law during surgery (-1 until decided), and
    // which product atom each edited atom came from (-1 for an atom the template adds).
    int   edited_h[EDITED_BUCKET];
    int   edited_from_product[EDITED_BUCKET];
    // Whether the RHS names this atom. The RHS *is* the template's enumeration of what
    // comes back, so a component carrying none of them is debris, not a precursor.
    uchar described[EDITED_BUCKET];
    // The RHS group each edited atom came from, 0 for none. A `uchar` because a template
    // writes a handful of groups, and because a second word-wide EDITED_BUCKET array in
    // this struct costs every thread.
    uchar edited_group[EDITED_BUCKET];
    // Whether each component root has any RHS-named atom, accumulated in one pass rather
    // than rescanned per root.
    uchar root_described[EDITED_BUCKET];
    // Total hydrogen pinned by the hydrogen law during surgery; perception takes it
    // verbatim rather than re-deriving it.
    int   comp_ph[BUCKET];
    // Edge rows of the whole fused ring system a face is being counted against.
    ulong span_rows[BUCKET];
    // How many chordless faces each edge belongs to, for the circuit boundary.
    uchar face_incidence[BUCKET][BUCKET];
    // Examined cycles as GF(2) vectors over the edges they span, and the elimination.
    ulong gf2_vec[GF2_MAX_VECTORS][GF2_WORDS];
    ulong gf2_pivot[GF2_MAX_VECTORS][GF2_WORDS];
    int   gf2_lead[GF2_MAX_VECTORS];
    uchar gf2_u[GF2_MAX_EDGES];
    uchar gf2_v[GF2_MAX_EDGES];

    // component_token scratch.
    uchar identity_bonds[BUCKET][BUCKET];
    ulong labels[BUCKET];
    ulong next_labels[BUCKET];
    ulong degree[BUCKET];
    ulong payload[BUCKET + 4];
};

// Working state for the standalone WL-hash kernel (wl_hash_molecules), which runs
// one threadgroup of WL_THREADS lanes per emitted precursor. It needs only the
// subset of fields component_token touches; kept in (small) threadgroup memory so
// many precursor threadgroups stay resident. Field names match Scratch so the WL
// helpers read `s.comp_words` / `s.comp_bonds` / etc. unchanged.
struct WLScratch {
    uint  comp_words[BUCKET][ATOM_WORD_COUNT];
    uchar comp_bonds[BUCKET][BUCKET];
    ulong adjacency_rows[BUCKET];
    uchar identity_bonds[BUCKET][BUCKET];
    ulong labels[BUCKET];
    ulong next_labels[BUCKET];
    ulong degree[BUCKET];
    ulong payload[BUCKET + 4];
};

// --- 64-bit hash primitives (mirror invariants.py / cuda_retro.py) ---

inline ulong mix_uint64(ulong value) {
    value += SPLITMIX64_GAMMA;
    value = (value ^ (value >> 30)) * SPLITMIX64_MUL1;
    value = (value ^ (value >> 27)) * SPLITMIX64_MUL2;
    return value ^ (value >> 31);
}

inline ulong combine_uint64(ulong left, ulong right) {
    const ulong combined =
        left ^ (right + WL_COMBINE_CONSTANT + (left << 6) + (left >> 2));
    return mix_uint64(combined);
}

inline ulong rotl64(ulong value, int shift) {
    return (value << shift) | (value >> (64 - shift));
}

inline ulong xxh64_round(ulong acc, ulong input) {
    acc += input * PRIME64_2;
    acc = rotl64(acc, 31);
    acc *= PRIME64_1;
    return acc;
}

inline ulong xxh64_merge_round(ulong acc, ulong value) {
    value = xxh64_round(0ul, value);
    acc ^= value;
    acc = (acc * PRIME64_1) + PRIME64_4;
    return acc;
}

inline ulong xxh64_avalanche(ulong hash) {
    hash ^= hash >> 33;
    hash *= PRIME64_2;
    hash ^= hash >> 29;
    hash *= PRIME64_3;
    hash ^= hash >> 32;
    return hash;
}

ulong xxh64_uint64_array(threadgroup const ulong* data, int count, ulong seed) {
    ulong h64;
    int idx = 0;
    const ulong len = (ulong)count * 8ul;

    if (count >= 4) {
        ulong v1 = seed + PRIME64_1 + PRIME64_2;
        ulong v2 = seed + PRIME64_2;
        ulong v3 = seed;
        ulong v4 = seed - PRIME64_1;
        const int limit = count - 4;
        do {
            v1 = xxh64_round(v1, data[idx]); idx += 1;
            v2 = xxh64_round(v2, data[idx]); idx += 1;
            v3 = xxh64_round(v3, data[idx]); idx += 1;
            v4 = xxh64_round(v4, data[idx]); idx += 1;
        } while (idx <= limit);
        h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h64 = xxh64_merge_round(h64, v1);
        h64 = xxh64_merge_round(h64, v2);
        h64 = xxh64_merge_round(h64, v3);
        h64 = xxh64_merge_round(h64, v4);
    } else {
        h64 = seed + PRIME64_5;
    }

    h64 += len;
    while (idx < count) {
        ulong k1 = data[idx];
        idx += 1;
        k1 *= PRIME64_2;
        k1 = rotl64(k1, 31);
        k1 *= PRIME64_1;
        h64 ^= k1;
        h64 = rotl64(h64, 27);
        h64 = (h64 * PRIME64_1) + PRIME64_4;
    }
    return xxh64_avalanche(h64);
}

// --- atom-mask decoders (mirror cuda_retro.py) ---

inline int ctz_u32_or_zero(uint value) {
    return value == 0u ? 0 : ctz(value);
}

inline bool is_single_bit_u32(uint value) {
    return value != 0u && (value & (value - 1u)) == 0u;
}

inline int exact_z_from_words(uint w0, uint w1, uint w2, uint w3) {
    if (w0 != 0u) return ctz_u32_or_zero(w0);
    if (w1 != 0u) return ctz_u32_or_zero(w1) + 32;
    if (w2 != 0u) return ctz_u32_or_zero(w2) + 64;
    if (w3 != 0u) return ctz_u32_or_zero(w3) + 96;
    return 0;
}

inline uint charge_mask_exact(int charge) {
    if (charge < -3) charge = -3;
    if (charge > 4) charge = 4;
    return 1u << (uint)(charge + 3);
}

inline int charge_from_mask(uint mask) {
    if (mask == 0u || (mask & (mask - 1u)) != 0u) return 0;
    return ctz_u32_or_zero(mask) - 3;
}

inline int exact_bucket_from_mask(uint mask) {
    return ctz_u32_or_zero(mask);
}

inline uint bucket_mask_exact(int value) {
    if (value < 0) value = 0;
    if (value > 7) value = 7;
    return 1u << value;
}

inline int heavy_valence_half_from_bond(uchar bond) {
    if (bond == BOND_SINGLE) return 2;
    if (bond == BOND_DOUBLE) return 4;
    if (bond == BOND_TRIPLE) return 6;
    if (bond == BOND_AROMATIC) return 3;
    return 0;
}

// --- closed-shell configurations (mirror vesma/configurations.py) ---
//
// An atom's chemistry is fixed by which closed-shell configuration it occupies, and the
// set of configurations is data rather than code: configurations_for(z, charge) is a pure
// function of two small integers, so the host generates the whole product from that one
// function and hands it here as two buffers. Generating rather than transcribing is what
// keeps the device from drifting -- in particular the isoelectronic asymmetry (standard
// valence looked up at z - charge, but outer electrons and octet-expansion eligibility at
// z itself) survives without the shader having to know the rule at all.
//
// Selection is a lookup with three outcomes -- one configuration, none, or several -- and
// the last two are reported rather than resolved. Nothing here ever picks the first row
// that fits.

#define CONFIG_CHARGE_MIN (-3)
#define CONFIG_CHARGE_SPAN 8
#define CONFIG_Z_SPAN 128
#define CONFIG_PI_BIAS 2

// PiContribution, stored unbiased. Note these values differ from the CPU PiDonor enum,
// which numbers its non-numeric states the other way round; the device carries exactly one
// representation so there is never a conversion to get wrong.
#define PI_INVALID (-2)
#define PI_AMBIGUOUS (-1)
#define PI_NONE 0
#define PI_ONE 1
#define PI_TWO 2

// ConfigurationFailure, shifted by one so zero can mean "selected".
#define CONFIG_OK 0u
#define CONFIG_INCOMPATIBLE 1u
#define CONFIG_AMBIGUOUS 2u
#define CONFIG_UNDECLARED 3u

struct ConfigRow {
    int sigma_capacity;
    int graph_valence;
    int pi;
};

inline ConfigRow config_row_empty() {
    ConfigRow row;
    row.sigma_capacity = 0;
    row.graph_valence = 0;
    row.pi = PI_INVALID;
    return row;
}

inline ConfigRow config_row_unpack(uint word) {
    ConfigRow row;
    row.sigma_capacity = (int)(word & 0xFFu);
    row.graph_valence = (int)((word >> 8) & 0xFFu);
    row.pi = (int)((word >> 16) & 0xFFu) - CONFIG_PI_BIAS;
    return row;
}

// The declared configurations for one (element, charge). A count of zero means the table
// says nothing about this element and charge at all, which is a gap in the model -- a
// different failure from declared states that simply do not fit, which is a claim about
// the molecule.
// Returned packed exactly as the table stores it -- offset in the low half, count in the
// high half -- rather than through an out-parameter, so there is one value to reason about
// and no aliasing question about where the second answer was written.
inline uint config_span(device const uint* config_index, int z, int charge) {
    // Both coordinates are bounds-tested as unsigned, so one comparison catches the
    // negative and the too-large case, and the subscript is computed from exactly the
    // values that were tested.
    const uint z_slot = (uint)z;
    const uint charge_slot = (uint)(charge - CONFIG_CHARGE_MIN);
    if (z_slot >= (uint)CONFIG_Z_SPAN) return 0u;
    if (charge_slot >= (uint)CONFIG_CHARGE_SPAN) return 0u;
    return config_index[z_slot * (uint)CONFIG_CHARGE_SPAN + charge_slot];
}

inline uint config_span_offset(uint span) { return span & 0xFFFFu; }
inline uint config_span_count(uint span) { return span >> 16; }

// Mirrors configurations.select_configuration. A delocalized atom has no bond orders to
// read -- that is what delocalization means -- so its sigma frame constrains it, together
// with the fact that it demonstrably belongs to a circuit. A concrete atom is constrained
// by its sigma frame and its bond-order sum, which is what separates an sp3 centre from an
// sp2 one of the same element.
inline uint config_select(
    device const uint* config_index,
    device const uint* config_rows,
    int z,
    int charge,
    int sigma_degree,
    int bond_order_sum,
    int total_h,
    bool delocalized,
    thread ConfigRow& out
) {
    out = config_row_empty();
    const uint span = config_span(config_index, z, charge);
    const uint offset = config_span_offset(span);
    const uint count = config_span_count(span);
    if (count == 0u) return CONFIG_UNDECLARED;
    const int sigma = sigma_degree + total_h;
    uint found = 0u;
    for (uint i = 0u; i < count; ++i) {
        const ConfigRow row = config_row_unpack(config_rows[offset + i]);
        if (row.sigma_capacity != sigma) continue;
        const bool consistent = delocalized
            ? (row.pi != PI_INVALID)
            : (row.graph_valence == bond_order_sum + total_h);
        if (!consistent) continue;
        if (found == 0u) out = row;
        found += 1u;
    }
    if (found == 0u) { out = config_row_empty(); return CONFIG_INCOMPATIBLE; }
    if (found > 1u) { out = config_row_empty(); return CONFIG_AMBIGUOUS; }
    return CONFIG_OK;
}

// Mirrors configurations.select_circuit_configuration. Deliberately blind to bond orders:
// concrete orders are a *spelling*, so the same ring written delocalized has none at all,
// and letting them select the configuration would make perception depend on notation --
// the one thing this model exists to prevent.
inline uint config_select_circuit(
    device const uint* config_index,
    device const uint* config_rows,
    int z,
    int charge,
    int sigma_degree,
    int total_h,
    thread ConfigRow& out
) {
    out = config_row_empty();
    const uint span = config_span(config_index, z, charge);
    const uint offset = config_span_offset(span);
    const uint count = config_span_count(span);
    if (count == 0u) return CONFIG_UNDECLARED;
    const int sigma = sigma_degree + total_h;
    uint found = 0u;
    for (uint i = 0u; i < count; ++i) {
        const ConfigRow row = config_row_unpack(config_rows[offset + i]);
        if (row.sigma_capacity != sigma) continue;
        if (row.pi == PI_INVALID) continue;
        if (found == 0u) out = row;
        found += 1u;
    }
    if (found == 0u) { out = config_row_empty(); return CONFIG_INCOMPATIBLE; }
    if (found > 1u) { out = config_row_empty(); return CONFIG_AMBIGUOUS; }
    return CONFIG_OK;
}

// Mirrors configurations.unwritten_hydrogen_configuration together with
// unwritten_hydrogens. An omitted hydrogen count is not a free variable to solve for: it
// means the fewest hydrogens the atom's declared configurations allow given the bonding
// that *was* written. The consequence worth naming is that octet expansion must be
// evidenced by bonding and can never be bought with hydrogen.
inline uint config_unwritten(
    device const uint* config_index,
    device const uint* config_rows,
    int z,
    int charge,
    int sigma_degree,
    int bond_order_sum,
    bool delocalized,
    thread ConfigRow& out,
    thread int& hydrogens
) {
    out = config_row_empty();
    hydrogens = 0;
    const uint span = config_span(config_index, z, charge);
    const uint offset = config_span_offset(span);
    const uint count = config_span_count(span);
    if (count == 0u) return CONFIG_UNDECLARED;
    int fewest = 0x7FFFFFFF;
    uint found = 0u;
    for (uint i = 0u; i < count; ++i) {
        const ConfigRow row = config_row_unpack(config_rows[offset + i]);
        int reachable;
        if (delocalized) {
            // The atom demonstrably belongs to a circuit, so a configuration that cannot
            // join one is not a candidate, and a delocalized bond carries no order to
            // subtract -- its sigma frame is what constrains it.
            if (row.pi == PI_INVALID) continue;
            reachable = row.sigma_capacity - sigma_degree;
        } else {
            reachable = row.graph_valence - bond_order_sum;
            if (row.sigma_capacity != sigma_degree + reachable) continue;
        }
        if (reachable < 0) continue;
        if (reachable < fewest) {
            fewest = reachable;
            found = 1u;
            out = row;
        } else if (reachable == fewest) {
            found += 1u;
        }
    }
    if (found == 0u) { out = config_row_empty(); return CONFIG_INCOMPATIBLE; }
    hydrogens = fewest;
    if (found > 1u) { out = config_row_empty(); hydrogens = 0; return CONFIG_AMBIGUOUS; }
    return CONFIG_OK;
}

// --- valence / aromaticity device functions (mirror cuda_retro.py) ---

inline int smallest_aliphatic_valence_geq(int z, int occupied) {
    switch (z) {
        case 5:  return occupied <= 3 ? 3 : -1;
        case 6:  return occupied <= 4 ? 4 : -1;
        case 7:  return occupied <= 3 ? 3 : (occupied <= 5 ? 5 : -1);
        case 8:  return occupied <= 2 ? 2 : -1;
        case 9:  return occupied <= 1 ? 1 : -1;
        case 14: return occupied <= 4 ? 4 : -1;
        case 15: return occupied <= 3 ? 3 : (occupied <= 5 ? 5 : -1);
        case 16: return occupied <= 2 ? 2 : (occupied <= 4 ? 4 : (occupied <= 6 ? 6 : -1));
        case 17: return occupied <= 1 ? 1 : -1;
        case 33: return occupied <= 3 ? 3 : (occupied <= 5 ? 5 : -1);
        case 34: return occupied <= 2 ? 2 : (occupied <= 4 ? 4 : (occupied <= 6 ? 6 : -1));
        case 35: return occupied <= 1 ? 1 : -1;
        case 53: return occupied <= 1 ? 1 : -1;
        default: return -1;
    }
}

inline int arom_default_valence(int z) {
    switch (z) {
        case 4: return 2;  case 5: return 3;  case 6: return 4;  case 7: return 3;
        case 8: return 2;  case 14: return 4; case 15: return 3; case 16: return 2;
        case 32: return 4; case 33: return 3; case 34: return 2; case 50: return 4;
        case 51: return 3; case 52: return 2; default: return -1;
    }
}

inline int arom_outer_electrons(int z) {
    switch (z) {
        case 5: return 3;  case 6: return 4;  case 7: return 5;  case 8: return 6;
        case 15: return 5; case 16: return 6; case 33: return 5; case 34: return 6;
        case 52: return 6; default: return -1;
    }
}

inline float arom_electronegativity(int z) {
    switch (z) {
        case 5: return 2.04f;  case 6: return 2.55f;  case 7: return 3.04f;
        case 8: return 3.44f;  case 9: return 3.98f;  case 14: return 1.90f;
        case 15: return 2.19f; case 16: return 2.58f; case 17: return 3.16f;
        case 33: return 2.18f; case 34: return 2.55f; case 35: return 2.96f;
        case 52: return 2.10f; case 53: return 2.66f; default: return -1.0f;
    }
}

inline bool arom_electrons_stolen(int donor_z, int acceptor_z) {
    const float donor = arom_electronegativity(donor_z);
    const float acceptor = arom_electronegativity(acceptor_z);
    return donor >= 0.0f && acceptor >= 0.0f && acceptor > donor;
}

inline int arom_bond_order(uchar bond) {
    bond = bond_class(bond);
    if (bond == BOND_SINGLE) return 1;
    if (bond == BOND_DOUBLE) return 2;
    if (bond == BOND_TRIPLE) return 3;
    return 0;
}

inline int arom_aromatic_implicit_hydrogens(int z, int degree) {
    if (z == 5 || z == 6) {
        const int h = 3 - degree;
        return h > 0 ? h : 0;
    }
    return 0;
}

// Total hydrogens for a canonical identity/WL component atom, mirroring
// total_hydrogens() in vesma/hydrogens.py so device WL tokens match the CPU model:
// source-aromatic B/C are degree-derived (3 - degree, even when charged, so
// cyclopentadienyl / tropylium ring carbons keep their H); an aromatic heteroatom, a
// fixed explicit H, or a formal charge leaves no free implicit H; every other atom
// closes valence from concrete bond orders (signed-charge occupancy).
inline int arom_component_hydrogens(
    int z, int charge, int explicit_h, int degree_h, int order_sum, bool aromatic_atom
) {
    if (aromatic_atom && (z == 5 || z == 6)) {
        return arom_aromatic_implicit_hydrogens(z, degree_h);
    }
    if (explicit_h > 0 || charge != 0 || aromatic_atom) {
        return explicit_h;
    }
    const int occupied = order_sum + explicit_h - charge;
    const int target = smallest_aliphatic_valence_geq(z, occupied);
    const int implicit_h = target < 0 ? 0 : target - occupied;
    return explicit_h + implicit_h;
}

// Canonical bond form: an accepted circuit's edge carries pure DELOCALIZED, every other
// present edge carries its concrete order. An edge asserting delocalization that no
// accepted circuit claimed is NOT quietly resolved to a single bond -- that would invent a
// bond order the input never gave and hand back a different molecule than the one asked
// for. The caller reports it as DELOCALIZED_EDGE_NOT_CYCLIC.
inline uchar arom_canonical_bond(uchar raw, bool aromatic_edge) {
    if (aromatic_edge) return BOND_AROMATIC;
    const uchar order = raw & (BOND_SINGLE | BOND_DOUBLE | BOND_TRIPLE);
    if (order != 0) return order;
    return BOND_NONE;
}

// Pi electrons an atom brings to a candidate circuit, or PI_INVALID if it cannot join one.
//
// What the atom *can* bring is a property of its configuration, so it is read from the
// table rather than re-derived here. What remains is the one question the table cannot
// answer, because it is about the circuit rather than the atom: when the atom has a single
// pi bond, does that bond lie inside this circuit? An endocyclic pi bond puts its electron
// in; a sole exocyclic one engages the p orbital elsewhere and puts nothing in. That is a
// topological fact, so carbon, nitrogen and oxygen partners are treated alike and no
// electronegativity comparison is needed -- tropone counts six and delocalizes, fulvene
// counts four and does not.
int arom_pi_donor(
    device Scratch& s,
    int atom,
    device const ulong* cyc_row,
    int n,
    device const uint* config_index,
    device const uint* config_rows
) {
    ConfigRow configuration;
    const uint status = config_select_circuit(
        config_index, config_rows, s.z[atom], s.charge[atom],
        s.heavy_degree[atom], s.total_h[atom], configuration);
    if (status != CONFIG_OK) return PI_INVALID;
    if (configuration.pi == PI_INVALID) return PI_INVALID;
    if (configuration.pi == PI_ONE) {
        // Relative to the whole eligible system, not to one face of it. A concrete double
        // bond drawn *inside* the system is a Kekule spelling of bonds the delocalized
        // spelling does not write at all, so letting its placement decide the count would
        // make perception depend on notation.
        for (int neighbor = 0; neighbor < n; ++neighbor) {
            if (s.comp_bonds[atom][neighbor] == BOND_NONE) continue;
            if (((cyc_row[atom] >> neighbor) & 1ul) != 0ul) continue;
            if (arom_bond_order(s.comp_bonds[atom][neighbor]) >= 2) return PI_NONE;
        }
        return PI_ONE;
    }
    // PI_AMBIGUOUS never reaches here: no row in the generated table carries it, which the
    // host asserts when it builds the table. Treating it as invalid is therefore not a
    // simplification of the model, only of the code.
    return configuration.pi;
}

// Whether a circuit is delocalized. An atom that cannot take part rules it out outright;
// otherwise the electron count must satisfy Huckel. Because every contribution is decided
// relative to this circuit, a molecule perceives identically however its bonds were spelled.
bool arom_system_is_aromatic(
    device Scratch& s,
    ulong atom_mask,
    device const ulong* cyc_row,
    int n,
    device const uint* config_index,
    device const uint* config_rows
) {
    int electron_count = 0;
    for (int atom = 0; atom < n; ++atom) {
        if (((atom_mask >> atom) & 1ul) == 0ul) continue;
        const int donor =
            arom_pi_donor(s, atom, cyc_row, n, config_index, config_rows);
        if (donor == PI_INVALID) return false;
        electron_count += donor;
    }
    return electron_count >= 2 && ((electron_count - 2) % 4 == 0);
}

bool arom_cycles_share_edge(
    device const uchar* left_path, int left_len,
    device const uchar* right_path, int right_len
) {
    for (int i = 0; i < left_len; ++i) {
        const int la = left_path[i];
        const int lb = left_path[(i + 1) % left_len];
        for (int j = 0; j < right_len; ++j) {
            const int ra = right_path[j];
            const int rb = right_path[(j + 1) % right_len];
            if ((la == ra && lb == rb) || (la == rb && lb == ra)) return true;
        }
    }
    return false;
}

// Edge rows of one enumerated cycle.
inline void arom_cycle_rows(device Scratch& s, int cycle, device ulong* rows, int n) {
    for (int i = 0; i < n; ++i) rows[i] = 0ul;
    const int length = s.cycle_len[cycle];
    for (int i = 0; i < length; ++i) {
        const int u = s.cycle_path[cycle][i];
        const int v = s.cycle_path[cycle][(i + 1) % length];
        rows[u] |= 1ul << v;
        rows[v] |= 1ul << u;
    }
}

// Whether a cycle has no edge joining two atoms that are not its neighbours. A cycle
// carrying a chord is not an elementary face of the ring system; it is two smaller cycles
// seen from outside, and counting it as a face would cancel the very edges it is made of.
bool arom_cycle_is_chordless(device Scratch& s, int cycle) {
    const int length = s.cycle_len[cycle];
    for (int left = 0; left < length; ++left) {
        for (int right = left + 1; right < length; ++right) {
            if (right - left == 1) continue;
            if (left == 0 && right == length - 1) continue;
            const int u = s.cycle_path[cycle][left];
            const int v = s.cycle_path[cycle][right];
            if (s.comp_bonds[u][v] != BOND_NONE) return false;
        }
    }
    return true;
}

// Record the cycle whose edge rows sit in s.degree as a GF(2) vector over a basis of the
// edges examined so far, growing the basis as new edges appear. Every edge of the cycle
// must get a position: quietly skipping one would turn a cycle into an open path, still a
// nonzero vector, which would raise the measured rank and hide the very ring the check
// exists to catch. Returns the new basis size, or -1 if the basis could not hold the cycle.
inline int arom_add_examined_cycle(
    device Scratch& s, int n, int vector_index, int basis_edges
) {
    for (int w = 0; w < GF2_WORDS; ++w) s.gf2_vec[vector_index][w] = 0ul;
    for (int u = 0; u < n; ++u) {
        ulong row = s.degree[u];
        while (row != 0ul) {
            const int v = ctz(row);
            row &= row - 1ul;
            if (v <= u) continue;
            int slot = -1;
            for (int e = 0; e < basis_edges && slot < 0; ++e) {
                if (s.gf2_u[e] == (uchar)u && s.gf2_v[e] == (uchar)v) slot = e;
            }
            if (slot < 0) {
                if (basis_edges >= GF2_MAX_EDGES) return -1;
                slot = basis_edges;
                s.gf2_u[slot] = (uchar)u;
                s.gf2_v[slot] = (uchar)v;
                basis_edges += 1;
            }
            s.gf2_vec[vector_index][slot >> 6] ^= 1ul << (slot & 63);
        }
    }
    return basis_edges;
}

// GF(2) rank of the examined cycles, by elimination on fixed-width bitmasks. Addition over
// GF(2) is exclusive or, so this is Gaussian elimination with no arithmetic to go wrong and
// no tolerance to choose.
inline int arom_gf2_rank(device Scratch& s, int vector_count) {
    int pivots = 0;
    for (int v = 0; v < vector_count; ++v) {
        ulong current[GF2_WORDS];
        for (int w = 0; w < GF2_WORDS; ++w) current[w] = s.gf2_vec[v][w];
        for (int p = 0; p < pivots; ++p) {
            // Reduce by the pivot when doing so clears the pivot's leading bit, which is
            // what `min(vector, vector ^ pivot)` expresses on arbitrary-width integers.
            int lead = s.gf2_lead[p];
            if (((current[lead >> 6] >> (lead & 63)) & 1ul) == 0ul) continue;
            for (int w = 0; w < GF2_WORDS; ++w) current[w] ^= s.gf2_pivot[p][w];
        }
        int lead = -1;
        for (int w = GF2_WORDS - 1; w >= 0 && lead < 0; --w) {
            if (current[w] == 0ul) continue;
            for (int bit = 63; bit >= 0; --bit) {
                if (((current[w] >> bit) & 1ul) != 0ul) { lead = w * 64 + bit; break; }
            }
        }
        if (lead < 0) continue;
        for (int w = 0; w < GF2_WORDS; ++w) s.gf2_pivot[pivots][w] = current[w];
        s.gf2_lead[pivots] = lead;
        pivots += 1;
        if (pivots >= GF2_MAX_VECTORS) break;
    }
    return pivots;
}

// The nine metals the configuration table declares. Bonds to a metal are authored template
// graph bonds and every declared metal configuration carries no hydrogen, so completion
// stops here rather than reading a valence off the graph.
inline bool z_is_metal(int z) {
    return z == 3 || z == 11 || z == 12 || z == 19 || z == 25
        || z == 29 || z == 30 || z == 50 || z == 76;
}

// Which refusal a configuration failure becomes when the atom's own state is what did not
// occupy one. Distinct from the hydrogen mapping below: the same failure means a different
// thing depending on what was asking.
inline uint reject_for_state_failure(uint status) {
    if (status == CONFIG_AMBIGUOUS) return REJECT_CONFIGURATION_AMBIGUOUS;
    if (status == CONFIG_UNDECLARED) return REJECT_UNDECLARED_CONFIGURATION;
    return REJECT_CONFIGURATION_INCOMPATIBLE;
}

// Which refusal a configuration failure becomes when it is hydrogen that needed the
// configuration. An element and charge the table says nothing about is a gap in the model;
// declared states that simply do not fit are a claim about the molecule.
inline uint reject_for_config_failure(uint status) {
    if (status == CONFIG_AMBIGUOUS) return REJECT_HYDROGEN_COMPLETION_AMBIGUOUS;
    if (status == CONFIG_UNDECLARED) return REJECT_UNDECLARED_CONFIGURATION;
    return REJECT_HYDROGEN_COMPLETION_INCOMPATIBLE;
}

// Perceive delocalization on s.comp_bonds (contiguous component, indices 0..n-1) under the
// closed-shell circuit model. Total hydrogen is *pinned* by the caller -- the hydrogen law
// decided it during surgery -- and is taken verbatim here.
//
// Writes s.atom_mask and s.edge_rows. Returns REJECT_NONE, or the reason the component is
// refused. The order of the steps is load-bearing: hydrogen before eligibility, eligibility
// before grouping, grouping before counting, each ring before the fused system, and the
// asserted circuits after automatic perception.
uint perceive_aromaticity_device(
    device Scratch& s,
    int n,
    device const uint* config_index,
    device const uint* config_rows,
    thread uint& profile_cycles,
    uint profile_mode
) {
    s.atom_mask = 0ul;
    for (int i = 0; i < BUCKET; ++i) s.edge_rows[i] = 0ul;
    if (n <= 0) return REJECT_NONE;

    ulong source_aromatic = 0ul;
    for (int atom = 0; atom < n; ++atom) {
        int degree = 0;
        ulong row = 0ul;
        for (int other = 0; other < n; ++other) {
            const uchar bond = s.comp_bonds[atom][other];
            if (bond == BOND_NONE) continue;
            degree += 1;
            if ((bond & BOND_AROMATIC) != 0) source_aromatic |= 1ul << atom;
            row |= 1ul << other;
        }
        s.heavy_degree[atom] = degree;
        s.adjacency[atom] = row;
        s.total_h[atom] = s.comp_ph[atom];
    }

    // Whether an atom can belong to a circuit at all is a property of its configuration,
    // not of any candidate circuit, so it is settled before cycles are grouped. Grouping
    // first would let a saturated ring that merely shares a bond with an aromatic one be
    // fused into the same system.
    ulong can_delocalize = 0ul;
    for (int atom = 0; atom < n; ++atom) {
        ConfigRow configuration;
        const uint status = config_select_circuit(
            config_index, config_rows, s.z[atom], s.charge[atom],
            s.heavy_degree[atom], s.total_h[atom], configuration);
        if (status == CONFIG_OK && configuration.pi != PI_INVALID) {
            can_delocalize |= 1ul << atom;
        }
    }

    // An atom carrying an asserted delocalized bond whose configuration cannot join a
    // circuit is a configuration failure, and saying so here is the difference between a
    // true cause and a downstream symptom: left alone it surfaces later as an orphan edge,
    // which reports that the bond is not cyclic when the ring is plainly cyclic.
    if ((source_aromatic & ~can_delocalize) != 0ul) return REJECT_CONFIGURATION_INCOMPATIBLE;

    // Enumerate over the eligible subgraph rather than the whole component. A cycle
    // containing an atom that cannot join a circuit is discarded either way, so this is the
    // same set of cycles -- but the walk no longer descends into saturated chains and
    // substituents to build cycles it will then throw away, which is most of what a typical
    // precursor is made of.
    ulong eligible_adjacency[BUCKET];
    for (int atom = 0; atom < n; ++atom) {
        eligible_adjacency[atom] = (((can_delocalize >> atom) & 1ul) != 0ul)
            ? (s.adjacency[atom] & can_delocalize)
            : 0ul;
    }

    int cycle_count = 0;
    for (int root = 0; root < n; ++root) {
        if (eligible_adjacency[root] == 0ul) continue;
        int path[8];
        ulong remaining[8];
        int depth = 0;
        ulong visited = 1ul << root;
        path[0] = root;
        remaining[0] = eligible_adjacency[root];
        while (depth >= 0) {
            if (remaining[depth] == 0ul) {
                visited &= ~(1ul << path[depth]);
                depth -= 1;
                continue;
            }
            // Visit set bits in ascending order, as the index cursor did, while
            // skipping non-neighbours instead of testing all n atoms per step.
            const int neighbor = ctz(remaining[depth]);
            remaining[depth] &= remaining[depth] - 1ul;
            const int length = depth + 1;
            if (neighbor == root) {
                if (length >= 3 && length <= 8 && path[1] < path[depth]) {
                    // Only cycles every one of whose atoms could delocalize are candidates.
                    // Every atom reached is eligible by construction of the subgraph.
                    ulong mask = 0ul;
                    for (int i = 0; i < length; ++i) mask |= 1ul << path[i];
                    // Overflowing the cycle store is a machine limit, not an answer. The
                    // old kernel fell through to "fully aliphatic", which under this model
                    // is a silently wrong molecule rather than a refusal.
                    if (cycle_count >= AROM_MAX_CYCLES) return REJECT_PRECURSOR_CAPACITY;
                    for (int i = 0; i < length; ++i) {
                        s.cycle_path[cycle_count][i] = (uchar)path[i];
                    }
                    s.cycle_len[cycle_count] = length;
                    s.cycle_mask[cycle_count] = mask;
                    cycle_count += 1;
                }
                continue;
            }
            if (length == 8) continue;
            if (neighbor < root) continue;
            if (((visited >> neighbor) & 1ul) != 0ul) continue;
            depth += 1;
            path[depth] = neighbor;
            remaining[depth] = eligible_adjacency[neighbor];
            visited |= 1ul << neighbor;
        }
    }

    profile_cycles = (uint)cycle_count;
    if (profile_mode == 5u) return REJECT_NONE;
    for (int c = 0; c < cycle_count; ++c) s.clabel[c] = c;
    bool changed = true;
    while (changed) {
        changed = false;
        for (int a = 0; a < cycle_count; ++a) {
            for (int b = 0; b < cycle_count; ++b) {
                if (a == b || s.clabel[b] >= s.clabel[a]) continue;
                if (arom_cycles_share_edge(
                        s.cycle_path[a], s.cycle_len[a],
                        s.cycle_path[b], s.cycle_len[b])) {
                    s.clabel[a] = s.clabel[b];
                    changed = true;
                }
            }
        }
    }

    for (int root_cycle = 0; root_cycle < cycle_count; ++root_cycle) {
        if (s.clabel[root_cycle] != root_cycle) continue;

        // The exocyclic classification is made against the whole fused component, not
        // against the single face being counted: a bond into the neighbouring ring is not
        // exocyclic to the conjugated system, and calling it so made the count depend on
        // where a Kekule spelling happened to put its double bonds.
        for (int i = 0; i < n; ++i) s.span_rows[i] = 0ul;
        int member_count = 0;
        ulong union_mask = 0ul;
        for (int c = 0; c < cycle_count; ++c) {
            if (s.clabel[c] != root_cycle) continue;
            member_count += 1;
            union_mask |= s.cycle_mask[c];
            const int length = s.cycle_len[c];
            for (int i = 0; i < length; ++i) {
                const int u = s.cycle_path[c][i];
                const int v = s.cycle_path[c][(i + 1) % length];
                s.span_rows[u] |= 1ul << v;
                s.span_rows[v] |= 1ul << u;
            }
        }

        // Each ring is asked on its own first. A ring that carries an acceptable count by
        // itself is an accepted circuit, and a bond two such rings share belongs to both of
        // them -- naphthalene's fusion bond is delocalized because it lies on two circuits,
        // not because the pair of rings together came to a total.
        bool all_accepted = true;
        for (int c = 0; c < cycle_count; ++c) {
            if (s.clabel[c] != root_cycle) continue;
            if (arom_system_is_aromatic(
                    s, s.cycle_mask[c], s.span_rows, n, config_index, config_rows)) {
                s.atom_mask |= s.cycle_mask[c];
                arom_cycle_rows(s, c, s.degree, n);
                for (int i = 0; i < n; ++i) s.edge_rows[i] |= s.degree[i];
            } else {
                all_accepted = false;
            }
        }
        if (all_accepted || member_count <= 1) continue;

        // Only where a ring cannot stand alone is the fused system asked as a whole. Then
        // the accepted circuit is the system's perimeter, and the bonds between rings are
        // interior to it rather than on it. Azulene is the textbook case: five and seven
        // electrons separately, ten around the rim, and a transannular bond that measures
        // like an ordinary single bond.
        if (!arom_system_is_aromatic(
                s, union_mask, s.span_rows, n, config_index, config_rows)) {
            continue;
        }
        // The atoms are flagged below, from the circuit's own edges rather than from the
        // union of the cycles that proposed it. Taking the union marks an atom the perimeter
        // never reaches, which is a claim that it delocalizes across nothing -- and on this
        // arm nothing refuses it, so it is written into the atom word and read back two ways
        // at once: `config_select` takes the aliphatic configuration from the bonds while the
        // hydrogen split below takes the aromatic branch from the word. Mirrors
        // `aromaticity.py`, where the host has the same two lines.

        // The boundary is the edges belonging to exactly one elementary face. Only
        // chordless cycles count as faces, since the enumeration also yields the perimeter
        // of a pair of fused rings, and including that dependent cycle would cancel the
        // edges it is made of.
        int face_count = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) s.face_incidence[i][j] = 0;
        }
        for (int c = 0; c < cycle_count; ++c) {
            if (s.clabel[c] != root_cycle) continue;
            if (!arom_cycle_is_chordless(s, c)) continue;
            face_count += 1;
            const int length = s.cycle_len[c];
            for (int i = 0; i < length; ++i) {
                const int u = s.cycle_path[c][i];
                const int v = s.cycle_path[c][(i + 1) % length];
                s.face_incidence[u][v] += 1;
                s.face_incidence[v][u] += 1;
            }
        }
        if (face_count == 0) {
            // Nothing elementary to bound; keep the whole system rather than emit no
            // circuit at all, which would read as "these atoms delocalize across no bond".
            // Every atom of the union carries a span edge here, so the union is the right
            // atom set in this branch and only in this branch.
            for (int i = 0; i < n; ++i) s.edge_rows[i] |= s.span_rows[i];
            s.atom_mask |= union_mask;
            continue;
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (s.face_incidence[i][j] == 1) {
                    s.edge_rows[i] |= 1ul << j;
                    s.atom_mask |= 1ul << i;
                }
            }
        }
    }

    // Circuits the input asserted that perception has not accounted for. An asserted
    // delocalized edge carries no bond order, so a connected run of them is either a
    // circuit or a claim that cannot be one. Only a component in which every atom has
    // exactly two asserted neighbours is a simple cycle readable by a walk; anything else
    // is left alone and reported as an orphan, because resolving it would mean searching
    // for which of its cycles the author meant.
    int examined_vectors = 0;
    int basis_edges = 0;
    for (int c = 0; c < cycle_count; ++c) {
        arom_cycle_rows(s, c, s.degree, n);
        basis_edges = arom_add_examined_cycle(s, n, examined_vectors, basis_edges);
        if (basis_edges < 0) return REJECT_PRECURSOR_CAPACITY;
        examined_vectors += 1;
        if (examined_vectors >= GF2_MAX_VECTORS) break;
    }

    ulong asserted[BUCKET];
    for (int i = 0; i < n; ++i) {
        asserted[i] = 0ul;
        for (int j = 0; j < n; ++j) {
            const uchar bond = s.comp_bonds[i][j];
            if (bond == BOND_NONE) continue;
            if ((bond & (BOND_SINGLE | BOND_DOUBLE | BOND_TRIPLE)) != 0) continue;
            if (((s.edge_rows[i] >> j) & 1ul) != 0ul) continue;
            asserted[i] |= 1ul << j;
        }
    }

    ulong asserted_seen = 0ul;
    for (int root = 0; root < n; ++root) {
        if (asserted[root] == 0ul) continue;
        if (((asserted_seen >> root) & 1ul) != 0ul) continue;
        // Collect the connected run of asserted edges containing this atom.
        ulong component = 1ul << root;
        bool grew = true;
        while (grew) {
            grew = false;
            for (int atom = 0; atom < n; ++atom) {
                if (((component >> atom) & 1ul) == 0ul) continue;
                const ulong reachable = asserted[atom] & ~component;
                if (reachable != 0ul) { component |= reachable; grew = true; }
            }
        }
        asserted_seen |= component;

        int size = 0;
        bool every_degree_two = true;
        for (int atom = 0; atom < n; ++atom) {
            if (((component >> atom) & 1ul) == 0ul) continue;
            size += 1;
            if (popcount(asserted[atom]) != 2) every_degree_two = false;
        }
        if (size < 3 || !every_degree_two) continue;

        // Walk it, so an authored ring is validated at whatever size it was written rather
        // than being bounded by what enumeration can afford to search.
        for (int i = 0; i < n; ++i) s.degree[i] = 0ul;
        int start = -1;
        for (int atom = 0; atom < n && start < 0; ++atom) {
            if (((component >> atom) & 1ul) != 0ul) start = atom;
        }
        int previous = start;
        int current = ctz(asserted[start]);
        s.degree[start] |= 1ul << current;
        s.degree[current] |= 1ul << start;
        while (current != start) {
            const ulong onward = asserted[current] & ~(1ul << previous);
            const int step = ctz(onward);
            s.degree[current] |= 1ul << step;
            s.degree[step] |= 1ul << current;
            previous = current;
            current = step;
        }

        if (!arom_system_is_aromatic(s, component, s.degree, n, config_index, config_rows)) {
            // The circuit was found by walking the asserted edges, so it is cyclic by
            // construction; what failed is the electron count. Falling through would leave
            // its edges orphaned and report the opposite of what is true.
            return REJECT_DELOCALIZED_SYSTEM_INVALID;
        }
        s.atom_mask |= component;
        for (int i = 0; i < n; ++i) s.edge_rows[i] |= s.degree[i];
        if (examined_vectors < GF2_MAX_VECTORS) {
            basis_edges = arom_add_examined_cycle(s, n, examined_vectors, basis_edges);
            if (basis_edges < 0) return REJECT_PRECURSOR_CAPACITY;
            examined_vectors += 1;
        }
    }

    // Refuse a ring that could delocalize but is larger than perception looks. Only the
    // *existence* of such a ring has to be decided: restricted to the subgraph of atoms
    // that could belong to a circuit at all, that subgraph's cycle rank E - V + components
    // is the number of independent cycles it contains, and the cycles perception examined
    // span a subspace whose dimension is their GF(2) rank. If the whole space is larger,
    // some independent cycle was never examined. Comparing ranks rather than deleting edges
    // is what makes this exact.
    int eligible_edges = 0;
    int eligible_nodes = 0;
    for (int atom = 0; atom < n; ++atom) {
        if (((can_delocalize >> atom) & 1ul) == 0ul) continue;
        const ulong row = s.adjacency[atom] & can_delocalize;
        if (row == 0ul) continue;
        eligible_nodes += 1;
        eligible_edges += popcount(row);
    }
    eligible_edges /= 2;
    if (eligible_edges > 0) {
        int label[BUCKET];
        for (int atom = 0; atom < n; ++atom) label[atom] = atom;
        bool merged = true;
        while (merged) {
            merged = false;
            for (int atom = 0; atom < n; ++atom) {
                if (((can_delocalize >> atom) & 1ul) == 0ul) continue;
                ulong row = s.adjacency[atom] & can_delocalize;
                while (row != 0ul) {
                    const int other = ctz(row);
                    row &= row - 1ul;
                    if (label[other] < label[atom]) { label[atom] = label[other]; merged = true; }
                    else if (label[atom] < label[other]) { label[other] = label[atom]; merged = true; }
                }
            }
        }
        int components = 0;
        for (int atom = 0; atom < n; ++atom) {
            if (((can_delocalize >> atom) & 1ul) == 0ul) continue;
            if ((s.adjacency[atom] & can_delocalize) == 0ul) continue;
            if (label[atom] == atom) components += 1;
        }
        const int cycle_rank = eligible_edges - eligible_nodes + components;
        if (cycle_rank > 0 && arom_gf2_rank(s, examined_vectors) < cycle_rank) {
            return REJECT_REQUIRES_DELOCALIZED_ANNOTATION;
        }
    }

    // Delocalization is a property of a circuit, so an edge asserting it outside one
    // asserts something false.
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const uchar bond = s.comp_bonds[i][j];
            if (bond == BOND_NONE) continue;
            if ((bond & (BOND_SINGLE | BOND_DOUBLE | BOND_TRIPLE)) != 0) continue;
            if (((s.edge_rows[i] >> j) & 1ul) == 0ul) return REJECT_DELOCALIZED_EDGE_NOT_CYCLIC;
        }
    }

    return REJECT_NONE;
}

// Ring-or-chain topology for every present edge of the component, for any ring size.
// A bond is a ring bond iff it is not a bridge, so this is a bridge finder, not a ring
// enumerator; bond_in_short_cycle_rows below answers the different, bounded question the
// WL identity token asks ("is this edge in a 3-8 ring") and misses rings of nine or more.
//
// Mirrors CPU ring_edge_mask (vesma/rings.py): build a spanning forest, then for every
// non-tree edge mark that edge plus the tree path joining its endpoints. The union over
// all non-tree edges is exactly the non-bridge set whatever forest is chosen, so the
// device does not have to reproduce the CPU's traversal order to agree with it.
//
// Root-to-atom tree paths are carried as bitmasks: tree edge (parent[v], v) is named by
// bit v, so path[v] = path[parent[v]] | (1 << v) and the tree path between u and v is
// path[u] ^ path[v] -- the prefix they share above their lowest common ancestor cancels.
// O(n + e) 64-bit operations: no recursion, and no per-edge search.
void ring_bond_rows(device Scratch& s, int n) {
    for (int v = 0; v < n; ++v) {
        s.ring_rows[v] = 0ul;
        s.ring_path[v] = 0ul;
        s.ring_children[v] = 0ul;
        s.ring_parent_bit[v] = 0ul;
    }
    ulong visited = 0ul;
    for (int root = 0; root < n; ++root) {
        if ((visited & (1ul << root)) != 0ul) continue;
        visited |= 1ul << root;
        ulong frontier = 1ul << root;
        while (frontier != 0ul) {
            ulong next = 0ul;
            for (ulong a = frontier; a != 0ul; a &= a - 1ul) next |= s.adjacency[ctz(a)];
            next &= ~visited;
            for (ulong a = next; a != 0ul; a &= a - 1ul) {
                const int v = ctz(a);
                // The parent is drawn from the previous level, whose path mask is final.
                const int parent = ctz(s.adjacency[v] & frontier);
                s.ring_path[v] = s.ring_path[parent] | (1ul << v);
                s.ring_parent_bit[v] = 1ul << parent;
                s.ring_children[parent] |= 1ul << v;
            }
            visited |= next;
            frontier = next;
        }
    }

    // Every non-tree edge closes a cycle; fold its tree path into the cyclic-edge set.
    // Each chord is reached from both of its ends, but OR is idempotent, so counting it
    // twice costs two instructions and saves a shift by 64 when filtering to one end.
    ulong cyclic = 0ul;  // bit v set iff tree edge (parent[v], v) lies on some cycle
    for (int v = 0; v < n; ++v) {
        const ulong tree = s.ring_children[v] | s.ring_parent_bit[v];
        for (ulong a = s.adjacency[v] & ~tree; a != 0ul; a &= a - 1ul) {
            cyclic |= s.ring_path[v] ^ s.ring_path[ctz(a)];
        }
    }
    for (int v = 0; v < n; ++v) {
        const ulong tree = s.ring_children[v] | s.ring_parent_bit[v];
        s.ring_rows[v] =
            (s.adjacency[v] & ~tree)                                   // chords are cyclic
            | (s.ring_children[v] & cyclic)                            // cyclic child edges
            | (((cyclic >> v) & 1ul) != 0ul ? s.ring_parent_bit[v] : 0ul);
    }
}

inline ulong horton_upper_row(device const Scratch& s, int u) {
    return s.ring_rows[u] & ~((2ul << u) - 1ul);
}

inline int horton_lead(ulong lo, ulong hi) {
    if (hi != 0ul) return 127 - (int)clz(hi);
    if (lo != 0ul) return 63 - (int)clz(lo);
    return -1;
}

inline int horton_edge_index(device const Scratch& s, int a, int b) {
    const int lo = min(a, b);
    const int hi = max(a, b);
    return s.horton_prefix[lo] + (int)popcount(horton_upper_row(s, lo) & ((1ul << hi) - 1ul));
}

// The neighbours of `atom` one step nearer `root` -- its predecessors in the shortest-path DAG --
// that are not below the root: a cycle is collected from its lowest atom only.
inline ulong horton_pred(device const Scratch& s, int root, int atom) {
    const int d = s.horton_dist[atom];
    ulong found = 0ul;
    for (ulong walk = s.ring_rows[atom] & ~((1ul << root) - 1ul); walk != 0ul; walk &= walk - 1ul) {
        const int v = ctz(walk);
        if ((int)s.horton_dist[v] + 1 == d) found |= 1ul << v;
    }
    return found;
}

void path_start(device Scratch& s, int w, int root, int target) {
    s.path_level[w] = 0;
    s.path_fresh[w] = 1;
    s.path_node[w][0] = (uchar)target;
    s.path_atoms[w][0] = 1ul << target;
    s.path_lo[w][0] = 0ul;
    s.path_hi[w][0] = 0ul;
    s.path_choice[w][0] = horton_pred(s, root, target);
}

// The next shortest path from the enumerator's target to `root`, left at path_*[w][level];
// false when there are no more, or the budget ran out (path_budget < 0 tells them apart).
bool path_next(device Scratch& s, int w, int root) {
    int l = s.path_level[w];
    if (s.path_fresh[w] != 0) {
        s.path_fresh[w] = 0;
    } else {
        if (l == 0) return false;
        l -= 1;  // back off the root the last path ended on
    }
    for (;;) {
        if (s.path_node[w][l] == root) { s.path_level[w] = l; return true; }
        if (--s.path_budget < 0) return false;
        const ulong choices = s.path_choice[w][l];
        if (choices == 0ul) {
            if (l == 0) return false;
            l -= 1;
            continue;
        }
        const int u = ctz(choices);
        s.path_choice[w][l] = choices & (choices - 1ul);
        const int e = horton_edge_index(s, u, s.path_node[w][l]);
        s.path_node[w][l + 1] = (uchar)u;
        s.path_atoms[w][l + 1] = s.path_atoms[w][l] | (1ul << u);
        s.path_lo[w][l + 1] = s.path_lo[w][l] | (e < 64 ? 1ul << e : 0ul);
        s.path_hi[w][l + 1] = s.path_hi[w][l] | (e >= 64 ? 1ul << (e - 64) : 0ul);
        s.path_choice[w][l + 1] = horton_pred(s, root, u);
        l += 1;
    }
}

// R (relevant cycles through each atom) and r (the smallest), from s.ring_rows, into
// s.ring_count / s.ring_smallest; false when a cap was reached and the values are unknown.
//
// The same algorithm as vesma.rings.relevant_ring_membership, so a precursor built here and
// the same molecule built on the host carry the same R and r. A relevant cycle is isometric, so
// from any root on it it is two shortest paths plus the edge joining their ends: candidates
// come from *every* pair of shortest paths -- one canonical path per atom would miss cycles
// that differ only where paths tie, and the counts would follow atom order. They are built
// size by size (a root's paths to the ends of edge (x, y) close a cycle of d(x) + d(y) + 1),
// each size tested by GF(2) against the basis of the smaller ones, until the basis reaches the
// cycle rank. Each cycle is collected only from its lowest atom -- isometric, it is two shortest
// paths from there too -- so paths never step below the root, while distances stay the graph's.
// A ring system that is one lone ring is answered directly: R = 1, r = its size.
bool ring_membership(device Scratch& s, int n) {
    for (int a = 0; a < n; ++a) { s.ring_count[a] = 0; s.ring_smallest[a] = 0; }
    int edge_count = 0;
    ulong active = 0ul;
    for (int u = 0; u < n; ++u) {
        if (s.ring_rows[u] != 0ul) active |= 1ul << u;
        s.horton_prefix[u] = (uchar)min(edge_count, 255);
        for (ulong walk = horton_upper_row(s, u); walk != 0ul; walk &= walk - 1ul) {
            if (edge_count < HORTON_EDGE_CAP) {
                s.horton_eu[edge_count] = (uchar)u;
                s.horton_ev[edge_count] = (uchar)ctz(walk);
            }
            ++edge_count;
        }
    }
    if (edge_count > HORTON_EDGE_CAP) return false;

    int mu = 0;
    ulong complex_atoms = 0ul;
    ulong seen = 0ul;
    for (ulong pending = active; pending != 0ul; pending = active & ~seen) {
        ulong component = 1ul << ctz(pending);
        ulong frontier = component;
        seen |= frontier;
        while (frontier != 0ul) {
            ulong reach = 0ul;
            for (ulong walk = frontier; walk != 0ul; walk &= walk - 1ul) reach |= s.ring_rows[ctz(walk)];
            frontier = reach & ~seen;
            seen |= frontier;
            component |= frontier;
        }
        int degree_sum = 0;
        for (ulong walk = component; walk != 0ul; walk &= walk - 1ul) {
            degree_sum += (int)popcount(s.ring_rows[ctz(walk)]);
        }
        const int atoms_c = (int)popcount(component);
        const int mu_c = degree_sum / 2 - atoms_c + 1;
        if (mu_c == 1) {
            for (ulong walk = component; walk != 0ul; walk &= walk - 1ul) {
                const int a = ctz(walk);
                s.ring_count[a] = 1;
                s.ring_smallest[a] = (uchar)atoms_c;
            }
        } else {
            mu += mu_c;
            complex_atoms |= component;
        }
    }
    if (mu == 0) return true;


    for (int i = 0; i < HORTON_EDGE_CAP; ++i) s.horton_lead_slot[i] = -1;
    s.path_budget = HORTON_PATH_BUDGET;
    int rank = 0;
    for (int size = 3; size <= n && rank < mu; ++size) {
        int cand_count = 0;
        for (ulong roots = complex_atoms; roots != 0ul; roots &= roots - 1ul) {
            const int root = ctz(roots);
            const ulong root_bit = 1ul << root;
            // Graph distances from the root -- through every atom, lower ones included.
            for (int a = 0; a < n; ++a) s.horton_dist[a] = HORTON_FAR;
            s.horton_dist[root] = 0;
            ulong visited = root_bit;
            ulong frontier = root_bit;
            for (int d = 1; frontier != 0ul; ++d) {
                ulong next = 0ul;
                for (ulong walk = frontier; walk != 0ul; walk &= walk - 1ul) next |= s.ring_rows[ctz(walk)];
                next &= ~visited;
                for (ulong walk = next; walk != 0ul; walk &= walk - 1ul) s.horton_dist[ctz(walk)] = (uchar)d;
                visited |= next;
                frontier = next;
            }
            for (int index = 0; index < edge_count; ++index) {
                const int x = s.horton_eu[index];
                const int y = s.horton_ev[index];
                if (x < root) continue;  // the cycle's lowest atom is the root; y > x
                const int dx = s.horton_dist[x];
                const int dy = s.horton_dist[y];
                if (dx == HORTON_FAR || dy == HORTON_FAR || dx + dy + 1 != size) continue;
                path_start(s, 0, root, x);
                while (path_next(s, 0, root)) {
                    const int lx = s.path_level[0];
                    path_start(s, 1, root, y);
                    while (path_next(s, 1, root)) {
                        const int ly = s.path_level[1];
                        if ((s.path_atoms[0][lx] & s.path_atoms[1][ly]) != root_bit) continue;
                        const ulong lo = s.path_lo[0][lx] | s.path_lo[1][ly] | (index < 64 ? 1ul << index : 0ul);
                        const ulong hi = s.path_hi[0][lx] | s.path_hi[1][ly] | (index >= 64 ? 1ul << (index - 64) : 0ul);
                        bool duplicate = false;
                        for (int c = 0; c < cand_count; ++c) {
                            if (s.horton_cand_lo[c] == lo && s.horton_cand_hi[c] == hi) { duplicate = true; break; }
                        }
                        if (duplicate) continue;
                        if (cand_count == HORTON_CAND_CAP) return false;
                        s.horton_cand_lo[cand_count] = lo;
                        s.horton_cand_hi[cand_count] = hi;
                        ++cand_count;
                    }
                    if (s.path_budget < 0) return false;
                }
                if (s.path_budget < 0) return false;
            }
        }

        const int frozen = rank;
        for (int c = 0; c < cand_count; ++c) {
            ulong lo = s.horton_cand_lo[c];
            ulong hi = s.horton_cand_hi[c];
            for (;;) {  // relevance: reduce against the basis of the smaller sizes only
                const int lead = horton_lead(lo, hi);
                if (lead < 0) break;
                const int slot = s.horton_lead_slot[lead];
                if (slot < 0 || slot >= frozen) break;
                lo ^= s.horton_basis_lo[slot];
                hi ^= s.horton_basis_hi[slot];
            }
            if (horton_lead(lo, hi) < 0) continue;
            ulong atoms = 0ul;
            for (ulong w = s.horton_cand_lo[c]; w != 0ul; w &= w - 1ul) {
                const int e = ctz(w);
                atoms |= (1ul << s.horton_eu[e]) | (1ul << s.horton_ev[e]);
            }
            for (ulong w = s.horton_cand_hi[c]; w != 0ul; w &= w - 1ul) {
                const int e = 64 + ctz(w);
                atoms |= (1ul << s.horton_eu[e]) | (1ul << s.horton_ev[e]);
            }
            for (; atoms != 0ul; atoms &= atoms - 1ul) {
                const int a = ctz(atoms);
                s.ring_count[a] += 1;
                if (s.ring_smallest[a] == 0) s.ring_smallest[a] = (uchar)size;
            }
            for (;;) {  // then grow the basis with what is left
                const int lead = horton_lead(lo, hi);
                if (lead < 0) break;
                const int slot = s.horton_lead_slot[lead];
                if (slot < 0) {
                    if (rank == HORTON_BASIS_CAP) return false;
                    s.horton_basis_lo[rank] = lo;
                    s.horton_basis_hi[rank] = hi;
                    s.horton_lead_slot[lead] = (char)rank;
                    ++rank;
                    break;
                }
                lo ^= s.horton_basis_lo[slot];
                hi ^= s.horton_basis_hi[slot];
            }
        }
    }
    return true;
}

// Re-perceive aromaticity and refresh degree/H/valence atom invariants on the
// contiguous component s.comp_words / s.comp_bonds. Mirrors
// normalize_aromaticity_and_hydrogens in cuda_retro.py.
uint normalize_aromaticity_and_hydrogens(
    device Scratch& s,
    int n,
    device const uint* config_index,
    device const uint* config_rows,
    thread uint& profile_cycles,
    uint profile_mode,
    thread bool& profile_stopped
) {
    for (int atom = 0; atom < n; ++atom) {
        s.z[atom] = exact_z_from_words(
            s.comp_words[atom][0], s.comp_words[atom][1],
            s.comp_words[atom][2], s.comp_words[atom][3]);
        s.charge[atom] = charge_from_mask(s.comp_words[atom][5]);
    }

    const uint refusal =
        perceive_aromaticity_device(s, n, config_index, config_rows,
                                    profile_cycles, profile_mode);
    if (refusal != REJECT_NONE) return refusal;
    if (profile_mode == 4u || profile_mode == 5u) {
        profile_stopped = true;
        return REJECT_NONE;
    }

    for (int atom = 0; atom < n; ++atom) {
        s.comp_words[atom][4] =
            (((s.atom_mask >> atom) & 1ul) != 0ul) ? AROMATIC : ALIPHATIC;
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const bool aromatic_edge = ((s.edge_rows[i] >> j) & 1ul) != 0ul;
            s.comp_bonds[i][j] = arom_canonical_bond(s.comp_bonds[i][j], aromatic_edge);
        }
    }

    // arom_canonical_bond can clear an edge the aromatic reclass refused, so ring
    // topology is read off the final graph rather than the pre-reclass adjacency.
    for (int atom = 0; atom < n; ++atom) {
        ulong row = 0ul;
        for (int other = 0; other < n; ++other) {
            if (s.comp_bonds[atom][other] != BOND_NONE) row |= 1ul << other;
        }
        s.adjacency[atom] = row;
    }
    ring_bond_rows(s, n);
    const bool rings_known = ring_membership(s, n);

    for (int atom = 0; atom < n; ++atom) {
        int degree = 0;
        int heavy_half = 0;
        for (int other = 0; other < n; ++other) {
            const uchar bond = s.comp_bonds[atom][other];
            if (bond != BOND_NONE) {
                degree += 1;
                heavy_half += heavy_valence_half_from_bond(bond);
            }
        }
        const int heavy_valence = (heavy_half + 1) / 2;
        // Hydrogen is molecular state decided by the hydrogen law during surgery, not
        // something perception infers, so this is the pinned count carried through.
        const int h_count = s.comp_ph[atom];
        const int explicit_count = s.comp_eh[atom] < 0 ? 0 : s.comp_eh[atom];

        // Every atom the edit produced must occupy a declared closed-shell configuration.
        // Checking it here, on the perceived graph, is what turns "a template asked for a
        // state the model does not declare" into a named refusal instead of a molecule
        // nobody can justify -- an [NH2] with no other bond is an aminyl radical, and the
        // model has nothing to say about it. The graph valence comes off the selected
        // configuration rather than being counted beside it, so `v` cannot report a
        // valence that merely adds up.
        int order_sum = 0;
        bool delocalized = false;
        for (int other = 0; other < n; ++other) {
            const uchar bond = s.comp_bonds[atom][other];
            if (bond == BOND_NONE) continue;
            order_sum += arom_bond_order(bond);
            if ((bond & BOND_AROMATIC) != 0) delocalized = true;
        }
        ConfigRow state;
        const uint state_status = config_select(
            config_index, config_rows, s.z[atom], s.charge[atom], degree, order_sum,
            h_count, delocalized, state);
        if (state_status != CONFIG_OK) return reject_for_state_failure(state_status);
        // Heteroaromatic H is explicit ([nH]-style); aromatic B/C store all H as
        // implicit (degree-derived, never explicit); aliphatic atoms carry the
        // remainder. Mirrors cmol_from_seed_data in vesma/cmol.py.
        int inferred_h;
        if (s.comp_words[atom][4] == AROMATIC && s.z[atom] != 5 && s.z[atom] != 6) {
            inferred_h = 0;
        } else if (s.comp_words[atom][4] == AROMATIC) {
            inferred_h = h_count;
        } else {
            inferred_h = h_count - explicit_count;
            if (inferred_h < 0) inferred_h = 0;
        }
        s.comp_words[atom][6] = bucket_mask_exact(degree);
        s.comp_words[atom][7] = bucket_mask_exact(h_count);
        s.comp_words[atom][8] = bucket_mask_exact(inferred_h);
        s.comp_words[atom][9] = bucket_mask_exact(state.graph_valence);
        s.comp_words[atom][10] = bucket_mask_exact(degree + h_count);
        // R and r: the relevant cycles, as on the host; unknown only past ring_membership's caps.
        s.comp_words[atom][11] = rings_known ? bucket_mask_exact(min((int)s.ring_count[atom], 7)) : WILDCARD;
        s.comp_words[atom][12] = rings_known ? bucket_mask_exact(min((int)s.ring_smallest[atom], 7)) : WILDCARD;
        const int ring_degree = (int)popcount(s.ring_rows[atom]);
        s.comp_words[atom][ISOTOPE_WORD] =
            (s.comp_words[atom][ISOTOPE_WORD] & ISOTOPE_FIELD) | bucket_mask_exact(ring_degree);
    }
    return REJECT_NONE;
}

// An edge lies on a 3-8-member cycle iff deleting it leaves a path of 2-7
// edges between its endpoints. Bounded bitset BFS mirrors CUDA's
// bond_in_short_cycle_rows; each shortest path is simple by construction.
bool bond_in_short_cycle_rows(threadgroup WLScratch& s, int left, int right) {
    const ulong target = 1ul << right;
    ulong visited = 1ul << left;
    ulong frontier = visited;
    for (int distance = 1; distance <= 7; ++distance) {
        ulong neighbors = 0ul;
        ulong active = frontier;
        while (active != 0ul) {
            const int atom = ctz(active);
            ulong row = s.adjacency_rows[atom];
            if (atom == left) row &= ~target;
            neighbors |= row;
            active &= active - 1ul;
        }
        frontier = neighbors & ~visited;
        if (distance >= 2 && (frontier & target) != 0ul) return true;
        if (frontier == 0ul) return false;
        visited |= frontier;
    }
    return false;
}

inline uchar identity_bond_class(threadgroup WLScratch& s, int left, int right,
                                 bool scan_short_ring) {
    const uchar bond = bond_class(s.comp_bonds[left][right]);
    if (bond == BOND_NONE || bond == BOND_TRIPLE) return bond;
    if (scan_short_ring && bond_in_short_cycle_rows(s, left, right)) return BOND_AROMATIC;
    const uchar order = bond & (BOND_SINGLE | BOND_DOUBLE | BOND_TRIPLE);
    if (order != 0) return order;
    return BOND_AROMATIC;
}

// Short-ring WL identity token for the contiguous component, bit-identical to CPU
// short_ring_wl_molecule_id. H is re-derived from the canonical bonds
// (total_hydrogens_for_identity) so identity is input-invariant.
//
// Cooperative over WL_THREADS lanes of one threadgroup (one precursor per group): the
// per-atom / per-pair loops are lane-striped with threadgroup barriers between phases;
// the cheap tail (label sort + payload + xxh64) runs on lane 0. Result (lo/hi) is valid
// only on lane 0. `partial_edges` is a threadgroup scratch array of WL_THREADS ints for
// the edge-count reduction. Every lane in the group must call this (uniform barriers).
void component_token(
    threadgroup WLScratch& s, int n, uint tid,
    threadgroup int* partial_edges,
    uint profile_mode,
    thread ulong& lo_out, thread ulong& hi_out
) {
    // Phase 1: canonical identity bonds (each unordered pair handled once by lane
    // owning `left`; symmetric writes are disjoint, so no race) + per-lane edge count.
    int local_edges = 0;
    for (int left = (int)tid; left < n; left += WL_THREADS) {
        s.identity_bonds[left][left] = BOND_NONE;
        for (int right = left + 1; right < n; ++right) {
            const uchar bond_class = identity_bond_class(
                s, left, right, profile_mode != 4u);
            s.identity_bonds[left][right] = bond_class;
            s.identity_bonds[right][left] = bond_class;
            if (bond_class != BOND_NONE) local_edges += 1;
        }
    }
    partial_edges[tid] = local_edges;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (profile_mode == 1u || profile_mode == 4u) {
        if (tid == 0) {
            int edge_count = 0;
            for (uint t = 0; t < WL_THREADS; ++t) edge_count += partial_edges[t];
            lo_out = (ulong)edge_count;
        }
        return;
    }

    // Phase 2: initial per-atom WL labels (lane-striped over atoms).
    for (int local = (int)tid; local < n; local += WL_THREADS) {
        const int z = exact_z_from_words(
            s.comp_words[local][0], s.comp_words[local][1],
            s.comp_words[local][2], s.comp_words[local][3]);
        const int charge = charge_from_mask(s.comp_words[local][5]);
        // Identity hydrogen is the stored total (word 7), which under the closed-shell
        // model is the count the hydrogen law decided rather than anything perception
        // inferred. CPU's total_hydrogens_for_identity reaches the same value by the same
        // route: an identity seed built from a molecule writes its hydrogen explicitly, and
        // a written count is taken exactly rather than re-derived.
        //
        // This replaces a re-derivation from the canonical bonds. That existed because the
        // old model's stored H was input-variant -- it carried stale aromatic bits through
        // surgery -- so the same molecule reached by two routes could hash two ways. The
        // law removes the cause rather than compensating for it: H is molecular state, is
        // decided once per atom, and survives the split unchanged.
        const int h_count = exact_bucket_from_mask(s.comp_words[local][7]);
        ulong deg = 0ul;
        for (int other = 0; other < n; ++other) {
            if (s.identity_bonds[local][other] != BOND_NONE) deg += 1ul;
        }
        s.degree[local] = deg;
        ulong label = mix_uint64((ulong)z);
        label = combine_uint64(label, (ulong)(long)charge);
        label = combine_uint64(label, (ulong)h_count);
        label = combine_uint64(label, deg);
        s.labels[local] = label;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (profile_mode == 2u) {
        if (tid == 0) {
            ulong checksum = 0ul;
            for (int local = 0; local < n; ++local) checksum ^= s.labels[local];
            lo_out = checksum;
        }
        return;
    }

    // Phase 3: WL message-passing rounds (read all labels, write own next_labels).
    for (int round = 0; round < GPU_WL_ROUNDS; ++round) {
        for (int local = (int)tid; local < n; local += WL_THREADS) {
            ulong aggregate = 0ul;
            for (int other = 0; other < n; ++other) {
                const uchar bond = s.identity_bonds[local][other];
                if (bond != BOND_NONE) {
                    aggregate += combine_uint64(s.labels[other], (ulong)bond);
                }
            }
            ulong label = combine_uint64(s.labels[local], s.degree[local]);
            label = combine_uint64(label, aggregate);
            s.next_labels[local] = mix_uint64(label);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int local = (int)tid; local < n; local += WL_THREADS) {
            s.labels[local] = s.next_labels[local];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (profile_mode == 3u) {
        if (tid == 0) {
            ulong checksum = 0ul;
            for (int local = 0; local < n; ++local) checksum ^= s.labels[local];
            lo_out = checksum;
        }
        return;
    }

    // Phase 4: canonical sort + payload + xxh64 (serial on lane 0; cheap vs the above).
    if (tid == 0) {
        int edge_count = 0;
        for (uint t = 0; t < WL_THREADS; ++t) edge_count += partial_edges[t];
        for (int left = 0; left < n; ++left) {
            for (int right = left + 1; right < n; ++right) {
                if (s.labels[right] < s.labels[left]) {
                    const ulong tmp = s.labels[left];
                    s.labels[left] = s.labels[right];
                    s.labels[right] = tmp;
                }
            }
        }
        s.payload[0] = GPU_WL_HASH_VERSION;
        s.payload[1] = GPU_WL_ROUNDS;
        s.payload[2] = (ulong)n;
        s.payload[3] = (ulong)edge_count;
        for (int idx = 0; idx < n; ++idx) {
            s.payload[4 + idx] = s.labels[idx];
        }
        lo_out = xxh64_uint64_array(s.payload, 4 + n, WL_FINAL_LOW_SEED);
        hi_out = xxh64_uint64_array(s.payload, 4 + n, WL_FINAL_HIGH_SEED);
    }
}

// Reports the compile-time constants this shader was built with.
//
// MSL lives in a raw string here and cannot `#include` the host's header, so every one of these
// is declared twice: once as a `#define` above and once in `vesma.limits`. A disagreement is not a
// wrong answer a parity test would catch -- it is a buffer indexed past its end, or a row
// truncated in silence. Asking the shader what it compiled turns that into a test failure.
kernel void constants_probe(device uint* out [[buffer(0)]],
                            uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    out[0] = (uint)BUCKET;
    out[1] = (uint)EDITED_BUCKET;
    out[2] = (uint)ATOM_WORD_COUNT;
    out[3] = (uint)MAX_COMPONENTS;
    out[4] = (uint)WL_THREADS;
    out[5] = (uint)MATCH_ATOM_CAP;
    out[6] = (uint)MAX_REACTION_EDGES;
}

// Reports the device-side sizeof(Scratch) the kernel uses to stride scratch_pool,
// so the host allocates the pool with the exact same element size.
kernel void scratch_size_probe(device uint* out [[buffer(0)]],
                               uint tid [[thread_position_in_grid]]) {
    if (tid == 0) out[0] = (uint)sizeof(Scratch);
}

// Runs the three configuration selectors on device for a batch of written atom states, so
// the shader's own answers can be compared against configurations.py rather than inferred
// from whether perception happened to agree. One thread per query; 6 ints in, 16 uints out.
kernel void config_probe(
    device const uint* config_index [[buffer(0)]],
    device const uint* config_rows [[buffer(1)]],
    device const int* queries [[buffer(2)]],
    device uint* out [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const int z = queries[gid * 6 + 0];
    const int charge = queries[gid * 6 + 1];
    const int sigma_degree = queries[gid * 6 + 2];
    const int bond_order_sum = queries[gid * 6 + 3];
    const int total_h = queries[gid * 6 + 4];
    const bool delocalized = queries[gid * 6 + 5] != 0;

    device uint* slot = out + gid * 16;
    for (uint i = 0u; i < 16u; ++i) slot[i] = 0u;

    ConfigRow row = config_row_empty();
    slot[0] = config_select(config_index, config_rows, z, charge, sigma_degree,
                            bond_order_sum, total_h, delocalized, row);
    slot[1] = (uint)row.sigma_capacity;
    slot[2] = (uint)row.graph_valence;
    slot[3] = (uint)(row.pi + CONFIG_PI_BIAS);

    row = config_row_empty();
    slot[4] = config_select_circuit(config_index, config_rows, z, charge, sigma_degree,
                                    total_h, row);
    slot[5] = (uint)row.sigma_capacity;
    slot[6] = (uint)row.graph_valence;
    slot[7] = (uint)(row.pi + CONFIG_PI_BIAS);

    row = config_row_empty();
    int hydrogens = 0;
    slot[8] = config_unwritten(config_index, config_rows, z, charge, sigma_degree,
                              bond_order_sum, delocalized, row, hydrogens);
    slot[9] = (uint)row.sigma_capacity;
    slot[10] = (uint)row.graph_valence;
    slot[11] = (uint)(row.pi + CONFIG_PI_BIAS);
    slot[12] = (uint)hydrogens;

    // The raw span the selectors worked from, so a disagreement can be attributed to the
    // table lookup or to the filter rather than guessed at. This is not decoration: it is
    // what showed that a signed subscript here was reading the wrong table slot while the
    // table itself was correct, which the selector answers alone could not have told apart.
    slot[13] = config_span(config_index, z, charge);
}

kernel void retro_construct(
    device const uint* product_atoms [[buffer(0)]],
    device const uchar* product_bonds [[buffer(1)]],
    device const uint* product_atom_counts [[buffer(2)]],
    device const ReactionData* reactions [[buffer(3)]],
    device const int* assignments [[buffer(4)]],
    device const uint* pair_product_indices [[buffer(5)]],
    device const uint* pair_reaction_indices [[buffer(6)]],
    // Three parallel outputs rather than one interleaved `GpuMol`, which is the layout every
    // molecule on this arm already has and the one CUDA's retro kernel has always written. The
    // struct existed so a fused WL pass could write `hash_lo`/`hash_hi` in place on it; with the
    // keyer its own entry point over a batch, nothing needs the interleaving -- and the host stops
    // de-interleaving it straight back into these three arrays.
    device uint* out_atoms [[buffer(7)]],
    device uchar* out_bonds [[buffer(8)]],
    device uint* out_atom_counts [[buffer(9)]],
    device RetroStatus* output_status [[buffer(10)]],
    device atomic_uint* output_counter [[buffer(11)]],
    device Scratch* scratch_pool [[buffer(12)]],
    device const uint* config_index [[buffer(13)]],
    device const uint* config_rows [[buffer(14)]],
    constant uint& profile_mode [[buffer(15)]],
    uint pair_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    // Per-assignment working state in device memory (was threadgroup). This frees the
    // ~27 KB threadgroup footprint that capped occupancy at one threadgroup per core,
    // letting many assignments run concurrently. One slot per assignment (pair_idx);
    // the host bounds pair_count per dispatch so the pool stays memory-bounded.
    if (tid != 0) return;
    device Scratch& s = scratch_pool[pair_idx];

    const uint p_idx = pair_product_indices[pair_idx];
    const uint r_idx = pair_reaction_indices[pair_idx];
    device const ReactionData& rxn = reactions[r_idx];
    const int product_atom_count = (int)product_atom_counts[p_idx];
    const int rhs_atom_count = (int)rxn.rhs_atom_count;
    device const int* assign = assignments + (pair_idx * BUCKET);

    output_status[pair_idx].component_count = 0;
    output_status[pair_idx].overflow = 0;
    output_status[pair_idx].atom_overflow = 0;
    output_status[pair_idx].edited_atom_overflow = 0;
    output_status[pair_idx].output_base = 0;
    // Zero is itself a declared reason, so "not refused" has to be written, not left.
    output_status[pair_idx].rejection_reason = REJECT_NONE;
    output_status[pair_idx].undescribed_components = 0u;
    output_status[pair_idx].orphaned_components = 0u;
    output_status[pair_idx].grouping_violations = 0u;
    if (profile_mode != 0u) {
        output_status[pair_idx].profile_stage = 0u;
        output_status[pair_idx].profile_edited_atoms = 0u;
        output_status[pair_idx].profile_component_merges = 0u;
        output_status[pair_idx].profile_kept_components = 0u;
        output_status[pair_idx].profile_perceived_atoms = 0u;
        output_status[pair_idx].profile_aromatic_atoms = 0u;
        output_status[pair_idx].profile_cycles = 0u;
    }

    // The edited graph cannot exceed the surviving product atoms plus the atoms the RHS
    // adds, so only that prefix is ever read. Clearing the whole 256-atom scratch instead
    // wrote 64 KB of bond matrix per assignment whatever its size -- for a typical edit of
    // a few dozen atoms, over an order of magnitude more work than the edit itself.
    int edited_bound = product_atom_count + rhs_atom_count;
    if (edited_bound > EDITED_BUCKET) edited_bound = EDITED_BUCKET;
    for (int atom = 0; atom < edited_bound; ++atom) {
        s.rhs_to_edited[atom] = -1;
        s.comp_labels[atom] = atom;
        s.explicit_h[atom] = 0;
        s.implicit_h[atom] = -1;
        s.edited_h[atom] = -1;
        s.edited_from_product[atom] = -1;
        s.described[atom] = 0;
        s.edited_group[atom] = 0;
        s.edited_map[atom] = 0;
        for (int word = 0; word < ATOM_WORD_COUNT; ++word) s.edited_atoms[atom][word] = 0u;
        for (int other = 0; other < edited_bound; ++other) s.edited_bonds[atom][other] = BOND_NONE;
    }
    for (int atom = 0; atom < BUCKET; ++atom) {
        s.keep_product[atom] = 0;
        s.product_to_edited[atom] = -1;
        s.assignment[atom] = assign[atom];
    }

    for (int atom = 0; atom < product_atom_count; ++atom) s.keep_product[atom] = 1;
    for (uint i = 0; i < rxn.deleted_lhs_count; ++i) {
        const int lhs_idx = rxn.deleted_lhs_indices[i];
        const int product_atom = s.assignment[lhs_idx];
        if (product_atom >= 0 && product_atom < product_atom_count) {
            s.keep_product[product_atom] = 0;
        }
    }

    int edited_count = 0;
    for (int atom = 0; atom < product_atom_count; ++atom) {
        if (!s.keep_product[atom]) continue;
        s.product_to_edited[atom] = edited_count;
        for (int word = 0; word < ATOM_WORD_COUNT; ++word) {
            s.edited_atoms[edited_count][word] =
                product_atoms[p_idx * BUCKET * ATOM_WORD_COUNT + atom * ATOM_WORD_COUNT + word];
        }
        if (s.edited_atoms[edited_count][4] == AROMATIC) {
            int eh = exact_bucket_from_mask(s.edited_atoms[edited_count][7])
                   - exact_bucket_from_mask(s.edited_atoms[edited_count][8]);
            if (eh < 0) eh = 0;
            s.explicit_h[edited_count] = eh;
        }
        s.edited_from_product[edited_count] = atom;
        edited_count += 1;
    }
    for (int left = 0; left < product_atom_count; ++left) {
        if (!s.keep_product[left]) continue;
        const int el = s.product_to_edited[left];
        for (int right = 0; right < product_atom_count; ++right) {
            if (!s.keep_product[right]) continue;
            const int er = s.product_to_edited[right];
            // Surgery and perception reason about bond classes, and perception writes
            // ring-or-chain topology back onto the emitted bonds. Masking here keeps a
            // re-fed precursor's topology bits out of the bond orders, as CUDA's
            // retro_construct_assignments does at the same point.
            s.edited_bonds[el][er] = bond_class(
                product_bonds[p_idx * BUCKET * BUCKET + left * BUCKET + right]);
        }
    }

    for (uint i = 0; i < rxn.common_count; ++i) {
        const int lhs_idx = rxn.common_lhs_indices[i];
        const int rhs_idx = rxn.common_rhs_indices[i];
        const int product_atom = s.assignment[lhs_idx];
        if (product_atom < 0 || product_atom >= product_atom_count) {
            output_status[pair_idx].inconsistent_assignment = 1u;
            output_status[pair_idx].rejection_reason = REJECT_UNSUPPORTED_TEMPLATE;
            return;
        }
        const int edited_atom = s.product_to_edited[product_atom];
        if (edited_atom < 0) {
            output_status[pair_idx].inconsistent_assignment = 2u;
            output_status[pair_idx].rejection_reason = REJECT_UNSUPPORTED_TEMPLATE;
            return;
        }
        s.rhs_to_edited[rhs_idx] = edited_atom;
        s.edited_map[edited_atom] = rxn.common_maps[i];

        // An atom the RHS writes takes its hydrogen count from the edit, not from the state it
        // had in the product. The survivor carry above copies the product's explicit count for
        // aromatics -- an `[nH]` keeps its hydrogen explicit so it writes back as `[nH]` -- and
        // leaving that in place hands `explicit = 1` to an atom the RHS turns into, say, a
        // nitrile nitrogen with no room for it. Mirrors `explicit_h[mapped_edited] = 0` in
        // reaction.py; the `is_single_bit_u32(rhs_h)` branch below already did this for the
        // narrower case where the RHS states a count outright.
        s.explicit_h[edited_atom] = 0;

        // Nuclear conservation: a mapped atom keeps the product's nucleus. A RHS that asserts
        // another isotope refuses the application, as reaction.py does.
        const uint rhs_mass = rxn.rhs_atoms[rhs_idx * ATOM_WORD_COUNT + ISOTOPE_WORD] >> ISOTOPE_SHIFT;
        const uint product_mass = s.edited_atoms[edited_atom][ISOTOPE_WORD] >> ISOTOPE_SHIFT;
        if (rhs_mass != ISOTOPE_ANY && rhs_mass != product_mass) {
            output_status[pair_idx].rejection_reason = REJECT_NUCLEAR_CONSERVATION;
            return;
        }
        const uint rhs_a = rxn.rhs_atoms[rhs_idx * ATOM_WORD_COUNT + 4];
        const uint rhs_h = rxn.rhs_atoms[rhs_idx * ATOM_WORD_COUNT + 7];
        if (rhs_a != WILDCARD && rhs_a != 0u) s.edited_atoms[edited_atom][4] = rhs_a;
        // Charge comes from the RHS only where the RHS wrote one. Reading it off the mask
        // instead cannot distinguish "unconstrained" from "explicitly neutral", and taking
        // the first for the second turns every atom that carried a charge into a charge
        // change -- refusing the application for something the template never asked.
        if (rxn.common_charge_explicit[i] != 0) {
            s.edited_atoms[edited_atom][5] = charge_mask_exact(rxn.common_charge[i]);
        }
        if (is_single_bit_u32(rhs_h)) {
            // The RHS wrote a count. It is an instruction, taken exactly.
            const int h_count = exact_bucket_from_mask(rhs_h);
            s.explicit_h[edited_atom] = 0;
            s.implicit_h[edited_atom] = h_count;
            s.edited_h[edited_atom] = h_count;
        }
    }
    for (int rhs_idx = 0; rhs_idx < rhs_atom_count; ++rhs_idx) {
        if (s.rhs_to_edited[rhs_idx] >= 0) continue;
        if (edited_count >= EDITED_BUCKET) {
            output_status[pair_idx].edited_atom_overflow = 1u;
            return;
        }
        s.rhs_to_edited[rhs_idx] = edited_count;
        for (int word = 0; word < ATOM_WORD_COUNT; ++word) {
            s.edited_atoms[edited_count][word] =
                rxn.rhs_atoms[rhs_idx * ATOM_WORD_COUNT + word];
        }
        // An omitted RHS charge means neutral for an added atom, not a query wildcard.
        s.edited_atoms[edited_count][5] = charge_mask_exact(
            charge_from_mask(s.edited_atoms[edited_count][5]));
        if (s.edited_atoms[edited_count][4] == WILDCARD || s.edited_atoms[edited_count][4] == 0u) {
            s.edited_atoms[edited_count][4] = ALIPHATIC;
        }
        // An added atom is the template's nucleus: the mass it names, else unlabelled.
        if ((s.edited_atoms[edited_count][ISOTOPE_WORD] >> ISOTOPE_SHIFT) == ISOTOPE_ANY) {
            s.edited_atoms[edited_count][ISOTOPE_WORD] &= ~ISOTOPE_FIELD;
        }
        s.edited_map[edited_count] = rxn.rhs_atom_maps[rhs_idx];
        s.edited_from_product[edited_count] = -1;
        if (is_single_bit_u32(s.edited_atoms[edited_count][7])) {
            const int h_count = exact_bucket_from_mask(s.edited_atoms[edited_count][7]);
            s.explicit_h[edited_count] = 0;
            s.implicit_h[edited_count] = h_count;
            s.edited_h[edited_count] = h_count;
        }
        edited_count += 1;
    }

    for (int rhs_idx = 0; rhs_idx < rhs_atom_count; ++rhs_idx) {
        const int edited_atom = s.rhs_to_edited[rhs_idx];
        if (edited_atom < 0) continue;
        s.described[edited_atom] = 1;
        s.edited_group[edited_atom] = (uchar)rxn.rhs_atom_groups[rhs_idx];
    }

    for (uint e = 0; e < rxn.lhs_edge_count; ++e) {
        const int lhs_left = rxn.lhs_edge_left[e];
        const int lhs_right = rxn.lhs_edge_right[e];
        const int product_left = s.assignment[lhs_left];
        const int product_right = s.assignment[lhs_right];
        if (product_left < 0 || product_right < 0) continue;
        const int el = s.product_to_edited[product_left];
        const int er = s.product_to_edited[product_right];
        if (el < 0 || er < 0) continue;
        s.edited_bonds[el][er] = BOND_NONE;
        s.edited_bonds[er][el] = BOND_NONE;
    }
    for (uint e = 0; e < rxn.rhs_edge_count; ++e) {
        const int rhs_left = rxn.rhs_edge_left[e];
        const int rhs_right = rxn.rhs_edge_right[e];
        const int el = s.rhs_to_edited[rhs_left];
        const int er = s.rhs_to_edited[rhs_right];
        if (el < 0 || er < 0) {
            output_status[pair_idx].inconsistent_assignment = 3u;
            output_status[pair_idx].rejection_reason = REJECT_UNSUPPORTED_TEMPLATE;
            return;
        }
        const uchar bond_type = rxn.rhs_edge_bond_types[e];
        s.edited_bonds[el][er] = bond_type;
        s.edited_bonds[er][el] = bond_type;
    }

    // The hydrogen law. Every atom the edit produces has its count decided here, from
    // three sources in decreasing authority: an explicit RHS count *sets* it (already
    // taken above), an unchanged charge and bond environment *copies* it, and a changed
    // environment *completes* it from the configuration the matched atom was already in.
    // Deciding it here is what lets perception take hydrogen as molecular state rather
    // than inferring it -- so no edit can re-select an atom into a different oxidation
    // state behind the caller's back.
    for (int atom = 0; atom < edited_count; ++atom) {
        if (s.edited_h[atom] >= 0) continue;

        const int z = exact_z_from_words(
            s.edited_atoms[atom][0], s.edited_atoms[atom][1],
            s.edited_atoms[atom][2], s.edited_atoms[atom][3]);
        const int charge_after = charge_from_mask(s.edited_atoms[atom][5]);

        int sigma_after = 0;
        int order_after = 0;
        bool delocalized_after = false;
        int histogram_after[4] = {0, 0, 0, 0};
        for (int other = 0; other < edited_count; ++other) {
            const uchar bond = bond_class(s.edited_bonds[atom][other]);
            if (bond == BOND_NONE) continue;
            sigma_after += 1;
            order_after += arom_bond_order(bond);
            if ((bond & BOND_AROMATIC) != 0) delocalized_after = true;
            if (bond & BOND_SINGLE) histogram_after[0] += 1;
            if (bond & BOND_DOUBLE) histogram_after[1] += 1;
            if (bond & BOND_TRIPLE) histogram_after[2] += 1;
            if (bond & BOND_AROMATIC) histogram_after[3] += 1;
        }

        const int source = s.edited_from_product[atom];
        if (source < 0) {
            // An atom the template adds exists only as what the RHS wrote, so its hydrogen
            // is the fewest its declared configurations allow given that bonding. Octet
            // expansion must be evidenced by bonding, never bought with hydrogen.
            ConfigRow configuration;
            int hydrogens = 0;
            const uint status = config_unwritten(
                config_index, config_rows, z, charge_after, sigma_after, order_after,
                delocalized_after, configuration, hydrogens);
            if (status != CONFIG_OK) {
                output_status[pair_idx].rejection_reason =
                    reject_for_config_failure(status);
                return;
            }
            s.edited_h[atom] = hydrogens;
            continue;
        }

        int sigma_before = 0;
        int order_before = 0;
        bool delocalized_before = false;
        int histogram_before[4] = {0, 0, 0, 0};
        for (int other = 0; other < product_atom_count; ++other) {
            const uchar bond = bond_class(
                product_bonds[p_idx * BUCKET * BUCKET + source * BUCKET + other]);
            if (bond == BOND_NONE) continue;
            sigma_before += 1;
            order_before += arom_bond_order(bond);
            if ((bond & BOND_AROMATIC) != 0) delocalized_before = true;
            if (bond & BOND_SINGLE) histogram_before[0] += 1;
            if (bond & BOND_DOUBLE) histogram_before[1] += 1;
            if (bond & BOND_TRIPLE) histogram_before[2] += 1;
            if (bond & BOND_AROMATIC) histogram_before[3] += 1;
        }
        const uint source_words_charge =
            product_atoms[p_idx * BUCKET * ATOM_WORD_COUNT + source * ATOM_WORD_COUNT + 5];
        const uint source_words_h =
            product_atoms[p_idx * BUCKET * ATOM_WORD_COUNT + source * ATOM_WORD_COUNT + 7];
        const int charge_before = charge_from_mask(source_words_charge);
        const int total_h_before = exact_bucket_from_mask(source_words_h);

        if (charge_after != charge_before) {
            // Configurations are keyed by charge, so the matched atom's configuration says
            // nothing about the atom after the change, and inventing one would be a search
            // for a target state.
            output_status[pair_idx].rejection_reason =
                REJECT_HYDROGEN_OMITTED_WITH_CHARGE_CHANGE;
            return;
        }

        // Decided on the per-kind bond histogram rather than the degree, because a single
        // becoming a double changes the environment without changing the neighbour count.
        // Splitting a molecule into components is not a bond change, so context atoms come
        // through unaltered.
        bool bonding_changed = false;
        for (int kind = 0; kind < 4; ++kind) {
            if (histogram_before[kind] != histogram_after[kind]) bonding_changed = true;
        }
        if (!bonding_changed) {
            s.edited_h[atom] = total_h_before;
            continue;
        }

        ConfigRow configuration;
        const uint status = config_select(
            config_index, config_rows, z, charge_before, sigma_before, order_before,
            total_h_before, delocalized_before, configuration);
        if (status != CONFIG_OK) {
            output_status[pair_idx].rejection_reason = reject_for_config_failure(status);
            return;
        }
        // The configuration is copied from the matched atom, so there is no choice among
        // valence options here: the edited local graph plus one declared configuration
        // determine exactly one count.
        int completed = z_is_metal(z)
            ? 0
            : (delocalized_after ? configuration.sigma_capacity - sigma_after
                                 : configuration.graph_valence - order_after);
        if (completed < 0) {
            output_status[pair_idx].rejection_reason = REJECT_HYDROGEN_COMPLETION_NEGATIVE;
            return;
        }
        s.edited_h[atom] = completed;
    }

    if (profile_mode != 0u) {
        output_status[pair_idx].profile_stage = 1u;
        output_status[pair_idx].profile_edited_atoms = (uint)edited_count;
        if (profile_mode == 1u) return;
    }

    // Scan the edited adjacency once. Union roots by their smallest atom index so
    // component IDs, ordering, and overflow tie-breaking match min-label propagation.
    uint unions = 0u;
    for (int atom = 0; atom < edited_count; ++atom) {
        for (int other = atom + 1; other < edited_count; ++other) {
            if (s.edited_bonds[atom][other] == BOND_NONE) continue;
            int left = atom;
            int right = other;
            while (s.comp_labels[left] != left) left = s.comp_labels[left];
            while (s.comp_labels[right] != right) right = s.comp_labels[right];
            if (left == right) continue;
            s.comp_labels[max(left, right)] = min(left, right);
            unions += 1u;
        }
    }
    for (int atom = 0; atom < edited_count; ++atom) {
        int root = atom;
        while (s.comp_labels[root] != root) root = s.comp_labels[root];
        s.comp_labels[atom] = root;
    }
    if (profile_mode != 0u) output_status[pair_idx].profile_component_merges = unions;

    // What the RHS's parentheses claimed about its own output, against the output. Same
    // non-zero group, same component; different groups, different components. Mirrors
    // `grouping_agrees_with_components` on the host, and refuses by the same rule as the
    // orphan below: an application that disagrees is not a partial result to hand back.
    bool grouping_ok = true;
    for (int atom = 0; atom < edited_count && grouping_ok; ++atom) {
        const uchar group = s.edited_group[atom];
        if (group == 0u) continue;
        for (int other = atom + 1; other < edited_count; ++other) {
            const uchar other_group = s.edited_group[other];
            if (other_group == 0u) continue;
            const bool together = (s.comp_labels[atom] == s.comp_labels[other]);
            if ((group == other_group) != together) { grouping_ok = false; break; }
        }
    }
    if (!grouping_ok) output_status[pair_idx].grouping_violations = 1u;

    // Component sizes (heavy-atom counts) per root, and which to emit. When an
    // assignment splits into more than MAX_COMPONENTS components we keep only the
    // heaviest ones (largest heavy-atom count; ties broken by lowest atom index) and
    // flag the overflow -- the dropped pieces are typically salts / small fragments,
    // so the emitted output stays useful while the flag records the truncation.
    int kept_candidate_count = 0;
    for (int root = 0; root < edited_count; ++root) {
        s.root_size[root] = 0;
        s.root_described[root] = 0;
    }
    for (int atom = 0; atom < edited_count; ++atom) {
        s.root_size[s.comp_labels[atom]] += 1;
        if (s.described[atom]) s.root_described[s.comp_labels[atom]] = 1;
    }
    for (int root = 0; root < edited_count; ++root) {
        bool is_root = (s.comp_labels[root] == root);
        if (!is_root) {
            s.keep_comp[root] = 0;
            continue;
        }
        if (!s.root_described[root]) {
            // Not a precursor: the RHS enumerates what comes back and never named it. If the
            // edit *severed* it, the disconnection is incomplete and the host refuses the
            // whole application; if it was already its own molecule, it is an unreacted
            // reagent -- what a unimolecular forward rule run against a union of precursors
            // always leaves behind -- and dropping it is right.
            //
            // Severance is a question about the *product's* bonds, not the edited ones:
            // this component is severed if one of its atoms was bonded, before the edit, to
            // something no longer in it. Computed here rather than precomputed per root,
            // because an undescribed component is rare and a second `EDITED_BUCKET` array
            // in this struct costs every thread. Mirrors `severed()` in `reaction.py`.
            s.keep_comp[root] = 0;
            output_status[pair_idx].undescribed_components += 1u;
            bool was_attached = false;
            for (int atom = 0; atom < edited_count && !was_attached; ++atom) {
                if (s.comp_labels[atom] != root) continue;
                const int source = s.edited_from_product[atom];
                if (source < 0) continue;
                for (int other = 0; other < product_atom_count; ++other) {
                    if (other == source) continue;
                    if (bond_class(product_bonds[
                            p_idx * BUCKET * BUCKET + source * BUCKET + other]) == 0u) {
                        continue;
                    }
                    const int neighbour = s.product_to_edited[other];
                    if (neighbour < 0 || s.comp_labels[neighbour] != root) {
                        was_attached = true;
                        break;
                    }
                }
            }
            if (was_attached) {
                output_status[pair_idx].orphaned_components += 1u;
            }
            continue;
        }
        if (s.root_size[root] > BUCKET) {
            s.keep_comp[root] = 0;
            output_status[pair_idx].atom_overflow += 1u;
            continue;
        }
        s.keep_comp[root] = 1;
        kept_candidate_count += 1;
    }
    if (kept_candidate_count > MAX_COMPONENTS) {
        output_status[pair_idx].overflow = 1u;
        for (int root = 0; root < edited_count; ++root) s.keep_comp[root] = 0;
        for (int k = 0; k < MAX_COMPONENTS; ++k) {
            int best_root = -1;
            int best_size = -1;
            for (int root = 0; root < edited_count; ++root) {
                if (s.comp_labels[root] != root || s.keep_comp[root]) continue;
                if (s.root_size[root] > BUCKET) continue;
                if (!s.root_described[root]) continue;
                if (s.root_size[root] > best_size) {
                    best_size = s.root_size[root];
                    best_root = root;
                }
            }
            if (best_root >= 0) s.keep_comp[best_root] = 1;
        }
    }

    // Reserve a contiguous, compacted span in the shared output buffer for this
    // assignment's kept components (atomic bump-allocator, like the prefilter kernel),
    // so the host copies back only the components actually produced instead of the
    // full pair_count * MAX_COMPONENTS fixed-stride buffer.
    const uint kept_count =
        (uint)((kept_candidate_count > MAX_COMPONENTS) ? MAX_COMPONENTS : kept_candidate_count);
    if (profile_mode != 0u) {
        output_status[pair_idx].profile_stage = 2u;
        output_status[pair_idx].profile_kept_components = kept_count;
        if (profile_mode == 2u) return;
    }
    uint out_base = 0u;
    if (kept_count > 0u && profile_mode != 4u && profile_mode != 5u) {
        out_base = atomic_fetch_add_explicit(output_counter, kept_count, memory_order_relaxed);
    }
    output_status[pair_idx].output_base = out_base;

    uint component_index = 0;
    for (int root = 0; root < edited_count; ++root) {
        if (s.comp_labels[root] != root) continue;
        if (!s.keep_comp[root]) continue;  // dropped (overflow: a lighter fragment)

        int comp_count = 0;
        for (int atom = 0; atom < edited_count; ++atom) {
            if (s.comp_labels[atom] != root) continue;
            s.comp_atoms[comp_count] = atom;
            comp_count += 1;
        }
        if (comp_count > BUCKET) continue;
        if (profile_mode != 0u) {
            output_status[pair_idx].profile_perceived_atoms += (uint)comp_count;
            for (int i = 0; i < comp_count; ++i) {
                if (s.edited_atoms[s.comp_atoms[i]][4] == AROMATIC)
                    output_status[pair_idx].profile_aromatic_atoms += 1u;
            }
        }

        for (int i = 0; i < comp_count; ++i) {
            const int e = s.comp_atoms[i];
            for (int word = 0; word < ATOM_WORD_COUNT; ++word) {
                s.comp_words[i][word] = s.edited_atoms[e][word];
            }
            s.comp_eh[i] = s.explicit_h[e];
            s.comp_ih[i] = s.implicit_h[e];
            s.comp_ph[i] = s.edited_h[e];
            for (int j = 0; j < comp_count; ++j) {
                s.comp_bonds[i][j] = s.edited_bonds[e][s.comp_atoms[j]];
            }
        }

        uint cycles = 0u;
        bool profile_stopped = false;
        const uint refusal = normalize_aromaticity_and_hydrogens(
            s, comp_count, config_index, config_rows, cycles,
            profile_mode, profile_stopped);
        if (profile_mode != 0u) output_status[pair_idx].profile_cycles += cycles;
        if (profile_stopped) {
            continue;
        }
        if (refusal != REJECT_NONE) {
            // The application produced no precursor set, so the span reserved for it holds
            // nothing. Zeroing every slot (not just the ones still to come) retracts the
            // components already written for earlier components of the same application.
            output_status[pair_idx].rejection_reason = refusal;
            output_status[pair_idx].component_count = 0u;
            for (uint k = 0; k < kept_count; ++k) {
                out_atom_counts[out_base + k] = 0u;
            }
            return;
        }

        // Emit the perceived component. No identity is written here: the keyer is its own
        // dispatch over whatever batch it is handed, so a constructed precursor is keyed by the
        // same call a frontier is.
        //
        // `atom_map` is not written either, and that is a deletion rather than an omission -- the
        // `GpuMol` field it filled was read by no host code and no test, 256 bytes a record for
        // nobody.
        const uint out_idx = out_base + component_index;
        const ulong atom_base = (ulong)out_idx * BUCKET * ATOM_WORD_COUNT;
        const ulong bond_base = (ulong)out_idx * BUCKET * BUCKET;
        out_atom_counts[out_idx] = (uint)comp_count;
        for (int i = 0; i < comp_count; ++i) {
            for (int word = 0; word < ATOM_WORD_COUNT; ++word) {
                out_atoms[atom_base + i * ATOM_WORD_COUNT + word] = s.comp_words[i][word];
            }
            for (int j = 0; j < comp_count; ++j) {
                const uchar bond = s.comp_bonds[i][j];
                const uchar topology = ((s.ring_rows[i] >> j) & 1ul) != 0ul
                    ? BOND_TOPOLOGY_RING : BOND_TOPOLOGY_CHAIN;
                out_bonds[bond_base + i * BUCKET + j] =
                    bond == BOND_NONE ? BOND_NONE : (uchar)(bond | topology);
            }
        }

        component_index += 1;
        output_status[pair_idx].component_count = component_index;
    }
    if (profile_mode != 0u)
        output_status[pair_idx].profile_stage = profile_mode >= 4u ? profile_mode : 3u;
}
// The keyer, as a kernel over a *batch* rather than over the struct a sibling kernel happens to
// have produced. It replaced an in-place sibling that took `device GpuMol*` and wrote
// `hash_lo`/`hash_hi` onto the record, and so could key only what `retro_construct` had just
// built -- a frontier had no record to write into.
//
// This is CUDA's signature (`wl_hash_molecules` in `cuda_retro.py`): parallel arrays in, fresh hash
// arrays out. Nothing about it is precursor-specific, so it keys any batch the arm can hold --
// One deliberate divergence from CUDA, which takes `rows_global`: this derives the adjacency rows
// in-kernel. Metal batches keep `rows` lazy -- a precursor batch carries `rows=None` -- and the
// load loop below already walks all n^2 bond bytes, so taking rows would force a host packbits
// pass on every keyed batch to save nothing.
kernel void wl_hash_molecules(
    device const uint* atoms [[buffer(0)]],
    device const uchar* bonds [[buffer(1)]],
    device const uint* atom_counts [[buffer(2)]],
    device const uint* count [[buffer(3)]],
    device ulong* hash_lo [[buffer(4)]],
    device ulong* hash_hi [[buffer(5)]],
    constant uint& profile_mode [[buffer(6)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    if (gid >= count[0]) return;  // uniform across the group
    threadgroup WLScratch w;
    threadgroup int partial_edges[WL_THREADS];
    const int n = (int)atom_counts[gid];
    if (n <= 0) {
        if (tid == 0) { hash_lo[gid] = 0ul; hash_hi[gid] = 0ul; }
        return;  // uniform across the group
    }
    const ulong atom_base = (ulong)gid * BUCKET * ATOM_WORD_COUNT;
    const ulong bond_base = (ulong)gid * BUCKET * BUCKET;
    for (int local = (int)tid; local < n; local += WL_THREADS) {
        for (int word = 0; word < ATOM_WORD_COUNT; ++word) {
            w.comp_words[local][word] = atoms[atom_base + local * ATOM_WORD_COUNT + word];
        }
        ulong row = 0ul;
        for (int other = 0; other < n; ++other) {
            const uchar bond = bonds[bond_base + local * BUCKET + other];
            w.comp_bonds[local][other] = bond;
            if (bond != BOND_NONE) row |= 1ul << other;
        }
        w.adjacency_rows[local] = row;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong lo = 0ul;
    ulong hi = 0ul;
    component_token(w, n, tid, partial_edges, profile_mode, lo, hi);
    if (tid == 0) {
        hash_lo[gid] = lo;
        hash_hi[gid] = hi;
    }
}

)";

// One device, one queue, one shader library and every pipeline built from it, shared by every
// entry point in this translation unit.
//
// These were function-local statics inside `vesma_metal_retro_construct`, so a second entry point
// could not dispatch a kernel from the same library without making a second `MTLDevice`, a second
// `MTLCommandQueue` and a second compile of the 2250-line shader -- which is what
// `metal_pathfp.mm` does for its two entries today. Hoisting them is what lets the keyer be its
// own export rather than a passenger on the constructor's dispatch.
static id<MTLDevice> s_device = nil;
static id<MTLComputePipelineState> s_pipeline = nil;
static id<MTLComputePipelineState> s_wl_molecules_pipeline = nil;
static id<MTLCommandQueue> s_queue = nil;
static size_t s_scratch_bytes = 0;
static dispatch_once_t s_once;
static bool s_init_failed = false;

static bool metal_retro_context(void) {
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

            id<MTLFunction> function = [library newFunctionWithName:@"retro_construct"];
            if (!function) {
                snprintf(g_last_error, sizeof(g_last_error), "Function retro_construct not found");
                s_init_failed = true;
                return;
            }

            s_pipeline = [s_device newComputePipelineStateWithFunction:function error:&error];
            if (!s_pipeline) {
                snprintf(g_last_error, sizeof(g_last_error), "Pipeline creation failed: %s", [[error localizedDescription] UTF8String]);
                s_init_failed = true;
                return;
            }

            id<MTLFunction> wl_mol_fn = [library newFunctionWithName:@"wl_hash_molecules"];
            if (!wl_mol_fn) {
                snprintf(g_last_error, sizeof(g_last_error), "Function wl_hash_molecules not found");
                s_init_failed = true;
                return;
            }
            s_wl_molecules_pipeline = [s_device newComputePipelineStateWithFunction:wl_mol_fn error:&error];
            if (!s_wl_molecules_pipeline) {
                snprintf(g_last_error, sizeof(g_last_error), "wl_hash_molecules pipeline failed: %s", [[error localizedDescription] UTF8String]);
                s_init_failed = true;
                return;
            }

            s_queue = [s_device newCommandQueue];
            if (!s_queue) {
                snprintf(g_last_error, sizeof(g_last_error), "Command queue creation failed");
                s_init_failed = true;
                return;
            }

            // Query the shader's own sizeof(Scratch) so the host sizes the device
            // scratch pool with exactly the stride the kernel indexes by.
            id<MTLFunction> size_fn = [library newFunctionWithName:@"scratch_size_probe"];
            id<MTLComputePipelineState> size_pipe = [s_device newComputePipelineStateWithFunction:size_fn error:&error];
            if (!size_pipe) {
                snprintf(g_last_error, sizeof(g_last_error), "scratch_size_probe pipeline failed: %s", [[error localizedDescription] UTF8String]);
                s_init_failed = true;
                return;
            }
            uint32_t sz = 0;
            id<MTLBuffer> size_buf = [s_device newBufferWithBytes:&sz length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cb = [s_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            cb.label = @"vesma.scratch_size_probe";
            enc.label = @"scratch_size_probe";
            [enc setComputePipelineState:size_pipe];
            [enc setBuffer:size_buf offset:0 atIndex:0];
            [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            s_scratch_bytes = (size_t)(*(uint32_t*)[size_buf contents]);
            if (s_scratch_bytes == 0) {
                snprintf(g_last_error, sizeof(g_last_error), "scratch_size_probe returned 0");
                s_init_failed = true;
            }
        }
    });
    return !s_init_failed;
}

// A probe kernel's pipeline, or nil with the error string set.
//
// `newComputePipelineStateWithFunction:` raises an uncaught `NSInvalidArgumentException` when
// handed a nil function, which aborts the process instead of taking the `g_last_error` path every
// entry point here uses. A missing function is exactly what a rename or a shader edit produces, so
// the probes that look kernels up by name check first. `metal_pathfp.mm`'s `pipeline_for` block
// does the same thing for the two production kernels.
static id<MTLComputePipelineState> probe_pipeline(
    id<MTLDevice> device, id<MTLLibrary> library, const char* name
) {
    id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!function) {
        snprintf(g_last_error, sizeof(g_last_error), "Function %s not found", name);
        return nil;
    }
    NSError* error = nil;
    id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function error:&error];
    if (!state) {
        snprintf(g_last_error, sizeof(g_last_error), "Pipeline creation failed for %s: %s", name, [[error localizedDescription] UTF8String]);
    }
    return state;
}

extern "C" {

int vesma_metal_retro_construct(
    const uint32_t* product_atoms,
    const uint8_t* product_bonds,
    const uint32_t* product_atom_counts,
    const ReactionData* reactions,
    const int32_t* assignments,
    const uint32_t* pair_product_indices,
    const uint32_t* pair_reaction_indices,
    uint32_t* out_atoms,
    uint8_t* out_bonds,
    uint32_t* out_atom_counts,
    uint32_t* out_produced,
    RetroStatus* output_status,
    const uint32_t* config_index,
    uint32_t config_index_count,
    const uint32_t* config_rows,
    uint32_t config_row_count,
    uint64_t reaction_token,
    uint64_t products_token,
    double* out_host_stages,
    int unique_product_count,
    int unique_reaction_count,
    int pair_count,
    int threads_per_group,
    double* out_construct_seconds,
    int profile_mode
) {
        // Per-kernel GPU seconds (construct vs wl_hash), measured from the command
        // buffers' own GPUStartTime/GPUEndTime. Zeroed up front so early returns and the
        // produced==0 case report 0 rather than leaving the caller's buffers uninitialized.
        if (out_construct_seconds) *out_construct_seconds = 0.0;
        // Host-side stages of one dispatch: buffer setup, the two command-buffer round
        // trips, and the copy back. Timed separately because "host overhead" as a single
        // number says only that the GPU was not the problem, not which part of the host is.
        if (out_host_stages) {
            for (int i = 0; i < 4; ++i) out_host_stages[i] = 0.0;
        }

        if (!metal_retro_context()) return 0;

        if (pair_count == 0) return 1; // Nothing to do

        // How wide to hold the device lock, stated here once because this is the widest case in
        // the package and the other two translation units point at it.
        //
        // ctypes releases the GIL for the duration of these calls, so two Python threads can be
        // inside one `extern "C"` entry at the same time. What that costs depends on what the
        // process-global buffers are:
        //
        //   * Cached *input* buffers, immutable once uploaded -- the matcher's patterns, the
        //     reaction table here. A concurrent replacement makes a *new* buffer and rebinds the
        //     static, so a dispatch that already read the static still holds a live, correct
        //     buffer. The lock need only cover the token check plus the binds, which is the
        //     narrow lock `metal_matcher.mm` takes.
        //
        //   * Buffers the kernel *writes*, or that the host mutates in place -- the output atoms,
        //     bonds, counts, the status rows, the atomic counter and the scratch pool. Two
        //     dispatches sharing those corrupt each other's results no matter what the tokens
        //     say: the `memset` of the status rows and the store of zero into the counter below
        //     both clobber a dispatch already in flight, and both kernels then pack components
        //     into the same output prefix. Nothing narrower than the whole dispatch is safe, so
        //     this holds the lock across commit and `waitUntilCompleted`.
        //
        // No caller fans out Metal dispatches today -- `AOSTAR_HOST_JOBS` parallelises template
        // parsing and the rdkit pool, not this -- so this is insurance against the next caller
        // rather than a fix for a live bug. It costs an uncontended `@synchronized` per dispatch.
        @synchronized (s_device) {
        @autoreleasepool {
            id<MTLDevice> device = s_device;
            id<MTLComputePipelineState> pipeline = s_pipeline;
            id<MTLCommandQueue> queue = s_queue;


            const auto stage_clock = []() {
                return std::chrono::duration<double>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
            };
            const double t_setup_start = stage_clock();

            size_t p_atoms_size = unique_product_count * 14 * 64 * sizeof(uint32_t);
            size_t p_bonds_size = unique_product_count * 64 * 64 * sizeof(uint8_t);
            size_t p_counts_size = unique_product_count * sizeof(uint32_t);
            size_t rxn_size = unique_reaction_count * sizeof(ReactionData);
            size_t assign_size = pair_count * 64 * sizeof(int32_t);
            size_t pair_idx_size = pair_count * sizeof(uint32_t);
            const size_t out_rows = (size_t)pair_count * MAX_COMPONENTS;
            size_t out_atoms_size = out_rows * 14 * 64 * sizeof(uint32_t);
            size_t out_bonds_size = out_rows * 64 * 64 * sizeof(uint8_t);
            size_t out_counts_size = out_rows * sizeof(uint32_t);
            size_t status_size = pair_count * sizeof(RetroStatus);

            // The products are identical across every chunk of one match() call and the
            // reactions across the whole matcher, so both are uploaded per token rather
            // than per dispatch.
            if (s_p_atoms_buffer == nil || products_token == 0
                || products_token != s_products_token || p_atoms_size != s_p_atoms_size) {
                s_p_atoms_buffer = [device newBufferWithBytes:product_atoms length:p_atoms_size options:MTLResourceStorageModeShared];
                s_p_bonds_buffer = [device newBufferWithBytes:product_bonds length:p_bonds_size options:MTLResourceStorageModeShared];
                s_p_counts_buffer = [device newBufferWithBytes:product_atom_counts length:p_counts_size options:MTLResourceStorageModeShared];
                s_products_token = products_token;
                s_p_atoms_size = p_atoms_size;
            }
            if (s_rxn_buffer == nil || reaction_token == 0
                || reaction_token != s_rxn_token || rxn_size != s_rxn_size) {
                s_rxn_buffer = [device newBufferWithBytes:reactions length:rxn_size options:MTLResourceStorageModeShared];
                s_rxn_token = reaction_token;
                s_rxn_size = rxn_size;
            }
            id<MTLBuffer> buf_p_atoms = s_p_atoms_buffer;
            id<MTLBuffer> buf_p_bonds = s_p_bonds_buffer;
            id<MTLBuffer> buf_p_counts = s_p_counts_buffer;
            id<MTLBuffer> buf_rxn = s_rxn_buffer;
            id<MTLBuffer> buf_assign = [device newBufferWithBytes:assignments length:assign_size options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_pair_p = [device newBufferWithBytes:pair_product_indices length:pair_idx_size options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_pair_r = [device newBufferWithBytes:pair_reaction_indices length:pair_idx_size options:MTLResourceStorageModeShared];

            // Output buffer holds the worst case, but the kernel atomically packs the
            // components it actually emits into the front of it (recording each
            // assignment's start in RetroStatus.output_base), so no memset is needed
            // and only the produced prefix is copied back below.
            if (s_out_atoms_buffer == nil || s_out_atoms_capacity < out_atoms_size) {
                s_out_atoms_buffer = [device newBufferWithLength:out_atoms_size options:MTLResourceStorageModeShared];
                s_out_atoms_capacity = out_atoms_size;
            }
            if (s_out_bonds_buffer == nil || s_out_bonds_capacity < out_bonds_size) {
                s_out_bonds_buffer = [device newBufferWithLength:out_bonds_size options:MTLResourceStorageModeShared];
                s_out_bonds_capacity = out_bonds_size;
            }
            if (s_out_counts_buffer == nil || s_out_counts_capacity < out_counts_size) {
                s_out_counts_buffer = [device newBufferWithLength:out_counts_size options:MTLResourceStorageModeShared];
                s_out_counts_capacity = out_counts_size;
            }
            if (s_out_status_buffer == nil || s_out_status_capacity < status_size) {
                s_out_status_buffer = [device newBufferWithLength:status_size options:MTLResourceStorageModeShared];
                s_out_status_capacity = status_size;
            }
            if (s_counter_buffer == nil) {
                s_counter_buffer = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
            }
            id<MTLBuffer> buf_out_atoms = s_out_atoms_buffer;
            id<MTLBuffer> buf_out_bonds = s_out_bonds_buffer;
            id<MTLBuffer> buf_out_counts = s_out_counts_buffer;
            id<MTLBuffer> buf_out_status = s_out_status_buffer;
            id<MTLBuffer> buf_out_counter = s_counter_buffer;
            // Only this dispatch's prefix is read back, so only it needs clearing.
            memset([buf_out_status contents], 0, status_size);
            *(uint32_t*)[buf_out_counter contents] = 0u;

            // Per-assignment device scratch pool (one slot per assignment), sized by the
            // shader's own sizeof(Scratch). The kernel initializes every slot it reads, so
            // no zeroing is needed. The caller bounds pair_count per dispatch so this stays
            // memory-bounded.
            size_t scratch_size = (size_t)pair_count * s_scratch_bytes;
            if (s_scratch_buffer == nil || s_scratch_capacity < scratch_size) {
                s_scratch_buffer = [device newBufferWithLength:scratch_size options:MTLResourceStorageModePrivate];
                s_scratch_capacity = scratch_size;
            }
            id<MTLBuffer> buf_scratch = s_scratch_buffer;

            // The closed-shell configuration table: ~6 KB shared by every assignment in the
            // dispatch, so it rides in its own buffers rather than in the per-assignment
            // scratch, and in the process-wide cache rather than being re-uploaded per
            // dispatch. See the declaration for why it needs no token.
            if (s_config_index_buffer == nil) {
                s_config_index_buffer = [device newBufferWithBytes:config_index length:config_index_count * sizeof(uint32_t) options:MTLResourceStorageModeShared];
                s_config_rows_buffer = [device newBufferWithBytes:config_rows length:config_row_count * sizeof(uint32_t) options:MTLResourceStorageModeShared];
                s_config_index_count = config_index_count;
                s_config_row_count = config_row_count;
            } else if (config_index_count != s_config_index_count || config_row_count != s_config_row_count) {
                snprintf(g_last_error, sizeof(g_last_error),
                         "configuration table changed shape within one process: had %u/%u, got %u/%u",
                         s_config_index_count, s_config_row_count, config_index_count, config_row_count);
                return 0;
            }
            id<MTLBuffer> buf_config_index = s_config_index_buffer;
            id<MTLBuffer> buf_config_rows = s_config_rows_buffer;

            const double t_setup_end = stage_clock();
            if (out_host_stages) out_host_stages[0] = t_setup_end - t_setup_start;

            id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            commandBuffer.label = @"vesma.retro_construct";
            encoder.label = @"retro_construct";

            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:buf_p_atoms offset:0 atIndex:0];
            [encoder setBuffer:buf_p_bonds offset:0 atIndex:1];
            [encoder setBuffer:buf_p_counts offset:0 atIndex:2];
            [encoder setBuffer:buf_rxn offset:0 atIndex:3];
            [encoder setBuffer:buf_assign offset:0 atIndex:4];
            [encoder setBuffer:buf_pair_p offset:0 atIndex:5];
            [encoder setBuffer:buf_pair_r offset:0 atIndex:6];
            [encoder setBuffer:buf_out_atoms offset:0 atIndex:7];
            [encoder setBuffer:buf_out_bonds offset:0 atIndex:8];
            [encoder setBuffer:buf_out_counts offset:0 atIndex:9];
            [encoder setBuffer:buf_out_status offset:0 atIndex:10];
            [encoder setBuffer:buf_out_counter offset:0 atIndex:11];
            [encoder setBuffer:buf_scratch offset:0 atIndex:12];
            [encoder setBuffer:buf_config_index offset:0 atIndex:13];
            [encoder setBuffer:buf_config_rows offset:0 atIndex:14];
            const uint32_t profile_mode_u32 = (uint32_t)(profile_mode >= 6 ? 3 : profile_mode);
            [encoder setBytes:&profile_mode_u32 length:sizeof(profile_mode_u32) atIndex:15];

            MTLSize gridSize = MTLSizeMake(pair_count, 1, 1);
            NSUInteger tg = threads_per_group > 0 ? (NSUInteger)threads_per_group : 1;
            MTLSize threadgroupSize = MTLSizeMake(tg, 1, 1);

            [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];

            if (commandBuffer.error) {
                snprintf(g_last_error, sizeof(g_last_error), "GPU execution failed: %s", [[commandBuffer.error localizedDescription] UTF8String]);
                return 0;
            }
            if (out_construct_seconds) *out_construct_seconds = commandBuffer.GPUEndTime - commandBuffer.GPUStartTime;
            const double t_construct_done = stage_clock();
            if (out_host_stages) out_host_stages[1] = t_construct_done - t_setup_end;

            uint32_t produced = *(const uint32_t*)[buf_out_counter contents];

            // Stage 2 was the fused WL round trip, which is now `vesma_metal_wl_hash`. Kept as a
            // zero so the four host stages stay positionally stable for callers that read them.
            const double t_wl_done = t_construct_done;
            if (out_host_stages) out_host_stages[2] = 0.0;

            // Copy back only the compacted prefix the kernel wrote, and say how long it is rather
            // than leaving the host to infer it from the status rows.
            if ((size_t)produced > out_rows) produced = (uint32_t)out_rows;  // paranoia
            if (out_produced) *out_produced = produced;
            memcpy(out_atoms, [buf_out_atoms contents], (size_t)produced * 14 * 64 * sizeof(uint32_t));
            memcpy(out_bonds, [buf_out_bonds contents], (size_t)produced * 64 * 64 * sizeof(uint8_t));
            memcpy(out_atom_counts, [buf_out_counts contents], (size_t)produced * sizeof(uint32_t));
            memcpy(output_status, [buf_out_status contents], status_size);
            if (out_host_stages) out_host_stages[3] = stage_clock() - t_wl_done;

            return 1;
        }
        }
    }

// The keyer as its own entry point, over any packed batch.
//
// `vesma_metal_retro_construct` used to dispatch a WL pass of its own over the `GpuMol` records it
// had just built, which is why a frontier could not be keyed: that kernel wrote in place on a
// struct a frontier never has. It keys nothing now. This dispatches `wl_hash_molecules` over the
// batch layout every molecule here uses, so the arm's `key` block is total -- frontier, gather,
// union and concat alike.
//
// No device lock, unlike the construct entry above: every buffer this touches is allocated per
// call, so two concurrent keyings share nothing. That is the first case in the lock-width note --
// there is no process-global buffer here to cache, let alone to write.
//
// `profile_mode` here is the WL stage 1..4 directly, not the 6..9 the constructor's entry remaps.
// That remapping exists because one `profile_mode` argument there had to address two kernels; with
// the keyer separate there is only one kernel to address, so the caller does the translation.
int vesma_metal_wl_hash(
    const uint32_t* atoms,
    const uint8_t* bonds,
    const uint32_t* atom_counts,
    int n,
    uint64_t* out_hash_lo,
    uint64_t* out_hash_hi,
    double* out_seconds,
    int profile_mode
) {
    if (out_seconds) *out_seconds = 0.0;
    if (n <= 0) return 1;  // nothing to key is not a failure
    if (!metal_retro_context()) return 0;

    @autoreleasepool {
        const size_t atoms_size = (size_t)n * 14 * 64 * sizeof(uint32_t);
        const size_t bonds_size = (size_t)n * 64 * 64 * sizeof(uint8_t);
        const size_t counts_size = (size_t)n * sizeof(uint32_t);
        const size_t hash_size = (size_t)n * sizeof(uint64_t);

        // Shared storage throughout, as everywhere else in this file: on Apple Silicon this is the
        // memory the host already holds, so these are views rather than transfers.
        id<MTLBuffer> buf_atoms = [s_device newBufferWithBytes:atoms length:atoms_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_bonds = [s_device newBufferWithBytes:bonds length:bonds_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_counts = [s_device newBufferWithBytes:atom_counts length:counts_size options:MTLResourceStorageModeShared];
        uint32_t count = (uint32_t)n;
        id<MTLBuffer> buf_count = [s_device newBufferWithBytes:&count length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_lo = [s_device newBufferWithLength:hash_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_hi = [s_device newBufferWithLength:hash_size options:MTLResourceStorageModeShared];
        if (!buf_atoms || !buf_bonds || !buf_counts || !buf_count || !buf_lo || !buf_hi) {
            snprintf(g_last_error, sizeof(g_last_error), "wl_hash buffer allocation failed");
            return 0;
        }

        uint32_t mode = (uint32_t)profile_mode;
        id<MTLCommandBuffer> cb = [s_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        cb.label = @"vesma.wl_hash_molecules";
        enc.label = @"wl_hash_molecules";
        [enc setComputePipelineState:s_wl_molecules_pipeline];
        [enc setBuffer:buf_atoms offset:0 atIndex:0];
        [enc setBuffer:buf_bonds offset:0 atIndex:1];
        [enc setBuffer:buf_counts offset:0 atIndex:2];
        [enc setBuffer:buf_count offset:0 atIndex:3];
        [enc setBuffer:buf_lo offset:0 atIndex:4];
        [enc setBuffer:buf_hi offset:0 atIndex:5];
        [enc setBytes:&mode length:sizeof(uint32_t) atIndex:6];
        // One threadgroup per molecule, 32 lanes -- one SIMD group on Apple GPUs, which is the
        // width `component_token` stripes over.
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger)WL_THREADS_PER_GROUP, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (cb.status != MTLCommandBufferStatusCompleted) {
            snprintf(g_last_error, sizeof(g_last_error), "wl_hash command buffer failed");
            return 0;
        }
        if (out_seconds) *out_seconds = cb.GPUEndTime - cb.GPUStartTime;
        memcpy(out_hash_lo, [buf_lo contents], hash_size);
        memcpy(out_hash_hi, [buf_hi contents], hash_size);
    }
    return 1;
}

    const char* vesma_metal_retro_last_error() {
        return g_last_error;
    }

    // Runs the device configuration selectors over a batch of written atom states so a
    // test can compare the shader's answers directly against configurations.py. Compiles
    // its own library per call, like vesma_metal_retro_probe -- it runs once in a test,
    // never on a hot path.
    int vesma_metal_config_probe(
        const uint32_t* config_index,
        uint32_t config_index_count,
        const uint32_t* config_rows,
        uint32_t config_row_count,
        const int32_t* queries,
        uint32_t query_count,
        uint32_t* out
    ) {
        @autoreleasepool {
            if (query_count == 0) return 1;
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) {
                snprintf(g_last_error, sizeof(g_last_error), "No Metal device found");
                return 0;
            }
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kShaderSource] options:nil error:&error];
            if (!library) {
                snprintf(g_last_error, sizeof(g_last_error), "Shader compilation failed: %s", [[error localizedDescription] UTF8String]);
                return 0;
            }
            id<MTLFunction> function = [library newFunctionWithName:@"config_probe"];
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            if (!pipeline) {
                snprintf(g_last_error, sizeof(g_last_error), "config_probe pipeline failed: %s", [[error localizedDescription] UTF8String]);
                return 0;
            }
            id<MTLCommandQueue> queue = [device newCommandQueue];

            const size_t out_size = (size_t)query_count * 16 * sizeof(uint32_t);
            id<MTLBuffer> buf_index = [device newBufferWithBytes:config_index length:config_index_count * sizeof(uint32_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_rows = [device newBufferWithBytes:config_rows length:config_row_count * sizeof(uint32_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_queries = [device newBufferWithBytes:queries length:(size_t)query_count * 6 * sizeof(int32_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_out = [device newBufferWithLength:out_size options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            cb.label = @"vesma.config_probe";
            enc.label = @"config_probe";
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:buf_index offset:0 atIndex:0];
            [enc setBuffer:buf_rows offset:0 atIndex:1];
            [enc setBuffer:buf_queries offset:0 atIndex:2];
            [enc setBuffer:buf_out offset:0 atIndex:3];
            [enc dispatchThreads:MTLSizeMake(query_count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) {
                snprintf(g_last_error, sizeof(g_last_error), "config_probe execution failed: %s", [[cb.error localizedDescription] UTF8String]);
                return 0;
            }
            memcpy(out, [buf_out contents], out_size);
            return 1;
        }
    }

    // Diagnostic probe for the retro_construct pipeline's occupancy limits. Fills
    // out[0..3] = {device max threadgroup memory, this kernel's static threadgroup
    // memory (the Scratch footprint), max total threads per threadgroup, thread
    // execution width}. Returns 1 on success.
    // Occupancy and pressure limits for both retro pipelines, plus the device-side
    // sizeof(Scratch) the pool is strided by. maxTotalThreadsPerThreadgroup is the
    // observable proxy for register pressure: the compiler lowers it below the device
    // maximum exactly when a kernel's register footprint will not fit more lanes.
    //
    // out[0]  device max threadgroup memory        out[5] wl max threads/threadgroup
    // out[1]  retro static threadgroup memory      out[6] wl thread execution width
    // out[2]  retro max threads/threadgroup        out[7] device sizeof(Scratch)
    // out[3]  retro thread execution width         out[8] recommended working set
    // out[4]  wl static threadgroup memory         out[9] max buffer length
    int vesma_metal_retro_probe(uint64_t* out) {
        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) {
                snprintf(g_last_error, sizeof(g_last_error), "No Metal device found");
                return 0;
            }
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kShaderSource] options:nil error:&error];
            if (!library) {
                snprintf(g_last_error, sizeof(g_last_error), "Shader compilation failed: %s", [[error localizedDescription] UTF8String]);
                return 0;
            }
            id<MTLComputePipelineState> retro = probe_pipeline(device, library, "retro_construct");
            if (!retro) return 0;
            id<MTLComputePipelineState> wl = probe_pipeline(device, library, "wl_hash_molecules");
            if (!wl) return 0;
            id<MTLComputePipelineState> sizer = probe_pipeline(device, library, "scratch_size_probe");
            if (!sizer) return 0;
            out[0] = (uint64_t)[device maxThreadgroupMemoryLength];
            out[1] = (uint64_t)[retro staticThreadgroupMemoryLength];
            out[2] = (uint64_t)[retro maxTotalThreadsPerThreadgroup];
            out[3] = (uint64_t)[retro threadExecutionWidth];
            out[4] = (uint64_t)[wl staticThreadgroupMemoryLength];
            out[5] = (uint64_t)[wl maxTotalThreadsPerThreadgroup];
            out[6] = (uint64_t)[wl threadExecutionWidth];

            uint32_t sz = 0;
            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLBuffer> size_buf = [device newBufferWithBytes:&sz length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            cb.label = @"vesma.retro_scratch_probe";
            enc.label = @"retro_scratch_probe";
            [enc setComputePipelineState:sizer];
            [enc setBuffer:size_buf offset:0 atIndex:0];
            [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            out[7] = (uint64_t)(*(uint32_t*)[size_buf contents]);
            out[8] = (uint64_t)[device recommendedMaxWorkingSetSize];
            out[9] = (uint64_t)[device maxBufferLength];

            // The shader's own constants, rather than a comment claiming they match the host.
            id<MTLComputePipelineState> constants = probe_pipeline(device, library, "constants_probe");
            if (!constants) return 0;
            const NSUInteger constant_count = 7;
            id<MTLBuffer> const_buf = [device newBufferWithLength:constant_count * sizeof(uint32_t) options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> const_cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> const_enc = [const_cb computeCommandEncoder];
            const_cb.label = @"vesma.retro_constants_probe";
            const_enc.label = @"retro_constants_probe";
            [const_enc setComputePipelineState:constants];
            [const_enc setBuffer:const_buf offset:0 atIndex:0];
            [const_enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [const_enc endEncoding];
            [const_cb commit];
            [const_cb waitUntilCompleted];
            const uint32_t* reported = (const uint32_t*)[const_buf contents];
            for (NSUInteger i = 0; i < constant_count; ++i) out[10 + i] = (uint64_t)reported[i];

            // This file's own host-side declarations of four of those, so the test compares the
            // shader against the host as well as both against `vesma.limits`.
            out[17] = (uint64_t)MAX_COMPONENTS;
            out[18] = (uint64_t)MAX_REACTION_EDGES;
            out[19] = (uint64_t)EDITED_BUCKET;
            out[20] = (uint64_t)WL_THREADS_PER_GROUP;
            return 1;
        }
    }
}

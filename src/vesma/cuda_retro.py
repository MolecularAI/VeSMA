from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .configurations import configuration_table
from .cuda_prune import (
    _ATOM_WORD_COUNT,
    _CUDA_BUCKET_SIZE,
    CudaAssignmentResult,
    CudaOutputCapacityError,
    CudaPruneBatch,
    _pack_adjacency_rows_u64,
    _pack_atom_masks_u32,
    _pack_bond_matrix_u8,
    _require_cupy,
    _to_numpy,
)
from .gmol import GMol
from .gpu_atom_masks import pack_atom_masks
from .limits import CUDA_EDITED_ATOM_CAP, GPU_MATCH_ATOM_CAP, GPU_PRECURSOR_ATOM_CAP
from .reaction import ReactionPattern
from .rejection import RejectionCounts, no_rejections

__all__ = [
    "CudaMoleculeBatch",
    "CudaMoleculeStore",
    "CudaReactionBatch",
    "CudaRetroResult",
    "CudaStock",
    "REJECTION_NONE",
    "construct_retro_cuda",
    "get_wl_hash_kernel",
    "perceive_cuda_retro",
    "probe_cuda_configurations",
    "probe_cuda_perception",
    "prepare_cuda_reaction_batch",
    "wl_hash_kernel",
]

_CUDA_RETRO_MODULE: Any | None = None
_CUDA_RETRO_KERNEL: Any | None = None
_CUDA_PERCEPTION_KERNEL: Any | None = None
_CUDA_CONFIGURATION_PROBE_KERNEL: Any | None = None
_CUDA_PERCEPTION_PROBE_KERNEL: Any | None = None
_MAX_REACTION_EDGES = 512
REJECTION_NONE = np.uint32(0xFFFFFFFF)
_CUDA_CONFIGURATION_TABLE: tuple[Any, Any] | None = None


def cuda_configuration_table() -> tuple[Any, Any]:
    global _CUDA_CONFIGURATION_TABLE
    if _CUDA_CONFIGURATION_TABLE is None:
        cp = _require_cupy()
        index, rows = configuration_table()
        _CUDA_CONFIGURATION_TABLE = (
            cp.asarray(index, dtype=cp.uint32),
            cp.asarray(rows, dtype=cp.uint32),
        )
    return _CUDA_CONFIGURATION_TABLE


# Smallest aliphatic valence option >= occupied for atomic number z, or -1 when none
# Existence test for "edge (left_local, right_local) lies on a simple cycle of
# size 3-8", expressed as a bounded, allocation-free per-edge search for an
# alternate simple path of length 2-7 (depth 1-6 -> cycle size depth+2). This is
# the device-natural equivalent of CPU short_cycle_edge_mask (vesma/rings.py),
# which enumerates every 3-8 cycle once and marks its edges; both yield the same
# FLEX-edge set. Standalone so ring-parity tests compile the exact function.
_DEVICE_SHORT_CYCLE_SOURCE = r"""
__device__ bool bond_in_short_cycle(
    unsigned char bonds[CUDA_BUCKET_SIZE][CUDA_BUCKET_SIZE],
    const int* comp_atoms,
    int comp_count,
    int left_local,
    int right_local
) {
    int path[8];
    int next_neighbor[8];
    int depth = 0;
    unsigned long long visited = 1ull << left_local;
    path[0] = left_local;
    next_neighbor[0] = 0;

    while (depth >= 0) {
        if (next_neighbor[depth] >= comp_count) {
            visited &= ~(1ull << path[depth]);
            depth -= 1;
            continue;
        }

        const int neighbor_local = next_neighbor[depth];
        next_neighbor[depth] += 1;
        const int current = comp_atoms[path[depth]];
        const int neighbor = comp_atoms[neighbor_local];
        if (bonds[current][neighbor] == BOND_NONE) continue;
        if (depth == 0 && neighbor_local == right_local) continue;

        if (neighbor_local == right_local) {
            if (depth >= 1 && depth <= 6) return true;
            continue;
        }
        if (depth == 6 || (visited & (1ull << neighbor_local)) != 0ull) {
            continue;
        }

        depth += 1;
        path[depth] = neighbor_local;
        next_neighbor[depth] = 0;
        visited |= 1ull << neighbor_local;
    }
    return false;
}

__device__ bool bond_in_short_cycle_local(
    unsigned char bonds[CUDA_BUCKET_SIZE][CUDA_BUCKET_SIZE],
    int comp_count,
    int left_local,
    int right_local
) {
    int path[8];
    int next_neighbor[8];
    int depth = 0;
    unsigned long long visited = 1ull << left_local;
    path[0] = left_local;
    next_neighbor[0] = 0;

    while (depth >= 0) {
        if (next_neighbor[depth] >= comp_count) {
            visited &= ~(1ull << path[depth]);
            depth -= 1;
            continue;
        }
        const int neighbor_local = next_neighbor[depth];
        next_neighbor[depth] += 1;
        if (bonds[path[depth]][neighbor_local] == BOND_NONE) continue;
        if (depth == 0 && neighbor_local == right_local) continue;
        if (neighbor_local == right_local) {
            if (depth >= 1 && depth <= 6) return true;
            continue;
        }
        if (depth == 6 || (visited & (1ull << neighbor_local)) != 0ull) {
            continue;
        }
        depth += 1;
        path[depth] = neighbor_local;
        next_neighbor[depth] = 0;
        visited |= 1ull << neighbor_local;
    }
    return false;
}
"""

_DEVICE_FROZEN_CHEMISTRY_SOURCE = r"""
#define AROM_MAX_CYCLES 48
#define GF2_MAX_EDGES 1024
#define GF2_WORDS 16
#define GF2_MAX_VECTORS (AROM_MAX_CYCLES + 8)
#define WILDCARD 0xFFFFFFFFu
#define ISOTOPE_WORD 13
#define ISOTOPE_SHIFT 16u
#define ISOTOPE_ANY 0xFFFFu
#define ISOTOPE_FIELD 0xFFFF0000u
#define HORTON_EDGE_CAP 128
#define HORTON_CAND_CAP 128
#define HORTON_BASIS_CAP 64
#define HORTON_FAR 255
// Steps the shortest-path enumerators may take per component before giving R and r up.
#define HORTON_PATH_BUDGET (1 << 20)


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

struct ChemistryScratch {
    unsigned int comp_words[CUDA_BUCKET_SIZE][ATOM_WORD_COUNT];
    unsigned char comp_bonds[CUDA_BUCKET_SIZE][CUDA_BUCKET_SIZE];
    int comp_eh[CUDA_BUCKET_SIZE];
    int comp_ph[CUDA_BUCKET_SIZE];
    int z[CUDA_BUCKET_SIZE];
    int charge[CUDA_BUCKET_SIZE];
    int heavy_degree[CUDA_BUCKET_SIZE];
    int total_h[CUDA_BUCKET_SIZE];
    unsigned long long degree[CUDA_BUCKET_SIZE];
    unsigned long long adjacency[CUDA_BUCKET_SIZE];
    unsigned long long edge_rows[CUDA_BUCKET_SIZE];
    unsigned long long atom_mask;
    unsigned char cycle_path[AROM_MAX_CYCLES][8];
    int cycle_len[AROM_MAX_CYCLES];
    unsigned long long cycle_mask[AROM_MAX_CYCLES];
    int clabel[AROM_MAX_CYCLES];
    unsigned long long span_rows[CUDA_BUCKET_SIZE];
    unsigned char face_incidence[CUDA_BUCKET_SIZE][CUDA_BUCKET_SIZE];
    unsigned long long gf2_vec[GF2_MAX_VECTORS][GF2_WORDS];
    unsigned long long gf2_pivot[GF2_MAX_VECTORS][GF2_WORDS];
    int gf2_lead[GF2_MAX_VECTORS];
    unsigned char gf2_u[GF2_MAX_EDGES];
    unsigned char gf2_v[GF2_MAX_EDGES];

    // ring_bond_rows working state and output: the non-bridge edge set of the component.
    unsigned long long ring_rows[CUDA_BUCKET_SIZE];
    unsigned long long ring_path[CUDA_BUCKET_SIZE];
    unsigned long long ring_children[CUDA_BUCKET_SIZE];
    unsigned long long ring_parent_bit[CUDA_BUCKET_SIZE];

    // ring_membership working state and output: R and r per atom from the relevant cycles.
    unsigned char ring_count[CUDA_BUCKET_SIZE];
    unsigned char ring_smallest[CUDA_BUCKET_SIZE];
    unsigned char horton_prefix[CUDA_BUCKET_SIZE];
    unsigned char horton_dist[CUDA_BUCKET_SIZE];  // from the current root; HORTON_FAR if unreached
    unsigned char horton_eu[HORTON_EDGE_CAP];
    unsigned char horton_ev[HORTON_EDGE_CAP];
    signed char horton_lead_slot[HORTON_EDGE_CAP];
    // Two shortest-path enumerators (one per end of an edge), each a DFS stack down the
    // shortest-path DAG from its target to the root: the node, the predecessors still to try,
    // and the atom and edge bits of the path so far.
    unsigned char path_node[2][CUDA_BUCKET_SIZE];
    unsigned long long path_choice[2][CUDA_BUCKET_SIZE];
    unsigned long long path_atoms[2][CUDA_BUCKET_SIZE];
    unsigned long long path_lo[2][CUDA_BUCKET_SIZE];
    unsigned long long path_hi[2][CUDA_BUCKET_SIZE];
    int   path_level[2];
    int   path_fresh[2];
    int   path_budget;
    unsigned long long horton_cand_lo[HORTON_CAND_CAP];
    unsigned long long horton_cand_hi[HORTON_CAND_CAP];
    unsigned long long horton_basis_lo[HORTON_BASIS_CAP];
    unsigned long long horton_basis_hi[HORTON_BASIS_CAP];

};

// --- closed-shell configurations (mirror vesma/configurations.py) ---
//
// An atom's chemistry is fixed by which closed-shell configuration it occupies, and the
// set of configurations is data rather than code: configurations_for(z, charge) is a pure
// function of two small integers, so the host generates the whole product from that one
// function and hands it here as two buffers. Generating rather than transcribing is what
// keeps the from drifting -- in particular the isoelectronic asymmetry (standard
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
// which numbers its non-numeric states the other way round; the carries exactly one
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

__device__ __forceinline__ int exact_z_from_words(
    unsigned int w0, unsigned int w1, unsigned int w2, unsigned int w3
) {
    if (w0 != 0u) return ctz_u32_or_zero(w0);
    if (w1 != 0u) return ctz_u32_or_zero(w1) + 32;
    if (w2 != 0u) return ctz_u32_or_zero(w2) + 64;
    if (w3 != 0u) return ctz_u32_or_zero(w3) + 96;
    return 0;
}

__device__ __forceinline__ ConfigRow config_row_empty() {
    ConfigRow row;
    row.sigma_capacity = 0;
    row.graph_valence = 0;
    row.pi = PI_INVALID;
    return row;
}

__device__ __forceinline__ ConfigRow config_row_unpack(unsigned int word) {
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
__device__ __forceinline__ unsigned int config_span(const unsigned int* config_index, int z, int charge) {
    // Both coordinates are bounds-tested as unsigned, so one comparison catches the
    // negative and the too-large case, and the subscript is computed from exactly the
    // values that were tested.
    const unsigned int z_slot = (unsigned int)z;
    const unsigned int charge_slot = (unsigned int)(charge - CONFIG_CHARGE_MIN);
    if (z_slot >= (unsigned int)CONFIG_Z_SPAN) return 0u;
    if (charge_slot >= (unsigned int)CONFIG_CHARGE_SPAN) return 0u;
    return config_index[z_slot * (unsigned int)CONFIG_CHARGE_SPAN + charge_slot];
}

__device__ __forceinline__ unsigned int config_span_offset(unsigned int span) { return span & 0xFFFFu; }
__device__ __forceinline__ unsigned int config_span_count(unsigned int span) { return span >> 16; }

// Mirrors configurations.select_configuration. A delocalized atom has no bond orders to
// read -- that is what delocalization means -- so its sigma frame constrains it, together
// with the fact that it demonstrably belongs to a circuit. A concrete atom is constrained
// by its sigma frame and its bond-order sum, which is what separates an sp3 centre from an
// sp2 one of the same element.
__device__ __forceinline__ unsigned int config_select(
    const unsigned int* config_index,
    const unsigned int* config_rows,
    int z,
    int charge,
    int sigma_degree,
    int bond_order_sum,
    int total_h,
    bool delocalized,
    ConfigRow& out
) {
    out = config_row_empty();
    const unsigned int span = config_span(config_index, z, charge);
    const unsigned int offset = config_span_offset(span);
    const unsigned int count = config_span_count(span);
    if (count == 0u) return CONFIG_UNDECLARED;
    const int sigma = sigma_degree + total_h;
    unsigned int found = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
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
__device__ __forceinline__ unsigned int config_select_circuit(
    const unsigned int* config_index,
    const unsigned int* config_rows,
    int z,
    int charge,
    int sigma_degree,
    int total_h,
    ConfigRow& out
) {
    out = config_row_empty();
    const unsigned int span = config_span(config_index, z, charge);
    const unsigned int offset = config_span_offset(span);
    const unsigned int count = config_span_count(span);
    if (count == 0u) return CONFIG_UNDECLARED;
    const int sigma = sigma_degree + total_h;
    unsigned int found = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
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
__device__ __forceinline__ unsigned int config_unwritten(
    const unsigned int* config_index,
    const unsigned int* config_rows,
    int z,
    int charge,
    int sigma_degree,
    int bond_order_sum,
    bool delocalized,
    ConfigRow& out,
    int& hydrogens
) {
    out = config_row_empty();
    hydrogens = 0;
    const unsigned int span = config_span(config_index, z, charge);
    const unsigned int offset = config_span_offset(span);
    const unsigned int count = config_span_count(span);
    if (count == 0u) return CONFIG_UNDECLARED;
    int fewest = 0x7FFFFFFF;
    unsigned int found = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
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

// --- valence / aromaticity functions (mirror cuda_retro.py) ---

__device__ __forceinline__ int smallest_aliphatic_valence_geq(int z, int occupied) {
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

__device__ __forceinline__ int arom_default_valence(int z) {
    switch (z) {
        case 4: return 2;  case 5: return 3;  case 6: return 4;  case 7: return 3;
        case 8: return 2;  case 14: return 4; case 15: return 3; case 16: return 2;
        case 32: return 4; case 33: return 3; case 34: return 2; case 50: return 4;
        case 51: return 3; case 52: return 2; default: return -1;
    }
}

__device__ __forceinline__ int arom_outer_electrons(int z) {
    switch (z) {
        case 5: return 3;  case 6: return 4;  case 7: return 5;  case 8: return 6;
        case 15: return 5; case 16: return 6; case 33: return 5; case 34: return 6;
        case 52: return 6; default: return -1;
    }
}

__device__ __forceinline__ float arom_electronegativity(int z) {
    switch (z) {
        case 5: return 2.04f;  case 6: return 2.55f;  case 7: return 3.04f;
        case 8: return 3.44f;  case 9: return 3.98f;  case 14: return 1.90f;
        case 15: return 2.19f; case 16: return 2.58f; case 17: return 3.16f;
        case 33: return 2.18f; case 34: return 2.55f; case 35: return 2.96f;
        case 52: return 2.10f; case 53: return 2.66f; default: return -1.0f;
    }
}

__device__ __forceinline__ bool arom_electrons_stolen(int donor_z, int acceptor_z) {
    const float donor = arom_electronegativity(donor_z);
    const float acceptor = arom_electronegativity(acceptor_z);
    return donor >= 0.0f && acceptor >= 0.0f && acceptor > donor;
}

__device__ __forceinline__ int arom_bond_order(unsigned char bond) {
    bond = bond_class(bond);
    if (bond == BOND_SINGLE) return 1;
    if (bond == BOND_DOUBLE) return 2;
    if (bond == BOND_TRIPLE) return 3;
    return 0;
}

__device__ __forceinline__ int arom_aromatic_implicit_hydrogens(int z, int degree) {
    if (z == 5 || z == 6) {
        const int h = 3 - degree;
        return h > 0 ? h : 0;
    }
    return 0;
}

// Total hydrogens for a canonical identity/WL component atom, mirroring
// total_hydrogens() in vesma/hydrogens.py so WL tokens match the CPU model:
// source-aromatic B/C are degree-derived (3 - degree, even when charged, so
// cyclopentadienyl / tropylium ring carbons keep their H); an aromatic heteroatom, a
// fixed explicit H, or a formal charge leaves no free implicit H; every other atom
// closes valence from concrete bond orders (signed-charge occupancy).
__device__ __forceinline__ int arom_component_hydrogens(
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
__device__ __forceinline__ unsigned char arom_canonical_bond(unsigned char raw, bool aromatic_edge) {
    if (aromatic_edge) return BOND_AROMATIC;
    const unsigned char order = raw & (BOND_SINGLE | BOND_DOUBLE | BOND_TRIPLE);
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
    ChemistryScratch& s,
    int atom,
    const unsigned long long* cyc_row,
    int n,
    const unsigned int* config_index,
    const unsigned int* config_rows
) {
    ConfigRow configuration;
    const unsigned int status = config_select_circuit(
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
            if (((cyc_row[atom] >> neighbor) & 1ull) != 0ull) continue;
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
__device__ bool arom_system_is_aromatic(
    ChemistryScratch& s,
    unsigned long long atom_mask,
    const unsigned long long* cyc_row,
    int n,
    const unsigned int* config_index,
    const unsigned int* config_rows
) {
    int electron_count = 0;
    for (int atom = 0; atom < n; ++atom) {
        if (((atom_mask >> atom) & 1ull) == 0ull) continue;
        const int donor =
            arom_pi_donor(s, atom, cyc_row, n, config_index, config_rows);
        if (donor == PI_INVALID) return false;
        electron_count += donor;
    }
    return electron_count >= 2 && ((electron_count - 2) % 4 == 0);
}

__device__ bool arom_cycles_share_edge(
    const unsigned char* left_path, int left_len,
    const unsigned char* right_path, int right_len
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
__device__ __forceinline__ void arom_cycle_rows(ChemistryScratch& s, int cycle, unsigned long long* rows, int n) {
    for (int i = 0; i < n; ++i) rows[i] = 0ull;
    const int length = s.cycle_len[cycle];
    for (int i = 0; i < length; ++i) {
        const int u = s.cycle_path[cycle][i];
        const int v = s.cycle_path[cycle][(i + 1) % length];
        rows[u] |= 1ull << v;
        rows[v] |= 1ull << u;
    }
}

// Whether a cycle has no edge joining two atoms that are not its neighbours. A cycle
// carrying a chord is not an elementary face of the ring system; it is two smaller cycles
// seen from outside, and counting it as a face would cancel the very edges it is made of.
__device__ bool arom_cycle_is_chordless(ChemistryScratch& s, int cycle) {
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
__device__ __forceinline__ int arom_add_examined_cycle(
    ChemistryScratch& s, int n, int vector_index, int basis_edges
) {
    for (int w = 0; w < GF2_WORDS; ++w) s.gf2_vec[vector_index][w] = 0ull;
    for (int u = 0; u < n; ++u) {
        unsigned long long row = s.degree[u];
        while (row != 0ull) {
            const int v = (__ffsll((long long)row) - 1);
            row &= row - 1ull;
            if (v <= u) continue;
            int slot = -1;
            for (int e = 0; e < basis_edges && slot < 0; ++e) {
                if (s.gf2_u[e] == (unsigned char)u && s.gf2_v[e] == (unsigned char)v) slot = e;
            }
            if (slot < 0) {
                if (basis_edges >= GF2_MAX_EDGES) return -1;
                slot = basis_edges;
                s.gf2_u[slot] = (unsigned char)u;
                s.gf2_v[slot] = (unsigned char)v;
                basis_edges += 1;
            }
            s.gf2_vec[vector_index][slot >> 6] ^= 1ull << (slot & 63);
        }
    }
    return basis_edges;
}

// GF(2) rank of the examined cycles, by elimination on fixed-width bitmasks. Addition over
// GF(2) is exclusive or, so this is Gaussian elimination with no arithmetic to go wrong and
// no tolerance to choose.
__device__ __forceinline__ int arom_gf2_rank(ChemistryScratch& s, int vector_count) {
    int pivots = 0;
    for (int v = 0; v < vector_count; ++v) {
        unsigned long long current[GF2_WORDS];
        for (int w = 0; w < GF2_WORDS; ++w) current[w] = s.gf2_vec[v][w];
        for (int p = 0; p < pivots; ++p) {
            // Reduce by the pivot when doing so clears the pivot's leading bit, which is
            // what `min(vector, vector ^ pivot)` expresses on arbitrary-width integers.
            int lead = s.gf2_lead[p];
            if (((current[lead >> 6] >> (lead & 63)) & 1ull) == 0ull) continue;
            for (int w = 0; w < GF2_WORDS; ++w) current[w] ^= s.gf2_pivot[p][w];
        }
        int lead = -1;
        for (int w = GF2_WORDS - 1; w >= 0 && lead < 0; --w) {
            if (current[w] == 0ull) continue;
            for (int bit = 63; bit >= 0; --bit) {
                if (((current[w] >> bit) & 1ull) != 0ull) { lead = w * 64 + bit; break; }
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
__device__ __forceinline__ bool z_is_metal(int z) {
    return z == 3 || z == 11 || z == 12 || z == 19 || z == 25
        || z == 29 || z == 30 || z == 50 || z == 76;
}

// Which refusal a configuration failure becomes when the atom's own state is what did not
// occupy one. Distinct from the hydrogen mapping below: the same failure means a different
// thing depending on what was asking.
__device__ __forceinline__ unsigned int reject_for_state_failure(unsigned int status) {
    if (status == CONFIG_AMBIGUOUS) return REJECT_CONFIGURATION_AMBIGUOUS;
    if (status == CONFIG_UNDECLARED) return REJECT_UNDECLARED_CONFIGURATION;
    return REJECT_CONFIGURATION_INCOMPATIBLE;
}

// Which refusal a configuration failure becomes when it is hydrogen that needed the
// configuration. An element and charge the table says nothing about is a gap in the model;
// declared states that simply do not fit are a claim about the molecule.
__device__ __forceinline__ unsigned int reject_for_config_failure(unsigned int status) {
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
__device__ unsigned int perceive_frozen_aromaticity_device(
    ChemistryScratch& s,
    int n,
    const unsigned int* config_index,
    const unsigned int* config_rows
) {
    s.atom_mask = 0ull;
    for (int i = 0; i < CUDA_BUCKET_SIZE; ++i) s.edge_rows[i] = 0ull;
    if (n <= 0) return REJECT_NONE;

    unsigned long long source_aromatic = 0ull;
    for (int atom = 0; atom < n; ++atom) {
        int degree = 0;
        unsigned long long row = 0ull;
        for (int other = 0; other < n; ++other) {
            const unsigned char bond = s.comp_bonds[atom][other];
            if (bond == BOND_NONE) continue;
            degree += 1;
            if ((bond & BOND_AROMATIC) != 0) source_aromatic |= 1ull << atom;
            row |= 1ull << other;
        }
        s.heavy_degree[atom] = degree;
        s.adjacency[atom] = row;
        s.total_h[atom] = s.comp_ph[atom];
    }

    // Whether an atom can belong to a circuit at all is a property of its configuration,
    // not of any candidate circuit, so it is settled before cycles are grouped. Grouping
    // first would let a saturated ring that merely shares a bond with an aromatic one be
    // fused into the same system.
    unsigned long long can_delocalize = 0ull;
    for (int atom = 0; atom < n; ++atom) {
        ConfigRow configuration;
        const unsigned int status = config_select_circuit(
            config_index, config_rows, s.z[atom], s.charge[atom],
            s.heavy_degree[atom], s.total_h[atom], configuration);
        if (status == CONFIG_OK && configuration.pi != PI_INVALID) {
            can_delocalize |= 1ull << atom;
        }
    }

    // An atom carrying an asserted delocalized bond whose configuration cannot join a
    // circuit is a configuration failure, and saying so here is the difference between a
    // true cause and a downstream symptom: left alone it surfaces later as an orphan edge,
    // which reports that the bond is not cyclic when the ring is plainly cyclic.
    if ((source_aromatic & ~can_delocalize) != 0ull) return REJECT_CONFIGURATION_INCOMPATIBLE;

    // Enumerate over the eligible subgraph rather than the whole component. A cycle
    // containing an atom that cannot join a circuit is discarded either way, so this is
    // the same set of cycles -- without walking saturated chains and substituents merely
    // to discard them afterward.
    unsigned long long eligible_adjacency[CUDA_BUCKET_SIZE];
    for (int atom = 0; atom < n; ++atom) {
        eligible_adjacency[atom] = (((can_delocalize >> atom) & 1ull) != 0ull)
            ? (s.adjacency[atom] & can_delocalize)
            : 0ull;
    }

    int cycle_count = 0;
    for (int root = 0; root < n; ++root) {
        if (eligible_adjacency[root] == 0ull) continue;
        int path[8];
        int cursor[8];
        int depth = 0;
        unsigned long long visited = 1ull << root;
        path[0] = root;
        cursor[0] = 0;
        while (depth >= 0) {
            const int neighbor = cursor[depth];
            if (neighbor >= n) {
                visited &= ~(1ull << path[depth]);
                depth -= 1;
                continue;
            }
            cursor[depth] = neighbor + 1;
            const int current = path[depth];
            if (((eligible_adjacency[current] >> neighbor) & 1ull) == 0ull) continue;
            const int length = depth + 1;
            if (neighbor == root) {
                if (length >= 3 && length <= 8 && path[1] < path[depth]) {
                    // Every atom reached is eligible by construction of the subgraph.
                    unsigned long long mask = 0ull;
                    for (int i = 0; i < length; ++i) mask |= 1ull << path[i];
                    // Overflowing the cycle store is a machine limit, not an answer. The
                    // old kernel fell through to "fully aliphatic", which under this model
                    // is a silently wrong molecule rather than a refusal.
                    if (cycle_count >= AROM_MAX_CYCLES) return REJECT_PRECURSOR_CAPACITY;
                    for (int i = 0; i < length; ++i) {
                        s.cycle_path[cycle_count][i] = (unsigned char)path[i];
                    }
                    s.cycle_len[cycle_count] = length;
                    s.cycle_mask[cycle_count] = mask;
                    cycle_count += 1;
                }
                continue;
            }
            if (length == 8) continue;
            if (neighbor < root) continue;
            if (((visited >> neighbor) & 1ull) != 0ull) continue;
            depth += 1;
            path[depth] = neighbor;
            cursor[depth] = 0;
            visited |= 1ull << neighbor;
        }
    }

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
        for (int i = 0; i < n; ++i) s.span_rows[i] = 0ull;
        int member_count = 0;
        unsigned long long union_mask = 0ull;
        for (int c = 0; c < cycle_count; ++c) {
            if (s.clabel[c] != root_cycle) continue;
            member_count += 1;
            union_mask |= s.cycle_mask[c];
            const int length = s.cycle_len[c];
            for (int i = 0; i < length; ++i) {
                const int u = s.cycle_path[c][i];
                const int v = s.cycle_path[c][(i + 1) % length];
                s.span_rows[u] |= 1ull << v;
                s.span_rows[v] |= 1ull << u;
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
                    s.edge_rows[i] |= 1ull << j;
                    s.atom_mask |= 1ull << i;
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

    unsigned long long asserted[CUDA_BUCKET_SIZE];
    for (int i = 0; i < n; ++i) {
        asserted[i] = 0ull;
        for (int j = 0; j < n; ++j) {
            const unsigned char bond = s.comp_bonds[i][j];
            if (bond == BOND_NONE) continue;
            if ((bond & (BOND_SINGLE | BOND_DOUBLE | BOND_TRIPLE)) != 0) continue;
            if (((s.edge_rows[i] >> j) & 1ull) != 0ull) continue;
            asserted[i] |= 1ull << j;
        }
    }

    unsigned long long asserted_seen = 0ull;
    for (int root = 0; root < n; ++root) {
        if (asserted[root] == 0ull) continue;
        if (((asserted_seen >> root) & 1ull) != 0ull) continue;
        // Collect the connected run of asserted edges containing this atom.
        unsigned long long component = 1ull << root;
        bool grew = true;
        while (grew) {
            grew = false;
            for (int atom = 0; atom < n; ++atom) {
                if (((component >> atom) & 1ull) == 0ull) continue;
                const unsigned long long reachable = asserted[atom] & ~component;
                if (reachable != 0ull) { component |= reachable; grew = true; }
            }
        }
        asserted_seen |= component;

        int size = 0;
        bool every_degree_two = true;
        for (int atom = 0; atom < n; ++atom) {
            if (((component >> atom) & 1ull) == 0ull) continue;
            size += 1;
            if (__popcll(asserted[atom]) != 2) every_degree_two = false;
        }
        if (size < 3 || !every_degree_two) continue;

        // Walk it, so an authored ring is validated at whatever size it was written rather
        // than being bounded by what enumeration can afford to search.
        for (int i = 0; i < n; ++i) s.degree[i] = 0ull;
        int start = -1;
        for (int atom = 0; atom < n && start < 0; ++atom) {
            if (((component >> atom) & 1ull) != 0ull) start = atom;
        }
        int previous = start;
        int current = (__ffsll((long long)asserted[start]) - 1);
        s.degree[start] |= 1ull << current;
        s.degree[current] |= 1ull << start;
        while (current != start) {
            const unsigned long long onward = asserted[current] & ~(1ull << previous);
            const int step = (__ffsll((long long)onward) - 1);
            s.degree[current] |= 1ull << step;
            s.degree[step] |= 1ull << current;
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
        if (((can_delocalize >> atom) & 1ull) == 0ull) continue;
        const unsigned long long row = s.adjacency[atom] & can_delocalize;
        if (row == 0ull) continue;
        eligible_nodes += 1;
        eligible_edges += __popcll(row);
    }
    eligible_edges /= 2;
    if (eligible_edges > 0) {
        int label[CUDA_BUCKET_SIZE];
        for (int atom = 0; atom < n; ++atom) label[atom] = atom;
        bool merged = true;
        while (merged) {
            merged = false;
            for (int atom = 0; atom < n; ++atom) {
                if (((can_delocalize >> atom) & 1ull) == 0ull) continue;
                unsigned long long row = s.adjacency[atom] & can_delocalize;
                while (row != 0ull) {
                    const int other = (__ffsll((long long)row) - 1);
                    row &= row - 1ull;
                    if (label[other] < label[atom]) { label[atom] = label[other]; merged = true; }
                    else if (label[atom] < label[other]) { label[other] = label[atom]; merged = true; }
                }
            }
        }
        int components = 0;
        for (int atom = 0; atom < n; ++atom) {
            if (((can_delocalize >> atom) & 1ull) == 0ull) continue;
            if ((s.adjacency[atom] & can_delocalize) == 0ull) continue;
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
            const unsigned char bond = s.comp_bonds[i][j];
            if (bond == BOND_NONE) continue;
            if ((bond & (BOND_SINGLE | BOND_DOUBLE | BOND_TRIPLE)) != 0) continue;
            if (((s.edge_rows[i] >> j) & 1ull) == 0ull) return REJECT_DELOCALIZED_EDGE_NOT_CYCLIC;
        }
    }

    return REJECT_NONE;
}

// Ring-or-chain topology for every present edge of the component, for any ring size.
// A bond is a ring bond iff it is not a bridge, so this is a bridge finder, not a ring
// enumerator; bond_in_short_cycle_rows answers the different, bounded question the WL
// identity token asks ("is this edge in a 3-8 ring") and misses rings of nine or more.
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
__device__ void ring_bond_rows(ChemistryScratch& s, int n) {
    for (int v = 0; v < n; ++v) {
        s.ring_rows[v] = 0ull;
        s.ring_path[v] = 0ull;
        s.ring_children[v] = 0ull;
        s.ring_parent_bit[v] = 0ull;
    }

    unsigned long long visited = 0ull;
    for (int root = 0; root < n; ++root) {
        if ((visited & (1ull << root)) != 0ull) continue;
        visited |= 1ull << root;
        unsigned long long frontier = 1ull << root;
        while (frontier != 0ull) {
            unsigned long long next = 0ull;
            for (unsigned long long a = frontier; a != 0ull; a &= a - 1ull) {
                next |= s.adjacency[__ffsll((long long)a) - 1];
            }
            next &= ~visited;
            for (unsigned long long a = next; a != 0ull; a &= a - 1ull) {
                const int v = __ffsll((long long)a) - 1;
                // The parent is drawn from the previous level, whose path mask is final.
                const int parent =
                    __ffsll((long long)(s.adjacency[v] & frontier)) - 1;
                s.ring_path[v] = s.ring_path[parent] | (1ull << v);
                s.ring_parent_bit[v] = 1ull << parent;
                s.ring_children[parent] |= 1ull << v;
            }
            visited |= next;
            frontier = next;
        }
    }

    // Every non-tree edge closes a cycle; fold its tree path into the cyclic-edge set.
    // Each chord is reached from both of its ends, but OR is idempotent, so counting it
    // twice costs two instructions and saves a shift by 64 when filtering to one end.
    unsigned long long cyclic = 0ull;  // bit v iff tree edge (parent[v], v) is on a cycle
    for (int v = 0; v < n; ++v) {
        const unsigned long long tree = s.ring_children[v] | s.ring_parent_bit[v];
        for (unsigned long long a = s.adjacency[v] & ~tree; a != 0ull; a &= a - 1ull) {
            cyclic |= s.ring_path[v] ^ s.ring_path[__ffsll((long long)a) - 1];
        }
    }
    for (int v = 0; v < n; ++v) {
        const unsigned long long tree = s.ring_children[v] | s.ring_parent_bit[v];
        s.ring_rows[v] =
            (s.adjacency[v] & ~tree)                                   // chords are cyclic
            | (s.ring_children[v] & cyclic)                            // cyclic child edges
            | (((cyclic >> v) & 1ull) != 0ull ? s.ring_parent_bit[v] : 0ull);
    }
}

__device__ __forceinline__ unsigned long long horton_upper_row(const ChemistryScratch& s, int u) {
    return s.ring_rows[u] & ~((2ull << u) - 1ull);
}

__device__ __forceinline__ int horton_lead(unsigned long long lo, unsigned long long hi) {
    if (hi != 0ull) return 127 - (int)__clzll(hi);
    if (lo != 0ull) return 63 - (int)__clzll(lo);
    return -1;
}

__device__ __forceinline__ int horton_edge_index(const ChemistryScratch& s, int a, int b) {
    const int lo = min(a, b);
    const int hi = max(a, b);
    return s.horton_prefix[lo] + (int)__popcll(horton_upper_row(s, lo) & ((1ull << hi) - 1ull));
}

// The neighbours of `atom` one step nearer `root` -- its predecessors in the shortest-path DAG --
// that are not below the root: a cycle is collected from its lowest atom only.
__device__ __forceinline__ unsigned long long horton_pred(const ChemistryScratch& s, int root, int atom) {
    const int d = s.horton_dist[atom];
    unsigned long long found = 0ull;
    for (unsigned long long walk = s.ring_rows[atom] & ~((1ull << root) - 1ull); walk != 0ull; walk &= walk - 1ull) {
        const int v = (__ffsll((long long)(walk)) - 1);
        if ((int)s.horton_dist[v] + 1 == d) found |= 1ull << v;
    }
    return found;
}

__device__ void path_start(ChemistryScratch& s, int w, int root, int target) {
    s.path_level[w] = 0;
    s.path_fresh[w] = 1;
    s.path_node[w][0] = (unsigned char)target;
    s.path_atoms[w][0] = 1ull << target;
    s.path_lo[w][0] = 0ull;
    s.path_hi[w][0] = 0ull;
    s.path_choice[w][0] = horton_pred(s, root, target);
}

// The next shortest path from the enumerator's target to `root`, left at path_*[w][level];
// false when there are no more, or the budget ran out (path_budget < 0 tells them apart).
__device__ bool path_next(ChemistryScratch& s, int w, int root) {
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
        const unsigned long long choices = s.path_choice[w][l];
        if (choices == 0ull) {
            if (l == 0) return false;
            l -= 1;
            continue;
        }
        const int u = (__ffsll((long long)(choices)) - 1);
        s.path_choice[w][l] = choices & (choices - 1ull);
        const int e = horton_edge_index(s, u, s.path_node[w][l]);
        s.path_node[w][l + 1] = (unsigned char)u;
        s.path_atoms[w][l + 1] = s.path_atoms[w][l] | (1ull << u);
        s.path_lo[w][l + 1] = s.path_lo[w][l] | (e < 64 ? 1ull << e : 0ull);
        s.path_hi[w][l + 1] = s.path_hi[w][l] | (e >= 64 ? 1ull << (e - 64) : 0ull);
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
__device__ bool ring_membership(ChemistryScratch& s, int n) {
    for (int a = 0; a < n; ++a) { s.ring_count[a] = 0; s.ring_smallest[a] = 0; }
    int edge_count = 0;
    unsigned long long active = 0ull;
    for (int u = 0; u < n; ++u) {
        if (s.ring_rows[u] != 0ull) active |= 1ull << u;
        s.horton_prefix[u] = (unsigned char)min(edge_count, 255);
        for (unsigned long long walk = horton_upper_row(s, u); walk != 0ull; walk &= walk - 1ull) {
            if (edge_count < HORTON_EDGE_CAP) {
                s.horton_eu[edge_count] = (unsigned char)u;
                s.horton_ev[edge_count] = (unsigned char)(__ffsll((long long)(walk)) - 1);
            }
            ++edge_count;
        }
    }
    if (edge_count > HORTON_EDGE_CAP) return false;

    int mu = 0;
    unsigned long long complex_atoms = 0ull;
    unsigned long long seen = 0ull;
    for (unsigned long long pending = active; pending != 0ull; pending = active & ~seen) {
        unsigned long long component = 1ull << (__ffsll((long long)(pending)) - 1);
        unsigned long long frontier = component;
        seen |= frontier;
        while (frontier != 0ull) {
            unsigned long long reach = 0ull;
            for (unsigned long long walk = frontier; walk != 0ull; walk &= walk - 1ull) reach |= s.ring_rows[(__ffsll((long long)(walk)) - 1)];
            frontier = reach & ~seen;
            seen |= frontier;
            component |= frontier;
        }
        int degree_sum = 0;
        for (unsigned long long walk = component; walk != 0ull; walk &= walk - 1ull) {
            degree_sum += (int)__popcll(s.ring_rows[(__ffsll((long long)(walk)) - 1)]);
        }
        const int atoms_c = (int)__popcll(component);
        const int mu_c = degree_sum / 2 - atoms_c + 1;
        if (mu_c == 1) {
            for (unsigned long long walk = component; walk != 0ull; walk &= walk - 1ull) {
                const int a = (__ffsll((long long)(walk)) - 1);
                s.ring_count[a] = 1;
                s.ring_smallest[a] = (unsigned char)atoms_c;
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
        for (unsigned long long roots = complex_atoms; roots != 0ull; roots &= roots - 1ull) {
            const int root = (__ffsll((long long)(roots)) - 1);
            const unsigned long long root_bit = 1ull << root;
            // Graph distances from the root -- through every atom, lower ones included.
            for (int a = 0; a < n; ++a) s.horton_dist[a] = HORTON_FAR;
            s.horton_dist[root] = 0;
            unsigned long long visited = root_bit;
            unsigned long long frontier = root_bit;
            for (int d = 1; frontier != 0ull; ++d) {
                unsigned long long next = 0ull;
                for (unsigned long long walk = frontier; walk != 0ull; walk &= walk - 1ull) next |= s.ring_rows[(__ffsll((long long)(walk)) - 1)];
                next &= ~visited;
                for (unsigned long long walk = next; walk != 0ull; walk &= walk - 1ull) s.horton_dist[(__ffsll((long long)(walk)) - 1)] = (unsigned char)d;
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
                        const unsigned long long lo = s.path_lo[0][lx] | s.path_lo[1][ly] | (index < 64 ? 1ull << index : 0ull);
                        const unsigned long long hi = s.path_hi[0][lx] | s.path_hi[1][ly] | (index >= 64 ? 1ull << (index - 64) : 0ull);
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
            unsigned long long lo = s.horton_cand_lo[c];
            unsigned long long hi = s.horton_cand_hi[c];
            for (;;) {  // relevance: reduce against the basis of the smaller sizes only
                const int lead = horton_lead(lo, hi);
                if (lead < 0) break;
                const int slot = s.horton_lead_slot[lead];
                if (slot < 0 || slot >= frozen) break;
                lo ^= s.horton_basis_lo[slot];
                hi ^= s.horton_basis_hi[slot];
            }
            if (horton_lead(lo, hi) < 0) continue;
            unsigned long long atoms = 0ull;
            for (unsigned long long w = s.horton_cand_lo[c]; w != 0ull; w &= w - 1ull) {
                const int e = (__ffsll((long long)(w)) - 1);
                atoms |= (1ull << s.horton_eu[e]) | (1ull << s.horton_ev[e]);
            }
            for (unsigned long long w = s.horton_cand_hi[c]; w != 0ull; w &= w - 1ull) {
                const int e = 64 + (__ffsll((long long)(w)) - 1);
                atoms |= (1ull << s.horton_eu[e]) | (1ull << s.horton_ev[e]);
            }
            for (; atoms != 0ull; atoms &= atoms - 1ull) {
                const int a = (__ffsll((long long)(atoms)) - 1);
                s.ring_count[a] += 1;
                if (s.ring_smallest[a] == 0) s.ring_smallest[a] = (unsigned char)size;
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
__device__ unsigned int normalize_frozen_aromaticity_and_hydrogens(
    ChemistryScratch& s,
    int n,
    const unsigned int* config_index,
    const unsigned int* config_rows
) {
    for (int atom = 0; atom < n; ++atom) {
        s.z[atom] = exact_z_from_words(
            s.comp_words[atom][0], s.comp_words[atom][1],
            s.comp_words[atom][2], s.comp_words[atom][3]);
        s.charge[atom] = charge_from_mask(s.comp_words[atom][5]);
    }

    const unsigned int refusal =
        perceive_frozen_aromaticity_device(s, n, config_index, config_rows);
    if (refusal != REJECT_NONE) return refusal;

    for (int atom = 0; atom < n; ++atom) {
        s.comp_words[atom][4] =
            (((s.atom_mask >> atom) & 1ull) != 0ull) ? AROMATIC : ALIPHATIC;
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const bool aromatic_edge = ((s.edge_rows[i] >> j) & 1ull) != 0ull;
            s.comp_bonds[i][j] = arom_canonical_bond(s.comp_bonds[i][j], aromatic_edge);
        }
    }

    // arom_canonical_bond can clear an edge the aromatic reclass refused, so ring
    // topology is read off the final graph rather than the pre-reclass adjacency.
    for (int atom = 0; atom < n; ++atom) {
        unsigned long long row = 0ull;
        for (int other = 0; other < n; ++other) {
            if (s.comp_bonds[atom][other] != BOND_NONE) row |= 1ull << other;
        }
        s.adjacency[atom] = row;
    }
    ring_bond_rows(s, n);
    const bool rings_known = ring_membership(s, n);

    for (int atom = 0; atom < n; ++atom) {
        int degree = 0;
        int heavy_half = 0;
        for (int other = 0; other < n; ++other) {
            const unsigned char bond = s.comp_bonds[atom][other];
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
            const unsigned char bond = s.comp_bonds[atom][other];
            if (bond == BOND_NONE) continue;
            order_sum += arom_bond_order(bond);
            if ((bond & BOND_AROMATIC) != 0) delocalized = true;
        }
        ConfigRow state;
        const unsigned int state_status = config_select(
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
        // Relevant cycles, identical to CPU/Metal; caps leave R/r unknown rather than guess.
        s.comp_words[atom][11] = rings_known
            ? bucket_mask_exact(min((int)s.ring_count[atom], 7)) : WILDCARD;
        s.comp_words[atom][12] = rings_known
            ? bucket_mask_exact(min((int)s.ring_smallest[atom], 7)) : WILDCARD;
        s.comp_words[atom][ISOTOPE_WORD] =
            (s.comp_words[atom][ISOTOPE_WORD] & ISOTOPE_FIELD)
            | bucket_mask_exact(__popcll(s.ring_rows[atom]));
    }
    return REJECT_NONE;
}

"""


_CUDA_RETRO_SOURCE = r"""
#define CUDA_BUCKET_SIZE 64
#define CUDA_EDITED_BUCKET_SIZE 128
#define CUDA_PRECURSOR_ATOM_CAP 64
#define ATOM_WORD_COUNT 14
#define MAX_REACTION_EDGES 512
#define GPU_WL_HASH_VERSION 6ull
#define GPU_WL_ROUNDS 8

#define BOND_NONE 0
#define BOND_SINGLE 1
#define BOND_DOUBLE 2
#define BOND_TRIPLE 4
#define BOND_AROMATIC 8
#define BOND_KIND_MASK 15
// Matcher topology field: 01 ring, 10 chain, 11 either; 00 only for no edge.
#define BOND_TOPOLOGY_RING 0x10
#define BOND_TOPOLOGY_CHAIN 0x20

#define ALIPHATIC 1u
#define AROMATIC 2u

__device__ __forceinline__ unsigned char bond_class(unsigned char flags) {
    return flags & BOND_KIND_MASK;
}

#define WL_COMBINE_CONSTANT 0x517CC1B727220A95ull
#define WL_FINAL_LOW_SEED 0x4D6F7267616E574Cull
#define WL_FINAL_HIGH_SEED 0x7665736D615F776Cull
#define SPLITMIX64_GAMMA 0x9E3779B97F4A7C15ull
#define SPLITMIX64_MUL1 0xBF58476D1CE4E5B9ull
#define SPLITMIX64_MUL2 0x94D049BB133111EBull

#define PRIME64_1 11400714785074694791ull
#define PRIME64_2 14029467366897019727ull
#define PRIME64_3 1609587929392839161ull
#define PRIME64_4 9650029242287828579ull
#define PRIME64_5 2870177450012600261ull

extern "C" {

__device__ __forceinline__ unsigned long long rotl64(
    unsigned long long value,
    int shift
) {
    return (value << shift) | (value >> (64 - shift));
}

__device__ __forceinline__ unsigned long long xxh64_round(
    unsigned long long acc,
    unsigned long long input
) {
    acc += input * PRIME64_2;
    acc = rotl64(acc, 31);
    acc *= PRIME64_1;
    return acc;
}

__device__ __forceinline__ unsigned long long xxh64_merge_round(
    unsigned long long acc,
    unsigned long long value
) {
    value = xxh64_round(0ull, value);
    acc ^= value;
    acc = (acc * PRIME64_1) + PRIME64_4;
    return acc;
}

__device__ __forceinline__ unsigned long long xxh64_avalanche(
    unsigned long long hash
) {
    hash ^= hash >> 33;
    hash *= PRIME64_2;
    hash ^= hash >> 29;
    hash *= PRIME64_3;
    hash ^= hash >> 32;
    return hash;
}

__device__ unsigned long long xxh64_uint64_array(
    const unsigned long long* data,
    int count,
    unsigned long long seed
) {
    unsigned long long h64;
    int idx = 0;
    const unsigned long long len = (unsigned long long)count * 8ull;

    if (count >= 4) {
        unsigned long long v1 = seed + PRIME64_1 + PRIME64_2;
        unsigned long long v2 = seed + PRIME64_2;
        unsigned long long v3 = seed;
        unsigned long long v4 = seed - PRIME64_1;
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
        unsigned long long k1 = data[idx];
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

__device__ __forceinline__ unsigned long long mix_uint64(
    unsigned long long value
) {
    value += SPLITMIX64_GAMMA;
    value = (value ^ (value >> 30)) * SPLITMIX64_MUL1;
    value = (value ^ (value >> 27)) * SPLITMIX64_MUL2;
    return value ^ (value >> 31);
}

__device__ __forceinline__ unsigned long long combine_uint64(
    unsigned long long left,
    unsigned long long right
) {
    const unsigned long long combined =
        left ^ (right + WL_COMBINE_CONSTANT + (left << 6) + (left >> 2));
    return mix_uint64(combined);
}

__device__ __forceinline__ int ctz_u32_or_zero(unsigned int value) {
    return value == 0u ? 0 : (__ffs((int)value) - 1);
}

__device__ __forceinline__ bool is_single_bit_u32(unsigned int value) {
    return value != 0u && (value & (value - 1u)) == 0u;
}

__device__ int exact_z_from_atom(const unsigned int atom[ATOM_WORD_COUNT]) {
    if (atom[0] != 0u) return ctz_u32_or_zero(atom[0]);
    if (atom[1] != 0u) return ctz_u32_or_zero(atom[1]) + 32;
    if (atom[2] != 0u) return ctz_u32_or_zero(atom[2]) + 64;
    if (atom[3] != 0u) return ctz_u32_or_zero(atom[3]) + 96;
    return 0;
}

__device__ int charge_from_mask(unsigned int mask) {
    if (mask == 0u || (mask & (mask - 1u)) != 0u) {
        return 0;
    }
    return ctz_u32_or_zero(mask) - 3;
}

__device__ unsigned int charge_mask_exact(int charge) {
    if (charge < -3) charge = -3;
    if (charge > 4) charge = 4;
    return 1u << (unsigned int)(charge + 3);
}

__device__ int exact_bucket_from_mask(unsigned int mask) {
    return ctz_u32_or_zero(mask);
}

__device__ unsigned int bucket_mask_exact(int value) {
    if (value < 0) value = 0;
    if (value > 7) value = 7;
    return 1u << value;
}

__device__ int heavy_valence_half_from_bond(unsigned char bond) {
    const unsigned char order = bond & (BOND_SINGLE | BOND_DOUBLE | BOND_TRIPLE);
    if (order == BOND_SINGLE) return 2;
    if (order == BOND_DOUBLE) return 4;
    if (order == BOND_TRIPLE) return 6;
    if ((bond & BOND_AROMATIC) != 0) return 3;
    return 0;
}

// @DEVICE_VALENCE_SOURCE@

// @DEVICE_SHORT_CYCLE_SOURCE@

// @DEVICE_PERCEIVE_SOURCE@

__device__ unsigned char identity_bond_class(
    unsigned char bonds[CUDA_BUCKET_SIZE][CUDA_BUCKET_SIZE],
    const int* comp_atoms,
    int comp_count,
    int left_local,
    int right_local
) {
    const unsigned char bond = bond_class(
        bonds[comp_atoms[left_local]][comp_atoms[right_local]]
    );
    if (bond == BOND_NONE || bond == BOND_TRIPLE) return bond;
    if (bond_in_short_cycle(
        bonds,
        comp_atoms,
        comp_count,
        left_local,
        right_local
    )) {
        return BOND_AROMATIC;
    }
    if ((bond & (BOND_SINGLE | BOND_DOUBLE | BOND_TRIPLE)) != 0) {
        return bond & (BOND_SINGLE | BOND_DOUBLE | BOND_TRIPLE);
    }
    return BOND_AROMATIC;
}

__device__ bool token_in_stock(
    unsigned long long lo,
    unsigned long long hi,
    const unsigned long long* stock_lo,
    const unsigned long long* stock_hi,
    int stock_count
) {
    int left = 0;
    int right = stock_count;
    while (left < right) {
        const int mid = left + ((right - left) >> 1);
        const unsigned long long mid_lo = stock_lo[mid];
        const unsigned long long mid_hi = stock_hi[mid];
        if (mid_lo < lo || (mid_lo == lo && mid_hi < hi)) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left < stock_count && stock_lo[left] == lo && stock_hi[left] == hi;
}

__device__ unsigned int normalize_aromaticity_and_hydrogens(
    unsigned int atoms[CUDA_BUCKET_SIZE][ATOM_WORD_COUNT],
    unsigned char bonds[CUDA_BUCKET_SIZE][CUDA_BUCKET_SIZE],
    int explicit_h[CUDA_BUCKET_SIZE],
    int implicit_h[CUDA_BUCKET_SIZE],
    int atom_count,
    const unsigned int* config_index,
    const unsigned int* config_rows
) {
    ChemistryScratch s;
    for (int atom = 0; atom < atom_count; ++atom) {
        for (int word = 0; word < ATOM_WORD_COUNT; ++word) {
            s.comp_words[atom][word] = atoms[atom][word];
        }
        for (int other = 0; other < atom_count; ++other) {
            // Perception reasons about bond classes and writes ring-or-chain topology
            // back onto the same array, so it masks on the way in and stays idempotent
            // rather than reading its own output as a bond order.
            s.comp_bonds[atom][other] = bond_class(bonds[atom][other]);
        }
        s.comp_eh[atom] = explicit_h[atom];
        const int explicit_count = explicit_h[atom] < 0 ? 0 : explicit_h[atom];
        s.comp_ph[atom] = implicit_h[atom] < 0
            ? exact_bucket_from_mask(atoms[atom][7])
            : explicit_count + implicit_h[atom];
    }
    const unsigned int refusal = normalize_frozen_aromaticity_and_hydrogens(
        s, atom_count, config_index, config_rows
    );
    if (refusal != REJECT_NONE) return refusal;
    for (int atom = 0; atom < atom_count; ++atom) {
        for (int word = 0; word < ATOM_WORD_COUNT; ++word) {
            atoms[atom][word] = s.comp_words[atom][word];
        }
        for (int other = 0; other < atom_count; ++other) {
            const unsigned char bond = s.comp_bonds[atom][other];
            const unsigned char topology =
                ((s.ring_rows[atom] >> other) & 1ull) != 0ull
                    ? BOND_TOPOLOGY_RING : BOND_TOPOLOGY_CHAIN;
            bonds[atom][other] =
                bond == BOND_NONE ? BOND_NONE : (unsigned char)(bond | topology);
        }
    }
    return REJECT_NONE;
}

__device__ void component_token(
    unsigned int atoms[CUDA_BUCKET_SIZE][ATOM_WORD_COUNT],
    unsigned char bonds[CUDA_BUCKET_SIZE][CUDA_BUCKET_SIZE],
    const int* comp_atoms,
    int comp_count,
    unsigned long long* lo_out,
    unsigned long long* hi_out,
    unsigned long long labels[CUDA_BUCKET_SIZE],
    unsigned long long next_labels[CUDA_BUCKET_SIZE],
    unsigned long long degree[CUDA_BUCKET_SIZE],
    unsigned char identity_bonds[CUDA_BUCKET_SIZE][CUDA_BUCKET_SIZE],
    unsigned long long payload[CUDA_BUCKET_SIZE + 4]
) {
    int edge_count = 0;

    for (int left = 0; left < comp_count; ++left) {
        identity_bonds[left][left] = BOND_NONE;
        for (int right = left + 1; right < comp_count; ++right) {
            const unsigned char bond_class = identity_bond_class(
                bonds,
                comp_atoms,
                comp_count,
                left,
                right
            );
            identity_bonds[left][right] = bond_class;
            identity_bonds[right][left] = bond_class;
            if (bond_class != BOND_NONE) edge_count += 1;
        }
    }

    for (int local = 0; local < comp_count; ++local) {
        const int atom = comp_atoms[local];
        unsigned int atom_words[ATOM_WORD_COUNT];
        for (int word = 0; word < ATOM_WORD_COUNT; ++word) {
            atom_words[word] = atoms[atom][word];
        }
        const int z = exact_z_from_atom(atom_words);
        const int charge = charge_from_mask(atoms[atom][5]);
        // The hydrogen law made total H input-invariant during surgery, so identity
        // reads the stored molecular state instead of re-selecting a valence here.
        const int h_count = exact_bucket_from_mask(atoms[atom][7]);
        unsigned long long deg = 0ull;
        for (int other_local = 0; other_local < comp_count; ++other_local) {
            const unsigned char bond = identity_bonds[local][other_local];
            if (bond == BOND_NONE) continue;
            deg += 1ull;
        }
        degree[local] = deg;
        unsigned long long label = mix_uint64((unsigned long long)z);
        label = combine_uint64(label, (unsigned long long)((long long)charge));
        label = combine_uint64(label, (unsigned long long)h_count);
        label = combine_uint64(label, deg);
        labels[local] = label;
    }

    for (int round = 0; round < GPU_WL_ROUNDS; ++round) {
        for (int local = 0; local < comp_count; ++local) {
            unsigned long long aggregate = 0ull;
            for (int other_local = 0; other_local < comp_count; ++other_local) {
                const unsigned char bond =
                    identity_bonds[local][other_local];
                if (bond != BOND_NONE) {
                    aggregate += combine_uint64(
                        labels[other_local],
                        (unsigned long long)bond
                    );
                }
            }
            unsigned long long label = combine_uint64(labels[local], degree[local]);
            label = combine_uint64(label, aggregate);
            next_labels[local] = mix_uint64(label);
        }
        for (int local = 0; local < comp_count; ++local) {
            labels[local] = next_labels[local];
        }
    }

    for (int left = 0; left < comp_count; ++left) {
        for (int right = left + 1; right < comp_count; ++right) {
            if (labels[right] < labels[left]) {
                const unsigned long long tmp = labels[left];
                labels[left] = labels[right];
                labels[right] = tmp;
            }
        }
    }

    payload[0] = GPU_WL_HASH_VERSION;
    payload[1] = GPU_WL_ROUNDS;
    payload[2] = (unsigned long long)comp_count;
    payload[3] = (unsigned long long)edge_count;
    for (int idx = 0; idx < comp_count; ++idx) {
        payload[4 + idx] = labels[idx];
    }
    *lo_out = xxh64_uint64_array(payload, 4 + comp_count, WL_FINAL_LOW_SEED);
    *hi_out = xxh64_uint64_array(payload, 4 + comp_count, WL_FINAL_HIGH_SEED);
}

__global__ void retro_construct_assignments(
    const unsigned int* product_atoms_global,
    const unsigned char* product_bonds_global,
    const int* product_atom_counts,
    const int* match_pair_pattern_index,
    const int* match_pair_molecule_index,
    const int* assignment_pair_ids,
    const int* assignments,
    const int* reaction_lhs_atom_counts,
    const int* reaction_rhs_atom_counts,
    const int* deleted_lhs_counts,
    const int* deleted_lhs_indices,
    const int* common_counts,
    const int* common_lhs_indices,
    const int* common_rhs_indices,
    const int* common_charge,
    const int* common_charge_explicit,
    const int* lhs_edge_counts,
    const int* lhs_edge_left,
    const int* lhs_edge_right,
    const int* rhs_edge_counts,
    const int* rhs_edge_left,
    const int* rhs_edge_right,
    const unsigned char* rhs_edge_bond_types,
    const unsigned int* rhs_atoms_global,
    // Which zero-level group each RHS atom was written in; 0 for none. The postcondition
    // below is the only reader.
    const unsigned int* rhs_atom_groups_global,
    const unsigned int* config_index,
    const unsigned int* config_rows,
    const int assignment_index_offset,
    const int assignment_count,
    const unsigned long long precursor_capacity,
    unsigned long long* precursor_required_out,
    unsigned long long* precursor_atom_overflow_components_out,
    unsigned long long* precursor_atom_overflow_assignments_out,
    unsigned long long* edited_atom_overflow_assignments_out,
    unsigned long long* undescribed_components_out,
    unsigned long long* orphaned_components_out,
    unsigned int* rejection_reasons_out,
    int* precursor_assignment_ids_out,
    int* precursor_atom_counts_out,
    unsigned int* precursor_atoms_out,
    unsigned char* precursor_bonds_out,
    unsigned long long* precursor_rows_out,
    int* precursor_explicit_h_out,
    int* precursor_implicit_h_out
) {
    const int assignment_idx = blockIdx.x;
    const int tid = threadIdx.x;
    if (assignment_idx >= assignment_count || tid != 0) {
        return;
    }

    const int pair_id = assignment_pair_ids[assignment_idx];
    const int reaction_idx = match_pair_pattern_index[pair_id];
    const int product_idx = match_pair_molecule_index[pair_id];
    const int product_atom_count = product_atom_counts[product_idx];
    const int rhs_atom_count = reaction_rhs_atom_counts[reaction_idx];

    unsigned int edited_atoms[CUDA_EDITED_BUCKET_SIZE][ATOM_WORD_COUNT];
    unsigned char edited_bonds[CUDA_EDITED_BUCKET_SIZE][CUDA_EDITED_BUCKET_SIZE];
    __shared__ bool keep_product[CUDA_BUCKET_SIZE];
    __shared__ int product_to_edited[CUDA_BUCKET_SIZE];
    int rhs_to_edited[CUDA_EDITED_BUCKET_SIZE];
    int comp_labels[CUDA_EDITED_BUCKET_SIZE];
    int comp_atoms[CUDA_EDITED_BUCKET_SIZE];
    __shared__ int assignment[CUDA_BUCKET_SIZE];
    int explicit_h[CUDA_EDITED_BUCKET_SIZE];
    int implicit_h[CUDA_EDITED_BUCKET_SIZE];
    int edited_h[CUDA_EDITED_BUCKET_SIZE];
    int edited_from_product[CUDA_EDITED_BUCKET_SIZE];
    bool described[CUDA_EDITED_BUCKET_SIZE];
    unsigned char edited_group[CUDA_EDITED_BUCKET_SIZE];

    for (int atom = 0; atom < CUDA_EDITED_BUCKET_SIZE; ++atom) {
        rhs_to_edited[atom] = -1;
        comp_labels[atom] = atom;
        explicit_h[atom] = 0;
        implicit_h[atom] = -1;
        edited_h[atom] = -1;
        edited_from_product[atom] = -1;
        described[atom] = false;
        edited_group[atom] = 0u;
        for (int word = 0; word < ATOM_WORD_COUNT; ++word) {
            edited_atoms[atom][word] = 0u;
        }
        for (int other = 0; other < CUDA_EDITED_BUCKET_SIZE; ++other) {
            edited_bonds[atom][other] = BOND_NONE;
        }
    }
    for (int atom = 0; atom < CUDA_BUCKET_SIZE; ++atom) {
        keep_product[atom] = false;
        product_to_edited[atom] = -1;
    }
    const size_t assignment_base =
        (size_t)assignment_idx * CUDA_BUCKET_SIZE;
    for (int atom = 0; atom < CUDA_BUCKET_SIZE; ++atom) {
        assignment[atom] = assignments[assignment_base + atom];
    }

    for (int atom = 0; atom < product_atom_count; ++atom) {
        keep_product[atom] = true;
    }
    const int deleted_count = deleted_lhs_counts[reaction_idx];
    for (int idx = 0; idx < deleted_count; ++idx) {
        const int lhs_idx = deleted_lhs_indices[reaction_idx * CUDA_BUCKET_SIZE + idx];
        const int product_atom = assignment[lhs_idx];
        if (product_atom >= 0 && product_atom < product_atom_count) {
            keep_product[product_atom] = false;
        }
    }

    int edited_count = 0;
    for (int atom = 0; atom < product_atom_count; ++atom) {
        if (!keep_product[atom]) continue;
        product_to_edited[atom] = edited_count;
        for (int word = 0; word < ATOM_WORD_COUNT; ++word) {
            edited_atoms[edited_count][word] =
                product_atoms_global[product_idx * CUDA_BUCKET_SIZE * ATOM_WORD_COUNT + atom * ATOM_WORD_COUNT + word];
        }
        if (edited_atoms[edited_count][4] == AROMATIC) {
            explicit_h[edited_count] =
                exact_bucket_from_mask(edited_atoms[edited_count][7])
                - exact_bucket_from_mask(edited_atoms[edited_count][8]);
            if (explicit_h[edited_count] < 0) {
                explicit_h[edited_count] = 0;
            }
        }
        edited_from_product[edited_count] = atom;
        edited_count += 1;
    }
    for (int left = 0; left < product_atom_count; ++left) {
        if (!keep_product[left]) continue;
        const int edited_left = product_to_edited[left];
        for (int right = 0; right < product_atom_count; ++right) {
            if (!keep_product[right]) continue;
            const int edited_right = product_to_edited[right];
            edited_bonds[edited_left][edited_right] = bond_class(
                product_bonds_global[
                    product_idx * CUDA_BUCKET_SIZE * CUDA_BUCKET_SIZE
                    + left * CUDA_BUCKET_SIZE
                    + right
                ]
            );
        }
    }

    const int common_count = common_counts[reaction_idx];
    for (int idx = 0; idx < common_count; ++idx) {
        const int lhs_idx = common_lhs_indices[reaction_idx * CUDA_BUCKET_SIZE + idx];
        const int rhs_idx = common_rhs_indices[reaction_idx * CUDA_BUCKET_SIZE + idx];
        const int product_atom = assignment[lhs_idx];
        if (product_atom < 0 || product_atom >= product_atom_count) {
            return;
        }
        const int edited_atom = product_to_edited[product_atom];
        if (edited_atom < 0) {
            return;
        }
        rhs_to_edited[rhs_idx] = edited_atom;
        // An atom the RHS writes takes its hydrogen from the edit, never from the state it
        // held in the product. The survivor loop above carried `H - h` forward for aromatic
        // atoms so an `[nH]` keeps its explicit hydrogen; an atom the RHS then rewrites --
        // a triazole N-H becoming a nitrile nitrogen -- must not keep it, or the edited
        // total of 0 is split as explicit 1 and implicit -1. Unconditional, matching the
        // host (`reaction.py`) and the Metal shader: the RHS-pinned branch below would only
        // clear it where the RHS states a count, and the defect is in the other branch.
        explicit_h[edited_atom] = 0;

        const size_t rhs_base =
            (size_t)reaction_idx * CUDA_EDITED_BUCKET_SIZE * ATOM_WORD_COUNT
            + rhs_idx * ATOM_WORD_COUNT;
        const unsigned int rhs_mass = rhs_atoms_global[rhs_base + ISOTOPE_WORD] >> ISOTOPE_SHIFT;
        const unsigned int product_mass = edited_atoms[edited_atom][ISOTOPE_WORD] >> ISOTOPE_SHIFT;
        if (rhs_mass != ISOTOPE_ANY && rhs_mass != product_mass) {
            rejection_reasons_out[assignment_idx] = REJECT_NUCLEAR_CONSERVATION;
            return;
        }
        const unsigned int rhs_a = rhs_atoms_global[rhs_base + 4];
        if (rhs_a != 0xFFu && rhs_a != 0u) edited_atoms[edited_atom][4] = rhs_a;
        const unsigned int rhs_h = rhs_atoms_global[rhs_base + 7];
        const size_t common_offset =
            (size_t)reaction_idx * CUDA_BUCKET_SIZE + idx;
        if (common_charge_explicit[common_offset] != 0) {
            edited_atoms[edited_atom][5] =
                charge_mask_exact(common_charge[common_offset]);
        }
        if (is_single_bit_u32(rhs_h)) {
            const int h_count = exact_bucket_from_mask(rhs_h);
            explicit_h[edited_atom] = 0;
            implicit_h[edited_atom] = h_count;
            edited_h[edited_atom] = h_count;
        }
    }
    for (int rhs_idx = 0; rhs_idx < rhs_atom_count; ++rhs_idx) {
        if (rhs_to_edited[rhs_idx] >= 0) continue;
        if (edited_count >= CUDA_EDITED_BUCKET_SIZE) {
            atomicAdd(edited_atom_overflow_assignments_out, 1ull);
            return;
        }
        rhs_to_edited[rhs_idx] = edited_count;
        edited_from_product[edited_count] = -1;
        for (int word = 0; word < ATOM_WORD_COUNT; ++word) {
            edited_atoms[edited_count][word] =
                rhs_atoms_global[reaction_idx * CUDA_EDITED_BUCKET_SIZE * ATOM_WORD_COUNT + rhs_idx * ATOM_WORD_COUNT + word];
        }
        // An omitted RHS charge means neutral for an added atom, not a query wildcard.
        edited_atoms[edited_count][5] = charge_mask_exact(
            charge_from_mask(edited_atoms[edited_count][5])
        );
        if (edited_atoms[edited_count][4] == 0xFFu) {
            edited_atoms[edited_count][4] = ALIPHATIC;
        }
        if ((edited_atoms[edited_count][ISOTOPE_WORD] >> ISOTOPE_SHIFT) == ISOTOPE_ANY) {
            edited_atoms[edited_count][ISOTOPE_WORD] &= ~ISOTOPE_FIELD;
        }
        if (is_single_bit_u32(edited_atoms[edited_count][7])) {
            const int h_count = exact_bucket_from_mask(edited_atoms[edited_count][7]);
            explicit_h[edited_count] = 0;
            implicit_h[edited_count] = h_count;
            edited_h[edited_count] = h_count;
        }
        edited_count += 1;
    }

    for (int rhs_idx = 0; rhs_idx < rhs_atom_count; ++rhs_idx) {
        const int edited_atom = rhs_to_edited[rhs_idx];
        if (edited_atom < 0) continue;
        described[edited_atom] = true;
        edited_group[edited_atom] = (unsigned char)rhs_atom_groups_global[
            reaction_idx * CUDA_EDITED_BUCKET_SIZE + rhs_idx
        ];
    }

    const int lhs_edges = lhs_edge_counts[reaction_idx];
    for (int edge = 0; edge < lhs_edges; ++edge) {
        const int lhs_left = lhs_edge_left[reaction_idx * MAX_REACTION_EDGES + edge];
        const int lhs_right = lhs_edge_right[reaction_idx * MAX_REACTION_EDGES + edge];
        const int product_left = assignment[lhs_left];
        const int product_right = assignment[lhs_right];
        if (product_left < 0 || product_right < 0) continue;
        const int edited_left = product_to_edited[product_left];
        const int edited_right = product_to_edited[product_right];
        if (edited_left < 0 || edited_right < 0) continue;
        edited_bonds[edited_left][edited_right] = BOND_NONE;
        edited_bonds[edited_right][edited_left] = BOND_NONE;
    }

    const int rhs_edges = rhs_edge_counts[reaction_idx];
    for (int edge = 0; edge < rhs_edges; ++edge) {
        const int rhs_left = rhs_edge_left[reaction_idx * MAX_REACTION_EDGES + edge];
        const int rhs_right = rhs_edge_right[reaction_idx * MAX_REACTION_EDGES + edge];
        const int edited_left = rhs_to_edited[rhs_left];
        const int edited_right = rhs_to_edited[rhs_right];
        if (edited_left < 0 || edited_right < 0) {
            return;
        }
        const unsigned char bond_type = rhs_edge_bond_types[reaction_idx * MAX_REACTION_EDGES + edge];
        edited_bonds[edited_left][edited_right] = bond_type;
        edited_bonds[edited_right][edited_left] = bond_type;
    }

    // Decide every atom's total hydrogen before perception: an RHS count sets it,
    // an unchanged environment copies it, and a changed environment completes it from
    // the matched atom's original closed-shell configuration.
    for (int atom = 0; atom < edited_count; ++atom) {
        if (edited_h[atom] >= 0) continue;

        const int z = exact_z_from_words(
            edited_atoms[atom][0], edited_atoms[atom][1],
            edited_atoms[atom][2], edited_atoms[atom][3]
        );
        const int charge_after = charge_from_mask(edited_atoms[atom][5]);
        int sigma_after = 0;
        int order_after = 0;
        bool delocalized_after = false;
        int histogram_after[4] = {0, 0, 0, 0};
        for (int other = 0; other < edited_count; ++other) {
            const unsigned char bond = bond_class(edited_bonds[atom][other]);
            if (bond == BOND_NONE) continue;
            sigma_after += 1;
            order_after += arom_bond_order(bond);
            if ((bond & BOND_AROMATIC) != 0) delocalized_after = true;
            if (bond & BOND_SINGLE) histogram_after[0] += 1;
            if (bond & BOND_DOUBLE) histogram_after[1] += 1;
            if (bond & BOND_TRIPLE) histogram_after[2] += 1;
            if (bond & BOND_AROMATIC) histogram_after[3] += 1;
        }

        const int source = edited_from_product[atom];
        if (source < 0) {
            ConfigRow configuration;
            int hydrogens = 0;
            const unsigned int status = config_unwritten(
                config_index, config_rows, z, charge_after, sigma_after,
                order_after, delocalized_after, configuration, hydrogens
            );
            if (status != CONFIG_OK) {
                rejection_reasons_out[assignment_idx] =
                    reject_for_config_failure(status);
                return;
            }
            edited_h[atom] = hydrogens;
            continue;
        }

        int sigma_before = 0;
        int order_before = 0;
        bool delocalized_before = false;
        int histogram_before[4] = {0, 0, 0, 0};
        for (int other = 0; other < product_atom_count; ++other) {
            const unsigned char bond = bond_class(
                product_bonds_global[
                    product_idx * CUDA_BUCKET_SIZE * CUDA_BUCKET_SIZE
                    + source * CUDA_BUCKET_SIZE + other
                ]
            );
            if (bond == BOND_NONE) continue;
            sigma_before += 1;
            order_before += arom_bond_order(bond);
            if ((bond & BOND_AROMATIC) != 0) delocalized_before = true;
            if (bond & BOND_SINGLE) histogram_before[0] += 1;
            if (bond & BOND_DOUBLE) histogram_before[1] += 1;
            if (bond & BOND_TRIPLE) histogram_before[2] += 1;
            if (bond & BOND_AROMATIC) histogram_before[3] += 1;
        }
        const size_t source_base =
            (size_t)product_idx * CUDA_BUCKET_SIZE * ATOM_WORD_COUNT
            + source * ATOM_WORD_COUNT;
        const int charge_before = charge_from_mask(
            product_atoms_global[source_base + 5]
        );
        const int total_h_before = exact_bucket_from_mask(
            product_atoms_global[source_base + 7]
        );
        if (charge_after != charge_before) {
            rejection_reasons_out[assignment_idx] =
                REJECT_HYDROGEN_OMITTED_WITH_CHARGE_CHANGE;
            return;
        }

        bool bonding_changed = false;
        for (int kind = 0; kind < 4; ++kind) {
            if (histogram_before[kind] != histogram_after[kind]) {
                bonding_changed = true;
            }
        }
        if (!bonding_changed) {
            edited_h[atom] = total_h_before;
            continue;
        }

        ConfigRow configuration;
        const unsigned int status = config_select(
            config_index, config_rows, z, charge_before, sigma_before,
            order_before, total_h_before, delocalized_before, configuration
        );
        if (status != CONFIG_OK) {
            rejection_reasons_out[assignment_idx] =
                reject_for_config_failure(status);
            return;
        }
        const int completed = z_is_metal(z)
            ? 0
            : (delocalized_after
                ? configuration.sigma_capacity - sigma_after
                : configuration.graph_valence - order_after);
        if (completed < 0) {
            rejection_reasons_out[assignment_idx] =
                REJECT_HYDROGEN_COMPLETION_NEGATIVE;
            return;
        }
        edited_h[atom] = completed;
    }

    for (int atom = 0; atom < edited_count; ++atom) {
        const int explicit_count = explicit_h[atom] < 0 ? 0 : explicit_h[atom];
        implicit_h[atom] = edited_h[atom] - explicit_count;
        if (implicit_h[atom] < 0) implicit_h[atom] = 0;
    }

    for (int iter = 0; iter < CUDA_EDITED_BUCKET_SIZE; ++iter) {
        bool changed = false;
        for (int atom = 0; atom < edited_count; ++atom) {
            int best = comp_labels[atom];
            for (int other = 0; other < edited_count; ++other) {
                if (edited_bonds[atom][other] != BOND_NONE && comp_labels[other] < best) {
                    best = comp_labels[other];
                }
            }
            if (best < comp_labels[atom]) {
                comp_labels[atom] = best;
                changed = true;
            }
        }
        if (!changed) break;
    }

    // What the RHS's parentheses claimed about its own output, against the output. Same
    // non-zero group, same component; different groups, different components. Mirrors
    // `grouping_agrees_with_components` on the host and the same test in `metal_retro.mm`.
    for (int atom = 0; atom < edited_count; ++atom) {
        const unsigned char group = edited_group[atom];
        if (group == 0u) continue;
        for (int other = atom + 1; other < edited_count; ++other) {
            const unsigned char other_group = edited_group[other];
            if (other_group == 0u) continue;
            const bool together = (comp_labels[atom] == comp_labels[other]);
            if ((group == other_group) != together) {
                return;
            }
        }
    }

    int kept_component_count = 0;
    int atom_overflow_components = 0;
    int orphaned_components = 0;
    for (int root = 0; root < edited_count; ++root) {
        if (comp_labels[root] != root) continue;
        int comp_count = 0;
        bool component_described = false;
        for (int atom = 0; atom < edited_count; ++atom) {
            if (comp_labels[atom] != root) continue;
            comp_count += 1;
            if (described[atom]) component_described = true;
        }
        if (!component_described) {
            atomicAdd(undescribed_components_out, 1ull);
            // Not a precursor, and whether that refuses the application depends on how it
            // came to be its own piece. If the *edit* severed it, the disconnection is
            // incomplete. If it was already its own molecule, it is an unreacted reagent --
            // what a unimolecular forward rule run against a union of precursors always
            // leaves behind -- and dropping it is right. Refusing those would refuse every
            // selectivity probe.
            //
            // Severance is a question about the *product's* bonds, not the edited ones:
            // severed iff one of the component's atoms was bonded, before the edit, to
            // something no longer in it. Mirrors `severed()` in `reaction.py` and the same
            // test in `metal_retro.mm`.
            bool was_attached = false;
            for (int atom = 0; atom < edited_count && !was_attached; ++atom) {
                if (comp_labels[atom] != root) continue;
                const int source = edited_from_product[atom];
                if (source < 0) continue;
                for (int other = 0; other < product_atom_count; ++other) {
                    if (other == source) continue;
                    if (bond_class(product_bonds_global[
                            product_idx * CUDA_BUCKET_SIZE * CUDA_BUCKET_SIZE
                            + source * CUDA_BUCKET_SIZE + other]) == 0u) {
                        continue;
                    }
                    const int neighbour = product_to_edited[other];
                    if (neighbour < 0 || comp_labels[neighbour] != root) {
                        was_attached = true;
                        break;
                    }
                }
            }
            if (was_attached) orphaned_components += 1;
            continue;
        }
        if (comp_count > CUDA_PRECURSOR_ATOM_CAP) {
            atom_overflow_components += 1;
        } else {
            kept_component_count += 1;
        }
    }
    if (atom_overflow_components > 0) {
        atomicAdd(
            precursor_atom_overflow_components_out,
            (unsigned long long)atom_overflow_components
        );
        atomicAdd(precursor_atom_overflow_assignments_out, 1ull);
    }
    // A reaction is defined by all of its named precursors, so one that does not fit
    // refuses the whole application rather than emitting the pieces that did. A partial
    // set presents an impossible disconnection as a complete one. Same rule as the
    // edited-atom cap above, and as the CPU reference and Metal.
    //
    // An orphaned component is that failure from the other side: not a piece too large to
    // hold, but a piece the template's precursor side never named. The counts above are
    // recorded first so a refused assignment still says *why* on both grounds when both
    // apply.
    if (orphaned_components > 0) {
        atomicAdd(orphaned_components_out, (unsigned long long)orphaned_components);
    }
    if (atom_overflow_components > 0 || orphaned_components > 0) {
        return;
    }
    const unsigned long long precursor_base = atomicAdd(
        precursor_required_out,
        (unsigned long long)kept_component_count
    );

    int component_offset = 0;
    for (int root = 0; root < edited_count; ++root) {
        if (comp_labels[root] != root) continue;
        int comp_count = 0;
        bool component_described = false;
        for (int atom = 0; atom < edited_count; ++atom) {
            if (comp_labels[atom] == root) {
                comp_atoms[comp_count] = atom;
                comp_count += 1;
                if (described[atom]) component_described = true;
            }
        }
        if (!component_described) continue;

        const unsigned long long precursor_idx =
            precursor_base + (unsigned long long)component_offset;
        component_offset += 1;
        if (precursor_idx >= precursor_capacity) continue;

        precursor_assignment_ids_out[precursor_idx] =
            assignment_index_offset + assignment_idx;
        precursor_atom_counts_out[precursor_idx] = comp_count;

        for (int local = 0; local < comp_count; ++local) {
            const int atom = comp_atoms[local];
            unsigned long long row_bits = 0ull;
            for (int word = 0; word < ATOM_WORD_COUNT; ++word) {
                precursor_atoms_out[
                    precursor_idx * CUDA_BUCKET_SIZE * ATOM_WORD_COUNT +
                    local * ATOM_WORD_COUNT +
                    word
                ] = edited_atoms[atom][word];
            }
            precursor_explicit_h_out[
                precursor_idx * CUDA_BUCKET_SIZE + local
            ] = explicit_h[atom];
            precursor_implicit_h_out[
                precursor_idx * CUDA_BUCKET_SIZE + local
            ] = implicit_h[atom];
            for (int other_local = 0; other_local < comp_count; ++other_local) {
                const int other = comp_atoms[other_local];
                const unsigned char bond = edited_bonds[atom][other];
                precursor_bonds_out[
                    precursor_idx * CUDA_BUCKET_SIZE * CUDA_BUCKET_SIZE +
                    local * CUDA_BUCKET_SIZE +
                    other_local
                ] = bond;
                if (bond != BOND_NONE) {
                    row_bits |= 1ull << other_local;
                }
            }
            precursor_rows_out[
                precursor_idx * CUDA_BUCKET_SIZE + local
            ] = row_bits;
        }
    }
}

__global__ void perceive_component_molecules(
    unsigned int* atoms_global,
    unsigned char* bonds_global,
    const int* atom_counts,
    const int* explicit_h_global,
    const int* implicit_h_global,
    const unsigned int* config_index,
    const unsigned int* config_rows,
    unsigned long long* component_rejections,
    const int* precursor_assignment_ids,
    const int assignment_index_offset,
    const int molecule_count
) {
    const int molecule_idx = blockIdx.x;
    const int tid = threadIdx.x;
    if (molecule_idx >= molecule_count || tid != 0) {
        return;
    }

    const int atom_count = atom_counts[molecule_idx];
    if (atom_count <= 0) {
        return;
    }

    __shared__ unsigned int atoms[CUDA_BUCKET_SIZE][ATOM_WORD_COUNT];
    __shared__ unsigned char bonds[CUDA_BUCKET_SIZE][CUDA_BUCKET_SIZE];
    __shared__ int explicit_h[CUDA_BUCKET_SIZE];
    __shared__ int implicit_h[CUDA_BUCKET_SIZE];

    for (int atom = 0; atom < CUDA_BUCKET_SIZE; ++atom) {
        explicit_h[atom] =
            explicit_h_global[molecule_idx * CUDA_BUCKET_SIZE + atom];
        implicit_h[atom] =
            implicit_h_global[molecule_idx * CUDA_BUCKET_SIZE + atom];
        for (int word = 0; word < ATOM_WORD_COUNT; ++word) {
            atoms[atom][word] = atoms_global[
                molecule_idx * CUDA_BUCKET_SIZE * ATOM_WORD_COUNT +
                atom * ATOM_WORD_COUNT +
                word
            ];
        }
        for (int other = 0; other < CUDA_BUCKET_SIZE; ++other) {
            bonds[atom][other] = bonds_global[
                molecule_idx * CUDA_BUCKET_SIZE * CUDA_BUCKET_SIZE +
                atom * CUDA_BUCKET_SIZE +
                other
            ];
        }
    }

    const unsigned int refusal = normalize_aromaticity_and_hydrogens(
        atoms,
        bonds,
        explicit_h,
        implicit_h,
        atom_count,
        config_index,
        config_rows
    );
    if (refusal != REJECT_NONE) {
        // Several components of one application can fail for different reasons. CPU
        // construction visits components in root order and reports the first one, so
        // pack the emitted component position ahead of the reason and reduce that key.
        // Reducing the reason alone would silently impose enum order instead.
        const unsigned long long decision =
            ((unsigned long long)molecule_idx << 32) | refusal;
        atomicMin(
            component_rejections
                + precursor_assignment_ids[molecule_idx]
                - assignment_index_offset,
            decision
        );
        return;
    }

    for (int atom = 0; atom < atom_count; ++atom) {
        for (int word = 0; word < ATOM_WORD_COUNT; ++word) {
            atoms_global[
                molecule_idx * CUDA_BUCKET_SIZE * ATOM_WORD_COUNT +
                atom * ATOM_WORD_COUNT +
                word
            ] = atoms[atom][word];
        }
        for (int other = 0; other < atom_count; ++other) {
            bonds_global[
                molecule_idx * CUDA_BUCKET_SIZE * CUDA_BUCKET_SIZE +
                atom * CUDA_BUCKET_SIZE +
                other
            ] = bonds[atom][other];
        }
    }
}

__device__ bool bond_in_short_cycle_rows(
    const unsigned long long* rows,
    int comp_count,
    int left,
    int right
) {
    const unsigned long long valid =
        comp_count == 64 ? ~0ull : ((1ull << comp_count) - 1ull);
    const unsigned long long target = 1ull << right;
    unsigned long long visited = 1ull << left;
    unsigned long long frontier = visited;

    for (int distance = 1; distance <= 7; ++distance) {
        unsigned long long neighbors = 0ull;
        unsigned long long active = frontier;
        while (active != 0ull) {
            const int atom = __ffsll((long long)active) - 1;
            unsigned long long row = rows[atom];
            if (atom == left) row &= ~target;
            neighbors |= row;
            active &= active - 1ull;
        }
        frontier = neighbors & valid & ~visited;
        if (distance >= 2 && (frontier & target) != 0ull) return true;
        if (frontier == 0ull) return false;
        visited |= frontier;
    }
    return false;
}

__global__ void probe_configurations(
    const unsigned int* config_index,
    const unsigned int* config_rows,
    const int* queries,
    unsigned int* out,
    const int query_count
) {
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= query_count) return;
    const int z = queries[gid * 6 + 0];
    const int charge = queries[gid * 6 + 1];
    const int sigma_degree = queries[gid * 6 + 2];
    const int bond_order_sum = queries[gid * 6 + 3];
    const int total_h = queries[gid * 6 + 4];
    const bool delocalized = queries[gid * 6 + 5] != 0;
    unsigned int* slot = out + gid * 16;
    for (int i = 0; i < 16; ++i) slot[i] = 0u;

    ConfigRow row = config_row_empty();
    slot[0] = config_select(
        config_index, config_rows, z, charge, sigma_degree, bond_order_sum,
        total_h, delocalized, row
    );
    slot[1] = (unsigned int)row.sigma_capacity;
    slot[2] = (unsigned int)row.graph_valence;
    slot[3] = (unsigned int)(row.pi + CONFIG_PI_BIAS);
    row = config_row_empty();
    slot[4] = config_select_circuit(
        config_index, config_rows, z, charge, sigma_degree, total_h, row
    );
    slot[5] = (unsigned int)row.sigma_capacity;
    slot[6] = (unsigned int)row.graph_valence;
    slot[7] = (unsigned int)(row.pi + CONFIG_PI_BIAS);
    row = config_row_empty();
    int hydrogens = 0;
    slot[8] = config_unwritten(
        config_index, config_rows, z, charge, sigma_degree, bond_order_sum,
        delocalized, row, hydrogens
    );
    slot[9] = (unsigned int)row.sigma_capacity;
    slot[10] = (unsigned int)row.graph_valence;
    slot[11] = (unsigned int)(row.pi + CONFIG_PI_BIAS);
    slot[12] = (unsigned int)hydrogens;
    slot[13] = config_span(config_index, z, charge);
}

__global__ void probe_frozen_perception(
    const int* z_global,
    const int* charge_global,
    const int* total_h_global,
    const unsigned char* bonds_global,
    const int* atom_counts,
    const unsigned int* config_index,
    const unsigned int* config_rows,
    const int molecule_count,
    unsigned char* out_a_global,
    unsigned char* out_bonds_global,
    unsigned int* status_global
) {
    const int molecule_idx = blockIdx.x;
    if (molecule_idx >= molecule_count || threadIdx.x != 0) return;
    const int n = atom_counts[molecule_idx];
    ChemistryScratch s;
    for (int atom = 0; atom < n; ++atom) {
        const size_t atom_offset = (size_t)molecule_idx * CUDA_BUCKET_SIZE + atom;
        s.z[atom] = z_global[atom_offset];
        s.charge[atom] = charge_global[atom_offset];
        s.total_h[atom] = total_h_global[atom_offset];
        s.comp_ph[atom] = total_h_global[atom_offset];
        for (int other = 0; other < n; ++other) {
            s.comp_bonds[atom][other] = bonds_global[
                (size_t)molecule_idx * CUDA_BUCKET_SIZE * CUDA_BUCKET_SIZE
                + atom * CUDA_BUCKET_SIZE + other
            ];
        }
    }
    const unsigned int refusal = perceive_frozen_aromaticity_device(
        s, n, config_index, config_rows
    );
    status_global[molecule_idx] = refusal == REJECT_NONE ? 0u : refusal + 1u;
    if (refusal != REJECT_NONE) return;

    for (int atom = 0; atom < n; ++atom) {
        out_a_global[(size_t)molecule_idx * CUDA_BUCKET_SIZE + atom] =
            ((s.atom_mask >> atom) & 1ull) != 0ull ? AROMATIC : ALIPHATIC;
        for (int other = 0; other < n; ++other) {
            const bool aromatic_edge = ((s.edge_rows[atom] >> other) & 1ull) != 0ull;
            out_bonds_global[
                (size_t)molecule_idx * CUDA_BUCKET_SIZE * CUDA_BUCKET_SIZE
                + atom * CUDA_BUCKET_SIZE + other
            ] = arom_canonical_bond(s.comp_bonds[atom][other], aromatic_edge);
        }
    }
}

__global__ void wl_hash_molecules(
    const unsigned int* atoms_global,
    const unsigned char* bonds_global,
    const unsigned long long* rows_global,
    const int* atom_counts,
    int n,
    unsigned long long* hash_lo_out,
    unsigned long long* hash_hi_out
) {
    const int mol_idx = blockIdx.x;
    const int tid = threadIdx.x;
    if (mol_idx >= n || tid >= 32) return;

    const int comp_count = atom_counts[mol_idx];
    if (comp_count <= 0) {
        if (tid == 0) {
            hash_lo_out[mol_idx] = 0ull;
            hash_hi_out[mol_idx] = 0ull;
        }
        return;
    }

    __shared__ unsigned long long labels[CUDA_BUCKET_SIZE];
    __shared__ unsigned long long next_labels[CUDA_BUCKET_SIZE];
    __shared__ unsigned long long degree[CUDA_BUCKET_SIZE];
    __shared__ unsigned char identity_bonds[CUDA_BUCKET_SIZE][CUDA_BUCKET_SIZE];
    __shared__ unsigned long long payload[CUDA_BUCKET_SIZE + 4];

    const unsigned int* atoms =
        atoms_global + (size_t)mol_idx * CUDA_BUCKET_SIZE * ATOM_WORD_COUNT;
    const unsigned char* bonds =
        bonds_global + (size_t)mol_idx * CUDA_BUCKET_SIZE * CUDA_BUCKET_SIZE;
    const unsigned long long* rows =
        rows_global + (size_t)mol_idx * CUDA_BUCKET_SIZE;

    for (int atom = tid; atom < comp_count; atom += 32) {
        identity_bonds[atom][atom] = BOND_NONE;
    }
    for (int left = 0; left < comp_count; ++left) {
        for (int right = left + 1 + tid; right < comp_count; right += 32) {
            const unsigned char bond = bond_class(
                bonds[left * CUDA_BUCKET_SIZE + right]
            );
            unsigned char identity_class;
            if (bond == BOND_NONE || bond == BOND_TRIPLE) {
                identity_class = bond;
            } else if (bond_in_short_cycle_rows(rows, comp_count, left, right)) {
                identity_class = BOND_AROMATIC;
            } else if ((bond & (BOND_SINGLE | BOND_DOUBLE | BOND_TRIPLE)) != 0) {
                identity_class =
                    bond & (BOND_SINGLE | BOND_DOUBLE | BOND_TRIPLE);
            } else {
                identity_class = BOND_AROMATIC;
            }
            identity_bonds[left][right] = identity_class;
            identity_bonds[right][left] = identity_class;
        }
    }
    __syncthreads();

    for (int local = tid; local < comp_count; local += 32) {
        unsigned int atom_words[ATOM_WORD_COUNT];
        for (int word = 0; word < ATOM_WORD_COUNT; ++word) {
            atom_words[word] = atoms[local * ATOM_WORD_COUNT + word];
        }
        const int z = exact_z_from_atom(atom_words);
        const int charge = charge_from_mask(atoms[local * ATOM_WORD_COUNT + 5]);
        const int h_count = exact_bucket_from_mask(
            atoms[local * ATOM_WORD_COUNT + 7]
        );
        const unsigned long long deg = (unsigned long long)__popcll(rows[local]);
        degree[local] = deg;
        unsigned long long label = mix_uint64((unsigned long long)z);
        label = combine_uint64(label, (unsigned long long)((long long)charge));
        label = combine_uint64(label, (unsigned long long)h_count);
        label = combine_uint64(label, deg);
        labels[local] = label;
    }
    __syncthreads();

    for (int round = 0; round < GPU_WL_ROUNDS; ++round) {
        for (int local = tid; local < comp_count; local += 32) {
            unsigned long long aggregate = 0ull;
            unsigned long long neighbors = rows[local];
            while (neighbors != 0ull) {
                const int other = __ffsll((long long)neighbors) - 1;
                aggregate += combine_uint64(
                    labels[other],
                    (unsigned long long)identity_bonds[local][other]
                );
                neighbors &= neighbors - 1ull;
            }
            unsigned long long label = combine_uint64(labels[local], degree[local]);
            label = combine_uint64(label, aggregate);
            next_labels[local] = mix_uint64(label);
        }
        __syncthreads();
        for (int local = tid; local < comp_count; local += 32) {
            labels[local] = next_labels[local];
        }
        __syncthreads();
    }

    for (int local = tid + comp_count; local < CUDA_BUCKET_SIZE; local += 32) {
        labels[local] = ~0ull;
    }
    __syncthreads();
    for (int width = 2; width <= CUDA_BUCKET_SIZE; width <<= 1) {
        for (int stride = width >> 1; stride > 0; stride >>= 1) {
            for (int local = tid; local < CUDA_BUCKET_SIZE; local += 32) {
                const int other = local ^ stride;
                if (other > local) {
                    const bool ascending = (local & width) == 0;
                    const unsigned long long left = labels[local];
                    const unsigned long long right = labels[other];
                    if ((ascending && left > right) || (!ascending && left < right)) {
                        labels[local] = right;
                        labels[other] = left;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int idx = tid; idx < comp_count; idx += 32) {
        payload[4 + idx] = labels[idx];
    }
    __syncthreads();
    if (tid != 0) return;

    const unsigned long long valid =
        comp_count == 64 ? ~0ull : ((1ull << comp_count) - 1ull);
    int edge_count = 0;
    for (int atom = 0; atom < comp_count; ++atom) {
        edge_count += __popcll(rows[atom] & valid);
    }
    edge_count /= 2;
    payload[0] = GPU_WL_HASH_VERSION;
    payload[1] = GPU_WL_ROUNDS;
    payload[2] = (unsigned long long)comp_count;
    payload[3] = (unsigned long long)edge_count;
    hash_lo_out[mol_idx] = xxh64_uint64_array(
        payload, 4 + comp_count, WL_FINAL_LOW_SEED
    );
    hash_hi_out[mol_idx] = xxh64_uint64_array(
        payload, 4 + comp_count, WL_FINAL_HIGH_SEED
    );
}

}
"""

_CUDA_RETRO_SOURCE = (
    _CUDA_RETRO_SOURCE.replace("// @DEVICE_VALENCE_SOURCE@", "")
    .replace("// @DEVICE_SHORT_CYCLE_SOURCE@", _DEVICE_SHORT_CYCLE_SOURCE.strip())
    .replace("// @DEVICE_PERCEIVE_SOURCE@", _DEVICE_FROZEN_CHEMISTRY_SOURCE.strip())
)


@dataclass(frozen=True)
class CudaStock:
    tokens: frozenset[tuple[int, int]]
    token_lo_device: Any
    token_hi_device: Any

    @property
    def count(self) -> int:
        return len(self.tokens)

    @staticmethod
    def empty() -> CudaStock:
        return CudaStock.from_tokens(())

    @staticmethod
    def from_tokens(tokens: Sequence[tuple[int, int]]) -> CudaStock:
        cp = _require_cupy()
        sorted_tokens = tuple(sorted((int(lo), int(hi)) for lo, hi in tokens))
        lo = np.asarray([token[0] for token in sorted_tokens], dtype=np.uint64)
        hi = np.asarray([token[1] for token in sorted_tokens], dtype=np.uint64)
        return CudaStock(
            tokens=frozenset(sorted_tokens),
            token_lo_device=cp.asarray(lo, dtype=cp.uint64),
            token_hi_device=cp.asarray(hi, dtype=cp.uint64),
        )


@dataclass
class CudaMoleculeStore:
    tokens: set[tuple[int, int]]

    @staticmethod
    def empty() -> CudaMoleculeStore:
        return CudaMoleculeStore(tokens=set())

    def add_tokens(
        self,
        values: Sequence[tuple[int, int]],
    ) -> tuple[tuple[int, int], ...]:
        new_tokens: list[tuple[int, int]] = []
        for token in values:
            normalized = (int(token[0]), int(token[1]))
            if normalized in self.tokens:
                continue
            self.tokens.add(normalized)
            new_tokens.append(normalized)
        return tuple(new_tokens)

    def add_host_molecules(
        self,
        values: Sequence[tuple[tuple[int, int], GMol]],
    ) -> None:
        self.add_tokens(tuple(token for token, _ in values))

    def add_cuda_precursors(
        self,
        result: CudaRetroResult,
        *,
        exclude: frozenset[tuple[int, int]] = frozenset(),
    ) -> CudaMoleculeBatch | None:
        cp = _require_cupy()
        token_lo = result.precursor_hash_lo_host()
        token_hi = result.precursor_hash_hi_host()
        atom_counts = result.precursor_atom_counts_host()
        precursor_indices: list[int] = []
        tokens_to_add: list[tuple[int, int]] = []
        for precursor_idx, (lo, hi) in enumerate(zip(token_lo, token_hi, strict=True)):
            token = (int(lo), int(hi))
            if token in self.tokens or token in exclude:
                continue
            if int(atom_counts[precursor_idx]) <= 0:
                continue
            self.tokens.add(token)
            tokens_to_add.append(token)
            precursor_indices.append(precursor_idx)

        if not tokens_to_add:
            return None

        precursor_device = cp.asarray(precursor_indices, dtype=cp.int32)
        # perceive_component_molecules writes ring-or-chain topology onto each bond
        # byte, so a generated precursor answers `@`/`!@` as exactly as a host-packed one.
        bonds = result.precursor_bonds_device[precursor_device]
        return CudaMoleculeBatch(
            tokens=tuple(tokens_to_add),
            atom_counts_device=result.precursor_atom_counts_device[precursor_device],
            atoms_device=result.precursor_atoms_device[precursor_device],
            bonds_device=bonds,
            rows_device=result.precursor_rows_device[precursor_device],
        )


@dataclass(frozen=True)
class CudaMoleculeBatch:
    tokens: tuple[tuple[int, int], ...]
    atom_counts_device: Any
    atoms_device: Any
    bonds_device: Any
    rows_device: Any

    @property
    def count(self) -> int:
        # Keys may live in a separate host array (de-fused keyer), so a batch can
        # carry empty tokens; fall back to the structural atom-count axis.
        if self.tokens:
            return len(self.tokens)
        return int(self.atom_counts_device.shape[0])

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, selector: Any) -> CudaMoleculeBatch:
        """Pure-device gather: select a subset of molecules by bool mask, int index
        array (numpy or cupy), or slice. Returns a new batch whose device arrays are
        fancy-indexed on the GPU (no host round-trip of atom/bond data). Tokens are
        gathered only when present; an empty selection yields an empty batch."""
        cp = _require_cupy()
        count = self.count
        if isinstance(selector, (int, np.integer)):
            raise TypeError(
                "CudaMoleculeBatch selection must preserve the batch axis; "
                "use a slice or one-dimensional index array."
            )
        if isinstance(selector, slice):
            host_indices = np.arange(count, dtype=np.int64)[selector]
        else:
            raw = np.asarray(selector.get() if hasattr(selector, "get") else selector)
            if raw.ndim != 1:
                raise IndexError("CudaMoleculeBatch selector must be one-dimensional.")
            if raw.dtype == bool:
                if len(raw) != count:
                    raise IndexError(
                        f"boolean mask length {len(raw)} != batch count {count}"
                    )
                host_indices = np.nonzero(raw)[0].astype(np.int64)
            elif np.issubdtype(raw.dtype, np.integer):
                host_indices = raw.astype(np.int64, copy=False)
            else:
                raise TypeError(
                    "CudaMoleculeBatch selector must contain booleans or integers."
                )
        if np.any(host_indices < -count) or np.any(host_indices >= count):
            raise IndexError("CudaMoleculeBatch index is out of range.")
        host_indices = np.where(host_indices < 0, host_indices + count, host_indices)
        device_indices = cp.asarray(host_indices)
        tokens = tuple(self.tokens[int(i)] for i in host_indices) if self.tokens else ()
        return CudaMoleculeBatch(
            tokens=tokens,
            atom_counts_device=self.atom_counts_device[device_indices],
            atoms_device=self.atoms_device[device_indices],
            bonds_device=self.bonds_device[device_indices],
            rows_device=self.rows_device[device_indices],
        )

    @staticmethod
    def from_parts(
        parts: Sequence[CudaMoleculeBatch],
    ) -> CudaMoleculeBatch | None:
        cp = _require_cupy()
        non_empty = tuple(part for part in parts if part.count)
        if not non_empty:
            return None
        if len(non_empty) == 1:
            return non_empty[0]
        # Tokens are optional metadata: a de-fused CUDA batch legitimately has none.
        # A mixed concatenation cannot retain only the keyed parts because ``count``
        # would then read the shortened token axis instead of the structural axis.
        tokens = (
            tuple(token for part in non_empty for token in part.tokens)
            if all(len(part.tokens) == part.count for part in non_empty)
            else ()
        )
        return CudaMoleculeBatch(
            tokens=tokens,
            atom_counts_device=cp.concatenate(
                [part.atom_counts_device for part in non_empty],
            ),
            atoms_device=cp.concatenate(
                [part.atoms_device for part in non_empty],
            ),
            bonds_device=cp.concatenate(
                [part.bonds_device for part in non_empty],
            ),
            rows_device=cp.concatenate(
                [part.rows_device for part in non_empty],
            ),
        )


def prepare_cuda_molecule_batch(
    values: Sequence[tuple[tuple[int, int], GMol]],
) -> CudaMoleculeBatch:
    cp = _require_cupy()
    if not values:
        raise ValueError("values must not be empty.")

    tokens = tuple((int(token[0]), int(token[1])) for token, _ in values)
    counts = np.asarray([gmol.atom_count for _, gmol in values], dtype=np.int32)
    if np.any(counts > GPU_MATCH_ATOM_CAP):
        raise ValueError("CUDA molecule batches support molecules up to 64 atoms.")
    atoms = np.stack(
        [
            _pack_atom_masks_u32(gmol).reshape(
                _CUDA_BUCKET_SIZE,
                _ATOM_WORD_COUNT,
            )
            for _, gmol in values
        ]
    )
    bonds = np.stack(
        [_pack_bond_matrix_u8(gmol, annotate_topology=True) for _, gmol in values]
    )
    rows = np.stack([_pack_adjacency_rows_u64(gmol) for _, gmol in values])
    return CudaMoleculeBatch(
        tokens=tokens,
        atom_counts_device=cp.asarray(counts, dtype=cp.int32),
        atoms_device=cp.asarray(atoms, dtype=cp.uint32),
        bonds_device=cp.asarray(bonds, dtype=cp.uint8),
        rows_device=cp.asarray(rows, dtype=cp.uint64),
    )


def pack_atom_masks_u32(gmol: GMol, bucket_size: int) -> np.ndarray:
    return pack_atom_masks(gmol, bucket_size, aromaticity_any=0xFF)


@dataclass(frozen=True)
class CudaReactionBatch:
    reaction_count: int
    lhs_atom_counts_device: Any
    rhs_atom_counts_device: Any
    deleted_lhs_counts_device: Any
    deleted_lhs_indices_device: Any
    common_counts_device: Any
    common_lhs_indices_device: Any
    common_rhs_indices_device: Any
    common_charge_device: Any
    common_charge_explicit_device: Any
    lhs_edge_counts_device: Any
    lhs_edge_left_device: Any
    lhs_edge_right_device: Any
    rhs_edge_counts_device: Any
    rhs_edge_left_device: Any
    rhs_edge_right_device: Any
    rhs_edge_bond_types_device: Any
    rhs_atoms_device: Any
    rhs_atom_groups_device: Any


@dataclass(frozen=True)
class CudaRetroResult:
    precursor_capacity: int
    precursor_required: int
    overflow_retries: int
    precursor_atom_overflow_components: int
    precursor_atom_overflow_assignments: int
    edited_atom_overflow_assignments: int
    assignment_start: int
    rejection_reasons_device: Any
    rejections: RejectionCounts
    undescribed_components: int
    #: The subset of `undescribed_components` that the edit severed, which is what refuses
    #: the application. The superset counts spectators too -- a component the edit never
    #: touched -- so it cannot say why an application was refused, and this can.
    orphaned_components: int
    precursor_assignment_ids_device: Any
    precursor_hash_lo_device: Any
    precursor_hash_hi_device: Any
    precursor_atom_counts_device: Any
    precursor_atoms_device: Any
    precursor_bonds_device: Any
    precursor_rows_device: Any
    precursor_explicit_h_device: Any
    precursor_implicit_h_device: Any

    def precursor_assignment_ids_host(self) -> np.ndarray:
        return _to_numpy(self.precursor_assignment_ids_device).astype(
            np.int32,
            copy=False,
        )

    def rejection_reasons_host(self) -> np.ndarray:
        return _to_numpy(self.rejection_reasons_device).astype(np.uint32, copy=False)

    def precursor_hash_lo_host(self) -> np.ndarray:
        return _to_numpy(self.precursor_hash_lo_device).astype(np.uint64, copy=False)

    def precursor_hash_hi_host(self) -> np.ndarray:
        return _to_numpy(self.precursor_hash_hi_device).astype(np.uint64, copy=False)

    def precursor_atom_counts_host(self) -> np.ndarray:
        return _to_numpy(self.precursor_atom_counts_device).astype(
            np.int32,
            copy=False,
        )


def prepare_cuda_reaction_batch(rules: Sequence[ReactionPattern]) -> CudaReactionBatch:
    cp = _require_cupy()
    reaction_count = len(rules)
    lhs_atom_counts = np.asarray(
        [rule.lhs.atom_count for rule in rules], dtype=np.int32
    )
    rhs_atom_counts = np.asarray(
        [rule.rhs.atom_count for rule in rules], dtype=np.int32
    )
    if np.any(lhs_atom_counts > GPU_MATCH_ATOM_CAP):
        raise ValueError("CUDA retro supports LHS patterns up to 64 atoms.")
    if np.any(rhs_atom_counts > CUDA_EDITED_ATOM_CAP):
        raise ValueError(
            f"CUDA retro supports RHS patterns up to {CUDA_EDITED_ATOM_CAP} atoms."
        )
    deleted_lhs_counts = np.zeros(reaction_count, dtype=np.int32)
    deleted_lhs_indices = np.full(
        (reaction_count, _CUDA_BUCKET_SIZE), -1, dtype=np.int32
    )
    common_counts = np.zeros(reaction_count, dtype=np.int32)
    common_lhs_indices = np.full(
        (reaction_count, _CUDA_BUCKET_SIZE), -1, dtype=np.int32
    )
    common_rhs_indices = np.full(
        (reaction_count, _CUDA_BUCKET_SIZE), -1, dtype=np.int32
    )
    common_charge = np.zeros((reaction_count, _CUDA_BUCKET_SIZE), dtype=np.int32)
    common_charge_explicit = np.zeros(
        (reaction_count, _CUDA_BUCKET_SIZE), dtype=np.int32
    )
    lhs_edge_counts = np.zeros(reaction_count, dtype=np.int32)
    lhs_edge_left = np.full((reaction_count, _MAX_REACTION_EDGES), -1, dtype=np.int32)
    lhs_edge_right = np.full((reaction_count, _MAX_REACTION_EDGES), -1, dtype=np.int32)
    rhs_edge_counts = np.zeros(reaction_count, dtype=np.int32)
    rhs_edge_left = np.full((reaction_count, _MAX_REACTION_EDGES), -1, dtype=np.int32)
    rhs_edge_right = np.full((reaction_count, _MAX_REACTION_EDGES), -1, dtype=np.int32)
    rhs_edge_bond_types = np.zeros(
        (reaction_count, _MAX_REACTION_EDGES), dtype=np.uint8
    )
    rhs_atoms = np.zeros(
        (reaction_count, CUDA_EDITED_ATOM_CAP, _ATOM_WORD_COUNT), dtype=np.uint32
    )
    rhs_atom_groups = np.zeros((reaction_count, CUDA_EDITED_ATOM_CAP), dtype=np.uint32)

    for reaction_idx, rule in enumerate(rules):
        deleted = rule.edit.deleted_lhs_indices
        if len(deleted) > _CUDA_BUCKET_SIZE:
            raise ValueError("CUDA retro supports at most 64 deleted LHS atoms.")
        deleted_lhs_counts[reaction_idx] = len(deleted)
        deleted_lhs_indices[reaction_idx, : len(deleted)] = deleted

        common = rule.edit.mapped_lhs_indices
        if len(common) > _CUDA_BUCKET_SIZE:
            raise ValueError("CUDA retro supports at most 64 mapped atoms.")
        common_counts[reaction_idx] = len(common)
        common_lhs_indices[reaction_idx, : len(common)] = common
        common_rhs_indices[reaction_idx, : len(common)] = rule.edit.mapped_rhs_indices
        common_charge[reaction_idx, : len(common)] = rule.edit.mapped_rhs_c
        common_charge_explicit[reaction_idx, : len(common)] = (
            rule.edit.mapped_rhs_c_is_explicit
        )

        lhs_edges = rule.edit.removed_lhs_edges
        if len(lhs_edges) > _MAX_REACTION_EDGES:
            raise ValueError("CUDA retro supports at most 512 LHS edit edges.")
        lhs_edge_counts[reaction_idx] = len(lhs_edges)
        if len(lhs_edges):
            lhs_edge_left[reaction_idx, : len(lhs_edges)] = lhs_edges[:, 0]
            lhs_edge_right[reaction_idx, : len(lhs_edges)] = lhs_edges[:, 1]

        rhs_edges = rule.edit.added_rhs_edges
        if len(rhs_edges) > _MAX_REACTION_EDGES:
            raise ValueError("CUDA retro supports at most 512 RHS edit edges.")
        rhs_edge_counts[reaction_idx] = len(rhs_edges)
        if len(rhs_edges):
            rhs_edge_left[reaction_idx, : len(rhs_edges)] = rhs_edges[:, 0]
            rhs_edge_right[reaction_idx, : len(rhs_edges)] = rhs_edges[:, 1]
            rhs_edge_bond_types[reaction_idx, : len(rhs_edges)] = (
                rule.edit.added_rhs_bond_types
            )

        rhs_atoms[reaction_idx] = pack_atom_masks_u32(
            rule.rhs, CUDA_EDITED_ATOM_CAP
        ).reshape(
            CUDA_EDITED_ATOM_CAP,
            _ATOM_WORD_COUNT,
        )
        rhs_atom_groups[reaction_idx, : rule.rhs.atom_count] = rule.rhs.atom_group

    return CudaReactionBatch(
        reaction_count=reaction_count,
        lhs_atom_counts_device=cp.asarray(lhs_atom_counts, dtype=cp.int32),
        rhs_atom_counts_device=cp.asarray(rhs_atom_counts, dtype=cp.int32),
        deleted_lhs_counts_device=cp.asarray(deleted_lhs_counts, dtype=cp.int32),
        deleted_lhs_indices_device=cp.asarray(deleted_lhs_indices, dtype=cp.int32),
        common_counts_device=cp.asarray(common_counts, dtype=cp.int32),
        common_lhs_indices_device=cp.asarray(common_lhs_indices, dtype=cp.int32),
        common_rhs_indices_device=cp.asarray(common_rhs_indices, dtype=cp.int32),
        common_charge_device=cp.asarray(common_charge, dtype=cp.int32),
        common_charge_explicit_device=cp.asarray(
            common_charge_explicit, dtype=cp.int32
        ),
        lhs_edge_counts_device=cp.asarray(lhs_edge_counts, dtype=cp.int32),
        lhs_edge_left_device=cp.asarray(lhs_edge_left, dtype=cp.int32),
        lhs_edge_right_device=cp.asarray(lhs_edge_right, dtype=cp.int32),
        rhs_edge_counts_device=cp.asarray(rhs_edge_counts, dtype=cp.int32),
        rhs_edge_left_device=cp.asarray(rhs_edge_left, dtype=cp.int32),
        rhs_edge_right_device=cp.asarray(rhs_edge_right, dtype=cp.int32),
        rhs_edge_bond_types_device=cp.asarray(rhs_edge_bond_types, dtype=cp.uint8),
        rhs_atoms_device=cp.asarray(rhs_atoms, dtype=cp.uint32),
        rhs_atom_groups_device=cp.asarray(rhs_atom_groups, dtype=cp.uint32),
    )


def construct_retro_cuda(
    batch: CudaPruneBatch,
    match: CudaAssignmentResult,
    reactions: CudaReactionBatch,
    *,
    threads_per_block: int = 32,
    perceive: bool = True,
    compute_wl: bool = True,
    assignment_start: int = 0,
    assignment_stop: int | None = None,
    precursor_capacity: int | None = None,
    precursor_capacity_limit: int | None = None,
) -> CudaRetroResult:
    """Run bond edits and emit a flat, compact precursor molecule batch.

    When ``compute_wl`` is False the internal WL-hash pass is skipped and
    ``precursor_hash_lo/hi_device`` stay zero. Output capacity is estimated from
    assignment count; the kernel keeps counting after overflow and is rerun once
    with the exact required capacity.
    """
    cp = _require_cupy()
    if threads_per_block <= 0:
        raise ValueError("threads_per_block must be positive.")

    total_assignment_count = match.assignment_required
    stop = total_assignment_count if assignment_stop is None else assignment_stop
    if assignment_start < 0 or stop < assignment_start or stop > total_assignment_count:
        raise ValueError(
            "assignment slice is outside compact assignment range: "
            f"[{assignment_start}:{stop}] of {total_assignment_count}"
        )
    assignment_count = stop - assignment_start
    initial_capacity = (
        max(int(np.ceil(assignment_count * 2.2)), 1)
        if precursor_capacity is None
        else precursor_capacity
    )
    if initial_capacity <= 0:
        raise ValueError("precursor_capacity must be positive.")
    if precursor_capacity_limit is not None:
        if precursor_capacity_limit <= 0:
            raise ValueError("precursor_capacity_limit must be positive.")
        initial_capacity = min(initial_capacity, precursor_capacity_limit)

    raw_result = _launch_construct_retro_cuda(
        batch,
        match,
        reactions,
        assignment_start=assignment_start,
        assignment_stop=stop,
        precursor_capacity=initial_capacity,
        threads_per_block=threads_per_block,
    )
    if raw_result.precursor_required > initial_capacity:
        required = raw_result.precursor_required
        if precursor_capacity_limit is not None and required > precursor_capacity_limit:
            raise CudaOutputCapacityError(
                "precursor",
                required,
                precursor_capacity_limit,
            )
        raw_result = _launch_construct_retro_cuda(
            batch,
            match,
            reactions,
            assignment_start=assignment_start,
            assignment_stop=stop,
            precursor_capacity=required,
            threads_per_block=threads_per_block,
        )
        if raw_result.precursor_required != required:
            raise RuntimeError(
                "CUDA precursor count changed between capacity retry passes: "
                f"{required} != {raw_result.precursor_required}"
            )
        raw_result = CudaRetroResult(
            precursor_capacity=required,
            precursor_required=required,
            overflow_retries=1,
            precursor_atom_overflow_components=raw_result.precursor_atom_overflow_components,
            precursor_atom_overflow_assignments=raw_result.precursor_atom_overflow_assignments,
            edited_atom_overflow_assignments=raw_result.edited_atom_overflow_assignments,
            assignment_start=raw_result.assignment_start,
            rejection_reasons_device=raw_result.rejection_reasons_device,
            rejections=raw_result.rejections,
            undescribed_components=raw_result.undescribed_components,
            orphaned_components=raw_result.orphaned_components,
            precursor_assignment_ids_device=raw_result.precursor_assignment_ids_device,
            precursor_hash_lo_device=raw_result.precursor_hash_lo_device,
            precursor_hash_hi_device=raw_result.precursor_hash_hi_device,
            precursor_atom_counts_device=raw_result.precursor_atom_counts_device,
            precursor_atoms_device=raw_result.precursor_atoms_device,
            precursor_bonds_device=raw_result.precursor_bonds_device,
            precursor_rows_device=raw_result.precursor_rows_device,
            precursor_explicit_h_device=raw_result.precursor_explicit_h_device,
            precursor_implicit_h_device=raw_result.precursor_implicit_h_device,
        )
    warn_cuda_atom_overflow(raw_result)
    if perceive:
        raw_result = perceive_cuda_retro(
            raw_result, threads_per_block=threads_per_block
        )

    # Compute WL hashes from the component atoms/bonds via the standalone kernel.
    # Skipped when the keyer is de-fused (the matcher hashes the precursor batch itself).
    if compute_wl and raw_result.precursor_required:
        n_total = raw_result.precursor_required
        wl_lo = cp.zeros(n_total, dtype=cp.uint64)
        wl_hi = cp.zeros(n_total, dtype=cp.uint64)
        wl_kernel = get_wl_hash_kernel(cp)
        wl_kernel(
            (n_total,),
            (32,),
            (
                raw_result.precursor_atoms_device,
                raw_result.precursor_bonds_device,
                raw_result.precursor_rows_device,
                raw_result.precursor_atom_counts_device,
                np.int32(n_total),
                wl_lo,
                wl_hi,
            ),
        )
        component_hash_lo = wl_lo
        component_hash_hi = wl_hi
    else:
        component_hash_lo = raw_result.precursor_hash_lo_device
        component_hash_hi = raw_result.precursor_hash_hi_device

    return CudaRetroResult(
        precursor_capacity=raw_result.precursor_capacity,
        precursor_required=raw_result.precursor_required,
        overflow_retries=raw_result.overflow_retries,
        precursor_atom_overflow_components=raw_result.precursor_atom_overflow_components,
        precursor_atom_overflow_assignments=raw_result.precursor_atom_overflow_assignments,
        edited_atom_overflow_assignments=raw_result.edited_atom_overflow_assignments,
        assignment_start=raw_result.assignment_start,
        rejection_reasons_device=raw_result.rejection_reasons_device,
        rejections=raw_result.rejections,
        undescribed_components=raw_result.undescribed_components,
        orphaned_components=raw_result.orphaned_components,
        precursor_assignment_ids_device=raw_result.precursor_assignment_ids_device,
        precursor_hash_lo_device=component_hash_lo,
        precursor_hash_hi_device=component_hash_hi,
        precursor_atom_counts_device=raw_result.precursor_atom_counts_device,
        precursor_atoms_device=raw_result.precursor_atoms_device,
        precursor_bonds_device=raw_result.precursor_bonds_device,
        precursor_rows_device=raw_result.precursor_rows_device,
        precursor_explicit_h_device=raw_result.precursor_explicit_h_device,
        precursor_implicit_h_device=raw_result.precursor_implicit_h_device,
    )


def _launch_construct_retro_cuda(
    batch: CudaPruneBatch,
    match: CudaAssignmentResult,
    reactions: CudaReactionBatch,
    *,
    assignment_start: int,
    assignment_stop: int,
    precursor_capacity: int,
    threads_per_block: int,
) -> CudaRetroResult:
    cp = _require_cupy()
    assignment_count = assignment_stop - assignment_start
    precursor_required = cp.zeros(1, dtype=cp.uint64)
    precursor_atom_overflow_components = cp.zeros(1, dtype=cp.uint64)
    precursor_atom_overflow_assignments = cp.zeros(1, dtype=cp.uint64)
    edited_atom_overflow_assignments = cp.zeros(1, dtype=cp.uint64)
    undescribed_components = cp.zeros(1, dtype=cp.uint64)
    orphaned_components = cp.zeros(1, dtype=cp.uint64)
    precursor_assignment_ids = cp.full(precursor_capacity, -1, dtype=cp.int32)
    component_atom_counts = cp.zeros(precursor_capacity, dtype=cp.int32)
    component_atoms = cp.zeros(
        (precursor_capacity, _CUDA_BUCKET_SIZE, _ATOM_WORD_COUNT),
        dtype=cp.uint32,
    )
    component_bonds = cp.zeros(
        (precursor_capacity, _CUDA_BUCKET_SIZE, _CUDA_BUCKET_SIZE),
        dtype=cp.uint8,
    )
    component_rows = cp.zeros(
        (precursor_capacity, _CUDA_BUCKET_SIZE),
        dtype=cp.uint64,
    )
    component_explicit_h = cp.zeros(
        (precursor_capacity, _CUDA_BUCKET_SIZE),
        dtype=cp.int32,
    )
    component_implicit_h = cp.full(
        (precursor_capacity, _CUDA_BUCKET_SIZE),
        -1,
        dtype=cp.int32,
    )
    rejection_reasons = cp.full(assignment_count, REJECTION_NONE, dtype=cp.uint32)
    config_index, config_rows = cuda_configuration_table()

    if assignment_count:
        kernel = _get_cuda_retro_kernel(cp)
        kernel(
            (assignment_count,),
            # This kernel is one CUDA thread per assignment; launching extra
            # threads only multiplies its large per-thread local scratch.
            (1,),
            (
                batch.molecule_atoms_u32,
                batch.molecule_bonds_u8,
                batch.molecule_atom_counts_device,
                batch.pair_pattern_index_device,
                batch.pair_molecule_index_device,
                match.assignment_pair_ids_device[assignment_start:assignment_stop],
                match.assignments_device[assignment_start:assignment_stop],
                reactions.lhs_atom_counts_device,
                reactions.rhs_atom_counts_device,
                reactions.deleted_lhs_counts_device,
                reactions.deleted_lhs_indices_device,
                reactions.common_counts_device,
                reactions.common_lhs_indices_device,
                reactions.common_rhs_indices_device,
                reactions.common_charge_device,
                reactions.common_charge_explicit_device,
                reactions.lhs_edge_counts_device,
                reactions.lhs_edge_left_device,
                reactions.lhs_edge_right_device,
                reactions.rhs_edge_counts_device,
                reactions.rhs_edge_left_device,
                reactions.rhs_edge_right_device,
                reactions.rhs_edge_bond_types_device,
                reactions.rhs_atoms_device,
                reactions.rhs_atom_groups_device,
                config_index,
                config_rows,
                np.int32(assignment_start),
                np.int32(assignment_count),
                np.uint64(precursor_capacity),
                precursor_required,
                precursor_atom_overflow_components,
                precursor_atom_overflow_assignments,
                edited_atom_overflow_assignments,
                undescribed_components,
                orphaned_components,
                rejection_reasons,
                precursor_assignment_ids,
                component_atom_counts,
                component_atoms,
                component_bonds,
                component_rows,
                component_explicit_h,
                component_implicit_h,
            ),
        )

    required = int(_to_numpy(precursor_required).reshape(-1)[0])
    atom_overflow_components = int(
        _to_numpy(precursor_atom_overflow_components).reshape(-1)[0]
    )
    atom_overflow_assignments = int(
        _to_numpy(precursor_atom_overflow_assignments).reshape(-1)[0]
    )
    edited_overflow_assignments = int(
        _to_numpy(edited_atom_overflow_assignments).reshape(-1)[0]
    )
    undescribed_count = int(_to_numpy(undescribed_components).reshape(-1)[0])
    orphaned_count = int(_to_numpy(orphaned_components).reshape(-1)[0])
    stored = min(required, precursor_capacity)
    return CudaRetroResult(
        precursor_capacity=precursor_capacity,
        precursor_required=required,
        overflow_retries=0,
        precursor_atom_overflow_components=atom_overflow_components,
        precursor_atom_overflow_assignments=atom_overflow_assignments,
        edited_atom_overflow_assignments=edited_overflow_assignments,
        assignment_start=assignment_start,
        rejection_reasons_device=rejection_reasons,
        rejections=no_rejections(),
        undescribed_components=undescribed_count,
        orphaned_components=orphaned_count,
        precursor_assignment_ids_device=precursor_assignment_ids[:stored],
        precursor_hash_lo_device=cp.zeros(stored, dtype=cp.uint64),
        precursor_hash_hi_device=cp.zeros(stored, dtype=cp.uint64),
        precursor_atom_counts_device=component_atom_counts[:stored],
        precursor_atoms_device=component_atoms[:stored],
        precursor_bonds_device=component_bonds[:stored],
        precursor_rows_device=component_rows[:stored],
        precursor_explicit_h_device=component_explicit_h[:stored],
        precursor_implicit_h_device=component_implicit_h[:stored],
    )


def warn_cuda_atom_overflow(result: CudaRetroResult) -> None:
    if result.precursor_atom_overflow_components:
        warnings.warn(
            "CUDA retro: refused "
            f"{result.precursor_atom_overflow_assignments} assignment(s) whose "
            f"precursors include {result.precursor_atom_overflow_components} "
            f"component(s) over {GPU_PRECURSOR_ATOM_CAP} atoms.",
            UserWarning,
            stacklevel=2,
        )
    if result.edited_atom_overflow_assignments:
        warnings.warn(
            "CUDA retro: dropped "
            f"{result.edited_atom_overflow_assignments} assignment(s) over "
            f"{CUDA_EDITED_ATOM_CAP} edited atoms.",
            UserWarning,
            stacklevel=2,
        )


_CUDA_WL_HASH_KERNEL: Any | None = None


def _get_cuda_retro_module(cp: Any) -> Any:
    global _CUDA_RETRO_MODULE
    if _CUDA_RETRO_MODULE is None:
        _CUDA_RETRO_MODULE = cp.RawModule(
            code=_CUDA_RETRO_SOURCE,
            options=("-std=c++11",),
        )
    return _CUDA_RETRO_MODULE


def _get_cuda_retro_kernel(cp: Any) -> Any:
    global _CUDA_RETRO_KERNEL
    if _CUDA_RETRO_KERNEL is None:
        _CUDA_RETRO_KERNEL = _get_cuda_retro_module(cp).get_function(
            "retro_construct_assignments"
        )
    return _CUDA_RETRO_KERNEL


def _get_cuda_perception_kernel(cp: Any) -> Any:
    global _CUDA_PERCEPTION_KERNEL
    if _CUDA_PERCEPTION_KERNEL is None:
        _CUDA_PERCEPTION_KERNEL = _get_cuda_retro_module(cp).get_function(
            "perceive_component_molecules"
        )
    return _CUDA_PERCEPTION_KERNEL


def probe_cuda_configurations(queries: np.ndarray) -> np.ndarray:
    """Run the three frozen configuration selectors on CUDA for ``(N, 6)`` rows."""
    global _CUDA_CONFIGURATION_PROBE_KERNEL
    cp = _require_cupy()
    values = np.ascontiguousarray(queries, dtype=np.int32)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("queries must have shape (N, 6).")
    if _CUDA_CONFIGURATION_PROBE_KERNEL is None:
        _CUDA_CONFIGURATION_PROBE_KERNEL = _get_cuda_retro_module(cp).get_function(
            "probe_configurations"
        )
    count = values.shape[0]
    out = cp.zeros((count, 16), dtype=cp.uint32)
    config_index, config_rows = cuda_configuration_table()
    threads = 128
    _CUDA_CONFIGURATION_PROBE_KERNEL(
        ((count + threads - 1) // threads,),
        (threads,),
        (
            config_index,
            config_rows,
            cp.asarray(values),
            out,
            np.int32(count),
        ),
    )
    return out.get()


def probe_cuda_perception(
    *,
    z: np.ndarray,
    charge: np.ndarray,
    total_h: np.ndarray,
    bonds: np.ndarray,
    atom_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run frozen circuit perception on padded ``(N, 64)`` molecule arrays."""
    global _CUDA_PERCEPTION_PROBE_KERNEL
    cp = _require_cupy()
    molecule_count = int(atom_counts.shape[0])
    expected_atoms = (molecule_count, _CUDA_BUCKET_SIZE)
    expected_bonds = (molecule_count, _CUDA_BUCKET_SIZE, _CUDA_BUCKET_SIZE)
    if z.shape != expected_atoms or charge.shape != expected_atoms:
        raise ValueError(f"z and charge must have shape {expected_atoms}.")
    if total_h.shape != expected_atoms or bonds.shape != expected_bonds:
        raise ValueError(
            f"total_h must have shape {expected_atoms} and bonds {expected_bonds}."
        )
    if _CUDA_PERCEPTION_PROBE_KERNEL is None:
        _CUDA_PERCEPTION_PROBE_KERNEL = _get_cuda_retro_module(cp).get_function(
            "probe_frozen_perception"
        )
    out_a = cp.zeros(expected_atoms, dtype=cp.uint8)
    out_bonds = cp.zeros(expected_bonds, dtype=cp.uint8)
    status = cp.zeros(molecule_count, dtype=cp.uint32)
    config_index, config_rows = cuda_configuration_table()
    _CUDA_PERCEPTION_PROBE_KERNEL(
        (molecule_count,),
        (1,),
        (
            cp.asarray(z, dtype=cp.int32),
            cp.asarray(charge, dtype=cp.int32),
            cp.asarray(total_h, dtype=cp.int32),
            cp.asarray(bonds, dtype=cp.uint8),
            cp.asarray(atom_counts, dtype=cp.int32),
            config_index,
            config_rows,
            np.int32(molecule_count),
            out_a,
            out_bonds,
            status,
        ),
    )
    return out_a.get(), out_bonds.get(), status.get()


def perceive_cuda_retro(
    result: CudaRetroResult,
    *,
    threads_per_block: int = 32,
) -> CudaRetroResult:
    """Perceive aromaticity and hydrogens for raw precursor components in-place."""
    cp = _require_cupy()
    molecule_count = int(result.precursor_atom_counts_device.size)
    if molecule_count == 0:
        reasons = result.rejection_reasons_host()
        counts = np.bincount(
            reasons[reasons != REJECTION_NONE].astype(np.int64, copy=False),
            minlength=len(no_rejections()),
        )
        return replace(result, rejections=tuple(int(value) for value in counts))
    config_index, config_rows = cuda_configuration_table()
    component_rejections = cp.full(
        result.rejection_reasons_device.size,
        np.uint64(0xFFFFFFFFFFFFFFFF),
        dtype=cp.uint64,
    )
    kernel = _get_cuda_perception_kernel(cp)
    kernel(
        (molecule_count,),
        (threads_per_block,),
        (
            result.precursor_atoms_device.reshape(
                molecule_count,
                _CUDA_BUCKET_SIZE,
                _ATOM_WORD_COUNT,
            ),
            result.precursor_bonds_device.reshape(
                molecule_count,
                _CUDA_BUCKET_SIZE,
                _CUDA_BUCKET_SIZE,
            ),
            result.precursor_atom_counts_device.reshape(molecule_count),
            result.precursor_explicit_h_device.reshape(
                molecule_count,
                _CUDA_BUCKET_SIZE,
            ),
            result.precursor_implicit_h_device.reshape(
                molecule_count,
                _CUDA_BUCKET_SIZE,
            ),
            config_index,
            config_rows,
            component_rejections,
            result.precursor_assignment_ids_device,
            np.int32(result.assignment_start),
            np.int32(molecule_count),
        ),
    )
    cp.cuda.Stream.null.synchronize()
    reasons = result.rejection_reasons_host().copy()
    component_decisions = _to_numpy(component_rejections).astype(np.uint64, copy=False)
    perceived = component_decisions != np.uint64(0xFFFFFFFFFFFFFFFF)
    construction_succeeded = reasons == REJECTION_NONE
    use_perception = perceived & construction_succeeded
    reasons[use_perception] = (
        component_decisions[use_perception] & np.uint64(0xFFFFFFFF)
    ).astype(np.uint32)
    rejection_reasons_device = cp.asarray(reasons, dtype=cp.uint32)
    counts = np.bincount(
        reasons[reasons != REJECTION_NONE].astype(np.int64, copy=False),
        minlength=len(no_rejections()),
    )
    local_assignment_ids = result.precursor_assignment_ids_device - np.int32(
        result.assignment_start
    )
    keep = rejection_reasons_device[local_assignment_ids] == REJECTION_NONE
    valid_count = int(cp.count_nonzero(keep).get())
    return replace(
        result,
        precursor_required=valid_count,
        rejection_reasons_device=rejection_reasons_device,
        rejections=tuple(int(value) for value in counts),
        precursor_assignment_ids_device=result.precursor_assignment_ids_device[keep],
        precursor_hash_lo_device=result.precursor_hash_lo_device[keep],
        precursor_hash_hi_device=result.precursor_hash_hi_device[keep],
        precursor_atom_counts_device=result.precursor_atom_counts_device[keep],
        precursor_atoms_device=result.precursor_atoms_device[keep],
        precursor_bonds_device=result.precursor_bonds_device[keep],
        precursor_rows_device=result.precursor_rows_device[keep],
        precursor_explicit_h_device=result.precursor_explicit_h_device[keep],
        precursor_implicit_h_device=result.precursor_implicit_h_device[keep],
    )


def get_wl_hash_kernel(cp: Any) -> Any:
    global _CUDA_WL_HASH_KERNEL
    if _CUDA_WL_HASH_KERNEL is None:
        _CUDA_WL_HASH_KERNEL = _get_cuda_retro_module(cp).get_function(
            "wl_hash_molecules"
        )
    return _CUDA_WL_HASH_KERNEL


def wl_hash_kernel(batch: CudaMoleculeBatch) -> tuple[Any, Any]:
    """Compute WL hashes for each molecule in a CudaMoleculeBatch.

    Returns (hash_lo, hash_hi) as cp.ndarray of shape (n,) uint64.
    """
    cp = _require_cupy()
    n = batch.count
    if n == 0:
        return cp.empty(0, dtype=cp.uint64), cp.empty(0, dtype=cp.uint64)
    hash_lo = cp.zeros(n, dtype=cp.uint64)
    hash_hi = cp.zeros(n, dtype=cp.uint64)
    kernel = get_wl_hash_kernel(cp)
    kernel(
        (n,),
        (32,),
        (
            batch.atoms_device.astype(cp.uint32, copy=False),
            batch.bonds_device.astype(cp.uint8, copy=False),
            batch.rows_device.astype(cp.uint64, copy=False),
            batch.atom_counts_device.astype(cp.int32, copy=False),
            np.int32(n),
            hash_lo,
            hash_hi,
        ),
    )
    return hash_lo, hash_hi

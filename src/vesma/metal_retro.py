"""Metal retrosynthesis construction and key generation.

The ``retro_construct`` kernel in ``metal_retro.mm`` performs bond edits, precursor
splitting, hydrogen reconstruction, and ring/aromaticity perception. The ``wl_hash_molecules``
kernel computes short-ring WL identity over a *batch* -- parallel arrays in, fresh hash arrays
out, which is CUDA's shape -- so it keys a frontier as readily as a precursor.

Two exports rather than one pass, and that is what makes the arm's keyer total. The old
``wl_hash_precursors`` wrote hashes in place on the records construction had just built, so a
batch the device had not built had nowhere to put them; ``retro_construct`` keys nothing now.

Cross-backend aromaticity/bond-model agreement (Phase-0 audit, branch
``feature/metal-search``). There is **no canonical backend**; correctness is judged
against chemistry. State as of the ``1fe2a21`` "Rework aromaticity perception and
bond-class model" commit and the audit that followed:

- CPU (``aromaticity.py`` delocalized-pi model, Option-C single-class bonds) and the
  CUDA device perception (``cuda_retro.py:perceive_aromaticity_device``) were reworked
  together; CUDA is a documented bit-exact mirror of CPU, but is *unvalidated* on real
  CUDA hardware (none here). The Metal matcher's bond rules were updated and verified
  on-device (CPU = Metal = RDKit on the bench); only the Metal **retro** aromaticity
  port was deferred — this module closes that gap.
- RDKit diverges from CPU on a small (~0.045%), documented, test-locked set
  (fused-amine bridgeheads, boroxine; see ``test_aromaticity.py`` /
  ``test_rdkit_implication.py``). These are accepted, not bugs.
- The Metal kernel mirrors the CPU/CUDA model and is validated directly against CPU
  (``test_metal_retro.py``). The ``v``-word (word 9) follows the CUDA GPU formula
  (``heavy_valence + total_h``), which diverges from CPU ``infer_exact_valence`` for
  aromatic atoms — harmless because ``v`` is never hashed and precursors are never
  decoded to ``CMol`` (matchers re-ingest the packed precursor arrays directly).
- WL hydrogen: the key kernel re-derives total H from the *canonical* component bonds
  (``component_token``, mirroring CPU ``total_hydrogens_for_identity``) rather than reading
  the stored perception H. This keeps identity *input-invariant* — the same molecule hashes
  identically however its bonds were encoded — so a ring-opening precursor cannot fragment
  into two sids. (An earlier attempt to key all backends on the stored perception H was
  rejected: it broke input-invariance, caught by ``test_golden_search``.) Dropping H entirely
  was also rejected (~3% stock collisions: aromatic rings collapse onto saturated ones, since
  short-ring bond classes erase ring bond orders and H is the sole saturation carrier).
"""

from __future__ import annotations

import ctypes
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bonds import QBOND_DEFAULT, BondKind
from .configurations import (
    CONFIG_CHARGE_MAX,
    CONFIG_CHARGE_MIN,
    CONFIG_PI_BIAS,
    CONFIG_Z_SPAN,
    configuration_table,
)
from .gmol import GMol
from .gmol_atoms import AROMATIC
from .limits import (
    ATOM_MASK_WORD_COUNT,
    GPU_MATCH_ATOM_CAP,
    GPU_PRECURSOR_ATOM_CAP,
    MAX_COMPONENTS_PER_ASSIGNMENT,
    METAL_EDITED_ATOM_CAP,
)
from .metal_matcher import (
    _pack_atom_masks,
    _pack_bond_matrix,
    pair_index_array,
)
from .metal_runtime import i32_ptr, load_metal_dylib, u8_ptr, u32_ptr
from .reaction import ReactionPattern
from .rejection import RejectionCounts, no_rejections

__all__ = [
    "CONFIG_CHARGE_MAX",
    "CONFIG_CHARGE_MIN",
    "CONFIG_PI_BIAS",
    "CONFIG_Z_SPAN",
    "MAX_COMPONENTS_PER_ASSIGNMENT",
    "REJECTION_NONE",
    "MetalRetroConstructor",
    "ReactionData",
    "configuration_table",
    "probe_configurations",
    "probe_retro_pipelines",
    "reaction_data_from_pattern",
]

_METAL_SOURCE_PATH = Path(__file__).with_name("metal_retro.mm")

MAX_REACTION_EDGES = 512

# Written into RetroStatus.rejection_reason for an application that was not refused. Zero
# cannot serve: it is itself a declared RejectionReason.
REJECTION_NONE = 0xFFFFFFFF


class ReactionData(ctypes.Structure):
    _fields_ = [
        ("lhs_atom_count", ctypes.c_uint32),
        ("rhs_atom_count", ctypes.c_uint32),
        ("deleted_lhs_count", ctypes.c_uint32),
        ("deleted_lhs_indices", ctypes.c_int32 * GPU_MATCH_ATOM_CAP),
        ("common_count", ctypes.c_uint32),
        ("common_lhs_indices", ctypes.c_int32 * GPU_MATCH_ATOM_CAP),
        ("common_rhs_indices", ctypes.c_int32 * GPU_MATCH_ATOM_CAP),
        ("common_maps", ctypes.c_int32 * GPU_MATCH_ATOM_CAP),
        ("lhs_edge_count", ctypes.c_uint32),
        ("lhs_edge_left", ctypes.c_int32 * MAX_REACTION_EDGES),
        ("lhs_edge_right", ctypes.c_int32 * MAX_REACTION_EDGES),
        ("rhs_edge_count", ctypes.c_uint32),
        ("rhs_edge_left", ctypes.c_int32 * MAX_REACTION_EDGES),
        ("rhs_edge_right", ctypes.c_int32 * MAX_REACTION_EDGES),
        ("rhs_edge_bond_types", ctypes.c_uint8 * MAX_REACTION_EDGES),
        ("rhs_atoms", ctypes.c_uint32 * (ATOM_MASK_WORD_COUNT * METAL_EDITED_ATOM_CAP)),
        ("rhs_atom_maps", ctypes.c_int32 * METAL_EDITED_ATOM_CAP),
        ("rhs_atom_groups", ctypes.c_int32 * METAL_EDITED_ATOM_CAP),
        # What charge the RHS wrote for each mapped atom, and whether it wrote one at all.
        ("common_charge", ctypes.c_int32 * GPU_MATCH_ATOM_CAP),
        ("common_charge_explicit", ctypes.c_int32 * GPU_MATCH_ATOM_CAP),
        ("padding1", ctypes.c_uint32),
        ("padding2", ctypes.c_uint32),
    ]


class RetroStatus(ctypes.Structure):
    _fields_ = [
        ("component_count", ctypes.c_uint32),
        # Non-zero if this assignment produced more components than
        # MAX_COMPONENTS_PER_ASSIGNMENT (the heaviest were kept, lighter ones dropped).
        ("overflow", ctypes.c_uint32),
        # Count of this assignment's precursor components over GPU_PRECURSOR_ATOM_CAP.
        # Non-zero refuses the whole assignment, not just the oversized components.
        ("atom_overflow", ctypes.c_uint32),
        # Non-zero if edited bond-edit scratch exceeded METAL_EDITED_ATOM_CAP.
        ("edited_atom_overflow", ctypes.c_uint32),
        # Start index of this assignment's components in the compacted output_mols
        # buffer (the kernel atomically bump-allocates a span per assignment).
        ("output_base", ctypes.c_uint32),
        # RejectionReason for a refused application, or REJECTION_NONE. A refused
        # application produces no precursors at all: the reason is the outcome.
        ("rejection_reason", ctypes.c_uint32),
        # Components dropped because no RHS atom names them. The RHS enumerates what comes
        # back, so this counts templates that orphan rather than molecules that were built.
        ("undescribed_components", ctypes.c_uint32),
        # The subset the *edit* severed. Only these refuse the application: a component the
        # template never touched is an unreacted reagent in the pot, as a unimolecular
        # forward rule run against a union of precursors always leaves behind.
        ("orphaned_components", ctypes.c_uint32),
        # Non-zero if the product disagreed with the RHS's zero-level grouping: a rule
        # written `>>(B).(C)` that handed back one molecule, or a `>>(B.C)` that fell
        # apart. Refuses the application, like the orphan above.
        ("grouping_violations", ctypes.c_uint32),
        # Which structural give-up refused this assignment, if one did: 1 a mapped LHS atom
        # landing outside the product, 2 a mapped product atom with no edited slot, 3 an RHS edge
        # endpoint that never got one. These also set `rejection_reason` to
        # `UNSUPPORTED_TEMPLATE` -- not a chemistry claim but "this engine cannot act on this
        # template" -- so the assignment is a counted refusal rather than silently neither. This
        # code is the diagnostic the reason cannot carry: which of the three it was.
        ("inconsistent_assignment", ctypes.c_uint32),
        ("profile_stage", ctypes.c_uint32),
        ("profile_edited_atoms", ctypes.c_uint32),
        ("profile_component_merges", ctypes.c_uint32),
        ("profile_kept_components", ctypes.c_uint32),
        ("profile_perceived_atoms", ctypes.c_uint32),
        ("profile_aromatic_atoms", ctypes.c_uint32),
        ("profile_cycles", ctypes.c_uint32),
    ]


def reaction_data_from_pattern(rule: ReactionPattern) -> ReactionData:
    if rule.lhs.atom_count > GPU_MATCH_ATOM_CAP:
        raise ValueError("Metal retro supports LHS patterns up to 64 atoms.")
    if rule.rhs.atom_count > METAL_EDITED_ATOM_CAP:
        raise ValueError("Metal retro supports RHS patterns up to 256 atoms.")
    if len(rule.edit.deleted_lhs_indices) > GPU_MATCH_ATOM_CAP:
        raise ValueError("Metal retro supports at most 64 deleted LHS atoms.")
    if len(rule.edit.mapped_lhs_indices) > GPU_MATCH_ATOM_CAP:
        raise ValueError("Metal retro supports at most 64 mapped atoms.")
    if len(rule.edit.removed_lhs_edges) > MAX_REACTION_EDGES:
        raise ValueError("Metal retro supports at most 512 LHS edit edges.")
    if len(rule.edit.added_rhs_edges) > MAX_REACTION_EDGES:
        raise ValueError("Metal retro supports at most 512 RHS edit edges.")

    data = ReactionData()
    data.lhs_atom_count = rule.lhs.atom_count
    data.rhs_atom_count = rule.rhs.atom_count

    data.deleted_lhs_count = len(rule.edit.deleted_lhs_indices)
    for idx, lhs_idx in enumerate(rule.edit.deleted_lhs_indices):
        data.deleted_lhs_indices[idx] = int(lhs_idx)

    data.common_count = len(rule.edit.mapped_rhs_map)
    for idx, map_id in enumerate(rule.edit.mapped_rhs_map):
        data.common_lhs_indices[idx] = int(rule.edit.mapped_lhs_indices[idx])
        data.common_rhs_indices[idx] = int(rule.edit.mapped_rhs_indices[idx])
        data.common_maps[idx] = int(map_id)
        data.common_charge[idx] = int(rule.edit.mapped_rhs_c[idx])
        data.common_charge_explicit[idx] = int(rule.edit.mapped_rhs_c_is_explicit[idx])

    removed_lhs_edges = rule.edit.removed_lhs_edges
    data.lhs_edge_count = len(removed_lhs_edges)
    for idx, edge in enumerate(removed_lhs_edges):
        data.lhs_edge_left[idx] = int(edge[0])
        data.lhs_edge_right[idx] = int(edge[1])

    added_rhs_edges = rule.edit.added_rhs_edges
    data.rhs_edge_count = len(added_rhs_edges)
    for idx, edge in enumerate(added_rhs_edges):
        data.rhs_edge_left[idx] = int(edge[0])
        data.rhs_edge_right[idx] = int(edge[1])
        data.rhs_edge_bond_types[idx] = int(rule.edit.added_rhs_bond_types[idx])

    packed_rhs_atoms = _pack_atom_masks(rule.rhs, METAL_EDITED_ATOM_CAP)
    ctypes.memmove(
        ctypes.addressof(data) + ReactionData.rhs_atoms.offset,
        packed_rhs_atoms.ctypes.data,
        packed_rhs_atoms.nbytes,
    )

    for idx, atom_map in enumerate(rule.rhs.atom_map):
        data.rhs_atom_maps[idx] = int(atom_map)

    for idx, group in enumerate(rule.rhs.atom_group):
        data.rhs_atom_groups[idx] = int(group)

    return data


_LIBRARY_INSTANCE = None


def _load_metal_library() -> ctypes.CDLL:
    global _LIBRARY_INSTANCE
    if _LIBRARY_INSTANCE is not None:
        return _LIBRARY_INSTANCE

    lib = load_metal_dylib(_METAL_SOURCE_PATH)

    lib.vesma_metal_retro_construct.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),  # product_atoms
        ctypes.POINTER(ctypes.c_uint8),  # product_bonds
        ctypes.POINTER(ctypes.c_uint32),  # product_atom_counts
        ctypes.POINTER(ReactionData),  # reactions
        ctypes.POINTER(ctypes.c_int32),  # assignments
        ctypes.POINTER(ctypes.c_uint32),  # pair_product_indices
        ctypes.POINTER(ctypes.c_uint32),  # pair_reaction_indices
        ctypes.POINTER(ctypes.c_uint32),  # out_atoms
        ctypes.POINTER(ctypes.c_uint8),  # out_bonds
        ctypes.POINTER(ctypes.c_uint32),  # out_atom_counts
        ctypes.POINTER(ctypes.c_uint32),  # out_produced
        ctypes.POINTER(RetroStatus),  # output_status
        ctypes.POINTER(ctypes.c_uint32),  # config_index
        ctypes.c_uint32,  # config_index_count
        ctypes.POINTER(ctypes.c_uint32),  # config_rows
        ctypes.c_uint32,  # config_row_count
        ctypes.c_uint64,  # reaction_token
        ctypes.c_uint64,  # products_token
        ctypes.POINTER(ctypes.c_double),  # out_host_stages[4]
        ctypes.c_int,  # unique_product_count
        ctypes.c_int,  # unique_reaction_count
        ctypes.c_int,  # pair_count
        ctypes.c_int,  # threads_per_group
        ctypes.POINTER(ctypes.c_double),  # out_construct_seconds
        ctypes.c_int,  # 0=normal, 1..5=constructor phases, 6..9=WL phases
    ]
    lib.vesma_metal_retro_construct.restype = ctypes.c_int

    lib.vesma_metal_retro_last_error.argtypes = []
    lib.vesma_metal_retro_last_error.restype = ctypes.c_char_p

    lib.vesma_metal_wl_hash.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),  # atoms
        ctypes.POINTER(ctypes.c_uint8),  # bonds
        ctypes.POINTER(ctypes.c_uint32),  # atom_counts
        ctypes.c_int,  # n
        ctypes.POINTER(ctypes.c_uint64),  # out_hash_lo
        ctypes.POINTER(ctypes.c_uint64),  # out_hash_hi
        ctypes.POINTER(ctypes.c_double),  # out_seconds
        ctypes.c_int,  # profile_mode: the WL stage 1..4, or 0
    ]
    lib.vesma_metal_wl_hash.restype = ctypes.c_int

    lib.vesma_metal_config_probe.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),  # config_index
        ctypes.c_uint32,  # config_index_count
        ctypes.POINTER(ctypes.c_uint32),  # config_rows
        ctypes.c_uint32,  # config_row_count
        ctypes.POINTER(ctypes.c_int32),  # queries
        ctypes.c_uint32,  # query_count
        ctypes.POINTER(ctypes.c_uint32),  # out
    ]
    lib.vesma_metal_config_probe.restype = ctypes.c_int

    lib.vesma_metal_retro_probe.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    lib.vesma_metal_retro_probe.restype = ctypes.c_int

    _LIBRARY_INSTANCE = lib
    return lib


# One row of `probe_configurations` output, per selector: the failure code (0 selected,
# 1 incompatible, 2 ambiguous, 3 undeclared) and the configuration it landed on.
PROBE_WORDS_PER_QUERY = 16


PROBE_FIELDS = (
    "device_max_threadgroup_memory",
    "retro_static_threadgroup_memory",
    "retro_max_threads_per_threadgroup",
    "retro_thread_execution_width",
    "wl_static_threadgroup_memory",
    "wl_max_threads_per_threadgroup",
    "wl_thread_execution_width",
    "scratch_bytes",
    "recommended_working_set",
    "max_buffer_length",
    # The shader's own `#define`s, from `constants_probe`. Each is declared twice -- MSL lives in
    # a raw string and cannot include the host's header -- so `tests/vesma/test_shader_constants.py`
    # compares these against `vesma.limits` instead of trusting the comments beside them.
    "shader_bucket",
    "shader_edited_bucket",
    "shader_atom_word_count",
    "shader_max_components",
    "shader_wl_threads",
    "shader_match_atom_cap",
    "shader_max_reaction_edges",
    # And this file's own host-side C++ declarations of four of those, which are a third copy:
    # the shader source is a raw string and the host code beside it cannot share a `#define`.
    "host_max_components",
    "host_max_reaction_edges",
    "host_edited_bucket",
    "host_wl_threads",
)


@dataclass(frozen=True)
class MetalRetroOutput:
    """What one construct dispatch produced: the status rows, and the precursors it packed.

    The three arrays are read-only *views* into the constructor's grow-only buffers, trimmed to the
    prefix the kernel actually wrote -- `produced` rows, which the device now reports rather than
    leaving the host to infer from the status rows. Read-only because the buffers are reused by the
    next dispatch: handing out a writable alias to recycled storage was previously true and
    documented only by comment.
    """

    status: Any
    atoms: np.ndarray
    bonds: np.ndarray
    atom_counts: np.ndarray


@dataclass(frozen=True)
class MetalWlHash:
    """A keyed batch's tokens, split into halves, and what the dispatch cost.

    A frozen result rather than a `last_seconds` attribute on the function, matching
    `MetalPrefilterResult` and `CudaRetroResult`: the timing belongs to the call that produced it,
    where a module-level mutable would be shared by every caller and every thread.
    """

    hash_lo: np.ndarray
    hash_hi: np.ndarray
    seconds: float


def metal_wl_hash(
    atoms: np.ndarray,
    bonds: np.ndarray,
    atom_counts: np.ndarray,
    *,
    profile_mode: int = 0,
) -> MetalWlHash:
    """Short-ring WL identity for a packed batch, on the device.

    The keyer, over the layout every molecule in this arm already has: `(n, 896)` atom words,
    `(n, 4096)` bond bytes, `(n,)` atom counts. It is the same `component_token` the retro
    pipeline's in-place pass runs, so the tokens are the same tokens -- what differs is that this
    reads a batch instead of the `GpuMol` records one kernel happens to have produced, and so keys
    a frontier as readily as a precursor.

    Topology annotation on the bond bytes does not matter. `identity_bond_class` masks the bond to
    its class and re-derives short-ring membership from the adjacency itself, so a host-packed
    batch carrying `annotate_topology=True` and a device-built one carrying device perception give
    the same token for the same molecule. That is what lets one keyer serve every batch here
    without stripping or negotiating a layout.

    `profile_mode` is the WL stage directly (1..4), not the 6..9 the constructor's entry point
    remaps -- see `vesma_metal_wl_hash`. Non-zero modes return diagnostic checksums, not identities.
    """
    if atoms.ndim != 2 or atoms.shape[1] != 14 * 64:
        raise ValueError(f"atoms must be (n, {14 * 64}) uint32; got {atoms.shape}")
    if bonds.ndim != 2 or bonds.shape[1] != 64 * 64:
        raise ValueError(f"bonds must be (n, {64 * 64}) uint8; got {bonds.shape}")
    if atom_counts.ndim != 1:
        raise ValueError(f"atom_counts must be (n,) uint32; got {atom_counts.shape}")
    count = int(atoms.shape[0])
    if bonds.shape[0] != count or atom_counts.shape[0] != count:
        raise ValueError(
            "atoms, bonds and atom_counts must agree on the batch axis; got "
            f"{atoms.shape[0]}, {bonds.shape[0]}, {atom_counts.shape[0]}"
        )
    if not 0 <= profile_mode <= 4:
        raise ValueError(
            f"profile_mode must be 0..4 for the WL kernel; got {profile_mode}"
        )
    # The kernel loads a molecule into `threadgroup WLScratch`, whose arrays are `[BUCKET][...]`.
    # A count over the bucket writes past the threadgroup allocation, which is silent corruption
    # and a wrong token rather than an error -- the one value here that can do that. No current
    # caller can produce it, since `metal_stack_mols` and `metal_combine` both bound by the same
    # bucket, but this is a public entry point over raw arrays now and the bound is cheap.
    if count and int(atom_counts.max()) > GPU_MATCH_ATOM_CAP:
        raise ValueError(
            f"the WL kernel holds one molecule in {GPU_MATCH_ATOM_CAP} atoms of threadgroup "
            f"scratch; got a count of {int(atom_counts.max())}"
        )

    lo = np.zeros(count, dtype=np.uint64)
    hi = np.zeros(count, dtype=np.uint64)
    if count == 0:
        return MetalWlHash(hash_lo=lo, hash_hi=hi, seconds=0.0)

    packed_atoms = np.ascontiguousarray(atoms, dtype=np.uint32)
    packed_bonds = np.ascontiguousarray(bonds, dtype=np.uint8)
    packed_counts = np.ascontiguousarray(atom_counts, dtype=np.uint32)
    seconds = ctypes.c_double(0.0)
    library = _load_metal_library()
    ok = library.vesma_metal_wl_hash(
        u32_ptr(packed_atoms),
        u8_ptr(packed_bonds),
        u32_ptr(packed_counts),
        ctypes.c_int(count),
        lo.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        hi.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        ctypes.byref(seconds),
        ctypes.c_int(profile_mode),
    )
    if not ok:
        raise RuntimeError(
            f"metal wl_hash failed: {library.vesma_metal_retro_last_error().decode()}"
        )
    return MetalWlHash(hash_lo=lo, hash_hi=hi, seconds=seconds.value)


def probe_retro_pipelines() -> dict[str, int]:
    """Occupancy and pressure limits for the two retro kernels.

    ``*_max_threads_per_threadgroup`` is the readable proxy for register pressure: the
    compiler lowers it beneath the device maximum exactly when a kernel's register
    footprint will not fit more lanes, so a value well under 1024 says the kernel is
    register-bound rather than bound by anything the source makes visible.
    """
    library = _load_metal_library()
    out = (ctypes.c_uint64 * len(PROBE_FIELDS))()
    if library.vesma_metal_retro_probe(out) != 1:
        raise RuntimeError(_metal_last_error(library))
    return dict(zip(PROBE_FIELDS, list(out), strict=True))


def probe_configurations(queries: np.ndarray) -> np.ndarray:
    """Run the device configuration selectors over written atom states.

    ``queries`` is ``(n, 6)`` of ``(z, charge, sigma_degree, bond_order_sum, total_h,
    delocalized)``; the result is ``(n, 16)``, holding for each of the three selectors a
    status word followed by the selected configuration, the unwritten-hydrogen count in
    word 12, and the raw table span the selectors read in word 13. Exists so a test can ask
    the shader what it thinks rather than inferring it from whether perception happened to
    agree -- the two are not the same question, and the first is far easier to act on.
    """
    library = _load_metal_library()
    index, rows = configuration_table()
    flat = np.ascontiguousarray(np.asarray(queries, dtype=np.int32).reshape(-1))
    query_count = flat.size // 6
    out = np.zeros(query_count * PROBE_WORDS_PER_QUERY, dtype=np.uint32)
    ok = library.vesma_metal_config_probe(
        u32_ptr(index),
        ctypes.c_uint32(index.size),
        u32_ptr(rows),
        ctypes.c_uint32(rows.size),
        i32_ptr(flat),
        ctypes.c_uint32(query_count),
        u32_ptr(out),
    )
    if ok != 1:
        raise RuntimeError(_metal_last_error(library))
    return out.reshape(query_count, PROBE_WORDS_PER_QUERY)


def _metal_last_error(lib: ctypes.CDLL) -> str:
    err_bytes = lib.vesma_metal_retro_last_error()
    if err_bytes:
        return err_bytes.decode("utf-8")
    return "Unknown error"


class MetalRetroConstructor:
    """Drives `retro_construct`, one threadgroup per assignment.

    `threads_per_group` is correctness-neutral, and measured to make no difference.

    Correctness-neutral by construction: the kernel's scratch is `scratch_pool[pair_idx]`,
    indexed by threadgroup *position*, and the body returns on `tid != 0`. Every lane but the
    first does nothing, and no lane shares state with another.

    Performance was the open question, and the answer is that 1, 32, 64 and 128 all land in one
    band. Measured on the device's own clock rather than the wall, over the 100-molecule parity
    corpus, interleaved so drift could not favour whichever ran first: the spread between widths
    was smaller than the spread between repeats of a single width, and the one slow repeat
    landed on whichever arm happened to meet it. No absolute figure is recorded here on purpose
    -- the host was under other load, and a number pinned in a docstring only drifts.

    So the value does not matter and 64 is kept because it was already there. What a reader might
    expect -- that 64 idling lanes cost occupancy a single lane would not -- does not show up:
    there are far more assignments than the GPU can host at once, so it schedules threadgroups,
    and the idle lanes never become the constraint.
    """

    def __init__(self, threads_per_group: int = 64) -> None:
        self._library = _load_metal_library()
        self.threads_per_group = threads_per_group
        # Number of assignments in the most recent construct call that overflowed
        # MAX_COMPONENTS_PER_ASSIGNMENT (kept the heaviest components, flagged).
        self.last_overflow_count = 0
        self.last_atom_overflow_component_count = 0
        self.last_atom_overflow_count = 0
        self.last_edited_atom_overflow_count = 0
        #: Per-code counts of the three structural give-ups in `RetroStatus`, last dispatch.
        self.last_inconsistent_assignments: tuple[int, ...] = ()
        # Per-application refusals from the most recent construct call: the reason code for
        # every refused application, and the same counts folded into the CPU's counter
        # layout so the two are directly comparable.
        self.last_rejected_applications: np.ndarray = np.zeros(0, dtype=np.int64)
        self.last_rejections: RejectionCounts = no_rejections()
        # Components dropped as debris because no RHS atom named them.
        self.last_undescribed_components = 0
        # The subset of those the edit severed, which is what actually refuses the application.
        # A component the template never touched is a spectator, counted above but not here.
        self.last_orphaned_components = 0
        # The product disagreeing with the RHS's zero-level grouping, which also refuses.
        self.last_grouping_violations = 0
        self._out_atoms: np.ndarray | None = None
        self._out_bonds: np.ndarray | None = None
        self._out_counts: np.ndarray | None = None
        self._status_buffer: ctypes.Array[RetroStatus] | None = None
        self._rxn_array: ctypes.Array[ReactionData] | None = None
        self._rxn_array_token = 0
        # GPU seconds of the two retro command buffers in the most recent
        # construct_packed call (construct = bond edits + split+perception; wl = identity hash).
        self.last_construct_seconds = 0.0
        # The constructor keys nothing now -- `metal_wl_hash` is its own entry point and the keyer
        # that calls it owns that cost, reporting it on `MetalWlHash.seconds`. Kept at zero rather
        # than removed so every reader of the four host stages and this field keeps working, and
        # so the number says "no WL work happened here" rather than going missing.
        self.last_wl_seconds = 0.0
        self.last_host_stages: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)

    def _reaction_array(
        self, reactions: list[ReactionData], token: int
    ) -> ctypes.Array[ReactionData]:
        """The reaction table as one contiguous ctypes array, built once per token.

        Marshalling a Python list of ReactionData into a contiguous array copies every
        struct, and at ~25 KB each a 42k-rule library is over a gigabyte -- so doing it per
        dispatch cost more than the kernel did. The token is the same promise the device
        buffer relies on: equal tokens mean equal bytes.
        """
        if token and token == self._rxn_array_token and self._rxn_array is not None:
            return self._rxn_array
        array = (ReactionData * len(reactions))(*reactions)
        if token:
            self._rxn_array = array
            self._rxn_array_token = token
        return array

    def _host_outputs(self, slots: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Grow-only host arrays for the three precursor outputs.

        One interleaved `GpuMolStruct` array before, which the caller then de-interleaved straight
        back into these three. Numpy rather than ctypes because the gather downstream is a numpy
        fancy-index either way, and because `setflags(write=False)` on the views handed out is a
        real guarantee where a comment about not writing through them was not.
        """
        if self._out_atoms is None or self._out_atoms.shape[0] < slots:
            self._out_atoms = np.zeros((slots, 14 * 64), dtype=np.uint32)
            self._out_bonds = np.zeros((slots, 64 * 64), dtype=np.uint8)
            self._out_counts = np.zeros(slots, dtype=np.uint32)
        assert self._out_bonds is not None and self._out_counts is not None
        return self._out_atoms, self._out_bonds, self._out_counts

    def _host_status(self, rows: int) -> ctypes.Array[RetroStatus]:
        if self._status_buffer is None or len(self._status_buffer) < rows:
            self._status_buffer = (RetroStatus * rows)()
        return self._status_buffer

    def construct_batch(
        self,
        products: list[GMol],
        reactions: list[ReactionData],
        assignments: np.ndarray,
        pair_indices: np.ndarray | Sequence[tuple[int, int]],
        *,
        profile_mode: int = 0,
    ) -> MetalRetroOutput:
        """
        products: Unique GMol products
        reactions: Unique ReactionData rules
        assignments: [pair_count, 64] int32 array
        pair_indices: (reaction_idx, product_idx) for each assignment

        Returns a `MetalRetroOutput`: the status rows, and the precursors as the three parallel
        arrays every molecule on this arm uses, trimmed to what the kernel produced.

        ``profile_mode`` is for benchmarks. Modes 1, 2, 4, and 5 stop before
        producing molecules; mode 3 records counters while producing normal output.
        Modes 6–9 profile WL stages and return diagnostic keys.
        """
        if not products:
            return MetalRetroOutput(
                status=(RetroStatus * 0)(),
                atoms=np.zeros((0, 14 * 64), dtype=np.uint32),
                bonds=np.zeros((0, 64 * 64), dtype=np.uint8),
                atom_counts=np.zeros(0, dtype=np.uint32),
            )
        bucket_size = 64
        p_atoms = np.concatenate([_pack_atom_masks(p, bucket_size) for p in products])
        p_bonds = np.concatenate([_pack_bond_matrix(p, bucket_size) for p in products])
        p_counts = np.array([p.atom_count for p in products], dtype=np.uint32)
        return self.construct_packed(
            product_atoms=p_atoms,
            product_bonds=p_bonds,
            product_counts=p_counts,
            unique_product_count=len(products),
            reactions=reactions,
            assignments=assignments,
            pair_indices=pair_indices,
            profile_mode=profile_mode,
        )

    def construct_packed(
        self,
        *,
        product_atoms: np.ndarray,
        product_bonds: np.ndarray,
        product_counts: np.ndarray,
        unique_product_count: int,
        reactions: list[ReactionData],
        assignments: np.ndarray,
        pair_indices: np.ndarray | Sequence[tuple[int, int]],
        reaction_token: int = 0,
        products_token: int = 0,
        profile_mode: int = 0,
    ) -> MetalRetroOutput:
        """Run bond edits + perception + WL on pre-packed product arrays (the flat
        `_pack_*` forms), so a molecule batch held as packed arrays can be split
        without rebuilding GMols. ``product_bonds`` must contain class bits only:
        perception derives ring-or-chain topology itself and writes it onto the emitted
        precursor bonds, so an annotated input would only be overwritten.

        ``pair_indices`` is `(reaction_idx, product_idx)` per assignment, as a
        ``(N, 2)`` array (fast path) or a sequence of tuples.

        ``reaction_token`` and ``products_token`` let the device keep the uploaded
        reaction table and product arrays across dispatches. A nonzero token is a promise
        that the same token always means the same bytes; zero means "upload every time",
        which is correct but re-copies a table that for a 42k-rule library is over a
        gigabyte -- and a caller that cannot make the promise should pay that rather than
        risk serving a stale one.

        ``profile_mode`` uses the same benchmark-only modes as ``construct_batch``;
        6, 7, and 8 run full construction and stop WL after bond identity,
        initial labels, and refinement rounds, respectively. Mode 9 stops after
        bond identity with the short-ring search omitted. Their keys are
        diagnostic checksums, not molecule identities.
        """
        pairs = pair_index_array(
            pair_indices,
            first_count=len(reactions),
            second_count=unique_product_count,
            first_name="reaction",
            second_name="product",
        )
        pair_count = pairs.shape[0]
        if profile_mode not in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
            raise ValueError("profile_mode must be between 0 and 9.")
        if pair_count == 0:
            return MetalRetroOutput(
                status=(RetroStatus * 0)(),
                atoms=np.zeros((0, 14 * 64), dtype=np.uint32),
                bonds=np.zeros((0, 64 * 64), dtype=np.uint8),
                atom_counts=np.zeros(0, dtype=np.uint32),
            )
        if assignments.shape[0] != pair_count:
            raise ValueError(
                "assignments and pair_indices must have the same first dimension."
            )

        rxn_array = self._reaction_array(reactions, reaction_token)
        pair_r_idx = np.ascontiguousarray(pairs[:, 0], dtype=np.uint32)
        pair_p_idx = np.ascontiguousarray(pairs[:, 1], dtype=np.uint32)

        # Grow-only host buffers. ctypes zero-fills on construction, and at ~8 KB per
        # precursor slot the worst-case output array is hundreds of megabytes -- paying
        # that per dispatch cost more than the kernel it was collecting from.
        out_atoms, out_bonds, out_counts = self._host_outputs(
            pair_count * MAX_COMPONENTS_PER_ASSIGNMENT
        )
        output_status = self._host_status(pair_count)

        assign_flat = np.ascontiguousarray(assignments.astype(np.int32).reshape(-1))
        product_atoms = np.ascontiguousarray(product_atoms, dtype=np.uint32)
        product_bonds = np.ascontiguousarray(product_bonds, dtype=np.uint8)
        product_counts = np.ascontiguousarray(product_counts, dtype=np.uint32)

        construct_seconds = ctypes.c_double(0.0)
        produced = ctypes.c_uint32(0)
        host_stages = (ctypes.c_double * 4)()
        config_index, config_rows = configuration_table()
        ok = self._library.vesma_metal_retro_construct(
            u32_ptr(product_atoms),
            u8_ptr(product_bonds),
            u32_ptr(product_counts),
            rxn_array,
            ctypes.cast(assign_flat.ctypes.data, ctypes.POINTER(ctypes.c_int32)),
            u32_ptr(pair_p_idx),
            u32_ptr(pair_r_idx),
            u32_ptr(out_atoms),
            u8_ptr(out_bonds),
            u32_ptr(out_counts),
            ctypes.byref(produced),
            output_status,
            u32_ptr(config_index),
            ctypes.c_uint32(config_index.size),
            u32_ptr(config_rows),
            ctypes.c_uint32(config_rows.size),
            ctypes.c_uint64(reaction_token),
            ctypes.c_uint64(products_token),
            host_stages,
            unique_product_count,
            len(reactions),
            pair_count,
            self.threads_per_group,
            ctypes.byref(construct_seconds),
            profile_mode,
        )

        if ok != 1:
            raise RuntimeError(_metal_last_error(self._library))
        self.last_construct_seconds = construct_seconds.value
        # buffer setup, construct round trip, (the WL round trip, now always zero -- the keyer is
        # its own entry point), copy back
        self.last_host_stages = tuple(host_stages)

        # Overflow accounting: when an assignment splits into more than
        # MAX_COMPONENTS_PER_ASSIGNMENT components the kernel emits only the heaviest
        # ones (by heavy-atom count) and sets a per-assignment overflow flag. The
        # matcher rejects this assignment rather than using a partial precursor set.
        # The flag rides in `RetroStatus.overflow` for callers to inspect.
        status_words = np.frombuffer(
            output_status,
            dtype=np.uint32,
            count=pair_count * (ctypes.sizeof(RetroStatus) // 4),
        ).reshape(pair_count, -1)
        overflow = status_words[:, 1]
        atom_overflow = status_words[:, 2]
        edited_atom_overflow = status_words[:, 3]
        self.last_overflow_count = int(np.count_nonzero(overflow))
        self.last_atom_overflow_component_count = int(np.sum(atom_overflow))
        self.last_atom_overflow_count = int(np.count_nonzero(atom_overflow))
        self.last_edited_atom_overflow_count = int(
            np.count_nonzero(edited_atom_overflow)
        )

        # A refusal names a cause from the same closed set the CPU uses, so the two report
        # in one vocabulary and a sweep can compare them reason for reason.
        reasons = status_words[:, 5]
        refused = reasons != REJECTION_NONE
        self.last_rejected_applications = np.flatnonzero(refused).astype(np.int64)
        counts = list(no_rejections())
        for reason in reasons[refused].tolist():
            counts[int(reason)] += 1
        self.last_rejections = tuple(counts)
        self.last_undescribed_components = int(np.sum(status_words[:, 6]))
        # Columns 7 and 8 of `RetroStatus`. The shader has always filled them and the host has
        # always dropped them, so a Metal record could say an application was refused but not
        # which of the three refusal grounds fired: too many atoms, an orphan, or a grouping
        # disagreement. `undescribed_components` alone cannot answer it, being the superset that
        # also counts spectators -- the components a unimolecular forward rule leaves untouched
        # beside the one it edits, which are not refused at all.
        self.last_orphaned_components = int(np.sum(status_words[:, 7]))
        self.last_grouping_violations = int(np.sum(status_words[:, 8]))
        # The structural give-ups, counted but not yet named: see `RetroStatus`. Not warned
        # about, because the point of this counter is to find out whether it ever fires at all.
        self.last_inconsistent_assignments = tuple(
            int(np.count_nonzero(status_words[:, 9] == code)) for code in (1, 2, 3)
        )
        if self.last_overflow_count:
            warnings.warn(
                f"Metal retro: {self.last_overflow_count} of {pair_count} assignment(s) "
                f"split into more than {MAX_COMPONENTS_PER_ASSIGNMENT} components; kept "
                "the heaviest and flagged the assignment for rejection.",
                stacklevel=2,
            )
        if self.last_atom_overflow_count:
            warnings.warn(
                f"Metal retro: refused {self.last_atom_overflow_count} assignment(s) "
                f"whose precursors include {int(np.sum(atom_overflow))} component(s) "
                f"over {GPU_PRECURSOR_ATOM_CAP} atoms.",
                stacklevel=2,
            )
        if self.last_edited_atom_overflow_count:
            warnings.warn(
                f"Metal retro: dropped {self.last_edited_atom_overflow_count} "
                f"assignment(s) over {METAL_EDITED_ATOM_CAP} edited atoms.",
                stacklevel=2,
            )

        rows = int(produced.value)
        atoms = out_atoms[:rows]
        bonds = out_bonds[:rows]
        counts = out_counts[:rows]
        for view in (atoms, bonds, counts):
            view.setflags(write=False)
        return MetalRetroOutput(
            status=output_status, atoms=atoms, bonds=bonds, atom_counts=counts
        )


def graph_uses_aromaticity(graph: GMol) -> bool:
    # A default/omitted query bond (QBOND_DEFAULT) carries the aromatic bit only as
    # "single or aromatic"; it does not by itself require aromatic handling.
    aromatic_bond = (graph.bonds & np.uint8(BondKind.DELOCALIZED)) != 0
    real_aromatic_bond = aromatic_bond & (graph.bonds != np.uint8(QBOND_DEFAULT))
    return bool(
        np.any(graph.atom_masks.A == np.uint8(AROMATIC)) or np.any(real_aromatic_bond)
    )

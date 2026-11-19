from __future__ import annotations

import os
from array import array
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from threading import Lock
from time import perf_counter
from typing import Any, Literal
from weakref import finalize

import numpy as np

from .cmol import CMol
from .cmol_builder import gmol_from_cmol
from .cuda_pathfp import (
    compact_folded_prefilter_pairs_cuda,
    compute_folded_path_fingerprints_cuda,
)
from .cuda_prune import (
    CudaAssignmentResult,
    CudaOutputCapacityError,
    CudaPredicateProgram,
    CudaUnavailableError,
    _pack_collection_adjacency_rows,
    _pack_collection_atom_masks,
    _pack_collection_bond_matrices,
    _pair_indices,
    apply_cuda_predicates,
    match_fused_assignments_cuda,
    match_fused_cuda,
    pack_collection_atom_groups,
    prepare_cuda_predicates,
    prepare_cuda_prune_batch,
    prepare_cuda_prune_batch_from_device,
    prepare_cuda_prune_batch_from_device_pairs,
)
from .cuda_retro import (
    _ATOM_WORD_COUNT,
    _CUDA_BUCKET_SIZE,
    CudaMoleculeBatch,
    CudaReactionBatch,
    construct_retro_cuda,
    get_wl_hash_kernel,
    perceive_cuda_retro,
    prepare_cuda_molecule_batch,
    prepare_cuda_reaction_batch,
    wl_hash_kernel,
)
from .gmol import GMol
from .matcher import (
    HitArray,
    Matcher,
    MatchResult,
    MatchStatus,
    Pair,
    concat_hits,
    empty_hits,
)
from .pathfp import FoldedFingerprintConfig
from .reaction import ReactionPattern
from .rejection import RejectionCounts, no_rejections

__all__ = [
    "CudaFrontierMode",
    "CudaFoldedPrefilteredMatcher",
    "CUDA_HOST_FRONTIER_MAX_PAIRS_PER_LAUNCH",
    "CUDA_FRONTIER_MAX_PAIRS_PER_LAUNCH",
    "CudaMatchDiagnostics",
    "CudaMatcher",
    "CudaMatchOnlyPrefilterDiagnostics",
    "CudaReactionPrefilterDiagnostics",
    "CudaMoleculeDeviceLogicalBatch",
    "CudaMoleculeHostBatch",
    "CudaFoldedPrefilteredReactionMatcher",
    "CudaReactionMatcher",
    "CudaUnavailableError",
    "cuda_short_ring_wl_key",
    "cuda_concat_mols",
    "cuda_concat_host_mols",
    "cuda_stack_mols",
]

_STATUS_NAMES: dict[int, MatchStatus] = {
    0: "no_match",
    1: "match",
    2: "unknown",
}


class CudaMatcher(Matcher):
    def __init__(
        self,
        *,
        threads_per_block: int = 32,
        max_prune_rounds: int = 64,
        max_leaf_checks: int = 4096,
        max_prefix_depth: int = 16,
    ) -> None:
        self.threads_per_block = threads_per_block
        self.max_prune_rounds = max_prune_rounds
        self.max_leaf_checks = max_leaf_checks
        self.max_prefix_depth = max_prefix_depth

    def has_match(self, pattern: GMol, molecule: GMol) -> bool:
        batch = prepare_cuda_prune_batch(
            [pattern],
            [molecule],
            max_prune_rounds=self.max_prune_rounds,
            max_leaf_checks=self.max_leaf_checks,
            max_prefix_depth=self.max_prefix_depth,
        )
        result = match_fused_cuda(
            batch,
            threads_per_block=self.threads_per_block,
            max_prune_rounds=self.max_prune_rounds,
            max_leaf_checks=self.max_leaf_checks,
            max_prefix_depth=self.max_prefix_depth,
        )
        return int(result.status_codes_host()[0, 0]) == 1

    def find_matches(
        self,
        pattern: GMol,
        molecule: GMol,
        *,
        max_assignments: int | None = None,
        **kwargs: Any,
    ) -> MatchResult:
        del kwargs
        batch = prepare_cuda_prune_batch(
            [pattern],
            [molecule],
            max_prune_rounds=self.max_prune_rounds,
            max_leaf_checks=self.max_leaf_checks,
            max_prefix_depth=self.max_prefix_depth,
        )
        result = match_fused_assignments_cuda(
            batch,
            threads_per_block=self.threads_per_block,
            max_prune_rounds=self.max_prune_rounds,
            max_leaf_checks=self.max_leaf_checks,
            max_prefix_depth=self.max_prefix_depth,
        )
        status_code = int(result.status_codes_host()[0, 0])
        count = int(result.assignment_counts_host()[0, 0])
        returned_count = (
            count if max_assignments is None else min(count, max_assignments)
        )
        assignments = result.assignments_host()[
            :returned_count, : pattern.atom_count
        ].copy()
        return MatchResult(
            status=_STATUS_NAMES[status_code],
            assignments=assignments,
            prunings=int(result.prune_rounds_host()[0, 0]),
            pruning_complete=bool(result.pruning_complete_host()[0, 0]),
            search_complete=bool(result.search_complete_host()[0, 0])
            and returned_count == count,
            leaves_explored=int(result.leaf_checks_host()[0, 0]),
        )

    def has_match_many(
        self,
        patterns: list[GMol],
        molecules: list[GMol],
        **kwargs: Any,
    ) -> np.ndarray:
        del kwargs
        if not patterns or not molecules:
            return np.zeros((len(patterns), len(molecules)), dtype=bool)
        batch = prepare_cuda_prune_batch(
            patterns,
            molecules,
            max_prune_rounds=self.max_prune_rounds,
            max_leaf_checks=self.max_leaf_checks,
            max_prefix_depth=self.max_prefix_depth,
        )
        result = match_fused_cuda(
            batch,
            threads_per_block=self.threads_per_block,
            max_prune_rounds=self.max_prune_rounds,
            max_leaf_checks=self.max_leaf_checks,
            max_prefix_depth=self.max_prefix_depth,
        )
        return result.status_codes_host() == 1

    def assignment_batch_device(
        self,
        patterns: list[GMol],
        molecules: list[GMol],
    ) -> CudaAssignmentResult:
        batch = prepare_cuda_prune_batch(
            patterns,
            molecules,
            max_prune_rounds=self.max_prune_rounds,
            max_leaf_checks=self.max_leaf_checks,
            max_prefix_depth=self.max_prefix_depth,
        )
        return match_fused_assignments_cuda(
            batch,
            threads_per_block=self.threads_per_block,
            max_prune_rounds=self.max_prune_rounds,
            max_leaf_checks=self.max_leaf_checks,
            max_prefix_depth=self.max_prefix_depth,
        )


@dataclass(frozen=True)
class CudaMatchOnlyPrefilterDiagnostics:
    dense_pair_count: int
    candidate_pair_count: int
    pruned_pair_count: int
    prune_ratio: float
    fingerprint_seconds: float
    inner_match_seconds: float


@dataclass(frozen=True)
class CudaReactionPrefilterDiagnostics:
    dense_pair_count: int
    candidate_pair_count: int
    pruned_pair_count: int
    prune_ratio: float
    prefilter_seconds: float


class CudaFoldedPrefilteredMatcher(Matcher):
    def __init__(
        self,
        matcher: CudaMatcher | None = None,
        *,
        fingerprint_config: FoldedFingerprintConfig = FoldedFingerprintConfig(),
    ) -> None:
        self.matcher = CudaMatcher() if matcher is None else matcher
        self.fingerprint_config = fingerprint_config
        self.last_diagnostics: CudaMatchOnlyPrefilterDiagnostics | None = None

    def has_match(self, pattern: GMol, molecule: GMol) -> bool:
        return bool(self.has_match_many([pattern], [molecule])[0, 0])

    def find_matches(
        self,
        pattern: GMol,
        molecule: GMol,
        *,
        max_assignments: int | None = None,
        **kwargs: Any,
    ) -> MatchResult:
        return self.matcher.find_matches(
            pattern,
            molecule,
            max_assignments=max_assignments,
            **kwargs,
        )

    def has_match_many(
        self,
        patterns: list[GMol],
        molecules: list[GMol],
        **kwargs: Any,
    ) -> np.ndarray:
        del kwargs
        if not patterns or not molecules:
            self.last_diagnostics = CudaMatchOnlyPrefilterDiagnostics(
                dense_pair_count=len(patterns) * len(molecules),
                candidate_pair_count=0,
                pruned_pair_count=len(patterns) * len(molecules),
                prune_ratio=1.0 if patterns or molecules else 0.0,
                fingerprint_seconds=0.0,
                inner_match_seconds=0.0,
            )
            return np.zeros((len(patterns), len(molecules)), dtype=bool)

        cp = import_cupy()
        pattern_batch = cuda_stack_gmols(patterns)
        molecule_batch = cuda_stack_gmols(molecules)

        started = perf_counter()
        pattern_fps = compute_folded_path_fingerprints_cuda(
            pattern_batch,
            query=True,
            config=self.fingerprint_config,
        )
        molecule_fps = compute_folded_path_fingerprints_cuda(
            molecule_batch,
            query=False,
            config=self.fingerprint_config,
        )
        prefiltered = compact_folded_prefilter_pairs_cuda(
            molecule_fingerprints_device=molecule_fps,
            pattern_fingerprints_device=pattern_fps,
            molecule_atom_counts_device=molecule_batch.atom_counts_device,
            pattern_atom_counts_device=pattern_batch.atom_counts_device,
        )
        cp.cuda.Stream.null.synchronize()
        fingerprint_seconds = perf_counter() - started

        out = np.zeros((len(patterns), len(molecules)), dtype=bool)
        if prefiltered.pair_count == 0:
            self.last_diagnostics = CudaMatchOnlyPrefilterDiagnostics(
                dense_pair_count=prefiltered.dense_pair_count,
                candidate_pair_count=0,
                pruned_pair_count=prefiltered.dense_pair_count,
                prune_ratio=(0.0 if prefiltered.dense_pair_count == 0 else 1.0),
                fingerprint_seconds=fingerprint_seconds,
                inner_match_seconds=0.0,
            )
            return out

        started = perf_counter()
        batch = prepare_cuda_prune_batch_from_device_pairs(
            len(patterns),
            pattern_batch.atom_counts_device,
            pattern_batch.atoms_device,
            pattern_batch.rows_device,
            pattern_batch.bonds_device,
            pattern_groups_u32_device=cp.asarray(
                pack_collection_atom_groups(gmols=list(patterns)), dtype=cp.uint32
            ),
            molecule_atom_counts_device=molecule_batch.atom_counts_device,
            molecule_atoms_u32=molecule_batch.atoms_device,
            molecule_rows_u64=molecule_batch.rows_device,
            molecule_bonds_u8=molecule_batch.bonds_device,
            pair_pattern_index_device=prefiltered.pair_pattern_index_device,
            pair_molecule_index_device=prefiltered.pair_molecule_index_device,
            pair_count=prefiltered.pair_count,
        )
        batch = apply_cuda_predicates(
            batch,
            prepare_cuda_predicates(list(patterns)),
            max_prune_rounds=self.matcher.max_prune_rounds,
            max_leaf_checks=self.matcher.max_leaf_checks,
            max_prefix_depth=self.matcher.max_prefix_depth,
        )
        result = match_fused_cuda(
            batch,
            threads_per_block=self.matcher.threads_per_block,
            max_prune_rounds=self.matcher.max_prune_rounds,
            max_leaf_checks=self.matcher.max_leaf_checks,
            max_prefix_depth=self.matcher.max_prefix_depth,
        )
        cp.cuda.Stream.null.synchronize()
        inner_match_seconds = perf_counter() - started

        statuses = host_array(result.status_codes_device).astype(np.uint8, copy=False)
        pattern_indices = host_array(
            prefiltered.pair_pattern_index_device,
        ).astype(np.int64, copy=False)
        molecule_indices = host_array(
            prefiltered.pair_molecule_index_device,
        ).astype(np.int64, copy=False)
        out[pattern_indices, molecule_indices] = statuses == 1

        pruned_pair_count = prefiltered.dense_pair_count - prefiltered.pair_count
        self.last_diagnostics = CudaMatchOnlyPrefilterDiagnostics(
            dense_pair_count=prefiltered.dense_pair_count,
            candidate_pair_count=prefiltered.pair_count,
            pruned_pair_count=pruned_pair_count,
            prune_ratio=(
                0.0
                if prefiltered.dense_pair_count == 0
                else pruned_pair_count / prefiltered.dense_pair_count
            ),
            fingerprint_seconds=fingerprint_seconds,
            inner_match_seconds=inner_match_seconds,
        )
        return out


def cuda_stack_gmols(gmols: Sequence[GMol]) -> CudaMoleculeBatch:
    gmol_list = list(gmols)
    if not gmol_list:
        return _empty_cuda_molecule_batch()
    return prepare_cuda_molecule_batch(
        [((index, 0), gmol) for index, gmol in enumerate(gmol_list)]
    )


# --------------------------------------------------------------------------- #
# Reaction-matcher layer: the fused CUDA retro pipeline (match + bond edits).
# --------------------------------------------------------------------------- #
CUDA_RUNTIME_RESERVE_BYTES = 256 * 1024**2
# Bound leaf outputs so capacity retries split work instead of claiming all VRAM.
CUDA_INITIAL_ASSIGNMENT_CAPACITY = 65_536
CUDA_MAX_ASSIGNMENTS_PER_RESULT = 1_000_000
CUDA_INITIAL_PRECURSOR_CAPACITY = 1_024
CUDA_MAX_PRECURSORS_PER_RESULT = 250_000
# Expected precursor components produced per matched assignment. Used to right-size
# the initial retro output buffer from the known assignment count so a launch does
# not allocate a tiny buffer, overflow, and re-run retro (wasteful under a large
# resident device frontier). ~1.3 observed; 2.0 leaves margin to avoid retries.
CUDA_PRECURSORS_PER_ASSIGNMENT = 2.0
# Keep per-launch scratch bounded even while the full frontier remains on device.
CUDA_FRONTIER_MAX_PAIRS_PER_LAUNCH = 10_000
# Retain the previously exported host-specific name for callers importing it.
CUDA_HOST_FRONTIER_MAX_PAIRS_PER_LAUNCH = CUDA_FRONTIER_MAX_PAIRS_PER_LAUNCH
# Frontier rows preallocated per positive pair. The device frontier stores the
# *deduplicated* unique precursors (WL-keyed), not the raw retro output, so this
# estimates uniques-per-pair (~1-2 in practice) rather than raw precursors-per-pair
# (~3-4). Over-estimating preallocates a needlessly large resident frontier, leaving
# less room for per-launch scratch; under-estimating raises a clean
# CudaOutputCapacityError (the "device mode needs more VRAM for this payload" case).
CUDA_DEVICE_FRONTIER_ROWS_PER_POSITIVE_PAIR = 2.0
CUDA_DEVICE_FRONTIER_DENSE_PAIR_ASSIGNMENT_FACTOR = 1.2
# Multiplier over one launch's peak match+retro output buffers when reserving
# device memory for scratch. We keep enough free that the resident frontier never
# crowds out the per-launch retro/assignment buffers (which would otherwise trigger
# capacity-overflow splits); the headroom covers the transient gather, the dedup
# input batch, and pool fragmentation. If the frontier cannot fit alongside this
# reserve, device mode fails cleanly and the caller should use host mode instead.
CUDA_DEVICE_FRONTIER_SCRATCH_HEADROOM = 2.0
type CudaFrontierMode = Literal["device", "host"]
CUDA_DIAGNOSTIC_THREADS = 256
CUDA_DIAGNOSTIC_MAX_BLOCKS = 384
CUDA_DIAGNOSTIC_KERNEL: Any | None = None
CUDA_DIAGNOSTIC_SOURCE = r"""
extern "C" __global__ void reduce_match_diagnostics(
    const unsigned char* status,
    const int* prune_rounds,
    const int* assignment_counts,
    const unsigned char* search_complete,
    const int* leaf_checks,
    const unsigned long long* candidate_assignments,
    const unsigned char* candidate_overflow,
    const int* branches_explored,
    unsigned long long count,
    unsigned long long* output
) {
    unsigned long long status_no = 0;
    unsigned long long status_yes = 0;
    unsigned long long status_maybe = 0;
    unsigned long long incomplete = 0;
    unsigned long long prune_total = 0;
    unsigned long long prune_max = 0;
    unsigned long long assignment_total = 0;
    unsigned long long assignment_max = 0;
    unsigned long long leaf_total = 0;
    unsigned long long candidate_total = 0;
    unsigned long long candidate_max = 0;
    unsigned long long overflow_total = 0;
    unsigned long long branches_total = 0;
    unsigned long long branches_max = 0;

    const unsigned long long start =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride =
        (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long index = start; index < count; index += stride) {
        const unsigned char code = status[index];
        status_no += code == 0;
        status_yes += code == 1;
        status_maybe += code == 2;
        incomplete += search_complete[index] == 0;
        const unsigned long long prune = (unsigned long long)prune_rounds[index];
        prune_total += prune;
        prune_max = max(prune_max, prune);
        const unsigned long long assignments =
            (unsigned long long)assignment_counts[index];
        assignment_total += assignments;
        assignment_max = max(assignment_max, assignments);
        leaf_total += (unsigned long long)leaf_checks[index];
        const unsigned long long candidates = candidate_assignments[index];
        candidate_total += candidates;
        candidate_max = max(candidate_max, candidates);
        overflow_total += candidate_overflow[index] != 0;
        const unsigned long long branches = (unsigned long long)branches_explored[index];
        branches_total += branches;
        branches_max = max(branches_max, branches);
    }

    #define WARP_SUM(value) \
        for (int offset = 16; offset > 0; offset >>= 1) \
            value += __shfl_down_sync(0xffffffffu, value, offset)
    #define WARP_MAX(value) \
        for (int offset = 16; offset > 0; offset >>= 1) \
            value = max(value, __shfl_down_sync(0xffffffffu, value, offset))

    WARP_SUM(status_no);
    WARP_SUM(status_yes);
    WARP_SUM(status_maybe);
    WARP_SUM(incomplete);
    WARP_SUM(prune_total);
    WARP_MAX(prune_max);
    WARP_SUM(assignment_total);
    WARP_MAX(assignment_max);
    WARP_SUM(leaf_total);
    WARP_SUM(candidate_total);
    WARP_MAX(candidate_max);
    WARP_SUM(overflow_total);
    WARP_SUM(branches_total);
    WARP_MAX(branches_max);

    if ((threadIdx.x & 31) == 0) {
        atomicAdd(output + 1, status_no);
        atomicAdd(output + 2, status_yes);
        atomicAdd(output + 3, status_maybe);
        atomicAdd(output + 4, incomplete);
        atomicAdd(output + 5, prune_total);
        atomicMax(output + 6, prune_max);
        atomicAdd(output + 7, assignment_total);
        atomicMax(output + 8, assignment_max);
        atomicAdd(output + 9, leaf_total);
        atomicAdd(output + 10, candidate_total);
        atomicMax(output + 11, candidate_max);
        atomicAdd(output + 12, overflow_total);
        atomicAdd(output + 13, branches_total);
        atomicMax(output + 14, branches_max);
    }
}
"""


def host_array(array: Any) -> np.ndarray:
    if hasattr(array, "get"):
        return array.get()
    return np.asarray(array)


def import_cupy() -> Any:
    import cupy as cp  # ty: ignore[unresolved-import]  # optional 'cuda' extra

    return cp


def import_cupyx() -> Any:
    import cupyx  # ty: ignore[unresolved-import]  # optional 'cuda' extra

    return cupyx


PINNED_HOST_LIMIT_BYTES = 1 << 30
PINNED_HOST_LOCK = Lock()
PINNED_HOST_BYTES = 0
PINNED_HOST_ALLOCATED_BYTES = 0
PINNED_HOST_FREE: dict[int, list[Any]] = {}


def release_pinned_host_memory(memory: Any, size: int) -> None:
    global PINNED_HOST_BYTES
    with PINNED_HOST_LOCK:
        PINNED_HOST_BYTES -= size
        PINNED_HOST_FREE.setdefault(size, []).append(memory)


def acquire_pinned_host_memory(size: int) -> tuple[Any, int] | None:
    global PINNED_HOST_BYTES, PINNED_HOST_ALLOCATED_BYTES
    # Fixed size classes let differently sized batches reuse the same allocation.
    capacity = 1 << (size - 1).bit_length()
    with PINNED_HOST_LOCK:
        if PINNED_HOST_ALLOCATED_BYTES > PINNED_HOST_LIMIT_BYTES or (
            not PINNED_HOST_FREE.get(capacity)
            and PINNED_HOST_ALLOCATED_BYTES + capacity > PINNED_HOST_LIMIT_BYTES
        ):
            PINNED_HOST_FREE.clear()
            PINNED_HOST_ALLOCATED_BYTES = PINNED_HOST_BYTES
        if PINNED_HOST_BYTES + capacity > PINNED_HOST_LIMIT_BYTES:
            return None
        free = PINNED_HOST_FREE.get(capacity)
        if free:
            memory = free.pop()
        else:
            cp = import_cupy()
            memory = cp.cuda.PinnedMemoryPointer(cp.cuda.PinnedMemory(capacity), 0)
            PINNED_HOST_ALLOCATED_BYTES += capacity
        PINNED_HOST_BYTES += capacity
        return memory, capacity


def pinned_host_array(array: Any) -> np.ndarray:
    """Deliver independently owned host storage without a second host memcpy.

    Retained arrays and their views keep their pinned allocation alive. Limit all
    active and cached allocations to 1 GiB; further arrays use ordinary pageable
    delivery. Freed buffers are reused, and cached buffers are evicted as needed.
    """
    if not hasattr(array, "get"):
        return np.asarray(array)
    if not array.flags.c_contiguous or not array.nbytes:
        return array.get()
    allocation = acquire_pinned_host_memory(array.nbytes)
    if allocation is None:
        return array.get()
    memory, capacity = allocation
    try:
        result = np.ndarray(array.shape, dtype=array.dtype, buffer=memory)
    except BaseException:
        release_pinned_host_memory(memory, capacity)
        raise
    finalize(result, release_pinned_host_memory, memory, capacity)
    array.get(out=result)
    return result


def is_cuda_memory_error(exc: Exception) -> bool:
    cp = import_cupy()
    return isinstance(
        exc, (CudaOutputCapacityError, cp.cuda.memory.OutOfMemoryError)
    ) or (exc.__class__.__name__ == "CUDADriverError" and "OUT_OF_MEMORY" in str(exc))


def cuda_diagnostic_kernel(cp: Any) -> Any:
    global CUDA_DIAGNOSTIC_KERNEL
    if CUDA_DIAGNOSTIC_KERNEL is None:
        CUDA_DIAGNOSTIC_KERNEL = cp.RawKernel(
            CUDA_DIAGNOSTIC_SOURCE,
            "reduce_match_diagnostics",
            options=("-std=c++11",),
        )
    return CUDA_DIAGNOSTIC_KERNEL


@dataclass(frozen=True)
class CudaMatchDiagnostics:
    """Per-call match diagnostics, surfaced on `CudaReactionMatcher.last_diagnostics`.

    `status_maybe` / `incomplete_pairs` flag (mol, rule) pairs whose answer may be
    incomplete because leaf enumeration was truncated.
    """

    pair_count: int
    status_no: int
    status_yes: int
    status_maybe: int
    incomplete_pairs: int
    prune_rounds_total: int
    prune_rounds_max: int
    assignments_total: int
    assignments_max: int
    leaf_checks_total: int
    candidate_assignments_total: int
    candidate_assignments_max: int
    candidate_overflow_pairs: int
    branches_total: int
    branches_max: int
    pair_launches: int
    assignment_launches: int
    usable_device_bytes: int
    peak_estimated_workspace_bytes: int
    resident_rule_bytes: int
    assignment_capacity: int = 0
    assignment_required: int = 0
    assignment_overflow_retries: int = 0
    precursor_capacity: int = 0
    precursor_required: int = 0
    precursor_overflow_retries: int = 0
    precursor_atom_overflow_components: int = 0
    precursor_atom_overflow_assignments: int = 0
    edited_atom_overflow_assignments: int = 0
    rejections: RejectionCounts = no_rejections()
    undescribed_components: int = 0
    #: The subset of `undescribed_components` the edit severed, which is the counter that
    #: actually refuses an application. The kernel has always computed it -- `cuda_retro.cu`
    #: needs it to decide -- and then discarded it, so no CUDA record could say whether an
    #: application was refused for orphaning a component or merely counted one as undescribed.
    #: The superset includes spectators, which are not refused, so it cannot stand in for this.
    orphaned_components: int = 0
    #: Joining the per-part precursor batches into the one batch `match` returns. Zero when the
    #: caller pulls `match_batches` itself, because then there is no join to pay for.
    precursor_concat_seconds: float = 0.0
    #: Wall spent parked at a `yield` in `match_batches` while the consumer worked, excluded from
    #: `total_seconds` and recorded here instead. A generator's clock does not stop while it is
    #: suspended, so stamping a total as elapsed-since-entry bills the consumer's work to the
    #: matcher -- on Metal that showed up as 39.6 s of reported matcher time against stages
    #: summing to 28.8 s, the gap being the caller keying the batches it had just been handed.
    #: Zero for `match`, whose consumer only appends to a list. Added with the generator rather
    #: than after it, because the defect is invisible until something slow consumes the pieces.
    consumer_seconds: float = 0.0
    frontier_mode: str = "device"
    input_molecule_count: int = 0
    input_dense_bytes: int = 0
    output_dense_bytes: int = 0
    raw_precursor_count: int = 0
    unique_precursor_count: int = 0
    duplicate_precursor_count: int = 0
    atom_count_max: int = 0
    atom_count_histogram: tuple[int, ...] = ()
    host_to_device_bytes: int = 0
    device_to_host_bytes: int = 0
    host_to_device_seconds: float = 0.0
    device_to_host_seconds: float = 0.0
    match_seconds: float = 0.0
    retro_seconds: float = 0.0
    bond_edit_split_seconds: float = 0.0
    perceive_seconds: float = 0.0
    key_seconds: float = 0.0
    wl_kernel_seconds: float = 0.0
    hash_device_to_host_seconds: float = 0.0
    key_materialize_seconds: float = 0.0
    host_dedup_seconds: float = 0.0
    atom_count_device_to_host_seconds: float = 0.0
    retained_gather_seconds: float = 0.0
    retained_device_to_host_seconds: float = 0.0
    diagnostic_seconds: float = 0.0
    diagnostic_reduce_seconds: float = 0.0
    diagnostic_device_to_host_seconds: float = 0.0
    hit_device_seconds: float = 0.0
    hit_device_to_host_seconds: float = 0.0
    hit_materialize_seconds: float = 0.0
    hit_accumulate_seconds: float = 0.0
    atom_histogram_seconds: float = 0.0
    memory_snapshot_seconds: float = 0.0
    memory_release_seconds: float = 0.0
    pipeline_overhead_seconds: float = 0.0
    unclassified_seconds: float = 0.0
    host_store_seconds: float = 0.0
    materialize_seconds: float = 0.0
    peak_driver_used_bytes: int = 0
    peak_cupy_pool_used_bytes: int = 0
    peak_cupy_pool_total_bytes: int = 0
    peak_process_rss_bytes: int = 0
    retained_molecule_count: int = 0
    retained_dense_bytes: int = 0
    keys_generated_count: int = 0
    keys_transferred_count: int = 0
    atom_counts_transferred_count: int = 0
    molecules_transferred_to_host_count: int = 0
    molecules_retained_on_device_count: int = 0
    total_seconds: float = 0.0


_EMPTY_DIAGNOSTICS = CudaMatchDiagnostics(
    pair_count=0,
    status_no=0,
    status_yes=0,
    status_maybe=0,
    incomplete_pairs=0,
    prune_rounds_total=0,
    prune_rounds_max=0,
    assignments_total=0,
    assignments_max=0,
    leaf_checks_total=0,
    candidate_assignments_total=0,
    candidate_assignments_max=0,
    candidate_overflow_pairs=0,
    branches_total=0,
    branches_max=0,
    pair_launches=0,
    assignment_launches=0,
    usable_device_bytes=0,
    peak_estimated_workspace_bytes=0,
    resident_rule_bytes=0,
    assignment_capacity=0,
    assignment_required=0,
    assignment_overflow_retries=0,
    precursor_capacity=0,
    precursor_required=0,
    precursor_overflow_retries=0,
    precursor_atom_overflow_components=0,
    precursor_atom_overflow_assignments=0,
    edited_atom_overflow_assignments=0,
)


def _cuda_match_diagnostics(
    match_result,
    *,
    assignment_launches: int,
    usable_device_bytes: int,
    peak_estimated_workspace_bytes: int,
    resident_rule_bytes: int,
    precursor_capacity: int = 0,
    precursor_required: int = 0,
    precursor_overflow_retries: int = 0,
    precursor_atom_overflow_components: int = 0,
    precursor_atom_overflow_assignments: int = 0,
    edited_atom_overflow_assignments: int = 0,
    rejections: RejectionCounts = no_rejections(),
    undescribed_components: int = 0,
    orphaned_components: int = 0,
    match_seconds: float = 0.0,
    retro_seconds: float = 0.0,
    bond_edit_split_seconds: float = 0.0,
    perceive_seconds: float = 0.0,
    hit_device_seconds: float = 0.0,
    hit_device_to_host_seconds: float = 0.0,
    hit_materialize_seconds: float = 0.0,
    pipeline_overhead_seconds: float = 0.0,
) -> CudaMatchDiagnostics:
    cp = import_cupy()
    status = match_result.status_codes_device
    prune_rounds = match_result.prune_rounds_device
    assignment_counts = match_result.assignment_counts_device
    search_complete = match_result.search_complete_device
    leaf_checks = match_result.leaf_checks_device
    candidate_assignments = match_result.candidate_assignments_device
    candidate_overflow = match_result.candidate_overflow_device
    branches_explored = match_result.branches_explored_device
    reduce_started = perf_counter()
    values_device = cp.zeros(15, dtype=cp.uint64)
    values_device[0] = status.size
    block_count = min(
        max(
            (status.size + CUDA_DIAGNOSTIC_THREADS - 1) // CUDA_DIAGNOSTIC_THREADS,
            1,
        ),
        CUDA_DIAGNOSTIC_MAX_BLOCKS,
    )
    cuda_diagnostic_kernel(cp)(
        (block_count,),
        (CUDA_DIAGNOSTIC_THREADS,),
        (
            status,
            prune_rounds,
            assignment_counts,
            search_complete,
            leaf_checks,
            candidate_assignments,
            candidate_overflow,
            branches_explored,
            np.uint64(status.size),
            values_device,
        ),
    )
    cp.cuda.Stream.null.synchronize()
    diagnostic_reduce_seconds = perf_counter() - reduce_started
    copy_started = perf_counter()
    values = values_device.get()
    diagnostic_device_to_host_seconds = perf_counter() - copy_started
    diagnostic_seconds = diagnostic_reduce_seconds + diagnostic_device_to_host_seconds
    return CudaMatchDiagnostics(
        pair_count=int(values[0]),
        status_no=int(values[1]),
        status_yes=int(values[2]),
        status_maybe=int(values[3]),
        incomplete_pairs=int(values[4]),
        prune_rounds_total=int(values[5]),
        prune_rounds_max=int(values[6]),
        assignments_total=int(values[7]),
        assignments_max=int(values[8]),
        leaf_checks_total=int(values[9]),
        candidate_assignments_total=int(values[10]),
        candidate_assignments_max=int(values[11]),
        candidate_overflow_pairs=int(values[12]),
        branches_total=int(values[13]),
        branches_max=int(values[14]),
        pair_launches=1,
        assignment_launches=assignment_launches,
        usable_device_bytes=usable_device_bytes,
        peak_estimated_workspace_bytes=peak_estimated_workspace_bytes,
        resident_rule_bytes=resident_rule_bytes,
        assignment_capacity=match_result.assignment_capacity,
        assignment_required=match_result.assignment_required,
        assignment_overflow_retries=match_result.overflow_retries,
        precursor_capacity=precursor_capacity,
        precursor_required=precursor_required,
        precursor_overflow_retries=precursor_overflow_retries,
        precursor_atom_overflow_components=precursor_atom_overflow_components,
        precursor_atom_overflow_assignments=precursor_atom_overflow_assignments,
        edited_atom_overflow_assignments=edited_atom_overflow_assignments,
        rejections=rejections,
        undescribed_components=undescribed_components,
        orphaned_components=orphaned_components,
        match_seconds=match_seconds,
        retro_seconds=retro_seconds,
        bond_edit_split_seconds=bond_edit_split_seconds,
        perceive_seconds=perceive_seconds,
        diagnostic_seconds=diagnostic_seconds,
        diagnostic_reduce_seconds=diagnostic_reduce_seconds,
        diagnostic_device_to_host_seconds=diagnostic_device_to_host_seconds,
        hit_device_seconds=hit_device_seconds,
        hit_device_to_host_seconds=hit_device_to_host_seconds,
        hit_materialize_seconds=hit_materialize_seconds,
        pipeline_overhead_seconds=pipeline_overhead_seconds,
    )


def _empty_cuda_molecule_batch() -> CudaMoleculeBatch:
    cp = import_cupy()
    return CudaMoleculeBatch(
        tokens=(),
        atom_counts_device=cp.empty((0,), dtype=cp.int32),
        atoms_device=cp.empty(
            (0, _CUDA_BUCKET_SIZE, _ATOM_WORD_COUNT), dtype=cp.uint32
        ),
        bonds_device=cp.empty(
            (0, _CUDA_BUCKET_SIZE, _CUDA_BUCKET_SIZE), dtype=cp.uint8
        ),
        rows_device=cp.empty((0, _CUDA_BUCKET_SIZE), dtype=cp.uint64),
    )


def _cuda_molecule_storage(capacity: int) -> CudaMoleculeBatch:
    if capacity < 0:
        raise ValueError("capacity must be non-negative.")
    cp = import_cupy()
    return CudaMoleculeBatch(
        tokens=(),
        atom_counts_device=cp.empty((capacity,), dtype=cp.int32),
        atoms_device=cp.empty(
            (capacity, _CUDA_BUCKET_SIZE, _ATOM_WORD_COUNT), dtype=cp.uint32
        ),
        bonds_device=cp.empty(
            (capacity, _CUDA_BUCKET_SIZE, _CUDA_BUCKET_SIZE), dtype=cp.uint8
        ),
        rows_device=cp.empty((capacity, _CUDA_BUCKET_SIZE), dtype=cp.uint64),
    )


def _cuda_molecule_prefix_view(
    batch: CudaMoleculeBatch, count: int
) -> CudaMoleculeBatch:
    if count < 0 or count > batch.count:
        raise IndexError("prefix count is outside the CUDA molecule batch.")
    return CudaMoleculeBatch(
        tokens=(),
        atom_counts_device=batch.atom_counts_device[:count],
        atoms_device=batch.atoms_device[:count],
        bonds_device=batch.bonds_device[:count],
        rows_device=batch.rows_device[:count],
    )


def _scatter_cuda_molecule_rows(
    destination: CudaMoleculeBatch,
    start: int,
    source: CudaMoleculeBatch,
    host_indices: np.ndarray,
) -> int:
    """Gather ``host_indices`` rows of ``source`` directly into ``destination``
    at ``start`` and return how many rows were written.

    Unlike materializing ``source[host_indices]`` into a temporary batch (four
    gathered arrays live at once), this assigns one array at a time, so the peak
    transient is a single gathered field rather than a whole intermediate batch.
    """
    cp = import_cupy()
    count = int(host_indices.shape[0])
    stop = start + count
    if start < 0 or stop > destination.count:
        raise IndexError("destination slice is outside the CUDA molecule batch.")
    device_indices = cp.asarray(host_indices)
    target = slice(start, stop)
    destination.atom_counts_device[target] = source.atom_counts_device[device_indices]
    destination.atoms_device[target] = source.atoms_device[device_indices]
    destination.bonds_device[target] = source.bonds_device[device_indices]
    destination.rows_device[target] = source.rows_device[device_indices]
    return count


@dataclass(frozen=True)
class CudaMoleculeDeviceLogicalBatch:
    physical: CudaMoleculeBatch
    logical_indices: np.ndarray

    @property
    def physical_count(self) -> int:
        return len(self.physical)

    @property
    def count(self) -> int:
        return int(self.logical_indices.shape[0])

    def __len__(self) -> int:
        return self.count

    @property
    def nbytes(self) -> int:
        return cuda_molecule_batch_bytes(self.physical) + int(
            self.logical_indices.nbytes
        )

    def __getitem__(self, selector: Any) -> CudaMoleculeBatch:
        indices = batch_indices(self.count, selector)
        if not len(indices):
            return _empty_cuda_molecule_batch()
        return self.physical[self.logical_indices[indices]]


@dataclass(frozen=True)
class CudaMoleculeHostPart:
    keys: tuple[tuple[int, int], ...]
    atom_counts: np.ndarray
    atoms: np.ndarray
    bonds: np.ndarray
    rows: np.ndarray

    @property
    def count(self) -> int:
        return len(self.atom_counts)

    @property
    def nbytes(self) -> int:
        return int(
            self.atom_counts.nbytes
            + self.atoms.nbytes
            + self.bonds.nbytes
            + self.rows.nbytes
        )

    @classmethod
    def from_cuda(
        cls,
        batch: CudaMoleculeBatch,
        keys: Sequence[tuple[int, int]],
    ) -> CudaMoleculeHostPart:
        return cls(
            keys=tuple(keys),
            atom_counts=pinned_host_array(batch.atom_counts_device),
            atoms=pinned_host_array(batch.atoms_device),
            bonds=pinned_host_array(batch.bonds_device),
            rows=pinned_host_array(batch.rows_device),
        )

    def select(self, indices: np.ndarray) -> CudaMoleculeHostPart:
        return CudaMoleculeHostPart(
            keys=(
                tuple(self.keys[int(index)] for index in indices) if self.keys else ()
            ),
            atom_counts=self.atom_counts[indices],
            atoms=self.atoms[indices],
            bonds=self.bonds[indices],
            rows=self.rows[indices],
        )


@dataclass(frozen=True)
class CudaMoleculeHostBatch:
    parts: tuple[CudaMoleculeHostPart, ...]
    logical_indices: np.ndarray | None = None

    @property
    def physical_count(self) -> int:
        return sum(part.count for part in self.parts)

    @property
    def count(self) -> int:
        return (
            self.physical_count
            if self.logical_indices is None
            else len(self.logical_indices)
        )

    def __len__(self) -> int:
        return self.count

    @property
    def nbytes(self) -> int:
        return sum(part.nbytes for part in self.parts) + (
            0 if self.logical_indices is None else self.logical_indices.nbytes
        )

    def __getitem__(self, selector: Any) -> CudaMoleculeHostBatch:
        indices = batch_indices(self.count, selector)
        if not len(indices):
            return CudaMoleculeHostBatch(())
        physical_indices = (
            indices if self.logical_indices is None else self.logical_indices[indices]
        )
        return self.select_physical(physical_indices)

    def select_physical(self, indices: np.ndarray) -> CudaMoleculeHostBatch:
        offsets = np.cumsum(
            np.asarray([0, *(part.count for part in self.parts)], dtype=np.int64)
        )
        selected_parts: list[CudaMoleculeHostPart] = []
        run_part = -1
        run_rows: list[int] = []
        for index in indices:
            part_index = int(np.searchsorted(offsets, index, side="right") - 1)
            row_index = int(index - offsets[part_index])
            if part_index != run_part and run_rows:
                selected_parts.append(
                    self.parts[run_part].select(np.asarray(run_rows, dtype=np.int64))
                )
                run_rows = []
            run_part = part_index
            run_rows.append(row_index)
        if run_rows:
            selected_parts.append(
                self.parts[run_part].select(np.asarray(run_rows, dtype=np.int64))
            )
        return CudaMoleculeHostBatch(tuple(selected_parts))

    def to_cuda(self) -> CudaMoleculeBatch:
        if self.logical_indices is not None:
            return self.select_physical(self.logical_indices).to_cuda()
        if not self.parts:
            return _empty_cuda_molecule_batch()
        cp = import_cupy()
        keys = (
            tuple(key for part in self.parts for key in part.keys)
            if all(len(part.keys) == part.count for part in self.parts)
            else ()
        )
        if len(self.parts) == 1:
            part = self.parts[0]
            return CudaMoleculeBatch(
                tokens=keys,
                atom_counts_device=cp.asarray(part.atom_counts),
                atoms_device=cp.asarray(part.atoms),
                bonds_device=cp.asarray(part.bonds),
                rows_device=cp.asarray(part.rows),
            )
        return CudaMoleculeBatch(
            tokens=keys,
            atom_counts_device=cp.asarray(
                np.concatenate([part.atom_counts for part in self.parts])
            ),
            atoms_device=cp.asarray(
                np.concatenate([part.atoms for part in self.parts])
            ),
            bonds_device=cp.asarray(
                np.concatenate([part.bonds for part in self.parts])
            ),
            rows_device=cp.asarray(np.concatenate([part.rows for part in self.parts])),
        )


def batch_indices(count: int, selector: Any) -> np.ndarray:
    if isinstance(selector, (int, np.integer)):
        raise TypeError(
            "molecule batch selection must preserve the batch axis; "
            "use a slice or one-dimensional index array."
        )
    if isinstance(selector, slice):
        return np.arange(count, dtype=np.int64)[selector]

    raw = np.asarray(selector.get() if hasattr(selector, "get") else selector)
    if raw.ndim != 1:
        raise IndexError("molecule batch selector must be one-dimensional.")
    if raw.dtype == bool:
        if len(raw) != count:
            raise IndexError(f"boolean mask length {len(raw)} != batch count {count}")
        return np.nonzero(raw)[0].astype(np.int64)
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("molecule batch selector must contain booleans or integers.")
    indices = raw.astype(np.int64, copy=False)
    if np.any(indices < -count) or np.any(indices >= count):
        raise IndexError("molecule batch index is out of range.")
    return np.where(indices < 0, indices + count, indices)


def cuda_short_ring_wl_key(batch: CudaMoleculeBatch) -> list[tuple[int, int]]:
    """De-fused short-ring WL keyer: hash a precursor batch into WL tokens (lo, hi).

    The CUDA analogue of `short_ring_wl_keyer.encode`; tokens are bit-exact
    with the CPU codec (Phase 5 parity). This is the separate stage-5 keyer — the
    reaction matcher emits un-keyed precursor mols and this turns them into keys, so
    Python pipelines mirror the CPU/rdkit `match` + `key` split.
    """
    if batch.count == 0:
        return []
    hash_lo, hash_hi = wl_hash_kernel(batch)
    lo = host_array(hash_lo).astype(np.uint64).tolist()
    hi = host_array(hash_hi).astype(np.uint64).tolist()
    return [(int(a), int(b)) for a, b in zip(lo, hi, strict=True)]


def cuda_event_seconds(call: Callable[[], Any]) -> tuple[Any, float]:
    cp = import_cupy()
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()
    result = call()
    stop.record()
    stop.synchronize()
    return result, float(cp.cuda.get_elapsed_time(start, stop) / 1000.0)


def cuda_stack_mols(mols: Sequence[CMol]) -> CudaMoleculeBatch:
    """`stack_mols` for the CUDA backend: build a device-resident frontier batch from
    CMols. Tokens are placeholders — the matcher uses positional mol_idx and keys are
    tracked host-side by `search`."""
    mol_list = list(mols)
    if not mol_list:
        return _empty_cuda_molecule_batch()
    return prepare_cuda_molecule_batch(
        [((index, 0), gmol_from_cmol(mol)) for index, mol in enumerate(mol_list)]
    )


def cuda_concat_mols(parts: Sequence[CudaMoleculeBatch]) -> CudaMoleculeBatch:
    """Join device molecule batches while preserving an empty batch result.

    ``CudaMoleculeBatch.from_parts`` uses ``None`` to signal that every part was empty,
    which is convenient inside the reaction matcher but not for the search batch protocol.
    AO* needs concatenation to remain a molecule batch even when its frontier becomes empty.
    """
    combined = CudaMoleculeBatch.from_parts(parts)
    return combined if combined is not None else _empty_cuda_molecule_batch()


def cuda_concat_host_mols(
    parts: Sequence[
        CudaMoleculeBatch | CudaMoleculeDeviceLogicalBatch | CudaMoleculeHostBatch
    ],
) -> CudaMoleculeHostBatch:
    """Join search frontier parts in pinned host memory.

    Device inputs are copied once. Host logical views are materialized so subsequent gathers
    address a compact physical batch. This is the bounded-VRAM counterpart to
    :func:`cuda_concat_mols`: selected batches return to CUDA for matching, while the much larger
    set of open AO* leaves waits on the host.
    """
    host_parts: list[CudaMoleculeHostPart] = []
    for part in parts:
        if isinstance(part, CudaMoleculeHostBatch):
            materialized = (
                part
                if part.logical_indices is None
                else part.select_physical(part.logical_indices)
            )
            host_parts.extend(materialized.parts)
            continue
        device = (
            part[np.arange(len(part), dtype=np.int64)]
            if isinstance(part, CudaMoleculeDeviceLogicalBatch)
            else part
        )
        if len(device):
            host_parts.append(CudaMoleculeHostPart.from_cuda(device, device.tokens))
    return CudaMoleculeHostBatch(tuple(host_parts))


@dataclass(frozen=True)
class _CudaRuleChunk:
    """One rule group with its pattern arrays + reaction batch pre-compiled on device.

    `rule_ids[local_pattern_index]` maps a chunk-local pattern index back to the
    dense global rule index, so hits remain stable regardless of chunking.
    """

    rule_ids: tuple[int, ...]
    pattern_count: int
    pattern_atom_counts_device: Any
    pattern_atoms_u32_device: Any
    pattern_rows_u64_device: Any
    pattern_bonds_u8_device: Any
    pattern_groups_u32_device: Any
    pattern_fingerprints_device: Any | None
    reaction_batch: CudaReactionBatch
    predicate_program: CudaPredicateProgram | None = None


def _pack_rule_chunk(
    rule_ids: Sequence[int],
    patterns: Sequence[ReactionPattern],
    *,
    fingerprint_config: FoldedFingerprintConfig | None = None,
) -> _CudaRuleChunk:
    cp = import_cupy()
    lhs = [pattern.lhs for pattern in patterns]
    pattern_atom_counts_device = cp.asarray(
        np.asarray([gmol.atom_count for gmol in lhs], dtype=np.int32),
        dtype=cp.int32,
    )
    pattern_atoms_u32_device = cp.asarray(
        _pack_collection_atom_masks(lhs), dtype=cp.uint32
    )
    pattern_rows_u64_device = cp.asarray(
        _pack_collection_adjacency_rows(gmols=lhs), dtype=cp.uint64
    )
    pattern_bonds_u8_device = cp.asarray(
        _pack_collection_bond_matrices(gmols=lhs), dtype=cp.uint8
    )
    pattern_groups_u32_device = cp.asarray(
        pack_collection_atom_groups(gmols=lhs), dtype=cp.uint32
    )
    pattern_fingerprints_device = None
    if fingerprint_config is not None:
        pattern_batch = CudaMoleculeBatch(
            tokens=(),
            atom_counts_device=pattern_atom_counts_device,
            atoms_device=pattern_atoms_u32_device,
            bonds_device=pattern_bonds_u8_device,
            rows_device=pattern_rows_u64_device,
        )
        pattern_fingerprints_device = compute_folded_path_fingerprints_cuda(
            pattern_batch,
            query=True,
            config=fingerprint_config,
        )
    return _CudaRuleChunk(
        rule_ids=tuple(rule_ids),
        pattern_count=len(patterns),
        pattern_groups_u32_device=pattern_groups_u32_device,
        pattern_atom_counts_device=pattern_atom_counts_device,
        pattern_atoms_u32_device=pattern_atoms_u32_device,
        pattern_rows_u64_device=pattern_rows_u64_device,
        pattern_bonds_u8_device=pattern_bonds_u8_device,
        pattern_fingerprints_device=pattern_fingerprints_device,
        reaction_batch=prepare_cuda_reaction_batch(list(patterns)),
        predicate_program=prepare_cuda_predicates(lhs),
    )


def _aggregate_diagnostics(
    diagnostics: Sequence[CudaMatchDiagnostics],
) -> CudaMatchDiagnostics:
    if not diagnostics:
        return _EMPTY_DIAGNOSTICS
    return CudaMatchDiagnostics(
        pair_count=sum(d.pair_count for d in diagnostics),
        status_no=sum(d.status_no for d in diagnostics),
        status_yes=sum(d.status_yes for d in diagnostics),
        status_maybe=sum(d.status_maybe for d in diagnostics),
        incomplete_pairs=sum(d.incomplete_pairs for d in diagnostics),
        prune_rounds_total=sum(d.prune_rounds_total for d in diagnostics),
        prune_rounds_max=max(d.prune_rounds_max for d in diagnostics),
        assignments_total=sum(d.assignments_total for d in diagnostics),
        assignments_max=max(d.assignments_max for d in diagnostics),
        leaf_checks_total=sum(d.leaf_checks_total for d in diagnostics),
        candidate_assignments_total=sum(
            d.candidate_assignments_total for d in diagnostics
        ),
        candidate_assignments_max=max(d.candidate_assignments_max for d in diagnostics),
        candidate_overflow_pairs=sum(d.candidate_overflow_pairs for d in diagnostics),
        branches_total=sum(d.branches_total for d in diagnostics),
        branches_max=max(d.branches_max for d in diagnostics),
        pair_launches=sum(d.pair_launches for d in diagnostics),
        assignment_launches=sum(d.assignment_launches for d in diagnostics),
        usable_device_bytes=min(d.usable_device_bytes for d in diagnostics),
        peak_estimated_workspace_bytes=max(
            d.peak_estimated_workspace_bytes for d in diagnostics
        ),
        resident_rule_bytes=max(d.resident_rule_bytes for d in diagnostics),
        assignment_capacity=sum(d.assignment_capacity for d in diagnostics),
        assignment_required=sum(d.assignment_required for d in diagnostics),
        assignment_overflow_retries=sum(
            d.assignment_overflow_retries for d in diagnostics
        ),
        precursor_capacity=sum(d.precursor_capacity for d in diagnostics),
        precursor_required=sum(d.precursor_required for d in diagnostics),
        precursor_overflow_retries=sum(
            d.precursor_overflow_retries for d in diagnostics
        ),
        precursor_atom_overflow_components=sum(
            d.precursor_atom_overflow_components for d in diagnostics
        ),
        precursor_atom_overflow_assignments=sum(
            d.precursor_atom_overflow_assignments for d in diagnostics
        ),
        edited_atom_overflow_assignments=sum(
            d.edited_atom_overflow_assignments for d in diagnostics
        ),
        rejections=tuple(
            sum(d.rejections[index] for d in diagnostics)
            for index in range(len(no_rejections()))
        ),
        undescribed_components=sum(d.undescribed_components for d in diagnostics),
        orphaned_components=sum(d.orphaned_components for d in diagnostics),
        match_seconds=sum(d.match_seconds for d in diagnostics),
        retro_seconds=sum(d.retro_seconds for d in diagnostics),
        bond_edit_split_seconds=sum(d.bond_edit_split_seconds for d in diagnostics),
        perceive_seconds=sum(d.perceive_seconds for d in diagnostics),
        diagnostic_seconds=sum(d.diagnostic_seconds for d in diagnostics),
        diagnostic_reduce_seconds=sum(d.diagnostic_reduce_seconds for d in diagnostics),
        diagnostic_device_to_host_seconds=sum(
            d.diagnostic_device_to_host_seconds for d in diagnostics
        ),
        hit_device_seconds=sum(d.hit_device_seconds for d in diagnostics),
        hit_device_to_host_seconds=sum(
            d.hit_device_to_host_seconds for d in diagnostics
        ),
        hit_materialize_seconds=sum(d.hit_materialize_seconds for d in diagnostics),
        pipeline_overhead_seconds=sum(d.pipeline_overhead_seconds for d in diagnostics),
    )


def cuda_molecule_batch_bytes(batch: CudaMoleculeBatch) -> int:
    return int(
        batch.atom_counts_device.nbytes
        + batch.atoms_device.nbytes
        + batch.bonds_device.nbytes
        + batch.rows_device.nbytes
    )


def cuda_molecule_row_bytes() -> int:
    atom_count_bytes = np.dtype(np.int32).itemsize
    atom_bytes = _CUDA_BUCKET_SIZE * _ATOM_WORD_COUNT * np.dtype(np.uint32).itemsize
    bond_bytes = _CUDA_BUCKET_SIZE**2 * np.dtype(np.uint8).itemsize
    row_bytes = _CUDA_BUCKET_SIZE * np.dtype(np.uint64).itemsize
    return atom_count_bytes + atom_bytes + bond_bytes + row_bytes


def raise_cuda_device_frontier_capacity(required: int, capacity: int) -> None:
    row_bytes = cuda_molecule_row_bytes()
    required_gib = required * row_bytes / 1024**3
    capacity_gib = capacity * row_bytes / 1024**3
    raise CudaOutputCapacityError(
        f"device frontier ({required_gib:.2f} GiB dense, "
        f"{capacity_gib:.2f} GiB allocated)",
        required,
        capacity,
    )


def cuda_memory_snapshot() -> tuple[int, int, int]:
    cp = import_cupy()
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    return (
        int(total_bytes - free_bytes),
        int(pool.used_bytes()),
        int(pool.total_bytes()),
    )


def process_rss_bytes() -> int:
    with open("/proc/self/statm", encoding="utf-8") as handle:
        fields = handle.read().split()
    if len(fields) < 2:
        raise ValueError("/proc/self/statm must include resident page count.")
    return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")


def device_array_bytes(value: Any) -> int:
    return int(getattr(value, "nbytes", 0))


def rule_chunk_bytes(chunk: _CudaRuleChunk) -> int:
    predicates = chunk.predicate_program
    predicate_bytes = (
        0
        if predicates is None
        else device_array_bytes(predicates.pattern_predicates)
        + sum(device_array_bytes(value) for value in predicates.device.values())
    )
    return (
        predicate_bytes
        + sum(
            device_array_bytes(value)
            for value in vars(chunk).values()
            if hasattr(value, "nbytes")
        )
        + sum(
            device_array_bytes(value)
            for value in vars(chunk.reaction_batch).values()
            if hasattr(value, "nbytes")
        )
    )


def match_pair_workspace_bytes() -> int:
    expected_assignment = (
        _CUDA_BUCKET_SIZE * np.dtype(np.int32).itemsize + np.dtype(np.int32).itemsize
    )
    pair_indices = 2 * np.dtype(np.int32).itemsize
    diagnostics = (
        3 * np.dtype(np.int32).itemsize
        + 4 * np.dtype(np.uint8).itemsize
        + np.dtype(np.uint64).itemsize
    )
    return expected_assignment + pair_indices + diagnostics


def retro_precursor_workspace_bytes() -> int:
    atom_bytes = _CUDA_BUCKET_SIZE * _ATOM_WORD_COUNT * np.dtype(np.uint32).itemsize
    bond_bytes = _CUDA_BUCKET_SIZE**2 * np.dtype(np.uint8).itemsize
    row_bytes = _CUDA_BUCKET_SIZE * np.dtype(np.uint64).itemsize
    hydrogen_bytes = 2 * _CUDA_BUCKET_SIZE * np.dtype(np.int32).itemsize
    component_bytes = (
        np.dtype(np.int32).itemsize
        + atom_bytes
        + bond_bytes
        + row_bytes
        + hydrogen_bytes
        + 2 * np.dtype(np.uint64).itemsize
    )
    return component_bytes + np.dtype(np.int32).itemsize


def cuda_device_scratch_reserve_bytes() -> int:
    """Device memory to keep free for one match+retro launch's scratch.

    Sized to the per-result capacity caps (the largest a single launch's
    assignment and precursor output buffers can grow to) times a headroom factor
    for the transient gather, dedup input batch, and pool fragmentation. The
    device frontier is preallocated only out of what remains *after* this reserve,
    so a resident frontier never starves the scratch that makes device mode fast.
    """
    assignment_bytes = CUDA_MAX_ASSIGNMENTS_PER_RESULT * match_pair_workspace_bytes()
    precursor_bytes = CUDA_MAX_PRECURSORS_PER_RESULT * retro_precursor_workspace_bytes()
    return int(
        (assignment_bytes + precursor_bytes) * CUDA_DEVICE_FRONTIER_SCRATCH_HEADROOM
    )


class CudaReactionMatcher:
    """Reaction matcher backed by the CUDA retro pipeline.

    Conforms to the `vesma.matcher.ReactionMatcher` protocol:
    `match(mols, pairs=None)`. `match` runs the fused Ullmann-refinement +
    injectivity kernel, then bond edits + connected-component split, and returns
    `(precursors, hits)` where `precursors` is a device-resident `CudaMoleculeBatch`
    and `hits` are flat `(mol_idx, rule_id, app_idx, precursor_idx)` rows indexing it.

    There is **no dedup and no keying** here: this backend emits one precursor per hit
    (the precursor column is the identity), identical precursors collapse later in
    sid space (the store's step grouping + `expanded` filter), and WL tokens are produced by the
    separate `cuda_short_ring_wl_key` keyer. `pairs=None` means all mols x all rules;
    an explicit `(mol_idx, rule_id)` pair set restricts the work (selective match) by
    feeding a smaller pair-index array to the same kernels.
    """

    def __init__(
        self,
        rules: Sequence[ReactionPattern],
        *,
        frontier_mode: CudaFrontierMode = "device",
        max_leaf_checks: int = 4096,
        max_prune_rounds: int = 64,
        max_prefix_depth: int = 16,
        threads_per_block: int = 32,
        max_device_memory_gb: float | None = None,
        max_pairs_per_launch: int | None = None,
        fingerprint_config: FoldedFingerprintConfig | None = None,
    ) -> None:
        if frontier_mode not in {"device", "host"}:
            raise ValueError("frontier_mode must be 'device' or 'host'.")
        if max_device_memory_gb is not None and max_device_memory_gb <= 0:
            raise ValueError("max_device_memory_gb must be positive when provided.")
        if max_pairs_per_launch is not None and max_pairs_per_launch <= 0:
            raise ValueError("max_pairs_per_launch must be positive when provided.")
        self._frontier_mode = frontier_mode
        self._max_leaf_checks = max_leaf_checks
        self._max_prune_rounds = max_prune_rounds
        self._max_prefix_depth = max_prefix_depth
        self._threads_per_block = threads_per_block
        self._fingerprint_config = fingerprint_config
        self._max_device_memory_bytes = (
            None
            if max_device_memory_gb is None
            else int(max_device_memory_gb * 1024**3)
        )
        self._max_pairs_per_launch = (
            CUDA_FRONTIER_MAX_PAIRS_PER_LAUNCH
            if max_pairs_per_launch is None
            else max_pairs_per_launch
        )
        self._last_diagnostics: CudaMatchDiagnostics | None = None
        self._last_prefilter_diagnostics: CudaReactionPrefilterDiagnostics | None = None
        patterns = list(rules)
        self._patterns = patterns
        rule_ids = list(range(len(patterns)))
        cp = import_cupy()
        cuda_diagnostic_kernel(cp).compile()
        get_wl_hash_kernel(cp).compile()

        initial_pair_capacity = self._pair_capacity(resident_rule_bytes=0)
        rule_chunk_size = min(16_384, initial_pair_capacity)
        self._rule_chunks = [
            _pack_rule_chunk(
                rule_ids[start : start + rule_chunk_size],
                patterns[start : start + rule_chunk_size],
                fingerprint_config=fingerprint_config,
            )
            for start in range(0, len(patterns), rule_chunk_size)
        ]
        self._resident_rule_bytes = sum(
            rule_chunk_bytes(chunk) for chunk in self._rule_chunks
        )
        if (
            self._max_device_memory_bytes is not None
            and self._resident_rule_bytes >= self._max_device_memory_bytes
        ):
            raise ValueError(
                "compiled CUDA rules exceed max_device_memory_gb: "
                f"{self._resident_rule_bytes / 1024**3:.3f} GiB >= "
                f"{self._max_device_memory_bytes / 1024**3:.3f} GiB"
            )
        self._pair_capacity(resident_rule_bytes=self._resident_rule_bytes)
        self._chunk_by_rule = {
            rule_id: (chunk, local_index)
            for chunk in self._rule_chunks
            for local_index, rule_id in enumerate(chunk.rule_ids)
        }

    def _usable_device_bytes(
        self,
        *,
        resident_rule_bytes: int,
        active_workspace_bytes: int = 0,
    ) -> int:
        cp = import_cupy()
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
        pool = cp.get_default_memory_pool()
        reusable_pool_bytes = int(pool.total_bytes() - pool.used_bytes())
        free_workspace = max(
            int(free_bytes) + reusable_pool_bytes - CUDA_RUNTIME_RESERVE_BYTES,
            0,
        )
        if self._max_device_memory_bytes is None:
            return free_workspace
        configured_workspace = max(
            self._max_device_memory_bytes
            - resident_rule_bytes
            - active_workspace_bytes,
            0,
        )
        return min(free_workspace, configured_workspace)

    def _pair_capacity(self, *, resident_rule_bytes: int) -> int:
        usable = self._usable_device_bytes(resident_rule_bytes=resident_rule_bytes)
        bytes_per_pair = match_pair_workspace_bytes()
        capacity = usable // bytes_per_pair
        if self._max_pairs_per_launch is not None:
            capacity = min(capacity, self._max_pairs_per_launch)
        if capacity < 1:
            raise ValueError(
                "CUDA memory limit cannot fit one sparse match pair; "
                f"usable={usable} bytes, "
                f"required={bytes_per_pair} bytes."
            )
        return int(capacity)

    def _device_frontier_capacity(
        self,
        *,
        input_molecule_count: int,
        pair_count: int | None,
    ) -> int:
        estimated_rows = (
            int(np.ceil(pair_count * CUDA_DEVICE_FRONTIER_ROWS_PER_POSITIVE_PAIR))
            if pair_count is not None
            else int(
                np.ceil(
                    input_molecule_count
                    * len(self._patterns)
                    * CUDA_DEVICE_FRONTIER_DENSE_PAIR_ASSIGNMENT_FACTOR
                    * CUDA_DEVICE_FRONTIER_ROWS_PER_POSITIVE_PAIR
                )
            )
        )
        if estimated_rows <= 0:
            return 0

        usable = self._usable_device_bytes(
            resident_rule_bytes=self._resident_rule_bytes
        )
        allocable = max(usable - cuda_device_scratch_reserve_bytes(), 0)
        memory_capacity = allocable // cuda_molecule_row_bytes()
        if memory_capacity <= 0:
            raise_cuda_device_frontier_capacity(estimated_rows, 0)
        return int(min(estimated_rows, memory_capacity))

    def _ensure_folded_fingerprints(
        self,
        config: FoldedFingerprintConfig,
    ) -> None:
        if self._fingerprint_config == config and all(
            chunk.pattern_fingerprints_device is not None for chunk in self._rule_chunks
        ):
            return

        updated_chunks: list[_CudaRuleChunk] = []
        for chunk in self._rule_chunks:
            pattern_batch = CudaMoleculeBatch(
                tokens=(),
                atom_counts_device=chunk.pattern_atom_counts_device,
                atoms_device=chunk.pattern_atoms_u32_device,
                bonds_device=chunk.pattern_bonds_u8_device,
                rows_device=chunk.pattern_rows_u64_device,
            )
            updated_chunks.append(
                replace(
                    chunk,
                    pattern_fingerprints_device=compute_folded_path_fingerprints_cuda(
                        pattern_batch,
                        query=True,
                        config=config,
                    ),
                )
            )
        self._rule_chunks = updated_chunks
        self._fingerprint_config = config
        self._resident_rule_bytes = sum(
            rule_chunk_bytes(chunk) for chunk in self._rule_chunks
        )
        if (
            self._max_device_memory_bytes is not None
            and self._resident_rule_bytes >= self._max_device_memory_bytes
        ):
            raise ValueError(
                "compiled CUDA rules exceed max_device_memory_gb after folded "
                "fingerprints: "
                f"{self._resident_rule_bytes / 1024**3:.3f} GiB >= "
                f"{self._max_device_memory_bytes / 1024**3:.3f} GiB"
            )
        self._chunk_by_rule = {
            rule_id: (chunk, local_index)
            for chunk in self._rule_chunks
            for local_index, rule_id in enumerate(chunk.rule_ids)
        }

    @property
    def last_diagnostics(self) -> CudaMatchDiagnostics | None:
        return self._last_diagnostics

    @property
    def last_prefilter_diagnostics(self) -> CudaReactionPrefilterDiagnostics | None:
        return self._last_prefilter_diagnostics

    @property
    def max_pairs_per_launch(self) -> int | None:
        return self._max_pairs_per_launch

    @property
    def resident_rule_bytes(self) -> int:
        """Device memory retained by compiled patterns and their prefilter data."""
        return self._resident_rule_bytes

    def match(
        self,
        mols: Sequence[CMol] | CudaMoleculeBatch,
        pairs: Sequence[Pair] | None = None,
        fingerprint_config: FoldedFingerprintConfig | None = None,
    ) -> tuple[CudaMoleculeBatch, HitArray]:
        """Apply the pairs and return one joined precursor batch with its hits.

        A fold over `match_batches`, which is where the work is. It used to be a third copy of
        the chunking loop -- `_match_all`, `_match_selective` and `_match_all_folded_prefiltered`
        walked rule chunks and molecule slices themselves, in parallel with the generators
        `match_keyed` uses, and the two sized their molecule chunks by different arithmetic. Two
        loops over one pipeline is how the joined path and the batched path come to disagree
        about what they measured; worse, only the generators carry the out-of-memory retry that
        halves a part and tries again, so `match` was the one entry point that turned a
        recoverable allocation failure into a raised error. Folding it over the generator gives
        it that retry and deletes the second loop.

        Prefer `match_batches` where the result is about to be collapsed anyway:
        `matcher.keyed_match_folding` does that, and on this backend `match_keyed` has always
        done its own version internally.
        """
        parts: list[CudaMoleculeBatch] = []
        hit_parts: list[HitArray] = []
        for part, part_hits in self.match_batches(
            mols, pairs, fingerprint_config=fingerprint_config
        ):
            parts.append(part)
            hit_parts.append(part_hits)

        concat_started = perf_counter()
        joined = CudaMoleculeBatch.from_parts(parts)
        result = (
            joined if joined is not None else _empty_cuda_molecule_batch(),
            concat_hits(hit_parts, [part.count for part in parts]),
        )
        concat_seconds = perf_counter() - concat_started
        # The generator stamped every other phase and stopped its clock before this join, which
        # exists only for a caller that wants one array -- so it is counted here, and reads zero
        # for everyone who folds instead. Without this the join would vanish from the budget the
        # way it did on Metal, where it was the largest single line at full corpus.
        if self._last_diagnostics is not None:
            self._last_diagnostics = replace(
                self._last_diagnostics,
                precursor_concat_seconds=concat_seconds,
                total_seconds=self._last_diagnostics.total_seconds + concat_seconds,
            )
        return result

    def match_batches(
        self,
        mols: Sequence[CMol] | CudaMoleculeBatch | CudaMoleculeHostBatch,
        pairs: Sequence[Pair] | None = None,
        *,
        fingerprint_config: FoldedFingerprintConfig | None = None,
    ) -> Iterator[tuple[CudaMoleculeBatch, HitArray]]:
        """Yield each pipeline part's precursors as they land, per `matcher.BatchedMatch`.

        The generators this walks already satisfy the contract: they thread `app_base` across
        parts, so hit columns 0-2 are global and only column 3 is part-local. That is what lets
        `matcher.keyed_match_folding` collapse identity here the way `match_keyed` does it
        internally against `key_to_physical` -- the difference being that the fold is composed by
        the caller and works for any backend, where `key_to_physical` is this one's own.

        A part whose precursors are None contributed no molecules and is not yielded; its
        diagnostics are still counted, so an abandoned or empty call still reports what ran.

        Abandoning the iterator leaves the remaining parts un-run and `last_diagnostics`
        describing only the parts that were pulled. The diagnostics are stamped in a `finally`
        so that an abandoned call is reported rather than silently absent or, worse, left
        reading as the previous call's numbers.
        """
        total_started = perf_counter()
        input_batch: CudaMoleculeBatch | CudaMoleculeHostBatch
        if isinstance(mols, (CudaMoleculeBatch, CudaMoleculeHostBatch)):
            input_batch = mols
        else:
            input_batch = cuda_stack_mols(mols)

        if (
            not len(input_batch)
            or not self._rule_chunks
            or (pairs is not None and not pairs)
        ):
            self._last_diagnostics = replace(
                _EMPTY_DIAGNOSTICS,
                frontier_mode=self._frontier_mode,
                input_molecule_count=len(input_batch),
                total_seconds=perf_counter() - total_started,
            )
            return

        transfer_metrics: dict[str, int | float] = {
            "host_to_device_bytes": 0,
            "host_to_device_seconds": 0.0,
            "memory_release_seconds": 0.0,
        }
        input_batch, part_iterator, _candidate_pairs = self._select_part_iterator(
            input_batch,
            pairs,
            transfer_metrics,
            fingerprint_config=fingerprint_config,
        )

        diagnostics: list[CudaMatchDiagnostics] = []
        consumer_s = 0.0
        try:
            for precursors, part_hits, _assignments, diagnostic in part_iterator:
                diagnostics.append(diagnostic)
                if precursors is None:
                    continue
                # Timed in try/finally because the clock keeps running while this generator is
                # suspended; the `finally` also catches GeneratorExit, so an abandoned pull is
                # not billed to the matcher either.
                parked_started = perf_counter()
                try:
                    yield precursors, part_hits
                finally:
                    consumer_s += perf_counter() - parked_started
        finally:
            aggregate = (
                _aggregate_diagnostics(diagnostics)
                if diagnostics
                else _EMPTY_DIAGNOSTICS
            )
            self._last_diagnostics = replace(
                aggregate,
                frontier_mode=self._frontier_mode,
                input_molecule_count=len(input_batch),
                consumer_seconds=consumer_s,
                total_seconds=perf_counter() - total_started - consumer_s,
            )

    def match_keyed(
        self,
        mols: Sequence[CMol] | CudaMoleculeBatch | CudaMoleculeHostBatch,
        pairs: Sequence[Pair] | None = None,
        *,
        fingerprint_config: FoldedFingerprintConfig | None = None,
    ) -> tuple[
        CudaMoleculeBatch | CudaMoleculeDeviceLogicalBatch | CudaMoleculeHostBatch,
        Sequence[tuple[int, int]],
        HitArray,
    ]:
        """Match, key each bounded CUDA result, and store the next frontier."""
        total_started = perf_counter()
        if isinstance(mols, (CudaMoleculeBatch, CudaMoleculeHostBatch)):
            input_batch = mols
        else:
            input_batch = cuda_stack_mols(mols)
        if not len(input_batch) or not self._rule_chunks:
            self._last_diagnostics = replace(
                _EMPTY_DIAGNOSTICS,
                frontier_mode=self._frontier_mode,
                input_molecule_count=len(input_batch),
                total_seconds=perf_counter() - total_started,
            )
            empty = (
                CudaMoleculeHostBatch(())
                if self._frontier_mode == "host"
                else _empty_cuda_molecule_batch()
            )
            return empty, (), empty_hits()
        if pairs is not None and not pairs:
            self._last_diagnostics = replace(
                _EMPTY_DIAGNOSTICS,
                frontier_mode=self._frontier_mode,
                input_molecule_count=len(input_batch),
                total_seconds=perf_counter() - total_started,
            )
            empty = (
                CudaMoleculeHostBatch(())
                if self._frontier_mode == "host"
                else _empty_cuda_molecule_batch()
            )
            return empty, (), empty_hits()

        device_frontier: CudaMoleculeBatch | None = None
        host_parts: list[CudaMoleculeHostPart] = []
        key_to_physical: dict[tuple[int, int], int] = {}
        physical_keys: list[tuple[int, int]] = []
        logical_indices = array("q")
        raw_precursor_count = 0
        retained_molecule_count = 0
        retained_dense_bytes = 0
        molecules_transferred_to_host_count = 0
        molecules_retained_on_device_count = 0
        keys_transferred_count = 0
        atom_counts_transferred_count = 0
        hit_parts: list[HitArray] = []
        diagnostics: list[CudaMatchDiagnostics] = []
        transfer_metrics = {
            "host_to_device_bytes": 0,
            "host_to_device_seconds": 0.0,
            "memory_release_seconds": 0.0,
        }
        output_dense_bytes = 0
        device_to_host_bytes = 0
        device_to_host_seconds = 0.0
        key_seconds = 0.0
        wl_kernel_seconds = 0.0
        hash_device_to_host_seconds = 0.0
        key_materialize_seconds = 0.0
        host_dedup_seconds = 0.0
        atom_count_device_to_host_seconds = 0.0
        retained_gather_seconds = 0.0
        retained_device_to_host_seconds = 0.0
        hit_accumulate_seconds = 0.0
        atom_histogram_seconds = 0.0
        memory_snapshot_seconds = 0.0
        host_store_seconds = 0.0
        atom_count_histogram = np.zeros(_CUDA_BUCKET_SIZE + 1, dtype=np.int64)
        peak_driver_used_bytes = 0
        peak_cupy_pool_used_bytes = 0
        peak_cupy_pool_total_bytes = 0
        peak_process_rss_bytes = 0
        input_batch, part_iterator, candidate_pair_count = self._select_part_iterator(
            input_batch,
            pairs,
            transfer_metrics,
            fingerprint_config=fingerprint_config,
        )
        if self._frontier_mode == "device":
            frontier_capacity = self._device_frontier_capacity(
                input_molecule_count=len(input_batch),
                pair_count=(
                    candidate_pair_count
                    if pairs is None and fingerprint_config is not None
                    else (None if pairs is None else len(pairs))
                ),
            )
            cp = import_cupy()
            try:
                device_frontier = _cuda_molecule_storage(frontier_capacity)
            except cp.cuda.memory.OutOfMemoryError:
                cp.get_default_memory_pool().free_all_blocks()
                device_frontier = _cuda_molecule_storage(frontier_capacity)
        for (
            precursors,
            part_hits,
            _,
            diagnostic,
        ) in part_iterator:
            diagnostics.append(diagnostic)
            if precursors is not None:
                retained: CudaMoleculeBatch | None = None
                output_dense_bytes += cuda_molecule_batch_bytes(precursors)
                (hash_lo, hash_hi), kernel_elapsed = cuda_event_seconds(
                    lambda precursors=precursors: wl_hash_kernel(precursors)
                )
                wl_kernel_seconds += kernel_elapsed
                started = perf_counter()
                host_lo = hash_lo.get()
                host_hi = hash_hi.get()
                hash_copy_elapsed = perf_counter() - started
                hash_device_to_host_seconds += hash_copy_elapsed
                device_to_host_seconds += hash_copy_elapsed
                device_to_host_bytes += hash_lo.nbytes + hash_hi.nbytes
                started = perf_counter()
                part_keys = list(zip(host_lo.tolist(), host_hi.tolist(), strict=True))
                key_materialize_seconds += perf_counter() - started
                keys_transferred_count += len(part_keys)
                raw_precursor_count += len(part_keys)
                started = perf_counter()
                hit_parts.append(part_hits)
                hit_accumulate_seconds += perf_counter() - started
                if self._frontier_mode == "host":
                    dedup_started = perf_counter()
                    new_rows: list[int] = []
                    new_keys: list[tuple[int, int]] = []
                    for row_index, key in enumerate(part_keys):
                        physical_index = key_to_physical.get(key)
                        if physical_index is None:
                            physical_index = retained_molecule_count + len(new_rows)
                            key_to_physical[key] = physical_index
                            new_rows.append(row_index)
                            new_keys.append(key)
                            physical_keys.append(key)
                        logical_indices.append(physical_index)
                    host_dedup_seconds += perf_counter() - dedup_started

                    started = perf_counter()
                    counts = host_array(precursors.atom_counts_device)
                    counts_elapsed = perf_counter() - started
                    atom_count_device_to_host_seconds += counts_elapsed
                    device_to_host_seconds += counts_elapsed
                    device_to_host_bytes += counts.nbytes
                    atom_counts_transferred_count += int(counts.size)

                    if new_rows:
                        retained, gather_elapsed = cuda_event_seconds(
                            lambda precursors=precursors, new_rows=new_rows: precursors[
                                np.asarray(new_rows, dtype=np.int64)
                            ]
                        )
                        retained_gather_seconds += gather_elapsed
                        started = perf_counter()
                        host_part = CudaMoleculeHostPart.from_cuda(retained, new_keys)
                        host_elapsed = perf_counter() - started
                        retained_device_to_host_seconds += host_elapsed
                        host_store_seconds += gather_elapsed + host_elapsed
                        device_to_host_seconds += host_elapsed
                        device_to_host_bytes += host_part.nbytes
                        retained_molecule_count += host_part.count
                        molecules_transferred_to_host_count += host_part.count
                        retained_dense_bytes += host_part.nbytes
                        host_parts.append(host_part)
                else:
                    dedup_started = perf_counter()
                    new_rows = []
                    for row_index, key in enumerate(part_keys):
                        physical_index = key_to_physical.get(key)
                        if physical_index is None:
                            physical_index = retained_molecule_count + len(new_rows)
                            key_to_physical[key] = physical_index
                            new_rows.append(row_index)
                            physical_keys.append(key)
                        logical_indices.append(physical_index)
                    host_dedup_seconds += perf_counter() - dedup_started

                    started = perf_counter()
                    counts = host_array(precursors.atom_counts_device)
                    counts_elapsed = perf_counter() - started
                    atom_count_device_to_host_seconds += counts_elapsed
                    device_to_host_seconds += counts_elapsed
                    device_to_host_bytes += counts.nbytes
                    atom_counts_transferred_count += int(counts.size)

                    if new_rows:
                        if device_frontier is None:
                            raise RuntimeError("device frontier is not initialized.")
                        stop = retained_molecule_count + len(new_rows)
                        if stop > device_frontier.count:
                            raise_cuda_device_frontier_capacity(
                                stop,
                                device_frontier.count,
                            )

                        new_row_indices = np.asarray(new_rows, dtype=np.int64)
                        stored_count, gather_elapsed = cuda_event_seconds(
                            lambda retained_molecule_count=retained_molecule_count, precursors=precursors, new_row_indices=new_row_indices: (
                                _scatter_cuda_molecule_rows(
                                    device_frontier,
                                    retained_molecule_count,
                                    precursors,
                                    new_row_indices,
                                )
                            )
                        )
                        retained_gather_seconds += gather_elapsed
                        retained_molecule_count += stored_count
                        molecules_retained_on_device_count += stored_count
                        retained_dense_bytes += stored_count * cuda_molecule_row_bytes()
                histogram_started = perf_counter()
                atom_count_histogram += np.bincount(
                    counts.astype(np.int64, copy=False),
                    minlength=_CUDA_BUCKET_SIZE + 1,
                )[: _CUDA_BUCKET_SIZE + 1]
                atom_histogram_seconds += perf_counter() - histogram_started
                snapshot_started = perf_counter()
                driver_used, pool_used, pool_total = cuda_memory_snapshot()
                peak_driver_used_bytes = max(peak_driver_used_bytes, driver_used)
                peak_cupy_pool_used_bytes = max(peak_cupy_pool_used_bytes, pool_used)
                peak_cupy_pool_total_bytes = max(peak_cupy_pool_total_bytes, pool_total)
                peak_process_rss_bytes = max(
                    peak_process_rss_bytes, process_rss_bytes()
                )
                memory_snapshot_seconds += perf_counter() - snapshot_started
                release_started = perf_counter()
                del hash_lo, hash_hi
                if retained is not None:
                    del retained
                del precursors
                import_cupy().get_default_memory_pool().free_all_blocks()
                transfer_metrics["memory_release_seconds"] += (
                    perf_counter() - release_started
                )
        # Each pipeline part numbers its precursor rows densely. Concatenate in that raw
        # space, then map the hit column to the physical batch deduplicated above. The keyed
        # matcher contract permits a precursor batch shorter than the hits; doing this here
        # avoids sending duplicate keys through the Python interner and expansion store.
        hits = concat_hits(hit_parts, [len(part) for part in hit_parts])
        physical_indices = np.frombuffer(logical_indices, dtype=np.int64)
        if len(hits):
            hits[:, 3] = physical_indices[hits[:, 3]]
        materialize_started = perf_counter()
        if self._frontier_mode == "host":
            output_batch: (
                CudaMoleculeBatch
                | CudaMoleculeDeviceLogicalBatch
                | CudaMoleculeHostBatch
            ) = CudaMoleculeHostBatch(
                tuple(host_parts),
            )
        else:
            device_batch = (
                _cuda_molecule_prefix_view(device_frontier, retained_molecule_count)
                if device_frontier is not None and retained_molecule_count
                else _empty_cuda_molecule_batch()
            )
            output_batch = replace(device_batch, tokens=tuple(physical_keys))
        materialize_seconds = perf_counter() - materialize_started

        unique_precursor_count = len(key_to_physical)
        input_dense_bytes = (
            input_batch.nbytes
            if isinstance(input_batch, CudaMoleculeHostBatch)
            else cuda_molecule_batch_bytes(input_batch)
        )
        aggregate = _aggregate_diagnostics(diagnostics)
        device_to_host_seconds += (
            aggregate.hit_device_to_host_seconds
            + aggregate.diagnostic_device_to_host_seconds
        )
        device_to_host_bytes += (
            len(hits) * 3 * np.dtype(np.int32).itemsize
            + aggregate.pair_launches * 15 * np.dtype(np.uint64).itemsize
        )
        key_seconds = (
            wl_kernel_seconds + hash_device_to_host_seconds + key_materialize_seconds
        )
        classified_seconds = (
            aggregate.match_seconds
            + aggregate.retro_seconds
            + aggregate.diagnostic_reduce_seconds
            + aggregate.diagnostic_device_to_host_seconds
            + aggregate.hit_device_seconds
            + aggregate.hit_device_to_host_seconds
            + aggregate.hit_materialize_seconds
            + aggregate.pipeline_overhead_seconds
            + wl_kernel_seconds
            + hash_device_to_host_seconds
            + key_materialize_seconds
            + host_dedup_seconds
            + atom_count_device_to_host_seconds
            + retained_gather_seconds
            + retained_device_to_host_seconds
            + hit_accumulate_seconds
            + atom_histogram_seconds
            + memory_snapshot_seconds
            + float(transfer_metrics["memory_release_seconds"])
            + materialize_seconds
            + float(transfer_metrics["host_to_device_seconds"])
        )
        total_seconds = perf_counter() - total_started
        self._last_diagnostics = replace(
            aggregate,
            frontier_mode=self._frontier_mode,
            input_molecule_count=len(input_batch),
            input_dense_bytes=input_dense_bytes,
            output_dense_bytes=output_dense_bytes,
            raw_precursor_count=raw_precursor_count,
            unique_precursor_count=unique_precursor_count,
            duplicate_precursor_count=raw_precursor_count - unique_precursor_count,
            atom_count_max=(
                int(np.flatnonzero(atom_count_histogram)[-1])
                if np.any(atom_count_histogram)
                else 0
            ),
            atom_count_histogram=tuple(int(v) for v in atom_count_histogram),
            host_to_device_bytes=int(transfer_metrics["host_to_device_bytes"]),
            device_to_host_bytes=device_to_host_bytes,
            host_to_device_seconds=float(transfer_metrics["host_to_device_seconds"]),
            device_to_host_seconds=device_to_host_seconds,
            key_seconds=key_seconds,
            wl_kernel_seconds=wl_kernel_seconds,
            hash_device_to_host_seconds=hash_device_to_host_seconds,
            key_materialize_seconds=key_materialize_seconds,
            host_dedup_seconds=host_dedup_seconds,
            atom_count_device_to_host_seconds=atom_count_device_to_host_seconds,
            retained_gather_seconds=retained_gather_seconds,
            retained_device_to_host_seconds=retained_device_to_host_seconds,
            hit_accumulate_seconds=hit_accumulate_seconds,
            atom_histogram_seconds=atom_histogram_seconds,
            memory_snapshot_seconds=memory_snapshot_seconds,
            memory_release_seconds=float(transfer_metrics["memory_release_seconds"]),
            host_store_seconds=host_store_seconds,
            materialize_seconds=materialize_seconds,
            peak_driver_used_bytes=peak_driver_used_bytes,
            peak_cupy_pool_used_bytes=peak_cupy_pool_used_bytes,
            peak_cupy_pool_total_bytes=peak_cupy_pool_total_bytes,
            peak_process_rss_bytes=peak_process_rss_bytes,
            retained_molecule_count=retained_molecule_count,
            retained_dense_bytes=retained_dense_bytes,
            keys_generated_count=raw_precursor_count,
            keys_transferred_count=keys_transferred_count,
            atom_counts_transferred_count=atom_counts_transferred_count,
            molecules_transferred_to_host_count=molecules_transferred_to_host_count,
            molecules_retained_on_device_count=molecules_retained_on_device_count,
            unclassified_seconds=max(total_seconds - classified_seconds, 0.0),
            total_seconds=total_seconds,
        )
        return output_batch, physical_keys, hits

    def _select_part_iterator(
        self,
        input_batch: CudaMoleculeBatch | CudaMoleculeHostBatch,
        pairs: Sequence[Pair] | None,
        transfer_metrics: dict[str, int | float],
        *,
        fingerprint_config: FoldedFingerprintConfig | None,
    ) -> tuple[
        CudaMoleculeBatch | CudaMoleculeHostBatch,
        Iterator[tuple[CudaMoleculeBatch | None, HitArray, int, CudaMatchDiagnostics]],
        int | None,
    ]:
        """Pick the part generator for this call, and the batch the caller should go on using.

        Three ways in -- prefiltered all-vs-all, plain all-vs-all, and an explicit pair list --
        and the choice also decides whether the prefilter diagnostics exist and whether a host
        batch has to be moved to the device first. Shared rather than copied because
        `match_keyed` and `match_batches` must pick the *same* iterator for the same arguments:
        the moment they differ, the two entry points are measuring different work and no A/B
        between them means anything.

        Returns the batch as well because the selective path may replace a host batch with its
        device copy, and the caller's later byte accounting has to describe the batch that was
        actually used.
        """
        if pairs is None and fingerprint_config is not None:
            prepared_parts, prefilter_diagnostics = self._prepare_prefiltered_parts(
                input_batch,
                transfer_metrics,
                fingerprint_config=fingerprint_config,
            )
            self._last_prefilter_diagnostics = prefilter_diagnostics
            return (
                input_batch,
                self._iter_prepared_pipeline_parts(prepared_parts),
                prefilter_diagnostics.candidate_pair_count,
            )
        if pairs is None:
            self._last_prefilter_diagnostics = None
            return (
                input_batch,
                self._iter_all_pipeline_parts(input_batch, transfer_metrics),
                None,
            )
        self._last_prefilter_diagnostics = None
        if isinstance(input_batch, CudaMoleculeHostBatch):
            host_batch_bytes = input_batch.nbytes
            started = perf_counter()
            input_batch = input_batch.to_cuda()
            transfer_metrics["host_to_device_seconds"] += perf_counter() - started
            transfer_metrics["host_to_device_bytes"] += host_batch_bytes
        return (
            input_batch,
            self._iter_selective_pipeline_parts(input_batch, pairs, transfer_metrics),
            None,
        )

    def _iter_all_pipeline_parts(
        self,
        mols: CudaMoleculeBatch | CudaMoleculeHostBatch,
        transfer_metrics: dict[str, int | float],
    ) -> Iterator[tuple[CudaMoleculeBatch | None, HitArray, int, CudaMatchDiagnostics]]:
        molecule_count = len(mols)
        mol_start = 0
        app_base = [0]
        while mol_start < molecule_count:
            capacities = [
                (
                    chunk,
                    self._pair_capacity(resident_rule_bytes=self._resident_rule_bytes),
                )
                for chunk in self._rule_chunks
            ]
            molecule_chunk_size = min(
                max(1, capacity // min(chunk.pattern_count, capacity))
                for chunk, capacity in capacities
            )
            mol_end = min(mol_start + molecule_chunk_size, molecule_count)
            selected = mols[slice(mol_start, mol_end)]
            if isinstance(selected, CudaMoleculeHostBatch):
                started = perf_counter()
                sub_batch = selected.to_cuda()
                transfer_metrics["host_to_device_seconds"] += perf_counter() - started
                transfer_metrics["host_to_device_bytes"] += selected.nbytes
            else:
                sub_batch = selected
            sub_count = mol_end - mol_start

            for chunk, pair_capacity in capacities:
                for pattern_start in range(0, chunk.pattern_count, pair_capacity):
                    active_pattern_count = min(
                        pair_capacity,
                        chunk.pattern_count - pattern_start,
                    )
                    pair_pattern_index, pair_molecule_index = _pair_indices(
                        active_pattern_count,
                        sub_count,
                    )
                    pair_pattern_index += pattern_start
                    yield from self._iter_pipeline_parts(
                        sub_batch,
                        chunk,
                        pair_pattern_index,
                        pair_molecule_index,
                        mol_offset=mol_start,
                        app_base=app_base,
                    )
                    if self._frontier_mode == "host":
                        release_started = perf_counter()
                        import_cupy().get_default_memory_pool().free_all_blocks()
                        transfer_metrics["memory_release_seconds"] += (
                            perf_counter() - release_started
                        )
            if self._frontier_mode == "host":
                del sub_batch
                release_started = perf_counter()
                import_cupy().get_default_memory_pool().free_all_blocks()
                transfer_metrics["memory_release_seconds"] += (
                    perf_counter() - release_started
                )
            mol_start = mol_end

    def _prepare_prefiltered_parts(
        self,
        mols: CudaMoleculeBatch | CudaMoleculeHostBatch,
        transfer_metrics: dict[str, int | float],
        *,
        fingerprint_config: FoldedFingerprintConfig,
    ) -> tuple[
        list[tuple[CudaMoleculeBatch, _CudaRuleChunk, Any, Any, int]],
        CudaReactionPrefilterDiagnostics,
    ]:
        cp = import_cupy()
        prepared: list[tuple[CudaMoleculeBatch, _CudaRuleChunk, Any, Any, int]] = []
        dense_pair_count = 0
        candidate_pair_count = 0
        prefilter_seconds = 0.0
        molecule_count = len(mols)
        mol_start = 0
        while mol_start < molecule_count:
            capacities = [
                (
                    chunk,
                    self._pair_capacity(resident_rule_bytes=self._resident_rule_bytes),
                )
                for chunk in self._rule_chunks
            ]
            molecule_chunk_size = min(
                max(1, capacity // min(chunk.pattern_count, capacity))
                for chunk, capacity in capacities
            )
            mol_end = min(mol_start + molecule_chunk_size, molecule_count)
            selected = mols[slice(mol_start, mol_end)]
            if isinstance(selected, CudaMoleculeHostBatch):
                started = perf_counter()
                sub_batch = selected.to_cuda()
                transfer_metrics["host_to_device_seconds"] += perf_counter() - started
                transfer_metrics["host_to_device_bytes"] += selected.nbytes
            else:
                sub_batch = selected
            sub_count = mol_end - mol_start
            cp.cuda.Stream.null.synchronize()
            started = perf_counter()
            sub_fps = compute_folded_path_fingerprints_cuda(
                sub_batch,
                query=False,
                config=fingerprint_config,
            )
            for chunk, pair_capacity in capacities:
                pattern_fps = chunk.pattern_fingerprints_device
                if pattern_fps is None:
                    raise ValueError("CUDA folded fingerprints are not initialized.")
                for pattern_start in range(0, chunk.pattern_count, pair_capacity):
                    active_pattern_count = min(
                        pair_capacity,
                        chunk.pattern_count - pattern_start,
                    )
                    prefiltered = compact_folded_prefilter_pairs_cuda(
                        molecule_fingerprints_device=sub_fps,
                        pattern_fingerprints_device=pattern_fps[
                            pattern_start : pattern_start + active_pattern_count
                        ],
                        molecule_atom_counts_device=sub_batch.atom_counts_device,
                        pattern_atom_counts_device=chunk.pattern_atom_counts_device[
                            pattern_start : pattern_start + active_pattern_count
                        ],
                        pattern_index_offset=pattern_start,
                    )
                    dense_pair_count += active_pattern_count * sub_count
                    candidate_pair_count += prefiltered.pair_count
                    if prefiltered.pair_count:
                        prepared.append(
                            (
                                sub_batch,
                                chunk,
                                prefiltered.pair_pattern_index_device,
                                prefiltered.pair_molecule_index_device,
                                mol_start,
                            )
                        )
            cp.cuda.Stream.null.synchronize()
            prefilter_seconds += perf_counter() - started
            mol_start = mol_end

        pruned_pair_count = dense_pair_count - candidate_pair_count
        return prepared, CudaReactionPrefilterDiagnostics(
            dense_pair_count=dense_pair_count,
            candidate_pair_count=candidate_pair_count,
            pruned_pair_count=pruned_pair_count,
            prune_ratio=(
                0.0 if dense_pair_count == 0 else pruned_pair_count / dense_pair_count
            ),
            prefilter_seconds=prefilter_seconds,
        )

    def _iter_prepared_pipeline_parts(
        self,
        prepared: Sequence[tuple[CudaMoleculeBatch, _CudaRuleChunk, Any, Any, int]],
    ) -> Iterator[tuple[CudaMoleculeBatch | None, HitArray, int, CudaMatchDiagnostics]]:
        app_base = [0]
        for sub_batch, chunk, pattern_indices, molecule_indices, mol_offset in prepared:
            yield from self._iter_pipeline_parts(
                sub_batch,
                chunk,
                pattern_indices,
                molecule_indices,
                mol_offset=mol_offset,
                app_base=app_base,
            )

    def _iter_selective_pipeline_parts(
        self,
        mols: CudaMoleculeBatch,
        pairs: Sequence[Pair],
        transfer_metrics: dict[str, int | float],
    ) -> Iterator[tuple[CudaMoleculeBatch | None, HitArray, int, CudaMatchDiagnostics]]:
        pairs_by_chunk: dict[int, list[tuple[int, int]]] = {}
        for mol_idx, rule_id in pairs:
            if mol_idx < 0 or mol_idx >= mols.count:
                raise IndexError(
                    f"molecule index {mol_idx} is outside batch of length {mols.count}"
                )
            try:
                chunk, local_index = self._chunk_by_rule[rule_id]
            except KeyError as exc:
                raise KeyError(f"unknown CUDA reaction rule id {rule_id!r}") from exc
            pairs_by_chunk.setdefault(id(chunk), []).append((mol_idx, local_index))

        chunk_by_identity = {id(chunk): chunk for chunk in self._rule_chunks}
        pair_capacity = self._pair_capacity(
            resident_rule_bytes=self._resident_rule_bytes
        )
        app_base = [0]
        for chunk_id, chunk_pairs in pairs_by_chunk.items():
            chunk = chunk_by_identity[chunk_id]
            for start in range(0, len(chunk_pairs), pair_capacity):
                pair_slice = chunk_pairs[start : start + pair_capacity]
                pair_molecule_index = np.asarray(
                    [mol_idx for mol_idx, _ in pair_slice],
                    dtype=np.int32,
                )
                pair_pattern_index = np.asarray(
                    [local_index for _, local_index in pair_slice],
                    dtype=np.int32,
                )
                yield from self._iter_pipeline_parts(
                    mols,
                    chunk,
                    pair_pattern_index,
                    pair_molecule_index,
                    mol_offset=0,
                    app_base=app_base,
                )
                if self._frontier_mode == "host":
                    release_started = perf_counter()
                    import_cupy().get_default_memory_pool().free_all_blocks()
                    transfer_metrics["memory_release_seconds"] += (
                        perf_counter() - release_started
                    )

    def _iter_pipeline_parts(
        self,
        mol_batch: CudaMoleculeBatch,
        chunk: _CudaRuleChunk,
        pair_pattern_index: np.ndarray,
        pair_molecule_index: np.ndarray,
        *,
        mol_offset: int,
        app_base: list[int],
    ) -> Iterator[tuple[CudaMoleculeBatch | None, HitArray, int, CudaMatchDiagnostics]]:
        should_split = False
        try:
            result = self._run_pipeline(
                mol_batch,
                chunk,
                pair_pattern_index,
                pair_molecule_index,
                mol_offset=mol_offset,
                app_base=app_base[0],
                split_on_capacity=False,
            )
        except Exception as exc:
            if not is_cuda_memory_error(exc):
                raise
            should_split = True

        if should_split:
            import_cupy().get_default_memory_pool().free_all_blocks()
            pair_count = len(pair_pattern_index)
            if pair_count <= 1:
                raise ValueError(
                    "CUDA memory limit cannot fit the compact output for one "
                    "molecule-rule pair."
                ) from None
            middle = pair_count // 2
            yield from self._iter_pipeline_parts(
                mol_batch,
                chunk,
                pair_pattern_index[:middle],
                pair_molecule_index[:middle],
                mol_offset=mol_offset,
                app_base=app_base,
            )
            yield from self._iter_pipeline_parts(
                mol_batch,
                chunk,
                pair_pattern_index[middle:],
                pair_molecule_index[middle:],
                mol_offset=mol_offset,
                app_base=app_base,
            )
            return

        app_base[0] += result[2]
        yield result

    def _run_pipeline(
        self,
        mol_batch: CudaMoleculeBatch,
        chunk: _CudaRuleChunk,
        pair_pattern_index: np.ndarray,
        pair_molecule_index: np.ndarray,
        *,
        mol_offset: int,
        app_base: int,
        split_on_capacity: bool = True,
    ) -> tuple[CudaMoleculeBatch | None, HitArray, int, CudaMatchDiagnostics]:
        """Run prune -> fused match -> retro split for one (mol chunk x rule chunk),
        emitting precursors + global-id hits. `mol_offset`/`app_base` translate the
        chunk-local molecule and assignment indices into globally unique ids so the
        accumulated hits group correctly when the store builds steps. Returns
        (precursors_or_None, hits, assignment_count, diagnostics)."""
        pipeline_started = perf_counter()
        pair_count = int(pair_pattern_index.shape[0])
        usable_device_bytes = self._usable_device_bytes(
            resident_rule_bytes=self._resident_rule_bytes
        )
        match_workspace_bytes = pair_count * match_pair_workspace_bytes()
        assignment_row_bytes = (
            _CUDA_BUCKET_SIZE * np.dtype(np.int32).itemsize
            + np.dtype(np.int32).itemsize
        )
        assignment_capacity_limit = max(
            min(
                usable_device_bytes // assignment_row_bytes,
                CUDA_MAX_ASSIGNMENTS_PER_RESULT,
            ),
            1,
        )
        assignment_capacity = min(
            max(pair_count, 1),
            CUDA_INITIAL_ASSIGNMENT_CAPACITY,
            assignment_capacity_limit,
        )
        if hasattr(pair_pattern_index, "get"):
            batch = prepare_cuda_prune_batch_from_device_pairs(
                chunk.pattern_count,
                chunk.pattern_atom_counts_device,
                chunk.pattern_atoms_u32_device,
                chunk.pattern_rows_u64_device,
                chunk.pattern_bonds_u8_device,
                pattern_groups_u32_device=chunk.pattern_groups_u32_device,
                molecule_atom_counts_device=mol_batch.atom_counts_device,
                molecule_atoms_u32=mol_batch.atoms_device,
                molecule_rows_u64=mol_batch.rows_device,
                molecule_bonds_u8=mol_batch.bonds_device,
                pair_pattern_index_device=pair_pattern_index,
                pair_molecule_index_device=pair_molecule_index,
                pair_count=pair_count,
            )
        else:
            batch = prepare_cuda_prune_batch_from_device(
                chunk.pattern_count,
                chunk.pattern_atom_counts_device,
                chunk.pattern_atoms_u32_device,
                chunk.pattern_rows_u64_device,
                chunk.pattern_bonds_u8_device,
                pattern_groups_u32_device=chunk.pattern_groups_u32_device,
                molecule_atom_counts_device=mol_batch.atom_counts_device,
                molecule_atoms_u32=mol_batch.atoms_device,
                molecule_rows_u64=mol_batch.rows_device,
                molecule_bonds_u8=mol_batch.bonds_device,
                pair_pattern_index=pair_pattern_index,
                pair_molecule_index=pair_molecule_index,
            )

        match_started = perf_counter()
        batch = apply_cuda_predicates(
            batch,
            chunk.predicate_program,
            max_prune_rounds=self._max_prune_rounds,
            max_leaf_checks=self._max_leaf_checks,
            max_prefix_depth=self._max_prefix_depth,
        )
        try:
            match_result = match_fused_assignments_cuda(
                batch,
                threads_per_block=self._threads_per_block,
                max_prune_rounds=self._max_prune_rounds,
                max_leaf_checks=self._max_leaf_checks,
                max_prefix_depth=self._max_prefix_depth,
                assignment_capacity=assignment_capacity,
                assignment_capacity_limit=assignment_capacity_limit,
            )
        except CudaOutputCapacityError:
            if not split_on_capacity:
                raise
            return self._split_pipeline(
                mol_batch,
                chunk,
                pair_pattern_index,
                pair_molecule_index,
                mol_offset=mol_offset,
                app_base=app_base,
            )
        match_seconds = perf_counter() - match_started

        assignment_count = match_result.assignment_required
        if assignment_count == 0:
            diagnostic = _cuda_match_diagnostics(
                match_result,
                assignment_launches=0,
                usable_device_bytes=usable_device_bytes,
                peak_estimated_workspace_bytes=match_workspace_bytes,
                resident_rule_bytes=self._resident_rule_bytes,
                match_seconds=match_seconds,
            )
            return None, empty_hits(), 0, diagnostic

        retro_usable_bytes = self._usable_device_bytes(
            resident_rule_bytes=self._resident_rule_bytes,
            active_workspace_bytes=(
                len(pair_pattern_index) * match_pair_workspace_bytes()
                + match_result.assignment_capacity * assignment_row_bytes
            ),
        )
        precursor_capacity_limit = max(
            min(
                retro_usable_bytes // retro_precursor_workspace_bytes(),
                CUDA_MAX_PRECURSORS_PER_RESULT,
            ),
            1,
        )
        expected_precursors = int(assignment_count * CUDA_PRECURSORS_PER_ASSIGNMENT)
        precursor_capacity = min(
            max(expected_precursors, CUDA_INITIAL_PRECURSOR_CAPACITY),
            precursor_capacity_limit,
        )
        # Bond edits + connected-component split, then hydrogen/aromaticity perception, are
        # separate kernel launches. Time them separately (perceive=False here, then run
        # perceive_cuda_retro ourselves) so the paper can attribute time to each stage;
        # retro_seconds stays their sum so downstream classified/total math is unchanged.
        bond_edit_split_started = perf_counter()
        try:
            retro = construct_retro_cuda(
                batch,
                match_result,
                chunk.reaction_batch,
                threads_per_block=self._threads_per_block,
                perceive=False,
                compute_wl=False,
                precursor_capacity=precursor_capacity,
                precursor_capacity_limit=precursor_capacity_limit,
            )
        except CudaOutputCapacityError:
            if not split_on_capacity:
                raise
            return self._split_pipeline(
                mol_batch,
                chunk,
                pair_pattern_index,
                pair_molecule_index,
                mol_offset=mol_offset,
                app_base=app_base,
            )
        import_cupy().cuda.Stream.null.synchronize()
        bond_edit_split_seconds = perf_counter() - bond_edit_split_started
        perceive_started = perf_counter()
        retro = perceive_cuda_retro(retro, threads_per_block=self._threads_per_block)
        import_cupy().cuda.Stream.null.synchronize()
        perceive_seconds = perf_counter() - perceive_started
        retro_seconds = bond_edit_split_seconds + perceive_seconds
        precursor_count = retro.precursor_required
        precursors = CudaMoleculeBatch(
            tokens=(),
            atom_counts_device=retro.precursor_atom_counts_device,
            atoms_device=retro.precursor_atoms_device,
            bonds_device=retro.precursor_bonds_device,
            rows_device=retro.precursor_rows_device,
        )
        cp = import_cupy()
        hit_device_started = perf_counter()
        assignment_indices_device = retro.precursor_assignment_ids_device
        pair_ids_device = match_result.assignment_pair_ids_device[
            assignment_indices_device
        ]
        hit_columns_device = cp.stack(
            (
                batch.pair_molecule_index_device[pair_ids_device],
                batch.pair_pattern_index_device[pair_ids_device],
                assignment_indices_device,
            ),
            axis=1,
        )
        cp.cuda.Stream.null.synchronize()
        hit_device_seconds = perf_counter() - hit_device_started
        hit_copy_started = perf_counter()
        hit_columns_host = hit_columns_device.get()
        hit_device_to_host_seconds = perf_counter() - hit_copy_started
        hit_materialize_started = perf_counter()
        rule_ids = np.asarray(chunk.rule_ids, dtype=np.int64)
        hit_count = hit_columns_host.shape[0]
        hits = np.empty((hit_count, 4), dtype=np.int64)
        hits[:, 0] = mol_offset + hit_columns_host[:, 0]
        hits[:, 1] = rule_ids[hit_columns_host[:, 1]]
        hits[:, 2] = app_base + hit_columns_host[:, 2]
        # The retro pipeline emits one precursor per hit, in hit order: the precursor
        # column is the identity, and `concat_hits` rebases it per chunk.
        hits[:, 3] = np.arange(hit_count, dtype=np.int64)
        hit_materialize_seconds = perf_counter() - hit_materialize_started

        peak_workspace_bytes = (
            match_workspace_bytes
            + match_result.assignment_capacity
            * (
                _CUDA_BUCKET_SIZE * np.dtype(np.int32).itemsize
                + np.dtype(np.int32).itemsize
            )
            + retro.precursor_capacity * retro_precursor_workspace_bytes()
        )
        accounted_before_diagnostics = (
            match_seconds
            + retro_seconds
            + hit_device_seconds
            + hit_device_to_host_seconds
            + hit_materialize_seconds
        )
        diagnostic = _cuda_match_diagnostics(
            match_result,
            assignment_launches=1 + retro.overflow_retries,
            usable_device_bytes=usable_device_bytes,
            peak_estimated_workspace_bytes=peak_workspace_bytes,
            resident_rule_bytes=self._resident_rule_bytes,
            precursor_capacity=retro.precursor_capacity,
            precursor_required=precursor_count,
            precursor_overflow_retries=retro.overflow_retries,
            precursor_atom_overflow_components=retro.precursor_atom_overflow_components,
            precursor_atom_overflow_assignments=retro.precursor_atom_overflow_assignments,
            edited_atom_overflow_assignments=retro.edited_atom_overflow_assignments,
            rejections=retro.rejections,
            undescribed_components=retro.undescribed_components,
            orphaned_components=retro.orphaned_components,
            match_seconds=match_seconds,
            retro_seconds=retro_seconds,
            bond_edit_split_seconds=bond_edit_split_seconds,
            perceive_seconds=perceive_seconds,
            hit_device_seconds=hit_device_seconds,
            hit_device_to_host_seconds=hit_device_to_host_seconds,
            hit_materialize_seconds=hit_materialize_seconds,
        )
        pipeline_total = perf_counter() - pipeline_started
        diagnostic = replace(
            diagnostic,
            pipeline_overhead_seconds=max(
                pipeline_total
                - accounted_before_diagnostics
                - diagnostic.diagnostic_seconds,
                0.0,
            ),
        )
        return precursors, hits, assignment_count, diagnostic

    def _split_pipeline(
        self,
        mol_batch: CudaMoleculeBatch,
        chunk: _CudaRuleChunk,
        pair_pattern_index: np.ndarray,
        pair_molecule_index: np.ndarray,
        *,
        mol_offset: int,
        app_base: int,
    ) -> tuple[CudaMoleculeBatch | None, HitArray, int, CudaMatchDiagnostics]:
        pair_count = len(pair_pattern_index)
        if pair_count <= 1:
            raise ValueError(
                "CUDA memory limit cannot fit the compact output for one "
                "molecule-rule pair."
            )

        middle = pair_count // 2
        left = self._run_pipeline(
            mol_batch,
            chunk,
            pair_pattern_index[:middle],
            pair_molecule_index[:middle],
            mol_offset=mol_offset,
            app_base=app_base,
        )
        right = self._run_pipeline(
            mol_batch,
            chunk,
            pair_pattern_index[middle:],
            pair_molecule_index[middle:],
            mol_offset=mol_offset,
            app_base=app_base + left[2],
        )
        halves = [
            (batch, hits)
            for batch, hits in ((left[0], left[1]), (right[0], right[1]))
            if batch is not None
        ]
        precursors = CudaMoleculeBatch.from_parts([batch for batch, _ in halves])
        return (
            precursors,
            concat_hits(
                [hits for _, hits in halves], [batch.count for batch, _ in halves]
            ),
            left[2] + right[2],
            _aggregate_diagnostics((left[3], right[3])),
        )


class CudaFoldedPrefilteredReactionMatcher:
    """CUDA folded-linear-fingerprint prefilter wrapper for frontier search.

    The wrapper preserves the `ReactionMatcher` shape while keeping all-vs-all
    frontier prefiltering on device. Explicit selective pairs are delegated to
    the wrapped matcher for now; replay pair sets are already sparse.
    """

    def __init__(
        self,
        matcher: CudaReactionMatcher,
        *,
        fingerprint_config: FoldedFingerprintConfig = FoldedFingerprintConfig(),
    ) -> None:
        self.matcher = matcher
        self.fingerprint_config = fingerprint_config
        self.matcher._ensure_folded_fingerprints(fingerprint_config)

    @property
    def last_diagnostics(self) -> CudaMatchDiagnostics | None:
        return self.matcher.last_diagnostics

    @property
    def last_prefilter_diagnostics(self) -> CudaReactionPrefilterDiagnostics | None:
        return self.matcher.last_prefilter_diagnostics

    def match(
        self,
        mols: Sequence[CMol] | CudaMoleculeBatch,
        pairs: Sequence[Pair] | None = None,
    ) -> tuple[CudaMoleculeBatch, HitArray]:
        """The wrapped matcher's joined result, over the pairs that survived the screen.

        Screening decides which pairs reach the kernel, not how the results come back, so it is
        passed down as the fingerprint config and the delivery is whatever the inner matcher
        does. An explicit pair list has already chosen its pairs, so there is nothing to screen.
        """
        if pairs is not None:
            return self.matcher.match(mols, pairs)
        return self.matcher.match(mols, fingerprint_config=self.fingerprint_config)

    def match_batches(
        self,
        mols: Sequence[CMol] | CudaMoleculeBatch | CudaMoleculeHostBatch,
        pairs: Sequence[Pair] | None = None,
    ) -> Iterator[tuple[CudaMoleculeBatch, HitArray]]:
        """The wrapped matcher's parts, over the pairs that survived the screen."""
        if pairs is not None:
            return self.matcher.match_batches(mols, pairs)
        return self.matcher.match_batches(
            mols, fingerprint_config=self.fingerprint_config
        )

    def match_keyed(
        self,
        mols: Sequence[CMol] | CudaMoleculeBatch | CudaMoleculeHostBatch,
    ) -> tuple[
        CudaMoleculeBatch | CudaMoleculeDeviceLogicalBatch | CudaMoleculeHostBatch,
        Sequence[tuple[int, int]],
        HitArray,
    ]:
        return self.matcher.match_keyed(
            mols,
            fingerprint_config=self.fingerprint_config,
        )

"""High-level Metal reaction matcher: the orchestration layer that turns the
low-level Metal matcher, precursor-construction kernel, and WL-key kernel into a
`vesma.matcher.ReactionMatcher` for retrosynthesis search and replay.

Mirrors `cuda_matcher.py` (`CudaReactionMatcher`, `cuda_stack_mols`,
`cuda_short_ring_wl_key`, `CudaFoldedPrefilteredReactionMatcher`) and, like CUDA,
keeps precursors as **packed arrays** rather than decoding them back to `GMol`s: the
construction kernel emits each precursor in the same packed layout the matcher
consumes, so a subset of one step's precursors is gathered straight into the next
frontier and re-ingested with no `GMol` round-trip. A separate WL kernel fills the
precursor keys after construction.
"""

from __future__ import annotations

import ctypes
import itertools
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import numpy as np

from .adjacency import bond_kind_matrix
from .cmol import CMol
from .cmol_builder import gmol_from_cmol
from .gmol_builder import gmol_from_smarts
from .limits import ATOM_MASK_WORD_COUNT, GPU_MATCH_ATOM_CAP
from .matcher import HitArray, Pair, PairArray, concat_hits, empty_hits
from .metal_matcher import (
    MetalMatcher,
    _pack_adjacency_rows,
    _pack_atom_masks,
    _pack_bond_matrix,
    evaluate_predicate_roots,
    metal_predicate_program,
    pack_atom_groups,
    pack_collection_atom_predicates,
    pair_index_array,
)
from .metal_pathfp import (
    compact_folded_prefilter_pairs_metal,
    compute_folded_path_fingerprints_metal,
)
from .metal_retro import (
    MetalRetroConstructor,
    RetroStatus,
    metal_wl_hash,
    reaction_data_from_pattern,
)
from .pathfp import FoldedFingerprintConfig
from .predicates import compile_predicate_table
from .reaction import ReactionPattern
from .rejection import ChemistryRejection, RejectionReason

__all__ = [
    "MetalFoldedPrefilteredReactionMatcher",
    "metal_combine",
    "metal_concat_batches",
    "MetalMoleculeBatch",
    "MetalReactionDiagnostics",
    "MetalReactionMatcher",
    "metal_short_ring_wl_key",
    "metal_stack_mols",
]

#: This module's names for the two packed-layout dimensions. Aliases rather than restatements:
#: a disagreement with the shader is a buffer overrun, not a wrong answer.
#: Rows per WL keying dispatch. One row uploads `14 * 64 * 4` atom bytes plus `64 * 64` bond bytes
#: plus a count -- 7,684 -- so this slice is ~492 MB in flight, the same order as the retro
#: constructor's own 512 MB scratch budget and far inside any Apple Silicon working set. The keyer
#: is independent per molecule, so the slicing changes no token.
_WL_KEY_ROWS_PER_DISPATCH = 64 * 1024
#: Pairs per chunk when screening an explicit pair set. One chunk gathers two `(pairs, words)`
#: arrays of 64-bit words -- 256 bytes a pair at 2,048 bits -- so this caps the temporaries at
#: ~16 MB apiece however many pairs a layer offers.
_SCREEN_PAIRS_PER_CHUNK = 64 * 1024
_BUCKET = GPU_MATCH_ATOM_CAP
_ATOM_WORD_COUNT = ATOM_MASK_WORD_COUNT
# Cap the matcher's dense (pairs, max_assignments, 64) int32 assignment buffer per
# call so the reaction matcher stays memory-bounded as the pair count grows.
_ASSIGNMENT_BUFFER_BUDGET = 512 * 1024 * 1024
# The retro kernel allocates one device scratch slot per assignment; cap assignments
# per construct dispatch so that pool stays bounded. The exact stride is probed by
# metal_retro.mm; this estimate only chooses a conservative host chunk size.
_SCRATCH_BYTES_PER_ASSIGNMENT = 132 * 1024
# Monotonic tokens identifying an upload's contents. Distinct values never share bytes, so
# the device can cache on them; they are never reused, so a stale buffer cannot be served.
_UPLOAD_TOKENS = itertools.count(1)

# RetroStatus as uint32 words, so the gather can read the rows as an array.
_STATUS_WORDS = ctypes.sizeof(RetroStatus) // ctypes.sizeof(ctypes.c_uint32)
_STATUS_COMPONENT_COUNT = 0
_STATUS_COMPONENT_OVERFLOW = 1
_STATUS_ATOM_OVERFLOW = 2
_STATUS_EDITED_ATOM_OVERFLOW = 3
_STATUS_OUTPUT_BASE = 4
_STATUS_ORPHANED_COMPONENTS = 7
_STATUS_GROUPING_VIOLATIONS = 8

_RETRO_ASSIGNMENTS_PER_DISPATCH = max(
    1, (512 * 1024 * 1024) // _SCRATCH_BYTES_PER_ASSIGNMENT
)
# Swept, and it sets the smallest workload that runs at full speed rather than the speed of a
# large one: a call that fills one dispatch is the cheapest per precursor, so at 993/3971/15888
# assignments a dispatch the per-precursor minimum sat at >100, ~5-25 and 25 molecules on a corpus
# yielding ~500-860 assignments a molecule. That threshold is this constant divided by the corpus
# and owes nothing to GPU core count -- the 8-core M5 and the 20-core M5 Pro put it in the same
# place. Raising it is not free: the scratch pool is this many times
# `_SCRATCH_BYTES_PER_ASSIGNMENT`, so 15888 means a 2 GB device allocation out of the same unified
# memory the host results use. Left where it is because the measured gain was 4.40 -> 4.21
# us/precursor at 100 molecules, against 1.72x at full corpus size for delivering the precursors
# in pieces (`match_batches`), which is where the throughput actually was.
WlToken = tuple[int, int]


def _batch_indices(count: int, selector: Any) -> np.ndarray:
    """Resolve a `__getitem__` selector to a 1-D index array (mirrors
    `cuda_matcher.batch_indices`): reject scalar ints so the batch axis is kept."""
    if isinstance(selector, (int, np.integer)):
        raise TypeError(
            "molecule batch selection must preserve the batch axis; "
            "use a slice or one-dimensional index array."
        )
    if isinstance(selector, slice):
        return np.arange(count, dtype=np.int64)[selector]
    raw = np.asarray(selector)
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


def _adjacency_rows_from_bonds(bonds: np.ndarray) -> np.ndarray:
    """Per-atom adjacency bitsets (N, 64) from packed bond matrices (N, 64*64).

    `packbits` straight on the bond bytes, which allocates nothing but the result. Setting bit `j`
    of a word *is* what packing a row into bytes does, so saying it as a multiply by `1 << j`
    followed by a reduction asks numpy to materialise an `(N, 64, 64)` array of **uint64** to hold
    products it only wants in order to collapse them -- 1.6 GB at fifty thousand molecules, to
    produce 25 MB. Bit-identical and some 13 to 27 times faster from N = 10k to 200k, the gap
    widening with N because the old shape was bandwidth-bound on its own temporary.

    No `!= 0` either, which would cost an `(N, 64, 64)` bool. `packbits` sets a bit for any nonzero
    byte rather than testing the low bit -- checked over all 256 values -- and a bond byte is a
    union of `BondKind` flags (1, 2, 4, 8), so "present" and "nonzero" are the same question.

    `bitorder="little"` is the one load-bearing argument: bit `j` must mean atom `j`, and the
    default big-endian order reverses each group of eight, which permutes a molecule's neighbours
    while keeping the bit count right.

    **Slice before calling this.** The cost is linear in the batch, `__getitem__` carries `rows`
    forward as `None`, and a caller that needs adjacency for some precursors should gather them
    first rather than pay for the layer and discard most of it.
    """
    packed = np.packbits(bonds.reshape(-1, _BUCKET, _BUCKET), axis=2, bitorder="little")
    return packed.view(np.uint64)[:, :, 0]


@dataclass(frozen=True)
class MetalMoleculeBatch:
    r"""A host-side frontier batch of packed molecule arrays (the same `_pack_*` layout
    the Metal matcher and retro kernel consume).

    `atoms`/`bonds`/`rows` are `(N, 64*14)` / `(N, 64*64)` / `(N, 64)`. `rows` is lazy
    (`None`) for precursor batches and recomputed from `bonds` only when a (usually
    small) subset is re-matched, so building the full precursor batch stays cheap.
    Satisfies the search `MoleculeBatch` protocol (`len`, `__getitem__` with
    bool/int/slice).

    **Molecules only -- identity is not a field here.** `metal_short_ring_wl_key` keys whatever
    batch it is handed, on the device, so a frontier, a gather, a `metal_combine` union and a
    `metal_concat_batches` result are all just batches. That is what lets this arm lift through
    `keyed_match` exactly as the cpu arm does, and the matcher stay ignorant of identity.

    It was not always so, and the history is worth keeping because two of its conclusions were
    wrong. A `keys` field used to ride along, filled only by a fused WL pass over the `GpuMol`
    records `retro_construct` built -- so the keyer raised on anything the device had not built,
    which made the arm's `key` block **partial**. That is worse than slow: `vertsy.selectivity`
    keys a frontier gather to name a step's intended product, so no assembly containing a
    frontier-keying probe could be built at all.

    The first wrong conclusion was the fix. This docstring said the answer was "a packing and an
    export" -- manufacture an 8 KB interleaved record for a frontier that had no other use for one.
    CUDA had the shape all along: `wl_hash_molecules` takes parallel arrays and writes *fresh* hash
    arrays, so it keys any batch, and `tests/vesma/test_cuda_retro.py` keys a host-built batch on
    device bit-exact against the CPU codec. Metal's kernel now has that signature, and `GpuMol` is
    gone with the pass that needed it.

    The second was a measurement. An earlier revision claimed host keying cost "37s against 3.6s,
    a 10x regression paid by every run". Both figures came from `grep -oE "in [0-9]+\.[0-9]+s"`,
    which matches `1 failed in 3.6s` as readily as a pass -- the fast run was crashing early on
    this very refusal rather than running fast. Re-measured with the exit status checked: host
    keying passed in 35s where its absence failed in 3.6s. There was never a 10x regression to
    weigh. Keying on the device runs the same two-target assemblies at the cost the arm had
    before any of this, which is the claim worth recording; the absolute seconds are not, because
    every measurement in this file's history was taken on a host doing other work.
    """

    atom_counts: np.ndarray
    atoms: np.ndarray
    bonds: np.ndarray
    rows: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.atom_counts.shape[0])

    def adjacency_rows(self) -> np.ndarray:
        return (
            self.rows
            if self.rows is not None
            else _adjacency_rows_from_bonds(self.bonds)
        )

    def __getitem__(self, selector: Any) -> MetalMoleculeBatch:
        idx = _batch_indices(len(self), selector)
        return MetalMoleculeBatch(
            atom_counts=self.atom_counts[idx],
            atoms=self.atoms[idx],
            bonds=self.bonds[idx],
            rows=None if self.rows is None else self.rows[idx],
        )


def _empty_batch() -> MetalMoleculeBatch:
    return MetalMoleculeBatch(
        atom_counts=np.zeros(0, dtype=np.uint32),
        atoms=np.zeros((0, _ATOM_WORD_COUNT * _BUCKET), dtype=np.uint32),
        bonds=np.zeros((0, _BUCKET * _BUCKET), dtype=np.uint8),
        rows=np.zeros((0, _BUCKET), dtype=np.uint64),
    )


@dataclass(frozen=True)
class MetalReactionDiagnostics:
    component_overflow_assignments: int = 0
    precursor_atom_overflow_components: int = 0
    precursor_atom_overflow_assignments: int = 0
    edited_atom_overflow_assignments: int = 0
    # Per-stage seconds for the match+split pipeline. The Metal retro kernel fuses bond
    # bond edits + connected-component split + hydrogen/ring/aromaticity re-perception into a
    # single `retro_construct` kernel (not separable without de-fusing), so those share
    # `retro_construct_kernel_seconds`; the WL identity hash is a separate kernel
    # (`wl_kernel_seconds`, both GPU-measured via command-buffer GPU timestamps). The host
    # stages (`assignment_collect_seconds`, `precursor_gather_seconds`) and the matcher
    # dispatch (`match_seconds`) are wall-clocked; `match_kernel_seconds` is GPU-only.
    # `retro_seconds` is the wall time of the
    # construct dispatches (GPU kernels + host buffer setup/copy-back).
    match_seconds: float = 0.0
    # Recursive-SMARTS roots for the batch, every level, ahead of the matcher dispatch.
    predicate_seconds: float = 0.0
    match_buffer_seconds: float = 0.0
    match_dispatch_seconds: float = 0.0
    match_kernel_seconds: float = 0.0
    match_copyback_seconds: float = 0.0
    match_other_host_seconds: float = 0.0
    match_template_upload_bytes: int = 0
    match_molecule_upload_bytes: int = 0
    match_pair_upload_bytes: int = 0
    match_output_copyback_bytes: int = 0
    match_assignment_count: int = 0
    match_positive_pairs: int = 0
    match_reserved_slots: int = 0
    match_retry_pairs: int = 0
    assignment_collect_seconds: float = 0.0
    retro_seconds: float = 0.0
    retro_construct_kernel_seconds: float = 0.0
    wl_kernel_seconds: float = 0.0
    """Retired, and permanently zero.

    The matcher used to dispatch a WL pass over the records `retro_construct` had just built, and
    this timed it. Keying is `metal_short_ring_wl_key` now -- the arm's `key` block, a plain
    `Batch -> Sequence[K]` function that the matcher does not call and should not know about -- so
    there is no WL dispatch here left to time. The field stays at zero rather than being deleted so
    that records written before the split still compare.

    Measured before retiring it, because a cost going from timed to untimed is worth a number
    rather than a shrug: over a 17,026-row batch the whole keyer is 2.3% of the match it follows
    (0.041 s against 1.837 s), of which the kernel is 0.014 s and the rest upload and copy-back.
    That is why it gets a sentence here instead of an accumulator on the arm.
    """
    precursor_gather_seconds: float = 0.0
    # Applications the device refused, counted by reason in the same closed-set layout the
    # CPU uses, so the two backends report refusals in one vocabulary.
    rejections: tuple[int, ...] = ()
    undescribed_components: int = 0
    #: Assignments the kernel gave up on structurally rather than chemically, by code: a mapped
    #: LHS atom landing outside the product, a mapped product atom with no edited slot, an RHS
    #: edge endpoint that never got one. They are refusals -- `UNSUPPORTED_TEMPLATE`, so they
    #: land in `rejections` like any other -- and this says which of the three, which the reason
    #: cannot. Three counters, one per code, merged elementwise by `merged_rejections` -- the
    #: same shape and the same merge as `rejections`, for the same reason. Measured zero over the
    #: parity corpus, 110,449 applications of 42,554 rules, so the branches are unreachable in
    #: practice; counted so that stops being an assumption.
    inconsistent_assignments: tuple[int, ...] = ()
    #: Undescribed components the edit *severed*, which is the subset that refuses the
    #: application. `undescribed_components` is the superset and includes spectators -- a
    #: component the edit never touched, as a unimolecular forward rule leaves beside the one it
    #: edits -- which are not refused. So the superset alone cannot say why an application was
    #: refused, and this is the counter that can.
    orphaned_components: int = 0
    #: Applications whose product disagreed with the RHS's zero-level grouping, the third ground
    #: for refusal beside an orphan and the atom cap.
    grouping_violations: int = 0
    #: Joining the per-dispatch precursor batches into the one returned batch. Zero when the
    #: caller streams, and it used to be invisible: `total_seconds` was stamped before the join,
    #: so a 2,881-molecule call reported 14.9 s of a 26.4 s wall and the missing 11.5 s was the
    #: single largest line in the budget.
    precursor_concat_seconds: float = 0.0
    #: Wall spent parked at a `yield` in `match_batches` while the consumer worked, excluded from
    #: `total_seconds` and recorded here instead. A generator's clock keeps running while it is
    #: suspended, so stamping `total_seconds` as elapsed-since-entry billed the consumer's keyer
    #: to the matcher: a folded 100-target route recovery reported 39.6 s of matcher time whose
    #: own stages summed to 28.8 s, and the missing 10.8 s was `keyed_match_folding` keying the
    #: batches it had just been handed. Zero for `match`, whose consumer only appends to a list.
    consumer_seconds: float = 0.0
    total_seconds: float = 0.0


def merged_rejections(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    """Elementwise sum of two refusal counter tuples, either of which may be unset.

    The counters are a tuple rather than a mapping because merging is elementwise addition
    and the layout is the same array of counters the device kernel keeps.
    """
    if not left:
        return right
    if not right:
        return left
    return tuple(a + b for a, b in zip(left, right, strict=True))


def add_metal_reaction_diagnostics(
    left: MetalReactionDiagnostics,
    right: MetalReactionDiagnostics,
) -> MetalReactionDiagnostics:
    return MetalReactionDiagnostics(
        component_overflow_assignments=left.component_overflow_assignments
        + right.component_overflow_assignments,
        precursor_atom_overflow_components=left.precursor_atom_overflow_components
        + right.precursor_atom_overflow_components,
        precursor_atom_overflow_assignments=left.precursor_atom_overflow_assignments
        + right.precursor_atom_overflow_assignments,
        edited_atom_overflow_assignments=left.edited_atom_overflow_assignments
        + right.edited_atom_overflow_assignments,
        match_seconds=left.match_seconds + right.match_seconds,
        predicate_seconds=left.predicate_seconds + right.predicate_seconds,
        match_buffer_seconds=left.match_buffer_seconds + right.match_buffer_seconds,
        match_dispatch_seconds=left.match_dispatch_seconds
        + right.match_dispatch_seconds,
        match_kernel_seconds=left.match_kernel_seconds + right.match_kernel_seconds,
        match_copyback_seconds=left.match_copyback_seconds
        + right.match_copyback_seconds,
        match_other_host_seconds=left.match_other_host_seconds
        + right.match_other_host_seconds,
        match_template_upload_bytes=left.match_template_upload_bytes
        + right.match_template_upload_bytes,
        match_molecule_upload_bytes=left.match_molecule_upload_bytes
        + right.match_molecule_upload_bytes,
        match_pair_upload_bytes=left.match_pair_upload_bytes
        + right.match_pair_upload_bytes,
        match_output_copyback_bytes=left.match_output_copyback_bytes
        + right.match_output_copyback_bytes,
        match_assignment_count=left.match_assignment_count
        + right.match_assignment_count,
        match_positive_pairs=left.match_positive_pairs + right.match_positive_pairs,
        match_reserved_slots=left.match_reserved_slots + right.match_reserved_slots,
        match_retry_pairs=left.match_retry_pairs + right.match_retry_pairs,
        assignment_collect_seconds=left.assignment_collect_seconds
        + right.assignment_collect_seconds,
        retro_seconds=left.retro_seconds + right.retro_seconds,
        retro_construct_kernel_seconds=left.retro_construct_kernel_seconds
        + right.retro_construct_kernel_seconds,
        wl_kernel_seconds=left.wl_kernel_seconds + right.wl_kernel_seconds,
        rejections=merged_rejections(left.rejections, right.rejections),
        undescribed_components=left.undescribed_components
        + right.undescribed_components,
        orphaned_components=left.orphaned_components + right.orphaned_components,
        grouping_violations=left.grouping_violations + right.grouping_violations,
        inconsistent_assignments=merged_rejections(
            left.inconsistent_assignments, right.inconsistent_assignments
        ),
        precursor_gather_seconds=left.precursor_gather_seconds
        + right.precursor_gather_seconds,
        precursor_concat_seconds=left.precursor_concat_seconds
        + right.precursor_concat_seconds,
        consumer_seconds=left.consumer_seconds + right.consumer_seconds,
        total_seconds=left.total_seconds + right.total_seconds,
    )


def metal_combine(
    precursors: MetalMoleculeBatch,
    members: np.ndarray,
    indptr: np.ndarray,
) -> tuple[MetalMoleculeBatch, np.ndarray]:
    """One packed molecule per group, each the disjoint union of the group's members.

    What a multi-component forward pattern has to match across: a subgraph match is a match into one
    graph, so two starting materials have to be one graph first. Every per-atom property survives
    standing two molecules side by side -- ring membership included, since a disjoint union creates
    no ring -- so each group is a copy of its members' atom blocks at successive offsets and their
    bond matrices down the diagonal, with no bond written across.

    ``members`` and ``indptr`` are a CSR grouping of rows in ``precursors``, the same shape
    ``vertsy.tropical.Shape`` keeps its precursors in. **No molecule is ever unpacked**: this reads
    and writes the packed layout the matcher already consumes, which is the point of doing it here
    rather than through :func:`vesma.cmol.cmol_union` and a re-pack.

    Answers the batch and a **buildable mask**. A group whose atoms exceed the 64-atom bucket is
    written as an empty row rather than truncated -- a truncated molecule is a different molecule
    and would be measured as one -- and its mask entry is false. The mask is the part that matters:
    an empty row matches nothing, so a caller that could not tell it apart from a real row would
    read *silent* and charge a device atom budget as a broken pairing.
    """
    groups = len(indptr) - 1
    buildable = np.ones(groups, dtype=np.bool_)
    atom_counts = np.zeros(groups, dtype=np.uint32)
    atoms = np.zeros((groups, _ATOM_WORD_COUNT * _BUCKET), dtype=np.uint32)
    bonds = np.zeros((groups, _BUCKET * _BUCKET), dtype=np.uint8)
    source_atoms = precursors.atoms.reshape(-1, _BUCKET, _ATOM_WORD_COUNT)
    source_bonds = precursors.bonds.reshape(-1, _BUCKET, _BUCKET)
    out_atoms = atoms.reshape(groups, _BUCKET, _ATOM_WORD_COUNT)
    out_bonds = bonds.reshape(groups, _BUCKET, _BUCKET)
    for group in range(groups):
        rows = members[indptr[group] : indptr[group + 1]]
        sizes = precursors.atom_counts[rows].astype(np.int64)
        if int(sizes.sum()) > _BUCKET:
            buildable[group] = False
            continue
        at = 0
        for row, size in zip(rows.tolist(), sizes.tolist()):
            out_atoms[group, at : at + size] = source_atoms[row, :size]
            out_bonds[group, at : at + size, at : at + size] = source_bonds[
                row, :size, :size
            ]
            at += size
        atom_counts[group] = at
    return (
        MetalMoleculeBatch(
            atom_counts=atom_counts,
            atoms=atoms,
            bonds=bonds,
            rows=None,
        ),
        buildable,
    )


def metal_screen(
    patterns: Sequence[str],
) -> Callable[[MetalMoleculeBatch, Sequence[tuple[int, int]]], np.ndarray]:
    """A ``vertsy.filters.Screen`` on the device: which ``(pattern, molecule row)`` pairs match.

    Satisfies that protocol without naming it, exactly as :func:`metal_combine` satisfies
    ``vertsy.selectivity.Combine``: the injected operation is declared up in :mod:`vertsy`, where the
    algebra is, and implemented down here, where the packed layout is. Patterns arrive as SMARTS
    rather than as veto rows so nothing in :mod:`vesma` has to know what a veto is.

    **No kernel changes, and no molecule is unpacked.** A veto is a plain subgraph question, so this
    is ``dispatch_match_packed`` over the batch's own arrays with ``max_assignments=1``: one matching
    assignment answers "does this pattern occur", and asking for more would enumerate every
    occurrence to throw them away. The patterns are packed once and uploaded under a stable token,
    so the per-layer cost is the dispatch and nothing else.

    A batch row wider than the bucket cannot occur here -- every packed structure is already
    ``_BUCKET`` slots -- and an oversized *pattern* is refused at build time, like every other
    pattern this backend compiles.
    """
    compiled = [gmol_from_smarts(text)[0] for text in patterns]
    table = compile_predicate_table(compiled)
    program = metal_predicate_program(table)
    matcher = MetalMatcher()
    oversized = [
        (text, pattern.atom_count)
        for text, pattern in zip(patterns, compiled, strict=True)
        if pattern.atom_count > _BUCKET
    ]
    if oversized:
        raise ValueError(
            f"a device screen holds patterns of at most {_BUCKET} atoms; "
            f"{oversized[0][0]!r} has {oversized[0][1]}."
        )
    if compiled:
        p_atoms = np.concatenate([_pack_atom_masks(p, _BUCKET) for p in compiled])
        p_rows = np.concatenate([_pack_adjacency_rows(p, _BUCKET) for p in compiled])
        p_bonds = np.concatenate([_pack_bond_matrix(p, _BUCKET) for p in compiled])
        p_groups = np.concatenate([pack_atom_groups(p, _BUCKET) for p in compiled])
        p_preds = pack_collection_atom_predicates(table, compiled, _BUCKET)
        p_counts = np.array([p.atom_count for p in compiled], dtype=np.uint32)
    # A stable, process-unique token for the packed pattern contents, which are immutable for this
    # screen's lifetime -- the same contract `MetalFoldedPrefilteredReactionMatcher` uses for its
    # rule table, and from the same counter, so no two packed uploads in flight can share a token.
    token = next(_UPLOAD_TOKENS)

    def screen(
        batch: MetalMoleculeBatch, pairs: Sequence[tuple[int, int]]
    ) -> np.ndarray:
        if not pairs or not compiled:
            return np.zeros(len(pairs), dtype=bool)
        m_atoms = batch.atoms.reshape(-1)
        m_rows = batch.adjacency_rows().reshape(-1)
        m_bonds = batch.bonds.reshape(-1)
        m_counts = np.ascontiguousarray(batch.atom_counts, dtype=np.uint32)
        status, _ = matcher.dispatch_match_packed(
            p_atoms=p_atoms,
            p_rows=p_rows,
            p_bonds=p_bonds,
            p_groups=p_groups,
            p_preds=p_preds,
            p_counts=p_counts,
            m_atoms=m_atoms,
            m_rows=m_rows,
            m_bonds=m_bonds,
            m_counts=m_counts,
            pair_indices=pairs,
            max_assignments=1,
            roots=evaluate_predicate_roots(
                matcher,
                program,
                m_atoms=m_atoms,
                m_rows=m_rows,
                m_bonds=m_bonds,
                m_counts=m_counts,
            ),
            pattern_token=token,
        )
        return np.asarray(status[:, 0], dtype=np.int64) > 0

    return screen


def metal_stack_mols(mols: Sequence[CMol]) -> MetalMoleculeBatch:
    """Pack CMols into a Metal frontier with exact bond topology.

    Refuses a molecule larger than the bucket rather than letting the pack fail. Every packed
    structure here is ``_BUCKET`` atom slots wide, so an oversized molecule used to reach
    ``_pack_atom_masks`` and raise `could not broadcast input array from shape (68,) into shape
    (64,)` from inside numpy -- a crash that took the whole sweep with it, where an oversized
    *precursor* has always been refused and counted. `MetalMatcher.match` guards its own inputs;
    this is the frontier path, which did not.

    The refusal carries ``PRECURSOR_CAPACITY_EXCEEDED``, whose name predates this use: the cause
    is the same 64-slot capacity, and a caller counting refusals by reason wants them together.
    """
    gmols = [gmol_from_cmol(mol) for mol in mols]
    if not gmols:
        return _empty_batch()
    oversized = [g.atom_count for g in gmols if g.atom_count > _BUCKET]
    if oversized:
        raise ChemistryRejection(
            f"the Metal frontier packs {_BUCKET} atoms per molecule and was given "
            f"{', '.join(str(count) for count in sorted(oversized, reverse=True))}; "
            "search this molecule on the cpu arm, which has no such cap",
            reason=RejectionReason.PRECURSOR_CAPACITY_EXCEEDED,
        )
    atoms = np.stack([_pack_atom_masks(g, _BUCKET) for g in gmols])
    bonds = np.stack(
        [_pack_bond_matrix(g, _BUCKET, annotate_topology=True) for g in gmols]
    )
    rows = np.stack([_pack_adjacency_rows(g, _BUCKET) for g in gmols])
    counts = np.array([g.atom_count for g in gmols], dtype=np.uint32)
    return MetalMoleculeBatch(atom_counts=counts, atoms=atoms, bonds=bonds, rows=rows)


def metal_short_ring_wl_key(batch: MetalMoleculeBatch) -> list[WlToken]:
    """Short-ring WL identity for any batch this arm can hold, computed on the device.

    The arm's `key` block, and the whole of it: a `Batch -> Sequence[K]` function, the same type
    the cpu arm's keyer has, so both lift through `keyed_match` and the matcher stays ignorant of
    identity. Nothing is stored and nothing can be missing -- a frontier, a gather, a `metal_combine`
    union and a `metal_concat_batches` result are all just batches, and all four key.

    This used to read a `keys` field that `retro_construct`'s fused pass had filled, and so raised
    on anything the device had not built. That made the block *partial*, which is a worse thing
    than slow: no assembly containing a frontier-keying probe could be built at all, because
    `vertsy.selectivity` keys a frontier gather to name a step's intended product.

    Chunked for the same reason it is total. Each dispatch uploads the rows it keys -- 7,684 bytes
    apiece across atoms, bonds and counts -- and `metal_wl_hash` allocates fresh buffers per call,
    so an unchunked keyer over a large enough batch would ask for a buffer the device refuses and
    fail on a batch construction itself handles by chunking. A keyer that raises past some size is
    partial again, just at a different boundary, so it does not get to be: the kernel is
    independent per molecule, which makes slicing exact rather than approximate.

    The bound is generous -- this machine caps a single buffer at 8.9 GiB, about 2.3M rows, against
    the ~17k of the largest batch measured and ~1.3k per real construct dispatch -- so the chunk
    size is chosen to keep one dispatch's upload well inside the working set rather than to track
    any limit precisely. Keying is 2.3% of a match either way (0.041 s of 1.837 s over 17,026 rows,
    of which the kernel is 0.014 s), so there is no throughput reason to prefer a bigger slice.
    """
    rows = len(batch)
    if rows <= _WL_KEY_ROWS_PER_DISPATCH:
        keyed = metal_wl_hash(batch.atoms, batch.bonds, batch.atom_counts)
        return list(zip(keyed.hash_lo.tolist(), keyed.hash_hi.tolist(), strict=True))

    tokens: list[WlToken] = []
    for start in range(0, rows, _WL_KEY_ROWS_PER_DISPATCH):
        stop = start + _WL_KEY_ROWS_PER_DISPATCH
        keyed = metal_wl_hash(
            batch.atoms[start:stop],
            batch.bonds[start:stop],
            batch.atom_counts[start:stop],
        )
        tokens.extend(zip(keyed.hash_lo.tolist(), keyed.hash_hi.tolist(), strict=True))
    return tokens


class MetalReactionMatcher:
    """Reaction matcher backed by the Metal matcher and construction/key kernels.

    Conforms to `vesma.matcher.ReactionMatcher`: `match(mols, pairs=None)` runs the
    fused matcher to enumerate assignments, ``retro_construct`` to edit and perceive
    precursor graphs, and the standalone WL kernel to fill their keys. It returns
    `(precursors, hits)`, where precursors is a packed `MetalMoleculeBatch` and hits
    are `(mol_idx, rule_id, app_idx, precursor_idx)` rows indexing it (co-precursors of
    one assignment share `app_idx`). `pairs=None` means all mols x all rules; an
    explicit `(mol_idx, rule_id)` set restricts the work. No deduplication occurs here,
    so the precursor column is the identity: one precursor row per hit. Matching first
    reserves `assignment_probe_cap` slots per pair and replays saturated pairs at
    `max_assignments`, preserving the full per-pair limit with smaller usual buffers.
    """

    def __init__(
        self,
        rules: Sequence[ReactionPattern],
        *,
        max_assignments: int = 256,
        assignment_probe_cap: int = 8,
        matcher: MetalMatcher | None = None,
        constructor: MetalRetroConstructor | None = None,
    ) -> None:
        if max_assignments < 1:
            raise ValueError("max_assignments must be positive")
        if assignment_probe_cap < 1:
            raise ValueError("assignment_probe_cap must be positive")
        self._rules = list(rules)
        self._reaction_data = [reaction_data_from_pattern(rule) for rule in self._rules]
        # The reaction table is built once and never mutated, so it earns a stable token:
        # the device uploads it on the first dispatch and reuses it for every later one.
        self._reaction_token = next(_UPLOAD_TOKENS)
        self._matcher = matcher if matcher is not None else MetalMatcher()
        self._constructor = (
            constructor if constructor is not None else MetalRetroConstructor()
        )
        self._max_assignments = max_assignments
        self._assignment_probe_cap = min(assignment_probe_cap, max_assignments)
        self.last_diagnostics: MetalReactionDiagnostics | None = None

        patterns = [rule.lhs for rule in self._rules]
        self._n_patterns = len(patterns)
        # Built once with the rule table; only the roots are per batch.
        self._predicates = compile_predicate_table(patterns)
        self._predicate_program = metal_predicate_program(self._predicates)
        if patterns:
            self._p_atoms = np.concatenate(
                [_pack_atom_masks(p, _BUCKET) for p in patterns]
            )
            self._p_rows = np.concatenate(
                [_pack_adjacency_rows(p, _BUCKET) for p in patterns]
            )
            self._p_bonds = np.concatenate(
                [_pack_bond_matrix(p, _BUCKET) for p in patterns]
            )
            self._p_groups = np.concatenate(
                [pack_atom_groups(p, _BUCKET) for p in patterns]
            )
            self._p_preds = pack_collection_atom_predicates(
                self._predicates, patterns, _BUCKET
            )
            self._p_counts = np.array([p.atom_count for p in patterns], dtype=np.uint32)
        else:
            self._p_atoms = np.zeros(0, dtype=np.uint32)
            self._p_rows = np.zeros(0, dtype=np.uint64)
            self._p_bonds = np.zeros(0, dtype=np.uint8)
            self._p_groups = np.zeros(0, dtype=np.uint32)
            self._p_preds = np.zeros(0, dtype=np.uint32)
            self._p_counts = np.zeros(0, dtype=np.uint32)

    def match(
        self,
        mols: Sequence[CMol] | MetalMoleculeBatch,
        pairs: PairArray | Sequence[Pair] | None = None,
    ) -> tuple[MetalMoleculeBatch, HitArray]:
        """Apply every requested (molecule, rule) pair and return the precursors and hits.

        A fold over `match_batches`, which is where the work is. Joining the pieces is what costs
        memory rather than time: a precursor row is ~7,684 bytes on the host, so applying 2,881
        products to 42,554 rules holds 25 GB of them and peaks near 50 GB with the join's
        destination allocated beside the pieces. On a 64 GiB machine that measured 3.56M
        compressor pages and 21,908 swapouts and cost 1.72x against folding the pieces instead;
        on a 16 GiB machine, 3.4x.

        So prefer `match_batches` where the result is about to be collapsed anyway --
        `matcher.keyed_match_folding` does exactly that, and it is what the search uses. This
        stays for the callers that genuinely want one array: parity checks the whole multiset,
        `vertsy.selectivity` keys a frontier gather, and a test wants something to compare against.

        Gather-plus-join costs a flat ~0.5 us a precursor when the pieces are folded and grows 9x
        (0.48 -> 4.42 us) over 1 to 2,881 molecules when they are joined here. Absolute wall times
        on this host drift by a quarter between measurement sessions, so compare the two within
        one run and never against a recorded number. `make bench-metal-precursor-streaming` is the
        A/B; `docs/reports/metal-workload-sweet-spot-2026-11-05.md` is how the mechanism was
        pinned down.
        """
        parts: list[MetalMoleculeBatch] = []
        hit_parts: list[HitArray] = []
        for chunk_batch, chunk_hits in self.match_batches(mols, pairs):
            parts.append(chunk_batch)
            hit_parts.append(chunk_hits)

        concat_started = perf_counter()
        if parts:
            joined = (
                metal_concat_batches(parts),
                concat_hits(hit_parts, [len(part) for part in parts]),
            )
        else:
            joined = (_empty_batch(), empty_hits())
        concat_s = perf_counter() - concat_started
        # The generator stamped everything else. This phase is the join, which only exists for a
        # caller that asked for one array, so it is counted here rather than in there -- and reads
        # zero for everyone who folds instead.
        if self.last_diagnostics is not None:
            self.last_diagnostics = replace(
                self.last_diagnostics,
                precursor_concat_seconds=concat_s,
                total_seconds=self.last_diagnostics.total_seconds + concat_s,
            )
        return joined

    def match_batches(
        self,
        mols: Sequence[CMol] | MetalMoleculeBatch,
        pairs: PairArray | Sequence[Pair] | None = None,
    ) -> Iterator[tuple[MetalMoleculeBatch, HitArray]]:
        """Apply the pairs, yielding each `retro_construct` sub-dispatch's precursors as gathered.

        Device work per sub-dispatch is fixed -- `_RETRO_ASSIGNMENTS_PER_DISPATCH` assignments
        however large the call -- so yielding rather than accumulating is what keeps the *host*
        side size-independent too: nothing is retained between pieces, and the allocator reuses
        the same pages for the next gather instead of faulting fresh ones.

        A yielded batch is freshly gathered and owned by the consumer, so retaining it is safe;
        the arrays past each precursor's `atom_count` are whatever the reused output mirror held
        and carry no meaning. Hit rows index **the batch they arrive with** in column 3, while
        columns 0-2 are already global, which is the contract `matcher.BatchedMatch` states.

        Abandoning the iterator leaves the remaining sub-dispatches un-run and `last_diagnostics`
        describing only the pieces that were pulled. It is still recorded, so an abandoned call is
        reported rather than silently absent.
        """
        total_started = perf_counter()
        batch = mols if isinstance(mols, MetalMoleculeBatch) else metal_stack_mols(mols)
        n_mols = len(batch)
        if n_mols == 0 or not self._rules:
            self.last_diagnostics = MetalReactionDiagnostics()
            return

        if pairs is None:
            # All (pattern_idx, mol_idx) pairs, rule-major, built vectorized.
            rule_col = np.repeat(np.arange(self._n_patterns), n_mols)
            mol_col = np.tile(np.arange(n_mols), self._n_patterns)
            matcher_pairs = np.stack([rule_col, mol_col], axis=1).astype(np.int64)
        else:
            # ReactionMatcher pairs are (mol_idx, rule_id); the matcher wants
            # (pattern_idx, mol_idx). Accept a (N, 2) array (fast path, no Python
            # round-trip) or a sequence of tuples, then swap columns.
            pairs_arr = pair_index_array(
                pairs,
                first_count=n_mols,
                second_count=self._n_patterns,
                first_name="molecule",
                second_name="rule",
            )
            matcher_pairs = np.ascontiguousarray(pairs_arr[:, ::-1])
        if matcher_pairs.shape[0] == 0:
            self.last_diagnostics = MetalReactionDiagnostics(
                total_seconds=perf_counter() - total_started
            )
            return

        # Most pairs have no assignments. Probe with a small dense buffer, then
        # retry only pairs that filled it using the full per-pair limit.
        chunk_size = max(
            1,
            _ASSIGNMENT_BUFFER_BUDGET // (self._assignment_probe_cap * _BUCKET * 4),
        )
        m_atoms = batch.atoms.reshape(-1)
        m_rows = batch.adjacency_rows().reshape(-1)
        m_bonds = batch.bonds.reshape(-1)
        m_counts = np.ascontiguousarray(batch.atom_counts, dtype=np.uint32)
        predicate_started = perf_counter()
        roots = evaluate_predicate_roots(
            self._matcher,
            self._predicate_program,
            m_atoms=m_atoms,
            m_rows=m_rows,
            m_bonds=m_bonds,
            m_counts=m_counts,
        )
        predicate_s = perf_counter() - predicate_started
        product_atoms = batch.atoms.reshape(-1)
        # Reaction construction/perception operates on class-only scratch.
        product_bonds = bond_kind_matrix(batch.bonds).reshape(-1)

        diagnostics = MetalReactionDiagnostics()
        # Per-stage wall/GPU-second accumulators, summed across matcher/retro chunks and
        # folded into `diagnostics` at the end (see MetalReactionDiagnostics).
        match_s = collect_s = retro_s = 0.0
        match_buffer_s = match_dispatch_s = match_kernel_s = match_copyback_s = 0.0
        match_other_host_s = 0.0
        match_template_bytes = match_molecule_bytes = match_pair_bytes = 0
        match_output_bytes = 0
        match_assignment_count = match_positive_pairs = match_reserved_slots = 0
        match_retry_pairs = 0
        construct_kernel_s = wl_kernel_s = gather_s = consumer_s = 0.0
        app_offset = 0
        try:
            for start in range(0, len(matcher_pairs), chunk_size):
                chunk_pairs = matcher_pairs[start : start + chunk_size]
                match_started = perf_counter()
                status_out, assignments_out = self._matcher.dispatch_match_packed(
                    p_atoms=self._p_atoms,
                    p_rows=self._p_rows,
                    p_bonds=self._p_bonds,
                    p_groups=self._p_groups,
                    p_preds=self._p_preds,
                    p_counts=self._p_counts,
                    roots=roots,
                    m_atoms=m_atoms,
                    m_rows=m_rows,
                    m_bonds=m_bonds,
                    m_counts=m_counts,
                    pair_indices=chunk_pairs,
                    max_assignments=self._assignment_probe_cap,
                    copy_valid_only=True,
                    pattern_token=self._reaction_token,
                )
                match_s += perf_counter() - match_started
                profiles = [self._matcher.last_dispatch_profile]

                collect_started = perf_counter()
                counts = status_out[:, 0].astype(np.int64)
                if counts.size and int(counts.max()) > self._assignment_probe_cap:
                    raise RuntimeError(
                        "MetalReactionMatcher assignment cap exceeded "
                        f"({int(counts.max())} > {self._assignment_probe_cap}); "
                        "raise max_assignments."
                    )
                saturated = (
                    np.flatnonzero(counts == self._assignment_probe_cap)
                    if self._assignment_probe_cap < self._max_assignments
                    else np.empty(0, dtype=np.int64)
                )
                match_retry_pairs += len(saturated)
                match_reserved_slots += (
                    len(counts) * self._assignment_probe_cap
                    + len(saturated) * self._max_assignments
                )
                # A saturated pair is discarded from the probe result and replayed at
                # the full cap. Preserve pair-major order for downstream application IDs.
                counts[saturated] = 0
                valid = np.arange(self._assignment_probe_cap)[None, :] < counts[:, None]
                assignment_parts = [assignments_out[valid]]
                pair_positions = [np.repeat(np.arange(len(counts)), counts)]
                retry_positive_pairs = 0
                retry_wall_s = 0.0
                retry_chunk_size = max(
                    1,
                    _ASSIGNMENT_BUFFER_BUDGET // (self._max_assignments * _BUCKET * 4),
                )
                for retry_start in range(0, len(saturated), retry_chunk_size):
                    retry_positions = saturated[
                        retry_start : retry_start + retry_chunk_size
                    ]
                    match_started = perf_counter()
                    retry_status, retry_out = self._matcher.dispatch_match_packed(
                        p_atoms=self._p_atoms,
                        p_rows=self._p_rows,
                        p_bonds=self._p_bonds,
                        p_groups=self._p_groups,
                        p_preds=self._p_preds,
                        p_counts=self._p_counts,
                        roots=roots,
                        m_atoms=m_atoms,
                        m_rows=m_rows,
                        m_bonds=m_bonds,
                        m_counts=m_counts,
                        pair_indices=chunk_pairs[retry_positions],
                        max_assignments=self._max_assignments,
                        copy_valid_only=True,
                        pattern_token=self._reaction_token,
                    )
                    elapsed = perf_counter() - match_started
                    match_s += elapsed
                    retry_wall_s += elapsed
                    profiles.append(self._matcher.last_dispatch_profile)
                    retry_counts = retry_status[:, 0].astype(np.int64)
                    retry_positive_pairs += int(np.count_nonzero(retry_counts))
                    retry_valid = (
                        np.arange(self._max_assignments)[None, :]
                        < retry_counts[:, None]
                    )
                    assignment_parts.append(retry_out[retry_valid])
                    pair_positions.append(np.repeat(retry_positions, retry_counts))

                for profile in profiles:
                    match_buffer_s += profile.buffer_seconds
                    match_dispatch_s += profile.dispatch_seconds
                    match_kernel_s += profile.kernel_seconds
                    match_copyback_s += profile.copyback_seconds
                    match_other_host_s += profile.other_host_seconds
                    match_template_bytes += profile.template_upload_bytes
                    match_molecule_bytes += profile.molecule_upload_bytes
                    match_pair_bytes += profile.pair_upload_bytes
                    match_output_bytes += profile.output_copyback_bytes

                if len(saturated):
                    assignments = np.concatenate(assignment_parts)
                    positions = np.concatenate(pair_positions)
                    order = np.argsort(positions, kind="stable")
                    assignments = assignments[order]
                    positions = positions[order]
                else:
                    assignments = assignment_parts[0]
                    positions = pair_positions[0]
                match_assignment_count += len(assignments)
                match_positive_pairs += (
                    int(np.count_nonzero(counts)) + retry_positive_pairs
                )
                if not len(assignments):
                    collect_s += perf_counter() - collect_started - retry_wall_s
                    continue
                # retro_pairs: (n_assignments, 2) = (rule_idx, mol_idx) per assignment.
                retro_pairs = np.asarray(chunk_pairs[positions], dtype=np.int64)
                collect_s += perf_counter() - collect_started - retry_wall_s

                # The retro kernel allocates a device scratch slot per assignment
                # (~126 KiB currently); sub-chunk so that pool stays memory-bounded regardless of
                # how many assignments this matcher chunk produced.
                # The product arrays do not change across this call's chunks, so one token
                # covers them all and the upload happens once rather than per chunk.
                products_token = next(_UPLOAD_TOKENS)
                # Refusals accumulate across sub-dispatches: each reports only its own slice.
                for sub in range(
                    0, retro_pairs.shape[0], _RETRO_ASSIGNMENTS_PER_DISPATCH
                ):
                    sub_assign = assignments[
                        sub : sub + _RETRO_ASSIGNMENTS_PER_DISPATCH
                    ]
                    sub_pairs = retro_pairs[sub : sub + _RETRO_ASSIGNMENTS_PER_DISPATCH]
                    retro_started = perf_counter()
                    produced = self._constructor.construct_packed(
                        reaction_token=self._reaction_token,
                        products_token=products_token,
                        product_atoms=product_atoms,
                        product_bonds=product_bonds,
                        product_counts=m_counts,
                        unique_product_count=n_mols,
                        reactions=self._reaction_data,
                        assignments=sub_assign,
                        pair_indices=sub_pairs,
                    )
                    retro_s += perf_counter() - retro_started
                    construct_kernel_s += self._constructor.last_construct_seconds
                    wl_kernel_s += self._constructor.last_wl_seconds
                    diagnostics = add_metal_reaction_diagnostics(
                        diagnostics,
                        MetalReactionDiagnostics(
                            component_overflow_assignments=self._constructor.last_overflow_count,
                            precursor_atom_overflow_components=self._constructor.last_atom_overflow_component_count,
                            precursor_atom_overflow_assignments=self._constructor.last_atom_overflow_count,
                            edited_atom_overflow_assignments=self._constructor.last_edited_atom_overflow_count,
                            rejections=self._constructor.last_rejections,
                            undescribed_components=(
                                self._constructor.last_undescribed_components
                            ),
                            orphaned_components=(
                                self._constructor.last_orphaned_components
                            ),
                            grouping_violations=(
                                self._constructor.last_grouping_violations
                            ),
                            inconsistent_assignments=(
                                self._constructor.last_inconsistent_assignments
                            ),
                        ),
                    )
                    gather_started = perf_counter()
                    chunk_batch, chunk_hits = _precursor_batch(produced, sub_pairs)
                    gather_s += perf_counter() - gather_started
                    if len(chunk_batch):
                        chunk_hits[:, 2] += app_offset  # globally unique app_idx
                        # Handed over and dropped. Not retaining it is the point: the allocator
                        # reuses the same pages for the next sub-dispatch instead of faulting fresh
                        # ones, so the gather stops touching new memory. A consumer that wants to
                        # keep it may -- it is freshly gathered, not a view of the output mirror.
                        #
                        # Timed in a `try/finally` because the clock does not stop while a
                        # generator is suspended: everything the consumer does between pulls would
                        # otherwise land in `total_seconds` as if the matcher had spent it. The
                        # `finally` also catches `GeneratorExit`, so an abandoned pull is not
                        # billed either.
                        parked_started = perf_counter()
                        try:
                            yield chunk_batch, chunk_hits
                        finally:
                            consumer_s += perf_counter() - parked_started
                        del chunk_batch, chunk_hits
                    app_offset += sub_pairs.shape[0]

        finally:
            # In a `finally` because this is a generator: a consumer that stops pulling
            # part-way leaves the remaining sub-dispatches un-run, and what it did run
            # should still be reported rather than vanishing. `precursor_concat_seconds`
            # is deliberately absent -- the join is `match`'s, and it stamps it there.
            diagnostics = replace(
                diagnostics,
                match_seconds=match_s,
                predicate_seconds=predicate_s,
                match_buffer_seconds=match_buffer_s,
                match_dispatch_seconds=match_dispatch_s,
                match_kernel_seconds=match_kernel_s,
                match_copyback_seconds=match_copyback_s,
                match_other_host_seconds=match_other_host_s,
                match_template_upload_bytes=match_template_bytes,
                match_molecule_upload_bytes=match_molecule_bytes,
                match_pair_upload_bytes=match_pair_bytes,
                match_output_copyback_bytes=match_output_bytes,
                match_assignment_count=match_assignment_count,
                match_positive_pairs=match_positive_pairs,
                match_reserved_slots=match_reserved_slots,
                match_retry_pairs=match_retry_pairs,
                assignment_collect_seconds=collect_s,
                retro_seconds=retro_s,
                retro_construct_kernel_seconds=construct_kernel_s,
                wl_kernel_seconds=wl_kernel_s,
                precursor_gather_seconds=gather_s,
                consumer_seconds=consumer_s,
                total_seconds=perf_counter() - total_started - consumer_s,
            )
            self.last_diagnostics = diagnostics


def metal_concat_batches(batches: Sequence[MetalMoleculeBatch]) -> MetalMoleculeBatch:
    """`concat_mols` for the Metal backend: join precursor batches end to end.

    Public because a search that carries a frontier across passes needs it — the counterpart
    of `metal_stack_mols`, and the reason AO* can run here without ever holding a molecule
    on its own.

    Joining nothing gives the empty batch. That is not a courtesy: `matcher.keyed_match_folding`
    injects this as its `concat` and calls it with no parts when a layer produced no precursors
    at all -- a frontier whose pairs all screened out, or one the rules do not touch -- so
    raising here would turn an empty answer into a crashed search. `np.concatenate` refuses an
    empty sequence, which is exactly the case to take before reaching it.
    """
    if not batches:
        return _empty_batch()
    if len(batches) == 1:
        return batches[0]
    # Three arrays and a dropped `rows` cache, which is the whole of it now. This used to carry
    # an all-or-nothing `keys` tuple as well, and the awkward part of concatenation was that a
    # frontier built from CMols had none while precursors did -- so a mixed concat had to drop
    # every key rather than leave the tuple shorter than the batch. Keys are computed from a
    # batch rather than stored on one, so a concat is just a concat.
    return MetalMoleculeBatch(
        atom_counts=np.concatenate([b.atom_counts for b in batches]),
        atoms=np.concatenate([b.atoms for b in batches]),
        bonds=np.concatenate([b.bonds for b in batches]),
        rows=None,
    )


def _precursor_batch(
    produced: Any, retro_pairs: PairArray
) -> tuple[MetalMoleculeBatch, HitArray]:
    """Gather the retro kernel's valid precursor components into a packed batch and hits.

    Vectorised over the three output arrays, with no per-precursor molecule construction. It used
    to view one interleaved `GpuMol` array and pull four fields back out of it; the kernel writes
    the arrays directly now, so the de-interleaving is gone along with the struct.
    """
    n_assign = len(retro_pairs)
    # Read the status rows as one array rather than touching each ctypes struct: at tens of
    # thousands of assignments per dispatch the attribute access dominates the gather.
    status_words = np.frombuffer(
        produced.status, dtype=np.uint32, count=n_assign * _STATUS_WORDS
    ).reshape(n_assign, _STATUS_WORDS)
    # A reaction needs every named precursor. Keeping just the pieces that fit
    # can turn an impossible disconnection into a false stock route.
    #
    # An orphaned component is the same failure arriving from the other direction: not a
    # piece too large to hold, but a piece the template's precursor side never named, so
    # the disconnection is incomplete rather than merely truncated. The kernel flags it
    # and keeps going; refusing is this line's job, as it already is for the three
    # overflows, and the host arm does the same in `reaction.py`.
    complete = (
        (status_words[:, _STATUS_COMPONENT_OVERFLOW] == 0)
        & (status_words[:, _STATUS_ATOM_OVERFLOW] == 0)
        & (status_words[:, _STATUS_EDITED_ATOM_OVERFLOW] == 0)
        & (status_words[:, _STATUS_ORPHANED_COMPONENTS] == 0)
        & (status_words[:, _STATUS_GROUPING_VIOLATIONS] == 0)
    )
    comp_counts = np.where(
        complete, status_words[:, _STATUS_COMPONENT_COUNT], 0
    ).astype(np.int64)
    if int(comp_counts.sum()) == 0:
        return _empty_batch(), empty_hits()

    active = np.nonzero(comp_counts > 0)[0]
    # The kernel packs each assignment's components contiguously at status.output_base in
    # the compacted output buffer, so gather via those bases (not a fixed stride). Built as
    # one ragged arange -- base repeated per component, plus the offset within the run --
    # instead of one small array per assignment.
    bases = status_words[active, _STATUS_OUTPUT_BASE].astype(np.int64)
    repeats = comp_counts[active]
    total = int(repeats.sum())
    run_starts = np.zeros(repeats.size, dtype=np.int64)
    np.cumsum(repeats[:-1], out=run_starts[1:])
    out_indices = (
        np.repeat(bases, repeats)
        + np.arange(total, dtype=np.int64)
        - np.repeat(run_starts, repeats)
    )
    # Three gathers over the arrays the kernel wrote, where this used to view one interleaved
    # `GpuMol` array and de-interleave four fields out of it -- roughly twice the traffic, since
    # the struct carried the hash fields and a dead `atom_map` through the gather as well.
    atoms = np.ascontiguousarray(produced.atoms[out_indices], dtype=np.uint32)
    # retro_construct perceives ring-or-chain topology per edge and emits it on the bond
    # byte, so a generated precursor answers `@`/`!@` as exactly as a host-packed one.
    bonds = np.ascontiguousarray(produced.bonds[out_indices], dtype=np.uint8)
    counts = np.ascontiguousarray(produced.atom_counts[out_indices], dtype=np.uint32)

    retro = np.asarray(
        retro_pairs, dtype=np.int64
    )  # (n_assign, 2) = (rule_idx, mol_idx)
    hit_rule = np.repeat(retro[active, 0], repeats)
    hit_mol = np.repeat(retro[active, 1], repeats)
    hit_app = np.repeat(active, repeats)
    # The gather emits one precursor per hit, in hit order: the precursor column is the
    # identity. Deduplicating the batch here would only have to renumber this column.
    hit_precursor = np.arange(len(out_indices), dtype=np.int64)
    hits = np.stack([hit_mol, hit_rule, hit_app, hit_precursor], axis=1).astype(
        np.int64
    )

    batch = MetalMoleculeBatch(atom_counts=counts, atoms=atoms, bonds=bonds, rows=None)
    return batch, hits


class MetalFoldedPrefilteredReactionMatcher:
    """`MetalReactionMatcher` wrapped in folded path-fingerprint prefiltering (the Metal
    analogue of `CudaFoldedPrefilteredReactionMatcher`).

    With `pairs=None` the prefilter shrinks the all-pairs cross product to candidates before
    matching. With **explicit** pairs it screens the pairs it was given instead of ignoring them,
    which is what the forward probe needs: the probe asks for a sparse, deliberately chosen pair
    set -- roughly 64 candidate rules per application on the competitor channel -- and used to
    hand every one of them to the matcher kernel unscreened.

    Screening a given pair set is sound for the same reason screening the cross product is:
    the fingerprint test is **subset-sound**, so it can only let through more pairs than a sharper
    test would, never fewer. `tests/vesma/test_prefilter_soundness.py` is what holds that up.
    """

    def __init__(
        self,
        inner: MetalReactionMatcher,
        *,
        fingerprint_config: FoldedFingerprintConfig = FoldedFingerprintConfig(),
    ) -> None:
        self._inner = inner
        self._fingerprint_config = fingerprint_config
        # Pattern fingerprints depend only on the (fixed) rule set, so compute them
        # once and reuse across frontier steps instead of recomputing every call.
        self._pattern_fps: np.ndarray | None = None
        # Prefilter cost + how many candidate pairs survived it in the most recent
        # all-pairs (pairs=None) call. Lets a caller attribute the prune stage and see
        # the candidate count that actually reached the matcher kernel.
        self.last_prefilter_seconds = 0.0
        self.last_candidate_pair_count = 0

    @property
    def last_diagnostics(self) -> MetalReactionDiagnostics | None:
        """The wrapped matcher's per-stage diagnostics from the most recent `match`."""
        return self._inner.last_diagnostics

    def _pattern_fingerprints(self) -> np.ndarray:
        if self._pattern_fps is None:
            inner = self._inner
            self._pattern_fps = compute_folded_path_fingerprints_metal(
                inner._p_atoms,
                inner._p_bonds,
                inner._p_rows,
                inner._p_counts,
                query=True,
                config=self._fingerprint_config,
            )
        return self._pattern_fps

    def match(
        self,
        mols: Sequence[CMol] | MetalMoleculeBatch,
        pairs: PairArray | Sequence[Pair] | None = None,
    ) -> tuple[MetalMoleculeBatch, HitArray]:
        return self._inner.match(*self._screened(mols, pairs))

    def match_batches(
        self,
        mols: Sequence[CMol] | MetalMoleculeBatch,
        pairs: PairArray | Sequence[Pair] | None = None,
    ) -> Iterator[tuple[MetalMoleculeBatch, HitArray]]:
        """The wrapped matcher's pieces, over the pairs that survived the prune.

        Screening is a property of which pairs reach the kernel, not of how its results come
        back, so it happens once here and the delivery is whatever the inner matcher does.
        """
        return self._inner.match_batches(*self._screened(mols, pairs))

    def _screened(
        self,
        mols: Sequence[CMol] | MetalMoleculeBatch,
        pairs: PairArray | Sequence[Pair] | None,
    ) -> tuple[MetalMoleculeBatch, PairArray | Sequence[Pair]]:
        """Stack the molecules and prune the pairs, recording what the prune cost and kept."""
        batch = mols if isinstance(mols, MetalMoleculeBatch) else metal_stack_mols(mols)
        if pairs is None:
            prefilter_started = perf_counter()
            pairs = self._prefilter_pairs(batch)
            self.last_prefilter_seconds = perf_counter() - prefilter_started
            self.last_candidate_pair_count = int(len(pairs))
        else:
            prefilter_started = perf_counter()
            pairs = self._screen_pairs(batch, pairs)
            self.last_prefilter_seconds = perf_counter() - prefilter_started
            self.last_candidate_pair_count = int(len(pairs))
        return batch, pairs

    def _screen_pairs(
        self, batch: MetalMoleculeBatch, pairs: PairArray | Sequence[Pair]
    ) -> PairArray:
        """Drop the given pairs the folded fingerprints already rule out, order preserved.

        The same two tests the dense prefilter kernel applies -- the pattern cannot have more
        atoms than the molecule, and every path feature the pattern needs must be present -- just
        gathered over an explicit pair list instead of a tiled cross product. Host-side and
        vectorised, because for a sparse pair set the gather *is* the work; there is no kernel to
        write.

        Chunked so the temporaries stay bounded: one `(pairs, words)` gather of 64-bit words is
        256 bytes a pair at the default 2,048-bit width, which a whole layer's pair set would turn
        into hundreds of megabytes.
        """
        inner = self._inner
        given = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        if not len(given) or inner._n_patterns == 0 or len(batch) == 0:
            return given

        pattern_fps = self._pattern_fingerprints()
        molecule_fps = compute_folded_path_fingerprints_metal(
            batch.atoms.reshape(-1),
            batch.bonds.reshape(-1),
            batch.adjacency_rows().reshape(-1),
            np.ascontiguousarray(batch.atom_counts, dtype=np.int32),
            query=False,
            config=self._fingerprint_config,
        )
        pattern_atoms = inner._p_counts.astype(np.int64)
        molecule_atoms = np.asarray(batch.atom_counts, dtype=np.int64)

        keep = np.empty(len(given), dtype=np.bool_)
        for start in range(0, len(given), _SCREEN_PAIRS_PER_CHUNK):
            stop = start + _SCREEN_PAIRS_PER_CHUNK
            mol = given[start:stop, 0]
            rule = given[start:stop, 1]
            fits = pattern_atoms[rule] <= molecule_atoms[mol]
            covered = ~np.any(pattern_fps[rule] & ~molecule_fps[mol], axis=1)
            keep[start:stop] = fits & covered
        return given[keep]

    def _prefilter_pairs(self, batch: MetalMoleculeBatch) -> PairArray:
        """Candidate `(mol_idx, rule_id)` pairs as a ``(N, 2)`` array (no Python
        tuple round-trip; consumed directly by `MetalReactionMatcher.match`)."""
        inner = self._inner
        if len(batch) == 0 or inner._n_patterns == 0:
            return np.empty((0, 2), dtype=np.int64)

        pattern_fps = self._pattern_fingerprints()
        molecule_fps = compute_folded_path_fingerprints_metal(
            batch.atoms.reshape(-1),
            batch.bonds.reshape(-1),
            batch.adjacency_rows().reshape(-1),
            np.ascontiguousarray(batch.atom_counts, dtype=np.int32),
            query=False,
            config=self._fingerprint_config,
        )
        prefiltered = compact_folded_prefilter_pairs_metal(
            molecule_fingerprints=molecule_fps,
            pattern_fingerprints=pattern_fps,
            molecule_atom_counts=np.ascontiguousarray(
                batch.atom_counts, dtype=np.int32
            ),
            pattern_atom_counts=inner._p_counts.astype(np.int32),
        )
        if prefiltered.pair_count == 0:
            return np.empty((0, 2), dtype=np.int64)
        # (mol_idx, rule_id) columns.
        return np.stack(
            [
                prefiltered.pair_molecule_index_device,
                prefiltered.pair_pattern_index_device,
            ],
            axis=1,
        ).astype(np.int64)

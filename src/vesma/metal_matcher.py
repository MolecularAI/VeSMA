from __future__ import annotations

import ctypes
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .adjacency import bond_presence
from .gmol import GMol
from .gpu_atom_masks import ISOTOPE_ANY as ISOTOPE_ANY
from .gpu_atom_masks import ISOTOPE_SHIFT as ISOTOPE_SHIFT
from .gpu_atom_masks import ISOTOPE_WORD as ISOTOPE_WORD
from .gpu_atom_masks import isotope_field as isotope_field
from .gpu_atom_masks import pack_atom_masks
from .gpu_predicates import NO_PREDICATE as NO_PREDICATE
from .gpu_predicates import TERM_NEGATED as TERM_NEGATED
from .gpu_predicates import TERM_SUBPATTERN as TERM_SUBPATTERN
from .gpu_predicates import GpuPredicateProgram as MetalPredicateProgram
from .gpu_predicates import UnresolvedPredicateError as UnresolvedPredicateError
from .gpu_predicates import compile_predicate_program
from .gpu_predicates import literal_box as literal_box
from .gpu_predicates import literal_sort_key as literal_sort_key
from .gpu_predicates import pack_atom_predicates as pack_atom_predicates
from .gpu_predicates import (
    pack_collection_atom_predicates as pack_collection_atom_predicates,
)
from .gpu_predicates import sorted_literals_key as sorted_literals_key
from .gpu_predicates import sorted_terms_key as sorted_terms_key
from .gpu_predicates import term_box_row as term_box_row
from .limits import ATOM_MASK_WORD_COUNT
from .matcher import Matcher, MatchResult, MatchStatus
from .metal_pathfp import (
    compact_folded_prefilter_pairs_metal,
    compute_folded_path_fingerprints_metal,
)
from .metal_runtime import i32_ptr, load_metal_dylib, u8_ptr, u32_ptr, u64_ptr
from .pathfp import FoldedFingerprintConfig
from .predicates import PredicateTable, compile_predicate_table
from .rings import annotate_bond_topology

__all__ = [
    "MetalMatcher",
    "MetalMatchDispatchProfile",
    "MetalFoldedPrefilteredMatcher",
    "MetalMatchOnlyPrefilterDiagnostics",
    "GpuUnavailableError",
    "UnresolvedPredicateError",
    "probe_metal",
    "GpuBackendInfo",
    "pair_index_array",
]

_ATOM_WORD_COUNT = ATOM_MASK_WORD_COUNT
# A single Metal dispatch maps each (pattern, molecule) pair to one threadgroup;
# very large grids exceed device limits and silently drop work (matches diverge above
# ~4M pairs), so cap pairs per dispatch and loop. 4M is below the empirically-good
# bound; fixed reaction patterns may be cached by token across chunks.
_MAX_PAIRS_PER_DISPATCH = 4_000_000
_STATUS_NAMES: dict[int, MatchStatus] = {0: "no_match", 1: "match", 2: "unknown"}
_METAL_SOURCE_PATH = Path(__file__).with_name("metal_matcher.mm")


@dataclass(frozen=True)
class MetalMatchDispatchProfile:
    """One packed matcher call, separating GPU work from allocation and copies."""

    total_seconds: float = 0.0
    buffer_seconds: float = 0.0
    dispatch_seconds: float = 0.0
    kernel_seconds: float = 0.0
    copyback_seconds: float = 0.0
    other_host_seconds: float = 0.0
    template_upload_bytes: int = 0
    molecule_upload_bytes: int = 0
    pair_upload_bytes: int = 0
    output_copyback_bytes: int = 0
    native_calls: int = 0


_ADJACENCY_BITS = np.left_shift(np.uint64(1), np.arange(64, dtype=np.uint64))


class GpuUnavailableError(ValueError):
    pass


@dataclass(frozen=True)
class GpuBackendInfo:
    backend_type: str
    adapter_type: str
    name: str
    vendor: str
    architecture: str
    device: str
    description: str


def pair_index_array(
    pair_indices: np.ndarray | Sequence[tuple[int, int]],
    *,
    first_count: int,
    second_count: int,
    first_name: str,
    second_name: str,
) -> np.ndarray:
    pairs = np.asarray(pair_indices, dtype=np.int64)
    if pairs.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pair_indices must be a (N, 2) array or sequence of 2-tuples.")

    first = pairs[:, 0]
    bad_first = (first < 0) | (first >= first_count)
    if bool(np.any(bad_first)):
        bad = int(first[np.nonzero(bad_first)[0][0]])
        raise IndexError(
            f"{first_name} index {bad} out of bounds for {first_count} {first_name}s."
        )

    second = pairs[:, 1]
    bad_second = (second < 0) | (second >= second_count)
    if bool(np.any(bad_second)):
        bad = int(second[np.nonzero(bad_second)[0][0]])
        raise IndexError(
            f"{second_name} index {bad} out of bounds for "
            f"{second_count} {second_name}s."
        )

    return np.ascontiguousarray(pairs)


class MetalMatcher(Matcher):
    """Metal implementation of the Matcher protocol."""

    def __init__(
        self,
        max_prunings: int = 64,
        max_leaf_checks: int = 65536,
        threads_per_group: int = 256,
        max_prefix_depth: int = 16,
    ):
        if max_prunings <= 0:
            raise ValueError("max_prunings must be positive.")
        if max_leaf_checks <= 0:
            raise ValueError("max_leaf_checks must be positive.")
        if threads_per_group <= 0:
            raise ValueError("threads_per_group must be positive.")
        if max_prefix_depth < 0:
            raise ValueError("max_prefix_depth must be non-negative.")
        self.max_prunings = max_prunings
        self.max_leaf_checks = max_leaf_checks
        self.threads_per_group = threads_per_group
        self.max_prefix_depth = max_prefix_depth
        self._library = _load_metal_library()
        self.last_dispatch_profile = MetalMatchDispatchProfile()
        self.last_predicate_seconds = 0.0
        self.last_predicate_gpu_seconds = 0.0

    def has_match(self, pattern: GMol, molecule: GMol) -> bool:
        return self.find_matches(pattern, molecule, max_assignments=1).status == "match"

    def find_matches(
        self,
        pattern: GMol,
        molecule: GMol,
        *,
        max_assignments: int | None = None,
        **kwargs: Any,
    ) -> MatchResult:
        max_assignments = max_assignments or 256

        # Prepare single pair batch
        results = self._run_batch(
            [pattern],
            [molecule],
            [(0, 0)],
            max_assignments=max_assignments,
        )
        return results[0]

    def has_match_many(
        self,
        patterns: list[GMol],
        molecules: list[GMol],
        **kwargs: Any,
    ) -> np.ndarray:
        if not patterns or not molecules:
            return np.zeros((len(patterns), len(molecules)), dtype=bool)

        pair_indices = []
        for i in range(len(patterns)):
            for j in range(len(molecules)):
                pair_indices.append((i, j))

        status_out, _ = self._run_batch_raw(
            patterns,
            molecules,
            pair_indices,
            max_assignments=1,
            collect_assignments=False,
        )

        # status 1 is 'match'
        matches_flat = status_out[:, 1] == 1
        return matches_flat.reshape(len(patterns), len(molecules))

    def _run_batch_raw(
        self,
        patterns: list[GMol],
        molecules: list[GMol],
        pair_indices: np.ndarray | Sequence[tuple[int, int]],
        max_assignments: int,
        *,
        collect_assignments: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        bucket_size = 64
        pair_count = len(pair_indices)
        if pair_count == 0:
            return (
                np.zeros((0, 4), dtype=np.int32),
                np.zeros((0, max_assignments, 64), dtype=np.int32),
            )

        for p in patterns:
            if p.atom_count > bucket_size:
                raise ValueError(
                    f"MetalMatcher only supports up to {bucket_size} atoms. "
                    f"Pattern has {p.atom_count}."
                )
        for m in molecules:
            if m.atom_count > bucket_size:
                raise ValueError(
                    f"MetalMatcher only supports up to {bucket_size} atoms. "
                    f"Molecule has {m.atom_count}."
                )

        # Pack unique patterns and molecules
        p_atoms = np.concatenate([_pack_atom_masks(p, bucket_size) for p in patterns])
        m_atoms = np.concatenate([_pack_atom_masks(m, bucket_size) for m in molecules])
        p_rows = np.concatenate(
            [_pack_adjacency_rows(p, bucket_size) for p in patterns]
        )
        m_rows = np.concatenate(
            [_pack_adjacency_rows(m, bucket_size) for m in molecules]
        )
        p_bonds = np.concatenate([_pack_bond_matrix(p, bucket_size) for p in patterns])
        p_groups = np.concatenate([pack_atom_groups(p, bucket_size) for p in patterns])
        m_bonds = np.concatenate(
            [
                _pack_bond_matrix(m, bucket_size, annotate_topology=True)
                for m in molecules
            ]
        )
        p_counts = np.array([p.atom_count for p in patterns], dtype=np.uint32)
        m_counts = np.array([m.atom_count for m in molecules], dtype=np.uint32)
        table = compile_predicate_table(patterns)
        p_preds = pack_collection_atom_predicates(table, patterns, bucket_size)
        roots = evaluate_predicate_roots(
            self,
            metal_predicate_program(table),
            m_atoms=m_atoms,
            m_rows=m_rows,
            m_bonds=m_bonds,
            m_counts=m_counts,
        )

        return self.dispatch_match_packed(
            p_atoms=p_atoms,
            p_rows=p_rows,
            p_bonds=p_bonds,
            p_groups=p_groups,
            p_preds=p_preds,
            p_counts=p_counts,
            roots=roots,
            m_atoms=m_atoms,
            m_rows=m_rows,
            m_bonds=m_bonds,
            m_counts=m_counts,
            pair_indices=pair_indices,
            max_assignments=max_assignments,
            collect_assignments=collect_assignments,
        )

    def dispatch_match_packed(
        self,
        *,
        p_atoms: np.ndarray,
        p_rows: np.ndarray,
        p_bonds: np.ndarray,
        p_groups: np.ndarray,
        p_preds: np.ndarray | None,
        p_counts: np.ndarray,
        m_atoms: np.ndarray,
        m_rows: np.ndarray,
        m_bonds: np.ndarray,
        m_counts: np.ndarray,
        pair_indices: np.ndarray | Sequence[tuple[int, int]],
        max_assignments: int,
        roots: np.ndarray | None = None,
        collect_assignments: bool = True,
        copy_valid_only: bool = False,
        pattern_token: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Dispatch the matcher over pre-packed pattern + molecule arrays (the flat,
        concatenated `_pack_*` forms), so a molecule batch held as packed arrays can be
        re-ingested without rebuilding GMols. Pattern/molecule bonds are passed as-is;
        every present bond must carry a nonempty topology-state mask.

        ``pair_indices`` is `(pattern_idx, molecule_idx)` per pair, as a ``(N, 2)``
        array (fast path, no Python round-trip) or a sequence of tuples.
        A nonzero ``pattern_token`` identifies immutable packed pattern contents for
        reuse across calls; zero uploads them for each native dispatch.
        ``copy_valid_only`` leaves unused assignment slots unspecified and is intended
        for callers that select rows using the returned counts.

        ``p_groups`` is :func:`pack_atom_groups` per pattern, concatenated. It has no molecule
        counterpart: a target's components are a closure over adjacency the kernel already
        holds. It is required rather than defaulted because the pattern buffers are cached
        behind ``pattern_token``, so an omission would be served a stale buffer rather than an
        error.

        ``p_preds`` is :func:`pack_collection_atom_predicates`, required for the same reason --
        ``None`` is a statement that no pattern names a predicate, and nothing is uploaded -- and
        ``roots`` is what :func:`evaluate_predicate_roots` made of this molecule batch: a
        ``(molecule, predicate)`` table of atom bitsets, required whenever any pattern atom names
        a predicate."""
        bucket_size = 64
        pairs = pair_index_array(
            pair_indices,
            first_count=len(p_counts),
            second_count=len(m_counts),
            first_name="pattern",
            second_name="molecule",
        )
        pair_count = pairs.shape[0]
        named = (
            np.zeros(0, dtype=np.uint32)
            if p_preds is None
            else p_preds[p_preds != NO_PREDICATE]
        )
        if roots is None:
            roots = np.zeros((len(m_counts), 0), dtype=np.uint64)
        roots = np.ascontiguousarray(roots, dtype=np.uint64)
        if roots.shape[0] != len(m_counts) or (
            named.size and int(named.max()) >= roots.shape[1]
        ):
            raise ValueError(
                f"roots must be one row per molecule and cover every predicate a pattern names; "
                f"got {roots.shape} for {len(m_counts)} molecules."
            )
        if pair_count == 0:
            self.last_dispatch_profile = MetalMatchDispatchProfile()
            return (
                np.zeros((0, 4), dtype=np.int32),
                np.zeros((0, max_assignments, 64), dtype=np.int32),
            )

        dispatch_started = perf_counter()
        timings_total = np.zeros(4, dtype=np.float64)
        bytes_total = np.zeros(4, dtype=np.uint64)
        pair_p_idx_all = np.ascontiguousarray(pairs[:, 0], dtype=np.uint32)
        pair_m_idx_all = np.ascontiguousarray(pairs[:, 1], dtype=np.uint32)

        if collect_assignments:
            assignments_out = (
                np.empty((pair_count, max_assignments, 64), dtype=np.int32)
                if copy_valid_only
                else np.full((pair_count, max_assignments, 64), -1, dtype=np.int32)
            )
        else:
            assignments_out = np.zeros((pair_count, 0, 64), dtype=np.int32)
        status_out = np.zeros((pair_count, 4), dtype=np.int32)

        for start in range(0, pair_count, _MAX_PAIRS_PER_DISPATCH):
            stop = min(start + _MAX_PAIRS_PER_DISPATCH, pair_count)
            chunk_count = stop - start
            pair_p_idx = np.ascontiguousarray(pair_p_idx_all[start:stop])
            pair_m_idx = np.ascontiguousarray(pair_m_idx_all[start:stop])
            chunk_status = status_out[start:stop]
            if collect_assignments:
                chunk_assignments = assignments_out[start:stop]
                chunk_assignments_flat = chunk_assignments.reshape(-1)
                assignments_ptr = i32_ptr(chunk_assignments_flat)
            else:
                chunk_assignments = None
                assignments_ptr = ctypes.POINTER(ctypes.c_int32)()
            chunk_status_flat = chunk_status.reshape(-1)
            timings = (ctypes.c_double * 4)()
            bytes_copied = (ctypes.c_uint64 * 4)()

            ok = self._library.vesma_metal_match(
                u32_ptr(p_atoms),
                u32_ptr(m_atoms),
                u64_ptr(p_rows),
                u64_ptr(m_rows),
                u32_ptr(pair_p_idx),
                u32_ptr(pair_m_idx),
                u32_ptr(p_counts),
                u32_ptr(m_counts),
                u8_ptr(p_bonds),
                u8_ptr(m_bonds),
                u32_ptr(p_groups),
                (
                    u32_ptr(p_preds)
                    if p_preds is not None
                    else ctypes.POINTER(ctypes.c_uint32)()
                ),
                u64_ptr(roots) if roots.size else ctypes.POINTER(ctypes.c_uint64)(),
                ctypes.c_int(roots.shape[1] if p_preds is not None else 0),
                ctypes.c_int(bucket_size),
                ctypes.c_int(_ATOM_WORD_COUNT),
                ctypes.c_int(chunk_count),
                ctypes.c_int(self.max_prunings),
                ctypes.c_int(self.max_leaf_checks),
                ctypes.c_int(max_assignments),
                ctypes.c_int(1 if collect_assignments else 0),
                ctypes.c_int(1 if copy_valid_only else 0),
                ctypes.c_int(self.threads_per_group),
                ctypes.c_int(self.max_prefix_depth),
                ctypes.c_int(len(p_counts)),
                ctypes.c_uint64(pattern_token),
                assignments_ptr,
                i32_ptr(chunk_status_flat),
                timings,
                bytes_copied,
            )

            if ok != 1:
                raise RuntimeError(_metal_last_error(self._library))

            timings_total += np.asarray(timings)
            bytes_total += np.asarray(bytes_copied)

        total_seconds = perf_counter() - dispatch_started
        self.last_dispatch_profile = MetalMatchDispatchProfile(
            total_seconds=total_seconds,
            buffer_seconds=float(timings_total[0]),
            dispatch_seconds=float(timings_total[1]),
            copyback_seconds=float(timings_total[2]),
            kernel_seconds=float(timings_total[3]),
            other_host_seconds=max(total_seconds - float(timings_total[:3].sum()), 0.0),
            template_upload_bytes=int(bytes_total[0]),
            molecule_upload_bytes=int(bytes_total[1]),
            pair_upload_bytes=int(bytes_total[2]),
            output_copyback_bytes=int(bytes_total[3]),
            native_calls=-(-pair_count // _MAX_PAIRS_PER_DISPATCH),
        )
        return status_out, assignments_out

    def _run_batch(
        self,
        patterns: list[GMol],
        molecules: list[GMol],
        pair_indices: list[tuple[int, int]],
        max_assignments: int,
    ) -> list[MatchResult]:
        status_out, assignments_out = self._run_batch_raw(
            patterns, molecules, pair_indices, max_assignments
        )

        results = []
        for i in range(len(pair_indices)):
            p_idx, m_idx = pair_indices[i]
            p_atom_count = patterns[p_idx].atom_count
            count, status, prunings, leaf_checks = status_out[i]

            res = MatchResult(
                status=_STATUS_NAMES.get(status, "unknown"),
                assignments=assignments_out[i, :count, :p_atom_count].copy(),
                prunings=prunings,
                pruning_complete=True,
                search_complete=(status != 2),
                leaves_explored=leaf_checks,
            )
            results.append(res)
        return results


def probe_metal() -> GpuBackendInfo:
    if sys.platform != "darwin":
        raise GpuUnavailableError("Metal backend is only available on macOS.")

    library = _load_metal_library()
    name_buffer = ctypes.create_string_buffer(256)
    max_threads = ctypes.c_uint32()
    ok = library.vesma_metal_probe(
        name_buffer,
        ctypes.c_int(len(name_buffer)),
        ctypes.byref(max_threads),
    )
    if ok != 1:
        raise GpuUnavailableError(_metal_last_error(library))
    return GpuBackendInfo(
        backend_type="Metal",
        adapter_type="IntegratedGPU",
        name=name_buffer.value.decode("utf-8"),
        vendor="Apple",
        architecture="",
        device=name_buffer.value.decode("utf-8"),
        description=f"max_threads_per_threadgroup={max_threads.value}",
    )


def _pack_atom_masks(gmol: GMol, bucket_size: int) -> np.ndarray:
    cache_key = f"atom_masks_{bucket_size}"
    if cache_key in gmol.metal_cache:
        return gmol.metal_cache[cache_key]

    res = pack_atom_masks(gmol, bucket_size)
    gmol.metal_cache[cache_key] = res
    return res


def _pack_adjacency_rows(gmol: GMol, bucket_size: int) -> np.ndarray:
    cache_key = f"adjacency_rows_{bucket_size}"
    if cache_key in gmol.metal_cache:
        return gmol.metal_cache[cache_key]

    edges = bond_presence(gmol.bonds)
    rows = np.zeros(bucket_size, dtype=np.uint64)
    count = gmol.atom_count
    rows[:count] = np.sum(
        edges[:count, :count].astype(np.uint64, copy=False) * _ADJACENCY_BITS[:count],
        axis=1,
        dtype=np.uint64,
    )
    gmol.metal_cache[cache_key] = rows
    return rows


def pack_atom_groups(gmol: GMol, bucket_size: int) -> np.ndarray:
    """A pattern's zero-level group ids, one per atom slot, padded to the bucket.

    Public where its siblings are not, because every caller of
    :meth:`MetalMatcher.dispatch_match_packed` must supply this array and none may leave it out:
    the native side caches pattern buffers behind an upload token, so a caller that passed
    nothing while another passed groups under the same token would be answered from a stale
    buffer -- a wrong answer with no error.
    """
    cache_key = f"atom_groups_{bucket_size}"
    if cache_key in gmol.metal_cache:
        return gmol.metal_cache[cache_key]

    groups = np.zeros(bucket_size, dtype=np.uint32)
    groups[: gmol.atom_count] = gmol.atom_group
    gmol.metal_cache[cache_key] = groups
    return groups


def metal_predicate_program(table: PredicateTable) -> MetalPredicateProgram:
    return compile_predicate_program(
        table,
        pack_atoms=partial(_pack_atom_masks, bucket_size=64),
        pack_rows=partial(_pack_adjacency_rows, bucket_size=64),
        pack_bonds=partial(_pack_bond_matrix, bucket_size=64),
        pack_groups=partial(pack_atom_groups, bucket_size=64),
    )


def evaluate_predicate_roots(
    matcher: MetalMatcher,
    program: MetalPredicateProgram,
    *,
    m_atoms: np.ndarray,
    m_rows: np.ndarray,
    m_bonds: np.ndarray,
    m_counts: np.ndarray,
) -> np.ndarray:
    """Every residual of a program over a packed molecule batch: ``(molecules, residuals)``.

    All of it on the device in one command buffer -- probe bits; then per level the pinned
    body matches, their fold into root bitsets, and the level's residual expressions -- with one
    readback at the end. A level's pairs are implied (every body of the level at every molecule
    atom), so no grid size waits on the device; a root its body's first atom cannot take is
    dropped before any work. Bodies are searched to their first embedding: enumerated
    assignments would be capped, and the roots then wrong.
    """
    n_mols = len(m_counts)
    roots = np.zeros((n_mols, program.residual_count), dtype=np.uint64)
    if program.residual_count == 0 or n_mols == 0:
        matcher.last_predicate_seconds = 0.0
        matcher.last_predicate_gpu_seconds = 0.0
        return roots
    started = perf_counter()
    counts = np.ascontiguousarray(m_counts, dtype=np.uint32)
    timings = (ctypes.c_double * 3)()
    unresolved = ctypes.c_uint32(0)
    ok = matcher._library.vesma_metal_predicate_roots(
        u32_ptr(np.ascontiguousarray(m_atoms, dtype=np.uint32)),
        u64_ptr(np.ascontiguousarray(m_rows, dtype=np.uint64)),
        u8_ptr(np.ascontiguousarray(m_bonds, dtype=np.uint8)),
        u32_ptr(counts),
        ctypes.c_int(n_mols),
        u32_ptr(program.body_atoms),
        u64_ptr(program.body_rows),
        u8_ptr(program.body_bonds),
        u32_ptr(program.body_groups),
        u32_ptr(program.body_preds),
        u32_ptr(program.body_counts),
        ctypes.c_int(len(program.body_counts)),
        u32_ptr(program.probes),
        ctypes.c_int(len(program.probes) // _ATOM_WORD_COUNT),
        u32_ptr(program.clause_start),
        u32_ptr(program.term_start),
        u32_ptr(program.factor_start),
        u32_ptr(program.factors),
        ctypes.c_int(program.residual_count),
        ctypes.c_int(len(program.term_start) - 1),
        ctypes.c_int(len(program.factor_start) - 1),
        ctypes.c_int(len(program.factors)),
        u32_ptr(program.residual_level_start),
        u32_ptr(program.body_level_start),
        ctypes.c_int(program.depth),
        ctypes.c_int(64),
        ctypes.c_int(_ATOM_WORD_COUNT),
        ctypes.c_int(matcher.max_prunings),
        ctypes.c_int(matcher.max_leaf_checks),
        ctypes.c_int(matcher.threads_per_group),
        ctypes.c_int(matcher.max_prefix_depth),
        ctypes.c_int(_MAX_PAIRS_PER_DISPATCH),
        u64_ptr(roots.reshape(-1)),
        ctypes.byref(unresolved),
        timings,
    )
    if ok != 1:
        raise RuntimeError(_metal_last_error(matcher._library))
    if unresolved.value:
        raise UnresolvedPredicateError(
            f"{unresolved.value} recursive-SMARTS body search(es) exhausted the matcher's "
            "budget, so where the body holds is unknown; raise max_leaf_checks or "
            "max_prefix_depth."
        )
    matcher.last_predicate_seconds = perf_counter() - started
    matcher.last_predicate_gpu_seconds = float(timings[2])
    return roots


def _pack_bond_matrix(
    gmol: GMol, bucket_size: int, *, annotate_topology: bool = False
) -> np.ndarray:
    cache_key = f"bond_matrix_{bucket_size}_topology_{annotate_topology}"
    if cache_key in gmol.metal_cache:
        return gmol.metal_cache[cache_key]

    bonds = annotate_bond_topology(gmol.bonds) if annotate_topology else gmol.bonds
    res = np.zeros((bucket_size, bucket_size), dtype=np.uint8)
    res[: gmol.atom_count, : gmol.atom_count] = bonds
    res_flat = res.reshape(-1)
    gmol.metal_cache[cache_key] = res_flat
    return res_flat


def _metal_last_error(library: ctypes.CDLL) -> str:
    msg = library.vesma_metal_last_error()
    return msg.decode("utf-8") if msg else "Unknown Metal error"


def _load_metal_library() -> ctypes.CDLL:
    lib = load_metal_dylib(_METAL_SOURCE_PATH)
    lib.vesma_metal_last_error.restype = ctypes.c_char_p
    lib.vesma_metal_probe.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.vesma_metal_probe.restype = ctypes.c_int
    lib.vesma_metal_match.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),  # pattern_atoms
        ctypes.POINTER(ctypes.c_uint32),  # molecule_atoms
        ctypes.POINTER(ctypes.c_uint64),  # pattern_rows
        ctypes.POINTER(ctypes.c_uint64),  # molecule_rows
        ctypes.POINTER(ctypes.c_uint32),  # pair_p_indices
        ctypes.POINTER(ctypes.c_uint32),  # pair_m_indices
        ctypes.POINTER(ctypes.c_uint32),  # p_counts
        ctypes.POINTER(ctypes.c_uint32),  # m_counts
        ctypes.POINTER(ctypes.c_uint8),  # p_bonds
        ctypes.POINTER(ctypes.c_uint8),  # m_bonds
        ctypes.POINTER(ctypes.c_uint32),  # p_groups
        ctypes.POINTER(ctypes.c_uint32),  # p_preds
        ctypes.POINTER(ctypes.c_uint64),  # roots, or NULL
        ctypes.c_int,  # pred_stride
        ctypes.c_int,  # bucket_size
        ctypes.c_int,  # atom_word_count
        ctypes.c_int,  # pair_count
        ctypes.c_int,  # max_prunings
        ctypes.c_int,  # max_leaf_checks
        ctypes.c_int,  # max_assignments
        ctypes.c_int,  # collect_assignments
        ctypes.c_int,  # copy_valid_only
        ctypes.c_int,  # threads_per_group
        ctypes.c_int,  # max_prefix_depth
        ctypes.c_int,  # pattern_count
        ctypes.c_uint64,  # pattern_token; zero disables caching
        ctypes.POINTER(ctypes.c_int32),  # assignments_out
        ctypes.POINTER(ctypes.c_int32),  # status_out
        ctypes.POINTER(ctypes.c_double),  # native stage timings
        ctypes.POINTER(ctypes.c_uint64),  # copied bytes by category
    ]
    lib.vesma_metal_match.restype = ctypes.c_int
    u32p = ctypes.POINTER(ctypes.c_uint32)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    lib.vesma_metal_predicate_roots.argtypes = [
        u32p,  # molecule atoms
        u64p,  # molecule rows
        u8p,  # molecule bonds
        u32p,  # molecule atom counts
        ctypes.c_int,  # molecules
        u32p,  # body atoms
        u64p,  # body rows
        u8p,  # body bonds
        u32p,  # body groups
        u32p,  # body predicate ids
        u32p,  # body atom counts
        ctypes.c_int,  # bodies
        u32p,  # probes
        ctypes.c_int,  # probe count
        u32p,  # clause start
        u32p,  # term start
        u32p,  # factor start
        u32p,  # factors
        ctypes.c_int,  # residuals
        ctypes.c_int,  # clauses
        ctypes.c_int,  # terms
        ctypes.c_int,  # factors
        u32p,  # residual level start
        u32p,  # body level start
        ctypes.c_int,  # depth
        ctypes.c_int,  # bucket size
        ctypes.c_int,  # atom word count
        ctypes.c_int,  # max prunings
        ctypes.c_int,  # max leaf checks
        ctypes.c_int,  # threads per group
        ctypes.c_int,  # max prefix depth
        ctypes.c_int,  # max pairs per dispatch
        u64p,  # roots out
        ctypes.POINTER(ctypes.c_uint32),  # unresolved searches out
        ctypes.POINTER(ctypes.c_double),  # timings
    ]
    lib.vesma_metal_predicate_roots.restype = ctypes.c_int
    return lib


@dataclass(frozen=True)
class MetalMatchOnlyPrefilterDiagnostics:
    dense_pair_count: int
    candidate_pair_count: int
    pruned_pair_count: int
    prune_ratio: float
    fingerprint_seconds: float
    inner_match_seconds: float
    # Breakdown of the prefilter stage (sums to ~fingerprint_seconds): pattern-fp
    # compute, molecule-fp compute, and the compare/compaction kernel.
    pattern_fingerprint_seconds: float = 0.0
    molecule_fingerprint_seconds: float = 0.0
    prefilter_compare_seconds: float = 0.0
    pair_materialization_seconds: float = 0.0


class MetalFoldedPrefilteredMatcher(Matcher):
    def __init__(
        self,
        matcher: MetalMatcher | None = None,
        *,
        fingerprint_config: FoldedFingerprintConfig = FoldedFingerprintConfig(),
    ) -> None:
        self.matcher = MetalMatcher() if matcher is None else matcher
        self.fingerprint_config = fingerprint_config
        self.last_diagnostics: MetalMatchOnlyPrefilterDiagnostics | None = None

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
            self.last_diagnostics = MetalMatchOnlyPrefilterDiagnostics(
                dense_pair_count=len(patterns) * len(molecules),
                candidate_pair_count=0,
                pruned_pair_count=len(patterns) * len(molecules),
                prune_ratio=1.0 if patterns or molecules else 0.0,
                fingerprint_seconds=0.0,
                inner_match_seconds=0.0,
            )
            return np.zeros((len(patterns), len(molecules)), dtype=bool)

        bucket_size = 64
        # Pack + fingerprint patterns (timed separately from molecule fp + compare).
        pattern_started = perf_counter()
        p_atoms = np.concatenate([_pack_atom_masks(p, bucket_size) for p in patterns])
        p_bonds = np.concatenate([_pack_bond_matrix(p, bucket_size) for p in patterns])
        p_rows = np.concatenate(
            [_pack_adjacency_rows(p, bucket_size) for p in patterns]
        )
        p_counts = np.array([p.atom_count for p in patterns], dtype=np.int32)

        pattern_fps = compute_folded_path_fingerprints_metal(
            p_atoms,
            p_bonds,
            p_rows,
            p_counts,
            query=True,
            config=self.fingerprint_config,
        )
        pattern_fingerprint_seconds = perf_counter() - pattern_started

        # Pack + fingerprint molecules.
        molecule_started = perf_counter()
        m_atoms = np.concatenate([_pack_atom_masks(m, bucket_size) for m in molecules])
        m_bonds = np.concatenate(
            [
                _pack_bond_matrix(m, bucket_size, annotate_topology=True)
                for m in molecules
            ]
        )
        m_rows = np.concatenate(
            [_pack_adjacency_rows(m, bucket_size) for m in molecules]
        )
        m_counts = np.array([m.atom_count for m in molecules], dtype=np.int32)

        molecule_fps = compute_folded_path_fingerprints_metal(
            m_atoms,
            m_bonds,
            m_rows,
            m_counts,
            query=False,
            config=self.fingerprint_config,
        )
        molecule_fingerprint_seconds = perf_counter() - molecule_started

        # Compare/compaction kernel.
        compare_started = perf_counter()
        prefiltered = compact_folded_prefilter_pairs_metal(
            molecule_fingerprints=molecule_fps,
            pattern_fingerprints=pattern_fps,
            molecule_atom_counts=m_counts,
            pattern_atom_counts=p_counts,
        )
        prefilter_compare_seconds = perf_counter() - compare_started
        fingerprint_seconds = (
            pattern_fingerprint_seconds
            + molecule_fingerprint_seconds
            + prefilter_compare_seconds
        )

        out = np.zeros((len(patterns), len(molecules)), dtype=bool)
        if prefiltered.pair_count == 0:
            self.last_diagnostics = MetalMatchOnlyPrefilterDiagnostics(
                dense_pair_count=prefiltered.dense_pair_count,
                candidate_pair_count=0,
                pruned_pair_count=prefiltered.dense_pair_count,
                prune_ratio=(0.0 if prefiltered.dense_pair_count == 0 else 1.0),
                fingerprint_seconds=fingerprint_seconds,
                inner_match_seconds=0.0,
                pattern_fingerprint_seconds=pattern_fingerprint_seconds,
                molecule_fingerprint_seconds=molecule_fingerprint_seconds,
                prefilter_compare_seconds=prefilter_compare_seconds,
            )
            return out

        # Candidate pairs flow as a single (N, 2) (pattern_idx, molecule_idx) array
        # straight into the matcher dispatch — no numpy -> tuples -> numpy round-trip.
        materialize_started = perf_counter()
        pair_array = np.stack(
            [
                prefiltered.pair_pattern_index_device,
                prefiltered.pair_molecule_index_device,
            ],
            axis=1,
        ).astype(np.uint32, copy=False)
        pair_materialization_seconds = perf_counter() - materialize_started

        started = perf_counter()
        status_out, _ = self.matcher._run_batch_raw(
            patterns,
            molecules,
            pair_array,
            max_assignments=1,
            collect_assignments=False,
        )
        inner_match_seconds = perf_counter() - started

        self.last_diagnostics = MetalMatchOnlyPrefilterDiagnostics(
            dense_pair_count=prefiltered.dense_pair_count,
            candidate_pair_count=prefiltered.pair_count,
            pruned_pair_count=prefiltered.dense_pair_count - prefiltered.pair_count,
            prune_ratio=1.0 - (prefiltered.pair_count / prefiltered.dense_pair_count),
            fingerprint_seconds=fingerprint_seconds,
            inner_match_seconds=inner_match_seconds,
            pattern_fingerprint_seconds=pattern_fingerprint_seconds,
            molecule_fingerprint_seconds=molecule_fingerprint_seconds,
            prefilter_compare_seconds=prefilter_compare_seconds,
            pair_materialization_seconds=pair_materialization_seconds,
        )

        out[pair_array[:, 0], pair_array[:, 1]] = status_out[:, 1] == 1
        return out

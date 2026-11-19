from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

import numpy as np

from .gmol import GMol

MatchStatus = Literal["match", "no_match", "unknown"]
AssignmentMatrix = np.ndarray[tuple[int, int], np.dtype[np.int32]]
CandidateMatrix = np.ndarray[
    tuple[int, int], np.dtype[np.bool_]
]  # pattern×molecule / edge booleans


@dataclass(frozen=True)
class MatchResult:
    """Unified matcher result with status, assignments, and search diagnostics."""

    status: MatchStatus
    assignments: AssignmentMatrix
    prunings: int
    pruning_complete: bool
    search_complete: bool
    leaves_explored: int
    metadata: dict[str, Any] | None = None


@runtime_checkable
class Matcher(Protocol):
    """Protocol for subgraph isomorphism matching backends.

    This is the *low-level* stage of the pipeline: Ullmann refinement, injectivity,
    and assignment enumeration. Reaction application (bond edits, split into
    connected components, aromaticity inference) lives one layer up in
    `ReactionMatcher`.
    """

    def has_match(self, pattern: GMol, molecule: GMol) -> bool:
        """Return True if the pattern matches the molecule."""
        ...

    def find_matches(
        self,
        pattern: GMol,
        molecule: GMol,
        *,
        max_assignments: int | None = None,
        **kwargs: Any,
    ) -> MatchResult:
        """Find matches and return detailed results."""
        ...

    def has_match_many(
        self,
        patterns: list[GMol],
        molecules: list[GMol],
        **kwargs: Any,
    ) -> np.ndarray:
        """Perform cross-product matching for many patterns and molecules."""
        ...


# --------------------------------------------------------------------------- #
# Reaction-matcher layer
#
# The reaction matcher is the *full* per-molecule retro pipeline: run the
# subgraph-isomorphism `Matcher`, then split each assignment into precursor
# molecules (bond edits + connected components + aromaticity inference).
#
# The one primitive — `match(mols, pairs)` — is unified across CPU, rdkit, and
# CUDA. `pairs` is an explicit set of `(mol_idx, rule_id)` evaluations;
# `pairs=None` means every molecule against every rule. all-vs-all and
# some-vs-some differ *only* in the `pairs` argument, so replay is identical
# across backends.
#
# Hits carry an explicit precursor index, so the precursor batch is *not* tied to the
# hit count: a backend may return every precursor it built (one row per hit), collapse
# only some duplicates, or return one row per distinct precursor in the batch. Near
# synonymous rules produce identical precursors, so within-batch dedup is a large win
# in device memory and transfer — it is a backend's choice, and no live backend takes
# it today. Identical *steps* still collapse downstream in key space (the search `seen`
# set + the store's `(product, rule, app)` grouping).
# --------------------------------------------------------------------------- #

# The shared domain type-variable vocabulary. Defined once, here, and imported by both `vesma`
# and `vertsy` (never re-declared) so the same letter means the same thing everywhere. Bounds
# live on the TypeVar, so use sites write bare `M`/`K` (no per-scope `K: Hashable` repetition).
# molecule repr (CMol / rdkit.Mol / CUDA-batch element); opaque to the matcher, so unbounded
M = TypeVar("M")
# key / identity token (SMILES str, WL tuple[int, int], a future 64-bit hash)
K = TypeVar("K", bound=Hashable)
# Rule id is concrete int today, used only in concrete positions (Hit/Pair/prefilter). It is
# deliberately not a shared TypeVar: `R` still conflates a stable rule *identifier* with the
# matcher's dense rule *index*, a split left for a later rule-interner pass.
type R = int

# (mol_idx, rule_id, app_idx, precursor_idx): app_idx groups co-precursors of one
# reaction application (one disconnection); precursor_idx points at the precursor batch
# row. The conceptual element type; a batch of hits is carried as the
# structure-of-arrays ``HitArray`` below (one row per element).
Hit = tuple[int, R, int, int]
# (N, 4) int array of (mol_idx, rule_id, app_idx, precursor_idx) rows. The first three
# columns are the application key; the fourth indexes the returned precursor batch,
# which may be shorter than the hit array (see the block comment above). Rule ids are
# int here — the search/store edge relation is integer-keyed.
HitArray = np.ndarray[
    tuple[int, int], np.dtype[np.int64]
]  # (n_hits, 4): molecule, rule, application, precursor
# (molecule_index, reaction_pattern_index) evaluation to perform. Pairs are
# assumed unique.
Pair = tuple[int, R]
# The same (mol_idx, rule_id) pairs packed as a dense (N, 2) int64 array, the fast path
# the GPU matchers consume directly (no Python tuple round-trip).
PairArray = np.ndarray[tuple[int, int], np.dtype[np.int64]]


def empty_hits() -> HitArray:
    """The (0, 4) hit array a match with no applications returns."""
    return np.zeros((0, 4), dtype=np.int64)


def hit_array(rows: Sequence[Hit]) -> HitArray:
    """Build an (N, 4) hit array from ``(mol_idx, rule_id, app_idx, precursor_idx)``."""
    if not rows:
        return empty_hits()
    hits = np.asarray(rows, dtype=np.int64)
    if hits.ndim != 2 or hits.shape[1] != 4:
        raise ValueError(f"hit rows must be 4-tuples, got shape {hits.shape}")
    return hits


def concat_hits(parts: Sequence[HitArray], precursor_counts: Sequence[int]) -> HitArray:
    """Concatenate per-chunk hit arrays, rebasing each chunk's precursor column.

    ``precursor_counts`` is each part's precursor batch size, in the same order the
    parts' precursor batches are concatenated: part ``i``'s precursor indices shift by
    the number of precursors the earlier parts contributed. It is required rather than
    inferred because the hit count no longer implies the precursor count.
    """
    if len(parts) != len(precursor_counts):
        raise ValueError(
            "one precursor count per hit part is required: "
            f"{len(precursor_counts)} counts for {len(parts)} parts"
        )
    if not parts:
        return empty_hits()
    offsets = np.cumsum([0, *precursor_counts[:-1]])
    rebased = [
        part if not offset else part + np.array([0, 0, 0, offset], dtype=np.int64)
        for part, offset in zip(parts, offsets.tolist(), strict=True)
    ]
    return np.concatenate(rebased)


def compact_precursor_index(hits: HitArray) -> tuple[np.ndarray, HitArray]:
    """Drop precursor rows no hit references, renumbering the hit precursor column.

    Returns the ascending precursor rows still in use and the rebased hits. Callers
    gather their own precursor container (and key sequence) with the returned indices;
    the helper stays container-agnostic because every backend batch differs.
    """
    used, inverse = np.unique(hits[:, 3], return_inverse=True)
    compacted = hits.copy()
    compacted[:, 3] = inverse.reshape(-1)
    return used, compacted


# Engine-facing callables. `search`/`step` take a KeyedMatch. The key sequence is
# parallel to the *precursor batch* (one key per precursor row), not to the hits.
Match = Callable[[Sequence[Any]], tuple[np.ndarray, HitArray]]
KeyedMatch = Callable[
    [Sequence[Any]],
    tuple[Any, Sequence[Hashable], HitArray],
]
# A match delivered in pieces: one `(precursor batch, hits)` per piece, in order, with each
# piece's hit rows indexing *that* piece. Columns 0-2 are already global -- a backend numbers
# applications across its own pieces -- so only column 3 is piece-local, and `concat_hits` or
# `keyed_match_folding` is what reconciles it.
#
# Why a generator rather than a callback: the consumer decides when to pull, which is what lets
# `keyed_match_folding` fold without the matcher knowing anything about identity, and what lets
# `match` be written as a fold over `match_batches` instead of a second copy of the same loop.
BatchedMatch = Callable[[Sequence[Any]], Iterator[tuple[Any, HitArray]]]


@runtime_checkable
class ReactionMatcher(Protocol[M]):
    """Apply a compiled rule set to molecules, returning precursor mols + hits.

    Implementations receive compiled rules in their constructor. `match` returns a
    precursor batch alongside the `Hit` provenance rows that point into it.

    The precursor batch *may* be deduplicated: a backend can return one row per hit,
    collapse only the duplicates it cheaply recognises, or return one row per distinct
    precursor. Consumers must therefore index — `precursors[hit[3]]` — and never zip.
    The only invariants are that every ``hit[3]`` is a valid precursor row and that
    every precursor row is referenced by at least one hit (no orphans, which would
    otherwise put molecules on the search frontier that no application produced).
    """

    def match(
        self,
        mols: Sequence[M],
        pairs: Sequence[Pair] | None = None,
    ) -> tuple[np.ndarray, HitArray]:
        """Evaluate `pairs` (or every mol × every rule when None)."""
        ...


@runtime_checkable
class BatchedReactionMatcher(ReactionMatcher[M], Protocol[M]):
    """A `ReactionMatcher` that can also deliver its precursors in pieces.

    The reason to want pieces is that the whole is large: a batch row costs thousands of bytes,
    and a layer of a real search produces millions of them, so a matcher that can only hand back
    one array forces its peak footprint on every caller -- including the callers that are about to
    collapse the result to distinct molecules anyway.

    ``match`` and ``match_batches`` answer the same question, and the natural implementation makes
    the first a fold over the second rather than a second copy of the loop (see
    `metal_reaction_matcher.MetalReactionMatcher`). A backend with no internal chunking satisfies
    this by yielding once.

    A caller that abandons the iterator part-way leaves the rest of the work undone, and anything
    the matcher reports about the call -- diagnostics, counters -- describes only the pieces that
    were actually pulled.
    """

    def match_batches(
        self,
        mols: Sequence[M],
        pairs: Sequence[Pair] | None = None,
    ) -> Iterator[tuple[Any, HitArray]]:
        """Evaluate `pairs` (or every mol × every rule when None), one piece at a time."""
        ...


def all_vs_all(matcher: ReactionMatcher[M]) -> Match:
    """Lift a reaction matcher to an all-vs-all match callable."""

    def run(mols: Sequence[M]) -> tuple[np.ndarray, HitArray]:
        return matcher.match(mols, None)

    return run


def keyed_match(
    match: Callable[[Sequence[M]], tuple[Any, HitArray]],
    key: Callable[[Any], Sequence[K]],
) -> KeyedMatch:
    """Compose reaction matching and molecule identity for search.

    Keys come out parallel to the precursor batch the matcher returned, so a
    deduplicated batch is keyed once per distinct precursor.
    """

    def run(mols: Sequence[M]) -> tuple[Any, Sequence[K], HitArray]:
        precursors, hits = match(mols)
        return precursors, key(precursors), hits

    return run


def all_vs_all_batches(matcher: BatchedReactionMatcher[M]) -> BatchedMatch:
    """`all_vs_all` for a matcher that delivers its precursors in batches."""

    def run(mols: Sequence[M]) -> Iterator[tuple[Any, HitArray]]:
        return matcher.match_batches(mols, None)

    return run


def keyed_match_folding(
    batched: BatchedMatch,
    key: Callable[[Any], Sequence[K]],
    concat: Callable[[Sequence[Any]], Any],
) -> KeyedMatch:
    """`keyed_match`, folding identity in as batches arrive instead of after.

    Same type as :func:`keyed_match` and the same answer, assembled differently: each batch is
    keyed as it is delivered and collapsed against the keys already seen, so what accumulates is
    **one row per distinct precursor** rather than one per application. On the full corpus that is
    the difference between holding every precursor a layer produced and holding only the molecules
    it actually found -- the batch a matcher hands over is ~7,684 bytes a row, and a layer's raw
    precursors outnumber its distinct molecules by most of an order of magnitude.

    Deduplicating here rather than in a backend is the point. `ReactionMatcher` has always allowed
    a batch shorter than the hit array (see the block comment above), `cuda_matcher` has always
    returned one, and this is that behaviour written once against the keyer every arm already
    lifts through -- so the matcher stays ignorant of identity, which
    `metal_reaction_matcher.metal_short_ring_wl_key` records the reason for at length.

    `concat` joins the retained parts: a batch is a backend's own type, so the join is the
    backend's own function (`metal_concat_batches`, `cuda_concat_mols`,
    `vertsy.search.concat_object_batches`). It is called once, on the collapsed parts.

    The hits' precursor column is rewritten to index the returned batch directly, so no
    :func:`concat_hits` rebasing happens on this path -- that stays the primitive for joining
    batches whose rows are all kept. Columns 0-2 are untouched: a backend already numbers
    applications globally across its own batches, and an application id means what it meant.
    """

    def run(mols: Sequence[M]) -> tuple[Any, Sequence[K], HitArray]:
        seen: dict[Hashable, int] = {}
        kept_keys: list[K] = []
        parts: list[Any] = []
        hit_parts: list[HitArray] = []

        for batch, hits in batched(mols):
            batch_keys = key(batch)
            if len(batch_keys) != len(batch):
                raise ValueError(
                    "keyer returned one key per precursor row: "
                    f"{len(batch_keys)} keys for {len(batch)} rows"
                )
            # Local row -> row in the batch this returns. A first sighting mints the next row and
            # is retained; a repeat points at whatever row already holds that molecule, which may
            # have arrived in an earlier batch.
            target = np.empty(len(batch_keys), dtype=np.int64)
            first_seen: list[int] = []
            for local, batch_key in enumerate(batch_keys):
                row = seen.get(batch_key)
                if row is None:
                    row = len(seen)
                    seen[batch_key] = row
                    first_seen.append(local)
                    kept_keys.append(batch_key)
                target[local] = row
            if first_seen:
                parts.append(batch[np.asarray(first_seen, dtype=np.int64)])
            if len(hits):
                rebased = np.asarray(hits, dtype=np.int64).reshape(-1, 4).copy()
                rebased[:, 3] = target[rebased[:, 3]]
                hit_parts.append(rebased)

        if not parts:
            # No precursors at all, so there is no batch to concatenate and nothing to key. The
            # empty batch has to come from somewhere backend-shaped, and `concat` of nothing is
            # the only such thing in scope.
            return concat(parts), [], empty_hits()
        return (
            concat(parts) if len(parts) > 1 else parts[0],
            kept_keys,
            np.concatenate(hit_parts) if hit_parts else empty_hits(),
        )

    return run


__all__ = [
    "AssignmentMatrix",
    "BatchedMatch",
    "BatchedReactionMatcher",
    "Hit",
    "HitArray",
    "K",
    "KeyedMatch",
    "M",
    "Match",
    "MatchResult",
    "MatchStatus",
    "Matcher",
    "Pair",
    "PairArray",
    "R",
    "ReactionMatcher",
    "all_vs_all",
    "all_vs_all_batches",
    "compact_precursor_index",
    "concat_hits",
    "empty_hits",
    "hit_array",
    "keyed_match",
    "keyed_match_folding",
]

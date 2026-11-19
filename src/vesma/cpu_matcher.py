from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .adjacency import BondMatrix, bond_presence, bond_type_compatibility_matrix
from .cmol import CMol
from .cmol_builder import gmol_from_cmol
from .connected_components import (
    component_labels_from_bonds,
    grouping_agrees_with_components,
)
from .gmol import AtomMapVector, GMol
from .gmol_atoms import MASS_ANY, Aromaticity, AtomIndices, AtomMask
from .matcher import (
    CandidateMatrix,
    Hit,
    HitArray,
    Matcher,
    MatchResult,
    Pair,
    concat_hits,
    hit_array,
)
from .parallel import (
    ResidentForkPool,
    chunked,
    parallel_chunks,
    parallel_map_fork,
    resolve_n_jobs,
)
from .predicates import predicate_atoms
from .reaction import (
    DEFAULT_REACTION_PROCESSING,
    MoleculeProcessing,
    PrecursorOverflowDiagnostics,
    ReactionEdit,
    ReactionMapping,
    ReactionPattern,
    ReactionProcessing,
    RetroOutcome,
    add_precursor_overflow_diagnostics,
    precursor_signature,
    precursors_from_mapping_result,
    stamped_undescribed,
    warn_precursor_overflow,
)
from .rejection import (
    ChemistryRejection,
    RejectionReason,
    counted,
    merged,
    no_rejections,
)
from .rings import RingDescriptors, annotate_bond_topology

__all__ = [
    "CpuMatcher",
    "CpuReactionMatcher",
    "MatchResult",
    "find_matches",
    "generate_precursors",
    "has_match",
    "has_match_many",
]

# One refused application: (mol_idx, rule_id, app_idx, reason). Deliberately the same
# shape as a Hit, because the audit needs accepted and refused applications side by side:
# a single (molecule, rule) pair routinely contains both, so a pair-level status cannot
# describe it.
RejectedApplication = tuple[int, int, int, int]
RejectedApplicationArray = np.ndarray[tuple[int, int], np.dtype[np.int64]]


def rejected_application_array(
    rows: Sequence[RejectedApplication],
) -> RejectedApplicationArray:
    if not rows:
        return np.empty((0, 4), dtype=np.int64)
    return np.array(rows, dtype=np.int64)


AssignmentVector = np.ndarray[tuple[int], np.dtype[np.int32]]
AssignmentMatrix = np.ndarray[tuple[int, int], np.dtype[np.int32]]


# --------------------------------------------------------------------------- #
# Subgraph-isomorphism matcher (Ullmann refinement + assignment enumeration).
# --------------------------------------------------------------------------- #


class CpuMatcher(Matcher):
    """Reference CPU implementation of the Matcher protocol."""

    def has_match(self, pattern: GMol, molecule: GMol) -> bool:
        return has_match(pattern, molecule)

    def find_matches(
        self,
        pattern: GMol,
        molecule: GMol,
        *,
        max_assignments: int | None = None,
        prefix_depth: int | None = None,
        max_prune_rounds: int | None = None,
        max_leaf_checks: int | None = None,
        **kwargs: Any,
    ) -> MatchResult:
        return find_matches(
            pattern,
            molecule,
            return_first=max_assignments == 1,
            prefix_depth=prefix_depth,
            max_prune_rounds=max_prune_rounds,
            max_leaf_checks=max_leaf_checks,
        )

    def has_match_many(
        self,
        patterns: Sequence[GMol],
        molecules: Sequence[GMol],
        *,
        n_jobs: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        return has_match_many(patterns, molecules, n_jobs=n_jobs)


@dataclass
class SearchStats:
    assignments: list[AssignmentVector] = field(default_factory=list)
    prunings: int = 0
    pruning_complete: bool = True
    search_complete: bool = True
    branches_explored: int = 0
    leaves_explored: int = 0


def overlap_matrix(left: np.ndarray, right: np.ndarray) -> CandidateMatrix:
    return (left[:, None] & right[None, :]) != 0


def mass_compatibility(
    pattern_mass: np.ndarray, molecule_mass: np.ndarray
) -> CandidateMatrix:
    """Isotope is exact-or-any, so this is an equality test rather than a mask overlap.

    A query that says nothing about the nucleus (MASS_ANY) accepts every atom; a query
    that asserts a mass accepts only that mass, including MASS_UNLABELED.
    """
    return (pattern_mass[:, None] == MASS_ANY) | (
        pattern_mass[:, None] == molecule_mass[None, :]
    )


def initialize_candidates(pattern: GMol, molecule: GMol) -> CandidateMatrix:
    return mask_candidates(pattern, molecule) & predicate_candidates(pattern, molecule)


def predicate_candidates(pattern: GMol, molecule: GMol) -> CandidateMatrix:
    """Per pattern atom, the molecule atoms its recursive predicate allows.

    The reference evaluation: each body is matched afresh, recursively, for this one pair. The
    device backends batch the same work level by level instead.
    """
    rows = [
        predicate_atoms(
            predicate,
            molecule,
            {
                body: subpattern_roots(graph, molecule)
                for body, graph in predicate.subpatterns
            },
        )
        for predicate in pattern.predicates
    ]
    # Index -1, an atom with no predicate, picks the trailing all-allowed row.
    table = np.vstack([*rows, np.ones(molecule.atom_count, dtype=bool)])
    return table[pattern.atom_predicate]


def subpattern_roots(subpattern: GMol, molecule: GMol) -> AtomMask:
    """The molecule atoms a recursive body embeds at, rooted at the body's first atom.

    Each root candidate is pinned in turn and searched to its first embedding. Reading the
    roots off enumerated assignments instead would be wrong wherever enumeration is capped.
    """
    candidates = initialize_candidates(subpattern, molecule)
    roots = np.zeros(molecule.atom_count, dtype=bool)
    for root in np.flatnonzero(candidates[0]):
        pinned = candidates.copy()
        pinned[0] = False
        pinned[0, root] = True
        status = search_candidates(
            subpattern, molecule, pinned, return_first=True
        ).status
        if status == "unknown":
            raise RuntimeError(
                "A recursive-SMARTS body search ended without an answer; its roots are unknown."
            )
        roots[root] = status == "match"
    return roots


def mask_candidates(pattern: GMol, molecule: GMol) -> CandidateMatrix:
    p = pattern.atom_masks
    m = molecule.atom_masks
    return (
        (overlap_matrix(p.Z_lo, m.Z_lo) | overlap_matrix(p.Z_hi, m.Z_hi))
        & overlap_matrix(p.A, m.A)
        & overlap_matrix(p.C, m.C)
        & overlap_matrix(p.D, m.D)
        & overlap_matrix(p.H, m.H)
        & overlap_matrix(p.h, m.h)
        & overlap_matrix(p.v, m.v)
        & overlap_matrix(p.X, m.X)
        & overlap_matrix(p.R, m.R)
        & overlap_matrix(p.r, m.r)
        & overlap_matrix(p.x, m.x)
        & mass_compatibility(p.M, m.M)
    )


def prune(
    candidates: CandidateMatrix,
    pattern_edges: CandidateMatrix,
    molecule_edges: CandidateMatrix,
    *,
    max_prune_rounds: int | None = None,
) -> tuple[CandidateMatrix, int, bool]:
    """Prune one candidate matrix to fixpoint.

    `max_prune_rounds` bounds the number of refinement rounds in this single
    prune invocation. It is not a search-wide global pruning budget.
    """
    if candidates.size == 0:
        return candidates.copy(), 0, True

    current = candidates.copy()
    prune_rounds = 0
    pattern_edges_i16 = pattern_edges.astype(np.int16, copy=False)
    molecule_edges_i16 = molecule_edges.astype(np.int16, copy=False)
    pattern_degree = pattern_edges_i16.sum(axis=1, keepdims=True, dtype=np.int16)

    while True:
        if max_prune_rounds is not None and prune_rounds >= max_prune_rounds:
            return current, prune_rounds, False

        support = (current.astype(np.int16, copy=False) @ molecule_edges_i16.T) > 0
        satisfied_neighbors = pattern_edges_i16 @ support.astype(np.int16, copy=False)
        refined = current & (satisfied_neighbors == pattern_degree)
        prune_rounds += 1
        if np.array_equal(refined, current):
            return refined, prune_rounds, True
        current = refined
        if not has_support(current):
            return current, prune_rounds, True


def enumerate_assignments(
    candidates: CandidateMatrix,
    pattern_bonds: BondMatrix,
    molecule_bonds: BondMatrix,
    *,
    prefix_depth: int | None = None,
    return_first: bool = False,
    max_prune_rounds: int | None = None,
    max_leaf_checks: int | None = None,
    pattern_aromaticity: Aromaticity | None = None,
    molecule_aromaticity: Aromaticity | None = None,
    pattern_groups: AtomMapVector | None = None,
    molecule_components: AtomIndices | None = None,
) -> MatchResult:
    validate_bounds(
        prefix_depth=prefix_depth,
        max_prune_rounds=max_prune_rounds,
        max_leaf_checks=max_leaf_checks,
    )
    stats = SearchStats()
    pattern_edges = bond_presence(pattern_bonds)
    molecule_edges = bond_presence(molecule_bonds)
    search_assignments(
        candidates,
        pattern_edges,
        molecule_edges,
        pattern_bonds,
        molecule_bonds,
        pattern_aromaticity,
        molecule_aromaticity,
        pattern_groups,
        molecule_components,
        stats,
        prefix_depth=prefix_depth,
        return_first=return_first,
        max_prune_rounds=max_prune_rounds,
        max_leaf_checks=max_leaf_checks,
    )
    return finalize_result(candidates.shape[0], stats)


def find_matches(
    pattern: GMol,
    molecule: GMol,
    *,
    prefix_depth: int | None = None,
    return_first: bool = False,
    max_prune_rounds: int | None = None,
    max_leaf_checks: int | None = None,
) -> MatchResult:
    """Run the full matcher pipeline and return assignments plus diagnostics."""

    validate_bounds(
        prefix_depth=prefix_depth,
        max_prune_rounds=max_prune_rounds,
        max_leaf_checks=max_leaf_checks,
    )
    return search_candidates(
        pattern,
        molecule,
        initialize_candidates(pattern, molecule),
        prefix_depth=prefix_depth,
        return_first=return_first,
        max_prune_rounds=max_prune_rounds,
        max_leaf_checks=max_leaf_checks,
    )


def search_candidates(
    pattern: GMol,
    molecule: GMol,
    candidates: CandidateMatrix,
    *,
    prefix_depth: int | None = None,
    return_first: bool = False,
    max_prune_rounds: int | None = None,
    max_leaf_checks: int | None = None,
) -> MatchResult:
    """The search from given initial candidates, so a caller can narrow them first."""
    pattern_edges = bond_presence(pattern.bonds)
    molecule_edges = bond_presence(molecule.bonds)
    # Resolve the molecule's ANY topology masks to exact ring-or-chain states.
    molecule_bonds = annotate_bond_topology(molecule.bonds)
    # Zero-level grouping, and only when the pattern writes one: an unparenthesised pattern pays
    # for no traversal and reaches an untouched `accept_assignment`.
    grouped = bool(np.any(pattern.atom_group))
    pattern_groups = pattern.atom_group if grouped else None
    molecule_components = (
        component_labels_from_bonds(molecule.bonds) if grouped else None
    )
    pruned, prune_rounds, pruning_complete = prune(
        candidates,
        pattern_edges,
        molecule_edges,
        max_prune_rounds=max_prune_rounds,
    )
    stats = SearchStats(
        prunings=prune_rounds,
        pruning_complete=pruning_complete,
    )
    search_assignments(
        pruned,
        pattern_edges,
        molecule_edges,
        pattern.bonds,
        molecule_bonds,
        pattern.atom_masks.A,
        molecule.atom_masks.A,
        pattern_groups,
        molecule_components,
        stats,
        prefix_depth=prefix_depth,
        return_first=return_first,
        max_prune_rounds=max_prune_rounds,
        max_leaf_checks=max_leaf_checks,
    )
    return finalize_result(candidates.shape[0], stats)


def has_match(pattern: GMol, molecule: GMol) -> bool:
    """Return only whether the pattern matches the molecule."""

    return find_matches(pattern, molecule, return_first=True).status == "match"


_G_HAS_MATCH_PATTERNS: tuple[GMol, ...] | None = None
_G_HAS_MATCH_MOLECULES: tuple[GMol, ...] | None = None


def _init_has_match_globals(
    patterns: tuple[GMol, ...], molecules: tuple[GMol, ...]
) -> None:
    global _G_HAS_MATCH_PATTERNS, _G_HAS_MATCH_MOLECULES
    _G_HAS_MATCH_PATTERNS = patterns
    _G_HAS_MATCH_MOLECULES = molecules


def _eval_has_match_row(pattern_idx: int) -> AtomMask:
    if _G_HAS_MATCH_PATTERNS is None or _G_HAS_MATCH_MOLECULES is None:
        raise RuntimeError(
            "has_match worker globals unset; run _init_has_match_globals in the worker first."
        )
    pattern = _G_HAS_MATCH_PATTERNS[pattern_idx]
    return np.asarray(
        [has_match(pattern, molecule) for molecule in _G_HAS_MATCH_MOLECULES],
        dtype=bool,
    )


def has_match_many(
    patterns: Sequence[GMol],
    molecules: Sequence[GMol],
    *,
    n_jobs: int | None = None,
) -> CandidateMatrix:
    """Return a boolean matrix of pairwise match existence."""

    jobs = resolve_n_jobs(n_jobs)
    matches = np.zeros((len(patterns), len(molecules)), dtype=bool)
    if jobs != 1 and len(patterns) and len(molecules):
        rows = parallel_map_fork(
            _eval_has_match_row,
            tuple(range(len(patterns))),
            initializer=_init_has_match_globals,
            initargs=(tuple(patterns), tuple(molecules)),
            n_jobs=jobs,
        )
        return np.vstack(rows).astype(bool, copy=False)

    for i, pattern in enumerate(patterns):
        for j, molecule in enumerate(molecules):
            matches[i, j] = has_match(pattern, molecule)
    return matches


def has_support(candidates: CandidateMatrix) -> bool:
    return bool(np.all(candidates.any(axis=1)))


def validate_bounds(
    *,
    prefix_depth: int | None,
    max_prune_rounds: int | None,
    max_leaf_checks: int | None,
) -> None:
    if prefix_depth is not None and prefix_depth < 0:
        raise ValueError("prefix_depth must be non-negative when provided.")
    if max_prune_rounds is not None and max_prune_rounds <= 0:
        raise ValueError("max_prune_rounds must be positive when provided.")
    if max_leaf_checks is not None and max_leaf_checks <= 0:
        raise ValueError("max_leaf_checks must be positive when provided.")


def search_assignments(
    candidates: CandidateMatrix,
    pattern_edges: CandidateMatrix,
    molecule_edges: CandidateMatrix,
    pattern_bonds: BondMatrix,
    molecule_bonds: BondMatrix,
    pattern_aromaticity: Aromaticity | None,
    molecule_aromaticity: Aromaticity | None,
    pattern_groups: AtomMapVector | None,
    molecule_components: AtomIndices | None,
    stats: SearchStats,
    *,
    prefix_depth: int | None,
    return_first: bool,
    max_prune_rounds: int | None,
    max_leaf_checks: int | None,
) -> bool:
    atom_count = candidates.shape[0]
    if atom_count == 0:
        return accept_assignment(
            np.zeros(0, dtype=np.int32),
            pattern_edges,
            molecule_edges,
            pattern_bonds,
            molecule_bonds,
            pattern_aromaticity,
            molecule_aromaticity,
            pattern_groups,
            molecule_components,
            stats,
            return_first=return_first,
            max_leaf_checks=max_leaf_checks,
        )

    if not has_support(candidates):
        return False

    row_counts = np.sum(candidates, axis=1)
    if np.all(row_counts == 1):
        return accept_assignment(
            np.argmax(candidates, axis=1).astype(np.int32, copy=False),
            pattern_edges,
            molecule_edges,
            pattern_bonds,
            molecule_bonds,
            pattern_aromaticity,
            molecule_aromaticity,
            pattern_groups,
            molecule_components,
            stats,
            return_first=return_first,
            max_leaf_checks=max_leaf_checks,
        )

    if prefix_depth == 0:
        return enumerate_leaves(
            candidates,
            pattern_edges,
            molecule_edges,
            pattern_bonds,
            molecule_bonds,
            pattern_aromaticity,
            molecule_aromaticity,
            pattern_groups,
            molecule_components,
            stats,
            return_first=return_first,
            max_leaf_checks=max_leaf_checks,
        )

    branch_row = int(
        np.flatnonzero(row_counts == np.min(row_counts[row_counts > 1]))[0]
    )
    branch_columns = np.flatnonzero(candidates[branch_row])
    next_prefix_depth = None if prefix_depth is None else prefix_depth - 1

    for branch_column in branch_columns:
        stats.branches_explored += 1
        branch = candidates.copy()
        branch[branch_row, :] = False
        branch[:, branch_column] = False
        branch[branch_row, branch_column] = True
        pruned, prune_rounds, pruning_complete = prune(
            branch,
            pattern_edges,
            molecule_edges,
            max_prune_rounds=max_prune_rounds,
        )
        stats.prunings += prune_rounds
        stats.pruning_complete = stats.pruning_complete and pruning_complete
        if search_assignments(
            pruned,
            pattern_edges,
            molecule_edges,
            pattern_bonds,
            molecule_bonds,
            pattern_aromaticity,
            molecule_aromaticity,
            pattern_groups,
            molecule_components,
            stats,
            prefix_depth=next_prefix_depth,
            return_first=return_first,
            max_prune_rounds=max_prune_rounds,
            max_leaf_checks=max_leaf_checks,
        ):
            return True
        if not stats.search_complete and return_first:
            return True

    return False


def enumerate_leaves(
    candidates: CandidateMatrix,
    pattern_edges: CandidateMatrix,
    molecule_edges: CandidateMatrix,
    pattern_bonds: BondMatrix,
    molecule_bonds: BondMatrix,
    pattern_aromaticity: Aromaticity | None,
    molecule_aromaticity: Aromaticity | None,
    pattern_groups: AtomMapVector | None,
    molecule_components: AtomIndices | None,
    stats: SearchStats,
    *,
    return_first: bool,
    max_leaf_checks: int | None,
) -> bool:
    row_counts = np.sum(candidates, axis=1)
    fixed_rows = np.flatnonzero(row_counts == 1)
    ambiguous_rows = np.flatnonzero(row_counts > 1)
    if ambiguous_rows.size == 0:
        return accept_assignment(
            np.argmax(candidates, axis=1).astype(np.int32, copy=False),
            pattern_edges,
            molecule_edges,
            pattern_bonds,
            molecule_bonds,
            pattern_aromaticity,
            molecule_aromaticity,
            pattern_groups,
            molecule_components,
            stats,
            return_first=return_first,
            max_leaf_checks=max_leaf_checks,
        )

    order = np.lexsort((ambiguous_rows, row_counts[ambiguous_rows]))
    ordered_rows = ambiguous_rows[order]
    ordered_columns = [np.flatnonzero(candidates[row]) for row in ordered_rows]

    current = np.full(candidates.shape[0], -1, dtype=np.int32)
    if fixed_rows.size:
        current[fixed_rows] = np.argmax(candidates[fixed_rows], axis=1).astype(
            np.int32, copy=False
        )

    def walk(depth: int) -> bool:
        if not stats.search_complete:
            return True
        if depth == len(ordered_rows):
            return accept_assignment(
                current.copy(),
                pattern_edges,
                molecule_edges,
                pattern_bonds,
                molecule_bonds,
                pattern_aromaticity,
                molecule_aromaticity,
                pattern_groups,
                molecule_components,
                stats,
                return_first=return_first,
                max_leaf_checks=max_leaf_checks,
            )

        row = int(ordered_rows[depth])
        for column in ordered_columns[depth]:
            current[row] = int(column)
            if walk(depth + 1):
                return True
        current[row] = -1
        return False

    return walk(0)


def accept_assignment(
    assignment: AssignmentVector,
    pattern_edges: CandidateMatrix,
    molecule_edges: CandidateMatrix,
    pattern_bonds: BondMatrix,
    molecule_bonds: BondMatrix,
    pattern_aromaticity: Aromaticity | None,
    molecule_aromaticity: Aromaticity | None,
    pattern_groups: AtomMapVector | None,
    molecule_components: AtomIndices | None,
    stats: SearchStats,
    *,
    return_first: bool,
    max_leaf_checks: int | None,
) -> bool:
    if max_leaf_checks is not None and stats.leaves_explored >= max_leaf_checks:
        stats.search_complete = False
        return True

    stats.leaves_explored += 1

    if len(np.unique(assignment)) != assignment.size:
        return False

    matched_edges = molecule_edges[np.ix_(assignment, assignment)]
    if not np.all(pattern_edges <= matched_edges):
        return False

    matched_bonds = molecule_bonds[np.ix_(assignment, assignment)]
    if not assignment_bond_types_compatible(
        pattern_bonds,
        matched_bonds,
    ):
        return False

    if pattern_groups is not None and molecule_components is not None:
        if not grouping_agrees_with_components(
            pattern_groups, molecule_components[assignment]
        ):
            return False

    stats.assignments.append(assignment)
    if return_first:
        stats.search_complete = False
        return True
    return False


def assignment_bond_types_compatible(
    pattern_bonds: BondMatrix,
    matched_bonds: BondMatrix,
) -> bool:
    compatible = bond_type_compatibility_matrix(pattern_bonds, matched_bonds)
    return bool(np.all(compatible))


def finalize_result(atom_count: int, stats: SearchStats) -> MatchResult:
    if stats.assignments:
        assignments = np.stack(stats.assignments).astype(np.int32, copy=False)
        status = "match"
    else:
        assignments = np.zeros((0, atom_count), dtype=np.int32)
        status = (
            "no_match"
            if stats.pruning_complete and stats.search_complete
            else "unknown"
        )
    return MatchResult(
        status=status,
        assignments=assignments,
        prunings=stats.prunings,
        pruning_complete=stats.pruning_complete,
        search_complete=stats.search_complete,
        leaves_explored=stats.leaves_explored,
        metadata={"branches_explored": stats.branches_explored},
    )


# --------------------------------------------------------------------------- #
# Reaction-matcher layer: subgraph match + bond edits into precursor mols.
# --------------------------------------------------------------------------- #


class CpuReactionMatcher:
    """Reaction matcher over `CMol` molecules backed by the CPU isomorphism matcher.

    Compiles a rule set to `ReactionPattern`s; `match` runs `find_matches` for each
    `(mol_idx, rule_id)` pair and splits every assignment into precursor molecules
    via `precursors_from_mapping`. No deduplication — identical steps collapse
    downstream in key space.
    """

    def __init__(
        self,
        rules: Sequence[ReactionPattern],
        *,
        matcher: Matcher | None = None,
        max_assignments: int = 256,
        max_precursor_atoms: int | None = None,
        processing: ReactionProcessing = DEFAULT_REACTION_PROCESSING,
    ) -> None:
        validate_cpu_reaction_processing(processing)
        self._patterns = tuple(rules)
        self._matcher: Matcher = matcher if matcher is not None else CpuMatcher()
        self._max_assignments = max_assignments
        self._max_precursor_atoms = max_precursor_atoms
        self.processing = processing
        self.last_diagnostics: PrecursorOverflowDiagnostics | None = None
        # Accepted applications are described by the returned hits; refused ones by these
        # rows. Together they account for every application attempted, which is what an
        # audit needs -- one (molecule, rule) pair routinely contains both.
        self.last_rejected_applications: RejectedApplicationArray = (
            rejected_application_array([])
        )

    def match_batches(
        self,
        mols: Sequence[CMol],
        pairs: Sequence[Pair] | None = None,
        *,
        n_jobs: int | None = None,
        pairs_per_batch: int = 4096,
    ) -> Iterator[tuple[np.ndarray, HitArray]]:
        """Deliver bounded pair chunks, with no joined precursor array or identity work.

        One worker window is retained at a time. Workers inherit the rules and molecule
        graphs once per call, as in the joined parallel path; they are not respawned for
        each chunk. Diagnostics describe delivered chunks, including when the iterator
        is closed early. Precursor indices are local to each delivered batch.
        """
        if pairs_per_batch <= 0:
            raise ValueError("pairs_per_batch must be positive")
        if pairs is None:
            pairs = [
                (mol_idx, rule_id)
                for mol_idx in range(len(mols))
                for rule_id in range(len(self._patterns))
            ]
        gmols = {
            index: gmol_from_cmol(mols[index])
            for index in {index for index, _ in pairs}
        }
        diagnostics = PrecursorOverflowDiagnostics(
            precursor_atom_cap=self._max_precursor_atoms
        )
        refused: list[RejectedApplicationArray] = []
        pair_chunks = chunked(pairs, pairs_per_batch)
        try:
            with ResidentForkPool(
                initializer=_init_react_globals,
                initargs=(
                    tuple(mols),
                    gmols,
                    self._patterns,
                    self._matcher,
                    self._max_assignments,
                    self._max_precursor_atoms,
                    self.processing,
                ),
                n_jobs=n_jobs,
            ) as workers:
                for window in chunked(pair_chunks, workers.n_workers):
                    results = workers.map(_eval_react_chunk, window)
                    for precursors, hits, partial, rejected in results:
                        diagnostics = add_precursor_overflow_diagnostics(
                            diagnostics, partial
                        )
                        if rejected.size:
                            refused.append(rejected)
                        if len(hits):
                            yield precursors, hits
                        del precursors, hits, partial, rejected
                    del results
        finally:
            self.last_diagnostics = diagnostics
            self.last_rejected_applications = (
                np.concatenate(refused) if refused else rejected_application_array([])
            )
            warn_precursor_overflow(diagnostics, backend="CPU retro")

    def match(
        self,
        mols: Sequence[CMol],
        pairs: Sequence[Pair] | None = None,
        *,
        n_jobs: int | None = None,
    ) -> tuple[np.ndarray, HitArray]:
        if pairs is None:
            pairs = [
                (mol_idx, rule_id)
                for mol_idx in range(len(mols))
                for rule_id in range(len(self._patterns))
            ]

        jobs = resolve_n_jobs(n_jobs)
        if jobs != 1:
            precursors, hits, diagnostics, refused_rows = match_cpu_reactions_parallel(
                mols=tuple(mols),
                pairs=tuple(pairs),
                patterns=self._patterns,
                matcher=self._matcher,
                max_assignments=self._max_assignments,
                max_precursor_atoms=self._max_precursor_atoms,
                processing=self.processing,
                n_jobs=jobs,
            )
            self.last_diagnostics = diagnostics
            self.last_rejected_applications = refused_rows
            warn_precursor_overflow(diagnostics, backend="CPU retro")
            return precursors, hits

        gmols: dict[int, GMol] = {}
        precursors: list[CMol] = []
        hits: list[Hit] = []
        rejections = no_rejections()
        refused: list[RejectedApplication] = []
        diagnostics = PrecursorOverflowDiagnostics(
            precursor_atom_cap=self._max_precursor_atoms
        )

        for mol_idx, rule_id in pairs:
            pattern = self._patterns[rule_id]
            gmol = gmols.get(mol_idx)
            if gmol is None:
                gmol = gmol_from_cmol(mols[mol_idx])
                gmols[mol_idx] = gmol

            result = self._matcher.find_matches(
                pattern.lhs,
                gmol,
                max_assignments=self._max_assignments,
            )
            if result.status != "match" or result.assignments.size == 0:
                # The matcher produced nothing for this pair. "no_match" is an answer;
                # "unknown" means it gave up, and counting the two separately is what
                # keeps an exhausted search from looking like an absent one.
                diagnostics = replace(
                    diagnostics,
                    pairs_declined=diagnostics.pairs_declined + 1,
                    pairs_search_incomplete=diagnostics.pairs_search_incomplete
                    + int(result.status == "unknown"),
                )
                continue

            diagnostics = replace(
                diagnostics,
                applications_attempted=diagnostics.applications_attempted
                + int(result.assignments.shape[0]),
            )
            for app_idx, mapping in enumerate(result.assignments):
                mapped = applied_precursors(
                    product=mols[mol_idx],
                    reaction=pattern,
                    mapping=mapping,
                    ring_descriptors=post_split_ring_descriptors(self.processing),
                    max_precursor_atoms=self._max_precursor_atoms,
                )
                if isinstance(mapped, RejectionReason):
                    rejections = merged(rejections, counted(mapped))
                    refused.append((mol_idx, rule_id, app_idx, int(mapped)))
                    continue
                diagnostics = add_precursor_overflow_diagnostics(
                    diagnostics,
                    stamped_undescribed(mapped.diagnostics, mol_idx, rule_id, app_idx),
                )
                for precursor in mapped.precursors:
                    hits.append((mol_idx, rule_id, app_idx, len(precursors)))
                    precursors.append(precursor)

        diagnostics = replace(diagnostics, rejections=rejections)
        self.last_diagnostics = diagnostics
        self.last_rejected_applications = rejected_application_array(refused)
        warn_precursor_overflow(diagnostics, backend="CPU retro")
        return np.array(precursors, dtype=object), hit_array(hits)


def applied_precursors(
    *,
    product: CMol,
    reaction: ReactionPattern,
    mapping: np.ndarray,
    ring_descriptors: RingDescriptors,
    max_precursor_atoms: int | None,
):
    """Apply one assignment, or the reason the chemistry refused it.

    Batch callers use this so an application the model cannot express costs only itself:
    every other assignment, molecule and rule in the run keeps its result, and the
    refusal is counted under the reason it named rather than as an anonymous failure.
    `generate_precursors` deliberately calls the underlying function instead, so a caller
    diagnosing one template against one molecule is told why rather than handed an empty
    list.
    """
    try:
        return precursors_from_mapping_result(
            product=product,
            reaction=reaction,
            mapping=mapping,
            ring_descriptors=ring_descriptors,
            max_precursor_atoms=max_precursor_atoms,
        )
    except ChemistryRejection as rejection:
        return rejection.reason


@dataclass(frozen=True)
class CpuReactionChunk:
    mols: tuple[CMol, ...]
    gmols: dict[int, GMol]
    patterns: tuple[ReactionPattern, ...]
    pairs: tuple[Pair, ...]
    matcher: Matcher
    max_assignments: int
    max_precursor_atoms: int | None
    processing: ReactionProcessing


_G_REACT_MOLS: tuple[CMol, ...] | None = None
_G_REACT_GMOLS: dict[int, GMol] | None = None
_G_REACT_PATTERNS: tuple[ReactionPattern, ...] | None = None
_G_REACT_MATCHER: Matcher | None = None
_G_REACT_MAX_ASSIGNMENTS: int | None = None
_G_REACT_MAX_PRECURSOR_ATOMS: int | None = None
_G_REACT_PROCESSING: ReactionProcessing | None = None


def _init_react_globals(
    mols: tuple[CMol, ...],
    gmols: dict[int, GMol],
    patterns: tuple[ReactionPattern, ...],
    matcher: Matcher,
    max_assignments: int,
    max_precursor_atoms: int | None,
    processing: ReactionProcessing,
) -> None:
    global _G_REACT_MOLS, _G_REACT_GMOLS, _G_REACT_PATTERNS
    global _G_REACT_MATCHER, _G_REACT_MAX_ASSIGNMENTS, _G_REACT_MAX_PRECURSOR_ATOMS
    global _G_REACT_PROCESSING
    _G_REACT_MOLS = mols
    _G_REACT_GMOLS = gmols
    _G_REACT_PATTERNS = patterns
    _G_REACT_MATCHER = matcher
    _G_REACT_MAX_ASSIGNMENTS = max_assignments
    _G_REACT_MAX_PRECURSOR_ATOMS = max_precursor_atoms
    _G_REACT_PROCESSING = processing


def _eval_react_chunk(
    pairs: tuple[Pair, ...],
) -> tuple[
    np.ndarray, HitArray, PrecursorOverflowDiagnostics, RejectedApplicationArray
]:
    if (
        _G_REACT_MOLS is None
        or _G_REACT_GMOLS is None
        or _G_REACT_PATTERNS is None
        or _G_REACT_MATCHER is None
        or _G_REACT_MAX_ASSIGNMENTS is None
        or _G_REACT_PROCESSING is None
    ):
        raise RuntimeError(
            "react worker globals unset; run _init_react_globals in the worker first."
        )
    max_precursor_atoms = _G_REACT_MAX_PRECURSOR_ATOMS

    precursors: list[CMol] = []
    hits: list[Hit] = []
    rejections = no_rejections()
    refused: list[RejectedApplication] = []
    diagnostics = PrecursorOverflowDiagnostics(precursor_atom_cap=max_precursor_atoms)
    for mol_idx, rule_id in pairs:
        pattern = _G_REACT_PATTERNS[rule_id]
        result = _G_REACT_MATCHER.find_matches(
            pattern.lhs,
            _G_REACT_GMOLS[mol_idx],
            max_assignments=_G_REACT_MAX_ASSIGNMENTS,
        )
        if result.status != "match" or result.assignments.size == 0:
            diagnostics = replace(
                diagnostics,
                pairs_declined=diagnostics.pairs_declined + 1,
                pairs_search_incomplete=diagnostics.pairs_search_incomplete
                + int(result.status == "unknown"),
            )
            continue
        diagnostics = replace(
            diagnostics,
            applications_attempted=diagnostics.applications_attempted
            + int(result.assignments.shape[0]),
        )
        for app_idx, mapping in enumerate(result.assignments):
            mapped = applied_precursors(
                product=_G_REACT_MOLS[mol_idx],
                reaction=pattern,
                mapping=mapping,
                ring_descriptors=post_split_ring_descriptors(_G_REACT_PROCESSING),
                max_precursor_atoms=max_precursor_atoms,
            )
            if isinstance(mapped, RejectionReason):
                rejections = merged(rejections, counted(mapped))
                refused.append((mol_idx, rule_id, app_idx, int(mapped)))
                continue
            diagnostics = add_precursor_overflow_diagnostics(
                diagnostics,
                # Stamped here as well as in the serial path, which is what its docstring
                # already claims keeps the two identical. It did not: stamping moves the
                # dropped fragments' formulas into `undescribed_rows` with the
                # application that produced them, and the merge carries `undescribed_rows`
                # but not `undescribed_formulas` -- so an unstamped worker's formulas were
                # silently discarded and the attribution was empty whenever `n_jobs != 1`.
                stamped_undescribed(mapped.diagnostics, mol_idx, rule_id, app_idx),
            )
            for precursor in mapped.precursors:
                hits.append((mol_idx, rule_id, app_idx, len(precursors)))
                precursors.append(precursor)
    diagnostics = replace(diagnostics, rejections=rejections)
    return (
        np.array(precursors, dtype=object),
        hit_array(hits),
        diagnostics,
        rejected_application_array(refused),
    )


def match_cpu_reactions_parallel(
    *,
    mols: tuple[CMol, ...],
    pairs: tuple[Pair, ...],
    patterns: tuple[ReactionPattern, ...],
    matcher: Matcher,
    max_assignments: int,
    max_precursor_atoms: int | None,
    processing: ReactionProcessing,
    n_jobs: int,
) -> tuple[
    np.ndarray, HitArray, PrecursorOverflowDiagnostics, RejectedApplicationArray
]:
    gmols = {
        mol_idx: gmol_from_cmol(mols[mol_idx])
        for mol_idx in {mol_idx for mol_idx, _ in pairs}
    }
    pair_chunks = parallel_chunks(pairs, n_jobs=n_jobs)
    results = parallel_map_fork(
        _eval_react_chunk,
        pair_chunks,
        initializer=_init_react_globals,
        initargs=(
            mols,
            gmols,
            patterns,
            matcher,
            max_assignments,
            max_precursor_atoms,
            processing,
        ),
        n_jobs=n_jobs,
    )
    diagnostics = PrecursorOverflowDiagnostics(precursor_atom_cap=max_precursor_atoms)
    for _, _, result_diagnostics, _ in results:
        diagnostics = add_precursor_overflow_diagnostics(
            diagnostics,
            result_diagnostics,
        )
    refused = [rows for _, _, _, rows in results if rows.size]
    return (
        concat_object_arrays(tuple(precursors for precursors, _, _, _ in results)),
        concat_hits(
            tuple(hits for _, hits, _, _ in results),
            tuple(len(precursors) for precursors, _, _, _ in results),
        ),
        diagnostics,
        np.concatenate(refused) if refused else np.empty((0, 4), dtype=np.int64),
    )


def validate_cpu_reaction_processing(processing: ReactionProcessing) -> None:
    if processing.before_match != MoleculeProcessing.NONE:
        raise ValueError("CPU reaction processing before_match is not implemented yet.")
    required = MoleculeProcessing.AROMATICITY
    if processing.after_split & required != required:
        raise ValueError(
            "CPU reaction processing after_split must include AROMATICITY."
        )
    supported = MoleculeProcessing.AROMATICITY | MoleculeProcessing.RING_DESCRIPTORS
    if int(processing.after_split) & ~int(supported):
        raise ValueError(
            "CPU reaction processing after_split contains unsupported steps."
        )


def post_split_ring_descriptors(processing: ReactionProcessing) -> RingDescriptors:
    """The per-atom ring descriptors this policy asks a split precursor to carry."""
    selected = RingDescriptors.NONE
    if processing.after_split & MoleculeProcessing.RING_CONNECTIVITY:
        selected |= RingDescriptors.CONNECTIVITY
    if processing.after_split & MoleculeProcessing.RING_MEMBERSHIP:
        selected |= RingDescriptors.MEMBERSHIP
    return selected


def concat_object_arrays(parts: Sequence[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.array([], dtype=object)
    if not any(len(part) for part in parts):
        return np.array([], dtype=object)
    return np.concatenate(parts).astype(object, copy=False)


def edit_signature(
    edit: ReactionEdit, mapping: ReactionMapping
) -> tuple[Any, Any, Any]:
    """What an assignment will *do* to the product, as a hashable key.

    An embedding and a permutation of it are different injective maps and the matcher is right to
    return both -- injectivity is not symmetry. But where a pattern names a reagent more than
    once, the permutations all rewrite the product identically, and building precursors for each
    before deduplicating them costs the automorphism count. `FWD MC 3` written with three
    interchangeable SnCl2 and six interchangeable HCl is 34,560 embeddings of which one survives:
    4.4 seconds to find them and twenty-seven more to construct and discard them.

    Three parts, and between them they settle the rewrite completely:

    * where each *mapped* atom lands, in order, since each carries its own right-hand-side
      properties and bonds;
    * which product atoms are *deleted*, as a set, because a deleted atom takes its bonds with it
      and nothing distinguishes one deletion from another;
    * which product *bonds* are removed, as a set of unordered pairs.

    Everything else the edit adds -- new atoms, new bonds -- is invented the same way every time
    and attaches at the mapped positions, so it is implied by the first part. Two assignments
    sharing this key therefore produce the same precursors, which is what makes skipping the
    second one sound rather than merely cheaper.

    This does not replace the outcome-level deduplication below it. Two assignments can rewrite
    the product differently and still arrive at the same precursors, and only comparing the
    results catches that.
    """
    mapped = tuple(mapping[edit.mapped_lhs_indices].tolist())
    deleted = frozenset(mapping[edit.deleted_lhs_indices].tolist())
    if edit.removed_lhs_edges.size:
        ends = np.sort(mapping[edit.removed_lhs_edges], axis=1)
        removed = frozenset(map(tuple, ends.tolist()))
    else:
        removed = frozenset()
    return mapped, deleted, removed


def generate_precursors(
    product: CMol,
    reaction: ReactionPattern,
    max_mappings: int | None = None,
    matcher: Matcher | None = None,
    ring_descriptors: RingDescriptors = RingDescriptors.ALL,
    max_precursor_atoms: int | None = None,
) -> Sequence[RetroOutcome]:
    if matcher is None:
        matcher = CpuMatcher()

    product_gmol = gmol_from_cmol(product)
    result = matcher.find_matches(
        reaction.lhs,
        product_gmol,
        max_assignments=max_mappings,
    )
    assignments = result.assignments
    if max_mappings is not None:
        assignments = assignments[:max_mappings]
    if assignments.size == 0:
        return []

    outcomes: list[RetroOutcome] = []
    seen: set[Any] = set()
    # Assignments that rewrite the product the same way, before paying to find out -- see
    # `edit_signature`. Kept apart from `seen` because the two catch different things.
    #
    # One consequence worth knowing: a suppressed assignment never reaches
    # `add_precursor_overflow_diagnostics`, so these counts are per *distinct edit* now rather
    # than per assignment. A symmetric pattern that orphans a component used to report its
    # automorphism count -- "refused 288 application(s)" for one disconnection -- and now reports
    # one. More honest, and a change to what that warning's numbers mean.
    #
    # The batch path does *not* share this filter. `CpuReactionMatcher.react` builds precursors
    # for every assignment because `app_idx` is `enumerate(result.assignments)` and those indices
    # key the `Hit` and `RejectedApplication` rows a run audits, so skipping there would renumber
    # that stream. Measured, the symmetric nitro reduction's forward rule yields 288 identical
    # applications and 1152 precursor molecules for one real disconnection. The *retro* direction
    # is unaffected either way: a retro rule's symmetry sits in reagents on its right-hand side
    # while its left-hand side is one component, and the same rule measures 1 application there.
    # So the forward probe has a real opportunity here, and it is not a pure optimisation --
    # collapsing those applications changes what the selectivity channels count.
    edits: set[tuple[Any, Any, Any]] = set()
    diagnostics = PrecursorOverflowDiagnostics(precursor_atom_cap=max_precursor_atoms)
    for mapping in assignments:
        edit = edit_signature(reaction.edit, mapping)
        if edit in edits:
            continue
        edits.add(edit)
        mapped = precursors_from_mapping_result(
            product=product,
            reaction=reaction,
            mapping=mapping,
            ring_descriptors=ring_descriptors,
            max_precursor_atoms=max_precursor_atoms,
        )
        diagnostics = add_precursor_overflow_diagnostics(
            diagnostics,
            mapped.diagnostics,
        )
        precursors = mapped.precursors
        if not precursors:
            continue
        signature = precursor_signature(precursors)
        if signature in seen:
            continue
        seen.add(signature)
        outcomes.append(RetroOutcome(mapping=mapping.copy(), precursors=precursors))

    warn_precursor_overflow(diagnostics, backend="CPU retro")
    return outcomes

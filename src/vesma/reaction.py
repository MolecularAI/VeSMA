from __future__ import annotations

import hashlib
import os
import pickle
import platform
import tempfile
import warnings
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import IntFlag, auto
from pathlib import Path

import numpy as np

from .adjacency import (
    BondMatrix,
    bond_delocalization,
    bond_order_matrix,
    bond_presence,
)
from .bonds import (
    EXACT_BOND_KINDS,
    BondKind,
    bond_kind_flags,
    resolve_default_bond_flags,
)
from .cmol import CMol
from .cmol_builder import cmol_from_seed_data
from .connected_components import (
    connected_components_from_bonds,
    grouping_agrees_with_components,
)
from .gmol import GMol
from .gmol_atoms import (
    AROMATIC,
    BUCKET_DEFAULT_MASKS,
    MASS_ANY,
    MASS_UNLABELED,
    AtomIndices,
    AtomVector,
    Charge,
    bucket_mask_exact,
)
from .gmol_builder import gmol_from_smarts
from .hydrogens import (
    HYDROGEN_UNPINNED,
    REASON_BY_HYDROGEN_FAILURE,
    HydrogenFailure,
    hydrogens_after_edit,
    hydrogens_for_written_state,
)
from .loading import cache_dir, cached_function, parallel_map, resolve_cache
from .parsing.errors import SmartsError
from .rejection import (
    ChemistryRejection,
    RejectionCounts,
    RejectionReason,
    by_name,
    merged,
    no_rejections,
)
from .rings import RingDescriptors
from .symbols import Z_TO_ELEMENT

AtomIndexVector = np.ndarray[tuple[int], np.dtype[np.int32]]
AtomIndexPairs = np.ndarray[tuple[int, int], np.dtype[np.int32]]  # (n_edges, 2)
AtomMapVector = np.ndarray[tuple[int], np.dtype[np.int32]]
ReactionMapping = np.ndarray[tuple[int], np.dtype[np.int32]]
PrecursorSignature = tuple[tuple[bytes, bytes, bytes, bytes], ...]


class MoleculeProcessing(IntFlag):
    """Selectable molecule-processing operations around reaction application.

    Backends own the ordering and implementation of the selected operations.  Keeping
    the selections separate from their execution stage lets the policy grow to cover
    pre-match perception and representation transforms without replacing matcher APIs.
    """

    NONE = 0
    AROMATICITY = auto()
    # x, the incident ring-bond count, which every backend can compute exactly.
    RING_CONNECTIVITY = auto()
    # R and r from the relevant cycles, independent of atom order on every backend.
    RING_MEMBERSHIP = auto()
    RING_DESCRIPTORS = RING_CONNECTIVITY | RING_MEMBERSHIP


@dataclass(frozen=True)
class ReactionProcessing:
    """Molecule processing requested before matching and after precursor splitting."""

    before_match: MoleculeProcessing = MoleculeProcessing.NONE
    after_split: MoleculeProcessing = (
        MoleculeProcessing.AROMATICITY | MoleculeProcessing.RING_DESCRIPTORS
    )


DEFAULT_REACTION_PROCESSING = ReactionProcessing()
# Both device backends now perceive the same ring descriptors as the CPU reference.
GPU_PARITY_REACTION_PROCESSING = DEFAULT_REACTION_PROCESSING


class UnsupportedReactionError(SmartsError, ChemistryRejection):
    """A template application the model cannot express.

    It is a ``SmartsError`` because a caller parsing one template against one molecule
    wants it with the rest of the SMARTS failures, and a ``ChemistryRejection`` because a
    batch caller wants to count it by reason. The default reason covers a template vesma
    does not support at all; the chemistry raise sites name a more specific one.
    """


@dataclass(frozen=True)
class ReactionEdit:
    deleted_lhs_indices: AtomIndexVector

    mapped_lhs_indices: AtomIndexVector
    mapped_rhs_indices: AtomIndexVector
    mapped_rhs_z: np.ndarray
    mapped_rhs_m: np.ndarray
    mapped_rhs_c: np.ndarray
    mapped_rhs_c_is_explicit: np.ndarray
    mapped_rhs_h_total: np.ndarray
    mapped_rhs_map: AtomMapVector

    added_rhs_indices: AtomIndexVector
    added_rhs_z: np.ndarray
    added_rhs_m: np.ndarray
    added_rhs_c: np.ndarray
    added_rhs_h_total: np.ndarray
    added_rhs_map: AtomMapVector

    removed_lhs_edges: AtomIndexPairs
    added_rhs_edges: AtomIndexPairs
    added_rhs_bond_types: np.ndarray


@dataclass(frozen=True)
class ReactionPattern:
    lhs: GMol
    rhs: GMol
    edit: ReactionEdit


@dataclass(frozen=True)
class RetroOutcome:
    mapping: ReactionMapping
    precursors: Sequence[CMol]


# The RHS properties vesma reads but does not obey.
#
# An RHS is a rewrite program, not an assertion set. Its operative vocabulary is narrow:
# which atoms exist, their nucleus (``Z``, ``M``), formal charge, total hydrogen ``H``,
# and bond kind. Everything below is *derived* from the result of applying that program,
# so vesma recomputes it and reports where the template's spelling disagreed, rather than
# refusing the application.
#
# The reason is provenance, not chemistry. These fields arrive from whichever extractor
# emitted the templates; enforcing them makes vesma's output a function of that
# extractor's notation conventions -- conventions we neither control nor can currently
# regenerate. A Kekule RHS for a ring that aromatizes is then read as a claim about
# aromaticity, when it is only a claim about how someone once wrote the ring.
#
# Aromaticity in particular is a consequence of the resulting electronic graph, never a
# writable atom bit: ``[c:1]>>[C:1]`` on benzene rewrites no bond and changes no state,
# so benzene stays aromatic. Note that RHS atom case is still *read* during construction,
# where it resolves an omitted bond to single or delocalized -- see
# ``_resolved_rhs_bond_types``. Not obeying a property is not the same as discarding it.
RHS_ANNOTATIONS: tuple[str, ...] = ("A", "D", "X", "v", "R", "r", "x")

# One counter per annotation, laid out like ``RejectionCounts`` so merging is elementwise.
AnnotationCounts = tuple[int, ...]


def no_annotation_mismatches() -> AnnotationCounts:
    return (0,) * len(RHS_ANNOTATIONS)


def merged_annotations(
    left: AnnotationCounts, right: AnnotationCounts
) -> AnnotationCounts:
    return tuple(
        a + b
        for a, b in zip(
            left or no_annotation_mismatches(),
            right or no_annotation_mismatches(),
            strict=True,
        )
    )


@dataclass(frozen=True)
class PrecursorOverflowDiagnostics:
    precursor_atom_cap: int | None = None
    edited_atom_cap: int | None = None
    atom_overflow_components: int = 0
    rejections: RejectionCounts = ()
    atom_overflow_assignments: int = 0
    edited_atom_overflow_assignments: int = 0
    # Neutral telemetry: successful operations, not failures. Completion is the one
    # hydrogen decision that is neither copied verbatim nor written by the template, so
    # how often the model has to make it is worth knowing on its own.
    hydrogen_completion_atoms: int = 0
    hydrogen_completion_applications: int = 0
    # Also neutral: the application was built and returned. This says the template's
    # derived-property spelling did not survive its own edit, which is a fact about the
    # template rather than about the chemistry. See ``RHS_ANNOTATIONS``.
    annotation_mismatches: AnnotationCounts = ()
    annotation_mismatch_applications: int = 0
    # Components the edit disconnected that carry no RHS atom, so the template never
    # described them and they are not precursors. Also a fact about the template: it
    # deleted an atom that turned out to sit between two things it kept.
    undescribed_components: int = 0
    undescribed_component_applications: int = 0
    # The subset of the above that the *edit* severed, rather than components that were
    # already their own molecule. Only these refuse the application: a component the
    # template never touched is an unreacted reagent in the pot, which is what a
    # unimolecular forward rule run against a `cmol_union` of precursors always leaves
    # behind, and refusing that would refuse every selectivity probe.
    orphaned_components: int = 0
    orphaned_component_applications: int = 0
    # Applications whose product disagreed with the RHS's zero-level grouping: a rule written
    # `>>(B).(C)` that opened a ring instead of cutting a bond in two and handed back one
    # molecule, or a `>>(B.C)` that fell apart. Counted here rather than as a `RejectionReason`
    # because this is the same kind of statement as the orphan above -- a component
    # postcondition, refusing by the same `refuse` flag -- and because the rejection counters
    # are mirrored into both device kernels, where an appended member would silently shift
    # every counter recorded before it.
    grouping_violations: int = 0
    grouping_violation_applications: int = 0
    # What was dropped, as a formula per component, so the count can be audited rather
    # than taken on trust. Filled in here for one application; the matcher stamps the
    # (molecule, rule, application) it belongs to into ``undescribed_rows``, because
    # only the matcher knows those.
    undescribed_formulas: tuple[str, ...] = ()
    undescribed_rows: tuple[tuple[int, int, int, str], ...] = ()
    # The severed subset of the above, which is what refuses an application. Kept apart because
    # the two answer different questions and a report that confuses them sends a reader after the
    # wrong molecule: a spectator reagent sitting in a probe's pot is undescribed every time, and
    # naming it as the cause of a refusal it did not cause is worse than saying nothing.
    orphaned_formulas: tuple[str, ...] = ()
    orphaned_rows: tuple[tuple[int, int, int, str], ...] = ()
    # What the matcher enumerated, so that "every application is accounted for" is a
    # statement an audit can check rather than one the code makes true by construction.
    # ``applications_attempted`` must equal accepted + refused + unkeyable downstream.
    # ``pairs_declined`` counts pairs the matcher answered with no assignments at all,
    # and ``pairs_search_incomplete`` the subset where it declined to answer rather than
    # answering "no" -- an exhausted search budget, which would otherwise vanish.
    applications_attempted: int = 0
    pairs_declined: int = 0
    pairs_search_incomplete: int = 0

    @property
    def rejected_applications(self) -> int:
        """Applications the chemistry refused, across every reason."""
        return sum(self.rejections)

    @property
    def rejections_by_reason(self) -> dict[str, int]:
        """The non-zero rejection counts, keyed by reason name."""
        return by_name(self.rejections or no_rejections())

    @property
    def annotation_mismatches_by_property(self) -> dict[str, int]:
        """The non-zero annotation mismatches, keyed by the property that disagreed."""
        return {
            name: count
            for name, count in zip(
                RHS_ANNOTATIONS,
                self.annotation_mismatches or no_annotation_mismatches(),
                strict=True,
            )
            if count
        }

    @property
    def has_overflow(self) -> bool:
        return (
            self.atom_overflow_components > 0
            or self.atom_overflow_assignments > 0
            or self.edited_atom_overflow_assignments > 0
        )


@dataclass(frozen=True)
class PrecursorMappingResult:
    precursors: Sequence[CMol]
    diagnostics: PrecursorOverflowDiagnostics


@dataclass(frozen=True)
class ReactionTemplate:
    id: str
    smarts: str
    pattern: ReactionPattern
    metadata: dict[str, str | int | float]


# 6: compiled patterns carry `GMol.atom_group`, so a pickle written by 5 unpickles into the old
# shape and would read as "no grouping anywhere" rather than failing.
# 7: compiled patterns carry `GMol.atom_predicate` and `GMol.predicates` (recursive SMARTS).
# 8: predicates hold box-form expressions, and alternatives tying properties compile to them.
REACTION_CACHE_VERSION = 8


def reaction_pickle_cache_path(smarts: Sequence[str]) -> Path:
    """Cache the full parsed rule tuple using pickle's C unpickler.

    Joblib's NumPy-aware loader visits millions of tiny arrays in a large rule set;
    pickle protocol 5 loads that same object graph substantially faster. The model
    and parser schema versions remain part of the cache identity.
    """
    digest = hashlib.sha256()
    digest.update(f"{REACTION_CACHE_VERSION}:{platform.python_version()}".encode())
    for row in smarts:
        encoded = row.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return cache_dir() / "reaction-pickle" / f"{digest.hexdigest()}.pkl"


def parse_reaction_smarts(smarts: str) -> ReactionPattern:
    text = smarts.strip()
    parts = text.split(">")
    if len(parts) != 3:
        raise UnsupportedReactionError(
            "Reaction SMARTS must use 'lhs>agents>rhs' or 'lhs>>rhs' form."
        )

    lhs_text, _, rhs_text = parts
    lhs_text = lhs_text.strip()
    rhs_text = rhs_text.strip()
    if not lhs_text or not rhs_text:
        raise UnsupportedReactionError(
            "Reaction SMARTS requires both LHS and RHS queries."
        )

    lhs, _ = gmol_from_smarts(lhs_text)
    rhs, _ = gmol_from_smarts(rhs_text)
    if rhs.predicates:
        raise UnsupportedReactionError(
            "Recursive SMARTS $(...), and properties tied together like [C,n], constrain "
            "matching only; the right-hand side is built, not matched, so it cannot carry them."
        )

    # Validate RHS semantics (added atoms must be concrete, mapped atoms can inherit)
    lhs_maps = lhs.atom_map[lhs.atom_map > 0]
    rhs_added_mask = (rhs.atom_map <= 0) | ~np.isin(rhs.atom_map, lhs_maps)

    try:
        rhs_z = rhs.atom_masks.exact_z()
        rhs.atom_masks.exact_c()
    except ValueError as exc:
        raise UnsupportedReactionError(str(exc)) from exc

    if np.any(rhs_z[rhs_added_mask] == -1):
        raise UnsupportedReactionError("RHS-added atoms require a concrete element.")

    rhs_edges = np.argwhere(np.triu(bond_presence(rhs.bonds), k=1)).astype(np.int32)
    if len(rhs_edges) > 0:
        types = _resolved_rhs_bond_types(rhs, rhs_edges)
        types_single = (types > 0) & ((types & (types - 1)) == 0)
        if not np.all(types_single):
            raise UnsupportedReactionError(
                "RHS bond edits require a concrete bond type."
            )

    return ReactionPattern(lhs=lhs, rhs=rhs, edit=reaction_edit_from_graphs(lhs, rhs))


def parse_reaction_smarts_items(
    smarts: tuple[str, ...],
    cache_version: int,
    python_version: str,
    n_jobs: int | None,
) -> tuple[ReactionPattern, ...]:
    del python_version
    if cache_version != REACTION_CACHE_VERSION:
        raise ValueError("Unsupported reaction cache version.")
    return parallel_map(
        parse_reaction_smarts,
        smarts,
        n_jobs=n_jobs,
    )


def reaction_patterns_from_smarts(
    smarts: Sequence[str],
    *,
    n_jobs: int | None = None,
    cache: bool | None = None,
) -> Sequence[ReactionPattern]:
    rows = tuple(smarts)
    if not resolve_cache(cache):
        return parse_reaction_smarts_items(
            rows,
            REACTION_CACHE_VERSION,
            platform.python_version(),
            n_jobs,
        )

    pickle_path = reaction_pickle_cache_path(rows)
    if pickle_path.exists():
        with pickle_path.open("rb") as file:
            return pickle.load(file)

    cached = cached_function(parse_reaction_smarts_items, ignore=("n_jobs",))
    args = (rows, REACTION_CACHE_VERSION, platform.python_version(), n_jobs)
    # Read an existing Joblib entry once when upgrading an established cache. New
    # entries only use pickle, avoiding two large copies of each rule set on disk.
    if cached.check_call_in_cache(*args):
        patterns = cached(*args)
    else:
        patterns = parse_reaction_smarts_items(*args)

    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=pickle_path.parent, delete=False) as file:
        temporary_path = Path(file.name)
        try:
            pickle.dump(patterns, file, protocol=5)
        except BaseException:
            temporary_path.unlink()
            raise
    os.replace(temporary_path, pickle_path)
    return patterns


def _resolved_rhs_bond_types(rhs: GMol, rhs_edges: AtomIndexPairs) -> np.ndarray:
    """Concrete bond type per RHS edge, resolving omitted/default bonds.

    RHS pattern bonds are written into concrete precursor molecules, so an omitted
    (``QBOND_DEFAULT``) bond must resolve to a single or aromatic order via the RHS
    atoms' aromatic intent before being treated as a concrete bond type.
    """
    a_intent = rhs.atom_masks.exact_a()
    resolved = np.empty(len(rhs_edges), dtype=np.uint8)
    for k in range(len(rhs_edges)):
        i = int(rhs_edges[k, 0])
        j = int(rhs_edges[k, 1])
        flags = resolve_default_bond_flags(
            int(rhs.bonds[i, j]),
            left_aromatic=int(a_intent[i]) == AROMATIC,
            right_aromatic=int(a_intent[j]) == AROMATIC,
        )
        resolved[k] = np.uint8(bond_kind_flags(flags))
    return resolved


def reaction_edit_from_graphs(lhs: GMol, rhs: GMol) -> ReactionEdit:
    lhs_map_to_idx = map_to_index(lhs.atom_map)
    rhs_map_to_idx = map_to_index(rhs.atom_map)

    common_maps = np.array(
        sorted(lhs_map_to_idx.keys() & rhs_map_to_idx.keys()), dtype=np.int32
    )
    common_lhs_indices = np.array(
        [lhs_map_to_idx[m] for m in common_maps], dtype=np.int32
    )
    common_rhs_indices = np.array(
        [rhs_map_to_idx[m] for m in common_maps], dtype=np.int32
    )

    rhs_map_ids = np.array(list(rhs_map_to_idx), dtype=np.int32)
    deleted_lhs_indices = np.flatnonzero(
        (lhs.atom_map <= 0) | ~np.isin(lhs.atom_map, rhs_map_ids)
    ).astype(np.int32)

    added_maps = np.array(
        sorted(rhs_map_to_idx.keys() - lhs_map_to_idx.keys()), dtype=np.int32
    )
    added_rhs_indices = np.array(
        [rhs_map_to_idx[m] for m in added_maps], dtype=np.int32
    )

    unmapped_rhs_indices = np.flatnonzero(rhs.atom_map <= 0).astype(np.int32)
    all_added_rhs_indices = np.sort(
        np.concatenate([added_rhs_indices, unmapped_rhs_indices])
    ).astype(np.int32)

    rhs_z = rhs.atom_masks.exact_z()
    rhs_m = rhs.atom_masks.exact_m()
    rhs_c = rhs.atom_masks.exact_c()
    rhs_h_total = rhs.atom_masks.exact_h_total()

    mapped_z = rhs_z[common_rhs_indices]
    mapped_m = rhs_m[common_rhs_indices]
    mapped_c = rhs_c[common_rhs_indices]
    mapped_h_total = rhs_h_total[common_rhs_indices]
    mapped_c_masks = rhs.atom_masks.C[common_rhs_indices]
    mapped_c_is_explicit = (mapped_c_masks > 0) & (
        (mapped_c_masks & (mapped_c_masks - 1)) == 0
    )
    added_z = rhs_z[all_added_rhs_indices]
    added_m = rhs_m[all_added_rhs_indices]
    added_c = rhs_c[all_added_rhs_indices]
    added_h_total = rhs_h_total[all_added_rhs_indices]

    lhs_edges = np.argwhere(np.triu(bond_presence(lhs.bonds), k=1)).astype(np.int32)
    rhs_edges = np.argwhere(np.triu(bond_presence(rhs.bonds), k=1)).astype(np.int32)

    rhs_bond_types = np.array([], dtype=np.uint8)
    if len(rhs_edges) > 0:
        rhs_bond_types = _resolved_rhs_bond_types(rhs, rhs_edges)

    return ReactionEdit(
        deleted_lhs_indices=deleted_lhs_indices,
        mapped_lhs_indices=common_lhs_indices,
        mapped_rhs_indices=common_rhs_indices,
        mapped_rhs_z=mapped_z,
        mapped_rhs_m=mapped_m,
        mapped_rhs_c=mapped_c,
        mapped_rhs_c_is_explicit=mapped_c_is_explicit,
        mapped_rhs_h_total=mapped_h_total,
        mapped_rhs_map=rhs.atom_map[common_rhs_indices].copy(),
        added_rhs_indices=all_added_rhs_indices,
        added_rhs_z=added_z,
        added_rhs_m=added_m,
        added_rhs_c=added_c,
        added_rhs_h_total=added_h_total,
        added_rhs_map=rhs.atom_map[all_added_rhs_indices].copy(),
        removed_lhs_edges=lhs_edges,
        added_rhs_edges=rhs_edges,
        added_rhs_bond_types=rhs_bond_types,
    )


def map_to_index(atom_map: AtomMapVector) -> dict[int, int]:
    indices = np.flatnonzero(atom_map > 0).astype(np.int32)
    map_ids = atom_map[indices]
    unique_map_ids, counts = np.unique(map_ids, return_counts=True)
    duplicate_map_ids = unique_map_ids[counts > 1]
    if duplicate_map_ids.size:
        raise UnsupportedReactionError(
            f"Duplicate atom map number: {duplicate_map_ids[0]}"
        )
    return dict(zip(map_ids.tolist(), indices.tolist(), strict=True))


def precursors_from_mapping(
    *,
    product: CMol,
    reaction: ReactionPattern,
    mapping: ReactionMapping,
    ring_descriptors: RingDescriptors = RingDescriptors.ALL,
    max_precursor_atoms: int | None = None,
    max_edited_atoms: int | None = None,
) -> Sequence[CMol]:
    result = precursors_from_mapping_result(
        product=product,
        reaction=reaction,
        mapping=mapping,
        ring_descriptors=ring_descriptors,
        max_precursor_atoms=max_precursor_atoms,
        max_edited_atoms=max_edited_atoms,
    )
    warn_precursor_overflow(result.diagnostics, backend="CPU retro")
    return result.precursors


def precursors_from_mapping_result(
    *,
    product: CMol,
    reaction: ReactionPattern,
    mapping: ReactionMapping,
    ring_descriptors: RingDescriptors = RingDescriptors.ALL,
    max_precursor_atoms: int | None = None,
    max_edited_atoms: int | None = None,
) -> PrecursorMappingResult:
    validate_atom_cap(max_precursor_atoms, name="max_precursor_atoms")
    validate_atom_cap(max_edited_atoms, name="max_edited_atoms")
    edit = reaction.edit

    keep = np.ones(product.atom_count, dtype=bool)
    if edit.deleted_lhs_indices.size:
        keep[mapping[edit.deleted_lhs_indices]] = False

    survivor_idx = np.flatnonzero(keep).astype(np.int32)
    survivor_count = survivor_idx.size
    added_count = edit.added_rhs_indices.size
    edited_count = survivor_count + added_count
    if max_edited_atoms is not None and edited_count > max_edited_atoms:
        return PrecursorMappingResult(
            precursors=(),
            diagnostics=PrecursorOverflowDiagnostics(
                precursor_atom_cap=max_precursor_atoms,
                edited_atom_cap=max_edited_atoms,
                edited_atom_overflow_assignments=1,
            ),
        )

    prod_to_edited = np.full(product.atom_count, -1, dtype=np.int32)
    prod_to_edited[survivor_idx] = np.arange(survivor_count, dtype=np.int32)

    rhs_to_edited = np.full(reaction.rhs.atom_count, -1, dtype=np.int32)
    mapped_product = mapping[edit.mapped_lhs_indices]
    mapped_edited = prod_to_edited[mapped_product]

    if np.any(mapped_edited < 0):
        raise UnsupportedReactionError(
            "Reaction mapping produced a deleted atom for a required RHS map."
        )

    rhs_to_edited[edit.mapped_rhs_indices] = mapped_edited

    added_edited = np.arange(survivor_count, edited_count, dtype=np.int32)
    rhs_to_edited[edit.added_rhs_indices] = added_edited

    Z = np.zeros(edited_count, dtype=np.int16)
    M = np.zeros(edited_count, dtype=np.int16)
    C = np.zeros(edited_count, dtype=np.int8)
    explicit_h = np.zeros(edited_count, dtype=np.int16)
    implicit_h = np.full(edited_count, -1, dtype=np.int16)
    atom_map = np.zeros(edited_count, dtype=np.int32)

    if survivor_count > 0:
        Z[:survivor_count] = product.Z[survivor_idx]
        M[:survivor_count] = product.M[survivor_idx]
        C[:survivor_count] = product.C[survivor_idx]
        explicit_h[:survivor_count] = np.where(
            product.A[survivor_idx] == AROMATIC,
            product.H[survivor_idx] - product.h[survivor_idx],
            0,
        )
        atom_map[:survivor_count] = product.atom_map[survivor_idx]

    # An atom the RHS writes gets its hydrogen count from the edit, not from the state it had in
    # the product. Carrying the product's explicit count into one is what let an aromatic `[nH]`
    # -- stored with its hydrogen explicit on purpose, so it writes back as `[nH]` and not `n` --
    # hand `explicit = 1` to an atom the RHS turns into a nitrile nitrogen, whose edited total is
    # 0. `cmol_from_seed_data` then computed `total_h - explicit` as `0 - 1` and stored an
    # impossible split: total right, explicit/implicit not. `Hartenfeller 15`, `16a` and `16b` on
    # an N-H triazole did exactly that, and the result reached a WL keyer as acetonitrile with
    # `h = -1`.
    #
    # Zeroing is safe rather than lossy because the split is re-derived downstream from the
    # *edited* total: an aromatic heteroatom takes `h = 0, H = total`, so `H - h` returns the
    # explicit count and `[nH]` still round-trips, while the nitrile nitrogen resolves to 0 --
    # which `hydrogens_for_written_state` knew all along. Only atoms the RHS leaves alone keep
    # what the product said about them.
    explicit_h[mapped_edited] = 0

    Z[mapped_edited] = np.where(
        edit.mapped_rhs_z != -1, edit.mapped_rhs_z, Z[mapped_edited]
    )
    C[mapped_edited] = np.where(
        edit.mapped_rhs_c_is_explicit,
        edit.mapped_rhs_c,
        C[mapped_edited],
    )
    mapped_explicit_h, mapped_implicit_h = rhs_hydrogen_seed(
        h_total=edit.mapped_rhs_h_total,
        default_explicit_h=explicit_h[mapped_edited],
        default_implicit_h=implicit_h[mapped_edited],
    )
    explicit_h[mapped_edited] = mapped_explicit_h
    implicit_h[mapped_edited] = mapped_implicit_h
    atom_map[mapped_edited] = edit.mapped_rhs_map

    # Nuclear conservation: a mapped RHS atom may assert its nucleus but never
    # transmute it. Z and M are carried from the matched atom, and an assertion that
    # disagrees rejects rather than silently rewriting the nucleus.
    if np.any(
        (edit.mapped_rhs_z != -1) & (edit.mapped_rhs_z != product.Z[mapped_product])
    ):
        raise UnsupportedReactionError(
            "Mapped RHS atom changed element identity.",
            reason=RejectionReason.NUCLEAR_CONSERVATION_VIOLATED,
        )
    if np.any(
        (edit.mapped_rhs_m != MASS_ANY)
        & (edit.mapped_rhs_m != product.M[mapped_product])
    ):
        raise UnsupportedReactionError(
            "Mapped RHS atom changed isotope identity.",
            reason=RejectionReason.NUCLEAR_CONSERVATION_VIOLATED,
        )

    Z[added_edited] = edit.added_rhs_z
    M[added_edited] = np.where(
        edit.added_rhs_m == MASS_ANY, MASS_UNLABELED, edit.added_rhs_m
    )
    C[added_edited] = edit.added_rhs_c
    added_explicit_h, added_implicit_h = rhs_hydrogen_seed(
        h_total=edit.added_rhs_h_total,
        default_explicit_h=np.zeros_like(edit.added_rhs_h_total),
        default_implicit_h=np.full_like(edit.added_rhs_h_total, -1),
    )
    explicit_h[added_edited] = added_explicit_h
    implicit_h[added_edited] = added_implicit_h
    atom_map[added_edited] = edit.added_rhs_map

    bonds = np.zeros((edited_count, edited_count), dtype=np.uint8)
    if survivor_count > 0:
        bonds[:survivor_count, :survivor_count] = product.bonds[
            np.ix_(survivor_idx, survivor_idx)
        ]

    if edit.removed_lhs_edges.size:
        removed_prod = mapping[edit.removed_lhs_edges]
        removed_edited = prod_to_edited[removed_prod]
        valid = (removed_edited >= 0).all(axis=1)
        valid_edges = removed_edited[valid]
        bonds[valid_edges[:, 0], valid_edges[:, 1]] = BondKind.NONE
        bonds[valid_edges[:, 1], valid_edges[:, 0]] = BondKind.NONE

    if edit.added_rhs_edges.size:
        added_edges = rhs_to_edited[edit.added_rhs_edges]
        if np.any(added_edges < 0):
            raise UnsupportedReactionError(
                "Reaction edit did not assign every RHS bond endpoint."
            )
        types = edit.added_rhs_bond_types
        bonds[added_edges[:, 0], added_edges[:, 1]] = types
        bonds[added_edges[:, 1], added_edges[:, 0]] = types

    total_h, completed_atoms = edited_total_hydrogens(
        product=product,
        edit=edit,
        bonds=bonds,
        survivor_idx=survivor_idx,
        mapped_edited=mapped_edited,
        mapped_product=mapped_product,
        added_edited=added_edited,
        C=C,
        edited_count=edited_count,
    )

    mismatches = edited_annotation_mismatches(
        reaction=reaction,
        rhs_to_edited=rhs_to_edited,
        bonds=bonds,
        total_h=total_h,
    )

    components = connected_components_from_bonds(bonds)
    precursors: list[CMol] = []
    undescribed_components = 0
    orphaned_components = 0
    undescribed_formulas: list[str] = []
    orphaned_formulas: list[str] = []
    # Where each edited atom ended up, so a postcondition that needs perception can be
    # read off the precursor that actually carries it.
    precursor_of_edited = np.full(edited_count, -1, dtype=np.int32)
    edited_to_precursor = np.full(edited_count, -1, dtype=np.int32)

    # The RHS enumerates the precursors, so a component carrying no RHS atom is not one.
    described = np.zeros(edited_count, dtype=bool)
    described[rhs_to_edited[rhs_to_edited >= 0]] = True

    # Whether the *edit* disconnected a component, or it was already its own molecule.
    #
    # The distinction is the difference between an under-specified template and an
    # unreacted reagent, and the RHS test alone cannot see it. A unimolecular forward rule
    # run against a two-component pot -- which is exactly how the selectivity probes work,
    # `cmol_union` of an application's precursors -- leaves the other component untouched
    # and unnamed by the RHS. That is a *spectator*, not an orphan: TMSCN sitting beside
    # the glycosyl acetate it will go on to attack. Refusing those would refuse the whole
    # forward probe.
    #
    # An orphan, by contrast, is a piece the edit severed: in the product its atoms were
    # bonded to something outside the component, and a deleted atom is what cut them
    # loose. So the question is asked of the *product's* bonds, not the edited ones.
    product_bonds = bond_presence(product.bonds)

    def severed(component: AtomIndices) -> bool:
        """Was this component attached to the rest of the molecule before the edit?"""
        if component.size >= survivor_count:
            return False
        inside = survivor_idx[component]
        outside = np.ones(product.atom_count, dtype=bool)
        outside[inside] = False
        return bool(product_bonds[np.ix_(inside, np.flatnonzero(outside))].any())

    named: list[AtomIndices] = []
    for component in components:
        if component.size == 0:
            continue
        if not described[component].any():
            # A piece of the molecule that ended up in no precursor the template named.
            # Across uspto42k only 41 templates delete a mapped atom at all, and only one
            # of those orphans anything; the rest delete leaving atoms pinned to one
            # neighbour, where nothing can be orphaned. Where the deleted atom *bridged*
            # two things the template kept, though, the orphan is not debris: the
            # measured case sheds `C17`, seventeen carbons of a molecule the route still
            # needs and did not get.
            #
            # So this refuses the application, by the rule stated below for the atom cap
            # and for exactly its reason -- a partial set presents an impossible
            # disconnection as a complete one. The earlier reading, that the orphan is
            # "the leaving group with an atom amputated and the wound capped with
            # hydrogen -- an ethyl ester yields methane, not ethanol", is true only for
            # the pinned case, and the two are not distinguishable here without deciding
            # how large a fragment stops being debris. Refusing both is the honest
            # choice, and a template that orphans is under-specified either way:
            # `vertsy.template_lint` finds the same fault in the template text.
            undescribed_components += 1
            formula = heavy_atom_formula(Z[component])
            undescribed_formulas.append(formula)
            if severed(component):
                orphaned_components += 1
                orphaned_formulas.append(formula)
            continue
        named.append(component)

    # A reaction is defined by all of its named precursors, so one that does not fit
    # refuses the whole application rather than handing back the pieces that did. A
    # partial set presents an impossible disconnection as a complete one, which is worst
    # exactly where it is least visible: in a multi-layer search, where the missing piece
    # would have been the molecule that failed to reach stock. This is the same rule the
    # edited-atom cap above applies, and CUDA and Metal apply it too.
    atom_overflow_components = (
        sum(1 for component in named if component.size > max_precursor_atoms)
        if max_precursor_atoms is not None
        else 0
    )

    # What the RHS's parentheses claimed about its own output, checked against the output. A
    # grouping cannot be *required* of a product the way it is required of a match -- whether
    # two pieces come apart depends on the bonds, not on the query -- so it is a postcondition,
    # and a rule that says `>>(B).(C)` and produces one molecule is refused rather than silently
    # handing back a single precursor. That is the macrocycle class: a bimolecular disconnection
    # that opened a ring instead of cutting a bond in two.
    grouping_violations = 0
    if np.any(reaction.rhs.atom_group):
        labels = np.zeros(edited_count, dtype=np.int64)
        for label, component in enumerate(components):
            labels[component] = label
        described_rhs = rhs_to_edited >= 0
        edited_group = np.zeros(edited_count, dtype=np.int32)
        edited_group[rhs_to_edited[described_rhs]] = reaction.rhs.atom_group[
            described_rhs
        ]
        grouping_violations = int(
            not grouping_agrees_with_components(edited_group, labels)
        )

    refuse = (
        atom_overflow_components > 0
        or orphaned_components > 0
        or grouping_violations > 0
    )
    for component in named if not refuse else ():
        idx = component.astype(np.int32)
        sub_bonds = bonds[np.ix_(idx, idx)]

        precursor = cmol_from_seed_data(
            Z=Z[idx],
            M=M[idx],
            C=C[idx],
            explicit_h=explicit_h[idx],
            implicit_h=implicit_h[idx],
            atom_map=atom_map[idx],
            bonds=sub_bonds,
            ring_descriptors=ring_descriptors,
            pinned_total_h=total_h[idx],
        )
        precursors.append(precursor)
        edited_to_precursor[idx] = np.arange(idx.size, dtype=np.int32)
        precursor_of_edited[idx] = len(precursors) - 1

    mismatches = merged_annotations(
        mismatches,
        perceived_annotation_mismatches(
            reaction=reaction,
            rhs_to_edited=rhs_to_edited,
            precursors=precursors,
            precursor_of_edited=precursor_of_edited,
            edited_to_precursor=edited_to_precursor,
        ),
    )

    return PrecursorMappingResult(
        precursors=tuple(precursors),
        diagnostics=PrecursorOverflowDiagnostics(
            precursor_atom_cap=max_precursor_atoms,
            edited_atom_cap=max_edited_atoms,
            atom_overflow_components=atom_overflow_components,
            atom_overflow_assignments=int(atom_overflow_components > 0),
            hydrogen_completion_atoms=completed_atoms,
            hydrogen_completion_applications=int(completed_atoms > 0),
            annotation_mismatches=mismatches,
            annotation_mismatch_applications=int(any(mismatches)),
            undescribed_components=undescribed_components,
            undescribed_component_applications=int(undescribed_components > 0),
            orphaned_components=orphaned_components,
            orphaned_component_applications=int(orphaned_components > 0),
            grouping_violations=grouping_violations,
            grouping_violation_applications=grouping_violations,
            undescribed_formulas=tuple(undescribed_formulas),
            orphaned_formulas=tuple(orphaned_formulas),
        ),
    )


def stamped_undescribed(
    diagnostics: PrecursorOverflowDiagnostics,
    mol_idx: int,
    rule_id: int,
    app_idx: int,
) -> PrecursorOverflowDiagnostics:
    """Attach the application a dropped component came from.

    ``precursors_from_mapping_result`` knows *what* it dropped and the matcher knows
    *where* from, so the row is only complete once both have spoken. Stamping here rather
    than threading a separate list keeps the serial and parallel paths identical: both
    already merge diagnostics.
    """
    if not diagnostics.undescribed_formulas and not diagnostics.orphaned_formulas:
        return diagnostics
    return replace(
        diagnostics,
        undescribed_rows=tuple(
            (mol_idx, rule_id, app_idx, formula)
            for formula in diagnostics.undescribed_formulas
        ),
        undescribed_formulas=(),
        orphaned_rows=tuple(
            (mol_idx, rule_id, app_idx, formula)
            for formula in diagnostics.orphaned_formulas
        ),
        orphaned_formulas=(),
    )


def heavy_atom_formula(Z: AtomVector) -> str:
    """Hill-order element counts for a set of atoms, hydrogen excluded.

    Enough to recognise what a dropped component was -- ``C`` for the methane an ethyl
    ester sheds, ``C8O3`` for a larger fragment -- without perceiving a molecule vesma
    has already decided is not a precursor.
    """
    counts = Counter(int(z) for z in Z.tolist())
    ordered = sorted(
        counts,
        key=lambda z: (Z_TO_ELEMENT.get(z, "") not in ("C",), Z_TO_ELEMENT.get(z, "")),
    )
    return "".join(
        f"{Z_TO_ELEMENT.get(z, f'Z{z}')}{counts[z] if counts[z] > 1 else ''}"
        for z in ordered
    )


def edited_annotation_mismatches(
    *,
    reaction: ReactionPattern,
    rhs_to_edited: AtomIndexVector,
    bonds: BondMatrix,
    total_h: AtomVector,
) -> AnnotationCounts:
    """Count where the edited graph disagrees with a derived property the RHS wrote.

    ``D`` and ``X`` can be read from the edited graph: splitting into components removes
    no bonds, so degree and total connectivity are the same before and after it. The
    properties that need perception are counted in ``perceived_annotation_mismatches``
    instead, once the precursors exist.
    """
    masks = reaction.rhs.atom_masks
    degrees = bond_presence(bonds).sum(axis=1)
    counts = [0] * len(RHS_ANNOTATIONS)
    for rhs_atom, edited in enumerate(rhs_to_edited.tolist()):
        if edited < 0:
            continue
        degree = int(degrees[edited])
        for name, observed in (
            ("D", degree),
            ("X", degree + int(total_h[edited])),
        ):
            mask = int(getattr(masks, name)[rhs_atom])
            if mask == BUCKET_DEFAULT_MASKS[name]:
                continue
            if not mask & int(bucket_mask_exact(observed)):
                counts[RHS_ANNOTATIONS.index(name)] += 1
    return tuple(counts)


def perceived_annotation_mismatches(
    *,
    reaction: ReactionPattern,
    rhs_to_edited: AtomIndexVector,
    precursors: Sequence[CMol],
    precursor_of_edited: AtomIndexVector,
    edited_to_precursor: AtomIndexVector,
) -> AnnotationCounts:
    """Count the same disagreements for the properties that need perception first.

    Aromaticity is the one that occurs: an atom's case is written by every template, so
    ``A`` is nominally constrained on every RHS atom in the corpus whether or not its
    author meant it as a constraint. It is compared against the perceived precursor
    rather than the edited graph because aromaticity does not exist until perception has
    run, and it is a count rather than a refusal because the RHS does not get to declare
    it -- see ``RHS_ANNOTATIONS``.
    """
    masks = reaction.rhs.atom_masks
    counts = [0] * len(RHS_ANNOTATIONS)
    for rhs_atom, edited in enumerate(rhs_to_edited.tolist()):
        if edited < 0:
            continue
        precursor_index = int(precursor_of_edited[edited])
        if precursor_index < 0:
            # The atom's component overflowed the cap and was dropped, so there is no
            # perceived state to compare against. Capacity is reported on its own.
            continue
        precursor = precursors[precursor_index]
        atom = int(edited_to_precursor[edited])
        for name, observed in (
            ("A", int(precursor.A[atom])),
            ("v", int(precursor.v[atom])),
            ("R", int(precursor.R[atom])),
            ("r", int(precursor.r[atom])),
            ("x", int(precursor.x[atom])),
        ):
            mask = int(getattr(masks, name)[rhs_atom])
            if mask == BUCKET_DEFAULT_MASKS[name]:
                continue
            if observed < 0:
                # Ring descriptors are absent when the caller skipped ring perception.
                continue
            agrees = (
                mask & observed if name == "A" else mask & bucket_mask_exact(observed)
            )
            if not agrees:
                counts[RHS_ANNOTATIONS.index(name)] += 1
    return tuple(counts)


def bond_kind_histogram(bonds: BondMatrix, atom: int) -> tuple[int, ...]:
    """Counts of each exact bond kind incident to one atom.

    The hydrogen law compares this before and after the edit. It is a histogram rather
    than a degree because replacing a single bond with a double one changes an atom's
    local environment without changing how many neighbours it has.
    """
    row = bonds[atom]
    return tuple(int(np.count_nonzero(row == int(kind))) for kind in EXACT_BOND_KINDS)


def edited_total_hydrogens(
    *,
    product: CMol,
    edit: ReactionEdit,
    bonds: BondMatrix,
    survivor_idx: AtomIndexVector,
    mapped_edited: AtomIndexVector,
    mapped_product: AtomIndexVector,
    added_edited: AtomIndexVector,
    C: Charge,
    edited_count: int,
) -> tuple[AtomVector, int]:
    """Total hydrogen for every atom of the edited graph, and how many were completed.

    Three cases decide a surviving atom: an explicit RHS count sets it, an unchanged
    charge and incident bond histogram copy it, and a changed environment completes it
    from the configuration the matched atom was already in. Every atom is decided here,
    so perception never infers a precursor's hydrogen.
    """
    total_h = np.full(edited_count, HYDROGEN_UNPINNED, dtype=np.int16)
    completed_atoms = 0
    orders = bond_order_matrix(bonds)
    presence = bond_presence(bonds)
    delocalized_after = bond_delocalization(bonds).any(axis=1)

    product_orders = bond_order_matrix(product.bonds)
    product_presence = bond_presence(product.bonds)
    product_delocalized = bond_delocalization(product.bonds).any(axis=1)

    pinned_by_rhs = np.zeros(edited_count, dtype=bool)
    if mapped_edited.size:
        pinned_by_rhs[mapped_edited] = edit.mapped_rhs_h_total >= 0
        total_h[mapped_edited] = np.where(
            edit.mapped_rhs_h_total >= 0, edit.mapped_rhs_h_total, HYDROGEN_UNPINNED
        )
    if added_edited.size:
        # An added atom has no matched atom to copy from, so a written H is taken exactly
        # and an omitted one is decided by the same law molecule literals use: the fewest
        # hydrogens its declared configurations allow for the bonding the RHS wrote.
        for position, atom in enumerate(added_edited.tolist()):
            written = int(edit.added_rhs_h_total[position])
            if written >= 0:
                total_h[atom] = np.int16(written)
                continue
            resolved = hydrogens_for_written_state(
                z=int(edit.added_rhs_z[position]),
                charge=int(C[atom]),
                sigma_degree=int(presence[atom].sum()),
                bond_order_sum=int(orders[atom].sum()),
                delocalized=bool(delocalized_after[atom]),
            )
            if isinstance(resolved, HydrogenFailure):
                raise UnsupportedReactionError(
                    f"Added RHS atom has no determined hydrogen count: "
                    f"{resolved.name.lower()}.",
                    reason=REASON_BY_HYDROGEN_FAILURE[resolved],
                )
            total_h[atom] = np.int16(resolved)

    charge_before = np.zeros(edited_count, dtype=np.int8)
    charge_before[: survivor_idx.size] = product.C[survivor_idx]

    for position, original in enumerate(survivor_idx.tolist()):
        if pinned_by_rhs[position]:
            continue
        before = bond_kind_histogram(product.bonds, original)
        after = bond_kind_histogram(bonds, position)
        resolved = hydrogens_after_edit(
            z=int(product.Z[original]),
            charge_before=int(charge_before[position]),
            charge_after=int(C[position]),
            sigma_degree_before=int(product_presence[original].sum()),
            bond_order_sum_before=int(product_orders[original].sum()),
            total_h_before=int(product.H[original]),
            delocalized_before=bool(product_delocalized[original]),
            sigma_degree_after=int(presence[position].sum()),
            bond_order_sum_after=int(orders[position].sum()),
            delocalized_after=bool(delocalized_after[position]),
            bonding_changed=before != after,
        )
        if isinstance(resolved, HydrogenFailure):
            raise UnsupportedReactionError(
                f"Hydrogen could not be determined after the edit: "
                f"{resolved.name.lower()}.",
                reason=REASON_BY_HYDROGEN_FAILURE[resolved],
            )
        total_h[position] = np.int16(resolved)
        if before != after:
            completed_atoms += 1

    return total_h, completed_atoms


def rhs_hydrogen_seed(
    *,
    h_total: AtomVector,
    default_explicit_h: AtomVector,
    default_implicit_h: AtomVector,
) -> tuple[AtomVector, AtomVector]:
    explicit_h = default_explicit_h.copy()
    implicit_h = default_implicit_h.copy()
    specified = h_total >= 0
    explicit_h[specified] = 0
    implicit_h[specified] = h_total[specified]
    return explicit_h, implicit_h


def validate_atom_cap(value: int | None, *, name: str) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive when provided.")


def add_precursor_overflow_diagnostics(
    left: PrecursorOverflowDiagnostics,
    right: PrecursorOverflowDiagnostics,
) -> PrecursorOverflowDiagnostics:
    precursor_atom_cap = left.precursor_atom_cap
    if precursor_atom_cap is None:
        precursor_atom_cap = right.precursor_atom_cap
    edited_atom_cap = left.edited_atom_cap
    if edited_atom_cap is None:
        edited_atom_cap = right.edited_atom_cap
    return PrecursorOverflowDiagnostics(
        precursor_atom_cap=precursor_atom_cap,
        edited_atom_cap=edited_atom_cap,
        atom_overflow_components=left.atom_overflow_components
        + right.atom_overflow_components,
        atom_overflow_assignments=left.atom_overflow_assignments
        + right.atom_overflow_assignments,
        edited_atom_overflow_assignments=left.edited_atom_overflow_assignments
        + right.edited_atom_overflow_assignments,
        rejections=merged(
            left.rejections or no_rejections(), right.rejections or no_rejections()
        ),
        hydrogen_completion_atoms=left.hydrogen_completion_atoms
        + right.hydrogen_completion_atoms,
        hydrogen_completion_applications=left.hydrogen_completion_applications
        + right.hydrogen_completion_applications,
        annotation_mismatches=merged_annotations(
            left.annotation_mismatches, right.annotation_mismatches
        ),
        annotation_mismatch_applications=left.annotation_mismatch_applications
        + right.annotation_mismatch_applications,
        undescribed_components=left.undescribed_components
        + right.undescribed_components,
        undescribed_component_applications=left.undescribed_component_applications
        + right.undescribed_component_applications,
        orphaned_components=left.orphaned_components + right.orphaned_components,
        orphaned_component_applications=left.orphaned_component_applications
        + right.orphaned_component_applications,
        grouping_violations=left.grouping_violations + right.grouping_violations,
        grouping_violation_applications=left.grouping_violation_applications
        + right.grouping_violation_applications,
        undescribed_rows=left.undescribed_rows + right.undescribed_rows,
        orphaned_rows=left.orphaned_rows + right.orphaned_rows,
        applications_attempted=left.applications_attempted
        + right.applications_attempted,
        pairs_declined=left.pairs_declined + right.pairs_declined,
        pairs_search_incomplete=left.pairs_search_incomplete
        + right.pairs_search_incomplete,
    )


def warn_precursor_overflow(
    diagnostics: PrecursorOverflowDiagnostics,
    *,
    backend: str,
) -> None:
    if diagnostics.atom_overflow_components:
        warnings.warn(
            f"{backend}: refused {diagnostics.atom_overflow_assignments} "
            f"assignment(s) whose precursors include "
            f"{diagnostics.atom_overflow_components} component(s) over "
            f"{diagnostics.precursor_atom_cap} atoms.",
            UserWarning,
            stacklevel=2,
        )
    if diagnostics.edited_atom_overflow_assignments:
        warnings.warn(
            f"{backend}: dropped {diagnostics.edited_atom_overflow_assignments} "
            f"assignment(s) over {diagnostics.edited_atom_cap} edited atoms.",
            UserWarning,
            stacklevel=2,
        )
    if diagnostics.orphaned_components:
        # Named loudly and with the fragment's formula, because the fix is to the
        # *template*, not to the molecule or the run: a template whose product side maps
        # an atom its precursor side never mentions orphans whatever hung off that atom.
        # `vertsy.template_lint` finds the same fault in the template text, before
        # anything is matched, and is the cheaper check by the number of applications a
        # run makes.
        dropped = (
            ", ".join(
                f"{formula} (molecule {mol_idx}, rule {rule_id}, application {app_idx})"
                for mol_idx, rule_id, app_idx, formula in diagnostics.orphaned_rows[:5]
            )
            or f"{diagnostics.undescribed_components} fragment(s)"
        )
        more = (
            f" and {len(diagnostics.orphaned_rows) - 5} more"
            if len(diagnostics.orphaned_rows) > 5
            else ""
        )
        warnings.warn(
            f"{backend}: refused "
            f"{diagnostics.orphaned_component_applications} application(s) that "
            f"orphaned {diagnostics.orphaned_components} component(s) -- a piece of "
            f"the molecule the template's precursor side never mentions, so the "
            f"disconnection is incomplete rather than selective. Dropped: {dropped}"
            f"{more}. Run `vertsy.template_lint` over the rule set to find the same "
            f"fault in the template text.",
            UserWarning,
            stacklevel=2,
        )


def precursor_signature(precursors: Iterable[CMol]) -> PrecursorSignature:
    """What makes two precursor sets the same set, for the deduplication in
    :func:`vesma.cpu_matcher.generate_precursors`.

    **Bond orders, not just bond presence.** Using `bond_presence` here discarded the one thing a
    bond-order-only rule changes, so every application of such a rule had the same signature and
    all but the first were dropped: pentane, pent-2-ene and pent-3-ene were one molecule by this
    measure. `MC 79` -- hydrogenation, `[CH1,CH2,CH3:1]-[CH1,CH2,CH3:2]>>[C:1]=[C:2]` -- found
    twenty assignments on a 33-atom macrocycle and returned **one** outcome, so the reference step
    it should have reproduced was scored a miss. Every oxidation, hydrogenation and aromatisation
    was affected the same way, and the per-step coverage numbers measured through this path are
    under-estimates by however much that cost.

    Kekule spelling is not a reason to go back: a `CMol` stores a perceived-aromatic bond as the
    single `DELOCALIZED` class rather than as alternating orders, so two spellings of one aromatic
    ring already carry identical bonds here.

    Hydrogen too, because `Z` + charge + bonds leaves it free -- an `[nH]` and an `n` differ in
    nothing else -- and a signature that cannot tell them apart would drop one of them.
    """
    return tuple(
        (p.Z.tobytes(), p.C.tobytes(), p.H.tobytes(), p.bonds.tobytes())
        for p in precursors
    )


__all__ = [
    "PrecursorMappingResult",
    "PrecursorOverflowDiagnostics",
    "ReactionEdit",
    "ReactionPattern",
    "ReactionTemplate",
    "RetroOutcome",
    "add_precursor_overflow_diagnostics",
    "parse_reaction_smarts",
    "precursors_from_mapping",
    "precursors_from_mapping_result",
    "precursor_signature",
    "reaction_patterns_from_smarts",
    "reaction_pickle_cache_path",
    "validate_atom_cap",
    "warn_precursor_overflow",
]

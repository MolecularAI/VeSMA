"""Builders for the Vesma scaling benchmarks.

Each backend is exposed as a ``build_*`` function returning ``(target, finalize)``:

- ``target`` is a zero-argument callable wrapping ONLY the matching call. It is the
  region timed by ``pytest-benchmark`` (catalog/library/matcher construction happens
  before it and is excluded from timing).
- ``finalize(result)`` runs once after timing and returns the non-timing ``extra_info``
  dict (match counts, corpus sizes, diagnostics) attached to the benchmark record.

The corpus loaders are shared with ``bench_match_scaling.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import lru_cache, partial
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import FilterCatalog, rdSubstructLibrary

from tests.datadirs import external_data_root
from tests.datasets import (
    load_compatible_templates,
    load_match_molecules,
    load_paroutes_n1_d2_match_uspto42k_positive_pairs,
    read_paroutes_n1_d2_f0_smiles,
)
from vesma.cpu_matcher import CpuMatcher, CpuReactionMatcher
from vesma.limits import GPU_PRECURSOR_ATOM_CAP
from vesma.parallel import resolve_n_jobs
from vesma.rdkit_matcher import RdkitReactionMatcher, rdkit_reactions_from_smarts
from vesma.reaction import (
    DEFAULT_REACTION_PROCESSING,
    GPU_PARITY_REACTION_PROCESSING,
    MoleculeProcessing,
    ReactionProcessing,
    ReactionTemplate,
)

PAROUTES_DIR = external_data_root() / "external" / "paroutes" / "processed"

Builder = Callable[..., "tuple[Callable[[], Any], Callable[[Any], dict[str, Any]]]"]


@dataclass(frozen=True)
class MatchCorpus:
    source_molecule_indices: tuple[int, ...]
    smiles: tuple[str, ...]
    rd_mols: tuple[Any, ...]
    cmols: tuple[Any, ...]
    gmols: tuple[Any, ...]
    templates: tuple[ReactionTemplate, ...]
    lhs_queries: tuple[Any, ...]
    skipped_smiles: int
    skipped_templates: int

    @property
    def pair_count(self) -> int:
        return len(self.rd_mols) * len(self.templates)


# --------------------------------------------------------------------------- #
# Corpus loading
# --------------------------------------------------------------------------- #
def load_match_corpus(bench_mols: int) -> MatchCorpus:
    return load_match_corpus_by_indices(molecule_indices(bench_mols))


@lru_cache
def load_match_corpus_by_indices(molecule_indices_: tuple[int, ...]) -> MatchCorpus:
    all_smiles = load_frontier_smiles()
    smiles = tuple(all_smiles[index] for index in molecule_indices_)
    molecules = load_match_molecules(smiles, n_jobs=-1, cache=True)
    templates = load_templates()
    lhs_queries = load_rdkit_lhs_queries()

    if not molecules:
        raise ValueError("Benchmark corpus contains no valid frontier molecules.")
    if not templates:
        raise ValueError("Benchmark corpus contains no compatible templates.")

    return MatchCorpus(
        source_molecule_indices=molecule_indices_,
        smiles=tuple(molecule.smiles for molecule in molecules),
        rd_mols=tuple(molecule.rd_mol for molecule in molecules),
        cmols=tuple(molecule.cmol for molecule in molecules),
        gmols=tuple(molecule.gmol for molecule in molecules),
        templates=templates,
        lhs_queries=tuple(lhs_queries),
        skipped_smiles=0,
        skipped_templates=0,
    )


def molecule_indices(bench_mols: int) -> tuple[int, ...]:
    smiles = load_frontier_smiles()
    if bench_mols < 0:
        return tuple(range(len(smiles)))
    if bench_mols == 0:
        raise ValueError("bench_mols must be positive or -1 for all molecules.")
    if bench_mols > len(smiles):
        raise ValueError("bench_mols exceeds the benchmark molecule count.")
    return tuple(range(bench_mols))


@lru_cache
def load_frontier_smiles() -> tuple[str, ...]:
    path = PAROUTES_DIR / "n1-d2-f0.smi.gz"
    if not path.exists():
        pytest.skip(f"Missing benchmark frontier file: {path}")
    return read_paroutes_n1_d2_f0_smiles()


@lru_cache
def load_templates() -> tuple[ReactionTemplate, ...]:
    return load_compatible_templates()


@lru_cache
def load_rdkit_lhs_queries() -> tuple[Any, ...]:
    templates = load_templates()
    lhs_queries = tuple(
        Chem.MolFromSmarts(lhs_smarts(template)) for template in templates
    )
    if any(query is None for query in lhs_queries):
        raise ValueError(
            "RDKit-compatible template filtering produced a null LHS query."
        )
    return lhs_queries


def load_positive_pairs(corpus: MatchCorpus) -> list[tuple[int, int]]:
    return load_paroutes_n1_d2_match_uspto42k_positive_pairs(
        corpus.source_molecule_indices
    )


def lhs_smarts(template: ReactionTemplate) -> str:
    parts = template.smarts.partition(">>")
    if parts[1] != ">>":
        raise ValueError(f"Reaction template is missing >> separator: {template.id}")
    return parts[0].strip()


def template_smarts(corpus: MatchCorpus) -> tuple[str, ...]:
    return tuple(template.smarts for template in corpus.templates)


# --------------------------------------------------------------------------- #
# extra_info helpers
# --------------------------------------------------------------------------- #
def match_only_info(
    corpus: MatchCorpus,
    *,
    matches: int,
    path: str,
    build_seconds: float | None = None,
    **extra: Any,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "matches": matches,
        "molecules": len(corpus.rd_mols),
        "rules": len(corpus.templates),
        "pairs": corpus.pair_count,
        "path": path,
    }
    if build_seconds is not None:
        info["build_seconds"] = build_seconds
    info.update(extra)
    return info


#: What a split leg does about precursor identity, inside the region it times. The family is
#: meant to measure rule application and precursor materialisation, and identity is a separate
#: stage the caller composes -- `matcher.keyed_match_folding` is what made it liftable rather than
#: a property of the backend. But the legs do not agree, and for a long time nothing said so:
#: `cuda-host` and `cuda-device` call `match_keyed`, which runs a WL pass on the device and folds
#: a dedup map on the host, while `rdkit-serial`, the `cpu-*` legs and `metal-raw` call plain
#: `match` and do neither. Two rows of that table were doing strictly more work than the rest.
#: `CudaReactionMatcher.match` exists and skips all of it, so the keyed entry point was a choice;
#: moving those legs onto it needs a CUDA host to re-measure, which is why they are labelled here
#: rather than changed. Declared per leg and required, so a new leg cannot quietly join the
#: wrong group.
PRECURSOR_IDENTITY_NONE = "none"
PRECURSOR_IDENTITY_KEYED_DEDUPED = "keyed-deduped"
PRECURSOR_IDENTITY_OUTCOME_DEDUPED = "outcome-deduped"


def split_info(
    corpus: MatchCorpus,
    positive_pairs: list[tuple[int, int]],
    *,
    captured: dict[str, Any],
    compile_seconds: float,
    precursor_identity: str,
    **extra: Any,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "molecules": len(corpus.rd_mols),
        "rules": len(corpus.templates),
        "positive_pairs": len(positive_pairs),
        "precursors": captured.get("precursors"),
        "hits": captured.get("hits"),
        "compile_seconds": compile_seconds,
        "precursor_identity": precursor_identity,
    }
    info.update(extra)
    return info


def wrapper_parallelism_metrics(
    n_jobs: int, *, parallelism: str = "fork-processes"
) -> dict[str, Any]:
    return {
        "n_jobs": n_jobs,
        "wrapper_parallelism": "serial" if n_jobs == 1 else parallelism,
    }


# --------------------------------------------------------------------------- #
# Match-only builders
# --------------------------------------------------------------------------- #
def build_rdkit_filtercatalog(corpus: MatchCorpus):
    started = perf_counter()
    params = FilterCatalog.FilterCatalogParams()
    catalog = FilterCatalog.FilterCatalog(params)
    for index, query in enumerate(corpus.lhs_queries):
        matcher = FilterCatalog.SmartsMatcher(f"R_{index}", query)
        if not matcher.IsValid():
            raise ValueError(f"RDKit FilterCatalog rejected query {index}.")
        catalog.AddEntry(FilterCatalog.FilterCatalogEntry(f"R_{index}", matcher))
    build_seconds = perf_counter() - started
    smiles = list(corpus.smiles)

    def target() -> int:
        rows = FilterCatalog.RunFilterCatalog(catalog, smiles, numThreads=0)
        return sum(len(row) for row in rows)

    def finalize(result: Any) -> dict[str, Any]:
        return match_only_info(
            corpus,
            matches=int(result),
            path="FilterCatalog rules -> frontier SMILES",
            build_seconds=build_seconds,
        )

    return target, finalize


def build_rdkit_substructlibrary_frontier(corpus: MatchCorpus):
    started = perf_counter()
    library = rdSubstructLibrary.SubstructLibrary(
        rdSubstructLibrary.MolHolder(),
        rdSubstructLibrary.PatternHolder(),
    )
    for mol in corpus.rd_mols:
        library.AddMol(mol)
    build_seconds = perf_counter() - started
    queries = list(corpus.lhs_queries)
    max_results = len(corpus.rd_mols)

    def target() -> int:
        matches = 0
        for query in queries:
            matches += len(
                library.GetMatches(
                    query,
                    recursionPossible=True,
                    useChirality=False,
                    useQueryQueryMatches=False,
                    numThreads=0,
                    maxResults=max_results,
                )
            )
        return matches

    def finalize(result: Any) -> dict[str, Any]:
        return match_only_info(
            corpus,
            matches=int(result),
            path="SubstructLibrary frontier -> cached rule LHS queries",
            build_seconds=build_seconds,
        )

    return target, finalize


def build_cpu_match_only(corpus: MatchCorpus, *, n_jobs: int | None = None):
    matcher = CpuMatcher()
    n_jobs = resolve_n_jobs(n_jobs)
    patterns = [template.pattern.lhs for template in corpus.templates]
    gmols = list(corpus.gmols)

    def target() -> Any:
        return matcher.has_match_many(patterns, gmols, n_jobs=n_jobs)

    def finalize(result: Any) -> dict[str, Any]:
        return match_only_info(
            corpus,
            matches=int(np.asarray(result).sum()),
            path="Vesma CPU low-level matcher",
            **wrapper_parallelism_metrics(n_jobs),
        )

    return target, finalize


def build_cuda_raw_match_only(corpus: MatchCorpus):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CuPy cannot access CUDA on this machine.")
    from vesma.cuda_matcher import CudaMatcher, cuda_stack_gmols
    from vesma.cuda_prune import (
        match_fused_cuda,
        prepare_cuda_prune_batch,
        prepare_cuda_prune_batch_from_device,
    )

    started = perf_counter()
    matcher = CudaMatcher()
    patterns = [template.pattern.lhs for template in corpus.templates]
    gmols = list(corpus.gmols)
    pattern_batch = prepare_cuda_prune_batch(patterns, [])
    cp.cuda.Stream.null.synchronize()
    build_seconds = perf_counter() - started

    def target() -> Any:
        molecule_batch = cuda_stack_gmols(gmols)
        batch = prepare_cuda_prune_batch_from_device(
            len(patterns),
            pattern_batch.pattern_atom_counts_device,
            pattern_batch.pattern_atoms_u32,
            pattern_batch.pattern_rows_u64,
            pattern_batch.pattern_bonds_u8,
            pattern_groups_u32_device=pattern_batch.pattern_groups_u32,
            molecule_atom_counts_device=molecule_batch.atom_counts_device,
            molecule_atoms_u32=molecule_batch.atoms_device,
            molecule_rows_u64=molecule_batch.rows_device,
            molecule_bonds_u8=molecule_batch.bonds_device,
        )
        result = match_fused_cuda(
            batch,
            threads_per_block=matcher.threads_per_block,
            max_prune_rounds=matcher.max_prune_rounds,
            max_leaf_checks=matcher.max_leaf_checks,
        )
        matches = result.status_codes_host() == 1
        cp.cuda.Stream.null.synchronize()
        return matches

    def finalize(result: Any) -> dict[str, Any]:
        return match_only_info(
            corpus,
            matches=int(np.asarray(result).sum()),
            path="Vesma CUDA low-level matcher without folded prefilter",
            build_seconds=build_seconds,
        )

    return target, finalize


def build_cuda_prefiltered_match_only(corpus: MatchCorpus):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CuPy cannot access CUDA on this machine.")
    from vesma.cuda_matcher import (
        CudaFoldedPrefilteredMatcher,
        CudaMatchOnlyPrefilterDiagnostics,
        cuda_stack_gmols,
        host_array,
    )
    from vesma.cuda_pathfp import (
        compact_folded_prefilter_pairs_cuda,
        compute_folded_path_fingerprints_cuda,
    )
    from vesma.cuda_prune import (
        match_fused_cuda,
        pack_collection_atom_groups,
        prepare_cuda_prune_batch,
        prepare_cuda_prune_batch_from_device_pairs,
    )
    from vesma.cuda_retro import CudaMoleculeBatch

    started = perf_counter()
    matcher = CudaFoldedPrefilteredMatcher()
    patterns = [template.pattern.lhs for template in corpus.templates]
    gmols = list(corpus.gmols)
    pattern_prune_batch = prepare_cuda_prune_batch(patterns, [])
    pattern_batch = CudaMoleculeBatch(
        tokens=(),
        atom_counts_device=pattern_prune_batch.pattern_atom_counts_device,
        atoms_device=pattern_prune_batch.pattern_atoms_u32,
        bonds_device=pattern_prune_batch.pattern_bonds_u8,
        rows_device=pattern_prune_batch.pattern_rows_u64,
    )
    pattern_fps = compute_folded_path_fingerprints_cuda(
        pattern_batch,
        query=True,
        config=matcher.fingerprint_config,
    )
    cp.cuda.Stream.null.synchronize()
    build_seconds = perf_counter() - started
    captured: dict[str, Any] = {}

    def target() -> Any:
        molecule_batch = cuda_stack_gmols(gmols)
        started = perf_counter()
        molecule_fps = compute_folded_path_fingerprints_cuda(
            molecule_batch,
            query=False,
            config=matcher.fingerprint_config,
        )
        prefiltered = compact_folded_prefilter_pairs_cuda(
            molecule_fingerprints_device=molecule_fps,
            pattern_fingerprints_device=pattern_fps,
            molecule_atom_counts_device=molecule_batch.atom_counts_device,
            pattern_atom_counts_device=pattern_batch.atom_counts_device,
        )
        cp.cuda.Stream.null.synchronize()
        fingerprint_seconds = perf_counter() - started

        matches = np.zeros((len(patterns), len(gmols)), dtype=bool)
        if prefiltered.pair_count == 0:
            captured["diagnostics"] = CudaMatchOnlyPrefilterDiagnostics(
                dense_pair_count=prefiltered.dense_pair_count,
                candidate_pair_count=0,
                pruned_pair_count=prefiltered.dense_pair_count,
                prune_ratio=(0.0 if prefiltered.dense_pair_count == 0 else 1.0),
                fingerprint_seconds=fingerprint_seconds,
                inner_match_seconds=0.0,
            )
            return matches

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
        result = match_fused_cuda(
            batch,
            threads_per_block=matcher.matcher.threads_per_block,
            max_prune_rounds=matcher.matcher.max_prune_rounds,
            max_leaf_checks=matcher.matcher.max_leaf_checks,
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
        matches[pattern_indices, molecule_indices] = statuses == 1

        pruned_pair_count = prefiltered.dense_pair_count - prefiltered.pair_count
        captured["diagnostics"] = CudaMatchOnlyPrefilterDiagnostics(
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
        return matches

    def finalize(result: Any) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if "diagnostics" in captured:
            extra["prefilter_diagnostics"] = captured["diagnostics"].__dict__
        return match_only_info(
            corpus,
            matches=int(np.asarray(result).sum()),
            path="Vesma CUDA folded-fingerprint prefiltered match-only",
            build_seconds=build_seconds,
            **extra,
        )

    return target, finalize


def build_metal_raw_match_only(corpus: MatchCorpus):
    MetalMatcher = _require_metal("MetalMatcher")
    matcher = MetalMatcher()
    patterns = [template.pattern.lhs for template in corpus.templates]
    gmols = list(corpus.gmols)

    def target() -> Any:
        return matcher.has_match_many(patterns, gmols)

    def finalize(result: Any) -> dict[str, Any]:
        return match_only_info(
            corpus,
            matches=int(np.asarray(result).sum()),
            path="Vesma Metal low-level matcher without folded prefilter",
        )

    return target, finalize


def build_metal_prefiltered_match_only(corpus: MatchCorpus):
    MetalFoldedPrefilteredMatcher = _require_metal("MetalFoldedPrefilteredMatcher")
    matcher = MetalFoldedPrefilteredMatcher()
    patterns = [template.pattern.lhs for template in corpus.templates]
    gmols = list(corpus.gmols)

    def target() -> Any:
        return matcher.has_match_many(patterns, gmols)

    def finalize(result: Any) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if matcher.last_diagnostics is not None:
            extra["prefilter_diagnostics"] = matcher.last_diagnostics.__dict__
        return match_only_info(
            corpus,
            matches=int(np.asarray(result).sum()),
            path="Vesma Metal folded-fingerprint prefiltered match-only",
            **extra,
        )

    return target, finalize


# --------------------------------------------------------------------------- #
# Match-plus-split builders
# --------------------------------------------------------------------------- #
def build_rdkit_match_plus_split(
    corpus: MatchCorpus,
    positive_pairs: list[tuple[int, int]],
    *,
    n_jobs: int | None = None,
):
    n_jobs = resolve_n_jobs(n_jobs)
    started = perf_counter()
    # Workers are part of the matcher now, so they are stood up here with the rules and the
    # measured region below pays only for the molecules.
    matcher = RdkitReactionMatcher(
        rdkit_reactions_from_smarts(template_smarts(corpus)), n_jobs=n_jobs
    )
    compile_seconds = perf_counter() - started
    rd_mols = corpus.rd_mols
    captured: dict[str, Any] = {}

    def target() -> None:
        precursors, hits = matcher.match(rd_mols, positive_pairs)
        captured["precursors"] = len(precursors)
        captured["hits"] = len(hits)

    def finalize(result: Any) -> dict[str, Any]:
        return split_info(
            corpus,
            positive_pairs,
            captured=captured,
            compile_seconds=compile_seconds,
            # `RunReactants` plus `split_outcome_or_error`: sanitised, never canonicalised and
            # never keyed. `match` routes through `match_keyed` with no key, so the worker's
            # `collapse_by_key` does not run.
            precursor_identity=PRECURSOR_IDENTITY_NONE,
            **wrapper_parallelism_metrics(n_jobs, parallelism="fork-processes"),
        )

    return target, finalize


@lru_cache
def load_rdchiral_reactions(smarts: tuple[str, ...]) -> tuple[Any, ...]:
    """Compile the template collection once per session.

    Compiling all 42,554 templates costs roughly three minutes, and the scaling sweep
    builds a backend per molecule size and per enantiomer variant, so the compile is
    cached across those builder calls. Reuse is safe: ``rdchiralRun`` resets each
    template's atom map numbers before applying it.
    """
    from vesma.rdchiral_matcher import rdchiral_reactions_from_smarts

    return rdchiral_reactions_from_smarts(smarts)


def build_rdchiral_match_plus_split(
    corpus: MatchCorpus,
    positive_pairs: list[tuple[int, int]],
    *,
    combine_enantiomers: bool = True,
):
    pytest.importorskip("rdchiral")
    from vesma.rdchiral_matcher import (
        RdchiralReactionMatcher,
        rdchiral_reactants_from_smiles,
        rdchiral_version,
    )

    started = perf_counter()
    reactions = load_rdchiral_reactions(template_smarts(corpus))
    compile_seconds = perf_counter() - started

    # RDChiral parses each molecule into a chiral and an achiral copy. That is its
    # analogue of the prebuilt rd_mols/cmols the other backends receive, so it is built
    # outside the timed region and reported separately rather than hidden.
    started = perf_counter()
    reactants = rdchiral_reactants_from_smiles(corpus.smiles)
    reactants_build_seconds = perf_counter() - started

    matcher = RdchiralReactionMatcher(
        reactions, combine_enantiomers=combine_enantiomers
    )
    captured: dict[str, Any] = {}

    def target() -> None:
        precursors, hits = matcher.match(reactants, positive_pairs)
        captured["precursors"] = len(precursors)
        captured["hits"] = len(hits)

    def finalize(result: Any) -> dict[str, Any]:
        extra: dict[str, Any] = {
            "reactants_build_seconds": reactants_build_seconds,
            "rdchiral_version": rdchiral_version(),
        }
        if matcher.last_diagnostics is not None:
            extra["rdchiral_diagnostics"] = matcher.last_diagnostics.__dict__
        return split_info(
            corpus,
            positive_pairs,
            captured=captured,
            compile_seconds=compile_seconds,
            # RDChiral merges stereoisomeric outcomes itself, so its count is
            # outcome-deduplicated rather than a raw application multiset -- not a WL identity
            # fold, but not a raw count either.
            precursor_identity=PRECURSOR_IDENTITY_OUTCOME_DEDUPED,
            **extra,
            **wrapper_parallelism_metrics(1),
        )

    return target, finalize


def build_cpu_match_plus_split(
    corpus: MatchCorpus,
    positive_pairs: list[tuple[int, int]],
    *,
    n_jobs: int | None = None,
    max_precursor_atoms: int | None = None,
    processing: ReactionProcessing = DEFAULT_REACTION_PROCESSING,
):
    n_jobs = resolve_n_jobs(n_jobs)
    started = perf_counter()
    matcher = CpuReactionMatcher(
        [template.pattern for template in corpus.templates],
        max_precursor_atoms=max_precursor_atoms,
        processing=processing,
    )
    compile_seconds = perf_counter() - started
    cmols = corpus.cmols
    captured: dict[str, Any] = {}

    def target() -> None:
        precursors, hits = matcher.match(cmols, positive_pairs, n_jobs=n_jobs)
        captured["precursors"] = len(precursors)
        captured["hits"] = len(hits)

    def finalize(result: Any) -> dict[str, Any]:
        extra: dict[str, Any] = wrapper_parallelism_metrics(
            n_jobs, parallelism="fork-processes"
        )
        if matcher.last_diagnostics is not None:
            extra["cpu_retro_diagnostics"] = matcher.last_diagnostics.__dict__
        extra["reaction_processing"] = {
            "before_match": processing_names(processing.before_match),
            "after_split": processing_names(processing.after_split),
        }
        return split_info(
            corpus,
            positive_pairs,
            captured=captured,
            compile_seconds=compile_seconds,
            # Plain `match`: one precursor row per application, no identity computed.
            precursor_identity=PRECURSOR_IDENTITY_NONE,
            **extra,
        )

    return target, finalize


def processing_names(processing: MoleculeProcessing) -> list[str]:
    return [
        operation.name.lower().replace("_", "-")
        for operation in MoleculeProcessing
        if operation is not MoleculeProcessing.NONE and processing & operation
    ]


def build_cuda_match_plus_split(
    corpus: MatchCorpus,
    positive_pairs: list[tuple[int, int]],
    *,
    frontier_mode: Literal["device", "host"] = "device",
):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CuPy cannot access CUDA on this machine.")
    from vesma.cuda_matcher import CudaReactionMatcher

    started = perf_counter()
    matcher = CudaReactionMatcher(
        [template.pattern for template in corpus.templates],
        frontier_mode=frontier_mode,
    )
    compile_seconds = perf_counter() - started
    cmols = corpus.cmols
    captured: dict[str, Any] = {}

    def target() -> None:
        precursors, precursor_keys, hits = matcher.match_keyed(cmols, positive_pairs)
        cp.cuda.Stream.null.synchronize()
        captured["precursors"] = len(precursors)
        captured["hits"] = len(hits)

    def finalize(result: Any) -> dict[str, Any]:
        extra: dict[str, Any] = {
            "cuda_split_runner": f"match-keyed-{frontier_mode}-frontier"
        }
        if matcher.last_diagnostics is not None:
            extra["cuda_diagnostics"] = matcher.last_diagnostics.__dict__
        return split_info(
            corpus,
            positive_pairs,
            captured=captured,
            compile_seconds=compile_seconds,
            # `match_keyed`, not `match`: a WL pass on the device and a `key_to_physical` fold on
            # the host, both inside the timed region, in either frontier mode. So these two rows
            # do strictly more than every other leg in this family, and `len(precursors)` reads
            # as the application count because the batch returned is a logical view over
            # deduplicated physical storage. `CudaReactionMatcher.match` skips all of it; moving
            # these legs onto it wants a CUDA host to re-measure, so for now the record says
            # which group the row belongs to.
            precursor_identity=PRECURSOR_IDENTITY_KEYED_DEDUPED,
            **extra,
        )

    return target, finalize


def build_metal_match_plus_split(
    corpus: MatchCorpus,
    positive_pairs: list[tuple[int, int]],
):
    """Metal reaction application, precursors joined into one returned batch.

    Plain `match`, to stand beside the `cpu-*` and `rdkit-serial` rows: rules applied and
    precursors materialised, with identity left to whatever the caller composes on top. Two other
    deliveries exist -- pulling `match_batches` and dropping each piece, and folding identity in
    as the pieces arrive -- and both were legs here for a while. They are not legs here any more:
    streaming keeps nothing a consumer can use, the fold does strictly more work than every
    unkeyed row in this family, and measuring either against the rest answers no question this
    table asks. `make bench-metal-precursor-streaming` compares all three in one interleaved
    process, which is where a delivery A/B belongs; route recovery is where the fold's effect on
    a real search is measured.
    """
    MetalReactionMatcher = _require_metal_reaction_matcher()
    started = perf_counter()
    matcher = MetalReactionMatcher([template.pattern for template in corpus.templates])
    compile_seconds = perf_counter() - started
    cmols = corpus.cmols
    captured: dict[str, Any] = {}

    def target() -> None:
        precursors, hits = matcher.match(cmols, positive_pairs)
        captured["precursors"] = len(precursors)
        captured["hits"] = len(hits)

    def finalize(result: Any) -> dict[str, Any]:
        extra: dict[str, Any] = {"metal_precursor_delivery": "returned"}
        # The Metal matcher computes a full phase breakdown on every call and used to throw it
        # away here, which is why no record explained the shape of the scaling curve. The CPU
        # and CUDA builders above both attach theirs; this is the same line.
        if matcher.last_diagnostics is not None:
            extra["metal_diagnostics"] = asdict(matcher.last_diagnostics)
        return split_info(
            corpus,
            positive_pairs,
            captured=captured,
            compile_seconds=compile_seconds,
            # Plain `match`: one precursor row per application, no identity computed.
            precursor_identity=PRECURSOR_IDENTITY_NONE,
            **extra,
        )

    return target, finalize


# --------------------------------------------------------------------------- #
# Metal import/probe guards (skip cleanly off-darwin or when unavailable)
# --------------------------------------------------------------------------- #
def _require_metal(class_name: str):
    import sys

    if sys.platform != "darwin":
        pytest.skip("Metal backend is only available on macOS.")
    try:
        import vesma.metal_matcher as metal_matcher
        from vesma.metal_matcher import GpuUnavailableError, probe_metal
    except ImportError as exc:  # pragma: no cover - import guard
        pytest.skip(f"Metal backend unavailable: {exc}")
    try:
        probe_metal()
    except (GpuUnavailableError, RuntimeError) as exc:
        pytest.skip(f"Metal backend not available: {exc}")
    return getattr(metal_matcher, class_name)


def _require_metal_reaction_matcher():
    import sys

    if sys.platform != "darwin":
        pytest.skip("Metal backend is only available on macOS.")
    try:
        from vesma.metal_matcher import GpuUnavailableError, probe_metal
        from vesma.metal_reaction_matcher import MetalReactionMatcher
    except ImportError as exc:  # pragma: no cover - import guard
        pytest.skip(f"Metal backend unavailable: {exc}")
    try:
        probe_metal()
    except (GpuUnavailableError, RuntimeError) as exc:
        pytest.skip(f"Metal backend not available: {exc}")
    return MetalReactionMatcher


# --------------------------------------------------------------------------- #
# Builder registries
# --------------------------------------------------------------------------- #
# `cpu-parallel` binds no `n_jobs`, so `vesma.parallel.resolve_n_jobs` reads `VESMA_N_JOBS` and
# falls back to -1 when it is unset -- which is what every record so far was measured at, so the
# default is unchanged. Binding -1 here instead made the variable silently inert: `resolve_n_jobs`
# returns an explicit argument without consulting the environment, so a run asked for four workers
# got ten and recorded "-1", and only the record was honest about it.
#
# The `*-serial` entries stay pinned at 1. Those are not a parallelism *setting* but the definition
# of the backend, and an environment variable must not be able to turn the serial reference into a
# parallel one.
#
# RDKit's own threading is deliberately left alone. `VESMA_N_JOBS` names this library's worker
# count; `rdkit-filtercatalog-rules` and `rdkit-substructlibrary-frontier` pass `numThreads=0` to
# RDKit's C++, which means all cores and is not ours to reinterpret. So limiting VESMA_N_JOBS
# narrows the CPU backends and not the RDKit ones -- worth knowing before comparing a capped run.
MATCH_ONLY_BUILDERS: dict[str, Builder] = {
    "rdkit-filtercatalog-rules": build_rdkit_filtercatalog,
    "rdkit-substructlibrary-frontier": build_rdkit_substructlibrary_frontier,
    "cpu-serial": partial(build_cpu_match_only, n_jobs=1),
    "cpu-parallel": build_cpu_match_only,
    "metal": build_metal_prefiltered_match_only,
    "metal-raw": build_metal_raw_match_only,
    "cuda": build_cuda_prefiltered_match_only,
    "cuda-raw": build_cuda_raw_match_only,
}

SPLIT_BUILDERS: dict[str, Builder] = {
    "rdkit-serial": partial(build_rdkit_match_plus_split, n_jobs=1),
    # RDChiral deduplicates outcomes internally, so its precursor count is
    # outcome-deduplicated rather than a raw application multiset; the enantiomer
    # variants differ only in whether RDChiral merges stereoisomeric outcomes itself.
    "rdchiral-serial": build_rdchiral_match_plus_split,
    "rdchiral-serial-nocombine": partial(
        build_rdchiral_match_plus_split, combine_enantiomers=False
    ),
    "cpu-serial": partial(build_cpu_match_plus_split, n_jobs=1),
    "cpu-parallel": build_cpu_match_plus_split,
    "cpu-serial-gpu-parity": partial(
        build_cpu_match_plus_split,
        n_jobs=1,
        max_precursor_atoms=GPU_PRECURSOR_ATOM_CAP,
        processing=GPU_PARITY_REACTION_PROCESSING,
    ),
    "cpu-parallel-gpu-parity": partial(
        build_cpu_match_plus_split,
        max_precursor_atoms=GPU_PRECURSOR_ATOM_CAP,
        processing=GPU_PARITY_REACTION_PROCESSING,
    ),
    "metal-raw": build_metal_match_plus_split,
    "cuda-host": partial(build_cuda_match_plus_split, frontier_mode="host"),
    "cuda-device": partial(build_cuda_match_plus_split, frontier_mode="device"),
}

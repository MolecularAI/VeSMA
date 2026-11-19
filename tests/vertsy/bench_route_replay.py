from __future__ import annotations

import csv
import gc
import gzip
import json
import os
import resource
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import joblib
import numpy as np
import pytest
from rdkit import Chem

from tests.datadirs import external_data_root
from tests.vertsy.paper_provenance import source_artifact
from tests.vertsy.paroutes_reference import (
    clean_reaction_smarts,
    load_raw_template_rows,
    load_route_template_rows,
    raw68k_rule_index,
    route_recovery_rule_index,
    uspto42k_rule_index,
)
from vertsy.interner import MISSING_SID, DictInterner
from vertsy.interop import route_from_aizynth_dict
from vertsy.replay import precursor_replay_search, replay_search, route_precursor_sets
from vertsy.routes import (
    Route,
    encode_route,
    iter_solved_routes,
    leaves,
    library_route_replayed,
    loose_route_replayed,
    reaction_count,
    reaction_depth,
    solved_targets,
    strict_route_replayed,
)
from vertsy.search import Expansions, MoleculeBatch, expansion, object_mol_array
from vertsy.store.numpy_store import NumpyExpansionStore
from vesma.cmol import CMol
from vesma.interop import cmol_from_mol
from vesma.keyer import (
    Keyer,
    cmol_rdkit_smiles_keyer,
    rdkit_mol_canonical_smiles,
    rdkit_smiles_keyer,
    short_ring_wl_keyer,
)
from vesma.matcher import HitArray, Pair
from vesma.parallel import (
    effective_n_jobs,
    parallel_chunks,
    validate_n_jobs,
)
from vesma.rdkit_matcher import (
    RdkitReactionMatcher,
    rdkit_reactions_from_smarts,
)
from vesma.reaction import (
    ReactionPattern,
    parse_reaction_smarts,
    reaction_patterns_from_smarts,
)

pytestmark = [pytest.mark.perf, pytest.mark.paper_benchmark]

PAROUTES_DIR = external_data_root() / "external" / "paroutes"
# Committed (reaction_hash, retro_template) subset filtered to the templates used by
# n1+n5 routes; replaces the 116 MB raw library for replay. See
# paroutes/build_used14k_reactions.py.
ROUTE_TEMPLATES_FILE = "used14k-reactions.tsv.gz"
# Full raw USPTO template library deduplicated to its 67,557 unique retro_template SMARTS
# ("raw68k"); committed so precursor replay can run the whole raw library without the 116 MB
# download. Ids do not align with the routes (agnostic-only). See build_raw68k_templates.py.
RAW68K_TEMPLATES_FILE = "raw68k-templates.tsv.gz"

DEFAULT_SIZES = (1, 10, 100)
DEFAULT_BACKENDS = ("rdkit",)
DEFAULT_MODES = ("guided", "topology")
# guided/topology build the matcher from the route's own reactions ("route-reactions");
# precursor replay reads a full library and filters by precursor set, so it needs a real rule
# set. See rule_set_smarts / skip_invalid_replay_combo.
DEFAULT_RULE_SETS = ("route-reactions",)
SUPPORTED_RULE_SETS = {"route-reactions", "used14k", "raw68k", "uspto42k", "curated"}
ROUTE_REACTION_RULE_SET = "route-reactions"
RULE_SET_TEMPLATE_SOURCE = {
    "route-reactions": ROUTE_TEMPLATES_FILE,
    "used14k": "used14k-reactions",
    "raw68k": "raw68k-templates",
    "uspto42k": "uspto_unique_templates",
    "curated": "templates-curated",
}
# The hand-written collection, as one library. Precursor replay matches on precursor *sets*
# rather than rule ids, so these need no shared numbering with the reference -- which is what
# makes "does this collection reproduce the reference disconnection" askable of a set whose ids
# have nothing to do with PaRoutes'.
# Paths rather than names because the sets do not all live in one place. All of them are
# hand-written, the hartenfeller file included: its flattened non-recursive *retro* forms were
# written here from the forward templates, which came via chemdots from the paper's SI -- which
# is why it sits under external/ despite not being upstream data as shipped. medchem was then
# built as an add-on topping up the robust general chemistry hartenfeller had gaps in, so the
# two belong in one library rather than being compared against each other. All share the
# `template_code,retro_template,classification` columns.
CURATED_RULE_FILES = (
    ("templates", "templates-hartenfeller.csv"),
    ("templates", "templates-medchem.csv"),
    ("templates", "templates-protecting-groups.csv"),
    ("templates", "templates-carbohydrate.csv"),
    ("templates", "templates-selenium.csv"),
    ("templates", "templates-phosphorus.csv"),
)


def curated_smarts() -> tuple[str, ...]:
    """Every `retro_template` in the curated CSVs, in file then row order, deduplicated."""
    seen: dict[str, None] = {}
    for parts in CURATED_RULE_FILES:
        path = external_data_root().joinpath(*parts)
        if not path.exists():
            pytest.skip(f"Missing curated template file: {path}")
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                smarts = (row.get("retro_template") or "").strip()
                if smarts:
                    seen.setdefault(smarts, None)
    return tuple(seen)


# Precursor replay prunes fanout to reference edges, but a disconnection reproduced by several
# rules still multiplies route trees; cap enumeration so a pathological route cannot blow up.
PRECURSOR_ROUTE_CAP = 100_000

type Key = Any
type Mol = Any


class ReplayBackend(Protocol):
    name: str
    compile_seconds: float

    def prepare_case(self, case: ReplayCase) -> BackendCase: ...


@dataclass(frozen=True)
class ReplayCase:
    target_smiles: str
    route: Route[str]
    expected: Route[str]
    smarts: tuple[str, ...]
    patterns: tuple[ReactionPattern, ...]


@dataclass(frozen=True)
class BackendCase:
    target_key: Key
    target_mol: Mol
    mapped_route: Route[Key]
    stock: frozenset[Key]


@dataclass(frozen=True)
class ReplayRunResult:
    routes: frozenset[Route[Key]]
    diagnostics: Counter[str]


@dataclass(frozen=True)
class ReplayBenchmarkResult:
    case_count: int
    route_count: int
    rule_count: int
    reaction_count: int
    strict_count: int
    loose_count: int
    library_count: int
    # Per-route recovery flags (0/1), indexed by position in the prepared corpus, so a
    # downstream consumer can pick out exactly which routes diverged. route_set and
    # template_source in the extra_info identify the corpus the indices refer to.
    strict_recoveries: tuple[int, ...]
    loose_recoveries: tuple[int, ...]
    library_recoveries: tuple[int, ...]
    any_solved_recoveries: tuple[int, ...]
    any_solved_count: int
    same_leaves_different_intermediates: int
    same_reactions_different_leaves: int
    generated_route_count: int
    diagnostics: Counter[str]
    cuda_memory: dict[str, int]


@dataclass(frozen=True)
class RdkitReplayBackend:
    name: str
    matcher: RdkitReactionMatcher
    stock: frozenset[Key]
    compile_seconds: float

    def prepare_case(self, case: ReplayCase) -> BackendCase:
        target = Chem.MolFromSmiles(case.target_smiles)
        if target is None:
            raise ValueError("invalid-target-smiles")
        return BackendCase(
            target_key=rdkit_mol_canonical_smiles(target),
            target_mol=target,
            mapped_route=case.expected,
            stock=self.stock,
        )

    def stack(self, mols: Sequence[Mol]) -> MoleculeBatch[Mol]:
        return object_mol_array(mols)

    def match(
        self, mols: MoleculeBatch[Mol], pairs: Sequence[Pair]
    ) -> tuple[MoleculeBatch[Mol], Sequence[Key], HitArray]:
        precursors, hits = self.matcher.match(mols, pairs)
        precursor_keys = [rdkit_mol_canonical_smiles(mol) for mol in precursors]
        return object_mol_array(precursors), precursor_keys, hits


@dataclass(frozen=True)
class ObjectReplayBackend:
    name: str
    matcher: Any
    codec: Keyer[Any, Key]
    stock: frozenset[Key]
    compile_seconds: float

    def prepare_case(self, case: ReplayCase) -> BackendCase:
        target = target_cmol(case)
        mapped_route = encode_route(case.route, self.codec)
        return BackendCase(
            target_key=self.codec.encode(target),
            target_mol=target,
            mapped_route=mapped_route,
            stock=self.stock,
        )

    def stack(self, mols: Sequence[Mol]) -> MoleculeBatch[Mol]:
        return object_mol_array(mols)

    def match(
        self, mols: MoleculeBatch[Mol], pairs: Sequence[Pair]
    ) -> tuple[MoleculeBatch[Mol], Sequence[Key], HitArray]:
        precursors, hits = self.matcher.match(mols, pairs)
        precursor_keys = [self.codec.encode(mol) for mol in precursors]
        return object_mol_array(precursors), precursor_keys, hits


@dataclass(frozen=True)
class CudaReplayBackend:
    name: str
    matcher: Any
    codec: Keyer[Any, Key]
    stock: frozenset[Key]
    compile_seconds: float
    stack_mols: Callable[[Sequence[CMol]], MoleculeBatch[Mol]]
    key_mols: Callable[[MoleculeBatch[Mol]], Sequence[Key]]
    cupy: Any
    cuda_diagnostics: list[Any]

    def prepare_case(self, case: ReplayCase) -> BackendCase:
        target = target_cmol(case)
        mapped_route = encode_route(case.route, self.codec)
        return BackendCase(
            target_key=self.codec.encode(target),
            target_mol=target,
            mapped_route=mapped_route,
            stock=self.stock,
        )

    def stack(self, mols: Sequence[Mol]) -> MoleculeBatch[Mol]:
        return self.stack_mols(mols)

    def match(
        self, mols: MoleculeBatch[Mol], pairs: Sequence[Pair]
    ) -> tuple[MoleculeBatch[Mol], Sequence[Key], HitArray]:
        precursors, hits = self.matcher.match(mols, pairs)
        self.cupy.cuda.Stream.null.synchronize()
        diagnostic = self.matcher.last_diagnostics
        self.cuda_diagnostics.append(cuda_memory_snapshot(self.cupy, diagnostic))
        return precursors, self.key_mols(precursors), hits


@dataclass(frozen=True)
class MetalReplayBackend:
    name: str
    matcher: Any
    codec: Keyer[Any, Key]
    stock: frozenset[Key]
    compile_seconds: float
    stack_mols: Callable[[Sequence[CMol]], MoleculeBatch[Mol]]
    key_mols: Callable[[MoleculeBatch[Mol]], Sequence[Key]]

    def prepare_case(self, case: ReplayCase) -> BackendCase:
        target = target_cmol(case)
        mapped_route = encode_route(case.route, self.codec)
        return BackendCase(
            target_key=self.codec.encode(target),
            target_mol=target,
            mapped_route=mapped_route,
            stock=self.stock,
        )

    def stack(self, mols: Sequence[Mol]) -> MoleculeBatch[Mol]:
        return self.stack_mols(mols)

    def match(
        self, mols: MoleculeBatch[Mol], pairs: Sequence[Pair]
    ) -> tuple[MoleculeBatch[Mol], Sequence[Key], HitArray]:
        precursors, hits = self.matcher.match(mols, pairs)
        return precursors, self.key_mols(precursors), hits


def test_bench_route_replay_n1(
    benchmark: Any,
    route_replay_backend: str,
    route_replay_mode: str,
    route_replay_rule_set: str,
    size: int,
) -> None:
    skip_invalid_replay_combo(route_replay_mode, route_replay_rule_set)
    cleanup_cuda_memory()
    backend: ReplayBackend | None = None
    target: Callable[[], Any] | None = None
    try:
        corpus = load_route_replay_corpus(size)
        backend = build_route_replay_backend(
            route_replay_backend, corpus.cases, route_replay_rule_set
        )
        compile_seconds = backend.compile_seconds

        if route_replay_mode == "precursor":

            def target() -> PrecursorReplayResult:
                if backend is None:
                    raise RuntimeError("route replay backend was released")
                reset_cuda_diagnostics(backend)
                try:
                    jobs = replay_jobs()
                    if route_replay_backend in ("rdkit", "cpu") and jobs != 1:
                        return run_precursor_replay_benchmark_parallel(
                            backend,
                            corpus.cases,
                            jobs,
                        )
                    return run_precursor_replay_benchmark(
                        backend, corpus.cases, route_replay_rule_set
                    )
                finally:
                    cleanup_cuda_memory()

            result = benchmark.pedantic(
                target, rounds=rounds(), warmup_rounds=warmup_rounds(), iterations=1
            )
            benchmark.extra_info.update(
                precursor_replay_extra_info(
                    result,
                    backend=route_replay_backend,
                    rule_set=route_replay_rule_set,
                    size=size,
                    compile_seconds=compile_seconds,
                    corpus=corpus,
                    replay_workers=replay_worker_count(route_replay_backend),
                )
            )
            return

        def target() -> ReplayBenchmarkResult:
            if backend is None:
                raise RuntimeError("route replay backend was released")
            reset_cuda_diagnostics(backend)
            try:
                jobs = replay_jobs()
                if route_replay_backend in ("rdkit", "cpu") and jobs != 1:
                    return run_replay_benchmark_parallel(
                        route_replay_backend, corpus.cases, route_replay_mode, jobs
                    )
                return run_replay_benchmark(backend, corpus.cases, route_replay_mode)
            finally:
                cleanup_cuda_memory()

        result = benchmark.pedantic(
            target,
            rounds=rounds(),
            warmup_rounds=warmup_rounds(),
            iterations=1,
        )
        benchmark.extra_info.update(
            replay_extra_info(
                result,
                backend=route_replay_backend,
                mode=route_replay_mode,
                size=size,
                compile_seconds=compile_seconds,
                corpus=corpus,
                replay_workers=replay_worker_count(route_replay_backend),
            )
        )
    finally:
        target = None
        backend = None
        cleanup_cuda_memory()


@dataclass(frozen=True)
class RouteReplayCorpus:
    route_nodes_requested: int
    cases: tuple[ReplayCase, ...]
    skipped_routes: Counter[str]


@lru_cache
def load_route_replay_corpus(size: int) -> RouteReplayCorpus:
    route_nodes = load_route_nodes(size)
    cases, skipped = replay_cases(route_nodes)
    return RouteReplayCorpus(
        route_nodes_requested=len(route_nodes),
        cases=cases,
        skipped_routes=skipped,
    )


def load_route_nodes(limit: int) -> tuple[dict[str, Any], ...]:
    routes_path = PAROUTES_DIR / "processed" / "n1-routes.json.gz"
    if not routes_path.exists():
        pytest.skip(f"Missing benchmark route file: {routes_path}")

    with gzip.open(routes_path, "rt", encoding="utf-8") as handle:
        rows = json.load(handle)

    nodes: list[dict[str, Any]] = []
    for row in rows:
        target_smiles = row["smiles"]
        for child in row.get("children", []):
            nodes.append({"type": "mol", "smiles": target_smiles, "children": [child]})
            if limit > 0 and len(nodes) >= limit:
                return tuple(nodes)
    return tuple(nodes)


@lru_cache
def load_stock_smiles() -> tuple[str, ...]:
    path = PAROUTES_DIR / "processed" / "n1-stock.smi.gz"
    if not path.exists():
        pytest.skip(f"Missing benchmark stock file: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return tuple(line.strip().split()[0] for line in handle if line.strip())


@lru_cache
def stock_keys(codec: Keyer[Any, Key]) -> frozenset[Key]:
    return frozenset(codec.from_smiles(smiles) for smiles in load_stock_smiles())


def replay_cases(
    route_nodes: tuple[dict[str, Any], ...],
) -> tuple[tuple[ReplayCase, ...], Counter[str]]:
    rows = load_replay_template_rows(PAROUTES_DIR)
    cases: list[ReplayCase] = []
    skipped: Counter[str] = Counter()
    for node in route_nodes:
        try:
            cases.append(replay_case(node, rows))
        except ValueError as exc:
            skipped[str(exc)] += 1
    return tuple(cases), skipped


def load_replay_template_rows(directory: Path) -> dict[str, dict[str, str]]:
    committed_path = directory / "processed" / ROUTE_TEMPLATES_FILE
    if committed_path.exists():
        return load_route_template_rows(committed_path)

    raw_path = directory / "raw" / "uspto_raw_template_library.csv.gz"
    if raw_path.exists():
        return load_raw_template_rows(directory / "raw")

    reference_path = directory / "processed" / "n1-reference-reactions.tsv.gz"
    if not reference_path.exists():
        pytest.skip(
            f"Route replay requires processed/{ROUTE_TEMPLATES_FILE}, "
            "raw/uspto_raw_template_library.csv.gz, or "
            "processed/n1-reference-reactions.tsv.gz."
        )
    with gzip.open(reference_path, "rt", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        required = ("reaction_hash", "smiles", "rsmi", "id", "ring_breaker")
        if tuple(rows.fieldnames or ()) != required:
            raise ValueError(
                f"Unexpected reference reaction columns: {rows.fieldnames}"
            )
        return {
            row["reaction_hash"]: {
                "retro_template": row["smiles"],
                "template_hash": row["reaction_hash"],
                "classification": row["id"],
            }
            for row in rows
        }


def replay_case(
    node: dict[str, Any],
    rows: dict[str, dict[str, str]],
) -> ReplayCase:
    reaction_hash_to_rule: dict[str, int] = {}
    smarts: list[str] = []
    patterns: list[ReactionPattern] = []
    for metadata in reaction_metadata(node):
        reaction_hash = str(metadata["reaction_hash"])
        if reaction_hash in reaction_hash_to_rule:
            continue
        row = rows.get(reaction_hash)
        if row is None:
            raise ValueError("missing-template-row")
        cleaned = clean_reaction_smarts(row["retro_template"])
        try:
            pattern = parse_reaction_smarts(cleaned)
        except Exception as exc:
            raise ValueError("unsupported-template") from exc
        if pattern.lhs.atom_count > 64 or pattern.rhs.atom_count > 64:
            raise ValueError("oversize-template")
        reaction_hash_to_rule[reaction_hash] = len(smarts)
        smarts.append(cleaned)
        patterns.append(pattern)

    if not smarts:
        raise ValueError("route-has-no-reactions")

    route = route_from_aizynth_dict(
        node,
        lambda metadata: reaction_hash_to_rule[str(metadata["reaction_hash"])],
    )
    return ReplayCase(
        target_smiles=node["smiles"],
        route=route,
        expected=encode_route(route, cmol_rdkit_smiles_keyer),
        smarts=tuple(smarts),
        patterns=tuple(patterns),
    )


def reaction_metadata(node: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    if node["type"] == "reaction":
        records = [dict(node["metadata"])]
    else:
        records = []
    for child in node.get("children", []):
        records.extend(reaction_metadata(child))
    return tuple(records)


def build_route_replay_backend(
    backend: str,
    cases: tuple[ReplayCase, ...],
    rule_set: str = ROUTE_REACTION_RULE_SET,
) -> ReplayBackend:
    if backend == "rdkit":
        started = perf_counter()
        matcher = RdkitReactionMatcher(
            rdkit_reactions_from_smarts(rule_set_smarts(rule_set, cases))
        )
        return RdkitReplayBackend(
            name=backend,
            matcher=matcher,
            stock=stock_keys(rdkit_smiles_keyer),
            compile_seconds=perf_counter() - started,
        )
    if backend == "cpu":
        from vesma.cpu_matcher import CpuReactionMatcher

        started = perf_counter()
        matcher = CpuReactionMatcher(rule_set_patterns(rule_set, cases))
        return ObjectReplayBackend(
            name=backend,
            matcher=matcher,
            codec=cmol_rdkit_smiles_keyer,
            stock=stock_keys(cmol_rdkit_smiles_keyer),
            compile_seconds=perf_counter() - started,
        )
    if backend in {"cuda", "cuda-prefilter"}:
        cp = pytest.importorskip("cupy")
        if not cp.cuda.is_available():
            pytest.skip("CuPy cannot access CUDA on this machine.")
        from vesma.cuda_matcher import (
            CudaFoldedPrefilteredReactionMatcher,
            CudaReactionMatcher,
            cuda_short_ring_wl_key,
            cuda_stack_mols,
        )

        # cuda-prefilter only changes the all-pairs (precursor-mode) path; guided/topology
        # replay passes explicit pairs, which the wrapper delegates straight to the matcher.
        codec = short_ring_wl_keyer
        started = perf_counter()
        matcher: Any = CudaReactionMatcher(rule_set_patterns(rule_set, cases))
        if backend == "cuda-prefilter":
            matcher = CudaFoldedPrefilteredReactionMatcher(matcher)
        cp.cuda.Stream.null.synchronize()
        return CudaReplayBackend(
            name=backend,
            matcher=matcher,
            codec=codec,
            stock=stock_keys(codec),
            compile_seconds=perf_counter() - started,
            stack_mols=cuda_stack_mols,
            key_mols=cuda_short_ring_wl_key,
            cupy=cp,
            cuda_diagnostics=[],
        )
    if backend in {"metal", "metal-prefilter"}:
        require_metal()
        from vesma.metal_reaction_matcher import (
            MetalFoldedPrefilteredReactionMatcher,
            MetalReactionMatcher,
            metal_short_ring_wl_key,
            metal_stack_mols,
        )

        # metal-prefilter only changes the all-pairs (precursor-mode) path; guided/topology
        # replay passes explicit pairs, which the wrapper delegates straight to the matcher.
        codec = short_ring_wl_keyer
        started = perf_counter()
        matcher: Any = MetalReactionMatcher(rule_set_patterns(rule_set, cases))
        if backend == "metal-prefilter":
            matcher = MetalFoldedPrefilteredReactionMatcher(matcher)
        return MetalReplayBackend(
            name=backend,
            matcher=matcher,
            codec=codec,
            stock=stock_keys(codec),
            compile_seconds=perf_counter() - started,
            stack_mols=metal_stack_mols,
            key_mols=metal_short_ring_wl_key,
        )
    raise ValueError(f"Unsupported BENCH_ROUTE_REPLAY_BACKENDS entry: {backend}")


def all_smarts(cases: tuple[ReplayCase, ...]) -> tuple[str, ...]:
    return tuple(smarts for case in cases for smarts in case.smarts)


def all_patterns(cases: tuple[ReplayCase, ...]) -> tuple[ReactionPattern, ...]:
    return tuple(pattern for case in cases for pattern in case.patterns)


def rule_set_smarts(rule_set: str, cases: tuple[ReplayCase, ...]) -> tuple[str, ...]:
    """SMARTS templates for a rule set: the routes' own reactions or a full library.

    ``route-reactions`` reuses the guided/topology matcher (rule ids align with the reference).
    ``used14k`` / ``raw68k`` / ``uspto42k`` return whole libraries for precursor replay, which
    matches on precursor *sets* rather than rule ids — so id alignment is irrelevant here
    (used14k's ids do align, as the routes' own reactions; raw68k/uspto42k's do not).
    """
    if rule_set == ROUTE_REACTION_RULE_SET:
        return all_smarts(cases)
    if rule_set == "used14k":
        path = PAROUTES_DIR / "processed" / ROUTE_TEMPLATES_FILE
        if not path.exists():
            pytest.skip(f"Missing benchmark template file: {path}")
        return route_recovery_rule_index(path).smarts
    if rule_set == "raw68k":
        path = PAROUTES_DIR / "processed" / RAW68K_TEMPLATES_FILE
        if not path.exists():
            pytest.skip(f"Missing benchmark template file: {path}")
        return raw68k_rule_index(
            path,
            PAROUTES_DIR / "used14k-reactions.tsv.gz",
        ).smarts
    if rule_set == "uspto42k":
        return uspto42k_rule_index().smarts
    if rule_set == "curated":
        return curated_smarts()
    raise ValueError(f"Unsupported replay rule set: {rule_set}")


def rule_set_patterns(
    rule_set: str, cases: tuple[ReplayCase, ...]
) -> Sequence[ReactionPattern]:
    if rule_set == ROUTE_REACTION_RULE_SET:
        return all_patterns(cases)
    return reaction_patterns_from_smarts(rule_set_smarts(rule_set, cases), cache=True)


def case_rule_offset(cases: tuple[ReplayCase, ...], case_index: int) -> int:
    return sum(len(case.patterns) for case in cases[:case_index])


def offset_route_rules(route: Route[Key], offset: int) -> Route[Key]:
    rule = None if route.rule is None else route.rule + offset
    return Route(
        product=route.product,
        rule=rule,
        precursors=tuple(
            offset_route_rules(child, offset) for child in route.precursors
        ),
    )


def run_replay_benchmark(
    backend: ReplayBackend,
    cases: tuple[ReplayCase, ...],
    mode: str,
) -> ReplayBenchmarkResult:
    diagnostics: Counter[str] = Counter()
    strict_count = 0
    loose_count = 0
    library_count = 0
    any_solved_count = 0
    same_leaves_different_intermediates = 0
    same_reactions_different_leaves = 0
    generated_route_count = 0
    strict_recoveries: list[int] = []
    loose_recoveries: list[int] = []
    library_recoveries: list[int] = []
    any_solved_recoveries: list[int] = []

    for index, case in enumerate(cases):
        offset = case_rule_offset(cases, index)
        backend_case = backend.prepare_case(case)
        mapped_route = offset_route_rules(backend_case.mapped_route, offset)
        backend_case = BackendCase(
            target_key=backend_case.target_key,
            target_mol=backend_case.target_mol,
            mapped_route=mapped_route,
            stock=backend_case.stock,
        )
        result = run_replay_case(backend, backend_case, mode)
        diagnostics.update(result.diagnostics)
        routes = result.routes
        generated_route_count += len(routes)

        strict = strict_route_replayed(mapped_route, routes)
        loose = loose_route_replayed(mapped_route, routes)
        library = library_route_replayed(mapped_route, routes)
        any_solved = bool(routes)

        strict_count += int(strict)
        loose_count += int(loose)
        library_count += int(library)
        strict_recoveries.append(int(strict))
        loose_recoveries.append(int(loose))
        library_recoveries.append(int(library))
        any_solved_recoveries.append(int(any_solved))
        any_solved_count += int(any_solved)
        same_leaves_different_intermediates += int(loose and not strict)
        same_reactions_different_leaves += int(library and not loose)

        if not any_solved:
            diagnostics["not-solved"] += 1
        elif not strict:
            diagnostics["strict-reference-route-not-generated"] += 1

    return ReplayBenchmarkResult(
        case_count=len(cases),
        route_count=len(cases),
        rule_count=sum(len(case.patterns) for case in cases),
        reaction_count=sum(reaction_count(case.route) for case in cases),
        strict_count=strict_count,
        loose_count=loose_count,
        library_count=library_count,
        strict_recoveries=tuple(strict_recoveries),
        loose_recoveries=tuple(loose_recoveries),
        library_recoveries=tuple(library_recoveries),
        any_solved_recoveries=tuple(any_solved_recoveries),
        any_solved_count=any_solved_count,
        same_leaves_different_intermediates=same_leaves_different_intermediates,
        same_reactions_different_leaves=same_reactions_different_leaves,
        generated_route_count=generated_route_count,
        diagnostics=diagnostics,
        cuda_memory=cuda_memory_extra(cuda_diagnostics(backend)),
    )


def run_replay_benchmark_parallel(
    backend_name: str,
    cases: tuple[ReplayCase, ...],
    mode: str,
    n_jobs: int,
) -> ReplayBenchmarkResult:
    """Replay `cases` across worker processes, one chunk per task, then merge.

    Each chunk builds its own matcher (compile is cheap) and runs the sequential
    `run_replay_benchmark` verbatim, so per-case rule offsetting stays chunk-relative and the
    merged result is bit-identical to the sequential run. Processes (loky) are required: RDKit
    replay is GIL-bound, so threads give no speedup (and RDKit is not thread-safe to share).
    """
    chunks = parallel_chunks(cases, n_jobs=n_jobs)
    if len(chunks) <= 1:  # size 1 / n_jobs 1 — not worth spawning a pool
        return _replay_chunk(backend_name, cases, mode)
    parts = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
        joblib.delayed(_replay_chunk)(backend_name, chunk, mode) for chunk in chunks
    )
    return merge_replay_results(parts)


def _replay_chunk(
    backend_name: str,
    cases: tuple[ReplayCase, ...],
    mode: str,
) -> ReplayBenchmarkResult:
    from rdkit import RDLogger

    # Route-level parallelism owns the cores, so pin the inner match to one thread (avoid
    # oversubscription). Silence RDKit's per-template warnings so N workers don't flood stderr.
    os.environ["VESMA_N_JOBS"] = "1"
    RDLogger.DisableLog("rdApp.*")
    backend = build_route_replay_backend(backend_name, cases)
    return run_replay_benchmark(backend, cases, mode)


def merge_replay_results(
    parts: Sequence[ReplayBenchmarkResult],
) -> ReplayBenchmarkResult:
    """Combine per-chunk results in order: sum scalars/Counters, concatenate recovery flags.

    joblib preserves task order and `parallel_chunks` yields contiguous ordered slices, so the
    concatenated recovery tuples reproduce the sequential per-route ordering.
    """
    diagnostics: Counter[str] = Counter()
    for part in parts:
        diagnostics.update(part.diagnostics)
    return ReplayBenchmarkResult(
        case_count=sum(p.case_count for p in parts),
        route_count=sum(p.route_count for p in parts),
        rule_count=sum(p.rule_count for p in parts),
        reaction_count=sum(p.reaction_count for p in parts),
        strict_count=sum(p.strict_count for p in parts),
        loose_count=sum(p.loose_count for p in parts),
        library_count=sum(p.library_count for p in parts),
        strict_recoveries=tuple(x for p in parts for x in p.strict_recoveries),
        loose_recoveries=tuple(x for p in parts for x in p.loose_recoveries),
        library_recoveries=tuple(x for p in parts for x in p.library_recoveries),
        any_solved_recoveries=tuple(x for p in parts for x in p.any_solved_recoveries),
        any_solved_count=sum(p.any_solved_count for p in parts),
        same_leaves_different_intermediates=sum(
            p.same_leaves_different_intermediates for p in parts
        ),
        same_reactions_different_leaves=sum(
            p.same_reactions_different_leaves for p in parts
        ),
        generated_route_count=sum(p.generated_route_count for p in parts),
        diagnostics=diagnostics,
        cuda_memory={},
    )


def run_replay_case(
    backend: ReplayBackend,
    case: BackendCase,
    mode: str,
) -> ReplayRunResult:
    if mode == "guided":
        return run_guided_replay_case(backend, case)
    if mode == "topology":
        diagnostics: Counter[str] = Counter()
        routes = topology_replay_routes(
            backend,
            case.target_mol,
            case.target_key,
            case.mapped_route,
            case.stock,
            diagnostics,
            {},
        )
        return ReplayRunResult(frozenset(routes), diagnostics)
    raise ValueError(f"Unsupported BENCH_ROUTE_REPLAY_MODES entry: {mode}")


def run_guided_replay_case(
    backend: ReplayBackend,
    case: BackendCase,
) -> ReplayRunResult:
    diagnostics: Counter[str] = Counter()
    reference_stock = leaves(case.mapped_route)
    expansions = replay_search(
        case.mapped_route,
        [case.target_key],
        backend.stack([case.target_mol]),
        reference_stock,
        reaction_depth(case.mapped_route),
        Expansions(NumpyExpansionStore(), DictInterner()),
        match=backend.match,
    )
    routes = frozenset(iter_solved_routes(case.target_key, expansions, reference_stock))
    diagnostics["generated_routes"] += len(routes)
    return ReplayRunResult(routes, diagnostics)


# --------------------------------------------------------------- precursor-based replay
#
# Applies every library rule to the frontier and keeps only applications whose precursor set
# reproduces the reference route's precursors at that product. Rule ids never have to align, so
# this is the only mode that runs against a foreign library (uspto42k). Termination is at the
# reference-route leaves, so a solved route is a full rule-agnostic precursor recovery; a
# separate per-disconnection count gives partial credit for routes that are not fully recovered.


@dataclass(frozen=True)
class PrecursorRouteReplayRecord:
    route_index: int
    target_smiles: str
    reaction_count: int
    any_solved: int
    covered_disconnections: int
    total_disconnections: int
    generated_route_count: int
    diagnostics: Counter[str]


@dataclass(frozen=True)
class PrecursorReplayResult:
    case_count: int
    reaction_count: int
    any_solved_count: int
    any_solved_recoveries: tuple[int, ...]
    covered_disconnections: int
    total_disconnections: int
    generated_route_count: int
    diagnostics: Counter[str]
    route_records: tuple[PrecursorRouteReplayRecord, ...]
    cuda_memory: dict[str, int]


def run_precursor_replay_benchmark(
    backend: ReplayBackend,
    cases: tuple[ReplayCase, ...],
    rule_set: str,
) -> PrecursorReplayResult:
    """Replay every case against a full rule library, pruned to reference precursor sets.

    One shared library backend, no per-case rule offset (rule ids are ignored): each case seeds
    its target, applies all rules, and keeps only the reference disconnections. A solved route
    means every disconnection on some path was reproduced by *some* library rule.
    """
    records = tuple(
        run_precursor_replay_case(backend, index, case)
        for index, case in enumerate(cases)
    )
    return precursor_replay_result(
        records,
        cuda_memory=cuda_memory_extra(cuda_diagnostics(backend)),
    )


def run_precursor_replay_case(
    backend: ReplayBackend,
    route_index: int,
    case: ReplayCase,
) -> PrecursorRouteReplayRecord:
    diagnostics: Counter[str] = Counter()

    # `precursor_replay_search` takes a `Match`, which answers an `Expansion`; the backend still
    # speaks the raw three-tuple, so it is lifted here rather than at every call inside.
    match = expansion(lambda mols: backend.match(mols, None))

    backend_case = backend.prepare_case(case)
    route = backend_case.mapped_route
    allowed = route_precursor_sets(route)
    reference_stock = leaves(route)
    expansions = precursor_replay_search(
        allowed,
        [backend_case.target_key],
        backend.stack([backend_case.target_mol]),
        reference_stock,
        reaction_depth(route),
        Expansions(NumpyExpansionStore(), DictInterner()),
        match=match,
    )

    solved = bool(
        solved_targets([backend_case.target_key], expansions, reference_stock)
    )
    if not solved:
        diagnostics["not-solved"] += 1

    covered, total = disconnection_coverage(allowed, expansions)
    diagnostics["matched_disconnections"] += covered

    generated = sum(
        1
        for _ in iter_solved_routes(
            backend_case.target_key,
            expansions,
            reference_stock,
            cap=PRECURSOR_ROUTE_CAP,
        )
    )
    diagnostics["generated_routes"] += generated

    return PrecursorRouteReplayRecord(
        route_index=route_index,
        target_smiles=case.target_smiles,
        reaction_count=reaction_count(case.route),
        any_solved=int(solved),
        covered_disconnections=covered,
        total_disconnections=total,
        generated_route_count=generated,
        diagnostics=diagnostics,
    )


def run_precursor_replay_benchmark_parallel(
    backend: ReplayBackend,
    cases: tuple[ReplayCase, ...],
    n_jobs: int,
) -> PrecursorReplayResult:
    indexed_cases = tuple(enumerate(cases))
    chunks = parallel_chunks(indexed_cases, n_jobs=n_jobs, chunks_per_worker=1)
    if not chunks:
        return precursor_replay_result((), cuda_memory={})
    if len(chunks) <= 1:
        return _precursor_replay_chunk(backend, chunks[0])
    parts = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
        joblib.delayed(_precursor_replay_chunk)(backend, chunk) for chunk in chunks
    )
    return merge_precursor_replay_results(parts)


def _precursor_replay_chunk(
    backend: ReplayBackend,
    indexed_cases: tuple[tuple[int, ReplayCase], ...],
) -> PrecursorReplayResult:
    from rdkit import RDLogger

    os.environ["VESMA_N_JOBS"] = "1"
    RDLogger.DisableLog("rdApp.*")
    records = tuple(
        run_precursor_replay_case(backend, route_index, case)
        for route_index, case in indexed_cases
    )
    return precursor_replay_result(records, cuda_memory={})


def merge_precursor_replay_results(
    parts: Sequence[PrecursorReplayResult],
) -> PrecursorReplayResult:
    records = tuple(record for part in parts for record in part.route_records)
    return precursor_replay_result(records, cuda_memory={})


def precursor_replay_result(
    records: Sequence[PrecursorRouteReplayRecord],
    *,
    cuda_memory: dict[str, int],
) -> PrecursorReplayResult:
    ordered_records = tuple(sorted(records, key=lambda record: record.route_index))
    diagnostics: Counter[str] = Counter()
    for record in ordered_records:
        diagnostics.update(record.diagnostics)
    return PrecursorReplayResult(
        case_count=len(ordered_records),
        reaction_count=sum(record.reaction_count for record in ordered_records),
        any_solved_count=sum(record.any_solved for record in ordered_records),
        any_solved_recoveries=tuple(record.any_solved for record in ordered_records),
        covered_disconnections=sum(
            record.covered_disconnections for record in ordered_records
        ),
        total_disconnections=sum(
            record.total_disconnections for record in ordered_records
        ),
        generated_route_count=sum(
            record.generated_route_count for record in ordered_records
        ),
        diagnostics=diagnostics,
        route_records=ordered_records,
        cuda_memory=cuda_memory,
    )


def disconnection_coverage(
    allowed_sets: dict[Key, frozenset[frozenset[Key]]],
    expansions: Expansions,
) -> tuple[int, int]:
    """Reference reactions reproduced in the pruned graph, and the total reference reactions.

    ``allowed_sets`` has one entry per non-leaf reference product; a product carries a matched
    disconnection when it was reached and kept at least one application. Because a product is
    only reached once its ancestors matched, this credits the matched prefix of each route
    rather than each disconnection in isolation.
    """
    total = len(allowed_sets)
    covered = 0
    for product in allowed_sets:
        sid = int(expansions.lookup([product])[0])
        if sid == MISSING_SID:
            continue
        if expansions.steps(sid):
            covered += 1
    return covered, total


def topology_replay_routes(
    backend: ReplayBackend,
    product_mol: Mol,
    product_key: Key,
    route: Route[Key],
    stock: frozenset[Key],
    diagnostics: Counter[str],
    cache: dict[tuple[Key, object], tuple[Route[Key], ...]],
) -> tuple[Route[Key], ...]:
    return topology_replay_batch_routes(
        backend,
        backend.stack([product_mol]),
        product_key,
        route,
        stock,
        diagnostics,
        cache,
    )


def topology_replay_batch_routes(
    backend: ReplayBackend,
    product_mols: MoleculeBatch[Mol],
    product_key: Key,
    route: Route[Key],
    stock: frozenset[Key],
    diagnostics: Counter[str],
    cache: dict[tuple[Key, object], tuple[Route[Key], ...]],
) -> tuple[Route[Key], ...]:
    cache_key = (product_key, reference_route_key(route))
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    if route.rule is None:
        if product_key in stock:
            routes = (Route(product_key, None),)
            cache[cache_key] = routes
            return routes
        diagnostics["leaf-not-in-stock"] += 1
        cache[cache_key] = ()
        return ()

    precursor_mols, precursor_keys, hits = backend.match(
        product_mols,
        [(0, route.rule)],
    )
    app_indices = application_indices(hits)
    if not app_indices:
        diagnostics["no-applications"] += 1
        cache[cache_key] = ()
        return ()

    solved_routes: list[Route[Key]] = []
    for indices in app_indices:
        diagnostics["applications"] += 1
        unique_indices = unique_precursor_indices(indices, precursor_keys)
        diagnostics["duplicate_precursors_collapsed"] += len(indices) - len(
            unique_indices
        )

        child_options = relaxed_topology_children(
            backend,
            precursor_mols,
            precursor_keys,
            unique_indices,
            route.precursors,
            stock,
            diagnostics,
            cache,
        )
        for children in child_options:
            solved_routes.append(Route(product_key, route.rule, children))

    if not solved_routes:
        diagnostics["no-child-assignment"] += 1
    routes = tuple(solved_routes)
    cache[cache_key] = routes
    return routes


def application_indices(hits: HitArray) -> tuple[tuple[int, ...], ...]:
    """Group each application's precursor batch rows, in hit order."""
    grouped: dict[int, list[int]] = {}
    for hit in np.asarray(hits, dtype=np.int64):
        mol_idx, _rule_id, app_idx, precursor_index = hit.tolist()
        if mol_idx != 0:
            raise ValueError("Topology replay expects one product molecule per match.")
        grouped.setdefault(int(app_idx), []).append(int(precursor_index))
    return tuple(tuple(indices) for _, indices in sorted(grouped.items()))


def unique_precursor_indices(
    indices: tuple[int, ...],
    precursor_keys: Sequence[Key],
) -> tuple[int, ...]:
    seen: set[Key] = set()
    unique: list[int] = []
    for index in indices:
        key = precursor_keys[index]
        if key in seen:
            continue
        seen.add(key)
        unique.append(index)
    return tuple(unique)


def relaxed_topology_children(
    backend: ReplayBackend,
    precursor_mols: MoleculeBatch[Mol],
    precursor_keys: Sequence[Key],
    indices: tuple[int, ...],
    children: tuple[Route[Key], ...],
    stock: frozenset[Key],
    diagnostics: Counter[str],
    cache: dict[tuple[Key, object], tuple[Route[Key], ...]],
) -> tuple[tuple[Route[Key], ...], ...]:
    child_options = tuple(
        relaxed_precursor_options(
            backend,
            precursor_mols,
            precursor_keys,
            precursor_index,
            children,
            stock,
            diagnostics,
            cache,
        )
        for precursor_index in indices
    )
    if any(not options for options in child_options):
        return ()
    return tuple(tuple(options) for options in cartesian_product(child_options))


def relaxed_precursor_options(
    backend: ReplayBackend,
    precursor_mols: MoleculeBatch[Mol],
    precursor_keys: Sequence[Key],
    precursor_index: int,
    children: tuple[Route[Key], ...],
    stock: frozenset[Key],
    diagnostics: Counter[str],
    cache: dict[tuple[Key, object], tuple[Route[Key], ...]],
) -> tuple[Route[Key], ...]:
    precursor_key = precursor_keys[precursor_index]
    options: list[Route[Key]] = []
    if precursor_key in stock:
        diagnostics["leaf-in-stock"] += 1
        options.append(Route(precursor_key, None))
    else:
        diagnostics["leaf-not-in-stock"] += 1

    for child in children:
        if child.rule is None:
            continue
        options.extend(
            topology_replay_batch_routes(
                backend,
                one_molecule_batch(precursor_mols, precursor_index),
                precursor_key,
                child,
                stock,
                diagnostics,
                cache,
            )
        )
    return tuple(options)


def one_molecule_batch(
    mols: MoleculeBatch[Mol],
    index: int,
) -> MoleculeBatch[Mol]:
    return mols[np.array([index], dtype=np.int64)]


def cartesian_product(
    groups: tuple[tuple[Route[Key], ...], ...],
) -> tuple[tuple[Route[Key], ...], ...]:
    if not groups:
        return ((),)
    first = groups[0]
    rest = cartesian_product(groups[1:])
    return tuple((item,) + suffix for item in first for suffix in rest)


def reference_route_key(
    route: Route[Key],
) -> tuple[Key, int | None, tuple[object, ...]]:
    return (
        route.product,
        route.rule,
        tuple(reference_route_key(child) for child in route.precursors),
    )


def replay_extra_info(
    result: ReplayBenchmarkResult,
    *,
    backend: str,
    mode: str,
    size: int,
    compile_seconds: float,
    corpus: RouteReplayCorpus,
    replay_workers: int = 1,
) -> dict[str, Any]:
    return {
        "experiment": "route-replay",
        "backend": backend,
        "mode": mode,
        "size": size,
        # Route-level worker processes used for this run (1 = sequential). CPU model and
        # physical/usable core counts live in machine_info.cpu_detail.
        "replay_jobs": replay_workers,
        "route_set": "paroutes-n1",
        "template_source": ROUTE_TEMPLATES_FILE,
        "stock_policy": (
            "reference-route-leaves" if mode == "guided" else "paroutes-n1-stock"
        ),
        "routes_requested": corpus.route_nodes_requested,
        "routes_prepared": len(corpus.cases),
        "skipped_routes": dict(corpus.skipped_routes),
        "routes": result.route_count,
        "rules": result.rule_count,
        "reactions": result.reaction_count,
        "generated_routes": result.generated_route_count,
        "any_solved_routes": result.any_solved_count,
        "strict_reference_recoveries": result.strict_count,
        "loose_reference_recoveries": result.loose_count,
        "library_reference_recoveries": result.library_count,
        # Per-route 0/1 flags, indexed by prepared-corpus position (see route_set /
        # template_source), so diverging routes can be located without a custom wrapper.
        "strict_reference_recoveries_list": list(result.strict_recoveries),
        "loose_reference_recoveries_list": list(result.loose_recoveries),
        "library_reference_recoveries_list": list(result.library_recoveries),
        "any_solved_routes_list": list(result.any_solved_recoveries),
        "not_solved_route_indices": missing_indices(result.any_solved_recoveries),
        "strict_reference_missing_indices": missing_indices(result.strict_recoveries),
        "loose_reference_missing_indices": missing_indices(result.loose_recoveries),
        "library_reference_missing_indices": missing_indices(result.library_recoveries),
        "same_leaves_different_intermediates": (
            result.same_leaves_different_intermediates
        ),
        "same_reactions_different_leaves": result.same_reactions_different_leaves,
        "any_solved_rate": rate(result.any_solved_count, result.case_count),
        "strict_reference_recovery_rate": rate(result.strict_count, result.case_count),
        "loose_reference_recovery_rate": rate(result.loose_count, result.case_count),
        "library_reference_recovery_rate": rate(
            result.library_count, result.case_count
        ),
        "compile_seconds": compile_seconds,
        "diagnostics": dict(result.diagnostics),
        "cuda_memory": result.cuda_memory,
        "source_artifacts": source_artifacts(),
    }


def precursor_replay_extra_info(
    result: PrecursorReplayResult,
    *,
    backend: str,
    rule_set: str,
    size: int,
    compile_seconds: float,
    corpus: RouteReplayCorpus,
    replay_workers: int = 1,
) -> dict[str, Any]:
    """Serialize a precursor-replay run.

    Rule ids do not align with the reference for a foreign library, so the strict/loose/library
    comparators are omitted (``reaction_comparators_feasible`` is False). ``any_solved`` here
    means full rule-agnostic precursor recovery; ``disconnection_coverage_rate`` is the
    partial-credit fraction of reference reactions reproduced by some library rule.
    """
    return {
        "experiment": "route-replay",
        "backend": backend,
        "mode": "precursor",
        "rule_set": rule_set,
        "reaction_comparators_feasible": False,
        "size": size,
        "replay_jobs": replay_workers,
        "route_set": "paroutes-n1",
        "template_source": RULE_SET_TEMPLATE_SOURCE.get(rule_set, rule_set),
        "stock_policy": "reference-route-leaves",
        "routes_requested": corpus.route_nodes_requested,
        "routes_prepared": len(corpus.cases),
        "skipped_routes": dict(corpus.skipped_routes),
        "routes": result.case_count,
        "routes_detail": [
            precursor_route_record(record) for record in result.route_records
        ],
        "reactions": result.reaction_count,
        "generated_routes": result.generated_route_count,
        "any_solved_routes": result.any_solved_count,
        "any_solved_routes_list": list(result.any_solved_recoveries),
        "not_solved_route_indices": missing_indices(result.any_solved_recoveries),
        "any_solved_rate": rate(result.any_solved_count, result.case_count),
        # any_solved is a full precursor recovery here (leaves are forced); alias for clarity.
        "precursor_recovery_rate": rate(result.any_solved_count, result.case_count),
        "matched_disconnections": result.covered_disconnections,
        "total_disconnections": result.total_disconnections,
        "disconnection_coverage_rate": rate(
            result.covered_disconnections, result.total_disconnections
        ),
        "compile_seconds": compile_seconds,
        "diagnostics": dict(result.diagnostics),
        "cuda_memory": result.cuda_memory,
        "source_artifacts": source_artifacts(),
    }


def precursor_route_record(record: PrecursorRouteReplayRecord) -> dict[str, Any]:
    return {
        "route_index": record.route_index,
        "target_smiles": record.target_smiles,
        "reaction_count": record.reaction_count,
        "any_solved": record.any_solved,
        "covered_disconnections": record.covered_disconnections,
        "total_disconnections": record.total_disconnections,
        "generated_route_count": record.generated_route_count,
        "diagnostics": dict(record.diagnostics),
    }


def missing_indices(flags: tuple[int, ...]) -> list[int]:
    return [index for index, flag in enumerate(flags) if not flag]


def reset_cuda_diagnostics(backend: ReplayBackend) -> None:
    diagnostics = getattr(backend, "cuda_diagnostics", None)
    if diagnostics is not None:
        diagnostics.clear()


def cleanup_cuda_memory() -> None:
    gc.collect()
    cupy = sys.modules.get("cupy")
    if cupy is None:
        return
    cupy.cuda.Stream.null.synchronize()
    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()


def cuda_diagnostics(backend: ReplayBackend) -> tuple[Any, ...]:
    diagnostics = getattr(backend, "cuda_diagnostics", None)
    if diagnostics is None:
        return ()
    return tuple(diagnostics)


def cuda_memory_snapshot(cupy: Any, diagnostic: Any | None) -> dict[str, int]:
    free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
    pool = cupy.get_default_memory_pool()
    return {
        "match_calls": 1,
        "peak_driver_used_bytes": int(total_bytes - free_bytes),
        "peak_cupy_pool_used_bytes": int(pool.used_bytes()),
        "peak_cupy_pool_total_bytes": int(pool.total_bytes()),
        "peak_process_rss_bytes": process_rss_bytes(),
        "resident_rule_bytes": (
            0 if diagnostic is None else int(diagnostic.resident_rule_bytes)
        ),
        "peak_estimated_workspace_bytes": (
            0 if diagnostic is None else int(diagnostic.peak_estimated_workspace_bytes)
        ),
        "usable_device_bytes_min": (
            int(free_bytes)
            if diagnostic is None
            else int(diagnostic.usable_device_bytes)
        ),
    }


def process_rss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


def cuda_memory_extra(diagnostics: Sequence[dict[str, int]]) -> dict[str, int]:
    if not diagnostics:
        return {}
    return {
        "match_calls": sum(d["match_calls"] for d in diagnostics),
        "peak_driver_used_bytes": max(d["peak_driver_used_bytes"] for d in diagnostics),
        "peak_cupy_pool_used_bytes": max(
            d["peak_cupy_pool_used_bytes"] for d in diagnostics
        ),
        "peak_cupy_pool_total_bytes": max(
            d["peak_cupy_pool_total_bytes"] for d in diagnostics
        ),
        "peak_process_rss_bytes": max(d["peak_process_rss_bytes"] for d in diagnostics),
        "resident_rule_bytes": max(d["resident_rule_bytes"] for d in diagnostics),
        "peak_estimated_workspace_bytes": max(
            d["peak_estimated_workspace_bytes"] for d in diagnostics
        ),
        "usable_device_bytes_min": min(
            d["usable_device_bytes_min"] for d in diagnostics
        ),
    }


def rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def source_artifacts() -> dict[str, Any]:
    return {
        "routes": source_artifact(PAROUTES_DIR / "processed" / "n1-routes.json.gz"),
        "route_templates": source_artifact(
            PAROUTES_DIR / "processed" / ROUTE_TEMPLATES_FILE
        ),
        "reference_reactions": source_artifact(
            PAROUTES_DIR / "processed" / "n1-reference-reactions.tsv.gz"
        ),
        "stock": source_artifact(PAROUTES_DIR / "processed" / "n1-stock.smi.gz"),
    }


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "size" in metafunc.fixturenames:
        metafunc.parametrize("size", scaling_sizes())
    if "route_replay_backend" in metafunc.fixturenames:
        metafunc.parametrize("route_replay_backend", selected_backends())
    if "route_replay_mode" in metafunc.fixturenames:
        metafunc.parametrize("route_replay_mode", selected_modes())
    if "route_replay_rule_set" in metafunc.fixturenames:
        metafunc.parametrize("route_replay_rule_set", selected_rule_sets())


def scaling_sizes() -> tuple[int, ...]:
    raw = os.environ.get("BENCH_ROUTE_REPLAY_SIZES")
    if not raw:
        return DEFAULT_SIZES
    return tuple(int(value) for value in raw.replace(",", " ").split() if value.strip())


def selected_backends() -> tuple[str, ...]:
    raw = os.environ.get("BENCH_ROUTE_REPLAY_BACKENDS")
    values = (
        DEFAULT_BACKENDS
        if not raw
        else tuple(value.strip() for value in raw.replace(",", " ").split())
    )
    supported = {"rdkit", "cpu", "cuda", "cuda-prefilter", "metal", "metal-prefilter"}
    unsupported = [value for value in values if value not in supported]
    if unsupported:
        raise ValueError(
            f"BENCH_ROUTE_REPLAY_BACKENDS has unsupported entries: {unsupported}"
        )
    return tuple(value for value in values if value)


def selected_modes() -> tuple[str, ...]:
    raw = os.environ.get("BENCH_ROUTE_REPLAY_MODES")
    values = (
        DEFAULT_MODES
        if not raw
        else tuple(value.strip() for value in raw.replace(",", " ").split())
    )
    supported = {"guided", "topology", "precursor"}
    unsupported = [value for value in values if value not in supported]
    if unsupported:
        raise ValueError(
            f"BENCH_ROUTE_REPLAY_MODES has unsupported entries: {unsupported}"
        )
    return tuple(value for value in values if value)


def selected_rule_sets() -> tuple[str, ...]:
    raw = os.environ.get("BENCH_ROUTE_REPLAY_RULE_SETS")
    values = (
        DEFAULT_RULE_SETS
        if not raw
        else tuple(value.strip() for value in raw.replace(",", " ").split())
    )
    unsupported = [value for value in values if value not in SUPPORTED_RULE_SETS]
    if unsupported:
        raise ValueError(
            f"BENCH_ROUTE_REPLAY_RULE_SETS has unsupported entries: {unsupported}"
        )
    return tuple(value for value in values if value)


def skip_invalid_replay_combo(mode: str, rule_set: str) -> None:
    """Only precursor replay uses a rule set; guided/topology always use route reactions."""
    if mode in ("guided", "topology") and rule_set != ROUTE_REACTION_RULE_SET:
        pytest.skip(f"{mode} replay uses route reactions, not rule set {rule_set!r}")
    if mode == "precursor" and rule_set == ROUTE_REACTION_RULE_SET:
        pytest.skip("precursor replay needs a full rule library (used14k or uspto42k)")


def rounds() -> int:
    value = int(os.environ.get("BENCH_ROUNDS", "5"))
    if value <= 0:
        raise ValueError("BENCH_ROUNDS must be positive.")
    return value


def warmup_rounds() -> int:
    """Warm-up rounds for pedantic benchmarking (BENCH_WARMUP_ROUNDS, default 1).

    ``benchmark.pedantic`` ignores pytest-benchmark's global ``--benchmark-warmup`` CLI flag
    and takes its own ``warmup_rounds`` argument, so warm-up is controlled here rather than on
    the command line. Set ``BENCH_WARMUP_ROUNDS=0`` for a single measured round with no warm-up:
    precursor replay is minutes-long, so a warm-up round just doubles wall time for no benefit.
    """
    value = int(os.environ.get("BENCH_WARMUP_ROUNDS", "1"))
    if value < 0:
        raise ValueError("BENCH_WARMUP_ROUNDS must be non-negative.")
    return value


def replay_jobs() -> int:
    """Route-level worker count for rdkit/cpu replay (joblib n_jobs; -1 = all cores).

    Separate from `VESMA_N_JOBS` (which controls inner-match parallelism) so the two do not
    oversubscribe. Defaults to -1: parallel on. Set `BENCH_ROUTE_REPLAY_JOBS=1` for sequential.
    """
    raw = os.environ.get("BENCH_ROUTE_REPLAY_JOBS")
    if raw is None or not raw.strip():
        return -1
    return validate_n_jobs(int(raw))


def replay_worker_count(backend: str) -> int:
    """Effective worker processes actually used, for provenance reporting."""
    jobs = replay_jobs()
    if backend not in ("rdkit", "cpu") or jobs == 1:
        return 1
    return effective_n_jobs(jobs)


def target_cmol(case: ReplayCase) -> CMol:
    rd_mol = Chem.MolFromSmiles(case.target_smiles)
    if rd_mol is None:
        raise ValueError(f"Invalid target SMILES: {case.target_smiles}")
    return cmol_from_mol(rd_mol)


def require_metal() -> None:
    if sys.platform != "darwin":
        pytest.skip("Metal backend is only available on macOS.")
    try:
        from vesma.metal_matcher import GpuUnavailableError, probe_metal
    except ImportError as exc:
        pytest.skip(f"Metal backend unavailable: {exc}")
    try:
        probe_metal()
    except (GpuUnavailableError, RuntimeError) as exc:
        pytest.skip(f"Metal backend not available: {exc}")

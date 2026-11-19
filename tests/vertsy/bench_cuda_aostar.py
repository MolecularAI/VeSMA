"""CUDA AO* experiment over a seeded, stratified PaRoutes n1 target sample.

The reaction matcher, its pattern fingerprints, and the stock circular fingerprints are built
once and retained in VRAM across targets. Open leaves wait in pinned host memory; each selected
batch returns to CUDA for matching and heuristic scoring. Each target gets an independent search
and a fresh 0.001 -> 1.0 anytime schedule.

Run with ``make bench-aostar-cuda``.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar

import numpy as np
import pytest

from tests.datadirs import external_data_root
from tests.vertsy.paper_provenance import source_artifact
from tests.vertsy.paroutes_reference import (
    load_route_template_rows,
    route_recovery_rule_index_from_rows,
)
from vertsy.aostar import (
    anytime_step,
    aostar,
    bottleneck_fold,
    committed_application_leaves,
)
from vertsy.cuda_similarity import (
    cuda_stock_similarity_from_fingerprints,
)
from vertsy.expansions import Expansions
from vertsy.host_similarity import NearestStock
from vertsy.interner import DictInterner
from vertsy.interop import aizynth_depth
from vertsy.routes import reaction_count
from vertsy.search import expansion
from vertsy.store.numpy_store import NumpyExpansionStore
from vertsy.template_lint import lint
from vesma import cmol_from_smiles, gmol_from_cmol, reaction_patterns_from_smarts
from vesma.circularfp import (
    CircularFingerprintConfig,
    circular_fingerprint_words,
)
from vesma.cuda_matcher import (
    CudaFoldedPrefilteredReactionMatcher,
    CudaReactionMatcher,
    cuda_concat_host_mols,
    cuda_stack_mols,
)
from vesma.keyer import short_ring_wl_keyer
from vesma.loading import cached_function
from vesma.parallel import parallel_map

pytestmark = [
    pytest.mark.perf,
    pytest.mark.gpu,
    pytest.mark.cuda,
    pytest.mark.external_data,
    pytest.mark.filterwarnings(
        "ignore:Setting the shape on a NumPy array has been deprecated.*:DeprecationWarning:joblib.numpy_pickle"
    ),
]

PAROUTES = external_data_root() / "external" / "paroutes" / "processed"
STOCK_FILE = PAROUTES / "n1-stock.smi.gz"
ROUTES_FILE = PAROUTES / "n1-routes.json.gz"
TEMPLATE_FILES = {
    "used14k": PAROUTES / "used14k-reactions.tsv.gz",
    "raw68k": PAROUTES / "raw68k-templates.tsv.gz",
}

TARGETS_BY_DEPTH = {5: 34, 6: 33, 7: 33}
TARGET_SEED = 0
GREEDY_STEP = 0.001
RELAXED_STEP = 1.0
HEURISTIC_WEIGHT = 1.0
FINGERPRINT = CircularFingerprintConfig(n_bits=2048, radius=2)
STOCK_CACHE_VERSION = 1

R = TypeVar("R")


@dataclass(frozen=True)
class TargetResult:
    target: str
    reference_depth: int
    solved: bool
    seconds: float
    width: int
    pass_limit: int
    passes: int
    molecules_expanded: int
    molecules_seen: int
    routes: int
    best_reaction_count: int | None
    phase_seconds: dict[str, float]
    matcher_seconds: dict[str, float]
    dense_pairs: int
    candidate_pairs: int
    positive_pairs: int
    assignments: int
    raw_precursors: int
    unique_precursors: int
    precursor_atom_overflow_components: int
    precursor_atom_overflow_assignments: int


@dataclass(frozen=True)
class Setup:
    stock: frozenset[tuple[int, int]]
    heuristic: NearestStock
    matcher: CudaFoldedPrefilteredReactionMatcher
    rule_set: str
    template_file: Path
    template_row_count: int
    template_count_input: int
    template_count_rejected: int
    rejected_template_indices: tuple[int, ...]
    template_load_seconds: float
    template_parse_seconds: float
    template_compile_seconds: float
    stock_prepare_seconds: float


def require_files(template_file: Path) -> None:
    missing = [
        path for path in (STOCK_FILE, ROUTES_FILE, template_file) if not path.exists()
    ]
    if missing:
        pytest.skip(f"missing benchmark inputs: {missing}")


def read_stock_smiles() -> tuple[str, ...]:
    with gzip.open(STOCK_FILE, "rt", encoding="utf-8") as handle:
        return tuple(line.split()[0] for line in handle if line.strip())


def read_templates(path: Path) -> tuple[str, ...]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return tuple(
            row["retro_template"] for row in csv.DictReader(handle, delimiter="\t")
        )


def stock_key_fingerprint(
    smiles: str,
    *,
    config: CircularFingerprintConfig,
) -> tuple[tuple[int, int], np.ndarray]:
    molecule = cmol_from_smiles(smiles)[0]
    return (
        short_ring_wl_keyer(molecule),
        circular_fingerprint_words(gmol_from_cmol(molecule), config),
    )


def prepare_stock_data(
    smiles: tuple[str, ...],
    config: CircularFingerprintConfig,
    cache_version: int,
    n_jobs: int | None,
) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    if cache_version != STOCK_CACHE_VERSION:
        raise ValueError("Unsupported CUDA AO* stock cache version.")
    rows = parallel_map(
        partial(stock_key_fingerprint, config=config),
        smiles,
        n_jobs=n_jobs,
    )
    keys, fingerprints = zip(*rows, strict=True)
    return tuple(keys), np.stack(fingerprints)


def cached_stock_data(
    smiles: tuple[str, ...],
) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    cached = cached_function(prepare_stock_data, ignore=("n_jobs",))
    return cached(smiles, FINGERPRINT, STOCK_CACHE_VERSION, None)


def interleave_target_buckets(
    buckets: Sequence[Sequence[tuple[int, str]]],
) -> tuple[tuple[int, str], ...]:
    """Round-robin depth buckets so every prefix remains approximately stratified."""
    return tuple(
        bucket[index]
        for index in range(max(map(len, buckets), default=0))
        for bucket in buckets
        if index < len(bucket)
    )


def sampled_targets() -> tuple[tuple[int, str], ...]:
    """Seeded 34/33/33 sample whose prefixes remain depth-stratified."""
    with gzip.open(ROUTES_FILE, "rt", encoding="utf-8") as handle:
        references = json.load(handle)
    buckets: dict[int, list[str]] = {depth: [] for depth in TARGETS_BY_DEPTH}
    for reference in references:
        depth = aizynth_depth(reference)
        if depth in buckets:
            buckets[depth].append(reference["smiles"])

    rng = np.random.default_rng(TARGET_SEED)
    chosen: list[tuple[tuple[int, str], ...]] = []
    for depth, count in TARGETS_BY_DEPTH.items():
        bucket = sorted(buckets[depth])
        indices = np.sort(rng.choice(len(bucket), count, replace=False))
        chosen.append(tuple((depth, bucket[int(index)]) for index in indices))
    return interleave_target_buckets(chosen)


def load_template_library(
    rule_set: str,
    path: Path,
) -> tuple[int, tuple[str, ...]]:
    """Load the same canonical rule collections used by route-recovery benchmarks."""
    if rule_set == "used14k":
        rows = load_route_template_rows(path)
        return len(rows), route_recovery_rule_index_from_rows(rows).smarts
    templates = read_templates(path)
    return len(templates), templates


def build_setup(cp: Any, rule_set: str, template_file: Path) -> Setup:
    template_started = perf_counter()
    template_row_count, templates = load_template_library(rule_set, template_file)
    rejected = tuple(lint(templates))
    rejected_indices = frozenset(index for index, _, _ in rejected)
    retained = tuple(
        smarts
        for index, smarts in enumerate(templates)
        if index not in rejected_indices
    )
    template_load_seconds = perf_counter() - template_started
    print(
        f"rule_set={rule_set} templates rows={template_row_count} "
        f"rules={len(templates)} "
        f"rejected_mass_losing={len(rejected)} compiled={len(retained)}",
        flush=True,
    )

    parse_started = perf_counter()
    patterns = reaction_patterns_from_smarts(retained, cache=True)
    template_parse_seconds = perf_counter() - parse_started

    compile_started = perf_counter()
    raw_matcher = CudaReactionMatcher(patterns, frontier_mode="device")
    matcher = CudaFoldedPrefilteredReactionMatcher(raw_matcher)
    cp.cuda.Stream.null.synchronize()
    template_compile_seconds = perf_counter() - compile_started

    stock_started = perf_counter()
    stock_smiles = read_stock_smiles()
    stock_keys, stock_fingerprints = cached_stock_data(stock_smiles)
    stock = frozenset(stock_keys)
    heuristic = cuda_stock_similarity_from_fingerprints(
        stock_fingerprints,
        weight=HEURISTIC_WEIGHT,
        config=FINGERPRINT,
    )
    cp.cuda.Stream.null.synchronize()
    stock_prepare_seconds = perf_counter() - stock_started

    return Setup(
        stock=stock,
        heuristic=heuristic,
        matcher=matcher,
        rule_set=rule_set,
        template_file=template_file,
        template_row_count=template_row_count,
        template_count_input=len(templates),
        template_count_rejected=len(rejected),
        rejected_template_indices=tuple(index for index, _, _ in rejected),
        template_load_seconds=template_load_seconds,
        template_parse_seconds=template_parse_seconds,
        template_compile_seconds=template_compile_seconds,
        stock_prepare_seconds=stock_prepare_seconds,
    )


def timed(
    phase_seconds: dict[str, float],
    name: str,
    call: Callable[..., R],
) -> Callable[..., R]:
    def measured(*args: Any, **kwargs: Any) -> R:
        started = perf_counter()
        result = call(*args, **kwargs)
        phase_seconds[name] = phase_seconds.get(name, 0.0) + perf_counter() - started
        return result

    return measured


def run_target(
    cp: Any,
    setup: Setup,
    reference_depth: int,
    smiles: str,
    width: int,
    passes: int,
) -> TargetResult:
    phases: dict[str, float] = {}
    matcher_seconds: Counter[str] = Counter()
    matcher_counts: Counter[str] = Counter()
    expansions = Expansions(NumpyExpansionStore(), DictInterner())
    price, scheduled = anytime_step(
        greedy=GREEDY_STEP,
        relaxed=RELAXED_STEP,
        select=committed_application_leaves(),
    )
    select_calls = 0
    measured_select = timed(phases, "select", scheduled)

    def select(*args: Any, **kwargs: Any) -> list[int]:
        nonlocal select_calls
        select_calls += 1
        return measured_select(*args, **kwargs)

    def match(molecules: Any) -> Any:
        started = perf_counter()
        result = expansion(setup.matcher.match_keyed)(molecules)
        phases["match"] = phases.get("match", 0.0) + perf_counter() - started
        diagnostic = setup.matcher.last_diagnostics
        prefilter = setup.matcher.last_prefilter_diagnostics
        if diagnostic is not None:
            for name in (
                "match_seconds",
                "retro_seconds",
                "key_seconds",
                "wl_kernel_seconds",
                "host_to_device_seconds",
                "device_to_host_seconds",
                "host_store_seconds",
                "materialize_seconds",
                "total_seconds",
            ):
                matcher_seconds[name] += float(getattr(diagnostic, name))
            matcher_counts.update(
                positive_pairs=diagnostic.status_yes,
                assignments=diagnostic.assignments_total,
                raw_precursors=diagnostic.raw_precursor_count,
                unique_precursors=diagnostic.unique_precursor_count,
                precursor_atom_overflow_components=(
                    diagnostic.precursor_atom_overflow_components
                ),
                precursor_atom_overflow_assignments=(
                    diagnostic.precursor_atom_overflow_assignments
                ),
            )
        if prefilter is not None:
            matcher_seconds["prefilter_seconds"] += prefilter.prefilter_seconds
            matcher_counts.update(
                dense_pairs=prefilter.dense_pair_count,
                candidate_pairs=prefilter.candidate_pair_count,
            )
        return result

    cp.cuda.Stream.null.synchronize()
    started = perf_counter()
    routes = aostar(
        short_ring_wl_keyer.from_smiles(smiles),
        target_mol=cmol_from_smiles(smiles)[0],
        stock=setup.stock,
        expansions=expansions,
        heuristic=timed(phases, "heuristic", setup.heuristic),
        match=match,
        stack_mols=cuda_stack_mols,
        concat_mols=cuda_concat_host_mols,
        width=width,
        passes=passes,
        max_depth=passes,
        select=select,
        fold=timed(phases, "fold", bottleneck_fold),
        price=price,
    )
    cp.cuda.Stream.null.synchronize()
    elapsed = perf_counter() - started
    return TargetResult(
        target=smiles,
        reference_depth=reference_depth,
        solved=bool(routes),
        seconds=elapsed,
        width=width,
        pass_limit=passes,
        passes=select_calls,
        molecules_expanded=int(expansions.store.expanded().size),
        molecules_seen=expansions.sid_count(),
        routes=len(routes),
        best_reaction_count=(
            min(reaction_count(route) for route in routes) if routes else None
        ),
        phase_seconds=phases,
        matcher_seconds=dict(matcher_seconds),
        dense_pairs=matcher_counts["dense_pairs"],
        candidate_pairs=matcher_counts["candidate_pairs"],
        positive_pairs=matcher_counts["positive_pairs"],
        assignments=matcher_counts["assignments"],
        raw_precursors=matcher_counts["raw_precursors"],
        unique_precursors=matcher_counts["unique_precursors"],
        precursor_atom_overflow_components=matcher_counts[
            "precursor_atom_overflow_components"
        ],
        precursor_atom_overflow_assignments=matcher_counts[
            "precursor_atom_overflow_assignments"
        ],
    )


def summarize_results(results: Sequence[TargetResult]) -> dict[str, Any]:
    solved = [result for result in results if result.solved]
    total_seconds = sum(result.seconds for result in results)
    phase_names = sorted({name for result in results for name in result.phase_seconds})
    return {
        "targets": len(results),
        "solved_targets": len(solved),
        "solve_rate": len(solved) / len(results),
        "search_seconds": total_seconds,
        "targets_per_second": len(results) / total_seconds,
        "latency_seconds_mean": float(np.mean([result.seconds for result in results])),
        "latency_seconds_median": float(
            np.median([result.seconds for result in results])
        ),
        "latency_seconds_p95": float(
            np.percentile([result.seconds for result in results], 95)
        ),
        "expanded_per_target_mean": float(
            np.mean([result.molecules_expanded for result in results])
        ),
        "seen_per_target_mean": float(
            np.mean([result.molecules_seen for result in results])
        ),
        "phase_seconds": {
            name: sum(result.phase_seconds.get(name, 0.0) for result in results)
            for name in phase_names
        },
        "solved_by_reference_depth": dict(
            sorted(Counter(result.reference_depth for result in solved).items())
        ),
    }


def parse_efforts() -> tuple[tuple[int, int], ...]:
    configured = os.environ.get("AOSTAR_CUDA_EFFORTS", "").strip()
    if configured:
        efforts = tuple(
            tuple(int(value) for value in item.lower().split("x"))
            for item in configured.split(",")
        )
    else:
        efforts = (
            (
                int(os.environ.get("AOSTAR_CUDA_WIDTH", "32")),
                int(os.environ.get("AOSTAR_CUDA_PASSES", "8")),
            ),
        )
    if any(len(effort) != 2 for effort in efforts):
        raise ValueError("AOSTAR_CUDA_EFFORTS must contain width x passes pairs.")
    if any(width <= 0 or passes <= 0 for width, passes in efforts):
        raise ValueError("AO* widths and passes must be positive.")
    if len(set(efforts)) != len(efforts):
        raise ValueError("AOSTAR_CUDA_EFFORTS must not contain duplicates.")
    return efforts


def test_bench_cuda_aostar_n1(benchmark: Any) -> None:
    rule_set = os.environ.get("AOSTAR_CUDA_RULE_SET", "used14k")
    if rule_set not in TEMPLATE_FILES:
        raise ValueError(
            f"AOSTAR_CUDA_RULE_SET must be one of {tuple(TEMPLATE_FILES)}, "
            f"got {rule_set!r}."
        )
    template_file = TEMPLATE_FILES[rule_set]
    require_files(template_file)
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CuPy cannot access CUDA on this machine.")

    efforts = parse_efforts()
    setup = build_setup(cp, rule_set, template_file)
    target_limit = int(os.environ.get("AOSTAR_CUDA_TARGETS", "1"))
    if target_limit <= 0 or target_limit > 100:
        raise ValueError("AOSTAR_CUDA_TARGETS must be between 1 and 100.")
    targets = sampled_targets()[:target_limit]
    peak_pool_total_bytes = int(cp.get_default_memory_pool().total_bytes())

    def run() -> tuple[TargetResult, ...]:
        nonlocal peak_pool_total_bytes
        results = []
        for width, passes in efforts:
            effort = f"{width}x{passes}"
            for index, (depth, smiles) in enumerate(targets, start=1):
                print(
                    f"[{effort} {index}/{len(targets)}] start "
                    f"depth={depth} target={smiles}",
                    flush=True,
                )
                result = run_target(cp, setup, depth, smiles, width, passes)
                results.append(result)
                measured = sum(result.phase_seconds.values())
                orchestration = max(result.seconds - measured, 0.0)
                print(
                    f"[{effort} {index}/{len(targets)}] done {result.seconds:.3f}s "
                    f"solved={result.solved} expanded={result.molecules_expanded} "
                    f"seen={result.molecules_seen} "
                    f"match={result.phase_seconds.get('match', 0.0):.3f}s "
                    f"heuristic={result.phase_seconds.get('heuristic', 0.0):.3f}s "
                    f"fold={result.phase_seconds.get('fold', 0.0):.3f}s "
                    f"select={result.phase_seconds.get('select', 0.0):.3f}s "
                    f"orchestration={orchestration:.3f}s "
                    f"overflow_components={result.precursor_atom_overflow_components}",
                    flush=True,
                )
                peak_pool_total_bytes = max(
                    peak_pool_total_bytes,
                    int(cp.get_default_memory_pool().total_bytes()),
                )
        return tuple(results)

    rounds = int(os.environ.get("BENCH_ROUNDS", "1"))
    results = benchmark.pedantic(run, rounds=rounds, warmup_rounds=0, iterations=1)
    summaries = {
        f"{width}x{passes}": summarize_results(
            tuple(
                result
                for result in results
                if result.width == width and result.pass_limit == passes
            )
        )
        for width, passes in efforts
    }
    raw_matcher = setup.matcher.matcher
    benchmark.extra_info.update(
        {
            "experiment": "cuda-aostar-anytime",
            "select": "committed-application+global",
            "backend": "cuda-hybrid-prefilter",
            "rule_set": setup.rule_set,
            "efforts": tuple(
                {"width": width, "passes": passes, "max_depth": passes}
                for width, passes in efforts
            ),
            "greedy_step": GREEDY_STEP,
            "relaxed_step": RELAXED_STEP,
            "fold": "max",
            "fingerprint": asdict(FINGERPRINT),
            "heuristic_weight": HEURISTIC_WEIGHT,
            "target_seed": TARGET_SEED,
            "target_counts_by_reference_depth": TARGETS_BY_DEPTH,
            "selected_target_counts_by_reference_depth": dict(
                sorted(Counter(depth for depth, _ in targets).items())
            ),
            "target_order": "depth-interleaved",
            "target_limit": target_limit,
            "template_row_count": setup.template_row_count,
            "template_count_input": setup.template_count_input,
            "template_count_rejected": setup.template_count_rejected,
            "template_count_compiled": (
                setup.template_count_input - setup.template_count_rejected
            ),
            "rejected_template_indices": setup.rejected_template_indices,
            "template_load_seconds": setup.template_load_seconds,
            "template_parse_seconds": setup.template_parse_seconds,
            "template_compile_seconds": setup.template_compile_seconds,
            "stock_prepare_seconds": setup.stock_prepare_seconds,
            "stock_size": len(setup.stock),
            "stock_fingerprint_bytes": setup.heuristic.nbytes,
            "resident_rule_bytes": raw_matcher.resident_rule_bytes,
            "peak_cupy_pool_total_bytes": peak_pool_total_bytes,
            "summaries": summaries,
            "records": [asdict(result) for result in results],
            "inputs": {
                "stock": source_artifact(STOCK_FILE),
                "routes": source_artifact(ROUTES_FILE),
                "templates": source_artifact(setup.template_file),
            },
        }
    )
    print(json.dumps(summaries, indent=2, sort_keys=True), flush=True)

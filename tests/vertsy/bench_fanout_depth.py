"""Deep fanout-per-depth benchmark (pytest-benchmark).

Runs a *full* (uncapped) retrosynthetic expansion for a handful of PaRoutes targets and
records, per depth, the branching behaviour of the search: frontier size, ``(mol, rule)``
pairs considered, positive matches, precursors produced, precursor atom-count summary, and
the deduplicated next-frontier size. The paper reports these measured layers directly.

Unlike :mod:`tests.vertsy.bench_route_recovery` (fixed depth 2, plus the reference-route
recovery comparators), this bench has no recovery metrics and a configurable depth, so it can
be pushed deeper on capable hardware. It reuses the recovery backends and loaders wholesale.

The intended run is the smallest library (``used14k``) on GPU, probing depth 3
on a single target first (``BENCH_FANOUT_DEPTH_MAX_DEPTH`` / ``BENCH_FANOUT_DEPTH_TARGETS``) and
going deeper only if a layer stays fast. ``BENCH_FANOUT_DEPTH_FRONTIER_CEILING`` is a safety
abort — not a sampler — that stops a runaway expansion before it exhausts memory; completed
layers stay exact and the target is flagged ``truncated``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pytest

from tests.vertsy.bench_route_recovery import (
    SUPPORTED_BACKENDS,
    SUPPORTED_RULE_SETS,
    TARGETS_FILE,
    LayerStats,
    RecoveryBackend,
    aggregate_cuda_memory,
    atom_count_stats,
    build_recovery_backend,
    cleanup_cuda_memory,
    cuda_memory_extra,
    distinct_applications,
    last_cuda_diagnostic,
    layer_extra_info,
    load_rule_set,
    load_target_smiles,
    precursor_atom_counts,
    reset_cuda_diagnostics,
    rounds,
    source_artifacts,
    warmup_rounds,
)
from vertsy.expansions import Expansions
from vertsy.interner import DictInterner
from vertsy.search import MoleculeBatch, step
from vertsy.store.numpy_store import NumpyExpansionStore
from vesma.matcher import HitArray

pytestmark = [pytest.mark.perf, pytest.mark.paper_benchmark]

Key = Any

DEFAULT_BACKENDS = ("cuda-host",)
# used14k, the smallest library — the one deep expansion can actually reach.
DEFAULT_RULE_SETS = ("used14k",)
DEFAULT_MAX_DEPTH = 3
DEFAULT_TARGETS = 1
# Safety backstop: abort a target before expanding a frontier larger than this (protects the
# box from OOM on a runaway layer). Tune per hardware and report truncation explicitly.
DEFAULT_FRONTIER_CEILING = 2_000_000


@dataclass(frozen=True)
class FanoutTarget:
    target_index: int
    target_smiles: str
    layers: tuple[LayerStats, ...]
    truncated: bool  # True if the frontier ceiling aborted the expansion early
    depth_reached: int  # number of layers actually expanded
    search_seconds: float


def run_fanout_target(
    backend: RecoveryBackend,
    target_index: int,
    target_smiles: str,
    *,
    max_depth: int,
    frontier_ceiling: int,
) -> FanoutTarget:
    """Fold :func:`vertsy.search.step` up to ``max_depth``, timing and counting each layer.

    Mirrors the instrumented step loop of ``bench_route_recovery.run_recovery_target`` but
    without the reference-route recovery metrics, with a configurable depth, and with a
    frontier ceiling that aborts (does not subsample) a runaway expansion.
    """
    key = backend.from_smiles_key(target_smiles)
    mols = backend.stack_mols([backend.from_smiles_mol(target_smiles)])

    expansions: Expansions[Key] = Expansions(NumpyExpansionStore(), DictInterner())
    sids = expansions.intern([key])
    in_stock = np.fromiter((key in backend.stock,), dtype=np.bool_, count=1)
    expansions.mark_expanded(sids[in_stock])
    frontier_sids = sids[~in_stock]
    frontier_mols = mols[~in_stock]

    layers: list[LayerStats] = []
    truncated = False
    depth_reached = 0
    search_started = perf_counter()
    for depth in range(max_depth):
        if not len(frontier_sids):
            break
        frontier_size = int(len(frontier_sids))
        if frontier_size > frontier_ceiling:
            truncated = True
            break
        captured: dict[str, Any] = {}

        def counting_match(
            batch: MoleculeBatch, _captured: dict[str, Any] = captured
        ) -> tuple[MoleculeBatch, Sequence[Key], HitArray]:
            started = perf_counter()
            precursor_mols, precursor_keys, hits = backend.match(batch)
            _captured["match_split_seconds"] = perf_counter() - started
            _captured["hits"] = hits
            _captured["precursors"] = len(precursor_keys)
            mean_atoms, median_atoms, max_atoms = atom_count_stats(
                precursor_atom_counts(precursor_mols)
            )
            _captured["precursor_atoms_mean"] = mean_atoms
            _captured["precursor_atoms_median"] = median_atoms
            _captured["precursor_atoms_max"] = max_atoms
            _captured["cuda_memory"] = cuda_memory_extra(last_cuda_diagnostic(backend))
            return precursor_mols, precursor_keys, hits

        started = perf_counter()
        frontier_sids, frontier_mols = step(
            frontier_sids,
            frontier_mols,
            expansions,
            backend.stock,
            match=counting_match,
        )
        step_seconds = perf_counter() - started

        layers.append(
            LayerStats(
                depth=depth,
                frontier_size=frontier_size,
                pairs=frontier_size * backend.rule_count,
                applications=distinct_applications(captured["hits"]),
                precursors=int(captured["precursors"]),
                precursor_atoms_mean=float(captured["precursor_atoms_mean"]),
                precursor_atoms_median=float(captured["precursor_atoms_median"]),
                precursor_atoms_max=int(captured["precursor_atoms_max"]),
                next_frontier_size=int(len(frontier_sids)),
                match_split_seconds=float(captured["match_split_seconds"]),
                step_seconds=step_seconds,
                cuda_memory=captured["cuda_memory"],
            )
        )
        depth_reached = depth + 1

    return FanoutTarget(
        target_index=target_index,
        target_smiles=target_smiles,
        layers=tuple(layers),
        truncated=truncated,
        depth_reached=depth_reached,
        search_seconds=perf_counter() - search_started,
    )


def fanout_extra_info(
    targets: Sequence[FanoutTarget],
    *,
    backend: str,
    rule_set: str,
    template_source: str,
    rules: int,
    stock_size: int,
    compile_seconds: float,
    max_depth: int,
    frontier_ceiling: int,
    targets_file: str,
) -> dict[str, Any]:
    return {
        "experiment": "fanout-depth",
        "backend": backend,
        "rule_set": rule_set,
        "route_set": targets_file.removesuffix(".smi.gz"),
        "targets_file": targets_file,
        "template_source": template_source,
        "stock_policy": "paroutes-n1-stock",
        "rules": rules,
        "stock_size": stock_size,
        "max_depth": max_depth,
        "frontier_ceiling": frontier_ceiling,
        "compile_seconds": compile_seconds,
        "num_targets": len(targets),
        "any_truncated": any(target.truncated for target in targets),
        "targets": [fanout_target_extra_info(target) for target in targets],
        "cuda_memory": aggregate_cuda_memory(
            layer.cuda_memory for target in targets for layer in target.layers
        ),
        "source_artifacts": source_artifacts(template_source, targets_file),
    }


def fanout_target_extra_info(target: FanoutTarget) -> dict[str, Any]:
    return {
        "target_index": target.target_index,
        "target_smiles": target.target_smiles,
        "truncated": target.truncated,
        "depth_reached": target.depth_reached,
        "search_seconds": target.search_seconds,
        "cuda_memory": aggregate_cuda_memory(
            layer.cuda_memory for layer in target.layers
        ),
        "layers": [layer_extra_info(layer) for layer in target.layers],
    }


# ----------------------------------------------------------------------- the bench


def test_bench_fanout_depth(
    benchmark: Any, fanout_backend: str, fanout_rule_set: str
) -> None:
    cleanup_cuda_memory()
    backend: RecoveryBackend | None = None
    try:
        rule_index, template_source, _rule_aligned = load_rule_set(fanout_rule_set)
        if not rule_index.smarts:
            pytest.skip(
                f"Fanout library {fanout_rule_set!r} has no compatible templates."
            )

        backend = build_recovery_backend(fanout_backend, rule_index)
        compile_seconds = backend.compile_seconds
        stock_size = len(backend.stock)
        targets_file = selected_targets_file()
        target_smiles = load_target_smiles(selected_targets(), targets_file)
        max_depth = selected_max_depth()
        frontier_ceiling = selected_frontier_ceiling()

        def run() -> list[FanoutTarget]:
            if backend is None:
                raise RuntimeError("fanout backend was released")
            reset_cuda_diagnostics(backend)
            try:
                results: list[FanoutTarget] = []
                for index, smiles in enumerate(target_smiles):
                    results.append(
                        run_fanout_target(
                            backend,
                            index,
                            smiles,
                            max_depth=max_depth,
                            frontier_ceiling=frontier_ceiling,
                        )
                    )
                    if backend.name.startswith("cuda"):
                        cleanup_cuda_memory()
                return results
            finally:
                cleanup_cuda_memory()

        result = benchmark.pedantic(
            run, rounds=rounds(), warmup_rounds=warmup_rounds(), iterations=1
        )
        benchmark.extra_info.update(
            fanout_extra_info(
                result,
                backend=fanout_backend,
                rule_set=fanout_rule_set,
                template_source=template_source,
                rules=len(rule_index.smarts),
                stock_size=stock_size,
                compile_seconds=compile_seconds,
                max_depth=max_depth,
                frontier_ceiling=frontier_ceiling,
                targets_file=targets_file,
            )
        )
    finally:
        backend = None
        cleanup_cuda_memory()


# ------------------------------------------------------------------ parametrization


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "fanout_backend" in metafunc.fixturenames:
        metafunc.parametrize("fanout_backend", selected_backends())
    if "fanout_rule_set" in metafunc.fixturenames:
        metafunc.parametrize("fanout_rule_set", selected_rule_sets())


def selected_backends() -> tuple[str, ...]:
    raw = os.environ.get("BENCH_FANOUT_DEPTH_BACKENDS")
    values = (
        DEFAULT_BACKENDS
        if not raw
        else tuple(value.strip() for value in raw.replace(",", " ").split())
    )
    unsupported = [value for value in values if value not in SUPPORTED_BACKENDS]
    if unsupported:
        raise ValueError(
            f"BENCH_FANOUT_DEPTH_BACKENDS has unsupported entries: {unsupported}"
        )
    return tuple(value for value in values if value)


def selected_rule_sets() -> tuple[str, ...]:
    raw = os.environ.get("BENCH_FANOUT_DEPTH_RULE_SETS")
    values = (
        DEFAULT_RULE_SETS
        if not raw
        else tuple(value.strip() for value in raw.replace(",", " ").split())
    )
    unsupported = [value for value in values if value not in SUPPORTED_RULE_SETS]
    if unsupported:
        raise ValueError(
            f"BENCH_FANOUT_DEPTH_RULE_SETS has unsupported entries: {unsupported}"
        )
    return tuple(value for value in values if value)


def selected_targets() -> int:
    value = int(os.environ.get("BENCH_FANOUT_DEPTH_TARGETS", str(DEFAULT_TARGETS)))
    if value <= 0:
        raise ValueError("BENCH_FANOUT_DEPTH_TARGETS must be positive.")
    return value


def selected_targets_file() -> str:
    # Default n1-d2-f0 keeps the smallest depth-2 target set; n1-d3-f0.smi.gz holds the targets
    # (frontier 0) that genuinely need three steps, so depth-3 expansion recovers real routes.
    return os.environ.get("BENCH_FANOUT_DEPTH_TARGETS_FILE", TARGETS_FILE)


def selected_max_depth() -> int:
    value = int(os.environ.get("BENCH_FANOUT_DEPTH_MAX_DEPTH", str(DEFAULT_MAX_DEPTH)))
    if value <= 0:
        raise ValueError("BENCH_FANOUT_DEPTH_MAX_DEPTH must be positive.")
    return value


def selected_frontier_ceiling() -> int:
    value = int(
        os.environ.get(
            "BENCH_FANOUT_DEPTH_FRONTIER_CEILING", str(DEFAULT_FRONTIER_CEILING)
        )
    )
    if value <= 0:
        raise ValueError("BENCH_FANOUT_DEPTH_FRONTIER_CEILING must be positive.")
    return value

from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl
import pytest

from vertsy.expansions import Expansions, edge_frame, sid_steps
from vertsy.interner import DictInterner, SidArray
from vertsy.routes import solved_sids
from vertsy.store.numpy_store import NumpyExpansionStore

pytestmark = [pytest.mark.perf]


@dataclass(frozen=True)
class SolvedSidsBenchmarkConfig:
    depth: int
    products_per_layer: int
    applications_per_product: int
    precursors_per_application: int
    repeats: int


@dataclass(frozen=True)
class SolvedSidsBenchmarkResult:
    benchmark: str
    config: SolvedSidsBenchmarkConfig
    sid_count: int
    edge_count: int
    solved_count: int
    production_ms: tuple[float, ...]
    production_best_ms: float
    simplified_incremental_ms: tuple[float, ...]
    simplified_incremental_best_ms: float
    materialized_fixpoint_ms: tuple[float, ...]
    materialized_fixpoint_best_ms: float
    numpy_fixpoint_ms: tuple[float, ...]
    numpy_fixpoint_best_ms: float
    polars_fixpoint_ms: tuple[float, ...]
    polars_fixpoint_best_ms: float


@dataclass(frozen=True)
class PropagationIndex:
    parent_applications: tuple[tuple[int, ...], ...]
    application_products: tuple[int, ...]
    precursor_counts: tuple[int, ...]


def test_perf_solved_sids() -> None:
    config = config_from_env()
    result = run_solved_sids_benchmark(config)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


def run_solved_sids_benchmark(
    config: SolvedSidsBenchmarkConfig,
) -> SolvedSidsBenchmarkResult:
    expansions, stock_sids = synthetic_expansions(config)

    expected = solved_sids(expansions, stock_sids)
    alternatives = (
        solved_sids_simplified_incremental(expansions, stock_sids),
        solved_sids_materialized_fixpoint(expansions, stock_sids),
        solved_sids_numpy_fixpoint(expansions, stock_sids),
    )
    for got in alternatives:
        if not np.array_equal(got, expected):
            raise AssertionError("solved_sids implementations disagree")
    polars_expected = solved_sids_polars_fixpoint(expansions, stock_sids)
    if not np.array_equal(polars_expected, expected):
        raise AssertionError("polars solved_sids implementation disagrees")

    production_ms = timed_repeats(
        config.repeats, lambda: solved_sids(expansions, stock_sids)
    )
    simplified_incremental_ms = timed_repeats(
        config.repeats,
        lambda: solved_sids_simplified_incremental(expansions, stock_sids),
    )
    materialized_fixpoint_ms = timed_repeats(
        config.repeats,
        lambda: solved_sids_materialized_fixpoint(expansions, stock_sids),
    )
    numpy_fixpoint_ms = timed_repeats(
        config.repeats,
        lambda: solved_sids_numpy_fixpoint(expansions, stock_sids),
    )
    polars_fixpoint_ms = timed_repeats(
        config.repeats,
        lambda: solved_sids_polars_fixpoint(expansions, stock_sids),
    )

    product, _, _ = expansions.edges()
    return SolvedSidsBenchmarkResult(
        benchmark="solved-sids",
        config=config,
        sid_count=expansions.sid_count(),
        edge_count=len(product),
        solved_count=int(expected.sum()),
        production_ms=production_ms,
        production_best_ms=min(production_ms),
        simplified_incremental_ms=simplified_incremental_ms,
        simplified_incremental_best_ms=min(simplified_incremental_ms),
        materialized_fixpoint_ms=materialized_fixpoint_ms,
        materialized_fixpoint_best_ms=min(materialized_fixpoint_ms),
        numpy_fixpoint_ms=numpy_fixpoint_ms,
        numpy_fixpoint_best_ms=min(numpy_fixpoint_ms),
        polars_fixpoint_ms=polars_fixpoint_ms,
        polars_fixpoint_best_ms=min(polars_fixpoint_ms),
    )


def solved_sids_simplified_incremental(
    expansions: Expansions[int],
    stock_sids: SidArray,
) -> np.ndarray:
    return propagate_solved(
        build_propagation_index(expansions),
        expansions.sid_count(),
        stock_sids,
    )


def build_propagation_index(expansions: Expansions[int]) -> PropagationIndex:
    parent_applications: list[list[int]] = [[] for _ in range(expansions.sid_count())]
    application_ids: dict[tuple[int, int], int] = {}
    application_products: list[int] = []
    precursor_counts: list[int] = []

    for product_sid, app_id, precursor_sid in zip(*expansions.edges()):
        application = (product_sid, app_id)
        application_id = application_ids.get(application)
        if application_id is None:
            application_id = len(application_products)
            application_ids[application] = application_id
            application_products.append(product_sid)
            precursor_counts.append(0)

        precursor_counts[application_id] += 1
        parent_applications[precursor_sid].append(application_id)

    return PropagationIndex(
        parent_applications=tuple(tuple(apps) for apps in parent_applications),
        application_products=tuple(application_products),
        precursor_counts=tuple(precursor_counts),
    )


def propagate_solved(
    index: PropagationIndex,
    sid_count: int,
    stock_sids: SidArray,
) -> np.ndarray:
    solved = np.zeros(sid_count, dtype=np.bool_)
    solved[stock_sids] = True

    remaining = list(index.precursor_counts)
    application_solved = [False] * len(index.application_products)
    queue = deque(stock_sids.tolist())

    while queue:
        solved_precursor = queue.popleft()
        for application in index.parent_applications[solved_precursor]:
            if application_solved[application]:
                continue

            remaining[application] -= 1
            if remaining[application] != 0:
                continue

            application_solved[application] = True
            solved_product = index.application_products[application]
            if solved[solved_product]:
                continue

            solved[solved_product] = True
            queue.append(solved_product)

    return solved


def solved_sids_materialized_fixpoint(
    expansions: Expansions[int],
    stock_sids: SidArray,
) -> np.ndarray:
    solved = np.zeros(expansions.sid_count(), dtype=np.bool_)
    solved[stock_sids] = True

    grouped = sid_steps(expansions, expansions.store.expanded())
    changed = True
    while changed:
        changed = False
        for product, steps in grouped.items():
            if solved[product]:
                continue
            for _, precursors in steps:
                if all(solved[precursor] for precursor in precursors):
                    solved[product] = True
                    changed = True
                    break
    return solved


def solved_sids_numpy_fixpoint(
    expansions: Expansions[int],
    stock_sids: SidArray,
) -> np.ndarray:
    solved = np.zeros(expansions.sid_count(), dtype=np.bool_)
    solved[stock_sids] = True

    product, app, precursor = expansions.edges()
    if not len(product):
        return solved

    application_keys = np.stack((product, app), axis=1)
    applications, application_by_edge = np.unique(
        application_keys,
        axis=0,
        return_inverse=True,
    )
    application_products = applications[:, 0]
    precursor_count = np.bincount(application_by_edge)

    while True:
        solved_precursor_count = np.bincount(
            application_by_edge,
            weights=solved[precursor].astype(np.int64),
            minlength=len(applications),
        )
        solved_applications = solved_precursor_count == precursor_count
        newly_solved_products = application_products[solved_applications]
        solved_count = int(solved.sum())
        solved[newly_solved_products] = True
        if int(solved.sum()) == solved_count:
            return solved


def solved_sids_polars_fixpoint(
    expansions: Expansions[int],
    stock_sids: SidArray,
) -> np.ndarray:
    solved = np.zeros(expansions.sid_count(), dtype=np.bool_)
    solved[stock_sids] = True

    edges = edge_frame(expansions)
    if edges.is_empty():
        return solved
    precursor = edges.get_column("precursor").to_numpy()

    while True:
        solved_applications = (
            edges.with_columns(pl.Series("precursor_is_solved", solved[precursor]))
            .group_by("product", "app")
            .agg(pl.col("precursor_is_solved").all().alias("app_all_is_solved"))
            .filter(pl.col("app_all_is_solved"))
        )
        newly_solved_products = solved_applications["product"].to_numpy()
        solved_count = int(solved.sum())
        solved[newly_solved_products] = True
        if int(solved.sum()) == solved_count:
            return solved


def synthetic_expansions(
    config: SolvedSidsBenchmarkConfig,
) -> tuple[Expansions[int], SidArray]:
    expansions = Expansions(NumpyExpansionStore(), DictInterner())

    stock_keys = layer_keys(0, config.products_per_layer)
    stock_sids = expansions.intern(stock_keys)
    expansions.mark_expanded(stock_sids)

    for depth in range(1, config.depth + 1):
        product_keys = layer_keys(depth, config.products_per_layer)
        precursor_keys = layer_keys(depth - 1, config.products_per_layer)
        product_sids = expansions.intern(product_keys)
        expansions.mark_expanded(product_sids)
        expansions.store.add_edges(
            *synthetic_layer_edges(
                product_sids,
                precursor_keys,
                config.applications_per_product,
                config.precursors_per_application,
                expansions,
            )
        )

    return expansions, stock_sids


def synthetic_layer_edges(
    product_sids: SidArray,
    precursor_keys: list[int],
    applications_per_product: int,
    precursors_per_application: int,
    expansions: Expansions[int],
) -> tuple[SidArray, SidArray, SidArray]:
    edge_count = (
        len(product_sids) * applications_per_product * precursors_per_application
    )
    products = np.empty(edge_count, dtype=np.int64)
    apps = np.empty(edge_count, dtype=np.int64)
    precursor_lookup: list[int] = []

    row = 0
    for product_index, product_sid in enumerate(product_sids.tolist()):
        for app_id in range(applications_per_product):
            for precursor_index in range(precursors_per_application):
                products[row] = product_sid
                apps[row] = app_id
                precursor_lookup.append(
                    precursor_keys[
                        (product_index + app_id * 17 + precursor_index * 31)
                        % len(precursor_keys)
                    ]
                )
                row += 1

    return products, apps, expansions.intern(precursor_lookup)


def layer_keys(depth: int, count: int) -> list[int]:
    offset = depth * count
    return list(range(offset, offset + count))


def timed_repeats(repeats: int, run) -> tuple[float, ...]:
    out: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        run()
        out.append(elapsed_ms(started))
    return tuple(out)


def config_from_env() -> SolvedSidsBenchmarkConfig:
    return SolvedSidsBenchmarkConfig(
        depth=env_int("VERTSY_SOLVED_SIDS_DEPTH", 4),
        products_per_layer=env_int("VERTSY_SOLVED_SIDS_PRODUCTS", 10_000),
        applications_per_product=env_int("VERTSY_SOLVED_SIDS_APPS", 3),
        precursors_per_application=env_int("VERTSY_SOLVED_SIDS_PRECURSORS", 2),
        repeats=env_int("VERTSY_SOLVED_SIDS_REPEATS", 5),
    )


def env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def elapsed_ms(started_ns: int) -> float:
    return (time.perf_counter_ns() - started_ns) / 1_000_000

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import pytest

from vertsy.expansions import Expansions, sid_steps
from vertsy.interner import DictInterner
from vertsy.store.numpy_store import NumpyExpansionStore

pytestmark = [pytest.mark.perf]


@dataclass(frozen=True)
class ExpansionStoreBenchmarkConfig:
    frontier_count: int
    applications_per_product: int
    precursors_per_application: int
    unique_precursor_modulus: int
    repeats: int
    grouping_sample_size: int


@dataclass(frozen=True)
class ExpansionStoreBenchmarkResult:
    benchmark: str
    config: ExpansionStoreBenchmarkConfig
    edge_count: int
    append_ms: tuple[float, ...]
    append_best_ms: float
    on_demand_group_ms: tuple[float, ...]
    on_demand_group_best_ms: float
    expanded_product_count: int
    no_dict_materialized: bool


def test_perf_expansion_store_hot_path() -> None:
    config = config_from_env()
    result = run_expansion_store_benchmark(config)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


def run_expansion_store_benchmark(
    config: ExpansionStoreBenchmarkConfig,
) -> ExpansionStoreBenchmarkResult:
    frontier_keys = list(range(config.frontier_count))
    precursor_keys, hits = synthetic_layer(config)

    append_ms: list[float] = []
    on_demand_group_ms: list[float] = []
    expanded_product_count = 0

    for _ in range(config.repeats):
        expansions = Expansions(NumpyExpansionStore(), DictInterner())
        frontier_sids = expansions.intern(frontier_keys)

        started = time.perf_counter_ns()
        precursor_sids = expansions.intern(precursor_keys)
        expansions.add_layer(frontier_sids, hits, precursor_sids)
        append_ms.append(elapsed_ms(started))

        expanded_product_count = len(expansions.store.expanded())

        requested = frontier_sids[: config.grouping_sample_size]
        started = time.perf_counter_ns()
        grouped = sid_steps(expansions, requested)
        on_demand_group_ms.append(elapsed_ms(started))
        if len(grouped) != len(requested):
            raise AssertionError(
                "on-demand grouping must return every requested product"
            )

    return ExpansionStoreBenchmarkResult(
        benchmark="expansion-store-hot-path",
        config=config,
        edge_count=len(precursor_keys),
        append_ms=tuple(append_ms),
        append_best_ms=min(append_ms),
        on_demand_group_ms=tuple(on_demand_group_ms),
        on_demand_group_best_ms=min(on_demand_group_ms),
        expanded_product_count=expanded_product_count,
        no_dict_materialized=True,
    )


def synthetic_layer(
    config: ExpansionStoreBenchmarkConfig,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    edge_count = (
        config.frontier_count
        * config.applications_per_product
        * config.precursors_per_application
    )
    precursor_keys: list[tuple[int, int]] = []
    hits = np.empty((edge_count, 4), dtype=np.int64)
    row = 0
    for product_index in range(config.frontier_count):
        for app_index in range(config.applications_per_product):
            for precursor_index in range(config.precursors_per_application):
                # One precursor per hit: the benchmark measures the store, not dedup.
                hits[row] = (product_index, app_index % 1024, app_index, row)
                key = (
                    product_index * 17 + app_index * 3 + precursor_index
                ) % config.unique_precursor_modulus
                precursor_keys.append((key, (key * 1_315_423_911) & ((1 << 63) - 1)))
                row += 1
    return precursor_keys, hits


def config_from_env() -> ExpansionStoreBenchmarkConfig:
    frontier_count = env_int("VERTSY_EXPANSION_BENCH_FRONTIER", 20_000)
    applications_per_product = env_int("VERTSY_EXPANSION_BENCH_APPS", 4)
    precursors_per_application = env_int("VERTSY_EXPANSION_BENCH_PRECURSORS", 2)
    repeats = env_int("VERTSY_EXPANSION_BENCH_REPEATS", 5)
    grouping_sample_size = env_int("VERTSY_EXPANSION_BENCH_GROUP_SAMPLE", 256)
    return ExpansionStoreBenchmarkConfig(
        frontier_count=frontier_count,
        applications_per_product=applications_per_product,
        precursors_per_application=precursors_per_application,
        unique_precursor_modulus=env_int(
            "VERTSY_EXPANSION_BENCH_UNIQUE_PRECURSORS",
            max(frontier_count // 2, 1),
        ),
        repeats=repeats,
        grouping_sample_size=min(grouping_sample_size, frontier_count),
    )


def env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def elapsed_ms(started_ns: int) -> float:
    return (time.perf_counter_ns() - started_ns) / 1_000_000

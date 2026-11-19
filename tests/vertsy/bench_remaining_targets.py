"""Larger-budget probes for targets a campaign has not solved, writing results beside them.

Requires FRONTIER_EXPERIMENT_DIR and FRONTIER_STOCK_CACHE as documented by
bench_bottleneck_selection, and reads its targets from FRONTIER_TARGETS (a filename within
that directory). Nothing here names a campaign: the directory, its targets and its results
all live outside the repository. Retains the existing committed selector for comparison with
the earlier failed probes; does not change ranking and budget simultaneously.
"""

import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pytest

from vertsy.aostar import committed_application_leaves

from .bench_host_aostar import (
    TEMPLATE_FILES,
    Stock,
    cached_stock,
    load_library,
    metal_arm,
    run_target,
)

pytestmark = [pytest.mark.perf, pytest.mark.metal]


@pytest.fixture(scope="module")
def probe_inputs():
    if not os.environ.get("FRONTIER_EXPERIMENT_DIR") or not os.environ.get(
        "FRONTIER_STOCK_CACHE"
    ):
        pytest.skip("set FRONTIER_EXPERIMENT_DIR and FRONTIER_STOCK_CACHE")
    root = Path(os.environ["FRONTIER_EXPERIMENT_DIR"])
    library = load_library(
        "raw68k", TEMPLATE_FILES["raw68k"], (root / "extra-templates.smarts",)
    )
    stock = joblib.load(os.environ["FRONTIER_STOCK_CACHE"])
    targets = [
        line.split()[0]
        for line in (root / os.environ.get("FRONTIER_TARGETS", "targets.smi"))
        .read_text()
        .splitlines()
    ]
    return root, library, stock, targets


@pytest.mark.parametrize(
    "target,width,passes,selenium", [(15, 64, 16, True), (2, 128, 16, False)]
)
def test_remaining_target(probe_inputs, target, width, passes, selenium):
    root, library, stock, targets = probe_inputs
    extra_keys = ()
    if selenium:
        extra_path = root / "records/frontier-stock-20260920/extra-selenium-stock.smi"
        extra = cached_stock(tuple(extra_path.read_text().splitlines()), 1)
        extra_keys = extra.rdkit_keys
        stock = Stock(
            stock.wl_keys + extra.wl_keys,
            np.concatenate((stock.wl_fingerprints, extra.wl_fingerprints)),
            stock.rdkit_keys + extra.rdkit_keys,
            np.concatenate((stock.rdkit_fingerprints, extra.rdkit_fingerprints)),
            stock.row_count + extra.row_count,
            dict(Counter(stock.refusals) + Counter(extra.refusals)),
        )
    arm = metal_arm(
        library, stock, heuristic_kind="stock", scan_kind="metal", weight=1.0, n_jobs=1
    )
    selector = committed_application_leaves()
    iteration = 0

    def progress(*args, **kwargs):
        nonlocal iteration
        chosen = selector(*args, **kwargs)
        iteration += 1
        print(
            f"target={target} pass={iteration} selected={len(chosen)} seen={args[1].expansions.sid_count()}",
            flush=True,
        )
        return chosen

    print(
        f"START target={target} effort={width}x{passes} extra_stock={extra_keys}",
        flush=True,
    )
    result = run_target(
        arm, targets[target - 1], width, passes, library.smarts, selection=progress
    )
    record = {
        "target_index": target,
        "selector": "committed",
        "extra_stock": extra_keys,
        "template_count": len(library.smarts),
        "extra_template_counts": library.extra_counts,
        **asdict(result),
    }
    destination = root / "records/high-effort-20260920"
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "results.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    (destination / f"routes-{target}-{width}x{passes}.md").write_text(
        f"# Target {target}, {width}x{passes}\n\n"
        f"- Unique precursor routes: "
        f"{'at least ' if result.unique_route_count_capped else ''}"
        f"{result.unique_route_count}\n"
        f"- Replayed routes shown: {len(result.readable_route_trees)}\n"
        f"- Replay failures: {result.replay_failures}\n\n"
        + "\n\n".join(
            "```text\n" + route + "\n```"
            for route in (result.readable_route_trees or result.route_trees)
        )
    )
    print(
        "RESULT",
        {
            key: value
            for key, value in record.items()
            if key not in ("route_trees", "readable_route_trees", "matcher_totals")
        },
        flush=True,
    )
    assert result.error is None

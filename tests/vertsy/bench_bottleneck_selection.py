"""Controlled selector comparison; all arms use identical templates and stock.

Set FRONTIER_EXPERIMENT_DIR and FRONTIER_STOCK_CACHE, then run with pytest -m perf.
FRONTIER_EFFORTS accepts e.g. 16x8,20x8,4nx8 (default 32x8).
FRONTIER_RULE_SETS is a comma-separated rule-set list (default the curated medchem toolkit ahead
of Hartenfeller); FRONTIER_SELECTORS and FRONTIER_TARGET_INDICES narrow the sweep.
Results append to FRONTIER_EXPERIMENT_DIR/frontier-selection.jsonl.

**What the arms are for.** `global` ranks every reachable open leaf; `committed` is production,
which pursues the target's best live application and falls back to `global`; `exact` scores leaves
by the exact bottleneck cost rather than the additive upper bound the sweep computes.

That bound is the reason the last three arms exist. Under a max fold the scalar context charges
*every* sibling along a leaf's path where the objective charges only the worst, and the
over-charge is linear in depth -- 0.375, 0.750, 1.125, 1.500 at depths one to four on a branched
chain with one 0.375 sibling per level. So the ranking carries a depth penalty that is an artefact
of the representation, not of the chemistry, and a deep route is ranked worse than it is.

`exact` removes that artefact but needs a uniform reaction price, so it cannot be combined with a
substrate-aware one (:class:`vertsy.costs.StepPrice` refuses, deliberately, on the flag rather
than on the numbers). `weighted` only shrinks the context's contribution, which is crude but costs
nothing and stays compatible with selectivity -- which is why both are measured, not just the
exact one. `committed-exact` is the cell that was missing: AND-completion *and* an exact score.
"""

import json
import os
from dataclasses import asdict
from functools import partial
from pathlib import Path

import joblib
import pytest

from vertsy.aostar import (
    bottleneck_leaves,
    committed_application_leaves,
    global_leaves,
    weighted,
)

from .bench_host_aostar import (
    TEMPLATE_FILES,
    load_merged_library,
    metal_arm,
    parse_efforts,
    parse_inputs,
    run_target,
)

pytestmark = [pytest.mark.perf, pytest.mark.metal]

CONTEXT_WEIGHT = 0.5
"""How far to discount the context half. One is `by_total`; zero is inside-only.

A single intermediate point, because this is a probe of whether the over-charge matters at all,
not a tuning sweep. If half helps, the shape is worth fitting; if it does not, no other value of
one scalar is going to rescue it.
"""

SELECTORS = {
    "global": global_leaves,
    "committed": committed_application_leaves(),
    "exact": bottleneck_leaves,
    "committed-exact": committed_application_leaves(fill=bottleneck_leaves),
    "global-weighted": partial(global_leaves, rank=weighted(CONTEXT_WEIGHT)),
    "committed-weighted": committed_application_leaves(
        fill=partial(global_leaves, rank=weighted(CONTEXT_WEIGHT))
    ),
}


def selector_names() -> list[str]:
    chosen = os.environ.get("FRONTIER_SELECTORS") or ",".join(SELECTORS)
    names = [name.strip() for name in chosen.split(",") if name.strip()]
    unknown = [name for name in names if name not in SELECTORS]
    if unknown:
        raise ValueError(f"unknown selectors {unknown}; expected {sorted(SELECTORS)}")
    return names


def target_indices() -> list[int]:
    chosen = os.environ.get("FRONTIER_TARGET_INDICES") or "11,14"
    return [int(part) for part in chosen.split(",") if part.strip()]


@pytest.fixture(scope="module")
def inputs():
    if not os.environ.get("FRONTIER_EXPERIMENT_DIR") or not os.environ.get(
        "FRONTIER_STOCK_CACHE"
    ):
        pytest.skip("set FRONTIER_EXPERIMENT_DIR and FRONTIER_STOCK_CACHE")
    root = Path(os.environ["FRONTIER_EXPERIMENT_DIR"])
    # The curated toolkit first, so it wins any duplicate, then Hartenfeller for its lineage.
    # raw68k was what this bench originally loaded and is no longer trusted enough to compare
    # selectors through: a selector difference read off a noisy library is not a selector result.
    library = load_merged_library(
        parse_inputs(
            os.environ.get("FRONTIER_RULE_SETS") or "medchem,hartenfeller",
            TEMPLATE_FILES,
            "FRONTIER_RULE_SETS",
        ),
        (root / "extra-templates.smarts",),
    )
    stock = joblib.load(os.environ["FRONTIER_STOCK_CACHE"])
    arm = metal_arm(
        library, stock, heuristic_kind="stock", scan_kind="metal", weight=1.0, n_jobs=1
    )
    targets = [
        line.split()[0]
        for line in (root / os.environ.get("FRONTIER_TARGETS", "targets.smi"))
        .read_text()
        .splitlines()
    ]
    return root, library, arm, targets


@pytest.mark.parametrize(
    "width,passes", parse_efforts(os.environ.get("FRONTIER_EFFORTS") or "32x8")
)
@pytest.mark.parametrize("target", target_indices())
@pytest.mark.parametrize("selector", selector_names())
def test_selector_comparison(inputs, target, selector, width, passes):
    root, library, arm, targets = inputs
    result = run_target(
        arm,
        targets[target - 1],
        width,
        passes,
        library.smarts,
        selection=SELECTORS[selector],
    )
    record = {"target_index": target, "selector": selector, **asdict(result)}
    with (root / "frontier-selection.jsonl").open("a") as output:
        output.write(json.dumps(record) + "\n")
    print(
        {
            key: value
            for key, value in record.items()
            if key not in ("route_trees", "readable_route_trees", "matcher_totals")
        },
        flush=True,
    )
    assert result.error is None

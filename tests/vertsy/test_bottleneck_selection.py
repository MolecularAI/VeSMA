"""Check constrained max costs against exhaustive bounded derivation trees."""

from itertools import product

import numpy as np
import pytest

from vertsy.aostar import bottleneck_leaves, constant_step
from vertsy.costs import by_rule
from vertsy.tropical import bottleneck_leaf_scores, tropical_valuation

from .test_tropical import GRAPHS, leaf_array, prepared


def trees(node, graph, stock, depth, estimates, cost):
    if node in stock:
        return [(0.0, frozenset())]
    if depth == 0:
        return []
    if node not in graph:
        return [(estimates[node], frozenset([node]))]
    result = []
    for _, children in graph[node]:
        options = [
            trees(child, graph, stock, depth - 1, estimates, cost) for child in children
        ]
        for combination in product(*options):
            result.append(
                (
                    cost + max(item[0] for item in combination),
                    frozenset().union(*(item[1] for item in combination)),
                )
            )
    return result


@pytest.mark.parametrize(
    "graph,stock",
    [
        *GRAPHS.values(),
        ({"T": [(0, ["A", "B"]), (1, ["C"])], "C": [(2, ["A", "D"])]}, set()),
        ({"T": [(0, ["A", "B"])], "A": [(1, ["T"]), (2, ["C"])]}, set()),
    ],
)
@pytest.mark.parametrize("depth", [0, 1, 2, 4])
@pytest.mark.parametrize("cost", [0.0, 0.001, 1.0])
def test_scores_match_exhaustive_trees(graph, stock, depth, cost):
    expansions, state, target = prepared(graph, stock)
    leaves = leaf_array(state, expansions.sid_count())
    names = set(graph) | {
        child for apps in graph.values() for _, children in apps for child in children
    }
    estimates = {name: (index + 1) / 17 for index, name in enumerate(sorted(names))}
    ids = {name: int(expansions.lookup([name])[0]) for name in names}
    for name, sid in ids.items():
        if name not in graph and name not in stock:
            leaves[sid] = estimates[name]
    value = tropical_valuation(
        state,
        leaf_value=leaves,
        price=by_rule(lambda _: cost),
        max_depth=depth,
        conjoin=max,
    )
    expected = {}
    for score, open_nodes in trees(
        next(iter(graph)), graph, stock, depth, estimates, cost
    ):
        for name in open_nodes:
            sid = ids[name]
            expected[sid] = min(expected.get(sid, np.inf), score)
    assert bottleneck_leaf_scores(
        target, state, value, max_depth=depth
    ) == pytest.approx(expected)


def test_two_siblings_are_not_added_and_best_application_can_change():
    expansions, state, target = prepared({"T": [(0, ["A", "B"]), (1, ["C"])]}, set())
    ids = {name: int(expansions.lookup([name])[0]) for name in "ABC"}
    leaves = leaf_array(state, expansions.sid_count())
    cost = constant_step(0.001)
    for estimates, best in [((0.52, 0.53, 0.7), "AB"), ((0.8, 0.9, 0.7), "C")]:
        for name, estimate in zip("ABC", estimates, strict=True):
            leaves[ids[name]] = estimate
        value = tropical_valuation(
            state, leaf_value=leaves, price=cost, max_depth=4, conjoin=max
        )
        scores = bottleneck_leaf_scores(target, state, value, max_depth=4)
        assert scores[ids["A"]] == pytest.approx(max(estimates[:2]) + 0.001)
        selected = bottleneck_leaves(
            target, state, value, max_depth=4, limit=len(best), price=cost
        )
        assert set(selected) == {ids[name] for name in best}


@pytest.mark.parametrize(
    "conjoin,cost,message",
    [
        (max, by_rule(lambda rule: rule + 1.0), "uniform"),
        (max, by_rule(lambda rule: np.inf), "finite"),
        (max, by_rule(lambda rule: -1.0), "nonnegative"),
        (sum, by_rule(lambda rule: 1.0), "max fold"),
    ],
)
def test_rejects_unsupported_cost_models(conjoin, cost, message):
    import operator

    _, state, target = prepared({"T": [(0, ["A"]), (1, ["B"])]}, set())
    value = tropical_valuation(
        state,
        leaf_value=np.ones(3),
        price=cost,
        max_depth=3,
        conjoin=operator.add if conjoin is sum else conjoin,
    )
    with pytest.raises(ValueError, match=message):
        bottleneck_leaf_scores(target, state, value, max_depth=3)


def test_stock_root_terminates_despite_existing_expansion():
    expansions, state, target = prepared({"T": [(0, ["A", "B"])]}, {"T"})
    value = tropical_valuation(
        state,
        leaf_value=leaf_array(state, expansions.sid_count()),
        price=constant_step(1.0),
        max_depth=3,
        conjoin=max,
    )
    assert bottleneck_leaf_scores(target, state, value, max_depth=3) == {}


def test_bottleneck_selection_refuses_a_substrate_aware_price_before_measuring_anything():
    """Refused on the configuration, not on the numbers — or it fails on the ninth pass.

    Nothing in this graph has been measured, so every application still costs exactly 1.0 and the
    array check downstream would happily pass. The selector still has to refuse here, because the
    first mixture the probe finds would break the closed form mid-run, when a search has already
    spent its budget.
    """
    from vertsy.aostar import bottleneck_leaves
    from vertsy.costs import selective, steps_equivalent

    expansions, state, target = prepared({"T": [(0, ["A"]), (1, ["B"])]}, {"A"})
    aware = selective(constant_step(1.0), {}, {"sites": steps_equivalent})
    value = tropical_valuation(
        state,
        leaf_value=leaf_array(state, expansions.sid_count()),
        price=aware,
        max_depth=3,
        conjoin=max,
    )
    assert np.all(value.shape.app_cost == 1.0)  # the data alone would not object

    with pytest.raises(ValueError, match="uniform"):
        bottleneck_leaves(target, state, value, max_depth=3, limit=2, price=aware)


def test_global_leaves_is_unaffected_by_a_substrate_aware_price():
    """The refusal is the bottleneck selector's alone; production ranks with this one."""
    from vertsy.aostar import global_leaves
    from vertsy.costs import selective, steps_equivalent

    expansions, state, target = prepared({"T": [(0, ["A"]), (1, ["B"])]}, set())
    aware = selective(constant_step(1.0), {}, {"sites": steps_equivalent})
    value = tropical_valuation(
        state,
        leaf_value=leaf_array(state, expansions.sid_count()),
        price=aware,
        max_depth=3,
        conjoin=max,
    )
    assert global_leaves(target, state, value, max_depth=3, limit=2, price=aware)

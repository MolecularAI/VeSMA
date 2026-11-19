"""The array fold must answer what the closure fold answers.

Every test here is a comparison rather than a pinned number: the generic fold in
:mod:`vertsy.metrics` is the specification, and this module is a specialisation of it. Values are
compared with a tolerance because the array form sums an application's siblings by subtracting one
precursor from the total instead of looping over the others, which reassociates the addition.
Infinities are compared exactly, because "unreachable" is a decision rather than a quantity.

Both aggregations are checked. Summing an application's precursors makes a route's value its
reaction count; taking their maximum makes it the longest linear sequence. The array form differs
between them in the outside sweep — addition subtracts one precursor from an application's total,
maximum has no inverse and rebuilds the sibling term from prefix and suffix maxima — so the `max`
arm is testing genuinely different code, not the same code with a different operator.
"""

from __future__ import annotations

import operator
from functools import partial

import numpy as np
import pytest

from vertsy.aostar import iter_best_sid_routes
from vertsy.costs import application_price, by_rule
from vertsy.expansions import Expansions, expansions_of
from vertsy.interner import DictInterner
from vertsy.metrics import TROPICAL, outside_valuation, prepare_graph, valuation
from vertsy.store.numpy_store import NumpyExpansionStore
from vertsy.tropical import shape_of, tropical_valuation

from .handbuilt import add_application

# Every shape the fold has to get right, in one place: two ways to a target, a repeated
# precursor, a shared intermediate, a dead end, and a cycle.
GRAPHS = {
    "two-routes": (
        {"T": [(0, ["A", "B"]), (1, ["C"])], "C": [(2, ["D"])]},
        {"A", "B", "D"},
    ),
    "open-leaf": ({"T": [(0, ["A", "U"]), (1, ["C"])], "C": [(2, ["V"])]}, {"A"}),
    "repeated": ({"T": [(0, ["U", "U"])]}, set()),
    "diamond": (
        {"T": [(0, ["L", "R"])], "L": [(1, ["S"])], "R": [(2, ["S"])]},
        {"S"},
    ),
    "dead-end": ({"T": [(0, ["X"])], "X": []}, set()),
    "cycle": ({"X": [(0, ["Y"])], "Y": [(1, ["X"])]}, set()),
    "convergent": (
        {"T": [(0, ["M", "M"])], "M": [(1, ["S"]), (2, ["S", "S"])]},
        {"S"},
    ),
    # Arity three, and every position of it distinct. `leave_one_out_max` handles the first and
    # last of an application by one accumulate each and the *middle* by combining both, so without
    # a three-precursor application that branch is never executed. Two of the three are open leaves
    # and one is stock, so the leave-one-out answer differs at each position.
    "three-way": (
        {"T": [(0, ["A", "U", "V"])], "U": [(1, ["W"])]},
        {"A"},
    ),
}
ESTIMATE = 0.375  # a value no sum of unit costs can accidentally equal


UNIT = by_rule(lambda rule: 1.0)
# A non-unit prior, so route costs stop being small exactly-representable integers.
PRICED = by_rule(lambda rule: 1.0 + rule / 7.0)


def scalar_cost(price):
    """Drive the closure fold through the same price the array fold used.

    The two folds only mean the same thing if they are handed the same cost model, and
    ``rule_cost`` cannot express a substrate-aware one at all.
    """
    return partial(application_price, price)


def prepared(graph, stock):
    expansions = expansions_of(graph)
    state = prepare_graph(expansions, stock)
    root = next(iter(graph))
    return expansions, state, int(expansions.lookup([root])[0])


def leaf_array(state, count: int) -> np.ndarray:
    """A dense estimate for every unexpanded, non-stock sid — what the search hands over."""
    leaves = np.full(count, TROPICAL.zero, dtype=np.float64)
    for sid in range(count):
        if sid not in state.steps and sid not in state.stock_sids:
            leaves[sid] = ESTIMATE
    return leaves


@pytest.mark.parametrize("name", sorted(GRAPHS))
@pytest.mark.parametrize("price", [UNIT, PRICED], ids=["unit", "priced"])
@pytest.mark.parametrize("max_depth", [0, 1, 3, 8])
@pytest.mark.parametrize("conjoin", [operator.add, max], ids=["sum", "max"])
def test_inside_matches_the_generic_fold(name, price, max_depth, conjoin) -> None:
    graph, stock = GRAPHS[name]
    expansions, state, _ = prepared(graph, stock)
    count = expansions.sid_count()
    leaves = leaf_array(state, count)

    scalar = valuation(
        state,
        TROPICAL,
        leaf_value=lambda sid: float(leaves[sid]),
        step_cost=scalar_cost(price),
        conjoin=conjoin,
    )
    array = tropical_valuation(
        state,
        leaf_value=leaves,
        price=price,
        max_depth=max_depth,
        conjoin=conjoin,
    )

    for sid in range(count):
        for depth in range(max_depth + 1):
            want, got = scalar.node(sid, depth), array.node(sid, depth)
            assert np.isinf(want) == np.isinf(got), (sid, depth, want, got)
            if not np.isinf(want):
                assert got == pytest.approx(want), (sid, depth)


@pytest.mark.parametrize("name", sorted(GRAPHS))
@pytest.mark.parametrize("price", [UNIT, PRICED], ids=["unit", "priced"])
@pytest.mark.parametrize("max_depth", [1, 3, 8])
@pytest.mark.parametrize("conjoin", [operator.add, max], ids=["sum", "max"])
def test_open_leaves_match_the_generic_sweep(name, price, max_depth, conjoin) -> None:
    from vertsy.tropical import outside_valuation_of

    graph, stock = GRAPHS[name]
    expansions, state, target = prepared(graph, stock)
    leaves = leaf_array(state, expansions.sid_count())

    scalar_value = valuation(
        state,
        TROPICAL,
        leaf_value=lambda sid: float(leaves[sid]),
        step_cost=scalar_cost(price),
        conjoin=conjoin,
    )
    scalar = outside_valuation(
        target, state, scalar_value, max_depth=max_depth, step_cost=scalar_cost(price)
    )
    array_value = tropical_valuation(
        state,
        leaf_value=leaves,
        price=price,
        max_depth=max_depth,
        conjoin=conjoin,
    )
    array = outside_valuation_of(target, state, array_value, max_depth=max_depth)

    # The same leaves are reachable, at the same totals and the same contexts. The attaining
    # horizon may differ on a tie, which is documented and read by nobody.
    assert set(array.leaves) == set(scalar.leaves), name
    for sid, (total, context, _) in scalar.leaves.items():
        got_total, got_context, _ = array.leaves[sid]
        assert got_total == pytest.approx(total), (name, sid)
        assert got_context == pytest.approx(context), (name, sid)


@pytest.mark.parametrize("name", sorted(GRAPHS))
@pytest.mark.parametrize("conjoin", [operator.add, max], ids=["sum", "max"])
def test_delta_matches_the_generic_sweep_entry_for_entry(name, conjoin) -> None:
    from vertsy.tropical import outside_valuation_of

    graph, stock = GRAPHS[name]
    expansions, state, target = prepared(graph, stock)
    leaves = leaf_array(state, expansions.sid_count())
    scalar_value = valuation(
        state,
        TROPICAL,
        leaf_value=lambda sid: float(leaves[sid]),
        step_cost=scalar_cost(UNIT),
        conjoin=conjoin,
    )
    array_value = tropical_valuation(
        state, leaf_value=leaves, price=UNIT, max_depth=4, conjoin=conjoin
    )

    scalar = outside_valuation(
        target, state, scalar_value, max_depth=4, step_cost=scalar_cost(UNIT)
    )
    array = outside_valuation_of(target, state, array_value, max_depth=4)

    # `delta` holds only finite entries on both sides, so the key sets must agree exactly: a
    # disagreement would mean one of them thinks a product is reachable and the other does not.
    assert set(array.delta) == set(scalar.delta), name
    for key, context in scalar.delta.items():
        assert array.delta[key] == pytest.approx(context), (name, key)


def test_a_disconnection_with_several_templates_takes_the_cheapest() -> None:
    """Min-plus over an application's templates, which is the one place `app_cost` is decided."""
    expansions: Expansions[str] = Expansions(NumpyExpansionStore(), DictInterner())
    target, stock = expansions.intern(["T", "S"]).tolist()
    expansions.mark_expanded(np.array([target], dtype=np.int64))
    app = add_application(expansions, target, 5, [stock])
    add_application(expansions, target, 2, [stock], app=app)  # cheaper under `priced`
    state = prepare_graph(expansions, {"S"})

    array = tropical_valuation(
        state,
        leaf_value=np.zeros(expansions.sid_count()),
        price=PRICED,
        max_depth=2,
    )
    assert array.node(target, 2) == pytest.approx(1.0 + 2 / 7.0)


def test_extraction_labels_a_step_with_the_template_that_set_its_price() -> None:
    """The fold prices a disconnection at its cheapest template, so the route must name that one.

    Rule 5 is listed first and rule 2 is cheaper under `priced`. Labelling by lowest id would
    report a route whose stated chemistry costs more than the value the search chose it for.
    """
    expansions: Expansions[str] = Expansions(NumpyExpansionStore(), DictInterner())
    target, stock = expansions.intern(["T", "S"]).tolist()
    expansions.mark_expanded(np.array([target], dtype=np.int64))
    app = add_application(expansions, target, 2, [stock])
    add_application(expansions, target, 5, [stock], app=app)
    state = prepare_graph(expansions, {"S"})
    assert state.applications(target) == (((2, 5), (stock,)),)  # rule 2 is listed first

    # Priced so that the *second* template is the cheap one, which is the case a lowest-id label
    # gets wrong: cost 5 - 3 = 2 against rule 2's 1 + 2/7.
    def inverted(rule: int) -> float:
        return float(5 - rule)

    inverted_price = by_rule(inverted)

    value = tropical_valuation(
        state,
        leaf_value=np.zeros(expansions.sid_count()),
        price=inverted_price,
        max_depth=2,
    )
    (route,) = iter_best_sid_routes(target, state, value, 2, price=inverted_price)

    assert value.node(target, 2) == pytest.approx(inverted(5))
    assert route.rule == 5  # the attaining template, not `rules[0]`


def test_shape_rejects_a_gap_in_the_application_ids() -> None:
    """Dense ids are what makes `bincount` a valid CSR offset; a gap would silently misgroup."""
    expansions: Expansions[str] = Expansions(NumpyExpansionStore(), DictInterner())
    target, stock = expansions.intern(["T", "S"]).tolist()
    # Written straight past `add_layer`, which is the only way to produce this: app ids 0..2 have
    # no rows, so a `bincount` offset would put app 3's precursors at the wrong slice.
    expansions.store.add_edges(
        np.array([target], dtype=np.int64),
        np.array([3], dtype=np.int64),
        np.array([stock], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="dense"):
        shape_of(expansions, UNIT)


def test_an_empty_graph_folds_to_the_target_estimate_alone() -> None:
    """The first pass has no applications at all, which every array operation has to survive."""
    expansions: Expansions[str] = Expansions(NumpyExpansionStore(), DictInterner())
    (target,) = expansions.intern(["T"]).tolist()
    state = prepare_graph(expansions, set())
    leaves = np.array([ESTIMATE])

    from vertsy.tropical import outside_valuation_of

    array = tropical_valuation(state, leaf_value=leaves, price=UNIT, max_depth=4)
    assert array.node(target, 4) == pytest.approx(ESTIMATE)
    outside = outside_valuation_of(target, state, array, max_depth=4)
    assert outside.leaves == {target: (pytest.approx(ESTIMATE), 0.0, 4)}

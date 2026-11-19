from __future__ import annotations

import numpy as np

from tests.vertsy.handbuilt import add_application
from vertsy.expansions import Expansions, materialize, sid_steps
from vertsy.interner import DictInterner
from vertsy.routes import (
    Route,
    iter_solved_sid_routes,
    leaves,
    reaction_count,
    reaction_depth,
    solved_sids,
)
from vertsy.store.numpy_store import NumpyExpansionStore


def fresh_expansions() -> Expansions[str]:
    return Expansions(NumpyExpansionStore(), DictInterner())


def test_expansions_steps_group_append_rows_into_tuple_sid_steps() -> None:
    expansions = fresh_expansions()
    target, left, right = expansions.intern(["target", "left", "right"]).tolist()
    expansions.mark_expanded(np.array([target], dtype=np.int64))
    add_application(expansions, target, 7, [left, right])

    assert expansions.steps(target) == (((7,), (left, right)),)


def test_sid_steps_returns_requested_products_without_edges() -> None:
    expansions = fresh_expansions()
    leaf = expansions.intern(["leaf"])[0]
    expansions.mark_expanded(np.array([leaf], dtype=np.int64))

    assert sid_steps(expansions, np.array([leaf], dtype=np.int64)) == {leaf: ()}
    assert expansions.steps(leaf) == ()


def test_materialize_preserves_key_space_debug_view() -> None:
    expansions = fresh_expansions()
    target, left, right = expansions.intern(["target", "left", "right"]).tolist()
    expansions.mark_expanded(np.array([target], dtype=np.int64))
    add_application(expansions, target, 7, [left, right])

    assert materialize(expansions) == {
        "target": frozenset({(7, frozenset({"left", "right"}))})
    }


def test_solved_sid_routes_keep_repeated_precursor_occurrences_independent() -> None:
    expansions = fresh_expansions()
    target, repeated, stock_a, stock_b = expansions.intern(
        ["target", "repeated", "stock-a", "stock-b"]
    ).tolist()
    expansions.mark_expanded(np.array([target, repeated], dtype=np.int64))
    add_application(expansions, target, 0, [repeated, repeated])
    add_application(expansions, repeated, 1, [stock_a])
    add_application(expansions, repeated, 2, [stock_b])

    solved = solved_sids(expansions, np.array([stock_a, stock_b], dtype=np.int64))
    routes = tuple(
        iter_solved_sid_routes(
            target,
            expansions,
            solved,
            frozenset({stock_a, stock_b}),
        )
    )

    assert len(routes) == 4
    assert {tuple(child.rule for child in route.precursors) for route in routes} == {
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    }


def test_tree_metrics_distinguish_reaction_count_and_reaction_depth() -> None:
    route = Route(
        "target",
        0,
        (
            Route("left", 1, (Route("stock-a", None),)),
            Route("right", 2, (Route("stock-b", None),)),
        ),
    )

    assert reaction_count(route) == 3
    assert reaction_depth(route) == 2
    assert leaves(route) == frozenset({"stock-a", "stock-b"})


def test_solved_sid_routes_reject_ancestry_cycles() -> None:
    expansions = fresh_expansions()
    left, right = expansions.intern(["left", "right"]).tolist()
    expansions.mark_expanded(np.array([left, right], dtype=np.int64))
    add_application(expansions, left, 0, [right])
    add_application(expansions, right, 1, [left])

    solved = np.ones(expansions.sid_count(), dtype=np.bool_)

    assert tuple(iter_solved_sid_routes(left, expansions, solved, frozenset())) == ()

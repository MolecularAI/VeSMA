from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tests.vertsy.handbuilt import add_application
from vertsy import Route, is_solved, leaves, reaction_count, reaction_depth
from vertsy.expansions import Expansions, expansions_of
from vertsy.interner import DictInterner
from vertsy.routes import (
    SidRoute,
    iter_solved_routes,
    pretty_string,
    sid_route_combinations,
    solved_targets,
)
from vertsy.store.numpy_store import NumpyExpansionStore


@dataclass(frozen=True)
class Token:
    name: str


def token(name: str) -> Token:
    return Token(name)


def route_signature(route: Route[Token]) -> tuple[str, int | None, tuple[object, ...]]:
    return (
        route.product.name,
        route.rule,
        tuple(sorted((route_signature(child) for child in route.precursors), key=repr)),
    )


def route_signatures(routes: frozenset[Route[Token]]) -> set[object]:
    return {route_signature(route) for route in routes}


def solved_routes(
    target: Token,
    expansions,
    stock: set[Token],
) -> frozenset[Route[Token]]:
    return frozenset(iter_solved_routes(target, expansions, stock))


def test_route_tree_is_hashable_value() -> None:
    s = token("s")
    x = token("x")
    route = Route(x, 0, (Route(s, None),))

    assert route == Route(x, 0, (Route(s, None),))
    assert len({route, Route(x, 0, (Route(s, None),))}) == 1


def test_iter_solved_routes_for_missing_target_yields_nothing() -> None:
    target = token("target")
    assert solved_routes(target, expansions_of({}), set()) == frozenset()


def test_iter_solved_routes_for_dead_end_target_yields_nothing() -> None:
    target = token("target")
    assert solved_routes(target, expansions_of({target: frozenset()}), set()) == (
        frozenset()
    )


def test_iter_solved_routes_for_simple_chain_yields_one_route() -> None:
    c = token("C")
    cc = token("CC")
    ccc = token("CCC")
    expansions = expansions_of(
        {
            ccc: frozenset({(0, frozenset({cc, c}))}),
            cc: frozenset({(1, frozenset({c}))}),
        }
    )

    route_set = solved_routes(ccc, expansions, {c})

    assert route_signatures(route_set) == {
        (
            "CCC",
            0,
            (
                ("C", None, ()),
                ("CC", 1, (("C", None, ()),)),
            ),
        )
    }


def test_iter_solved_routes_for_multiple_steps_yields_multiple_routes() -> None:
    target = token("target")
    a = token("a")
    b = token("b")
    expansions = expansions_of(
        {
            target: frozenset(
                {
                    (0, frozenset({a})),
                    (1, frozenset({b})),
                }
            )
        }
    )

    route_set = solved_routes(target, expansions, {a, b})

    assert route_signatures(route_set) == {
        ("target", 0, (("a", None, ()),)),
        ("target", 1, (("b", None, ()),)),
    }


def test_iter_solved_routes_can_limit_reaction_depth() -> None:
    expansions = expansions_of(
        {
            "target": frozenset({(0, frozenset({"short"})), (1, frozenset({"mid"}))}),
            "mid": frozenset({(2, frozenset({"deep"}))}),
        }
    )

    routes = tuple(
        iter_solved_routes("target", expansions, {"short", "deep"}, max_depth=1)
    )

    assert routes == (Route("target", 0, (Route("short", None),)),)


def test_bounded_routes_round_robin_across_disconnections() -> None:
    expansions = Expansions(NumpyExpansionStore(), DictInterner())
    target, branch, stock_a, stock_b, stock_c = expansions.intern(
        ["target", "branch", "stock-a", "stock-b", "stock-c"]
    ).tolist()
    expansions.mark_expanded(np.array([target, branch], dtype=np.int64))
    add_application(expansions, target, 0, [branch])
    add_application(expansions, target, 1, [stock_c])
    add_application(expansions, branch, 2, [stock_a])
    add_application(expansions, branch, 3, [stock_b])

    routes = tuple(
        iter_solved_routes(
            "target",
            expansions,
            {"stock-a", "stock-b", "stock-c"},
            cap=3,
            max_depth=2,
        )
    )

    assert tuple(route.rule for route in routes) == (0, 1, 0)
    assert tuple(route.precursors[0].product for route in routes) == (
        "branch",
        "stock-c",
        "branch",
    )


def test_route_cartesian_product_yields_without_pooling_child_route_spaces() -> None:
    generated: list[int] = []

    def options(precursor: int):
        for choice in range(1_000):
            generated.append(precursor)
            yield SidRoute(choice, None)

    first = next(sid_route_combinations((10, 20), options))

    assert first == (SidRoute(0, None), SidRoute(0, None))
    assert generated == [10, 20]


def test_repeated_precursor_occurrences_choose_routes_independently() -> None:
    target = token("target")
    x = token("x")
    s1 = token("s1")
    s2 = token("s2")
    expansions = Expansions(NumpyExpansionStore(), DictInterner())
    target_sid, x_sid, s1_sid, s2_sid = expansions.intern([target, x, s1, s2]).tolist()
    expansions.mark_expanded(np.array([target_sid, x_sid], dtype=np.int64))
    add_application(expansions, target_sid, 0, [x_sid, x_sid])
    add_application(expansions, x_sid, 1, [s1_sid])
    add_application(expansions, x_sid, 2, [s2_sid])

    route_set = solved_routes(target, expansions, {s1, s2})

    assert route_signatures(route_set) == {
        (
            "target",
            0,
            (("x", 1, (("s1", None, ()),)), ("x", 1, (("s1", None, ()),))),
        ),
        (
            "target",
            0,
            (("x", 1, (("s1", None, ()),)), ("x", 2, (("s2", None, ()),))),
        ),
        (
            "target",
            0,
            (("x", 2, (("s2", None, ()),)), ("x", 2, (("s2", None, ()),))),
        ),
    }


def test_cycle_does_not_recurse_forever_and_yields_no_routes() -> None:
    a = token("A")
    b = token("B")
    expansions = expansions_of(
        {
            a: frozenset({(0, frozenset({b}))}),
            b: frozenset({(1, frozenset({a}))}),
        }
    )

    assert solved_routes(a, expansions, set()) == frozenset()


def test_cycle_with_alternative_leaf_route_still_yields_route() -> None:
    a = token("A")
    b = token("B")
    s = token("s")
    expansions = expansions_of(
        {
            a: frozenset(
                {
                    (0, frozenset({b})),
                    (1, frozenset({s})),
                }
            ),
            b: frozenset({(2, frozenset({a}))}),
        }
    )

    route_set = solved_routes(b, expansions, {s})

    assert route_signatures(route_set) == {("B", 2, (("A", 1, (("s", None, ()),)),))}


def test_solved_targets_returns_requested_solved_products_only() -> None:
    target = token("target")
    mid = token("mid")
    stock = token("stock")
    dead = token("dead")
    expansions = expansions_of(
        {
            target: frozenset({(0, frozenset({mid})), (1, frozenset({dead}))}),
            mid: frozenset({(2, frozenset({stock}))}),
            dead: frozenset(),
        }
    )

    assert solved_targets({target, mid, dead}, expansions, {stock}) == {target, mid}


def test_leaves_is_solved_and_metrics_match_tree_definition() -> None:
    c = token("C")
    cc = token("CC")
    o = token("O")
    ccc = token("CCC")
    route = Route(
        ccc,
        0,
        (
            Route(cc, 1, (Route(c, None),)),
            Route(o, 2, (Route(c, None),)),
        ),
    )

    assert leaves(route) == frozenset({c})
    assert is_solved(route, {c})
    assert not is_solved(route, set())
    assert reaction_count(route) == 3
    assert reaction_depth(route) == 2


def test_pretty_string() -> None:
    route = Route(
        "M1",
        1,
        (
            Route("M2", 2, (Route("M4", None),)),
            Route("M3", None),
        ),
    )

    assert pretty_string(route) == "M1\n  1\n    M2\n      2\n        M4\n    M3"

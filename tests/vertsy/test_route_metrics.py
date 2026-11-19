from __future__ import annotations

import sys

import numpy as np

from tests.vertsy.handbuilt import add_application
from vertsy.expansions import Expansions, expansions_of
from vertsy.interner import DictInterner
from vertsy.metrics import (
    compare_routes,
    library,
    loose,
    min_route_depth,
    precursor_route_count_bounded,
    precursor_route_counts_by_first_disconnection,
    precursor_route_depth_counter,
    precursor_route_depth_profile,
    precursors_recovered,
    prepare_graph,
    route_solved,
    solved_disconnection_count,
    solved_route_count,
    solved_route_count_bounded,
    starting_material_recovered,
    starting_material_route_count_bounded,
    strict,
)
from vertsy.routes import (
    Route,
    count_recoveries,
    iter_solved_routes,
    library_route_matches,
    loose_route_matches,
    strict_route_matches,
)
from vertsy.store.numpy_store import NumpyExpansionStore


def test_strict_loose_library_match_materialized_comparators() -> None:
    target = "target"
    mid = "mid"
    stock = "stock"
    expansions = expansions_of(
        {
            target: frozenset({(0, frozenset({mid}))}),
            mid: frozenset({(1, frozenset({stock}))}),
        }
    )
    reference = Route(target, 0, (Route(mid, 1, (Route(stock, None),)),))
    routes = frozenset(iter_solved_routes(target, expansions, {stock}))

    assert compare_routes(expansions, {stock}, strict(reference)) == count_recoveries(
        reference, routes, strict_route_matches
    )
    assert compare_routes(expansions, {stock}, loose(reference)) == count_recoveries(
        reference, routes, loose_route_matches
    )
    assert compare_routes(expansions, {stock}, library(reference)) == count_recoveries(
        reference, routes, library_route_matches
    )


def test_loose_allows_different_intermediate_with_same_leaf_and_skeleton() -> None:
    target = "target"
    generated_mid = "generated-mid"
    reference_mid = "reference-mid"
    stock = "stock"
    expansions = expansions_of(
        {
            target: frozenset({(0, frozenset({generated_mid}))}),
            generated_mid: frozenset({(1, frozenset({stock}))}),
        }
    )
    reference = Route(target, 0, (Route(reference_mid, 1, (Route(stock, None),)),))

    assert compare_routes(expansions, {stock}, strict(reference)) == 0
    assert compare_routes(expansions, {stock}, loose(reference)) == 1
    assert compare_routes(expansions, {stock}, library(reference)) == 1


def test_library_allows_different_leaf_set() -> None:
    target = "target"
    mid = "mid"
    generated_stock = "generated-stock"
    reference_stock = "reference-stock"
    expansions = expansions_of(
        {
            target: frozenset({(0, frozenset({mid}))}),
            mid: frozenset({(1, frozenset({generated_stock}))}),
        }
    )
    reference = Route(target, 0, (Route(mid, 1, (Route(reference_stock, None),)),))

    assert compare_routes(expansions, {generated_stock}, loose(reference)) == 0
    assert compare_routes(expansions, {generated_stock}, library(reference)) == 1


def test_missing_rule_id_is_not_recovered() -> None:
    target = "target"
    stock = "stock"
    expansions = expansions_of({target: frozenset({(0, frozenset({stock}))})})
    reference = Route(target, -1, (Route(stock, None),))

    assert compare_routes(expansions, {stock}, strict(reference)) == 0
    assert compare_routes(expansions, {stock}, loose(reference)) == 0
    assert compare_routes(expansions, {stock}, library(reference)) == 0


def test_cycles_terminate_for_count_and_min_depth() -> None:
    a = "a"
    b = "b"
    stock = "stock"
    expansions = expansions_of(
        {
            a: frozenset({(0, frozenset({b})), (2, frozenset({stock}))}),
            b: frozenset({(1, frozenset({a}))}),
        }
    )

    assert compare_routes(expansions, {stock}, solved_route_count(b, cap=10)) == 1
    assert compare_routes(expansions, {stock}, min_route_depth(b)) == 2


def test_cycle_only_graph_is_unsolved() -> None:
    a = "a"
    b = "b"
    expansions = expansions_of(
        {
            a: frozenset({(0, frozenset({b}))}),
            b: frozenset({(1, frozenset({a}))}),
        }
    )

    assert compare_routes(expansions, set(), solved_route_count(a, cap=10)) == 0
    assert compare_routes(expansions, set(), min_route_depth(a)) is None


def test_solved_route_count_saturates_for_repeated_precursor_choices() -> None:
    expansions = Expansions(NumpyExpansionStore(), DictInterner())
    target, repeated, stock_a, stock_b = expansions.intern(
        ["target", "repeated", "stock-a", "stock-b"]
    ).tolist()
    expansions.mark_expanded(np.array([target, repeated], dtype=np.int64))
    add_application(expansions, target, 0, [repeated, repeated])
    add_application(expansions, repeated, 1, [stock_a])
    add_application(expansions, repeated, 2, [stock_b])

    assert (
        compare_routes(
            expansions,
            {"stock-a", "stock-b"},
            solved_route_count("target", cap=3),
        )
        == 3
    )


def test_precursor_route_count_bounded_collapses_duplicate_reactions() -> None:
    # Two rules disconnect target into the SAME precursor set {mid}; mid -> stock.
    expansions = expansions_of(
        {
            "target": frozenset({(0, frozenset({"mid"})), (1, frozenset({"mid"}))}),
            "mid": frozenset({(2, frozenset({"stock"}))}),
        }
    )
    # The rule-aware count counts both reactions; the precursors comparator collapses them.
    assert (
        compare_routes(
            expansions,
            {"stock"},
            solved_route_count_bounded("target", cap=10, max_depth=2),
        )
        == 2
    )
    assert (
        compare_routes(
            expansions,
            {"stock"},
            precursor_route_count_bounded("target", cap=10, max_depth=2),
        )
        == 1
    )
    # The depth bound still applies: a two-reaction route needs max_depth >= 2.
    assert (
        compare_routes(
            expansions,
            {"stock"},
            precursor_route_count_bounded("target", cap=10, max_depth=1),
        )
        == 0
    )


def test_first_disconnection_counts_share_the_total_and_collapse_rules() -> None:
    expansions = expansions_of(
        {
            "target": frozenset(
                {
                    (0, frozenset({"a"})),
                    (1, frozenset({"a"})),
                    (2, frozenset({"b"})),
                }
            ),
            "a": frozenset({(3, frozenset({"stock-a"})), (4, frozenset({"stock-b"}))}),
            "b": frozenset({(5, frozenset({"stock-a"}))}),
        }
    )
    stock = {"stock-a", "stock-b"}
    counts = compare_routes(
        expansions,
        stock,
        precursor_route_counts_by_first_disconnection("target", cap=100, max_depth=2),
    )
    assert counts == ((("a",), 2), (("b",), 1))
    assert sum(count for _, count in counts) == compare_routes(
        expansions,
        stock,
        precursor_route_count_bounded("target", cap=100, max_depth=2),
    )
    profile = compare_routes(
        expansions,
        stock,
        precursor_route_depth_profile("target", cap=100, extra_depths=2),
    )
    for depth_index, count in enumerate(profile.counts):
        assert count.at_most == sum(
            root_counts[depth_index].at_most
            for _, root_counts in profile.first_disconnections
        )


def test_shortest_depth_counts_include_longer_shallow_branch_options() -> None:
    expansions = expansions_of(
        {
            "target": frozenset({(0, frozenset({"a", "b"}))}),
            "a": frozenset({(1, frozenset({"stock"})), (2, frozenset({"x"}))}),
            "b": frozenset({(3, frozenset({"y"}))}),
            "x": frozenset({(4, frozenset({"stock"}))}),
            "y": frozenset({(5, frozenset({"stock"}))}),
        }
    )
    state = prepare_graph(expansions, {"stock"})
    profile = precursor_route_depth_profile("target", cap=100)(state)
    assert profile.shortest_depth == 3
    assert profile.counts[0].exact == 2
    assert profile.first_disconnections[0][1][0].exact == 2
    target_sid = int(expansions.lookup(["target"])[0])
    counter = precursor_route_depth_counter(state, cap=100)
    for depth in range(5):
        assert counter.node(target_sid, depth) == precursor_route_count_bounded(
            "target", cap=100, max_depth=depth
        )(state)


def test_depth_profile_handles_stock_cycles_and_saturation() -> None:
    expansions = expansions_of(
        {
            "target": frozenset(
                {
                    (0, frozenset({"stock"})),
                    (1, frozenset({"stock"})),
                    (2, frozenset({"loop"})),
                }
            ),
            "loop": frozenset({(3, frozenset({"target"})), (4, frozenset({"stock"}))}),
        }
    )
    state = prepare_graph(expansions, {"stock"})
    profile = precursor_route_depth_profile("target", cap=2)(state)
    assert profile.shortest_depth == 1
    assert profile.counts[0].exact == 1  # Duplicate template IDs collapse.
    assert profile.counts[1].exact is None  # The cumulative count reached the cap.
    assert profile.counts[1].capped
    stock_profile = precursor_route_depth_profile("stock", cap=100)(state)
    assert stock_profile.shortest_depth == 0
    assert stock_profile.counts[0].exact == 1
    assert precursor_route_depth_profile("missing", cap=100)(state).counts == ()


def test_reference_agnostic_recovery_metrics() -> None:
    # Two rule-distinct routes: target -> a -> s1 and target -> b -> s2.
    expansions = expansions_of(
        {
            "target": frozenset({(0, frozenset({"a"})), (1, frozenset({"b"}))}),
            "a": frozenset({(2, frozenset({"s1"}))}),
            "b": frozenset({(3, frozenset({"s2"}))}),
        }
    )
    stock = {"s1", "s2"}

    assert compare_routes(expansions, stock, route_solved("target")) == 1
    assert compare_routes(expansions, stock, route_solved("missing")) == 0

    # Starting materials: each single-route leaf set is recovered; their union is not (no one
    # route uses both), and an out-of-graph leaf is not.
    sm = starting_material_recovered
    assert compare_routes(expansions, stock, sm("target", {"s1"}, 2)) == 1
    assert compare_routes(expansions, stock, sm("target", {"s2"}, 2)) == 1
    assert compare_routes(expansions, stock, sm("target", {"s1", "s2"}, 2)) == 0
    assert compare_routes(expansions, stock, sm("target", {"missing"}, 2)) == 0
    # Depth bound: the leaves are two reactions deep.
    assert compare_routes(expansions, stock, sm("target", {"s1"}, 1)) == 0

    # Precursors (intermediates + leaves) must match exactly.
    pr = precursors_recovered
    assert compare_routes(expansions, stock, pr("target", {"a", "s1"}, 2)) == 1
    assert compare_routes(expansions, stock, pr("target", {"b", "s2"}, 2)) == 1
    assert compare_routes(expansions, stock, pr("target", {"a", "s2"}, 2)) == 0
    assert compare_routes(expansions, stock, pr("target", {"a", "s1", "s2"}, 2)) == 0
    assert compare_routes(expansions, stock, pr("target", {"a", "s1"}, 1)) == 0


def test_starting_material_route_count_bounded_counts_distinct_leaf_sets() -> None:
    # Two routes to different starting materials -> two distinct leaf-sets.
    distinct = expansions_of(
        {
            "target": frozenset({(0, frozenset({"a"})), (1, frozenset({"b"}))}),
            "a": frozenset({(2, frozenset({"s1"}))}),
            "b": frozenset({(3, frozenset({"s2"}))}),
        }
    )
    assert (
        compare_routes(
            distinct,
            {"s1", "s2"},
            starting_material_route_count_bounded("target", cap=10, max_depth=2),
        )
        == 2
    )

    # Two rule-distinct routes that reach the SAME starting material collapse to one leaf-set.
    shared = expansions_of(
        {
            "target": frozenset({(0, frozenset({"a"})), (1, frozenset({"b"}))}),
            "a": frozenset({(2, frozenset({"s"}))}),
            "b": frozenset({(3, frozenset({"s"}))}),
        }
    )
    assert (
        compare_routes(
            shared, {"s"}, solved_route_count_bounded("target", cap=10, max_depth=2)
        )
        == 2
    )
    assert (
        compare_routes(
            shared,
            {"s"},
            starting_material_route_count_bounded("target", cap=10, max_depth=2),
        )
        == 1
    )


def test_solved_route_count_bounded_respects_max_depth() -> None:
    expansions = expansions_of(
        {
            "target": frozenset({(0, frozenset({"mid"}))}),
            "mid": frozenset({(1, frozenset({"stock"}))}),
        }
    )

    assert (
        compare_routes(
            expansions,
            {"stock"},
            solved_route_count_bounded("target", cap=10, max_depth=1),
        )
        == 0
    )
    assert (
        compare_routes(
            expansions,
            {"stock"},
            solved_route_count_bounded("target", cap=10, max_depth=2),
        )
        == 1
    )


def test_min_route_depth_handles_stock_and_missing_targets() -> None:
    expansions = expansions_of({"target": frozenset({(0, frozenset({"stock"}))})})

    assert compare_routes(expansions, {"stock"}, min_route_depth("stock")) == 0
    assert compare_routes(expansions, {"stock"}, min_route_depth("target")) == 1
    assert compare_routes(expansions, {"stock"}, min_route_depth("missing")) is None


def _recovered(reference, routes, matches) -> int:
    """Existence (0/1) of a materialized route matching under ``matches``.

    The graph comparators are existence checks, so they equal ``count_recoveries > 0`` — not
    the raw count, which can exceed 1 when several routes match loosely.
    """
    return int(count_recoveries(reference, routes, matches) > 0)


def test_comparators_match_materialized_on_branching_graph() -> None:
    target, mid, stock_a, stock_b = "target", "mid", "stock-a", "stock-b"
    expansions = expansions_of(
        {
            target: frozenset({(0, frozenset({mid}))}),
            # Two applications with the same rule but different leaves: same skeleton, so
            # both routes match the reference under library, only one under loose.
            mid: frozenset({(1, frozenset({stock_a})), (1, frozenset({stock_b}))}),
        }
    )
    stock = {stock_a, stock_b}
    reference = Route(target, 0, (Route(mid, 1, (Route(stock_a, None),)),))
    routes = frozenset(iter_solved_routes(target, expansions, stock))

    assert len(routes) == 2
    assert count_recoveries(reference, routes, library_route_matches) == 2

    for metric, matches in (
        (strict(reference), strict_route_matches),
        (loose(reference), loose_route_matches),
        (library(reference), library_route_matches),
    ):
        assert compare_routes(expansions, stock, metric) == _recovered(
            reference, routes, matches
        )

    assert compare_routes(expansions, stock, solved_route_count(target, cap=10)) == len(
        routes
    )
    assert compare_routes(expansions, stock, min_route_depth(target)) == 2


def test_whole_graph_metrics_are_memoized_not_exponential() -> None:
    # A diamond chain: each n_i reaches n_{i+1} by two applications, so there are 2**k solved
    # routes over only k products. Trail recursion would be O(2**k) (a hang); the memoized
    # metrics are O(k). This is the shape that made the real depth-2 recovery graph explode.
    k = 30
    names = [f"n{i}" for i in range(k)] + ["stock"]
    expansions = Expansions(NumpyExpansionStore(), DictInterner())
    sids = expansions.intern(names).tolist()
    expansions.mark_expanded(np.array(sids[:k], dtype=np.int64))

    for i in range(k):
        # Two *distinct* disconnections would collapse, since they name the same precursor. Two
        # templates on one disconnection is the honest way to double the rule-aware count here.
        app = add_application(expansions, sids[i], 2 * i, [sids[i + 1]])
        add_application(expansions, sids[i], 2 * i + 1, [sids[i + 1]], app=app)

    assert (
        compare_routes(expansions, {"stock"}, solved_route_count("n0", cap=100)) == 100
    )
    assert (
        compare_routes(
            expansions,
            {"stock"},
            solved_route_count_bounded("n0", cap=100, max_depth=10),
        )
        == 0
    )
    assert compare_routes(expansions, {"stock"}, min_route_depth("n0")) == k


def test_min_route_depth_reports_a_depth_wherever_the_graph_is_solved() -> None:
    # `target <- p <- a <- stock`, with `a <- p` closing a cycle, and a dead first application on
    # the target whose only job is to make the walk reach `a` before `p`. A single postorder pass
    # then cuts `p` as a back edge and leaves it valueless, which propagates: a measured Retro*
    # target printed 12 reactions and no depth at all. This function and `is_solved` must not be
    # able to disagree about whether a graph has a route.
    expansions = expansions_of(
        {
            "target": frozenset({(0, frozenset({"a", "dead"})), (1, frozenset({"p"}))}),
            "a": frozenset({(2, frozenset({"p"})), (3, frozenset({"stock"}))}),
            "p": frozenset({(4, frozenset({"a"}))}),
        }
    )
    assert compare_routes(expansions, {"stock"}, min_route_depth("target")) == 3


def test_whole_graph_metrics_survive_a_chain_deeper_than_the_stack() -> None:
    # A plain chain longer than CPython's recursion limit. Both metrics used to walk the solved
    # subgraph by recursion, so a 540k-molecule raw68k graph 992 levels deep raised RecursionError
    # *after* the search had finished — a whole target's compute lost to reporting its depth.
    k = sys.getrecursionlimit() + 500
    names = [f"n{i}" for i in range(k)] + ["stock"]
    expansions = Expansions(NumpyExpansionStore(), DictInterner())
    sids = expansions.intern(names).tolist()
    expansions.mark_expanded(np.array(sids[:k], dtype=np.int64))
    for i in range(k):
        add_application(expansions, sids[i], i, [sids[i + 1]])

    assert compare_routes(expansions, {"stock"}, min_route_depth("n0")) == k
    assert compare_routes(expansions, {"stock"}, solved_route_count("n0", cap=100)) == 1


def test_strict_recovers_duplicate_precursor_reaction() -> None:
    expansions = Expansions(NumpyExpansionStore(), DictInterner())
    target, mid, stock = expansions.intern(["target", "mid", "stock"]).tolist()
    expansions.mark_expanded(np.array([target, mid], dtype=np.int64))
    # target -rule0-> (mid, mid): one application emitting two copies of one precursor.
    add_application(expansions, target, 0, [mid, mid])
    add_application(expansions, mid, 1, [stock])
    reference = Route(
        "target",
        0,
        (
            Route("mid", 1, (Route("stock", None),)),
            Route("mid", 1, (Route("stock", None),)),
        ),
    )
    routes = frozenset(iter_solved_routes("target", expansions, {"stock"}))

    # Strict is stoichiometry-free: the duplicated precursor collapses and still matches.
    assert compare_routes(expansions, {"stock"}, strict(reference)) == 1
    assert count_recoveries(reference, routes, strict_route_matches) == 1


def test_solved_disconnection_count_is_ways_to_begin_that_lead_somewhere() -> None:
    """The diversity measure that is both meaningful and affordable: distinct first disconnections
    a route actually runs through.

    ``T`` splits three ways. ``A+B`` are both stock, ``C`` reaches stock in one reaction, and ``D``
    is a dead end nothing expands. Two of three lead anywhere, and that filter is what makes the
    number mean "ways to begin" rather than "templates that matched" — measured on n1, only about a
    fifth of a target's disconnections lead to a solved graph.
    """
    expansions = expansions_of(
        {"T": [(0, ["A", "B"]), (1, ["C"]), (2, ["D"])], "C": [(3, ["E"])]}
    )
    state = prepare_graph(expansions, frozenset({"A", "B", "E"}))

    assert len(state.applications(int(expansions.lookup(["T"])[0]))) == 3
    assert solved_disconnection_count("T")(state) == 2
    # A target the graph never saw has no ways to begin, rather than raising.
    assert solved_disconnection_count("ZZZ")(state) == 0


def test_solved_disconnection_count_counts_a_disconnection_once_per_precursor_set() -> (
    None
):
    """Two templates achieving one disconnection are one way to begin, not two.

    Written through :func:`tests.vertsy.handbuilt.add_application` with an explicit ``app``, because
    that is the shape the *search* produces: `layer_tables` keys a layer on
    ``(product, precursor multiset)`` and hangs every template achieving it off one application.
    ``expansions_of`` does not collapse — it is a direct builder — so this property belongs to the
    ingest path and has to be tested through it.

    Template multiplicity then lives on ``len(rules)``, where it means the reaction was extracted
    independently more than once. That is evidence the chemistry is real, not evidence of another
    route. How much it matters is a property of the library: on used14k the median disconnection
    carries one template, on raw68k the same one is reachable by 28 to 49.
    """
    expansions: Expansions[str] = Expansions(NumpyExpansionStore(), DictInterner())
    target, left, right = (int(sid) for sid in expansions.intern(["T", "A", "B"]))
    app = add_application(expansions, target, 0, [left, right])
    add_application(expansions, target, 1, [left, right], app=app)
    expansions.mark_expanded(np.array([target], dtype=np.int64))
    state = prepare_graph(expansions, frozenset({"A", "B"}))

    (application,) = state.applications(target)
    assert application[0] == (0, 1)  # both templates, one disconnection
    assert solved_disconnection_count("T")(state) == 1

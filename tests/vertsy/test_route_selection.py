from __future__ import annotations

import numpy as np
import pytest

from vertsy.costs import (
    SILENT,
    by_rule,
    cost_of_factor,
    doubted,
    selective,
    steps_equivalent,
)
from vertsy.expansions import Expansions, expansions_of
from vertsy.interner import DictInterner
from vertsy.metrics import prepare_graph
from vertsy.route_selection import DEFAULT_PRICE, select_solved_routes
from vertsy.routes import decode_route, reaction_count, reaction_depth
from vertsy.store.numpy_store import NumpyExpansionStore

from .handbuilt import add_application


def select(mapping, stock, target="target", **kwargs):
    expansions = expansions_of(mapping)
    state = prepare_graph(expansions, stock)
    target_sid = int(expansions.lookup([target])[0])
    result = select_solved_routes(target_sid, state, **kwargs)
    routes = tuple(decode_route(item.route, expansions) for item in result.routes)
    return result, routes


BRANCHED_OR_LONG = {
    # Four reactions across three parallel branches, two levels deep...
    "target": {
        (0, frozenset({"a", "b", "c"})),
        (1, frozenset({"d"})),
    },
    "a": {(2, frozenset({"stock"}))},
    "b": {(3, frozenset({"stock"}))},
    "c": {(4, frozenset({"stock"}))},
    # ...against three reactions in one chain, three levels deep.
    "d": {(5, frozenset({"e"}))},
    "e": {(6, frozenset({"stock"}))},
}


def test_cost_precedes_shape_and_scores_are_decomposed() -> None:
    """Three reactions cost less than four, so the report now says so. Depth follows cost.

    This ordering is **reversed** from what this module used to report, and the reversal is the
    point rather than a side effect. The key was ``(depth, reactions, utility)``, which preferred
    the shallower four-reaction route -- and, more importantly, meant no route preference of any
    strength could ever prefer a longer route, because it lost on the second key before the third
    was read. A selectivity penalty reaches only that third key, so it could not have worked at
    all.

    Depth is still reported and still breaks ties. What it no longer does is outrank the price.
    """
    result, routes = select(BRANCHED_OR_LONG, {"stock"})

    assert [(reaction_depth(route), reaction_count(route)) for route in routes] == [
        (3, 3),
        (2, 4),
    ]
    assert (
        result.best_shortest_reactions == 4
    )  # still the fewest within the shortest depth
    assert result.routes[0].score.utility == pytest.approx(0.9**3)
    assert result.routes[0].score.stock_factor_product == 1.0
    assert result.routes[0].score.reaction_factor_product == 1.0


def test_ranking_by_cost_reproduces_ranking_by_reaction_count() -> None:
    """The safety net, and the reason cost-first is not a new opinion.

    With nothing but reactions priced, a route's cost is ``reactions * -log(0.9)`` -- strictly
    increasing in the reaction count. So the new primary key is a monotone function of the old
    secondary one, and the two orderings coincide. They can only differ where something *else* is
    priced, which is exactly where they should.
    """
    result, routes = select(BRANCHED_OR_LONG, {"stock"})
    # Utility is `exp(-cost)`, so descending utility *is* ascending cost.
    utilities = [item.score.utility for item in result.routes]
    reactions = [reaction_count(route) for route in routes]

    assert utilities == sorted(utilities, reverse=True)
    assert reactions == sorted(reactions)


def test_common_first_step_exposes_deeper_disconnections() -> None:
    result, routes = select(
        {
            "target": {(0, frozenset({"mid"}))},
            "mid": {
                (1, frozenset({"a"})),
                (2, frozenset({"b"})),
                (3, frozenset({"c"})),
            },
        },
        {"a", "b", "c"},
    )
    assert len(routes) == 3
    assert {route.precursors[0].precursors[0].product for route in routes} == {
        "a",
        "b",
        "c",
    }
    assert result.routes[0].first_distinct_decision is None
    assert {item.first_distinct_decision for item in result.routes[1:]} == {2}


def test_duplicate_templates_choose_best_price_and_count_once() -> None:
    """Two templates, one disconnection: it is priced and labelled at the cheaper of them."""
    result, routes = select(
        {
            "target": {
                (7, frozenset({"stock"})),
                (2, frozenset({"stock"})),
            }
        },
        {"stock"},
        price=by_rule(
            lambda rule: cost_of_factor(0.9 * (0.5 if rule == 2 else 1.0)),
        ),
        stock_factor=lambda key: 0.8,
    )
    assert len(routes) == 1
    assert routes[0].rule == 7
    assert result.routes[0].score.utility == pytest.approx(0.9 * 0.8)


def test_cycles_are_excluded_from_displayed_routes() -> None:
    result, routes = select(
        {
            "target": {
                (0, frozenset({"mid"})),
                (1, frozenset({"stock"})),
            },
            "mid": {(2, frozenset({"target"}))},
        },
        {"stock"},
    )
    assert len(routes) == 1
    assert routes[0].rule == 1
    assert result.shortest_depth == 1


def test_many_tied_routes_are_bounded_and_distinct() -> None:
    result, routes = select(
        {"target": {(rule, frozenset({f"stock-{rule}"})) for rule in range(30)}},
        {f"stock-{rule}" for rule in range(30)},
        candidate_limit=20,
    )
    assert len(routes) == 20
    assert len({route.precursors[0].product for route in routes}) == 20
    assert result.truncated
    assert result.candidates_considered == 20


def test_graph_guided_variants_fill_report_without_partial_tree_search() -> None:
    mapping = {"target": {(rule, frozenset({f"mid-{rule}"})) for rule in range(16)}}
    mapping.update(
        {
            f"mid-{rule}": {
                (100 + rule, frozenset({f"stock-{rule}"})),
            }
            | ({(200 + rule, frozenset({f"extra-{rule}"}))} if rule < 4 else set())
            for rule in range(16)
        }
    )
    result, routes = select(
        mapping,
        {f"stock-{rule}" for rule in range(16)}
        | {f"extra-{rule}" for rule in range(4)},
        state_limit=1,
    )
    assert len(routes) == 20
    assert result.partial_states_explored == 0
    assert result.stopped_after_limit
    assert not result.truncated
    assert all(reaction_depth(route) == 2 for route in routes)


def test_stock_target_and_unsolved_graph() -> None:
    stock_result, stock_routes = select({"target": frozenset()}, {"target"})
    assert len(stock_routes) == 1
    assert stock_result.routes[0].score.depth == 0
    unsolved, routes = select({"target": {(0, frozenset({"missing"}))}}, {"stock"})
    assert routes == ()
    assert unsolved.shortest_depth is None


def test_work_limit_reports_truncation_without_claiming_graph_is_unsolved() -> None:
    result, routes = select(
        {
            "target": {(0, frozenset({"mid"}))},
            "mid": {(1, frozenset({"stock"}))},
        },
        {"stock"},
        state_limit=1,
    )
    assert len(routes) == 1
    assert reaction_depth(routes[0]) == 2
    assert result.shortest_depth == 2
    assert result.truncated


# --------------------------------------------------------- the point of the whole mechanism

# The xylitol shape, in sids. Rule 0 makes the chlorinated polyol from the free polyol in **one**
# step, and that is the step a chemist objects to: chlorinating one primary hydroxyl of a free
# pentitol selectively is optimistic. Rules 1, 2 and 3 are the route they would run instead --
# acetalise to block two hydroxyls, chlorinate the one left, remove the acetal -- three reactions,
# each of them clean. `expansions_of` mints one application per step in iteration order, so rule 0's
# application is 0.
UNSELECTIVE_OR_PROTECTED = {
    "target": {(0, frozenset({"polyol"})), (1, frozenset({"protected-chloride"}))},
    "protected-chloride": {(2, frozenset({"protected"}))},
    "protected": {(3, frozenset({"polyol"}))},
}


def measured_select(outcomes, *, penalty=None, **kwargs):
    """Build the graph, write ``{rule: outcomes}`` into it, and price it with ``penalty``.

    Keyed by rule and not by application, because an application id is minted in iteration order
    and a set has no order to depend on. The id is looked up from the template table instead, which
    is also how a real probe would find it.
    """
    expansions = expansions_of(UNSELECTIVE_OR_PROTECTED)
    _products, apps, rules = expansions.rules()
    app_of = dict(zip(rules.tolist(), apps.tolist()))
    for rule, count in outcomes.items():
        expansions.store.add_selectivity(
            "sites",
            np.array([app_of[rule]], dtype=np.int64),
            np.array([rule], dtype=np.int64),
            np.array([count], dtype=np.int64),
        )
    state = prepare_graph(expansions, {"polyol"})
    if penalty is not None:
        kwargs["price"] = selective(DEFAULT_PRICE, state.outcomes, {"sites": penalty})
    target_sid = int(expansions.lookup(["target"])[0])
    result = select_solved_routes(target_sid, state, **kwargs)
    routes = tuple(decode_route(item.route, expansions) for item in result.routes)
    return result, routes


def test_an_unselective_step_loses_to_a_protection_detour_once_it_is_measured() -> None:
    """The whole mechanism, end to end on a hand-built graph.

    Unmeasured, the one-step route wins on price and should: one reaction beats three. Measured as a
    three-way mixture it is worth four steps, and the detour -- three clean reactions -- becomes the
    cheaper plan. Nothing about the graph changed between the two calls; only what is known about
    one of its steps.
    """
    _, clean = measured_select({})
    assert reaction_count(clean[0]) == 1  # today's answer, pinned

    result, routes = measured_select({0: 3}, penalty=steps_equivalent)

    assert reaction_count(routes[0]) == 3  # acetalise, chlorinate, deprotect
    assert reaction_count(routes[1]) == 1  # still reported, just no longer first
    # And the score says why: the one-step route's price is no longer just its length.
    assert result.routes[1].score.reaction_factor_product < 1.0
    assert result.routes[0].score.reaction_factor_product == 1.0


def test_a_two_way_mixture_ties_with_the_detour_rather_than_beating_it() -> None:
    """Deliberate: the probe counts products the chemistry can make, not the ratio it makes them in.

    A two-way split is as often a clean 20:1 as a 1:1, so `steps_equivalent` prices it at exactly
    three steps and lets the shape tie-break decide. Deciding it either way round would claim
    information the probe does not have.
    """
    result, routes = measured_select({0: 2}, penalty=steps_equivalent)

    assert {reaction_count(route) for route in routes} == {1, 3}
    assert result.routes[0].score.utility == pytest.approx(
        result.routes[1].score.utility
    )


def test_a_detour_two_reactions_longer_is_inside_the_default_candidate_box() -> None:
    """It has to be *considered* before it can be preferred, which the old slack did not ensure.

    ``reaction_slack = 2`` admitted a three-step detour on a one-reaction target by arithmetic
    accident and excluded one everywhere else, which made the detour invisible rather than merely
    unpreferred. A slack counted in step-equivalents admits one wherever the target sits.
    """
    _, routes = measured_select({})
    assert sorted(reaction_count(route) for route in routes) == [1, 3]


def test_a_silent_step_is_charged_and_the_decomposition_stays_a_decomposition() -> None:
    """Silence is priced, and the score's three factors still multiply to its utility.

    The decomposition divides out what the same templates would cost *unmeasured*, and asking for
    that with a vector of zeros asks what a **silent** step costs -- so the moment silence is not
    free, the denominator exceeds the real cost and the remainder goes negative. That surfaced as a
    refusal from `factor_of_cost` in the middle of a benchmark, which is exactly the wrong place to
    find out, so the invariant is pinned here: every factor is a real utility in ``(0, 1]``, and
    their product is the score.
    """
    result, routes = measured_select(
        {0: SILENT}, penalty=doubted(steps_equivalent, silent=2.0)
    )

    assert routes, "a solved graph must still report routes when a step is silent"
    for item in result.routes:
        score = item.score
        for factor in (
            score.base_step_factor,
            score.stock_factor_product,
            score.reaction_factor_product,
        ):
            assert 0.0 < factor <= 1.0
        assert score.utility == pytest.approx(
            score.base_step_factor
            * score.stock_factor_product
            * score.reaction_factor_product
        )
    # And the silent step was actually charged: some route's price is more than its length.
    charged = [
        item for item in result.routes if item.score.reaction_factor_product < 1.0
    ]
    assert charged, "silence was priced at 2.0 and nothing paid it"


def test_silence_costs_nothing_until_it_is_doubted() -> None:
    """The sentinel alone moves no number, which is what makes the change measurable."""
    plain, _ = measured_select({0: SILENT}, penalty=steps_equivalent)
    unmeasured, _ = measured_select({}, penalty=steps_equivalent)
    assert plain.routes[0].score.utility == pytest.approx(
        unmeasured.routes[0].score.utility
    )


def test_two_equivalents_of_one_precursor_survive_into_the_route_and_the_price() -> (
    None
):
    """A disconnection may legitimately need two equivalents of the same starting material.

    Grouping applications by a `frozenset` collapsed `(A, A)` to `(A,)`, which cost twice: the
    emitted route lost a reactant, and -- because a measurement is stored against the precursor
    *multiset* -- the price looked up a key nothing was written under, so a step measured as a
    three-way mixture read as unprobed and was charged nothing.
    """
    expansions: Expansions[str] = Expansions(NumpyExpansionStore(), DictInterner())
    target, feedstock = expansions.intern(["dimer", "monomer"]).tolist()
    expansions.mark_expanded(np.array([target], dtype=np.int64))
    add_application(
        expansions, target, 0, [feedstock, feedstock], outcomes={"sites": 3}
    )
    state = prepare_graph(expansions, {"monomer"})

    # Both equivalents reach the route.
    plain = select_solved_routes(target, state)
    assert len(plain.routes[0].route.precursors) == 2

    # And the measurement is found, so the step is charged rather than read as unprobed.
    priced = select_solved_routes(
        target,
        state,
        price=selective(DEFAULT_PRICE, state.outcomes, {"sites": steps_equivalent}),
    )
    assert priced.routes[0].score.reaction_factor_product < 1.0


def test_the_utility_denominator_is_the_price_multiplicative_identity() -> None:
    """A clean step must never cost less than its own denominator.

    The decomposition splits a route's utility into length and chemistry by dividing the priced
    cost by what the same templates would cost with nothing charged. That denominator has to be the
    count every multiplier shape charges exactly 1.0 for, or the remainder goes negative and
    `factor_of_cost` refuses it.

    `1` is the only such count. Zero is a measurement -- silent, which `doubted` charges for -- and
    `UNPROBED` is a measurement of a different kind -- nobody looked, which `uncurated` charges for.
    This denominator used to be `UNPROBED` and was correct only while nothing priced it.
    """
    from vertsy.costs import (
        SILENT,
        UNPROBED,
        doubted,
        flat,
        steps_equivalent,
        uncurated,
    )

    shapes = {
        "steps": steps_equivalent,
        "flat": flat(3.0),
        "doubted": doubted(steps_equivalent, silent=2.0),
        "uncurated": uncurated(steps_equivalent, unprobed=1.2),
        "both": uncurated(doubted(flat(3.0), silent=2.0), unprobed=1.2),
    }

    assert all(multiplier(1) == 1.0 for multiplier in shapes.values())
    # And the two counts that are *not* neutral, so this test fails if someone swaps them back.
    assert shapes["uncurated"](UNPROBED) > 1.0
    assert shapes["doubted"](SILENT) > 1.0


# ------------------------------------------------------------------ the diversity floor


def _two_near_identical_routes():
    """A graph whose two routes share their first disconnection and differ only below it.

    `T -> A + B`, then `A -> C` or `A -> D`, with `B`, `C`, `D` buyable. The second route is what a
    naive "next best" always finds: the same plan with its least significant step swapped.
    """
    import numpy as np

    from vertsy.expansions import Expansions
    from vertsy.interner import DictInterner
    from vertsy.metrics import prepare_graph
    from vertsy.store.numpy_store import NumpyExpansionStore

    expansions = Expansions(NumpyExpansionStore(), DictInterner())
    target, a, b, c, d = expansions.intern(["T", "A", "B", "C", "D"]).tolist()
    expansions.add_layer(
        np.array([target]), [(0, 0, 0, 0), (0, 0, 0, 1)], np.array([a, b])
    )
    expansions.add_layer(np.array([a]), [(0, 1, 0, 0), (0, 2, 1, 1)], np.array([c, d]))
    return target, prepare_graph(expansions, ["B", "C", "D"])


def test_without_a_floor_a_trivial_variation_is_reported() -> None:
    """Today's behaviour, and the thing the floor exists to stop reporting."""
    target, state = _two_near_identical_routes()

    selection = select_solved_routes(target, state, limit=5)

    assert len(selection.routes) == 2
    # The second shares the first decision and differs only at the second.
    assert [route.first_distinct_decision for route in selection.routes] == [None, 2]
    assert selection.too_similar == ()


def test_a_floor_of_one_demands_a_different_first_disconnection() -> None:
    """And says which route the rejected one resembled, so the floor can be widened knowingly."""
    target, state = _two_near_identical_routes()

    selection = select_solved_routes(target, state, limit=5, diversity_cutoff=1)

    assert len(selection.routes) == 1
    assert selection.routes[0].first_distinct_decision is None
    (dropped,) = selection.too_similar
    assert dropped.first_distinct_decision == 2
    assert dropped.resembled == 1


def test_the_floor_never_rejects_on_a_template_synonym() -> None:
    """Diversity is measured over precursors, not templates, and that matters for real tables.

    Some rule sets carry many near-duplicate synonym templates. Two routes differing only in which
    one was credited are the *same plan*, and `route_tokens` is template-insensitive, so they
    collapse before the diversity index is computed -- a synonym can never look like diversity, and
    can never be rejected as similarity either.
    """
    import numpy as np

    from vertsy.expansions import Expansions
    from vertsy.interner import DictInterner
    from vertsy.metrics import prepare_graph
    from vertsy.store.numpy_store import NumpyExpansionStore

    expansions = Expansions(NumpyExpansionStore(), DictInterner())
    target, a, b = expansions.intern(["T", "A", "B"]).tolist()
    # One disconnection, two co-canonical templates claiming it.
    expansions.add_layer(
        np.array([target]),
        [(0, 7, 0, 0), (0, 7, 0, 1), (0, 9, 1, 0), (0, 9, 1, 1)],
        np.array([a, b]),
    )
    state = prepare_graph(expansions, ["A", "B"])

    selection = select_solved_routes(target, state, limit=5, diversity_cutoff=1)

    assert len(selection.routes) == 1
    assert selection.too_similar == ()


# ------------------------------------------------------------- the extraction validator


def test_a_censored_route_is_replaced_and_its_reason_is_carried_out() -> None:
    """The second of two veto layers: the search prices a row, extraction refuses it.

    Refusing here rather than in the search is what makes the refusal reportable -- the graph still
    holds the condemned step -- and what lets the next candidate take the place of the one lost
    instead of the report coming back short.
    """
    target, state = _two_near_identical_routes()
    first = select_solved_routes(target, state, limit=5).routes[0].route

    selection = select_solved_routes(
        target,
        state,
        limit=1,
        validate=lambda route: "B 30" if route == first else None,
    )

    assert len(selection.routes) == 1
    assert selection.routes[0].route != first
    (lost,) = selection.censored
    assert lost.route == first
    assert lost.reason == "B 30"


def test_a_censored_route_is_not_what_the_next_route_must_differ_from() -> None:
    """A route nobody can make must not become the thing diversity is measured against.

    Both routes here share their first disconnection, so a floor of one admits only the first of
    them. Censor that first one and the second is the only route a reader will see -- so it is the
    *best* route, not a near-duplicate of a route that was never reported.
    """
    target, state = _two_near_identical_routes()
    first = select_solved_routes(target, state, limit=5).routes[0].route

    selection = select_solved_routes(
        target,
        state,
        limit=5,
        diversity_cutoff=1,
        validate=lambda route: "B 30" if route == first else None,
    )

    assert len(selection.routes) == 1
    assert selection.routes[0].route != first
    assert selection.routes[0].first_distinct_decision is None
    # Nothing was dropped for resembling a route that was never reported.
    assert selection.too_similar == ()


def test_a_validator_that_refuses_nothing_changes_nothing() -> None:
    """The safety net: a validator is only ever allowed to remove."""
    target, state = _two_near_identical_routes()

    plain = select_solved_routes(target, state, limit=5)
    validated = select_solved_routes(
        target, state, limit=5, validate=lambda route: None
    )

    assert [item.route for item in validated.routes] == [
        item.route for item in plain.routes
    ]
    assert validated.censored == ()


def test_a_validator_that_refuses_everything_reports_every_refusal() -> None:
    """A report that came back empty has to say why, rather than reading as an unsolved target."""
    target, state = _two_near_identical_routes()

    selection = select_solved_routes(
        target, state, limit=5, validate=lambda route: "every step refused"
    )

    assert selection.routes == ()
    assert len(selection.censored) == 2
    assert {item.reason for item in selection.censored} == {"every step refused"}


def test_a_slack_is_counted_in_a_typical_step_not_the_cheapest_one() -> None:
    """One well-curated template must not shrink the detour budget for every route in the graph.

    `cost_slack = 3` was calibrated against a protect/react/deprotect detour -- three *ordinary*
    steps. Counting it in the cheapest step available made that exact under a uniform price and
    collapses it under a per-rule one: a single template at a 0.95 yield is less than a quarter of
    the cost of a typical 0.80 one, so the budget falls to 23% of what it was meant to be, with
    nothing about the chemistry having changed.

    The work bound keeps the minimum, and should: a budget divided by the cheapest step is the most
    reactions that budget could hold, which is what makes the heap bound correct.
    """
    from vertsy.costs import by_rule, cost_of_factor
    from vertsy.route_selection import cheapest_application, typical_application

    class Graph:
        steps = ("P",)

        def applications(self, product: str):
            return [((rule,), (f"p{rule}",)) for rule in range(5)]

    yields = {0: 0.95, 1: 0.80, 2: 0.80, 3: 0.65, 4: 0.50}
    base = by_rule(lambda rule: cost_of_factor(yields[rule]), uniform=False).base

    def cost(rule: int, precursors: tuple[str, ...]) -> float:
        return base(rule)

    graph = Graph()
    assert cheapest_application(graph, cost) == pytest.approx(cost_of_factor(0.95))
    assert typical_application(graph, cost) == pytest.approx(cost_of_factor(0.80))
    # The budget the old reading would have allowed, against the one intended.
    assert (
        3 * cheapest_application(graph, cost) < 3 * typical_application(graph, cost) / 4
    )


def test_under_a_uniform_price_the_two_units_are_one_number() -> None:
    """Which is why the split changes nothing already recorded.

    Every measurement in `docs/note-forward-selectivity.md` was taken under a constant base, where
    the median application cost and the minimum are the same, so `steps` runs are unaffected.
    """
    from vertsy.route_selection import cheapest_application, typical_application

    class Graph:
        steps = ("P",)

        def applications(self, product: str):
            return [((rule,), (f"p{rule}",)) for rule in range(4)]

    def cost(rule: int, precursors: tuple[str, ...]) -> float:
        return 0.105

    graph = Graph()
    assert (
        cheapest_application(graph, cost) == typical_application(graph, cost) == 0.105
    )

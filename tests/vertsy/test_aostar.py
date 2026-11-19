from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import cast

import numpy as np
import pytest

from vertsy.aostar import (
    anytime_step,
    aostar,
    by_context,
    by_inside,
    by_total,
    closure_fold,
    committed_application_leaves,
    constant_step,
    descent_leaves,
    global_leaves,
    weighted,
)
from vertsy.costs import application_price, by_rule
from vertsy.expansions import Expansions, expansions_of
from vertsy.heuristics import zero_heuristic
from vertsy.interner import DictInterner
from vertsy.metrics import (
    BOOLEAN,
    TROPICAL,
    log_semiring,
    outside_valuation,
    prepare_graph,
    valuation,
)
from vertsy.routes import (
    collapsed_route_key,
    iter_solved_routes,
    leaves,
    reaction_count,
)
from vertsy.search import concat_object_batches, expansion, object_mol_array
from vertsy.store import SidStep
from vertsy.store.numpy_store import NumpyExpansionStore
from vertsy.tropical import tropical_valuation


def scalar_price(base: Callable[[int], float]) -> Callable[[SidStep], float]:
    """A per-rule cost as the per-application cost the generic fold now takes."""
    return partial(application_price, by_rule(base))


# ------------------------------------------------------------------ semiring algebra
#
# Each semiring is exercised over its own carrier: BOOLEAN lives on {0, 1} and says nothing
# about 7.5, so testing it there would only prove the test wrong.

SEMIRINGS = {
    "tropical": (TROPICAL, (0.0, 0.5, 1.0, 3.0, 7.5)),
    "boolean": (BOOLEAN, (0.0, 1.0)),
    "log-1.0": (log_semiring(1.0), (0.0, 0.5, 1.0, 3.0, 7.5)),
    "log-0.1": (log_semiring(0.1), (0.0, 0.5, 1.0, 3.0, 7.5)),
}


@pytest.mark.parametrize("name", sorted(SEMIRINGS))
def test_semiring_identities(name: str) -> None:
    """zero and one are the identities the fold's empty base cases lean on."""
    semiring, carrier = SEMIRINGS[name]
    for value in carrier:
        assert semiring.oplus(semiring.zero, value) == pytest.approx(value)
        assert semiring.otimes(semiring.one, value) == pytest.approx(value)


@pytest.mark.parametrize("name", sorted(SEMIRINGS))
def test_semiring_is_associative_and_distributive(name: str) -> None:
    """The laws that let the fold combine applications in any order."""
    semiring, carrier = SEMIRINGS[name]
    for a in carrier:
        for b in carrier:
            for c in carrier:
                assert semiring.oplus(semiring.oplus(a, b), c) == pytest.approx(
                    semiring.oplus(a, semiring.oplus(b, c))
                )
                assert semiring.otimes(a, semiring.oplus(b, c)) == pytest.approx(
                    semiring.oplus(semiring.otimes(a, b), semiring.otimes(a, c))
                )


def test_tropical_is_selective_and_log_is_not() -> None:
    """The property that decides which semiring may guide a search.

    AO* needs ``oplus`` to *choose* an alternative. Softmin accumulates them instead, which
    is the point of it — two equally good routes really are more route mass than one — but
    it means "the best application" has no meaning under it.
    """
    assert TROPICAL.oplus(3.0, 3.0) == 3.0
    assert log_semiring(1.0).oplus(3.0, 3.0) < 3.0


def test_log_semiring_anneals_to_tropical() -> None:
    """Min-plus is the zero-temperature limit of the log semiring, not a separate algebra."""
    assert log_semiring(0.0) is TROPICAL
    values = (1.0, 1.2, 5.0)
    approached = [
        log_semiring(t).oplus(values[0], log_semiring(t).oplus(*values[1:]))
        for t in (1.0, 0.1, 0.01)
    ]
    exact = TROPICAL.oplus(values[0], TROPICAL.oplus(*values[1:]))
    assert approached == sorted(approached)  # rises monotonically toward the min
    assert approached[-1] == pytest.approx(exact, abs=1e-6)


def test_log_semiring_rejects_negative_temperature() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        log_semiring(-1.0)


# ------------------------------------------------------------------- the shared fixture
#
# T is reachable two ways: one reaction to two stock molecules, or two reactions through C.
# Cheapest is 1. Every fold test below reads off this graph.

GRAPH = {"T": [(0, ["A", "B"]), (1, ["C"])], "C": [(2, ["D"])]}
STOCK = frozenset({"A", "B", "D"})

# A unit step throughout, because these tests read a route's value *as* its reaction count and
# assert on the number. That is the one job a unit step is right for; `aostar` requires the choice
# to be stated rather than defaulted precisely because it is wrong for search.
UNIT = constant_step(1.0)


def graph_state():
    expansions = expansions_of(GRAPH)
    state = prepare_graph(expansions, STOCK)
    sids = {key: int(expansions.lookup([key])[0]) for key in ("T", "A", "B", "C", "D")}
    return expansions, state, sids


def test_conjoining_precursors_with_max_values_the_longest_sequence() -> None:
    """``conjoin=max`` prices a convergent route by its deepest branch, not its total steps.

    A branched graph where the two readings disagree by construction: ``T`` needs ``P`` and ``W``,
    ``P`` is three reactions above stock and ``W`` is one. Summing calls that 5 reactions, which is
    the reaction count. Taking the worst branch calls it 4, the longest linear sequence — and the
    difference is exactly ``W``'s one reaction, which runs in parallel with ``P``'s three.
    """
    expansions = expansions_of(
        {
            "T": [(0, ["P", "W"])],
            "P": [(1, ["Q"])],
            "Q": [(2, ["R"])],
            "R": [(3, ["S"])],
            "W": [(4, ["S"])],
        }
    )
    state = prepare_graph(expansions, frozenset({"S"}))
    target = int(expansions.lookup(["T"])[0])
    summed = valuation(
        state, TROPICAL, leaf_value=lambda p: 0.0, rule_cost=lambda r: 1.0
    )
    worst = valuation(
        state, TROPICAL, leaf_value=lambda p: 0.0, rule_cost=lambda r: 1.0, conjoin=max
    )

    assert summed.node(target, 9) == 5.0  # 4 up the P branch plus W's one
    assert worst.node(target, 9) == 4.0  # the P branch alone; W is free in parallel
    # And each fold reports the rule it folded by, so the outside sweep cannot disagree with it.
    assert summed.conjoin(2.0, 3.0) == 5.0
    assert worst.conjoin(2.0, 3.0) == 3.0


def test_max_conjoin_does_not_penalise_splitting_a_molecule() -> None:
    """The measured failure, in miniature: a good disconnection that summing calls a bad one.

    ``T`` has two options — split into two unexpanded halves estimated at 0.4 each, or one step to a
    single unexpanded molecule at 0.7. Both halves are closer to the stock than anything the second
    option reaches, and the split is the move a chemist wants. Summing prices it at 0.8 + step and
    prefers the worse option; taking the worst branch prices it at 0.4 + step and prefers the split.
    """
    expansions = expansions_of({"T": [(0, ["X", "Y"]), (1, ["Z"])]})
    state = prepare_graph(expansions, frozenset())
    estimate = {"X": 0.4, "Y": 0.4, "Z": 0.7}
    sid_estimate = {
        int(expansions.lookup([key])[0]): value for key, value in estimate.items()
    }
    target = int(expansions.lookup(["T"])[0])
    step = 0.1

    summed = valuation(
        state,
        TROPICAL,
        leaf_value=lambda p: sid_estimate.get(p, 0.0),
        rule_cost=lambda r: step,
    )
    worst = valuation(
        state,
        TROPICAL,
        leaf_value=lambda p: sid_estimate.get(p, 0.0),
        rule_cost=lambda r: step,
        conjoin=max,
    )

    assert summed.node(target, 5) == pytest.approx(
        0.8
    )  # 0.1 + 0.7, the WORSE option wins
    assert worst.node(target, 5) == pytest.approx(0.5)  # 0.1 + 0.4, the split wins


def test_committed_application_keeps_bottleneck_branches_in_the_batch() -> None:
    """A scalar additive outside score overcharges siblings of a max-fold application.

    The fold correctly prices X+Y at the worse leaf, 0.531, but the global sweep assigns each
    leaf 1.051 by adding both estimates. Z therefore wins the global ranking even though its
    application loses the inside fold. The committed selector reserves X and Y together and keeps
    that choice after their estimates change on a later pass.
    """
    expansions = expansions_of({"T": [(0, ["X", "Y"]), (1, ["Z"])]})
    state = prepare_graph(expansions, frozenset())
    sids = {key: int(expansions.lookup([key])[0]) for key in ("T", "X", "Y", "Z")}
    step = 0.001

    def values(estimates: dict[str, float]):
        by_sid = {sids[key]: estimate for key, estimate in estimates.items()}
        return valuation(
            state,
            TROPICAL,
            leaf_value=lambda sid: by_sid.get(sid, 0.0),
            rule_cost=lambda rule: step,
            conjoin=max,
        )

    first = values({"X": 0.52, "Y": 0.53, "Z": 0.70})
    outside = outside_valuation(
        sids["T"], state, first, max_depth=2, step_cost=scalar_price(lambda rule: step)
    )
    kwargs = {"max_depth": 2, "limit": 2, "price": by_rule(lambda rule: step)}

    assert first.node(sids["T"], 2) == pytest.approx(0.531)
    assert outside.leaves[sids["X"]][0] == pytest.approx(1.051)
    assert outside.leaves[sids["Y"]][0] == pytest.approx(1.051)
    assert global_leaves(sids["T"], state, first, **kwargs)[0] == sids["Z"]

    select = committed_application_leaves()
    assert select(sids["T"], state, first, **kwargs) == [sids["X"], sids["Y"]]

    later = values({"X": 0.80, "Y": 0.90, "Z": 0.10})
    assert select(sids["T"], state, later, **kwargs) == [sids["X"], sids["Y"]]

    expansions.mark_expanded(np.array([sids["X"]], dtype=np.int64))
    dead_state = prepare_graph(expansions, frozenset())
    by_sid = {sids["Y"]: 0.90, sids["Z"]: 0.10}
    after_dead_end = valuation(
        dead_state,
        TROPICAL,
        leaf_value=lambda sid: by_sid.get(sid, 0.0),
        rule_cost=lambda rule: step,
        conjoin=max,
    )
    assert select(sids["T"], dead_state, after_dead_end, **kwargs) == [sids["Z"]]


def test_tropical_fold_finds_the_cheapest_route_cost() -> None:
    _, state, sids = graph_state()
    value = valuation(
        state, TROPICAL, leaf_value=lambda p: 0.0, rule_cost=lambda r: 1.0
    )
    assert value.node(sids["T"], 5) == 1.0  # A+B, not the two-reaction path through C
    assert value.node(sids["C"], 5) == 1.0
    assert value.node(sids["A"], 5) == TROPICAL.one  # already buyable


def test_fold_respects_the_depth_budget() -> None:
    """Past the horizon a product is ``zero``: a route that does not fit does not count."""
    _, state, sids = graph_state()
    value = valuation(
        state, TROPICAL, leaf_value=lambda p: 0.0, rule_cost=lambda r: 1.0
    )
    assert value.node(sids["T"], 1) == 1.0
    assert value.node(sids["T"], 0) == TROPICAL.zero


def test_boolean_fold_reproduces_the_solved_sids_fixpoint() -> None:
    """The same statement as :func:`vertsy.routes.solved_sids`, in the semiring instead.

    This is the claim that the fold generalises what the graph already knew — checked
    against the Polars fixpoint the route extractor actually trusts.
    """
    _, state, sids = graph_state()
    value = valuation(state, BOOLEAN, leaf_value=lambda p: BOOLEAN.zero)
    for sid in sids.values():
        assert bool(value.node(sid, 10)) == state.is_solved(sid)


def test_log_fold_scores_route_mass_below_the_best_route() -> None:
    """Two routes to T score better than the single cheapest one — that is the mass."""
    _, state, sids = graph_state()
    tropical = valuation(
        state, TROPICAL, leaf_value=lambda p: 0.0, rule_cost=lambda r: 1.0
    )
    soft = valuation(
        state, log_semiring(1.0), leaf_value=lambda p: 0.0, rule_cost=lambda r: 1.0
    )
    assert soft.node(sids["T"], 5) < tropical.node(sids["T"], 5)


def test_fold_terminates_on_a_cyclic_graph() -> None:
    """X and Y make each other forever; the reaction budget is what ends it."""
    expansions = expansions_of({"X": [(0, ["Y"])], "Y": [(1, ["X"])]})
    state = prepare_graph(expansions, frozenset())
    value = valuation(
        state, TROPICAL, leaf_value=lambda p: 0.0, rule_cost=lambda r: 1.0
    )
    assert value.node(int(expansions.lookup(["X"])[0]), 50) == TROPICAL.zero


def test_unexpanded_product_takes_its_heuristic() -> None:
    """The one place search knowledge enters the fold."""
    expansions = expansions_of({"T": [(0, ["U"])]})  # U is named but never expanded
    state = prepare_graph(expansions, frozenset())
    value = valuation(
        state, TROPICAL, leaf_value=lambda p: 0.25, rule_cost=lambda r: 1.0
    )
    assert not state.is_expanded(int(expansions.lookup(["U"])[0]))
    assert value.node(int(expansions.lookup(["T"])[0]), 5) == 1.25


# ------------------------------------------------------------------------ AO* search
#
# A synthetic backend: molecules are their own keys and reactions come from a dict, so the
# search is exercised without dragging chemistry into it.


def make_match(reactions: dict, expanded: list[str]):
    def match(mols):
        keys: list[str] = []
        hits: list[tuple[int, int, int, int]] = []
        for product_idx, mol in enumerate(list(mols)):
            expanded.append(mol)
            for app, (rule, precursors) in enumerate(reactions.get(mol, [])):
                for precursor in precursors:
                    hits.append((product_idx, rule, app, len(keys)))
                    keys.append(precursor)
        return (
            object_mol_array(keys),
            keys,
            np.array(hits, dtype=np.int64).reshape(-1, 4),
        )

    return expansion(match)


def run_aostar(target, reactions, stock, heuristic=zero_heuristic, **kwargs):
    expanded: list[str] = []
    kwargs.setdefault("price", UNIT)
    routes = aostar(
        target,
        target_mol=target,
        stock=stock,
        expansions=kwargs.pop("expansions", None)
        or Expansions(NumpyExpansionStore(), DictInterner()),
        heuristic=heuristic,
        match=make_match(reactions, expanded),
        stack_mols=object_mol_array,
        concat_mols=concat_object_batches,
        width=kwargs.pop("width", 8),
        passes=kwargs.pop("passes", 6),
        **kwargs,
    )
    return routes, expanded


def test_aostar_returns_the_cheapest_route_under_an_admissible_heuristic() -> None:
    """With an admissible estimate one pass already finds A+B, without opening C.

    That is the whole content of the old proof-then-stop exit: the estimate could rule the C
    branch out without expanding it. What changed is that ruling it out no longer ends the
    search — see :func:`test_aostar_spends_the_budget_on_a_solved_target`.
    """
    routes, expanded = run_aostar("T", GRAPH, STOCK, passes=1)
    assert routes
    best = min(routes, key=reaction_count)
    assert reaction_count(best) == 1
    assert sorted(leaves(best)) == ["A", "B"]
    assert expanded == ["T"]


def test_aostar_spends_the_budget_on_a_solved_target() -> None:
    """A proved-best route does not end the search; the rest of the budget buys alternatives.

    T is solved in one reaction after the first pass and the old loop stopped there. Many routes
    are the goal now, so C is opened too and the graph ends up holding both routes — while the
    *returned* list still holds only the cheapest, which is what `iter_best_sid_routes` means.
    """
    expansions = cast(
        Expansions[str], Expansions(NumpyExpansionStore(), DictInterner())
    )
    routes, expanded = run_aostar("T", GRAPH, STOCK, expansions=expansions, passes=3)

    assert expanded == ["T", "C"]
    assert [reaction_count(route) for route in routes] == [1]
    assert len(frozenset(iter_solved_routes("T", expansions, STOCK))) == 2


def test_stop_when_solved_buys_the_same_verdict_for_less_work() -> None:
    """The inverse of :func:`test_aostar_spends_the_budget_on_a_solved_target`, opted into.

    Same graph, same budget: the default spends it and opens C for a second route, while this
    returns at the top of pass two having opened only T. A template-coverage sweep wants exactly
    that -- *reachable or not* is the whole question, and the alternatives the repair phase buys
    are what makes the full schedule cost an order of magnitude more per target.

    The verdict has to agree, which is the half worth pinning: stopping early must not turn a
    solvable target into an unsolved one.
    """
    expansions = cast(
        Expansions[str], Expansions(NumpyExpansionStore(), DictInterner())
    )
    routes, expanded = run_aostar(
        "T", GRAPH, STOCK, expansions=expansions, passes=3, stop_when_solved=True
    )

    assert expanded == ["T"]  # C is never opened; the default run opens it
    assert prepare_graph(expansions, STOCK).is_solved(int(expansions.lookup(["T"])[0]))
    assert [reaction_count(route) for route in routes] == [1]


def test_stop_when_solved_does_not_cut_an_unsolved_search_short() -> None:
    """Nothing is solved, so the flag must be inert and the whole budget still spent."""
    unreachable = frozenset({"A", "B"})  # D is gone, so neither disconnection closes
    routes, expanded = run_aostar(
        "T", {"T": [(1, ["C"])], "C": [(2, ["D"])]}, unreachable, passes=3
    )
    stopped_routes, stopped_expanded = run_aostar(
        "T",
        {"T": [(1, ["C"])], "C": [(2, ["D"])]},
        unreachable,
        passes=3,
        stop_when_solved=True,
    )

    assert routes == stopped_routes == []
    assert expanded == stopped_expanded == ["T", "C", "D"]


def test_aostar_keeps_searching_past_a_solved_upper_bound() -> None:
    """A solved upper bound is not final while an open alternative can still beat it.

    T disconnects two ways, tied at one reaction each while both are still estimates. At
    ``width=1`` the ranking is redone after every single expansion, and expanding A settles the
    A+C branch at two reactions — so B, still estimated at one, overtakes C and is opened next.
    The cheapest route found is B's, and C is only opened once nothing cheaper is left.
    """
    reactions = {
        "T": [(0, ["A", "C"]), (1, ["B"])],
        "A": [(2, ["S"])],
        "C": [(3, ["S"])],
        "B": [(4, ["S"])],
    }

    routes, expanded = run_aostar(
        "T", reactions, frozenset({"S"}), max_routes=1, width=1, passes=4
    )

    assert reaction_count(routes[0]) == 2
    assert expanded == ["T", "A", "B", "C"]


def test_aostar_respects_max_depth_in_an_existing_graph() -> None:
    expansions = cast(
        Expansions[str],
        expansions_of({"T": [(0, frozenset({"A"}))], "A": [(1, frozenset({"S"}))]}),
    )

    routes = aostar(
        "T",
        target_mol="T",
        stock=frozenset({"S"}),
        expansions=expansions,
        heuristic=zero_heuristic,
        match=make_match({}, []),
        stack_mols=object_mol_array,
        concat_mols=concat_object_batches,
        price=UNIT,
        # "A" arrives as an open leaf of the caller's graph, so its molecule comes with it.
        resume=(["A"], object_mol_array(["A"])),
        max_depth=1,
        width=10,
        passes=10,
    )

    assert routes == []


def test_aostar_refuses_a_resume_that_omits_an_open_leaf() -> None:
    """A missing restart molecule raises rather than quietly costing every route through it.

    The graph names U as a precursor but this search never built its molecule and cannot: keys
    need not be invertible. A dense estimate array reads an unscored sid as unreachable, so
    without this check selection would simply never offer U and the target would come back
    unsolved — the failure mode the resume contract exists to prevent.
    """
    expansions = cast(Expansions[str], expansions_of({"T": [(0, frozenset({"U"}))]}))

    with pytest.raises(KeyError, match="U"):
        aostar(
            "T",
            target_mol="T",
            stock=frozenset({"S"}),
            expansions=expansions,
            heuristic=zero_heuristic,
            match=make_match({"U": [(1, ["S"])]}, []),
            stack_mols=object_mol_array,
            concat_mols=concat_object_batches,
            price=UNIT,
            width=4,
            passes=4,
        )


def test_aostar_ignores_open_leaves_of_another_target_in_a_shared_store() -> None:
    """A store may hold several targets' graphs; only this target's leaves are this search's.

    Sharing a store is the point of one — a shared intermediate is interned once. Demanding a
    molecule for `U2`, which nothing reachable from `T1` names, would refuse a perfectly
    well-formed resume.
    """
    expansions = cast(
        Expansions[str],
        expansions_of(
            {"T1": [(0, frozenset({"U1"}))], "T2": [(1, frozenset({"U2"}))]},
        ),
    )
    expanded: list[str] = []

    routes = aostar(
        "T1",
        target_mol="T1",
        stock=frozenset({"S"}),
        expansions=expansions,
        heuristic=zero_heuristic,
        match=make_match({"U1": [(2, ["S"])]}, expanded),
        stack_mols=object_mol_array,
        concat_mols=concat_object_batches,
        price=UNIT,
        resume=(["U1"], object_mol_array(["U1"])),
        width=4,
        passes=4,
    )

    assert expanded == ["U1"]
    assert reaction_count(routes[0]) == 2


def test_aostar_ignores_an_open_leaf_it_cannot_reach_within_the_horizon() -> None:
    """A leaf past the horizon reads as unreachable whatever its estimate, so none is needed.

    `B` sits two reactions down. At `max_depth=1` the fold never asks what it is worth — it has
    no budget left by the time it arrives — so requiring its molecule would be requiring one that
    cannot change any answer.
    """
    expansions = cast(
        Expansions[str],
        expansions_of({"T": [(0, frozenset({"A"}))], "A": [(1, frozenset({"B"}))]}),
    )

    routes = aostar(
        "T",
        target_mol="T",
        stock=frozenset({"S"}),
        expansions=expansions,
        heuristic=zero_heuristic,
        match=make_match({}, []),
        stack_mols=object_mol_array,
        concat_mols=concat_object_batches,
        price=UNIT,
        width=4,
        passes=1,
        max_depth=1,
    )

    assert routes == []  # nothing solvable that shallow, but nothing raised either


def test_aostar_accepts_a_horizon_deeper_than_the_new_passes_when_resuming() -> None:
    """A preloaded graph already has levels, so the horizon is not bounded by the new passes.

    `T -> A -> B` needs one pass to open B, but a horizon of one cannot see a three-reaction
    route. The ceiling is a fact about graphs this search builds from nothing, not about every
    graph it is handed.
    """
    expansions = cast(
        Expansions[str],
        expansions_of(
            {"T": [(0, frozenset({"A"}))], "A": [(1, frozenset({"B"}))]},
        ),
    )
    expanded: list[str] = []

    routes = aostar(
        "T",
        target_mol="T",
        stock=frozenset({"S"}),
        expansions=expansions,
        heuristic=zero_heuristic,
        match=make_match({"B": [(2, ["S"])]}, expanded),
        stack_mols=object_mol_array,
        concat_mols=concat_object_batches,
        price=UNIT,
        resume=(["B"], object_mol_array(["B"])),
        width=4,
        passes=1,
        max_depth=3,
    )

    assert expanded == ["B"]  # one new pass was all it took
    assert reaction_count(routes[0]) == 3  # and the horizon could see the whole route


def test_aostar_still_bounds_the_horizon_for_a_graph_it_builds_itself() -> None:
    """The ceiling stays where it is meaningful: nothing preloaded, so depth cannot exceed passes."""
    with pytest.raises(ValueError, match="between 0 and passes"):
        run_aostar("T", GRAPH, STOCK, passes=2, max_depth=3)


def test_aostar_scores_open_leaves_in_an_existing_graph() -> None:
    expansions = cast(Expansions[str], expansions_of({"T": [(0, frozenset({"U"}))]}))
    expanded: list[str] = []

    routes = aostar(
        "T",
        target_mol="T",
        stock=frozenset({"S"}),
        expansions=expansions,
        heuristic=zero_heuristic,
        match=make_match({"U": [(1, ["S"])]}, expanded),
        stack_mols=object_mol_array,
        concat_mols=concat_object_batches,
        price=UNIT,
        # The graph arrives with "U" already an open leaf: this search never built its
        # molecule, so the caller that did hands it over alongside the graph.
        resume=(["U"], object_mol_array(["U"])),
        max_depth=2,
        width=10,
        passes=10,
    )

    assert reaction_count(routes[0]) == 2
    assert expanded == ["U"]


def test_aostar_expands_a_leaf_discovered_several_passes_earlier() -> None:
    """The frontier carries molecules forward, so an old leaf is still expandable.

    A leaf is selected by sid out of a batch that has been narrowed and rejoined once per
    pass since it arrived. If the sids and the batch ever fell out of step, this would expand
    the wrong molecule — silently, and only for leaves that waited.
    """
    # width=1 forces one expansion per pass, so W — discovered in the first pass alongside P —
    # has to survive on the frontier until its turn. At the default width both would go in a
    # single pass and nothing would be carried anywhere.
    #
    # How *long* a leaf can wait is bounded by admissibility, not by the frontier: an estimate
    # capped at one reaction can only ever push a leaf one place down the order, so a test that
    # demanded a long wait would be testing an inadmissible heuristic instead.
    reactions = {
        "T": [(0, ["P", "W"])],
        "P": [(1, ["Q"])],
        "Q": [(2, ["R"])],
        "R": [(3, ["S"])],
        "W": [(4, ["S"])],
    }

    routes, expanded = run_aostar("T", reactions, frozenset({"S"}), width=1, passes=6)

    assert routes and reaction_count(routes[0]) == 5
    assert set(expanded) == {"T", "P", "Q", "R", "W"}
    # W waited at least one pass, and the molecule that came back was still the right one —
    # a frontier out of step with its sids would have expanded something else and the route
    # would not have closed.
    assert expanded.index("W") > expanded.index("P")


def test_aostar_never_expands_the_same_molecule_twice() -> None:
    """A precursor that is already an open leaf must not be filed on the frontier again.

    `step` deduplicates within one call, so a leaf reachable from two different products
    comes back a second time; without a check against the standing frontier it would be
    stored twice and, worse, expanded twice.
    """
    # U is a precursor of both A and B, and is discovered twice in consecutive passes.
    reactions = {
        "T": [(0, ["A", "B"])],
        "A": [(1, ["U"])],
        "B": [(2, ["U"])],
        "U": [(3, ["S"])],
    }

    _, expanded = run_aostar("T", reactions, frozenset({"S"}), passes=4)

    assert expanded.count("U") == 1
    assert len(expanded) == len(set(expanded))


def test_aostar_returns_the_trivial_route_for_a_stock_target() -> None:
    routes, expanded = run_aostar("A", GRAPH, STOCK)
    assert reaction_count(routes[0]) == 0
    assert expanded == []


def test_aostar_reports_no_routes_for_an_unreachable_target() -> None:
    routes, expanded = run_aostar("ZZZ", GRAPH, STOCK)
    assert routes == []
    assert expanded == ["ZZZ"]  # tried once, proved dead, stopped


def test_aostar_terminates_on_a_cycle() -> None:
    routes, expanded = run_aostar(
        "X", {"X": [(0, ["Y"])], "Y": [(1, ["X"])]}, frozenset(), width=1, passes=20
    )
    assert routes == []
    # Running dry ended it, not the budget: X and Y make each other and nothing else is open.
    assert len(expanded) < 20


def test_aostar_honours_the_expansion_budget() -> None:
    """A chain longer than the budget stops at the budget rather than running away."""
    chain = {f"M{i}": [(i, [f"M{i + 1}"])] for i in range(20)}
    routes, expanded = run_aostar("M0", chain, frozenset({"M19"}), width=1, passes=3)
    assert routes == []
    assert len(expanded) == 3


def test_aostar_expands_and_siblings_as_one_batch() -> None:
    """An application is a conjunction, so its precursors form a batch by construction.

    This is what lets the search hand the matcher real batches instead of single molecules.
    """
    batches: list[int] = []
    reactions = {"T": [(0, ["P", "Q", "R"])], "P": [], "Q": [], "R": []}

    def counting_match(mols):
        batches.append(len(list(mols)))
        return make_match(reactions, [])(mols)

    aostar(
        "T",
        target_mol="T",
        stock=frozenset(),
        expansions=Expansions(NumpyExpansionStore(), DictInterner()),
        heuristic=zero_heuristic,
        match=counting_match,
        stack_mols=object_mol_array,
        concat_mols=concat_object_batches,
        price=UNIT,
        width=8,
        passes=5,
    )
    assert batches[0] == 1  # the target alone
    assert batches[1] == 3  # P, Q and R together, in one step()


@pytest.mark.parametrize(
    "bad",
    [
        {"width": 0},
        {"passes": 0},
        {"max_routes": -1},
        {"max_depth": -1},
        # A horizon deeper than the budget can reach is unreachable by construction, so it is
        # a mistake rather than wasted fold depth.
        {"max_depth": 7, "passes": 6},
    ],
)
def test_aostar_rejects_nonsense_budgets(bad: dict) -> None:
    with pytest.raises(ValueError, match="must be"):
        run_aostar("T", GRAPH, STOCK, **bad)


@pytest.mark.parametrize("cost", [0.0, -1.0])
def test_a_step_must_cost_something(cost: float) -> None:
    """A free step makes every route tie at zero, which leaves route extraction nothing to rank.

    The lower end of the step cost is where the interesting settings live, so the floor is worth
    guarding: 0.01 is a fine choice and 0 is a different algorithm.
    """
    with pytest.raises(ValueError, match="positive"):
        constant_step(cost)


@pytest.mark.parametrize(
    "reactions,stock",
    [
        (GRAPH, STOCK),
        (
            {"T": [(0, ["A", "C"]), (1, ["B"])], "A": [(2, ["S"])], "C": [(3, ["S"])]},
            {"S"},
        ),
        # A chain, a shared intermediate, and a repeated precursor: the three shapes where the
        # two folds could disagree about a sibling sum.
        (
            {
                "T": [(0, ["P", "P"]), (1, ["Q"])],
                "P": [(2, ["R"])],
                "Q": [(3, ["R", "R"])],
                "R": [(4, ["S"])],
            },
            {"S"},
        ),
    ],
    ids=["two-routes", "and-siblings", "repeated"],
)
@pytest.mark.parametrize("width", [1, 8])
def test_the_two_folds_search_identically(reactions, stock, width: int) -> None:
    """The array fold is a specialisation, so it must not change a single decision.

    Same expansions in the same order, and the same routes out. Values may differ in their last
    bits — the array form sums siblings by subtraction — so this asserts on what the search *did*,
    which is what the difference would have to show up in.
    """
    array = run_aostar(
        "T", reactions, frozenset(stock), width=width, fold=tropical_valuation
    )
    closure = run_aostar(
        "T", reactions, frozenset(stock), width=width, fold=closure_fold
    )

    assert array[1] == closure[1]  # the expansion order, molecule for molecule
    assert {collapsed_route_key(route) for route in array[0]} == {
        collapsed_route_key(route) for route in closure[0]
    }


def test_aostar_skips_extraction_when_no_routes_are_wanted() -> None:
    """Aggregate metrics come off the graph, so `max_routes=0` must not pay for enumeration."""
    expansions = cast(
        Expansions[str], Expansions(NumpyExpansionStore(), DictInterner())
    )
    routes, expanded = run_aostar(
        "T", GRAPH, STOCK, expansions=expansions, passes=3, max_routes=0
    )

    assert routes == []
    assert expanded == ["T", "C"]  # the search ran in full; only extraction was skipped
    assert len(frozenset(iter_solved_routes("T", expansions, STOCK))) == 2


# ------------------------------------------------------------------ the outside pass
#
# T is reachable two ways: through the open leaf U (alongside stock A), or through C and then
# the open leaf V. With h(U)=0.5 and h(V)=0.1 the cheapest estimate is the U branch at 1.5, so
# U — not the cheaper-looking V — is the leaf on a best solution graph.

OPEN_GRAPH = {"T": [(0, ["A", "U"]), (1, ["C"])], "C": [(2, ["V"])]}
OPEN_STOCK = frozenset({"A"})
OPEN_H = {"U": 0.5, "V": 0.1}


def open_graph_state(graph=OPEN_GRAPH, stock=OPEN_STOCK, estimates=OPEN_H):
    expansions = expansions_of(graph)
    state = prepare_graph(expansions, stock)
    keys = sorted(set(graph) | {p for v in graph.values() for p in v[0][1]})
    sids = {key: int(expansions.lookup([key])[0]) for key in keys}
    by_sid = {sid: key for key, sid in sids.items()}
    value = valuation(
        state,
        TROPICAL,
        leaf_value=lambda p: estimates[by_sid[p]],
        rule_cost=lambda r: 1.0,
    )
    return state, sids, value


def test_outside_charges_the_context_of_each_leaf() -> None:
    """delta is what a solution graph through a leaf needs *besides* that leaf's subtree."""
    state, sids, value = open_graph_state()
    outside = outside_valuation(
        sids["T"], state, value, max_depth=3, step_cost=scalar_price(lambda r: 1.0)
    )

    assert outside.delta[(sids["T"], 3)] == 0.0  # the target's context is empty
    assert value.node(sids["T"], 3) == pytest.approx(1.5)
    # U: one reaction above it, and its stock sibling A costs nothing.
    assert outside.delta[(sids["U"], 2)] == pytest.approx(1.0)
    assert outside.leaves[sids["U"]][0] == pytest.approx(1.5)
    # V sits under C, so two reactions stand above it.
    assert outside.delta[(sids["C"], 2)] == pytest.approx(1.0)
    assert outside.delta[(sids["V"], 1)] == pytest.approx(2.0)
    assert outside.leaves[sids["V"]][0] == pytest.approx(2.1)


def test_outside_charges_and_siblings() -> None:
    """An AND-sibling's own estimate is part of a leaf's context, not of its subtree."""
    state, sids, value = open_graph_state(
        graph={"T": [(0, ["W", "U"])]},
        stock=frozenset(),
        estimates={"W": 0.3, "U": 0.5},
    )
    outside = outside_valuation(
        sids["T"], state, value, max_depth=3, step_cost=scalar_price(lambda r: 1.0)
    )

    # One reaction plus the sibling W's 0.3 — where a buyable sibling contributed nothing.
    assert outside.delta[(sids["U"], 2)] == pytest.approx(1.3)
    assert outside.leaves[sids["U"]][0] == pytest.approx(1.8)


def test_outside_drops_a_leaf_that_only_exists_past_the_horizon() -> None:
    """A leaf reachable only below the depth budget is absent, not infinite.

    At ``max_depth=2`` no route can reach V, so every application leading to it is already
    ``zero`` and descending would record nothing but infinities. Absence says the same thing
    and spares the caller a filter.
    """
    state, sids, value = open_graph_state()
    outside = outside_valuation(
        sids["T"], state, value, max_depth=2, step_cost=scalar_price(lambda r: 1.0)
    )

    assert outside.leaves[sids["U"]][0] == pytest.approx(1.5)
    assert sids["V"] not in outside.leaves
    # And at a budget that does reach it, it comes back.
    deeper = outside_valuation(
        sids["T"], state, value, max_depth=3, step_cost=scalar_price(lambda r: 1.0)
    )
    assert deeper.leaves[sids["V"]][0] == pytest.approx(2.1)


def test_outside_is_unchanged_by_a_horizon_deeper_than_the_graph() -> None:
    """The sweep stops once a level is empty, and stopping changes no value.

    This matters because the horizon is now the caller's pass count, which at a high effort is
    far deeper than any graph an early pass has built — so most of the sweep's levels would
    otherwise be spent walking empty sets.
    """
    state, sids, value = open_graph_state()
    tight = outside_valuation(
        sids["T"], state, value, max_depth=3, step_cost=scalar_price(lambda r: 1.0)
    )
    deep = outside_valuation(
        sids["T"], state, value, max_depth=3, step_cost=scalar_price(lambda r: 1.0)
    )
    generous = outside_valuation(
        sids["T"], state, value, max_depth=32, step_cost=scalar_price(lambda r: 1.0)
    )

    assert tight.leaves == deep.leaves  # the sweep is deterministic to begin with
    # A deeper horizon reaches the same leaves. Their totals are attained at a larger
    # `depth_left`, since the sweep starts further from the goal, but the costs are identical.
    assert {sid: total for sid, (total, _, _) in generous.leaves.items()} == {
        sid: total for sid, (total, _, _) in tight.leaves.items()
    }


def test_outside_terminates_on_a_cycle_without_a_visited_guard() -> None:
    """X and Y make each other forever; the level sweep is what ends it."""
    expansions = expansions_of({"X": [(0, ["Y"])], "Y": [(1, ["X"])]})
    state = prepare_graph(expansions, frozenset())
    value = valuation(
        state, TROPICAL, leaf_value=lambda p: 0.0, rule_cost=lambda r: 1.0
    )
    target = int(expansions.lookup(["X"])[0])

    outside = outside_valuation(
        target, state, value, max_depth=50, step_cost=scalar_price(lambda r: 1.0)
    )

    assert outside.delta[(target, 50)] == 0.0
    assert outside.leaves == {}  # every product is expanded; there are no open leaves


def test_outside_counts_a_duplicate_precursor_twice() -> None:
    """An application naming the same precursor twice charges it twice, so removing one
    occurrence from the sibling sum must leave the other. Subtracting `rn(q)` would not.
    """
    expansions = expansions_of({"T": [(0, ["U", "U"])]})
    state = prepare_graph(expansions, frozenset())
    sids = {key: int(expansions.lookup([key])[0]) for key in ("T", "U")}
    value = valuation(
        state, TROPICAL, leaf_value=lambda p: 0.5, rule_cost=lambda r: 1.0
    )

    outside = outside_valuation(
        sids["T"], state, value, max_depth=3, step_cost=scalar_price(lambda r: 1.0)
    )

    # The reaction, plus the *other* copy of U at 0.5.
    assert outside.delta[(sids["U"], 2)] == pytest.approx(1.5)


def test_global_selection_matches_the_descent_at_k_one() -> None:
    """At limit 1 the global ranking picks a leaf on some best solution graph — AO*'s rule.

    Not necessarily the *same* leaf: ties are broken differently, and several leaves can sit
    on equally-best graphs. What must agree is the value, which is the property the descent
    was ever relied on for.
    """
    state, sids, value = open_graph_state()
    kwargs = {"max_depth": 3, "limit": 1, "price": UNIT}

    (chosen,) = global_leaves(sids["T"], state, value, **kwargs)
    (descended,) = descent_leaves(sids["T"], state, value, **kwargs)

    best = value.node(sids["T"], 3)
    outside = outside_valuation(
        sids["T"], state, value, max_depth=3, step_cost=scalar_price(lambda r: 1.0)
    )
    assert outside.leaves[chosen][0] == pytest.approx(best)
    assert outside.leaves[descended][0] == pytest.approx(best)


def test_the_cheapest_leaf_always_totals_the_graph_cost() -> None:
    """The optimality invariant: some open leaf is on a best solution graph, or we are done.

    This is what makes global ranking a faithful AO*, and it is the single strongest statement
    in this file — if it fails, ranking has invented a cheaper graph than the fold believes in.
    """
    state, sids, value = open_graph_state()
    outside = outside_valuation(
        sids["T"], state, value, max_depth=3, step_cost=scalar_price(lambda r: 1.0)
    )

    assert min(total for total, _, _ in outside.leaves.values()) == pytest.approx(
        value.node(sids["T"], 3)
    )


def test_global_selection_offers_at_least_what_the_descent_does() -> None:
    """Every leaf the descent would expand is reachable, so global ranking sees it too."""
    state, sids, value = open_graph_state()
    kwargs = {"max_depth": 3, "limit": 100, "price": UNIT}

    assert set(descent_leaves(sids["T"], state, value, **kwargs)) <= set(
        global_leaves(sids["T"], state, value, **kwargs)
    )


def test_only_global_selection_can_fill_a_batch() -> None:
    """The whole point: a descent offers one solution graph's leaves, however big `limit` is.

    On this graph the best solution graph exposes a single open leaf, so the descent returns
    one molecule at any limit — the behaviour that made every earlier k sweep a null result.
    Global ranking offers both.
    """
    state, sids, value = open_graph_state()
    kwargs = {"max_depth": 3, "limit": 8, "price": UNIT}

    assert len(descent_leaves(sids["T"], state, value, **kwargs)) == 1
    assert len(global_leaves(sids["T"], state, value, **kwargs)) == 2


def test_only_the_faithful_rank_keeps_the_k_one_guarantee() -> None:
    """`by_total` picks a leaf on a best solution graph; the halves need not.

    This is what makes the default the default. `by_inside` ignores how much stands above a
    leaf, so it happily opens V — cheaper in itself (0.1 against 0.5) but sitting under an
    extra reaction — where the faithful sum prefers U and matches the graph's own cost.
    """
    state, sids, value = open_graph_state()
    kwargs = {"max_depth": 3, "limit": 1, "price": UNIT}

    assert global_leaves(sids["T"], state, value, rank=by_total, **kwargs) == [
        sids["U"]
    ]
    assert global_leaves(sids["T"], state, value, rank=by_inside, **kwargs) == [
        sids["V"]
    ]
    # And the faithful one agrees with the graph's cost, which is the invariant that matters.
    assert value.node(sids["T"], 3) == pytest.approx(1.5)


def test_the_ranks_order_the_same_leaves_differently() -> None:
    """All three see the same frontier; they disagree only about what to open first."""
    state, sids, value = open_graph_state()
    kwargs = {"max_depth": 3, "limit": 8, "price": UNIT}

    orders = {
        name: global_leaves(sids["T"], state, value, rank=rank, **kwargs)
        for name, rank in (
            ("total", by_total),
            ("inside", by_inside),
            ("context", by_context),
        )
    }

    assert {frozenset(order) for order in orders.values()} == {
        frozenset({sids["U"], sids["V"]})
    }
    assert orders["total"] != orders["inside"]


def test_weighted_interpolates_between_the_halves() -> None:
    """`weighted(1)` is the faithful sum and `weighted(0)` ignores the context entirely."""
    state, sids, value = open_graph_state()
    kwargs = {"max_depth": 3, "limit": 8, "price": UNIT}

    assert global_leaves(
        sids["T"], state, value, rank=weighted(1.0), **kwargs
    ) == global_leaves(sids["T"], state, value, rank=by_total, **kwargs)
    assert global_leaves(
        sids["T"], state, value, rank=weighted(0.0), **kwargs
    ) == global_leaves(sids["T"], state, value, rank=by_inside, **kwargs)


def test_anytime_step_relaxes_only_once_a_route_exists() -> None:
    """Greedy until solved, then relaxed — and the switch lands on the pass *after* the solve.

    ``GRAPH`` is the fixture that can show this: ``T`` solves in one pass through the stock pair
    ``A+B``, while the ``C`` branch stays open for later passes to work on. A chain that solves only
    at its last expansion cannot demonstrate it at all, because the search ends the moment nothing
    is left open and no pass ever runs under the relaxed cost.

    The switch is deliberately after selection rather than during it: the valuation handed to a
    selection was folded under the current cost, so changing it mid-pass would rank leaves by a
    value computed in a different algebra.
    """
    seen: list[float] = []
    price, scheduled = anytime_step(greedy=0.001, relaxed=1.0)

    def watching(target, state, value, *, max_depth, limit, price):
        seen.append(price.base(0))
        return scheduled(
            target, state, value, max_depth=max_depth, limit=limit, price=price
        )

    routes, _ = run_aostar("T", GRAPH, STOCK, price=price, select=watching, passes=4)

    assert routes  # it solves, so the schedule has something to trigger on
    # One pass of lag, and it is inherent rather than incidental: a pass folds before it selects, so
    # the pass that first *sees* the target solved was already valued under the greedy cost. The
    # earliest a relaxed fold can happen is the pass after that.
    assert seen == [0.001, 0.001, 1.0]
    assert seen == sorted(seen)  # monotone: it never switches back


def test_anytime_step_stays_greedy_when_nothing_is_ever_solved() -> None:
    """An unsolved target never relaxes, which is why the schedule cannot cost solve rate."""
    seen: list[float] = []
    price, scheduled = anytime_step(greedy=0.001, relaxed=1.0)

    def watching(target, state, value, *, max_depth, limit, price):
        seen.append(price.base(0))
        return scheduled(
            target, state, value, max_depth=max_depth, limit=limit, price=price
        )

    # `X` and `Y` make each other forever and neither reaches stock.
    routes, _ = run_aostar(
        "X",
        {"X": [(0, ["Y"])], "Y": [(1, ["X"])]},
        frozenset(),
        price=price,
        select=watching,
        passes=4,
    )
    assert not routes
    assert set(seen) == {0.001}


def test_aostar_widens_each_iteration_without_carrying_unused_slots() -> None:
    from vertsy.aostar import global_leaves

    requested: list[int] = []
    selected_counts: list[int] = []

    def recording(*args, **kwargs):
        requested.append(kwargs["limit"])
        selected = global_leaves(*args, **kwargs)
        selected_counts.append(len(selected))
        return selected

    reactions = {"T": [(0, [f"M{i}"]) for i in range(30)]}
    run_aostar(
        "T", reactions, frozenset(), width=(4, 8, 12), passes=3, select=recording
    )
    assert requested == [4, 8, 12]
    assert selected_counts == [1, 8, 12]


@pytest.mark.parametrize("width", [(1, 2), (1, 0, 3), (), -1])
def test_aostar_rejects_invalid_width_schedule(width) -> None:
    with pytest.raises(ValueError, match="width"):
        run_aostar("T", {}, frozenset(), width=width, passes=3)

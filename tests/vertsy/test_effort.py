from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import cast

import numpy as np
import pytest

from vertsy.aostar import aostar, constant_step, global_leaves
from vertsy.effort import (
    Round,
    descent,
    format_effort,
    per_reaction,
    reached,
    recording_select,
    retention,
    route_effort,
    steering,
)
from vertsy.expansions import Expansions
from vertsy.heuristics import zero_heuristic
from vertsy.interner import DictInterner
from vertsy.routes import Route, reaction_count, reaction_products
from vertsy.search import concat_object_batches, expansion, object_mol_array
from vertsy.store.numpy_store import NumpyExpansionStore

# A chain with a branch: T needs P and W, and P is three reactions deep. At width=1 the search
# has to defer one of them, which is what makes the fresh/deferred split observable.
BRANCHED = {
    "T": [(0, ["P", "W"])],
    "P": [(1, ["Q"])],
    "Q": [(2, ["R"])],
    "R": [(3, ["S"])],
    "W": [(4, ["S"])],
}

# These tests read a route's value as its reaction count, which is the one job a unit step is right
# for. `aostar` requires the choice to be stated rather than defaulted; see its docstring.
UNIT = constant_step(1.0)


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


def run(reactions, stock, target="T", **kwargs):
    """Run a recorded search and hand back everything the report needs."""
    rounds: list[Round] = []
    expanded: list[str] = []
    expansions = cast(
        Expansions[str], Expansions(NumpyExpansionStore(), DictInterner())
    )
    routes = aostar(
        target,
        target_mol=target,
        stock=stock,
        expansions=expansions,
        heuristic=zero_heuristic,
        match=make_match(reactions, expanded),
        stack_mols=object_mol_array,
        concat_mols=concat_object_batches,
        price=UNIT,
        select=recording_select(global_leaves, rounds),
        width=kwargs.pop("width", 8),
        passes=kwargs.pop("passes", 6),
        **kwargs,
    )
    return routes, rounds, expanded, expansions


def test_recording_select_is_transparent() -> None:
    """Instrumentation that changed the search would measure itself, not the search."""
    plain_routes, _, plain_expanded, _ = run(BRANCHED, frozenset({"S"}))
    rounds: list[Round] = []
    recorded_expanded: list[str] = []
    recorded_routes = aostar(
        "T",
        target_mol="T",
        stock=frozenset({"S"}),
        expansions=Expansions(NumpyExpansionStore(), DictInterner()),
        heuristic=zero_heuristic,
        match=make_match(BRANCHED, recorded_expanded),
        stack_mols=object_mol_array,
        concat_mols=concat_object_batches,
        price=UNIT,
        select=recording_select(global_leaves, rounds),
        width=8,
        passes=6,
    )

    assert recorded_expanded == plain_expanded
    assert [reaction_count(route) for route in recorded_routes] == [
        reaction_count(route) for route in plain_routes
    ]
    # And the record accounts for exactly the expansions that happened, in order.
    assert sum(len(record.selected) for record in rounds) == len(recorded_expanded)


def test_rank_and_pass_are_read_off_the_selection() -> None:
    """A leaf's position in a pass is the width it needed; the pass index is the passes."""
    _, rounds, expanded, expansions = run(BRANCHED, frozenset({"S"}), width=8)

    # Pass 1 opens T alone; pass 2 opens P and W together, since an application is a conjunction.
    assert [len(record.selected) for record in rounds][:2] == [1, 2]
    costs = reached(rounds)
    sid = lambda key: int(expansions.lookup([key])[0])  # noqa: E731
    assert costs[sid("T")] == (1, 1)  # first of pass one
    assert costs[sid("P")] == (1, 2)  # first of pass two
    assert costs[sid("W")] == (2, 2)  # second of pass two, so it needed width 2
    assert expanded[:3] == ["T", "P", "W"]


def test_a_deferred_leaf_is_not_counted_as_fresh() -> None:
    """W is discovered in pass one and opened later, so it is deferred, not newly arrived."""
    _, rounds, expanded, _ = run(BRANCHED, frozenset({"S"}), width=1, passes=6)

    assert expanded == ["T", "P", "W", "Q", "R"]
    # T is new. P was born in pass 1 and taken in pass 2, so it never waited. W was born in pass 1
    # too but pass 2 preferred P, so W is deferred — and so is Q, born in pass 2 while pass 3 was
    # busy with W. R was born in pass 4 and taken in pass 5.
    assert [record.fresh for record in rounds] == [1, 1, 0, 0, 1]
    # One leaf per pass, and the sixth pass finds nothing left to open, so it is not a round.
    assert [len(record.selected) for record in rounds] == [1, 1, 1, 1, 1]


def test_open_leaves_counts_the_frontier_the_pass_chose_from() -> None:
    """The deferred frontier is what a wider pass would have drawn on."""
    _, rounds, _, _ = run(BRANCHED, frozenset({"S"}), width=1, passes=6)

    # Pass 1 sees T alone. Pass 2 sees P and W. Pass 3 sees W plus P's precursor Q, since taking P
    # deferred W and added Q. By pass 4 only Q is open: W's precursor S has been interned and is
    # stock, so it is subtracted rather than counted as a leaf.
    assert [record.open_leaves for record in rounds] == [1, 2, 2, 1, 1]


def test_reasons_show_the_ranking_being_driven_by_committed_cost() -> None:
    """With no heuristic, the sort key *is* the accumulated cost, and the depths say so.

    Every estimate is zero, so `context + estimate` is `context`, which is what the search has
    already committed above the leaf. The chain is hand-checkable: T at depth 0 costs nothing to
    reach; P and W at depth 1 both cost one reaction plus their sibling's estimate; and each
    expansion turns an estimate into a real cost, so the next pick's context is one higher.
    """
    _, rounds, expanded, _ = run(BRANCHED, frozenset({"S"}), width=1, passes=6)

    assert expanded == ["T", "P", "W", "Q", "R"]
    reasons = [reason for record in rounds for reason in record.reasons]
    assert [context for context, _, _ in reasons] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert [estimate for _, estimate, _ in reasons] == [0.0] * 5
    # T is the target; P and W are its precursors; Q and R sit one and two below P.
    assert [depth for _, _, depth in reasons] == [0, 1, 1, 2, 3]

    context, estimate, depths = steering(rounds)
    assert context == pytest.approx(2.0) and estimate == pytest.approx(0.0)
    assert depths == Counter({1: 2, 0: 1, 2: 1, 3: 1})

    # A heuristic that says nothing has no gradient, so there is nothing a reaction is worth to it.
    # Reporting zero here is the honest answer and the reason the column is worth printing: it says
    # the step cost is carrying the whole ranking.
    assert per_reaction(descent(rounds)) == pytest.approx(0.0)


def test_the_gradient_is_the_mean_fall_per_level() -> None:
    """``descent`` averages estimates within a depth, and ``per_reaction`` differences across them.

    Built by hand rather than searched, because the quantity under test is arithmetic over
    ``Round.reasons`` and a graph would only obscure which numbers went in. Depth 1 averages 0.6 and
    0.4 to 0.5; depth 3 sits at 0.2; so two levels carry a fall of 0.3 and a reaction is worth 0.15.
    """
    rounds = [
        Round(
            open_leaves=9,
            selected=(1, 2, 3),
            fresh=0,
            reasons=((0.0, 0.6, 1), (0.0, 0.4, 1), (0.0, 0.2, 3)),
        )
    ]
    assert descent(rounds) == {1: (2, pytest.approx(0.5)), 3: (1, pytest.approx(0.2))}
    assert per_reaction(descent(rounds)) == pytest.approx(0.15)

    # One level observed is no gradient observed, and reporting 0 would read as "flat" when it means
    # "unmeasured". The grid drops NaN from the column rather than averaging it in.
    single = [Round(open_leaves=1, selected=(1,), fresh=0, reasons=((0.0, 0.6, 1),))]
    assert per_reaction(descent(single)) != per_reaction(descent(single))
    assert per_reaction({}) != per_reaction({})


def test_reasons_can_be_switched_off_without_changing_the_search() -> None:
    """The second sweep is opt-out, and opting out must cost only the record."""
    plain_routes, plain_rounds, plain_expanded, _ = run(
        BRANCHED, frozenset({"S"}), width=1, passes=6
    )
    rounds: list[Round] = []
    expanded: list[str] = []
    routes = aostar(
        "T",
        target_mol="T",
        stock=frozenset({"S"}),
        expansions=Expansions(NumpyExpansionStore(), DictInterner()),
        heuristic=zero_heuristic,
        match=make_match(BRANCHED, expanded),
        stack_mols=object_mol_array,
        concat_mols=concat_object_batches,
        price=UNIT,
        select=recording_select(global_leaves, rounds, reasons=False),
        width=1,
        passes=6,
    )

    assert expanded == plain_expanded
    assert [reaction_count(r) for r in routes] == [
        reaction_count(r) for r in plain_routes
    ]
    assert [record.selected for record in rounds] == [
        record.selected for record in plain_rounds
    ]
    assert all(record.reasons == () for record in rounds)


def test_route_effort_is_the_max_over_the_reactions_it_needs() -> None:
    routes, rounds, _, expansions = run(BRANCHED, frozenset({"S"}), width=1, passes=6)

    assert routes
    route = routes[0]
    assert reaction_products(route) == frozenset({"T", "P", "Q", "R", "W"})
    # R was opened last, in pass 5; nothing needed a width beyond 1 at that width.
    assert route_effort(route, expansions, reached(rounds)) == (1, 5)


def test_route_effort_reports_no_cost_for_a_stock_target() -> None:
    routes, rounds, _, expansions = run(BRANCHED, frozenset({"T"}))
    assert route_effort(routes[0], expansions, reached(rounds)) == (0, 0)


def test_route_effort_does_not_require_keys_to_be_orderable() -> None:
    """``K`` is constrained to be hashable, and nothing more.

    A key flavour with no ``<`` is perfectly valid — the CUDA and Metal backends already use
    opaque WL tokens — so pricing a route must not sort its products. The answer is a max, which
    never needed an order.
    """

    @dataclass(frozen=True, eq=True)
    class Opaque:
        name: str

        __lt__ = None  # type: ignore[assignment]  # explicitly unorderable

    reactions = {
        Opaque("T"): [(0, [Opaque("M")])],
        Opaque("M"): [(1, [Opaque("S")])],
    }
    routes, rounds, _, expansions = run(
        reactions, frozenset({Opaque("S")}), target=Opaque("T")
    )

    assert routes
    with pytest.raises(TypeError):  # the fixture really is unorderable
        sorted([Opaque("a"), Opaque("b")])
    assert route_effort(routes[0], expansions, reached(rounds)) == (1, 2)


def test_route_effort_is_none_for_a_route_this_recording_did_not_build() -> None:
    """A route whose reactions were never chosen here cannot be priced against this effort."""
    _, rounds, _, expansions = run(BRANCHED, frozenset({"S"}))
    stranger = Route("Z", 9, (Route("S", None),))

    assert route_effort(stranger, expansions, reached(rounds)) is None


def test_retention_prices_each_route_against_a_smaller_budget() -> None:
    """Two routes needing different widths: the narrower budget keeps only one of them."""
    # T disconnects two ways. Both are one reaction, so both are opened in the same pass and
    # their ranks differ — which is exactly the "would width=1 have lost it" question.
    reactions = {
        "T": [(0, ["A"]), (1, ["B"])],
        "A": [(2, ["S"])],
        "B": [(3, ["S"])],
    }
    routes, rounds, _, expansions = run(
        reactions, frozenset({"S"}), width=8, passes=4, max_routes=50
    )

    counts = retention(routes, expansions, reached(rounds))
    assert sum(counts.values()) == len(routes) == 2
    # Both routes go through T (pass 1, rank 1) plus one of A or B, opened together in pass 2 at
    # ranks 1 and 2. So one route needs width 1 and the other width 2, both at 2 passes.
    assert counts == {(1, 2): 1, (2, 2): 1}


def test_format_effort_reports_the_shape_and_survives_an_empty_run() -> None:
    _, rounds, _, expansions = run(BRANCHED, frozenset({"S"}), width=1, passes=6)
    report = format_effort(rounds, [], expansions)

    assert "5 products over 5 passes (widest 1, filled in 5/5)" in report
    assert "3/5 picks freshly discovered, 2 deferred" in report
    assert "no routes to price" in report
    assert format_effort([], [], expansions) == "no passes ran"

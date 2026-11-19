"""Scores travel with an expansion, and each channel is optional in the way that costs nothing.

The payload has two axes and no third: one value per precursor row, one per hit row. The tests here
pin what that buys -- a wrapper adding a score changes nothing else, a search reads a score it was
handed instead of recomputing it, and a name nobody wrote stays absent rather than becoming a
neutral value that a price would then charge for.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from vertsy.aostar import aostar, constant_step
from vertsy.expansions import expansions_of
from vertsy.heuristics import zero_heuristic
from vertsy.search import (
    Expansion,
    concat_object_batches,
    expansion,
    measured,
    object_mol_array,
    scored_with,
)

GRAPH = {"T": [(0, ["A", "B"])], "A": [(1, ["S"])], "B": [(2, ["S"])]}


def make_match(reactions: dict, expanded: list[str]):
    """Molecules are their own keys and reactions come from a dict, as elsewhere in these tests."""

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


def run(match, **kwargs):
    return aostar(
        "T",
        target_mol="T",
        stock=frozenset({"S"}),
        expansions=expansions_of({}),
        match=match,
        stack_mols=object_mol_array,
        concat_mols=concat_object_batches,
        width=4,
        passes=4,
        price=constant_step(1.0),
        **kwargs,
    )


def test_a_backend_returning_the_three_tuple_needs_no_scores_at_all() -> None:
    """`expansion` is the whole adapter, and an unscored payload is the behaviour that existed."""
    found = make_match(GRAPH, [])(object_mol_array(["T"]))

    assert isinstance(found, Expansion)
    assert found.keys == ["A", "B"]
    assert found.molecule_scores == {} and found.reaction_scores == {}


def test_a_search_reads_an_estimate_it_was_handed_instead_of_recomputing_it() -> None:
    """What `estimated` buys: the heuristic runs in the wrapper, not inside the search loop.

    The heuristic passed to `aostar` here would raise if the search called it for a discovered
    leaf. It does not, because the layer arrived estimated -- which is the property that lets a
    device backend answer with a column of numbers and never hand a molecule back to the host.
    """
    calls: list[int] = []

    def counted(molecules):
        calls.append(len(molecules))
        return np.zeros(len(molecules), dtype=np.float32)

    def refuses(molecules):
        # The target and any resumed leaf still go through `heuristic`; nothing else may.
        if len(molecules) != 1:
            raise AssertionError("the search estimated a layer that arrived estimated")
        return np.zeros(len(molecules), dtype=np.float32)

    routes = run(
        measured(make_match(GRAPH, []), molecules={"estimate": counted}),
        heuristic=refuses,
    )

    assert routes  # it still solves
    assert calls  # and the wrapper did the estimating


def test_estimating_with_a_zero_heuristic_changes_no_route() -> None:
    """Transparency, earned before the wrapper is allowed to be a wrapper."""
    plain = run(make_match(GRAPH, []), heuristic=zero_heuristic)
    wrapped = run(
        measured(make_match(GRAPH, []), molecules={"estimate": zero_heuristic}),
        heuristic=zero_heuristic,
    )

    assert [route.rule for route in plain] == [route.rule for route in wrapped]


def test_an_estimate_column_follows_the_mask_its_molecules_follow() -> None:
    """A score paired with the wrong molecule is worse than no score, so the pairing is the test.

    The match returns three precursors of which one is stock, so `step`'s fresh mask drops a row.
    The estimate must lose the same row, and the two that survive must keep their own values.
    """
    from vertsy.metrics import prepare_graph
    from vertsy.search import step

    def match(mols):
        return (
            object_mol_array(["a", "b", "S"]),
            ["a", "b", "S"],
            np.array([(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2)], dtype=np.int64),
        )

    def by_position(molecules):
        return np.array([10.0, 20.0, 30.0][: len(molecules)], dtype=np.float64)

    expansions = expansions_of({})
    frontier = expansions.intern(["T"])
    layer = step(
        frontier,
        object_mol_array(["T"]),
        expansions,
        {"S"},
        match=measured(expansion(match), molecules={"estimate": by_position}),
    )

    assert expansions.decode(layer.sids) == ["a", "b"]
    assert layer.estimate is not None
    assert np.asarray(layer.estimate).tolist() == [10.0, 20.0]  # 30.0 left with "S"
    assert prepare_graph(expansions, {"S"}).outcomes == {}  # nothing was measured


def test_a_measurement_reaches_the_store_and_an_absent_one_does_not() -> None:
    """The two channels part company here, and the asymmetry is the design.

    An estimate is about a molecule the frontier will carry, so it travels with the frontier. A
    forward measurement is about an application the graph keeps, so it goes to the store and is
    never seen up here again.
    """
    from vertsy.metrics import prepare_graph
    from vertsy.search import step

    def match(mols):
        return (
            object_mol_array(["a"]),
            ["a"],
            np.array([(0, 7, 0, 0)], dtype=np.int64),
        )

    def measuring(products, mols, hits):
        return np.full(len(hits), 3, dtype=np.int64)

    expansions = expansions_of({})
    frontier = expansions.intern(["T"])
    layer = step(
        frontier,
        object_mol_array(["T"]),
        expansions,
        set(),
        match=measured(expansion(match), reactions={"outcomes": measuring}),
    )

    assert layer.estimate is None  # nobody estimated anything
    outcomes = prepare_graph(expansions, set()).outcomes
    assert list(outcomes) == ["outcomes"]  # the channel the wrapper named
    assert list(outcomes["outcomes"].values()) == [3]
    assert [rule for _precursors, rule in outcomes["outcomes"]] == [7]


def test_an_unmeasured_layer_leaves_the_store_as_it_was() -> None:
    """The no-op path, asserted rather than assumed."""
    from vertsy.metrics import prepare_graph
    from vertsy.search import step

    expansions = expansions_of({})
    frontier = expansions.intern(["T"])
    step(
        frontier,
        object_mol_array(["T"]),
        expansions,
        set(),
        match=make_match({"T": [(7, ["a"])]}, []),
    )

    assert expansions.channels() == ()
    assert prepare_graph(expansions, set()).outcomes == {}


def test_two_wrappers_on_one_name_refuse_rather_than_pick_a_winner() -> None:
    """Which of them wins is not a question with a good answer, so it is not answered."""
    found = make_match(GRAPH, [])(object_mol_array(["T"]))
    once = scored_with(found, molecules={"estimate": np.zeros(2)})

    with pytest.raises(ValueError, match="already measured"):
        scored_with(once, molecules={"estimate": np.ones(2)})


def test_a_search_prices_a_measurement_the_probe_wrote_into_its_own_graph() -> None:
    """End to end in one loop: measure while expanding, and pay for it on a later pass.

    `T` disconnects two ways at one reaction each, so nothing but the measurement can separate them.
    The probe reports a three-way mixture for the rule that reaches `A` and a clean step for the one
    that reaches `B`, and the search then returns the `B` route -- having learned that from its own
    expansion rather than from anything the caller knew in advance.
    """
    from vertsy.costs import selective, steps_equivalent

    reactions = {"T": [(0, ["A"]), (1, ["B"])], "A": [(2, ["S"])], "B": [(3, ["S"])]}
    mixtures = {0: 3}

    def measuring(products, precursors, hits):
        rows = np.asarray(hits, dtype=np.int64).reshape(-1, 4)
        return np.array(
            [mixtures.get(int(rule), 1) for rule in rows[:, 1]], dtype=np.int64
        )

    routes = run(
        measured(make_match(reactions, []), reactions={"sites": measuring}),
        heuristic=zero_heuristic,
        # The shape is the caller's now: a multiple of a step, over the `sites` channel.
        substrate=partial(selective, penalties={"sites": steps_equivalent}),
        max_routes=5,
    )

    assert routes
    # The clean disconnection, not the mixture: rule 1 reaches B.
    assert {route.rule for route in routes} == {1}


def test_without_a_penalty_a_measurement_is_written_and_never_read() -> None:
    """Which is what makes the probe safe to leave on: measuring costs nothing until it is priced."""

    reactions = {"T": [(0, ["A"]), (1, ["B"])], "A": [(2, ["S"])], "B": [(3, ["S"])]}

    def measuring(products, precursors, hits):
        rows = np.asarray(hits, dtype=np.int64).reshape(-1, 4)
        return np.full(rows.shape[0], 3, dtype=np.int64)

    plain = run(make_match(reactions, []), heuristic=zero_heuristic, max_routes=5)
    measured_only = run(
        measured(make_match(reactions, []), reactions={"outcomes": measuring}),
        heuristic=zero_heuristic,
        max_routes=5,
    )

    assert {route.rule for route in plain} == {route.rule for route in measured_only}

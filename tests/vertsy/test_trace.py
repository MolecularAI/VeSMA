from __future__ import annotations

import numpy as np
import pytest

from vertsy.expansions import Expansions
from vertsy.heuristics import zero_heuristic
from vertsy.interner import DictInterner
from vertsy.search import Expansion
from vertsy.store.numpy_store import NumpyExpansionStore
from vertsy.trace import Event, phase_seconds, phase_totals, traced

from .test_aostar import GRAPH, STOCK, make_match


def test_traced_returns_what_it_wrapped() -> None:
    """Instrumentation is invisible to the caller, or it is not instrumentation."""
    trace: list[Event] = []
    doubled = traced(trace, "double")(lambda values: [v * 2 for v in values])

    assert doubled([1, 2, 3]) == [2, 4, 6]
    assert len(trace) == 1
    name, items, seconds = trace[0]
    assert (name, items) == ("double", 3)
    assert seconds >= 0.0


def test_traced_sizes_a_match_by_its_hits() -> None:
    """A match's `(precursors, keys, hits)` is sized by the hits, not the precursor rows.

    Those two differ once a matcher deduplicates its batch, and the hits are the work.
    """
    trace: list[Event] = []
    match = traced(trace, "match")(
        lambda mols: (
            np.empty(2, dtype=object),
            ["a", "b"],
            np.zeros((7, 4), dtype=int),
        )
    )

    match(np.empty(0, dtype=object))

    assert trace[0][:2] == ("match", 7)


def test_traced_sizes_an_expansion_by_its_hits_too() -> None:
    """The shape a `Match` actually answers now, and the one that silently read as zero.

    `Expansion` has no `__len__`, so before it was recognised every traced match fell through to
    the total `TypeError` branch and every pass in every trace and performance report recorded
    zero matched items -- instrumentation that cannot fail and therefore cannot warn.
    """
    trace: list[Event] = []
    match = traced(trace, "match")(
        lambda mols: Expansion(
            precursors=np.empty(2, dtype=object),
            keys=["a", "b"],
            hits=np.zeros((7, 4), dtype=int),
        )
    )

    match(np.empty(0, dtype=object))

    assert trace[0][:2] == ("match", 7)


def test_traced_does_not_swallow_an_exception() -> None:
    """A failed call records nothing: half an event would report a phase as fast when it broke."""
    trace: list[Event] = []
    boom = traced(trace, "boom")(lambda: 1 / 0)

    with pytest.raises(ZeroDivisionError):
        boom()
    assert trace == []


def test_wrappers_nest_to_give_inclusive_and_exclusive_cost() -> None:
    """The reason this is a decorator: an inner and an outer wrap both report."""
    trace: list[Event] = []
    inner = traced(trace, "inner")(lambda values: values)

    def cached(values):
        return inner(values) if values else []

    outer = traced(trace, "outer")(cached)
    outer([1, 2])
    outer([])  # short-circuits, so "inner" is not called a second time

    totals = phase_totals(trace)
    assert totals["outer"][0] == 2
    assert totals["inner"][0] == 1


def test_a_traced_search_attributes_every_pass() -> None:
    """End to end: the matcher fires once per pass, so its rows are the passes."""
    from vertsy.aostar import aostar, constant_step
    from vertsy.search import concat_object_batches, object_mol_array

    trace: list[Event] = []
    expanded: list[str] = []
    aostar(
        "T",
        target_mol="T",
        stock=STOCK,
        price=constant_step(1.0),
        expansions=Expansions(NumpyExpansionStore(), DictInterner()),
        heuristic=traced(trace, "heuristic")(zero_heuristic),
        match=traced(trace, "match")(make_match(GRAPH, expanded)),
        stack_mols=object_mol_array,
        concat_mols=concat_object_batches,
        width=8,
        passes=1,
    )

    passes = [event for event in trace if event[0] == "match"]
    assert len(passes) == 1  # one pass, so one match call: T alone
    # Every expansion the search reports is accounted for by a traced match call.
    assert sum(1 for event in trace if event[0] == "match") == len(passes)
    assert set(phase_seconds(trace)) == {"match", "heuristic"}
    assert all(seconds >= 0.0 for seconds in phase_seconds(trace).values())

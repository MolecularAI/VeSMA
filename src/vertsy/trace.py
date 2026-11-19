"""Timing by wrapping the callables a search was already given.

`aostar` takes its matcher and its heuristic as arguments, so measuring them needs nothing
from the search itself — wrap what you pass in. That keeps attribution out of the algorithm,
which is the only way the numbers stay honest when the algorithm changes.

    trace: list[Event] = []
    routes = aostar(..., match=traced(trace, "match")(match),
                         heuristic=traced(trace, "heuristic")(heuristic))
    seconds = phase_seconds(trace)

Wrappers nest, deliberately: wrapping an inner callable and an outer one attributes both, and
wrapping the same callable inside and outside a cache gives cache-inclusive against
cache-exclusive cost — the hit rate — without either party knowing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from functools import wraps
from time import perf_counter
from typing import Any

__all__ = ["Event", "phase_seconds", "phase_totals", "traced"]

# One call of one wrapped callable: what it was, how much it handled, how long it took.
# Row order is call order, so a per-pass series needs no pass counter — the matcher fires
# once per pass, so its rows *are* the passes.
type Event = tuple[str, int, float]


def sized(result: Any) -> int:
    """How much a call handled, for the common shapes, or 0 when it says nothing.

    A match answers an :class:`vertsy.search.Expansion` and its size is the hits; a backend's raw
    ``(precursors, keys, hits)`` is the same thing before it is lifted; anything else with a length
    reports that. Deliberately total: a size that raised would turn instrumentation into a source of
    failure -- which is also why the `Expansion` case is a `hits` attribute test rather than an
    import, and why it comes first. `Expansion` has no `__len__`, so without it every traced match
    fell through to the `TypeError` branch and every pass reported zero.
    """
    hits = getattr(result, "hits", None)
    if hits is not None:
        return len(hits)
    if isinstance(result, tuple) and len(result) == 3:
        return len(result[2])
    try:
        return len(result)
    except TypeError:
        return 0


def traced[**P, T](
    trace: list[Event],
    name: str,
    size: Callable[[T], int] = sized,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Append one event per call to ``trace``, and otherwise change nothing.

    No ``try``/``finally``: an exception ends the run, and half an event is worse than none —
    it would report a phase as fast precisely when it failed.
    """

    def wrap(block: Callable[P, T]) -> Callable[P, T]:
        @wraps(block)
        def run(*args: P.args, **kwargs: P.kwargs) -> T:
            started = perf_counter()
            result = block(*args, **kwargs)
            trace.append((name, size(result), perf_counter() - started))
            return result

        return run

    return wrap


def phase_seconds(trace: Sequence[Event]) -> dict[str, float]:
    """Total seconds per wrapped callable."""
    totals: Counter[str] = Counter()
    for name, _, seconds in trace:
        totals[name] += seconds
    return dict(totals)


def phase_totals(trace: Sequence[Event]) -> dict[str, tuple[int, int, float]]:
    """Calls, items and seconds per wrapped callable."""
    calls: Counter[str] = Counter()
    items: Counter[str] = Counter()
    seconds: Counter[str] = Counter()
    for name, size, elapsed in trace:
        calls[name] += 1
        items[name] += size
        seconds[name] += elapsed
    return {name: (calls[name], items[name], seconds[name]) for name in calls}

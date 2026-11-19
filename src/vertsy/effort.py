"""What a search's ``width x passes`` effort actually bought.

:func:`vertsy.aostar.aostar` knows nothing about any of this. Its chooser is a callable it was
handed, so recording what each pass picked needs only a wrapper — the same trick
:mod:`vertsy.trace` plays on the matcher and the heuristic, and the reason ``Select`` is injected
at all::

    rounds: list[Round] = []
    routes = aostar(..., width=16, passes=8, select=recording_select(global_leaves, rounds))
    print(format_effort(rounds, routes, expansions))

Three cheap observations carry the whole module. A chooser returns leaves best first, so a leaf's
*position* is the ``width`` it needed. Sids are minted monotonically and never reused, so
comparing a sid against the previous pass's :meth:`Expansions.sid_count` watermark says whether it
was newly discovered or had been waiting. And the open frontier is arithmetic on counts the graph
already has, so measuring it costs nothing.

The point is a *reshaping* signal, not a score. Picks drawn from the deferred part of the frontier
say ``width`` was too small — with a wider pass they would have been opened when discovered. Picks
that are always freshly discovered, out of a frontier far larger than ``width``, say the effort
could be spent differently: the search never ran out of good candidates, so more of them are
arriving each pass than the budget can use.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .aostar import Select, reachable_leaves
from .core import K
from .costs import StepPrice
from .expansions import Expansions
from .metrics import GraphState, Valuation
from .routes import Route, reaction_products

__all__ = [
    "Round",
    "descent",
    "first_route",
    "format_effort",
    "kept_at",
    "per_reaction",
    "pick_order",
    "reached",
    "recording_select",
    "retention",
    "route_effort",
    "route_expansions",
    "steering",
]


@dataclass(frozen=True)
class Round:
    """One pass: what it could have opened, what it opened, and how new that was."""

    open_leaves: int
    """Open leaves when the pass began — interned, less expanded, less stock.

    Approximate in one case only: a target that is itself in stock is counted as expanded and
    as stock both, so the number can read one low. Worth the arithmetic being free.
    """

    selected: tuple[int, ...]
    """The chosen sids, best first, so an index *is* the ``width`` that leaf required.

    Never empty: a pass that chooses nothing has ended the search rather than spent any effort.
    """

    fresh: int
    """How many of ``selected`` the *immediately* preceding pass discovered.

    The rest were deferred, and deferral is per pass rather than cumulative: a leaf born two
    passes ago counts as deferred here even if it has only waited once, because what the number
    answers is "would a wider pass have taken this when it arrived".
    """

    reasons: tuple[tuple[float, float, int], ...]
    """Per pick, in ``selected`` order: ``(context, estimate, depth)`` — why it was chosen.

    ``context`` is Δ, what a solution graph through the leaf needs *besides* it; ``estimate`` is the
    leaf's own heuristic; ``depth`` is how many reactions below the target it sits. The three
    together say whether the ranking is being driven by the graph or by the heuristic, which a
    selected-sid list alone cannot: `context + estimate` is the sort key, so their *ratio* is the
    balance of power between committed cost and remaining guess.

    Empty when the chooser is not one this can read — see :func:`recording_select`.
    """


def recording_select(
    select: Select[K], rounds: list[Round], *, reasons: bool = True
) -> Select[K]:
    """Wrap a chooser so each pass appends one :class:`Round`, and change nothing else.

    ``reasons`` recomputes the sweep to record *why* each leaf was chosen. That is a second sweep
    per pass, which was unaffordable while the sweep was a quarter of the runtime and is not now
    that the array fold put it near 1%. It is still opt-out, and it is measured *outside* whatever
    timing wrapper the caller put on ``select`` only if they wrapped ``select`` first — see the note
    in ``perf_aostar_variants``, because getting that order wrong doubles the reported cost of
    selection and nothing complains.
    """
    marks: list[int] = []

    def choose(
        target: int,
        state: GraphState[K],
        value: Valuation[float],
        *,
        max_depth: int,
        limit: int,
        price: StepPrice,
    ) -> list[int]:
        marks.append(state.expansions.sid_count())
        # Sids at or above the *previous* pass's mark were minted during that pass. On the first
        # pass there is no previous mark and the only leaf is the target, which is new by
        # definition.
        born = marks[-2] if len(marks) > 1 else 0
        selected = select(
            target,
            state,
            value,
            max_depth=max_depth,
            limit=limit,
            price=price,
        )
        # A pass that chooses nothing is the search running dry, not a pass — recording it would
        # count a termination check as spent effort and drag the filled-width ratio down with it.
        if selected:
            why: tuple[tuple[float, float, int], ...] = ()
            if reasons:
                leaves = reachable_leaves(
                    target, state, value, max_depth=max_depth, price=price
                )
                # `depth_left` is the horizon the total was attained at, so the graph depth is
                # what the horizon has already been spent on.
                why = tuple(
                    (context, total - context, max_depth - depth_left)
                    for total, context, depth_left in (
                        leaves[sid] for sid in selected if sid in leaves
                    )
                )
            rounds.append(
                Round(
                    open_leaves=marks[-1] - len(state.steps) - len(state.stock_sids),
                    selected=tuple(selected),
                    fresh=sum(1 for sid in selected if sid >= born),
                    reasons=why,
                )
            )
        return selected

    return choose


def steering(rounds: Sequence[Round]) -> tuple[float, float, Counter[int]]:
    """``(mean context, mean estimate, depth histogram)`` over every pick.

    The question this answers is whether the ranking is being driven by the graph or by the
    heuristic. ``context + estimate`` is the sort key, so a mean context far above the mean estimate
    says the search is ordering by committed cost — which is depth — and the depth histogram is the
    visible consequence: picks pinned to the shallowest level still open, whatever the heuristic
    thinks of what lies below.
    """
    reasons = [reason for record in rounds for reason in record.reasons]
    if not reasons:
        return (float("nan"), float("nan"), Counter())
    return (
        sum(context for context, _, _ in reasons) / len(reasons),
        sum(estimate for _, estimate, _ in reasons) / len(reasons),
        Counter(depth for _, _, depth in reasons),
    )


def descent(rounds: Sequence[Round]) -> dict[int, tuple[int, float]]:
    """``pick depth -> (picks, mean estimate)`` — what a heuristic thinks each level down is worth.

    A pick at graph depth ``d`` has ``d`` reactions committed above it, so its estimate is the
    heuristic's guess at what remains. Reading the mean estimate against depth therefore reads the
    heuristic's *gradient*: how much closer to the stock it believes one reaction brings a molecule.

    Two things make this the right population despite being a biased one. It is the leaves the
    search actually ranked, so it measures the gradient where the ranking consumed it rather than
    over a corpus. And it needs no labels at all, so it says what a reaction is worth *to this
    heuristic, on this stock, with these rules* — which is exactly the quantity :func:`per_reaction`
    turns into a step cost.
    """
    totals: dict[int, tuple[int, float]] = {}
    for record in rounds:
        for _, estimate, depth in record.reasons:
            picks, total = totals.get(depth, (0, 0.0))
            totals[depth] = (picks + 1, total + estimate)
    return {depth: (picks, total / picks) for depth, (picks, total) in totals.items()}


def per_reaction(profile: Mapping[int, tuple[int, float]]) -> float:
    """What one reaction is worth to the heuristic, from the :func:`descent` profile.

    The mean decrement per level, which telescopes to the drop across the whole observed range
    divided by the levels spanned — so no weighting decision hides in it, and one sparsely picked
    deep level cannot dominate.

    Read it against the step cost the run charged. Charging more than a reaction is worth makes the
    ranking order by depth and the search breadth-first; charging much less leaves the step unable
    to break ties between candidates the heuristic scores alike. The point of reporting it is that
    the comparison is the user's to make, and with several heuristics it is the only principled way
    to weight them against each other: scale each so that what it charges matches what it buys.

    **It is an order-of-magnitude instrument, and it over-recommends.** Against PaRoutes n1 it reads
    0.077-0.084, stably across depths and effort shapes, and picking a step there catches an error
    of ten — but a step of 0.01 solves targets that 0.1 does not. The mean is the wrong statistic and
    the reason is structural: a route is only as findable as its **flattest** step, and real routes
    are far from uniform. One measured chain runs 0.559, 0.538, 0.557, 0.569, 0.422, 0 — four levels
    the fingerprint cannot tell apart, interior decrements of 0.021, 0.019 and 0.012, and the mean
    lifted to 0.11 by the last step into stock alone. A step above those interior values makes the
    step cost, not the heuristic, decide the order; worse, it can *manufacture* an uphill, since
    0.538 + 0.1 exceeds 0.559 where 0.538 + 0.01 does not.

    So treat this as a ceiling to come in under rather than a target to match, and prefer a low
    percentile of the per-level drop if the distribution is ever worth measuring properly.
    """
    depths = sorted(depth for depth, (picks, _) in profile.items() if picks)
    if len(depths) < 2:
        return float("nan")
    shallow, deep = depths[0], depths[-1]
    return (profile[shallow][1] - profile[deep][1]) / (deep - shallow)


def reached(rounds: Sequence[Round]) -> dict[int, tuple[int, int]]:
    """Expanded sid -> the smallest ``(width, passes)`` that would have opened it."""
    return {
        sid: (rank + 1, index + 1)
        for index, record in enumerate(rounds)
        for rank, sid in enumerate(record.selected)
    }


def pick_order(rounds: Sequence[Round]) -> dict[int, int]:
    """Expanded sid -> its position in the whole run's pick sequence, counting from one.

    The ``(width, passes)`` pair says what *shape* of effort would have reached a leaf; this says
    how much effort had been spent by the time it was. They are different questions and the second
    is the one a budget is set from.
    """
    return {
        sid: position
        for position, sid in enumerate(
            (sid for record in rounds for sid in record.selected), start=1
        )
    }


def route_expansions(
    route: Route[K],
    expansions: Expansions[K],
    order: Mapping[int, int],
) -> int | None:
    """How many products had been expanded when ``route`` became complete.

    A route is finished the moment its last remaining reaction is opened, so this is the greatest
    pick position among its products. Over the routes a search returned, the minimum is where the
    *first* one appeared — which is the number a budget should be set against, since everything
    after it is buying alternatives rather than an answer.
    """
    products = tuple(reaction_products(route))
    if not products:
        return 0  # a stock target was already solved before the first pass
    positions: list[int] = []
    for sid in expansions.lookup(products).tolist():
        position = order.get(int(sid))
        if position is None:
            return None
        positions.append(position)
    return max(positions)


def route_effort(
    route: Route[K],
    expansions: Expansions[K],
    reached: Mapping[int, tuple[int, int]],
) -> tuple[int, int] | None:
    """The smallest ``(width, passes)`` whose expansions cover every reaction in ``route``.

    ``None`` when some product of the route was never chosen by the recorded chooser, which
    means the route came from a graph this recording did not build.
    """
    # A tuple, not a sorted list: ``K`` is only constrained to be hashable, so a perfectly valid
    # custom key need not be orderable — and the answer here is a max over the products, which
    # does not care what order they arrive in.
    products = tuple(reaction_products(route))
    if not products:
        return (0, 0)  # a stock target: no reaction, so no effort
    costs: list[tuple[int, int]] = []
    for sid in expansions.lookup(products).tolist():
        cost = reached.get(int(sid))
        if cost is None:
            return None
        costs.append(cost)
    return (max(width for width, _ in costs), max(passes for _, passes in costs))


def retention(
    routes: Sequence[Route[K]],
    expansions: Expansions[K],
    reached: Mapping[int, tuple[int, int]],
) -> Counter[tuple[int, int]]:
    """How many routes each ``(width, passes)`` is the smallest sufficient effort for.

    Cumulative sums over this give the retention curve: the routes a smaller budget would still
    have produced. It is a **lower bound** on what that budget finds, not a prediction — a
    narrower or shorter run ranks differently from the first pass on, so a route counted lost
    here may well be found by a differently-guided search, and one counted kept only stays kept
    if the same molecules land in the same places.
    """
    counts: Counter[tuple[int, int]] = Counter()
    for route in routes:
        cost = route_effort(route, expansions, reached)
        if cost is not None:
            counts[cost] += 1
    return counts


def kept_at(
    counts: Counter[tuple[int, int]],
    *,
    width: int,
    passes: int,
    divisors: Sequence[int] = (2, 4, 8),
) -> tuple[list[float], list[float]]:
    """Share of priced routes surviving ``width/d`` and ``passes/d``, for each ``d``.

    Halves alone answer "was this budget roughly right"; a curve answers "what should the next one
    be", which is the question a caller actually has. Each axis is reported independently — a route
    needs both, so these are marginals and not a joint prediction.
    """
    priced = sum(counts.values())
    if not priced:
        return ([float("nan")] * len(divisors), [float("nan")] * len(divisors))
    by_width = [
        sum(n for (needed, _), n in counts.items() if needed <= max(1, width // d))
        / priced
        for d in divisors
    ]
    by_passes = [
        sum(n for (_, needed), n in counts.items() if needed <= max(1, passes // d))
        / priced
        for d in divisors
    ]
    return (by_width, by_passes)


def first_route(
    routes: Sequence[Route[K]],
    expansions: Expansions[K],
    order: Mapping[int, int],
) -> int | None:
    """Expansions spent before the earliest of ``routes`` was complete.

    The budget-setting number. Everything spent after it bought alternatives rather than an
    answer, which is worth having separately because a pool is the goal — but a caller sizing a
    run for *solve rate* wants this one.
    """
    positions = [
        position
        for position in (route_expansions(route, expansions, order) for route in routes)
        if position is not None
    ]
    return min(positions) if positions else None


def format_effort(
    rounds: Sequence[Round],
    routes: Sequence[Route[K]],
    expansions: Expansions[K],
) -> str:
    """A short report: what the effort cost, what bound it, and what a smaller one would keep."""
    if not rounds:
        return "no passes ran"

    spent = sum(len(record.selected) for record in rounds)
    fresh = sum(record.fresh for record in rounds)
    widest = max(len(record.selected) for record in rounds)
    filled = sum(1 for record in rounds if len(record.selected) == widest)
    counts = retention(routes, expansions, reached(rounds))

    lines = [
        f"{spent} products over {len(rounds)} passes "
        f"(widest {widest}, filled in {filled}/{len(rounds)})",
        f"{fresh}/{spent} picks freshly discovered, {spent - fresh} deferred",
        "open leaves per pass: "
        + " ".join(str(record.open_leaves) for record in rounds),
    ]
    if not counts:
        return "\n".join(lines + ["no routes to price"])

    earliest = first_route(routes, expansions, pick_order(rounds))
    if earliest is not None:
        lines.append(
            f"first route complete after {earliest} of {spent} expansions "
            f"({100 * earliest // max(spent, 1)}% of the budget); the rest bought alternatives"
        )
    by_width, by_passes = kept_at(counts, width=widest, passes=len(rounds))
    lines.append(
        "routes kept at 1/2, 1/4, 1/8 of the budget — width "
        + " ".join(f"{share:.0%}" for share in by_width)
        + ", passes "
        + " ".join(f"{share:.0%}" for share in by_passes)
    )
    lines.append(f"{sum(counts.values())} routes; cumulative retention:")
    for axis, label in ((0, "width "), (1, "passes")):
        bounds = sorted({cost[axis] for cost in counts})
        lines.append(
            f"  {label} "
            + " ".join(
                f"<={bound}:{sum(n for cost, n in counts.items() if cost[axis] <= bound)}"
                for bound in bounds
            )
        )
    return "\n".join(lines)

from __future__ import annotations

from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from functools import partial
from itertools import product as cartesian
from math import isclose
from typing import Protocol

import numpy as np

from .core import K, M
from .costs import (
    StepPrice,
    Substrate,
    Table,
    application_price,
    attaining_rule,
    by_rule,
    constant_step,
)
from .expansions import Expansions
from .heuristics import Heuristic
from .metrics import (
    TROPICAL,
    GraphState,
    Valuation,
    outside_valuation,
    prepare_graph,
    valuation,
)
from .routes import Route, SidRoute, decode_route
from .search import Layer, Match, MoleculeBatch, step
from .store import SidStep
from .tropical import (
    TropicalValuation,
    bottleneck_leaf_scores,
    open_leaf_totals,
    tropical_valuation,
)

__all__ = [
    "Expand",
    "Fold",
    "Prepare",
    "Rank",
    "Select",
    "bottleneck_fold",
    "bottleneck_leaves",
    "closure_fold",
    "by_context",
    "by_inside",
    "by_total",
    "constant_step",
    "committed_application_leaves",
    "weighted",
    "anytime_step",
    "aostar",
    "descent_leaves",
    "global_leaves",
    "iter_best_sid_routes",
    "reachable_leaves",
    "rows_of",
]


# How a leaf's two halves — the context above it and its own remaining value — combine into a
# sort key. The choice is the search strategy; see `global_leaves`.
type Rank = Callable[[float, float], float]


#: How a pass builds the graph state it values. Injected for the same reason `fold` is, and
#: defaulting to :func:`vertsy.metrics.prepare_graph`. Worth a name because it is called **once per
#: pass** on a graph that grows every pass, which is not obvious from the call site and was the
#: largest unattributed cost in the loop.
type Prepare[K] = Callable[[Expansions[K], Collection[K]], GraphState[K]]

#: How a pass expands its chosen leaves, defaulting to :func:`vertsy.search.step`. It *contains*
#: the injected `match`, so anything timing both must treat `match` as a component of this rather
#: than a sibling of it.
type Expand[M, K] = Callable[..., Layer[M]]


class Fold(Protocol[K]):
    """How a pass values its graph. Always min-plus; the choice is only how it is computed.

    Estimates arrive as a dense sid-indexed array rather than a callable, because the array form
    is what lets :func:`vertsy.tropical.tropical_valuation` stay in arrays — and a search that
    keeps them densely anyway loses nothing by saying so.
    """

    def __call__(
        self,
        state: GraphState[K],
        *,
        leaf_value: np.ndarray,
        price: StepPrice,
        max_depth: int,
    ) -> Valuation[float]: ...


def closure_fold(
    state: GraphState[K],
    *,
    leaf_value: np.ndarray,
    price: StepPrice,
    max_depth: int,
) -> Valuation[float]:
    """The generic semiring fold, in the search's terms.

    ``max_depth`` goes unused: this fold is lazy, so a horizon is something its caller asks for
    rather than something it prepares. The array fold has to know in advance, which is exactly the
    trade — laziness for vectorisation.
    """
    return valuation(
        state,
        TROPICAL,
        leaf_value=lambda sid: float(leaf_value[sid]),
        step_cost=partial(application_price, price),
    )


def bottleneck_fold(
    state: GraphState[K],
    *,
    leaf_value: np.ndarray,
    price: StepPrice,
    max_depth: int,
) -> Valuation[float]:
    """Min-plus, but an application costs its step plus its **worst** precursor, not their sum.

    Two things change together, and they are the same change seen from either end.

    As a *score*, a route's value stops being its reaction count and becomes its **longest linear
    sequence** — branches that could be run in parallel no longer add. That is a real objective in
    process chemistry rather than a convenience, and often the one that matters: a convergent route
    of eight steps in two branches of four ships sooner than a linear route of six.

    As *guidance*, it removes a penalty that summing cannot avoid. An estimate charged once per
    precursor makes every bimolecular disconnection look like it multiplies the work remaining.
    Measured on a PaRoutes route the search never recovered, the correct first disconnection split a
    molecule at distance 0.684 into halves at 0.709 and 0.581, so the sum read **1.290** — nearly
    double the parent — while the worst half read 0.709. Three of that route's molecules were never
    generated at all. Under ``max`` the disconnection prices at what its hardest branch needs.

    Admissibility survives in its own terms: the longest sequence below a molecule is at least the
    longest sequence below its hardest precursor, so an estimate bounded by the cheapest step is
    still optimistic *about this objective*. It is not admissible about reaction count, and it is
    not meant to be — a different question, honestly answered, rather than the same one approximated.

    **The outside ranking needs care.** On four PaRoutes targets at 32x8 it originally
    reproduced the summing fold expansion for expansion, verdict for verdict. Not because the fold is
    inert — 149 of 225 expanded products take different values, and one target reads 0.690 against
    0.837 — but because the two agree on the part of the frontier a selection reads. The rankings are
    identical for the first **19,855** of 23,018 leaves, and no width goes near that. An open leaf's
    own estimate has no precursors to conjoin, so only its *context* can differ, and the leaves that
    reach the top are shallow ones whose siblings are already buyable, where the sum of a stock
    sibling and its max are both zero. Roughly 9-22% of applications do have two or more non-stock
    precursors; they are simply never what a pass is choosing between.

    Hard22 exposed the missing case: two non-stock branches can be the best application, then be
    demoted by the scalar global-leaf context. Use :func:`committed_application_leaves` to retain
    work for both while the application remains live.

    The inside fold is exact and vectorised, so it costs what the summing fold costs. The outside
    sweep can compute a sibling maximum with :func:`vertsy.tropical.leave_one_out_max`, but a
    nested ``step + max`` context is a function of the selected leaf, not one additive scalar.
    Consequently :func:`global_leaves` uses an upper-bound ranking for this fold.
    """
    return tropical_valuation(
        state,
        leaf_value=leaf_value,
        price=price,
        max_depth=max_depth,
        conjoin=max,
    )


class Select(Protocol[K]):
    """How a pass chooses what to expand next, given the valued graph.

    The one part of AO* with a real choice in it, so the one part injected. Both
    implementations return sids, cheapest first, capped at ``limit``.
    """

    def __call__(
        self,
        target: int,
        state: GraphState[K],
        value: Valuation[float],
        *,
        max_depth: int,
        limit: int,
        price: StepPrice,
    ) -> list[int]: ...


def descent_leaves(
    target: int,
    state: GraphState[K],
    value: Valuation[float],
    *,
    max_depth: int,
    limit: int,
    price: StepPrice,
) -> list[int]:
    """The unexpanded products of the cheapest solution graph, most promising first.

    AO*'s selection half, and the reason expansion batches naturally: an application is a
    conjunction, so *every* precursor it names has to be solved, and they are all on the
    solution graph at once. Their molecules are exactly the batch the matcher wants. The
    batch is therefore discovered, not chosen — ``limit`` only caps it, and anything left
    over stays open for the next pass.

    Applications valued ``zero`` are skipped rather than descended into. One of their
    precursors is already dead or past the budget, so no amount of expanding their siblings
    can revive them; when every application of ``target`` is ``zero`` this returns nothing
    and the search has proved the target unreachable within ``max_depth``.

    Values come from the caller's :func:`vertsy.metrics.valuation`, the same fold that
    produced the cost — selection and valuation cannot disagree about which route is best
    because there is only one of them.
    """
    found: list[int] = []
    seen: set[tuple[int, int]] = set()

    def descend(product: int, depth_left: int) -> None:
        if product in state.stock_sids or depth_left <= 0:
            return
        key = (product, depth_left)
        if key in seen:
            return
        seen.add(key)
        if not state.is_expanded(product):
            found.append(product)
            return
        applications = state.applications(product)
        if not applications:
            return
        # Scored once and kept: `application` is the fold's unmemoized half, so asking for the
        # winner's value a second time would re-walk its precursors.
        best_value, best = min(
            ((value.application(step, depth_left), step) for step in applications),
            key=lambda scored: scored[0],
        )
        if best_value == TROPICAL.zero:
            return
        for precursor in best[1]:
            descend(precursor, depth_left - 1)

    descend(target, max_depth)
    # An unexpanded product has the same value at every live depth — its heuristic — so any
    # positive horizon ranks them. Cheapest first, so a truncated batch keeps the best of it.
    found.sort(key=lambda product: value.node(product, max_depth))
    return found[:limit]


def by_total(context: float, inside: float) -> float:
    """Rank by additive context plus inside value; exact when precursors are also summed."""
    return context + inside


def by_inside(context: float, inside: float) -> float:
    """Rank a leaf by what is left below it, ignoring its context. Greedy, depth-first-ish."""
    return inside


def by_context(context: float, inside: float) -> float:
    """Rank a leaf by what stands above it, ignoring the leaf. Breadth-first-ish."""
    return context


def weighted(context_weight: float) -> Rank:
    """``inside + w * context``: interpolate between inside-only and additive total ranking.

    ``w = 1`` is :func:`by_total`, ``w = 0`` is :func:`by_inside`. Values above one bias toward
    leaves in shallow contexts, which is a search decision rather than a valuation one — the
    graph's cost is unchanged, only the order in which leaves are opened.
    """

    def rank(context: float, inside: float) -> float:
        return inside + context_weight * context

    return rank


def reachable_leaves(
    target: int,
    state: GraphState[K],
    value: Valuation[float],
    *,
    max_depth: int,
    price: StepPrice,
) -> dict[int, tuple[float, float, int]]:
    """Every open leaf a solution graph could still run through: sid -> ``(total, context, depth)``.

    The sweep is taken from whichever fold produced ``value``, and this dispatch is a *performance*
    choice only: an array fold already holds the inside table the sweep needs, so asking it in
    arrays too avoids rebuilding it. Both sweeps aggregate siblings by ``value.conjoin``, and
    ``test_open_leaves_match_the_generic_sweep`` pins them equal entry for entry under the max fold
    as well as the sum. Neither is the consistent one; the value carries its own aggregation so
    that whichever sweep runs cannot pick a different algebra from the fold that produced it.

    Separate from :func:`global_leaves` because the ranking is not the only thing that wants this.
    Instrumentation wants to know *why* a leaf was chosen — its context against its own estimate,
    and how deep it sits — and recomputing the sweep to find out is affordable now that it is
    around 1% of a run (:mod:`vertsy.effort`).
    """
    if isinstance(value, TropicalValuation):
        return open_leaf_totals(target, state, value, max_depth=max_depth)
    return outside_valuation(
        target,
        state,
        value,
        max_depth=max_depth,
        step_cost=partial(application_price, price),
    ).leaves


def global_leaves(
    target: int,
    state: GraphState[K],
    value: Valuation[float],
    *,
    max_depth: int,
    limit: int,
    price: StepPrice,
    rank: Rank = by_total,
) -> list[int]:
    """The most promising open leaves anywhere in the graph, cheapest first.

    Where :func:`descent_leaves` can only offer what one solution graph exposes — often a single
    molecule, whatever ``limit`` says — this ranks *every* reachable open leaf and so can always
    fill a batch. With an additive precursor fold, each is scored by the cost of the cheapest
    solution graph constrained to contain it, ``delta + rn``: the context above it from
    :func:`vertsy.metrics.outside_valuation` plus its own value from the same fold that produced
    the target's cost. With a bottleneck fold, this is an upper bound; see below.

    **For the additive fold, at ``limit == 1`` this is exactly AO\\*.** The target's own total is
    the graph's cost ``f``;
    every total is the cost of *some* solution graph, so none is below ``f``; and on any
    ``f``-achieving graph each choice attains its node's value, so every node of it totals ``f``.
    The cheapest leaf therefore lies on some best solution graph — precisely the set a descent
    may choose from — unless no best graph has an open leaf left, in which case the solved and
    estimated values agree and the search has already finished. Beyond the first, leaves total
    more than ``f``: they are speculation, which is what a batch buys and what it costs.

    ``rank`` decides *which* of the two halves to order on, and so which search this is: the
    default sum is the faithful value when the fold also sums precursors, and the k=1 argument
    above holds for that combination alone. With a bottleneck fold, the scalar outside context is
    an additive upper bound: it charges a selected leaf and its siblings even though the fold takes
    their maximum. Use :func:`committed_application_leaves` when branches of a promising
    bottleneck application need to keep receiving work across passes. The
    alternatives are here to be measured against it — if :func:`by_inside` ranks nearly as well,
    the context sweep is not paying for the pass it costs.

    **That difference is invisible to retrosynthesis, and measurably so.** The context charges *the
    other* siblings of a leaf, not every precursor of its application, and sum over a singleton is
    max over a singleton — so for a two-precursor reaction the two folds agree exactly. Built
    minimally, with one application offering two leaves estimated 0.5 and another offering one at
    0.9, both folds give context 0.501 and the same order; add a third precursor and they split,
    0.501 against 1.001. Chemistry almost never supplies the third: of 261 steps taken from a route
    archive, 85 had one precursor, 171 had two and 5 had three. Switching the fold on a real target
    accordingly moved the search by one molecule in 47,000 seen.

    The trap this leaves is worth naming, since it cost a day. Scoring *applications* by a fold over
    all their precursors — the obvious offline experiment — measures something this function never
    computes, and max and sum really do differ there. Rank leaves, with their context, or the
    measurement will not transfer.

    Ties break by sid so a run is reproducible; leaves with no finite total never appear,
    whatever ``rank`` would have made of them.

    The sweep is taken from whichever fold produced ``value``; see :func:`reachable_leaves`.
    """
    leaves = reachable_leaves(target, state, value, max_depth=max_depth, price=price)
    ranked = sorted(
        (rank(context, total - context), sid)
        for sid, (total, context, _) in leaves.items()
        if total != TROPICAL.zero
    )
    return [sid for _, sid in ranked[:limit]]


def bottleneck_leaves(
    target: int,
    state: GraphState[K],
    value: Valuation[float],
    *,
    max_depth: int,
    limit: int,
    price: StepPrice,
) -> list[int]:
    """Rerank all open leaves by exact max-fold cost, without a root commitment.

    Requires an array bottleneck fold with uniform finite nonnegative reaction costs.
    Scores include sibling obligations using maximum rather than addition. Ties use
    sid order; this selector alone does not guarantee diversity between applications.

    **Refused on the price, not on the numbers.** The closed form behind
    :func:`vertsy.tropical.bottleneck_leaf_scores` multiplies one cost by a depth, so it needs
    every application to cost the same -- and a substrate-aware price is non-uniform from the
    first pass even while nothing has been measured yet and every cost is still equal. Checking
    the array would let such a search run happily for eight passes and fail on the ninth, the
    moment a mixture turns up. ``price.uniform`` says what the caller *configured*, so this fails
    at the first pass instead. The array check downstream stays as the backstop.

    Production is unaffected: it pairs the bottleneck fold with
    :func:`committed_application_leaves`, whose fill is :func:`global_leaves`, and that ranking
    reads an outside sweep which needs no uniformity at all.
    """
    if not isinstance(value, TropicalValuation):
        raise ValueError("bottleneck selection requires an array valuation")
    if not price.uniform:
        raise ValueError(
            "bottleneck selection requires uniform reaction costs, and this price is "
            "substrate-aware; rank with global_leaves or committed_application_leaves instead"
        )
    scores = bottleneck_leaf_scores(target, state, value, max_depth=max_depth)
    return sorted(scores, key=lambda sid: (scores[sid], sid))[:limit]


def committed_application_leaves(
    fill: Select[K] = global_leaves,
) -> Select[K]:
    """Keep pursuing every branch of the target's best live application.

    The returned selector commits when the target is first expanded. On later passes it reserves
    room for the open leaves on each precursor's current best solution graph, interleaving the
    branches so a large one cannot crowd out the others. Any remaining batch capacity comes from
    ``fill``. A commitment is replaced only when it becomes unreachable; once it solves, no
    reservation is needed because the target itself is solved.

    This is deliberately a selector factory: the commitment belongs to one search run, just like
    the mutable schedule returned by :func:`anytime_step`. It addresses the failure mode where a
    multi-precursor application wins the inside fold once, all its leaves are expanded, and the
    next global ranking abandons one branch before the application can finish.

    It is also a bounded compromise rather than a new valuation algebra. The reserved leaves
    follow the fold exactly through :func:`descent_leaves`; the fill ranking still supplies broad
    exploration and may still use an approximate scalar context for a bottleneck fold.
    """
    commitment: SidStep | None = None

    def select(
        target: int,
        state: GraphState[K],
        value: Valuation[float],
        *,
        max_depth: int,
        limit: int,
        price: StepPrice,
    ) -> list[int]:
        nonlocal commitment
        if state.is_expanded(target) and not state.is_solved(target):
            applications = state.applications(target)
            if (
                commitment is None
                or value.application(commitment, max_depth) == TROPICAL.zero
            ):
                live = (
                    (value.application(application, max_depth), application)
                    for application in applications
                )
                best_value, best = min(
                    live, default=(TROPICAL.zero, None), key=lambda pair: pair[0]
                )
                commitment = best if best_value != TROPICAL.zero else None

        branches: list[list[int]] = []
        if commitment is not None and not state.is_solved(target):
            branches = [
                descent_leaves(
                    precursor,
                    state,
                    value,
                    max_depth=max_depth - 1,
                    limit=limit,
                    price=price,
                )
                for precursor in commitment[1]
            ]

        reserved: list[int] = []
        seen: set[int] = set()
        for position in range(max((len(branch) for branch in branches), default=0)):
            for branch in branches:
                if position < len(branch) and branch[position] not in seen:
                    seen.add(branch[position])
                    reserved.append(branch[position])
                    if len(reserved) == limit:
                        return reserved

        broad = fill(
            target,
            state,
            value,
            max_depth=max_depth,
            limit=limit,
            price=price,
        )
        return (
            reserved
            + [sid for sid in broad if sid not in seen][: limit - len(reserved)]
        )

    return select


def scored(
    estimates: np.ndarray, sids: np.ndarray, scores: Sequence[float] | np.ndarray
) -> np.ndarray:
    """Record ``scores`` for ``sids``, growing the dense estimate array to fit.

    Grown by concatenation rather than with a capacity scheme: sids only ever arrive, the array is
    a few hundred kilobytes at any realistic size, and one memcpy per pass is far below the noise
    of the pass that produced them.
    """
    wanted = int(sids.max()) + 1 if len(sids) else 0
    if wanted > len(estimates):
        estimates = np.concatenate(
            [estimates, np.full(wanted - len(estimates), TROPICAL.zero)]
        )
    estimates[sids] = np.asarray(scores, dtype=np.float64)
    return estimates


def unscored_leaves(
    target: int,
    state: GraphState[K],
    scored_sids: np.ndarray,
    *,
    max_depth: int,
) -> list[int]:
    """Open leaves this search will actually consult that nobody has estimated.

    An open leaf is an interned product neither expanded nor buyable, and a caller handing over a
    populated graph has to hand over those molecules too — they cannot be rebuilt from keys, which
    need not be invertible. Silently skipping one would drop every route through it and report the
    target unsolved, so this is the honest failure.

    **Scoped to what the fold can reach**, by a breadth-first walk from ``target`` rather than a
    scan of the store. A store is shared: it may hold independent graphs for other targets, whose
    open leaves are none of this search's business, and demanding molecules for them would refuse
    a perfectly well-formed resume. The horizon bounds it the same way — a leaf at distance ``d``
    is consulted with ``max_depth - d`` of budget left, and a product past the horizon is
    unreachable whatever its estimate, so only ``d <= max_depth - 1`` is ever read.

    Each product is visited once, at its shortest distance, which is where it has the most budget
    left; reaching it again later could only ask less of it. That also makes a cycle finite without
    a separate guard.

    Only the initial state can be wrong. Everything the search finds afterwards is scored off the
    very batch the matcher returned, in the same call that discovered it.
    """
    scored = set(scored_sids.tolist())
    missing: list[int] = []
    seen = {target}
    level = [target]
    for _ in range(max_depth):
        below: list[int] = []
        for product in level:
            if product in state.stock_sids:
                continue
            if not state.is_expanded(product):
                if product not in scored:
                    missing.append(product)
                continue
            for _rules, precursors in state.applications(product):
                for precursor in precursors:
                    if precursor not in seen:
                        seen.add(precursor)
                        below.append(precursor)
        if not below:
            break
        level = below
    return missing


def rows_of(haystack: np.ndarray, needles: np.ndarray) -> np.ndarray:
    """Where each of ``needles`` sits in ``haystack``, which is unsorted but distinct.

    Both the frontier's sids and a selection out of it are plain arrays, so locating one in the
    other is a sort and two lookups rather than a Python loop. Raises if a needle is absent —
    that would mean the search selected a leaf it holds no molecule for, which is a bug here
    rather than a condition to paper over.
    """
    order = np.argsort(haystack)
    found = np.searchsorted(haystack, needles, sorter=order)
    if np.any(found >= len(haystack)):
        raise KeyError("selected a leaf that is not on the frontier.")
    rows = order[found]
    if not np.array_equal(haystack[rows], needles):
        raise KeyError("selected a leaf that is not on the frontier.")
    return rows


def anytime_step(
    *,
    greedy: float,
    relaxed: float,
    select: Select[K] = global_leaves,
    substrate: Substrate | None = None,
    tables: Mapping[str, Table] | None = None,
) -> tuple[StepPrice, Select[K]]:
    """Search greedily until a route exists, then relax the step cost and keep going.

    Returns the ``(price, select)`` pair to hand to :func:`aostar`. They come together because
    they share the schedule: the price reads it, the selection advances it, and a caller who took
    only one of them would get a search that never relaxes or one that relaxes nothing.

    ``substrate`` and ``tables`` make the scheduled step substrate-aware: a shape from
    :mod:`vertsy.costs` -- :func:`~vertsy.costs.selective` for a multiple of the step,
    :func:`~vertsy.costs.compounded` for a yield -- partially applied to its own parameters and
    handed the moving price here. Which shape is the caller's to choose, because the two mean
    different things about a measurement; *when* it is composed is not, which is why this takes a
    function rather than a finished price. Both or neither: a shape with nothing to measure charges
    nothing, and a measurement nothing prices is dead weight.

    **Why a schedule at all.** The ranking is ``s * steps + h``, which is weighted A* with effective
    weight ``1/s``. A constant ``s`` forces one choice for the whole run, and the two ends want
    opposite things: ``s = 0.001`` is weight 1000, greedy enough to dive straight to a route and
    never look sideways, while ``s = 1`` is admissible, broad, and often fails to arrive at all
    (34/48 against 44/48 on seeded depth-5 targets). Measured, the cost of choosing greedy is that
    **after the first route the rest of the budget buys nothing**: ways to begin, reaction count and
    longest sequence are all unchanged from 128 expansions to 512.

    So do what anytime weighted A* does — find something, then lower the weight. On six n1 depth-5
    targets at 32x16, ways to begin summed 231 flat against **783** with this schedule: 3.4x more
    chemistry for the same 481 expansions, every target still solved, and one target's longest
    sequence improved from 3 to 2.

    **Two rungs, and the switch on the first solution.** Longer ladders were measured and are not
    worth their complexity: 0.001/0.01/0.1/1.0 scores 760 because the middle rungs earn almost
    nothing, and adding 10.0 past admissible reproduces this arm *target for target* — at ``s = 1``
    the estimate is already negligible against the step, so relaxing further reorders nothing. The
    ladder has a natural top and this reaches it in one step.

    The relaxed phase costs wall clock the expansion count does not show — 1.6 to 2.1x, because a
    broad search interns more distinct molecules to fingerprint. It cannot cost solve rate, since it
    only begins once a route is in hand.
    """
    if not greedy > 0.0 or not relaxed > 0.0:
        raise ValueError(
            f"step costs must be positive, got {greedy!r} and {relaxed!r}."
        )
    if (substrate is None) != (tables is None):
        raise ValueError("pass a price shape together with what it prices, or neither.")
    schedule = [greedy]

    def base(rule: int) -> float:
        return schedule[0]

    price = by_rule(base)
    if substrate is not None and tables is not None:
        price = substrate(price, tables)

    def scheduled(
        target: int,
        state: GraphState[K],
        value: Valuation[float],
        *,
        max_depth: int,
        limit: int,
        price: StepPrice,
    ) -> list[int]:
        chosen = select(
            target, state, value, max_depth=max_depth, limit=limit, price=price
        )
        # Advance *after* selecting: the valuation in hand was folded under the current cost, so
        # switching mid-pass would rank leaves by a value computed in a different algebra. The new
        # cost takes effect at the next pass's fold, which keeps every pass internally consistent.
        if state.is_solved(target):
            schedule[0] = relaxed
        return chosen

    return price, scheduled


def iter_best_sid_routes(
    product: int,
    state: GraphState[K],
    value: Valuation[float],
    depth_left: int,
    trail: frozenset[int] = frozenset(),
    *,
    price: StepPrice,
) -> Iterator[SidRoute]:
    """Yield the cheapest solved routes represented by a bounded exact valuation.

    Routes are acyclic: ``trail`` carries the ancestors of ``product``, so no molecule repeats
    on a root-to-leaf path. The depth budget is a second, independent bound. One route per
    disconnection — see :func:`vertsy.routes.iter_solved_sid_routes` for why co-canonical
    templates are not alternative routes.

    A step is labelled with the template that **attained** its price, not the first of them. The
    fold values a disconnection at its cheapest template, so under any non-unit ``rule_cost`` the
    two differ, and reporting the wrong one would put a route's stated chemistry at odds with the
    cost the search selected it for. ``rule_cost`` therefore has to be the same one the valuation
    used; ties keep the lowest id, which is what unit costs always gave.

    Applications come from ``state``, not from :meth:`vertsy.expansions.Expansions.steps`, which
    rebuilds the whole edge relation per call — O(edges) at every node of every route, against
    the pre-grouped map's O(1). It only matters because a caller wanting many routes walks this
    many times.

    "Attains the optimum" is a comparison **within a tolerance**, not an exact float equality. It
    used to be exact and got away with it only because extraction runs on unit costs and free
    stock leaves, so every value was a small integer. Any real template prior — or a fold that
    sums in a different order, as the vectorised one does — makes an exact test silently return
    *no routes at all*, which is the worst possible failure for this function: a graph full of
    routes reported as unsolved.
    """
    if product in state.stock_sids:
        yield SidRoute(product, None)
        return
    if depth_left <= 0 or product in trail:
        return

    best = value.node(product, depth_left)
    if best == TROPICAL.zero:
        return

    next_trail = trail | frozenset({product})
    for application in state.applications(product):
        # `best` is finite here, so an unreachable application cannot pass as close to it.
        if not isclose(value.application(application, depth_left), best, rel_tol=1e-9):
            continue
        rules, precursor_sids = application
        child_options = tuple(
            iter_best_sid_routes(
                precursor,
                state,
                value,
                depth_left - 1,
                next_trail,
                price=price,
            )
            for precursor in precursor_sids
        )
        for children in cartesian(*child_options):
            yield SidRoute(product, attaining_rule(price, application), tuple(children))


def aostar(
    target: K,
    *,
    target_mol: M,
    stock: Collection[K],
    expansions: Expansions[K],
    heuristic: Heuristic[M],
    match: Match[M, K],
    stack_mols: Callable[[Sequence[M]], MoleculeBatch[M]],
    concat_mols: Callable[[Sequence[MoleculeBatch[M]]], MoleculeBatch[M]],
    width: int | Sequence[int],
    passes: int,
    max_depth: int | None = None,
    select: Select[K] = global_leaves,
    fold: Fold[K] = tropical_valuation,
    prepare: Prepare[K] = prepare_graph,
    expand: Expand[M, K] = step,
    price: StepPrice,
    substrate: Substrate | None = None,
    resume: tuple[Sequence[K], MoleculeBatch[M]] | None = None,
    max_routes: int = 5,
    stop_when_solved: bool = False,
) -> list[Route[K]]:
    """Search for routes to ``target`` by expanding only what the value function points at.

    AO* over the AND/OR graph, in the three parts Hughes separates: ``expansions`` generates
    the graph, :func:`vertsy.metrics.valuation` values it, and ``select`` chooses
    from it. Each pass values the graph from the target, takes the ``width`` most promising open
    leaves, and expands them in **one** batched :func:`vertsy.search.step` — the frontier is a
    batch by construction, which is the shape the vectorised matcher wants.

    Guidance is always min-plus. Ranking alternatives requires a selective ``oplus``, so
    there is no semiring parameter here: a non-selective one (:func:`log_semiring`) would
    make "the best application" meaningless. Score a finished graph with those instead.

    ``heuristic`` estimates the reactions still needed to reach ``stock``, and is evaluated
    in batch when new leaves appear, never inside the fold. If it never exceeds the cheapest
    ``rule_cost`` it is admissible, which is what makes the ranking trustworthy — it orders leaves
    by a value no real route can beat. A larger one searches greedier and deeper for the same
    effort, the usual weighted-A* bargain, at the cost of that ordering being optimistic in an
    unbounded way.

    **``rule_cost`` has no default, on purpose.** A step cost and a heuristic's range are one knob
    seen from two ends — only their ratio decides anything, since scaling every cost leaves min-plus
    ranking untouched — and there is no ratio that is right for every fingerprint, stock and rule
    set. The unit step this used to default to was not neutral: it made a reaction 10x more
    expensive than one is worth to an ECFP4 Jaccard distance, which orders the frontier by depth and
    searches breadth-first. On 48 seeded PaRoutes targets spanning reference depths 5 to 7 it solved
    34 against ``constant_step(0.1)``'s 44, the margin widening with depth, and two targets it could
    not reach at *any* effort fall on half the budget once the step is cheap.

    So state both weights and calibrate them: pass ``constant_step(s)`` here and ``weight`` to the
    heuristic, start near ``weight = 10 * s``, and read :func:`vertsy.effort.per_reaction` — it
    reports how far the estimate actually falls per reaction on *your* stock, from the run's own
    picks and with no corpus in it. When the charge matches the fall, the two are in balance.

    ``prepare`` and ``expand`` are injected for the same reason ``fold`` and ``select`` are, and
    both default to the function they replace. They exist because the pass loop's own cost was
    unattributable: a bench can time what it injects, and these two are where the time between the
    named phases actually went. ``prepare`` is called **once per pass**, which is the thing worth
    knowing about it -- a graph that grows every pass is re-indexed every pass -- and ``expand``
    contains ``match``, so a bench that times both must not add them together.

    Either way the routes returned are the cheapest **found**, not the cheapest that exist:
    optimality was a claim the old proof-then-stop loop could make, and a fixed effort cannot.

    Keys and molecules are the package's own ``K`` and ``M``, and nothing here reads either.
    Molecules are held **only in batches**, never one at a time: the open leaves live as a
    ``(frontier_sids, frontier_mols)`` pair kept parallel, and the only operations on the batch
    are the two :func:`vertsy.search.step` already uses — selection by index array and
    ``concat_mols``. That is what lets this run wherever the matcher does, since a batch on a
    device cannot be indexed down to an element at all (`CudaMoleculeBatch.__getitem__` raises on
    an ``int`` by design, and there is no decoder back from a packed row). It also means nothing
    is ever rebuilt from a key, so the key space need not be invertible — the WL hashes the CUDA
    and Metal backends use are not.

    ``heuristic`` scores each new leaf batch as it arrives, straight off the one ``step``
    returned, so no molecule is ever fetched to be estimated. Pass stock and target through the
    same keyer the matcher keys precursors with, or stock membership compares two different
    alphabets and quietly never matches.

    ``substrate`` turns the price substrate-aware, by composing a shape from :mod:`vertsy.costs`
    over the measurement channels the graph is carrying -- one entry per channel, since the two that
    exist measure different chemistry and deserve to be read differently.
    :func:`~vertsy.costs.selective` reads a measurement as a multiple of a step and
    :func:`~vertsy.costs.compounded` reads it as a fraction of the material; which of those is meant
    is the caller's to say, and nothing here depends on the answer. It is composed here
    rather than by the caller because what it reads belongs to the graph, and the graph is rebuilt
    every pass: a measurement taken while expanding pass three is priced from pass four onwards,
    which is the same one-pass lag the anytime schedule has and for the same reason. Without one,
    measurements sit in the store unread and every cost is exactly what it was.

    ``resume`` carries the open leaves of a graph this search did not build. A caller passing a
    populated ``expansions`` has to pass their molecules too — it built them, and they cannot be
    recovered from keys — so the restart state is the pair, not the graph alone. Omit it and any
    such leaf raises, which is the honest failure: silently skipping one would drop every route
    through it.

    ``rule_cost`` prices an application, and defaults to one per reaction so a route's value counts
    its reactions. It is also where a template prior belongs — an edge cost need not vanish at the
    goal the way a node heuristic must.

    **It is the natural home for the free parameter, too.** Min-plus ranking is invariant under
    scaling every cost by the same factor, so a cheap step with a ``[0, 1]`` heuristic is the *same
    search* as a unit step with a heuristic scaled up to match: ``rule_cost = 0.25`` against
    ``h = 1 - tanimoto`` orders identically to ``rule_cost = 1`` against ``h = 4 * (1 - tanimoto)``.
    The first reads better, because a fingerprint distance keeps its own range and the step carries
    the conversion — which is also how a log-probability cost arrives, a step worth ``-log 0.9``
    against a leaf penalty in the same units. ``max_depth`` is a hop count, so the horizon does not
    move under rescaling, and :func:`iter_best_sid_routes` compares with a *relative* tolerance, so
    co-optimality does not either.

    What rescaling does move is admissibility: an estimate is free of charge only while it stays
    below the **cheapest** ``rule_cost``, so halving the step halves the room the heuristic has
    before AO* stops promising the cheapest route. That is the weighted-A\\* bargain, stated in the
    units it is actually paid in.

    ``width`` can also be a sequence of positive widths, one per pass. This schedules
    expansion batch size by iteration, not by molecular depth. Unused slots are not
    carried forward, and the initial target counts as the first pass.

    **For a fixed width, effort is ``width`` and ``passes``**: open the best ``width`` leaves, ``passes`` times, for
    a budget of ``width * passes`` products. Two numbers rather than one because their *shape*
    matters and no single total can express it — the same 128 products spent as ``16 x 8``
    explore in parallel, as ``8 x 16`` exploit the leaders and reach deeper. ``16 x 8`` is a
    reasonable place to start, ``32 x 16`` and ``64 x 32`` the next rungs;
    :mod:`vertsy.effort` reports which of the two axes actually bound a run so the next one can
    be reshaped rather than guessed.

    ``passes`` also fixes the horizon: starting from nothing, ``p`` passes cannot build anything
    deeper than ``p`` reactions, so ``max_depth`` defaults to ``passes`` — the deepest route the
    budget could reach, with no promise it gets there. Override it *below* ``passes`` to search
    wide for shallow routes, which also makes the fold cheaper.

    That ceiling holds **only for a graph this search builds itself.** A caller resuming a graph
    that already has levels in it needs a horizon covering those levels plus the new passes: a
    ``T -> A -> B`` graph needs one pass to open ``B`` but a horizon of three to see the route,
    and the default of ``passes`` cannot. So the ceiling is enforced only when ``expansions``
    arrives empty, and a resumed search is expected to say how deep to look — there is no way to
    infer it, since a cyclic graph has no single depth.

    Termination is the budget, or a pass with nothing left worth opening — every remaining leaf
    unreachable within the horizon, which is a proved dead end. There is deliberately no exit on
    a proved-best route: one route is not the goal, and stopping at the first would forfeit the
    rest of a pool that later scoring has to choose from. For the same reason a batch is never
    torn in half to fit a remaining budget, and routes are returned whether or not a cheaper one
    might still have been out there — a fixed effort makes no optimality claim.

    ``max_routes`` caps extraction, and ``0`` skips it. Aggregate route properties come off the
    graph without materialising anything (:mod:`vertsy.metrics`), so a caller who only wants
    metrics should not pay for a cartesian product.

    ``stop_when_solved`` returns at the first pass that finds the target solved, abandoning the
    anytime schedule's repair phase. It answers a different question: *is this target reachable
    at all with these templates*, where the cheapest route found is beside the point and the
    remaining passes are pure cost. With it the width schedule becomes an upper bound rather
    than a budget actually spent, so timings from such a run are not comparable with the
    benchmarks here.

    ``fold`` is how the graph gets valued, and it is min-plus either way — the choice is only
    whether that happens in arrays (:func:`vertsy.tropical.tropical_valuation`, the default,
    which values every product at every horizon at once) or lazily through memoized closures
    (:func:`closure_fold`, the generic semiring fold every route metric uses). They agree to
    floating-point tolerance rather than bit for bit, because the array form sums an
    application's siblings by subtraction; the closure form is the specification and is kept as
    the control arm.
    """
    if passes <= 0:
        raise ValueError(f"passes must be positive, got {passes}.")
    widths = (width,) * passes if isinstance(width, int) else tuple(width)
    if len(widths) != passes or any(limit <= 0 for limit in widths):
        raise ValueError(
            "width must be positive, with one width per pass for a schedule."
        )
    preloaded = bool(expansions.store.expanded().size)
    if max_depth is None:
        max_depth = passes
    elif max_depth < 0:
        raise ValueError(f"max_depth must be non-negative, got {max_depth}.")
    elif not preloaded and max_depth > passes:
        raise ValueError(
            f"max_depth must be between 0 and passes ({passes}), got {max_depth}."
        )
    if max_routes < 0:
        raise ValueError(f"max_routes must be non-negative, got {max_routes}.")

    target_sid = int(expansions.intern([target])[0])

    # The open leaves, as a sid array and a molecule batch kept parallel. Every molecule this
    # search will ever need is here: an expanded product's own molecule is never wanted again,
    # because the graph records what it decomposed into.
    frontier_sids = np.array([target_sid], dtype=np.int64)
    frontier_mols = stack_mols([target_mol])
    if resume is not None:
        resume_keys, resume_mols = resume
        frontier_sids = np.concatenate([frontier_sids, expansions.intern(resume_keys)])
        frontier_mols = concat_mols([frontier_mols, resume_mols])
    # Estimates live in a dense sid-indexed array, not a dict: sids are minted from zero and never
    # reused, so the array *is* the natural shape, and the fold can then read them without going
    # through Python once per leaf.
    estimates = scored(
        np.full(0, TROPICAL.zero), frontier_sids, heuristic(frontier_mols)
    )
    # A dense array cannot report a gap the way a dict's KeyError did, so the gap is checked for
    # once, here, against the part of the caller's graph this search will read. Everything
    # discovered later is scored at birth, so this is the only place one can appear.
    unscored = unscored_leaves(
        target_sid,
        prepare(expansions, stock),
        frontier_sids,
        max_depth=max_depth,
    )
    if unscored:
        raise KeyError(
            "resume must supply a molecule for every open leaf reachable from the target; "
            f"missing {expansions.decode(unscored)}."
        )

    for pass_width in widths:
        state = prepare(expansions, stock)
        # Checked here because this pass's graph is already built and `is_solved` reads it without
        # a fold, an extraction or a second valuation -- so asking costs nothing, and asking at the
        # *top* of a pass means the answer is acted on before any further expansion is paid for.
        # Off by default: the anytime schedule's whole point is that a first solution is not the
        # last word, and every benchmark here measures the full budget. A caller that only wants a
        # solved/unsolved verdict -- a template-coverage sweep, say -- wants the opposite.
        if stop_when_solved and state.is_solved(target_sid):
            break
        # A substrate-aware price has to be composed here rather than by the caller, because what it
        # reads is a property of the graph and the graph is rebuilt every pass. A measurement taken
        # during pass three is priced from pass four onwards, which is the same one-pass lag the
        # anytime schedule has and for the same reason: a pass folds before it selects.
        pass_price = price if substrate is None else substrate(price, state.outcomes)
        estimated_value = fold(
            state,
            leaf_value=estimates,
            price=pass_price,
            max_depth=max_depth,
        )
        selected = select(
            target_sid,
            state,
            estimated_value,
            max_depth=max_depth,
            limit=pass_width,
            price=pass_price,
        )
        if not selected:
            break

        chosen = np.array(selected, dtype=np.int64)
        layer = expand(
            chosen,
            frontier_mols[rows_of(frontier_sids, chosen)],
            expansions,
            stock,
            match=match,
        )
        fresh_sids, fresh_mols = layer.sids, layer.mols

        # What was expanded leaves the frontier; what the matcher found joins it. `step`
        # deduplicates only within its own call, so a leaf discovered several passes ago can
        # come back as a precursor and would otherwise be filed twice.
        keep = ~np.isin(frontier_sids, chosen)
        kept_sids = frontier_sids[keep]
        novel = ~np.isin(fresh_sids, kept_sids)

        # Score the new leaves off the very batch the matcher produced: estimated once, at
        # birth, with no molecule ever fetched to be estimated and none scored twice. A layer
        # that arrives already estimated is read rather than re-scored, which is what lets a
        # device backend answer with a column and never hand a molecule back to the host.
        fresh_estimates = (
            heuristic(fresh_mols[novel])
            if layer.estimate is None
            else np.asarray(layer.estimate[novel], dtype=np.float64)
        )
        estimates = scored(estimates, fresh_sids[novel], fresh_estimates)

        # Both halves are batch-preserving, the same algebra `step` uses internally.
        frontier_sids = np.concatenate([kept_sids, fresh_sids[novel]])
        frontier_mols = concat_mols([frontier_mols[keep], fresh_mols[novel]])

    if not max_routes:
        return []
    state = prepare(expansions, stock)
    if not state.is_solved(target_sid):
        return []
    solved_price = price if substrate is None else substrate(price, state.outcomes)
    # Stock leaves cost nothing, so this values only what was actually found — the estimate the
    # loop searched on has no part in an answer, and the leaves attaining this value are the
    # cheapest routes the budget bought.
    solved_value = valuation(
        state,
        TROPICAL,
        leaf_value=lambda product: TROPICAL.zero,
        step_cost=partial(application_price, solved_price),
    )
    solved = solved_value.node(target_sid, max_depth)
    if solved == TROPICAL.zero:
        return []
    routes = []
    for sid_route in iter_best_sid_routes(
        target_sid,
        state,
        solved_value,
        max_depth,
        price=solved_price,
    ):
        routes.append(decode_route(sid_route, expansions))
        if len(routes) >= max_routes:
            break
    return routes

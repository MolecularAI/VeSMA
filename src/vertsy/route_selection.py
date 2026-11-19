"""Select short, structurally varied solved routes from an expansion graph.

The graph can encode more routes than can be materialized.  An admissible bound on
each unfinished tree lets a heap find complete routes in score order, while one
stream per first disconnection keeps the candidate sample from being monopolized
by a prolific application.  The final round robin follows the first reaction
choice at which candidate routes differ, ignoring template-only duplicates.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import cache
from itertools import count

from .core import K
from .costs import (
    StepPrice,
    constant_step,
    cost_of_factor,
    factor_of_cost,
    rule_price,
)
from .metrics import GraphState, minimum_route_depth
from .routes import SidRoute

__all__ = [
    "DEFAULT_PRICE",
    "RouteScore",
    "SelectedRoute",
    "TooSimilar",
    "RouteSelection",
    "application_costs",
    "cheapest_application",
    "typical_application",
    "disconnections",
    "select_solved_routes",
]


@dataclass(frozen=True)
class RouteScore:
    depth: int
    reactions: int
    utility: float
    base_step_factor: float
    """What the templates alone are worth, before any substrate measurement.

    `step_factor ** reactions` under a constant price, so it reads as a route length; the product of
    the rules' own yields under a per-rule base. Printed as "templates" because only the first of
    those is a length.
    """
    stock_factor_product: float
    reaction_factor_product: float


@dataclass(frozen=True)
class SelectedRoute:
    route: SidRoute
    score: RouteScore
    # One-based position in a canonical, top-down walk of molecule decisions.
    # None marks the first route in the report.
    first_distinct_decision: int | None


@dataclass(frozen=True)
class TooSimilar:
    """A route that was good enough to report and too much like one already reported.

    A first-class outcome rather than a tie-break, because the thing it rejects is the thing a
    naive "next best" would always pick: the same plan with its least significant step swapped.
    Carrying the resemblance out of the selector is what lets a report say *which* route this one
    looked like and *where* it finally differed, so a reader can widen ``diversity_cutoff``
    knowingly instead of wondering what was dropped.
    """

    route: SidRoute
    score: RouteScore
    first_distinct_decision: int
    resembled: int
    """One-based rank of the reported route it most resembled."""


@dataclass(frozen=True)
class Censored:
    """A route the engine could cost but could not make.

    The selector ranks a graph of *applications*: a template matched a molecule, which is a claim
    about patterns rather than about a product. Replay is where the claim is cashed, and a step
    that will not reproduce -- or that a refusing veto row condemns -- invalidates the route
    without saying anything about the one that replaces it. So a censored candidate is dropped and
    the next *diverse* one is taken, rather than the report silently coming back short.
    """

    route: SidRoute
    score: RouteScore
    reason: str
    """What the validator said, verbatim: a veto code, a replay mismatch, whatever censored it."""


@dataclass(frozen=True)
class RouteSelection:
    routes: tuple[SelectedRoute, ...]
    candidates_considered: int
    partial_states_explored: int
    truncated: bool
    shortest_depth: int | None
    best_shortest_reactions: int | None
    stopped_after_limit: bool = False
    too_similar: tuple[TooSimilar, ...] = ()
    """Candidates the diversity floor rejected, newest last. Empty when no floor was set."""
    censored: tuple[Censored, ...] = ()
    """Candidates a validator refused, newest last. Empty when no validator was passed."""


@dataclass(frozen=True)
class Pending:
    path: tuple[int, ...]
    product: int
    ancestors: frozenset[int]


@dataclass(frozen=True)
class ChosenStep:
    path: tuple[int, ...]
    product: int
    rule: int
    precursors: tuple[int, ...]


@dataclass(frozen=True)
class PartialRoute:
    pending: tuple[Pending, ...]
    steps: tuple[ChosenStep, ...]
    reactions: int
    depth: int
    reaction_log_cost: float
    stock_log_cost: float


@dataclass(frozen=True)
class Candidate:
    route: SidRoute
    score: RouteScore
    log_cost: float


def disconnections(
    state: GraphState[K],
    cost: Callable[[int, tuple[int, ...]], float],
    product: int,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Every distinct disconnection of one molecule, each labelled with its cheapest rule.

    One entry per distinct precursor *multiset*, so "three disconnections" means three different
    sets of starting materials and not three templates. Co-canonical templates collapse onto the
    cheapest of them, which is why a route is priced and labelled at the selective template where
    one exists.

    Keyed by the sorted *tuple*, never a set: a disconnection may legitimately need two equivalents
    of the same precursor, and a set would collapse ``(A, A)`` to ``(A,)``. That loses a reactant
    from the emitted route, and -- since a measurement is stored against the precursor multiset --
    makes the price look up a key nothing was ever written under, so a measured step reads as
    unprobed. The same canonical form :func:`vertsy.costs.measured` uses.

    Module-level, and memoised by the caller rather than here, because **two readers must agree**:
    :func:`select_solved_routes` chooses among these, and
    :mod:`vertsy.route_alternatives` reports on them. A second implementation would drift, and the
    report would then explain alternatives the selector never saw.
    """
    by_precursors: dict[tuple[int, ...], set[int]] = {}
    for rules, precursors in state.applications(product):
        by_precursors.setdefault(tuple(sorted(precursors)), set()).update(rules)
    return tuple(
        sorted(
            (
                (
                    min(rules, key=lambda rule: (cost(rule, precursors), rule)),
                    precursors,
                )
                for precursors, rules in by_precursors.items()
            ),
            key=lambda step: (step[1], step[0]),
        )
    )


def application_costs(
    state: GraphState[K], cost: Callable[[int, tuple[int, ...]], float]
) -> list[float]:
    """Every reaction the graph offers, priced. Sorted, because both callers want an order statistic."""
    return sorted(
        cost(rule, tuple(sorted(precursors)))
        for product in state.steps
        for rules, precursors in state.applications(product)
        for rule in rules
    )


def typical_application(
    state: GraphState[K], cost: Callable[[int, tuple[int, ...]], float]
) -> float:
    """The median reaction cost, which is the unit a slack should be counted in.

    A slack of "three extra steps" means three *ordinary* steps -- the protect/react/deprotect
    detour it was calibrated against. Counting it in the **cheapest** step available instead makes
    the budget collapse as soon as the steps stop being equal: under a per-rule base, one template
    curated at a 0.95 yield halves the cheapest step and so halves the detour the slack admits,
    anywhere in the graph, for every route. Nothing about the chemistry changed.

    Identical to :func:`cheapest_application` under a uniform price, where median and minimum are
    the same number -- which is why this was invisible until a base varied, and why swapping to it
    changes no recorded measurement.
    """
    costs = application_costs(state, cost)
    return costs[len(costs) // 2] if costs else math.inf


def cheapest_application(
    state: GraphState[K], cost: Callable[[int, tuple[int, ...]], float]
) -> float:
    """The cheapest reaction the graph offers, which bounds how many can fit in a budget.

    The *work* bound wants this and not the median: a budget divided by the cheapest step is the
    most reactions that budget could possibly hold, which is the bound the heap needs to be
    correct. What a slack is counted in is a different question -- see
    :func:`typical_application`.

    A graph with no affordable application at all returns ``inf``, and the caller then has no
    routes to bound.
    """
    costs = application_costs(state, cost)
    return costs[0] if costs else math.inf


def route_tokens(route: SidRoute) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Template-insensitive decisions in canonical top-down tree order."""
    if route.rule is None:
        return ()
    children = tuple(sorted(route.precursors, key=lambda child: child.product))
    return ((route.product, tuple(child.product for child in children)),) + tuple(
        token for child in children for token in route_tokens(child)
    )


def _resembled(
    tokens: tuple[tuple[int, tuple[int, ...]], ...],
    previous: Sequence[tuple[tuple[int, tuple[int, ...]], ...]],
) -> int:
    """One-based rank of the already-reported route sharing the longest prefix with ``tokens``."""
    shared = [
        next(
            (
                index
                for index, (left, right) in enumerate(zip(tokens, other))
                if left != right
            ),
            min(len(tokens), len(other)),
        )
        for other in previous
    ]
    return 1 + max(range(len(shared)), key=shared.__getitem__)


def diverse_order(candidates: tuple[Candidate, ...]) -> tuple[Candidate, ...]:
    """Round robin at the first decision that differs, then at deeper decisions."""
    keyed = [(route_tokens(item.route), item) for item in candidates]

    def arrange(
        items: list[tuple[tuple[tuple[int, tuple[int, ...]], ...], Candidate]],
        index: int,
    ) -> list[Candidate]:
        if len(items) <= 1:
            return [item for _, item in items]
        groups: dict[
            tuple[int, tuple[int, ...]] | None,
            list[tuple[tuple[tuple[int, tuple[int, ...]], ...], Candidate]],
        ] = {}
        for tokens, item in items:
            token = tokens[index] if index < len(tokens) else None
            groups.setdefault(token, []).append((tokens, item))
        if len(groups) == 1:
            token = next(iter(groups))
            if token is None:
                return [item for _, item in items]
            return arrange(items, index + 1)
        streams = deque(
            iter(arrange(group, index + 1))
            for group in sorted(
                groups.values(),
                key=lambda group: min(
                    (
                        item.log_cost,
                        item.score.depth,
                        item.score.reactions,
                        repr(tokens),
                    )
                    for tokens, item in group
                ),
            )
        )
        ordered = []
        while streams:
            stream = streams.popleft()
            try:
                ordered.append(next(stream))
            except StopIteration:
                continue
            streams.append(stream)
        return ordered

    return tuple(arrange(keyed, 0))


DEFAULT_PRICE = constant_step(cost_of_factor(0.9))
"""The report's incumbent step cost, written as the cost it always was.

``0.9`` per step is ``-log 0.9 = 0.105`` as a cost, which is also the step the *search* was
calibrated to. The two were arrived at independently and coincide; saying so once here is cheaper
than leaving two spellings of one number in two modules.
"""


def select_solved_routes(
    target: int,
    state: GraphState[K],
    *,
    limit: int = 20,
    price: StepPrice = DEFAULT_PRICE,
    stock_factor: Callable[[K], float] | None = None,
    cost_slack: float = 3.0,
    depth_slack: int = 2,
    candidate_limit: int = 400,
    state_limit: int = 20_000,
    diversity_cutoff: int | None = None,
    validate: Callable[[SidRoute], str | None] | None = None,
) -> RouteSelection:
    """Select solved routes by cost, then shape, then plan diversity.

    **Cost first, and that is the whole point of the ordering.** This used to rank
    ``(depth, reactions, utility)``, and under that key no route preference of any strength could
    prefer a longer route: a protection detour costs two extra reactions, so it lost on the second
    key before the third was ever consulted. Since a clean step and a stock leaf are the only
    things a default price charges for, and both are constant per reaction, ``log_cost`` was a
    strictly increasing function of ``reactions`` and the third key said nothing the second had
    not. **Ranking by cost reproduces that ordering exactly wherever nothing but reactions is
    priced**, and differs only where something else is -- which is the property that makes this
    safe rather than a new opinion.

    Depth and reaction count stay, as the shape tie-breakers and as the bounds that make
    enumeration finite. They are no longer the objective.

    ``price`` is the same :class:`vertsy.costs.StepPrice` the search folds with, so a substrate
    penalty applied to one is applied to both. It is read per ``(rule, precursors)``, which is why
    a disconnection reachable selectively by one template and unselectively by another is priced
    and labelled at the selective one.

    ``stock_factor`` stays its own parameter rather than joining the price. A terminal cost is
    something the search's fold does not have at all -- :func:`vertsy.metrics.valuation` returns
    the semiring ``one`` for stock -- so folding it in would claim an alignment that does not
    exist.

    ``cost_slack`` is what the report will consider beyond the cheapest route it found, in
    **step-equivalents** of the cheapest reaction the graph offers. It replaces the old
    ``reaction_slack`` because a count cannot express the thing being widened for: at a slack of 2
    a protect/react/deprotect detour fitted on a one-reaction target and was excluded on every
    other, which made it *invisible* rather than merely unpreferred. The default of 3 admits one
    such detour anywhere, and ``max_reactions`` is derived from it -- a route within ``max_cost``
    cannot hold more reactions than ``max_cost`` divided by the cheapest one.

    ``depth_slack`` remains a work bound on the *shape*: it caps how much longer a reported route's
    longest branch may be than the shortest one found, and therefore how many detours can stack.

    The limits bound memory and work, not chemistry. ``truncated`` reports when either bound
    prevented a complete scan of eligible routes. A route revisiting a molecule on one ancestry
    path is excluded, even though relaxed lower bounds may include such walks.

    **Measured against the lexicographic ordering it replaces**, on 30 PaRoutes targets at 32x8
    against medchem + hartenfeller + protecting-groups: search is untouched, so solve decisions were
    identical 30 for 30; 5 of the 17 solved targets reported their routes in a different order; and
    total selection time was unchanged at 0.03 s. Where an order changed it became **monotone in
    cost**, which the old key was not -- ``diverse_order`` interleaves streams, so a 2-reaction route
    could be reported after a 3-reaction one. On two hard-22 targets at 16x6 against raw68k, every
    reported route came out identical, which is the safety net doing its job: raw68k prices nothing
    but reactions, so the two orderings coincide exactly.

    ``validate`` is asked of each candidate in report order and returns **the reason it cannot be
    made**, or ``None`` to let it through. It is the second of two veto layers by design: the
    search may price a veto, which steers towards more promising branches while keeping every
    measurement, and extraction may refuse one, which is where a route is actually lost. Refusing
    here rather than in the search is what makes the refusal *reportable* -- the graph still holds
    the condemned step, so the report can name it as an alternative that was turned down.

    A refused candidate is replaced, not merely dropped: it never joins the set the diversity floor
    measures against, so the route that takes its place is still chosen for being different from
    the routes a reader will actually see. Because both the floor and the validator consume
    candidates after discovery, setting either makes discovery aim for ``candidate_limit`` rather
    than ``limit`` -- the caller's own work budget, spent only when something might eat into it.
    """
    if limit <= 0 or candidate_limit < limit or state_limit <= 0:
        raise ValueError(
            "route and work limits must be positive; candidate_limit >= limit"
        )
    if depth_slack < 0 or cost_slack < 0.0 or not math.isfinite(cost_slack):
        raise ValueError("route slacks must be non-negative and finite")
    stock_weight = stock_factor if stock_factor is not None else lambda key: 1.0
    shortest = minimum_route_depth(target, state)
    if shortest is None:
        return RouteSelection((), 0, 0, False, None, None)
    if target in state.stock_sids:
        # Answered before any bound is computed, because a buyable target has no reactions to
        # bound and no cheapest one to count a slack in.
        bought = factor_of_cost(
            cost_of_factor(stock_weight(*state.expansions.decode([target])))
        )
        score = RouteScore(0, 0, bought, 1.0, bought, 1.0)
        return RouteSelection(
            (SelectedRoute(SidRoute(target, None), score, None),), 1, 0, False, 0, 0
        )

    @cache
    def stock_log_cost(product: int) -> float:
        key = state.expansions.decode([product])[0]
        return cost_of_factor(stock_weight(key))

    @cache
    def step_log_cost(rule: int, precursors: tuple[int, ...]) -> float:
        return rule_price(price, rule, precursors)

    @cache
    def clean_log_cost(rule: int) -> float:
        """What the same template would cost with nothing charged against it.

        Not a cost the report charges -- it is the denominator that lets a route's utility be
        *decomposed*, so a reader can see how much of a route's price was its length and how much
        was its selectivity. It therefore has to be the price's **multiplicative identity**: the
        count every multiplier shape charges exactly 1.0 for. Any other choice can exceed the real
        cost and turn the decomposition's remainder negative, which `factor_of_cost` refuses.

        That identity is ``1`` -- one distinct product, a clean step -- and it is the *only* count
        that qualifies. Zero is a measurement: it says the forward chemistry ran and made nothing,
        which :func:`vertsy.costs.doubted` charges for. `UNPROBED` is also a measurement, of a
        different kind: it says nobody looked, which :func:`vertsy.costs.uncurated` charges for.
        This used to pass `UNPROBED` and was correct for as long as nothing priced it; the day a
        run did, a genuinely clean step cost *less* than its own denominator and the remainder went
        negative. With nothing pricing `UNPROBED` the two are identical, so the fix changes no
        recorded number.
        """
        return price.charge(rule, (1,) * len(price.channels))

    @cache
    def applications(product: int) -> tuple[tuple[int, tuple[int, ...]], ...]:
        # Memoised here, computed by the module-level `disconnections` so the report explains
        # exactly the alternatives this function chose among.
        return disconnections(state, step_log_cost, product)

    # Annotate each molecule and remaining depth with the cheapest total cost, and -- separately --
    # the fewest reactions. Both are memoized on `(product, depth_left)`, and the budget is what
    # makes either well founded on a cyclic molecule graph: every recursive call spends one, so a
    # molecule reached through itself simply runs out. Two folds rather than one lexicographic
    # tuple because they are used for different things, and a component of a lexicographic minimum
    # is not a minimum: the reaction count of the *cheapest* plan is no lower bound on reactions,
    # and pruning on it would drop routes that exist.
    best_steps: dict[tuple[int, int], tuple[int, tuple[int, ...]]] = {}

    @cache
    def best_plan(product: int, depth_left: int) -> float:
        """The cheapest total cost of finishing ``product`` within ``depth_left`` reactions.

        Admissible by construction: cost is additive over a route tree, so the cheapest completion
        of each pending branch sums to a lower bound on the cheapest completion of all of them.
        """
        if product in state.stock_sids:
            return stock_log_cost(product)
        if depth_left <= 0 or not state.is_solved(product):
            return math.inf
        best = math.inf
        for rule, precursors in applications(product):
            children = [best_plan(child, depth_left - 1) for child in precursors]
            if any(math.isinf(child) for child in children):
                continue
            value = step_log_cost(rule, precursors) + sum(children)
            if value < best:
                best = value
                best_steps[(product, depth_left)] = (rule, precursors)
        return best

    @cache
    def fewest_reactions(product: int, depth_left: int) -> float:
        """The fewest reactions that finish ``product`` within ``depth_left``. A work bound only."""
        if product in state.stock_sids:
            return 0.0
        if depth_left <= 0 or not state.is_solved(product):
            return math.inf
        return min(
            (
                1.0
                + sum(fewest_reactions(child, depth_left - 1) for child in precursors)
                for _rule, precursors in applications(product)
            ),
            default=math.inf,
        )

    def best_route(
        product: int, depth_left: int, ancestors: frozenset[int]
    ) -> SidRoute | None:
        if product in ancestors:
            return None
        if product in state.stock_sids:
            return SidRoute(product, None)
        step = best_steps.get((product, depth_left))
        if step is None:
            return None
        rule, precursors = step
        children = tuple(
            best_route(child, depth_left - 1, ancestors | {product})
            for child in precursors
        )
        if any(child is None for child in children):
            return None
        return SidRoute(
            product, rule, tuple(child for child in children if child is not None)
        )

    best_shortest = fewest_reactions(target, shortest)
    if math.isinf(best_shortest):
        return RouteSelection((), 0, 0, False, shortest, None)
    max_depth = shortest + depth_slack
    cheapest_step = cheapest_application(state, step_log_cost)
    typical_step = typical_application(state, step_log_cost)
    if not math.isfinite(cheapest_step) or cheapest_step <= 0.0:
        # No affordable reaction anywhere, so there is no unit to count a slack in and nothing to
        # enumerate. A free reaction is the same problem seen from the other side: every route
        # would fit every budget.
        return RouteSelection((), 0, 0, False, shortest, int(best_shortest))
    # A route costing at most `max_cost` cannot hold more reactions than that divided by the
    # cheapest one available, which turns one chemistry bound into the work bound the heap needs.
    # The slack is counted in a *typical* step and the work bound in the *cheapest* one. They are
    # the same number under a uniform price, and must not be under a per-rule one: a single
    # well-curated template would otherwise shrink the detour budget for every route in the graph.
    max_cost = best_plan(target, max_depth) + cost_slack * typical_step
    max_reactions = int(max_cost / cheapest_step) if cheapest_step > 0.0 else 0

    def bound(partial: PartialRoute) -> tuple[float, float, float]:
        # For each possible completed depth, the pending branches can be priced independently with
        # their own remaining-depth budgets.  This joint bound avoids combining a shallow route's
        # depth with a different route's low cost, which made the old heap chase impossible pairs.
        # Cost leads, so the heap pops in cost order and the first complete route out is the
        # cheapest one -- which is what makes the enumeration a ranking rather than a sample.
        for depth in range(partial.depth, max_depth + 1):
            budgets = [depth - len(pending.path) for pending in partial.pending]
            costs = [
                best_plan(pending.product, budget)
                for pending, budget in zip(partial.pending, budgets)
            ]
            if any(math.isinf(cost) for cost in costs):
                continue
            counts = [
                fewest_reactions(pending.product, budget)
                for pending, budget in zip(partial.pending, budgets)
            ]
            reactions = partial.reactions + sum(counts)
            total = partial.reaction_log_cost + partial.stock_log_cost + sum(costs)
            if reactions > max_reactions or total > max_cost:
                continue
            return (total, depth, reactions)
        return (math.inf, math.inf, math.inf)

    def apply(
        partial: PartialRoute, rule: int, precursors: tuple[int, ...]
    ) -> PartialRoute | None:
        pending = partial.pending[0]
        if any(
            child in pending.ancestors or child == pending.product
            for child in precursors
        ):
            return None
        children = tuple(
            Pending(
                pending.path + (index,), child, pending.ancestors | {pending.product}
            )
            for index, child in enumerate(precursors)
            if child not in state.stock_sids
        )
        stock_cost = sum(
            stock_log_cost(child) for child in precursors if child in state.stock_sids
        )
        return PartialRoute(
            children + partial.pending[1:],
            partial.steps
            + (ChosenStep(pending.path, pending.product, rule, precursors),),
            partial.reactions + 1,
            max(partial.depth, len(pending.path) + 1),
            partial.reaction_log_cost + step_log_cost(rule, precursors),
            partial.stock_log_cost + stock_cost,
        )

    def scored_route(route: SidRoute) -> Candidate:
        def parts(node: SidRoute) -> tuple[int, int, float, float, float]:
            if node.rule is None:
                return (0, 0, 0.0, 0.0, stock_log_cost(node.product))
            children = [parts(child) for child in node.precursors]
            precursors = tuple(sorted(child.product for child in node.precursors))
            return (
                1 + max((child[0] for child in children), default=0),
                1 + sum(child[1] for child in children),
                step_log_cost(node.rule, precursors)
                + sum(child[2] for child in children),
                clean_log_cost(node.rule) + sum(child[3] for child in children),
                sum(child[4] for child in children),
            )

        depth, reactions, reaction_cost, clean_cost, stock_cost = parts(route)
        # Three factors whose product is the route's utility, split so a reader can see which part
        # of a price came from the templates and which from the substrate. `base` is what the route
        # would be worth with nothing measured against any of its steps: `step_factor ** reactions`
        # under a constant price, which is why it used to be printed as "length", and the product of
        # the per-rule yields under a price whose base reads the template. It is the *templates'*
        # contribution either way, and is labelled that rather than as a length it no longer is.
        # `reaction_quality` is everything a measurement added on top.
        base = factor_of_cost(clean_cost)
        stock_quality = factor_of_cost(stock_cost)
        reaction_quality = factor_of_cost(reaction_cost - clean_cost)
        score = RouteScore(
            depth,
            reactions,
            base * stock_quality * reaction_quality,
            base,
            stock_quality,
            reaction_quality,
        )
        return Candidate(route, score, reaction_cost + stock_cost)

    def candidate(partial: PartialRoute) -> Candidate:
        chosen = {step.path: step for step in partial.steps}

        def build(path: tuple[int, ...], product: int) -> SidRoute:
            step = chosen.get(path)
            if step is None:
                return SidRoute(product, None)
            return SidRoute(
                product,
                step.rule,
                tuple(
                    build(path + (index,), child)
                    for index, child in enumerate(step.precursors)
                ),
            )

        return scored_route(build((), target))

    def one_step_variants(
        route: SidRoute, depth_left: int, ancestors: frozenset[int]
    ) -> Iterator[SidRoute]:
        """Vary one disconnection, completing its new branches from graph scores."""
        if route.rule is None:
            return
        next_ancestors = ancestors | {route.product}
        current = tuple(child.product for child in route.precursors)
        for rule, precursors in applications(route.product):
            if precursors == current:
                continue
            if any(child in next_ancestors for child in precursors):
                continue
            for child in precursors:
                best_plan(child, depth_left - 1)
                fewest_reactions(child, depth_left - 1)
            children = tuple(
                best_route(child, depth_left - 1, next_ancestors)
                for child in precursors
            )
            if all(child is not None for child in children):
                yield SidRoute(
                    route.product,
                    rule,
                    tuple(child for child in children if child is not None),
                )
        for index, child in enumerate(route.precursors):
            for alternative in one_step_variants(child, depth_left - 1, next_ancestors):
                children = (
                    route.precursors[:index]
                    + (alternative,)
                    + route.precursors[index + 1 :]
                )
                yield SidRoute(route.product, route.rule, children)

    serial = count()
    explored = 0
    exhausted_budget = False
    per_stream_limit = state_limit

    def stream(first: PartialRoute) -> Iterator[Candidate]:
        nonlocal explored, exhausted_budget
        heap: list[tuple[float, float, float, int, int, PartialRoute]] = []
        first_bound = bound(first)
        if math.isfinite(first_bound[0]):
            heapq.heappush(
                heap, (*first_bound, len(first.pending), next(serial), first)
            )
        stream_explored = 0
        while heap:
            if stream_explored >= per_stream_limit:
                exhausted_budget = True
                return
            _, _, _, _, _, partial = heapq.heappop(heap)
            stream_explored += 1
            explored += 1
            if not partial.pending:
                yield candidate(partial)
                continue
            product = partial.pending[0].product
            for rule, precursors in applications(product):
                next_partial = apply(partial, rule, precursors)
                if next_partial is None:
                    continue
                estimate = bound(next_partial)
                if not math.isfinite(estimate[0]):
                    continue
                heapq.heappush(
                    heap,
                    (*estimate, len(next_partial.pending), next(serial), next_partial),
                )

    witnesses: list[Candidate] = []
    for rule, precursors in applications(target):
        for depth in range(shortest, max_depth + 1):
            costs = [best_plan(child, depth - 1) for child in precursors]
            if any(math.isinf(cost) for cost in costs):
                continue
            counts = [fewest_reactions(child, depth - 1) for child in precursors]
            if 1 + sum(counts) > max_reactions:
                continue
            routes = tuple(
                best_route(child, depth - 1, frozenset({target}))
                for child in precursors
            )
            if all(route is not None for route in routes):
                witnesses.append(
                    scored_route(
                        SidRoute(
                            target,
                            rule,
                            tuple(route for route in routes if route is not None),
                        )
                    )
                )
                break
    witnesses.sort(
        key=lambda item: (
            item.log_cost,
            item.score.depth,
            item.score.reactions,
            route_tokens(item.route),
        )
    )
    discovered = witnesses[:candidate_limit]
    seen = {route_tokens(item.route) for item in discovered}
    # A diversity floor and a validator both reject candidates *after* discovery, so discovery
    # stopping at `limit` would guarantee a short report. Where neither is set this is `limit` and
    # no phase does a step more than it did before.
    wanted = limit if diversity_cutoff is None and validate is None else candidate_limit
    for witness in witnesses:
        if len(discovered) >= wanted:
            break
        for route in one_step_variants(witness.route, max_depth, frozenset()):
            tokens = route_tokens(route)
            if tokens in seen:
                continue
            seen.add(tokens)
            item = scored_route(route)
            if (
                item.score.depth <= max_depth
                and item.score.reactions <= max_reactions
                and item.log_cost <= max_cost
            ):
                discovered.append(item)
                if len(discovered) >= wanted:
                    break
    if len(discovered) < wanted:
        initial = PartialRoute((Pending((), target, frozenset()),), (), 0, 0, 0.0, 0.0)
        first_routes = tuple(
            first
            for rule, precursors in applications(target)
            if (first := apply(initial, rule, precursors)) is not None
            and math.isfinite(bound(first)[0])
        )
        per_stream_limit = max(1, state_limit // max(1, len(first_routes)))
        streams = deque(stream(first) for first in first_routes)
        while streams and len(discovered) < candidate_limit:
            source = streams.popleft()
            try:
                discovered.append(next(source))
            except StopIteration:
                continue
            streams.append(source)
        truncated = bool(streams) or exhausted_budget
    else:
        truncated = len(witnesses) > candidate_limit
    unique = {route_tokens(item.route): item for item in discovered}
    # No `[:limit]` here any more: with a diversity floor some candidates are rejected, so the
    # list has to stay long enough to replace them.
    ordered = diverse_order(tuple(unique.values()))
    previous: list[tuple[tuple[int, tuple[int, ...]], ...]] = []
    selected = []
    rejected: list[TooSimilar] = []
    censored: list[Censored] = []
    for item in ordered:
        if len(selected) >= limit:
            break
        # Validated before the floor, and that order is the point: a route nobody can make must
        # not become the thing the next route has to differ from.
        if validate is not None and (reason := validate(item.route)) is not None:
            censored.append(Censored(item.route, item.score, reason))
            continue
        tokens = route_tokens(item.route)
        shared = max(
            (
                next(
                    (
                        index
                        for index, (left, right) in enumerate(zip(tokens, other))
                        if left != right
                    ),
                    min(len(tokens), len(other)),
                )
                for other in previous
            ),
            default=-1,
        )
        distinct = shared + 1 if previous else None
        # The diversity floor. Only the *first* route strives to be best; after that a route earns
        # its place by being different, because the next-cheapest candidate is usually the same
        # plan with one leaf swapped -- a variation nobody asked for that crowds out a real
        # alternative. `first_distinct_decision` already measures this, and it measures it over
        # *precursors* rather than templates, because `route_tokens` is template-insensitive: a
        # near-duplicate synonym template never looks like diversity.
        if (
            diversity_cutoff is not None
            and distinct is not None
            and distinct > diversity_cutoff
        ):
            rejected.append(
                TooSimilar(
                    item.route,
                    item.score,
                    distinct,
                    _resembled(tokens, previous),
                )
            )
            continue
        selected.append(SelectedRoute(item.route, item.score, distinct))
        previous.append(tokens)
    return RouteSelection(
        tuple(selected),
        len(discovered),
        explored,
        truncated,
        shortest,
        int(best_shortest),
        len(discovered) >= limit and not truncated,
        tuple(rejected),
        tuple(censored),
    )

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import dataclass, field
from functools import cache, lru_cache
from typing import Generic, TypeVar

import numpy as np

from .core import K
from .expansions import (
    Expansions,
    sid_selectivity,
    sid_step_selectivity,
    sid_steps,
)
from .interner import MISSING_SID
from .routes import (
    Route,
    Skeleton,
    leaves,
    reaction_skeleton,
    solved_sids,
)
from .store import SidMask, SidStep

# Semiring carrier / metric value type (int counts, int | None depth, float cost, …). Local,
# non-vocabulary; the fold is generic over it. Bound-free.
V = TypeVar("V")

__all__ = [
    "BOOLEAN",
    "TROPICAL",
    "GraphState",
    "Outside",
    "RouteMetric",
    "Semiring",
    "Valuation",
    "compare_routes",
    "library",
    "log_semiring",
    "loose",
    "min_route_depth",
    "outside_valuation",
    "precursor_route_count_bounded",
    "precursor_route_counts_by_first_disconnection",
    "precursor_route_depth_counter",
    "precursor_route_depth_profile",
    "PrecursorDepthCounter",
    "DepthRouteCount",
    "DepthRouteProfile",
    "precursors_recovered",
    "prepare_graph",
    "route_solved",
    "solved_disconnection_count",
    "solved_route_count",
    "solved_route_count_bounded",
    "starting_material_recovered",
    "starting_material_route_count_bounded",
    "strict",
    "valuation",
]

type LeafSets = frozenset[frozenset[int]]


@dataclass(frozen=True)
class GraphState(Generic[K]):
    """Sid-space graph state prepared once, then shared by every metric.

    The grouped ``steps`` map is the important part: ``Expansions.steps`` rebuilds the whole
    edge relation on each call, so a metric that recursed through it would be O(edges) per
    node. Grouping every expanded product once here makes each application lookup O(1).
    """

    expansions: Expansions[K]
    stock_sids: frozenset[int]
    solved: SidMask
    steps: dict[int, tuple[SidStep, ...]]
    outcomes: dict[str, dict[tuple[tuple[int, ...], int], int]] = field(
        default_factory=dict
    )
    """What a forward probe measured, per channel, keyed by ``(precursor sids, rule)``.

    Empty for a graph nobody probed, which is every graph until one is -- so a default rather than
    an argument, and a price reading it then measures nothing and charges nothing. One table per
    channel because the two that exist measure different chemistry: ``sites`` is the intended
    reaction happening in more than one place, ``competitors`` is a different reaction the same
    starting materials could undergo instead.
    """
    step_outcomes: dict[str, dict[tuple[int, int, tuple[int, ...]], int]] = field(
        default_factory=dict
    )
    """The same measurements for channels that are about a *product*, keyed by the whole step.

    Empty unless a caller named channels to index this way. A verdict read on the product is not a
    property of the chemistry, so the key above cannot hold it: two applications sharing precursors
    and rule can condemn one product and clear another, and the pair is then dropped as
    self-contradictory. See :func:`vertsy.expansions.sid_step_selectivity`.
    """

    def applications(self, product: int) -> tuple[SidStep, ...]:
        return self.steps.get(product, ())

    def is_solved(self, product: int) -> bool:
        return 0 <= product < len(self.solved) and bool(self.solved[product])

    def is_expanded(self, product: int) -> bool:
        """Has this product been evaluated? Distinguishes a dead end from an open leaf.

        ``steps`` is keyed by every expanded product, including those that matched no rule,
        so membership is the expandedness bit and ``applications`` alone is ambiguous: it
        returns ``()`` both for a dead end and for a leaf nobody has expanded yet.
        """
        return product in self.steps


# Implicit-generic alias over K (route key) and V (metric value): a metric folds a prepared
# GraphState[K] to a value V. Plain assignment (not `type`) so the free TypeVars parametrize it.
RouteMetric = Callable[[GraphState[K]], V]


# ----------------------------------------------------------------- semiring valuation


@dataclass(frozen=True)
class Semiring(Generic[V]):
    """A semiring: two identities and two associative operations, nothing more.

    ``oplus`` aggregates the alternatives at an OR node (the reaction applications of one
    product); ``otimes`` combines the conjuncts at an AND node (the precursors one
    application demands, plus that application's own cost). ``zero`` and ``one`` are their
    identities, and they carry the base cases for free: folding no applications yields
    ``zero`` (a dead end is unsolvable) and folding no precursors yields ``one`` (nothing
    left to buy). Callers never special-case those — the algebra already says it.

    The operations are plain fields rather than methods, so an instance is a value: a
    semiring *is* the 4-tuple, and ``TROPICAL`` below reads as its own definition.
    """

    zero: V
    one: V
    oplus: Callable[[V, V], V]
    otimes: Callable[[V, V], V]


# Min-plus. ``value`` is the cost of the cheapest route; ``one = 0`` because a stock
# molecule costs nothing. Selective (``oplus`` is idempotent and returns one of its
# arguments), which is what lets AO* trust it to rank alternatives — see :func:`valuation`.
TROPICAL: Semiring[float] = Semiring(
    zero=math.inf, one=0.0, oplus=min, otimes=operator.add
)

# Solvability over {0.0, 1.0}: OR becomes max, AND becomes min. Folding this over a fully
# expanded graph reproduces :func:`vertsy.routes.solved_sids`, which is the same statement
# in Polars rather than in a recursion — ``tests/vertsy/test_aostar.py`` asserts they agree.
BOOLEAN: Semiring[float] = Semiring(zero=0.0, one=1.0, oplus=max, otimes=min)


def log_semiring(temperature: float) -> Semiring[float]:
    """Soft route mass: ``-t * log(sum(exp(-cost / t)))`` over the routes below a product.

    Where :data:`TROPICAL` reports the single best route, this reports how *much* good route
    there is — a product with many cheap routes scores better than one with a single cheap
    route — the ease-of-synthesis signal — and it is what ``otimes`` being
    addition on costs turns into a product of masses.

    ``temperature`` anneals between the two: this semiring's zero-temperature limit *is*
    min-plus, so ``log_semiring(0.0)`` returns :data:`TROPICAL` itself rather than an
    approximation of it.

    Not selective — ``oplus(3, 3) < 3``, because two routes really are more mass than one.
    That makes it a scoring semiring, not a search-guiding one: it cannot rank alternatives
    for AO*, so :func:`vertsy.aostar.aostar` does not accept it. Score a finished graph with
    it instead.
    """
    if temperature < 0.0:
        raise ValueError(f"temperature must be non-negative, got {temperature}.")
    if temperature == 0.0:
        return TROPICAL

    def soft_min(left: float, right: float) -> float:
        # logaddexp is the stable log(exp(a) + exp(b)), and it carries the identity for us:
        # -inf/t is -inf, and logaddexp(-inf, x) == x, so oplus(zero, v) == v exactly.
        return -temperature * float(
            np.logaddexp(-left / temperature, -right / temperature)
        )

    return Semiring(zero=math.inf, one=0.0, oplus=soft_min, otimes=operator.add)


@dataclass(frozen=True)
class Valuation(Generic[V]):
    """A semiring fold of the AND/OR graph, memoized, as two closures.

    ``node`` values a product and ``application`` values one of its reaction applications.
    Selection needs both — AO* compares applications, then descends into the winner — and
    they must agree, so they come from one fold rather than two look-alike recursions.
    """

    node: Callable[[int, int], V]
    application: Callable[[SidStep, int], V]

    conjoin: Callable[[V, V], V]
    """How the precursors of one application combine — carried so nothing can disagree with it.

    :func:`outside_valuation` has to aggregate the immediate *siblings* of a leaf the same way the
    inside fold aggregated them. Keeping it on the value rather than passing it separately makes
    that local mismatch unrepresentable. Only addition also composes with a scalar parent context;
    a nested max needs a richer representation for an exact outside value.
    """


def valuation(
    state: GraphState[K],
    semiring: Semiring[V],
    *,
    leaf_value: Callable[[int], V],
    rule_cost: Callable[[int], V] | None = None,
    step_cost: Callable[[SidStep], V] | None = None,
    conjoin: Callable[[V, V], V] | None = None,
) -> Valuation[V]:
    """Fold the AND/OR graph under ``semiring``, once, into memoized closures.

    The whole recursion is the semiring's own definition::

        node(p, d) = oplus over applications a of application(a, d)
        application((rules, precursors), d) =
            (oplus over r in rules of rule_cost(r)) otimes product of node(q, d-1)

    with three base cases: a stock product is ``one``, an unexpanded product is
    ``leaf_value(p)`` (its heuristic estimate — this is the only place search knowledge
    enters), and a product past the horizon is ``zero``. Swapping the semiring swaps the
    question: :data:`TROPICAL` asks for the cheapest route, :func:`log_semiring` for route
    mass, :data:`BOOLEAN` for solvability. The graph walk never changes.

    ``rule_cost`` weights an application and defaults to ``one`` (free). AO* passes a
    constant ``1.0`` under :data:`TROPICAL` so a route's value counts its reactions; it is
    also where a complexity penalty belongs, since an edge cost need not vanish at the goal
    the way a node heuristic must.

    ``step_cost`` prices the whole application instead, and is what a score depending on the
    *substrate* needs: a rule id cannot say which molecule it was applied to, and a whole
    :data:`~vertsy.store.SidStep` can. The two are exclusive -- passing both is a caller who has
    not decided which one is the price -- and ``rule_cost`` is exactly the special case
    ``lambda step: oplus over its rules``, so it stays for every semiring where multiplying a
    cost is not a defined operation at all: ``V`` may be an integer route count.

    ``depth_left`` is the reaction budget, and it alone makes the fold well-founded on the
    cyclic molecule graph: every recursive call spends one, so a product reached through
    itself simply runs out and contributes ``zero``. No on-stack cycle guard is needed —
    ``(product, depth_left)`` cannot recur while it is being computed, because the depth
    strictly decreases — and unlike the cycle-cutting in :func:`count_solved_routes`, that
    makes this fold exact: it is the true value over routes of at most ``depth_left``
    reactions. Memoized on ``(product, depth_left)``, so a shared subgraph is folded once per
    horizon rather than once per path.

    A product past the horizon is ``zero``, not its heuristic: the budget is a real
    constraint, so a route that does not fit does not count. The value is therefore optimal
    *among routes within the budget*, which is the claim AO* can honestly make here.

    ``conjoin`` overrides how the precursors of one application combine, and defaults to
    ``otimes`` — the semiring's own answer, and the only one that makes a route's value a *route*
    cost. Passing ``max`` under :data:`TROPICAL` asks a different and also chemically real question:
    the value becomes the **longest linear sequence** rather than the total reaction count, since
    branches that can be run in parallel no longer add. That matters for guidance as much as for
    scoring. Summing an estimate over a conjunction charges a bimolecular disconnection twice, so
    splitting a molecule at distance 0.68 into halves at 0.71 and 0.58 reads as *doubling* the work
    remaining (1.29) when both halves are in fact closer to the catalogue — measured on a PaRoutes
    route the search could not recover. ``max`` removes that penalty by construction.
    """
    if rule_cost is not None and step_cost is not None:
        raise ValueError(
            "pass rule_cost or step_cost, not both: they are two spellings of one price."
        )
    rule_price = (lambda rule: semiring.one) if rule_cost is None else rule_cost
    join = semiring.otimes if conjoin is None else conjoin
    memo: dict[tuple[int, int], V] = {}

    def node(product: int, depth_left: int) -> V:
        if product in state.stock_sids:
            return semiring.one
        if depth_left <= 0:
            return semiring.zero
        if not state.is_expanded(product):
            return leaf_value(product)
        key = (product, depth_left)
        if key in memo:
            return memo[key]
        total = semiring.zero
        for step in state.applications(product):
            total = semiring.oplus(total, application(step, depth_left))
        memo[key] = total
        return total

    def application(step: SidStep, depth_left: int) -> V:
        rules, precursor_sids = step
        # The templates of one disconnection are *alternatives*, so they combine under `oplus`
        # like any other choice — and that one line makes each semiring answer its own question
        # correctly. Min-plus takes the cheapest template; the counting semiring sums them, so a
        # rule-aware route count still counts every template; boolean solvability ignores the
        # multiplicity entirely.
        if step_cost is not None:
            cost = step_cost(step)
        else:
            cost = semiring.zero
            for rule in rules:
                cost = semiring.oplus(cost, rule_price(rule))
        # Split from the loop above so `conjoin` can differ from `otimes`. With the default the two
        # loops are one associative fold in the same left-to-right order, so nothing moves: `one` is
        # the identity of `otimes`, and the arithmetic is unchanged to the last bit.
        remaining = semiring.one
        for precursor in precursor_sids:
            remaining = join(remaining, node(precursor, depth_left - 1))
        return semiring.otimes(cost, remaining)

    return Valuation(node=node, application=application, conjoin=join)


@dataclass(frozen=True)
class Outside:
    """Context values: what a solution graph through a product needs *besides* its subtree.

    The dual of :func:`valuation`, which values what lies *below* a product. For an additive
    precursor fold, together they price a product in place: ``total`` is the cost of the cheapest
    solution graph constrained to contain it. For a bottleneck fold, ``total`` is an additive
    upper bound because a nested max context cannot be represented by one scalar.
    """

    delta: dict[tuple[int, int], float]
    leaves: dict[int, tuple[float, float, int]]
    """Open leaf -> its best ``total``, the ``context`` half of it, and the attaining depth.

    Both halves, not just their sum, because which half a caller ranks on *is* the search
    strategy: context alone is breadth-like, and the leaf's own value alone is greedy. Their sum
    is the faithful cost of a solution graph only when precursor values are also summed. A nested
    bottleneck context cannot be represented by this scalar. Recovering one half from the other
    would mean subtracting, and these are min-plus values where a subtraction can meet infinity.
    """


def outside_valuation(
    target: int,
    state: GraphState[K],
    value: Valuation[float],
    *,
    max_depth: int,
    step_cost: Callable[[SidStep], float],
) -> Outside:
    """Sweep context costs down from ``target``, and collect the open leaves on the way.

    The recursion, with ``rn`` for the inside values ``value`` already holds::

        delta(target, max_depth) = 0,  delta(m, max_depth) = zero otherwise
        delta(q, d-1) = min over parents (p, (rule, precursors)) naming q, of
                        delta(p, d) + step_cost(a) + conjoin over the other s of rn(s, d-1)
        total(m)      = min over d of delta(m, d) + rn(m, d)

    The sibling term is ``value.conjoin``, not ``+``: it is whatever the inside fold used, and a
    max fold is aggregated by max here too. Only the two outer compositions are fixed additions,
    which is the whole of the bottleneck approximation and is stated again at the end of this
    docstring. Reading ``sum`` into that middle term -- as this formula said until it was
    corrected -- makes the array and scalar sweeps look like they disagree. They do not:
    ``test_open_leaves_match_the_generic_sweep`` pins them equal under both folds.

    Written as a sweep from ``max_depth`` down to 1 rather than a recursion, so termination is
    syntactic: every relaxation moves strictly one level down, and a molecule cycle simply runs
    out of levels. No visited guard is needed, which is the dual of the property that makes
    :func:`valuation` well-founded on the same cyclic graph.

    Applications valued ``zero`` are skipped, which costs no reachable leaf: such an application
    already forces every one of its precursors to ``zero`` through it, so descending would only
    record infinities. ``leaves`` therefore holds exactly the open leaves with a *finite*
    ``total`` — a leaf reachable only below the depth budget is absent rather than infinite,
    which is the same statement and one the caller does not have to filter.

    Each precursor's inside value is read **once per application per level** and reused, which is
    where this pass spends its time: the fold memoizes ``node`` but not ``application``, so every
    call into it re-walks a precursor tuple. Asking directly whether any inside value is ``zero``
    replaces the ``application`` call that used to test the same thing, and the sibling sums then
    read a list instead of the fold. Calls into the fold drop from quadratic in arity to linear.

    The sibling aggregates stay explicit loops rather than "total minus ``rn(q)``", accumulating in the
    same order as before so the arithmetic is unchanged to the last bit — the selection ranks on
    these values and an invariant compares them for equality against the fold's own total, so a
    reassociated sum would be a real difference, not a rounding detail. It is also what lets the
    aggregation be ``value.conjoin`` rather than ``+``: a leave-one-out ``max`` cannot be had by
    subtraction. This computes the immediate sibling term, but composing it with an already
    scalar parent context remains additive and therefore only bounds a nested max objective.
    """
    delta: dict[tuple[int, int], float] = {(target, max_depth): TROPICAL.one}
    leaves: dict[int, tuple[float, float, int]] = {}
    level: set[int] = {target}
    # `step_cost` is a pure function of a disconnection, and a disconnection does not change
    # between levels, so its price is worth remembering across the sweep. Keyed by the whole step
    # and not by its template set: a substrate-aware price reads the precursors too, so two
    # disconnections sharing templates need not share a cost.
    priced: dict[SidStep, float] = {}

    def offer(product: int, depth_left: int, context: float) -> None:
        key = (product, depth_left)
        if context < delta.get(key, TROPICAL.zero):
            delta[key] = context

    for depth_left in range(max_depth, 0, -1):
        below: set[int] = set()
        # One level asks for the same precursor's inside value from every application that names
        # it, and a shared intermediate is named by many. The fold memoizes it, but reaching that
        # memo is still a call through a closure, so the level keeps its own answers.
        inside_at: dict[int, float] = {}
        for product in level:
            context = delta.get((product, depth_left), TROPICAL.zero)
            if context == TROPICAL.zero or product in state.stock_sids:
                continue
            if not state.is_expanded(product):
                total = context + value.node(product, depth_left)
                best = leaves.get(product)
                if best is None or total < best[0]:
                    leaves[product] = (total, context, depth_left)
                continue
            for step in state.applications(product):
                _rules, precursors = step
                inside = []
                for other in precursors:
                    # Values are floats, never None, so a miss is unambiguous.
                    got = inside_at.get(other)
                    if got is None:
                        got = value.node(other, depth_left - 1)
                        inside_at[other] = got
                    inside.append(got)
                if TROPICAL.zero in inside:
                    continue
                cheapest = priced.get(step)
                if cheapest is None:
                    # Min-plus, so a route here uses the disconnection's cheapest template.
                    cheapest = step_cost(step)
                    priced[step] = cheapest
                if cheapest == TROPICAL.zero:
                    continue
                for position, precursor in enumerate(precursors):
                    siblings = TROPICAL.one
                    for index, other_value in enumerate(inside):
                        if index != position:
                            siblings = value.conjoin(siblings, other_value)
                    offer(precursor, depth_left - 1, context + cheapest + siblings)
                    below.add(precursor)
        # Only products in `level` can put anything in `below`, so once a level is empty every
        # deeper one is too. Worth a check now that the horizon is the caller's pass count.
        if not below:
            break
        level = below

    return Outside(delta=delta, leaves=leaves)


def interned_stock_sids(expansions: Expansions[K], stock: Collection[K]) -> np.ndarray:
    """Which interned sids are stock, found by walking the *interner* rather than the stock.

    The same sids either way -- the answer is used as a set and `solved_sids` only scatters it
    into a mask -- but the two sides are nowhere near the same size, and the old direction asked
    the question of the big one. A search carries a stock of ~920,000 keys against an interner
    holding ~7,900 sids, and `prepare_graph` runs **once per pass**, so "is each stock key
    interned?" cost a dict lookup per stock key per pass: 16.6 million of them, measured at
    **1.64 s of a 7.4 s search -- 85% of `prepare_graph` and 22% of the whole search**. "Is each
    interned key in stock?" is the same question asked of the small side: 88.8 ms to 0.33 ms at
    those sizes, and the sid sets compared equal.

    `stock` is used for membership only, so it is converted unless it already answers in constant
    time -- which the search's own `frozenset` does, making the hot path free. A caller passing a
    sequence pays one pass over it, which is no worse than the per-key lookups this replaces.
    """
    count = expansions.sid_count()
    if not count:
        return np.zeros(0, dtype=np.int64)
    held: Collection[K] = (
        stock if isinstance(stock, (set, frozenset, dict)) else frozenset(stock)
    )
    in_stock = np.fromiter(
        (key in held for key in expansions.decode(range(count))),
        dtype=np.bool_,
        count=count,
    )
    return np.flatnonzero(in_stock).astype(np.int64)


def prepare_graph(
    expansions: Expansions[K],
    stock: Collection[K],
    *,
    step_keyed: Collection[str] = (),
) -> GraphState[K]:
    """Resolve stock, run the solved-sid fixpoint, and group all steps — once.

    ``step_keyed`` names the channels whose answer is about a *product* rather than about
    chemistry, and so need the second index. The caller names them because the caller mints the
    channels; nothing here knows what a channel means.

    A step-keyed channel is left **out** of :attr:`GraphState.outcomes` rather than written to
    both: under the chemistry key its answer is not merely redundant but wrong, since two products
    sharing precursors and rule can disagree, and an entry nobody may trust is worth neither the
    pass that builds it nor the memory that holds it.

    **Measured, and it is not free.** On a 50,000-application graph with 34 per-row channels,
    indexing everything by chemistry costs 1.31 s and splitting the two costs 1.53 s -- **+17%**,
    because each pass walks the edge columns and sorts each application's precursors for itself.
    The available fix is one pass writing both indexes, which would make the split cheaper than
    the single index it replaced.

    **And read "per target" carefully, which this docstring once did not.** It used to conclude
    "one to two percent of a target" from those figures, but :func:`vertsy.aostar.aostar` calls
    this **once per pass** -- sixteen times in the default schedule, on a graph that grows every
    pass -- so a per-call cost reasoned about as a whole-target cost understates it by that
    factor. Timed properly it was 24% of a search, 85% of it in the stock resolution alone, which
    :func:`interned_stock_sids` now does from the smaller side. The indexing figures above are
    what remains, and are genuinely small.
    """
    stock_sids_array = interned_stock_sids(expansions, stock)
    stock_sids = frozenset(stock_sids_array.tolist())
    solved = solved_sids(expansions, stock_sids_array)
    steps = sid_steps(expansions, expansions.store.expanded())
    return GraphState(
        expansions,
        stock_sids,
        solved,
        steps,
        sid_selectivity(expansions, exclude=step_keyed),
        sid_step_selectivity(expansions, step_keyed),
    )


def compare_routes(
    expansions: Expansions[K],
    stock: Collection[K],
    metric: RouteMetric[K, V],
) -> V:
    return metric(prepare_graph(expansions, stock))


def strict(reference: Route[K]) -> RouteMetric[K, int]:
    def metric(state: GraphState[K]) -> int:
        sid_route = route_to_sids(reference, state.expansions)
        if sid_route is None:
            return 0
        return int(strict_recovered(sid_route, state))

    return metric


def loose(reference: Route[K]) -> RouteMetric[K, int]:
    def metric(state: GraphState[K]) -> int:
        root_sid = lookup_one(state.expansions, reference.product)
        if root_sid is None:
            return 0
        leaf_sids = lookup_all(state.expansions, leaves(reference))
        if leaf_sids is None:
            return 0
        wanted_leaves = frozenset(leaf_sids)
        if not wanted_leaves.issubset(state.stock_sids):
            return 0
        return int(
            wanted_leaves
            in skeleton_leaf_sets(
                root_sid, reaction_skeleton(reference), state, wanted_leaves
            )
        )

    return metric


def library(reference: Route[K]) -> RouteMetric[K, int]:
    def metric(state: GraphState[K]) -> int:
        root_sid = lookup_one(state.expansions, reference.product)
        if root_sid is None:
            return 0
        return int(skeleton_recovered(root_sid, reaction_skeleton(reference), state))

    return metric


def solved_route_count(target: K, cap: int) -> RouteMetric[K, int]:
    if cap <= 0:
        raise ValueError("cap must be positive.")

    def metric(state: GraphState[K]) -> int:
        target_sid = lookup_one(state.expansions, target)
        if target_sid is None:
            return 0
        return count_solved_routes(target_sid, state, cap)

    return metric


def solved_route_count_bounded(
    target: K, cap: int, max_depth: int
) -> RouteMetric[K, int]:
    if cap <= 0:
        raise ValueError("cap must be positive.")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative.")

    def metric(state: GraphState[K]) -> int:
        target_sid = lookup_one(state.expansions, target)
        if target_sid is None:
            return 0
        return count_solved_routes_bounded(target_sid, state, cap, max_depth)

    return metric


def precursor_route_count_bounded(
    target: K, cap: int, max_depth: int
) -> RouteMetric[K, int]:
    """Solved routes using at most ``max_depth`` reactions, counted under the *precursors*
    comparator: two routes are the same when every step has the same product and the same set
    of precursors, regardless of which rule produced it (order and multiplicity ignored). This
    collapses template redundancy — many rules disconnecting a product into identical
    precursors count once — so it is a more chemically intuitive route count than the
    rule-aware :func:`solved_route_count_bounded`. Memoized on ``(product, depth)``, O(edges).
    """
    if cap <= 0:
        raise ValueError("cap must be positive.")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative.")

    def metric(state: GraphState[K]) -> int:
        target_sid = lookup_one(state.expansions, target)
        if target_sid is None:
            return 0
        return count_precursor_routes_bounded(target_sid, state, cap, max_depth)

    return metric


def starting_material_route_count_bounded(
    target: K, cap: int, max_depth: int
) -> RouteMetric[K, int]:
    """Distinct starting-material sets reachable by a solved route of at most ``max_depth``
    reactions, under the *starting-materials* comparator: two routes are the same when they
    share the target and the set of stock leaves, regardless of reactions or intermediates.
    This is the most collapsed, most chemically intuitive route count. It enumerates the
    distinct leaf-sets (a set-union valuation) and is capped; counting distinct set-unions is
    #P-hard in general, but the depth bound keeps it small and fast.
    """
    if cap <= 0:
        raise ValueError("cap must be positive.")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative.")

    def metric(state: GraphState[K]) -> int:
        target_sid = lookup_one(state.expansions, target)
        if target_sid is None:
            return 0
        return len(starting_material_sets_bounded(target_sid, state, cap, max_depth))

    return metric


def solved_disconnection_count(target: K) -> RouteMetric[K, int]:
    """How many genuinely different ways there are to *begin* the synthesis.

    The target's applications whose precursors are all solved — so a route exists through each one.
    The most collapsed diversity measure worth having, and the only one measured here that is both
    meaningful and affordable:

    - counting routes is meaningless. Both counters run past ``10**12`` on an ordinary depth-5
      graph of 19,455 molecules, and the log semiring puts the *competitive* route count near
      ``10**40``. AND-branching multiplies, so this number is astronomical for any solved target and
      says nothing about whether the chemistry is varied.
    - counting distinct starting-material sets is meaningful and costs ``O(cap**1.7)`` — measured at
      12.7 s, 40.3 s and 133.3 s for caps of 1k, 2k and 4k, saturated at every rung, against a 44 s
      search. There is no cap at which it both finishes and answers.
    - this is tens (42, 35, 32 on three n1 targets) and takes microseconds, because ``prepare_graph``
      has already run the fixpoint it needs.

    Distinct *precursor sets*, not distinct templates, and that distinction is the store's rather
    than this function's: a layer collapses on ``(product, precursor multiset)``, so two templates
    achieving the same disconnection become one application carrying both. Template multiplicity
    survives as ``len(rules)`` and belongs on ``rule_cost`` — forty independent extractions of one
    disconnection are evidence the reaction is real, not evidence of forty routes. On used14k the
    median disconnection has exactly one template and inflation is 1.0-1.2x; on raw68k the same
    disconnection is reachable by 28-49 template combinations, which is where that distinction pays.

    The ``solved`` filter does real work: only about a fifth of a target's disconnections lead
    anywhere (42 of 208, 35 of 97, 32 of 104), so the raw application count overstates by four to
    five times.
    """

    def metric(state: GraphState[K]) -> int:
        target_sid = lookup_one(state.expansions, target)
        if target_sid is None:
            return 0
        return sum(
            1
            for _, precursors in state.applications(target_sid)
            if all(state.is_solved(int(precursor)) for precursor in precursors)
        )

    return metric


def min_route_depth(target: K) -> RouteMetric[K, int | None]:
    def metric(state: GraphState[K]) -> int | None:
        target_sid = lookup_one(state.expansions, target)
        if target_sid is None:
            return None
        return minimum_route_depth(target_sid, state)

    return metric


def route_solved(target: K) -> RouteMetric[K, int]:
    """Whether ``target`` has at least one solved route (any reactions, any depth).

    The loosest recovery signal, and the only comparison possible when the reference route's
    reactions cannot be aligned to the search rules at all.
    """

    def metric(state: GraphState[K]) -> int:
        target_sid = lookup_one(state.expansions, target)
        if target_sid is None:
            return 0
        return int(state.is_solved(target_sid))

    return metric


def starting_material_recovered(
    target: K, reference_leaves: Collection[K], max_depth: int
) -> RouteMetric[K, int]:
    """Whether a solved route of at most ``max_depth`` reactions has exactly ``reference_leaves``
    as its stock leaves — the *starting-materials* comparator, which ignores reactions and
    intermediates. Reference-guided and exact (a bitmask over the small reference leaf set),
    so it needs no cap and works even when rule ids do not align.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative.")

    def metric(state: GraphState[K]) -> int:
        target_sid = lookup_one(state.expansions, target)
        if target_sid is None:
            return 0
        leaf_sids = lookup_all(state.expansions, reference_leaves)
        if leaf_sids is None or not leaf_sids.issubset(state.stock_sids):
            return 0
        return int(leaf_set_recovered(target_sid, leaf_sids, state, max_depth))

    return metric


def precursors_recovered(
    target: K, reference_precursors: Collection[K], max_depth: int
) -> RouteMetric[K, int]:
    """Whether a solved route of at most ``max_depth`` reactions has exactly
    ``reference_precursors`` (all intermediates and leaves) as its precursor set — the
    *precursors* comparator, which ignores reactions. Reference-guided bitmask over the small
    reference precursor set; exact and rule-id-agnostic.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative.")

    def metric(state: GraphState[K]) -> int:
        target_sid = lookup_one(state.expansions, target)
        if target_sid is None:
            return 0
        precursor_sids = lookup_all(state.expansions, reference_precursors)
        if precursor_sids is None:
            return 0
        return int(
            precursor_set_recovered(target_sid, precursor_sids, state, max_depth)
        )

    return metric


def route_to_sids(route: Route[K], expansions: Expansions[K]) -> Route[int] | None:
    product = lookup_one(expansions, route.product)
    if product is None:
        return None
    precursors: list[Route[int]] = []
    for child in route.precursors:
        sid_child = route_to_sids(child, expansions)
        if sid_child is None:
            return None
        precursors.append(sid_child)
    return Route(product, route.rule, tuple(precursors))


def lookup_one(expansions: Expansions[K], key: K) -> int | None:
    sid = int(expansions.lookup([key])[0])
    if sid == MISSING_SID:
        return None
    return sid


def lookup_all(expansions: Expansions[K], keys: Collection[K]) -> frozenset[int] | None:
    sids = expansions.lookup(keys)
    if np.any(sids == MISSING_SID):
        return None
    return frozenset(int(sid) for sid in sids.tolist())


def strict_recovered(reference: Route[int], state: GraphState[K]) -> bool:
    if reference.rule is None:
        return reference.product in state.stock_sids
    if not state.is_solved(reference.product):
        return False

    # Strict is stoichiometry-free (matching ``collapsed_route_key``): collapse the
    # reference precursors to the set of their product sids, match any application with the
    # same rule and that exact precursor-product set, then require every reference child to
    # be strictly recovered from its product sid. Each matched precursor's solvability is
    # enforced by the child recursion (a leaf must be stock, a branch must be solved).
    # A disconnection carries every template that achieves it, so the reference rule has to be
    # *among* them rather than equal to one. Same answer as when each template was its own
    # application — the reference rule was one of those and is now one of these.
    child_products = frozenset(child.product for child in reference.precursors)
    for rules, precursor_sids in state.applications(reference.product):
        if reference.rule not in rules:
            continue
        if frozenset(precursor_sids) != child_products:
            continue
        if all(strict_recovered(child, state) for child in reference.precursors):
            return True
    return False


def skeleton_recovered(product: int, skeleton: Skeleton, state: GraphState[K]) -> bool:
    if skeleton is None:
        return product in state.stock_sids
    if not state.is_solved(product):
        return False

    rule, children = skeleton
    for rules, precursor_sids in state.applications(product):
        if rule not in rules:
            continue
        if not all(state.is_solved(precursor) for precursor in precursor_sids):
            continue
        if skeleton_assignment_exists(tuple(children), tuple(precursor_sids), state):
            return True
    return False


def skeleton_assignment_exists(
    child_skeletons: Sequence[Skeleton],
    precursor_sids: tuple[int, ...],
    state: GraphState[K],
) -> bool:
    if not child_skeletons:
        return all(precursor in state.stock_sids for precursor in precursor_sids)

    child = child_skeletons[0]
    rest = child_skeletons[1:]
    for index, precursor in enumerate(precursor_sids):
        if not skeleton_recovered(precursor, child, state):
            continue
        remaining = precursor_sids[:index] + precursor_sids[index + 1 :]
        if skeleton_assignment_exists(rest, remaining, state):
            return True
    return False


def skeleton_leaf_sets(
    product: int,
    skeleton: Skeleton,
    state: GraphState[K],
    wanted_leaves: frozenset[int],
) -> LeafSets:
    @cache
    def recur(product: int, skeleton: Skeleton) -> LeafSets:
        if skeleton is None:
            if product in state.stock_sids and product in wanted_leaves:
                return frozenset({frozenset({product})})
            return frozenset()

        if not state.is_solved(product):
            return frozenset()

        rule, children = skeleton
        out: set[frozenset[int]] = set()
        for rules, precursor_sids in state.applications(product):
            if rule not in rules:
                continue
            if not all(state.is_solved(precursor) for precursor in precursor_sids):
                continue
            out.update(
                assignment_leaf_sets(tuple(children), tuple(precursor_sids), recur)
            )
        return frozenset(leaves for leaves in out if leaves.issubset(wanted_leaves))

    def assignment_leaf_sets(
        child_skeletons: Sequence[Skeleton],
        precursor_sids: tuple[int, ...],
        child_sets: Callable[[int, Skeleton], LeafSets],
    ) -> LeafSets:
        if not child_skeletons:
            if all(precursor in state.stock_sids for precursor in precursor_sids):
                leaf_set = frozenset(precursor_sids)
                if leaf_set.issubset(wanted_leaves):
                    return frozenset({leaf_set})
            return frozenset()

        child = child_skeletons[0]
        rest = child_skeletons[1:]
        out: set[frozenset[int]] = set()
        for index, precursor in enumerate(precursor_sids):
            for child_leaf_set in child_sets(precursor, child):
                remaining = precursor_sids[:index] + precursor_sids[index + 1 :]
                for rest_leaf_set in assignment_leaf_sets(rest, remaining, child_sets):
                    leaf_set = child_leaf_set | rest_leaf_set
                    if leaf_set.issubset(wanted_leaves):
                        out.add(leaf_set)
        return frozenset(out)

    return recur(product, skeleton)


def solved_postorder(target: int, state: GraphState[K]) -> list[int]:
    """Solved non-stock products under ``target``, every precursor before the product it makes.

    An explicit stack rather than recursion, because the depth of this walk is the longest chain
    in the solved subgraph and nothing bounds it. A 540k-molecule raw68k graph reached 992 levels
    and overflowed the interpreter's stack — after the search had finished, so a whole target's
    compute was lost to a metric. Any unbounded fold over a solved subgraph belongs here; the
    ``reach`` walks elsewhere in this module are safe only because ``max_depth`` bounds them.

    **Cycle handling falls out of the ordering** rather than needing a rule of its own. A back
    edge points at a product still on the stack, which therefore lands *after* this one in the
    result, so a fold reading values in this order finds nothing recorded for it — exactly the
    "a product revisited while still on the stack contributes the semiring's zero" the recursions
    used to spell out. Cross edges point at products already emitted and read normally.

    Emitting the order instead of folding over it is what lets two folds share one traversal: a
    caller supplies ``oplus`` and ``otimes`` as a plain loop body and keeps whatever saturation or
    ``None`` handling it needs, which is the part a generic :class:`Semiring` fold could not
    express and the reason both callers were separate recursions to begin with.
    """
    if target in state.stock_sids or not state.is_solved(target):
        return []

    def precursors(product: int) -> Iterator[int]:
        return (sid for _, group in state.applications(product) for sid in group)

    order: list[int] = []
    seen = {target}
    stack: list[tuple[int, Iterator[int]]] = [(target, precursors(target))]
    while stack:
        product, remaining = stack[-1]
        # `remaining` is resumed on the next visit, so each product's precursors are walked once
        # in order — which is what makes the emitted sequence a depth-first postorder.
        for precursor in remaining:
            if (
                precursor in seen
                or precursor in state.stock_sids
                or not state.is_solved(precursor)
            ):
                continue
            seen.add(precursor)
            stack.append((precursor, precursors(precursor)))
            break
        else:
            order.append(product)
            stack.pop()
    return order


def count_solved_routes(target: int, state: GraphState[K], cap: int) -> int:
    """Number of solved routes for ``target``, saturating at ``cap``.

    One value per product (O(edges)), so shared subgraphs are counted once rather than
    re-enumerated per path — the dense depth-2 recovery graph makes trail counting exponential.
    A product reachable only through a cycle contributes 0, giving simple-route counting on the
    shared DAG; the cap keeps the arithmetic bounded.

    This is the counting semiring — ``oplus`` is ``+`` and ``otimes`` is ``*`` — i.e. the same
    fold as :func:`valuation`, specialised and capped. It stays its own loop because the
    saturation threads through both levels, which a generic fold cannot express; see the note on
    :class:`Semiring` for the family it belongs to. :func:`solved_postorder` supplies the order.
    """
    if target in state.stock_sids:
        return 1
    counts: dict[int, int] = {}
    for product in solved_postorder(target, state):
        total = 0
        for rules, precursor_sids in state.applications(product):
            # Rule-aware identity, so every template that reaches this disconnection is its own
            # route. The store collapses them into one application; the multiplier is what keeps
            # this count answering the question its name asks.
            app_count = len(rules)
            for precursor in precursor_sids:
                app_count *= (
                    1 if precursor in state.stock_sids else counts.get(precursor, 0)
                )
                if app_count >= cap:
                    app_count = cap
                    break
            total += app_count
            if total >= cap:
                total = cap
                break
        counts[product] = total
    return counts.get(target, 0)


def count_solved_routes_bounded(
    target: int, state: GraphState[K], cap: int, max_depth: int
) -> int:
    """Number of solved routes using at most ``max_depth`` reactions, saturating at cap."""
    memo: dict[tuple[int, int], int] = {}
    computing: set[tuple[int, int]] = set()

    def count(product: int, depth_left: int) -> int:
        if product in state.stock_sids:
            return 1
        if depth_left <= 0:
            return 0
        key = (product, depth_left)
        if key in memo:
            return memo[key]
        if key in computing or not state.is_solved(product):
            return 0
        computing.add(key)
        total = 0
        for rules, precursor_sids in state.applications(product):
            app_count = len(rules)  # rule-aware: one route per template (see above)
            for precursor in precursor_sids:
                app_count *= count(precursor, depth_left - 1)
                if app_count >= cap:
                    app_count = cap
                    break
            total += app_count
            if total >= cap:
                total = cap
                break
        computing.discard(key)
        memo[key] = total
        return total

    return count(target, max_depth)


def count_precursor_routes_bounded(
    target: int, state: GraphState[K], cap: int, max_depth: int
) -> int:
    """Like :func:`count_solved_routes_bounded`, but applications are deduplicated by their
    precursor set (rule-agnostic), so co-canonical templates that yield the same precursors
    count once. Memoized on ``(product, depth_left)``; cycle-safe because depth strictly
    decreases.
    """
    return precursor_route_count_function(state, cap)(target, max_depth)


def precursor_route_count_function(
    state: GraphState[K], cap: int
) -> Callable[[int, int], int]:
    """Share one memo table across total and first-disconnection counts."""
    return precursor_route_depth_counter(state, cap).node


@dataclass(frozen=True)
class PrecursorDepthCounter:
    node: Callable[[int, int], int]
    application: Callable[[frozenset[int], int], int]
    cap: int


@dataclass(frozen=True)
class DepthRouteCount:
    depth: int
    at_most: int
    exact: int | None
    capped: bool


@dataclass(frozen=True)
class DepthRouteProfile(Generic[K]):
    shortest_depth: int | None
    counts: tuple[DepthRouteCount, ...]
    first_disconnections: tuple[tuple[tuple[K, ...], tuple[DepthRouteCount, ...]], ...]


def precursor_route_depth_counter(
    state: GraphState[K], cap: int
) -> PrecursorDepthCounter:
    """Memoized, template-insensitive counts for products and applications by depth."""
    if cap <= 0:
        raise ValueError("cap must be positive")
    memo: dict[tuple[int, int], int] = {}

    @lru_cache(maxsize=262_144)
    def application(precursors: frozenset[int], depth_left: int) -> int:
        if depth_left <= 0:
            return 0
        routes = 1
        for precursor in precursors:
            routes = min(cap, routes * count(precursor, depth_left - 1))
            if routes == 0:
                break
        return routes

    def count(product: int, depth_left: int) -> int:
        if product in state.stock_sids:
            return 1
        if depth_left <= 0:
            return 0
        key = (product, depth_left)
        if key in memo:
            return memo[key]
        if not state.is_solved(product):
            return 0
        total = 0
        precursor_sets = {
            frozenset(precursor_sids)
            for _, precursor_sids in state.applications(product)
        }
        for precursor_set in precursor_sets:
            total = min(cap, total + application(precursor_set, depth_left))
            if total == cap:
                break
        memo[key] = total
        return total

    return PrecursorDepthCounter(count, application, cap)


def precursor_route_depth_profile(
    target: K, cap: int, extra_depths: int = 3
) -> RouteMetric[K, DepthRouteProfile[K]]:
    """Shortest-depth route count and three later depth slices by default.

    ``exact`` is absent if either cumulative count used in its difference saturated.
    Applications with the same precursor set are counted once regardless of rule ID.
    """
    if cap <= 0 or extra_depths < 0:
        raise ValueError("cap must be positive and extra_depths non-negative")

    def metric(state: GraphState[K]) -> DepthRouteProfile[K]:
        target_sid = lookup_one(state.expansions, target)
        if target_sid is None:
            return DepthRouteProfile(None, (), ())
        shortest = minimum_route_depth(target_sid, state)
        if shortest is None:
            return DepthRouteProfile(None, (), ())
        counter = precursor_route_depth_counter(state, cap)
        depths = range(shortest, shortest + extra_depths + 1)

        def depth_counts(value: Callable[[int], int]) -> tuple[DepthRouteCount, ...]:
            results = []
            for depth in depths:
                current = value(depth)
                previous = value(depth - 1) if depth else 0
                exact = current - previous if current < cap and previous < cap else None
                results.append(
                    DepthRouteCount(
                        depth,
                        current,
                        exact,
                        current >= cap,
                    )
                )
            return tuple(results)

        counts = depth_counts(lambda depth: counter.node(target_sid, depth))
        grouped: set[frozenset[int]] = set()
        if target_sid not in state.stock_sids:
            grouped = {
                frozenset(precursors)
                for _, precursors in state.applications(target_sid)
            }
        first = tuple(
            (
                tuple(state.expansions.decode(sorted(precursors))),
                depth_counts(
                    lambda depth, precursors=precursors: counter.application(
                        precursors, depth
                    )
                ),
            )
            for precursors in grouped
        )
        return DepthRouteProfile(
            shortest,
            counts,
            tuple(sorted(first, key=lambda item: (-item[1][0].at_most, repr(item[0])))),
        )

    return metric


def precursor_route_counts_by_first_disconnection(
    target: K, cap: int, max_depth: int
) -> RouteMetric[K, tuple[tuple[tuple[K, ...], int], ...]]:
    """Exact bounded route count for each distinct first precursor set.

    This is a cheap post-search decomposition of the existing precursor-route count.
    It still counts graph-choice trees, not distinct chemical plans.
    """
    if cap <= 0 or max_depth < 0:
        raise ValueError("cap must be positive and max_depth non-negative")

    def metric(state: GraphState[K]) -> tuple[tuple[tuple[K, ...], int], ...]:
        target_sid = lookup_one(state.expansions, target)
        if target_sid is None or max_depth == 0:
            return ()
        count = precursor_route_count_function(state, cap)
        grouped = {
            frozenset(precursors) for _, precursors in state.applications(target_sid)
        }
        results = []
        for precursor_set in grouped:
            routes = 1
            for precursor in precursor_set:
                routes = min(cap, routes * count(precursor, max_depth - 1))
            if routes:
                keys = tuple(state.expansions.decode(sorted(precursor_set)))
                results.append((keys, routes))
        return tuple(sorted(results, key=lambda item: (-item[1], repr(item[0]))))

    return metric


def starting_material_sets_bounded(
    target: int, state: GraphState[K], cap: int, max_depth: int
) -> frozenset[frozenset[int]]:
    """Distinct stock leaf-sets reachable from ``target`` by a solved route of at most
    ``max_depth`` reactions. Set-union valuation: an application's leaf-sets are the pairwise
    unions across its precursors' leaf-sets; a product's are the union over its applications.
    Truncated at ``cap`` distinct sets. Memoized on ``(product, depth_left)``.
    """
    memo: dict[tuple[int, int], frozenset[frozenset[int]]] = {}

    def leaf_sets(product: int, depth_left: int) -> frozenset[frozenset[int]]:
        if product in state.stock_sids:
            return frozenset({frozenset({product})})
        if depth_left <= 0:
            return frozenset()
        key = (product, depth_left)
        if key in memo:
            return memo[key]
        if not state.is_solved(product):
            return frozenset()
        out: set[frozenset[int]] = set()
        for _, precursor_sids in state.applications(product):
            combos: set[frozenset[int]] = {frozenset()}
            for precursor in frozenset(precursor_sids):
                child = leaf_sets(precursor, depth_left - 1)
                if not child:
                    combos = set()
                    break
                combos = {a | b for a in combos for b in child}
                if len(combos) > cap:
                    combos = set(list(combos)[:cap])
            out |= combos
            if len(out) > cap:
                out = set(list(out)[:cap])
                break
        result = frozenset(out)
        memo[key] = result
        return result

    return leaf_sets(target, max_depth)


def leaf_set_recovered(
    target: int, wanted: frozenset[int], state: GraphState[K], max_depth: int
) -> bool:
    """Does some solved route (<= ``max_depth`` reactions) have exactly ``wanted`` as its leaf
    set? Bitmask over ``wanted``: any leaf outside it kills the route, so reaching the full
    mask means the leaves are exactly ``wanted``.
    """
    bit = {sid: 1 << index for index, sid in enumerate(sorted(wanted))}
    full = (1 << len(wanted)) - 1
    memo: dict[tuple[int, int], frozenset[int]] = {}

    def reach(product: int, depth: int) -> frozenset[int]:
        if product in state.stock_sids:
            return frozenset({bit[product]}) if product in bit else frozenset()
        if depth <= 0 or not state.is_solved(product):
            return frozenset()
        key = (product, depth)
        if key in memo:
            return memo[key]
        masks: set[int] = set()
        for _, precursor_sids in state.applications(product):
            combos = {0}
            for precursor in precursor_sids:
                child = reach(precursor, depth - 1)
                if not child:
                    combos = set()
                    break
                combos = {c | m for c in combos for m in child}
            masks |= combos
        result = frozenset(masks)
        memo[key] = result
        return result

    return full in reach(target, max_depth)


def precursor_set_recovered(
    target: int, wanted: frozenset[int], state: GraphState[K], max_depth: int
) -> bool:
    """Does some solved route (<= ``max_depth`` reactions) have exactly ``wanted`` as its full
    precursor set (intermediates + leaves)? Bitmask over ``wanted``; any molecule outside it
    kills the route, and ``target`` itself is excluded.
    """
    bit = {sid: 1 << index for index, sid in enumerate(sorted(wanted))}
    full = (1 << len(wanted)) - 1
    memo: dict[tuple[int, int], frozenset[int]] = {}

    def reach(product: int, depth: int) -> frozenset[int]:
        # Masks over ``wanted`` for the molecule set of the subtree rooted at (and including)
        # ``product``; empty if any molecule is outside ``wanted`` or it cannot be solved.
        if product not in bit:
            return frozenset()
        if product in state.stock_sids:
            return frozenset({bit[product]})
        if depth <= 0 or not state.is_solved(product):
            return frozenset()
        key = (product, depth)
        if key in memo:
            return memo[key]
        masks: set[int] = set()
        for _, precursor_sids in state.applications(product):
            combos = {0}
            for precursor in precursor_sids:
                child = reach(precursor, depth - 1)
                if not child:
                    combos = set()
                    break
                combos = {c | m for c in combos for m in child}
            masks |= {bit[product] | c for c in combos}
        result = frozenset(masks)
        memo[key] = result
        return result

    if max_depth <= 0:
        return False
    target_masks: set[int] = set()
    for _, precursor_sids in state.applications(target):
        combos = {0}
        for precursor in precursor_sids:
            child = reach(precursor, max_depth - 1)
            if not child:
                combos = set()
                break
            combos = {c | m for c in combos for m in child}
        target_masks |= combos
    return full in target_masks


def minimum_route_depth(target: int, state: GraphState[K]) -> int | None:
    """Shallowest solved route depth for ``target`` (``None`` if unsolved).

    One value per product (O(edges)). A product reachable only through a cycle yields no finite
    route through itself, so it contributes ``None``.

    This is the bottleneck semiring — ``oplus`` is ``min`` and ``otimes`` is ``max``, with each
    application adding one — since a route's depth is set by its deepest branch rather than by
    their sum. Another instance of the :class:`Semiring` family, kept as its own loop because
    ``None`` carries "unsolved" where a semiring would use ``zero``. :func:`solved_postorder`
    supplies the order, and an absent entry there *is* that ``None``.

    **Relaxed to a fixpoint rather than read straight off that order**, because one pass is only
    right on a DAG. Where there are cycles the postorder cuts back edges, and a product whose only
    route runs *through* one of them is left without a value — which then propagates up, so a
    measured Retro\\* target reported 12 reactions and no depth at all, from this function
    disagreeing with the ``is_solved`` it consults. Passing over the order until nothing improves
    is Bellman-Ford on the AND/OR graph, and it is exact here for the usual reason: values only
    fall, they are bounded below by one, and no derivation that revisits a product can beat one
    that does not when every application costs a level. The postorder is what makes it cheap —
    it leaves the first pass all but final, and the second is the one that confirms it.
    """
    if target in state.stock_sids:
        return 0
    order = solved_postorder(target, state)
    depths: dict[int, int] = {}
    improved = True
    while improved:
        improved = False
        for product in order:
            best: int | None = None
            for _, precursor_sids in state.applications(product):
                deepest = 0
                for precursor in precursor_sids:
                    child = (
                        0 if precursor in state.stock_sids else depths.get(precursor)
                    )
                    if child is None:
                        break
                    deepest = max(deepest, child)
                else:
                    candidate = 1 + deepest
                    best = candidate if best is None else min(best, candidate)
            if best is not None and best < depths.get(product, best + 1):
                depths[product] = best
                improved = True
    return depths.get(target)

"""Min-plus over the whole graph at once, in arrays instead of closures.

:func:`vertsy.metrics.valuation` is generic over the semiring and lazy, which is what every
route metric wants. The search wants the opposite: one semiring, known in advance, evaluated
everywhere. At a high effort that difference is the search's largest cost — 15.7 M calls through
a memoized closure, about a quarter of the runtime — and it is not the *number* of lookups that
hurts but that a Python frame stands behind each one. Halving the lookups changed nothing;
removing the interpreter from the loop is the only thing that can.

So this is a specialisation, not a replacement. The generic fold keeps answering every other
question, in every other algebra, and the two are checked against each other.

The two tables the store keeps — ``(product, app, precursor)`` and ``(product, app, rule)`` —
become CSR arrays, and each level of the fold is then a handful of ``reduceat`` calls::

    inside[d][p] = min over p's apps a of  cost[a] + sum over a's precursors q of inside[d-1][q]
    delta[d-1][q] = min over apps a naming q of  delta[d][product[a]] + cost[a] + siblings

Both reductions are segment-wise over a sorted index, which is what CSR is for. The sibling sum
is the one place the array form differs from the scalar sweep: it is the application's total minus
the one precursor, rather than a loop over the others. That reassociates the addition, so values
can differ in the last bits — which is why the agreement tests compare with a tolerance. It is
*arithmetically* sound because a dead application is excluded first, so every remaining term is
finite and the subtraction is ordinary; min-plus has no inverse for infinity, and this never
needs one.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .core import K
from .costs import UNPROBED, StepPrice, application_price
from .expansions import Expansions
from .interner import SidArray
from .metrics import TROPICAL, GraphState, Outside, Valuation

__all__ = [
    "Shape",
    "TropicalValuation",
    "inside_table",
    "row_outcomes",
    "bottleneck_leaf_scores",
    "open_leaf_totals",
    "outside_valuation_of",
    "shape_of",
    "tropical_valuation",
]

type Floats = np.ndarray[tuple[int], np.dtype[np.float64]]


@dataclass(frozen=True)
class Shape:
    """The graph's two tables as CSR arrays: everything a min-plus sweep needs, and nothing else.

    Applications are the rows. ``app_indptr`` slices ``precursor`` into one application's
    precursors — with multiplicity, because an application that names a molecule twice needs it
    twice — and ``app_order`` regroups the applications by product so an OR-reduction is another
    segment scan. ``by_precursor`` does the same for the scatter-min the outside sweep ends in.
    """

    app_product: SidArray
    """(A,) the product each application disconnects."""
    app_cost: Floats
    """(A,) the cheapest template achieving it; min-plus takes the cheapest of alternatives."""
    app_indptr: SidArray
    """(A+1,) start of each application's precursors in ``precursor``."""
    app_size: SidArray
    """(A,) precursor count per application, so a per-application value can be broadcast."""
    precursor: SidArray
    """(E,) precursor sids, application-major."""

    app_order: SidArray
    """(A,) applications sorted by product."""
    product_rows: SidArray
    """(P,) the products that have at least one application, in the same order."""
    product_indptr: SidArray
    """(P+1,) start of each product's applications within ``app_order``."""

    by_precursor: SidArray
    """(E,) edge rows sorted by precursor sid, for the outside sweep's scatter-min."""
    precursor_rows: SidArray
    """(Q,) the distinct precursor sids, in the same order."""
    precursor_indptr: SidArray
    """(Q+1,) start of each precursor's edge rows within ``by_precursor``."""

    sid_count: int

    @property
    def applications(self) -> int:
        return len(self.app_product)


def grouped(keys: SidArray) -> tuple[SidArray, SidArray, SidArray]:
    """Sort ``keys`` into contiguous runs: the permutation, the distinct keys, the run starts.

    ``reduceat`` needs group starts and no empty groups, which is exactly what a stable sort
    plus ``unique`` gives — and a stable sort keeps a run's original order, so a reduction that
    is not commutative would still see the rows in the order the store wrote them.
    """
    order = np.argsort(keys, kind="stable")
    distinct, starts = np.unique(keys[order], return_index=True)
    return order, distinct, np.concatenate([starts, [len(keys)]]).astype(np.int64)


def app_major(column: SidArray, rows: int) -> SidArray | None:
    """``None`` when ``column`` is already non-decreasing, else the permutation that makes it so.

    Both tables are written application-major — ids are minted in order within a layer and
    offset past the store's high-water mark — so the common case is that no sort is needed at
    all. Checking is O(rows) against a sort's O(rows log rows), and it keeps the invariant
    honest rather than assumed.
    """
    if rows and not bool(np.all(np.diff(column) >= 0)):
        return np.argsort(column, kind="stable")
    return None


# An `(app, rule)` pair packed into one int64 so a join is a `searchsorted` rather than a Python
# loop. Both halves are non-negative sids, and the bound is checked rather than assumed: silently
# wrapping would make two different templates read as one measurement.
KEY_STRIDE = 1 << 31


def packed_keys(app_ids: SidArray, rule_ids: SidArray) -> SidArray:
    """Pack ``(app, rule)`` into one sortable int64 key."""
    if len(app_ids) and (
        int(app_ids.max()) >= KEY_STRIDE or int(rule_ids.max()) >= KEY_STRIDE
    ):
        raise ValueError(
            f"application and rule ids must stay below {KEY_STRIDE} to share a packed key"
        )
    return app_ids * KEY_STRIDE + rule_ids


def row_outcomes(
    expansions: Expansions[K],
    rule_apps: SidArray,
    rule_ids: SidArray,
    channels: Sequence[str],
) -> np.ndarray:
    """``(rows, channels)`` of what the forward probe measured, and `UNPROBED` where it did not.

    One left join per channel, of that channel's sparse measurement table onto the template table
    on ``(app, rule)``. A missing row reads `UNPROBED`, so a graph nobody probed reads that
    throughout and every price charges what its ``base`` says -- which is how this whole mechanism
    stays a no-op until a measurement exists. A channel the graph holds no rows for reads it too,
    which is what lets one channel be wired and measured before the other exists.

    Deliberately not ``0``: the probe writes zero when it *ran* and the forward chemistry made
    nothing, which says the pairing is broken and is a thing a price may charge for. The two have to
    be different values here, or the join hands every unprobed row a verdict it never earned.
    """
    found = np.full((len(rule_apps), len(channels)), UNPROBED, dtype=np.int64)
    if not len(rule_apps) or not len(channels):
        return found
    wanted = packed_keys(rule_apps, rule_ids)
    for column, channel in enumerate(channels):
        apps, rules, outcomes = expansions.selectivity(channel)
        if not len(apps):
            continue
        keys = packed_keys(apps, rules)
        order = np.argsort(keys, kind="stable")
        keys, values = keys[order], outcomes[order]
        at = np.searchsorted(keys, wanted)
        # `searchsorted` can point one past the end, and points at the wrong key for an unmeasured
        # row; both are answered by comparing the key it landed on.
        inside = at < len(keys)
        hit = np.zeros(len(wanted), dtype=np.bool_)
        hit[inside] = keys[at[inside]] == wanted[inside]
        found[hit, column] = values[at[hit]]
    return found


def shape_of(expansions: Expansions[K], price: StepPrice) -> Shape:
    """Build the CSR arrays from the store's two tables.

    ``price.charge`` is called once per *distinct* ``(rule, counts)`` row present, not once per
    template row. A graph names far fewer templates than it has rows, a measurement is a small
    integer and there are two or three channels, so the distinct rows are few -- which is the
    property that keeps a substrate-aware price out of the per-row Python the array fold exists to
    avoid.
    """
    edge_products, edge_apps, precursors = expansions.edges()
    reorder = app_major(edge_apps, len(edge_apps))
    if reorder is not None:
        edge_products, edge_apps, precursors = (
            edge_products[reorder],
            edge_apps[reorder],
            precursors[reorder],
        )

    count = expansions.sid_count()
    if not len(edge_apps):
        empty = np.zeros(0, dtype=np.int64)
        return Shape(
            app_product=empty,
            app_cost=np.zeros(0, dtype=np.float64),
            app_indptr=np.zeros(1, dtype=np.int64),
            app_size=empty,
            precursor=empty,
            app_order=empty,
            product_rows=empty,
            product_indptr=np.zeros(1, dtype=np.int64),
            by_precursor=empty,
            precursor_rows=empty,
            precursor_indptr=np.zeros(1, dtype=np.int64),
            sid_count=count,
        )

    total_apps = int(edge_apps[-1]) + 1
    app_size = np.bincount(edge_apps, minlength=total_apps)
    if not bool(np.all(app_size > 0)):
        raise ValueError(
            "application ids must be dense: every id needs at least one precursor"
        )
    app_indptr = np.concatenate([[0], np.cumsum(app_size)]).astype(np.int64)
    app_product = edge_products[app_indptr[:-1]]

    rule_products, rule_apps, rule_ids = expansions.rules()
    measured = row_outcomes(expansions, rule_apps, rule_ids, price.channels)
    # One key per distinct `(rule, *counts)` row, so the price is a gather off a short table. A
    # graph names few templates, a measurement is a small integer and there are two or three
    # channels, so the distinct rows stay few however large the graph gets. `np.unique` over
    # stacked columns sorts lexicographically, which is all this needs.
    rows = np.column_stack((rule_ids, measured))
    distinct, inverse = np.unique(rows, axis=0, return_inverse=True)
    costs = np.fromiter(
        (
            price.charge(int(row[0]), tuple(int(count) for count in row[1:]))
            for row in distinct.tolist()
        ),
        dtype=np.float64,
        count=len(distinct),
    )
    # An application keeps the cheapest of its templates, so this is a segment min over the
    # template table grouped by application. An application with no template row would stay
    # unaffordable, which is the right reading of a disconnection nothing achieves.
    rule_order, rule_app_rows, rule_starts = grouped(rule_apps)
    app_cost = np.full(total_apps, TROPICAL.zero, dtype=np.float64)
    app_cost[rule_app_rows] = np.minimum.reduceat(
        costs[inverse.reshape(-1)][rule_order], rule_starts[:-1]
    )

    app_order, product_rows, product_indptr = grouped(app_product)
    by_precursor, precursor_rows, precursor_indptr = grouped(precursors)
    return Shape(
        app_product=app_product,
        app_cost=app_cost,
        app_indptr=app_indptr,
        app_size=app_size,
        precursor=precursors,
        app_order=app_order,
        product_rows=product_rows,
        product_indptr=product_indptr,
        by_precursor=by_precursor,
        precursor_rows=precursor_rows,
        precursor_indptr=precursor_indptr,
        sid_count=count,
    )


def masks_of(state: GraphState[K], count: int) -> tuple[np.ndarray, np.ndarray]:
    """Dense ``(stock, expanded)`` bit masks over sids, from the state's Python collections."""
    stock = np.zeros(count, dtype=np.bool_)
    if state.stock_sids:
        stock[
            np.fromiter(state.stock_sids, dtype=np.int64, count=len(state.stock_sids))
        ] = True
    expanded = np.zeros(count, dtype=np.bool_)
    if state.steps:
        expanded[np.fromiter(state.steps, dtype=np.int64, count=len(state.steps))] = (
            True
        )
    return stock, expanded


# The two ways an application may aggregate its precursors, and the numpy reduction for each.
# Addition makes a route's value its reaction count; maximum makes it the longest linear sequence.
# A dict rather than a branch so an unsupported aggregation fails where it is chosen, by name.
AGGREGATIONS: dict[Callable[[float, float], float], np.ufunc] = {
    operator.add: np.add,
    max: np.maximum,
}


def leave_one_out_max(
    values: np.ndarray, app_size: SidArray, app_indptr: SidArray
) -> np.ndarray:
    """Per edge row, the ``max`` over the *other* rows of its application. ``one`` if it has none.

    The sibling term the outside sweep needs, and the one place ``max`` cannot follow the summing
    fold's trick: that one takes the application's total and subtracts each precursor's own value,
    which no maximum admits. Prefix and suffix maxima give it instead — ``max`` of everything left
    of a position and everything right of it — and those are one ``accumulate`` each *provided the
    segments have equal length*, which is what the arity buckets are for.

    The loop runs over distinct arities, not applications: measured on used14k an application has
    one, two or at most three precursors, so this is two or three vector passes over the edge rows
    however large the graph gets. ``TROPICAL.one`` is zero, which is the identity of ``max`` over
    the non-negative costs and distances this fold carries, so an arity-one application needs no
    special case beyond being left alone.
    """
    out = np.full(values.shape, TROPICAL.one, dtype=np.float64)
    for arity in np.unique(app_size):
        width = int(arity)
        if width < 2:
            continue
        starts = app_indptr[:-1][app_size == arity]
        rows = starts[:, None] + np.arange(width)
        block = values[rows]
        prefix = np.maximum.accumulate(block, axis=1)
        suffix = np.maximum.accumulate(block[:, ::-1], axis=1)[:, ::-1]
        found = np.empty_like(block)
        found[:, 0] = suffix[:, 1]
        found[:, -1] = prefix[:, -2]
        if width > 2:
            found[:, 1:-1] = np.maximum(prefix[:, :-2], suffix[:, 2:])
        out[rows] = found
    return out


def inside_table(
    shape: Shape,
    *,
    leaf: Floats,
    stock: np.ndarray,
    expanded: np.ndarray,
    max_depth: int,
    aggregate: np.ufunc = np.add,
) -> np.ndarray:
    """``(max_depth + 1, sid_count)`` of ``rn``: the cheapest route below each product.

    One row per horizon, because a product's value depends on how much budget is left when it is
    reached, and the outside sweep asks for several horizons of the same product. The three base
    cases are the scalar fold's, applied as masks rather than as branches: stock is free, a
    product past the horizon is unreachable, an unexpanded one is worth its estimate.
    """
    table = np.empty((max_depth + 1, shape.sid_count), dtype=np.float64)
    table[0] = np.where(stock, TROPICAL.one, TROPICAL.zero)
    unopened = np.where(expanded, TROPICAL.zero, leaf)
    for depth in range(1, max_depth + 1):
        level = unopened.copy()
        if shape.applications:
            below = table[depth - 1][shape.precursor]
            app_value = shape.app_cost + aggregate.reduceat(
                below, shape.app_indptr[:-1]
            )
            level[shape.product_rows] = np.minimum.reduceat(
                app_value[shape.app_order], shape.product_indptr[:-1]
            )
        # Stock last, as the scalar fold checks it first: a buyable product costs nothing however
        # it was reached, and a dead end stays unreachable because `unopened` left it at zero.
        table[depth] = np.where(stock, TROPICAL.one, level)
    return table


def outside_table(
    shape: Shape,
    inside: np.ndarray,
    target: int,
    max_depth: int,
    aggregate: np.ufunc = np.add,
) -> np.ndarray:
    """``(max_depth + 1, sid_count)`` of ``delta``: what a graph through a product needs besides it.

    Swept down from the target, one level per iteration, exactly as the scalar version — the
    difference is that a level is now three array operations rather than a walk over products.
    Addition gives an exact outside value. With ``max``, each immediate sibling aggregate is exact
    but its composition with the scalar parent context is an additive upper bound.
    """
    table = np.full((max_depth + 1, shape.sid_count), TROPICAL.zero, dtype=np.float64)
    table[max_depth][target] = TROPICAL.one
    if not shape.applications:
        return table

    for depth in range(max_depth, 0, -1):
        below = inside[depth - 1][shape.precursor]
        reachable = np.isfinite(below)
        # An application is live only if its product is reachable, its template is affordable,
        # and every precursor of it is. Anything else contributes nothing to any precursor.
        dead = np.add.reduceat((~reachable).astype(np.int64), shape.app_indptr[:-1]) > 0
        if aggregate is np.add:
            # Leave-one-out by subtraction: the application's total less the precursor's own value.
            # `below` is finite wherever `base` is, so no infinity is ever subtracted from another;
            # where `base` is infinite the row is dead and stays so whatever it reads. Kept exactly
            # as it was, arithmetic and order both, because the scalar sweep is checked against it.
            app_sum = np.add.reduceat(below, shape.app_indptr[:-1])
            base = np.where(
                dead, TROPICAL.zero, table[depth][shape.app_product] + shape.app_cost
            )
            base = base + np.where(dead, 0.0, app_sum)
            rows = np.repeat(base, shape.app_size) - np.where(reachable, below, 0.0)
        else:
            # No subtraction available, so the sibling term is built directly. Masking the
            # infinities out first is safe for the same reason it is above: an application holding
            # one is already `dead`, so whatever its rows read is discarded.
            siblings = leave_one_out_max(
                np.where(reachable, below, TROPICAL.one),
                shape.app_size,
                shape.app_indptr,
            )
            base = np.where(
                dead, TROPICAL.zero, table[depth][shape.app_product] + shape.app_cost
            )
            rows = np.repeat(base, shape.app_size) + siblings
        table[depth - 1][shape.precursor_rows] = np.minimum.reduceat(
            rows[shape.by_precursor], shape.precursor_indptr[:-1]
        )
        if not np.isfinite(table[depth - 1]).any():
            break
    return table


def open_leaves(
    inside: np.ndarray,
    outside: np.ndarray,
    *,
    stock: np.ndarray,
    expanded: np.ndarray,
) -> dict[int, tuple[float, float, int]]:
    """The open leaves with a finite total, as :attr:`vertsy.metrics.Outside.leaves` holds them.

    A leaf is priced at every horizon it is reachable at and keeps its cheapest, which is one
    ``argmin`` down the horizon axis. The attaining horizon may differ from the scalar sweep's
    when two of them tie — the sweep descends and keeps the first strict improvement, so it
    reports the deepest of a tie, and ``argmin`` reports the shallowest. Only the total and its
    context are read by selection; the horizon is there to be looked at.
    """
    totals = np.where(stock | expanded, TROPICAL.zero, outside + inside)
    depths = np.argmin(totals, axis=0)
    columns = np.arange(totals.shape[1])
    best = totals[depths, columns]
    found = np.flatnonzero(np.isfinite(best))
    return {
        int(sid): (
            float(best[sid]),
            float(outside[depths[sid], sid]),
            int(depths[sid]),
        )
        for sid in found.tolist()
    }


@dataclass(frozen=True)
class TropicalValuation(Valuation[float]):
    """A :class:`vertsy.metrics.Valuation` backed by a table, plus the arrays that built it.

    ``node`` and ``application`` answer exactly as the generic fold does, by indexing instead of
    recursing, so anything that takes a ``Valuation`` keeps working. Carrying ``shape`` and
    ``inside`` is what lets a selection skip them entirely and stay in arrays.
    """

    shape: Shape
    inside: np.ndarray


def tropical_valuation(
    state: GraphState[K],
    *,
    leaf_value: Floats,
    price: StepPrice,
    max_depth: int,
    conjoin: Callable[[float, float], float] = operator.add,
) -> TropicalValuation:
    """Fold the whole graph under min-plus, at every horizon up to ``max_depth``, at once.

    ``leaf_value`` is dense and sid-indexed rather than a callable: the point is to have the
    estimates as an array, and a search that keeps them one way can hand them over directly.
    Sids past its end, or holding infinity, are unreachable estimates — the honest reading for
    a molecule nothing has scored.

    ``conjoin`` is how an application combines its precursors, and only the two aggregations in
    :data:`AGGREGATIONS` have an array form: ``operator.add`` for reaction count and ``max`` for
    longest linear sequence. Both are one segmented reduction in the inside table; they differ in
    the outside sweep, where addition recovers a sibling term by subtracting and ``max`` has to
    build it — see :func:`leave_one_out_max`.
    """
    if conjoin not in AGGREGATIONS:
        raise ValueError(
            f"no array form for {conjoin!r}; expected one of {sorted(AGGREGATIONS, key=repr)}."
        )
    aggregate = AGGREGATIONS[conjoin]
    shape = shape_of(state.expansions, price)
    stock, expanded = masks_of(state, shape.sid_count)
    leaf = np.full(shape.sid_count, TROPICAL.zero, dtype=np.float64)
    scored = min(len(leaf_value), shape.sid_count)
    leaf[:scored] = leaf_value[:scored]
    inside = inside_table(
        shape,
        leaf=leaf,
        stock=stock,
        expanded=expanded,
        max_depth=max_depth,
        aggregate=aggregate,
    )

    def node(product: int, depth_left: int) -> float:
        if depth_left <= 0:
            return TROPICAL.one if stock[product] else TROPICAL.zero
        return float(inside[min(depth_left, max_depth)][product])

    def application(
        step: tuple[tuple[int, ...], tuple[int, ...]], depth_left: int
    ) -> float:
        _rules, precursors = step
        remaining = TROPICAL.one
        for precursor in precursors:
            remaining = conjoin(remaining, node(precursor, depth_left - 1))
        return application_price(price, step) + remaining

    # Carried, not assumed: the outside sweep reads it back off the value to pick the same
    # segmented reduction the inside table was built with. The two halves of a leaf's price cannot
    # come from different algebras if only one of them is ever chosen.
    return TropicalValuation(
        node=node,
        application=application,
        conjoin=conjoin,
        shape=shape,
        inside=inside,
    )


def bottleneck_leaf_scores(
    target: int,
    state: GraphState[K],
    value: TropicalValuation,
    *,
    max_depth: int,
) -> dict[int, float]:
    """Exact constrained leaf costs for a max fold with uniform reaction cost.

    A path with accumulated reaction cost ``a`` and sibling floor ``b`` prices a leaf
    at ``max(a + leaf, b)``. Uniform reaction costs fix ``a`` at each depth, so OR paths
    merge by taking their smallest ``b``. This needs one scalar per node and depth,
    unlike the envelope of contexts required when reaction costs differ.

    Leaf estimates must be nonnegative. Stock nodes terminate paths, even if a
    resumed graph contains their expansions. The horizon bounds cycles in the
    same way as the inside fold.
    """
    shape = value.shape
    if value.conjoin is not max:
        raise ValueError("bottleneck leaf scoring requires a max fold")
    if max_depth < 0 or max_depth >= len(value.inside):
        raise ValueError("horizon must lie within the prepared inside table")
    if shape.applications and not np.all(shape.app_cost == shape.app_cost[0]):
        raise ValueError(
            "bottleneck leaf scoring requires uniform reaction costs; a substrate-aware price "
            "(see vertsy.costs.selective) makes them differ per application"
        )
    cost = float(shape.app_cost[0]) if shape.applications else 0.0
    if not np.isfinite(cost) or cost < 0.0:
        raise ValueError("bottleneck leaf scoring requires finite nonnegative costs")
    stock, expanded = masks_of(state, shape.sid_count)
    floors = np.full((max_depth + 1, shape.sid_count), np.inf)
    floors[max_depth, target] = 0.0
    for depth in range(max_depth, 0, -1):
        if not shape.applications:
            break
        below = value.inside[depth - 1, shape.precursor]
        finite = np.isfinite(below)
        dead = np.logical_or.reduceat(~finite, shape.app_indptr[:-1])
        dead |= stock[shape.app_product]
        siblings = leave_one_out_max(
            np.where(finite, below, 0.0), shape.app_size, shape.app_indptr
        )
        parent = np.where(dead, np.inf, floors[depth, shape.app_product])
        path_cost = (max_depth - depth + 1) * cost
        rows = np.maximum(np.repeat(parent, shape.app_size), path_cost + siblings)
        floors[depth - 1, shape.precursor_rows] = np.minimum.reduceat(
            rows[shape.by_precursor], shape.precursor_indptr[:-1]
        )
    path_costs = (max_depth - np.arange(max_depth + 1))[:, None] * cost
    totals = np.maximum(floors, value.inside[: max_depth + 1] + path_costs)
    totals[:, stock | expanded] = np.inf
    best = totals.min(axis=0)
    return {int(sid): float(best[sid]) for sid in np.flatnonzero(np.isfinite(best))}


def open_leaf_totals(
    target: int,
    state: GraphState[K],
    value: TropicalValuation,
    *,
    max_depth: int,
) -> dict[int, tuple[float, float, int]]:
    """Just the open leaves, which is all a selection reads — no ``delta`` dict is built."""
    stock, expanded = masks_of(state, value.shape.sid_count)
    outside = outside_table(
        value.shape,
        value.inside,
        target,
        max_depth,
        aggregate=AGGREGATIONS[value.conjoin],
    )
    return open_leaves(value.inside, outside, stock=stock, expanded=expanded)


def outside_valuation_of(
    target: int,
    state: GraphState[K],
    value: TropicalValuation,
    *,
    max_depth: int,
) -> Outside:
    """The array twin of :func:`vertsy.metrics.outside_valuation`, whole.

    ``delta`` comes back in the sparse dict shape its scalar counterpart uses, since callers index
    it by ``(product, depth)``. Only comparisons against that counterpart want it — a search reads
    the leaves and nothing else, so it calls :func:`open_leaf_totals` instead.
    """
    stock, expanded = masks_of(state, value.shape.sid_count)
    outside = outside_table(
        value.shape,
        value.inside,
        target,
        max_depth,
        aggregate=AGGREGATIONS[value.conjoin],
    )
    live = np.argwhere(np.isfinite(outside))
    return Outside(
        delta={
            (int(sid), int(depth)): float(outside[depth, sid])
            for depth, sid in live.tolist()
        },
        leaves=open_leaves(value.inside, outside, stock=stock, expanded=expanded),
    )

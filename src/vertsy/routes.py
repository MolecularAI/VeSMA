from __future__ import annotations

from collections import deque
from collections.abc import Callable, Collection, Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import polars as pl

from vesma.keyer import Keyer

from .core import K, M, RuleId
from .expansions import Expansions, edge_frame, sid_steps
from .interner import MISSING_SID, SidArray
from .store import SidMask, SidStep

# Target key type for `map_route_mols`, distinct from the source key `K` (a route may be
# re-keyed into a different key flavor). Local, non-vocabulary; bound named once here.
L = TypeVar("L", bound=Hashable)

# Nested (rule, child-skeletons) tree with leaves/intermediates dropped: the recovery
# invariant a route must match. ``None`` marks a terminal (leaf/stock) node.
type Skeleton = tuple[RuleId, Sequence[Skeleton]] | None

__all__ = [
    "Route",
    "SidRoute",
    "bounded_solvability",
    "collapsed_route_key",
    "count_recoveries",
    "decode_route",
    "encode_route",
    "forward_steps",
    "is_solved",
    "iter_solved_routes",
    "iter_solved_sid_routes",
    "iter_solved_sid_routes_bounded",
    "leaves",
    "library_route_matches",
    "library_route_replayed",
    "loose_route_matches",
    "loose_route_replayed",
    "map_route_mols",
    "pretty_string",
    "reaction_count",
    "reaction_depth",
    "reaction_products",
    "reaction_skeleton",
    "round_robin",
    "route_root",
    "solved_sids",
    "solved_targets",
    "strict_route_matches",
    "strict_route_replayed",
]


@dataclass(frozen=True)
class SidRoute:
    """Nested sid-space route tree.

    ``rule is None`` marks a terminal leaf, usually a stock molecule.
    """

    product: int
    rule: int | None
    precursors: tuple[SidRoute, ...] = ()


@dataclass(frozen=True, eq=False)
class Route(Generic[K]):
    """Nested key-space route tree."""

    product: K
    rule: int | None
    precursors: tuple[Route[K], ...] = ()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Route):
            return NotImplemented
        return route_key(self) == route_key(other)

    def __hash__(self) -> int:
        return hash(route_key(self))


def sid_route_combinations(
    precursor_sids: Sequence[int],
    route_options: Callable[[int], Iterator[SidRoute]],
    index: int = 0,
) -> Iterator[tuple[SidRoute, ...]]:
    """Lazy Cartesian product of precursor routes.

    ``itertools.product`` pools every input iterable before yielding. A route node can have an
    enormous child route space, so that defeats the public enumeration cap and can exhaust RAM
    while asking for one route. Re-entering the later generators trades some repeated traversal
    for bounded memory and immediate yielding.
    """
    if index == len(precursor_sids):
        yield ()
        return
    for route in route_options(int(precursor_sids[index])):
        for suffix in sid_route_combinations(precursor_sids, route_options, index + 1):
            yield (route,) + suffix


def round_robin(iterables: Sequence[Iterator[SidRoute]]) -> Iterator[SidRoute]:
    """Yield one item from each iterator in turn until all are exhausted."""
    active = deque(iter(iterable) for iterable in iterables)
    while active:
        iterator = active.popleft()
        try:
            yield next(iterator)
        except StopIteration:
            continue
        active.append(iterator)


def bounded_solvability(
    stock: frozenset[int],
    steps: Mapping[int, tuple[SidStep, ...]],
) -> Callable[[int, int], bool]:
    """Return a memoized test for a route to stock within a reaction depth.

    Route enumeration uses this to reject an application before descending into any
    precursor whose solved route exceeds the remaining depth. The ordinary ``solved`` mask
    has no depth information, so relying on it alone can traverse millions of ultimately
    empty route combinations before yielding a bounded route.
    """
    memo: dict[tuple[int, int], bool] = {}

    def can_solve(product: int, depth_left: int) -> bool:
        if product in stock:
            return True
        if depth_left <= 0:
            return False
        key = (product, depth_left)
        if key not in memo:
            memo[key] = any(
                all(
                    can_solve(int(precursor), depth_left - 1)
                    for precursor in precursors
                )
                for _, precursors in steps.get(product, ())
            )
        return memo[key]

    return can_solve


def solved_sids(expansions: Expansions[K], stock_sids: SidArray) -> SidMask:
    """Return a dense sid mask for products with at least one route to stock.

    This is a fixpoint over grouped append-only application rows. Routes
    themselves only consume the resulting sid mask.
    """
    solved = np.zeros(expansions.sid_count(), dtype=np.bool_)
    solved[stock_sids] = True

    edges = edge_frame(expansions)
    if edges.is_empty():
        return solved
    precursor = edges.get_column("precursor").to_numpy()

    while True:
        solved_applications = (
            edges.with_columns(pl.Series("precursor_is_solved", solved[precursor]))
            .group_by("product", "app")
            .agg(pl.col("precursor_is_solved").all().alias("app_all_is_solved"))
            .filter(pl.col("app_all_is_solved"))
        )
        newly_solved_products = solved_applications["product"].to_numpy()
        solved_count = int(solved.sum())
        solved[newly_solved_products] = True
        if int(solved.sum()) == solved_count:
            return solved


def iter_solved_sid_routes(
    product: int,
    expansions: Expansions[K],
    solved: SidMask,
    stock: frozenset[int],
    trail: frozenset[int] = frozenset(),
    steps: Mapping[int, tuple[SidStep, ...]] | None = None,
) -> Iterator[SidRoute]:
    """Yield solved sid-space route trees rooted at ``product``.

    One route per *disconnection*, labelled with the lowest-numbered template that achieves it.
    Co-canonical templates are not alternative routes — yielding one tree per template would
    return the same chemistry spelled several ways, which is the opposite of a diverse pool. Ask
    the graph (:meth:`vertsy.expansions.Expansions.steps`) when the full template set matters.
    """
    if steps is None:
        steps = sid_steps(expansions, expansions.store.expanded())
    if product in stock:
        yield SidRoute(product, None)
        return

    if product in trail or not solved[product]:
        return

    next_trail = trail | frozenset({product})

    for rules, precursor_sids in steps.get(product, ()):
        if not all(solved[precursor] for precursor in precursor_sids):
            continue

        def options(precursor: int) -> Iterator[SidRoute]:
            return iter_solved_sid_routes(
                precursor, expansions, solved, stock, next_trail, steps
            )

        for children in sid_route_combinations(precursor_sids, options):
            yield SidRoute(product, rules[0], tuple(children))


def iter_solved_sid_routes_bounded(
    product: int,
    expansions: Expansions[K],
    solved: SidMask,
    stock: frozenset[int],
    depth_left: int,
    trail: frozenset[int] = frozenset(),
    steps: Mapping[int, tuple[SidStep, ...]] | None = None,
    can_solve: Callable[[int, int], bool] | None = None,
) -> Iterator[SidRoute]:
    """Yield rule-collapsed solved routes using at most ``depth_left`` reactions."""
    if steps is None:
        steps = sid_steps(expansions, expansions.store.expanded())
    if can_solve is None:
        can_solve = bounded_solvability(stock, steps)
    if product in stock:
        yield SidRoute(product, None)
        return
    if product in trail or not solved[product] or not can_solve(product, depth_left):
        return

    next_trail = trail | frozenset({product})
    application_routes: list[Iterator[SidRoute]] = []
    for rules, precursor_sids in steps.get(product, ()):
        if not all(
            solved[precursor] and can_solve(int(precursor), depth_left - 1)
            for precursor in precursor_sids
        ):
            continue

        def routes_for_application(
            rules: tuple[int, ...] = rules,
            precursor_sids: Sequence[int] = precursor_sids,
        ) -> Iterator[SidRoute]:
            def options(child: int) -> Iterator[SidRoute]:
                return iter_solved_sid_routes_bounded(
                    child,
                    expansions,
                    solved,
                    stock,
                    depth_left - 1,
                    next_trail,
                    steps,
                    can_solve,
                )

            for children in sid_route_combinations(precursor_sids, options):
                yield SidRoute(product, rules[0], tuple(children))

        application_routes.append(routes_for_application())

    yield from round_robin(application_routes)


def iter_solved_routes(
    target: K,
    expansions: Expansions[K],
    stock: Collection[K],
    *,
    cap: int | None = None,
    max_depth: int | None = None,
) -> Iterator[Route[K]]:
    """Yield solved key-space route trees for one target.

    Enumeration is a cartesian product over per-precursor route options and can explode on
    cyclic motifs (protect/deprotect), so this is a cold path — prefer the graph metrics in
    :mod:`vertsy.metrics` for aggregates. ``cap`` bounds how many routes are yielded (``None``
    yields all); pass one when only a sample or a capped count is needed. ``max_depth``
    restricts each root-to-leaf reaction sequence.
    """
    target_sid = expansions.lookup([target])[0]
    if target_sid == MISSING_SID:
        return

    stock_sids = existing_sids(expansions.lookup(stock))
    solved = solved_sids(expansions, stock_sids)
    stock_set = frozenset(stock_sids.tolist())

    source = (
        iter_solved_sid_routes(target_sid, expansions, solved, stock_set)
        if max_depth is None
        else iter_solved_sid_routes_bounded(
            target_sid, expansions, solved, stock_set, max_depth
        )
    )
    for count, route in enumerate(source):
        if cap is not None and count >= cap:
            return
        yield decode_route(route, expansions)


def solved_targets(
    targets: Collection[K],
    expansions: Expansions[K],
    stock: Collection[K],
) -> frozenset[K]:
    """Return the requested targets that have at least one route to stock."""
    target_keys = tuple(targets)
    target_sids = expansions.lookup(target_keys)
    stock_sids = existing_sids(expansions.lookup(stock))
    solved = solved_sids(expansions, stock_sids)
    return frozenset(
        target
        for target, sid in zip(target_keys, target_sids)
        if sid != MISSING_SID and solved[sid]
    )


def decode_route(route: SidRoute, expansions: Expansions[K]) -> Route[K]:
    """Decode a sid-space route tree into key space."""
    product = expansions.decode([route.product])[0]
    return Route(
        product,
        route.rule,
        tuple(decode_route(child, expansions) for child in route.precursors),
    )


def leaves(route: Route[K]) -> frozenset[K]:
    """Return terminal route products."""
    if route.rule is None:
        return frozenset({route.product})

    out: set[K] = set()
    for child in route.precursors:
        out.update(leaves(child))
    return frozenset(out)


def reaction_products(route: Route[K]) -> frozenset[K]:
    """Return the products a route makes: every node carrying a reaction. Mirrors :func:`leaves`.

    These are exactly the molecules a search had to expand to produce the route, which is what
    lets :mod:`vertsy.effort` price it against the effort that was spent.
    """
    if route.rule is None:
        return frozenset()

    out: set[K] = {route.product}
    for child in route.precursors:
        out.update(reaction_products(child))
    return frozenset(out)


def is_solved(route: Route[K], stock: Collection[K]) -> bool:
    return leaves(route).issubset(stock)


def reaction_count(route: Route[K]) -> int:
    """Return total reaction steps in a route tree."""
    return int(route.rule is not None) + sum(
        reaction_count(child) for child in route.precursors
    )


def reaction_depth(route: Route[K]) -> int:
    """Return longest reaction sequence length in a route tree."""
    if route.rule is None:
        return 0
    return 1 + max(
        (reaction_depth(child) for child in route.precursors),
        default=0,
    )


def route_root(route: Route[K]) -> K:
    return route.product


def route_key(route: Route[K]) -> tuple[K, int | None, tuple[object, ...]]:
    return (
        route.product,
        route.rule,
        tuple(sorted((route_key(child) for child in route.precursors), key=repr)),
    )


def map_route_mols(route: Route[K], map_mol: Callable[[K], L]) -> Route[L]:
    return Route(
        map_mol(route.product),
        route.rule,
        tuple(map_route_mols(child, map_mol) for child in route.precursors),
    )


def encode_route(route: Route[str], codec: Keyer[M, K]) -> Route[K]:
    return map_route_mols(route, codec.from_smiles)


def pretty_string(route: Route[K], indent: str = "") -> str:
    """Pretty-print a nested route tree."""
    out = [f"{indent}{route.product}"]
    if route.rule is None:
        return "\n".join(out)

    out.append(f"{indent}  {route.rule}")
    for child in route.precursors:
        out.append(pretty_string(child, indent + "    "))
    return "\n".join(out)


def forward_steps(route: Route[K]) -> tuple[Route[K], ...]:
    """List reaction nodes with precursor-producing steps before their use."""
    return tuple(
        step for child in route.precursors for step in forward_steps(child)
    ) + ((route,) if route.rule is not None else ())


def existing_sids(sids: SidArray) -> SidArray:
    return sids[sids != MISSING_SID]


# ------------------------------------------------------------------ route comparison
#
# Ask whether a reference route is reproduced by a set of generated routes, at three
# increasingly permissive levels. Shared by the route-replay and route-recovery
# benchmarks. Rule ids on both sides must come from the same template numbering for the
# comparison to mean anything.


def collapsed_route_key(route: Route[K]) -> tuple[K, int | None, frozenset[object]]:
    """Route identity with each reaction's precursors collapsed to a set.

    Stoichiometry, duplicate precursors, and precursor order are ignored: a reaction that
    emits several identical molecules (e.g. two equivalents of water) is indistinguishable
    from one that emits a single copy. Used by the strict comparator.
    """
    return (
        route.product,
        route.rule,
        frozenset(collapsed_route_key(child) for child in route.precursors),
    )


def reaction_skeleton(route: Route[K]) -> Skeleton:
    """Nested ``(rule, child-skeletons)`` tuple, dropping leaves and intermediates."""
    if route.rule is None:
        return None
    children = tuple(
        child_skeleton
        for child_skeleton in (reaction_skeleton(child) for child in route.precursors)
        if child_skeleton is not None
    )
    return (route.rule, tuple(sorted(children, key=repr)))


def strict_route_matches(reference: Route[K], route: Route[K]) -> bool:
    """``route`` reproduces ``reference`` node-for-node, up to stoichiometry."""
    return collapsed_route_key(route) == collapsed_route_key(reference)


def loose_route_matches(reference: Route[K], route: Route[K]) -> bool:
    """Same target, leaves, and reaction skeleton; intermediates may differ."""
    return (route.product, leaves(route), reaction_skeleton(route)) == (
        reference.product,
        leaves(reference),
        reaction_skeleton(reference),
    )


def library_route_matches(reference: Route[K], route: Route[K]) -> bool:
    """Same target and reaction skeleton; leaves and intermediates may differ."""
    return (route.product, reaction_skeleton(route)) == (
        reference.product,
        reaction_skeleton(reference),
    )


def count_recoveries(
    reference: Route[K],
    routes: Collection[Route[K]],
    matches: Callable[[Route[K], Route[K]], bool],
) -> int:
    """Number of ``routes`` that reproduce ``reference`` under ``matches``."""
    return sum(1 for route in routes if matches(reference, route))


def strict_route_replayed(expected: Route[K], routes: Collection[Route[K]]) -> bool:
    return any(strict_route_matches(expected, route) for route in routes)


def loose_route_replayed(expected: Route[K], routes: Collection[Route[K]]) -> bool:
    return any(loose_route_matches(expected, route) for route in routes)


def library_route_replayed(expected: Route[K], routes: Collection[Route[K]]) -> bool:
    return any(library_route_matches(expected, route) for route in routes)

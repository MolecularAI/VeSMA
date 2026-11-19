from __future__ import annotations

from collections import Counter
from collections.abc import (
    Callable,
    Collection,
    Container,
    Hashable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from itertools import product as cartesian
from typing import Generic, TypeVar, cast

import numpy as np

from vesma.matcher import Pair, compact_precursor_index

from .core import HitArray, K, M
from .expansions import Expansions
from .interner import SidArray
from .routes import Route
from .search import Match, MoleculeBatch, one_per_sid

L = TypeVar("L", bound=Hashable)

__all__ = [
    "ReplayMismatch",
    "precursor_replay_search",
    "replay_search",
    "template_replay_routes",
    "route_precursor_sets",
]


@dataclass(frozen=True)
class ReplayMismatch(Generic[K]):
    """A node whose recorded precursors no reproduced application could match.

    Replay is guided: it fires the rule the route recorded and requires the emitted
    precursor keys to equal the recorded children's. When that never holds, the route is
    silently dropped, and the count of dropped routes is the only thing that survives. That
    is too little to act on -- a route lost because the matchers disagree on *which*
    precursors and a route lost because they disagree on *how many* are different bugs.

    ``emitted`` carries one key tuple per application the matcher produced, so a
    ``len(recorded) != len(emitted[i])`` reading says the two sides disagree on the
    arity of the disconnection rather than on its chemistry.
    """

    rule: int
    recorded: tuple[K, ...]
    emitted: tuple[tuple[K, ...], ...]

    @property
    def arity_differs(self) -> bool:
        """True when no application emitted as many precursors as the route recorded."""
        return all(len(keys) != len(self.recorded) for keys in self.emitted)

    def __str__(self) -> str:
        shapes = sorted({len(keys) for keys in self.emitted})
        return (
            f"rule {self.rule}: route recorded {len(self.recorded)} precursor(s), "
            f"replay emitted {len(self.emitted)} application(s) with "
            f"{shapes or [0]} precursor(s)"
        )


# Product key -> the reference precursor key set(s) accepted at that product. A frozenset of
# frozensets: usually one set per product, but a product reused across a batch maps to the
# union of its precursor sets so any of them is accepted.
AllowedPrecursorSets = Mapping[K, "frozenset[frozenset[K]]"]


SelectiveKeyedMatch = Callable[
    [MoleculeBatch[M], Sequence[Pair]],
    tuple[MoleculeBatch[M], Sequence[K], HitArray],
]


def template_replay_routes(
    route: Route[K],
    product_mol: M,
    *,
    match: SelectiveKeyedMatch[M, K],
    stack_mols: Callable[[Sequence[M]], MoleculeBatch[M]],
    output_key: Callable[[M], L],
    cap: int = 1,
    on_mismatch: Callable[[ReplayMismatch[K]], None] | None = None,
) -> tuple[Route[L], ...]:
    """Replay a keyed route with its templates and return readable output-key routes.

    This is the guided replay needed when the search identity cannot be decoded to a
    molecule, such as a WL hash. Each non-leaf is expanded only with the rule recorded in
    ``route``. Applications are retained when their precursor-key multiset equals the
    reference children's product-key multiset, then replay continues recursively with the
    actual precursor molecules. ``output_key`` independently chooses the returned identity,
    normally canonical SMILES.

    The replay verifies that another matcher reproduces the searched graph. It does not
    establish reaction feasibility. ``cap`` bounds alternative assignments caused by
    symmetric matches or collisions in the search key.
    """
    if cap <= 0:
        raise ValueError("template replay cap must be positive")

    def replay(node: Route[K], molecule: M) -> tuple[Route[L], ...]:
        product = output_key(molecule)
        if node.rule is None:
            return (Route(product, None),)

        precursor_mols, precursor_keys, hits = match(
            stack_mols([molecule]), [(0, node.rule)]
        )
        expected = Counter(child.product for child in node.precursors)
        applications: dict[int, list[int]] = {}
        for product_idx, _rule, app_idx, precursor_idx in np.asarray(
            hits, dtype=np.int64
        ).reshape(-1, 4):
            if product_idx != 0:
                raise ValueError("template replay expects one product molecule")
            applications.setdefault(int(app_idx), []).append(int(precursor_idx))

        found: list[Route[L]] = []
        matched_any = False
        for indices in applications.values():
            if Counter(precursor_keys[index] for index in indices) != expected:
                continue
            matched_any = True
            for assignment in child_assignments(
                node.precursors, tuple(indices), precursor_keys
            ):
                options = tuple(
                    replay(child, cast(M, precursor_mols[index]))
                    for child, index in zip(node.precursors, assignment, strict=True)
                )
                for children in cartesian(*options):
                    found.append(Route(product, node.rule, tuple(children)))
                    if len(found) >= cap:
                        return tuple(found)
        if not matched_any and on_mismatch is not None:
            on_mismatch(
                ReplayMismatch(
                    rule=node.rule,
                    recorded=tuple(child.product for child in node.precursors),
                    emitted=tuple(
                        tuple(precursor_keys[index] for index in indices)
                        for indices in applications.values()
                    ),
                )
            )
        return tuple(found)

    return replay(route, product_mol)


def step_precursor_names(
    product_mol: M,
    rules: Sequence[int],
    *,
    match: SelectiveKeyedMatch[M, K],
    stack_mols: Callable[[Sequence[M]], MoleculeBatch[M]],
    output_key: Callable[[M], L],
) -> dict[K, L]:
    """Search key -> readable key for every precursor these rules make from one real molecule.

    What a report needs and a replayed route cannot give it. Replay reproduces the route that was
    *chosen*, so every molecule on it has a name; the alternatives a report must show are other
    disconnections of those same molecules, which were never replayed, and under a hashing arm
    their search keys say nothing a reader can use. Applying their rules to the product -- which
    replay has already turned into a real molecule -- names them.

    Exactly, not approximately: the returned mapping is keyed by whatever ``match`` computes as the
    search key, so a name is attached to a molecule only where the keyer that named it and the
    keyer that searched agree. Nothing is inferred from position, and an application the arm
    refuses simply contributes no names.

    One matcher call for the whole step, by stacking the same molecule once per rule -- the shape
    :func:`replay_step` uses, where each slot in the batch carries its own rule.

    **A molecule the output keyer cannot express costs one name and nothing else.** This runs over
    every disconnection of every molecule on a reported route, including rules the search would
    never take, so it reaches products no replay ever had to stand behind -- and canonical SMILES
    goes through RDKit, which refuses to kekulise some of them. A report is an add-on: it may lose
    a name, and it may not take down the run it is describing. Measured: on a twenty-target sweep
    one such molecule appeared at target 8 and ended the run with a `KekulizeException`.
    """
    if not rules:
        return {}
    batch = stack_mols([product_mol] * len(rules))
    precursor_mols, precursor_keys, _hits = match(
        batch, [(index, rule) for index, rule in enumerate(rules)]
    )
    names: dict[K, L] = {}
    for index, key in enumerate(precursor_keys):
        try:
            names[key] = output_key(cast(M, precursor_mols[index]))
        except Exception:
            continue
    return names


def child_assignments(
    children: Sequence[Route[K]],
    indices: tuple[int, ...],
    precursor_keys: Sequence[K],
) -> tuple[tuple[int, ...], ...]:
    """Assign one emitted precursor to every reference child by search key."""
    if not children:
        return ((),) if not indices else ()
    first, *rest = children
    assignments: list[tuple[int, ...]] = []
    for position, index in enumerate(indices):
        if precursor_keys[index] != first.product:
            continue
        remaining = indices[:position] + indices[position + 1 :]
        assignments.extend(
            (index,) + suffix
            for suffix in child_assignments(rest, remaining, precursor_keys)
        )
    return tuple(assignments)


def replay_step(
    frontier_sids: SidArray,
    frontier_mols: MoleculeBatch[M],
    expansions: Expansions[K],
    mapped_route: Route,
    stock: Container[K],
    *,
    match: SelectiveKeyedMatch[M, K],
) -> tuple[SidArray, MoleculeBatch[M]]:
    frontier_keys = expansions.decode(frontier_sids)
    rules = route_rules(mapped_route)

    pairs = []
    for idx, key in enumerate(frontier_keys):
        rule = rules.get(key)
        if rule is not None:
            pairs.append((idx, rule))

    if not pairs:
        # Return empty frontier if no rules match the current frontier.
        return np.zeros(0, dtype=np.int64), frontier_mols[0:0]

    precursor_mols, precursor_keys, hits = match(frontier_mols, pairs)
    precursor_sids = expansions.intern(precursor_keys)
    expansions.add_layer(frontier_sids, hits, precursor_sids)

    in_stock = np.fromiter(
        (key in stock for key in precursor_keys),
        dtype=np.bool_,
        count=len(precursor_keys),
    )
    fresh = one_per_sid(
        precursor_sids, ~expansions.is_expanded(precursor_sids) & ~in_stock
    )
    return precursor_sids[fresh], precursor_mols[fresh]


def route_rules(route: Route[K]) -> dict[K, int]:
    """Return product -> rule for a tree route, rejecting conflicting repeats."""
    rules: dict[K, int] = {}
    add_route_rules(route, rules)
    return rules


def add_route_rules(route: Route[K], rules: dict[K, int]) -> None:
    if route.rule is not None:
        seen = rules.get(route.product)
        if seen is not None and seen != route.rule:
            raise ValueError(
                "Replay routes cannot reuse one product with different rules."
            )
        rules[route.product] = route.rule
    for precursor in route.precursors:
        add_route_rules(precursor, rules)


def replay_search(
    mapped_route: Route,
    frontier_keys: Sequence[K],
    frontier_mols: MoleculeBatch[M],
    stock: Container[K],
    max_depth: int,
    expansions: Expansions[K],
    *,
    match: SelectiveKeyedMatch[M, K],
) -> Expansions[K]:
    """Fold `replay_step` over the frontier, guided by `mapped_route`.

    Operates identically to `search()`, but strictly expands molecules
    using the rule defined in `mapped_route`. Molecules not in `mapped_route`
    are treated as dead ends.
    """
    sids = expansions.intern(frontier_keys)
    in_stock = np.fromiter(
        (key in stock for key in frontier_keys),
        dtype=np.bool_,
        count=len(frontier_keys),
    )
    expansions.mark_expanded(sids[in_stock])
    frontier_sids = sids[~in_stock]
    mols = frontier_mols[~in_stock]

    for _ in range(max_depth):
        if not len(frontier_sids):
            break
        frontier_sids, mols = replay_step(
            frontier_sids, mols, expansions, mapped_route, stock, match=match
        )
    return expansions


# --------------------------------------------------------------- precursor-based replay
#
# Unlike guided/topology replay, which follow the reference rule id, precursor replay applies
# *all* library rules to the frontier and keeps only the applications whose precursor *set*
# reproduces the reference route's precursors at that product. Fanout collapses to the
# reference edges, so the search walks the reference tree using any rule library that can
# reproduce its disconnections — rule ids never have to align.


def route_precursor_sets(route: Route[K]) -> dict[K, frozenset[frozenset[K]]]:
    """Map each non-leaf product to the set(s) of its reference precursor products.

    A product that appears more than once (across a batch of routes or via a shared
    intermediate) maps to the union of its precursor sets, so any of them is accepted.
    """
    allowed: dict[K, set[frozenset[K]]] = {}
    add_route_precursor_sets(route, allowed)
    return {product: frozenset(sets) for product, sets in allowed.items()}


def add_route_precursor_sets(
    route: Route[K], allowed: dict[K, set[frozenset[K]]]
) -> None:
    if route.rule is not None:
        precursors = frozenset(child.product for child in route.precursors)
        allowed.setdefault(route.product, set()).add(precursors)
    for child in route.precursors:
        add_route_precursor_sets(child, allowed)


def matching_application_rows(
    frontier_keys: Sequence[K],
    precursor_keys: Sequence[K],
    hits: HitArray,
    allowed_sets: AllowedPrecursorSets,
    *,
    ignored: Collection[K] = frozenset(),
) -> np.ndarray:
    """Hit row indices of applications whose precursor set matches the reference.

    Groups hit rows by ``(product_idx, rule, app)``, collapses each application's precursors to
    a key set (stoichiometry-free, matching the strict/precursors comparators), and keeps every
    row of an application whose set is an allowed precursor set for that frontier product. Rows
    come back in ascending order so the caller's ``hits`` stay in their emitted order. Keys are
    read through each hit's precursor column, not by row position.

    `ignored` names keys that are not disconnections -- reagents a template spells out and a
    recorded route does not, or the reverse. They are dropped from *both* sides before comparing,
    so a rule that names its hydride reproduces a reference step that only shows the substrate.
    Without it the comparison is exact set equality, which answers "does this library draw the
    reference edge in the reference's own vocabulary"; with it, "does it draw that edge at all".
    Reagents still enter the graph as precursors, so a caller that passes `ignored` wants them in
    `stock` too or they become dead ends on the frontier.
    """
    rows_array = np.asarray(hits, dtype=np.int64).reshape(-1, 4)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for row, (product_idx, rule_id, app_idx, _precursor_idx) in enumerate(
        rows_array.tolist()
    ):
        groups.setdefault((product_idx, rule_id, app_idx), []).append(row)

    spare = frozenset(ignored)
    kept: list[int] = []
    for (product_idx, _rule_id, _app_idx), rows in groups.items():
        allowed = allowed_sets.get(frontier_keys[product_idx])
        if not allowed:
            continue
        found = frozenset(precursor_keys[rows_array[row, 3]] for row in rows)
        if found in allowed:
            kept.extend(rows)
        elif spare and (found - spare) in frozenset(one - spare for one in allowed):
            kept.extend(rows)
    kept.sort()
    return np.asarray(kept, dtype=np.int64)


def precursor_replay_step(
    frontier_sids: SidArray,
    frontier_mols: MoleculeBatch[M],
    expansions: Expansions[K],
    allowed_sets: AllowedPrecursorSets,
    stock: Container[K],
    *,
    match: Match[M, K],
    ignored: Collection[K] = frozenset(),
) -> tuple[SidArray, MoleculeBatch[M]]:
    """One precursor-filtered expansion: match all rules, keep only reference precursor sets."""
    frontier_keys = expansions.decode(frontier_sids)
    found = match(frontier_mols)
    precursor_mols, precursor_keys, hits = found.precursors, found.keys, found.hits

    kept = matching_application_rows(
        frontier_keys, precursor_keys, hits, allowed_sets, ignored=ignored
    )
    # Dropping applications orphans their precursors; compacting keeps the batch free of
    # rows no surviving hit points at, so the frontier stays on the reference tree.
    used, kept_hits = compact_precursor_index(
        np.asarray(hits, dtype=np.int64).reshape(-1, 4)[kept]
    )
    kept_keys = [precursor_keys[row] for row in used.tolist()]
    kept_mols = precursor_mols[used]

    precursor_sids = expansions.intern(kept_keys)
    # A reaction score is one value per *hit row*, so it survives this filter by the same gather
    # the hits do. Dropping it would leave a graph whose every application reads as unprobed while
    # the matcher had in fact measured all of them -- a measured replay that silently prices like
    # an unmeasured one.
    expansions.add_layer(
        frontier_sids,
        kept_hits,
        precursor_sids,
        {
            channel: np.asarray(column)[kept]
            for channel, column in found.reaction_scores.items()
        }
        or None,
    )

    in_stock = np.fromiter(
        (key in stock for key in kept_keys),
        dtype=np.bool_,
        count=len(kept_keys),
    )
    fresh = one_per_sid(
        precursor_sids, ~expansions.is_expanded(precursor_sids) & ~in_stock
    )
    return precursor_sids[fresh], kept_mols[fresh]


def precursor_replay_search(
    allowed_sets: AllowedPrecursorSets,
    frontier_keys: Sequence[K],
    frontier_mols: MoleculeBatch[M],
    stock: Container[K],
    max_depth: int,
    expansions: Expansions[K],
    *,
    match: Match[M, K],
    ignored: Collection[K] = frozenset(),
) -> Expansions[K]:
    """Fold `precursor_replay_step` over the frontier, pruned to reference precursor sets.

    Operates like `search()`, but applies every library rule and discards any application whose
    precursor set is not an `allowed_sets` entry for the product it expands. Products with no
    allowed set (off the reference tree) are dead ends, so the surviving frontier stays on the
    reference route while accepting whichever rule reproduces each disconnection.
    """
    sids = expansions.intern(frontier_keys)
    in_stock = np.fromiter(
        (key in stock for key in frontier_keys),
        dtype=np.bool_,
        count=len(frontier_keys),
    )
    expansions.mark_expanded(sids[in_stock])
    frontier_sids = sids[~in_stock]
    mols = frontier_mols[~in_stock]

    for _ in range(max_depth):
        if not len(frontier_sids):
            break
        frontier_sids, mols = precursor_replay_step(
            frontier_sids,
            mols,
            expansions,
            allowed_sets,
            stock,
            match=match,
            ignored=ignored,
        )
    return expansions

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from tests.vertsy.bench_route_replay import (
    PrecursorRouteReplayRecord,
    library_route_replayed,
    loose_route_replayed,
    merge_precursor_replay_results,
    precursor_replay_result,
    precursor_route_record,
    strict_route_replayed,
    topology_replay_routes,
)
from vertsy.routes import Route
from vertsy.search import MoleculeBatch, expansion, object_mol_array
from vesma.matcher import hit_array


@dataclass(frozen=True)
class FakeReplayBackend:
    outcomes: dict[tuple[str, int], tuple[tuple[str, ...], ...]]

    def stack(self, mols: tuple[str, ...] | list[str]) -> MoleculeBatch[str]:
        return object_mol_array(mols)

    def match(
        self, mols: MoleculeBatch[str], pairs: list[tuple[int, int]]
    ) -> tuple[MoleculeBatch[str], list[str], Any]:
        precursors: list[str] = []
        hits: list[tuple[int, int, int, int]] = []
        for mol_idx, rule in pairs:
            mol = mols[mol_idx]
            for app_idx, outcome in enumerate(self.outcomes.get((mol, rule), ())):
                for precursor in outcome:
                    hits.append((mol_idx, rule, app_idx, len(precursors)))
                    precursors.append(precursor)
        return object_mol_array(precursors), precursors, hit_array(hits)


def test_route_replay_comparison_modes_separate_intermediates_and_leaves() -> None:
    expected = Route(
        "target",
        0,
        (Route("reference-intermediate", 1, (Route("stock-a", None),)),),
    )
    same_leaves = frozenset(
        {
            Route(
                "target",
                0,
                (Route("generated-intermediate", 1, (Route("stock-a", None),)),),
            )
        }
    )
    different_leaves = frozenset(
        {
            Route(
                "target",
                0,
                (Route("generated-intermediate", 1, (Route("stock-b", None),)),),
            )
        }
    )

    assert not strict_route_replayed(expected, same_leaves)
    assert loose_route_replayed(expected, same_leaves)
    assert library_route_replayed(expected, same_leaves)

    assert not strict_route_replayed(expected, different_leaves)
    assert not loose_route_replayed(expected, different_leaves)
    assert library_route_replayed(expected, different_leaves)


def test_topology_replay_continues_generated_nonreference_intermediate() -> None:
    backend = FakeReplayBackend(
        {
            ("target", 0): (("generated-intermediate",),),
            ("generated-intermediate", 1): (("stock-a",),),
        }
    )
    expected = Route(
        "target",
        0,
        (Route("reference-intermediate", 1, (Route("stock-a", None),)),),
    )

    routes = frozenset(
        topology_replay_routes(
            backend,
            "target",
            "target",
            expected,
            frozenset({"stock-a"}),
            Counter(),
            {},
        )
    )

    assert not strict_route_replayed(expected, routes)
    assert loose_route_replayed(expected, routes)
    assert library_route_replayed(expected, routes)


def test_topology_replay_allows_extra_stock_leaf_precursor() -> None:
    backend = FakeReplayBackend(
        {
            ("target", 0): (("generated-intermediate", "extra-leaf"),),
            ("generated-intermediate", 1): (("stock-a",),),
        }
    )
    expected = Route(
        "target",
        0,
        (Route("reference-intermediate", 1, (Route("stock-a", None),)),),
    )

    routes = frozenset(
        topology_replay_routes(
            backend,
            "target",
            "target",
            expected,
            frozenset({"stock-a", "extra-leaf"}),
            Counter(),
            {},
        )
    )

    assert routes
    assert not strict_route_replayed(expected, routes)
    assert not loose_route_replayed(expected, routes)
    assert library_route_replayed(expected, routes)


def test_topology_replay_rejects_extra_nonstock_leaf_precursor() -> None:
    backend = FakeReplayBackend(
        {
            ("target", 0): (("generated-intermediate", "extra-leaf"),),
            ("generated-intermediate", 1): (("stock-a",),),
        }
    )
    expected = Route(
        "target",
        0,
        (Route("reference-intermediate", 1, (Route("stock-a", None),)),),
    )

    routes = frozenset(
        topology_replay_routes(
            backend,
            "target",
            "target",
            expected,
            frozenset({"stock-a"}),
            Counter(),
            {},
        )
    )

    assert not routes


# --------------------------------------------------------------- precursor-based replay
#
# Precursor replay applies every library rule and keeps only the applications whose precursor
# set reproduces the reference route, so it never depends on rule ids matching the reference.


@dataclass(frozen=True)
class FakePrecursorBackend:
    """All-rules matcher: ``outcomes[mol]`` is a list of ``(rule_id, precursors)`` applications.

    ``pairs=None`` (what precursor replay passes) applies every application; a pair list applies
    only the requested rules, so the same fake also drives guided replay for cross-checks.
    """

    outcomes: dict[str, tuple[tuple[int, tuple[str, ...]], ...]]

    def stack(self, mols: tuple[str, ...] | list[str]) -> MoleculeBatch[str]:
        return object_mol_array(mols)

    def match(
        self,
        mols: MoleculeBatch[str],
        pairs: list[tuple[int, int]] | None = None,
    ) -> tuple[MoleculeBatch[str], list[str], Any]:
        wanted: dict[int, set[int]] | None = None
        if pairs is not None:
            wanted = {}
            for mol_idx, rule in pairs:
                wanted.setdefault(mol_idx, set()).add(rule)
        precursors: list[str] = []
        hits: list[tuple[int, int, int, int]] = []
        for mol_idx, mol in enumerate(mols.tolist()):
            for app_idx, (rule, outcome) in enumerate(self.outcomes.get(mol, ())):
                if wanted is not None and rule not in wanted.get(mol_idx, set()):
                    continue
                for precursor in outcome:
                    hits.append((mol_idx, rule, app_idx, len(precursors)))
                    precursors.append(precursor)
        return object_mol_array(precursors), precursors, hit_array(hits)


def run_precursor_replay(
    outcomes: dict[str, tuple[tuple[int, tuple[str, ...]], ...]],
    reference: Route[str],
    stock: frozenset[str],
    *,
    with_expansions: bool = False,
):
    from vertsy.expansions import Expansions
    from vertsy.interner import DictInterner
    from vertsy.replay import precursor_replay_search, route_precursor_sets
    from vertsy.routes import iter_solved_routes, reaction_depth
    from vertsy.store.numpy_store import NumpyExpansionStore

    backend = FakePrecursorBackend(outcomes)
    expansions = Expansions(NumpyExpansionStore(), DictInterner())
    precursor_replay_search(
        route_precursor_sets(reference),
        [reference.product],
        backend.stack([reference.product]),
        stock,
        reaction_depth(reference),
        expansions,
        match=expansion(lambda mols: backend.match(mols, None)),
    )
    routes = list(iter_solved_routes(reference.product, expansions, stock))
    return (routes, expansions) if with_expansions else routes


def test_precursor_replay_result_keeps_route_records_as_source_of_truth() -> None:
    first = PrecursorRouteReplayRecord(
        route_index=1,
        target_smiles="CCO",
        reaction_count=2,
        any_solved=1,
        covered_disconnections=2,
        total_disconnections=2,
        generated_route_count=3,
        diagnostics=Counter({"matched_disconnections": 2, "generated_routes": 3}),
    )
    second = PrecursorRouteReplayRecord(
        route_index=0,
        target_smiles="CC",
        reaction_count=1,
        any_solved=0,
        covered_disconnections=0,
        total_disconnections=1,
        generated_route_count=0,
        diagnostics=Counter({"not-solved": 1}),
    )

    left = precursor_replay_result((first,), cuda_memory={})
    right = precursor_replay_result((second,), cuda_memory={})
    result = merge_precursor_replay_results((left, right))

    assert tuple(record.route_index for record in result.route_records) == (0, 1)
    assert result.case_count == 2
    assert result.reaction_count == 3
    assert result.any_solved_count == 1
    assert result.any_solved_recoveries == (0, 1)
    assert result.covered_disconnections == 2
    assert result.total_disconnections == 3
    assert result.generated_route_count == 3
    assert result.diagnostics == Counter(
        {"matched_disconnections": 2, "generated_routes": 3, "not-solved": 1}
    )
    assert precursor_route_record(result.route_records[0]) == {
        "route_index": 0,
        "target_smiles": "CC",
        "reaction_count": 1,
        "any_solved": 0,
        "covered_disconnections": 0,
        "total_disconnections": 1,
        "generated_route_count": 0,
        "diagnostics": {"not-solved": 1},
    }


def linear_reference() -> Route[str]:
    return Route("T", 0, (Route("M", 1, (Route("A", None), Route("B", None))),))


def test_route_precursor_sets_maps_products_to_child_sets() -> None:
    from vertsy.replay import route_precursor_sets

    allowed = route_precursor_sets(linear_reference())

    assert allowed == {
        "T": frozenset({frozenset({"M"})}),
        "M": frozenset({frozenset({"A", "B"})}),
    }


def test_precursor_replay_keeps_matching_and_drops_other_precursor_sets() -> None:
    # Rule ids (10, 20) differ from the reference route (0, 1): precursor replay ignores them.
    outcomes = {
        "T": (
            (10, ("M",)),
            (11, ("X",)),
        ),  # {M} reproduces the reference; {X} is dropped
        "M": ((20, ("A", "B")),),
    }

    routes = run_precursor_replay(outcomes, linear_reference(), frozenset({"A", "B"}))

    assert len(routes) == 1
    route = routes[0]
    assert (route.product, route.rule) == ("T", 10)
    assert route.precursors[0].rule == 20


def test_precursor_replay_accepts_any_rule_reproducing_the_set() -> None:
    """Rules 20 and 21 reach the same precursors, so they are one disconnection, not two routes.

    Both templates are kept — on the graph, where the question "what found this" belongs — but
    enumeration yields a single route for them, labelled with the lower id. Two spellings of one
    disconnection were never two routes; they only used to look like it.
    """
    outcomes = {
        "T": ((10, ("M",)),),
        # Two distinct rules reproduce {A, B}; a third yields {A, C} and is dropped.
        "M": ((20, ("A", "B")), (21, ("A", "B")), (22, ("A", "C"))),
    }

    routes, expansions = run_precursor_replay(
        outcomes, linear_reference(), frozenset({"A", "B"}), with_expansions=True
    )

    assert len(routes) == 1
    assert routes[0].precursors[0].rule == 20
    mid = int(expansions.lookup(["M"])[0])
    assert {step[0] for step in expansions.steps(mid)} == {(20, 21)}


def test_precursor_replay_dead_ends_when_no_rule_reproduces_the_set() -> None:
    outcomes = {"T": ((11, ("X",)),), "M": ((20, ("A", "B")),)}

    routes = run_precursor_replay(outcomes, linear_reference(), frozenset({"A", "B"}))

    assert routes == []


def test_precursor_replay_recovers_route_under_foreign_rule_ids() -> None:
    # The reference route (rules 0, 1) is recovered entirely through unrelated library rule ids,
    # the uspto42k-style case where ids never align but the disconnections still reproduce.
    outcomes = {
        "T": ((900, ("M",)), (901, ("N",))),
        "M": ((910, ("A", "B")),),
        "N": (
            (920, ("A", "B")),
        ),  # off-route product; never reached, so never expanded
    }

    routes = run_precursor_replay(outcomes, linear_reference(), frozenset({"A", "B"}))

    assert len(routes) == 1
    assert (routes[0].product, routes[0].rule) == ("T", 900)


def test_precursor_replay_superset_of_guided_recovery() -> None:
    # Precursor replay keeps any rule reproducing the reference set, so whenever guided (which
    # follows the exact reference rule) recovers a route, precursor replay must too.
    from vertsy.expansions import Expansions
    from vertsy.interner import DictInterner
    from vertsy.replay import replay_search
    from vertsy.routes import iter_solved_routes, reaction_depth
    from vertsy.store.numpy_store import NumpyExpansionStore

    reference = linear_reference()
    stock = frozenset({"A", "B"})
    outcomes = {
        "T": (
            (0, ("M",)),
            (5, ("M",)),
        ),  # rule 0 is the reference; rule 5 also reproduces {M}
        "M": ((1, ("A", "B")), (6, ("A", "B"))),  # rule 1 is the reference
    }
    backend = FakePrecursorBackend(outcomes)

    guided_expansions = replay_search(
        reference,
        [reference.product],
        backend.stack([reference.product]),
        stock,
        reaction_depth(reference),
        Expansions(NumpyExpansionStore(), DictInterner()),
        match=backend.match,
    )
    guided_solved = bool(
        list(iter_solved_routes(reference.product, guided_expansions, stock))
    )
    precursor_solved = bool(run_precursor_replay(outcomes, reference, stock))

    assert guided_solved
    assert precursor_solved

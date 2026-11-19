"""Tests for the id-native Expansions facade, its store backends, and search."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from vertsy.expansions import Expansions, layer_tables, materialize
from vertsy.interner import MISSING_SID, DictInterner
from vertsy.routes import solved_targets
from vertsy.search import expansion, object_mol_array, search, step
from vertsy.store import ExpansionStore
from vertsy.store.numpy_store import NumpyExpansionStore, resized

STORE_BACKENDS: list[type[ExpansionStore]] = [NumpyExpansionStore]

# --------------------------------------------------------------------- interner


def test_interner_mints_monotonic_sids_and_roundtrips() -> None:
    interner = DictInterner()

    assert interner.intern(["a", "b", "a", "c"]).tolist() == [0, 1, 0, 2]
    # Re-interning is idempotent.
    assert interner.intern(["c", "a"]).tolist() == [2, 0]
    assert interner.decode([0, 1, 2]) == ["a", "b", "c"]
    assert len(interner) == 3


def test_interner_lookup_is_pure() -> None:
    interner = DictInterner()
    interner.intern(["known"])
    # ``lookup`` never mints: the unknown key reports MISSING and the sid space stays 1.
    assert interner.lookup(["known", "missing"]).tolist() == [0, MISSING_SID]
    assert len(interner) == 1


# ------------------------------------------------------------- pure helpers


def test_layer_tables_gather_products_and_mint_dense_apps() -> None:
    frontier = np.array([10, 11], dtype=np.int64)  # frontier sids
    tables = layer_tables(
        frontier, [(0, 5, 0, 0), (1, 6, 0, 1)], np.array([20, 21], dtype=np.int64)
    )
    edge_product, edge_app, precursor = tables.edges
    rule_product, rule_app, rule = tables.rules
    assert edge_product.tolist() == [10, 11]  # product_idx gathered into frontier sids
    assert precursor.tolist() == [20, 21]
    # App ids are minted densely across the layer, replacing the matcher's raw
    # (product, rule)-local index, and the caller shifts them past what the store holds.
    assert edge_app.tolist() == [0, 1]
    assert (rule_product.tolist(), rule_app.tolist(), rule.tolist()) == (
        [10, 11],
        [0, 1],
        [5, 6],
    )


def test_layer_tables_validate_product_and_precursor_indices() -> None:
    frontier = np.array([10], dtype=np.int64)
    with pytest.raises(IndexError, match="outside the precursor batch"):
        layer_tables(frontier, [(0, 1, 0, 0)], np.array([], dtype=np.int64))
    with pytest.raises(IndexError, match="outside the frontier"):
        layer_tables(frontier, [(1, 1, 0, 0)], np.array([20], dtype=np.int64))


def test_layer_tables_empty_is_a_noop() -> None:
    tables = layer_tables(
        np.array([10], dtype=np.int64), [], np.array([], dtype=np.int64)
    )
    edge_product, edge_app, precursor = tables.edges
    rule_product, rule_app, rule = tables.rules
    assert all(column.tolist() == [] for column in tables.edges + tables.rules)
    assert tables.selectivity == {}


def test_resized_grows_and_is_identity_when_large_enough() -> None:
    bits = np.array([True, False], dtype=np.bool_)
    assert resized(bits, 4).tolist() == [True, False, False, False]
    assert resized(bits, 2) is bits  # no copy when already large enough


# ----------------------------------------------------------------------- facade


@pytest.fixture(params=STORE_BACKENDS, ids=lambda backend: backend.__name__)
def store(request: pytest.FixtureRequest) -> ExpansionStore:
    return request.param()


def test_add_layer_groups_edges_into_steps(store: ExpansionStore) -> None:
    expansions = Expansions(store, DictInterner())
    target = expansions.intern(["target"])
    # Seed: target has one prior step (rule 1 -> {old}); known is an evaluated leaf.
    expansions.add_layer(target, [(0, 1, 0, 0)], expansions.intern(["old"]))
    expansions.mark_expanded(expansions.intern(["known"]))

    expansions.add_layer(
        target,
        [(0, 2, 0, 0), (0, 3, 1, 1), (0, 4, 2, 2), (0, 5, 3, 3)],
        expansions.intern(["fresh", "fresh", "stock", "known"]),
    )

    view = materialize(expansions)
    assert view["target"] == frozenset(
        {
            (1, frozenset({"old"})),
            (2, frozenset({"fresh"})),
            (3, frozenset({"fresh"})),
            (4, frozenset({"stock"})),
            (5, frozenset({"known"})),
        }
    )
    assert view["known"] == frozenset()
    assert "target" in view
    assert "old" not in view  # a precursor, never expanded


def test_add_layer_collapses_identical_steps(store: ExpansionStore) -> None:
    # Two applications of rule 7 yielding the same precursor set collapse to one Step.
    expansions = Expansions(store, DictInterner())
    expansions.add_layer(
        expansions.intern(["p"]),
        [(0, 7, 0, 0), (0, 7, 1, 1)],
        expansions.intern(["x", "x"]),
    )
    assert materialize(expansions)["p"] == frozenset({(7, frozenset({"x"}))})


def test_steps_stay_distinct_across_products_sharing_raw_app_index(
    store: ExpansionStore,
) -> None:
    # Two products, each rule 0 / app 0, with different precursors. App ids are now local,
    # so both rows carry app=0; the composite (product, rule, app) keeps them apart.
    expansions = Expansions(store, DictInterner())
    pq = expansions.intern(["p", "q"])
    expansions.add_layer(
        pq, [(0, 0, 0, 0), (1, 0, 0, 1)], expansions.intern(["x", "y"])
    )
    view = materialize(expansions)
    assert view["p"] == frozenset({(0, frozenset({"x"}))})
    assert view["q"] == frozenset({(0, frozenset({"y"}))})


def test_expanded_mask_reports_evaluated_products(store: ExpansionStore) -> None:
    expansions = Expansions(store, DictInterner())
    expansions.add_layer(
        expansions.intern(["target"]), [(0, 1, 0, 0)], expansions.intern(["old"])
    )
    expansions.mark_expanded(expansions.intern(["known"]))

    ids = expansions.intern(["fresh", "known", "old"])
    # fresh: never seen; known: a leaf; old: a precursor that was never expanded.
    assert expansions.is_expanded(ids).tolist() == [False, True, False]


def test_facade_reconstructs_from_prepopulated_store(store: ExpansionStore) -> None:
    # Serialization seam: populate the store directly, then wrap a fresh facade over it
    # plus a matching interner. The expanded set (including a zero-hit leaf) is recovered
    # from the store alone -- no separate `expanded` artifact needed.
    interner = DictInterner()
    target, leaf, prec = interner.intern(["target", "leaf", "prec"]).tolist()
    store.add_edges(np.array([target]), np.array([0]), np.array([prec]))
    store.add_rules(np.array([target]), np.array([0]), np.array([1]))
    store.mark_expanded(np.array([target, leaf]))

    expansions = Expansions(store, interner)
    assert materialize(expansions) == {
        "target": frozenset({(1, frozenset({"prec"}))}),
        "leaf": frozenset(),
    }
    assert expansions.is_expanded(np.array([target, leaf, prec])).tolist() == [
        True,
        True,
        False,
    ]


# --------------------------------------------------------------------- search


def fresh_expansions() -> Expansions:
    return Expansions(NumpyExpansionStore(), DictInterner())


def _match(precursor_mols: Sequence[str], keys, hits):
    def match(_frontier):
        return object_mol_array(list(precursor_mols)), list(keys), list(hits)

    return expansion(match)


def test_search_negative_depth_is_noop() -> None:
    calls = 0

    def match(frontier_mols):
        nonlocal calls
        calls += 1
        return frontier_mols, [], []

    result = search(
        ["target"],
        object_mol_array(["target mol"]),
        set(),
        -1,
        fresh_expansions(),
        match=expansion(match),
    )
    assert materialize(result) == {}
    assert calls == 0


def test_search_records_fresh_stock_target_as_buyable_leaf() -> None:
    def fail(_):
        raise AssertionError("stock targets must not be expanded")

    result = search(
        ["stock"],
        object_mol_array(["stock mol"]),
        {"stock"},
        1,
        fresh_expansions(),
        match=fail,
    )
    assert materialize(result) == {"stock": frozenset()}


def test_search_unions_restart_steps() -> None:
    expansions = fresh_expansions()
    expansions.add_layer(
        expansions.intern(["target"]), [(0, 1, 0, 0)], expansions.intern(["old"])
    )

    result = search(
        ["target"],
        object_mol_array(["target mol"]),
        set(),
        1,
        expansions,
        match=_match(["new mol"], ["new"], [(0, 2, 0, 0)]),
    )

    assert result is expansions
    assert materialize(result) == {
        "target": frozenset({(1, frozenset({"old"})), (2, frozenset({"new"}))})
    }


def test_search_two_layers_and_solving() -> None:
    # target -> {cc, c}; cc -> {c}; c is stock. Drives two layers then solves.
    def match(frontier_mols):
        frozen = list(frontier_mols.tolist())
        if frozen == ["target mol"]:
            return (
                object_mol_array(["cc mol", "c mol"]),
                ["cc", "c"],
                [(0, 0, 0, 0), (0, 0, 0, 1)],
            )
        if frozen == ["cc mol"]:
            return object_mol_array(["c mol"]), ["c"], [(0, 1, 0, 0)]
        raise AssertionError(f"unexpected frontier {frozen}")

    result = search(
        ["target"],
        object_mol_array(["target mol"]),
        {"c"},
        3,
        fresh_expansions(),
        match=expansion(match),
    )

    # 'c' is stock seen only as a precursor: filtered from the frontier and, matching
    # the dict model, never recorded as a product key (only stock *targets* are leaves).
    assert materialize(result) == {
        "target": frozenset({(0, frozenset({"cc", "c"}))}),
        "cc": frozenset({(1, frozenset({"c"}))}),
    }
    assert solved_targets({"cc", "target"}, result, {"c"}) == {"cc", "target"}


def test_step_returns_one_representative_per_fresh_key() -> None:
    expansions = fresh_expansions()
    frontier_ids = expansions.intern(["target"])
    layer = step(
        frontier_ids,
        object_mol_array(["target mol"]),
        expansions,
        {"stock"},
        match=_match(
            ["fresh first", "fresh second", "stock mol"],
            ["fresh", "fresh", "stock"],
            [(0, 2, 0, 0), (0, 3, 1, 1), (0, 4, 2, 2)],
        ),
    )
    # Exactly the single fresh key survives; stock is filtered.
    assert layer.sids.tolist() == expansions.intern(["fresh"]).tolist()
    # Any representative molecule of the surviving key is acceptable.
    assert layer.mols.tolist()[0] in {"fresh first", "fresh second"}
    # Nobody measured anything, so there is nothing to carry.
    assert layer.estimate is None

"""The deduplicated-precursor half of the `ReactionMatcher` contract.

Hits carry an explicit precursor index, so a matcher may return fewer precursor rows
than hit rows. No live backend does that yet, so this module covers it from both ends:
hand-built hit arrays that pin the exact index arithmetic, and a real CPU match run
through `DedupingReactionMatcher` in each of its three collapse modes, which must leave
search, route extraction, and precursor replay bit-identical to the dense run.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from tests.dedup_matcher import DEDUP_MODES, DedupingReactionMatcher, DedupMode
from vertsy.expansions import Expansions, layer_tables, materialize
from vertsy.filters import refusal_wins
from vertsy.interner import DictInterner
from vertsy.replay import precursor_replay_search, route_precursor_sets
from vertsy.routes import iter_solved_routes
from vertsy.search import expansion, init, object_mol_array, search
from vertsy.selectivity import unmeasured
from vertsy.store.numpy_store import NumpyExpansionStore
from vesma.cpu_matcher import CpuReactionMatcher
from vesma.interop import cmol_from_mol
from vesma.keyer import short_ring_wl_keyer
from vesma.matcher import (
    all_vs_all,
    compact_precursor_index,
    concat_hits,
    empty_hits,
    hit_array,
    keyed_match,
)
from vesma.reaction import parse_reaction_smarts

# Near-synonymous rules: several of these disconnect the same bond, so the batch is full
# of identical precursors -- exactly the redundancy within-batch dedup would remove.
SYNONYM_SMARTS = (
    "[C:1][C:2]>>[C:1].[C:2]",
    "[C;X4:1][C:2]>>[C:1].[C:2]",
    "[C:1][C;X4:2]>>[C:1].[C:2]",
    "[C:1][O:2]>>[C:1].[O:2]",
    "[C;X4:1][O:2]>>[C:1].[O:2]",
)
TARGET = "CCOC"
STOCK_SMILES = ("C", "CC", "O", "CO", "OC")


# ------------------------------------------------------------------ array helpers


def test_empty_hits_is_four_wide() -> None:
    assert empty_hits().shape == (0, 4)
    assert hit_array([]).shape == (0, 4)


def test_hit_array_rejects_three_tuples() -> None:
    with pytest.raises(ValueError, match="4-tuples"):
        hit_array([(0, 1, 2)])  # ty: ignore[invalid-argument-type]


def test_concat_hits_rebases_each_part_precursor_column() -> None:
    first = hit_array([(0, 0, 0, 0), (0, 0, 0, 1)])
    second = hit_array([(1, 0, 0, 0), (1, 0, 0, 0)])
    # The second part's two hits share its only precursor, which lands after the first
    # part's two rows in the concatenated batch.
    assert concat_hits([first, second], [2, 1]).tolist() == [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 2],
        [1, 0, 0, 2],
    ]


def test_concat_hits_requires_one_count_per_part() -> None:
    with pytest.raises(ValueError, match="one precursor count per hit part"):
        concat_hits([empty_hits()], [])


def test_compact_precursor_index_drops_orphans_and_renumbers() -> None:
    hits = hit_array([(0, 0, 0, 3), (0, 0, 0, 1), (0, 1, 0, 3)])
    used, compacted = compact_precursor_index(hits)

    assert used.tolist() == [
        1,
        3,
    ]  # rows 0 and 2 of the batch are referenced by nothing
    assert compacted[:, 3].tolist() == [1, 0, 1]
    assert compacted[:, :3].tolist() == hits[:, :3].tolist()


# -------------------------------------------------------------------- layer edges


def test_layer_tables_collapse_synonym_rules_into_one_disconnection() -> None:
    """Two templates reaching the same precursor are one disconnection, not two.

    This is the raw68k pathology in miniature: co-canonical templates multiply applications
    without adding chemistry. Collapsing at ingest is what keeps the fold's fan-out and the
    route count proportional to the disconnections rather than to the template library.
    """
    frontier = np.array([10, 11], dtype=np.int64)
    # Four hits (two products x two synonym rules) over a two-row precursor batch.
    hits = hit_array(
        [(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 1), (1, 1, 0, 1)],
    )
    tables = layer_tables(frontier, hits, np.array([20, 21], dtype=np.int64))
    edge_product, edge_app, precursor = tables.edges
    rule_product, rule_app, rule = tables.rules

    # One edge per product, not one per hit: four applications became two disconnections.
    assert edge_product.tolist() == [10, 11]
    assert edge_app.tolist() == [0, 1]
    assert precursor.tolist() == [20, 21]
    # Both templates survive, attached to the disconnection they found.
    assert rule_product.tolist() == [10, 10, 11, 11]
    assert rule_app.tolist() == [0, 0, 1, 1]
    assert rule.tolist() == [0, 1, 0, 1]


def test_layer_tables_keep_one_template_reached_twice_once() -> None:
    """One template can hit the same disconnection through two assignments; that is one row.

    Counting it twice would put the duplication straight back into every rule-aware count.
    """
    frontier = np.array([10], dtype=np.int64)
    hits = hit_array(
        [(0, 7, 0, 0), (0, 7, 1, 0)]
    )  # same rule, two app indices, same precursor
    tables = layer_tables(frontier, hits, np.array([20], dtype=np.int64))
    edge_product, edge_app, precursor = tables.edges
    rule_product, rule_app, rule = tables.rules

    assert edge_app.tolist() == [0] and precursor.tolist() == [20]
    assert rule_app.tolist() == [0] and rule.tolist() == [7]


def test_layer_tables_keep_distinct_precursor_sets_apart() -> None:
    """Collapsing is by precursor multiset, so genuinely different disconnections stay separate."""
    frontier = np.array([10], dtype=np.int64)
    hits = hit_array([(0, 0, 0, 0), (0, 1, 0, 1)])  # same product, different precursors
    tables = layer_tables(frontier, hits, np.array([20, 21], dtype=np.int64))
    edge_product, edge_app, precursor = tables.edges
    rule_product, rule_app, rule = tables.rules

    assert edge_app.tolist() == [0, 1]
    assert precursor.tolist() == [20, 21]
    assert rule_app.tolist() == [0, 1] and rule.tolist() == [0, 1]


def test_layer_tables_reject_an_out_of_range_precursor_index() -> None:
    frontier = np.array([10], dtype=np.int64)
    with pytest.raises(IndexError, match="outside the precursor batch"):
        layer_tables(frontier, [(0, 1, 0, 1)], np.array([20], dtype=np.int64))
    with pytest.raises(IndexError, match="outside the frontier"):
        layer_tables(frontier, [(1, 1, 0, 0)], np.array([20], dtype=np.int64))


def test_add_layer_from_a_deduplicated_batch_matches_the_dense_layer() -> None:
    # The same layer written twice: once with a precursor row per hit, once with the
    # duplicates collapsed. The stored steps must be indistinguishable.
    dense = Expansions(NumpyExpansionStore(), DictInterner())
    dense.add_layer(
        dense.intern(["p"]),
        hit_array([(0, 0, 0, 0), (0, 0, 0, 1), (0, 1, 0, 2), (0, 1, 0, 3)]),
        dense.intern(["x", "y", "x", "y"]),
    )

    deduped = Expansions(NumpyExpansionStore(), DictInterner())
    deduped.add_layer(
        deduped.intern(["p"]),
        hit_array([(0, 0, 0, 0), (0, 0, 0, 1), (0, 1, 0, 0), (0, 1, 0, 1)]),
        deduped.intern(["x", "y"]),
    )

    assert materialize(deduped) == materialize(dense)
    assert materialize(deduped)["p"] == frozenset(
        {(0, frozenset({"x", "y"})), (1, frozenset({"x", "y"}))}
    )


def test_search_over_a_deduplicated_match_matches_the_dense_match() -> None:
    def dense_match(_frontier):
        return (
            object_mol_array(["a mol", "b mol", "a mol", "b mol"]),
            ["a", "b", "a", "b"],
            hit_array([(0, 0, 0, 0), (0, 0, 0, 1), (0, 1, 0, 2), (0, 1, 0, 3)]),
        )

    def deduped_match(_frontier):
        return (
            object_mol_array(["a mol", "b mol"]),
            ["a", "b"],
            hit_array([(0, 0, 0, 0), (0, 0, 0, 1), (0, 1, 0, 0), (0, 1, 0, 1)]),
        )

    def run(match):
        return materialize(
            search(
                ["target"],
                object_mol_array(["target mol"]),
                {"a", "b"},
                1,
                Expansions(NumpyExpansionStore(), DictInterner()),
                match=expansion(match),
            )
        )

    assert run(deduped_match) == run(dense_match)


# ------------------------------------------------------- real backend, all three modes


def cpu_rules():
    return tuple(parse_reaction_smarts(smarts) for smarts in SYNONYM_SMARTS)


def keyed_cpu_match(mode: DedupMode | None):
    """A keyed all-vs-all CPU match, optionally deduplicating its precursor batch."""
    codec = short_ring_wl_keyer

    def key(precursors):
        return [codec.encode(precursor) for precursor in precursors]

    matcher = CpuReactionMatcher(cpu_rules())
    if mode is not None:
        matcher = DedupingReactionMatcher(matcher, key, mode)
    return expansion(keyed_match(all_vs_all(matcher), key))


def cpu_frontier(targets):
    return init(
        targets,
        from_smiles_mol=lambda smiles: cmol_from_mol(Chem.MolFromSmiles(smiles)),
        from_smiles_key=short_ring_wl_keyer.from_smiles,
    )


def stock_keys():
    return {short_ring_wl_keyer.from_smiles(smiles) for smiles in STOCK_SMILES}


def run_search(match, max_depth: int = 2):
    target_keys, frontier = cpu_frontier([TARGET])
    expansions = search(
        target_keys,
        frontier,
        stock_keys(),
        max_depth,
        Expansions(NumpyExpansionStore(), DictInterner()),
        match=match,
    )
    return target_keys[0], expansions


def test_dedup_wrapper_actually_shortens_the_precursor_batch() -> None:
    """The premise of the conformance tests below: the modes really do differ."""
    _, frontier = cpu_frontier([TARGET])
    sizes = {}
    for mode in DEDUP_MODES:
        found = keyed_cpu_match(mode)(frontier)
        precursors, keys, hits = found.precursors, found.keys, found.hits
        assert len(keys) == len(precursors)  # keys follow the batch, not the hits
        sizes[mode] = (len(precursors), len(hits))

    hit_count = sizes["none"][1]
    assert all(count == hit_count for _, count in sizes.values())  # hits are unchanged
    assert sizes["none"][0] == hit_count  # dense: one precursor row per hit
    assert sizes["full"][0] < sizes["partial"][0] < sizes["none"][0]


@pytest.mark.parametrize("mode", DEDUP_MODES)
def test_search_and_routes_survive_deduplicated_precursors(mode: DedupMode) -> None:
    target, dense = run_search(keyed_cpu_match(None))
    _, deduped = run_search(keyed_cpu_match(mode))

    assert materialize(deduped) == materialize(dense)
    dense_routes = frozenset(iter_solved_routes(target, dense, stock_keys()))
    assert dense_routes, "the dense CPU search failed to solve the target"
    assert frozenset(iter_solved_routes(target, deduped, stock_keys())) == dense_routes


@pytest.mark.parametrize("mode", DEDUP_MODES)
def test_precursor_replay_survives_deduplicated_precursors(mode: DedupMode) -> None:
    target, dense = run_search(keyed_cpu_match(None))
    route = next(iter(sorted(iter_solved_routes(target, dense, stock_keys()), key=str)))
    allowed = route_precursor_sets(route)

    def replay(match):
        target_keys, frontier = cpu_frontier([TARGET])
        return materialize(
            precursor_replay_search(
                allowed,
                target_keys,
                frontier,
                stock_keys(),
                2,
                Expansions(NumpyExpansionStore(), DictInterner()),
                match=match,
            )
        )

    assert replay(keyed_cpu_match(mode)) == replay(keyed_cpu_match(None))


def test_a_measurement_collapses_with_the_templates_it_belongs_to() -> None:
    """One measurement per ``(app, rule)``, off the same grouping that collapses synonyms.

    Two templates disconnect one product into one precursor, so the store keeps one application
    with two templates — and the measurement has to survive that collapse *per template*, because
    a disconnection reachable selectively by one and unselectively by the other is a selective
    disconnection and the fold has to be able to see which is which.
    """
    frontier = np.array([10], dtype=np.int64)
    hits = [(0, 7, 0, 0), (0, 8, 0, 0)]
    tables = layer_tables(
        frontier,
        hits,
        np.array([20], dtype=np.int64),
        outcomes={"sites": np.array([1, 4], dtype=np.int64)},
    )
    app, rule, outcomes = tables.selectivity["sites"]

    assert tables.rules[1].tolist() == [0, 0]  # one disconnection
    assert app.tolist() == [0, 0] and rule.tolist() == [7, 8]
    assert outcomes.tolist() == [1, 4]  # per template, not per application


def test_an_unmeasured_layer_writes_no_measurement_rows() -> None:
    """The no-op path: no probe, no rows, and nothing downstream to charge for."""
    tables = layer_tables(
        np.array([10], dtype=np.int64),
        [(0, 7, 0, 0)],
        np.array([20], dtype=np.int64),
    )
    assert tables.selectivity == {}


def test_a_measurement_that_disagrees_with_itself_within_one_application_raises() -> (
    None
):
    """Every row of one group is one template on one precursor set, so they cannot differ.

    Two hit rows, one per precursor of a bimolecular disconnection, measured differently. That is
    not a chemistry question with two answers — it is one probe that reported twice — and
    averaging it into something plausible would bury the bug.
    """
    with pytest.raises(ValueError, match="constant across"):
        layer_tables(
            np.array([10], dtype=np.int64),
            [(0, 7, 0, 0), (0, 7, 0, 1)],
            np.array([20, 21], dtype=np.int64),
            outcomes={"sites": np.array([1, 3], dtype=np.int64)},
        )


def test_a_channel_resolves_its_own_disagreement_and_the_search_carries_on() -> None:
    """Who owns the invariant, and the answer is: not the store.

    The matcher found the same template reaching the same precursor twice -- two of its own
    applications -- which the store collapses into one. If the probe gave them different answers,
    two values arrive for one key. That is a defect in *that channel*, and the search it was bolted
    onto is otherwise fine, so the store asks the channel what the value is and carries on.

    A count that contradicts itself measured nothing, so `unmeasured` returns the `UNPROBED`
    sentinel -- which already means that and already costs nothing.
    """
    tables = layer_tables(
        np.array([10], dtype=np.int64),
        [(0, 7, 0, 0), (0, 7, 1, 0)],
        np.array([20], dtype=np.int64),
        outcomes={"sites": np.array([1, 3], dtype=np.int64)},
        reduce={"sites": unmeasured},
    )

    assert tables.disputed == {"sites": 1}
    # `UNPROBED` rows are not stored at all: an absent row already reads as unprobed.
    assert "sites" not in tables.selectivity or not len(tables.selectivity["sites"][0])
    # And the disconnection itself survived -- one application, one rule.
    assert len(tables.rules[0]) == 1


def test_a_verdict_resolves_the_other_way_because_a_refusal_is_definite() -> None:
    """The same mechanism, the opposite policy -- which is why neither lives in the store."""
    tables = layer_tables(
        np.array([10], dtype=np.int64),
        [(0, 7, 0, 0), (0, 7, 1, 0)],
        np.array([20], dtype=np.int64),
        outcomes={"vetoed": np.array([0, 2], dtype=np.int64)},
        reduce={"vetoed": refusal_wins},
    )

    assert tables.disputed == {"vetoed": 1}
    _apps, _rules, counts = tables.selectivity["vetoed"]
    assert counts.tolist() == [2]


def test_a_channel_that_declares_no_reduction_and_then_disagrees_raises() -> None:
    """Absence of a policy is a wiring error, not a licence to pick whichever row was written."""
    with pytest.raises(ValueError, match="declared no `Reduce`"):
        layer_tables(
            np.array([10], dtype=np.int64),
            [(0, 7, 0, 0), (0, 7, 1, 0)],
            np.array([20], dtype=np.int64),
            outcomes={"sites": np.array([1, 3], dtype=np.int64)},
        )

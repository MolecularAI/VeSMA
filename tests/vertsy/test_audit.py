"""What the audit wrappers catch, and the real molecules that motivated each one."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from vertsy.aostar import bottleneck_fold, constant_step, global_leaves
from vertsy.audit import auditing_fold, auditing_select, floored
from vertsy.expansions import expansions_of
from vertsy.metrics import TROPICAL, prepare_graph
from vertsy.search import object_mol_array
from vertsy.similarity import colliding_stock_neighbours, stock_of
from vesma.keyer import rdkit_smiles_keyer as KEYER

# Three molecules are enough to reproduce the failure that motivated all of this: a C15 alkyl
# iodide is not in this stock, and binary ECFP4 gives it byte-identical bits to the C7 that is,
# because every interior methylene of a chain shares one radius-2 environment.
STOCK = ("CCO", "c1ccccc1", "CCCCCCI")
COLLIDES = "CCCCCCCCCCCCCCCI"
DISTINCT = "c1ccccc1O"


def prepared(graph, stock):
    expansions = expansions_of(graph)
    state = prepare_graph(expansions, stock)
    root = next(iter(graph))
    return expansions, state, int(expansions.lookup([root])[0])


def leaves_of(state, count: int, value: float) -> np.ndarray:
    """``value`` for every open leaf, the semiring's zero elsewhere — what the search hands over."""
    leaves = np.full(count, TROPICAL.zero, dtype=np.float64)
    for sid in range(count):
        if sid not in state.steps and sid not in state.stock_sids:
            leaves[sid] = value
    return leaves


def test_stock_of_agrees_with_the_keyer_and_vanishes_on_stock() -> None:
    keys, heuristic = stock_of(STOCK, KEYER, weight=1.0)

    assert keys == {KEYER.from_smiles(smi) for smi in STOCK}
    scored = heuristic(object_mol_array([Chem.MolFromSmiles(smi) for smi in STOCK]))
    assert list(scored) == [0.0, 0.0, 0.0]


def test_a_non_stock_molecule_can_still_score_zero() -> None:
    # The residue unifying the two stocks cannot remove: fingerprint equality is coarser than
    # molecular identity, so `h = 0` means "in the same ECFP4 class as something buyable".
    keys, heuristic = stock_of(STOCK, KEYER, weight=1.0)
    mols = [Chem.MolFromSmiles(COLLIDES), Chem.MolFromSmiles(DISTINCT)]
    collides, distinct = heuristic(object_mol_array(mols))

    assert KEYER.from_smiles(COLLIDES) not in keys
    assert collides == pytest.approx(0.0)
    assert distinct > 0.5


def test_collisions_name_the_molecule_and_its_match() -> None:
    mols = [Chem.MolFromSmiles(COLLIDES), Chem.MolFromSmiles(DISTINCT)]
    named = colliding_stock_neighbours(mols, STOCK)

    # Only the chain collides, and it is matched against the C7 iodide at index 2.
    assert named == [(0, 2, pytest.approx(1.0))]


def test_auditing_fold_raises_when_a_stock_molecule_is_not_free() -> None:
    # Equal molecules give equal fingerprints, so a buyable molecule's distance to the nearest
    # buyable molecule is zero by construction. A positive value means the heuristic and the
    # search are reading different catalogues, which no weighting choice can produce.
    _, state, _ = prepared(
        {"target": frozenset({(0, frozenset({"stock"}))})}, {"stock"}
    )
    leaves = np.zeros(4, dtype=np.float64)
    leaves[next(iter(state.stock_sids))] = 0.25

    with pytest.raises(ValueError, match="positive estimate"):
        auditing_fold(bottleneck_fold)(
            state, leaf_value=leaves, price=constant_step(1.0), max_depth=2
        )


def test_an_unscored_stock_molecule_is_not_a_disagreement() -> None:
    # A stock molecule is interned but never handed to the heuristic, so its slot holds the
    # semiring's zero -- infinity. The first version of this check read that as a positive
    # estimate and aborted a real run on 45 perfectly ordinary buyable molecules.
    graph = {"target": frozenset({(0, frozenset({"stock"}))})}
    expansions, state, _ = prepared(graph, {"stock"})
    leaves = leaves_of(state, expansions.sid_count(), 0.25)

    for sid in state.stock_sids:
        assert leaves[sid] == TROPICAL.zero
    auditing_fold(bottleneck_fold, report=lambda _: None)(
        state, leaf_value=leaves, price=constant_step(1.0), max_depth=2
    )


def test_auditing_fold_reports_zero_off_stock_without_changing_the_value() -> None:
    graph = {"target": frozenset({(0, frozenset({"mid"}))})}
    expansions, state, root = prepared(graph, {"stock"})
    count = expansions.sid_count()
    leaves = leaves_of(state, count, 0.0)

    found: list[str] = []
    plain = bottleneck_fold(
        state, leaf_value=leaves, price=constant_step(1.0), max_depth=3
    )
    audited = auditing_fold(bottleneck_fold, report=found.append)(
        state, leaf_value=leaves, price=constant_step(1.0), max_depth=3
    )

    assert len(found) == 1
    assert "estimated at exactly 0" in found[0]
    for sid in range(count):
        for depth in range(4):
            assert audited.node(sid, depth) == plain.node(sid, depth)


def test_floored_lifts_open_leaves_and_leaves_stock_free() -> None:
    graph = {"target": frozenset({(0, frozenset({"mid"}))})}
    expansions, state, root = prepared(graph, {"stock"})
    leaves = leaves_of(state, expansions.sid_count(), 0.0)

    value = floored(bottleneck_fold, floor=0.5)(
        state, leaf_value=leaves, price=constant_step(1.0), max_depth=3
    )

    # `mid` is an open leaf clamped to the floor; the target pays a step on top of it.
    mid = int(expansions.lookup(["mid"])[0])
    assert value.node(mid, 3) == pytest.approx(0.5)
    assert value.node(root, 3) == pytest.approx(1.5)
    # Stock needs no exemption from the clamp: `inside_table` overrides it afterwards.
    for sid in state.stock_sids:
        assert value.node(sid, 3) == pytest.approx(TROPICAL.one)


def test_the_audit_must_wrap_the_floor_to_see_what_the_floor_repairs() -> None:
    graph = {"target": frozenset({(0, frozenset({"mid"}))})}
    expansions, state, _ = prepared(graph, {"stock"})
    leaves = leaves_of(state, expansions.sid_count(), 0.0)
    args = {"leaf_value": leaves, "price": constant_step(1.0), "max_depth": 3}

    outside: list[str] = []
    auditing_fold(floored(bottleneck_fold, floor=0.5), report=outside.append)(
        state, **args
    )
    inside: list[str] = []
    floored(auditing_fold(bottleneck_fold, report=inside.append), floor=0.5)(
        state, **args
    )

    # Audit outermost sees the estimates the search believes; innermost sees them repaired, and
    # so reports nothing at all -- a check that has erased its own evidence.
    assert len(outside) == 1
    assert inside == []


def test_auditing_select_reports_a_plateau_it_could_not_break() -> None:
    # Eight separate disconnections, each into one identically-estimated precursor, and room for
    # two: the ranking chose the score and sid order chose the molecules.
    names = "abcdefgh"
    graph = {
        "target": frozenset(
            (rule, frozenset({name})) for rule, name in enumerate(names)
        )
    }
    expansions, state, root = prepared(graph, set())
    leaves = leaves_of(state, expansions.sid_count(), 0.25)
    value = bottleneck_fold(
        state, leaf_value=leaves, price=constant_step(1.0), max_depth=3
    )

    found: list[str] = []
    chosen = auditing_select(global_leaves, report=found.append)(
        root, state, value, max_depth=3, limit=2, price=constant_step(1.0)
    )

    assert len(chosen) == 2
    assert len(found) == 1
    assert f"out of {len(names)} tied" in found[0]


def test_auditing_select_reports_a_conjunction_torn_by_the_width() -> None:
    # AND-siblings tie *exactly* under `by_total` -- both carry the cost of the same solution
    # graph. A tie the size of one application's arity cannot exceed any real width, so it never
    # fires in practice; where it does, the pass genuinely took part of a conjunction and left
    # the rest, which biases against finishing a group a route needs whole. Worth reporting.
    graph = {"target": frozenset({(0, frozenset({"a", "b", "c", "d"}))})}
    expansions, state, root = prepared(graph, set())
    leaves = leaves_of(state, expansions.sid_count(), 0.25)
    value = bottleneck_fold(
        state, leaf_value=leaves, price=constant_step(1.0), max_depth=3
    )

    found: list[str] = []
    chosen = auditing_select(global_leaves, report=found.append)(
        root, state, value, max_depth=3, limit=2, price=constant_step(1.0)
    )

    assert len(chosen) == 2
    assert len(found) == 1
    assert "out of 4 tied" in found[0]


def test_a_sibling_tie_cannot_exceed_a_realistic_width() -> None:
    # The arithmetic that removed the `factor` parameter: a tie of one application's arity is
    # smaller than any batch a real run uses, so the plain "more tied than taken" test is quiet
    # on a healthy conjunction without needing to be scaled.
    graph = {"target": frozenset({(0, frozenset({"a", "b", "c", "d"}))})}
    expansions, state, root = prepared(graph, set())
    leaves = leaves_of(state, expansions.sid_count(), 0.25)
    value = bottleneck_fold(
        state, leaf_value=leaves, price=constant_step(1.0), max_depth=3
    )

    found: list[str] = []
    auditing_select(global_leaves, report=found.append)(
        root, state, value, max_depth=3, limit=128, price=constant_step(1.0)
    )

    assert found == []


def test_auditing_select_stays_quiet_when_the_ranking_decided() -> None:
    graph = {
        "target": frozenset({(0, frozenset({"a"})), (1, frozenset({"b"}))}),
    }
    expansions, state, root = prepared(graph, set())
    leaves = np.full(expansions.sid_count(), TROPICAL.zero, dtype=np.float64)
    for name, estimate in (("a", 0.25), ("b", 0.75)):
        leaves[int(expansions.lookup([name])[0])] = estimate
    value = bottleneck_fold(
        state, leaf_value=leaves, price=constant_step(1.0), max_depth=3
    )

    found: list[str] = []
    chosen = auditing_select(global_leaves, report=found.append)(
        root, state, value, max_depth=3, limit=1, price=constant_step(1.0)
    )

    assert len(chosen) == 1
    assert found == []

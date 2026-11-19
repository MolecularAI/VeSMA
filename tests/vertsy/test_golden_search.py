"""Phase 0 golden snapshot tests.

Run once to generate fixtures:
    pytest tests/vertsy/test_golden_search.py -q

The golden files are created in tests/data/golden/ and should be committed.
Delete a golden file and re-run to regenerate it.
"""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import pytest

from tests.vertsy.paroutes_reference import (
    reference_route_root,
    reference_template_set,
)
from vertsy.expansions import Expansions, materialize
from vertsy.interner import DictInterner
from vertsy.interop import route_from_aizynth_dict
from vertsy.replay import replay_search
from vertsy.routes import encode_route, iter_solved_routes, leaves, reaction_depth
from vertsy.search import expansion, init, object_mol_array, search
from vertsy.store.numpy_store import NumpyExpansionStore
from vesma import cmol_from_smiles
from vesma.cpu_matcher import CpuMatcher, CpuReactionMatcher
from vesma.keyer import cmol_rdkit_smiles_keyer, short_ring_wl_keyer
from vesma.matcher import all_vs_all, keyed_match

pytest.importorskip("rdkit.Chem")

pytestmark = pytest.mark.slow

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "data" / "golden"

MAX_DEPTH = 2
MAX_ASSIGNMENTS = 256


def _save_golden(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _load_golden(path: Path) -> object:
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh)


def _maybe_create_golden(path: Path, data: object) -> bool:
    """Write golden if absent. Returns True when a new golden was just created."""
    if not path.exists():
        _save_golden(path, data)
        return True
    return False


def _build_wl_search_result(corpus, templates):
    """Run WL-keyed CPU search on the small corpus and return all intermediate values."""
    codec = short_ring_wl_keyer

    rules = tuple(t.pattern for t in templates)

    def from_smiles_mol(smi: str):
        return cmol_from_smiles(smi)[0]

    def key_fn(mols):
        return [codec.encode(m) for m in mols]

    match = expansion(
        keyed_match(
            all_vs_all(
                CpuReactionMatcher(
                    rules, matcher=CpuMatcher(), max_assignments=MAX_ASSIGNMENTS
                )
            ),
            key_fn,
        )
    )

    stock_tokens = {codec.from_smiles(smi) for smi in corpus.stock_smiles}
    targets_tokens = {codec.from_smiles(smi) for smi in corpus.target_smiles}

    frontier_keys, frontier_mols = init(
        corpus.target_smiles,
        from_smiles_mol=from_smiles_mol,
        from_smiles_key=codec.from_smiles,
    )
    expansions = search(
        frontier_keys,
        frontier_mols,
        stock_tokens,
        MAX_DEPTH,
        Expansions(NumpyExpansionStore(), DictInterner()),
        match=match,
    )
    return expansions, targets_tokens, stock_tokens, rules, codec


def test_golden_search_wl_cpu(
    small_paroutes_corpus,
    small_paroutes_templates,
) -> None:
    """Expansion graph must match the pinned WL-keyed CPU golden."""
    golden_path = GOLDEN_DIR / "search_wl_cpu.pkl.gz"

    expansions, _, _, _, _ = _build_wl_search_result(
        small_paroutes_corpus, small_paroutes_templates
    )
    graph = materialize(expansions)

    if _maybe_create_golden(golden_path, graph):
        pytest.skip("Golden snapshot created; re-run to verify")

    expected = _load_golden(golden_path)
    assert graph == expected


def test_golden_routes_wl_cpu(
    small_paroutes_corpus,
    small_paroutes_templates,
) -> None:
    """Solved routes extracted from the expansion graph must match the pinned golden."""
    golden_path = GOLDEN_DIR / "routes_wl_cpu.pkl.gz"

    expansions, targets_tokens, stock_tokens, _, _ = _build_wl_search_result(
        small_paroutes_corpus, small_paroutes_templates
    )

    # Take the first 5 targets (deterministic: sort by repr of WlToken)
    sample_targets = sorted(targets_tokens, key=repr)[:5]
    snapshot = {
        target: frozenset(iter_solved_routes(target, expansions, stock_tokens))
        for target in sample_targets
    }

    if _maybe_create_golden(golden_path, snapshot):
        pytest.skip("Golden snapshot created; re-run to verify")

    expected = _load_golden(golden_path)
    assert snapshot == expected


def test_golden_replay_rdkit_cpu(
    small_paroutes_corpus,
    small_paroutes_templates,
) -> None:
    """Restricted replay with RDKit canonical keys must match the pinned golden."""
    golden_path = GOLDEN_DIR / "replay_rdkit_cpu.pkl.gz"
    route_nodes = tuple(
        reference_route_root(row) for row in small_paroutes_corpus.reference_routes
    )
    template_set = reference_template_set(
        small_paroutes_corpus.directory,
        route_nodes,
    )
    rules = tuple(template.pattern for template in template_set.templates)

    snapshot: dict = {}
    for row in small_paroutes_corpus.reference_routes[:3]:
        target_smi = row["smiles"]
        route_node = reference_route_root(row)
        smiles_route = route_from_aizynth_dict(
            route_node,
            lambda metadata: template_set.rule_id_by_reaction_hash[
                str(metadata["reaction_hash"])
            ],
        )
        mapped_route = encode_route(smiles_route, cmol_rdkit_smiles_keyer)
        target_cmol = cmol_from_smiles(target_smi)[0]
        target_token = cmol_rdkit_smiles_keyer.encode(target_cmol)

        matcher = CpuReactionMatcher(rules, max_assignments=MAX_ASSIGNMENTS)
        stock = leaves(mapped_route)

        def match_fn(mols, pairs, matcher=matcher):
            precursors, hits = matcher.match(mols, pairs)
            return (
                object_mol_array(precursors),
                [cmol_rdkit_smiles_keyer.encode(mol) for mol in precursors],
                hits,
            )

        replayed = replay_search(
            mapped_route,
            [target_token],
            object_mol_array([target_cmol]),
            stock,
            reaction_depth(mapped_route),
            Expansions(NumpyExpansionStore(), DictInterner()),
            match=match_fn,
        )
        routes = frozenset(iter_solved_routes(target_token, replayed, stock))
        if routes:
            snapshot[target_smi] = routes

    if not snapshot:
        pytest.skip("No solved routes found for replay targets in this corpus")

    if _maybe_create_golden(golden_path, snapshot):
        pytest.skip("Golden snapshot created; re-run to verify")

    expected = _load_golden(golden_path)
    assert snapshot == expected

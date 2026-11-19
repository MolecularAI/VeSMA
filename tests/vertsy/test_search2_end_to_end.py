"""End-to-end: drive the Expansions facade search through the real CPU matcher.

WL-keyed CPU pipeline over each store backend; checks the graph recovers routes and that
every backend agrees with the numpy reference, product for product.
"""

from __future__ import annotations

import pytest

from vertsy.expansions import Expansions, materialize
from vertsy.interner import DictInterner
from vertsy.routes import iter_solved_routes, solved_targets
from vertsy.search import init, search
from vertsy.store import ExpansionStore
from vertsy.store.numpy_store import NumpyExpansionStore
from vesma import cmol_from_smiles
from vesma.cpu_matcher import CpuMatcher, CpuReactionMatcher
from vesma.keyer import short_ring_wl_keyer
from vesma.matcher import all_vs_all, keyed_match

pytest.importorskip("rdkit.Chem")

pytestmark = pytest.mark.slow

STORE_BACKENDS: list[type[ExpansionStore]] = [NumpyExpansionStore]

MAX_DEPTH = 2
MAX_ASSIGNMENTS = 256


def _wl_pipeline(corpus, templates):
    """Build the WL-keyed CPU match callable, stock/target tokens, and a frontier factory."""
    codec = short_ring_wl_keyer
    rules = tuple(t.pattern for t in templates)
    # Explicit composition (no factory): reaction matcher -> all-vs-all -> keyed.
    match = keyed_match(
        all_vs_all(
            CpuReactionMatcher(
                rules, matcher=CpuMatcher(), max_assignments=MAX_ASSIGNMENTS
            )
        ),
        lambda mols: [codec.encode(m) for m in mols],
    )

    stock = {codec.from_smiles(smi) for smi in corpus.stock_smiles}
    targets = {codec.from_smiles(smi) for smi in corpus.target_smiles}

    def frontier():
        return init(
            corpus.target_smiles,
            from_smiles_mol=lambda smi: cmol_from_smiles(smi)[0],
            from_smiles_key=codec.from_smiles,
        )

    return match, stock, targets, frontier


def _run(store, corpus, templates):
    match, stock, targets, frontier = _wl_pipeline(corpus, templates)
    keys, mols = frontier()
    expansions = search(
        keys, mols, stock, MAX_DEPTH, Expansions(store, DictInterner()), match=match
    )
    return expansions, stock, targets


@pytest.mark.parametrize("backend", STORE_BACKENDS, ids=lambda b: b.__name__)
def test_search_recovers_routes_on_each_backend(
    backend, small_paroutes_corpus, small_paroutes_templates
) -> None:
    expansions, stock, targets = _run(
        backend(), small_paroutes_corpus, small_paroutes_templates
    )

    solved = solved_targets(targets, expansions, stock)
    assert solved, "expected at least one solved target at this depth"
    # A concrete route can be extracted from the facade.
    assert tuple(iter_solved_routes(min(solved, key=repr), expansions, stock))


def test_backends_agree_product_for_product(
    small_paroutes_corpus, small_paroutes_templates
) -> None:
    reference, _, _ = _run(
        NumpyExpansionStore(), small_paroutes_corpus, small_paroutes_templates
    )
    for backend in STORE_BACKENDS:
        if backend is NumpyExpansionStore:
            continue
        other, _, _ = _run(backend(), small_paroutes_corpus, small_paroutes_templates)
        other_expansions = materialize(other)
        reference_expansions = materialize(reference)
        assert other_expansions == reference_expansions, backend.__name__

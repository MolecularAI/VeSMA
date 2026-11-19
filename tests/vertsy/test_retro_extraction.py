import pytest

from vertsy import is_solved
from vertsy.expansions import Expansions, expansions_of
from vertsy.interner import DictInterner
from vertsy.routes import Route, iter_solved_routes, pretty_string
from vertsy.search import init, search
from vertsy.store.numpy_store import NumpyExpansionStore
from vesma import cmol_from_smiles
from vesma.cpu_matcher import CpuMatcher, CpuReactionMatcher
from vesma.keyer import cmol_rdkit_smiles_keyer
from vesma.matcher import all_vs_all, keyed_match


def test_pretty_string():
    r = Route("M1", 1, (Route("M2", 2, (Route("M4", None),)), Route("M3", None)))
    expected = "M1\n  1\n    M2\n      2\n        M4\n    M3"
    assert pretty_string(r) == expected


def _int_rule_ids(graph):
    """Encode the reference corpus's string reaction ids as dense ints (the store is int-native).

    Route equivalence is structural, so any consistent encoding preserves it; this just lets
    the string-keyed reference graph cross the int64 store boundary.
    """
    rule_ids: dict[object, int] = {}
    return {
        product: frozenset(
            (rule_ids.setdefault(rule, len(rule_ids)), precursors)
            for rule, precursors in steps
        )
        for product, steps in graph.items()
    }


def test_extraction_from_predefined_expansions(small_paroutes_corpus):
    frozen_expansions = expansions_of(
        _int_rule_ids(small_paroutes_corpus.reference_expansions)
    )
    stock = set(small_paroutes_corpus.stock_smiles)
    target = small_paroutes_corpus.target_smiles[0]

    routes = frozenset(iter_solved_routes(target, frozen_expansions, stock))

    assert routes, "No solved routes found"
    assert all(is_solved(route, stock) for route in routes)


@pytest.mark.slow
def test_end_to_end_retrosynthesis(small_paroutes_corpus, small_paroutes_templates):
    codec = cmol_rdkit_smiles_keyer
    rules = tuple(t.pattern for t in small_paroutes_templates)
    match = keyed_match(
        all_vs_all(
            CpuReactionMatcher(rules, matcher=CpuMatcher(), max_assignments=100)
        ),
        lambda mols: [codec.encode(m) for m in mols],
    )

    stock_tokens = {codec.from_smiles(s) for s in small_paroutes_corpus.stock_smiles}
    targets_tokens = {codec.from_smiles(t) for t in small_paroutes_corpus.target_smiles}

    frontier_keys, frontier_mols = init(
        small_paroutes_corpus.target_smiles,
        from_smiles_mol=lambda smi: cmol_from_smiles(smi)[0],
        from_smiles_key=codec.from_smiles,
    )
    expansions = search(
        frontier_keys,
        frontier_mols,
        stock_tokens,
        2,
        Expansions(NumpyExpansionStore(), DictInterner()),
        match=match,
    )

    found_routes = sum(
        bool(tuple(iter_solved_routes(target, expansions, stock_tokens)))
        for target in targets_tokens
    )
    assert found_routes > 0, "No routes extracted for any target"

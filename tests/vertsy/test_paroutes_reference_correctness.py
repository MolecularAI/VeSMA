from __future__ import annotations

from typing import Any

import pytest

from tests.vertsy.paroutes_reference import (
    ReferenceTemplateSet,
    rdkit_precursor_sets,
    reference_route_root,
    reference_rule_id_by_hash,
    reference_template_set,
    reference_templates,
    vesma_precursor_sets,
)
from vertsy import (
    Keyer,
    cmol_rdkit_smiles_keyer,
    short_ring_wl_keyer,
)
from vertsy.expansions import Expansions
from vertsy.interner import DictInterner
from vertsy.interop import (
    aizynth_depth,
    aizynth_reaction_metadata,
    aizynth_starting_smiles,
    route_from_aizynth_dict,
    route_to_aizynth_dict,
)
from vertsy.replay import replay_search
from vertsy.routes import (
    Route,
    encode_route,
    iter_solved_routes,
    leaves,
    reaction_depth,
    route_root,
)
from vertsy.search import expansion, init, object_mol_array, search
from vertsy.store.numpy_store import NumpyExpansionStore
from vesma import CMol, cmol_from_smiles
from vesma.cpu_matcher import CpuMatcher, CpuReactionMatcher
from vesma.keyer import rdkit_mol_canonical_smiles
from vesma.matcher import all_vs_all, keyed_match
from vesma.rdkit_matcher import (
    RdkitReactionMatcher,
    rdkit_reactions_from_smarts,
)
from vesma.reaction import ReactionTemplate

pytest.importorskip("rdkit.Chem")

pytestmark = pytest.mark.slow

EXPECTED_WL_ROUTE_MISS_TARGETS: tuple[str, ...] = ()


def first_reference_route(small_paroutes_corpus) -> dict[str, Any]:
    return reference_route_root(small_paroutes_corpus.reference_routes[0])


@pytest.fixture
def small_reference_routes(small_paroutes_corpus) -> tuple[dict[str, Any], ...]:
    routes = tuple(
        reference_route_root(row) for row in small_paroutes_corpus.reference_routes
    )
    assert len(routes) == 10
    return routes


@pytest.fixture
def small_reference_template_set(
    small_paroutes_corpus,
    small_reference_routes: tuple[dict[str, Any], ...],
) -> ReferenceTemplateSet:
    return reference_template_set(
        small_paroutes_corpus.directory,
        small_reference_routes,
    )


@pytest.fixture
def deep_reference_routes(deep_paroutes_corpus) -> tuple[dict[str, Any], ...]:
    routes = tuple(
        reference_route_root(row) for row in deep_paroutes_corpus.reference_routes
    )
    assert len(routes) == 10
    return routes


@pytest.fixture
def deep_reference_template_set(
    deep_paroutes_corpus,
    deep_reference_routes: tuple[dict[str, Any], ...],
) -> ReferenceTemplateSet:
    return reference_template_set(
        deep_paroutes_corpus.directory,
        deep_reference_routes,
    )


def first_reference_smiles_route(small_paroutes_corpus) -> Route:
    route = first_reference_route(small_paroutes_corpus)
    rule_ids = reference_rule_id_by_hash(route)
    return route_from_aizynth_dict(
        route,
        lambda metadata: rule_ids[str(metadata["reaction_hash"])],
    )


def reference_smiles_route(
    route: dict[str, Any],
    template_set: ReferenceTemplateSet,
) -> Route:
    return route_from_aizynth_dict(
        route,
        lambda metadata: template_set.rule_id_by_reaction_hash[
            str(metadata["reaction_hash"])
        ],
    )


def route_matches_reference(
    candidates: frozenset[Route],
    expected: Route,
) -> bool:
    return expected in candidates


def route_reaction_steps(route: Route) -> tuple[tuple[Any, int, frozenset[Any]], ...]:
    steps: list[tuple[Any, int, frozenset[Any]]] = []
    collect_route_reaction_steps(route, steps)
    return tuple(steps)


def collect_route_reaction_steps(
    route: Route,
    steps: list[tuple[Any, int, frozenset[Any]]],
) -> None:
    if route.rule is not None:
        steps.append(
            (
                route.product,
                route.rule,
                frozenset(child.product for child in route.precursors),
            )
        )
    for child in route.precursors:
        collect_route_reaction_steps(child, steps)


def rdkit_replay_solved_routes(
    smiles_route: Route,
    target_smiles: str,
    template_set: ReferenceTemplateSet,
) -> frozenset[Route]:
    from rdkit import Chem

    target = Chem.MolFromSmiles(target_smiles)
    assert target is not None
    mapped_route = encode_route(smiles_route, cmol_rdkit_smiles_keyer)
    target_key = rdkit_mol_canonical_smiles(target)
    stock = leaves(mapped_route)
    matcher = RdkitReactionMatcher(
        rdkit_reactions_from_smarts(
            tuple(template.smarts for template in template_set.templates)
        )
    )

    def match_fn(mols, pairs):
        precursors, hits = matcher.match(mols, pairs)
        return (
            object_mol_array(precursors),
            [rdkit_mol_canonical_smiles(mol) for mol in precursors],
            hits,
        )

    expansions = replay_search(
        mapped_route,
        [target_key],
        object_mol_array([target]),
        stock,
        reaction_depth(mapped_route),
        Expansions(NumpyExpansionStore(), DictInterner()),
        match=match_fn,
    )
    return frozenset(iter_solved_routes(target_key, expansions, stock))


def vesma_replay_solved_routes(
    smiles_route: Route,
    target_smiles: str,
    template_set: ReferenceTemplateSet,
) -> frozenset[Route]:
    target = cmol_from_smiles(target_smiles)[0]
    mapped_route = encode_route(smiles_route, cmol_rdkit_smiles_keyer)
    target_key = cmol_rdkit_smiles_keyer.encode(target)
    stock = leaves(mapped_route)
    matcher = CpuReactionMatcher(
        tuple(template.pattern for template in template_set.templates),
        max_assignments=256,
    )

    def match_fn(mols, pairs):
        precursors, hits = matcher.match(mols, pairs)
        return (
            object_mol_array(precursors),
            [cmol_rdkit_smiles_keyer.encode(mol) for mol in precursors],
            hits,
        )

    expansions = replay_search(
        mapped_route,
        [target_key],
        object_mol_array([target]),
        stock,
        reaction_depth(mapped_route),
        Expansions(NumpyExpansionStore(), DictInterner()),
        match=match_fn,
    )
    return frozenset(iter_solved_routes(target_key, expansions, stock))


def test_aizynth_reference_route_reducers(small_paroutes_corpus) -> None:
    route = first_reference_route(small_paroutes_corpus)

    assert aizynth_depth(route) == 2
    assert frozenset(aizynth_starting_smiles(route)) == {
        "COCCOCCOCCO",
        "O=S(Cl)Cl",
        "OCCOCCO",
    }
    assert len(aizynth_reaction_metadata(route)) == 2


def test_aizynth_reference_route_roundtrips_through_vertsy(
    small_paroutes_corpus,
) -> None:
    route = first_reference_route(small_paroutes_corpus)
    metadata = aizynth_reaction_metadata(route)
    smiles_route = first_reference_smiles_route(small_paroutes_corpus)
    roundtrip = route_to_aizynth_dict(
        smiles_route,
        str,
        lambda rule_id: metadata[rule_id],
    )
    rule_ids = reference_rule_id_by_hash(roundtrip)

    assert (
        route_from_aizynth_dict(
            roundtrip,
            lambda item: rule_ids[str(item["reaction_hash"])],
        )
        == smiles_route
    )


def test_rdkit_replays_small_paroutes_reference_steps(small_paroutes_corpus) -> None:
    route = first_reference_smiles_route(small_paroutes_corpus)
    templates = reference_templates(
        small_paroutes_corpus.directory,
        first_reference_route(small_paroutes_corpus),
    )

    for product, rule_id, precursors in route_reaction_steps(route):
        observed = rdkit_precursor_sets(str(product), templates[rule_id].smarts)

        expected = frozenset(
            cmol_rdkit_smiles_keyer.from_smiles(str(mol)) for mol in precursors
        )

        assert expected in observed


def test_vesma_cpu_replays_small_paroutes_reference_steps(
    small_paroutes_corpus,
) -> None:
    route = first_reference_smiles_route(small_paroutes_corpus)
    templates = reference_templates(
        small_paroutes_corpus.directory,
        first_reference_route(small_paroutes_corpus),
    )

    for product, rule_id, precursors in route_reaction_steps(route):
        observed = vesma_precursor_sets(str(product), templates[rule_id].smarts)

        expected = frozenset(
            cmol_rdkit_smiles_keyer.from_smiles(str(mol)) for mol in precursors
        )

        assert expected in observed


@pytest.mark.parametrize("route_index", range(10))
def test_rdkit_replays_small_paroutes_reference_routes(
    small_reference_routes: tuple[dict[str, Any], ...],
    small_reference_template_set: ReferenceTemplateSet,
    route_index: int,
) -> None:
    route = small_reference_routes[route_index]
    smiles_route = reference_smiles_route(route, small_reference_template_set)
    expected = encode_route(smiles_route, cmol_rdkit_smiles_keyer)

    replayed = rdkit_replay_solved_routes(
        smiles_route,
        route["smiles"],
        small_reference_template_set,
    )

    assert route_matches_reference(replayed, expected)


@pytest.mark.parametrize("route_index", range(10))
def test_vesma_cpu_replays_small_paroutes_reference_routes(
    small_reference_routes: tuple[dict[str, Any], ...],
    small_reference_template_set: ReferenceTemplateSet,
    route_index: int,
) -> None:
    assert vesma_replays_reference_route(
        small_reference_routes[route_index],
        small_reference_template_set,
    )


@pytest.mark.parametrize("route_index", range(10))
def test_cpu_expansion_recovers_small_paroutes_reference_routes(
    small_paroutes_corpus,
    small_reference_routes: tuple[dict[str, Any], ...],
    small_reference_template_set: ReferenceTemplateSet,
    route_index: int,
) -> None:
    assert cpu_expansion_recovers_reference_route(
        small_reference_routes[route_index],
        small_paroutes_corpus.stock_smiles,
        small_reference_template_set,
        cmol_rdkit_smiles_keyer,
    )


def test_cpu_expansion_recovers_all_small_paroutes_reference_routes_with_wl_codec(
    small_paroutes_corpus,
    small_reference_routes: tuple[dict[str, Any], ...],
    small_reference_template_set: ReferenceTemplateSet,
) -> None:
    misses = tuple(
        route["smiles"]
        for route in small_reference_routes
        if not cpu_expansion_recovers_reference_route(
            route,
            small_paroutes_corpus.stock_smiles,
            small_reference_template_set,
            short_ring_wl_keyer,
        )
    )

    assert misses == EXPECTED_WL_ROUTE_MISS_TARGETS


def test_deep_paroutes_reference_routes_have_depth_three_or_four(
    deep_reference_routes: tuple[dict[str, Any], ...],
) -> None:
    assert {aizynth_depth(route) for route in deep_reference_routes}.issubset({3, 4})


@pytest.mark.parametrize("route_index", range(10))
def test_rdkit_replays_deep_paroutes_reference_routes(
    deep_reference_routes: tuple[dict[str, Any], ...],
    deep_reference_template_set: ReferenceTemplateSet,
    route_index: int,
) -> None:
    route = deep_reference_routes[route_index]
    smiles_route = reference_smiles_route(route, deep_reference_template_set)
    expected = encode_route(smiles_route, cmol_rdkit_smiles_keyer)

    replayed = rdkit_replay_solved_routes(
        smiles_route,
        route["smiles"],
        deep_reference_template_set,
    )

    assert route_matches_reference(replayed, expected)


@pytest.mark.parametrize("route_index", range(10))
def test_vesma_cpu_replays_deep_paroutes_reference_routes(
    deep_reference_routes: tuple[dict[str, Any], ...],
    deep_reference_template_set: ReferenceTemplateSet,
    route_index: int,
) -> None:
    route = deep_reference_routes[route_index]

    assert vesma_replays_reference_route(route, deep_reference_template_set)


@pytest.mark.parametrize("route_index", range(10))
def test_cpu_expansion_recovers_deep_paroutes_reference_routes(
    deep_paroutes_corpus,
    deep_reference_routes: tuple[dict[str, Any], ...],
    deep_reference_template_set: ReferenceTemplateSet,
    route_index: int,
) -> None:
    assert cpu_expansion_recovers_reference_route(
        deep_reference_routes[route_index],
        deep_paroutes_corpus.stock_smiles,
        deep_reference_template_set,
        cmol_rdkit_smiles_keyer,
    )


@pytest.mark.parametrize("route_index", range(10))
def test_cpu_expansion_recovers_deep_paroutes_reference_routes_with_wl_codec(
    deep_paroutes_corpus,
    deep_reference_routes: tuple[dict[str, Any], ...],
    deep_reference_template_set: ReferenceTemplateSet,
    route_index: int,
) -> None:
    assert cpu_expansion_recovers_reference_route(
        deep_reference_routes[route_index],
        deep_paroutes_corpus.stock_smiles,
        deep_reference_template_set,
        short_ring_wl_keyer,
    )


def cpu_expansion_recovers_reference_route(
    route: dict[str, Any],
    stock_smiles: frozenset[str],
    template_set: ReferenceTemplateSet,
    codec: Keyer[CMol, Any],
) -> bool:
    smiles_route = reference_smiles_route(route, template_set)
    try:
        encoded_route = encode_route(smiles_route, codec)
    except ValueError:
        return False

    graph = cpu_expansion_for_route(
        route,
        stock_smiles=stock_smiles,
        templates=template_set.templates,
        codec=codec,
    )

    target = codec.from_smiles(route["smiles"])
    stock = {codec.from_smiles(smi) for smi in stock_smiles}

    return encoded_route in frozenset(iter_solved_routes(target, graph, stock))


def test_one_way_token_route_replay_recovers_reference_molecules(
    small_paroutes_corpus,
) -> None:
    route = first_reference_route(small_paroutes_corpus)
    smiles_route = first_reference_smiles_route(small_paroutes_corpus)
    templates = reference_templates(small_paroutes_corpus.directory, route)
    codec = short_ring_wl_keyer

    mapped_route = encode_route(smiles_route, codec)
    target_token = route_root(mapped_route)
    assert isinstance(target_token, tuple) and len(target_token) == 2

    target = cmol_from_smiles(route["smiles"])[0]
    matcher = CpuReactionMatcher(
        tuple(template.pattern for template in templates),
        max_assignments=256,
    )
    stock = leaves(mapped_route)

    def match_fn(mols, pairs):
        precursors, hits = matcher.match(mols, pairs)
        return object_mol_array(precursors), [codec.encode(m) for m in precursors], hits

    expansions = replay_search(
        mapped_route,
        [target_token],
        object_mol_array([target]),
        stock,
        reaction_depth(mapped_route),
        Expansions(NumpyExpansionStore(), DictInterner()),
        match=match_fn,
    )

    assert mapped_route in frozenset(
        iter_solved_routes(target_token, expansions, stock)
    )


def vesma_replays_reference_route(
    route: dict[str, Any],
    template_set: ReferenceTemplateSet,
) -> bool:
    smiles_route = reference_smiles_route(route, template_set)
    expected = encode_route(smiles_route, cmol_rdkit_smiles_keyer)
    replayed = vesma_replay_solved_routes(smiles_route, route["smiles"], template_set)
    return route_matches_reference(replayed, expected)


def cpu_expansion_for_route(
    route: dict[str, Any],
    *,
    stock_smiles: frozenset[str],
    templates: tuple[ReactionTemplate, ...],
    codec: Keyer[CMol, Any],
) -> Expansions[Any]:
    rules = tuple(template.pattern for template in templates)
    match = expansion(
        keyed_match(
            all_vs_all(
                CpuReactionMatcher(rules, matcher=CpuMatcher(), max_assignments=256)
            ),
            lambda mols: [codec.encode(m) for m in mols],
        )
    )
    stock_keys = {codec.from_smiles(smi) for smi in stock_smiles}
    frontier_keys, frontier_mols = init(
        [route["smiles"]],
        from_smiles_mol=lambda smi: cmol_from_smiles(smi)[0],
        from_smiles_key=codec.from_smiles,
    )
    return search(
        frontier_keys,
        frontier_mols,
        stock_keys,
        aizynth_depth(route),
        Expansions(NumpyExpansionStore(), DictInterner()),
        match=match,
    )

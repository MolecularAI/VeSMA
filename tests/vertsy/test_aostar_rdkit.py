from __future__ import annotations

import pytest

pytest.importorskip("rdkit.Chem")

from rdkit import Chem

from vertsy.aostar import (
    aostar,
    closure_fold,
    constant_step,
    descent_leaves,
    global_leaves,
)
from vertsy.expansions import Expansions
from vertsy.heuristics import zero_heuristic
from vertsy.interner import DictInterner
from vertsy.routes import collapsed_route_key, iter_solved_routes, reaction_count
from vertsy.search import (
    concat_object_batches,
    expansion,
    init,
    object_mol_array,
    search,
)
from vertsy.similarity import stock_similarity
from vertsy.store.numpy_store import NumpyExpansionStore
from vertsy.tropical import tropical_valuation
from vesma.keyer import rdkit_mol_canonical_smiles, rdkit_token_from_smiles
from vesma.matcher import all_vs_all, keyed_match
from vesma.rdkit_matcher import (
    RdkitReactionMatcher,
    rdkit_reactions_from_smarts,
)

# One template that splits any carbon-carbon bond, so an alkane decomposes to stock carbon.
# Deliberately trivial chemistry: the point is to compare two searches, not to do synthesis.
SPLIT_CC = ("[C:1][C:2]>>[C:1].[C:2]",)


def build_match():
    matcher = RdkitReactionMatcher(rdkit_reactions_from_smarts(SPLIT_CC))
    return expansion(
        keyed_match(
            all_vs_all(matcher),
            lambda mols: [rdkit_mol_canonical_smiles(mol) for mol in mols],
        )
    )


def fresh_expansions() -> Expansions:
    return Expansions(NumpyExpansionStore(), DictInterner())


def exhaustive_routes(smiles: str, stock: frozenset[str], max_depth: int):
    keys, mols = init(
        [smiles],
        from_smiles_mol=Chem.MolFromSmiles,
        from_smiles_key=rdkit_token_from_smiles,
    )
    expansions = search(
        keys, mols, stock, max_depth, fresh_expansions(), match=build_match()
    )
    target = rdkit_token_from_smiles(smiles)
    return list(iter_solved_routes(target, expansions, stock))


def heuristic_routes(
    smiles: str,
    stock: frozenset[str],
    max_depth: int,
    heuristic,
    select=global_leaves,
    width: int = 32,
    fold=tropical_valuation,
    price=constant_step(1.0),
):
    # `passes` is set from the budget rather than from `max_depth`, and `max_depth` is passed
    # explicitly: the descent arm and `width=1` open one leaf per pass, so a horizon-sized
    # `passes` would give them a budget of four expansions and nothing to compare.
    return aostar(
        rdkit_token_from_smiles(smiles),
        target_mol=Chem.MolFromSmiles(smiles),
        stock=stock,
        expansions=fresh_expansions(),
        heuristic=heuristic,
        match=build_match(),
        stack_mols=object_mol_array,
        concat_mols=concat_object_batches,
        select=select,
        fold=fold,
        price=price,
        width=width,
        passes=max(200 // width, max_depth),
        max_depth=max_depth,
        max_routes=50,
    )


@pytest.mark.parametrize("smiles", ["CC", "CCC", "CCCC"])
@pytest.mark.parametrize(
    "select", [global_leaves, descent_leaves], ids=["global", "descent"]
)
@pytest.mark.parametrize("width", [1, 32])
@pytest.mark.parametrize(
    "fold", [tropical_valuation, closure_fold], ids=["array", "closure"]
)
def test_aostar_agrees_with_exhaustive_search(
    smiles: str, select, width: int, fold
) -> None:
    """AO* is a way of not building the whole graph, not a different chemistry.

    Same templates, same stock, same keyer as the exhaustive fold, so anything AO* returns
    has to be a route the fold would also have found, and its best has to be the fold's best.
    If these ever disagree the heuristic is not pruning, it is inventing.
    """
    carbon = rdkit_token_from_smiles("C")
    stock = frozenset({carbon})

    exhaustive = exhaustive_routes(smiles, stock, max_depth=4)
    guided = heuristic_routes(
        smiles,
        stock,
        max_depth=4,
        heuristic=zero_heuristic,
        select=select,
        width=width,
        fold=fold,
    )

    assert exhaustive, "fixture must be solvable exhaustively"
    assert guided, "AO* must solve what exhaustive search solves"

    # Every route AO* reports is one exhaustive search also found.
    known = {collapsed_route_key(route) for route in exhaustive}
    assert {collapsed_route_key(route) for route in guided} <= known

    # And the cheapest route is the same cost: an admissible estimate does not lose the optimum.
    assert min(reaction_count(route) for route in guided) == min(
        reaction_count(route) for route in exhaustive
    )


def test_admissible_similarity_keeps_the_optimum() -> None:
    """weight <= 1 never overestimates a reaction, so the cheapest route survives guiding."""
    carbon = rdkit_token_from_smiles("C")
    stock = frozenset({carbon})
    similarity = stock_similarity(["C"], weight=1.0)

    guided = heuristic_routes("CCCC", stock, max_depth=4, heuristic=similarity)
    exhaustive = exhaustive_routes("CCCC", stock, max_depth=4)

    assert guided
    assert min(reaction_count(route) for route in guided) == min(
        reaction_count(route) for route in exhaustive
    )


# There is deliberately no "AO* prunes more than exhaustive search" test here. This
# chemistry cannot show it: with a single bond-splitting rule every fragment of an alkane
# has to be decomposed anyway, so there are no cheaper alternatives to prefer and both
# searches expand the same 4 molecules — the true minimum. Pruning is asserted where it can
# actually fail: `test_aostar.py` pins it exactly on a graph with a cheap and an expensive
# route (the expensive branch is never opened), and `bench_aostar.py` measures it on n1.

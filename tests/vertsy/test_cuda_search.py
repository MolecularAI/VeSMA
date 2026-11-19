"""CUDA driven through the generic `search()` — parity with the CPU backend.

The device-resident `CudaMoleculeBatch` frontier flows through the *same* `search()`
loop as CPU/rdkit (via `CudaMoleculeBatch.__getitem__` + `cuda_stack_mols`), so the
resulting `Expansions` must be byte-identical to the CPU backend.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from vertsy.expansions import Expansions, materialize
from vertsy.interner import DictInterner
from vertsy.routes import iter_solved_routes, solved_targets
from vertsy.search import expansion, init, search
from vertsy.store.numpy_store import NumpyExpansionStore
from vesma import cmol_from_smiles
from vesma.cpu_matcher import CpuReactionMatcher
from vesma.cuda_matcher import (
    CudaFoldedPrefilteredReactionMatcher,
    CudaReactionMatcher,
    cuda_stack_mols,
)
from vesma.keyer import short_ring_wl_keyer
from vesma.matcher import all_vs_all, keyed_match
from vesma.reaction import ReactionPattern, reaction_patterns_from_smarts


def _skip_if_no_cuda() -> None:
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CuPy cannot access CUDA on this machine.")


RULES: dict[int, str] = {
    0: "[C:1][C:2]>>[C:1].[C:2]",
    1: "[C:1][O:2]>>[C:1].[O:2]",
    2: "[C:1][N:2]>>[C:1].[N:2]",
}
TARGETS: tuple[str, ...] = ("OCC(N)CO", "CCCC", "CCO", "CC(N)CO")
STOCK: tuple[str, ...] = ("C", "O", "N", "CO", "CN", "CC")


@pytest.fixture(scope="module")
def compiled_rules() -> Sequence[ReactionPattern]:
    _skip_if_no_cuda()
    return reaction_patterns_from_smarts(
        tuple(RULES.values()),
        n_jobs=1,
        cache=False,
    )


def _rules_from_smarts(smarts: tuple[str, ...]) -> Sequence[ReactionPattern]:
    return reaction_patterns_from_smarts(smarts, n_jobs=1, cache=False)


def _cpu_search(targets, stock_keys, max_depth, cpu_matcher, codec):
    frontier_keys, frontier_mols = init(
        targets,
        from_smiles_mol=lambda smi: cmol_from_smiles(smi)[0],
        from_smiles_key=codec.from_smiles,
    )
    return search(
        frontier_keys,
        frontier_mols,
        stock_keys,
        max_depth,
        Expansions(NumpyExpansionStore(), DictInterner()),
        match=expansion(
            keyed_match(
                all_vs_all(cpu_matcher),
                lambda mols: [codec.encode(m) for m in mols],
            )
        ),
    )


def _cuda_search(targets, stock_keys, max_depth, *, matcher, codec):
    """Local CUDA driver, mirroring `_cpu_search`: device frontier + the generic search.

    The only CUDA-specific pieces are `cuda_stack_mols` (device-resident frontier) and
    the matcher's `match_keyed`; everything else is the shared `search` fold.
    """
    frontier_keys, frontier_mols = init(
        targets,
        from_smiles_mol=lambda smi: cmol_from_smiles(smi)[0],
        from_smiles_key=codec.from_smiles,
        stack_mols=cuda_stack_mols,
    )
    return search(
        frontier_keys,
        frontier_mols,
        stock_keys,
        max_depth,
        Expansions(NumpyExpansionStore(), DictInterner()),
        match=expansion(matcher.match_keyed),
    )


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_search_splits_ethane_to_stock_carbon() -> None:
    """Exact one-step golden: ethane -> {carbon} via the generic CUDA search."""
    _skip_if_no_cuda()
    codec = short_ring_wl_keyer
    carbon = codec.from_smiles("C")
    target = codec.from_smiles("CC")
    expansions = _cuda_search(
        ("CC",),
        {carbon},
        1,
        matcher=CudaReactionMatcher(
            _rules_from_smarts(("[C:1][C:2]>>[C:1].[C:2]",)),
        ),
        codec=codec,
    )
    assert materialize(expansions) == {target: frozenset({(0, frozenset({carbon}))})}


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_search_solves_two_step_route() -> None:
    """A 2-layer CUDA search recovers a propane -> carbon route."""
    _skip_if_no_cuda()
    codec = short_ring_wl_keyer
    carbon = codec.from_smiles("C")
    target = codec.from_smiles("CCC")
    expansions = _cuda_search(
        ("CCC",),
        {carbon},
        2,
        matcher=CudaReactionMatcher(
            _rules_from_smarts(("[C:1][C:2]>>[C:1].[C:2]",)),
        ),
        codec=codec,
    )
    assert len(materialize(expansions)) == 2
    assert tuple(iter_solved_routes(target, expansions, {carbon}))


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize("max_depth", [1, 2, 3])
@pytest.mark.parametrize("frontier_mode", ["device", "host"])
def test_cuda_search_matches_cpu_search(
    max_depth: int,
    frontier_mode: str,
    compiled_rules: Sequence[ReactionPattern],
) -> None:
    """Multi-step CUDA search via the generic loop == CPU search (bit-exact)."""
    codec = short_ring_wl_keyer
    stock_keys = {codec.from_smiles(smi) for smi in STOCK}

    cpu_expansions = _cpu_search(
        TARGETS,
        stock_keys,
        max_depth,
        CpuReactionMatcher(compiled_rules, max_assignments=64),
        codec,
    )
    cuda_expansions = _cuda_search(
        TARGETS,
        stock_keys,
        max_depth,
        matcher=CudaReactionMatcher(
            compiled_rules,
            frontier_mode=frontier_mode,
        ),
        codec=codec,
    )
    assert materialize(cuda_expansions) == materialize(cpu_expansions)


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_search_chunked_matches_cpu_search(
    compiled_rules: Sequence[ReactionPattern],
) -> None:
    """Tiny chunk sizes don't change the result of a multi-step search."""
    codec = short_ring_wl_keyer
    stock_keys = {codec.from_smiles(smi) for smi in STOCK}

    cpu_expansions = _cpu_search(
        TARGETS,
        stock_keys,
        3,
        CpuReactionMatcher(compiled_rules, max_assignments=64),
        codec,
    )
    cuda_expansions = _cuda_search(
        TARGETS,
        stock_keys,
        3,
        matcher=CudaReactionMatcher(
            compiled_rules,
            max_pairs_per_launch=1,
        ),
        codec=codec,
    )
    assert materialize(cuda_expansions) == materialize(cpu_expansions)


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_folded_prefilter_search_matches_cpu_search(
    compiled_rules: Sequence[ReactionPattern],
) -> None:
    codec = short_ring_wl_keyer
    stock_keys = {codec.from_smiles(smi) for smi in STOCK}

    cpu_expansions = _cpu_search(
        TARGETS,
        stock_keys,
        3,
        CpuReactionMatcher(compiled_rules, max_assignments=64),
        codec,
    )
    cuda_expansions = _cuda_search(
        TARGETS,
        stock_keys,
        3,
        matcher=CudaFoldedPrefilteredReactionMatcher(
            CudaReactionMatcher(compiled_rules)
        ),
        codec=codec,
    )
    assert materialize(cuda_expansions) == materialize(cpu_expansions)


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_search_one_layer_matches_cpu_on_small_paroutes(
    small_paroutes_corpus,
    small_paroutes_templates,
) -> None:
    """One layer of chunked CUDA search reproduces the CPU search exactly on the small
    PaRoutes reference subset. (Only the first layer is asserted bit-exact: at deeper
    layers CUDA re-matches its own GPU-perceived precursors, whose aromaticity is a
    faithful-but-not-identical port of the CPU perception — short-ring WL keys still
    agree, but SMARTS matching can diverge on aromatic intermediates. Route recovery is
    checked separately below.)"""
    _skip_if_no_cuda()
    codec = short_ring_wl_keyer
    rules = tuple(template.pattern for template in small_paroutes_templates)
    targets = tuple(small_paroutes_corpus.target_smiles)
    stock_keys = {codec.from_smiles(smi) for smi in small_paroutes_corpus.stock_smiles}

    cpu_expansions = _cpu_search(
        targets, stock_keys, 1, CpuReactionMatcher(rules, max_assignments=128), codec
    )
    cuda_expansions = _cuda_search(
        targets,
        stock_keys,
        1,
        matcher=CudaReactionMatcher(
            rules,
            max_leaf_checks=8192,
            max_pairs_per_launch=512,
        ),
        codec=codec,
    )
    assert materialize(cuda_expansions) == materialize(cpu_expansions)


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_search_recovers_routes_on_small_paroutes(
    small_paroutes_corpus,
    small_paroutes_templates,
) -> None:
    """A 2-layer chunked CUDA search via the generic loop recovers reference routes."""
    _skip_if_no_cuda()
    codec = short_ring_wl_keyer
    rules = tuple(template.pattern for template in small_paroutes_templates)
    targets = tuple(small_paroutes_corpus.target_smiles)
    stock_keys = {codec.from_smiles(smi) for smi in small_paroutes_corpus.stock_smiles}

    expansions = _cuda_search(
        targets,
        stock_keys,
        2,
        matcher=CudaReactionMatcher(
            rules,
            max_leaf_checks=8192,
            max_pairs_per_launch=512,
        ),
        codec=codec,
    )

    target_keys = {codec.from_smiles(smi) for smi in targets}
    assert solved_targets(target_keys, expansions, stock_keys)

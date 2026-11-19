from __future__ import annotations

import numpy as np
import pytest

from vertsy.aostar import anytime_step, aostar, bottleneck_fold
from vertsy.cuda_similarity import (
    cuda_stock_similarity,
    cuda_stock_similarity_from_fingerprints,
)
from vertsy.expansions import Expansions
from vertsy.interner import DictInterner
from vertsy.search import expansion
from vertsy.store.numpy_store import NumpyExpansionStore
from vesma import cmol_from_smiles
from vesma.circularfp import circular_fingerprint_words
from vesma.cmol_builder import gmol_from_cmol
from vesma.cuda_matcher import (
    CudaFoldedPrefilteredReactionMatcher,
    CudaReactionMatcher,
    cuda_concat_host_mols,
    cuda_stack_mols,
)
from vesma.keyer import short_ring_wl_keyer
from vesma.reaction import reaction_patterns_from_smarts


def skip_if_no_cuda() -> None:
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CuPy cannot access CUDA on this machine.")


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_stock_similarity_keeps_stock_on_device_and_scores_batches() -> None:
    skip_if_no_cuda()
    stock = [cmol_from_smiles(smiles)[0] for smiles in ("C", "CC", "CO")]
    heuristic = cuda_stock_similarity(stock, weight=1.0, batch_size=2)
    queries = cuda_stack_mols(
        [cmol_from_smiles(smiles)[0] for smiles in ("C", "CCC", "CO")]
    )

    scores = heuristic(queries)

    assert scores.dtype == np.float32
    assert scores.shape == (3,)
    assert scores[[0, 2]].tolist() == [0.0, 0.0]
    assert 0.0 < scores[1] <= 1.0
    assert heuristic.nbytes == 3 * 2048 // 8


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_stock_similarity_uploads_cached_fingerprints() -> None:
    skip_if_no_cuda()
    molecules = [cmol_from_smiles(smiles)[0] for smiles in ("C", "CC", "CO")]
    fingerprints = np.stack(
        [circular_fingerprint_words(gmol_from_cmol(molecule)) for molecule in molecules]
    )
    heuristic = cuda_stock_similarity_from_fingerprints(
        fingerprints,
        weight=1.0,
    )

    scores = heuristic(cuda_stack_mols(molecules))

    assert scores.tolist() == [0.0, 0.0, 0.0]
    assert heuristic.nbytes == fingerprints.nbytes


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_stock_similarity_guides_hybrid_frontier_aostar() -> None:
    skip_if_no_cuda()
    carbon = cmol_from_smiles("C")[0]
    heuristic = cuda_stock_similarity([carbon], weight=1.0)
    matcher = CudaFoldedPrefilteredReactionMatcher(
        CudaReactionMatcher(
            reaction_patterns_from_smarts(("[C:1][C:2]>>[C:1].[C:2]",), cache=False),
            frontier_mode="device",
        )
    )
    price, select = anytime_step(greedy=0.001, relaxed=1.0)

    routes = aostar(
        short_ring_wl_keyer.from_smiles("CCC"),
        target_mol=cmol_from_smiles("CCC")[0],
        stock=frozenset({short_ring_wl_keyer.from_smiles("C")}),
        expansions=Expansions(NumpyExpansionStore(), DictInterner()),
        heuristic=heuristic,
        match=expansion(matcher.match_keyed),
        stack_mols=cuda_stack_mols,
        concat_mols=cuda_concat_host_mols,
        width=2,
        passes=4,
        max_depth=4,
        select=select,
        fold=bottleneck_fold,
        price=price,
    )

    assert routes

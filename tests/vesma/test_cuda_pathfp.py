from __future__ import annotations

import numpy as np
import pytest

from vesma import cmol_from_smiles, gmol_from_cmol, gmol_from_smarts
from vesma.cuda_pathfp import compute_folded_path_fingerprints_cuda
from vesma.cuda_retro import prepare_cuda_molecule_batch
from vesma.pathfp import FoldedFingerprintConfig, folded_path_fingerprint_words


def skip_if_no_cuda() -> None:
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CuPy cannot access CUDA on this machine.")


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize(
    "smarts,query",
    [
        ("OCC(N)CO", False),
        ("C1CCCCC1", False),
        ("C1CCCCCCCC1", False),
        ("c1ccccc1", False),
        ("C[N,O]", True),
        ("C~C", True),
        ("C1~CC~CC~C1", True),
    ],
)
def test_cuda_folded_path_fingerprint_matches_cpu(
    smarts: str,
    query: bool,
) -> None:
    skip_if_no_cuda()
    graph = (
        gmol_from_smarts(smarts)[0]
        if query
        else gmol_from_cmol(cmol_from_smiles(smarts)[0])
    )
    batch = prepare_cuda_molecule_batch([((0, 0), graph)])
    config = FoldedFingerprintConfig(n_bits=2048, max_length=7)

    observed = compute_folded_path_fingerprints_cuda(
        batch,
        query=query,
        config=config,
    ).get()[0]
    expected = folded_path_fingerprint_words(graph, query=query, config=config)

    assert np.array_equal(observed, expected)

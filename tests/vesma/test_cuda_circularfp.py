from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from vesma import cmol_from_smiles, gmol_from_cmol
from vesma.circularfp import CircularFingerprintConfig, circular_fingerprint_words
from vesma.cuda_circularfp import (
    compute_circular_fingerprints_cuda,
    nearest_stock_distance_cuda,
)
from vesma.cuda_matcher import (
    CudaMoleculeHostBatch,
    cuda_concat_host_mols,
    cuda_concat_mols,
    cuda_stack_mols,
)


def skip_if_no_cuda() -> None:
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CuPy cannot access CUDA on this machine.")


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize(
    "smiles",
    [
        "C",
        "CCO",
        "CC(C)C",
        "c1ccccc1",
        "C1CCCCC1",
        "C[NH3+]",
        "O=C([O-])C",
    ],
)
def test_cuda_circular_fingerprint_matches_cpu(smiles: str) -> None:
    skip_if_no_cuda()
    molecule = cmol_from_smiles(smiles)[0]
    batch = cuda_stack_mols([molecule])

    observed = compute_circular_fingerprints_cuda(batch).get()[0]
    expected = circular_fingerprint_words(gmol_from_cmol(molecule))

    assert np.array_equal(observed, expected)


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_circular_fingerprint_preserves_batch_axis_and_empty_batch() -> None:
    skip_if_no_cuda()
    molecules = [cmol_from_smiles(smiles)[0] for smiles in ("C", "CC", "CCC")]
    batch = cuda_stack_mols(molecules)
    config = CircularFingerprintConfig(n_bits=256, radius=1)

    observed = compute_circular_fingerprints_cuda(batch, config).get()
    empty = compute_circular_fingerprints_cuda(batch[:0], config).get()

    assert observed.shape == (3, 4)
    assert empty.shape == (0, 4)


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_nearest_stock_distance_matches_cpu_popcount() -> None:
    skip_if_no_cuda()
    stock_molecules = [cmol_from_smiles(s)[0] for s in ("C", "CC", "CO", "c1ccccc1")]
    query_molecules = [cmol_from_smiles(s)[0] for s in ("C", "CCC", "c1ccccc1")]
    stock = compute_circular_fingerprints_cuda(cuda_stack_mols(stock_molecules))
    query = compute_circular_fingerprints_cuda(cuda_stack_mols(query_molecules))

    observed = nearest_stock_distance_cuda(query, stock, weight=0.75).get()
    stock_host = stock.get()
    query_host = query.get()
    expected = []
    for row in query_host:
        similarities = []
        for candidate in stock_host:
            intersection = int(np.bitwise_count(row & candidate).sum())
            union = int(np.bitwise_count(row | candidate).sum())
            similarities.append(intersection / union if union else 0.0)
        expected.append(0.75 * (1.0 - max(similarities)))

    assert np.allclose(observed, expected, rtol=1e-6)
    assert observed[0] == pytest.approx(0.0)
    assert observed[2] == pytest.approx(0.0)


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize("invalid_dtype", ["uint32", "float64"])
def test_cuda_nearest_stock_distance_rejects_non_uint64(
    invalid_dtype: str,
) -> None:
    skip_if_no_cuda()
    cp = pytest.importorskip("cupy")
    valid = cp.zeros((1, 4), dtype=cp.uint64)
    invalid = cp.zeros((1, 4), dtype=getattr(cp, invalid_dtype))

    with pytest.raises(ValueError, match="dtype uint64"):
        nearest_stock_distance_cuda(invalid, valid, weight=1.0)
    with pytest.raises(ValueError, match="dtype uint64"):
        nearest_stock_distance_cuda(valid, invalid, weight=1.0)


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_concat_mols_always_returns_a_batch() -> None:
    skip_if_no_cuda()
    left = cuda_stack_mols([cmol_from_smiles("C")[0]])
    right = cuda_stack_mols([cmol_from_smiles("CC")[0]])

    combined = cuda_concat_mols([left, right])
    empty = cuda_concat_mols([left[:0], right[:0]])

    assert len(combined) == 2
    assert len(empty) == 0


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_concat_host_mols_accepts_device_and_host_parts() -> None:
    skip_if_no_cuda()
    left = cuda_stack_mols([cmol_from_smiles("C")[0]])
    right = cuda_stack_mols([cmol_from_smiles("CC")[0]])

    first = cuda_concat_host_mols([left])
    combined = cuda_concat_host_mols([first, right])

    assert isinstance(combined, CudaMoleculeHostBatch)
    assert len(combined) == 2
    assert len(combined.to_cuda()) == 2


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_concat_mols_preserves_mixed_tokenless_batches() -> None:
    skip_if_no_cuda()
    keyed = cuda_stack_mols([cmol_from_smiles("C")[0]])
    tokenless = replace(
        cuda_stack_mols([cmol_from_smiles("CC")[0]]),
        tokens=(),
    )

    combined = cuda_concat_mols([keyed, tokenless])

    assert len(combined) == 2
    assert combined.atom_counts_device.shape[0] == 2
    assert combined.tokens == ()


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_concat_host_mols_selects_tokenless_batches() -> None:
    skip_if_no_cuda()
    tokenless = replace(
        cuda_stack_mols([cmol_from_smiles("C")[0], cmol_from_smiles("CC")[0]]),
        tokens=(),
    )

    combined = cuda_concat_host_mols([tokenless])
    selected = combined[np.array([1], dtype=np.int64)]

    assert len(selected) == 1
    assert len(selected.to_cuda()) == 1
    assert selected.parts[0].keys == ()

"""Swapping the scan onto the GPU must change the running time and nothing else.

:mod:`vertsy.metal_similarity` exists for speed — 94x against a catalogue-scale stock — and its whole
correctness claim is that it is invisible: the same heuristic classes, the same estimates, the same
admissibility properties :mod:`tests.vertsy.test_host_similarity` pins on the host path. So the tests
here are agreement tests against that path rather than a second set of property tests, because a
device scan that answered *nearly* the same would be the hard failure: AO* would rank precursors one
way on one backend and another way on the other, and it would surface as route quality rather than as
an error.

The one thing that is genuinely new is the resident catalogue, which introduces state a pure function
did not have. Two heuristics alive at once is the case that state can get wrong, so it is tested
directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from rdkit import Chem

from vertsy.host_similarity import (
    ecfp4_stock,
    packed_ecfp4_words,
    packed_stock,
)
from vertsy.metal_similarity import (
    MetalNearestScan,
    metal_ecfp4_stock,
    metal_packed_stock,
    metal_scan,
    metal_similarity_available,
)
from vertsy.search import object_mol_array
from vesma.circularfp import CircularFingerprintConfig

from .test_host_similarity import (
    CONFIG,
    QUERIES,
    STOCK,
    PackedFrontier,
    vesma_stock_words,
)

pytestmark = [pytest.mark.gpu, pytest.mark.metal]


def skip_if_no_metal_device() -> None:
    if not metal_similarity_available():
        pytest.skip("Metal backend not available")


def rdkit_stock_words(smiles: tuple[str, ...]) -> np.ndarray:
    return packed_ecfp4_words([Chem.MolFromSmiles(smi) for smi in smiles])


def rdkit_batch(smiles: tuple[str, ...]) -> Any:
    return object_mol_array([Chem.MolFromSmiles(smi) for smi in smiles])


# --------------------------------------------------------------------------- agreement


@pytest.mark.parametrize("weight", [0.0, 1.0, 3.5])
def test_the_packed_heuristic_scores_identically_on_the_gpu(weight: float) -> None:
    """Identically and not approximately — see the module docstring for why that is the bar."""
    skip_if_no_metal_device()
    stock = vesma_stock_words(STOCK)
    frontier = PackedFrontier(QUERIES)
    assert np.array_equal(
        metal_packed_stock(stock, weight=weight, config=CONFIG)(frontier),
        packed_stock(stock, weight=weight, config=CONFIG)(frontier),
    )


def test_the_ecfp4_heuristic_scores_identically_on_the_gpu() -> None:
    skip_if_no_metal_device()
    stock = rdkit_stock_words(STOCK)
    frontier = rdkit_batch(QUERIES)
    assert np.array_equal(
        metal_ecfp4_stock(stock, weight=2.0)(frontier),
        ecfp4_stock(stock, weight=2.0)(frontier),
    )


def test_the_estimate_still_vanishes_on_the_stock() -> None:
    """The property AO* depends on, re-checked on the backend that actually computes it."""
    skip_if_no_metal_device()
    heuristic = metal_packed_stock(vesma_stock_words(STOCK), weight=5.0, config=CONFIG)
    assert heuristic(PackedFrontier(STOCK)) == pytest.approx(np.zeros(len(STOCK)))


def test_an_empty_frontier_never_reaches_the_device() -> None:
    """A pass that discovered nothing new must not be a launch, let alone a failed one."""
    skip_if_no_metal_device()
    heuristic = metal_packed_stock(vesma_stock_words(STOCK), weight=1.0, config=CONFIG)
    assert heuristic(PackedFrontier(())).shape == (0,)


def test_a_narrower_fingerprint_agrees_too() -> None:
    """512 bits is 8 uint64 words, so it exercises a different tile than the 2048-bit default."""
    skip_if_no_metal_device()
    config = CircularFingerprintConfig(n_bits=512, radius=2)
    stock = np.stack(
        [words[:8] for words in vesma_stock_words(STOCK)]
    )  # not a real 512-bit fingerprint, but a valid packed stock of that width
    frontier = PackedFrontier(QUERIES)
    assert np.array_equal(
        metal_packed_stock(stock, weight=1.0, config=config)(frontier),
        packed_stock(stock, weight=1.0, config=config)(frontier),
    )


# --------------------------------------------------------------------------- resident catalogue


def test_two_heuristics_over_different_stocks_do_not_share_a_catalogue() -> None:
    """The failure the token exists to prevent, exercised the way a comparison run would hit it.

    Interleaved rather than sequential, because a cache that simply held the last catalogue would
    pass a sequential test and answer the previous stock's distances here.
    """
    skip_if_no_metal_device()
    first_stock = vesma_stock_words(STOCK)
    second_stock = vesma_stock_words(QUERIES)
    first = metal_packed_stock(first_stock, weight=1.0, config=CONFIG)
    second = metal_packed_stock(second_stock, weight=1.0, config=CONFIG)
    frontier = PackedFrontier(QUERIES)

    expected_first = packed_stock(first_stock, weight=1.0, config=CONFIG)(frontier)
    expected_second = packed_stock(second_stock, weight=1.0, config=CONFIG)(frontier)
    for _ in range(2):
        assert np.array_equal(first(frontier), expected_first)
        assert np.array_equal(second(frontier), expected_second)


def test_repeated_passes_over_a_resident_stock_stay_correct() -> None:
    """An AO* search calls its heuristic once per pass, so the second call is the normal case."""
    skip_if_no_metal_device()
    heuristic = metal_packed_stock(vesma_stock_words(STOCK), weight=1.0, config=CONFIG)
    frontier = PackedFrontier(QUERIES)
    expected = heuristic(frontier)
    for _ in range(3):
        assert np.array_equal(heuristic(frontier), expected)


def test_each_scan_mints_its_own_token() -> None:
    """Tokens are identities, so two scans must never be able to collide on one."""
    stock = np.zeros((4, 32), dtype=np.uint64)
    assert metal_scan(stock).stock_token != metal_scan(stock).stock_token


def test_a_scan_carries_the_bit_counts_the_kernel_needs() -> None:
    """Precomputed on the host precisely so it is not a per-pass numpy sweep of the catalogue."""
    stock = vesma_stock_words(STOCK)
    scan = metal_scan(stock)
    assert isinstance(scan, MetalNearestScan)
    assert np.array_equal(
        scan.stock_bit_counts, np.bitwise_count(stock).sum(axis=1, dtype=np.uint32)
    )

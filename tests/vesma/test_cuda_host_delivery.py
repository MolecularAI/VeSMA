from __future__ import annotations

import gc
from collections import Counter

import numpy as np
import pytest

from tests.vesma.bench_reaction_application import cuda_host_batches
from vesma import cmol_from_smiles, cuda_retro
from vesma import cuda_matcher as cuda
from vesma.reaction import parse_reaction_smarts

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


@pytest.fixture(scope="module", autouse=True)
def cuda_available() -> None:
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CUDA is unavailable")


def test_host_delivery_copies_all_properties_without_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rules = [parse_reaction_smarts("[C:1][O:2]>>[C:1].[O:2]")]
    mols = [
        cmol_from_smiles(smiles)[0] for smiles in ("CCO", "[13CH3][18OH]", "c1ccccc1CO")
    ]
    pairs = [(index, 0) for index in range(len(mols))]
    matcher = cuda.CudaReactionMatcher(
        rules, frontier_mode="host", max_pairs_per_launch=1
    )
    assert matcher.max_pairs_per_launch == 1
    expected = list(matcher.match_batches(mols, pairs))

    def unexpected_key(*args: object, **kwargs: object) -> None:
        pytest.fail("Host delivery must not compute WL keys")

    with monkeypatch.context() as patch:
        patch.setattr(cuda, "cuda_short_ring_wl_key", unexpected_key)
        patch.setattr(cuda_retro, "wl_hash_kernel", unexpected_key)
        actual = list(cuda_host_batches(matcher, mols, pairs))
    assert len(actual) == len(expected) > 0
    replay = cuda.CudaReactionMatcher([parse_reaction_smarts("[*:1]>>[*:1]")])
    for (host, hits), (device, expected_hits) in zip(actual, expected, strict=True):
        assert len(host) == len(device)
        assert len(host.parts) == 1
        part = host.parts[0]
        assert part.keys == ()
        np.testing.assert_array_equal(hits, expected_hits)
        for name in ("atom_counts", "atoms", "bonds", "rows"):
            np.testing.assert_array_equal(
                getattr(part, name), getattr(device, name + "_device").get()
            )
        # Host output is also a usable input frontier, not merely telemetry.
        host_result, host_hits = replay.match(host)
        device_result, device_hits = replay.match(device)
        assert len(host_result) > 0
        host_keys = cuda.cuda_short_ring_wl_key(host_result)
        device_keys = cuda.cuda_short_ring_wl_key(device_result)
        # Parallel assignment numbering may differ; chemical outcomes must agree.
        assert Counter(
            (int(h[0]), int(h[1]), host_keys[h[3]]) for h in host_hits
        ) == Counter((int(h[0]), int(h[1]), device_keys[h[3]]) for h in device_hits)


def test_pinned_output_survives_device_reuse_and_retained_views() -> None:
    cp = pytest.importorskip("cupy")
    baseline = cuda.PINNED_HOST_BYTES
    device = cp.arange(1024, dtype=cp.int32)
    host = cuda.pinned_host_array(device)
    view = host[::2]
    assert cuda.PINNED_HOST_BYTES == baseline + device.nbytes
    device.fill(-1)
    other = cuda.pinned_host_array(device)
    np.testing.assert_array_equal(host, np.arange(1024, dtype=np.int32))
    np.testing.assert_array_equal(other, -np.ones(1024, dtype=np.int32))
    del host, other
    gc.collect()
    assert cuda.PINNED_HOST_BYTES == baseline + device.nbytes
    np.testing.assert_array_equal(view, np.arange(0, 1024, 2, dtype=np.int32))
    del view
    gc.collect()
    assert cuda.PINNED_HOST_BYTES == baseline


def test_retained_output_over_budget_uses_pageable_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = pytest.importorskip("cupy")
    baseline = cuda.PINNED_HOST_BYTES
    monkeypatch.setattr(cuda, "PINNED_HOST_LIMIT_BYTES", baseline + 4096)
    device = cp.arange(1024, dtype=cp.int32)
    first = cuda.pinned_host_array(device)
    second = cuda.pinned_host_array(device)
    assert cuda.PINNED_HOST_BYTES == baseline + device.nbytes
    device.fill(-1)
    np.testing.assert_array_equal(first, np.arange(1024, dtype=np.int32))
    np.testing.assert_array_equal(second, first)
    del first
    gc.collect()
    assert cuda.PINNED_HOST_BYTES == baseline
    assert cuda.PINNED_HOST_ALLOCATED_BYTES <= cuda.PINNED_HOST_LIMIT_BYTES


def test_device_frontier_does_not_materialize_host_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_graph_copy(*args: object, **kwargs: object) -> None:
        pytest.fail("Device frontier must not copy graph arrays to host")

    monkeypatch.setattr(cuda, "pinned_host_array", unexpected_graph_copy)
    mol = cmol_from_smiles("CCO")[0]
    matcher = cuda.CudaReactionMatcher(
        [parse_reaction_smarts("[C:1][O:2]>>[C:1].[O:2]")],
        frontier_mode="device",
    )
    assert matcher.max_pairs_per_launch == cuda.CUDA_FRONTIER_MAX_PAIRS_PER_LAUNCH
    result, _, _ = matcher.match_keyed([mol])
    assert len(result) > 0
    assert matcher.last_diagnostics is not None
    assert matcher.last_diagnostics.molecules_transferred_to_host_count == 0
    assert matcher.last_diagnostics.retained_device_to_host_seconds == 0

from __future__ import annotations

from collections import Counter

import numpy as np

from tests.vertsy.bench_cuda_aostar import (
    STOCK_CACHE_VERSION,
    interleave_target_buckets,
    prepare_stock_data,
)
from vesma.circularfp import CircularFingerprintConfig


def test_interleave_target_buckets_keeps_small_prefixes_stratified() -> None:
    targets = interleave_target_buckets(
        tuple(
            tuple((depth, f"{depth}-{index}") for index in range(count))
            for depth, count in ((5, 34), (6, 33), (7, 33))
        )
    )

    assert len(targets) == 100
    assert Counter(depth for depth, _ in targets[:10]) == {5: 4, 6: 3, 7: 3}
    assert Counter(depth for depth, _ in targets) == {5: 34, 6: 33, 7: 33}


def test_prepare_stock_data_uses_fingerprint_config() -> None:
    radius_zero = CircularFingerprintConfig(n_bits=256, radius=0)
    radius_two = CircularFingerprintConfig(n_bits=256, radius=2)

    keys_zero, fingerprints_zero = prepare_stock_data(
        ("CCO",), radius_zero, STOCK_CACHE_VERSION, 1
    )
    keys_two, fingerprints_two = prepare_stock_data(
        ("CCO",), radius_two, STOCK_CACHE_VERSION, 1
    )

    assert keys_zero == keys_two
    assert fingerprints_zero.shape == fingerprints_two.shape == (1, 4)
    assert not np.array_equal(fingerprints_zero, fingerprints_two)

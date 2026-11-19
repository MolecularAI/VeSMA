"""The nearest-stock scan on an Apple GPU, as a drop-in for the numpy one.

:class:`vertsy.host_similarity.HostNearestScan`'s replacement, computing the same distances bit for
bit — see ``tests/vesma/test_metal_circularfp.py``, which pins them exactly rather than
approximately. Packed precursor fingerprinting also runs on Metal after AO* selects novel molecules;
the stock scan remains a separate kernel.

    at 2048 bits against a 988k-row Enamine catalogue, a full AO* pass's 19,296 precursors take
    **594 seconds** to score on the CPU and **6.3 seconds** here (94x, measured on an Apple M5,
    10 CPU cores / 8 GPU cores). The fingerprint stage is measured separately from this scan.

That flips which term dominates: before, a 64x16 search against a full catalogue spent ~2.5 hours in
the heuristic against ~5 minutes in the matcher, and after, the heuristic is ~2 minutes and the
matcher is the largest cost again. It also makes a catalogue-scale stock a reasonable default rather
than something to subsample around.

**The catalogue stays resident on the device**, which is why the scan is an object and not a
function: 241 MB is more expensive to upload than to scan (39 ms per call re-uploading against 18 ms
reusing), so it is uploaded once against a token minted here and reused for the life of the scan.
Build the scan and the stock together through :func:`metal_packed_stock_similarity` or
:func:`metal_ecfp4_stock_similarity` — the same reasoning as :func:`vertsy.similarity.stock_of`, that
a heuristic pointed at one catalogue and a token minted for another is a wrong answer nothing else
would catch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vesma.circularfp import CircularFingerprintConfig
from vesma.metal_circularfp import (
    compute_circular_fingerprints_metal,
    metal_circularfp_available,
    nearest_stock_distance_metal,
    next_stock_token,
    packed_bit_counts,
)

from .host_similarity import NearestStock, ecfp4_stock, packed_stock

__all__ = [
    "MetalNearestScan",
    "metal_ecfp4_stock",
    "metal_packed_stock",
    "metal_scan",
    "metal_similarity_available",
]

metal_similarity_available = metal_circularfp_available


@dataclass(frozen=True)
class MetalNearestScan:
    """A device scan bound to one resident catalogue.

    ``stock_token`` is that catalogue's identity, not a copy of it: the fingerprints are uploaded on
    the first call carrying this token and every later call reuses them. Mint tokens through
    :func:`metal_scan` rather than by hand, so two live catalogues can never be handed the same one.

    ``stock_bit_counts`` is the set-bit count per catalogue row, which the union identity needs and
    which never changes. Held here because deriving it inside the scan would be a numpy pass over the
    whole 241 MB catalogue on every AO* pass, for a quantity that was already known.
    """

    stock_token: int
    stock_bit_counts: np.ndarray

    def __call__(
        self, queries: np.ndarray, stock: np.ndarray, *, weight: float
    ) -> np.ndarray:
        return nearest_stock_distance_metal(
            queries,
            stock,
            weight=weight,
            stock_token=self.stock_token,
            stock_bit_counts=self.stock_bit_counts,
        )


def metal_scan(stock_fingerprints: np.ndarray) -> MetalNearestScan:
    """A device scan for this catalogue, with a fresh token and its bit counts precomputed.

    One scan per catalogue, held for as long as the search runs. Building a second scan for the same
    fingerprints is correct but wasteful: it mints a new token and so re-uploads 241 MB.
    """
    return MetalNearestScan(
        stock_token=next_stock_token(),
        stock_bit_counts=packed_bit_counts(stock_fingerprints),
    )


def metal_packed_stock(
    stock_fingerprints: np.ndarray,
    *,
    weight: float,
    config: CircularFingerprintConfig = CircularFingerprintConfig(),
) -> NearestStock[Any]:
    """Circular fingerprints on the GPU, scanned on the GPU, against a resident catalogue.

    Both stages move to the device and neither changes an answer, which is the whole claim the
    two-stage split makes checkable: the catalogue and the scan's token come from one array in one
    place, so the heuristic cannot end up measuring distance to a catalogue other than the resident
    one.
    """
    return packed_stock(
        stock_fingerprints,
        weight=weight,
        config=config,
        scan=metal_scan(stock_fingerprints),
        words=compute_circular_fingerprints_metal,
    )


def metal_ecfp4_stock(
    stock_fingerprints: np.ndarray,
    *,
    weight: float,
    n_bits: int = 2048,
) -> NearestStock[Any]:
    """RDKit ECFP4 on the host, scanned on the GPU -- the second stage moved and not the first."""
    return ecfp4_stock(
        stock_fingerprints,
        weight=weight,
        n_bits=n_bits,
        scan=metal_scan(stock_fingerprints),
    )

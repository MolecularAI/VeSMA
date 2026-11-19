"""CUDA stock-distance heuristic for device-resident AO* frontiers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from vesma.circularfp import CircularFingerprintConfig
from vesma.cmol import CMol
from vesma.cuda_circularfp import (
    compute_circular_fingerprints_cuda,
    nearest_stock_distance_cuda,
)
from vesma.cuda_matcher import CudaMoleculeHostBatch, cuda_stack_mols, import_cupy

from .host_similarity import Fingerprint, NearestStock

__all__ = [
    "cuda_circular",
    "cuda_scan",
    "cuda_stock_similarity",
    "cuda_stock_similarity_from_fingerprints",
]


def cuda_circular(
    config: CircularFingerprintConfig = CircularFingerprintConfig(),
) -> Fingerprint[Any]:
    """Circular fingerprints on the GPU, leaving the words in VRAM for the scan that follows.

    A host batch is uploaded first; a batch that is already device-resident is used where it is,
    which is the point of the CUDA arm -- nothing round-trips between the two stages.
    """

    def fingerprint(molecules: Any) -> Any:
        device = (
            molecules.to_cuda()
            if isinstance(molecules, CudaMoleculeHostBatch)
            else molecules
        )
        return compute_circular_fingerprints_cuda(device, config)

    return fingerprint


def cuda_scan(queries: Any, stock: Any, *, weight: float) -> np.ndarray:
    """The device popcount scan, returning the one array that does cross back to the host.

    A :class:`vertsy.host_similarity.NearestScan`: exactly the numpy scan's numbers, computed
    elsewhere. One float per molecule is all the search reads, so that is all that is copied.
    """
    return nearest_stock_distance_cuda(queries, stock, weight=weight).get()


def cuda_stock_similarity(
    stock_molecules: Sequence[CMol],
    *,
    weight: float,
    config: CircularFingerprintConfig = CircularFingerprintConfig(),
    batch_size: int = 1024,
) -> NearestStock[Any]:
    """Build an exact nearest-stock heuristic and keep only its fingerprints in VRAM."""
    if not stock_molecules:
        raise ValueError("stock_molecules must not be empty.")
    if weight < 0.0:
        raise ValueError(f"weight must be non-negative, got {weight!r}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size!r}.")

    cp = import_cupy()
    fingerprints = [
        compute_circular_fingerprints_cuda(
            cuda_stack_mols(stock_molecules[start : start + batch_size]),
            config,
        )
        for start in range(0, len(stock_molecules), batch_size)
    ]
    stock = fingerprints[0] if len(fingerprints) == 1 else cp.concatenate(fingerprints)
    return NearestStock(stock, weight, cuda_circular(config), cuda_scan)


def cuda_stock_similarity_from_fingerprints(
    stock_fingerprints: np.ndarray,
    *,
    weight: float,
    config: CircularFingerprintConfig = CircularFingerprintConfig(),
) -> NearestStock[Any]:
    """Upload cached packed fingerprints once and retain them in VRAM."""
    if stock_fingerprints.ndim != 2:
        raise ValueError("stock_fingerprints must be a two-dimensional array.")
    if stock_fingerprints.shape[0] == 0:
        raise ValueError("stock_fingerprints must not be empty.")
    if stock_fingerprints.shape[1] != config.n_bits // 64:
        raise ValueError(
            f"fingerprint width must be {config.n_bits // 64} uint64 words, "
            f"got {stock_fingerprints.shape[1]}."
        )
    if stock_fingerprints.dtype != np.uint64:
        raise ValueError("stock_fingerprints must have dtype uint64.")
    if weight < 0.0:
        raise ValueError(f"weight must be non-negative, got {weight!r}.")
    cp = import_cupy()
    return NearestStock(
        cp.asarray(stock_fingerprints), weight, cuda_circular(config), cuda_scan
    )

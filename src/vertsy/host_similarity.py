"""Stock-distance heuristics for frontiers that arrive as host arrays.

Two callables, one per molecule representation a host backend hands the search:

``PackedStockSimilarity``  a packed batch (the Metal matcher's precursors, or a CUDA host batch)
                          scored with :mod:`vesma.packed_circularfp` — vesma's own circular
                          fingerprint, so the estimate is the one the CUDA heuristic computes and
                          the two backends are comparable.
``Ecfp4StockSimilarity``   an object array of RDKit ``Mol`` scored with RDKit's ECFP4, which is what
                          the RDKit arm produces natively.

Both keep the stock as **packed bits**: ``n_bits / 8`` bytes per molecule rather than the
``4 * n_bits`` of :func:`vertsy.similarity.stock_similarity`'s float matrix. That is the whole
reason this module exists — a 988k-molecule Enamine catalogue at 2048 bits is 253 MB packed
against 8.1 GB unpacked, and only one of those fits on a laptop. It is a memory trade and not a
speed one: scoring is exact against the entire catalogue either way, and at catalogue scale it
is slow enough (~30 ms per query against 988k) that a large stock wants a device kernel. See
:func:`vesma.packed_circularfp.nearest_stock_distance`.

**Read that cost against the frontier before choosing a stock.** The estimate is priced per
(molecule, catalogue row) pair, so its cost is the product and neither factor is small: one AO*
pass at width 64 over a 67k-rule library discovers ~20,000 novel precursors, and 20,000 against
988k rows is ten CPU-minutes for one pass. A 64x16 search therefore spends hours in the heuristic
and minutes in the matcher — the opposite of the balance the search was tuned for. Against a stock
of tens of thousands (PaRoutes n1) the same estimate is free. Do not assume a bigger stock is only a
memory question.

**Which is what ``scan`` is for.** The scan is the whole cost and it is separable from the
fingerprinting, so it is a strategy rather than a hard-coded call: :class:`HostNearestScan` is the
numpy reference and :func:`vertsy.metal_similarity.metal_scan` is the same answer bit for bit on an
Apple GPU, measured at 94x against a 988k catalogue — 6.3 seconds per AO* pass where the host path
takes ten minutes. Reach for a catalogue-scale stock through that one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, cast

import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs

from vesma.circularfp import CircularFingerprintConfig, circular_fingerprint_words
from vesma.cmol_builder import gmol_from_cmol
from vesma.packed_circularfp import (
    NEAREST_BUDGET_BYTES,
    circular_fingerprints_packed,
    nearest_stock_distance,
)

from .core import M
from .search import MoleculeBatch
from .similarity import morgan_generator

__all__ = [
    "Fingerprint",
    "HostNearestScan",
    "NearestScan",
    "NearestStock",
    "PackedBatch",
    "cmol_circular",
    "cmol_stock",
    "ecfp4",
    "ecfp4_stock",
    "packed_circular",
    "packed_ecfp4_words",
    "packed_stock",
    "pack_bit_vectors",
]


class NearestScan(Protocol):
    """Nearest-stock distance for a batch of packed fingerprints, however it is computed.

    The seam between "what a fingerprint is" and "where the popcounts happen". Both are exact
    against the whole catalogue and both return the same numbers, so which one a search holds is a
    performance decision and never a modelling one — which is exactly why it is a swappable
    strategy and not a flag.
    """

    def __call__(
        self, queries: np.ndarray, stock: np.ndarray, *, weight: float
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class HostNearestScan:
    """The numpy scan: exact, portable, and the specification the device scans are held to.

    ``budget_bytes`` caps the working set rather than the row count, because an AO* pass hands the
    scan the several thousand precursors it just discovered and a stock-side knob leaves the
    intermediate growing with the batch. See
    :func:`vesma.packed_circularfp.nearest_stock_distance`.
    """

    budget_bytes: int = NEAREST_BUDGET_BYTES

    def __call__(
        self, queries: np.ndarray, stock: np.ndarray, *, weight: float
    ) -> np.ndarray:
        return nearest_stock_distance(
            queries, stock, weight=weight, budget_bytes=self.budget_bytes
        )


class PackedBatch(Protocol):
    """The three arrays a packed molecule batch exposes, whatever else it carries.

    Structural rather than a concrete type so this module needs neither the Metal nor the CUDA
    matcher imported to score their batches.
    """

    atoms: np.ndarray
    bonds: np.ndarray
    atom_counts: np.ndarray

    def __len__(self) -> int: ...


# A batch of molecules as a matrix of fingerprint words -- one row per molecule, in whatever
# representation the scan that follows wants. Deliberately untyped in its output: on the host that is
# a `(N, n_bits // 64)` uint64 array and on CUDA it is the device array of the same shape, and
# nothing between the two stages needs to know which.
type Fingerprint[M] = Callable[[MoleculeBatch[M]], Any]


@dataclass(frozen=True)
class NearestStock(Generic[M]):
    """Distance to the nearest stock molecule: a fingerprint, then a scan. Nothing else.

    **The two independent choices are the two fields.** What a fingerprint *is* -- circular over a
    packed graph, ECFP4 over an RDKit molecule, whatever a device kernel computes -- is one
    decision; *where the popcounts happen* is the other, and it is a performance decision that never
    changes an answer. Every combination used to be its own frozen dataclass -- packed host, packed
    Metal, RDKit ECFP4, vesma CMol, CUDA -- and all five were the same two lines with a different
    spelling of the first stage, which is exactly the structure four classes were hiding.

    Satisfies :class:`vertsy.heuristics.Heuristic`, and satisfies its contract: a molecule that *is*
    in the stock scores 0 and one sharing nothing with it scores ``weight``, so the estimate vanishes
    at the goal. ``weight`` converts a dimensionless distance into whatever unit a step costs, and
    only the ratio of the two decides anything -- see
    :func:`vertsy.similarity.stock_similarity` for the measurements behind that.

    Validation of the catalogue's *width* belongs to whoever knows it, which is the factory that also
    chose the fingerprint; this checks only what it can see.
    """

    stock: Any
    weight: float
    fingerprint: Fingerprint[M]
    scan: NearestScan = HostNearestScan()

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError(f"weight must be non-negative, got {self.weight!r}.")

    def __call__(self, molecules: MoleculeBatch[M]) -> np.ndarray:
        if not len(molecules):
            return np.zeros(0, dtype=np.float32)
        return self.scan(self.fingerprint(molecules), self.stock, weight=self.weight)

    @property
    def nbytes(self) -> int:
        return int(self.stock.nbytes)


def packed_circular(
    config: CircularFingerprintConfig = CircularFingerprintConfig(),
    words: Callable[
        [np.ndarray, np.ndarray, np.ndarray, CircularFingerprintConfig], np.ndarray
    ] = circular_fingerprints_packed,
) -> Fingerprint[Any]:
    """Circular fingerprints over a packed batch -- the shape a device kernel wants.

    ``words`` is the kernel: the numpy one by default, and
    :func:`vesma.metal_circularfp.compute_circular_fingerprints_metal` for the same numbers on a GPU.
    """

    def fingerprint(molecules: MoleculeBatch[Any]) -> np.ndarray:
        batch = cast("PackedBatch", molecules)
        return words(batch.atoms, batch.bonds, batch.atom_counts, config)

    return fingerprint


def cmol_circular(
    config: CircularFingerprintConfig = CircularFingerprintConfig(),
) -> Fingerprint[Any]:
    """The same circular fingerprints, for a frontier held one molecule per object-array slot.

    Per molecule and not vectorised on purpose: this serves the arm whose reason to exist is that it
    needs no GPU, and the fingerprint is a small fraction of a pass that also applies tens of
    thousands of templates.
    """

    def fingerprint(molecules: MoleculeBatch[Any]) -> np.ndarray:
        rows = cast("Sequence[Any]", molecules)
        return np.stack(
            [
                circular_fingerprint_words(gmol_from_cmol(molecule), config)
                for molecule in rows
            ]
        )

    return fingerprint


def ecfp4(n_bits: int = 2048) -> Fingerprint[Chem.Mol]:
    """Binary ECFP4 over RDKit molecules, packed into words.

    The same estimate :func:`vertsy.similarity.stock_similarity` computes with its default arguments
    -- binary ECFP4, single nearest neighbour, Jaccard -- over packed bits instead of a float matrix.
    Tanimoto is invariant under permuting bit positions, so the packing order RDKit's binary text
    happens to use never enters the answer.
    """

    def fingerprint(molecules: MoleculeBatch[Chem.Mol]) -> np.ndarray:
        return packed_ecfp4_words(cast("Sequence[Chem.Mol]", molecules), n_bits=n_bits)

    return fingerprint


def packed_stock(
    stock_fingerprints: np.ndarray,
    *,
    weight: float,
    config: CircularFingerprintConfig = CircularFingerprintConfig(),
    scan: NearestScan = HostNearestScan(),
    words: Callable[
        [np.ndarray, np.ndarray, np.ndarray, CircularFingerprintConfig], np.ndarray
    ] = circular_fingerprints_packed,
) -> NearestStock[Any]:
    """:class:`NearestStock` over a packed batch, with the catalogue width checked."""
    validate_packed_stock(stock_fingerprints, weight, config)
    return NearestStock(
        stock_fingerprints, weight, packed_circular(config, words), scan
    )


def cmol_stock(
    stock_fingerprints: np.ndarray,
    *,
    weight: float,
    config: CircularFingerprintConfig = CircularFingerprintConfig(),
    scan: NearestScan = HostNearestScan(),
) -> NearestStock[Any]:
    """:class:`NearestStock` over vesma ``CMol`` molecules, against the same catalogue."""
    validate_packed_stock(stock_fingerprints, weight, config)
    return NearestStock(stock_fingerprints, weight, cmol_circular(config), scan)


def ecfp4_stock(
    stock_fingerprints: np.ndarray,
    *,
    weight: float,
    n_bits: int = 2048,
    scan: NearestScan = HostNearestScan(),
) -> NearestStock[Chem.Mol]:
    """:class:`NearestStock` over RDKit molecules, with the catalogue width checked."""
    if n_bits <= 0 or n_bits % 64:
        raise ValueError("n_bits must be a positive multiple of 64.")
    if stock_fingerprints.dtype != np.uint64:
        raise ValueError("stock_fingerprints must have dtype uint64.")
    if stock_fingerprints.shape[1:] != (n_bits // 64,):
        raise ValueError(
            f"stock fingerprint width must be {n_bits // 64} uint64 words, "
            f"got {stock_fingerprints.shape[1:]}."
        )
    return NearestStock(stock_fingerprints, weight, ecfp4(n_bits), scan)


def pack_bit_vectors(vectors: Sequence[Any], n_bits: int) -> np.ndarray:
    """``(N, n_bits // 64)`` ``uint64`` from RDKit explicit bit vectors."""
    words = np.zeros((len(vectors), n_bits // 64), dtype=np.uint64)
    for index, vector in enumerate(vectors):
        raw = np.frombuffer(DataStructs.BitVectToBinaryText(vector), dtype=np.uint8)
        words[index] = raw.view(np.uint64)
    return words


def packed_ecfp4_words(
    molecules: Sequence[Chem.Mol], *, n_bits: int = 2048
) -> np.ndarray:
    """Packed binary ECFP4 for RDKit molecules, in the layout :func:`pack_bit_vectors` produces.

    Fingerprinted through :func:`vertsy.similarity.morgan_generator` rather than a generator of
    its own, so this and :func:`vertsy.similarity.stock_similarity` cannot drift into computing
    two different ECFP4s and then be compared as though they were the same estimate.
    """
    generator = morgan_generator(n_bits)
    return pack_bit_vectors(
        [generator.GetFingerprint(mol) for mol in molecules], n_bits
    )


def validate_packed_stock(
    fingerprints: np.ndarray,
    weight: float,
    config: CircularFingerprintConfig,
) -> None:
    """Refuse a stock that cannot be what the heuristic will compare queries against."""
    if weight < 0.0:
        raise ValueError(f"weight must be non-negative, got {weight!r}.")
    if fingerprints.ndim != 2:
        raise ValueError("stock_fingerprints must be a two-dimensional array.")
    if not fingerprints.shape[0]:
        raise ValueError("stock_fingerprints must not be empty.")
    if fingerprints.dtype != np.uint64:
        raise ValueError("stock_fingerprints must have dtype uint64.")
    if fingerprints.shape[1] != config.n_bits // 64:
        raise ValueError(
            f"fingerprint width must be {config.n_bits // 64} uint64 words, "
            f"got {fingerprints.shape[1]}."
        )

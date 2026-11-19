"""Weisfeiler-Lehman molecule-hash engine.

Produces a one-way 128-bit molecule identity token by refining per-atom labels
over their circular neighborhoods and hashing the sorted multiset of final
labels. The shared hash mixers and bond-classing live in ``invariants.py``; this
module owns only the vectorized refinement loop and the final token.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xxhash

from .adjacency import BondMatrix
from .gmol_atoms import AtomVector, Charge
from .invariants import (
    combine_uint64,
    mix_uint64,
    signed_values_as_uint64,
)

WlToken = tuple[int, int]

WL_DEFAULT_ROUNDS = 8

WL_FINAL_LOW_SEED = 0x4D6F7267616E574C
WL_FINAL_HIGH_SEED = 0x7665736D615F776C


@dataclass(frozen=True)
class WlMol:
    Z: AtomVector
    C: Charge
    H: AtomVector
    bonds: BondMatrix

    def __post_init__(self) -> None:
        atom_count = self.Z.size
        if self.C.shape != (atom_count,) or self.H.shape != (atom_count,):
            raise ValueError("WL atom invariant shapes must match atom count.")
        if self.bonds.shape != (atom_count, atom_count):
            raise ValueError("WL bond matrix shape must match atom count.")
        if np.any(self.H < 0):
            raise ValueError("WL identity requires exact total hydrogen counts.")


def wl_molecule_id(
    molecule: WlMol,
    *,
    schema: int,
    rounds: int,
) -> WlToken:
    """Generic Weisfeiler-Lehman 128-bit molecule identity token.

    ``schema`` namespaces tokens from different bond-classing conventions
    (short-ring vs aromatic); ``rounds`` is the neighborhood-refinement depth.

    Isotope is deliberately excluded: isotopologues are one molecule in key space, so a
    labelled and an unlabelled precursor collapse to the same identity. The nucleus is
    still exact state on `CMol` and is conserved across a reaction; it simply does not
    take part in identity.
    """
    if rounds < 0:
        raise ValueError("rounds must be non-negative.")

    adjacency = molecule.bonds != 0
    degree = adjacency.sum(axis=1).astype(np.uint64, copy=False)
    labels = mix_uint64(molecule.Z.astype(np.uint64, copy=False))
    labels = combine_uint64(labels, signed_values_as_uint64(molecule.C))
    labels = combine_uint64(labels, molecule.H.astype(np.uint64, copy=False))
    labels = combine_uint64(labels, degree)
    edge_classes = molecule.bonds.astype(np.uint64, copy=False)

    for _ in range(rounds):
        messages = combine_uint64(
            np.broadcast_to(labels[None, :], adjacency.shape),
            edge_classes,
        )
        aggregate = np.where(adjacency, messages, np.uint64(0)).sum(
            axis=1,
            dtype=np.uint64,
        )
        labels = mix_uint64(
            combine_uint64(
                combine_uint64(labels, degree),
                aggregate,
            )
        )

    edge_count = int(np.count_nonzero(np.triu(adjacency, k=1)))
    payload = np.concatenate(
        [
            np.asarray(
                (schema, rounds, molecule.Z.size, edge_count),
                dtype=np.uint64,
            ),
            np.sort(labels).astype(np.uint64, copy=False),
        ]
    )
    payload_bytes = payload.tobytes()
    return (
        xxhash.xxh64_intdigest(payload_bytes, seed=WL_FINAL_LOW_SEED),
        xxhash.xxh64_intdigest(payload_bytes, seed=WL_FINAL_HIGH_SEED),
    )


__all__ = [
    "WL_DEFAULT_ROUNDS",
    "WL_FINAL_HIGH_SEED",
    "WL_FINAL_LOW_SEED",
    "WlMol",
    "WlToken",
    "wl_molecule_id",
]

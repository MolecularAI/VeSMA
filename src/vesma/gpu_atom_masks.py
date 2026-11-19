"""The atom-word layout shared by the CUDA and Metal matchers."""

from __future__ import annotations

import numpy as np

from .gmol import GMol
from .gmol_atoms import MASS_ANY
from .limits import ATOM_MASK_WORD_COUNT

ISOTOPE_WORD = 13
ISOTOPE_SHIFT = 16
ISOTOPE_ANY = 0xFFFF


def isotope_field(mass: np.ndarray) -> np.ndarray:
    """Encode exact masses and the query wildcard in word 13's high half."""
    return np.where(mass == MASS_ANY, ISOTOPE_ANY, mass.astype(np.int64)).astype(
        np.uint32
    )


def pack_atom_masks(
    graph: GMol, bucket_size: int, *, aromaticity_any: int = 0xFFFFFFFF
) -> np.ndarray:
    """Pack atom masks, with ring connectivity and isotope in separate half words."""
    if graph.atom_count > bucket_size:
        raise ValueError(f"atom mask packer supports graphs up to {bucket_size} atoms.")
    masks = graph.atom_masks
    rows = np.zeros((bucket_size, ATOM_MASK_WORD_COUNT), dtype=np.uint32)
    count = graph.atom_count
    rows[:count, 0] = (masks.Z_lo & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    rows[:count, 1] = (masks.Z_lo >> np.uint64(32)).astype(np.uint32)
    rows[:count, 2] = (masks.Z_hi & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    rows[:count, 3] = (masks.Z_hi >> np.uint64(32)).astype(np.uint32)
    rows[:count, 4] = np.where(
        masks.A == 3, np.uint32(aromaticity_any), masks.A.astype(np.uint32)
    )
    for word, prop in enumerate(("C", "D", "H", "h", "v", "X", "R", "r", "x"), 5):
        rows[:count, word] = getattr(masks, prop).astype(np.uint32)
    rows[:count, ISOTOPE_WORD] |= isotope_field(masks.M) << np.uint32(ISOTOPE_SHIFT)
    return rows.reshape(-1)

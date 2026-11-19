from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .metal_runtime import i32_ptr, load_metal_dylib, u8_ptr, u32_ptr, u64_ptr
from .pathfp import FoldedFingerprintConfig

_FP_TILE_SIZE = 32

__all__ = [
    "PATHFP_CONSTANT_FIELDS",
    "MetalPrefilterResult",
    "compute_folded_path_fingerprints_metal",
    "compact_folded_prefilter_pairs_metal",
    "probe_pathfp_constants",
]

_METAL_LINEAR_SOURCE_PATH = Path(__file__).with_name("metal_pathfp.mm")


@dataclass(frozen=True)
class MetalPrefilterResult:
    pair_pattern_index_device: np.ndarray
    pair_molecule_index_device: np.ndarray
    pair_count: int
    dense_pair_count: int


def _load_metal_linear_library() -> ctypes.CDLL:
    lib = load_metal_dylib(_METAL_LINEAR_SOURCE_PATH)

    lib.vesma_metal_linear_last_error.restype = ctypes.c_char_p

    lib.vesma_metal_linear_fingerprints.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),  # atoms
        ctypes.POINTER(ctypes.c_uint8),  # bonds
        ctypes.POINTER(ctypes.c_uint64),  # rows
        ctypes.POINTER(ctypes.c_int32),  # atom_counts
        ctypes.POINTER(ctypes.c_uint64),  # out
        ctypes.c_uint32,  # molecule_count
        ctypes.c_uint32,  # query
        ctypes.c_uint32,  # max_length
        ctypes.c_uint32,  # word_count
    ]
    lib.vesma_metal_linear_fingerprints.restype = ctypes.c_int

    lib.vesma_metal_linear_prefilter.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),  # molecule_fps
        ctypes.POINTER(ctypes.c_uint64),  # pattern_fps
        ctypes.POINTER(ctypes.c_int32),  # molecule_atom_counts
        ctypes.POINTER(ctypes.c_int32),  # pattern_atom_counts
        ctypes.POINTER(ctypes.c_int32),  # pair_pattern_index
        ctypes.POINTER(ctypes.c_int32),  # pair_molecule_index
        ctypes.POINTER(ctypes.c_uint64),  # survivor_count
        ctypes.c_uint32,  # molecule_count
        ctypes.c_uint32,  # pattern_count
        ctypes.c_uint32,  # word_count
        ctypes.c_uint32,  # pattern_index_offset
    ]
    lib.vesma_metal_linear_prefilter.restype = ctypes.c_int

    lib.vesma_metal_linear_constants.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    lib.vesma_metal_linear_constants.restype = ctypes.c_int

    return lib


#: What `vesma_metal_linear_constants` reports, in order. Each of these is declared twice -- MSL
#: lives in a raw string and cannot include the host's header -- so a test compares them against
#: `vesma.limits` rather than trusting the comments beside them.
PATHFP_CONSTANT_FIELDS = (
    "shader_bucket",
    "shader_atom_word_count",
    "shader_fp_word_count",
    "shader_fp_tile_size",
    "shader_bond_class_mask",
)


def probe_pathfp_constants() -> dict[str, int]:
    """Ask the folded-path shader which compile-time constants it was built with."""
    library = _load_metal_linear_library()
    out = (ctypes.c_uint32 * len(PATHFP_CONSTANT_FIELDS))()
    if library.vesma_metal_linear_constants(out) != 1:
        raise RuntimeError(library.vesma_metal_linear_last_error().decode("utf-8"))
    return dict(zip(PATHFP_CONSTANT_FIELDS, list(out), strict=True))


def compute_folded_path_fingerprints_metal(
    batch_atoms: np.ndarray,
    batch_bonds: np.ndarray,
    batch_rows: np.ndarray,
    batch_atom_counts: np.ndarray,
    *,
    query: bool,
    config: FoldedFingerprintConfig = FoldedFingerprintConfig(),
) -> np.ndarray:
    if config.n_bits % 64 != 0:
        raise ValueError(
            "config.n_bits must be a multiple of 64 for Metal fingerprints."
        )
    if config.max_length > 7:
        raise ValueError("Metal folded path fingerprints support max_length <= 7.")

    molecule_count = int(batch_atom_counts.shape[0])
    word_count = config.n_bits // 64
    out = np.zeros((molecule_count, word_count), dtype=np.uint64)
    if molecule_count == 0:
        return out

    lib = _load_metal_linear_library()
    ok = lib.vesma_metal_linear_fingerprints(
        u32_ptr(batch_atoms.reshape(-1)),
        u8_ptr(batch_bonds.reshape(-1)),
        u64_ptr(batch_rows.reshape(-1)),
        i32_ptr(batch_atom_counts),
        u64_ptr(out.reshape(-1)),
        molecule_count,
        1 if query else 0,
        config.max_length,
        word_count,
    )
    if ok != 1:
        msg = lib.vesma_metal_linear_last_error()
        raise RuntimeError(msg.decode("utf-8") if msg else "Unknown Metal error")
    return out


def compact_folded_prefilter_pairs_metal(
    *,
    molecule_fingerprints: np.ndarray,
    pattern_fingerprints: np.ndarray,
    molecule_atom_counts: np.ndarray,
    pattern_atom_counts: np.ndarray,
    pattern_index_offset: int = 0,
) -> MetalPrefilterResult:
    molecule_count = int(molecule_fingerprints.shape[0])
    pattern_count = int(pattern_fingerprints.shape[0])
    if molecule_count == 0 or pattern_count == 0:
        return MetalPrefilterResult(
            pair_pattern_index_device=np.empty(0, dtype=np.int32),
            pair_molecule_index_device=np.empty(0, dtype=np.int32),
            pair_count=0,
            dense_pair_count=molecule_count * pattern_count,
        )
    if molecule_fingerprints.shape[1] != pattern_fingerprints.shape[1]:
        raise ValueError(
            "molecule and pattern fingerprints must have equal word counts."
        )

    dense_pair_count = molecule_count * pattern_count
    pair_pattern_index = np.empty(dense_pair_count, dtype=np.int32)
    pair_molecule_index = np.empty(dense_pair_count, dtype=np.int32)
    survivor_count = np.zeros(1, dtype=np.uint64)
    word_count = int(molecule_fingerprints.shape[1])

    if word_count != 32:
        raise ValueError("Metal folded prefilter expects 2048-bit fingerprints.")

    lib = _load_metal_linear_library()
    ok = lib.vesma_metal_linear_prefilter(
        u64_ptr(molecule_fingerprints.reshape(-1)),
        u64_ptr(pattern_fingerprints.reshape(-1)),
        i32_ptr(molecule_atom_counts),
        i32_ptr(pattern_atom_counts),
        i32_ptr(pair_pattern_index),
        i32_ptr(pair_molecule_index),
        u64_ptr(survivor_count),
        molecule_count,
        pattern_count,
        word_count,
        pattern_index_offset,
    )
    if ok != 1:
        msg = lib.vesma_metal_linear_last_error()
        raise RuntimeError(msg.decode("utf-8") if msg else "Unknown Metal error")

    pair_count = int(survivor_count[0])
    return MetalPrefilterResult(
        pair_pattern_index_device=pair_pattern_index[:pair_count],
        pair_molecule_index_device=pair_molecule_index[:pair_count],
        pair_count=pair_count,
        dense_pair_count=dense_pair_count,
    )

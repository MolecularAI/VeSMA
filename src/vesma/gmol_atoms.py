from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

ALL_Z_MASK = (1 << 128) - 1
ALIPHATIC = 0b0000_0001
AROMATIC = 0b0000_0010
ALL_A_MASK = ALIPHATIC | AROMATIC

# Isotope is exact-or-any rather than a bucket mask: the mass domain is far too wide to
# enumerate, and SMARTS only ever asserts one mass. A query carries MASS_ANY when it
# says nothing about the nucleus; a molecule carries MASS_UNLABELED for natural
# abundance, matching RDKit's GetIsotope() convention so the two round-trip.
MASS_ANY = -1
MASS_UNLABELED = 0

# OpenSMILES 3.1 requires a parser to accept isotope values from 0 to 999. Zero is the
# unlabelled sentinel above, so a *written* nuclide runs 1 to 999. Nothing heavier exists,
# and tools that write more are using the field to smuggle in attachment points.
MIN_ISOTOPE = 1
MAX_ISOTOPE = 999

# What the per-property masks can hold without clamping. A molecule states its properties
# exactly, so anything outside these is a value the encoding would silently round -- see
# `vesma.parsing.reading.validate_representable`. Bucket masks cover a value and,
# conventionally for a query, "or more" at the top; charge masks are offset by -CHARGE_MIN.
CHARGE_MIN = -3
CHARGE_MAX = 4
BUCKET_MAX = 7


type AtomPropertyName = Literal[
    "Z",
    "A",
    "C",
    "D",
    "H",
    "h",
    "v",
    "X",
    "x",
    "R",
    "r",
    "M",
    "AtomMap",
]
type AtomData = dict[AtomPropertyName, int]


# Shared per-atom array vocabulary (this leaf module is the home for atom-indexed
# array types, as adjacency.py is for bond-matrix types). Names describe the axis
# and role; the dtype/shape hint carries the representation.
AtomVector = np.ndarray[tuple[int], np.dtype[np.int16]]  # per-atom int16 property
Aromaticity = np.ndarray[tuple[int], np.dtype[np.uint8]]  # per-atom aromatic flag/mask
Charge = np.ndarray[tuple[int], np.dtype[np.int8]]  # per-atom formal charge
MaskVector = np.ndarray[
    tuple[int], np.dtype[np.uint8]
]  # per-atom property bitmask byte
ZMaskWords = np.ndarray[
    tuple[int], np.dtype[np.uint64]
]  # 64-bit halves of the Z bitmask
AtomIndices = np.ndarray[tuple[int], np.dtype[np.int64]]  # array of atom indices
AtomMask = np.ndarray[tuple[int], np.dtype[np.bool_]]  # per-atom boolean mask
AtomSelector = np.ndarray[tuple[int], np.dtype[np.int32]]  # int32 atom take-selector


BUCKET_DEFAULT_MASKS: dict[str, int] = {
    "Z": ALL_Z_MASK,
    "A": ALL_A_MASK,
    "C": 0xFF,
    "D": 0xFF,
    "H": 0xFF,
    "h": 0xFF,
    "v": 0xFF,
    "X": 0xFF,
    "R": 0xFF,
    "r": 0xFF,
    "x": 0xFF,
}


def _bucket_mask(value: int) -> int:
    v = min(max(value, 0), BUCKET_MAX)
    return 1 << v


def _charge_mask(charge: int) -> int:
    clamped = min(max(charge, CHARGE_MIN), CHARGE_MAX)
    return 1 << (clamped - CHARGE_MIN)


def z_mask_for_numbers(numbers: Iterable[int]) -> int:
    mask = 0
    for num in numbers:
        if 0 <= num < 128:
            mask |= 1 << num
    return mask


def bucket_mask_exact(value: int) -> int:
    return _bucket_mask(value)


def bucket_mask_at_least(min_value: int) -> int:
    start = min(max(min_value, 0), 7)
    return sum(1 << bit for bit in range(start, 8))


def bucket_mask_at_most(max_value: int) -> int:
    stop = min(max(max_value, 0), 7)
    return sum(1 << bit for bit in range(stop + 1))


def bucket_mask_range(min_value: int, max_value: int) -> int:
    if max_value < min_value:
        return 0
    return bucket_mask_at_least(min_value) & bucket_mask_at_most(max_value)


def bucket_mask_min(mask: int) -> int:
    mask = int(mask)
    if mask <= 0:
        raise ValueError("Bucket mask must be non-zero.")
    return (mask & -mask).bit_length() - 1


def bucket_mask_max(mask: int) -> int:
    mask = int(mask)
    if mask <= 0:
        raise ValueError("Bucket mask must be non-zero.")
    return mask.bit_length() - 1


def mask_for_charge(charge: int) -> int:
    return _charge_mask(charge)


def decode_single_mask_or_none(mask: int, *, offset: int = 0) -> int | None:
    mask = int(mask)
    if mask <= 0 or mask & (mask - 1):
        return None
    return mask.bit_length() - 1 + offset


def decode_single_mask(mask: int, *, name: str, offset: int = 0) -> int:
    value = decode_single_mask_or_none(mask, offset=offset)
    if value is None:
        raise ValueError(f"Operation requires single-bit {name} mask.")
    return value


def decode_aromatic_mask(a_mask: int, *, aromatic_from_bonds: bool) -> bool:
    if a_mask == AROMATIC:
        return True
    if a_mask == ALIPHATIC:
        return False
    if a_mask == ALL_A_MASK:
        return aromatic_from_bonds
    raise ValueError("Operation requires single-bit aromaticity mask or ALL_A mask.")


def decode_template_explicit_h(h_total_mask: int, h_implicit_mask: int) -> int:
    h_total = decode_single_mask_or_none(h_total_mask)
    h_implicit = decode_single_mask_or_none(h_implicit_mask)
    if h_total is None or h_implicit is None:
        return -1
    explicit_h = max(h_total - h_implicit, 0)
    if explicit_h == 0 and h_implicit > 0:
        return -1
    return explicit_h


@dataclass(frozen=True)
class AtomMasks:
    Z_lo: ZMaskWords
    Z_hi: ZMaskWords
    A: MaskVector
    C: MaskVector
    D: MaskVector
    H: MaskVector
    h: MaskVector
    v: MaskVector
    X: MaskVector
    R: MaskVector
    r: MaskVector
    x: MaskVector
    M: AtomVector

    def __post_init__(self) -> None:
        expected_dtypes = (
            ("Z_lo", np.uint64),
            ("Z_hi", np.uint64),
            ("A", np.uint8),
            ("C", np.uint8),
            ("D", np.uint8),
            ("H", np.uint8),
            ("h", np.uint8),
            ("v", np.uint8),
            ("X", np.uint8),
            ("R", np.uint8),
            ("r", np.uint8),
            ("x", np.uint8),
            ("M", np.int16),
        )
        size: int | None = None
        for name, dtype in expected_dtypes:
            values = getattr(self, name)
            if values.dtype != dtype:
                raise ValueError(f"{name} dtype must be {np.dtype(dtype)}.")
            if values.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional.")
            if size is None:
                size = int(values.shape[0])
            elif values.shape != (size,):
                raise ValueError(f"{name} shape must match atom count.")

    @property
    def size(self) -> int:
        return int(self.A.shape[0])

    def exact_z(self) -> np.ndarray:
        z_lo = self.Z_lo
        z_hi = self.Z_hi
        z_single = ((z_lo > 0) & ((z_lo & (z_lo - 1)) == 0) & (z_hi == 0)) | (
            (z_hi > 0) & ((z_hi & (z_hi - 1)) == 0) & (z_lo == 0)
        )
        z_inherit = (z_lo == np.iinfo(np.uint64).max) & (
            z_hi == np.iinfo(np.uint64).max
        )
        if not np.all(z_single | z_inherit):
            raise ValueError(
                "Concrete molecule conversion requires single-bit atomic number mask."
            )
        val = np.where(z_hi > 0, z_hi, z_lo)
        safe_val = np.where(val > 0, val, 1)
        res = np.where(
            z_inherit, -1, np.log2(safe_val) + np.where(z_hi > 0, 64, 0)
        ).astype(np.int16)
        return res

    def exact_c(self) -> np.ndarray:
        c = self.C
        c_single = (c > 0) & ((c & (c - 1)) == 0)
        c_has_neutral = (c & (1 << 3)) > 0
        if not np.all(c_single | c_has_neutral):
            raise ValueError(
                "Concrete molecule conversion requires single-bit charge mask or a mask that includes neutral charge."
            )
        safe_c = np.where(c > 0, c, 1)
        return np.where(c_single, np.log2(safe_c) - 3, 0).astype(np.int8)

    def exact_h_total(self) -> np.ndarray:
        h = self.H
        h_single = (h > 0) & ((h & (h - 1)) == 0)
        safe_h = np.where(h > 0, h, 1)
        return np.where(h_single, np.log2(safe_h), -1).astype(np.int16)

    def exact_h_implicit(self) -> np.ndarray:
        h = self.h
        h_single = (h > 0) & ((h & (h - 1)) == 0)
        safe_h = np.where(h > 0, h, 1)
        return np.where(h_single, np.log2(safe_h), -1).astype(np.int16)

    def exact_a(self) -> np.ndarray:
        a = self.A
        a_single = (a > 0) & ((a & (a - 1)) == 0)
        a_inherit = a == ALL_A_MASK
        if not np.all(a_single | a_inherit):
            raise ValueError(
                "Concrete molecule conversion requires single-bit aromaticity mask or ALL_A mask."
            )
        return a.copy()

    def exact_m(self) -> np.ndarray:
        """Required isotope per atom; MASS_ANY where the query is silent."""
        return self.M.copy()

    def exact_property(self, attr_name: str) -> np.ndarray:
        v = getattr(self, attr_name)
        single = (v > 0) & ((v & (v - 1)) == 0)
        safe_v = np.where(v > 0, v, 1)
        return np.where(single, np.log2(safe_v), -1).astype(np.int16)


def atom_masks_from_props(
    atom_props: Sequence[AtomData],
) -> AtomMasks:
    if not atom_props:
        empty_u64 = np.zeros(0, dtype=np.uint64)
        empty_u8 = np.zeros(0, dtype=np.uint8)
        return AtomMasks(
            Z_lo=empty_u64,
            Z_hi=empty_u64,
            A=empty_u8,
            C=empty_u8,
            D=empty_u8,
            H=empty_u8,
            h=empty_u8,
            v=empty_u8,
            X=empty_u8,
            R=empty_u8,
            r=empty_u8,
            x=empty_u8,
            M=np.zeros(0, dtype=np.int16),
        )

    z_masks = [int(props.get("Z", BUCKET_DEFAULT_MASKS["Z"])) for props in atom_props]
    z_lo = np.array([mask & ((1 << 64) - 1) for mask in z_masks], dtype=np.uint64)
    z_hi = np.array([mask >> 64 for mask in z_masks], dtype=np.uint64)

    return AtomMasks(
        Z_lo=z_lo,
        Z_hi=z_hi,
        A=np.array(
            [int(props.get("A", BUCKET_DEFAULT_MASKS["A"])) for props in atom_props],
            dtype=np.uint8,
        ),
        C=np.array(
            [int(props.get("C", BUCKET_DEFAULT_MASKS["C"])) for props in atom_props],
            dtype=np.uint8,
        ),
        D=np.array(
            [int(props.get("D", BUCKET_DEFAULT_MASKS["D"])) for props in atom_props],
            dtype=np.uint8,
        ),
        H=np.array(
            [int(props.get("H", BUCKET_DEFAULT_MASKS["H"])) for props in atom_props],
            dtype=np.uint8,
        ),
        h=np.array(
            [int(props.get("h", BUCKET_DEFAULT_MASKS["h"])) for props in atom_props],
            dtype=np.uint8,
        ),
        v=np.array(
            [int(props.get("v", BUCKET_DEFAULT_MASKS["v"])) for props in atom_props],
            dtype=np.uint8,
        ),
        X=np.array(
            [int(props.get("X", BUCKET_DEFAULT_MASKS["X"])) for props in atom_props],
            dtype=np.uint8,
        ),
        R=np.array(
            [int(props.get("R", BUCKET_DEFAULT_MASKS["R"])) for props in atom_props],
            dtype=np.uint8,
        ),
        r=np.array(
            [int(props.get("r", BUCKET_DEFAULT_MASKS["r"])) for props in atom_props],
            dtype=np.uint8,
        ),
        x=np.array(
            [int(props.get("x", BUCKET_DEFAULT_MASKS["x"])) for props in atom_props],
            dtype=np.uint8,
        ),
        M=np.array(
            [int(props.get("M", MASS_ANY)) for props in atom_props],
            dtype=np.int16,
        ),
    )


def atom_masks_take(atom_masks: AtomMasks, indices: AtomSelector) -> AtomMasks:
    return AtomMasks(
        Z_lo=atom_masks.Z_lo[indices].copy(),
        Z_hi=atom_masks.Z_hi[indices].copy(),
        A=atom_masks.A[indices].copy(),
        C=atom_masks.C[indices].copy(),
        D=atom_masks.D[indices].copy(),
        H=atom_masks.H[indices].copy(),
        h=atom_masks.h[indices].copy(),
        v=atom_masks.v[indices].copy(),
        X=atom_masks.X[indices].copy(),
        R=atom_masks.R[indices].copy(),
        r=atom_masks.r[indices].copy(),
        x=atom_masks.x[indices].copy(),
        M=atom_masks.M[indices].copy(),
    )

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Generic, Protocol, runtime_checkable

import numpy as np

from .core import K

__all__ = [
    "DictInterner",
    "Interner",
    "MISSING_SID",
    "SidArray",
]

# A 1-D array of dense ``int64`` storage ids (sids). Sids are minted from 0, so they
# double as indices into the facade's ``expanded`` bitvector.
type SidArray = np.ndarray[tuple[int], np.dtype[np.int64]]

# Returned by ``lookup`` for keys never interned. Negative so it never collides with a
# real sid and reads naturally in masks (``sids != MISSING_SID``).
MISSING_SID = np.int64(-1)


@runtime_checkable
class Interner(Protocol[K]):
    """A bijection between molecule keys and dense ``int64`` storage ids (sids).

    The only place a *key flavor* (str SMILES/InChI, 128-bit WL ``tuple[int, int]``, a
    future 64-bit hash) is materialised; every layer below speaks sids. ``intern`` mints
    sids for unseen keys (mutating); ``lookup`` is the pure variant that reports misses,
    so a Mapping's ``__contains__`` need not grow the id space.
    """

    def intern(self, keys: Collection[K]) -> SidArray:
        """Get-or-assign a sid for each key (mutating); sids are parallel to ``keys``.

        ``keys`` is any array-like (``list`` or ``ndarray``): the common denominator of
        the two is ``Collection`` (``Sequence`` excludes ``ndarray``).
        """
        ...

    def lookup(self, keys: Collection[K]) -> SidArray:
        """Pure key→sid lookup; unknown keys map to :data:`MISSING_SID`."""
        ...

    def decode(self, sids: Sequence[int] | SidArray) -> list[K]:
        """Reverse map sids back to their keys."""
        ...

    def __len__(self) -> int:
        """Number of distinct keys interned so far (== next sid to be minted)."""
        ...


class DictInterner(Generic[K]):
    """Reference interner: a ``dict`` plus a growable sid→key list.

    Correct for any hashable key flavor and fine to ~1M keys; at many millions the
    per-key ``dict`` traffic becomes the bottleneck, at which point a columnar interner
    (Polars anti-join, or a numba table over the two ``int64`` WL columns) slots in
    behind the same :class:`Interner` protocol.
    """

    __slots__ = ("_sids", "_keys")

    def __init__(self) -> None:
        self._sids: dict[K, int] = {}
        self._keys: list[K] = []

    def intern(self, keys: Collection[K]) -> SidArray:
        table = self._sids
        store = self._keys
        out = np.empty(len(keys), dtype=np.int64)
        for index, key in enumerate(keys):
            got = table.get(key)
            if got is None:
                got = len(store)
                table[key] = got
                store.append(key)
            out[index] = got
        return out

    def lookup(self, keys: Collection[K]) -> SidArray:
        table = self._sids
        return np.fromiter(
            (table.get(key, MISSING_SID) for key in keys),
            dtype=np.int64,
            count=len(keys),
        )

    def decode(self, sids: Sequence[int] | SidArray) -> list[K]:
        store = self._keys
        return [store[int(sid)] for sid in sids]

    def __len__(self) -> int:
        return len(self._keys)

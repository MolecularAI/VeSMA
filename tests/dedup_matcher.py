"""A `ReactionMatcher` wrapper that deduplicates the precursor batch.

The live backends all return one precursor row per hit. The contract allows a shorter,
deduplicated batch (`vesma.matcher.ReactionMatcher`), so this wrapper is what exercises
that half of it: wrap any matcher, key its precursors, and collapse duplicate keys down
to one batch row, leaving the hits' precursor column to point at the survivor.

`mode` picks how much is collapsed, covering the three shapes a consumer must handle:

* ``"none"`` — no collapsing (one row per hit), i.e. exactly what the live backends do.
* ``"partial"`` — every other duplicate group is collapsed, so the batch is neither
  fully dense nor fully unique. The arbitrary alternation is the point: consumers must
  not be able to assume any relationship between hit count and batch size.
* ``"full"`` — one row per distinct key.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from vesma.matcher import HitArray, Pair, compact_precursor_index

DedupMode = Literal["none", "partial", "full"]
DEDUP_MODES: tuple[DedupMode, ...] = ("none", "partial", "full")


def dedup_precursor_rows(
    keys: Sequence[Any],
    hits: HitArray,
    mode: DedupMode,
) -> tuple[np.ndarray, HitArray]:
    """Point every hit at a representative precursor row, then compact the batch.

    Returns the surviving precursor rows (ascending) and the rebased hits, exactly like
    :func:`vesma.matcher.compact_precursor_index`, whose renumbering it reuses.
    """
    rows = np.asarray(hits, dtype=np.int64).reshape(-1, 4)
    if mode == "none":
        return np.arange(len(keys), dtype=np.int64), rows.copy()

    first_row: dict[Any, int] = {}
    collapse_key: dict[Any, bool] = {}
    representative: dict[int, int] = {}
    collapse_next = True
    for row in np.unique(rows[:, 3]).tolist():
        key = keys[row]
        if key not in first_row:
            first_row[key] = row
            collapse_key[key] = mode == "full" or collapse_next
            collapse_next = not collapse_next
        representative[row] = first_row[key] if collapse_key[key] else row

    rebased = rows.copy()
    rebased[:, 3] = [representative[row] for row in rows[:, 3].tolist()]
    return compact_precursor_index(rebased)


@dataclass(frozen=True)
class DedupingReactionMatcher:
    """Wrap a `ReactionMatcher` and return a deduplicated precursor batch."""

    matcher: Any
    key: Callable[[Any], Sequence[Any]]
    mode: DedupMode = "full"

    def match(
        self,
        mols: Sequence[Any],
        pairs: Sequence[Pair] | None = None,
    ) -> tuple[Any, HitArray]:
        precursors, hits = self.matcher.match(mols, pairs)
        used, rebased = dedup_precursor_rows(self.key(precursors), hits, self.mode)
        return precursors[used], rebased

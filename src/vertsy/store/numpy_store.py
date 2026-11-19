from __future__ import annotations

import numpy as np

from ..interner import SidArray
from . import EdgeColumns, ExpansionStore, RuleColumns, SelectivityColumns, SidMask

__all__ = ["NumpyExpansionStore", "Table", "resized"]

_INITIAL_CAPACITY = 1024


class Table:
    """Three parallel growable int64 columns: an append-only relation, and nothing else.

    Every one of the store's tables is this shape — ``(product, app, precursor)``,
    ``(product, app, rule)`` and one ``(app, rule, outcomes)`` per measurement channel — so they
    share one implementation rather than a set of near-identical buffers and one ``_grow`` apiece.
    """

    __slots__ = ("_columns", "_n")

    def __init__(self) -> None:
        self._columns = tuple(
            np.empty(_INITIAL_CAPACITY, dtype=np.int64) for _ in range(3)
        )
        self._n = 0

    def append(self, first: SidArray, second: SidArray, third: SidArray) -> None:
        rows = (first, second, third)
        count = len(first)
        if not all(len(column) == count for column in rows):
            raise ValueError("table columns must be parallel and equal length")
        if count == 0:
            return
        capacity = self._columns[0].shape[0]
        needed = self._n + count
        if needed > capacity:
            while capacity < needed:
                capacity *= 2
            self._columns = tuple(
                np.resize(column, capacity) for column in self._columns
            )
        start, end = self._n, needed
        for column, values in zip(self._columns, rows, strict=True):
            column[start:end] = values
        self._n = end

    def columns(self) -> tuple[SidArray, SidArray, SidArray]:
        first, second, third = (column[: self._n] for column in self._columns)
        return first, second, third


class NumpyExpansionStore(ExpansionStore):
    """Reference :class:`~vertsy.store.ExpansionStore` over growable int64 columns.

    Two :class:`Table` relations in doubling buffers; appending is an O(batch) copy. The
    expanded-product set is a per-sid bool bitvector (so membership is a vectorised gather
    and the set is inherently unique). A memmap variant can subclass this and back the
    buffers with ``np.memmap`` to spill out of core.

    A plain class, not a dataclass: the store takes no constructor inputs — every field is
    freshly-allocated internal state — so a hand-written ``__init__`` is shorter and avoids
    exposing the private buffers as init parameters.
    """

    def __init__(self) -> None:
        self._edges = Table()
        self._rules = Table()
        self._selectivity: dict[str, Table] = {}
        self._next_app = 0
        self._expanded: SidMask = np.zeros(_INITIAL_CAPACITY, dtype=np.bool_)

    def add_edges(
        self,
        product_sids: SidArray,
        app_ids: SidArray,
        precursor_sids: SidArray,
    ) -> None:
        self._edges.append(product_sids, app_ids, precursor_sids)
        if len(app_ids):
            self._next_app = max(self._next_app, int(np.max(app_ids)) + 1)

    def next_app(self) -> int:
        return self._next_app

    def add_rules(
        self,
        product_sids: SidArray,
        app_ids: SidArray,
        rule_ids: SidArray,
    ) -> None:
        self._rules.append(product_sids, app_ids, rule_ids)

    def add_selectivity(
        self,
        channel: str,
        app_ids: SidArray,
        rule_ids: SidArray,
        outcomes: SidArray,
    ) -> None:
        self._selectivity.setdefault(channel, Table()).append(
            app_ids, rule_ids, outcomes
        )

    def mark_expanded(self, product_sids: SidArray) -> None:
        sids = np.asarray(product_sids, dtype=np.int64)
        if not len(sids):
            return
        needed = int(sids.max()) + 1
        capacity = self._expanded.shape[0]
        if needed > capacity:
            while capacity < needed:
                capacity *= 2
            self._expanded = resized(self._expanded, capacity)
        self._expanded[sids] = True

    def is_expanded(self, product_sids: SidArray) -> SidMask:
        sids = np.asarray(product_sids, dtype=np.int64)
        out = np.zeros(len(sids), dtype=np.bool_)
        in_range = sids < self._expanded.shape[0]  # sids minted past the bitvector
        out[in_range] = self._expanded[sids[in_range]]  # are unexpanded by construction
        return out

    def expanded(self) -> SidArray:
        return np.nonzero(self._expanded)[0]

    def edges(self) -> EdgeColumns:
        return self._edges.columns()

    def rules(self) -> RuleColumns:
        return self._rules.columns()

    def selectivity(self, channel: str) -> SelectivityColumns:
        table = self._selectivity.get(channel)
        if table is None:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty, empty
        return table.columns()

    def channels(self) -> tuple[str, ...]:
        return tuple(self._selectivity)


def resized(array: SidMask, size: int) -> SidMask:
    """Grow-only zero-extend of a 1-D bool array to ``size`` (pure; identity if large)."""
    if array.shape[0] >= size:
        return array
    grown = np.zeros(size, dtype=np.bool_)
    grown[: array.shape[0]] = array
    return grown

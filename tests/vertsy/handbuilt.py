"""Writing one application straight into a store, for tests that build a graph by hand.

A disconnection now spans two tables — ``(product, app, precursor)`` and ``(product, app, rule)``
— so appending one by hand takes two calls with a shared app id. That is the store's business,
not each test's, hence this.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from vertsy.expansions import Expansions
from vertsy.interner import SidArray


def add_application(
    expansions: Expansions,
    product: int,
    rule: int,
    precursors: Sequence[int],
    *,
    app: int | None = None,
    outcomes: Mapping[str, int] | None = None,
) -> int:
    """Append one disconnection of ``product`` into ``precursors``, found by ``rule``.

    Returns the app id used. Pass ``app`` explicitly to attach another template to a
    disconnection already written — that is the co-canonical case worth testing on purpose, and
    it writes **only** the template row: the precursors are already there, and adding them again
    would say the disconnection needs each of them twice.

    ``outcomes`` writes a measurement row per channel -- what a forward probe would have found for
    this template on these precursors. Omit a channel and the disconnection is *unmeasured* on it,
    which is what every graph looked like before a probe existed and must keep costing exactly what
    it did.
    """
    minting = app is None
    if app is None:
        app = expansions.store.next_app()
    count = len(precursors)
    if minting and count:
        expansions.store.add_edges(
            np.full(count, product, dtype=np.int64),
            np.full(count, app, dtype=np.int64),
            np.asarray(precursors, dtype=np.int64),
        )
    expansions.store.add_rules(
        np.array([product], dtype=np.int64),
        np.array([app], dtype=np.int64),
        np.array([rule], dtype=np.int64),
    )
    for channel, count in (outcomes or {}).items():
        expansions.store.add_selectivity(
            channel,
            np.array([app], dtype=np.int64),
            np.array([rule], dtype=np.int64),
            np.array([count], dtype=np.int64),
        )
    return app


def sids_of(expansions: Expansions, keys: Sequence[str]) -> SidArray:
    """Intern ``keys`` and return their sids as an array (the common fixture preamble)."""
    return expansions.intern(list(keys))

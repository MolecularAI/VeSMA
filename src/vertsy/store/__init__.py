from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..interner import SidArray

__all__ = [
    "SidMask",
    "EdgeColumns",
    "ExpansionStore",
    "RuleColumns",
    "SelectivityColumns",
    "SidStep",
]

# One materialised step in sid space: (rule_id, precursor sids).
# One materialised disconnection in sid space: the templates that achieve it, and its
# precursor sids. ``rules`` is sorted and never empty; several templates reaching the same
# precursors is the common case, not the exception.
type SidStep = tuple[tuple[int, ...], tuple[int, ...]]
# The store's append-only disconnection rows as three parallel int64 arrays:
# product, application, precursor.
type EdgeColumns = tuple[SidArray, SidArray, SidArray]
# The store's append-only template rows as three parallel int64 arrays:
# product, application, rule.
type RuleColumns = tuple[SidArray, SidArray, SidArray]
# The store's append-only measurement rows as three parallel int64 arrays: application, rule, and
# how many distinct products the forward chemistry makes from that application's precursors.
# Sparse -- only measured rows exist -- and no product column, because ``app`` determines it.
type SelectivityColumns = tuple[SidArray, SidArray, SidArray]
# A 1-D boolean array (a per-sid membership mask / bitvector).
type SidMask = np.ndarray[tuple[int], np.dtype[np.bool_]]


@runtime_checkable
class ExpansionStore(Protocol):
    """Swappable, sid-native storage for the retrosynthetic edge relation.

        Two append-only tables sharing one primary key, ``(product, app)``. An **edge** row is
        ``(product_sid, app_id, precursor_sid)``: one precursor of one *disconnection* of one
        product. A **rule** row is ``(product_sid, app_id, rule_id)``: one template that achieves
        that disconnection.

        Splitting them is what keeps the graph proportional to the chemistry. ``app_id`` used to
        come straight from the matcher and was local to a ``(product, rule)`` pair, so up to ~50
        co-canonical templates disconnecting a product into *identical* precursors became 50
        separate applications — fifty-fold OR fan-out for the fold to walk and fifty spellings of
        one route to enumerate. Now ``app_id`` is local to the product alone and identifies the
        precursor multiset, so a template is an attribute of a disconnection rather than a copy of
        it. It never recurs: a product is expanded at most once over the facade's lifetime.

        The store speaks only sids; the hot path is ``add_edges`` / ``add_rules`` (bulk append), and
        cold-path grouping happens above the store from the flat columns.

    A **sparse** table per measurement *channel* records what a forward probe found:
        ``(app_id, rule_id, outcomes)``, one row per template whose forward chemistry someone actually
        ran on that disconnection's precursors. Each needs no product column, because ``app_id`` already
        determines the product -- the same invariant that lets a template be an attribute of a
        disconnection rather than a copy of it.

        Alongside the tables the store owns the set of *expanded* product sids (a
        per-sid bitvector): a product can be evaluated yet emit zero edges (a stock leaf or a
        dead end), so the expanded set is what tells "evaluated, no steps" from "never seen".
        ``is_expanded`` answers the per-sid membership test the search hot path needs. Both tables
        plus the expanded set are the store's full, self-describing state (the serializable
        unit, together with the interner).
    """

    def add_edges(
        self,
        product_sids: SidArray,
        app_ids: SidArray,
        precursor_sids: SidArray,
    ) -> None:
        """Append a batch of edge rows; the three arrays are parallel and equal length."""
        ...

    def add_rules(
        self,
        product_sids: SidArray,
        app_ids: SidArray,
        rule_ids: SidArray,
    ) -> None:
        """Append a batch of template rows; the three arrays are parallel and equal length."""
        ...

    def add_selectivity(
        self,
        channel: str,
        app_ids: SidArray,
        rule_ids: SidArray,
        outcomes: SidArray,
    ) -> None:
        """Append a batch of measurement rows to one channel; the arrays are parallel.

        A measurement is what running forward templates on one application's precursors makes: the
        number of distinct products. ``1`` says the step is clean, more says it is a mixture, and a
        row that is simply **absent** says nobody measured it -- which is not the same claim and
        must not be priced as one.

        One relation per ``channel`` because the two that exist measure different chemistry.
        ``sites`` counts the products of the **intended** reaction, which is a mixture to separate.
        ``competitors`` counts the products of *other* reactions the same starting materials could
        undergo, which may mean the intended step does not work. A store keeps whichever channels
        were written and knows nothing else about them.

        Only measured rows are written. That keeps each table sparse exactly to the extent the
        forward library covers the retro library, which is the honest shape and also makes coverage
        readable straight off the tables rather than tracked separately.
        """
        ...

    def mark_expanded(self, product_sids: SidArray) -> None:
        """Record product sids as evaluated (whether or not they emitted any edges)."""
        ...

    def is_expanded(self, product_sids: SidArray) -> SidMask:
        """Per-sid membership predicate (vectorised; the search hot-path filter).

        Tests a *supplied* batch of sids; unseen sids (incl. any minted past the expanded
        bitvector) read ``False``. Distinct from :meth:`expanded`, which enumerates.
        """
        ...

    def expanded(self) -> SidArray:
        """Enumerate every evaluated product sid (the source of truth for "seen").

        Drives materialization and analytics that need the universe of evaluated products;
        distinct from :meth:`is_expanded`, which tests a supplied batch.
        """
        ...

    def edges(self) -> EdgeColumns:
        """Return product, application, precursor columns for cold-path analytics."""
        ...

    def rules(self) -> RuleColumns:
        """Return product, application, rule columns: which templates found each disconnection."""
        ...

    def selectivity(self, channel: str) -> SelectivityColumns:
        """Return one channel's application, rule and outcome columns.

        A channel nobody wrote returns three empty arrays, and every consumer's behaviour on those
        is its behaviour before measurements existed at all.
        """
        ...

    def channels(self) -> tuple[str, ...]:
        """Which measurement channels this store holds rows for, in insertion order."""
        ...

    def next_app(self) -> int:
        """The lowest application id not yet stored.

        Application ids are minted per *layer*, but a caller may seed a graph and then search
        over the same product — ``add_layer`` fires twice for it and both layers' disconnections
        must survive. Offsetting the second layer by this keeps ``(product, app)`` unique, and
        makes the ids globally dense besides, which is what a vectorised fold would index on.
        """
        ...

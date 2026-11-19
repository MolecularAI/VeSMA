from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Generic

import numpy as np
import polars as pl

from .core import HitArray, K, Step
from .costs import UNPROBED
from .interner import DictInterner, Interner, SidArray
from .store import (
    EdgeColumns,
    ExpansionStore,
    RuleColumns,
    SelectivityColumns,
    SidMask,
    SidStep,
)
from .store.numpy_store import NumpyExpansionStore

__all__ = [
    "EdgeColumns",
    "Expansions",
    "LayerTables",
    "RuleColumns",
    "SelectivityColumns",
    "Steps",
    "edge_frame",
    "expansions_of",
    "layer_tables",
    "materialize",
    "rule_frame",
    "selectivity_frame",
    "sid_selectivity",
    "sid_steps",
]

# Decoded key-space steps exposed only by explicit cold-path materialization.
Steps = frozenset[Step[K]]


@dataclass(eq=False)
class Expansions(Generic[K]):
    """Thin key/sid facade over a sid-native append-only expansion store.

    Search writes through ``intern`` / ``add_layer`` / ``mark_expanded``. Route extraction
    and solving stay sid-native through ``lookup`` / ``decode`` / ``steps`` and only cross
    back to keys at public API boundaries.
    """

    store: ExpansionStore
    interner: Interner[K]
    disputed: dict[str, int] = field(default_factory=dict)
    """Per channel, disconnections where several measurements arrived for one key and were reduced.

    Accumulated across layers for whoever owns the channel to report. The search writes it and
    reads nothing from it: a measurement contradicting itself is that measurement's defect, and
    stopping a search over it would make an add-on fatal to the thing it was added to.
    """

    def intern(self, keys: Collection[K]) -> SidArray:
        """Map a key batch to sids, minting unseen keys."""
        return self.interner.intern(keys)

    def lookup(self, keys: Collection[K]) -> SidArray:
        """Pure key-to-sid lookup; unknown keys map to ``MISSING_SID``."""
        return self.interner.lookup(keys)

    def decode(self, sids: Sequence[int] | SidArray) -> list[K]:
        """Decode sid-space ids back to key-space values."""
        return self.interner.decode(sids)

    def sid_count(self) -> int:
        """Number of interned sids, suitable for dense sid-indexed arrays."""
        return len(self.interner)

    def is_expanded(self, sids: SidArray) -> SidMask:
        """Per-sid: has this product already been evaluated?"""
        return self.store.is_expanded(sids)

    def add_layer(
        self,
        frontier_sids: SidArray,
        hits: HitArray,
        precursor_sids: SidArray,
        outcomes: Mapping[str, np.ndarray] | None = None,
        reduce: Mapping[str, Reduce] | None = None,
    ) -> None:
        """Record one matched layer: mark the frontier expanded and append the tables.

        ``precursor_sids`` is the interned precursor *batch* (which may be shorter than
        ``hits``); the hits' precursor column selects from it.

        ``outcomes`` maps a measurement channel to one count per **hit row**, and an absent channel
        is one nobody measured. Per hit rather than per application because that is the axis a
        matcher already returns and the axis every wrapper already gathers on; it is redundant by
        an application's arity, and collapsing it here is one reduction the grouping below performs
        anyway.

        ``reduce`` is how each channel resolves several values arriving for one disconnection --
        see :data:`Reduce`. Accumulated into :attr:`disputed` for whoever owns the channel to
        report; this layer neither warns nor raises about it, because what a measurement means is
        not its business.
        """
        # `layer_tables` mints app ids from zero, so shift them past whatever the store already
        # holds. Without this a caller who seeds a graph and then searches over the same product
        # would have the second layer's app 0 collide with the first's, silently welding two
        # disconnections into one.
        tables = layer_tables(frontier_sids, hits, precursor_sids, outcomes, reduce)
        for channel, count in tables.disputed.items():
            self.disputed[channel] = self.disputed.get(channel, 0) + count
        edge_products, edge_apps, precursors = tables.edges
        rule_products, rule_apps, rules = tables.rules
        base = self.store.next_app()
        self.store.mark_expanded(frontier_sids)
        self.store.add_edges(edge_products, edge_apps + base, precursors)
        self.store.add_rules(rule_products, rule_apps + base, rules)
        for channel, (apps, measured_rules, counts) in tables.selectivity.items():
            if len(apps):
                self.store.add_selectivity(channel, apps + base, measured_rules, counts)

    def mark_expanded(self, sids: SidArray) -> None:
        """Record ``sids`` as evaluated with no steps."""
        self.store.mark_expanded(sids)

    def steps(self, product_sid: int) -> tuple[SidStep, ...]:
        """Return grouped sid-native steps for one expanded product sid."""
        return sid_steps(self, np.array([product_sid], dtype=np.int64))[product_sid]

    def edges(self) -> EdgeColumns:
        """Return the flat append-only edge columns for cold-path solving."""
        return self.store.edges()

    def rules(self) -> RuleColumns:
        """Return the flat append-only template columns for cold-path solving."""
        return self.store.rules()

    def selectivity(self, channel: str) -> SelectivityColumns:
        """Return one channel's flat append-only measurement columns."""
        return self.store.selectivity(channel)

    def channels(self) -> tuple[str, ...]:
        """Which measurement channels this graph carries rows for."""
        return self.store.channels()


def expansions_of(
    mapping: Mapping[K, Iterable[Step[K]]],
    *,
    store: ExpansionStore | None = None,
    interner: Interner[K] | None = None,
) -> Expansions[K]:
    """Build :class:`Expansions` from a key-space ``product -> steps`` mapping.

    This helper is for tests, fixtures, and cold-path interop. It mirrors what the search
    fold writes: every mapping key is marked expanded, and each step becomes one application
    whose rows share ``(product, app)``, with its rule in the template table.

    Steps arrive already distinct, so nothing is collapsed here — a mapping that names one rule
    twice for the same precursors is the caller's statement that they are two applications.
    """
    actual_interner: Interner[K] = (
        interner if interner is not None else DictInterner[K]()
    )
    expansions: Expansions[K] = Expansions(
        store or NumpyExpansionStore(),
        actual_interner,
    )
    for product, steps in mapping.items():
        (product_sid,) = expansions.intern([product]).tolist()
        expansions.mark_expanded(np.array([product_sid], dtype=np.int64))
        for rule, precursors in steps:
            app = expansions.store.next_app()
            precursor_sids = expansions.intern(list(precursors))
            count = len(precursor_sids)
            expansions.store.add_edges(
                np.full(count, product_sid, dtype=np.int64),
                np.full(count, app, dtype=np.int64),
                precursor_sids,
            )
            expansions.store.add_rules(
                np.array([product_sid], dtype=np.int64),
                np.array([app], dtype=np.int64),
                np.array([rule], dtype=np.int64),
            )
    return expansions


def materialize(expansions: Expansions[K]) -> dict[K, Steps[K]]:
    """Decode all expanded products into the old dict-style key-space view."""
    sids = expansions.store.expanded()
    grouped = sid_steps(expansions, sids)
    keys = expansions.decode(sids)
    return {
        key: frozenset(
            (rule, frozenset(expansions.decode(precursor_sids)))
            for rules, precursor_sids in grouped[sid]
            for rule in rules
        )
        for key, sid in zip(keys, sids.tolist())
    }


def edge_frame(expansions: Expansions[K]) -> pl.DataFrame:
    """Return the append-only disconnection relation as a Polars dataframe."""
    product, app, precursor = expansions.edges()
    return pl.DataFrame({"product": product, "app": app, "precursor": precursor})


def rule_frame(expansions: Expansions[K]) -> pl.DataFrame:
    """Return the append-only template relation as a Polars dataframe."""
    product, app, rule = expansions.rules()
    return pl.DataFrame({"product": product, "app": app, "rule": rule})


def selectivity_frame(expansions: Expansions[K], channel: str) -> pl.DataFrame:
    """Return one channel's append-only measurement relation as a Polars dataframe."""
    app, rule, outcomes = expansions.selectivity(channel)
    return pl.DataFrame({"app": app, "rule": rule, "outcomes": outcomes})


def sid_steps(
    expansions: Expansions[K],
    product_sids: SidArray,
) -> dict[int, tuple[SidStep, ...]]:
    """Group requested products into deterministic sid-native route steps.

    One step per disconnection, carrying every template that achieves it — the two tables
    joined on their shared ``(product, app)`` key.
    """
    steps: dict[int, set[SidStep]] = {sid: set() for sid in product_sids.tolist()}
    if not len(product_sids):
        return {}

    grouped = (
        edge_frame(expansions)
        .filter(pl.col("product").is_in(product_sids))
        .group_by("product", "app")
        .agg(pl.col("precursor").sort())
        .join(
            rule_frame(expansions)
            .filter(pl.col("product").is_in(product_sids))
            .group_by("product", "app")
            .agg(pl.col("rule").sort().unique(maintain_order=True)),
            on=("product", "app"),
            how="left",
        )
    )

    for product, precursors, rules in zip(
        grouped["product"].to_list(),
        grouped["precursor"].to_list(),
        grouped["rule"].to_list(),
    ):
        steps[product].add((tuple(rules or ()), tuple(precursors)))

    return {
        product: tuple(sorted(product_steps, key=lambda step: (step[0], step[1])))
        for product, product_steps in steps.items()
    }


type Reduce = Callable[[np.ndarray], int]
"""How one channel resolves several values arriving for one disconnection.

The store's job is to collapse matcher applications onto disconnections, unconditionally and
without knowing what any measurement means. Two values can therefore arrive for one key, and what
to do about that is a question only the channel's owner can answer: a selectivity count that
contradicts itself measured nothing, so :func:`vertsy.selectivity.unmeasured` returns the
`UNPROBED` sentinel; a veto verdict resolves the other way, because a refusal is a refusal, so
:func:`vertsy.filters.refusal_wins` takes the strictest value.

Injected rather than defaulted, and a channel that supplies none is a wiring error rather than a
silent policy: see :func:`layer_tables`.
"""


@dataclass(frozen=True)
class LayerTables:
    """One matched layer as the store's three relations, named so an arity cannot be miscounted."""

    edges: EdgeColumns
    rules: RuleColumns
    selectivity: Mapping[str, SelectivityColumns]
    """One measurement relation per channel the layer was probed on; absent means unprobed."""
    disputed: Mapping[str, int] = field(default_factory=dict)
    """Per channel, disconnections where several values arrived for one key and were reduced.

    Reported and not raised, because a measurement is an **add-on**: a channel contradicting
    itself is a defect in that channel, and the search it was bolted onto is otherwise fine. The
    count belongs to whoever owns the channel -- it travels out of here and is reported beside that
    channel's own coverage, not by the search.
    """


def sid_selectivity(
    expansions: Expansions[K],
    exclude: Collection[str] = (),
) -> dict[str, dict[tuple[tuple[int, ...], int], int]]:
    """Group every channel's measurements by ``(precursor sids, rule)``, which prices a step.

    Keyed by the precursors and not by the application, because that is what the measurement is
    *about*: a forward reaction is determined by its rule and its starting materials. Two
    applications sharing both -- the same precursors disconnecting two different products by the
    same template -- share the chemistry, so they share the measurement, and they must agree. If
    they do not, one of them was measured wrongly -- and the honest answer is then that *neither*
    measured it, so the pair is dropped rather than resolved by whichever row the store wrote
    last. It used to raise, which made a probe defect fatal to a search that was otherwise fine;
    `layer_tables` now drops such a disconnection to `UNPROBED` before it ever reaches here, so
    this path is the backstop and counts rather than throws.

    ``exclude`` drops channels indexed elsewhere, which is not an optimisation so much as the
    absence of a lie: a channel whose answer depends on the product has no correct entry here, and
    building one costs a pass over every application to store a value nobody may trust. See
    :func:`sid_step_selectivity`.

    Empty in, empty out, and cheap: a store nobody probed returns three empty columns and this
    does no work at all.
    """
    skip = frozenset(exclude)
    channels = [channel for channel in expansions.channels() if channel not in skip]
    if not channels:
        return {}
    edge_apps_to_precursors: dict[int, list[int]] = {}
    _products, edge_apps, precursors = expansions.edges()
    for app, precursor in zip(edge_apps.tolist(), precursors.tolist()):
        edge_apps_to_precursors.setdefault(app, []).append(precursor)
    grouped: dict[str, dict[tuple[tuple[int, ...], int], int]] = {}
    for channel in channels:
        apps, rules, outcomes = expansions.selectivity(channel)
        table: dict[tuple[tuple[int, ...], int], int] = {}
        for app, rule, count in zip(apps.tolist(), rules.tolist(), outcomes.tolist()):
            key = (tuple(sorted(edge_apps_to_precursors.get(app, ()))), rule)
            seen = table.setdefault(key, count)
            if seen != count:
                # Neither value is trustworthy, so the key carries no measurement at all.
                table[key] = UNPROBED
        grouped[channel] = table
    return grouped


def sid_step_selectivity(
    expansions: Expansions[K],
    channels: Collection[str],
) -> dict[str, dict[tuple[int, int, tuple[int, ...]], int]]:
    """Group the named channels by ``(product, rule, precursor sids)`` -- one disconnection.

    The sibling of :func:`sid_selectivity`, for a measurement that is **not** a property of the
    chemistry alone. A forward probe asks what these starting materials do under this template, so
    its answer is keyed by exactly those two and the product is deliberately absent. A veto read on
    the *product* asks a different question, and two applications sharing precursors and rule can
    halogenate two different positions of two different molecules -- one condemned, one clean. Keyed
    without the product those two honestly disagree, and `sid_selectivity` drops the pair rather
    than trusting whichever was written last; keyed with it, each keeps its own answer.

    **The product-inclusive key cannot collide.** A product is expanded at most once over the
    facade's lifetime and `layer_tables` collapses its applications as they arrive, so one
    ``(product, rule, precursors)`` is one row here and the disagreement backstop has nothing to
    do. That is the whole reason this index exists rather than a policy bolted onto the other one.

    Only the named channels, because the index costs an entry per application and only a verdict
    needs it. The caller names them, for the same reason it supplies a :data:`Reduce`: the module
    that mints a channel owns what the channel is about.
    """
    wanted = [channel for channel in channels if channel in set(expansions.channels())]
    if not wanted:
        return {}
    products_of: dict[int, int] = {}
    precursors_of: dict[int, list[int]] = {}
    products, edge_apps, precursors = expansions.edges()
    for product, app, precursor in zip(
        products.tolist(), edge_apps.tolist(), precursors.tolist()
    ):
        products_of[app] = product
        precursors_of.setdefault(app, []).append(precursor)
    grouped: dict[str, dict[tuple[int, int, tuple[int, ...]], int]] = {}
    for channel in wanted:
        apps, rules, outcomes = expansions.selectivity(channel)
        table: dict[tuple[int, int, tuple[int, ...]], int] = {}
        for app, rule, count in zip(apps.tolist(), rules.tolist(), outcomes.tolist()):
            key = (
                products_of.get(app, -1),
                rule,
                tuple(sorted(precursors_of.get(app, ()))),
            )
            table[key] = count
        grouped[channel] = table
    return grouped


def layer_tables(
    frontier_sids: SidArray,
    hits: HitArray,
    precursor_sids: SidArray,
    outcomes: Mapping[str, np.ndarray] | None = None,
    reduce: Mapping[str, Reduce] | None = None,
) -> LayerTables:
    """Collapse one matched layer into disconnections, and record what found each.

    Each hit is ``(product_idx, rule_id, app_idx, precursor_idx)``: ``product_idx`` selects a
    frontier product, ``precursor_idx`` selects a precursor sid, and the matcher's ``app_idx``
    is local to a ``(product, rule)`` pair. So a product disconnected into identical precursors
    by fifty co-canonical templates arrives as fifty applications, and the whole point of this
    function is that it leaves as one, with fifty rules attached.

    **Layer-local deduplication is complete deduplication**: a product is expanded at most once
    over the facade's lifetime, so every application it will ever have is in this one batch.
    That is what makes collapsing here — rather than on every read — both correct and cheap.

    Applications are grouped by arity and collapsed with array operations, once per product ever;
    the fold this work spares runs once per pass, forever.

    ``outcomes`` maps each measurement channel to one count per hit row, and each collapses on the
    *same* grouping the template table already uses -- a measurement is a property of
    ``(rule, precursors)``, which is exactly what a ``(product, rule, app_idx)`` group is. It must
    be constant within a group, since every row of one group is one application of one template; a
    group that disagrees with itself is a probe that measured the same reaction twice and got two
    answers, so it raises rather than being averaged into something plausible.
    """
    rows = np.asarray(hits, dtype=np.int64).reshape(-1, 4)
    product_idx, rule_ids, app_ids, precursor_idx = (
        rows[:, 0],
        rows[:, 1],
        rows[:, 2],
        rows[:, 3],
    )
    if rows.shape[0]:
        if product_idx.min() < 0 or product_idx.max() >= len(frontier_sids):
            raise IndexError("hit product index is outside the frontier")
        if precursor_idx.min() < 0 or precursor_idx.max() >= len(precursor_sids):
            raise IndexError("hit precursor index is outside the precursor batch")
    products = frontier_sids[product_idx]
    precursors = precursor_sids[precursor_idx]

    # Sorting by the matcher's application key puts each application's rows together and its
    # precursors in a canonical order, which is what makes two applications' multisets
    # comparable at all. Multiplicity is preserved: an application naming one precursor twice
    # really does need it twice, and the fold charges it twice.
    order = np.lexsort((precursors, app_ids, rule_ids, products))
    products, rule_ids, app_ids, precursors = (
        column[order] for column in (products, rule_ids, app_ids, precursors)
    )
    reduce = reduce or {}
    measured = {
        channel: np.asarray(column, dtype=np.int64).reshape(-1)[order]
        for channel, column in (outcomes or {}).items()
    }
    keys = np.stack([products, rule_ids, app_ids], axis=1)
    breaks = np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1
    if not len(products):
        empty = np.empty(0, dtype=np.int64)
        return LayerTables((empty, empty, empty), (empty, empty, empty), {})

    starts = np.concatenate((np.array([0], dtype=np.int64), breaks))
    stops = np.concatenate((breaks, np.array([len(products)], dtype=np.int64)))
    arities = stops - starts
    edge_parts: list[EdgeColumns] = []
    template_parts: list[np.ndarray] = []
    app_base = 0
    for arity in np.unique(arities):
        group_rows = np.nonzero(arities == arity)[0]
        group_starts = starts[group_rows]
        positions = group_starts[:, None] + np.arange(arity, dtype=np.int64)
        application_keys = np.column_stack(
            (products[group_starts], precursors[positions])
        )
        unique_applications, inverse = np.unique(
            application_keys,
            axis=0,
            return_inverse=True,
        )
        application_ids = app_base + inverse
        unique_ids = np.arange(
            app_base,
            app_base + len(unique_applications),
            dtype=np.int64,
        )
        edge_parts.append(
            (
                np.repeat(unique_applications[:, 0], arity),
                np.repeat(unique_ids, arity),
                unique_applications[:, 1:].reshape(-1),
            )
        )
        # One further column per measured channel, so every measurement rides the same
        # `np.unique` that collapses co-canonical templates and needs no second grouping.
        template_parts.append(
            np.column_stack(
                (
                    products[group_starts],
                    application_ids,
                    rule_ids[group_starts],
                    *(column[group_starts] for column in measured.values()),
                )
            )
        )
        app_base += len(unique_applications)

    edges: EdgeColumns = (
        np.concatenate([part[0] for part in edge_parts]),
        np.concatenate([part[1] for part in edge_parts]),
        np.concatenate([part[2] for part in edge_parts]),
    )
    for channel, column in measured.items():
        # Groups are contiguous after the lexsort, so "constant within a group" is each row
        # against its own group's first row — one broadcast rather than a scan.
        if not bool(np.array_equal(np.repeat(column[starts], arities), column)):
            raise ValueError(
                "a forward measurement must be constant across one application's hit rows, "
                f"and channel {channel!r} is not"
            )
    templates = np.unique(np.concatenate(template_parts), axis=0)
    disputed: dict[str, int] = {}
    selectivity: dict[str, SelectivityColumns] = {}
    if measured:
        # Two matcher applications collapsing into one store application agree on their
        # measurements or they do not; if they do, `unique` left one row, and if they do not it
        # left two and `(app, rule)` is no longer a key. That is the same probe inconsistency the
        # check above catches within a group, seen across groups, and it fails the same way.
        keys = templates[:, 1:3]
        if len(np.unique(keys, axis=0)) != len(keys):
            # The store collapses; it does not judge. Several values for one disconnection is a
            # question about what the *channel* means, so the channel's own reduction answers it
            # and this code never learns what a count or a verdict is.
            seen, counts = np.unique(keys, axis=0, return_counts=True)
            culprits = seen[counts > 1]
            undeclared = [channel for channel in measured if channel not in reduce]
            for position, channel in enumerate(measured):
                column = templates[:, 3 + position]
                for app, rule in culprits.tolist():
                    where = (keys[:, 0] == app) & (keys[:, 1] == rule)
                    values = np.unique(column[where])
                    if len(values) == 1:
                        continue
                    if channel in reduce:
                        column[where] = reduce[channel](values)
                        disputed[channel] = disputed.get(channel, 0) + 1
                    else:
                        # Nobody said how this channel resolves, so there is no honest value to
                        # store and guessing one would price chemistry by whichever row the store
                        # wrote last. A wiring error, raised where it is cheapest to fix.
                        raise ValueError(
                            f"channel {channel!r} gave {values.tolist()} for one disconnection "
                            f"(application {app}, rule {rule}) and declared no `Reduce`, so the "
                            "store cannot collapse it. Measurements that can disagree have to say "
                            f"how they resolve; channels missing one: {sorted(undeclared)}."
                        )
            # With every disagreement resolved the duplicate rows are identical, so one survives.
            templates = np.unique(templates, axis=0)
            keys = templates[:, 1:3]
        # Store only what was examined. The relation is sparse on purpose: an absent row reads
        # `UNPROBED` and costs nothing, so "nobody looked" needs no encoding of its own, and a row
        # that *is* there carries a real answer -- including `SILENT`, which is an answer.
        selectivity = {}
        for position, channel in enumerate(measured):
            counts = templates[:, 3 + position]
            keep = counts >= 0
            selectivity[channel] = (
                templates[keep, 1],
                templates[keep, 2],
                counts[keep],
            )
    return LayerTables(
        edges,
        (templates[:, 0], templates[:, 1], templates[:, 2]),
        selectivity,
        disputed,
    )


def columns_of(
    rows: Sequence[tuple[int, int, int]],
) -> tuple[SidArray, SidArray, SidArray]:
    """Transpose ``(a, b, c)`` rows into three parallel int64 columns."""
    table = np.array(rows, dtype=np.int64).reshape(-1, 3)
    return table[:, 0], table[:, 1], table[:, 2]

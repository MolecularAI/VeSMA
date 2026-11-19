"""The expansion fold, and the two arrows every measurement on a layer is.

A retro split produces precursors, and several separate things then get computed about them. They
look unrelated until you ask which *axis* each one lives on, and then there are only two:

    MoleculeScore[M] = MoleculeBatch[M]                   -> Column   # one per precursor row
    ReactionScore[M] = (products, precursors, HitArray)   -> Column   # one per hit row

Everything a search wants to know about a layer is one of those. An identity key is a
``MoleculeScore`` whose column happens to hold tokens (:class:`vesma.keyer.Keyer`). A stock distance
is a ``MoleculeScore`` with an extra contract on top (:class:`vertsy.heuristics.Heuristic`: it must
vanish at the stock). A forward-selectivity count is a ``ReactionScore``
(:mod:`vertsy.selectivity`). :func:`measured` is the one place any of them goes in, and
:class:`Expansion` carries them under names on the axis they belong to.

**Composition is ordinary function application, and stages are ordinary functions.** A score is
rarely one kernel: a stock distance is *fingerprint then scan*, and a selectivity count is *group,
combine, match, key, count*. Those stages are separately swappable because they are separate
arguments -- ``NearestStock(fingerprint=..., scan=...)`` says which two kernels, and each can move to
a device without the other knowing. There is deliberately no ``>>`` operator and no pipeline object:
in Python that means a wrapper class with dunder methods, which buys nothing a keyword argument does
not already say, and costs the reader an indirection.

**What is not in this algebra, and cannot be.** Aromaticity, hydrogen counts and ring descriptors
are decided per *assignment* inside the split -- before a precursor row exists at all -- so they are
fused into the matcher (:class:`vesma.reaction.ReactionProcessing`) rather than composed onto its
output. That is the one real boundary here, and it is a boundary of fact: there is no batch to pass
because the batch is what perception produces.

**A column is not an array.** :class:`Column` promises a length and selection by an index array, the
same two operations :class:`MoleculeBatch` promises, which is what lets a measurement live wherever
the molecules it describes live. A device backend hands back a column of numbers and never a
molecule; whoever needs host arithmetic converts at its own boundary and says so.
"""

from __future__ import annotations

from collections.abc import Callable, Container, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Generic, Protocol, cast

import numpy as np

from .core import HitArray, K, M
from .expansions import Expansions, Reduce
from .interner import SidArray

__all__ = [
    "Column",
    "Expansion",
    "Expansions",
    "Layer",
    "Match",
    "MoleculeBatch",
    "MoleculeScore",
    "ReactionScore",
    "concat_object_batches",
    "expansion",
    "measured",
    "init",
    "object_mol_array",
    "one_per_sid",
    "scored_with",
    "search",
    "step",
]


# --------------------------------------------------------------- frontier container


class MoleculeBatch(Protocol[M]):
    def __len__(self) -> int: ...

    def __getitem__(self, selector: Any) -> MoleculeBatch[M]: ...


def object_mol_array(items: Sequence[M]) -> MoleculeBatch[M]:
    """Default `stack_mols`: pack molecule objects into a 1-D object array.

    Uses explicit element assignment so numpy never tries to nest array-like
    molecules into a 2-D array. Supports `arr[bool_mask]` / `arr[int_array]`
    selection, matching the `CudaMoleculeBatch.__getitem__` device gather.
    """
    array = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        array[index] = item
    return cast(MoleculeBatch[M], array)


def concat_object_batches(parts: Sequence[MoleculeBatch[M]]) -> MoleculeBatch[M]:
    """Default `concat_mols`: join object-array batches end to end.

    The inverse of splitting a batch, and the other half of what a search needs to carry a
    frontier: `object_mol_array` builds one, `batch[selector]` narrows one, this joins them.
    A device backend supplies its own (`CudaMoleculeBatch.from_parts`), which is why this is
    injected rather than a method — an object array cannot grow one.
    """
    if not parts:
        return cast(MoleculeBatch[M], np.empty(0, dtype=object))
    return cast(MoleculeBatch[M], np.concatenate([cast(Any, part) for part in parts]))


def init(
    targets: Iterable[str],
    *,
    from_smiles_mol: Callable[[str], M],
    from_smiles_key: Callable[[str], K],
    stack_mols: Callable[[Sequence[M]], MoleculeBatch[M]] = object_mol_array,
) -> tuple[list[K], MoleculeBatch[M]]:
    """Parse target SMILES into the (keys, mols) frontier the search folds over.

    A pure front door: no stock policy here — `search` decides what to expand. Keys
    are an ordinary sequence parallel to the backend-specific molecule container.

    targets         — target SMILES strings to retrosynthesise.
    from_smiles_mol — parse a SMILES to the molecule type M expected by match.
    from_smiles_key — encode a SMILES directly to a key K (faster than mol→key).
    stack_mols      — build the molecule container from a list of M (backend-specific;
                      defaults to a numpy object array).
    """
    targets = list(targets)
    keys = [from_smiles_key(smi) for smi in targets]
    mols = stack_mols([from_smiles_mol(smi) for smi in targets])
    return keys, mols


# ------------------------------------------------------------------- the search fold


class Column(Protocol):
    """A column of numbers parallel to one of an expansion's two axes.

    The same two operations :class:`MoleculeBatch` promises -- a length and selection by an index
    array -- and nothing more, which is the whole point: a column can then live wherever the
    molecules it parallels live, and a device backend can hand one back without materialising
    anything on the host. Deliberately not ``np.ndarray``, though one satisfies this; a consumer
    that needs arithmetic converts at its own boundary and says so, and a backend whose column
    cannot be converted is a backend that must supply a host column for that channel.

    ``__getitem__`` is typed loosely on purpose. Narrowing a column gives a column and indexing one
    position gives a scalar, and promising the former would be a lie for every array in numpy.
    """

    def __len__(self) -> int: ...

    def __getitem__(self, selector: Any) -> Any: ...


# Something worth knowing about each precursor the matcher returned.
type MoleculeScore[M] = Callable[[MoleculeBatch[M]], Column]
# Something worth knowing about each application it found, as one value per hit row.
#
# **Both sides of the application, because an application is a relation and not a molecule.** The
# precursors alone answer "is this reagent pair incompatible" and "how many products does the
# forward rule make", which is why this took one batch for as long as those were the only questions.
# They cannot answer anything comparing the two sides -- mass balance is the plain case, and a
# template that drops a mapped atom is invisible from the precursors alone, which is exactly the
# defect `vertsy.filters` exists to catch. `products` is the frontier the matcher was handed, which
# a hit row's column 0 indexes; `precursors` is what came back, which column 3 indexes.
type ReactionScore[M] = Callable[[MoleculeBatch[M], MoleculeBatch[M], HitArray], Column]


@dataclass(frozen=True)
class Expansion(Generic[M, K]):
    """Everything one match call found, and whatever anyone measured about it.

    Two score maps because the payload has exactly **two id spaces** and no third.
    ``molecule_scores`` are parallel to ``keys`` -- one value per precursor row, the axis the batch
    and the keyer agree on and the axis :func:`step` masks along. ``reaction_scores`` are parallel
    to ``hits`` -- one value per hit row, which is the axis a matcher already returns, the axis
    :func:`vertsy.filters.refused` already gathers on, and the axis
    :func:`vertsy.expansions.concat_hits` needs no rebasing for. Per hit is redundant by an
    application's arity, two or three rows, and buys the property that no wrapper needs to know
    what an application id is.

    Named maps rather than fixed fields, because the interesting scores are not knowable in
    advance. A stock distance and a forward-selectivity count are the two that exist; over-reaction
    and a negative-condition check are the same shape, and so is anything learned. A name that is
    **absent** is a measurement nobody took, which is not the same claim as a measurement that came
    back neutral -- and keeping those distinguishable is what lets an unprobed rule set cost exactly
    what it costs today rather than being quietly declared clean.
    """

    precursors: MoleculeBatch[M]
    keys: Sequence[K]
    hits: HitArray
    molecule_scores: Mapping[str, Column] = field(default_factory=dict)
    reaction_scores: Mapping[str, Column] = field(default_factory=dict)
    reaction_reduce: Mapping[str, Reduce] = field(default_factory=dict)
    """Per reaction channel, how several values for one disconnection resolve.

    Travels **with** the column, because the question is what that column *means*: a count that
    contradicts itself measured nothing, a verdict that contradicts itself still found a refusal.
    The store collapses matcher applications onto disconnections and so is where the question
    arises, but it cannot be where the answer lives -- it would have to know a count from a verdict.

    A channel with no entry here is one that claims it cannot disagree, and the store raises if it
    does, which is the honest failure for a wiring error.
    """


@dataclass(frozen=True)
class Layer(Generic[M]):
    """The fresh frontier one expansion produced, and what was known about it at birth.

    A named value rather than a wider tuple, because only :func:`vertsy.aostar.aostar` reads the
    third field and :func:`search` should not have to skip it. The asymmetry with an expansion's
    other channel is deliberate and worth stating: an estimate is about a *molecule* the frontier
    will carry onward, so it travels with the frontier; a forward measurement is about an
    *application* the graph keeps, so it goes to the store and is never seen again up here.
    """

    sids: SidArray
    mols: MoleculeBatch[M]
    estimate: Column | None = None


# A match callable: frontier container -> everything found about one layer.
# Implicit-generic over the repr M; plain assignment so the free M parametrizes it.
Match = Callable[[MoleculeBatch[M]], Expansion[M, K]]


def expansion(match: Callable[..., Any]) -> Match[M, K]:
    """Lift a backend's ``(precursors, keys, hits)`` into the payload a search folds over.

    The one line that keeps :mod:`vesma` out of this. A reaction matcher's job ends at "here is
    what the rules did"; which scores a particular search wants alongside that is a property of the
    search, and every backend would otherwise have to learn the vocabulary. So the tuple stays the
    backend seam -- exactly what :data:`vesma.matcher.KeyedMatch` says it is -- and this adapts it.
    """

    def matched(mols: MoleculeBatch[M]) -> Expansion[M, K]:
        precursors, keys, hits = match(mols)
        return Expansion(precursors=precursors, keys=keys, hits=hits)

    return matched


def measured(
    match: Match[M, K],
    *,
    molecules: Mapping[str, MoleculeScore[M]] | None = None,
    reactions: Mapping[str, ReactionScore[M]] | None = None,
    reduce: Mapping[str, Reduce] | None = None,
) -> Match[M, K]:
    """Wrap ``match`` so everything it finds arrives already measured.

    **The one wrapper, and the two axes are its two arguments.** Every measurement a layer can carry
    is one of two arrows -- ``MoleculeBatch -> Column`` on the precursor axis, or
    ``(products, precursors, HitArray) -> Column`` on the application axis -- so there is one
    function here rather than one per kind of thing anybody wants to measure. A stock distance, an identity key, a
    forward-selectivity count and anything learned are all the same shape, and this is where they go
    in.

    A ``Match -> Match`` wrapper, the idiom :func:`vertsy.filters.refused` and
    :func:`vertsy.trace.traced` already use on this callable, so a search that wants no measurements
    never runs one and several compose either by nesting or -- better -- by arriving together, since
    a mapping takes as many as you like.

    What is deliberately *not* expressible here is perception: aromaticity, hydrogen counts and ring
    descriptors are decided per *assignment* inside the split, before a precursor row exists at all,
    so they are fused into the matcher rather than composed onto its output. That is the one real
    boundary in this algebra, and it is a boundary of fact rather than of taste.
    """

    def matched(mols: MoleculeBatch[M]) -> Expansion[M, K]:
        found = match(mols)
        return scored_with(
            found,
            molecules={
                name: score(found.precursors)
                for name, score in (molecules or {}).items()
            },
            reactions={
                name: score(mols, found.precursors, found.hits)
                for name, score in (reactions or {}).items()
            },
            reduce=reduce,
        )

    return matched


def scored_with(
    found: Expansion[M, K],
    *,
    molecules: Mapping[str, Column] | None = None,
    reactions: Mapping[str, Column] | None = None,
    reduce: Mapping[str, Reduce] | None = None,
) -> Expansion[M, K]:
    """Add named columns to an expansion, refusing to overwrite one that is already there.

    Two wrappers measuring the same thing under one name is a composition nobody has defined --
    which of them wins is not a question with a good answer -- so it raises rather than picking.
    """
    for axis, incoming, existing in (
        ("molecule", molecules or {}, found.molecule_scores),
        ("reaction", reactions or {}, found.reaction_scores),
    ):
        clash = set(incoming) & set(existing)
        if clash:
            raise ValueError(
                f"{axis} score {sorted(clash)} was already measured by another wrapper"
            )
    return replace(
        found,
        molecule_scores={**found.molecule_scores, **(molecules or {})},
        reaction_scores={**found.reaction_scores, **(reactions or {})},
        reaction_reduce={**found.reaction_reduce, **(reduce or {})},
    )


def one_per_sid(sids: SidArray, mask: np.ndarray) -> SidArray:
    """Indices of one representative per distinct sid among the masked positions.

    Which representative is unspecified (whatever ``np.unique`` returns) — all positions
    with the same sid refer to the same molecule, so any one is interchangeable.
    """
    candidates = np.nonzero(mask)[0]
    if candidates.size == 0:
        return candidates
    _, representative = np.unique(sids[candidates], return_index=True)
    return candidates[representative]


def step(
    frontier_sids: SidArray,
    frontier_mols: MoleculeBatch[M],
    expansions: Expansions[K],
    stock: Container[K],
    *,
    match: Match[M, K],
    estimate: str = "estimate",
) -> Layer[M]:
    """One expansion transition, carried entirely as parallel ``(sids, mols)`` arrays.

    Match the frontier, intern the precursors to sids, append the layer's rows, then keep the fresh
    ones — unseen, not stock, one representative per distinct sid. Pure array algebra: two boolean
    masks and a gather. ``expansions`` grows in place.

    Everything here works in *precursor batch* space, so a matcher that deduplicated its batch
    simply presents fewer rows to intern and mask; only ``add_layer`` crosses back into hit space.

    ``estimate`` names the molecule column this function carries onward, and is a parameter rather
    than a constant so no magic string hides in here. It rides the same ``fresh`` mask the molecules
    do, so a score computed off the very batch the matcher returned stays paired with the molecule
    it was computed for.

    **Every reaction score is persisted, under its own name.** The reaction axis *is* the
    measurement axis: a per-hit-row column is a statement about an application, which is exactly
    what the store's measurement channels hold, so there is no list of blessed names here and no way
    for a probe to measure something the graph then loses. A layer nobody measured has no columns
    and writes no rows, which is the behaviour that existed before any of this.
    """
    found = match(frontier_mols)
    precursor_sids = expansions.intern(found.keys)
    expansions.add_layer(
        frontier_sids,
        found.hits,
        precursor_sids,
        {
            channel: np.asarray(column, dtype=np.int64)
            for channel, column in found.reaction_scores.items()
        },
        found.reaction_reduce,
    )

    in_stock = np.fromiter(
        (key in stock for key in found.keys),
        dtype=np.bool_,
        count=len(found.keys),
    )
    fresh = one_per_sid(
        precursor_sids, ~expansions.is_expanded(precursor_sids) & ~in_stock
    )
    scored = found.molecule_scores.get(estimate)
    return Layer(
        sids=precursor_sids[fresh],
        mols=found.precursors[fresh],
        estimate=None if scored is None else scored[fresh],
    )


def search(
    frontier_keys: Sequence[K],
    frontier_mols: MoleculeBatch[M],
    stock: Container[K],
    max_depth: int,
    expansions: Expansions[K],
    *,
    match: Match[M, K],
) -> Expansions[K]:
    """Fold :func:`step` over the frontier, up to ``max_depth`` times.

    Keys are interned once here; everything past this point is sid space. The caller
    injects ``expansions`` (and thus the store backend); pass an existing one to extend
    it (restart). Frontier keys already in ``stock`` are 0-depth buyable leaves: marked
    evaluated (no steps) and never expanded.
    """
    sids = expansions.intern(frontier_keys)
    in_stock = np.fromiter(
        (key in stock for key in frontier_keys),
        dtype=np.bool_,
        count=len(frontier_keys),
    )
    expansions.mark_expanded(sids[in_stock])
    frontier_sids = sids[~in_stock]
    mols = frontier_mols[~in_stock]

    for _ in range(max_depth):
        if not len(frontier_sids):
            break
        layer = step(frontier_sids, mols, expansions, stock, match=match)
        frontier_sids, mols = layer.sids, layer.mols
    return expansions

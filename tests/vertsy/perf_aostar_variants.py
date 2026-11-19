"""Exploratory grid over AO* variants: which effort shape, which ranking, which weight.

Not a benchmark — a notebook that happens to be a test. It writes no files and pins no
numbers; it prints a table so a human can decide what is worth turning into one. Promote a
finding to ``bench_aostar.py`` once it has stopped moving.

The headline question is the *shape* of an effort rather than its size. ``16x8`` and ``8x16``
both expand 128 products: the wide one explores in parallel, the tall one exploits its leaders
and can reach deeper. Nothing about a single expansion budget can express that difference, which
is why the two numbers are separate — and why every row reports what
:mod:`vertsy.effort` saw, so a run says which of its two axes actually bound it.

The three selections it compares are the two halves of one value. ``total`` is the faithful
``delta + rn`` and the only rank with the k=1 AO* guarantee; ``inside`` and ``context`` rank on
one half each and exist to price the context sweep — if ``inside`` searches nearly as well, the
extra top-down pass is not earning its keep. ``descent`` is textbook AO*, kept as a control that
cannot fill a batch at all.

``AOSTAR_STEP_COST`` is where the free parameter lives, not ``AOSTAR_WEIGHT``. Min-plus ranking is
invariant under scaling every cost, so a cheap step against a ``[0, 1]`` heuristic is the same search
as a unit step against a scaled-up one — and the first keeps a fingerprint distance inside its own
range instead of stretching it to 4. Leave the weight at 1 and make the step cheaper.

**The default row is a step of 0.1 against a unit weight**, which is the ratio the sweep settled on
and then replicated on a fresh sample: 48 seeded n1 targets over reference depths 5 to 7 at 32x8
solve 34/48 at a step of 1.0 and 44/48 at 0.1, the margin widening with depth (12→14, 12→15, 10→15).
Below about 0.05 the solve rate holds but route quality starts being paid for -- 3.38 reactions
against 3.25, and `fresh` climbing past three quarters as the search dives.

Reference depth is not difficulty. The search does not follow the reference route, so a depth-7
bucket solves 15/16 and depth 8 solves 16/16, returning routes about two reactions shorter than the
reference throughout. What grows with reference depth is `1st`: 34, 43, 96, 144 expansions to the
first route at depths 5 to 8.

``AOSTAR_SCHEDULE`` chooses whether the step cost is held or annealed. ``flat`` is a constant
``AOSTAR_STEP_COST``; ``anytime`` searches greedily at that cost until a route exists, then relaxes
to 1.0 for the remainder. The budget after the first route was otherwise buying nothing measurable —
ways to begin, reaction count and longest sequence are all flat from 128 to 512 expansions — and
annealing turned that tail into 3.4x the distinct first disconnections at identical expansions.

``AOSTAR_FILTER`` drops disconnections before the store sees them. ``mass`` rejects any whose
precursors are smaller than their product, which real chemistry cannot do: 71 of raw68k's 67,557
templates map a product atom no precursor mentions, so RDKit deletes it and everything attached, and
a search hunts for such rules because discarding most of a molecule lands straight in the catalogue.
Four SynthEx natural products "solved" in two or three steps that way and solve 0/2 with it on.
Defaults to ``none``, because dropping disconnections changes which library a benchmark ran against.

``AOSTAR_CORPUS`` selects targets *and* their stock together, which is a correctness matter: n1 and
n5 share only 6,415 of ~13.4k stock molecules, so the pairing must not be separable.

``AOSTAR_FINGERPRINT`` and ``AOSTAR_FLOOR`` exist for one measured failure. Binary ``ecfp4`` is a
*set* of circular environments, so every interior methylene of a chain contributes the same
radius-2 environment and past about five carbons chain length is invisible: ``C=CCCCCCCO`` and
``C=CCCCCCCCCCO`` share all sixteen bits. On a Retro* Hard target that put 127 of every 128 picks
at an estimate of exactly 0 -- molecules the heuristic called finished and the catalogue did not
contain -- for thirteen consecutive passes. ``ecfp4-count`` keeps the multiplicity and separates
them. ``AOSTAR_FLOOR`` raises every estimate to at least that much, which is admissible for a
non-stock molecule since it needs at least one reaction; it can only affect leaves already
*below* the floor, so it is a no-op wherever the collisions are not there to begin with. Both
default to the earlier behaviour so every row already in this file stays comparable.

Configure with ``AOSTAR_EFFORT`` (``WIDTHxPASSES`` shapes), ``AOSTAR_SELECT``, ``AOSTAR_FOLD``,
``AOSTAR_DEPTHS``, ``AOSTAR_CORPUS``, ``AOSTAR_TARGETS``, ``AOSTAR_SEED``, ``AOSTAR_RULES``,
``AOSTAR_JOBS``, ``AOSTAR_STEP_COST``, ``AOSTAR_SCHEDULE``, ``AOSTAR_FILTER``, ``AOSTAR_WEIGHT``,
``AOSTAR_FINGERPRINT``, ``AOSTAR_FLOOR``, ``AOSTAR_CAP``.
Run it with::

    .venv/bin/pytest tests/vertsy/perf_aostar_variants.py -s -m perf
"""

from __future__ import annotations

import csv
import gzip
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache, partial
from importlib import import_module
from time import perf_counter
from typing import Any

import numpy as np
import pytest

pytest.importorskip("rdkit.Chem")

from rdkit import Chem

from tests.datadirs import external_data_root
from vertsy.aostar import (
    anytime_step,
    aostar,
    bottleneck_fold,
    by_context,
    by_inside,
    by_total,
    closure_fold,
    constant_step,
    descent_leaves,
    global_leaves,
)
from vertsy.audit import floored
from vertsy.effort import (
    Round,
    descent,
    first_route,
    kept_at,
    per_reaction,
    pick_order,
    reached,
    recording_select,
    retention,
    steering,
)
from vertsy.expansions import Expansions
from vertsy.interner import MISSING_SID, DictInterner
from vertsy.interop import aizynth_depth
from vertsy.metrics import (
    GraphState,
    Valuation,
    min_route_depth,
    precursor_route_count_bounded,
    prepare_graph,
    route_solved,
    solved_disconnection_count,
    solved_route_count_bounded,
    starting_material_route_count_bounded,
    valuation,
)
from vertsy.routes import reaction_count, reaction_skeleton
from vertsy.search import (
    concat_object_batches,
    expansion,
    object_mol_array,
)
from vertsy.similarity import stock_similarity
from vertsy.stock import load_stock
from vertsy.store.numpy_store import NumpyExpansionStore
from vertsy.trace import Event, phase_seconds, traced
from vertsy.tropical import tropical_valuation
from vesma.keyer import rdkit_mol_canonical_smiles, rdkit_smiles_keyer
from vesma.rdkit_matcher import (
    RdkitFilterCatalogReactionMatcher,
    rdkit_reactions_from_smarts,
)

pytestmark = [pytest.mark.perf, pytest.mark.external_data]

PAROUTES = external_data_root() / "external" / "paroutes" / "processed"
SELECTORS = {
    "total": partial(global_leaves, rank=by_total),
    "inside": partial(global_leaves, rank=by_inside),
    "context": partial(global_leaves, rank=by_context),
    "descent": descent_leaves,
}
RULE_FILES = {
    "used14k": ("used14k-reactions.tsv.gz", "retro_template"),
    "raw68k": ("raw68k-templates.tsv.gz", "retro_template"),
}
# Min-plus either way. `array` values every product at every horizon in one sweep of numpy;
# `closure` is the generic semiring fold every route metric uses, kept as the control.
# `max` prices an application by its worst precursor rather than their sum, which makes a
# route's value its longest linear sequence and stops a bimolecular split reading as uphill.
FOLDS = {
    "array": tropical_valuation,
    "closure": closure_fold,
    "max": bottleneck_fold,
}


def env_strs(name: str, default: str) -> list[str]:
    return [part for part in (os.environ.get(name) or default).split(",") if part]


def env_ints(name: str, default: str) -> list[int]:
    return [int(part) for part in env_strs(name, default)]


def env_floats(name: str, default: str) -> list[float]:
    return [float(part) for part in env_strs(name, default)]


def env_shapes(name: str, default: str) -> list[tuple[int, int]]:
    """``"16x8,8x16"`` -> ``[(16, 8), (8, 16)]``."""
    shapes = []
    for part in env_strs(name, default):
        width, _, passes = part.partition("x")
        shapes.append((int(width), int(passes)))
    return shapes


@lru_cache
def rule_smarts(rule_set: str) -> tuple[str, ...]:
    filename, column = RULE_FILES[rule_set]
    path = PAROUTES / filename
    if not path.exists():
        pytest.skip(f"missing rule set: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return tuple(
            sorted({row[column] for row in csv.DictReader(handle, delimiter="\t")})
        )


@lru_cache
def matcher_for(rule_set: str, n_jobs: int) -> RdkitFilterCatalogReactionMatcher:
    """Built once per process: 15 s of compilation must not land inside a measurement."""
    smarts = rule_smarts(rule_set)
    return RdkitFilterCatalogReactionMatcher(
        rdkit_reactions_from_smarts(smarts),
        [text.split(">>")[0] for text in smarts],
        n_jobs=n_jobs,
        key=lambda mols: [rdkit_mol_canonical_smiles(mol) for mol in mols],
    )


# A corpus is targets *and* the stock they are priced against, and the two must never come from
# different ones: n1 and n5 share only 6,415 of ~13.4k stock molecules, so an n5 target searched
# against n1's catalogue is measuring nothing at all. Every accessor below takes the corpus, so the
# pairing cannot drift.
@lru_cache
def stock_smiles(corpus: str) -> tuple[str, ...]:
    path = PAROUTES / f"{corpus}-stock.smi.gz"
    if not path.exists():
        pytest.skip(f"missing dataset: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return tuple(line.split()[0] for line in handle if line.strip())


@lru_cache
def stock_keys(corpus: str) -> frozenset[str]:
    return frozenset(
        load_stock(PAROUTES / f"{corpus}-stock.smi.gz", rdkit_smiles_keyer)
    )


@lru_cache
def heuristic_for(corpus: str, weight: float, fingerprint: str = "ecfp4"):
    """One stock matrix per (corpus, weight, fingerprint); ~110 MB each, so cached not rebuilt."""
    return stock_similarity(
        stock_smiles(corpus), weight=weight, fingerprint=fingerprint
    )


@lru_cache
def targets_by_depth(corpus: str) -> dict[int, tuple[str, ...]]:
    """Targets bucketed by their reference route's reaction depth.

    Derived rather than read from ``{corpus}-dN-f0.smi.gz``: those files are the same construction —
    the routes of one depth, reduced to their targets — so bucketing here reaches every depth
    without new data files. n1 has 590 targets at depth 5, n5 has 1,256.
    """
    path = PAROUTES / f"{corpus}-routes.json.gz"
    if not path.exists():
        pytest.skip(f"missing dataset: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        references = json.load(handle)
    buckets: dict[int, list[str]] = {}
    for reference in references:
        buckets.setdefault(aizynth_depth(reference), []).append(reference["smiles"])
    return {depth: tuple(sorted(smiles)) for depth, smiles in sorted(buckets.items())}


@lru_cache
def reference_intermediates(corpus: str) -> dict[str, tuple[str, ...]]:
    """Target SMILES -> the non-target molecules of its reference route, stock included.

    Solving a target and recovering the chemist's route are different achievements, and until now
    only the first had a column. This is the cheap half of the second: the molecules the recorded
    route passes through, so a run can report how many of them it ever built.

    Deliberately *not* strict recovery. That needs the reference route's rule ids, and this file
    numbers rules by sorted-unique SMARTS while the recovery benchmark numbers them by file order
    over cleaned SMARTS — the same template gets different ids, so a naive comparison would be
    quietly wrong. Molecule identity is scheme-independent, which is why it is what gets reported.
    """
    path = PAROUTES / f"{corpus}-routes.json.gz"
    if not path.exists():
        pytest.skip(f"missing dataset: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        references = json.load(handle)

    found: dict[str, tuple[str, ...]] = {}
    for reference in references:
        molecules: list[str] = []
        stack = list(reference.get("children", ()))
        while stack:
            node = stack.pop()
            if node["type"] == "mol":
                molecules.append(node["smiles"])
            stack.extend(node.get("children", ()))
        found[reference["smiles"]] = tuple(dict.fromkeys(molecules))
    return found


def reference_coverage(
    smiles: str, corpus: str, expansions: Expansions[str], state: GraphState[str]
) -> tuple[float, float]:
    """``(fraction of reference intermediates the graph holds, fraction of those it solved)``.

    Two numbers because they fail differently. A molecule the search never interned was never even
    proposed by a template it tried; one it interned but did not solve was proposed and abandoned.
    """
    wanted = [
        rdkit_smiles_keyer.from_smiles(text)
        for text in reference_intermediates(corpus).get(smiles, ())
    ]
    if not wanted:
        return (float("nan"), float("nan"))
    sids = [int(sid) for sid in expansions.lookup(wanted).tolist()]
    present = [sid for sid in sids if sid != int(MISSING_SID)]
    solved = sum(1 for sid in present if state.is_solved(sid))
    return (len(present) / len(wanted), solved / len(wanted))


def sampled_targets(corpus: str, depth: int, count: int, seed: int) -> list[str]:
    """A seeded sample of the depth bucket, not its alphabetical prefix.

    The prefix is what this used to take, and it is not a sample: SMILES sort order tracks the
    leading atom and so the composition, which biased every deep-target claim made from three of
    them. Seeded so a row is reproducible without being systematic.
    """
    bucket = targets_by_depth(corpus).get(depth, ())
    if not bucket:
        pytest.skip(f"no {corpus} reference routes at depth {depth}")
    take = min(count, len(bucket))
    chosen = np.random.default_rng(seed).choice(len(bucket), take, replace=False)
    return [bucket[int(index)] for index in sorted(chosen.tolist())]


@dataclass
class Probe:
    """Where the patched blocks report to; ``run_one`` re-points it per run."""

    trace: list[Event] = field(default_factory=list)
    node_calls: int = 0
    application_calls: int = 0


PROBE = Probe()


def timed_prepare_graph(expansions, stock):
    """The real thing, timed. Monkeypatched in because `aostar` calls it directly."""
    started = perf_counter()
    state = prepare_graph(expansions, stock)
    PROBE.trace.append(("prepare", len(state.steps), perf_counter() - started))
    return state


def counted_valuation(state, semiring, **kwargs):
    """The real fold, with calls *into* it counted.

    Timing ``valuation`` itself says nothing — it returns closures and does no work until they
    are called. This counts the calls selection makes, not the fold's own recursion, which is
    closure-private and memoized. That is the useful half: an unmemoized ``application`` is
    re-walked on every one of these, and the sibling sums in the outside sweep are quadratic in
    arity, so this number *is* the cost the sweep imposes on the fold.
    """
    folded = valuation(state, semiring, **kwargs)

    def node(product: int, depth_left: int):
        PROBE.node_calls += 1
        return folded.node(product, depth_left)

    def application(step, depth_left: int):
        PROBE.application_calls += 1
        return folded.application(step, depth_left)

    # Pass the fold's own `conjoin` through: the outside sweep reads it off the value, so dropping
    # it here would silently switch a `max` fold back to summing its siblings.
    return Valuation(node=node, application=application, conjoin=folded.conjoin)


@dataclass(frozen=True)
class Run:
    """One target under one variant."""

    solved: bool
    expansions: int
    passes: int
    seconds: float
    phases: dict[str, float]
    reactions: int | None
    route_depth: int | None
    routes: int
    precursor_routes: int
    saturated: bool
    material_sets: int
    sets_saturated: bool
    signatures: int
    ways: int
    fresh: int
    picks: int
    kept_width: list[float]
    kept_passes: list[float]
    first: int | None
    priced: int
    applications: int
    node_calls: int
    application_calls: int
    context: float
    estimate: float
    drop: float
    ref_found: float
    ref_solved: float
    depths: Counter[int]


def run_one(
    smiles: str,
    *,
    corpus: str,
    matcher: Any,
    heuristic: Any,
    select: Any,
    fold: Any,
    step_cost: float,
    schedule: str,
    template_filter: str,
    width: int,
    passes: int,
    cap: int,
    set_cap: int,
) -> Run:
    trace: list[Event] = []
    PROBE.trace = trace
    PROBE.node_calls = 0
    PROBE.application_calls = 0
    rounds: list[Round] = []
    expansions = Expansions(NumpyExpansionStore(), DictInterner())
    target = rdkit_smiles_keyer.from_smiles(smiles)
    # Warmed *before* the clock, the same reason the matcher is built once per process: keying
    # 13.4k stock molecules takes seconds, and as an argument to `aostar` it would land inside the
    # timed region and be charged to the first row of every new corpus.
    stock = stock_keys(corpus)
    molecule = Chem.MolFromSmiles(smiles)

    # A filter drops whole disconnections before the store sees them. `mass` rejects any whose
    # precursors are smaller than their product, which cannot happen in real chemistry: 71 of
    # raw68k's 67,557 templates map a product atom no precursor mentions, and a search hunts for
    # them because throwing most of a molecule away lands straight in the catalogue. Four SynthEx
    # natural products "solved" in 2-3 steps that way, and solve 0/2 once it is on.
    matching = expansion(matcher.match_keyed)
    if template_filter == "mass":
        raise ValueError(
            "AOSTAR_FILTER=mass is gone with the mass veto it drove. 31% of the "
            "hand-written templates lose heavy atoms legitimately -- a reagent supplies "
            "them -- so the signal was wrong, and the defect it reached for is now "
            "refused at the root by `precursors_from_mapping`, which needs no filter."
        )

    # `anytime` searches greedily until a route exists, then relaxes the step cost to 1.0 and
    # spends the rest of the budget broadening. Measured on six n1 depth-5 targets at 32x16: ways to
    # begin summed 231 flat against 783 scheduled, same 481 expansions, all still solved. The cost
    # is wall clock rather than expansions -- a broad phase interns more distinct molecules.
    if schedule == "anytime":
        price, scheduled = anytime_step(greedy=step_cost, relaxed=1.0, select=select)
    else:
        price, scheduled = constant_step(step_cost), select

    started = perf_counter()
    routes = aostar(
        target,
        target_mol=molecule,
        stock=stock,
        expansions=expansions,
        heuristic=traced(trace, "heuristic")(heuristic),
        match=traced(trace, "match")(matching),
        stack_mols=object_mol_array,
        concat_mols=concat_object_batches,
        # `traced` goes *inside* `recording_select`, not outside: the recorder runs a second sweep
        # to note why each leaf was chosen, and wrapping the other way round would bill that to
        # selection and silently double its reported share.
        select=recording_select(traced(trace, "select")(scheduled), rounds),
        price=price,
        fold=traced(trace, "fold")(fold),
        width=width,
        passes=passes,
    )
    elapsed = perf_counter() - started

    # Everything below is scoring, so it lands after the clock stops. Route properties come off
    # the graph: no enumeration beyond the routes AO* already returned.
    state = prepare_graph(expansions, stock)
    counts = retention(routes, expansions, reached(rounds))
    priced = sum(counts.values())
    context, estimate, depths = steering(rounds)
    drop = per_reaction(descent(rounds))
    ref_found, ref_solved = reference_coverage(smiles, corpus, expansions, state)
    kept_width, kept_passes = kept_at(counts, width=width, passes=passes)
    rule_aware = solved_route_count_bounded(target, cap, passes)(state)
    by_precursors = precursor_route_count_bounded(target, cap, passes)(state)
    material_sets = starting_material_route_count_bounded(target, set_cap, passes)(
        state
    )
    return Run(
        solved=bool(route_solved(target)(state)),
        expansions=int(expansions.store.expanded().size),
        passes=len(rounds),
        seconds=elapsed,
        phases=phase_seconds(trace),
        reactions=min((reaction_count(route) for route in routes), default=None),
        route_depth=min_route_depth(target)(state),
        routes=rule_aware,
        precursor_routes=by_precursors,
        # A saturated count is a lower bound, so the ratio of two of them is meaningless — 10000
        # over 10000 reads as "no duplication" when it means "both overflowed".
        saturated=rule_aware >= cap or by_precursors >= cap,
        material_sets=material_sets,
        sets_saturated=material_sets >= set_cap,
        signatures=len({reaction_skeleton(route) for route in routes}),
        # Distinct first disconnections that lead somewhere: the diversity measure that is
        # both meaningful and affordable. `sets` costs ~40 s per target at its 2000 cap and
        # saturates in every row anyway; this is microseconds off a fixpoint already run.
        ways=solved_disconnection_count(target)(state),
        fresh=sum(record.fresh for record in rounds),
        picks=sum(len(record.selected) for record in rounds),
        kept_width=kept_width,
        kept_passes=kept_passes,
        first=first_route(routes, expansions, pick_order(rounds)),
        priced=priced,
        applications=(
            len(np.unique(np.stack(expansions.edges()[:2], axis=1), axis=0))
            if len(expansions.edges()[0])
            else 0
        ),
        node_calls=PROBE.node_calls,
        application_calls=PROBE.application_calls,
        context=context,
        estimate=estimate,
        drop=drop,
        ref_found=ref_found,
        ref_solved=ref_solved,
        depths=depths,
    )


HEADER = (
    f"{'corpus':>6} {'depth':>5} {'select':>8} {'fold':>7} {'step':>5} {'sched':>8} {'filter':>6} "
    f"{'fprint':>11} {'floor':>5} {'shape':>8} "
    f"{'solved':>7} {'expand':>7} {'passes':>6} {'s':>6} "
    f"{'match':>6} {'heur':>5} {'prep':>5} {'bld':>4} {'sel':>4} "
    f"{'rxn':>5} {'deep':>5} {'ways':>5} {'sets':>7} {'apps':>7} "
    f"{'ctx':>6} {'est':>6} {'drop':>6} {'ref':>5} {'refsol':>7} "
    f"{'pickdepth':>18} {'fresh':>6} "
    f"{'1st':>5} {'w 2/4/8':>14} {'p 2/4/8':>14}"
)


def summarize(label: str, runs: list[Run]) -> str:
    solved = [run for run in runs if run.solved]
    total = sum(run.seconds for run in runs)

    def mean(values) -> float:
        # NaN as well as None: a run whose picks all sat at one depth has no gradient to report,
        # and letting one such run poison a whole column would hide every run that does.
        values = [value for value in values if value is not None and value == value]
        return sum(values) / len(values) if values else float("nan")

    def share(name: str) -> float:
        got = sum(run.phases.get(name, 0.0) for run in runs)
        return got / total * 100 if total else 0.0

    picks = sum(run.picks for run in runs) or 1
    sets = mean([run.material_sets for run in solved])
    sets_mark = "+" if any(run.sets_saturated for run in solved) else " "
    # The depth histogram of picks, pooled: the share of picks at each graph depth, shallowest
    # first. This is the column that says whether the ranking ever leaves the top of the graph.
    pooled: Counter[int] = Counter()
    for run in runs:
        pooled.update(run.depths)
    spread = " ".join(
        f"{depth}:{100 * pooled[depth] // max(sum(pooled.values()), 1)}"
        for depth in sorted(pooled)[:5]
    )
    return (
        f"{label} {len(solved):3d}/{len(runs):<3d} "
        f"{mean([r.expansions for r in solved]):7.1f} "
        f"{mean([r.passes for r in solved]):6.1f} "
        f"{mean([r.seconds for r in solved]):6.1f} "
        f"{share('match'):5.0f}% {share('heuristic'):4.0f}% "
        f"{share('prepare'):4.0f}% {share('fold'):3.0f}% {share('select'):3.0f}% "
        f"{mean([r.reactions for r in solved]):5.2f} "
        f"{mean([r.route_depth for r in solved]):5.2f} "
        f"{mean([r.ways for r in solved]):5.0f} "
        f"{sets:6.0f}{sets_mark}"
        f"{mean([r.applications for r in solved]):7.0f} "
        f"{mean([r.context for r in runs]):6.2f} "
        f"{mean([r.estimate for r in runs]):6.2f} "
        f"{mean([r.drop for r in runs]):6.3f} "
        f"{mean([r.ref_found for r in runs]):5.0%} "
        f"{mean([r.ref_solved for r in runs]):7.0%} "
        f"{spread:>18} "
        f"{sum(r.fresh for r in runs) / picks * 100:5.0f}% "
        f"{mean([r.first for r in solved]):5.0f} "
        + " ".join(f"{mean([r.kept_width[d] for r in solved]):4.0%}" for d in range(3))
        + "  "
        + " ".join(f"{mean([r.kept_passes[d] for r in solved]):4.0%}" for d in range(3))
    )


def test_perf_aostar_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    """Print solve rate, effort accounting and route diversity for each variant."""
    rule_set = env_strs("AOSTAR_RULES", "used14k")[0]
    n_jobs = int(os.environ.get("AOSTAR_JOBS") or 4)
    weights = env_floats("AOSTAR_WEIGHT", "1.0")
    step_costs = env_floats("AOSTAR_STEP_COST", "0.1")
    schedules = env_strs("AOSTAR_SCHEDULE", "flat")
    # Defaults to none, for the reason `template_lint` is wired nowhere: dropping
    # disconnections changes which library a benchmark ran against.
    template_filters = env_strs("AOSTAR_FILTER", "none")
    corpora = env_strs("AOSTAR_CORPUS", "n1")
    seed = int(os.environ.get("AOSTAR_SEED") or 0)
    n_targets = int(os.environ.get("AOSTAR_TARGETS") or 4)
    # The two route counts are saturating integer arithmetic, so a huge cap costs nothing and
    # keeps `dup` meaningful. The set-union metric materialises frozensets and is #P-hard to
    # count exactly, so it stays modest and reports saturation with a trailing `+`.
    cap = int(os.environ.get("AOSTAR_CAP") or 10**12)
    set_cap = int(os.environ.get("AOSTAR_SET_CAP") or 2000)
    shapes = env_shapes("AOSTAR_EFFORT", "16x8,8x16")
    selectors = env_strs("AOSTAR_SELECT", "total")
    fold_names = env_strs("AOSTAR_FOLD", "array")
    # `ecfp4` is binary and cannot count repeated environments, so a long chain and a short one
    # of the same ends are one point to it; `ecfp4-count` keeps the multiplicity. `floor` raises
    # every estimate to at least that much, which is admissible for any non-stock molecule and
    # bounded by one step -- so 0.0 keeps every earlier measurement comparable.
    fingerprints = env_strs("AOSTAR_FINGERPRINT", "ecfp4")
    floors = env_floats("AOSTAR_FLOOR", "0.0")
    depths = env_ints("AOSTAR_DEPTHS", "2,3")

    # `aostar` imported `prepare_graph` by value, so its own namespace is the one to patch — and
    # the module has to come from `import_module`, because `vertsy.aostar` as an attribute is the
    # re-exported *function* of the same name.
    aostar_module = import_module("vertsy.aostar")
    monkeypatch.setattr(aostar_module, "prepare_graph", timed_prepare_graph)
    monkeypatch.setattr(aostar_module, "valuation", counted_valuation)
    matcher = matcher_for(rule_set, n_jobs)

    print(
        f"\n{rule_set} ({len(rule_smarts(rule_set))} rules), n_jobs={n_jobs}, "
        f"cap={cap:.0e}/{set_cap}, {n_targets} targets/depth, seed={seed}"
    )
    print(
        "\nstep = cost charged per reaction, with the heuristic left in its own [0, 1]. Only the"
        "\n  ratio matters: step 0.25 is the same search as a unit step against a 4x heuristic, and"
        "\n  a step below the heuristic's range is where admissibility stops."
        "\nctx, est = mean context (delta) against mean estimate over every pick. Their sum is the"
        "\n  sort key, so ctx >> est means the ranking is ordering by committed cost, i.e. depth."
        "\ndrop = how much the mean estimate falls per level of pick depth, i.e. what ONE reaction"
        "\n  is worth to this heuristic on this stock. Compare it against `step`: charging more than"
        "\n  a reaction buys orders the frontier by depth, charging much less leaves the step unable"
        "\n  to break ties. With several heuristics it is how to weight them — scale each so that"
        "\n  what it charges matches what it buys. No labels enter it, only the run's own picks."
        "\nref, refsol = of the reference route's own molecules, the share this run ever built,"
        "\n  and the share it built AND solved. Solving a target and recovering the chemist's route"
        "\n  are different achievements and these separate them: `solved` can be 16/16 while refsol"
        "\n  sits far below it, which means the search found its own route rather than the recorded"
        "\n  one. Molecule identity only — NOT strict recovery, which needs rule ids this file"
        "\n  numbers differently from bench_route_recovery."
        "\npickdepth = share of picks at each graph depth, shallowest first — the visible"
        "\n  consequence of that, and the column that says whether the search leaves the top."
        "\nways = distinct first disconnections of the target that lead to a solved graph -- how"
        "\n  many genuinely different ways there are to BEGIN. Bounded, cheap, and uncapped, unlike"
        "\n  the other two: route counts run past 1e12 and `sets` saturates at its cap in every row."
        "\nsets = distinct starting-material sets, capped and usually saturated; read `ways` instead."
        "\n1st = expansions spent before the FIRST returned route was complete; everything after"
        "\n  it bought alternatives, so it is the number a budget aimed at solve rate is set from."
        "\nw 2/4/8, p 2/4/8 = share of routes still reachable at a half, quarter and eighth of the"
        "\n  width, and of the passes. Marginals, not a joint prediction — a route needs both."
        "\nPhase shares do not sum to 100: the remainder is route extraction and the frontier."
    )
    print(f"\n{HEADER}")
    variants = [
        (
            corpus,
            depth,
            selector,
            fold_name,
            weight,
            step,
            schedule,
            template_filter,
            fingerprint,
            floor,
            shape,
        )
        for corpus in corpora
        for depth in depths
        for selector in selectors
        for fold_name in fold_names
        for weight in weights
        for step in step_costs
        for schedule in schedules
        for template_filter in template_filters
        for fingerprint in fingerprints
        for floor in floors
        for shape in shapes
    ]
    for (
        corpus,
        depth,
        selector,
        fold_name,
        weight,
        step,
        schedule,
        template_filter,
        fingerprint,
        floor,
        (width, passes),
    ) in variants:
        runs = [
            run_one(
                smiles,
                corpus=corpus,
                matcher=matcher,
                heuristic=heuristic_for(corpus, weight, fingerprint),
                select=SELECTORS[selector],
                fold=(
                    FOLDS[fold_name]
                    if not floor
                    else floored(FOLDS[fold_name], floor=floor)
                ),
                step_cost=step,
                schedule=schedule,
                template_filter=template_filter,
                width=width,
                passes=passes,
                cap=cap,
                set_cap=set_cap,
            )
            for smiles in sampled_targets(corpus, depth, n_targets, seed)
        ]
        label = (
            f"{corpus:>6} {depth:5d} {selector:>8} {fold_name:>7} {step:5.3f} "
            f"{schedule:>8} {template_filter:>6} {fingerprint:>11} {floor:5.3f} "
            f"{f'{width}x{passes}':>8}"
        )
        print(summarize(label, runs), flush=True)

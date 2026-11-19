"""Which targets a curated template collection cannot reach, and at what effort.

This asks one question of every target -- *is it reachable at all with these templates* -- and
is built to ask it nine thousand times. The unsolved tail is the output that matters: a target
no amount of effort reaches is a target whose chemistry the collection is missing, and naming
those gaps is the point of the exercise.

How it differs from :mod:`tests.vertsy.bench_host_aostar`, which measures the same engine
-----------------------------------------------------------------------------------------
That bench answers *how good are the routes*, so it pays for routes: extraction, exact route
counting over the whole graph, guided replay, and the forward-selectivity probe. Measured on 30
n1 targets at 8x8, search was 0.49 s a target and the wall clock 9 s -- roughly 95% of it spent
after the search had already decided. None of it changes a solved/unsolved verdict, so none of
it is done here: no extraction, no counting, no replay, no probe. What remains is the search,
one :func:`prepare_graph` and one ``is_solved``.

The other half is :data:`aostar`'s ``stop_when_solved``, which abandons the anytime repair phase
at the first pass that finds the target solved. The width schedule becomes an upper bound rather
than a budget spent, which is why **timings from this bench are not comparable with the paper
benchmarks** -- it is deliberately measuring a different thing.

The ladder
----------
Effort escalates, and each tier runs only what the tier before it left unsolved, so the easy
nine tenths are never searched at an effort they did not need. :func:`escalate` is the whole of
it. The tiers are configurable; the default 8x8 -> 16x16 -> 64x16 spends its budget where the
targets are hard.

Reading the output while it runs
--------------------------------
A sweep of this size is measured in hours, so every attempt is appended to ``attempts.jsonl``
and flushed before the next one starts, and a progress line carries a tier ETA. A run killed
halfway leaves a usable record of everything it reached.
"""

from __future__ import annotations

import gzip
import json
import os
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from shutil import copyfileobj
from time import perf_counter
from typing import Any

import pytest

from tests.vertsy.bench_host_aostar import (
    ARMS,
    GREEDY_STEP,
    PAROUTES,
    RELAXED_STEP,
    STOCK_FILES,
    TEMPLATE_FILES,
    Arm,
    effort_widths,
    env_int,
    env_text,
    load_merged_library,
    merge_stocks,
    parse_efforts,
    parse_inputs,
    prepare_labelled_slices,
    read_targets,
    skip_unless_metal,
    stock_slices,
    template_rows,
)
from tests.vertsy.paper_provenance import host_slug, source_artifact
from vertsy.aostar import (
    anytime_step,
    aostar,
    bottleneck_fold,
    committed_application_leaves,
    global_leaves,
)
from vertsy.costs import constant_step
from vertsy.expansions import Expansions
from vertsy.interner import DictInterner
from vertsy.metrics import prepare_graph
from vertsy.store.numpy_store import NumpyExpansionStore
from vesma import smiles_from_cmol
from vesma.keyer import short_ring_wl_keyer
from vesma.rejection import ChemistryRejection

__all__ = ["Attempt", "attempt_target", "escalate", "eta", "tier_name"]

pytestmark = [
    pytest.mark.perf,
    pytest.mark.external_data,
    pytest.mark.filterwarnings(
        "ignore:Setting the shape on a NumPy array has been deprecated.*:DeprecationWarning:joblib.numpy_pickle"
    ),
]


@dataclass(frozen=True)
class Attempt:
    """One target at one effort. ``solved`` is the whole finding; the rest sizes the work."""

    target: str
    tier: str
    width: int | str
    passes: int
    solved: bool
    seconds: float
    molecules_expanded: int
    molecules_seen: int
    error: str | None = None


def tier_name(width: int | str, passes: int) -> str:
    return f"{width}x{passes}"


def attempt_target(arm: Arm, smiles: str, width: int | str, passes: int) -> Attempt:
    """Search until the target is solved or the budget runs out, and report only which.

    Everything a route report would want is skipped, which is why this is a fifth of a second
    rather than nine seconds. ``max_routes=0`` skips extraction and ``stop_when_solved`` skips
    the repair passes; no price penalties are composed because no route is being ranked.
    """
    widths = effort_widths(width, passes)
    expansions: Expansions[Any] = Expansions(NumpyExpansionStore(), DictInterner())

    def heuristic(molecules: Any) -> Any:
        """A named parameter, which is what the `Heuristic` protocol wants and `Arm` is not."""
        return arm.heuristic(molecules)

    price, scheduled = anytime_step(
        greedy=GREEDY_STEP,
        relaxed=RELAXED_STEP,
        select=committed_application_leaves(fill=global_leaves),
    )
    started = perf_counter()
    try:
        target_key = arm.key_of(smiles)
        target_mol = arm.mol_of(smiles)
    except ChemistryRejection as exc:
        # An input the engine will not read is a finding about the target, not a reason to lose
        # the nine thousand that parse.
        return Attempt(
            target=smiles,
            tier=tier_name(width, passes),
            width=width,
            passes=passes,
            solved=False,
            seconds=perf_counter() - started,
            molecules_expanded=0,
            molecules_seen=0,
            error=f"{type(exc).__name__}: {exc}",
        )

    aostar(
        target_key,
        target_mol=target_mol,
        stock=arm.stock,
        expansions=expansions,
        heuristic=heuristic,
        match=arm.match,
        stack_mols=arm.stack_mols,
        concat_mols=arm.concat_mols,
        width=widths,
        passes=passes,
        max_depth=passes,
        select=scheduled,
        fold=bottleneck_fold,
        price=price,
        max_routes=0,
        stop_when_solved=True,
    )
    elapsed = perf_counter() - started
    state = prepare_graph(expansions, arm.stock)
    target_sid = int(expansions.lookup([target_key])[0])
    return Attempt(
        target=smiles,
        tier=tier_name(width, passes),
        width=width,
        passes=passes,
        solved=bool(state.is_solved(target_sid)),
        seconds=elapsed,
        molecules_expanded=int(expansions.store.expanded().size),
        molecules_seen=expansions.sid_count(),
    )


def escalate(
    targets: Sequence[str],
    tiers: Sequence[tuple[int | str, int]],
    attempt: Callable[[str, int | str, int], Attempt],
) -> Iterator[Attempt]:
    """Run each tier on only what the tier before it left unsolved.

    A generator rather than a list of tiers, so a caller writes and times each attempt as it
    lands instead of after a tier of several thousand finishes. The ladder's only state is the
    pending list, rebuilt from what was just yielded -- which is also what makes it honest: a
    target appears at tier *n+1* exactly when tier *n* failed on it.
    """
    pending = tuple(targets)
    for width, passes in tiers:
        if not pending:
            return
        unsolved: list[str] = []
        for smiles in pending:
            attempted = attempt(smiles, width, passes)
            yield attempted
            if not attempted.solved:
                unsolved.append(smiles)
        pending = tuple(unsolved)


def eta(started: float, done: int, total: int) -> str:
    """Remaining wall clock for the current tier, from its own mean so far."""
    if done == 0:
        return "?"
    remaining = (perf_counter() - started) / done * (total - done)
    hours, seconds = divmod(int(remaining), 3600)
    return (
        f"{hours}h{seconds // 60:02d}m"
        if hours
        else f"{seconds // 60}m{seconds % 60:02d}s"
    )


@dataclass(frozen=True)
class Inputs:
    """What both entry points need: the compiled rules, the catalogue, and a built arm."""

    library: Any
    stock: Any
    arm: Arm
    backend: str
    rule_sets: tuple[tuple[str, Path], ...]
    stocks: tuple[tuple[str, Path], ...]
    targets: tuple[str, ...]
    targets_file: Path
    run_dir: Path


def gather_inputs(targets_default: str) -> Inputs:
    """Resolve the environment into one built, GPU-resident arm and a target list.

    Shared by the sweep and the census so the two cannot drift into describing different runs,
    which is the whole risk when a diagnostic is read beside the numbers it explains.
    """
    rule_sets = parse_inputs(
        env_text(
            "GAPS_RULE_SET",
            "hartenfeller,medchem,protecting-groups,carbohydrate,selenium",
        ),
        TEMPLATE_FILES,
        "GAPS_RULE_SET",
    )
    stocks = parse_inputs(
        env_text("GAPS_STOCK", "n1,common-reagents"), STOCK_FILES, "GAPS_STOCK"
    )
    targets_file = Path(env_text("GAPS_TARGETS_FILE", targets_default)).expanduser()
    missing = [
        path
        for path in (*(p for _, p in rule_sets), *(p for _, p in stocks), targets_file)
        if not path.exists()
    ]
    if missing:
        pytest.skip(f"missing benchmark inputs: {missing}")

    # One backend, named by its own variable rather than the shared `AOSTAR_HOST_BACKENDS`: a
    # sweep this long is never a backend comparison, and sharing the list variable invites one.
    backend = env_text("GAPS_BACKEND", "metal")
    if backend not in ARMS:
        raise ValueError(f"unknown backend {backend!r}; have {sorted(ARMS)}")
    if backend == "metal":
        skip_unless_metal()

    targets = read_targets(
        targets_file,
        env_int("GAPS_TARGET_LIMIT", 0) or 10**9,
        env_int("GAPS_TARGET_OFFSET", 0),
    )
    if not targets:
        pytest.skip(f"no targets read from {targets_file}")

    out_root = Path(
        env_text("GAPS_OUT_DIR", "docs/paper2/benchmarks/template-gaps")
    ).expanduser()
    run_dir = out_root / f"{host_slug()}__{os.environ.get('GAPS_STAMP') or _stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    library = load_merged_library(rule_sets, ())
    slices = stock_slices([path for _, path in stocks], None)
    stock = merge_stocks([part for _, part in prepare_labelled_slices(slices, 1)])
    arm = ARMS[backend](
        library,
        stock,
        heuristic_kind=env_text("GAPS_HEURISTIC", "atoms"),
        scan_kind=env_text("GAPS_SCAN", "metal" if backend == "metal" else "host"),
        weight=1.0,
        n_jobs=1,
    )
    print(
        f"rule_sets={library.source_counts} compiled={len(library.smarts)} "
        f"stock={[n for n, _ in stocks]} rows={stock.row_count} "
        f"backend={backend} targets={len(targets)} run_dir={run_dir}",
        flush=True,
    )
    return Inputs(
        library=library,
        stock=stock,
        arm=arm,
        backend=backend,
        rule_sets=rule_sets,
        stocks=stocks,
        targets=targets,
        targets_file=targets_file,
        run_dir=run_dir,
    )


def test_bench_template_gaps() -> None:
    """The sweep. Not a benchmark fixture: it writes its own record and runs for hours."""
    gathered = gather_inputs(str(PAROUTES / "n1-targets.smi.gz"))
    library, arm, backend = gathered.library, gathered.arm, gathered.backend
    rule_sets, stocks = gathered.rule_sets, gathered.stocks
    targets, run_dir = gathered.targets, gathered.run_dir
    targets_file = gathered.targets_file
    stock = gathered.stock
    tiers = parse_efforts(env_text("GAPS_EFFORTS", "8x8,16x16,64x16"))

    manifest = {
        "experiment": "template-gaps-ladder",
        "backend": backend,
        "rule_sets": tuple(name for name, _ in rule_sets),
        "rule_set_contributions": library.source_counts,
        "template_count_compiled": len(library.smarts),
        "stock": tuple(name for name, _ in stocks),
        "stock_rows": stock.row_count,
        "tiers": tuple(
            {"width": w, "passes": p, "width_schedule": effort_widths(w, p)}
            for w, p in tiers
        ),
        "target_count": len(targets),
        "targets_file": str(targets_file),
        "stop_when_solved": True,
        "route_extraction": False,
        "forward_probe": False,
        "greedy_step": GREEDY_STEP,
        "relaxed_step": RELAXED_STEP,
        "inputs": {
            "stock": tuple(source_artifact(p) for _, p in stocks),
            "templates": tuple(source_artifact(p) for _, p in rule_sets),
            "targets": source_artifact(targets_file),
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    counts: dict[str, list[int]] = {}
    order: list[str] = []
    tier_started = perf_counter()
    current = ""
    done_in_tier = 0
    pending_in_tier = len(targets)

    with (run_dir / "attempts.jsonl").open("w") as records:
        for attempted in escalate(targets, tiers, partial(attempt_target, arm)):
            if attempted.tier != current:
                # A tier's size is the previous tier's unsolved count, which is only known once
                # that tier has finished -- so it is read here, at the boundary, not planned.
                pending_in_tier = len(targets) if not order else counts[order[-1]][1]
                current = attempted.tier
                order.append(current)
                counts[current] = [0, 0]
                tier_started, done_in_tier = perf_counter(), 0
                print(
                    f"\n=== tier {current}: {pending_in_tier} targets ===", flush=True
                )
            records.write(json.dumps(asdict(attempted)) + "\n")
            records.flush()
            counts[current][0 if attempted.solved else 1] += 1
            done_in_tier += 1
            if done_in_tier % 25 == 0 or done_in_tier == pending_in_tier:
                solved, unsolved = counts[current]
                print(
                    f"[{current} {done_in_tier}/{pending_in_tier}] "
                    f"solved={solved} unsolved={unsolved} "
                    f"({solved / max(1, done_in_tier):.1%}) "
                    f"eta={eta(tier_started, done_in_tier, pending_in_tier)}",
                    flush=True,
                )

    unsolved_final = _unsolved_after_last_tier(run_dir / "attempts.jsonl")
    (run_dir / "unsolved.smi").write_text("".join(f"{s}\n" for s in unsolved_final))
    # Plain while running so a killed sweep leaves a readable record, compressed once it cannot
    # grow: ten thousand rows of mostly-SMILES is 3 MB flat and 0.3 MB gzipped, and this lands in
    # a tracked directory.
    _compress(run_dir / "attempts.jsonl")
    summary = {
        "tiers": {
            name: {"solved": s, "unsolved": u} for name, (s, u) in counts.items()
        },
        "solved_total": len(targets) - len(unsolved_final),
        "unsolved_total": len(unsolved_final),
        "solve_rate": (len(targets) - len(unsolved_final)) / len(targets),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n{json.dumps(summary, indent=2)}\nrun directory: {run_dir}", flush=True)


@dataclass(frozen=True)
class RootCensus:
    """How many ways the collection can even *begin* on a target it could not finish."""

    target: str
    applications: int
    rules: tuple[int, ...]
    error: str | None = None


def root_census(arm: Arm, smiles: str) -> RootCensus:
    """Expand the target and nothing below it, then count the disconnections that produced.

    This splits the unsolved tail in two, and the halves want different work. ``applications
    == 0`` means no rule in the collection matches the molecule at all -- a gap at the root,
    readable straight off the SMILES. Anything above zero means the collection *starts* and
    then strands, which is a question about precursors and depth, not about the first bond.

    One pass at width one, so this is a single match call per target rather than a search.
    """
    expansions: Expansions[Any] = Expansions(NumpyExpansionStore(), DictInterner())

    def heuristic(molecules: Any) -> Any:
        return arm.heuristic(molecules)

    try:
        target_key = arm.key_of(smiles)
        target_mol = arm.mol_of(smiles)
    except ChemistryRejection as exc:
        return RootCensus(
            target=smiles,
            applications=0,
            rules=(),
            error=f"{type(exc).__name__}: {exc}",
        )
    aostar(
        target_key,
        target_mol=target_mol,
        stock=arm.stock,
        expansions=expansions,
        heuristic=heuristic,
        match=arm.match,
        stack_mols=arm.stack_mols,
        concat_mols=arm.concat_mols,
        width=1,
        passes=1,
        max_depth=1,
        price=constant_step(1.0),
        max_routes=0,
    )
    state = prepare_graph(expansions, arm.stock)
    target_sid = int(expansions.lookup([target_key])[0])
    steps = state.applications(target_sid)
    return RootCensus(
        target=smiles,
        applications=len(steps),
        rules=tuple(sorted({rule for step in steps for rule in step[0]})),
    )


@dataclass(frozen=True)
class Recovery:
    """Whether a goal-constrained case was recovered, and what the plain search found."""

    target: str
    starting_material: str
    solved: bool
    reached_starting_material: bool
    seconds: float
    error: str | None = None


def solution_uses(target_sid: int, wanted_sid: int, state: Any) -> bool:
    """Does **any** solution subgraph rooted at the target contain ``wanted_sid``?

    This is the goal-constrained question asked of the graph rather than of a sample of routes.
    Extraction would answer it for the handful of routes it materialises and say nothing about
    the rest; the graph encodes every route at once, so one bottom-up pass settles it exactly.

    A node contributes when it *is* the wanted molecule, and otherwise when some application of
    it has every precursor solved -- so the subgraph really is a solution -- and at least one
    precursor that contributes. A stocked node that is not the wanted molecule ends a route
    without using it, so it contributes nothing. Cycles contribute nothing either, which the
    in-flight set enforces: a product reachable from its own descendant adds no new leaf.
    """
    memo: dict[int, bool] = {}
    inflight: set[int] = set()

    def uses(product: int) -> bool:
        if product == wanted_sid:
            return True
        if product in memo:
            return memo[product]
        if product in state.stock_sids or product in inflight:
            return False
        inflight.add(product)
        found = any(
            all(state.solved[child] for child in precursors)
            and any(uses(child) for child in precursors)
            for _rules, precursors in state.steps.get(product, ())
        )
        inflight.discard(product)
        memo[product] = found
        return found

    return bool(state.solved[target_sid]) and uses(target_sid)


def recover_case(
    arm: Arm, target: str, starting_material: str, width: int | str, passes: int
) -> Recovery:
    """Search the target once, then ask the graph whether a solution reaches the given material."""
    widths = effort_widths(width, passes)
    expansions: Expansions[Any] = Expansions(NumpyExpansionStore(), DictInterner())

    def heuristic(molecules: Any) -> Any:
        return arm.heuristic(molecules)

    price, scheduled = anytime_step(
        greedy=GREEDY_STEP,
        relaxed=RELAXED_STEP,
        select=committed_application_leaves(fill=global_leaves),
    )
    started = perf_counter()
    try:
        target_key = arm.key_of(target)
        target_mol = arm.mol_of(target)
        wanted_key = arm.key_of(starting_material)
    except ChemistryRejection as exc:
        return Recovery(
            target=target,
            starting_material=starting_material,
            solved=False,
            reached_starting_material=False,
            seconds=perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    # Not `stop_when_solved`: the first route found need not be the one through this material,
    # so the budget is spent in full and the graph is read afterwards.
    #
    # Inside the refusal handler because a backend can refuse the *frontier*, not just the
    # keying: the metal arm packs 64 atoms a molecule and refuses a larger target. That costs
    # one case, which is a result, rather than the sweep.
    try:
        aostar(
            target_key,
            target_mol=target_mol,
            stock=arm.stock,
            expansions=expansions,
            heuristic=heuristic,
            match=arm.match,
            stack_mols=arm.stack_mols,
            concat_mols=arm.concat_mols,
            width=widths,
            passes=passes,
            max_depth=passes,
            select=scheduled,
            fold=bottleneck_fold,
            price=price,
            max_routes=0,
        )
    except ChemistryRejection as exc:
        return Recovery(
            target=target,
            starting_material=starting_material,
            solved=False,
            reached_starting_material=False,
            seconds=perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    elapsed = perf_counter() - started
    state = prepare_graph(expansions, arm.stock)
    target_sid = int(expansions.lookup([target_key])[0])
    wanted_sid = int(expansions.lookup([wanted_key])[0])
    reached = wanted_sid >= 0 and solution_uses(target_sid, wanted_sid, state)
    return Recovery(
        target=target,
        starting_material=starting_material,
        solved=bool(state.solved[target_sid]),
        reached_starting_material=reached,
        seconds=elapsed,
    )


def test_bench_goal_recovery() -> None:
    """Solvability, and then whether the route could have started from the specified material.

    `GAPS_PAIRS_FILE` is a `target<TAB>starting material` table -- DESP's benchmark sets, via
    `tests/data/external/desp/prepare.sh`. Load those starting materials as stock too, or none of
    them can end a route.
    """
    pairs_file = Path(env_text("GAPS_PAIRS_FILE", "")).expanduser()
    if not pairs_file.exists():
        pytest.skip(
            f"set GAPS_PAIRS_FILE to a target/starting-material table: {pairs_file}"
        )
    pairs = [
        (line.split("\t")[0], line.split("\t")[1])
        for line in pairs_file.read_text().splitlines()
        if line.strip() and "\t" in line
    ]
    gathered = gather_inputs(str(pairs_file))
    width, passes = parse_efforts(env_text("GAPS_EFFORTS", "64x16"))[0]

    print(f"goal-recovery over {len(pairs)} cases at {width}x{passes}", flush=True)

    run_dir, results = gathered.run_dir, []
    started = perf_counter()
    with (run_dir / "recovery.jsonl").open("w") as handle:
        for index, (target, material) in enumerate(pairs, start=1):
            row = recover_case(gathered.arm, target, material, width, passes)
            handle.write(json.dumps(asdict(row)) + "\n")
            handle.flush()
            results.append(row)
            if index % 10 == 0 or index == len(pairs):
                print(
                    f"[{index}/{len(pairs)}] solved={sum(r.solved for r in results)} "
                    f"reached_sm={sum(r.reached_starting_material for r in results)} "
                    f"eta={eta(started, index, len(pairs))}",
                    flush=True,
                )
    summary = {
        "cases": len(results),
        "solved": sum(r.solved for r in results),
        "reached_starting_material": sum(r.reached_starting_material for r in results),
        "rejected": sum(1 for r in results if r.error),
        "solve_rate": sum(r.solved for r in results) / max(1, len(results)),
        "recovery_rate": sum(r.reached_starting_material for r in results)
        / max(1, len(results)),
    }
    (run_dir / "recovery-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _compress(run_dir / "recovery.jsonl")
    print(f"\n{json.dumps(summary, indent=2)}\n{run_dir}", flush=True)


def test_bench_template_gap_census() -> None:
    """Why the unsolved tail is unsolved: does the collection start on it, or not at all?

    Reads a sweep's ``unsolved.smi`` -- point ``GAPS_TARGETS_FILE`` at one -- and writes the
    root census beside it. Cheap enough to run on the whole tail: one match call a target.
    """
    gathered = gather_inputs(str(PAROUTES / "n1-targets.smi.gz"))
    census = tuple(root_census(gathered.arm, smiles) for smiles in gathered.targets)
    no_start = tuple(row for row in census if row.applications == 0 and not row.error)
    rejected = tuple(row for row in census if row.error)

    fired = Counter(rule for row in census for rule in row.rules)
    codes = {
        row["rule_id"]: row["columns"].get("template_code", f"rule {row['rule_id']}")
        for row in template_rows(gathered.library, tuple(fired))
    }
    summary = {
        "targets": len(census),
        "no_root_application": len(no_start),
        "starts_but_strands": len(census) - len(no_start) - len(rejected),
        "parse_rejected": len(rejected),
        "root_applications": {
            "mean": sum(row.applications for row in census) / max(1, len(census)),
            "max": max((row.applications for row in census), default=0),
        },
        "rules_firing_at_root": [
            {"template_code": codes.get(rule, rule), "targets": count}
            for rule, count in fired.most_common(40)
        ],
    }
    run_dir = gathered.run_dir
    with (run_dir / "root-census.jsonl").open("w") as handle:
        for row in census:
            handle.write(json.dumps(asdict(row)) + "\n")
    (run_dir / "no-root-application.smi").write_text(
        "".join(f"{row.target}\n" for row in no_start)
    )
    (run_dir / "root-census-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"\n{json.dumps(summary, indent=2)}\ncensus directory: {run_dir}", flush=True)


@dataclass(frozen=True)
class DeadEnds:
    """A target's expanded-but-unexpandable leaves, named.

    These are the molecules the search *reached* and could neither buy nor take apart. For a
    target that strands, they are the question "what chemistry is missing" in its most concrete
    form: each one is a structure a chemist can look at.
    """

    target: str
    solved: bool
    expanded: int
    dead: tuple[str, ...]
    unnamed: int
    error: str | None = None


def dead_end_census(arm: Arm, smiles: str, width: int | str, passes: int) -> DeadEnds:
    """Search, then name every leaf that was expanded, matched nothing, and is not stocked.

    The naming is the part worth explaining. ``expansions`` stores WL keys, which are hashes, so
    the graph alone cannot say what a dead leaf *is*. Replay can, but replay needs a route and a
    stranded target has none. The way through is that ``match`` is the caller's own callback and
    is handed the molecules themselves -- and on this arm those are host-side :class:`CMol`,
    because vesma's matcher works on them directly. So every molecule the search expands is
    named on its way past, keyed the same way the store keys it, and the graph is joined to that
    table afterwards. No replay, no second search, no RDKit.
    """
    widths = effort_widths(width, passes)
    expansions: Expansions[Any] = Expansions(NumpyExpansionStore(), DictInterner())
    named: dict[Any, str] = {}

    def heuristic(molecules: Any) -> Any:
        return arm.heuristic(molecules)

    def match(molecules: Any) -> Any:
        for molecule in molecules:
            named[short_ring_wl_keyer.encode(molecule)] = smiles_from_cmol(molecule)
        return arm.match(molecules)

    price, scheduled = anytime_step(
        greedy=GREEDY_STEP,
        relaxed=RELAXED_STEP,
        select=committed_application_leaves(fill=global_leaves),
    )
    try:
        target_key = arm.key_of(smiles)
        target_mol = arm.mol_of(smiles)
    except ChemistryRejection as exc:
        return DeadEnds(
            target=smiles,
            solved=False,
            expanded=0,
            dead=(),
            unnamed=0,
            error=f"{type(exc).__name__}: {exc}",
        )
    aostar(
        target_key,
        target_mol=target_mol,
        stock=arm.stock,
        expansions=expansions,
        heuristic=heuristic,
        match=match,
        stack_mols=arm.stack_mols,
        concat_mols=arm.concat_mols,
        width=widths,
        passes=passes,
        max_depth=passes,
        select=scheduled,
        fold=bottleneck_fold,
        price=price,
        max_routes=0,
    )
    state = prepare_graph(expansions, arm.stock)
    target_sid = int(expansions.lookup([target_key])[0])
    # Expanded, produced no application, and not buyable. `state.steps` is keyed by every
    # expanded product including the ones that matched nothing, which is what makes this
    # readable off the graph rather than needing its own bookkeeping.
    dead = [
        sid
        for sid, steps in state.steps.items()
        if not steps and sid not in state.stock_sids
    ]
    keys = expansions.decode(dead)
    found = tuple(named[key] for key in keys if key in named)
    return DeadEnds(
        target=smiles,
        solved=bool(state.is_solved(target_sid)),
        expanded=len(state.steps),
        dead=found,
        unnamed=len(dead) - len(found),
    )


def test_bench_dead_end_census() -> None:
    """What the stranded targets could not take apart, as structures rather than hashes.

    Runs on the `cpu` arm by default and that is not incidental: it is the only arm whose
    molecules are host-side objects, so it is the only one where a dead leaf can be named
    without replaying a route that does not exist.
    """
    os.environ.setdefault("GAPS_BACKEND", "cpu")
    gathered = gather_inputs(str(PAROUTES / "n1-targets.smi.gz"))
    if gathered.backend == "metal":
        pytest.skip(
            "the metal arm's molecules are packed device arrays and cannot be named"
        )
    width, passes = parse_efforts(env_text("GAPS_EFFORTS", "16x16"))[0]

    run_dir, census = gathered.run_dir, []
    frequency: Counter[str] = Counter()
    started = perf_counter()
    with (run_dir / "dead-ends.jsonl").open("w") as handle:
        for index, smiles in enumerate(gathered.targets, start=1):
            row = dead_end_census(gathered.arm, smiles, width, passes)
            handle.write(json.dumps(asdict(row)) + "\n")
            handle.flush()
            census.append(row)
            frequency.update(row.dead)
            if index % 25 == 0 or index == len(gathered.targets):
                print(
                    f"[{index}/{len(gathered.targets)}] distinct_dead={len(frequency)} "
                    f"eta={eta(started, index, len(gathered.targets))}",
                    flush=True,
                )

    summary = {
        "targets": len(census),
        "solved_at_this_effort": sum(1 for row in census if row.solved),
        "distinct_dead_ends": len(frequency),
        "dead_ends_per_target": sum(len(row.dead) for row in census)
        / max(1, len(census)),
        "unnamed_total": sum(row.unnamed for row in census),
        "most_common": [
            {"smiles": smiles, "targets": count}
            for smiles, count in frequency.most_common(60)
        ],
    }
    (run_dir / "dead-end-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _compress(run_dir / "dead-ends.jsonl")
    print(f"\n{json.dumps(summary, indent=2)}\ncensus directory: {run_dir}", flush=True)


def _compress(path: Path) -> Path:
    """``path`` to ``path.gz``, removing the original. Returns where it went."""
    compressed = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as plain, gzip.open(compressed, "wb") as packed:
        copyfileobj(plain, packed)
    path.unlink()
    return compressed


def _stamp() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def _unsolved_after_last_tier(path: Path) -> tuple[str, ...]:
    """Targets whose *last* attempt failed -- the gap list, read back off the record."""
    last: dict[str, bool] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            last[row["target"]] = row["solved"]
    return tuple(target for target, solved in last.items() if not solved)

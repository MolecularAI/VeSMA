"""Which reference disconnections a template set can reproduce, one step at a time.

Route-level replay asks whether a collection reproduces a *whole* PaRoutes route, and that
question is dominated by arithmetic rather than chemistry: a seven-reaction route needs every
step to match, so per-step coverage of p gives route coverage near p**7, and a 414-rule curated
set is being compared with used14k's ~14,000 and raw68k's ~68,000. A low route-replay rate for
the small set is a size effect and says nothing about whether the chemistry in it is the right
chemistry.

This asks the size-fair question instead. Every reaction in every route is one independent
trial: given the product, does any rule in the set produce exactly the reference precursor set?
Each step is judged on its own, so the measure does not compound, and -- more to the point -- a
*miss* is a concrete (product, expected precursors) pair that a chemist can read and call either
routine medicinal chemistry the collection should have had, or something it is reasonable to
lack.

Comparing precursor sets needs both sides keyed the same way, so the reference SMILES are
parsed and keyed with the same keyer as the generated precursors. Stereochemistry is dropped by
that keyer, which is the intended behaviour here: a reference step and a rule that reproduces it
without stereocontrol are the same disconnection for this purpose.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pytest

from tests.datadirs import external_data_root
from vertsy.replay import route_precursor_sets
from vesma import cmol_from_smiles, generate_precursors, parse_reaction_smarts
from vesma.cmol import cmol_union
from vesma.cpu_matcher import CpuReactionMatcher
from vesma.keyer import cmol_smiles_keyer, short_ring_wl_keyer
from vesma.reaction import reaction_patterns_from_smarts
from vesma.rejection import ChemistryRejection

__all__ = [
    "Disconnection",
    "covered_by",
    "coverage_of",
    "forward_reach",
    "metal_coverage",
    "metal_forward_reach",
    "forward_rows",
    "load_disconnections",
    "rule_rows",
]

pytestmark = [pytest.mark.perf, pytest.mark.external_data]

PAROUTES = external_data_root() / "external" / "paroutes" / "processed"
TEMPLATES = external_data_root() / "templates"
HARTENFELLER = TEMPLATES / "templates-hartenfeller.csv"
FORWARD_TEMPLATES = TEMPLATES / "templates-forward.csv"
CURATED_FILES = (
    HARTENFELLER,
    TEMPLATES / "templates-medchem.csv",
    TEMPLATES / "templates-protecting-groups.csv",
    TEMPLATES / "templates-carbohydrate.csv",
    TEMPLATES / "templates-selenium.csv",
    TEMPLATES / "templates-phosphorus.csv",
)


@dataclass(frozen=True)
class Disconnection:
    """One reference reaction: a product and the precursor set the route says it came from."""

    product: str
    precursors: tuple[str, ...]
    route_index: int


@dataclass(frozen=True)
class Coverage:
    """Whether any rule reproduced a disconnection, in each direction, and which one did.

    The two directions are asked independently of the same reference step, which is the point:
    ``covered`` without ``forward`` is a disconnection the search can propose and the selectivity
    probe cannot examine, and an unprobed application is charged nothing. That is the gap the
    `sites` penalty turns into a gradient -- the engine learns to prefer the halogen whose forward
    rule nobody wrote -- so it is worth counting per rule rather than in aggregate.
    """

    product: str
    precursors: tuple[str, ...]
    route_index: int
    covered: bool
    by: str | None = None
    error: str | None = None
    forward: bool = False
    forward_by: str | None = None
    pot_atoms: int = 0
    pairing: str | None = None
    """For a retro-only step, why the forward side missed it, which decides the fix.

    ``unpaired`` -- no forward row names the covering retro rule, so the rule is unprobed by
    construction and somebody has to write its inverse. ``silent`` -- a forward row claims to
    invert it and did not rebuild this product, so the pairing exists and its *scope* is wrong,
    which is the harder and more interesting case. The same two words
    :class:`vertsy.selectivity.ProbeReport` uses, for the same reason.
    """


def rule_rows(paths: Sequence[Path]) -> tuple[tuple[str, str], ...]:
    """``(template_code, retro_template)`` for every row, so a hit can name the rule."""
    rows: list[tuple[str, str]] = []
    for path in paths:
        if not path.exists():
            pytest.skip(f"missing template file: {path}")
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                smarts = (row.get("retro_template") or "").strip()
                if smarts:
                    rows.append((row.get("template_code") or "?", smarts))
    return tuple(rows)


def probed_codes() -> frozenset[str]:
    """Every retro ``template_code`` some forward row claims, by ``probes`` or by ``scope``.

    Read here rather than taken from `load_forward_library` because that one resolves codes
    against a *compiled* rule set and drops any it did not compile, which is the right behaviour
    for a run and the wrong one for a census: a pairing naming a rule this sweep happens not to
    load is still a pairing somebody wrote.
    """
    if not FORWARD_TEMPLATES.exists():
        pytest.skip(f"missing forward template file: {FORWARD_TEMPLATES}")
    claimed: set[str] = set()
    with FORWARD_TEMPLATES.open(newline="") as handle:
        for row in csv.DictReader(handle):
            for column in ("probes", "scope"):
                for entry in (row.get(column) or "").split(";"):
                    if entry.strip():
                        claimed.add(entry.strip())
    return frozenset(claimed)


def forward_rows() -> tuple[tuple[str, str], ...]:
    """``(forward_code, forward_template)`` for every row of the forward table."""
    if not FORWARD_TEMPLATES.exists():
        pytest.skip(f"missing forward template file: {FORWARD_TEMPLATES}")
    rows: list[tuple[str, str]] = []
    with FORWARD_TEMPLATES.open(newline="") as handle:
        for row in csv.DictReader(handle):
            smarts = (row.get("forward_template") or "").strip()
            if smarts:
                rows.append((row.get("forward_code") or "?", smarts))
    return tuple(rows)


def forward_reach(
    disconnection: Disconnection,
    matcher: CpuReactionMatcher,
    codes: Sequence[str],
) -> tuple[bool, str | None, int]:
    """Does any forward rule rebuild the recorded product from the recorded precursors?

    The mirror image of :func:`coverage_of`, asked of the same reference step. The precursors are
    unioned into one molecule -- the pot -- because a bimolecular forward pattern has a
    two-component left-hand side and has to match across both starting materials at once, which
    is exactly what :func:`vertsy.selectivity.forward_probe` does in a run.

    Exact on the product, not subset-and-commodity like the retro side, and the asymmetry is
    real rather than an oversight: a retro rule is entitled to leave the bulk reagent implicit
    because it writes only the skeleton, while a forward rule is handed every recorded precursor
    including the reagents and so has no excuse for not producing the recorded product. What it
    *also* produces is not checked here; that is the selectivity channels' question.

    **No atom cap**, unlike :func:`tests.vertsy.bench_forward_calibration.outcome_of`, which skips
    pots over 64 atoms because that is what a device batch holds. This census is about whether the
    *library* has the rule, not whether a particular backend would have seen it, so the pot size is
    recorded instead of being used to exclude. A reader who wants the device's view filters on it.
    """
    try:
        parts = [cmol_from_smiles(smiles)[0] for smiles in disconnection.precursors]
        wanted = cmol_smiles_keyer(cmol_from_smiles(disconnection.product)[0])
    except ChemistryRejection:
        return False, None, 0
    pot_atoms = sum(part.atom_count for part in parts)
    try:
        products, hits = matcher.match([cmol_union(parts)], None, n_jobs=1)
    except ChemistryRejection:
        return False, None, pot_atoms

    rows = np.asarray(hits, dtype=np.int64).reshape(-1, 4)
    if not rows.shape[0]:
        return False, None, pot_atoms
    keys = [cmol_smiles_keyer(each) for each in products]
    # An outcome is one application's whole component multiset, the way the probe counts them, so
    # a rule that makes the product alongside something else still counts as reaching it.
    made: dict[tuple[int, int], list[Any]] = {}
    for _mol, rule, app, precursor in rows.tolist():
        made.setdefault((rule, app), []).append(keys[precursor])
    for (rule, _app), components in made.items():
        if wanted in components:
            return True, codes[rule], pot_atoms
    return False, None, pot_atoms


METAL_BUCKET = 64
"""The device's per-molecule atom budget. Measured over n1's first 500 routes, **no** reference
product and **no** reference pot exceeds it, so this costs the census nothing today; a step that
did would be reported as unexaminable rather than as a miss."""


def metal_coverage(
    steps: Sequence[Disconnection],
    matcher: Any,
    codes: Sequence[str],
    commodity: frozenset[Any],
) -> list[Coverage]:
    """:func:`coverage_of` for a whole chunk at once, on the device.

    Every ``(product, rule)`` pair of the chunk goes into one dispatch, which is the whole of the
    speedup: the host arm calls `generate_precursors` once per rule and rebuilds the product's
    GMol each time, 471 times a step. Measured on this workload, 186 ms a step becomes 4.3.

    The comparison moves from canonical SMILES to **WL tokens**, because that is what the device
    hands back and re-deriving SMILES would mean shipping the molecules home -- which is the one
    thing this arm exists not to do. Both keyers are blind to stereochemistry, deliberately here
    and by construction in the engine, so the verdict is the same question asked in a different
    alphabet; `test_the_two_arms_agree_on_the_same_steps` is what holds that claim up.

    Nothing readable is lost. The deliverables -- the missed steps and the forward gaps -- are
    written from the *reference* SMILES, which the route file supplies whatever the arm.
    """
    from vesma.metal_reaction_matcher import (
        metal_short_ring_wl_key,
        metal_stack_mols,
    )

    results: list[Coverage | None] = [None] * len(steps)
    products: list[Any] = []
    wanted: list[frozenset[Any]] = []
    rows_of: list[int] = []
    for index, step in enumerate(steps):
        try:
            product = cmol_from_smiles(step.product)[0]
            keys = frozenset(
                short_ring_wl_keyer(cmol_from_smiles(smiles)[0])
                for smiles in step.precursors
            )
        except ChemistryRejection as exc:
            results[index] = Coverage(
                product=step.product,
                precursors=step.precursors,
                route_index=step.route_index,
                covered=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            continue
        if product.atom_count > METAL_BUCKET:
            results[index] = Coverage(
                product=step.product,
                precursors=step.precursors,
                route_index=step.route_index,
                covered=False,
                error=f"product of {product.atom_count} atoms exceeds the device bucket",
            )
            continue
        products.append(product)
        wanted.append(keys)
        rows_of.append(index)

    if products:
        batch, hits = matcher.match(metal_stack_mols(products))
        keys = tuple(metal_short_ring_wl_key(batch))
        grouped: dict[int, dict[tuple[int, int], set[Any]]] = {}
        for mol_idx, rule, app, precursor in (
            np.asarray(hits, dtype=np.int64).reshape(-1, 4).tolist()
        ):
            grouped.setdefault(mol_idx, {}).setdefault((rule, app), set()).add(
                keys[precursor]
            )
        for slot, index in enumerate(rows_of):
            applications = (
                (codes[rule], frozenset(produced))
                for (rule, _app), produced in grouped.get(slot, {}).items()
            )
            found = covered_by(wanted[slot], commodity, applications)
            step = steps[index]
            results[index] = Coverage(
                product=step.product,
                precursors=step.precursors,
                route_index=step.route_index,
                covered=found is not None,
                by=found,
            )
    return [row for row in results if row is not None]


def metal_forward_reach(
    steps: Sequence[Disconnection], matcher: Any, codes: Sequence[str]
) -> list[tuple[bool, str | None, int]]:
    """:func:`forward_reach` for a whole chunk at once, on the device.

    One dispatch over every ``(pot, forward rule)`` pair. A pot the device cannot hold is reported
    unreached with its size, which is the honest answer rather than a miss -- the same discipline
    :data:`vertsy.costs.UNPROBED` draws, and measured to affect nothing in this corpus.
    """
    from vesma.metal_reaction_matcher import (
        metal_short_ring_wl_key,
        metal_stack_mols,
    )

    answers: list[tuple[bool, str | None, int]] = [(False, None, 0)] * len(steps)
    pots: list[Any] = []
    wanted: list[Any] = []
    rows_of: list[int] = []
    for index, step in enumerate(steps):
        try:
            parts = [cmol_from_smiles(smiles)[0] for smiles in step.precursors]
            target = short_ring_wl_keyer(cmol_from_smiles(step.product)[0])
        except ChemistryRejection:
            continue
        pot_atoms = sum(part.atom_count for part in parts)
        answers[index] = (False, None, pot_atoms)
        if pot_atoms > METAL_BUCKET:
            continue
        pots.append(cmol_union(parts))
        wanted.append(target)
        rows_of.append(index)

    if pots:
        batch, hits = matcher.match(metal_stack_mols(pots))
        keys = tuple(metal_short_ring_wl_key(batch))
        grouped: dict[int, dict[tuple[int, int], set[Any]]] = {}
        for mol_idx, rule, app, precursor in (
            np.asarray(hits, dtype=np.int64).reshape(-1, 4).tolist()
        ):
            grouped.setdefault(mol_idx, {}).setdefault((rule, app), set()).add(
                keys[precursor]
            )
        for slot, index in enumerate(rows_of):
            reached: str | None = None
            for (rule, _app), produced in grouped.get(slot, {}).items():
                if wanted[slot] in produced:
                    reached = codes[rule]
                    break
            _was, _by, pot_atoms = answers[index]
            answers[index] = (reached is not None, reached, pot_atoms)
    return answers


def load_disconnections(limit_routes: int) -> tuple[Disconnection, ...]:
    """Every reaction of the first ``limit_routes`` n1 reference routes, flattened.

    Read through :func:`route_precursor_sets`, which is what route replay prunes against, so
    this measures the same edges replay would have accepted rather than a second reading of the
    route format.
    """
    path = Path(
        os.environ.get("COVERAGE_ROUTES_FILE", "").strip()
        or str(PAROUTES / "n1-routes.json.gz")
    )
    if not path.exists():
        pytest.skip(f"missing route file: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = json.load(handle)

    found: list[Disconnection] = []
    for index, row in enumerate(rows[:limit_routes]):
        for product, allowed in _reference_edges(row):
            for precursors in allowed:
                found.append(
                    Disconnection(
                        product=product,
                        precursors=tuple(sorted(precursors)),
                        route_index=index,
                    )
                )
    return tuple(found)


def _reference_edges(
    row: dict[str, Any],
) -> Iterator[tuple[str, frozenset[frozenset[str]]]]:
    """``route_precursor_sets`` over one PaRoutes row, keyed by SMILES as the file stores them."""
    from vertsy.interop import route_from_aizynth_dict

    for child in row.get("children", []):
        node = {"type": "mol", "smiles": row["smiles"], "children": [child]}
        try:
            route = route_from_aizynth_dict(node, lambda metadata: 0)
        except KeyError, ValueError:
            continue
        yield from route_precursor_sets(route).items()


def commodity_keys(keyer: Any = cmol_smiles_keyer) -> frozenset[Any]:
    """Keys of the reagents a rule is entitled to leave implicit.

    ``stock-common-reagents.smi`` is exactly the right list for this: it is the bench's own
    statement of what is assumed to be on the shelf, so nothing here decides separately what
    counts as a reagent.

    ``keyer`` because these keys are compared against an arm's *own* keys, and the two arms speak
    different alphabets -- canonical SMILES on the host, WL tokens on the device. Keyed in the
    wrong one the set is simply never hit, the subset-plus-commodity allowance never fires, and
    coverage falls by the third that allowance is worth. Which is what happened, and is why the
    two arms are asserted to agree rather than assumed to.
    """
    path = TEMPLATES / "stock-common-reagents.smi"
    if not path.exists():
        pytest.skip(f"missing reagent stock: {path}")
    keys = set()
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                keys.add(keyer(cmol_from_smiles(line.split("\t")[0])[0]))
            except ChemistryRejection:
                continue
    return frozenset(keys)


def covered_by(
    wanted: frozenset[Any],
    commodity: frozenset[Any],
    applications: Iterable[tuple[str, frozenset[Any]]],
) -> str | None:
    """The covering rule, or ``None``. One definition, shared by every arm.

    **Not set equality, and symmetric about commodity reagents.** The two sides disagree about
    reagents in both directions, and for the same underlying reason -- that a reagent is not part
    of a disconnection's skeleton -- so both have to be forgiven.

    A rule may leave the reagent *implicit*: `MC 42` once took a nitroarene to the arene and said
    nothing about nitric acid, while the reference step lists it. Equality scored that a miss, and
    measured, it understated coverage by about a third.

    A rule may equally name a reagent the reference step omits, now that forty-six rows do name
    theirs. Keeping the one-sided test `produced <= wanted` turned exactly that improvement into a
    regression: measured over n1's first 500 routes, coverage fell from 64.70% to 56.54% -- 241
    steps -- purely because a rule that had become *more* explicit no longer looked like the
    reference.

    So a rule covers a step when the two precursor sets agree except for commodity reagents, in
    either direction. A leftover that is not a commodity is a real difference on either side: the
    rule found a different disconnection, not the same one written shorter or longer.

    Keys are opaque here, which is what lets the host arm compare canonical SMILES and the device
    arm compare WL tokens without the criterion knowing which it was handed.
    """
    for code, produced in applications:
        if (produced - wanted) <= commodity and (wanted - produced) <= commodity:
            return code
    return None


def coverage_of(
    disconnection: Disconnection,
    rules: Sequence[tuple[str, Any]],
    commodity: frozenset[Any] = frozenset(),
) -> Coverage:
    """Apply every rule to the product; covered when one reproduces the disconnection.

    **Not set equality**, and the reason is the whole measurement. Almost every rule here writes
    only the precursors that carry the skeleton and leaves the reagent implicit: ``MC 42`` takes
    a nitroarene to the arene and says nothing about nitric acid. A reference step lists the
    reagent as a precursor, so equality scores that rule a miss even though it reproduces the
    disconnection exactly -- measured, it understated coverage by about a third.

    So a rule covers a step when what it produces is a subset of the reference precursors and
    everything left over is a commodity reagent. A leftover that is *not* a commodity is a real
    difference: the rule found a different disconnection, not the same one written shorter.
    """
    try:
        product, _ = cmol_from_smiles(disconnection.product)
        wanted = frozenset(
            cmol_smiles_keyer(cmol_from_smiles(smiles)[0])
            for smiles in disconnection.precursors
        )
    except ChemistryRejection as exc:
        return Coverage(
            product=disconnection.product,
            precursors=disconnection.precursors,
            route_index=disconnection.route_index,
            covered=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    def applications() -> Iterator[tuple[str, frozenset[Any]]]:
        for code, pattern in rules:
            try:
                outcomes = generate_precursors(product, pattern)
            except ChemistryRejection:
                # A rule whose application does not describe a molecule -- `Cyc 2` without its
                # fusion guard raising OrphanDelocalizedBondError is the worked example -- has
                # not reproduced this disconnection, so it is a miss like any other. Narrow on
                # purpose: anything that is not a chemistry refusal is a bug and should escalate.
                continue
            for outcome in outcomes:
                yield (
                    code,
                    frozenset(
                        cmol_smiles_keyer(molecule) for molecule in outcome.precursors
                    ),
                )

    found = covered_by(wanted, commodity, applications())
    if found is not None:
        return Coverage(
            product=disconnection.product,
            precursors=disconnection.precursors,
            route_index=disconnection.route_index,
            covered=True,
            by=found,
        )
    return Coverage(
        product=disconnection.product,
        precursors=disconnection.precursors,
        route_index=disconnection.route_index,
        covered=False,
    )


def test_bench_disconnection_coverage() -> None:
    """Per-step coverage of the curated collection over n1 reference routes."""
    routes = int(os.environ.get("COVERAGE_ROUTES", "").strip() or 100)
    out_dir = Path(
        os.environ.get("COVERAGE_OUT_DIR", "").strip()
        or "docs/paper2/benchmarks/disconnection-coverage"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = (os.environ.get("COVERAGE_BACKEND", "").strip() or "cpu").lower()
    if backend not in ("cpu", "metal"):
        raise ValueError(f"COVERAGE_BACKEND must be cpu or metal, got {backend!r}")

    rows = rule_rows(CURATED_FILES)
    retro_codes = tuple(code for code, _ in rows)
    # The host arm wants parsed patterns one at a time; the device arm wants them compiled into
    # one table it can dispatch every pair against. Only the arm in use is built.
    rules = (
        [(code, parse_reaction_smarts(smarts)) for code, smarts in rows]
        if backend == "cpu"
        else []
    )
    retro_matcher: Any = None
    if backend == "metal":
        from vesma.metal_reaction_matcher import MetalReactionMatcher

        retro_matcher = MetalReactionMatcher(
            reaction_patterns_from_smarts(
                tuple(smarts for _, smarts in rows), cache=True
            )
        )
    # Keyed in the arm's own alphabet -- see `commodity_keys`.
    commodity = commodity_keys(
        short_ring_wl_keyer if backend == "metal" else cmol_smiles_keyer
    )
    disconnections = load_disconnections(routes)

    # The forward direction, off by default only because it roughly doubles the sweep: the whole
    # forward table runs against every step's pot. `COVERAGE_FORWARD=0` turns it off again.

    forward_wanted = (os.environ.get("COVERAGE_FORWARD", "").strip() or "1") != "0"
    forward_codes: tuple[str, ...] = ()
    forward_matcher: Any = None
    claimed: frozenset[str] = frozenset()
    if forward_wanted:
        forward_table = forward_rows()
        forward_codes = tuple(code for code, _ in forward_table)
        claimed = probed_codes()
        forward_patterns = reaction_patterns_from_smarts(
            tuple(smarts for _, smarts in forward_table), cache=True
        )
        if backend == "metal":
            from vesma.metal_reaction_matcher import MetalReactionMatcher

            forward_matcher = MetalReactionMatcher(forward_patterns)
        else:
            forward_matcher = CpuReactionMatcher(forward_patterns)
    print(
        f"backend={backend} rules={len(rows)} forward_rules={len(forward_codes)} "
        f"commodity_reagents={len(commodity)} "
        f"route_set={os.environ.get('COVERAGE_ROUTES_FILE') or 'n1'} "
        f"routes={routes} disconnections={len(disconnections)}",
        flush=True,
    )

    started = perf_counter()
    results: list[Coverage] = []
    # Chunked because the device wants every `(molecule, rule)` pair of a chunk in one dispatch,
    # which is where its 43x comes from. The host arm is still asked one step at a time, so the
    # chunk is only a flush boundary there and the progress line reads the same either way.
    chunk = int(os.environ.get("COVERAGE_CHUNK", "").strip() or 128)
    with (out_dir / "coverage.jsonl").open("w") as handle:
        for start in range(0, len(disconnections), chunk):
            window = disconnections[start : start + chunk]
            if backend == "metal":
                covers = metal_coverage(window, retro_matcher, retro_codes, commodity)
                reaches = (
                    metal_forward_reach(window, forward_matcher, forward_codes)
                    if forward_matcher is not None
                    else [(False, None, 0)] * len(window)
                )
            elif forward_matcher is not None:
                covers = [coverage_of(step, rules, commodity) for step in window]
                reaches = [
                    forward_reach(step, forward_matcher, forward_codes)
                    for step in window
                ]
            else:
                covers = [coverage_of(step, rules, commodity) for step in window]
                reaches = [(False, None, 0)] * len(window)
            for result, (reached, by, pot_atoms) in zip(covers, reaches, strict=True):
                if forward_matcher is not None:
                    pairing = None
                    if result.covered and not reached:
                        pairing = "silent" if result.by in claimed else "unpaired"
                    result = replace(
                        result,
                        forward=reached,
                        forward_by=by,
                        pot_atoms=pot_atoms,
                        pairing=pairing,
                    )
                handle.write(json.dumps(asdict(result)) + "\n")
                results.append(result)
            handle.flush()
            index = len(results)
            if True:
                hit = sum(1 for row in results if row.covered)
                rate = hit / len(results)
                gap = sum(1 for row in results if row.covered and not row.forward)
                per = (perf_counter() - started) / index
                print(
                    f"[{index}/{len(disconnections)}] covered={hit} ({rate:.1%}) "
                    f"retro-only={gap} "
                    f"{per:.2f}s/step eta={(len(disconnections) - index) * per / 60:.0f}m",
                    flush=True,
                )

    misses = [row for row in results if not row.covered and not row.error]
    gaps = [row for row in results if row.covered and not row.forward]
    total = max(1, len(results))
    summary = {
        "rules": len(rows),
        "backend": backend,
        "forward_rules": len(forward_codes),
        "routes": routes,
        "disconnections": len(results),
        "covered": sum(1 for row in results if row.covered),
        "coverage_rate": sum(1 for row in results if row.covered) / total,
        "unreadable_products": sum(1 for row in results if row.error),
        "by_rule": _top_rules(results),
    }
    if forward_matcher is not None:
        # The 2x2 over one corpus, which is the whole point of asking both directions of the same
        # step. `retro_only` is the arbitrage surface: a disconnection the search can propose and
        # the probe cannot examine, so its selectivity is charged nothing. `forward_only` is the
        # opposite and should be small -- a reaction the forward table can run but no retro rule
        # proposes is a disconnection we cannot reach.
        summary["directions"] = {
            "both": sum(1 for row in results if row.covered and row.forward),
            "retro_only": len(gaps),
            "forward_only": sum(
                1 for row in results if row.forward and not row.covered
            ),
            "neither": sum(1 for row in results if not row.covered and not row.forward),
            "forward_covered": sum(1 for row in results if row.forward),
            "forward_rate": sum(1 for row in results if row.forward) / total,
        }
        summary["unprobed_by_rule"] = _top_gaps(gaps)
        summary["directions"]["retro_only_unpaired"] = sum(
            1 for row in gaps if row.pairing == "unpaired"
        )
        summary["directions"]["retro_only_silent"] = sum(
            1 for row in gaps if row.pairing == "silent"
        )
    (out_dir / "coverage-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # The deliverable: the steps the collection cannot do, as reaction SMILES a chemist reads.
    (out_dir / "missed.rxn.smi").write_text(
        "".join(
            f"{'.'.join(row.precursors)}>>{row.product}\troute {row.route_index}\n"
            for row in misses
        )
    )
    # The second deliverable, and the one this direction was added for: steps the retro side
    # reproduces and the forward side cannot, tagged with the retro rule that covered them. Each
    # line is a forward rule somebody could write, named by the disconnection that wants it.
    if forward_matcher is not None:
        (out_dir / "forward-gaps.rxn.smi").write_text(
            "".join(
                f"{'.'.join(row.precursors)}>>{row.product}\t{row.by}\t"
                f"{row.pairing}\troute {row.route_index}\n"
                for row in gaps
            )
        )
    print(f"\n{json.dumps(summary, indent=2)}\nwrote {out_dir}", flush=True)


def _top_gaps(gaps: Sequence[Coverage]) -> list[dict[str, Any]]:
    """Which retro rules cover reference steps that no forward rule can rebuild.

    Ranked, because the fix is to write forward rules and this says which ones buy the most. It is
    also the list that makes a selectivity measurement interpretable: a penalty measured while
    these rules are unprobed is measuring the shape of the forward table as much as the chemistry.
    """
    from collections import Counter

    counted = Counter(row.by for row in gaps if row.by)
    pairing = {row.by: row.pairing for row in gaps if row.by}
    return [
        {
            "template_code": code,
            "unprobed_steps": count,
            "pairing": pairing.get(code),
        }
        for code, count in counted.most_common(30)
    ]


def _top_rules(results: Sequence[Coverage]) -> list[dict[str, Any]]:
    from collections import Counter

    counted = Counter(row.by for row in results if row.covered and row.by)
    return [{"template_code": code, "steps": n} for code, n in counted.most_common(30)]


@pytest.mark.metal
def test_the_two_arms_agree_on_the_same_steps() -> None:
    """The host and the device must reach the same verdict, step for step.

    The device arm exists because it is some thirty times faster -- 471 retro rules against one
    product cost 186 ms on the host and 4.3 ms on the device, which turns a four-hour sweep into
    minutes -- and it buys that by comparing **WL tokens** instead of canonical SMILES, since
    tokens are what it already has and shipping molecules home is the thing it avoids. That is a
    different alphabet for the same question, and the claim needs holding up rather than asserting.

    Two real defects came out of running this, which is the argument for having it:

    * the commodity reagent set was keyed in the host's alphabet whatever the arm, so on the device
      the subset-plus-commodity allowance could never fire and coverage fell by the third that
      allowance is worth;
    * `precursor_signature` discarded bond *order*, so `generate_precursors` collapsed every
      application of a bond-order-only rule into one -- a hydrogenation with twenty assignments
      returned a single outcome -- and the host under-reported coverage. The device arm, which does
      not go through that function, is what disagreed and was right.
    """
    from vesma.metal_matcher import GpuUnavailableError, probe_metal

    try:
        probe_metal()
    except (GpuUnavailableError, RuntimeError) as exc:
        pytest.skip(f"Metal backend not available: {exc}")

    steps = load_disconnections(6)
    rows = rule_rows(CURATED_FILES)
    forward_table = forward_rows()
    forward_codes = tuple(code for code, _ in forward_table)
    forward_patterns = reaction_patterns_from_smarts(
        tuple(smarts for _, smarts in forward_table), cache=True
    )

    from vesma.metal_reaction_matcher import MetalReactionMatcher

    host = [
        coverage_of(
            step,
            [(code, parse_reaction_smarts(smarts)) for code, smarts in rows],
            commodity_keys(cmol_smiles_keyer),
        )
        for step in steps
    ]
    device = metal_coverage(
        steps,
        MetalReactionMatcher(
            reaction_patterns_from_smarts(
                tuple(smarts for _, smarts in rows), cache=True
            )
        ),
        tuple(code for code, _ in rows),
        commodity_keys(short_ring_wl_keyer),
    )
    assert [row.covered for row in host] == [row.covered for row in device]
    assert [row.by for row in host] == [row.by for row in device]

    host_forward = [
        forward_reach(step, CpuReactionMatcher(forward_patterns), forward_codes)
        for step in steps
    ]
    device_forward = metal_forward_reach(
        steps, MetalReactionMatcher(forward_patterns), forward_codes
    )
    assert [reached for reached, _, _ in host_forward] == [
        reached for reached, _, _ in device_forward
    ]

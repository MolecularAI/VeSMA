"""Score a recorded literature route in our own price, beside the route the search found.

The instrument the price sweep asked for and did not have. `condemned_steps_on_the_top_route` --
the one sweep column independent of the mechanism being tuned -- reads 0 for every price on every
target, and `docs/note-vetoes.md:318-325` says why: 0.1 events per target after the template faults
the veto rows were written for were fixed. No affordable number of targets revives it.

PaRoutes n1 gives every target a recorded route. "How does our price rank the route a chemist
actually ran" is one event per target instead of 0.1, on ground truth written before any of these
searches existed. That is the comparison here.

Two halves, and the first is where the work is:

**Which of our templates would have drawn the recorded disconnection.** Not guessed from the
product -- `vertsy.replay.precursor_replay_search` exists for exactly this. It applies every rule
in the library and keeps only applications whose precursor set is the recorded one, so each
reference step is answered by whichever template reproduces it rather than by a pairing we assumed.
Reagents are `ignored` (see :func:`vertsy.replay.matching_application_rows`): a recorded route
names substrates, our templates increasingly name the hydride too, and that difference is not a
disconnection.

**What each route then costs.** Both columns are priced by the same `typical_yield` the search's
`AOSTAR_HOST_PRICE=yield` reads, so the numbers are directly comparable -- `base_step_factor` under
that price *is* the product of per-step yields. A reference step no template reproduces is scored at
the catch-all, and the report says how many, because a route whose every step is imputed is scoring
`YIELD_DEFAULT ** n` and nothing else.

Mixtures and mass coincidence are printed per step for both columns, from the same
:func:`tests.vertsy.bench_host_aostar.step_mixture` the route report uses, so a step can read as
good yield and poor recovery rather than as one muddled number.
"""

from __future__ import annotations

import gzip
import json
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.datadirs import external_data_root
from tests.vertsy.bench_host_aostar import (
    TEMPLATE_FILES,
    YIELD_DEFAULT,
    ForwardLibrary,
    Library,
    load_forward_library,
    load_merged_library,
    step_mixture,
)
from vertsy.interner import MISSING_SID, DictInterner
from vertsy.interop import route_from_aizynth_dict
from vertsy.mixture import ExpectedPeak
from vertsy.replay import precursor_replay_search, route_precursor_sets
from vertsy.routes import Route, encode_route, leaves, reaction_depth
from vertsy.search import Expansions, expansion, object_mol_array
from vertsy.store.numpy_store import NumpyExpansionStore
from vesma.cmol import CMol
from vesma.cmol_builder import cmol_from_smiles
from vesma.cpu_matcher import CpuReactionMatcher
from vesma.keyer import cmol_rdkit_smiles_keyer, short_ring_wl_keyer
from vesma.matcher import all_vs_all, keyed_match
from vesma.reaction import reaction_patterns_from_smarts

ROUTES = (
    external_data_root() / "external" / "paroutes" / "processed" / "n1-routes.json.gz"
)
REAGENTS = external_data_root() / "templates" / "stock-common-reagents.smi"
RULE_SETS = ("hartenfeller", "medchem", "protecting-groups")
RUN_DIR = os.environ.get("REFERENCE_ROUTES_RUN_DIR", "").strip()
IGNORE = os.environ.get("REFERENCE_ROUTES_IGNORE", "").strip() or "reagents"
"""Which precursors are not part of a disconnection: `reagents` or `stock`.

`reagents` -- the default -- drops only `stock-common-reagents.smi`: DCC, triphenylphosphine,
tetrahydroaluminate, the things a recorded route leaves implicit and our templates increasingly
name. None of them carries any of the target's skeleton, so dropping them cannot credit a rule
with chemistry it did not do.

`stock` drops every buyable precursor, which is the looser reading of "compare only what still
needs expanding". It is available because it is the natural thing to ask, but it over-credits, and
by a lot: benzaldehyde and acetaldehyde are both buyable and lead to different products, so a rule
that hands back the wrong one reads as having reproduced the step. At the limit a one-step route
whose every precursor is in stock compares the empty set with the empty set, and every rule that
fires on the target scores as reproducing it. Use it to bound coverage from above, not to score."""
REPORT = os.environ.get("REFERENCE_ROUTES_REPORT", "").strip()
ATOM_CAP = 64

type Key = Any


@dataclass(frozen=True)
class ScoredStep:
    """One disconnection, recorded or found, priced the way the search prices it."""

    product: str
    precursors: tuple[str, ...]
    rule: int | None
    label: str
    step_yield: float
    imputed: bool
    peaks: tuple[ExpectedPeak, ...]
    reagents: tuple[str, ...] = ()
    """Precursors the matching template names and the record does not.

    A recorded route lists what was drawn on the board, not what was in the flask: the hydride,
    the oxidant and the phosphine are left implicit. The replay matches anyway because those keys
    are `ignored`, but a mixture derived from the recorded list alone has no reagent to work with
    and falls back to the step's bare atom balance -- `MC 38` reading "H2 atoms lost" where its
    own forward rule would have said "water and chloride". So the matching rule is asked what it
    emits, and the difference is named here and used for the pot.
    """


@dataclass(frozen=True)
class ScoredRoute:
    steps: tuple[ScoredStep, ...]
    reached_from_the_target: int = 0
    """Recorded disconnections a replay seeded at the target alone reproduces -- see `Coverage`."""

    @property
    def route_yield(self) -> float:
        total = 1.0
        for step in self.steps:
            total *= step.step_yield
        return total

    @property
    def imputed_steps(self) -> int:
        return sum(1 for step in self.steps if step.imputed)


def recorded_roots() -> dict[Key, dict[str, Any]]:
    """Every n1 route tree, indexed by its target's key. First tree wins per target."""
    if not ROUTES.exists():
        pytest.skip(f"Missing reference route file: {ROUTES}")
    with gzip.open(ROUTES, "rt", encoding="utf-8") as handle:
        rows = json.load(handle)
    roots: dict[Key, dict[str, Any]] = {}
    for row in rows:
        key = key_of(row["smiles"])
        if key is not None:
            roots.setdefault(key, row)
    return roots


def key_of(smiles: str) -> Key | None:
    """The identity a replay compares on, or `None` for a SMILES this build cannot read."""
    try:
        molecule, _info = cmol_from_smiles(smiles)
    except Exception:
        return None
    return cmol_rdkit_smiles_keyer.encode(molecule)


def reagent_keys() -> frozenset[Key]:
    """Stock reagents, which a recorded route does not name and our templates sometimes do."""
    if not REAGENTS.exists():
        pytest.skip(f"Missing reagent stock file: {REAGENTS}")
    keys: set[Key] = set()
    for line in REAGENTS.read_text().splitlines():
        smiles = line.split("\t")[0].strip()
        if not smiles or smiles.startswith("#"):
            continue
        key = key_of(smiles)
        if key is not None:
            keys.add(key)
    return frozenset(keys)


def recorded_steps(node: dict[str, Any]) -> Iterator[tuple[str, tuple[str, ...]]]:
    """Each reaction in a route tree, as the product and its recorded precursors."""
    for reaction in node.get("children", ()):
        children = tuple(child["smiles"] for child in reaction.get("children", ()))
        yield node["smiles"], children
        for child in reaction.get("children", ()):
            yield from recorded_steps(child)


@dataclass(frozen=True)
class Coverage:
    """Which library rules reproduce each recorded disconnection, and how it was asked.

    `per_step` seeds every recorded product and asks each edge on its own. `from_the_target` seeds
    only the target, which is how `precursor_replay_search` is normally run and which credits only
    the matched *prefix* of a route: a product is reached in that graph only once its ancestors
    matched, so a step our library can draw perfectly well goes uncounted when the step above it
    failed (`vertsy.replay.disconnection_coverage` says so in as many words).

    Both are wanted, for different questions. Pricing a recorded route needs the per-step answer --
    a step's yield is a property of its chemistry, not of whether its parent was reachable. Asking
    whether we could have *run* that route needs the prefix answer. On these six targets the two
    differ by a factor of two, which is why reporting one of them alone would mislead.
    """

    per_step: dict[Key, frozenset[int]]
    from_the_target: int
    matcher: CpuReactionMatcher
    """Kept so a matched step can ask its rule what the flask held -- see `rule_precursors`."""


def covering_rules(
    node: dict[str, Any],
    merged: Library,
    ignored: frozenset[Key],
) -> Coverage:
    """Replay the recorded disconnections against the whole library, both ways round.

    The route's own leaves plus the reagents are the stock, because a template that names its
    hydride puts a precursor on the frontier the recorded route never mentions; without it in
    stock the replay would read that step as a dead end rather than as reproduced.
    """
    route: Route[Key] = encode_route(
        route_from_aizynth_dict(node, lambda _metadata: 0), cmol_rdkit_smiles_keyer
    )
    patterns = reaction_patterns_from_smarts(merged.smarts, cache=True)
    matcher = CpuReactionMatcher(patterns)

    def keys(precursors: Sequence[CMol]) -> Sequence[Key]:
        return [cmol_rdkit_smiles_keyer.encode(one) for one in precursors]

    allowed = route_precursor_sets(route)
    stock = frozenset(leaves(route)) | ignored
    match = expansion(keyed_match(all_vs_all(matcher), keys))
    spare = stock if IGNORE == "stock" else ignored

    def replay(seeds: Sequence[str], depth: int) -> Expansions[Key]:
        mols = [cmol_from_smiles(smiles)[0] for smiles in seeds]
        expansions: Expansions[Key] = Expansions(NumpyExpansionStore(), DictInterner())
        precursor_replay_search(
            allowed,
            [cmol_rdkit_smiles_keyer.encode(one) for one in mols],
            object_mol_array(mols),
            stock,
            depth,
            expansions,
            match=match,
            ignored=spare,
        )
        return expansions

    def drawn(expansions: Expansions[Key]) -> dict[Key, frozenset[int]]:
        found: dict[Key, frozenset[int]] = {}
        for product in allowed:
            sid = int(expansions.lookup([product])[0])
            if sid == MISSING_SID:
                continue
            rules = {rule for step in expansions.steps(sid) for rule in step[0]}
            if rules:
                found[product] = frozenset(rules)
        return found

    products = [product for product, _precursors in recorded_steps(node)]
    # One depth, because each product is its own seed: a second layer would re-derive edges the
    # other seeds already answer, and would reintroduce the prefix bias this split exists to avoid.
    per_step = drawn(replay(products, 1))
    prefix = drawn(replay([node["smiles"]], reaction_depth(route)))
    return Coverage(per_step=per_step, from_the_target=len(prefix), matcher=matcher)


def rule_precursors(
    matcher: CpuReactionMatcher,
    product: str,
    recorded: tuple[str, ...],
    rule: int,
    ignored: frozenset[Key],
) -> tuple[str, ...]:
    """What the matching template emits for this step, reagents included.

    Picks the application of `rule` whose substrates -- its precursors less the `ignored` reagents
    -- are the recorded ones, and returns that application's full precursor list. Falls back to
    the record where no application lines up, which keeps a step that matched through some other
    route from losing its peaks entirely.
    """
    molecule, _info = cmol_from_smiles(product)
    made, hits = matcher.match([molecule], [(0, rule)], n_jobs=1)
    rows = np.asarray(hits, dtype=np.int64).reshape(-1, 4)
    applications: dict[int, list[int]] = {}
    for _mol, _rule, app, precursor in rows.tolist():
        applications.setdefault(app, []).append(precursor)
    wanted = {key for key in (key_of(one) for one in recorded) if key not in ignored}
    for precursors in applications.values():
        keys = [cmol_rdkit_smiles_keyer.encode(made[one]) for one in precursors]
        if {key for key in keys if key not in ignored} == wanted:
            return tuple(keys)
    return recorded


def retro_label(merged: Library, rule: int) -> str:
    """A rule id back to the name its source file gave it.

    `Library.sources` is the only route from an id to a name -- nothing else survives the
    compile -- and `template_code` is the column the curated sets fill.
    """
    if not 0 <= rule < len(merged.sources):
        return f"#{rule}"
    source = merged.sources[rule]
    return source.columns.get("template_code") or f"{source.source}#{source.row}"


def best_rule(rules: frozenset[int], forward: ForwardLibrary) -> tuple[int, float]:
    """The highest-yielding template that draws this edge, and its yield.

    Max rather than mean for the reason `ForwardLibrary.yields` takes the max over a retro rule's
    forward partners: if any known way of running the step goes at 85%, the step can.
    """
    ranked = max(
        rules, key=lambda rule: (forward.yields.get(rule, YIELD_DEFAULT), -rule)
    )
    return ranked, forward.yields.get(ranked, YIELD_DEFAULT)


def score_recorded(
    node: dict[str, Any],
    merged: Library,
    forward: ForwardLibrary,
    mixture: Callable[[str, tuple[str, ...], int], tuple[ExpectedPeak, ...]],
    ignored: frozenset[Key],
) -> ScoredRoute:
    coverage = covering_rules(node, merged, ignored)
    steps: list[ScoredStep] = []
    for product, precursors in recorded_steps(node):
        key = key_of(product)
        rules = coverage.per_step.get(key) if key is not None else None
        reagents: tuple[str, ...] = ()
        if rules:
            rule, step_yield = best_rule(rules, forward)
            label = retro_label(merged, rule)
            others = len(rules) - 1
            if others:
                label = f"{label} (+{others} more drew it)"
            whole = rule_precursors(
                coverage.matcher, product, precursors, rule, ignored
            )
            recorded_keys = {key_of(one) for one in precursors}
            reagents = tuple(one for one in whole if key_of(one) not in recorded_keys)
            peaks = safe_mixture(mixture, product, whole, rule)
        else:
            rule, step_yield, label, peaks = None, YIELD_DEFAULT, "no template", ()
        steps.append(
            ScoredStep(
                product=product,
                precursors=precursors,
                rule=rule,
                label=label,
                step_yield=step_yield,
                imputed=rules is None or not rules,
                peaks=peaks,
                reagents=reagents,
            )
        )
    return ScoredRoute(tuple(steps), coverage.from_the_target)


def safe_mixture(
    mixture: Callable[[str, tuple[str, ...], int], tuple[ExpectedPeak, ...]],
    product: str,
    precursors: tuple[str, ...],
    rule: int,
) -> tuple[ExpectedPeak, ...]:
    """A mixture that cannot be derived is reported as absent, not as empty.

    The forward direction can refuse a step the retro direction drew -- that is what the
    confirmation probe measures -- and a report is not the place to raise about it.
    """
    try:
        return mixture(product, precursors, rule)
    except Exception:
        return ()


def rule_ids(merged: Library) -> dict[str, int]:
    """Template code -> rule id in *this* build.

    A dump carries `rule_label`, not an id, and ids are positions in whatever library the run
    merged. Reading the label back is what makes a dump from one run comparable to a library
    built here; a raw id would silently name a different template if the rule sets differ.
    """
    return {retro_label(merged, rule): rule for rule in range(len(merged.sources))}


def score_found(
    report: Any,
    merged: Library,
    forward: ForwardLibrary,
    mixture: Callable[[str, tuple[str, ...], int], tuple[ExpectedPeak, ...]],
) -> ScoredRoute:
    """The top extracted route, re-priced and re-mixed here from its own recorded templates.

    The dump carries the mixture its run derived, and this recomputes it instead. Both columns of
    the report then come from one forward table rather than one from the table and one from
    whenever the run happened -- and where they disagree it is the table that moved, which is a
    thing to see rather than a thing to inherit. The stored peaks stand in only for a step whose
    template this build cannot name.
    """
    ids = rule_ids(merged)
    steps: list[ScoredStep] = []
    for step in report.routes[0].retro_steps:
        label = step.chosen.rule_label or ""
        rule = ids.get(label)
        precursors = tuple(step.chosen.precursors)
        if rule is None:
            step_yield, imputed = YIELD_DEFAULT, True
            peaks = tuple(step.mixture)
        else:
            step_yield = forward.yields.get(rule, YIELD_DEFAULT)
            imputed = rule not in forward.yields
            peaks = safe_mixture(mixture, step.product, precursors, rule)
        steps.append(
            ScoredStep(
                product=step.product,
                precursors=precursors,
                rule=rule,
                label=label or f"#{step.chosen.rule}",
                step_yield=step_yield,
                imputed=imputed,
                peaks=peaks,
            )
        )
    return ScoredRoute(tuple(steps))


def render_peaks(peaks: tuple[ExpectedPeak, ...], pad: str) -> list[str]:
    if not peaks:
        return [f"{pad}no mixture derived"]
    lines = []
    for peak in peaks:
        mass = "       --" if peak.exact_mass is None else f"{peak.exact_mass:9.4f}"
        tier = {
            "same formula": "  <- isobaric with the product",
            "same nominal mass": "  <- same nominal mass, needs resolution",
        }.get(peak.coincidence, "")
        source = f" [{peak.source}]" if peak.source else ""
        lines.append(f"{pad}{mass}  {peak.formula:16} {peak.kind}{source}{tier}")
    return lines


def render_route(title: str, route: ScoredRoute, pad: str = "  ") -> list[str]:
    lines = [f"{pad}{title}"]
    for index, step in enumerate(route.steps, start=1):
        lines.append(f"{pad}  {index}. {step.product}")
        named = " + ".join(step.precursors)
        if step.reagents:
            named += "".join(
                f" + {one} (reagent, implicit in the record)" for one in step.reagents
            )
        lines.append(f"{pad}     <- {named}")
        flag = " (catch-all)" if step.imputed else ""
        lines.append(f"{pad}     {step.label}   yield {step.step_yield:.2f}{flag}")
        lines.extend(render_peaks(step.peaks, f"{pad}       "))
    lines.append(
        f"{pad}  = {route.route_yield:.3f} over {len(route.steps)} steps"
        f", {route.imputed_steps} on the catch-all"
    )
    return lines


def render_coverage(route: ScoredRoute, pad: str = "  ") -> list[str]:
    drawn = len(route.steps) - route.imputed_steps
    return [
        f"{pad}  {drawn} of {len(route.steps)} disconnections reproduced step by step,"
        f" {route.reached_from_the_target} replaying from the target"
    ]


def reports(run_dir: Path) -> Iterator[Any]:
    path = run_dir / "alternatives.jsonl"
    if not path.exists():
        pytest.skip(f"Missing route report dump: {path}")
    from vertsy.route_alternatives import report_from_json

    for line in path.read_text().splitlines():
        if line.strip():
            yield report_from_json(json.loads(line))


@pytest.mark.perf
def test_how_a_recorded_route_scores_beside_the_one_we_found(capsys: Any) -> None:
    """Measured, not asserted beyond the two things that would make the numbers meaningless."""
    if not RUN_DIR:
        pytest.skip("Set REFERENCE_ROUTES_RUN_DIR to a run's route report directory")
    run_dir = Path(RUN_DIR)
    merged = load_merged_library(
        tuple((name, TEMPLATE_FILES[name]) for name in RULE_SETS)
    )
    forward = load_forward_library(merged.sources)
    mixture = step_mixture(
        forward, short_ring_wl_keyer.encode, cmol_rdkit_smiles_keyer.encode
    )
    roots = recorded_roots()
    ignored = reagent_keys()

    lines: list[str] = []
    rows: list[tuple[str, ScoredRoute, ScoredRoute, float]] = []
    for report in reports(run_dir):
        if not report.routes:
            continue
        key = key_of(report.target)
        root = roots.get(key) if key is not None else None
        if root is None:
            lines.append(f"\n{report.target}\n  no recorded n1 route for this target")
            continue
        recorded = score_recorded(root, merged, forward, mixture, ignored)
        found = score_found(report, merged, forward, mixture)
        rows.append((report.target, recorded, found, report.routes[0].score.utility))
        lines.append("")
        lines.append("=" * 98)
        lines.append(report.target)
        lines.append("=" * 98)
        lines.extend(render_route("the route a chemist ran", recorded))
        lines.extend(render_coverage(recorded))
        lines.append("")
        lines.extend(render_route("the route the search found", found))
        recorded_in_dump = report.routes[0].score.base_step_factor
        if abs(recorded_in_dump - found.route_yield) > 1e-6:
            lines.append(
                f"    the dump scored this route at {recorded_in_dump:.3f}, not"
                f" {found.route_yield:.3f} -- that run was not priced by yield"
            )

    header = (
        f"\n=== {len(rows)} targets with a recorded n1 route, "
        f"{len(merged.smarts)} retro rules, {len(forward.smarts)} forward rules\n"
    )
    table = [
        f"{'target':<40} {'recorded':>22}   {'found':>22}",
        f"{'':<40} {'steps imputed yield':>22}   {'steps imputed yield':>22}",
    ]
    for target, recorded, found, utility in rows:
        table.append(
            f"{target[:40]:<40}"
            f" {len(recorded.steps):>5} {recorded.imputed_steps:>7}"
            f" {recorded.route_yield:>8.3f}  "
            f" {len(found.steps):>5} {found.imputed_steps:>7}"
            f" {found.route_yield:>8.3f}   utility {utility:.3f}"
        )
    better = sum(
        1
        for _t, recorded, found, _u in rows
        if found.route_yield > recorded.route_yield
    )
    table.append("")
    table.append(f"the search's route prices higher on {better} of {len(rows)} targets")
    imputed = sum(route.imputed_steps for _t, route, _f, _u in rows)
    total = sum(len(route.steps) for _t, route, _f, _u in rows)
    table.append(
        f"{imputed} of {total} recorded steps are scored at the {YIELD_DEFAULT} catch-all,"
        " so that column is partly an artefact of the default rather than curated chemistry"
    )
    prefix = sum(route.reached_from_the_target for _t, route, _f, _u in rows)
    table.append(
        f"{total - imputed} of {total} recorded disconnections are reproduced when each is asked"
        f" on its own, and {prefix} when the replay starts at the target -- the gap is steps whose"
        " parent we cannot draw, so the route is unreplayable while the step is not"
    )

    text = header + "\n".join(table) + "\n" + "\n".join(lines) + "\n"
    print(text)
    if REPORT:
        Path(REPORT).write_text(text)

    # No target in the dump had a recorded n1 route, so there was nothing to compare against.
    assert rows
    # No recorded disconnection was reproduced by any template. A route with no reproduced step
    # scores pure `YIELD_DEFAULT ** n`, which measures the default and not the chemistry.
    assert any(route.imputed_steps < len(route.steps) for _t, route, _f, _u in rows)
    # A recorded route names what was drawn on the board, never the hydride or the oxidant, so
    # some matched step must have recovered a reagent from its own rule. Without that the pot is
    # the record's own short list and the mixture falls back to the step's bare atom balance --
    # which is how `MC 38` came to report "H2 atoms lost" where its forward rule says water and
    # chloride. If this stops holding, either the reagents went implicit again or nothing matched.
    assert any(step.reagents for _t, route, _f, _u in rows for step in route.steps)

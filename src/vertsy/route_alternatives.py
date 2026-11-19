"""What else the engine could have done at each step of a reported route, and what it knew.

A route report that shows only the chosen steps cannot be acted on. The interesting question a
chemist asks of a proposed synthesis is not "what did you pick" but **"what else was there, and
why not that?"** -- and the engine already knows: every disconnection of every molecule in the
graph, its cost, how many products the forward probe says it makes, how many other reactions its
starting materials could undergo, and which veto rows fired on it. All of it is thrown away when
the route is emitted. This module keeps it.

The motivation is concrete. Two blind route judges, independently, said route quality is decided by
things the engine does not model, and separately a raised selectivity price was found to prefer
*worse* chemistry -- a glycidyl Grignard over an mCPBA epoxidation -- because the forward rule
paired with the bad step fires there and reports one clean product. No price can fix that; somebody
has to see that `FWD H 45` accepts epichlorohydrin and narrow it. A report that names the
measurement behind each alternative is how that becomes visible.

What a record does and does not claim
-------------------------------------
A per-step row says what an alternative **would cost and measure**. It does *not* say "the
optimiser rejected it because of this". The reported order comes from three selection phases and a
diversity floor, so a cheaper route legitimately appears after a dearer one -- by design, because
only the first route strives to be best and the rest earn their place by being *different*. So
route-level standing (:data:`RouteStanding`) is reported separately from per-step evidence
(:data:`Standing`), and the per-step rows are evidence a reader weighs rather than a cause the
engine asserts.

``UNPROBED`` is ``None`` here, and never ``0``. The stored relation is sparse on purpose -- an
absent row means nobody looked, while a stored zero means the probe ran and the reaction made
nothing. Collapsing those would make an unexamined rule read as a clean one, which is the one
confusion the whole measurement layer is built to prevent.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .core import K
from .costs import UNPROBED, StepPrice, rule_price
from .metrics import GraphState
from .mixture import ExpectedPeak
from .route_selection import RouteScore, RouteSelection, disconnections
from .routes import SidRoute

__all__ = [
    "ROUTE_STANDINGS",
    "STANDINGS",
    "Alternative",
    "ReportedRoute",
    "RetroStep",
    "RouteReport",
    "RouteStanding",
    "Standing",
    "VETO_CHANNEL",
    "fired_vetoes",
    "render_report_markdown",
    "route_report",
    "stored_veto_validator",
]

VETO_CHANNEL = "vetoed"
"""The aggregate verdict channel. Per-row verdicts are ``vetoed:<code>``, when a run measured them."""

type Standing = Literal[
    "chosen",
    "unsolved",
    "vetoed",
    "competing-sites",
    "competing-reaction",
    "costlier",
    "tie",
]
"""Why one alternative stands where it does *relative to the step that was taken*.

A closed vocabulary, declared in one place the way :class:`vesma.rejection.RejectionReason` is, so
every row names one of these and a report can be summarised without parsing prose.

Comparative on purpose. Labelling an alternative ``competing-sites`` when the chosen step makes
just as many products would be misleading, so these say how an alternative *differs* from the step
taken, and the raw counts travel alongside in :attr:`Alternative.measurements` for a reader who
wants to check.

``unsolved`` is first and is usually the commonest: most disconnections of most molecules lead to a
precursor nothing can buy or make, so they were never really alternatives at all. Saying so is what
stops a reader mistaking a dead branch for a rejected one.

Absent by design: *longer*. Deciding that an alternative yields a longer route needs the selector's
depth-budgeted ``fewest_reactions`` fold, which is entangled with its memo and its bounds. A
half-right version would be worse than none, so the honest signal here is
:attr:`Alternative.also_reported_in` -- whether the alternative is live in some other reported
route -- and route length stays on :class:`RouteScore` where it is exact.
"""

STANDINGS: frozenset[str] = frozenset(
    (
        "chosen",
        "unsolved",
        "vetoed",
        "competing-sites",
        "competing-reaction",
        "costlier",
        "tie",
    )
)

type RouteStanding = Literal["best", "diverse", "too-similar", "censored"]
"""Why a whole route is where it is, which is a different question from any one step.

``best`` is the first route, the only one that strives to be cheapest. ``diverse`` is a route that
earned its place by differing early -- the decision index is on the record. ``too-similar`` is a
candidate the diversity floor rejected: good enough to report, too much like one already reported.
``censored`` is a candidate a validator refused -- a step that would not replay, or one a refusing
veto row condemned -- which is a statement about that route and about no other.
"""

ROUTE_STANDINGS: frozenset[str] = frozenset(
    ("best", "diverse", "too-similar", "censored")
)

StepKey = tuple[int, int, tuple[str, ...]]
"""One disconnection in one position: retro-step index, rule, and its precursors by name.

Names, not sids, so the key means the same thing across two routes of the same report and
survives the JSON the report is written as.
"""


@dataclass(frozen=True)
class Alternative:
    """One disconnection of one molecule, with everything the graph knows about it."""

    rule: int
    rule_label: str
    """A chemist-readable name where the rule set gave one, else the template code, else the id."""
    precursors: tuple[str, ...]
    also_by: tuple[int, ...]
    """Other rules reaching these same precursors, cheapest-first after the credited one.

    Co-canonical templates are collapsed before anything is chosen, so these are never separate
    alternatives -- but a reader comparing a report against a rule table needs to know the
    disconnection was reachable more than one way and which one was credited.
    """
    log_cost: float
    cost_delta: float
    """Step cost minus the chosen step's, in step-equivalents. Zero for the chosen step."""
    measurements: Mapping[str, int | None]
    """Per channel, what the forward probe measured. ``None`` means **nobody looked**, not zero."""
    vetoes_fired: tuple[str, ...]
    standing: Standing
    also_reported_in: tuple[int, ...] = ()
    """Ranks of other reported routes that take this disconnection here."""


@dataclass(frozen=True)
class RetroStep:
    """One disconnection of a reported route, numbered in retro order.

    ``index`` 1 is the **first disconnection**, which is the **last** synthesis step. Counting
    backwards is how a route is read when the question is "what do I make last", and it matches the
    order the engine decided in. The reaction itself is still written forwards, ``A + B -> P``.
    """

    index: int
    product: str
    chosen: Alternative
    alternatives: tuple[Alternative, ...]
    mixture: tuple[ExpectedPeak, ...] = ()
    """What to expect in the pot for the step **taken**, with a mass for each.

    On the chosen step only, and that is the division of labour in this report: the table above
    explains the *decision*, comparing what was taken against what was not, while this explains
    the *consequence* of the decision a reader is about to act on. An alternative's mixture is a
    report about a reaction nobody is running.

    Empty when no mixture was supplied -- `route_report` takes the derivation as an argument,
    because naming the forward rules of a step is a question about pairings and conditions that
    the graph does not hold."""


@dataclass(frozen=True)
class ReportedRoute:
    rank: int
    standing: RouteStanding
    score: RouteScore
    first_distinct_decision: int | None
    differing_decisions: int | None
    """How many decisions differ from the nearest reported route -- evidence, not the floor's knob.

    The floor is tuned on :attr:`first_distinct_decision` because *where* routes diverge decides
    whether a plan is really different, while a count can be inflated by a route that diverges
    once and then re-converges. Recorded because it is cheap and worth reading.
    """
    resembled: int | None
    retro_steps: tuple[RetroStep, ...]
    reason: str | None = None
    """Why a ``censored`` route was refused, verbatim from the validator. ``None`` otherwise."""


@dataclass(frozen=True)
class RouteReport:
    target: str
    routes: tuple[ReportedRoute, ...]
    rejected: tuple[ReportedRoute, ...]
    """Candidates that were not reported: the diversity floor's, then a validator's.

    Both are here so the two reasons a report can come back shorter than asked are visible in it
    rather than inferred from a count: one is a knob a reader can widen, the other is chemistry.
    """
    truncated: bool
    stopped_after_limit: bool
    channels: tuple[str, ...] = ()
    """Measurement channels this run actually carried, so an empty column reads as "not measured"."""
    diversity_cutoff: int | None = None
    rule_yields: Mapping[int, float] = field(default_factory=dict)
    """Rule id -> the curated yield the price charged it, for the rules this report cites.

    The table showed a cost *delta* in step-equivalents, which says which option was dearer but
    not what it was dearer about. A yield is the number a chemist can disagree with directly, and
    a rule absent from this mapping is one nobody curated -- the same "nobody looked" signal
    `measurements` carries for the probe, on the other axis.
    """
    veto_rationales: Mapping[str, str] = field(default_factory=dict)
    """Veto code -> the chemistry it states, for the rows this report cites.

    The table's own `rationale` column, which `vertsy.filters` validates as non-empty and then
    nothing read -- its docstring said so: "`rationale` is not read by anything and is the only
    defence". A report that says a step was vetoed by `B 1` tells a reader which row to look up; a
    report that says *why* is one they can argue with, which is the whole point of writing the
    rationale down. Carried on the report rather than looked up at render time because the renderer
    works from the JSON and has no table.
    """


def _tokens(route: SidRoute) -> tuple[tuple[int, tuple[int, ...]], ...]:
    from .route_selection import route_tokens

    return route_tokens(route)


def _differing(
    route: SidRoute, others: Sequence[SidRoute]
) -> tuple[int | None, int | None]:
    """How many decisions differ from the nearest other route, and that route's position."""
    if not others:
        return None, None
    mine = _tokens(route)
    scored = [
        (
            sum(1 for left, right in zip(mine, _tokens(other)) if left != right)
            + abs(len(mine) - len(_tokens(other))),
            position,
        )
        for position, other in enumerate(others)
    ]
    count, nearest = min(scored)
    return count, nearest + 1


def _measurements(
    state: GraphState[K],
    channels: Sequence[str],
    precursors: tuple[int, ...],
    rule: int,
) -> dict[str, int | None]:
    return {
        channel: state.outcomes.get(channel, {}).get((precursors, rule))
        for channel in channels
    }


def fired_vetoes(
    state: GraphState[K], product: int, precursors: tuple[int, ...], rule: int
) -> tuple[str, ...]:
    """Which veto rows fired, by code, from the per-row columns when a run measured them.

    Read from the **step-keyed** index where a run built one, which is the exact answer: a verdict
    about a product belongs to one ``(product, rule, precursors)``, that key cannot collide, and
    nothing has to be inferred. See :func:`vertsy.expansions.sid_step_selectivity`.

    Everything below is the fallback for a run without it. A code with a trailing ``?`` is a row
    whose per-row verdict was **disputed**, and the reason is structural rather than a defect. A measurement is keyed by ``(precursor sids, rule)`` and
    deliberately not by product, because a forward reaction is determined by its rule and its
    starting materials -- but a veto read *on the product* is not. The same precursors under the
    same template, halogenating two different ring positions, give two products of which one is
    condemned and one is not, so the two applications disagree and
    :func:`vertsy.expansions.sid_selectivity` drops the pair to `UNPROBED` rather than trusting
    whichever was written last. The aggregate column can survive the same collapse by agreeing for
    another reason -- another row firing on the second application -- which is exactly the state
    that used to render as ``(unnamed)``: a veto fired, and nothing could say which.

    Naming the disputed rows is strictly better, because they are the only candidates: a row at
    `UNPROBED` here is one whose answer differed between the two products, and a curator opening
    `B 13?` is opening the right row. The complete answer is to re-run the table on that one
    application, which is what replay is for -- `note-vetoes.md` says so -- and needs a screen over
    real molecules that this module deliberately does not have.

    Falls back to the aggregate count last: a run without the per-row columns can still say *that*
    a veto fired, and saying so without naming it beats saying nothing.
    """
    if state.step_outcomes:
        step = (product, rule, tuple(sorted(precursors)))
        return tuple(
            sorted(
                channel.split(":", 1)[1]
                for channel, table in state.step_outcomes.items()
                if channel.startswith(f"{VETO_CHANNEL}:") and (table.get(step) or 0) > 0
            )
        )
    fired = tuple(
        sorted(
            channel.split(":", 1)[1]
            for channel, table in state.outcomes.items()
            if channel.startswith(f"{VETO_CHANNEL}:")
            and (table.get((precursors, rule)) or 0) > 0
        )
    )
    if fired:
        return fired
    aggregate = state.outcomes.get(VETO_CHANNEL, {}).get((precursors, rule)) or 0
    if aggregate <= 0:
        return ()
    disputed = tuple(
        sorted(
            f"{channel.split(':', 1)[1]}?"
            for channel, table in state.outcomes.items()
            if channel.startswith(f"{VETO_CHANNEL}:")
            and table.get((precursors, rule)) == UNPROBED
        )
    )
    return disputed or ("(unnamed)",)


def _standing(
    state: GraphState[K],
    measurements: Mapping[str, int | None],
    chosen: Mapping[str, int | None],
    vetoes: tuple[str, ...],
    precursors: tuple[int, ...],
    cost_delta: float,
) -> Standing:
    """Classify one alternative against the step that was taken. Order is the precedence."""
    if any(not state.is_solved(precursor) for precursor in precursors):
        return "unsolved"
    if vetoes:
        return "vetoed"
    for channel, label in (
        ("sites", "competing-sites"),
        ("competitors", "competing-reaction"),
    ):
        mine, theirs = measurements.get(channel), chosen.get(channel)
        if mine is not None and theirs is not None and mine > theirs:
            return label  # type: ignore[return-value]
    if cost_delta > 0.0:
        return "costlier"
    return "tie"


def stored_veto_validator(
    state: GraphState[K],
) -> Callable[[SidRoute], str | None]:
    """Refuse a candidate route whose steps a veto row already condemned, by lookup.

    The `validate` hook :func:`vertsy.route_selection.select_solved_routes` takes is
    ``SidRoute -> reason | None``, which is pure sid space -- so a validator does not have to own
    molecules. The only other implementation replays the route and re-runs the veto screen over
    real structures, which is the one way a ``geometry`` row can be answered at all and is also
    why it is host-only and costs a replay per candidate. Every shipped row is ``cost="pattern"``
    and was already evaluated per application during the search, so for those the verdict is
    sitting in the graph and this reads it.

    **The same lookup the report uses, deliberately.** :func:`fired_vetoes` is what fills an
    `Alternative.vetoes_fired`, so a route refused here is refused for the row the report prints
    beside it, keyed identically -- `(product, rule, sorted(precursors))` where the step-keyed
    columns exist, the chemistry key otherwise. Two queries would let the report explain a refusal
    that did not happen, or a refusal cite a row the report cannot show.

    Only the refusing channel is read, because :data:`VETO_CHANNEL` is what `fired_vetoes` looks
    at: a ``force="price"`` row belongs to the price and must not block a route here as well.

    A disputed code -- ``B 13?``, where the chemistry-keyed aggregate could not say which of two
    products a verdict belonged to -- refuses, and keeps its ``?`` in the reason. Acting on a
    doubtful attribution is the conservative half of the choice; saying so in the reason is what
    stops it being silent. With per-row columns written whenever vetoes are measured these should
    not arise at all.
    """

    def validate(route: SidRoute) -> str | None:
        condemned: list[str] = []

        def walk(node: SidRoute) -> None:
            if node.rule is None:
                return
            # Keyed exactly as `steps_of` keys it for the report, which is what makes the two
            # agree: the precursor sids sorted, the node's own product, the rule it credited.
            precursors = tuple(sorted(child.product for child in node.precursors))
            fired = fired_vetoes(state, node.product, precursors, node.rule)
            if fired:
                condemned.extend(fired)
            for child in node.precursors:
                walk(child)

        walk(route)
        if not condemned:
            return None
        return "vetoed: " + ", ".join(sorted(set(condemned)))

    return validate


def route_report(
    target: int,
    state: GraphState[K],
    selection: RouteSelection,
    *,
    price: StepPrice,
    rule_label: Callable[[int], str],
    name: Callable[[K], str] = str,
    channels: Sequence[str] = ("sites", "competitors"),
    mixture: (
        Callable[[str, tuple[str, ...], int], tuple[ExpectedPeak, ...]] | None
    ) = None,
    diversity_cutoff: int | None = None,
    veto_rationale: Mapping[str, str] | None = None,
    rule_yield: Mapping[int, float] | None = None,
) -> RouteReport:
    """Build the per-step alternatives report for one target's reported routes.

    ``rule_label`` and ``name`` are injected rather than derived, because neither answer belongs to
    the graph: a rule's readable name differs per rule set -- Hartenfeller writes ``classification``
    where the curated projection writes ``ri_name`` -- and a molecule's display form depends on the
    arm, since the Metal arm's search keys are WL hashes and only replay produces SMILES.

    Alternatives come from :func:`vertsy.route_selection.disconnections`, the same function the
    selector chose among, so the report cannot explain a disconnection the selector never saw.

    ``mixture`` is injected for the same reason the two namers are: deriving what else a pot can
    give needs the forward table, its pairings and its condition classes, none of which the graph
    holds. It is called with the step's product, its precursors **as named** and the rule that was
    credited -- the rule because the three classes are defined relative to the intended reaction,
    so without it "another reaction" has nothing to be another than. Given one, each step reports
    the mixture of the disconnection it *took* -- see :class:`RetroStep.mixture`.
    """
    molecule = {}

    def named(sid: int) -> str:
        if sid not in molecule:
            molecule[sid] = name(state.expansions.decode([sid])[0])
        return molecule[sid]

    def cost(rule: int, precursors: tuple[int, ...]) -> float:
        return rule_price(price, rule, precursors)

    def alternative(
        rule: int,
        precursors: tuple[int, ...],
        chosen_cost: float,
        chosen_measurements: Mapping[str, int | None],
        *,
        product: int,
        is_chosen: bool,
        rules_here: Mapping[tuple[int, ...], tuple[int, ...]],
    ) -> Alternative:
        measurements = _measurements(state, channels, precursors, rule)
        # The product is passed because a veto read on it is not a property of the chemistry: two
        # disconnections sharing precursors and rule can condemn one molecule and clear another.
        vetoes = fired_vetoes(state, product, precursors, rule)
        delta = cost(rule, precursors) - chosen_cost
        return Alternative(
            rule=rule,
            rule_label=rule_label(rule),
            precursors=tuple(named(sid) for sid in precursors),
            also_by=tuple(
                other for other in rules_here.get(precursors, ()) if other != rule
            ),
            log_cost=cost(rule, precursors),
            cost_delta=0.0 if is_chosen else delta,
            measurements=measurements,
            vetoes_fired=vetoes,
            standing=(
                "chosen"
                if is_chosen
                else _standing(
                    state, measurements, chosen_measurements, vetoes, precursors, delta
                )
            ),
        )

    def steps_of(
        route: SidRoute,
        elsewhere: Mapping[StepKey, tuple[int, ...]],
    ) -> tuple[RetroStep, ...]:
        found: list[RetroStep] = []

        def walk(node: SidRoute) -> None:
            if node.rule is None:
                return
            taken = tuple(sorted(child.product for child in node.precursors))
            here = disconnections(state, cost, node.product)
            rules_here = {
                precursors: tuple(
                    sorted(
                        {
                            rule
                            for rules, group in state.applications(node.product)
                            if tuple(sorted(group)) == precursors
                            for rule in rules
                        }
                    )
                )
                for _rule, precursors in here
            }
            chosen_cost = cost(node.rule, taken)
            chosen_measurements = _measurements(state, channels, taken, node.rule)
            chosen = alternative(
                node.rule,
                taken,
                chosen_cost,
                chosen_measurements,
                product=node.product,
                is_chosen=True,
                rules_here=rules_here,
            )
            others = tuple(
                alternative(
                    rule,
                    precursors,
                    chosen_cost,
                    chosen_measurements,
                    product=node.product,
                    is_chosen=False,
                    rules_here=rules_here,
                )
                for rule, precursors in here
                if precursors != taken
            )
            product_name = named(node.product)
            found.append(
                RetroStep(
                    index=len(found) + 1,
                    product=product_name,
                    chosen=chosen,
                    alternatives=others,
                    mixture=(
                        mixture(product_name, chosen.precursors, chosen.rule)
                        if mixture is not None
                        else ()
                    ),
                )
            )
            for child in node.precursors:
                walk(child)

        walk(route)
        # Attach "this alternative is live in route k" after the walk, where every route is known.
        return tuple(
            RetroStep(
                step.index,
                step.product,
                step.chosen,
                tuple(
                    replace(
                        option,
                        also_reported_in=elsewhere.get(
                            (step.index, option.rule, option.precursors), ()
                        ),
                    )
                    for option in step.alternatives
                ),
                step.mixture,
            )
            for step in found
        )

    taken_by: dict[StepKey, list[int]] = {}
    for rank, item in enumerate(selection.routes, start=1):
        index = 0

        def collect(node: SidRoute, rank: int = rank) -> None:
            nonlocal index
            if node.rule is None:
                return
            index += 1
            key = (
                index,
                node.rule,
                tuple(
                    named(sid)
                    for sid in sorted(child.product for child in node.precursors)
                ),
            )
            taken_by.setdefault(key, []).append(rank)
            for child in node.precursors:
                collect(child)

        collect(item.route)
    elsewhere = {key: tuple(ranks) for key, ranks in taken_by.items()}

    reported = []
    for rank, item in enumerate(selection.routes, start=1):
        others = [other.route for other in selection.routes[: rank - 1]]
        count, nearest = _differing(item.route, others)
        reported.append(
            ReportedRoute(
                rank=rank,
                standing="best" if rank == 1 else "diverse",
                score=item.score,
                first_distinct_decision=item.first_distinct_decision,
                differing_decisions=count,
                resembled=nearest,
                retro_steps=steps_of(
                    item.route,
                    {
                        key: tuple(r for r in ranks if r != rank)
                        for key, ranks in elsewhere.items()
                    },
                ),
            )
        )

    dropped = tuple(
        ReportedRoute(
            rank=0,
            standing="too-similar",
            score=item.score,
            first_distinct_decision=item.first_distinct_decision,
            differing_decisions=None,
            resembled=item.resembled,
            retro_steps=(),
        )
        for item in selection.too_similar
    ) + tuple(
        ReportedRoute(
            rank=0,
            standing="censored",
            score=item.score,
            first_distinct_decision=None,
            differing_decisions=None,
            resembled=None,
            retro_steps=(),
            reason=item.reason,
        )
        for item in selection.censored
    )
    # Every veto code any row of any reported route cites, with the `?` that marks a code the
    # chemistry-keyed aggregate could not settle stripped off -- a disputed row is still the row a
    # curator should open.
    cited_rules = {
        option.rule
        for route in reported
        for step in route.retro_steps
        for option in (step.chosen, *step.alternatives)
    }
    cited = {
        code.rstrip("?")
        for route in reported
        for step in route.retro_steps
        for option in (step.chosen, *step.alternatives)
        for code in option.vetoes_fired
    }
    return RouteReport(
        target=named(target),
        routes=tuple(reported),
        rejected=dropped,
        truncated=selection.truncated,
        stopped_after_limit=selection.stopped_after_limit,
        channels=tuple(channels),
        diversity_cutoff=diversity_cutoff,
        # Only the rows this report actually cites, so a reader is not handed the whole table and
        # a stored report stays about its own target.
        veto_rationales={
            code: text for code, text in (veto_rationale or {}).items() if code in cited
        },
        rule_yields={
            rule: value
            for rule, value in (rule_yield or {}).items()
            if rule in cited_rules
        },
    )


def report_from_json(data: Mapping[str, Any]) -> RouteReport:
    """Rebuild a report from the JSON it was written as.

    The inverse of ``asdict``, and the reason the whole format exists: extraction writes one object
    per target and *anything* may render it afterwards -- a Markdown page for a judge, a PDF for a
    chemist, a tally for a curator -- without re-running a search. Written out rather than derived
    from the annotations, because the shape is small and fixed, and a reflective loader would fail
    at the one place it matters: ``measurements`` is a mapping whose values are deliberately
    ``int | None``, and nothing in the JSON distinguishes it from the tuples around it.

    Tuples, not lists: JSON has no tuple, and the dataclasses are frozen and hashable.
    """

    def option(row: Mapping[str, Any]) -> Alternative:
        return Alternative(
            rule=row["rule"],
            rule_label=row["rule_label"],
            precursors=tuple(row["precursors"]),
            also_by=tuple(row["also_by"]),
            log_cost=row["log_cost"],
            cost_delta=row["cost_delta"],
            measurements=dict(row["measurements"]),
            vetoes_fired=tuple(row["vetoes_fired"]),
            standing=row["standing"],
            also_reported_in=tuple(row["also_reported_in"]),
        )

    # `.get` rather than `[...]`: a report written before rationales were carried has no such key,
    # and the same tolerance the peak reader below needs for its own older shape.
    def peak_of(row: Mapping[str, Any]) -> ExpectedPeak:
        """One peak, reading a report written before the coincidence tier replaced a bool.

        `isobaric` was a stored field and is now derived from `coincidence`, so an older row
        carries the bool and no tier. The bool said exactly "same formula", which is the top tier,
        so the translation is total rather than a guess -- and the rows it cannot describe are the
        ones that had no tier to lose.
        """
        if "coincidence" not in row and "isobaric" in row:
            fields = dict(row)
            fields["coincidence"] = (
                "same formula" if fields.pop("isobaric") else "resolved"
            )
            return ExpectedPeak(**fields)
        return ExpectedPeak(**row)

    def step(row: Mapping[str, Any]) -> RetroStep:
        return RetroStep(
            index=row["index"],
            product=row["product"],
            chosen=option(row["chosen"]),
            alternatives=tuple(option(one) for one in row["alternatives"]),
            # Absent in reports written before the mixture existed, which stay readable.
            mixture=tuple(peak_of(one) for one in row.get("mixture", ())),
        )

    def route(row: Mapping[str, Any]) -> ReportedRoute:
        return ReportedRoute(
            rank=row["rank"],
            standing=row["standing"],
            score=RouteScore(**row["score"]),
            first_distinct_decision=row["first_distinct_decision"],
            differing_decisions=row["differing_decisions"],
            resembled=row["resembled"],
            retro_steps=tuple(step(one) for one in row["retro_steps"]),
            reason=row.get("reason"),
        )

    return RouteReport(
        target=data["target"],
        routes=tuple(route(one) for one in data["routes"]),
        rejected=tuple(route(one) for one in data["rejected"]),
        truncated=data["truncated"],
        stopped_after_limit=data["stopped_after_limit"],
        channels=tuple(data["channels"]),
        diversity_cutoff=data["diversity_cutoff"],
        veto_rationales=dict(data.get("veto_rationales", {})),
        # Keys come back from JSON as strings; the report keys by rule id.
        rule_yields={int(k): v for k, v in data.get("rule_yields", {}).items()},
    )


def _cell(value: object) -> str:
    """A table cell, with ``None`` shown as an em dash so "not measured" is visible as such."""
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:+.3f}" if value else "0"
    if isinstance(value, tuple):
        return ", ".join(str(item) for item in value) if value else "--"
    return str(value)


def _mixture_lines(mixture: Sequence[ExpectedPeak]) -> list[str]:
    """What to expect in the pot for the step taken, as a table a chemist can read a spectrum by.

    Separate from the decision table above it and under its own heading, because it answers a
    different question. That one is "why this disconnection and not another"; this is "I am about
    to run it -- what will be in the flask". A reader doing the second thing does not want the
    alternatives interleaved, and a reader doing the first does not want masses.

    The isobaric note is the part worth printing even when nothing else is: a competitor sharing
    the product's formula will not appear as a peak of its own, so a clean-looking spectrum is not
    evidence the step was clean. Measured at 31.7% of competitor products, so it is the common
    case rather than a curiosity.
    """
    if not mixture:
        return []
    hidden = [one for one in mixture if one.isobaric]
    out = ["**Expect in the pot:**", ""]
    out += [
        "| compound | exact mass | formula | from |",
        "|---|---|---|---|",
    ]
    for one in mixture:
        mass = "--" if one.exact_mass is None else f"{one.exact_mass:.4f}"
        if one.coincidence == "same formula":
            mass += " (isobaric)"
        elif one.coincidence == "same nominal mass":
            # A different formula at the same integer mass: one peak at unit resolution, two on
            # better kit. Worth saying, and worth not saying with the same word as the first case.
            mass += " (same nominal mass)"
        structure = f"`{one.smiles}`" if one.smiles else "_composition only_"
        out += [f"| {structure} | {mass} | {one.formula} | {one.kind}{_from(one)} |"]
    out += [""]
    unaccounted = [one for one in mixture if one.kind == "atoms lost"]
    if unaccounted:
        out += [
            "`atoms lost` is the step's atom balance and not a compound to look for: those atoms "
            "left the pot as *something*, and only a template that declares its by-product says "
            "what. A Negishi's balance reads `Br2`, where the bromides in fact leave with the "
            "zinc.",
            "",
        ]
    if hidden:
        out += [
            f"{len(hidden)} of these share the product's formula, so no mass separates them from "
            "it: a single product peak is not evidence of a clean step.",
            "",
        ]
    # The second tier gets its own count rather than sharing the first's. A different formula at
    # the same integer mass is a *resolution* problem -- one peak on unit-resolution kit, two on
    # better -- where the first tier is not separable by mass at any resolution. Annotating each
    # peak and never counting them left a reader to tally "(same nominal mass)" out of the table
    # by eye, which is exactly what a summary line is for.
    unresolved = [one for one in mixture if one.coincidence == "same nominal mass"]
    if unresolved:
        out += [
            f"{len(unresolved)} more sit at the product's nominal mass with a different formula, "
            "so they separate on high resolution and not on unit resolution. Reported, not "
            "charged: no part of a route's price reads these counts yet.",
            "",
        ]
    return out


def _from(peak: ExpectedPeak) -> str:
    return f", {peak.source}" if peak.source else ""


def render_report_markdown(report: RouteReport) -> str:
    """One target's routes and their alternatives, as Markdown.

    Plain string building rather than Jinja, unlike
    :func:`vertsy.route_reports.render_route_steps`: that one renders *caller-supplied* templates
    and needs a sandbox for them, while this renders one fixed structure, so a template engine
    would buy nothing and would pull in an optional dependency to do it.

    Steps are numbered in **retro order** -- retro-step 1 is the first disconnection, which is the
    last synthesis step -- while each reaction is written **forwards**, ``A + B -> P``, because that
    is the direction a chemist runs it. The two orders are deliberately different and the heading
    says which is which.
    """
    out: list[str] = [f"# {report.target}", ""]
    measured = ", ".join(report.channels) or "none"
    note = f"{len(report.routes)} route(s) reported"
    similar = [one for one in report.rejected if one.standing == "too-similar"]
    censored = [one for one in report.rejected if one.standing == "censored"]
    if similar:
        note += f", {len(similar)} too similar to report"
    if censored:
        note += f", {len(censored)} refused at validation"
    out += [f"{note}. Channels measured: {measured}."]
    if report.diversity_cutoff is not None:
        out += [
            f"Diversity floor: every route after the first must differ by decision "
            f"{report.diversity_cutoff}."
        ]
    if report.truncated:
        out += [
            "Candidate discovery reached a resource limit, so other eligible plans may exist."
        ]
    out += [""]

    for route in report.routes:
        out += [f"## Route {route.rank} — {route.standing}", ""]
        score = route.score
        out += [
            f"{score.reactions} reaction(s), depth {score.depth}, utility "
            f"{score.utility:.3f} (templates {score.base_step_factor:.3f} x stock "
            f"{score.stock_factor_product:.3f} x selectivity "
            f"{score.reaction_factor_product:.3f})"
        ]
        if route.first_distinct_decision is not None:
            out += [
                f"Differs from route {route.resembled} at decision "
                f"{route.first_distinct_decision}"
                + (
                    f", in {route.differing_decisions} decision(s) overall."
                    if route.differing_decisions is not None
                    else "."
                )
            ]
        out += [""]

        for step in route.retro_steps:
            out += [f"### Retro-step {step.index} — {step.product}", ""]
            chosen = step.chosen
            out += [
                f"**Taken:** `{' + '.join(chosen.precursors)} -> {step.product}` "
                f"— {chosen.rule_label}"
            ]
            if chosen.also_by:
                out += [
                    f"Same disconnection also reachable by rule(s) "
                    f"{', '.join(str(rule) for rule in chosen.also_by)}; the cheapest was "
                    f"credited."
                ]
            out += [""]
            columns = [
                "standing",
                "reaction",
                "rule",
                "yield",
                "cost",
                *report.channels,
                "vetoes",
                "also in",
            ]
            out += [
                "| " + " | ".join(columns) + " |",
                "|" + "|".join(["---"] * len(columns)) + "|",
            ]
            # The chosen step is the table's first row, not only its heading. What the probe
            # measured *on the step taken* is the whole diagnostic: a veto row firing on a step
            # the report recommends, beside a `sites` count of 1, is a forward-pattern gap stated
            # in one line. Leaving it out of the table put that line nowhere a reader would look.
            for option in (chosen, *step.alternatives):
                cells = [
                    option.standing,
                    f"`{' + '.join(option.precursors)} -> {step.product}`",
                    option.rule_label,
                    _cell(report.rule_yields.get(option.rule)),
                    _cell(option.cost_delta),
                    *(
                        _cell(option.measurements.get(channel))
                        for channel in report.channels
                    ),
                    _cell(option.vetoes_fired),
                    _cell(tuple(f"route {rank}" for rank in option.also_reported_in)),
                ]
                out += ["| " + " | ".join(cells) + " |"]
            out += [""]
            if not step.alternatives:
                out += ["No other disconnection of this molecule is in the graph.", ""]
            # The chemistry behind any code the table just cited. A row name tells a reader which
            # entry to look up; the rationale is the part they can disagree with, which is the only
            # way a wrong veto gets found.
            fired = sorted(
                {
                    code.rstrip("?")
                    for option in (chosen, *step.alternatives)
                    for code in option.vetoes_fired
                }
            )
            stated = [
                f"- `{code}` — {report.veto_rationales[code]}"
                for code in fired
                if code in report.veto_rationales
            ]
            if stated:
                out += ["Vetoes cited above:", "", *stated, ""]
            out += _mixture_lines(step.mixture)

    if similar:
        out += ["## Rejected as too similar", ""]
        out += [
            "These were good enough to report and too much like a route already reported. Widen "
            "the diversity floor to see them.",
            "",
            "| resembled | first differed at | reactions | utility |",
            "|---|---|---|---|",
        ]
        for dropped in similar:
            out += [
                f"| route {dropped.resembled} | decision "
                f"{dropped.first_distinct_decision} | {dropped.score.reactions} | "
                f"{dropped.score.utility:.3f} |"
            ]
        out += [""]
    # Kept apart from the section above, because the two are not the same kind of absence: one is
    # a knob this reader can widen, the other is a route the engine says cannot be made.
    if censored:
        out += ["## Refused at validation", ""]
        out += [
            "These were costed and ranked, then refused when the plan was replayed against real "
            "molecules. Each was replaced by the next route, so nothing below was crowded out.",
            "",
            "| refused because | reactions | utility |",
            "|---|---|---|",
        ]
        for lost in censored:
            out += [
                f"| {lost.reason} | {lost.score.reactions} | {lost.score.utility:.3f} |"
            ]
        out += [""]
    return "\n".join(out)

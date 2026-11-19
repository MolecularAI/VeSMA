"""What one step costs, and the one place a substrate gets to change it.

:mod:`vertsy.metrics` prices an application at its cheapest template, and until now the only
thing that could set that price was ``rule_cost(rule)`` -- a pure function of a template id, and
therefore unable to say anything about the molecule the template was applied to. That limit is
the one :mod:`vertsy.filters` names in its own docstring, and it is what a selectivity, yield or
regiochemistry score needs to get past: those are properties of a *reaction on a particular
substrate*, not of a rule in the abstract.

A price is therefore a function of ``(rule, precursors)``, and deliberately **not** of the
product. A forward reaction is determined by its rule and its starting materials; the product is
what that reaction makes. Two products reached from the same precursors by the same rule
genuinely share the forward chemistry, so they must share the price -- and dropping the product
from the key is what keeps the lookup small enough for the hot path.

The substrate enters through exactly one number: ``outcomes``, a measurement of what the forward
chemistry does with those precursors. Everything downstream is a function of ``(rule, outcomes)``
and nothing else. That is the whole reason the array fold stays in arrays: a graph names few
distinct templates and measurements are small integers, so the distinct ``(rule, outcomes)``
pairs are few and the price is evaluated once per pair rather than once per row.

Prices are min-plus costs in whatever unit a heuristic is commensurable with. The empty template
set prices at infinity: a disconnection nothing achieves is unaffordable, which is the right
reading rather than an error.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .store import SidStep

__all__ = [
    "SILENT",
    "UNPROBED",
    "uncurated",
    "Aggregate",
    "Counts",
    "Multiplier",
    "Outcomes",
    "Penalties",
    "StepPrice",
    "Substrate",
    "application_price",
    "attaining_rule",
    "Factor",
    "Factors",
    "by_rule",
    "compounded",
    "constant_step",
    "cost_of_factor",
    "doubted",
    "extra_steps",
    "factor_of_cost",
    "flat",
    "graded",
    "Table",
    "lowest_rule",
    "measured",
    "rule_price",
    "selective",
    "banded",
    "shared",
    "steps_equivalent",
    "surcharged",
    "unmeasured",
]

# How many steps one step is worth, given a measurement of it. ``1.0`` is "worth exactly
# itself", which is what an unmeasured or clean step must return.
type Multiplier = Callable[[int], float]
# What was measured about applying one template to one precursor multiset, one count per channel in
# the price's declared order. The unit is "distinct products the forward chemistry makes from these
# precursors", with `UNPROBED` where no measurement exists. Empty when nothing is measured at all.
type Counts = tuple[int, ...]
type Outcomes = Callable[[int, tuple[int, ...]], Counts]
# What one measurement is worth, per channel, before the channels are combined.
type Penalties = Mapping[str, Multiplier]
# How several channels' worths become one number. A step is worth its extra steps, added up.
# Distinct from `vertsy.selectivity.Combine`, which merges molecules rather than numbers.
type Aggregate = Callable[[Sequence[float]], float]
# What fraction of the material survives a measurement of this channel: a yield in `(0, 1]`, where
# `1.0` is "loses nothing" and is what an unmeasured or clean step must return.
#
# The sibling of `Multiplier` and the other way to read the same measurement. A multiplier answers
# "how many steps is this worth", which is scale-free and has to be, because `m` is multiplied by
# whatever the caller's base happens to be. A factor answers "what share of the substrate becomes
# the product", which is *absolute*: a 70% step loses 30% of the material whatever the search's
# schedule charges per step. The two are not interchangeable, and `compounded` is the price that
# wants this one.
type Factor = Callable[[int], float]
# What each channel costs as a surviving fraction, per channel, in the price's declared order.
type Factors = Mapping[str, Factor]
# How a price becomes substrate-aware: given a base price and the graph's measurement tables, a
# price that reads them. `selective` and `compounded` are the two shapes, partially applied to
# their own parameters -- `partial(selective, penalties=...)` or `partial(compounded, factors=...)`.
#
# A function and not a finished price, because *when* the composition happens is not the caller's
# business: the tables are a property of the graph and the graph is rebuilt every pass, so the
# price has to be recomposed per pass. *Which* shape composes is entirely the caller's, because the
# two say different things about one measurement -- a multiple of a step, or a fraction of the
# material.
type Substrate = Callable[[StepPrice, Mapping[str, Table]], StepPrice]


UNPROBED = -1
"""No measurement exists for this ``(rule, precursors)`` on this channel.

Out of band rather than ``0`` because a count of zero is a *result*: the forward chemistry ran and
made nothing. Conflating the two is what let a broken pairing win a route -- a rule whose forward
partner could not fire looked exactly like a rule nobody had paired, and both were free.
"""

SILENT = 0
"""The probe ran and the forward chemistry made nothing from this application's own precursors.

A correct pairing cannot do this: the forward rule is the retro rule's inverse on this application,
so if the retro rule fired the forward rule must, and the intended product is itself one of the
outcomes. So silence is evidence of one of two things, and the count cannot separate them -- the
retro rule proposed partners the reaction does not accept, or the forward rule was flattened too
narrowly by hand. Hence :func:`doubted`: charged as low confidence, never refused.
"""


def unmeasured(rule: int, precursors: tuple[int, ...]) -> Counts:
    """Nothing was measured, on any channel. Distinct from a count of ``1``, which says "clean"."""
    return ()


def extra_steps(worths: Sequence[float]) -> float:
    """Several channels' worths as one: a step is worth its extra steps, added up.

    The alternative -- multiplying the worths -- reads a step that is both a three-way mixture and
    has one competing reaction as worth twelve steps, which is not a claim any of the measurements
    support. Adding the *extras* says four steps for the mixture and two more for the competitor:
    six, which is the arithmetic a chemist would do on a synthesis plan.

    Empty in, ``1.0`` out, so a price with no channels charges exactly its base.
    """
    return 1.0 + sum(worth - 1.0 for worth in worths)


@dataclass(frozen=True)
class StepPrice:
    """A step cost as its independent decisions, plus one fact about their composition."""

    base: Callable[[int], float]
    """What one template costs before any substrate is considered.

    A pure function of a rule id. Kept as its own field rather than recovered from ``charge`` at
    an unmeasured measurement so that wrapping a price is unambiguous: every shape below builds
    its ``charge`` out of ``base``, so a price carries one penalty rather than a stack of them
    whose composition nobody has defined.
    """

    charge: Callable[[int, Counts], float]
    """``(rule, counts) -> cost``: the whole price, and the only function a consumer needs.

    Two arguments and not one per channel, because a substrate reaches a price only through its
    measurements and those arrive as a tuple. That is what lets
    :func:`vertsy.tropical.shape_of` evaluate this once per distinct ``(rule, counts)`` row and
    gather the rest: a graph names few templates, a measurement is a small integer, and there are
    two or three channels, so the distinct rows stay few however large the graph gets.
    """

    channels: tuple[str, ...] = ()
    """What each position in ``counts`` measures, in order.

    Named because the two channels that exist measure different chemistry and deserve different
    penalties. **Competing sites** is the intended reaction happening in more than one place: the
    product is a mixture, and the cost is a separation. **Competing reactions** is a *different*
    reaction the same starting materials could undergo, which may mean the intended step does not
    work at all. Collapsing them to one number would price a separation and a failure alike.

    Ordered, rather than a mapping, so a count vector is hashable and the array fold can group on
    it. :func:`selective` derives the order from the penalties it is given, so the two cannot
    disagree.
    """

    outcomes: Outcomes = unmeasured
    """Where the measurements come from, keyed by ``(rule, precursors)``, in ``channels`` order.

    Defaults to measuring nothing, so a price built without one charges exactly what ``base``
    says and every existing caller is unaffected.
    """

    uniform: bool = True
    """Whether every application *necessarily* costs the same. Configuration, not data.

    A selectivity-aware price is non-uniform from the first pass, even while no application has
    yet been measured and every cost is still equal. Consumers that require uniformity --
    :func:`vertsy.tropical.bottleneck_leaf_scores` and the selector over it -- must refuse on
    this flag rather than on the numbers, or they work for eight passes and fail on the ninth,
    the moment the first mixture is measured.
    """


def rule_price(price: StepPrice, rule: int, precursors: tuple[int, ...]) -> float:
    """What one template costs on one precursor multiset."""
    return price.charge(rule, price.outcomes(rule, precursors))


def application_price(price: StepPrice, step: SidStep) -> float:
    """What one disconnection costs: its cheapest template, priced in context.

    Min-plus takes the cheapest of alternatives, and *in context* is the point -- a disconnection
    reachable selectively by one template and unselectively by another is a selective
    disconnection, and pricing it at the unselective template would charge chemistry nobody has
    to run. An empty template set is infinite: nothing achieves it.
    """
    rules, precursors = step
    if not rules:
        return math.inf
    return min(rule_price(price, rule, precursors) for rule in rules)


def attaining_rule(price: StepPrice, step: SidStep) -> int:
    """The template that attained :func:`application_price`, ties to the lowest id.

    A step must be labelled with the template its price was selected for, or a route's stated
    chemistry is at odds with the reason it was chosen. Under a uniform price every template ties
    and this is the lowest id, which is what unit costs always gave.
    """
    rules, precursors = step
    return min(rules, key=lambda rule: (rule_price(price, rule, precursors), rule))


def lowest_rule(step: SidStep) -> int:
    """The lowest template id of a disconnection: a labelling with no cost model in it.

    ``rules`` is sorted, so this *is* :func:`attaining_rule`'s tie-break under a uniform price. It
    exists so an enumerator that deliberately has no price --
    :func:`vertsy.routes.iter_solved_sid_routes` counts routes rather than ranking them -- can
    name its convention instead of leaving it to an index.
    """
    rules, _precursors = step
    return rules[0]


# One channel's measurements, keyed the way a price asks for them.
type Table = Mapping[tuple[tuple[int, ...], int], int]


def measured(tables: Mapping[str, Table], channels: Sequence[str]) -> Outcomes:
    """Read measurements out of the tables :attr:`vertsy.metrics.GraphState.outcomes` holds.

    Mappings rather than the graph itself, so a price never learns what a graph is -- which keeps
    this module under the fold rather than beside it, and lets a test write a table by hand.
    Precursors are sorted into the key because a multiset has no order and the callers that build
    one need not agree on an accident of iteration.

    A channel the graph carries no table for reads :data:`UNPROBED` throughout, which is "nobody
    measured this" and costs nothing. That is what lets one channel be wired before the other -- and
    it is deliberately *not* ``0``, which a probe writes when it ran and found nothing.
    """
    ordered = tuple(tables.get(channel, {}) for channel in channels)

    def outcomes_of(rule: int, precursors: tuple[int, ...]) -> Counts:
        key = (tuple(sorted(precursors)), rule)
        return tuple(table.get(key, UNPROBED) for table in ordered)

    return outcomes_of


def doubted(multiplier: Multiplier, *, silent: float) -> Multiplier:
    """``multiplier``, but charging ``silent`` where the probe came back silent.

    Silence says the reaction probably does not work, and *probably* is the whole reason this is a
    factor and not a veto: the same observation is produced by a retro rule proposing partners its
    reaction will not couple and by a forward rule this repo flattened too narrowly, and nothing in
    a product count separates them. A lint catches the second kind where an example exercises it
    (`tests/vertsy/test_forward_templates.py`), and cannot promise to catch all of it -- so the
    branch stays explorable, priced at low confidence, rather than being removed from the graph.

    Left out, silence costs exactly what it costs today, which is nothing. That is the wrong default
    for a route report and the right one for a change that has to be measurable against the old
    numbers, so the factor is a parameter and the composition is explicit.
    """

    def worth(outcomes: int) -> float:
        return silent if outcomes == SILENT else multiplier(outcomes)

    return worth


def uncurated(multiplier: Multiplier, *, unprobed: float) -> Multiplier:
    """``multiplier``, but charging ``unprobed`` where nobody measured the step at all.

    The sibling of :func:`doubted`, and a **different kind of claim**, which is why it is a separate
    combinator rather than a second argument. Silence is an observation about the chemistry: the
    probe ran and the reaction made nothing. `UNPROBED` is an observation about *us* -- this rule has
    no forward partner, so no one has curated what it does. Charging for it prices our own ignorance,
    not any property of the molecule.

    That is a legitimate policy and worth stating as one: it steers a search toward the part of the
    library whose selectivity has actually been examined, which is a reasonable preference when the
    uncurated part is where the scorer's gaps were found. It is **not** evidence that the unprobed
    chemistry is worse. A rule set nobody has written forward rules for would be charged throughout,
    uniformly and for no chemical reason, which is exactly the confusion the sentinel was introduced
    to prevent -- so the default is ``1.0`` and the charge has to be asked for.

    Keep it small. At ``1.2`` it breaks ties toward examined chemistry; at a value that could pay for
    a protection detour it would be buying rearrangements with a number that means nothing about the
    reaction.

    **Measured, and it charged the right answer.** On two targets a route judge had ranked, the
    correct step was an mCPBA epoxidation -- `MC 62`, which has no forward partner and so reads
    `UNPROBED` -- while the wrong step was a glycidyl Grignard whose forward rule fires and reports
    one clean product. Pricing `UNPROBED` therefore made the *correct* route more expensive and
    left the impossible one at 1.0, at both 1.2 and 3.0. Which side this knob penalises is decided
    by which rules happen to be curated, and that is uncorrelated with which chemistry is right. It
    stays available and documented, and the default stays ``1.0``.
    """
    if not math.isfinite(unprobed) or unprobed < 1.0:
        raise ValueError(
            f"an unprobed factor must be finite and at least 1, got {unprobed!r}."
        )

    def worth(outcomes: int) -> float:
        return unprobed if outcomes == UNPROBED else multiplier(outcomes)

    return worth


def by_rule(base: Callable[[int], float], *, uniform: bool = True) -> StepPrice:
    """A price that reads the template and nothing else -- a template prior, and no substrate.

    Where a per-template number exists and a per-substrate one does not: reaction-class frequency,
    a curated confidence, a learned feasibility over rules. :func:`constant_step` is the degenerate
    case, and :func:`selective` is how a substrate is added on top without disturbing this.

    ``uniform`` defaults to ``True`` because the two callers that existed when this was written
    both *are* uniform: :func:`constant_step` charges one number for every rule, and
    :func:`vertsy.aostar.anytime_step` reads the current rung, which every rule shares. A base that
    genuinely differs per template -- a curated yield, say -- must pass ``False``, or it claims
    costs cannot differ while they do.

    That claim is load-bearing rather than decorative. :attr:`StepPrice.uniform` exists so
    :func:`vertsy.tropical.bottleneck_leaf_scores` and the selector over it refuse on the
    *configuration* instead of on the numbers; a price that lies here passes the check at
    `vertsy.aostar` and then trips the array backstop mid-run, which is the "works for eight passes
    and fails on the ninth" failure the flag was added to prevent -- and it trips it with a message
    blaming :func:`selective`.
    """

    def charge(rule: int, counts: Counts) -> float:
        return base(rule)

    return StepPrice(base=base, charge=charge, uniform=uniform)


def constant_step(cost: float) -> StepPrice:
    """Charge ``cost`` per reaction, whatever the template and whatever the substrate.

    The shape a step cost takes until per-template or per-substrate data exists. Under it a
    route's value counts its reactions, scaled -- and min-plus ranking is invariant under that
    scaling, so what the number really fixes is its ratio to the heuristic's range.
    """
    if not cost > 0.0:
        raise ValueError(f"step cost must be positive, got {cost!r}.")
    return by_rule(lambda rule: cost)


def steps_equivalent(outcomes: int) -> float:
    """A step making ``n`` distinct products is worth ``n + 1`` steps. Clean steps are free.

    ``n`` products' worth of material, plus one separation. The offset is not decoration. The
    workaround for an unselective step is to protect, react and deprotect, which is **three**
    steps under either aggregation -- a protection detour is linear, so it costs the same whether
    an application's precursors are summed (reaction count) or maxed (longest linear sequence).
    A multiplier therefore has to *exceed* 3 to prefer the detour, and the bare ``m(n) = n`` ties
    at exactly the case this was written for: a pentitol has three distinct monosubstitution
    products, not five, because two of its five hydroxyls are equivalent.

    ====  ===  ============================
    n     m    against a three-step detour
    ====  ===  ============================
    -1    1    unprobed, costs nothing
    0     1    silent -- see :func:`doubted`
    1     1    clean, costs nothing
    2     3    ties
    3     4    the detour wins by one step
    5     6    the detour wins by three
    ====  ===  ============================

    ``0`` is left free here so that adding the sentinel changes no number on its own; wrap this in
    :func:`doubted` to charge it.

    Ties at ``n = 2`` on purpose. The measurement counts products the chemistry *can* make, not
    the ratio it makes them in, so a two-way split is as often a clean 20:1 as a 1:1; a penalty
    that decided it either way would claim information the probe does not have.
    """
    return 1.0 if outcomes < 2 else float(outcomes + 1)


def flat(multiple: float) -> Multiplier:
    """Any mixture is worth ``multiple`` steps; a clean step is worth itself.

    Coarser than :func:`steps_equivalent` -- it reads a two-way and a six-way ambiguity alike --
    and kept because it is the shape that states a single threshold. Pricing a mixture step at a
    utility of 0.5 where a clean one is 0.9 is this shape at ``multiple = 6.5``, since
    ``0.9 ** 6.5`` is about 0.5.
    """
    if not math.isfinite(multiple) or multiple < 1.0:
        raise ValueError(
            f"a multiplier must be finite and at least 1, got {multiple!r}."
        )

    def worth(outcomes: int) -> float:
        return multiple if outcomes >= 2 else 1.0

    return worth


def graded(slope: float, *, cap: float = math.inf) -> Multiplier:
    """A step making ``n`` distinct products is worth ``1 + slope*(n - 1)`` steps, up to ``cap``.

    The shape between :func:`flat`, which reads a two-way and a seven-way ambiguity alike, and
    :func:`steps_equivalent`, which charges ``n + 1`` and so prices a seven-way reading at eight
    steps of synthesis. Neither is right for the ``competitors`` channel: a count of seven there
    means *the forward table lists seven reactions these materials could undergo*, which is a
    warning worth more than a two-way split and much less than eight steps.

    **The law every shape here obeys, and the reason this one exists.** A multiplier is
    non-decreasing in ``n`` for ``n >= 1``: more products, or more competing reactions, may cost
    the same but may never cost *less*. Saturating is allowed, which is what ``cap`` is for, and
    is why `flat` is a legal member of the family -- it is this law at its coarsest, one step up
    at ``n = 2`` and a plateau thereafter.

    ====  ==========  =================  =====  ========
    n     slope=0.1   slope=0.1 cap=1.6  flat:1.2  steps
    ====  ==========  =================  =====  ========
    -1    1           1                  1      1
    0     1           1                  1      1
    1     1           1                  1      1
    2     1.1         1.1                1.2    3
    3     1.2         1.2                1.2    4
    7     1.6         1.6                1.2    8
    12    2.1         1.6                1.2    13
    ====  ==========  =================  =====  ========

    The sentinels are free, as everywhere: ``1`` is the multiplicative identity a route's utility
    decomposition needs, ``0`` is :data:`SILENT` and belongs to :func:`doubted`, ``-1`` is
    :data:`UNPROBED` and belongs to :func:`uncurated`. ``slope = 0`` is inert and legal, which is
    the useful null member of a sweep.

    **Why the shape was built.** On the five recommended steps a veto row condemned over six
    targets, the competitor counts were 7, 4, 1, 1 and 1 -- so under `flat:1.2` three of the five
    paid nothing, because the plateau starts at 2, and the seven-way step paid exactly what a
    two-way one pays. Flat's single step *up* lands at ``n = 2``, which on those targets is where
    the **correct** answer sits: the mCPBA epoxidation `MC 62` reads two competitors while the
    glycidyl Grignard that displaced it reads one.

    **And the measurement that says gradation is not the lever.** Those six targets were the ones
    the veto rows had been written while watching, so agreeing with them was never a blind test.
    On twenty fresh PaRoutes targets, eighteen of which reported a route, veto rows fired on the
    top route 12 times under `flat:1.2` and **12 times under `graded:0.1`** -- 13 under
    `graded:0.2` -- while the strong shapes fired 9 (`flat:3`), 10 (`graded:0.5` capped at 3) and
    11 (`steps`). What moves that number is the price's *magnitude*, not its shape, and magnitude
    is exactly what two blind route judges tied on (9-7, then 6-6, two-sided p = 1.00). So this
    shape is the right *statement* about a competitor count and is not, on the evidence, a better
    price than the flat one it was built to replace. The default stayed `flat:1.2`.
    """
    if not math.isfinite(slope) or slope < 0.0:
        raise ValueError(f"a slope must be finite and non-negative, got {slope!r}.")
    if math.isnan(cap) or cap < 1.0:
        raise ValueError(f"a cap must be at least 1, got {cap!r}.")

    def worth(outcomes: int) -> float:
        if outcomes < 2:
            return 1.0
        return min(cap, 1.0 + slope * (outcomes - 1))

    return worth


def shared(share: float, *, floor: float = 0.0) -> Factor:
    """Each competitor beyond the first takes a ``share`` of what is left: ``share ** (n - 1)``.

    The :class:`Factor` counterpart of :func:`graded`, and the honest reading of what a count can
    say. The count is of products the chemistry *can* make and not of the ratio it makes them in --
    :func:`steps_equivalent` declines to guess that ratio for exactly this reason -- so ``share`` is
    a **declared prior** and not a measurement, and naming it as a yield is what makes it arguable.
    A chemist can disagree with 0.8 per competing site; nobody can disagree with "worth three
    steps", which is the problem.

    ``floor`` stops the geometry running away on a six-way mixture, the way :func:`graded`'s ``cap``
    does: without it, eight competitors at 0.8 read as a 21% yield, which claims more resolution
    than a count has.

    Free below two, so :data:`UNPROBED` and :data:`SILENT` both survive intact and adding the
    sentinels changes no number on its own -- the same choice :func:`steps_equivalent` makes. Charge
    them by wrapping this, as the multipliers are wrapped.
    """
    if not math.isfinite(share) or not 0.0 < share <= 1.0:
        raise ValueError(f"a per-competitor share must be in (0, 1], got {share!r}.")
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"a yield floor must be in [0, 1], got {floor!r}.")

    def surviving(outcomes: int) -> float:
        if outcomes < 2:
            return 1.0
        return max(floor, share ** (outcomes - 1))

    return surviving


def compounded(
    price: StepPrice,
    tables: Mapping[str, Table],
    factors: Factors,
) -> StepPrice:
    """Charge ``price`` plus what each channel's surviving fraction costs, in log space.

    ``charge(rule, counts) = base(rule) + sum(-log f_c(count_c))``. Nothing is multiplied by
    anything, which is the whole point: a yield is a fraction of material, fractions compose by
    multiplying, and :func:`cost_of_factor` is the change of variable that turns that into the
    addition min-plus wants. So a step that is 90% selective at one site and 80% at another is
    worth 0.72 of itself, and the arithmetic says so in one line.

    **Why not :func:`selective`.** That shape charges ``base(rule) * m``, where ``m`` is "how many
    steps this is worth". It has to be scale-free, because the search and the report use different
    base scales and one multiplier must be correct under both. A yield is not scale-free and should
    not be: a 70% step loses 30% of the material whatever the schedule charges per step. Routing a
    yield through a step-multiple -- ``base + scale * (worth - 1)`` -- sums a quantity that
    multiplies and divides one that adds, and the units stop meaning anything.

    The consequence is real and is the price of saying something true: under
    :func:`vertsy.aostar.anytime_step`'s greedy rung a step costs 0.001, against which
    ``-log 0.7 = 0.36`` is not a nudge. The greedy dive is reordered by this where
    :func:`selective` left it alone. That makes the rung a *search schedule* rather than a
    chemistry scale, which is what it always was.

    Three properties, each of which something downstream depends on:

    * **Clean is free.** Every :class:`Factor` returns exactly ``1.0`` below two, so
      ``charge(rule, (1,) * n) == base(rule)``. :func:`vertsy.route_selection.clean_log_cost` needs
      no change and :func:`factor_of_cost` never sees a negative remainder.
    * **The report's factor is the product.** ``reaction_cost - clean_cost`` is exactly
      ``-log(prod f_c)``, so the printed selectivity *is* the product of the per-channel yields --
      for every rule, and independently of what its base happens to be.
    * **Non-uniform, and it says so**, since a channel measured on one application and not another
      makes costs differ.

    A channel with no table reads :data:`UNPROBED` throughout and costs nothing, so one channel can
    be wired and measured before the other exists -- the same affordance :func:`selective` has.
    """
    base = price.base
    channels = tuple(factors)
    surviving = tuple(factors[channel] for channel in channels)

    def charge(rule: int, counts: Counts) -> float:
        return base(rule) + sum(
            cost_of_factor(factor(count))
            for factor, count in zip(surviving, counts, strict=False)
        )

    return StepPrice(
        base=base,
        charge=charge,
        channels=channels,
        outcomes=measured(tables, channels),
        uniform=False,
    )


def selective(
    price: StepPrice,
    tables: Mapping[str, Table],
    penalties: Penalties,
    *,
    combine: Aggregate = extra_steps,
) -> StepPrice:
    """Charge a multiple of ``price`` where the measurements say the step is unselective.

    ``penalties`` names a multiplier per channel and, by its order, fixes the order of a count
    vector -- so the channels a price declares and the tables it reads cannot disagree, because
    both come from here. A channel with no table reads ``0`` throughout and costs nothing, which
    is what lets one channel be wired and measured before the other exists.

    ``combine`` turns several channels' worths into one, and defaults to :func:`extra_steps`.

    Scale-free, which is what makes one multiplier correct under both the search's step cost and
    the report's step utility -- and also its one limitation. Under
    :func:`vertsy.aostar.anytime_step`'s greedy rung a step costs 0.001, so a fourfold penalty is
    0.004 and negligible against a ``[0, 1]`` estimate: the search is reordered in the relaxed
    phase and not before. That is where the measured gain from relaxing lives anyway, but where
    it is not wanted, :func:`surcharged` anchors the charge to a fixed scale instead.

    Admissibility is untouched. A penalty only ever raises a cost, and an estimate is free of
    charge while it stays below the *cheapest* step -- which this cannot lower.
    """
    base = price.base
    channels = tuple(penalties)
    worth_of = tuple(penalties[channel] for channel in channels)

    def charge(rule: int, counts: Counts) -> float:
        return base(rule) * combine(
            [penalty(count) for penalty, count in zip(worth_of, counts)]
        )

    return StepPrice(
        base=base,
        charge=charge,
        channels=channels,
        outcomes=measured(tables, channels),
        uniform=False,
    )


def surcharged(
    price: StepPrice,
    tables: Mapping[str, Table],
    penalties: Penalties,
    *,
    scale: float,
    combine: Aggregate = extra_steps,
) -> StepPrice:
    """Like :func:`selective`, but the extra steps cost ``scale`` rather than ``base``.

    An unselective step costs ``base(rule) + scale * (m - 1)``: the same *number* of extra steps
    :func:`selective` charges, valued at a scale the caller fixes instead of at whatever the
    schedule currently charges. Pass the relaxed rung and a mixture costs three extra reactions
    from the first pass -- which is the point. It buys reordering during the greedy dive, and
    pays for it by no longer being invariant under rescaling the whole cost system.
    """
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"a surcharge scale must be finite and positive, got {scale!r}."
        )
    base = price.base
    channels = tuple(penalties)
    worth_of = tuple(penalties[channel] for channel in channels)

    def charge(rule: int, counts: Counts) -> float:
        worth = combine([penalty(count) for penalty, count in zip(worth_of, counts)])
        return base(rule) + scale * (worth - 1.0)

    return StepPrice(
        base=base,
        charge=charge,
        channels=channels,
        outcomes=measured(tables, channels),
        uniform=False,
    )


def banded(survivals: Sequence[float]) -> Factor:
    """A :class:`Factor` that reads a **band ordinal** rather than a count, from a lookup.

    ``survivals[0]`` is what band ``1`` keeps and has to be exactly ``1.0``, because a channel's
    ``1`` is the price's free reference -- see :func:`vertsy.route_selection.clean_log_cost`. Each
    entry after it is what the next band keeps, and the sequence has to be non-increasing: a worse
    band may keep the same share but never more. An ordinal past the end reads as the last entry,
    so a band vocabulary can grow without the curve lying about the bands it does not know.

    **Why a table and not a curve.** :func:`shared` and :func:`graded` are functions of *how many*,
    and they are the right shape for a count: competitors compound, so a geometry is the honest
    form. A severity band is not a quantity -- three mild competitors are not a worse band than one
    -- so there is nothing for a geometry to compound, and the relationship between band and
    surviving fraction is whatever the chemistry says it is. A lookup states that relationship
    without interpolating between bands nobody measured.

    This is what lets a *relative* quantity reach a :class:`Factor` at all. A factor is a function
    of one integer and cannot see which rule fired or what the intended reaction was, so a
    competition -- which is relative by nature -- looks unreachable. Computing the band where both
    reactions are known and letting the factor merely decode it is the way across, and
    :func:`vertsy.selectivity.severity_outcomes` is the other half.
    """
    table = tuple(float(one) for one in survivals)
    if not table:
        raise ValueError("a banded factor needs at least the clean band.")
    if table[0] != 1.0:
        raise ValueError(
            f"band 1 is the price's free reference and must keep 1.0, got {table[0]!r}."
        )
    for share in table:
        if not math.isfinite(share) or share <= 0.0 or share > 1.0:
            raise ValueError(f"a surviving fraction must be in (0, 1], got {share!r}.")
    for earlier, later in zip(table, table[1:], strict=False):
        if later > earlier:
            raise ValueError(
                "a worse band may keep the same share but never more; "
                f"{later!r} follows {earlier!r}."
            )

    def surviving(outcomes: int) -> float:
        if outcomes < 1:
            # `UNPROBED` and anything below the clean band: nothing was measured, so nothing is
            # charged. The same reading `measured` gives a channel the graph carries no table for.
            return 1.0
        return table[min(outcomes - 1, len(table) - 1)]

    return surviving


def cost_of_factor(factor: float) -> float:
    """A multiplicative utility in ``(0, 1]`` as an additive min-plus cost.

    The report states a step's worth as a factor and the search states it as a cost; these are one
    quantity and this is the change of variable between them. A factor of 1 is free, and a factor
    above 1 would be a reaction that makes a route better the more of them it has.
    """
    if not math.isfinite(factor) or factor <= 0.0 or factor > 1.0:
        raise ValueError(
            f"a route factor must be finite and in (0, 1], got {factor!r}."
        )
    return -math.log(factor)


def factor_of_cost(cost: float) -> float:
    """The inverse of :func:`cost_of_factor`: an additive cost as a multiplicative utility."""
    if not math.isfinite(cost) or cost < 0.0:
        raise ValueError(f"a step cost must be finite and non-negative, got {cost!r}.")
    return math.exp(-cost)

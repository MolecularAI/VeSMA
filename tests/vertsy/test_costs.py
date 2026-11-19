"""A step price is a cost model, and the first thing it has to earn is changing nothing.

Every existing number in this package was produced by ``rule_cost(rule)``. A price that can also
read the substrate is only trustworthy if, given nothing to read, it reproduces those numbers
exactly -- not to a tolerance, bit for bit, because a tolerance would hide precisely the
reassociation a new multiply could introduce.
"""

from __future__ import annotations

import math
import operator
from functools import partial

import numpy as np
import pytest

from vertsy.costs import (
    SILENT,
    UNPROBED,
    StepPrice,
    application_price,
    attaining_rule,
    by_rule,
    compounded,
    constant_step,
    cost_of_factor,
    doubted,
    extra_steps,
    factor_of_cost,
    flat,
    graded,
    lowest_rule,
    measured,
    rule_price,
    selective,
    shared,
    steps_equivalent,
    surcharged,
    uncurated,
)
from vertsy.expansions import Expansions
from vertsy.interner import DictInterner
from vertsy.metrics import TROPICAL, outside_valuation, prepare_graph, valuation
from vertsy.store.numpy_store import NumpyExpansionStore
from vertsy.tropical import shape_of, tropical_valuation

from .handbuilt import add_application


def sites(count: int | None) -> dict[str, int]:
    """One channel's measurement, or none at all. The channel a probe wrote is part of the data."""
    return {} if count is None else {"sites": count}


def graph(*, outcomes: dict[int, int] | None = None):
    """``T`` from two templates on one disconnection, plus a second disconnection.

    Two templates on one application is the case that decides whether a price is read per
    template or per disconnection, and it is the common case rather than the exotic one.
    """
    expansions: Expansions[str] = Expansions(NumpyExpansionStore(), DictInterner())
    target, stock, other = expansions.intern(["T", "S", "O"]).tolist()
    expansions.mark_expanded(np.array([target], dtype=np.int64))
    measured = outcomes or {}
    app = add_application(
        expansions, target, 2, [stock], outcomes=sites(measured.get(2))
    )
    add_application(
        expansions, target, 5, [stock], app=app, outcomes=sites(measured.get(5))
    )
    add_application(expansions, target, 7, [other], outcomes=sites(measured.get(7)))
    return expansions, prepare_graph(expansions, {"S"}), target


# Graphs whose applications are measured *differently*, which is the only way a per-application
# cost can be wrong in a way a uniform one cannot. Keyed by rule id, since that is what a
# measurement belongs to.
MEASURED_GRAPHS: dict[str, tuple[dict, set[str], dict[int, int]]] = {
    "one mixture": ({"T": [(0, ["A"]), (1, ["B"])]}, {"A"}, {0: 3}),
    "both measured": ({"T": [(0, ["A"]), (1, ["B"])]}, {"A", "B"}, {0: 3, 1: 2}),
    "and siblings": (
        {"T": [(0, ["A", "B"])], "A": [(1, ["C"])]},
        {"B", "C"},
        {0: 5, 1: 2},
    ),
    "convergent": (
        {"T": [(0, ["A", "B"])], "A": [(1, ["S"])], "B": [(2, ["S"])]},
        {"S"},
        {0: 2, 1: 4},
    ),
    "deep chain": (
        {"T": [(0, ["A"])], "A": [(1, ["B"])], "B": [(2, ["S"])]},
        {"S"},
        {1: 3},
    ),
    "cycle": ({"T": [(0, ["A"])], "A": [(1, ["T"]), (2, ["S"])]}, {"S"}, {1: 6}),
    "open leaf": ({"T": [(0, ["A"]), (1, ["B"])]}, set(), {0: 4}),
    # A *silent* measurement beside a clean one and an unprobed one, which is the arrangement that
    # found three separate places spelling "unmeasured" as zero: the sparse join's fill value, the
    # report's decomposition denominator, and the probe's own column. All three read `SILENT` as a
    # verdict once it is one, so the graph has to carry one.
    "silent beside clean": (
        {"T": [(0, ["A"]), (1, ["B"])], "A": [(2, ["S"])]},
        {"B", "S"},
        {0: SILENT, 1: 1},
    ),
}


def measured_graph(spec, stock, outcomes):
    """Build ``spec`` through `expansions_of`'s shape, writing a measurement per rule."""
    expansions: Expansions[str] = Expansions(NumpyExpansionStore(), DictInterner())
    for product, steps in spec.items():
        (product_sid,) = expansions.intern([product]).tolist()
        expansions.mark_expanded(np.array([product_sid], dtype=np.int64))
        for rule, precursors in steps:
            add_application(
                expansions,
                product_sid,
                rule,
                expansions.intern(list(precursors)).tolist(),
                outcomes=sites(outcomes.get(rule)),
            )
    state = prepare_graph(expansions, stock)
    return expansions, state, int(expansions.lookup([next(iter(spec))])[0])


# ----------------------------------------------------------------- the no-op properties


def test_an_unmeasured_price_charges_what_its_base_charges() -> None:
    """The property the whole mechanism rests on: measuring nothing changes nothing.

    A measurement of ``0`` is not a measurement of ``1``. "Nobody probed this" and "this step is
    clean" look alike only if the penalty is asked to price the first as though it were the
    second, and pricing an unprobed step as clean would quietly declare an unreviewed rule set
    perfectly selective.
    """
    price = selective(constant_step(0.105), {}, {"sites": steps_equivalent})
    assert rule_price(price, 3, (11,)) == 0.105
    assert application_price(price, ((3, 4), (11,))) == 0.105


def test_a_selective_price_with_no_measurements_matches_a_constant_one_bit_for_bit() -> (
    None
):
    """Asserted on `app_cost` and on the fold, because those are the two things that consume it."""
    _, state, target = graph()
    plain, aware = (
        constant_step(1.0),
        selective(constant_step(1.0), {}, {"sites": steps_equivalent}),
    )

    assert np.array_equal(
        shape_of(state.expansions, plain).app_cost,
        shape_of(state.expansions, aware).app_cost,
    )
    values = [
        tropical_valuation(
            state,
            leaf_value=np.zeros(state.expansions.sid_count()),
            price=p,
            max_depth=3,
        ).node(target, 3)
        for p in (plain, aware)
    ]
    assert values[0] == values[1]


def test_constant_step_and_by_rule_agree_with_the_closure_fold_exactly() -> None:
    """The array fold and the generic fold are each other's control arm, so they must agree here."""
    _, state, target = graph()
    price = by_rule(lambda rule: 1.0 + rule / 7.0)
    array = tropical_valuation(
        state,
        leaf_value=np.zeros(state.expansions.sid_count()),
        price=price,
        max_depth=3,
    )
    scalar = valuation(
        state,
        TROPICAL,
        leaf_value=lambda sid: 0.0,
        step_cost=partial(application_price, price),
    )
    assert array.node(target, 3) == pytest.approx(scalar.node(target, 3))


# ----------------------------------------------------------------- pricing an application


def test_a_disconnection_is_priced_at_its_selective_template_not_its_unselective_one() -> (
    None
):
    """Why the measurement is keyed per template and not per disconnection.

    Two templates reach ``S`` from ``T``. One is measured clean, the other as a three-way mixture.
    A chemist reaching that disconnection would run the clean one, so min-plus must too -- and a
    price keyed only by the disconnection could not tell them apart at all.
    """
    _, state, target = graph(outcomes={2: 3, 5: 1})
    price = selective(constant_step(1.0), state.outcomes, {"sites": steps_equivalent})
    (step,) = [s for s in state.applications(target) if len(s[0]) == 2]

    assert rule_price(price, 2, step[1]) == 4.0  # the mixture
    assert rule_price(price, 5, step[1]) == 1.0  # the clean template
    assert application_price(price, step) == 1.0
    assert attaining_rule(price, step) == 5


def test_a_step_is_labelled_with_the_template_its_price_was_selected_for() -> None:
    """Otherwise a route's stated chemistry costs more than the value it was chosen for."""
    _, state, target = graph(outcomes={2: 5, 5: 1})
    price = selective(constant_step(1.0), state.outcomes, {"sites": steps_equivalent})
    (step,) = [s for s in state.applications(target) if len(s[0]) == 2]

    assert lowest_rule(step) == 2  # what an index gives
    assert attaining_rule(price, step) == 5  # what the price gives
    assert attaining_rule(constant_step(1.0), step) == lowest_rule(
        step
    )  # they agree when equal


def test_a_disconnection_no_template_achieves_is_unaffordable() -> None:
    """Infinity rather than an error: min-plus already means "no route this way"."""
    assert application_price(constant_step(1.0), ((), (4,))) == math.inf


# ----------------------------------------------------------------- the multiplier shapes


@pytest.mark.parametrize(
    "outcomes,worth",
    [(0, 1.0), (1, 1.0), (2, 3.0), (3, 4.0), (5, 6.0)],
)
def test_steps_equivalent_clears_a_three_step_detour_from_three_products(
    outcomes, worth
) -> None:
    """The threshold is 3, and it is 3 under both aggregations.

    Protecting, reacting and deprotecting is a *linear* sequence, so it costs three steps whether
    an application's precursors are summed or maxed. A multiplier has to exceed 3 for the detour
    to win, which is why the shape is ``n + 1`` and not ``n``: the bare ``n`` ties at exactly the
    three-product case this exists for.
    """
    assert steps_equivalent(outcomes) == worth
    assert (worth > 3.0) == (outcomes >= 3)


def test_flat_reads_a_two_way_and_a_six_way_ambiguity_alike() -> None:
    """Coarser on purpose, and the shape that states one threshold rather than a slope."""
    worth = flat(6.5)
    assert worth(0) == worth(1) == 1.0
    assert worth(2) == worth(6) == 6.5
    with pytest.raises(ValueError, match="at least 1"):
        flat(0.5)


def test_graded_sits_between_a_threshold_and_a_step_count() -> None:
    """The shape for a reading that is a warning rather than a product count.

    Seven competing reactions is worth more than two and nowhere near eight steps of synthesis,
    which is the gap `flat` and `steps_equivalent` leave between them.
    """
    worth = graded(0.1)

    assert worth(UNPROBED) == worth(SILENT) == worth(1) == 1.0
    assert worth(2) == pytest.approx(1.1)
    assert worth(7) == pytest.approx(1.6)
    # Against the two shapes it sits between, on the reading that motivated it.
    assert flat(1.2)(7) == flat(1.2)(2)
    assert steps_equivalent(7) == 8.0


def test_graded_saturates_at_its_cap_and_never_drops() -> None:
    """A cap is a plateau, not a discount: the law below is what it has to keep."""
    worth = graded(0.5, cap=2.0)

    assert worth(2) == pytest.approx(1.5)
    assert worth(3) == worth(100) == 2.0
    with pytest.raises(ValueError, match="non-negative"):
        graded(-0.1)
    with pytest.raises(ValueError, match="at least 1"):
        graded(0.1, cap=0.5)


@pytest.mark.parametrize(
    "shape",
    [
        steps_equivalent,
        flat(1.2),
        flat(6.5),
        graded(0.0),
        graded(0.1),
        graded(0.5, cap=2.0),
        doubted(graded(0.1), silent=2.0),
        uncurated(flat(1.2), unprobed=1.2),
        uncurated(doubted(graded(0.2, cap=3.0), silent=2.0), unprobed=1.2),
    ],
)
def test_every_multiplier_is_non_decreasing_and_free_at_one(shape) -> None:
    """The law of the family: more may cost the same, and may never cost less.

    A price that charged a seven-way ambiguity less than a two-way one would make a report prefer
    the messier step, and a composed shape could introduce that without either half doing so --
    which is why this is asserted over the compositions and not only over the pieces.

    From ``n = 1`` upwards only. The sentinels below it are out of band: ``0`` says the probe ran
    and the reaction made nothing and ``-1`` says nobody looked, so neither is "fewer products"
    and :func:`doubted` and :func:`uncurated` deliberately charge them above a clean step.
    """
    assert shape(1) == 1.0, "one product is the multiplicative identity, in every shape"
    worths = [shape(n) for n in range(1, 25)]
    assert worths == sorted(worths)
    assert all(worth >= 1.0 for worth in worths)


def test_a_surcharge_prices_the_extra_steps_at_a_scale_the_caller_fixes() -> None:
    """The escape hatch from scale-freeness, which is what makes a greedy rung ignore a penalty.

    At a step of 0.001 a fourfold multiple is 0.004 and reorders nothing. The same *three extra
    steps*, priced at the relaxed rung, cost 3.0 and reorder from the first pass.
    """
    three_way = {"sites": {((), 0): 3}}
    scaled = selective(constant_step(0.001), three_way, {"sites": steps_equivalent})
    anchored = surcharged(
        constant_step(0.001), three_way, {"sites": steps_equivalent}, scale=1.0
    )
    assert rule_price(scaled, 0, ()) == pytest.approx(0.004)
    assert rule_price(anchored, 0, ()) == pytest.approx(0.001 + 3.0)


# ----------------------------------------------------------------- the factor/cost change of variable


@pytest.mark.parametrize("factor", [1.0, 0.9, 0.5, 0.105])
def test_a_factor_and_a_cost_are_one_quantity(factor) -> None:
    """The report states a step's worth as a factor and the search as a cost; these are inverses."""
    assert factor_of_cost(cost_of_factor(factor)) == pytest.approx(factor)


def test_the_report_factor_and_the_search_step_are_the_same_number() -> None:
    """0.9 per step and 0.105 per reaction were calibrated separately and coincide. Say so."""
    assert cost_of_factor(0.9) == pytest.approx(0.105, abs=5e-4)


@pytest.mark.parametrize("bad", [0.0, -1.0, 1.5, math.nan, math.inf])
def test_a_factor_outside_its_range_is_refused(bad) -> None:
    """A factor above 1 is a reaction that improves a route by existing, which is not a cost."""
    with pytest.raises(ValueError, match="in \\(0, 1\\]"):
        cost_of_factor(bad)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_step_cost_must_be_positive(bad) -> None:
    with pytest.raises(ValueError, match="positive"):
        constant_step(bad)


def test_a_price_says_whether_it_is_uniform_before_anything_is_measured() -> None:
    """Configuration, not data. A selector needing uniformity must refuse at pass one.

    Both prices below charge 1.0 for every application in this graph, because nothing has been
    measured yet. Only the flag distinguishes them, and only the flag can, which is the whole
    reason it is a field rather than a scan of the costs.
    """
    aware = selective(constant_step(1.0), {}, {"sites": steps_equivalent})
    assert constant_step(1.0).uniform and not aware.uniform
    assert StepPrice(base=aware.base, charge=aware.charge).uniform


def test_a_per_template_base_must_say_it_is_not_uniform() -> None:
    """The other way a price can be non-uniform, and it was claiming otherwise.

    `selective` makes a price non-uniform through the *substrate*. A base that differs per template
    -- a curated yield, a learned feasibility -- makes it non-uniform through the *rule*, and
    `by_rule` defaulted that flag to true however much its costs varied.

    Default still true, because both callers that existed are genuinely uniform: `constant_step`
    charges one number for every rule, and `anytime_step`'s base reads the current rung, which every
    rule shares. Asserted here so neither silently becomes a lie.
    """
    varying = {0: cost_of_factor(0.95), 1: cost_of_factor(0.60)}
    honest = by_rule(lambda rule: varying[rule], uniform=False)

    assert not honest.uniform
    # And the costs really do differ, which is what the flag is about.
    assert honest.charge(0, ()) != honest.charge(1, ())
    # The two uniform callers keep the default.
    assert constant_step(1.0).uniform
    assert by_rule(lambda rule: 1.0).uniform


# ----------------------------------------------------------------- the two folds, non-uniformly


@pytest.mark.parametrize("conjoin", [operator.add, max], ids=["sum", "max"])
@pytest.mark.parametrize("max_depth", [1, 3, 8])
@pytest.mark.parametrize("name", sorted(MEASURED_GRAPHS))
def test_the_two_folds_agree_under_non_uniform_application_costs(
    name, max_depth, conjoin
) -> None:
    """The control arm `tropical`'s own docstring demands, now that costs can differ per application.

    Every existing agreement test prices every application alike, so none of them can see a bug in
    how a *per-application* cost is gathered. ``max`` is the case that matters: the outside sweep
    recovers a sibling term by subtracting an application's total under ``add`` and has to build it
    under ``max`` (:func:`vertsy.tropical.leave_one_out_max`), so that is the path where a new
    multiply could diverge silently.
    """
    graph_spec, stock, outcomes = MEASURED_GRAPHS[name]
    expansions, state, target = measured_graph(graph_spec, stock, outcomes)
    # Silence is priced here so the sparse join's fill value is exercised: an unprobed row and
    # a silent one must charge differently, and both folds have to agree on which is which.
    price = selective(
        constant_step(1.0),
        state.outcomes,
        {"sites": doubted(steps_equivalent, silent=2.0)},
    )
    leaves = np.full(expansions.sid_count(), 0.375)

    array = tropical_valuation(
        state, leaf_value=leaves, price=price, max_depth=max_depth, conjoin=conjoin
    )
    scalar = valuation(
        state,
        TROPICAL,
        leaf_value=lambda sid: float(leaves[sid]),
        step_cost=partial(application_price, price),
        conjoin=conjoin,
    )

    for sid in range(expansions.sid_count()):
        for depth in range(max_depth + 1):
            assert array.node(sid, depth) == pytest.approx(scalar.node(sid, depth))
    for product in state.steps:
        for step in state.applications(product):
            assert array.application(step, max_depth) == pytest.approx(
                scalar.application(step, max_depth)
            )


@pytest.mark.parametrize("name", sorted(MEASURED_GRAPHS))
def test_the_two_outside_sweeps_agree_under_non_uniform_application_costs(name) -> None:
    """The same control arm for the context half, which is what a selection actually ranks on."""
    from vertsy.tropical import outside_valuation_of

    graph_spec, stock, outcomes = MEASURED_GRAPHS[name]
    expansions, state, target = measured_graph(graph_spec, stock, outcomes)
    # Silence is priced here so the sparse join's fill value is exercised: an unprobed row and
    # a silent one must charge differently, and both folds have to agree on which is which.
    price = selective(
        constant_step(1.0),
        state.outcomes,
        {"sites": doubted(steps_equivalent, silent=2.0)},
    )
    leaves = np.full(expansions.sid_count(), 0.375)

    scalar_value = valuation(
        state,
        TROPICAL,
        leaf_value=lambda sid: float(leaves[sid]),
        step_cost=partial(application_price, price),
    )
    scalar = outside_valuation(
        target,
        state,
        scalar_value,
        max_depth=4,
        step_cost=partial(application_price, price),
    )
    array_value = tropical_valuation(state, leaf_value=leaves, price=price, max_depth=4)
    array = outside_valuation_of(target, state, array_value, max_depth=4)

    assert set(scalar.leaves) == set(array.leaves)
    for sid, (total, context, _depth) in scalar.leaves.items():
        assert array.leaves[sid][0] == pytest.approx(total)
        assert array.leaves[sid][1] == pytest.approx(context)


def test_a_measured_step_costs_more_and_an_unmeasured_sibling_does_not() -> None:
    """The point of the whole thing, at its smallest: one application gets dearer, one does not."""
    _, state, target = graph(outcomes={2: 3, 5: 3})
    plain = constant_step(1.0)
    aware = selective(plain, state.outcomes, {"sites": steps_equivalent})
    before = shape_of(state.expansions, plain).app_cost
    after = shape_of(state.expansions, aware).app_cost

    assert sorted(before.tolist()) == [1.0, 1.0]
    assert sorted(after.tolist()) == [
        1.0,
        4.0,
    ]  # rule 7's disconnection was never probed


# ----------------------------------------------------------------- two channels, one price


def test_two_channels_add_their_extra_steps_rather_than_multiplying_their_worths() -> (
    None
):
    """A step that is both a three-way mixture and has one competing reaction is worth six steps.

    Multiplying the worths gives twelve, which is not a claim either measurement supports. Adding
    the *extras* -- three for the mixture, two for the competitor -- is the arithmetic a chemist
    would do on a synthesis plan, and it is what `extra_steps` does.
    """
    price = selective(
        constant_step(1.0),
        {"sites": {((4,), 0): 3}, "competitors": {((4,), 0): 2}},
        {"sites": steps_equivalent, "competitors": steps_equivalent},
    )

    assert price.channels == ("sites", "competitors")
    assert rule_price(price, 0, (4,)) == pytest.approx(6.0)
    assert extra_steps([4.0, 3.0]) == pytest.approx(6.0)
    assert extra_steps([]) == 1.0  # no channels, no charge


def test_a_channel_with_no_table_costs_nothing_so_one_can_be_wired_before_the_other() -> (
    None
):
    """Which is how the two get measured apart: they are different chemistry, not one number."""
    sites_only = selective(
        constant_step(1.0),
        {"sites": {((4,), 0): 3}},
        {"sites": steps_equivalent, "competitors": steps_equivalent},
    )
    assert rule_price(sites_only, 0, (4,)) == pytest.approx(4.0)

    competitors_only = selective(
        constant_step(1.0),
        {"competitors": {((4,), 0): 2}},
        {"sites": steps_equivalent, "competitors": steps_equivalent},
    )
    assert rule_price(competitors_only, 0, (4,)) == pytest.approx(3.0)


def test_each_channel_can_carry_its_own_shape() -> None:
    """Separate penalties because the chemistry differs: a separation is not a failure.

    A mixture of the intended reaction costs a separation, and `steps_equivalent` grades it. A
    competing reaction may mean the step does not work, so a flat threshold is a defensible shape
    for it -- and the point is that the choice is per channel rather than shared.
    """
    price = selective(
        constant_step(1.0),
        {"sites": {((4,), 0): 3}, "competitors": {((4,), 0): 2}},
        {"sites": steps_equivalent, "competitors": flat(6.5)},
    )
    # three extra for the mixture, five and a half for the competitor
    assert rule_price(price, 0, (4,)) == pytest.approx(1.0 + 3.0 + 5.5)


def test_the_array_fold_groups_on_the_whole_count_vector() -> None:
    """Two channels stay one `np.unique` over `(rule, *counts)`, so the fold stays in arrays."""
    expansions: Expansions[str] = Expansions(NumpyExpansionStore(), DictInterner())
    target, stock = expansions.intern(["T", "S"]).tolist()
    expansions.mark_expanded(np.array([target], dtype=np.int64))
    add_application(
        expansions, target, 2, [stock], outcomes={"sites": 3, "competitors": 2}
    )
    state = prepare_graph(expansions, {"S"})
    price = selective(
        constant_step(1.0),
        state.outcomes,
        {"sites": steps_equivalent, "competitors": steps_equivalent},
    )

    assert set(state.outcomes) == {"sites", "competitors"}
    assert shape_of(expansions, price).app_cost.tolist() == [6.0]


# ---------------------------------------------------- silence is not the same as no measurement


def test_an_unprobed_measurement_and_a_silent_one_are_different_values() -> None:
    """The distinction the whole doubting mechanism rests on, at the one place it is encoded.

    A price that cannot tell "nobody looked" from "we looked and the chemistry made nothing" charges
    a broken pairing exactly what it charges a clean step -- which is how a Stille whose forward rule
    wanted bromide beat a Suzuki honestly measured as a two-product mixture.
    """
    assert UNPROBED != SILENT
    assert UNPROBED < 0 <= SILENT


def test_a_missing_measurement_reads_unprobed_rather_than_zero() -> None:
    outcomes = measured({"sites": {((1, 2), 7): 3}}, ("sites",))
    assert outcomes(7, (1, 2)) == (3,)
    assert outcomes(7, (9,)) == (UNPROBED,)
    assert outcomes(8, (1, 2)) == (UNPROBED,)


def test_a_channel_with_no_table_at_all_reads_unprobed() -> None:
    """What lets one channel be wired before the other, and costs nothing while it is not."""
    outcomes = measured({}, ("sites", "competitors"))
    assert outcomes(0, ()) == (UNPROBED, UNPROBED)


@pytest.mark.parametrize("outcomes", [UNPROBED, 1])
def test_doubting_leaves_every_rung_but_silence_alone(outcomes: int) -> None:
    """A wrapper has to be transparent everywhere it does not claim to act."""
    assert doubted(steps_equivalent, silent=2.5)(outcomes) == steps_equivalent(outcomes)


def test_doubting_charges_its_factor_exactly_at_silence() -> None:
    assert doubted(steps_equivalent, silent=2.5)(SILENT) == 2.5
    # And unwrapped, silence is free -- so the sentinel alone changes no number.
    assert steps_equivalent(SILENT) == 1.0


def test_a_silence_factor_of_one_is_the_identity() -> None:
    """The spelling that restores the older behaviour, so the change is measurable against it."""
    for outcomes in (UNPROBED, SILENT, 1, 2, 5):
        assert doubted(steps_equivalent, silent=1.0)(outcomes) == steps_equivalent(
            outcomes
        )


def test_a_doubted_multiplier_still_composes_into_a_selective_price() -> None:
    """The point of being a `Multiplier` and not a special case inside `selective`."""
    price = selective(
        constant_step(1.0),
        {"sites": {((0,), 3): SILENT}},
        {"sites": doubted(steps_equivalent, silent=2.0)},
    )
    assert rule_price(price, 3, (0,)) == pytest.approx(2.0)
    # An unprobed application on the same channel pays the base and nothing more.
    assert rule_price(price, 4, (0,)) == pytest.approx(1.0)


# ----------------------------------------------------------------- yields, as fractions of material


@pytest.mark.parametrize("share", [1.0, 0.8, 0.5, 0.1])
def test_a_yield_curve_is_free_when_clean_and_never_rises(share: float) -> None:
    """The `Factor` counterpart of the multiplier law, in the other direction.

    A multiplier is `1.0` at a clean step and never falls; a factor is `1.0` and never rises. Both
    sentinels sit below the clean count so they survive intact, which is what lets the sentinels be
    priced by a wrapper rather than by the curve.
    """
    curve = shared(share)

    assert curve(UNPROBED) == 1.0 and curve(SILENT) == 1.0 and curve(1) == 1.0
    seen = [curve(n) for n in range(1, 9)]
    assert all(0.0 < value <= 1.0 for value in seen)
    assert seen == sorted(seen, reverse=True)


def test_a_yield_floor_stops_the_geometry_running_away() -> None:
    """Without it eight competitors at 0.8 read as 21%, which a count cannot support."""
    assert shared(0.8)(9) == pytest.approx(0.8**8)
    assert shared(0.8, floor=0.5)(9) == 0.5
    # And the floor cannot lift a clean step above itself.
    assert shared(0.8, floor=0.5)(1) == 1.0


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, math.inf, math.nan])
def test_a_share_outside_the_unit_interval_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="share"):
        shared(bad)


def test_a_compounded_step_is_free_when_every_channel_is_clean() -> None:
    """`clean_log_cost` passes `1` to every channel, so this has to equal the base exactly.

    Not approximately: `vertsy.route_selection.scored_route` subtracts the clean cost from the real
    one and hands the remainder to `factor_of_cost`, which refuses a negative. A clean step costing
    a hair more than its base is the mid-benchmark failure that function's docstring records.
    """
    for yielded in (0.9, 0.55, 0.999):
        base = by_rule(lambda rule, y=yielded: cost_of_factor(y), uniform=False)
        price = compounded(base, {}, {"sites": shared(0.8), "competitors": shared(0.5)})

        assert price.charge(0, (1, 1)) == cost_of_factor(yielded)
        assert not price.uniform


def test_what_a_compounded_step_charges_is_exactly_the_product_of_its_yields() -> None:
    """The claim the whole shape exists for, and the one `selective` cannot make.

    The report prints `selectivity` as `factor_of_cost(reaction_cost - clean_cost)`. Under this
    shape that remainder is `-log(prod of the channel yields)`, so the printed factor *is* their
    product -- and, the part that matters, it is their product **whatever the rule's own base is**.
    Under `selective` the remainder is `base(rule) * (m - 1)`, so the same measurement reads
    differently on a good template than on a bad one.
    """
    channels = {"sites": shared(0.8), "competitors": shared(0.5)}
    wanted = 0.8**2 * 0.5

    for yielded in (0.9, 0.55, 0.2):
        base = by_rule(lambda rule, y=yielded: cost_of_factor(y), uniform=False)
        price = compounded(base, {}, channels)

        charged = price.charge(0, (3, 2))
        clean = price.charge(0, (1, 1))
        assert factor_of_cost(charged - clean) == pytest.approx(wanted)


def test_a_compounded_channel_with_no_table_costs_nothing() -> None:
    """So one channel can be wired and measured before the other exists."""
    price = compounded(
        constant_step(0.105), {}, {"sites": shared(0.5), "competitors": shared(0.5)}
    )

    assert price.channels == ("sites", "competitors")
    assert price.outcomes(0, (7,)) == (UNPROBED, UNPROBED)
    assert rule_price(price, 0, (7,)) == pytest.approx(0.105)

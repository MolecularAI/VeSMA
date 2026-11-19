"""The per-step alternatives report: what else the engine could have done, and what it knew."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from vertsy.costs import UNPROBED
from vertsy.expansions import Expansions
from vertsy.filters import refusal_wins
from vertsy.interner import DictInterner
from vertsy.metrics import prepare_graph
from vertsy.mixture import ExpectedPeak
from vertsy.route_alternatives import (
    ROUTE_STANDINGS,
    STANDINGS,
    route_report,
    stored_veto_validator,
)
from vertsy.route_selection import DEFAULT_PRICE, select_solved_routes
from vertsy.selectivity import unmeasured
from vertsy.store.numpy_store import NumpyExpansionStore


def graph(outcomes=None, reduce=None):
    """`T` by two disconnections plus a dead one, so every standing has something to land on.

    `T -> A + B` (rule 0) and `T -> C + D` (rule 1) both complete; `T -> E` (rule 5) cannot,
    because `E` is neither buyable nor expanded.
    """
    expansions = Expansions(NumpyExpansionStore(), DictInterner())
    target, a, b, c, d, e = expansions.intern(["T", "A", "B", "C", "D", "E"]).tolist()
    expansions.add_layer(
        np.array([target]),
        [(0, 0, 0, 0), (0, 0, 0, 1), (0, 1, 1, 2), (0, 1, 1, 3), (0, 5, 2, 4)],
        np.array([a, b, c, d, e]),
        outcomes,
        reduce,
    )
    return target, prepare_graph(expansions, ["A", "B", "C", "D"])


def report_of(target, state, **kwargs):
    selection = select_solved_routes(target, state, limit=5, **kwargs)
    return route_report(
        target,
        state,
        selection,
        price=DEFAULT_PRICE,
        rule_label=lambda rule: f"rule {rule}",
        diversity_cutoff=kwargs.get("diversity_cutoff"),
    )


def test_retro_step_one_is_the_first_disconnection() -> None:
    """Which is the *last* synthesis step -- the order a chemist reads a plan backwards in."""
    target, state = graph()

    report = report_of(target, state)

    first = report.routes[0].retro_steps[0]
    assert first.index == 1
    assert first.product == "T"


def test_an_unreachable_disconnection_is_unsolved_and_not_rejected() -> None:
    """Most disconnections of most molecules lead nowhere; they were never alternatives.

    Saying so is what stops a reader mistaking a dead branch for chemistry the engine turned down.
    """
    target, state = graph()

    report = report_of(target, state)
    standings = {
        option.rule: option.standing
        for option in report.routes[0].retro_steps[0].alternatives
    }

    assert standings[5] == "unsolved"


def test_a_veto_that_fired_is_named_from_the_per_row_columns() -> None:
    """The aggregate verdict says only *that* one fired; the per-row columns say which."""
    target, state = graph(
        outcomes={
            "vetoed": np.array([0, 0, 2, 2, 0]),
            "vetoed:B 1": np.array([0, 0, 1, 1, 0]),
            "vetoed:B 7": np.array([0, 0, 1, 1, 0]),
        },
        reduce={
            "vetoed": refusal_wins,
            "vetoed:B 1": refusal_wins,
            "vetoed:B 7": refusal_wins,
        },
    )

    report = report_of(target, state)
    options = {
        option.rule: option for option in report.routes[0].retro_steps[0].alternatives
    }

    assert options[1].standing == "vetoed"
    assert options[1].vetoes_fired == ("B 1", "B 7")


def test_without_the_per_row_columns_a_veto_is_still_reported_unnamed() -> None:
    """Saying a veto fired without naming it beats saying nothing at all."""
    target, state = graph(
        outcomes={"vetoed": np.array([0, 0, 2, 2, 0])},
        reduce={"vetoed": refusal_wins},
    )

    report = report_of(target, state)
    options = {
        option.rule: option for option in report.routes[0].retro_steps[0].alternatives
    }

    assert options[1].standing == "vetoed"
    assert options[1].vetoes_fired == ("(unnamed)",)


def test_an_unmeasured_channel_reports_none_and_never_zero() -> None:
    """The distinction the whole measurement layer exists to keep.

    An absent row means nobody looked. A stored zero means the probe ran and the reaction made
    nothing. Collapsing them would make an unexamined rule read as a clean one.
    """
    target, state = graph(
        outcomes={"sites": np.array([1, 1, 0, 0, 1])},
        reduce={"sites": unmeasured},
    )

    report = report_of(target, state)
    step = report.routes[0].retro_steps[0]
    options = {option.rule: option for option in step.alternatives}

    # `sites` was measured and came back SILENT on rule 1 -- a real answer, stored as 0.
    assert options[1].measurements["sites"] == 0
    # `competitors` was never measured at all, on any step.
    assert step.chosen.measurements["competitors"] is None
    assert all(
        option.measurements["competitors"] is None for option in step.alternatives
    )


def test_a_standing_is_comparative_not_absolute() -> None:
    """An alternative is labelled by how it differs from the step taken, not in isolation.

    Calling an alternative `competing-sites` when the chosen step makes just as many products
    would be misleading, so the comparison is explicit and the raw counts travel alongside.
    """
    target, state = graph(
        outcomes={"sites": np.array([1, 1, 3, 3, 1])},
        reduce={"sites": unmeasured},
    )

    report = report_of(target, state)
    by_rank = {route.rank: route for route in report.routes}

    # Route 1 takes rule 0 (1 site); rule 1 makes three, so it is worse here.
    worse = {
        option.rule: option.standing
        for option in by_rank[1].retro_steps[0].alternatives
    }
    assert worse[1] == "competing-sites"
    # Route 2 takes rule 1 (3 sites); rule 0 makes fewer, so it is *not* labelled for competition.
    better = {
        option.rule: option.standing
        for option in by_rank[2].retro_steps[0].alternatives
    }
    assert better[0] != "competing-sites"


def test_an_alternative_live_in_another_route_says_so() -> None:
    """The honest substitute for "longer": this disconnection is used, just not here."""
    target, state = graph()

    report = report_of(target, state)
    options = {
        option.rule: option for option in report.routes[0].retro_steps[0].alternatives
    }

    assert options[1].also_reported_in == (2,)
    assert options[5].also_reported_in == ()


def test_the_diversity_floor_puts_its_rejects_in_the_report() -> None:
    """So a reader can widen the cutoff knowingly instead of wondering what was dropped."""
    target, state = graph()

    report = report_of(target, state, diversity_cutoff=1)

    # Both routes differ at decision 1, so the floor drops neither.
    assert len(report.routes) == 2
    assert report.diversity_cutoff == 1
    assert all(route.standing in ROUTE_STANDINGS for route in report.routes)
    assert report.routes[0].standing == "best"
    assert report.routes[1].standing == "diverse"


def test_every_standing_is_in_the_closed_vocabulary() -> None:
    """A closed set so a report can be summarised without parsing prose."""
    target, state = graph(
        outcomes={
            "sites": np.array([1, 1, 3, 3, 1]),
            "vetoed": np.array([0, 0, 1, 1, 0]),
        },
        reduce={"sites": unmeasured, "vetoed": refusal_wins},
    )

    report = report_of(target, state)
    seen = {
        option.standing
        for route in report.routes
        for step in route.retro_steps
        for option in (step.chosen, *step.alternatives)
    }

    assert seen <= STANDINGS
    assert "chosen" in seen


def test_the_report_round_trips_through_json() -> None:
    """It is an artefact, so it has to survive being written and read back."""
    target, state = graph(
        outcomes={"sites": np.array([1, 1, 3, 3, 1])}, reduce={"sites": unmeasured}
    )

    report = report_of(target, state)
    text = json.dumps(asdict(report))

    assert json.loads(text) == json.loads(json.dumps(asdict(report)))
    assert json.loads(text)["routes"][0]["retro_steps"][0]["chosen"]["rule_label"]


def test_the_markdown_numbers_backwards_and_draws_forwards() -> None:
    """The two orders are deliberately different, and the output has to make that unambiguous."""
    from vertsy.route_alternatives import render_report_markdown

    target, state = graph()
    text = render_report_markdown(report_of(target, state))

    assert "### Retro-step 1 — T" in text
    # The reaction itself reads in the synthesis direction.
    assert "-> T`" in text
    assert "T ->" not in text


def test_an_unmeasured_channel_renders_as_a_dash_not_a_zero() -> None:
    """A reader must be able to see "nobody looked" in the table, not infer it from a zero."""
    from vertsy.route_alternatives import render_report_markdown

    target, state = graph(
        outcomes={"sites": np.array([1, 1, 0, 0, 1])}, reduce={"sites": unmeasured}
    )
    text = render_report_markdown(report_of(target, state))

    assert "| competitors |" in text
    # `competitors` was never measured, so every one of its cells is a dash.
    prefixes = tuple(f"| {standing}" for standing in sorted(STANDINGS))
    rows = [line for line in text.splitlines() if line.startswith(prefixes)]
    assert rows
    assert all("--" in row for row in rows)


def test_a_named_veto_reaches_the_table() -> None:
    """The row code is the actionable part -- it is what a reader goes and reads the rationale of."""
    from vertsy.route_alternatives import render_report_markdown

    target, state = graph(
        outcomes={
            "vetoed": np.array([0, 0, 1, 1, 0]),
            "vetoed:B 7": np.array([0, 0, 1, 1, 0]),
        },
        reduce={"vetoed": refusal_wins, "vetoed:B 7": refusal_wins},
    )
    text = render_report_markdown(report_of(target, state))

    assert "B 7" in text


def test_a_censored_route_is_reported_with_the_reason_it_was_refused() -> None:
    """The two ways a report comes back short read differently, and both belong in it.

    A route dropped by the diversity floor is a knob a reader can widen; a route the validator
    refused is chemistry. Reporting them in one list with distinct standings is what stops the
    second being mistaken for the first.
    """
    target, state = graph()
    first = select_solved_routes(target, state, limit=5).routes[0].route
    selection = select_solved_routes(
        target, state, limit=1, validate=lambda route: "B 7" if route == first else None
    )

    report = route_report(
        target,
        state,
        selection,
        price=DEFAULT_PRICE,
        rule_label=lambda rule: f"rule {rule}",
    )

    (refused,) = report.rejected
    assert refused.standing == "censored"
    assert refused.reason == "B 7"
    # A route nobody can make has no steps worth drawing.
    assert refused.retro_steps == ()
    assert len(report.routes) == 1
    assert report.routes[0].standing == "best"


def test_the_chosen_step_carries_its_own_measurements_into_the_table() -> None:
    """The diagnostic the whole report exists for, in one row.

    Measured on a six-target run: the route reported for `C=C(C)C(=O)OC(C)CC1CO1` takes
    `CC=O + ClCC1CO1 -> CC(O)CC1CO1` under Hartenfeller 45a -- a Grignard onto an epoxide, which
    veto row B 30 condemns -- while the `sites` probe reports exactly one product. The veto and the
    forward pattern disagree about one reaction, and the forward pattern is the one that is wrong.
    A reader can only see that if the step *taken* shows its own counts, so it is the first row of
    the table rather than only the heading above it.
    """
    from vertsy.route_alternatives import render_report_markdown

    target, state = graph(
        outcomes={
            "sites": np.array([1, 1, 1, 1, 1]),
            "vetoed": np.array([1, 1, 0, 0, 0]),
            "vetoed:B 30": np.array([1, 1, 0, 0, 0]),
        },
        reduce={
            "sites": unmeasured,
            "vetoed": refusal_wins,
            "vetoed:B 30": refusal_wins,
        },
    )
    text = render_report_markdown(report_of(target, state))
    chosen_row = next(
        line for line in text.splitlines() if line.startswith("| chosen ")
    )

    assert "B 30" in chosen_row, "the row that condemns the step the report recommends"
    assert "| 1 |" in chosen_row, "beside the one clean product the probe measured"


def test_a_report_read_back_from_json_renders_the_same_page() -> None:
    """The decoupling the format is for: nothing but the file is needed to render a report."""
    from vertsy.route_alternatives import render_report_markdown, report_from_json

    target, state = graph(
        outcomes={"sites": np.array([1, 1, 3, 3, 1])}, reduce={"sites": unmeasured}
    )
    report = report_of(target, state)

    reloaded = report_from_json(json.loads(json.dumps(asdict(report))))

    assert reloaded == report
    assert render_report_markdown(reloaded) == render_report_markdown(report)


def test_a_disputed_veto_row_is_named_with_a_question_mark() -> None:
    """The state that used to render as `(unnamed)`, built the way a real run builds it.

    A measurement is keyed by precursors and rule, never by product, because a forward reaction is
    determined by those two -- but a veto read on the *product* is not. Two products disconnecting
    to the same materials by the same template, one condemned and one clean, therefore disagree,
    and `sid_selectivity` drops the pair to `UNPROBED` rather than trusting whichever was written
    last. The aggregate survives the same collapse whenever it agrees for another reason, which
    leaves "a veto fired and nothing can say which". The disputed rows are the only candidates.
    """
    expansions = Expansions(NumpyExpansionStore(), DictInterner())
    target, a, b, other = expansions.intern(["T", "A", "B", "OTHER"]).tolist()
    # The reported step: `T -> A + B`, where the row fired on this product.
    expansions.add_layer(
        np.array([target]),
        [(0, 0, 0, 0), (0, 0, 0, 1)],
        np.array([a, b]),
        {"vetoed": np.array([1, 1]), "vetoed:B 13": np.array([1, 1])},
        {"vetoed": refusal_wins, "vetoed:B 13": refusal_wins},
    )
    # The same materials and the same template reaching a *different* product, which the row does
    # not condemn -- while some other row does, so the aggregate still reads one.
    expansions.add_layer(
        np.array([other]),
        [(0, 0, 0, 0), (0, 0, 0, 1)],
        np.array([a, b]),
        {"vetoed": np.array([1, 1]), "vetoed:B 13": np.array([0, 0])},
        {"vetoed": refusal_wins, "vetoed:B 13": refusal_wins},
    )
    state = prepare_graph(expansions, ["A", "B"])

    report = route_report(
        target,
        state,
        select_solved_routes(target, state, limit=5),
        price=DEFAULT_PRICE,
        rule_label=lambda rule: f"rule {rule}",
    )
    step = report.routes[0].retro_steps[0]

    # The pair was dropped, while the aggregate agreed for another reason.
    assert state.outcomes["vetoed:B 13"][((a, b), 0)] == UNPROBED
    assert state.outcomes["vetoed"][((a, b), 0)] == 1
    assert step.chosen.vetoes_fired == ("B 13?",)


def test_the_step_keyed_index_names_the_row_exactly() -> None:
    """The same graph as the disputed case, read through the index that cannot collide.

    A verdict about a product belongs to one `(product, rule, precursors)`, and a product is
    expanded at most once over the store's lifetime, so that key has exactly one row and nothing
    has to be inferred. The chemistry-keyed index could only offer `B 13?` as a candidate here,
    because two products disagreed under one key; this one answers.
    """
    expansions = Expansions(NumpyExpansionStore(), DictInterner())
    target, a, b, other = expansions.intern(["T", "A", "B", "OTHER"]).tolist()
    for product, verdict in ((target, 1), (other, 0)):
        expansions.add_layer(
            np.array([product]),
            [(0, 0, 0, 0), (0, 0, 0, 1)],
            np.array([a, b]),
            {
                "vetoed": np.array([1, 1]),
                "vetoed:B 13": np.array([verdict, verdict]),
            },
            {"vetoed": refusal_wins, "vetoed:B 13": refusal_wins},
        )
    # The per-row channel only, which is the policy the bench uses: the aggregate stays under the
    # chemistry key, where a price can read it.
    state = prepare_graph(expansions, ["A", "B"], step_keyed=["vetoed:B 13"])

    report = route_report(
        target,
        state,
        select_solved_routes(target, state, limit=5),
        price=DEFAULT_PRICE,
        rule_label=lambda rule: f"rule {rule}",
    )

    # Not written under the chemistry key at all: there its answer is not merely redundant but
    # wrong, because the two products disagree and the pair would be dropped as untrustworthy.
    assert "vetoed:B 13" not in state.outcomes
    # The step-keyed index keeps both answers, so the row is named rather than guessed at.
    assert state.step_outcomes["vetoed:B 13"][(target, 0, (a, b))] == 1
    assert state.step_outcomes["vetoed:B 13"][(other, 0, (a, b))] == 0
    assert report.routes[0].retro_steps[0].chosen.vetoes_fired == ("B 13",)
    # And the aggregate stays where a price can still read it.
    assert state.outcomes["vetoed"][((a, b), 0)] == 1


def test_a_run_that_named_no_step_keyed_channel_still_reports() -> None:
    """The index is opt-in, so every older run and every caller that does not ask still works."""
    target, state = graph(
        outcomes={"vetoed": np.array([0, 0, 2, 2, 0])},
        reduce={"vetoed": refusal_wins},
    )

    assert state.step_outcomes == {}
    report = report_of(target, state)
    options = {
        option.rule: option for option in report.routes[0].retro_steps[0].alternatives
    }
    assert options[1].vetoes_fired == ("(unnamed)",)


# --- the mixture of the step actually taken -----------------------------------------------------

PEAKS = (
    ExpectedPeak("product", "C4H9NO", 87.0684, "CCNC(C)=O", "", "resolved"),
    ExpectedPeak("by-product", "HCl", 35.9767, "", "", "resolved"),
    # An ester at the amide's own formula: no instrument separates these two.
    ExpectedPeak(
        "another reaction, same site",
        "C4H9NO",
        87.0684,
        "CCOC(C)=O",
        "FWD X",
        "same formula",
    ),
)


def report_with_mixture(peaks=PEAKS):
    target, state = graph()
    return route_report(
        target,
        state,
        select_solved_routes(target, state, limit=5),
        price=DEFAULT_PRICE,
        rule_label=lambda rule: f"rule {rule}",
        mixture=(lambda product, precursors, rule: peaks) if peaks else None,
    )


def test_the_mixture_of_the_step_taken_is_rendered_under_its_own_heading() -> None:
    """The report explains the decision above and its consequence below.

    The table answers "why this disconnection and not another"; the mixture answers "I am about to
    run it -- what will be in the flask". A reader doing the second thing does not want
    alternatives interleaved, so it is a separate block, and it is on the chosen step only: an
    alternative's mixture is a report about a reaction nobody is running.
    """
    from vertsy.route_alternatives import render_report_markdown

    page = render_report_markdown(report_with_mixture())

    assert "**Expect in the pot:**" in page
    assert "| `CCNC(C)=O` | 87.0684 | C4H9NO | product |" in page
    assert "| _composition only_ | 35.9767 | HCl | by-product |" in page
    # The mass column is where a reader looks, so that is where the warning goes.
    assert "87.0684 (isobaric)" in page
    assert "FWD X" in page
    assert "a single product peak is not evidence of a clean step" in page


def test_the_mixture_is_only_on_the_step_taken() -> None:
    """An alternative's mixture would describe a reaction nobody is running."""
    report = report_with_mixture()

    for route in report.routes:
        for step in route.retro_steps:
            assert step.mixture == PEAKS
            assert not hasattr(step.chosen, "mixture")


def test_a_report_with_no_mixture_renders_no_block() -> None:
    """No forward table wired reads as "not derived", not as "nothing expected"."""
    from vertsy.route_alternatives import render_report_markdown

    page = render_report_markdown(report_with_mixture(peaks=()))

    assert "Expect in the pot" not in page


def test_a_mixture_round_trips_through_json() -> None:
    """Extraction writes one object per target and anything may render it later."""
    from vertsy.route_alternatives import report_from_json

    report = report_with_mixture()

    assert report_from_json(asdict(report)) == report


def test_a_report_written_before_the_mixture_existed_still_loads() -> None:
    """The field is absent from those, and a reader of an archive should not have to care."""
    from vertsy.route_alternatives import report_from_json

    data = asdict(report_with_mixture())
    for route in data["routes"]:
        for step in route["retro_steps"]:
            del step["mixture"]

    assert report_from_json(data).routes[0].retro_steps[0].mixture == ()


def test_the_second_coincidence_tier_is_counted_and_not_just_annotated() -> None:
    """Both tiers get a count, because they are answers to different questions.

    Same formula is not separable by mass at any resolution; same nominal mass with a different
    formula is one peak on unit-resolution kit and two on better. The table annotates each peak
    either way, and a reader should not have to tally the second kind by eye.
    """
    from vertsy.mixture import ExpectedPeak
    from vertsy.route_alternatives import _mixture_lines

    product = ExpectedPeak(
        kind="product",
        formula="C4H9NO",
        exact_mass=87.0684,
        smiles="CCCC(N)=O",
        source="",
        coincidence="resolved",
    )
    isobar = ExpectedPeak(
        kind="another reaction, same site",
        formula="C4H9NO",
        exact_mass=87.0684,
        smiles="CC(C)C(N)=O",
        source="FWD X",
        coincidence="same formula",
    )
    nominal = ExpectedPeak(
        kind="another reaction, same site",
        formula="C5H13N",
        exact_mass=87.1048,
        smiles="CCCCCN",
        source="FWD Y",
        coincidence="same nominal mass",
    )
    out = "\n".join(_mixture_lines((product, isobar, nominal)))
    assert "1 of these share the product's formula" in out
    assert "1 more sit at the product's nominal mass" in out
    # And it says plainly that nothing prices them, so the line is not read as a charge.
    assert "Reported, not charged" in out

    # With no second-tier peak the second line is absent rather than reading "0 more".
    without = "\n".join(_mixture_lines((product, isobar)))
    assert "1 of these share the product's formula" in without
    assert "nominal mass" not in without


def test_the_stored_column_validator_refuses_the_route_the_report_condemns() -> None:
    """Extraction can refuse a vetoed route by lookup, with no molecules and no replay.

    `select_solved_routes`' hook is `SidRoute -> reason | None`, pure sid space, and the only
    other implementation replays the route to re-run the veto screen over real structures. That
    is the only way a `geometry` row can be answered, and it costs a replay per candidate; a row
    evaluated per application during the search is already in the graph, so this reads it.

    The assertion that matters is the *agreement*: the validator and the report must cite the same
    row for the same step, because they go through one `fired_vetoes`. Two lookups could drift
    into a refusal the report cannot explain, or a report row nothing acts on -- which is the
    failure this test exists to catch, not whether a dict lookup works.
    """
    target, state = graph(
        outcomes={
            "vetoed": np.array([0, 0, 2, 2, 0]),
            "vetoed:B 1": np.array([0, 0, 1, 1, 0]),
            "vetoed:B 7": np.array([0, 0, 1, 1, 0]),
        },
        reduce={
            "vetoed": refusal_wins,
            "vetoed:B 1": refusal_wins,
            "vetoed:B 7": refusal_wins,
        },
    )
    validate = stored_veto_validator(state)

    selection = select_solved_routes(target, state, limit=5)
    by_rule = {item.route.rule: item.route for item in selection.routes}
    assert set(by_rule) == {0, 1}, "both disconnections should be candidates"

    # Rule 1 is the condemned one -- the veto column fires on its two applications.
    reason = validate(by_rule[1])
    assert reason == "vetoed: B 1, B 7", reason
    # Rule 0 is untouched, so a clean route is not collateral.
    assert validate(by_rule[0]) is None

    # The agreement: the rows the report prints for that step are the rows the refusal names.
    report = report_of(target, state)
    condemned = [
        option
        for route in report.routes
        for step in route.retro_steps
        for option in (step.chosen, *step.alternatives)
        if option.rule == 1
    ]
    assert condemned, "the report should carry the condemned disconnection"
    for option in condemned:
        assert option.vetoes_fired == ("B 1", "B 7")
        assert all(code in reason for code in option.vetoes_fired)


def test_the_stored_column_validator_ignores_a_priced_row() -> None:
    """A `force="price"` row belongs to the price and must not block a route here as well.

    `fired_vetoes` reads `VETO_CHANNEL` only, so a `discouraged` column -- which is the same
    table evaluated for the other force -- is invisible to this. Charging for a row *and*
    refusing by it would make one verdict cost twice.
    """
    target, state = graph(
        outcomes={"discouraged": np.array([0, 0, 2, 2, 0])},
        reduce={"discouraged": refusal_wins},
    )
    validate = stored_veto_validator(state)

    for item in select_solved_routes(target, state, limit=5).routes:
        assert validate(item.route) is None


def test_a_veto_is_attributed_to_its_own_application_not_to_shared_chemistry() -> None:
    """Two products, one rule, the same precursors -- and only one of them condemned.

    This is the case the chemistry key cannot represent. A veto is a statement about an
    *application*, and `(precursors, rule)` is not one: two products reached by the same rule from
    the same starting materials collapse into a single entry there, so a verdict earned by one
    becomes a verdict against both. `prepare_graph` routes the per-row `vetoed:<code>` columns
    through the step-keyed index for exactly this reason, and `fired_vetoes` prefers it.

    So the assertion is asymmetry: `P1` is refused and `P2` is not, from one column that fires on
    `P1`'s application alone. Under the chemistry key both would be refused, and a correct route
    would be thrown away with the bad one.
    """
    expansions = Expansions(NumpyExpansionStore(), DictInterner())
    p1, p2, a, b = expansions.intern(["P1", "P2", "A", "B"]).tolist()
    expansions.add_layer(
        np.array([p1, p2]),
        # (product row, rule, application, precursor row): both products, rule 7, A + B.
        [(0, 7, 0, 0), (0, 7, 0, 1), (1, 7, 1, 0), (1, 7, 1, 1)],
        np.array([a, b]),
        # Fires on application 0 -- P1's -- and not on application 1.
        {"vetoed": np.array([2, 2, 0, 0]), "vetoed:B 9": np.array([1, 1, 0, 0])},
        {"vetoed": refusal_wins, "vetoed:B 9": refusal_wins},
    )
    state = prepare_graph(expansions, ["A", "B"], step_keyed=["vetoed:B 9"])
    validate = stored_veto_validator(state)

    (first,) = select_solved_routes(p1, state, limit=1).routes
    (second,) = select_solved_routes(p2, state, limit=1).routes
    assert validate(first.route) == "vetoed: B 9"
    assert validate(second.route) is None, (
        "P2's application was never condemned; refusing it would mean the veto had been "
        "attributed to the chemistry rather than to the application"
    )

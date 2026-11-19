import re
from pathlib import Path

import pytest
from jinja2.exceptions import SecurityError, UndefinedError

from vertsy.route_reports import render_route_steps
from vertsy.routes import Route, forward_steps

from .bench_route_reports import write_report
from .route_report_data import (
    archive_route,
    benchmark_target_archive,
    diversity_summary,
    render_report_qmd,
    render_target_qmd,
    route_from_archive,
    rule_display_name,
    template_tables,
)


def sample_route() -> Route[str]:
    return Route(
        "CCO",
        7,
        (Route("CC", 8, (Route("C", None),)), Route("O", None)),
    )


def sample_archive() -> dict:
    route = sample_route()
    return {
        "target_index": 2,
        "input_target": "CCO",
        "searched_target": "CCO",
        "solved": True,
        "backend": "metal",
        "width": 32,
        "pass_limit": 8,
        "stock": "test stock",
        "templates": "test rules",
        "source": "run.jsonl",
        "preprocessing": "none",
        "replay_status": "verified",
        "unique_route_count": 2,
        "routes": [archive_route(route, ("",) * 7 + ("CO>>C.O", "C>>CC"))],
        "template_smarts": {"7": "CO>>C.O", "8": "C>>CC"},
        "diversity": diversity_summary((route,)),
    }


def test_structured_route_round_trip_and_forward_order() -> None:
    route = sample_route()
    record = archive_route(route, ("",) * 7 + ("CO>>C.O", "C>>CC"))
    assert route_from_archive(record) == route
    assert tuple(node.rule for node in forward_steps(route)) == (8, 7)


def test_report_contains_drawings_smiles_and_separate_pages() -> None:
    archive = sample_archive()
    archive["routes"].append(archive["routes"][0])
    qmd = render_target_qmd(archive)
    assert 'show_step(["C"], "CC")' in qmd
    assert "**Product:** `CCO`" in qmd
    assert "#pagebreak()" in qmd
    assert "Add chemist assessment" not in qmd
    assert "**Feasibility:** unreviewed" in render_target_qmd(archive, annotated=True)


def test_report_preserves_route_selection_provenance() -> None:
    archive = sample_archive()
    archive["route_selection"] = {
        "candidates_considered": 10,
        "partial_states_explored": 40,
        "truncated": True,
    }
    archive["routes"][0]["selection"] = {
        "depth": 2,
        "reactions": 2,
        "utility": 0.81,
        "base_step_factor": 0.81,
        "stock_factor_product": 1.0,
        "reaction_factor_product": 1.0,
        "first_distinct_decision": None,
    }
    qmd = render_target_qmd(archive)
    assert "Candidate discovery reached a resource limit" in qmd
    assert "**Longest sequence:** 2 reaction(s)" in qmd
    assert "**Relative utility:** 0.81" in qmd


@pytest.mark.parametrize("solved", [True, False])
def test_saved_benchmark_report_keeps_replay_and_template_provenance(solved) -> None:
    original = sample_archive()
    metadata = original | {
        "input_target": "C[OH2+]",
        "template_names": {"7": "Hartenfeller test reaction"},
        "templates": "Hartenfeller v2",
        "preprocessing": "free-base probe",
    }
    record = {
        "target": "CCO",
        "solved": solved,
        "backend": "metal",
        "width": 128,
        "pass_limit": 16,
        "seconds": 1.0,
        "routes": 2 if solved else 0,
        "replay_failures": 1 if solved else 0,
        "route_records": original["routes"] if solved else [],
    }
    archive = benchmark_target_archive(record, metadata)
    qmd = render_target_qmd(archive)
    assert "**Input target SMILES:** `C[OH2+]`" in qmd
    assert "**Searched target SMILES:** `CCO`" in qmd
    assert "**Template library:** Hartenfeller v2" in qmd
    assert archive["diversity"]["sampled_routes"] == (1 if solved else 0)
    if solved:
        assert "1/2 selected routes replayed through Vesma CPU; 1 failed" in qmd
        assert "Step 2: Hartenfeller test reaction" in qmd
    else:
        assert "No solved route was found" in qmd


def test_unsolved_target_report_and_sample_diversity() -> None:
    archive = sample_archive() | {
        "solved": False,
        "routes": [],
        "diversity": diversity_summary(()),
    }
    qmd = render_target_qmd(archive)
    assert "No solved route was found" in qmd
    assert "# Route 1" not in qmd


def test_step_template_uses_forward_order_and_plain_metadata() -> None:
    route = sample_route()
    result = render_route_steps(
        route,
        "{{ step_index }}:{{ precursor_smiles | join('+') }}>{{ product_smiles }} "
        "({{ metadata.name }})",
        lambda index, step: {"name": f"rule-{step.rule}"},
    )
    assert result.splitlines() == ["1:C>CC (rule-8)", "", "2:CC+O>CCO (rule-7)"]


def test_step_template_blocks_python_introspection_and_missing_fields() -> None:
    with pytest.raises((SecurityError, UndefinedError)):
        render_route_steps(sample_route(), "{{ product_smiles.__class__.__mro__ }}")
    with pytest.raises(UndefinedError):
        render_route_steps(sample_route(), "{{ metadata.missing }}")
    with pytest.raises(TypeError):
        render_route_steps(
            sample_route(), "{{ metadata.value }}", lambda i, s: {"value": object()}
        )


# --------------------------------------------------------------------------- #
# Naming a rule, and one document per run
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "columns,expected,why",
    [
        (
            {"classification": "Heck terminal vinyl"},
            "Heck terminal vinyl",
            "hartenfeller",
        ),
        (
            {"rc_code": "3.11.30.1", "ri_name": "C-C Coupling"},
            "C-C Coupling",
            "curated",
        ),
        ({"classification": "0.0 Unrecognized"}, None, "the placeholder is not a name"),
        ({}, None, "used14k carries no columns"),
        ({"template_code": "7"}, None, "an id is not a name"),
    ],
)
def test_a_rule_is_named_from_whichever_column_its_source_uses(
    columns: dict[str, str], expected: str | None, why: str
) -> None:
    """Sources disagree and none of them is wrong, so the columns are tried in order."""
    assert rule_display_name({"columns": columns}) == expected


def test_unnamed_rules_are_absent_rather_than_placeheld() -> None:
    """So the renderer's `Template <id>` fallback still reads as 'this has no name'."""
    rows = [
        {"rule_id": 1, "retro_template": "A>>B", "columns": {"ri_name": "Acylation"}},
        {"rule_id": 2, "retro_template": "C>>D", "columns": {}},
    ]

    smarts, names = template_tables(rows)

    assert smarts == {"1": "A>>B", "2": "C>>D"}, "every rule keeps its SMARTS"
    assert names == {1: "Acylation"}


def test_one_document_holds_every_target_under_a_single_preamble() -> None:
    """Quarto runs one kernel per file, so the setup is defined once and sections are content."""
    first = sample_archive() | {"target_index": 1}
    second = sample_archive() | {"target_index": 2}

    report = render_report_qmd((first, second))

    assert report.count("format: typst") == 1, "one front matter"
    assert report.count("def show_molecule") == 1, "one copy of the helpers"
    assert re.findall(r"^# Target \d+$", report, re.MULTILINE) == [
        "# Target 1",
        "# Target 2",
    ]


def test_a_targets_routes_sit_under_its_heading() -> None:
    """A standalone target starts at `# Route 1`; concatenated, that has to move down a level.

    Otherwise a route would be a sibling of the target it belongs to, and the PDF outline would
    read as twice as many top-level sections as there are targets.
    """
    report = render_report_qmd((sample_archive(),))

    assert "## Route 1" in report
    assert "\n# Route 1" not in report


def test_demotion_leaves_typst_and_cell_directives_alone() -> None:
    """`#show`, `#pagebreak()` and `#|` all start a line with `#` and none is a heading."""
    report = render_report_qmd((sample_archive(),))

    assert "#show raw.where" in report
    assert "#| include: false" in report
    assert "##show" not in report and "##|" not in report


def test_a_standalone_target_qmd_still_carries_its_own_setup() -> None:
    """The per-target form is unchanged, so the legacy archive path keeps working."""
    qmd = render_target_qmd(sample_archive())

    assert qmd.startswith("---")
    assert "def show_molecule" in qmd
    # Undemoted: a standalone target is its own document, so its routes stay at the top level.
    assert "# Route 1" in qmd


def test_an_annotated_report_is_never_overwritten(tmp_path: Path) -> None:
    """Regenerating the mechanical report must not discard a chemist's assessments.

    The guarantee lives in `write_report` rather than in its caller, so it holds for anyone
    who calls it -- the caller that used to hold it was a perf-marked bench that skips.
    """
    archives = (sample_archive(),)
    annotated = tmp_path / "report-annotated.qmd"

    write_report(archives, annotated, annotated=True, keep_existing=True)
    annotated.write_text("Chemist's notes")
    write_report(archives, annotated, annotated=True, keep_existing=True)

    assert annotated.read_text() == "Chemist's notes"


def test_the_mechanical_report_is_regenerated_every_time(tmp_path: Path) -> None:
    """Only the annotated copy is protected; the machine-written one must stay current."""
    report = tmp_path / "report.qmd"
    report.write_text("stale")

    write_report((sample_archive(),), report)

    assert report.read_text() != "stale"


def alternatives_report() -> dict:
    """One target's alternatives report, in the shape `alternatives.jsonl` carries."""
    option = {
        "rule": 7,
        "rule_label": "MC 62",
        "precursors": ["CC", "O"],
        "also_by": [],
        "log_cost": 0.105,
        "cost_delta": 0.0,
        "measurements": {"sites": 1, "competitors": None},
        "vetoes_fired": ["B 30"],
        "standing": "chosen",
        "also_reported_in": [],
    }
    return {
        "target": "CCO",
        "routes": [
            {
                "rank": 1,
                "standing": "best",
                "score": {"reactions": 1},
                "first_distinct_decision": None,
                "differing_decisions": None,
                "resembled": None,
                "retro_steps": [
                    {
                        "index": 1,
                        "product": "CCO",
                        "chosen": option,
                        "alternatives": [
                            option
                            | {
                                "standing": "competing-sites",
                                "rule_label": "Hartenfeller 45a",
                                "precursors": ["CC=O"],
                                "cost_delta": 0.021,
                                "measurements": {"sites": 3, "competitors": None},
                                "vetoes_fired": [],
                            }
                        ],
                    }
                ],
            }
        ],
        "rejected": [
            {
                "rank": 0,
                "standing": "censored",
                "score": {"reactions": 2},
                "first_distinct_decision": None,
                "differing_decisions": None,
                "resembled": None,
                "retro_steps": [],
                "reason": "rule 338: replay emitted 0 application(s)",
            }
        ],
        "truncated": False,
        "stopped_after_limit": False,
        "channels": ["sites", "competitors"],
        "diversity_cutoff": None,
    }


def test_the_pdf_carries_the_alternatives_and_what_was_measured_on_them() -> None:
    """The per-step evidence has to reach the document a chemist actually reads.

    An unmeasured channel renders as a dash, never a zero: a stored zero means the probe ran and
    the reaction made nothing, while an absent one means nobody looked, and a report that
    collapsed them would show an unexamined rule as a clean one.
    """
    archive = sample_archive() | {"alternatives": alternatives_report()}

    qmd = render_target_qmd(archive)

    assert "# Alternatives considered" in qmd
    assert "### Retro-step 1" in qmd
    assert "| chosen |" in qmd
    assert "Hartenfeller 45a" in qmd, "the alternative's rule is named"
    assert "B 30" in qmd, "and so is the veto row that fired on the step taken"
    chosen_row = next(line for line in qmd.splitlines() if line.startswith("| chosen "))
    assert chosen_row.endswith("| B 30 |")
    assert "| -- |" in chosen_row, "competitors was never measured"
    assert "rule 338: replay emitted 0 application(s)" in qmd


def test_a_report_without_alternatives_says_nothing_about_them() -> None:
    """Runs made before the report existed still render, and say nothing they cannot support."""
    qmd = render_target_qmd(sample_archive())

    assert "Alternatives considered" not in qmd

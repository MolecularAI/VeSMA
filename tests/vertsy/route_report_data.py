"""Experiment JSON records and target-level Quarto report assembly."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from functools import partial
from typing import Any

from vertsy.route_reports import render_route_steps
from vertsy.routes import Route, reaction_depth

__all__ = [
    "archive_route",
    "diversity_summary",
    "render_target_qmd",
    "route_from_archive",
]

RULE_LINE = re.compile(r"^( *)<- rule (\d+): (.*)$")


def archive_route(route: Route[str], smarts: Sequence[str]) -> dict[str, Any]:
    return {
        "smiles": route.product,
        "template_id": route.rule,
        "template_smarts": smarts[route.rule] if route.rule is not None else None,
        "precursors": [archive_route(child, smarts) for child in route.precursors],
    }


def route_from_archive(node: Mapping[str, Any]) -> Route[str]:
    return Route(
        node["smiles"],
        node["template_id"],
        tuple(route_from_archive(child) for child in node["precursors"]),
    )


def benchmark_target_archive(
    record: Mapping[str, Any],
    metadata: Mapping[str, Any],
    alternatives: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Adapt a completed host benchmark record without rerunning search or replay.

    ``alternatives`` is this target's entry from the run's `alternatives.jsonl`, joined by the
    caller. It arrives separately because it is written separately -- one per-disconnection report
    is larger than every timing in `targets.jsonl` put together -- and it is optional because a
    run made before the report existed, or one whose targets were never solved, has none.
    """
    routes = tuple(route_from_archive(node) for node in record["route_records"])
    selection = record.get("route_selection")
    if not record["solved"]:
        replay_status = "not applicable: no solved routes"
    elif record["backend"] == "metal":
        replay_status = (
            f"{len(routes)}/{record['routes']} selected routes replayed through Vesma CPU; "
            f"{record['replay_failures']} failed. "
            "Replay confirms graph consistency; chemical feasibility is unreviewed."
        )
    else:
        replay_status = (
            f"{len(routes)} native readable routes; chemical feasibility is unreviewed"
        )
    return dict(metadata) | {
        "schema_version": 1,
        "searched_target": record["target"],
        "solved": record["solved"],
        "backend": record["backend"],
        "width": record["width"],
        "pass_limit": record["pass_limit"],
        "seconds": record["seconds"],
        "replay_status": replay_status,
        "routes": record["route_records"],
        "unique_route_count": record.get("unique_route_count"),
        "unique_route_count_capped": record.get("unique_route_count_capped", False),
        # None when there is nothing the cap could have bounded, matching
        # the legacy per-target archive builder. The two builders write the same
        # schema and a reader should not have to know which produced a file.
        "route_count_cap": (
            "10^100" if record.get("first_disconnection_counts") else None
        ),
        "first_disconnection_counts": record.get("first_disconnection_counts", []),
        "depth_route_profile": record.get("depth_route_profile"),
        "route_selection": (
            selection | {"seconds": record.get("route_selection_seconds")}
            if selection is not None
            else None
        ),
        "sampled_route_depths": sorted({reaction_depth(route) for route in routes}),
        "diversity": diversity_summary(routes),
        # Starting-material SMILES -> every catalogue listing it. Supplied by the caller rather
        # than carried through the search, which holds a Collection of keys and nothing more.
        "stock_sources": metadata.get("stock_sources", {}),
        "alternatives": alternatives,
    }


def walk(route: Route[str]) -> Iterator[Route[str]]:
    yield route
    for child in route.precursors:
        yield from walk(child)


def route_leaves(route: Route[str]) -> frozenset[str]:
    return frozenset(node.product for node in walk(route) if node.rule is None)


def plan_key(route: Route[str]) -> tuple[str, ...]:
    """Purchased-reagent-blind intermediate multiset for a sampled route."""
    return tuple(sorted(node.product for node in walk(route) if node.rule is not None))


def diversity_summary(routes: Sequence[Route[str]]) -> dict[str, Any]:
    """Describe only the supplied sample; do not extrapolate to the search graph."""
    if not routes:
        return {
            "sampled_routes": 0,
            "sampled_plans": 0,
            "sampled_stock_sets": 0,
            "sampled_first_disconnections": 0,
            "plan_groups": [],
            "stock_set_groups": [],
            "first_disconnection_groups": [],
            "common_steps": [],
            "varying_steps": [],
        }
    step_sets = [
        frozenset(
            (node.product, frozenset(child.product for child in node.precursors))
            for node in walk(route)
            if node.rule is not None
        )
        for route in routes
    ]
    common = set.intersection(*(set(steps) for steps in step_sets))
    all_steps = set.union(*(set(steps) for steps in step_sets))

    def describe(step: tuple[str, frozenset[str]]) -> dict[str, Any]:
        return {
            "product": step[0],
            "precursors": sorted(step[1]),
            "routes": sum(step in steps for steps in step_sets),
            "template_ids": sorted(
                {
                    node.rule
                    for route in routes
                    for node in walk(route)
                    if node.rule is not None
                    and (
                        node.product,
                        frozenset(child.product for child in node.precursors),
                    )
                    == step
                }
            ),
        }

    def groups(keys: Sequence[Any]) -> list[dict[str, Any]]:
        indexed: dict[Any, list[int]] = {}
        for index, key in enumerate(keys, 1):
            indexed.setdefault(key, []).append(index)
        return [
            {
                "routes": indices,
                "key": sorted(key) if isinstance(key, frozenset) else key,
            }
            for key, indices in indexed.items()
        ]

    plan_groups = groups([plan_key(route) for route in routes])
    stock_groups = groups([route_leaves(route) for route in routes])
    first_groups = groups(
        [frozenset(child.product for child in route.precursors) for route in routes]
    )

    return {
        "sampled_routes": len(routes),
        "sampled_plans": len(plan_groups),
        "sampled_stock_sets": len(stock_groups),
        "sampled_first_disconnections": len(first_groups),
        "plan_groups": plan_groups,
        "stock_set_groups": stock_groups,
        "first_disconnection_groups": first_groups,
        "common_steps": [describe(step) for step in sorted(common)],
        "varying_steps": [describe(step) for step in sorted(all_steps - common)],
    }


def python_literal(value: str) -> str:
    return repr(value)


def qmd_alternatives(report: Mapping[str, Any]) -> list[str]:
    """One section per reported route: every other disconnection the engine could have taken.

    Rendered from the report's JSON rather than from its dataclasses, and with its own narrower
    table than the Markdown twin in :func:`vertsy.route_alternatives.render_report_markdown`: a
    page has a width, so the reaction is given as its precursors with the product in the heading
    already, and long SMILES are abbreviated. Nothing is dropped -- the full form is in
    `alternatives.jsonl` beside the archive.

    Retro-step 1 is the **first disconnection**, which is the **last** synthesis step. The
    numbering runs backwards because that is how a plan is read when the question is what to make
    last; the reactions themselves are still written forwards.
    """
    channels = list(report.get("channels", ()))
    if not report.get("routes"):
        return []
    lines = [
        "The engine costs every disconnection of every molecule it reaches, and measures the "
        "ones it probed. These are the alternatives it had at each step of each route reported "
        "above, with what it knew about them. A `standing` compares an alternative with the step "
        "*taken*, so `competing-sites` means more products than that step, not more than one; a "
        "dash means **nobody looked**, which is not the same as a measured zero.",
        "",
    ]
    for route in report["routes"]:
        lines += [f"## Route {route['rank']} — {route['standing']}", ""]
        for step in route["retro_steps"]:
            product = abbreviated_smiles(step["product"])
            lines += [
                f"### Retro-step {step['index']} — `{product}`",
                "",
                f"**Taken:** `{' + '.join(step['chosen']['precursors'])}` "
                f"— {step['chosen']['rule_label']}",
                "",
            ]
            columns = ["standing", "precursors", "rule", "cost", *channels, "vetoes"]
            lines += [
                "| " + " | ".join(columns) + " |",
                "|" + "|".join(["---"] * len(columns)) + "|",
            ]
            for option in (step["chosen"], *step["alternatives"]):
                measured = option["measurements"]
                cells = [
                    option["standing"],
                    "`"
                    + abbreviated_smiles(" + ".join(option["precursors"]), 48)
                    + "`",
                    option["rule_label"],
                    f"{option['cost_delta']:+.3f}" if option["cost_delta"] else "0",
                    *(
                        (
                            "--"
                            if measured.get(channel) is None
                            else str(measured[channel])
                        )
                        for channel in channels
                    ),
                    ", ".join(option["vetoes_fired"]) or "--",
                ]
                lines += ["| " + " | ".join(cells) + " |"]
            lines += [""]
    refused = [one for one in report["rejected"] if one["standing"] == "censored"]
    if refused:
        lines += [
            "## Refused at validation",
            "",
            "Costed and ranked, then refused when the plan was replayed against real molecules. "
            "Each was replaced by the next route, so nothing reported above was crowded out.",
            "",
            "| refused because | reactions |",
            "|---|---|",
        ]
        lines += [
            f"| {one['reason']} | {one['score']['reactions']} |" for one in refused
        ]
        lines += [""]
    return lines


def abbreviated_smiles(smiles: str, limit: int = 55) -> str:
    return smiles if len(smiles) <= limit else smiles[: limit - 1] + "…"


STEP_TEMPLATE = """## Step {{ step_index }}: {{ metadata.name }}

```{python}
show_step({{ precursor_smiles | tojson }}, {{ product_smiles | tojson }})
```

**Precursors:** {% for smiles in precursor_smiles %}`{{ smiles }}`{% if not loop.last %} + {% endif %}{% endfor %}{{ '  ' }}
**Product:** `{{ product_smiles }}`{{ '  ' }}
**Template ID:** {{ template_id }}{{ '  ' }}
**Retro SMARTS:** `{{ metadata.smarts }}`
{% if metadata.annotated %}
**Feasibility:** unreviewed{{ '  ' }}
**Comment:** {{ metadata.comment }}
{% endif %}"""


CONCISE_STEP_TEMPLATE = """## Step {{ step_index }}: {{ metadata.name }}

```{python}
show_step({{ precursor_smiles | tojson }}, {{ product_smiles | tojson }})
```

{% for smiles in precursor_smiles %}`{{ smiles }}`{% if not loop.last %} + {% endif %}{% endfor %} → `{{ product_smiles }}`
{% if metadata.annotated %}
**Feasibility:** unreviewed{{ '  ' }}
**Comment:** {{ metadata.comment }}
{% endif %}"""
"""A step as a chemist reads it: the drawing, and the line they paste into RDKit.

The detailed template also prints the template id and its retro SMARTS. Both are recoverable
from the JSON archive by anyone debugging a rule, and neither is what a reviewer is looking at.
"""


def qmd_step_metadata(
    index: int,
    node: Route[str],
    *,
    archive: Mapping[str, Any],
    names: Mapping[int, str],
    comments: Mapping[int, str],
    annotated: bool,
) -> dict[str, Any]:
    del index
    assert node.rule is not None
    return {
        "name": names.get(node.rule, f"Template {node.rule}"),
        "smarts": archive["template_smarts"].get(str(node.rule), ""),
        "annotated": annotated,
        "comment": comments.get(node.rule, "Add chemist assessment here."),
    }


#: Column names a source may use for a human-readable reaction class, best first.
#:
#: Tried in order because the sources disagree and none of them is wrong: Hartenfeller writes
#: `classification`, the curated raw68k projection writes `ri_name` beside a finer `rc_code`,
#: and uspto42k writes `classification` but fills it with `0.0 Unrecognized` on every row, which
#: is why the placeholder is rejected rather than printed.
NAME_COLUMNS = ("classification", "ri_name", "mapped_ri_name")

#: What AiZynthFinder ships in `classification` when it means "no label".
UNRECOGNIZED = "0.0 Unrecognized"


def rule_display_name(row: Mapping[str, Any]) -> str | None:
    """A readable reaction class for one `templates.json` row, or ``None``.

    Reads the source's own columns rather than a normalised field, because normalising them at
    load time would mean writing one vocabulary's value into another's column -- `rc_code` is an
    LLM-derived taxonomy and `classification` is NextMove-shaped, and they are not the same
    thing said twice.

    ``None`` rather than a placeholder, so the caller's existing `Template <id>` fallback still
    reads as "this rule has no name" instead of "this rule is called Unrecognized".
    """
    columns = row.get("columns", {})

    for column in NAME_COLUMNS:
        value = columns.get(column)

        if value and value != UNRECOGNIZED:
            return value

    return None


def template_tables(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[int, str]]:
    """`templates.json` as the two lookups the renderer wants: SMARTS by id, names by id.

    SMARTS is keyed by the *string* id and names by the int, matching what
    :func:`qmd_step_metadata` already reads. Unnamed rules are simply absent from the names
    table rather than present with a placeholder.
    """
    smarts = {str(row["rule_id"]): row["retro_template"] for row in rows}
    names = {
        int(row["rule_id"]): name
        for row in rows
        if (name := rule_display_name(row)) is not None
    }

    return smarts, names


#: An ATX heading: one to five hashes, a space, then something that is not a hash.
#:
#: The space is what keeps this off Typst's `#show` and `#pagebreak()` and Quarto's `#|` cell
#: directives, all of which start a line with `#` and none of which is a heading.
HEADING = re.compile(r"^(#{1,5}) (?=\S)", flags=re.MULTILINE)


def demoted(section: str) -> str:
    """Every heading one level deeper, so a target heading can contain a target's routes.

    A standalone target puts its name in the front matter and starts its body at `# Route 1`.
    Concatenated under one preamble that name has to become a heading, and it has to sit above
    the routes rather than beside them.
    """
    return HEADING.sub(r"#\1 ", section)


def render_report_qmd(
    archives: Sequence[Mapping[str, Any]],
    names: Mapping[int, str] | None = None,
    comments: Mapping[int, str] | None = None,
    *,
    annotated: bool = False,
    title: str = "Route review",
    concise: bool = True,
) -> str:
    """Every target of a run in one document.

    ``concise`` picks :func:`render_target_qmd_concise`, which is the default: a reviewer wants
    the molecule and the routes, and the run diagnostics were pushing both below the fold. Pass
    ``False`` for the full version when debugging a run rather than reading one.

    One file rather than one per target, because a reviewer reads a run rather than a molecule,
    and because the annotation cycle works on a single file: an agent reads this, adds
    assessments, saves it beside the original, and that renders to a second PDF.
    """
    render = render_target_qmd_concise if concise else render_target_qmd
    # A page break before every target but the first, for the same reason routes get one: a
    # reviewer reads one target at a time, and a target that starts halfway down the previous
    # one's last page reads as a continuation of it.
    sections = [
        ("" if index == 1 else "```{=typst}\n#pagebreak()\n```\n\n")
        + f"# Target {archive['target_index']}\n\n"
        + demoted(
            render(archive, names, comments, annotated=annotated, standalone=False)
        )
        for index, archive in enumerate(archives, 1)
    ]

    return "\n".join([*qmd_preamble(title), *sections])


def qmd_preamble(title: str) -> list[str]:
    """Front matter and the setup every target section depends on.

    Split out so several targets can share one document: Quarto runs one kernel per file, so
    the imports and the two drawing helpers are defined once and the per-target sections are
    pure content. A standalone per-target QMD still gets its own copy.
    """
    return [
        "---",
        f"title: {json.dumps(title)}",
        "format: typst",
        "jupyter: python3",
        "execute:",
        "  echo: false",
        "  warning: false",
        "---",
        "",
        "```{=typst}",
        "// Allow long SMILES to wrap without adding characters to the copied text.",
        "#show raw.where(block: false): it => {",
        '  set text(font: "DejaVu Sans Mono", size: 0.82em, hyphenate: false)',
        "  it.text.clusters().map(c => text(c)).intersperse(h(0.1pt)).join()",
        "}",
        "```",
        "",
        "```{python}",
        "#| include: false",
        "from IPython.display import display",
        "from rdkit import Chem",
        "from rdkit.Chem import Draw, rdChemReactions",
        "",
        "def show_molecule(smiles):",
        "    molecule = Chem.MolFromSmiles(smiles)",
        "    if molecule is None: raise ValueError(f'Invalid SMILES: {smiles}')",
        "    display(Draw.MolToImage(molecule, size=(700, 420)))",
        "",
        "def show_step(precursors, product):",
        "    reaction = rdChemReactions.ReactionFromSmarts('.'.join(precursors) + '>>' + product, useSmiles=True)",
        "    display(Draw.ReactionToImage(reaction, subImgSize=(500, 340), useSVG=False))",
        "```",
        "",
    ]


def render_target_qmd(
    archive: Mapping[str, Any],
    names: Mapping[int, str] | None = None,
    comments: Mapping[int, str] | None = None,
    *,
    annotated: bool = False,
    standalone: bool = True,
) -> str:
    """Generate a QMD whose execution never reads the source archive.

    ``standalone=False`` drops the front matter and the shared setup and opens with a target
    heading instead, so several of these concatenate into one document under a single
    :func:`qmd_preamble`.
    """
    names = (
        {int(rule): name for rule, name in archive.get("template_names", {}).items()}
        if names is None
        else names
    )
    comments = (
        {int(rule): note for rule, note in archive.get("template_comments", {}).items()}
        if comments is None
        else comments
    )
    routes = tuple(route_from_archive(node) for node in archive["routes"])
    title = f"Target {archive['target_index']} — route review"
    lines = [
        # No heading when embedded: `render_report_qmd` writes one *outside* the section, so
        # that it survives the demotion the body needs.
        *(qmd_preamble(title) if standalone else []),
        f"**Input target SMILES:** `{archive['input_target']}`  ",
        f"**Searched target SMILES:** `{archive['searched_target']}`  ",
        f"**Result:** {'graph solved' if archive['solved'] else 'unsolved'}  ",
        f"**Backend and effort:** {archive['backend']}, {archive['width']}×{archive['pass_limit']}  ",
        f"**Stock:** {archive['stock']}  ",
        f"**Template library:** {archive['templates']}  ",
        f"**Source:** `{archive['source']}`  ",
        f"**Preprocessing:** {archive['preprocessing']}  ",
        f"**Replay:** {archive['replay_status']}",
        "",
        "```{python}",
        f"show_molecule({python_literal(archive['searched_target'])})",
        "```",
        "",
    ]
    diversity = archive.get("diversity", {})
    if routes:
        selection = archive.get("route_selection")
        if selection is not None:
            if selection["truncated"]:
                coverage = (
                    "Candidate discovery reached a resource limit, so other "
                    "eligible plans may exist."
                )
            elif selection.get("stopped_after_limit"):
                coverage = (
                    "Selection stopped after enough representatives; other "
                    "eligible plans may exist."
                )
            else:
                coverage = "All eligible candidate routes were considered."
            work = (
                f"{selection['candidates_considered']:,} candidate routes considered; "
                f"{selection['partial_states_explored']:,} partial states explored"
            )
            if selection.get("seconds") is not None:
                work += f" in {selection['seconds']:.2f} s"
            lines += [
                "## Route selection",
                "",
                "Routes are selected by shortest longest sequence, then total reactions; "
                "distinct disconnection plans share the report slots.",
                "",
                f"{work}. {coverage}",
                "",
            ]
        lines += [
            "## Diversity of the reported sample",
            "",
            f"{diversity.get('sampled_routes', len(routes))} sampled routes; "
            f"{diversity.get('sampled_plans', 'unknown')} intermediate plans; "
            f"{diversity.get('sampled_stock_sets', 'unknown')} stock sets; "
            f"{diversity.get('sampled_first_disconnections', 'unknown')} first disconnections.",
            "",
        ]
        if archive.get("unique_route_count") is not None:
            bounded = archive.get("unique_route_count_capped", False)
            lines += [
                f"The full search graph contains {archive['unique_route_count']:,} depth-bounded "
                f"precursor-choice trees{' or more' if bounded else ''}. "
                "This count combines independent branch choices and "
                "is not a count of distinct chemical plans.",
                "",
            ]
        depth_profile = archive.get("depth_route_profile")
        if depth_profile and depth_profile["shortest_depth"] is not None:
            lines += [
                "### Route counts by longest reaction sequence",
                "",
                f"The shortest depth is {depth_profile['shortest_depth']}. "
                "Counts refer to distinct precursor-choice trees in this search graph.",
                "",
            ]
            sampled_depths = archive.get("sampled_route_depths", [])
            if sampled_depths:
                lines += [
                    "The displayed routes have depths "
                    + ", ".join(map(str, sampled_depths))
                    + "."
                    + (
                        " None of the displayed routes attains the graph's shortest depth."
                        if depth_profile["shortest_depth"] not in sampled_depths
                        else ""
                    ),
                    "",
                ]
            for count in depth_profile["counts"]:
                exact = (
                    f"{count['exact']:,}"
                    if count["exact"] is not None
                    else "unavailable (count cap reached)"
                )
                lines += [
                    f"- Depth {count['depth']}: {exact} routes at exactly this depth; "
                    f"{count['at_most']:,} at this depth or less."
                ]
            first_by_depth = [
                item
                for item in depth_profile["first_disconnections"]
                if item["counts"] and item["counts"][0]["at_most"]
            ]
            lines += [
                "",
                f"{len(first_by_depth)} first disconnections contribute at the shortest depth. "
                "The leading ones are:",
                "",
            ]
            for index, item in enumerate(first_by_depth[:12], 1):
                count = item["counts"][0]
                exact = (
                    f"{count['exact']:,}"
                    if count["exact"] is not None
                    else "at least the count cap"
                )
                sampled = item.get("sampled_precursor_smiles")
                description = (
                    "; sampled SMILES "
                    + " + ".join(f"`{abbreviated_smiles(s)}`" for s in sampled)
                    if sampled
                    else ""
                )
                lines += [
                    f"- Rank {index}: {exact} shortest-depth routes; "
                    f"{len(item['precursor_keys'])} precursor(s){description}"
                ]
            if len(first_by_depth) > 12:
                lines += [
                    "The complete first-disconnection profile for all four depths is "
                    "in the JSON archive."
                ]
            lines += [""]
        disconnections = archive.get("first_disconnection_counts", [])
        if disconnections:
            shown_disconnections = disconnections[:12]
            lines += [
                f"**Solved first disconnections in the full graph:** {len(disconnections)}. "
                "The counts below are exact depth-bounded precursor-choice trees beneath "
                f"each distinct first precursor set unless a count reached the "
                f"{archive.get('route_count_cap', 'recorded')} cap.",
                "",
            ]
            lines += [
                f"- Rank {index}: {item['route_count']:,} trees; "
                f"{len(item['precursor_keys'])} first precursor(s)"
                + (
                    "; sampled SMILES "
                    + " + ".join(
                        f"`{abbreviated_smiles(smiles)}`"
                        for smiles in item["sampled_precursor_smiles"]
                    )
                    if item.get("sampled_precursor_smiles")
                    else ""
                )
                for index, item in enumerate(shown_disconnections, 1)
            ]
            if len(disconnections) > len(shown_disconnections):
                lines += [
                    f"The remaining {len(disconnections) - len(shown_disconnections)} "
                    "counts and all opaque search keys are in the JSON archive."
                ]
            lines += [""]
        for label, key in (
            ("Intermediate skeleton", "plan_groups"),
            ("Stock-material set", "stock_set_groups"),
            ("First disconnection", "first_disconnection_groups"),
        ):
            groups = diversity.get(key, [])
            lines += [f"**{label} groups in this sample:**", ""]
            lines += [
                f"- Group {index}: routes {', '.join(map(str, group['routes']))}"
                for index, group in enumerate(groups, 1)
            ] or ["None."]
            lines += [""]
        for label, key in (
            ("Steps common to every sampled route", "common_steps"),
            ("Steps that vary across the sample", "varying_steps"),
        ):
            steps = diversity.get(key, [])
            shown = sorted(steps, key=lambda step: -step["routes"])[:12]
            lines += [f"### {label} ({len(steps)})", ""]
            lines += [
                f"- {step['routes']}/{len(routes)}: "
                f"product `{abbreviated_smiles(step['product'])}`; "
                f"{len(step['precursors'])} precursor(s); "
                f"template(s) {', '.join(map(str, step.get('template_ids', [])))}"
                for step in shown
            ] or ["None."]
            if len(steps) > len(shown):
                lines += [
                    f"{len(steps) - len(shown)} more sampled steps are recorded in the JSON archive; "
                    "full SMILES and templates appear in the route sections."
                ]
            lines += [""]
    else:
        message = (
            "No route was extracted for display."
            if archive["solved"]
            else "No solved route was found in this run."
        )
        lines += [message, ""]
    for route_index, route in enumerate(routes, 1):
        lines += ["```{=typst}", "#pagebreak()", "```", ""]
        lines += [f"# Route {route_index}", ""]
        selected = archive["routes"][route_index - 1].get("selection")
        if selected is not None:
            distinct = selected["first_distinct_decision"]
            lines += [
                f"**Longest sequence:** {selected['depth']} reaction(s)  ",
                f"**Total reactions:** {selected['reactions']}  ",
                f"**Relative utility:** {selected['utility']:.4g} "
                f"(step {selected['base_step_factor']:.4g} × "
                f"stock {selected['stock_factor_product']:.4g} × "
                f"reaction quality {selected['reaction_factor_product']:.4g})  ",
                "**Diversity:** "
                + (
                    "highest-ranked representative"
                    if distinct is None
                    else f"differs from a previous route at decision {distinct}"
                ),
                "",
            ]
        lines += ["### Starting materials", ""]
        for smiles in sorted(route_leaves(route)):
            lines += [f"- `{smiles}`"]
        lines += [""]
        lines += [
            render_route_steps(
                route,
                STEP_TEMPLATE,
                partial(
                    qmd_step_metadata,
                    archive=archive,
                    names=names,
                    comments=comments,
                    annotated=annotated,
                ),
            ),
            "",
        ]
    # After the routes, not between them: the alternatives are evidence a reader consults once
    # the plans themselves have been read, and the section is one page break away for that reason.
    alternatives = archive.get("alternatives")
    if alternatives:
        lines += ["```{=typst}", "#pagebreak()", "```", ""]
        lines += ["# Alternatives considered", ""]
        lines += qmd_alternatives(alternatives)
    return "\n".join(lines) + "\n"


def render_target_qmd_concise(
    archive: Mapping[str, Any],
    names: Mapping[int, str] | None = None,
    comments: Mapping[int, str] | None = None,
    *,
    annotated: bool = False,
    standalone: bool = True,
) -> str:
    """One target, for reading rather than for debugging.

    The detailed twin is :func:`render_target_qmd` and it stays: route-selection work, the
    diversity breakdown, first-disconnection profiles, backend, stock, preprocessing and replay
    status are all real and all answer questions a *run* raises. They are not what a chemist
    opens the document for, and putting them ahead of the chemistry meant the chemistry was
    several screens down.

    What survives is what gets acted on: whether it solved, how many routes, the molecule, and
    then the routes themselves. Every SMILES a reviewer might want to paste elsewhere is in a
    fenced block rather than inline, because inline code in a rendered PDF does not survive a
    copy cleanly once it wraps.
    """
    names = (
        {int(rule): name for rule, name in archive.get("template_names", {}).items()}
        if names is None
        else names
    )
    comments = (
        {int(rule): note for rule, note in archive.get("template_comments", {}).items()}
        if comments is None
        else comments
    )
    routes = tuple(route_from_archive(node) for node in archive["routes"])
    title = f"Target {archive['target_index']} — routes"

    headline = "solved" if archive["solved"] else "**unsolved**"
    counted = archive.get("unique_route_count")
    if counted is not None:
        capped = (
            " (at the count cap)" if archive.get("unique_route_count_capped") else ""
        )
        headline += f" · {counted:,} routes in the graph{capped}"
    headline += f" · {len(routes)} shown"

    lines = [
        *(qmd_preamble(title) if standalone else []),
        headline,
        "",
        "```{python}",
        f"show_molecule({python_literal(archive['searched_target'])})",
        "```",
        "",
        "```",
        archive["searched_target"],
        "```",
        "",
    ]
    if not routes:
        nothing = (
            "No route was extracted for display."
            if archive["solved"]
            else "No solved route was found in this run."
        )
        lines += [nothing, ""]
    for route_index, route in enumerate(routes, 1):
        lines += ["```{=typst}", "#pagebreak()", "```", ""]
        selected = archive["routes"][route_index - 1].get("selection")
        heading = f"# Route {route_index}"
        if selected is not None:
            heading += (
                f" — {selected['reactions']} reaction(s), "
                f"longest sequence {selected['depth']}"
            )
        lines += [heading, "", "### Starting materials", "", "```"]
        # Tab-separated SMILES and catalogue, which is the `.smi` format itself: the block pastes
        # straight into a file or a spreadsheet without anything to strip first. Every catalogue
        # carrying the molecule is listed, not only the one the search happened to match, because
        # "where can I buy this" is the question the list raises and one answer is rarely enough.
        sources = archive.get("stock_sources", {})
        for smiles in sorted(route_leaves(route)):
            found = sources.get(smiles) or []
            notes = "; ".join(found) if isinstance(found, list) else str(found)
            lines.append(f"{smiles}\t{notes}" if notes else smiles)
        lines += ["```", ""]
        lines += [
            render_route_steps(
                route,
                CONCISE_STEP_TEMPLATE,
                partial(
                    qmd_step_metadata,
                    archive=archive,
                    names=names,
                    comments=comments,
                    annotated=annotated,
                ),
            ),
            "",
        ]
    # After the routes, not between them: the alternatives are evidence a reader consults once
    # the plans themselves have been read, and the section is one page break away for that reason.
    alternatives = archive.get("alternatives")
    if alternatives:
        lines += ["```{=typst}", "#pagebreak()", "```", ""]
        lines += ["# Alternatives considered", ""]
        lines += qmd_alternatives(alternatives)
    return "\n".join(lines) + "\n"

from __future__ import annotations

import csv
import gzip
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.vertsy import bench_host_aostar
from tests.vertsy.bench_host_aostar import (
    ECFP4_BITS,
    FINGERPRINT,
    StockRow,
    ranking_change,
    read_extra_templates,
    stock_row,
)
from vertsy.costs import UNPROBED
from vertsy.expansions import Expansions
from vertsy.filters import CLEAR, Veto, host_screen, vetoes_in_scope
from vertsy.interner import DictInterner
from vertsy.route_selection import RouteScore, RouteSelection, SelectedRoute
from vertsy.routes import Route, SidRoute
from vertsy.search import expansion, object_mol_array
from vertsy.selectivity import object_combine
from vertsy.store.numpy_store import NumpyExpansionStore
from vesma import cmol_from_smiles
from vesma.keyer import rdkit_smiles_keyer


@pytest.mark.parametrize("suffix", [".csv", ".csv.gz", ".tsv", ".tsv.gz"])
def test_template_tables_preserve_smarts_and_source_order(
    tmp_path: Path, suffix: str
) -> None:
    path = tmp_path / f"templates{suffix}"
    templates = ("[C,N:1]-[O:2]>>[C,N:1].[O:2]", "[N:1]>>[N:1]")
    opener = gzip.open if suffix.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="," if ".csv" in suffix else "\t")
        writer.writerow(("template_code", "retro_template", "classification"))
        for index, smarts in enumerate(templates):
            writer.writerow((str(index), smarts, "name, with comma"))

    rows, smarts, sources = bench_host_aostar.load_template_library(
        "hartenfeller", path
    )

    assert (rows, smarts) == (2, templates)
    # The source row travels with each rule, because the rule's index is all that survives
    # into a route and these columns are the only place a readable name can come from.
    assert [source.row for source in sources] == [0, 1]
    assert all(source.source == "hartenfeller" for source in sources)
    assert sources[0].columns == {
        "template_code": "0",
        "classification": "name, with comma",
    }
    assert "retro_template" not in sources[0].columns, "the SMARTS is not duplicated"


def test_stock_row_uses_the_search_identity_for_stereochemical_smiles() -> None:
    smiles = "C[C@H](N)C(=O)O"
    row = stock_row(smiles, config=FINGERPRINT, n_bits=ECFP4_BITS)

    assert isinstance(row, StockRow)
    assert row.rdkit_key == rdkit_smiles_keyer.from_smiles(smiles)


def test_supplemental_templates_append_in_source_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "one.smarts"
    second = tmp_path / "two.smarts.gz"
    bad = tmp_path / "bad.smarts"
    first.write_text("# comment\n[C:1]-[O:2]>>[C:1].[O:2]\n")
    with gzip.open(second, "wt") as handle:
        handle.write("[C:1]-[N:2]>>[C:1].[N:2]\n")
    bad.write_text("[C:1]-[O:2]>>[C:1]\n")
    monkeypatch.setattr(
        bench_host_aostar,
        "load_template_library",
        lambda rule_set, path: (
            1,
            ("[C:1]-[C:2]>>[C:1].[C:2]",),
            (bench_host_aostar.RuleSource(rule_set, 0, {}),),
        ),
    )

    library = bench_host_aostar.load_library(
        "raw68k", tmp_path / "base", (first, second, bad)
    )

    assert library.smarts == (
        "[C:1]-[C:2]>>[C:1].[C:2]",
        *read_extra_templates(first),
        *read_extra_templates(second),
    )
    assert library.extra_counts == (1, 1, 1)
    assert library.rejected_indices == (3,)
    assert library.row_count == 1
    # Supplemental files have no columns to carry, and are recorded under their own path so
    # every rule id still has an origin.
    assert [source.source for source in library.sources] == [
        "raw68k",
        str(first),
        str(second),
    ]
    assert all(not source.columns for source in library.sources[1:])


@pytest.mark.parametrize(
    "text,expected",
    [
        ("16x8,4nx8", ((16, 8), ("4n", 8))),
        (" 04N x 8 ", (("4n", 8),)),
    ],
)
def test_parse_widening_efforts(text, expected) -> None:
    assert bench_host_aostar.parse_efforts(text) == expected


@pytest.mark.parametrize(
    "text", ["", "0nx8", "4nx0", "-4nx8", "nx8", "4nnx8", "4nx8,04nx8"]
)
def test_reject_invalid_widening_efforts(text) -> None:
    with pytest.raises(ValueError):
        bench_host_aostar.parse_efforts(text)


def test_widening_schedule_uses_one_based_iterations() -> None:
    assert bench_host_aostar.effort_widths("4n", 8) == (4, 8, 12, 16, 20, 24, 28, 32)
    assert bench_host_aostar.effort_widths(16, 8) == (16,) * 8


# --------------------------------------------------------------------------- #
# Several inputs of each kind
# --------------------------------------------------------------------------- #


def test_inputs_accept_registry_names_and_paths_interchangeably(tmp_path: Path) -> None:
    """So a shipped rule set and a one-off file can sit side by side in one list."""
    loose = tmp_path / "mine.smarts"
    loose.write_text("[C:1]>>[C:1]\n")
    registry = {"known": tmp_path / "known.tsv"}

    resolved = bench_host_aostar.parse_inputs(
        f"known, {loose}", registry, "AOSTAR_HOST_RULE_SET"
    )

    assert resolved == (("known", registry["known"]), ("mine", loose))


def test_an_unknown_input_fails_naming_the_valid_keys() -> None:
    """The alternative is a run that completes against a silently smaller library."""
    with pytest.raises(ValueError, match="neither one of"):
        bench_host_aostar.parse_inputs("nope", {"known": Path("x")}, "VAR")

    with pytest.raises(ValueError, match="is empty"):
        bench_host_aostar.parse_inputs(" , ", {"known": Path("x")}, "VAR")


def write_table(path: Path, rows: tuple[tuple[str, str], ...]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("template_code", "retro_template"))
        writer.writerows(rows)


def test_merging_rule_sets_is_first_wins_and_reports_each_contribution(
    tmp_path: Path,
) -> None:
    """Order is the whole precedence mechanism: the first source keeps the shared rule."""
    shared = "[C:1]-[O:2]>>[C:1].[O:2]"
    first = tmp_path / "first.tsv"
    second = tmp_path / "second.tsv"
    write_table(first, (("a", shared), ("b", "[N:1]>>[N:1]")))
    write_table(second, (("c", shared), ("d", "[S:1]>>[S:1]")))

    library = bench_host_aostar.load_merged_library(
        (("first", first), ("second", second))
    )

    assert library.smarts == (shared, "[N:1]>>[N:1]", "[S:1]>>[S:1]")
    assert library.source_counts == {"first": 2, "second": 1}
    # The surviving copy is the first source's row, so its columns -- hence its name -- win.
    assert library.sources[0].source == "first"
    assert library.sources[0].columns == {"template_code": "a"}
    assert library.duplicate_count == 1


def test_reversing_the_order_reverses_which_source_keeps_the_rule(
    tmp_path: Path,
) -> None:
    shared = "[C:1]-[O:2]>>[C:1].[O:2]"
    first = tmp_path / "first.tsv"
    second = tmp_path / "second.tsv"
    write_table(first, (("a", shared),))
    write_table(second, (("c", shared),))

    library = bench_host_aostar.load_merged_library(
        (("second", second), ("first", first))
    )

    assert library.sources[0].columns == {"template_code": "c"}
    assert library.source_counts == {"second": 1}


def test_a_rule_id_inverts_back_to_its_source_row(tmp_path: Path) -> None:
    """The point of carrying sources: a route's rule index must name a source row again."""
    path = tmp_path / "rules.tsv"
    write_table(
        path,
        (
            ("keep", "[N:1]>>[N:1]"),
            ("mass-losing", "[C:4]-[O:3]-[C:2]=[O:1]>>[O:1]=[C:2]-[O:3]"),
            ("also-keep", "[S:1]>>[S:1]"),
        ),
    )

    library = bench_host_aostar.load_merged_library((("rules", path),))

    assert library.rejected_count == 1, "the middle row loses an atom"
    # Rule 1 after compiling is row 2 in the file, which is only knowable from `sources`.
    assert library.smarts[1] == "[S:1]>>[S:1]"
    assert library.sources[1].row == 2
    assert library.sources[1].columns == {"template_code": "also-keep"}


def test_merged_stock_is_a_union_and_the_limit_applies_to_it(tmp_path: Path) -> None:
    """Two catalogues overlapping is the normal case, and the limit still means molecules."""
    first = tmp_path / "a.smi"
    second = tmp_path / "b.smi"
    first.write_text("CCO\nCCC\n")
    second.write_text("CCC\nCCN\n")

    assert bench_host_aostar.read_merged_stock([first, second], None) == (
        "CCO",
        "CCC",
        "CCN",
    )
    assert bench_host_aostar.read_merged_stock([first, second], 2) == ("CCO", "CCC")


def test_stock_slices_partition_the_same_union_by_file(tmp_path: Path) -> None:
    """Per-file caching only pays off if the slices are exactly the merged union, split up."""
    first = tmp_path / "a.smi"
    second = tmp_path / "b.smi"
    first.write_text("CCO\nCCC\n")
    second.write_text("CCC\nCCN\n")

    slices = bench_host_aostar.stock_slices([first, second], None)
    assert slices == ((first, ("CCO", "CCC")), (second, ("CCN",)))

    # concatenating the slices reproduces the merged union, which is what makes the split safe
    flat = tuple(row for _, rows in slices for row in rows)
    assert flat == bench_host_aostar.read_merged_stock([first, second], None)

    # the limit still applies to the union, so a later file can be cut short or emptied
    assert bench_host_aostar.stock_slices([first, second], 2) == (
        (first, ("CCO", "CCC")),
        (second, ()),
    )


def test_appending_a_stock_file_leaves_the_earlier_slices_untouched(
    tmp_path: Path,
) -> None:
    """The whole point: fifty commodity reagents must not re-key 988k Enamine rows.

    A slice is the cache key, so an unchanged slice is a cache hit. Appending is safe;
    inserting ahead of another file is not, and that asymmetry is worth pinning.
    """
    first = tmp_path / "a.smi"
    second = tmp_path / "b.smi"
    first.write_text("CCO\nCCC\n")
    second.write_text("CCN\n")

    before = bench_host_aostar.stock_slices([first], None)
    after = bench_host_aostar.stock_slices([first, second], None)
    assert after[0] == before[0]

    # inserting ahead of it changes what the later file contributes, so its cache is cold
    third = tmp_path / "c.smi"
    third.write_text("CCC\n")
    inserted = bench_host_aostar.stock_slices([third, first], None)
    assert inserted[1] == (first, ("CCO",))


def test_merging_prepared_stocks_matches_preparing_the_merged_rows(
    tmp_path: Path,
) -> None:
    """The refactor must not change the catalogue, only when it is computed."""
    first = tmp_path / "a.smi"
    second = tmp_path / "b.smi"
    first.write_text("CCO\nCCC\n")
    second.write_text("CCN\n")

    paths = [first, second]
    merged = bench_host_aostar.cached_stock(
        bench_host_aostar.read_merged_stock(paths, None), 1
    )
    per_file = bench_host_aostar.merge_stocks(
        [
            bench_host_aostar.cached_stock(rows, 1)
            for _, rows in bench_host_aostar.stock_slices(paths, None)
            if rows
        ]
    )

    assert per_file.wl_keys == merged.wl_keys
    assert per_file.rdkit_keys == merged.rdkit_keys
    assert per_file.row_count == merged.row_count
    assert per_file.refusals == merged.refusals
    assert (per_file.wl_fingerprints == merged.wl_fingerprints).all()
    assert (per_file.rdkit_fingerprints == merged.rdkit_fingerprints).all()


def test_an_annotated_catalogue_shares_its_stripped_sibling_s_cache(tmp_path) -> None:
    """The property the annotated Enamine download rests on, asserted rather than hoped for.

    `prepare.sh` writes two files from one download: SMILES only, which is committed, and SMILES
    plus the Enamine catalogue id, which is gitignored because it is an order of magnitude larger.
    Searching against either must be the same search, and must not re-key a million rows.

    It is, because `open_smiles` keeps the first whitespace-delimited column and the prepared-stock
    cache is keyed on the resulting tuple rather than on a path. Nothing needed hashing file
    contents or tracking which file is whose sibling; the annotation never reaches the cache key.
    """
    import joblib

    molecules = ["CCO", "CCN", "c1ccccc1"]
    bare = tmp_path / "bare.smi"
    annotated = tmp_path / "annotated.smi"
    bare.write_text("\n".join(molecules) + "\n")
    annotated.write_text(
        "\n".join(
            f"{smiles}\tEN300-{index:05d}" for index, smiles in enumerate(molecules)
        )
        + "\n"
    )

    assert tuple(bench_host_aostar.open_smiles(bare)) == tuple(molecules)
    assert tuple(bench_host_aostar.open_smiles(annotated)) == tuple(molecules)

    ((_, bare_rows),) = bench_host_aostar.stock_slices([bare], None)
    ((_, annotated_rows),) = bench_host_aostar.stock_slices([annotated], None)

    assert bare_rows == annotated_rows
    assert joblib.hash(bare_rows) == joblib.hash(annotated_rows)

    # and only the annotated one has anything for a report to print
    assert not bench_host_aostar.annotated_catalogue(bare)
    assert bench_host_aostar.annotated_catalogue(annotated)


def test_every_backend_can_count_its_own_atoms() -> None:
    """The estimate was metal-only by accident, and being the default made that a crash.

    `AtomsFromStock` read `molecules.atom_counts` off the batch, which only the packed Metal batch
    carries. The cpu and rdkit arms offered the same `AOSTAR_HOST_HEURISTIC` value and raised
    `AttributeError: 'numpy.ndarray' object has no attribute 'atom_counts'` on the first scoring
    call -- latent while the default was `stock`, and a broken `make bench-aostar-host
    AOSTAR_HOST_BACKENDS=cpu` once it was not.

    Counting is now supplied by the arm, because the three spell it three different ways and
    guessing is what made this backend-specific in the first place.
    """
    from rdkit import Chem

    from vesma import cmol_from_smiles

    # ethanol 3, phenol 7, acetanilide 10 -- heavy atoms, which is what both arms must report
    smiles = ("CCO", "c1ccccc1O", "CC(=O)Nc1ccccc1")
    expected = [3.0, 7.0, 10.0]

    cmols = bench_host_aostar.object_mol_array([cmol_from_smiles(s)[0] for s in smiles])
    mols = bench_host_aostar.object_mol_array([Chem.MolFromSmiles(s) for s in smiles])

    assert list(bench_host_aostar.cmol_atom_counts(cmols)) == expected
    assert list(bench_host_aostar.rdkit_atom_counts(mols)) == expected
    # and the two agree with each other, which is what makes the arms comparable at all
    assert list(bench_host_aostar.cmol_atom_counts(cmols)) == list(
        bench_host_aostar.rdkit_atom_counts(mols)
    )


def test_the_estimate_refuses_a_backend_that_cannot_count() -> None:
    """Better than reading an attribute off whatever arrives and failing deep in a search."""
    with pytest.raises(ValueError, match="count a batch's atoms"):
        bench_host_aostar.heuristic_of("atoms", lambda: None)

    # and the old spelling still resolves, since recorded manifests carry it
    assert isinstance(
        bench_host_aostar.heuristic_of(
            "unexplained", lambda: None, bench_host_aostar.cmol_atom_counts
        ),
        bench_host_aostar.AtomsFromStock,
    )


# --------------------------------------------------------------------------- #
# The run directory
# --------------------------------------------------------------------------- #


def test_rule_ids_are_collected_through_the_whole_route_tree() -> None:
    """A nested tree, not just its root: naming needs every step a route used."""
    records = [
        {
            "template_id": 1,
            "precursors": [
                {"template_id": 2, "precursors": []},
                {"template_id": None, "precursors": []},
            ],
        },
        {"template_id": 2, "precursors": []},
    ]

    assert bench_host_aostar.archived_rule_ids(records) == frozenset({1, 2})
    assert bench_host_aostar.archived_rule_ids([]) == frozenset()


def test_template_rows_name_only_the_rules_the_routes_used(tmp_path: Path) -> None:
    """Restricted on purpose: the full library is rebuildable from the manifest's hashes."""
    path = tmp_path / "rules.tsv"
    write_table(path, (("a", "[N:1]>>[N:1]"), ("b", "[S:1]>>[S:1]")))
    library = bench_host_aostar.load_merged_library((("rules", path),))

    rows = bench_host_aostar.template_rows(library, {1})

    assert rows == [
        {
            "rule_id": 1,
            "retro_template": "[S:1]>>[S:1]",
            "source": "rules",
            "row": 1,
            "columns": {"template_code": "b"},
        }
    ]


def test_the_run_directory_holds_what_a_report_needs(tmp_path: Path) -> None:
    """manifest.json and templates.json beside the records the run already appended."""
    path = tmp_path / "rules.tsv"
    write_table(path, (("a", "[N:1]>>[N:1]"), ("b", "[S:1]>>[S:1]")))
    library = bench_host_aostar.load_merged_library((("rules", path),))
    result = bench_host_aostar.TargetResult(
        backend="rdkit",
        target="CCO",
        solved=True,
        seconds=1.0,
        width=8,
        pass_limit=4,
        passes=1,
        molecules_expanded=1,
        molecules_seen=1,
        routes=1,
        unique_route_count=1,
        unique_route_count_capped=False,
        best_reaction_count=1,
        heuristic_queries=0,
        heuristic_calls=0,
        phase_seconds={},
        matcher_totals={},
        route_records=({"template_id": 0, "precursors": []},),
    )
    run_dir = tmp_path / "run"

    bench_host_aostar.write_run_directory(
        run_dir, manifest={"rule_sets": ["rules"]}, library=library, results=(result,)
    )

    assert json.loads((run_dir / "manifest.json").read_text()) == {
        "rule_sets": ["rules"]
    }
    templates = json.loads((run_dir / "templates.json").read_text())
    assert [row["rule_id"] for row in templates] == [0], "only the rule the route used"
    assert templates[0]["columns"] == {"template_code": "a"}


def test_a_run_that_solved_nothing_still_writes_its_manifest(tmp_path: Path) -> None:
    """An unsolved sweep is a result, and its inputs are what a reader needs most."""
    path = tmp_path / "rules.tsv"
    write_table(path, (("a", "[N:1]>>[N:1]"),))
    library = bench_host_aostar.load_merged_library((("rules", path),))
    run_dir = tmp_path / "run"

    bench_host_aostar.write_run_directory(
        run_dir, manifest={"rule_sets": ["rules"]}, library=library, results=()
    )

    assert json.loads((run_dir / "templates.json").read_text()) == []


def test_one_unreadable_stock_file_does_not_abort_the_run(tmp_path: Path) -> None:
    """Per-file preparation must not turn a harmless loss into an aborted benchmark.

    ``prepare_stock`` refuses a catalogue no row of which both toolkits can read. Before the
    stock was split per file that could only happen to the whole union; afterwards a single
    four-row supplement of unreadable rows would take the run down with it.
    """
    good = tmp_path / "good.smi"
    bad = tmp_path / "bad.smi"
    good.write_text("CCO\nCCC\n")
    bad.write_text("not-a-molecule\nalso-not-one\n")

    slices = bench_host_aostar.stock_slices([good, bad], None)
    parts = bench_host_aostar.prepare_labelled_slices(slices, 1)
    assert len(parts) == 1

    merged = bench_host_aostar.merge_stocks([part for _, part in parts])
    assert len(merged.wl_keys) == 2

    # the union still has to be non-empty, so a run against nothing readable is refused
    only_bad = bench_host_aostar.stock_slices([bad], None)
    with pytest.raises(ValueError, match="no stock row"):
        bench_host_aostar.merge_stocks(
            [part for _, part in bench_host_aostar.prepare_labelled_slices(only_bad, 1)]
        )


# ------------------------------------------------------- the paired selectivity re-ranking


def selection(*routes: tuple[SidRoute, int]) -> RouteSelection:
    """A RouteSelection carrying only what `ranking_change` reads: the routes and their scores."""
    return RouteSelection(
        routes=tuple(
            SelectedRoute(
                route=route,
                score=RouteScore(
                    depth=reactions,
                    reactions=reactions,
                    utility=0.9**reactions,
                    base_step_factor=0.9,
                    stock_factor_product=1.0,
                    reaction_factor_product=1.0,
                ),
                first_distinct_decision=None if index == 0 else index,
            )
            for index, (route, reactions) in enumerate(routes)
        ),
        candidates_considered=len(routes),
        partial_states_explored=len(routes),
        truncated=False,
        shortest_depth=min((r for _, r in routes), default=None),
        best_shortest_reactions=min((r for _, r in routes), default=None),
    )


def named_expansions(names: Sequence[str]) -> Expansions:
    """An Expansions whose interner maps sid `i` to `names[i]`, so decoding is readable."""
    expansions = Expansions(NumpyExpansionStore(), DictInterner())
    expansions.intern(list(names))
    return expansions


DIRECT = SidRoute(0, 1, (SidRoute(2, None),))
DETOUR = SidRoute(0, 3, (SidRoute(1, 4, (SidRoute(2, None),)),))


def test_an_unchanged_report_is_reported_as_unchanged() -> None:
    """The inert case is the one a reader must be able to trust: no change means no change.

    A run whose penalty moved nothing has measured that, which is a result and not a gap -- so it
    has to be distinguishable from a run that never compared.
    """
    expansions = named_expansions(["target", "protected", "stock"])
    one = selection((DIRECT, 1))
    change = ranking_change(one, one, expansions, ("", "rule one", "", "", ""))
    assert change["top_changed"] is False
    assert change["order_changed"] is False
    assert change["set_changed"] is False
    assert change["reactions_before"] == change["reactions_after"] == 1
    # No trees, because nothing changed and a payload proportional to what happened is the point.
    assert "tree_before" not in change


def test_a_changed_top_route_carries_both_trees_for_review() -> None:
    """The deliverable of the paired run: a before/after pair naming the templates involved."""
    expansions = named_expansions(["target", "protected", "stock"])
    smarts = ("", "direct", "", "deprotect", "protect")
    change = ranking_change(
        selection((DETOUR, 2), (DIRECT, 1)),
        selection((DIRECT, 1), (DETOUR, 2)),
        expansions,
        smarts,
    )
    assert change["top_changed"] is True
    assert change["order_changed"] is True
    # The same two routes in the other order, so the *set* is untouched -- which is the flag that
    # separates a re-ranking from a route the price newly admitted or newly excluded.
    assert change["set_changed"] is False
    assert (change["reactions_before"], change["reactions_after"]) == (1, 2)
    assert change["utility_before"] > change["utility_after"]
    assert "direct" in change["tree_before"] and "deprotect" in change["tree_after"]


def test_a_reordering_below_the_top_is_reported_without_trees() -> None:
    """Order without a top change is worth counting and not worth printing a tree for."""
    expansions = named_expansions(["target", "protected", "stock"])
    other = SidRoute(0, 2, (SidRoute(2, None),))
    change = ranking_change(
        selection((DIRECT, 1), (other, 1), (DETOUR, 2)),
        selection((DIRECT, 1), (DETOUR, 2), (other, 1)),
        expansions,
        ("", "a", "b", "c", "d"),
    )
    assert (change["top_changed"], change["order_changed"]) == (False, True)
    assert "tree_before" not in change


def test_a_route_the_price_newly_admits_shows_as_a_set_change() -> None:
    """`set_changed` is the one flag `order_changed` cannot express, so it is its own field."""
    expansions = named_expansions(["target", "protected", "stock"])
    change = ranking_change(
        selection((DETOUR, 2)),
        selection((DIRECT, 1)),
        expansions,
        ("", "a", "b", "c", "d"),
    )
    assert change["set_changed"] is True


def test_a_co_canonical_relabelling_is_a_ranking_change_but_not_a_chemistry_change() -> (
    None
):
    """The interesting case, and the reason two flags exist rather than one.

    One disconnection reached by two templates is what the per-template price is *for*: charging at
    the selective template is the mechanism working, not a different route. Reported as a changed
    top so the labelling move is visible, and as unchanged chemistry so nobody reads it as a
    reordered synthesis.
    """
    expansions = named_expansions(["target", "protected", "stock"])
    relabelled = SidRoute(0, 2, (SidRoute(2, None),))
    change = ranking_change(
        selection((relabelled, 1)),
        selection((DIRECT, 1)),
        expansions,
        ("", "selective variant", "unselective variant", "", ""),
    )
    assert change["top_changed"] is True
    assert change["chemistry_changed"] is False


def test_a_different_route_changes_the_chemistry_too() -> None:
    expansions = named_expansions(["target", "protected", "stock"])
    change = ranking_change(
        selection((DETOUR, 2)),
        selection((DIRECT, 1)),
        expansions,
        ("", "a", "b", "c", "d"),
    )
    assert change["top_changed"] is True
    assert change["chemistry_changed"] is True


# ------------------------------------------- the veto / competitor cross-tabulation


def _veto(code: str, scope: str, pattern: str) -> Veto:
    return Veto(
        code=code,
        cost="pattern",
        scope=frozenset(entry for entry in scope.split(";") if entry),
        on="precursors",
        force="refuse",
        sense="forbid",
        pattern=pattern,
        rationale="a test row",
    )


def _overlap_layer(competitors: Sequence[int]) -> tuple[dict, list]:
    """One layer of three applications, each on its own rule, with a competitor count supplied.

    Precursor 0 carries an ester and precursor 1 a boronic acid, so the two rows below condemn one
    application each -- which is what makes mis-attribution visible: a screen handed the wrong
    pattern index would credit both to whichever row it compiled first.
    """
    from vesma.cmol import cmol_union

    rules = ("MC 86", "MC 84a", "Hartenfeller 99")
    table = (
        _veto("B 2", "MC 86", "[CX3](=[OX1])-[OX2]-[CX4]"),
        _veto("B 1", "MC 84a", "[B]"),
    )
    layer = bench_host_aostar.VetoLayer(
        table=table,
        in_scope=vetoes_in_scope(table, rules),
        refuse=False,
    )
    precursors = [
        cmol_from_smiles(text)[0] for text in ("COC(=O)CC", "OB(O)c1ccccc1", "CCO")
    ]
    hits = np.array([(index, index, 0, index) for index in range(3)], dtype=np.int64)

    def match(mols):
        return object_mol_array(precursors), ["a", "b", "c"], hits

    overlap_reports: list = []
    built = bench_host_aostar.layered_match(
        expansion(match),
        probes={
            "competitors": lambda products, precursors, hits: np.asarray(
                competitors, dtype=np.int64
            )
        },
        vetoes=layer,
        veto_reports=[],
        overlap_reports=overlap_reports,
        make_screen=host_screen,
        combine=object_combine(cmol_union),
    )
    found = built(object_mol_array([cmol_from_smiles("CCC")[0]] * 3))
    return bench_host_aostar.overlap_tally(overlap_reports), found


def test_each_veto_row_is_credited_with_its_own_condemnations() -> None:
    """A per-row column needs a per-row screen, or every row reports row zero's pattern.

    The one bug this cross-tabulation can have and still look plausible: a screen closes over its
    compiled patterns and answers in its own numbering, so a column over one row handed the whole
    table's screen evaluates the wrong pattern. Here B 2 must claim the ester and B 1 the boron.
    """
    tally, _ = _overlap_layer([1, 0, 0])

    assert tally["applications"] == 3
    competitors = {name: cells["competitors"] for name, cells in tally["rows"].items()}
    assert competitors["B 2"] == {"unprobed": 0, "silent": 0, "one": 1, "many": 0}
    assert competitors["B 1"] == {"unprobed": 0, "silent": 1, "one": 0, "many": 0}
    # The whole table's column is the disjunction of the rows, so it sees both.
    assert competitors["any"] == {"unprobed": 0, "silent": 1, "one": 1, "many": 0}


def test_a_channel_that_never_ran_is_not_a_channel_that_found_nothing() -> None:
    """`UNPROBED` has to survive the join, or an unpaired rule reads as a clean one."""
    tally, _ = _overlap_layer([UNPROBED, UNPROBED, 0])

    zero = {"unprobed": 1, "silent": 0, "one": 0, "many": 0}
    assert tally["rows"]["B 2"]["competitors"] == zero
    assert tally["rows"]["B 1"]["competitors"] == zero


def test_cross_tabulating_changes_no_hit_and_no_column() -> None:
    """An observer observes. It sits inside the pipeline, so this is worth asserting."""
    _, found = _overlap_layer([1, 0, 0])

    assert found.hits.shape == (3, 4)
    assert found.reaction_scores["competitors"].tolist() == [1, 0, 0]
    assert found.reaction_scores["vetoed"].tolist() == [1, 1, CLEAR]


def test_the_cross_tabulation_carries_its_own_base_rate() -> None:
    """Without the denominator the numerators mean nothing, so it travels with them.

    A row whose condemnations carry a channel's signal 70% of the time has found nothing at all if
    the channel says 70% about every application. The first reading of the competitor channel in
    this repository turned on exactly that, which is why `(all)` is a row rather than a lookup.
    """
    tally, _ = _overlap_layer([1, 0, 0])

    base = tally["rows"]["(all)"]["competitors"]
    assert sum(base.values()) == tally["applications"] == 3
    assert base == {"unprobed": 0, "silent": 2, "one": 1, "many": 0}


def a_result(**overrides: Any) -> bench_host_aostar.TargetResult:
    """A `TargetResult` with every required field filled, for the writers to carry around."""
    fields: dict[str, Any] = {
        "backend": "rdkit",
        "target": "CCO",
        "solved": True,
        "seconds": 1.0,
        "width": 8,
        "pass_limit": 4,
        "passes": 1,
        "molecules_expanded": 1,
        "molecules_seen": 1,
        "routes": 1,
        "unique_route_count": 1,
        "unique_route_count_capped": False,
        "best_reaction_count": 1,
        "heuristic_queries": 0,
        "heuristic_calls": 0,
        "phase_seconds": {},
        "matcher_totals": {},
    }
    return bench_host_aostar.TargetResult(**(fields | overrides))


def one_step_report(
    *,
    standing: str,
    rule_label: str,
    vetoes: tuple[str, ...],
    sites: int | None,
    competitors: int | None = None,
) -> dict[str, Any]:
    """One target's alternatives report, hand-built to the shape the module emits."""
    option = {
        "rule": 0,
        "rule_label": rule_label,
        "precursors": ["CCO"],
        "also_by": [],
        "log_cost": 0.1,
        "cost_delta": 0.0,
        "measurements": {"sites": sites, "competitors": competitors},
        "vetoes_fired": list(vetoes),
        "standing": standing,
        "also_reported_in": [],
    }
    return {
        "target": "CCOC",
        "routes": [
            {
                "rank": 1,
                "standing": "best",
                "score": {},
                "first_distinct_decision": None,
                "differing_decisions": None,
                "resembled": None,
                "retro_steps": [
                    {
                        "index": 1,
                        "product": "CCOC",
                        "chosen": option,
                        "alternatives": [],
                    }
                ],
            }
        ],
        "rejected": [],
        "truncated": False,
        "stopped_after_limit": False,
        "channels": ["sites", "competitors"],
        "diversity_cutoff": None,
    }


def test_the_alternatives_report_is_written_beside_the_records_not_into_them(
    tmp_path: Path,
) -> None:
    """One per-disconnection report is larger than every timing in the sweep put together.

    `targets.jsonl` is the file read with `jq` to answer "how long did target 19 take"; burying a
    few hundred alternatives in each line would make that unreadable for no gain.
    """
    report = one_step_report(standing="chosen", rule_label="MC 62", vetoes=(), sites=1)
    result = a_result(alternatives=report)

    bench_host_aostar.append_record(tmp_path / "targets.jsonl", result, "rdkit 8x4")
    bench_host_aostar.append_alternatives(
        tmp_path / "alternatives.jsonl", result, "rdkit 8x4"
    )

    record = json.loads((tmp_path / "targets.jsonl").read_text())
    assert "alternatives" not in record
    assert record["target"] == "CCO", "the timings themselves are untouched"
    written = json.loads((tmp_path / "alternatives.jsonl").read_text())
    assert written["label"] == "rdkit 8x4"
    assert written["routes"][0]["retro_steps"][0]["chosen"]["rule_label"] == "MC 62"


def test_a_target_with_no_routes_writes_no_alternatives_line(tmp_path: Path) -> None:
    """An unsolved target has nothing to explain, and an empty line would read as a lost report."""
    bench_host_aostar.append_alternatives(
        tmp_path / "alternatives.jsonl", a_result(solved=False), "rdkit 8x4"
    )

    assert not (tmp_path / "alternatives.jsonl").exists()


def test_the_summary_names_the_template_and_veto_row_that_disagree(
    tmp_path: Path,
) -> None:
    """The worklist the whole report exists to produce.

    A veto row fired on a step the report *recommended*, while the `sites` probe called that same
    step one clean product. The veto rows were written by hand against named failures, so the
    disagreement is a forward-pattern gap -- and the pair named here is the pair a curator opens.
    """
    results = (
        a_result(
            alternatives=one_step_report(
                standing="chosen", rule_label="MC 62", vetoes=("B 30",), sites=1
            )
        ),
        a_result(
            alternatives=one_step_report(
                standing="vetoed", rule_label="MC 10", vetoes=("B 30",), sites=1
            )
        ),
    )

    summary = bench_host_aostar.alternatives_summary(results)

    assert summary["targets"] == 2
    assert summary["believed_clean"] == {"MC 62 / B 30": {"steps": 1, "silent": 1}}
    assert summary["vetoes_fired"] == {"B 30": 2}
    # The rejected alternative fired the same row, and is not a gap.
    assert summary["vetoes_fired_on_a_chosen_step"] == {"B 30": 1}
    assert summary["standings"]["chosen"] == 1
    assert summary["standings"]["route:best"] == 2
    assert summary["refused_at_validation"] == {}, "nothing was refused in this run"


def test_a_measured_mixture_is_not_a_curation_gap() -> None:
    """The forward table saw the mixture, so the veto and the probe agree and nothing is missing."""
    results = (
        a_result(
            alternatives=one_step_report(
                standing="chosen", rule_label="MC 62", vetoes=("B 30",), sites=3
            )
        ),
    )

    assert bench_host_aostar.alternatives_summary(results)["believed_clean"] == {}


def test_a_competitor_does_not_excuse_a_clean_sites_verdict() -> None:
    """The correction this came back wrong on, kept as a test.

    Requiring both channels silent found nothing on a run where five recommended steps fired a
    veto row -- all five reported one clean site and every one of them had competitors. The
    channels answer different questions, and only `sites` can contradict a veto row that says the
    intended reaction destroys its substrate. The other channel's verdict is recorded beside it,
    because a row the competitor channel also catches is a row that could be priced instead.
    """
    results = (
        a_result(
            alternatives=one_step_report(
                standing="chosen",
                rule_label="Hartenfeller 45a",
                vetoes=("B 30",),
                sites=1,
                competitors=4,
            )
        ),
    )

    summary = bench_host_aostar.alternatives_summary(results)

    assert summary["believed_clean"] == {
        "Hartenfeller 45a / B 30": {"steps": 1, "competed": 1}
    }


def test_replay_names_a_molecule_and_an_unreplayed_key_stays_a_key() -> None:
    """Only replay produces SMILES, and only along a route it reproduced.

    Everything else in the graph is a search key, and under the Metal arm that is a WL hash. A
    digest says so plainly; printing the key would invite reading a hash as a structure.
    """
    route = Route("CCOC", 0, (Route("CCO", None), Route("CO", None)))
    names = bench_host_aostar.replayed_names(
        [route], key_of=lambda smiles: f"k:{smiles}"
    )
    named = bench_host_aostar.molecule_name(names)

    assert named("k:CCO") == "CCO"
    assert named(("wl", 7)).startswith("#")
    # Stable across calls, so two reports of the same graph can be compared.
    assert named(("wl", 7)) == named(("wl", 7))
    assert named(("wl", 8)) != named(("wl", 7))


def test_the_summary_counts_the_routes_that_were_not_reported() -> None:
    """A report shorter than it was asked for has to say which of the two reasons applied."""
    report = one_step_report(standing="chosen", rule_label="MC 62", vetoes=(), sites=1)
    report["rejected"] = [
        {
            "rank": 0,
            "standing": "censored",
            "score": {},
            "first_distinct_decision": None,
            "differing_decisions": None,
            "resembled": None,
            "retro_steps": [],
            "reason": "rule 338: replay emitted 0 application(s)",
        },
        {
            "rank": 0,
            "standing": "too-similar",
            "score": {},
            "first_distinct_decision": 3,
            "differing_decisions": None,
            "resembled": 1,
            "retro_steps": [],
            "reason": None,
        },
    ]

    summary = bench_host_aostar.alternatives_summary((a_result(alternatives=report),))

    assert summary["standings"]["route:censored"] == 1
    assert summary["standings"]["route:too-similar"] == 1
    assert summary["refused_at_validation"] == {
        "rule 338: replay emitted 0 application(s)": 1
    }


def test_the_graded_spelling_parses_with_and_without_a_cap() -> None:
    """The tunable middle between a threshold and a step count, as a run can ask for it."""
    sloped = bench_host_aostar.multiplier_of("graded:0.1")
    capped = bench_host_aostar.multiplier_of("graded:0.5:2.0")

    assert [sloped(n) for n in (1, 2)] == [1.0, pytest.approx(1.1)]
    assert sloped(7) == pytest.approx(1.6)
    assert capped(3) == capped(100) == 2.0
    with pytest.raises(ValueError, match="graded:<slope>"):
        bench_host_aostar.multiplier_of("linear:0.1")


def test_a_sweep_entry_can_price_the_two_channels_differently() -> None:
    """The distinction a single-shape sweep could not test.

    `sites` counts the products the intended reagent makes, which is a product count and what
    `steps` was derived for; `competitors` counts rival reactions, which is a warning. An entry
    with no slash still means both channels, so the short spelling keeps working.
    """
    both = bench_host_aostar.penalties_of("graded:0.1", "graded:0.1")
    split = bench_host_aostar.penalties_of("steps", "graded:0.1")
    off = bench_host_aostar.penalties_of("steps", "none")

    assert both is not None and split is not None and off is not None
    assert both["sites"](7) == pytest.approx(1.6)
    assert split["sites"](7) == 8.0
    assert split["competitors"](7) == pytest.approx(1.6)
    assert "competitors" not in off


def test_the_price_sweep_summary_is_comparable_by_construction() -> None:
    """Every shape saw the same graphs, so the rows add up across targets without weighting.

    `vetoes_on_the_top_route` is the column worth reading: the veto rows are hand-written
    chemistry that neither selectivity channel reads, so they are the one criterion here that is
    not produced by the mechanism being tuned.
    """

    def entry(top: dict[str, int], changed: bool) -> dict[str, Any]:
        return {
            "top_changed": changed,
            "chemistry_changed": changed,
            "vetoes_on_the_top_route": top,
            "vetoes_on_any_reported_route": top,
            # One step, however many rows name it.
            "condemned_steps_on_the_top_route": 1 if top else 0,
            "condemned_steps_on_any_reported_route": 1 if top else 0,
            "routes": 3,
        }

    results = (
        a_result(
            price_sweep={
                "flat:1.2": entry({"B 30": 1}, True),
                "graded:0.1": entry({}, True),
            }
        ),
        a_result(
            price_sweep={
                "flat:1.2": entry({"B 30": 1, "B 31": 1}, False),
                "graded:0.1": entry({"B 31": 1}, True),
            }
        ),
    )

    summary = bench_host_aostar.price_sweep_summary(results)

    assert summary["flat:1.2"]["targets"] == summary["graded:0.1"]["targets"] == 2
    assert summary["flat:1.2"]["vetoes_on_the_top_route"] == 3
    assert summary["graded:0.1"]["vetoes_on_the_top_route"] == 1
    # Two steps against three codes: one of them named two rows, and a step is a step.
    assert summary["flat:1.2"]["condemned_steps_on_the_top_route"] == 2
    assert summary["graded:0.1"]["condemned_steps_on_the_top_route"] == 1
    assert summary["flat:1.2"]["by_row"] == {"B 30": 2, "B 31": 1}
    assert summary["flat:1.2"]["top_changed"] == 1
    assert summary["graded:0.1"]["top_changed"] == 2


def test_a_run_with_no_sweep_writes_no_sweep_file(tmp_path: Path) -> None:
    """The knob is off by default, and an empty file would read as a sweep that found nothing."""
    path = tmp_path / "rules.tsv"
    write_table(path, (("a", "[N:1]>>[N:1]"),))
    library = bench_host_aostar.load_merged_library((("rules", path),))
    run_dir = tmp_path / "run"

    bench_host_aostar.write_run_directory(
        run_dir, manifest={}, library=library, results=(a_result(),)
    )

    assert not (run_dir / "price-sweep-summary.json").exists()


def test_the_destructive_channel_keeps_exactly_the_destructive_rows() -> None:
    """`_competes_for` had no test at all, which is how an index confusion survived in its sibling.

    Asserted on `candidate` ids specifically. A `Competes` is handed `(intended, candidate)` where
    `intended` is a **retro** rule id and `candidate` a **forward** one -- `conditions_allow` says
    so and reads `requires[candidate]` accordingly -- and the severity lookup has to index the same
    way, because severity is a property of the candidate's own forward row. Getting that backwards
    is not hypothetical: `same_site` is built over the forward patterns and called with a retro
    `intended`, and its out-of-range guard returns `True`, so every retro rule beyond the forward
    table reads as same-site.
    """
    from tests.vertsy.bench_host_aostar import (
        TEMPLATE_FILES,
        _competes_for,
        load_forward_library,
        load_template_library,
    )

    _count, _smarts, sources = load_template_library(
        "medchem", TEMPLATE_FILES["medchem"]
    )
    forward = load_forward_library(sources)
    destructive = {
        rule for rule, band in enumerate(forward.severity) if band == "destructive"
    }
    assert destructive, "the table carries destructive rows"

    competes = _competes_for("competitors:destructive", forward)
    assert competes is not None

    # A pot whose intended reaction supplies a halogen source, which is what most of the
    # destructive family needs. Picked by what it provides rather than by code, so the test does
    # not break when the tables are recurated.
    halogenating = next(
        rule
        for rule, provides in sorted(forward.conditions.items())
        if "halogenating" in provides
    )
    kept = {
        candidate
        for candidate in range(len(forward.severity))
        if competes(halogenating, candidate)
    }

    # Exactly the destructive rows the conditions gate also admits -- and nothing else, which is
    # the half that matters: a competing row kept here would be priced as a destroyed substrate.
    admitted = {
        candidate
        for candidate in destructive
        if forward.requires[candidate] <= forward.conditions[halogenating]
    }
    assert kept == admitted
    assert kept, "a vacuous filter would pass the subset assertion above"
    assert kept <= destructive


def test_an_unknown_competitor_channel_is_refused() -> None:
    """The name is the predicate, so a misspelt channel must not silently become the crude one."""
    import pytest as _pytest

    from tests.vertsy.bench_host_aostar import (
        TEMPLATE_FILES,
        _competes_for,
        load_forward_library,
        load_template_library,
    )

    _count, _smarts, sources = load_template_library(
        "medchem", TEMPLATE_FILES["medchem"]
    )
    forward = load_forward_library(sources)

    with _pytest.raises(ValueError, match="competitors:destructive"):
        _competes_for("competitors:destuctive", forward)


def test_the_severity_band_is_priced_and_clean_stays_free() -> None:
    """The two properties a curve on this channel has to have, and where its numbers come from.

    `clean_log_cost` takes `charge(rule, (1,) * channels)` as the price's free reference, so a
    channel storing `1` for "nothing to worry about" costs nothing there -- which is why the
    severity column starts at one. A count channel cannot do this: `drop_intended` makes `0` clean
    there, so its `1` means *one competitor* and would be free.

    The shares are not a knob. They are the survivals of `y / (y + e)` over the four-band yield
    vocabulary both inputs come from, grouped into four bands -- four for the reason
    `typical_yield` has four, since thirteen distinct ratios from two class-level priors would
    claim a resolution nobody measured.
    """
    import math

    from tests.vertsy.bench_host_aostar import (
        SEVERITY_EDGES,
        SEVERITY_SURVIVALS,
        factors_of,
    )
    from vertsy.costs import by_rule, compounded, cost_of_factor, factor_of_cost

    # Band 1 is clean and keeps everything; every later band keeps strictly less.
    assert SEVERITY_SURVIVALS[0] == 1.0
    assert list(SEVERITY_SURVIVALS) == sorted(SEVERITY_SURVIVALS, reverse=True)
    # One survival per band, and one fewer edge than bands.
    assert len(SEVERITY_EDGES) == len(SEVERITY_SURVIVALS) - 2
    # Each edge admits at least one ratio the yield vocabulary can actually produce, so no band is
    # unreachable. The vocabulary is the four `typical_yield` values.
    ratios = {
        y / (y + e) for y in (0.90, 0.80, 0.65, 0.50) for e in (0.90, 0.80, 0.65, 0.50)
    }
    bands = {1 + sum(1 for edge in SEVERITY_EDGES if share < edge) for share in ratios}
    # Every band is reachable; an unreachable one would be a share nothing can ever charge.
    assert bands == set(range(1, len(SEVERITY_SURVIVALS)))

    factors = factors_of("none", "none", severity=True)
    assert factors is not None and list(factors) == ["competitors:severity"]
    assert factors_of("none", "none", severity=False) is None

    price = compounded(
        by_rule(lambda rule: cost_of_factor(0.80), uniform=False),
        {"competitors:severity": {}},
        factors,
    )
    clean = price.charge(0, (1,))
    # Clean is exactly the rule's own base: the channel adds nothing.
    assert factor_of_cost(clean) == pytest.approx(0.80)

    # And each band multiplies it by that band's survival, which is what `compounded` promises the
    # report's printed selectivity will be. Monotone, and the remainder never goes negative.
    previous = clean
    for band, share in enumerate(SEVERITY_SURVIVALS[1:], start=2):
        charged = price.charge(0, (band,))
        assert factor_of_cost(charged) == pytest.approx(0.80 * share)
        assert charged - clean == pytest.approx(-math.log(share))
        assert charged >= previous
        previous = charged

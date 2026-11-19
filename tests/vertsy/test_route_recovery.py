"""Fast correctness gate for the route-recovery benchmark.

The full benchmark (`bench_route_recovery.py`) parses the whole ~56k-template library and
can take many minutes on a slow host. This test instead builds a tiny library from just the
first target's own reference reactions plus a few decoys, so the same search/extraction/
comparison pipeline runs in well under a second while still proving that rule ids stay aligned
between the matcher and the reference route — i.e. that the reference route is actually
recovered.
"""

from __future__ import annotations

import pytest

from tests.vertsy.bench_route_recovery import (
    MAX_DEPTH,
    PAROUTES_DIR,
    ROUTE_TEMPLATES_FILE,
    build_recovery_backend,
    build_references,
    load_reference_nodes,
    load_target_smiles,
    recovery_extra_info,
    run_recovery,
)
from tests.vertsy.paroutes_reference import (
    clean_reaction_smarts,
    load_route_template_rows,
    raw68k_rule_index,
    route_recovery_rule_index_from_rows,
)
from vertsy.interop import aizynth_reaction_metadata


def test_route_recovery_recovers_reference_route() -> None:
    target_smiles = load_target_smiles(1)
    node = load_reference_nodes(target_smiles)[0]

    needed = [str(md["reaction_hash"]) for md in aizynth_reaction_metadata(node)]
    assert needed, "First n1-d2-f0 reference route has no reactions."

    rows = load_route_template_rows(PAROUTES_DIR / ROUTE_TEMPLATES_FILE)
    missing = [reaction_hash for reaction_hash in needed if reaction_hash not in rows]
    if missing:
        pytest.skip(f"Reference reactions absent from committed library: {missing}")

    decoys = [reaction_hash for reaction_hash in rows if reaction_hash not in needed][
        :3
    ]
    tiny_rows = {h: rows[h] for h in (*needed, *decoys)}
    rule_index = route_recovery_rule_index_from_rows(tiny_rows)
    assert rule_index.smarts, "Tiny route-recovery library is empty."

    backend = build_recovery_backend("rdkit", rule_index)
    references = build_references([node], rule_index, backend.codec, rule_aligned=True)
    result = run_recovery(backend, [node["smiles"]], references)
    target = result.targets[0]

    # n1-d2 reference routes are two reactions deep, so the search runs two layers.
    assert len(target.layers) == MAX_DEPTH
    assert target.layers[0].frontier_size == 1
    assert target.layers[0].pairs == 1 * len(rule_index.smarts)
    assert all(layer.applications >= 1 for layer in target.layers)
    assert all(
        layer.step_seconds >= layer.match_split_seconds for layer in target.layers
    )

    # The reference route's reactions are in the library, so it must be recovered, and the
    # comparators nest: strict matches imply loose imply library.
    assert target.solved_route_count_depth_bounded_capped >= 1
    assert target.min_route_depth == MAX_DEPTH
    assert result.strict >= 1
    assert result.loose >= result.strict
    assert result.library >= result.loose
    # Reaction-agnostic comparators recover the reference too.
    assert target.any_solved == 1
    assert target.starting_materials_recovered == 1
    assert target.precursors_recovered == 1

    extra = recovery_extra_info(
        result,
        backend="rdkit",
        rule_set="used14k",
        template_source="used14k-reactions",
        rules=len(rule_index.smarts),
        stock_size=len(backend.stock),
        compile_seconds=backend.compile_seconds,
        precursor_delivery=backend.precursor_delivery,
    )
    assert extra["reaction_comparators_feasible"] is True
    assert extra["comparator_results_per_target"] == [
        [target.strict, target.loose, target.library]
    ]
    assert extra["targets"][0]["layers"][0]["frontier_size"] == 1
    assert "match_split_seconds" in extra["targets"][0]["layers"][0]
    assert "step_seconds" in extra["targets"][0]["layers"][0]


def test_route_recovery_rule_index_deduplicates_smarts() -> None:
    rows = load_route_template_rows(PAROUTES_DIR / ROUTE_TEMPLATES_FILE)
    sample_hash, sample = next(iter(rows.items()))
    duplicate_hash = f"{sample_hash}#duplicate"
    tiny_rows = {
        sample_hash: sample,
        duplicate_hash: {"retro_template": sample["retro_template"]},
    }

    rule_index = route_recovery_rule_index_from_rows(tiny_rows)

    assert len(rule_index.rule_id_by_reaction_hash) == len(tiny_rows)
    assert len(rule_index.smarts) == len(
        {clean_reaction_smarts(row["retro_template"]) for row in tiny_rows.values()}
    )
    assert (
        rule_index.rule_id_by_reaction_hash[sample_hash]
        == rule_index.rule_id_by_reaction_hash[duplicate_hash]
    )


@pytest.mark.slow
def test_raw68k_aligns_every_reference_reaction() -> None:
    rule_index = raw68k_rule_index(
        PAROUTES_DIR / "raw68k-templates.tsv.gz",
        PAROUTES_DIR / ROUTE_TEMPLATES_FILE,
    )

    assert len(rule_index.smarts) == 67_557
    assert len(rule_index.rule_id_by_reaction_hash) == 56_657
    assert len(set(rule_index.rule_id_by_reaction_hash.values())) == 13_978

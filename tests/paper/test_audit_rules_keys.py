from __future__ import annotations

from tests.vesma.bench_audit_rules_keys import (
    summarize_identity_records,
    summarize_template_source,
)


def test_summarize_template_source_groups_rejections() -> None:
    smarts = ("A", "B", "C")
    outcomes = {
        "A": ("accepted", ""),
        "B": ("UnsupportedSmartsFeatureError", "unsupported B"),
        "C": ("UnsupportedSmartsFeatureError", "unsupported C"),
    }
    result = summarize_template_source(
        "example",
        5,
        smarts,
        outcomes,
        {"path": "example.tsv"},
    )
    assert result["source_rows"] == 5
    assert result["unique_templates"] == 3
    assert result["accepted_templates"] == 1
    assert result["rejected_templates"] == 2
    assert result["rejections_by_category"] == {"UnsupportedSmartsFeatureError": 2}


def test_summarize_identity_records_reports_both_discordance_directions() -> None:
    records = (
        {
            "molecule_index": 0,
            "smiles": "A",
            "categories": ["target"],
            "rdkit_identity": "r1",
            "wl_key": "w1",
        },
        {
            "molecule_index": 1,
            "smiles": "B",
            "categories": ["leaf"],
            "rdkit_identity": "r2",
            "wl_key": "w1",
        },
        {
            "molecule_index": 2,
            "smiles": "C",
            "categories": ["stock"],
            "rdkit_identity": "r1",
            "wl_key": "w2",
        },
    )
    result = summarize_identity_records(records)
    assert result["unique_rdkit_identities"] == 2
    assert result["unique_wl_keys"] == 2
    assert result["wl_false_merge_groups"] == 1
    assert result["rdkit_false_split_groups"] == 1

from __future__ import annotations

import csv
import gzip
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import tests.datasets as datasets
from vesma.reaction import reaction_patterns_from_smarts


def test_compatible_templates_file_is_one_column_csv() -> None:
    path = datasets.compatible_templates_path()
    assert path.exists()
    assert path.name == "uspto42k-compat.csv.gz"
    assert path.parent.name == "processed"

    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == ("retro_template",)
        rows = [row["retro_template"] for _, row in zip(range(3), reader)]

    assert rows
    patterns = reaction_patterns_from_smarts(tuple(rows), n_jobs=1, cache=False)
    assert all(pattern.lhs.atom_count <= 64 for pattern in patterns)


def test_load_compatible_templates_reads_single_column_csv(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    path = tmp_path / "uspto42k-compat.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("retro_template",))
        writer.writeheader()
        writer.writerow({"retro_template": "[C:1][O:2]>>[C:1].[O:2]"})

    monkeypatch.setattr(datasets, "compatible_templates_path", lambda: path)

    templates = datasets.load_compatible_templates()

    assert len(templates) == 1
    assert templates[0].id == "0"
    assert templates[0].smarts == "[C:1][O:2]>>[C:1].[O:2]"
    assert templates[0].metadata == {}
    assert templates[0].pattern.lhs.atom_count == 2


def test_positive_pairs_artifact_matches_sources() -> None:
    artifact = datasets.load_paroutes_n1_d2_match_uspto42k_artifact()

    assert datasets.paroutes_n1_d2_match_uspto42k_pairs_path().exists()
    assert artifact.target_count == 2881
    assert artifact.pattern_count == 42554
    assert artifact.positive_pairs.ndim == 2
    assert artifact.positive_pairs.shape[1] == 2

    pairs = artifact.positive_pairs
    if len(pairs) > 0:
        assert np.all(pairs[:, 0] >= 0)
        assert np.all(pairs[:, 0] < artifact.target_count)
        assert np.all(pairs[:, 1] >= 0)
        assert np.all(pairs[:, 1] < artifact.pattern_count)
        order = np.lexsort((pairs[:, 1], pairs[:, 0]))
        assert np.array_equal(order, np.arange(len(pairs)))


def test_positive_pairs_filter_by_molecule_selection_only() -> None:
    artifact = datasets.load_paroutes_n1_d2_match_uspto42k_artifact()
    selection = tuple(range(10))

    observed = datasets.load_paroutes_n1_d2_match_uspto42k_positive_pairs(selection)
    expected = artifact.positive_pairs[
        np.isin(artifact.positive_pairs[:, 0], np.asarray(selection))
    ]

    assert observed == [tuple(map(int, row)) for row in expected.tolist()]
    assert all(0 <= molecule_idx < 10 for molecule_idx, _ in observed)
    assert all(0 <= pattern_idx < artifact.pattern_count for _, pattern_idx in observed)


def test_positive_pairs_filter_remaps_arbitrary_molecule_selection() -> None:
    artifact = datasets.load_paroutes_n1_d2_match_uspto42k_artifact()
    original_indices = tuple(np.unique(artifact.positive_pairs[:, 0]).tolist())
    if len(original_indices) < 2:
        pytest.skip("Positive-pairs artifact has too few matched molecules.")

    selection = tuple(reversed(original_indices[:2]))
    observed = datasets.load_paroutes_n1_d2_match_uspto42k_positive_pairs(selection)

    assert observed
    assert {molecule_idx for molecule_idx, _ in observed} <= {0, 1}


def test_positive_pairs_hash_mismatch_raises() -> None:
    payload = datasets.read_positive_pairs_json(
        datasets.paroutes_n1_d2_match_uspto42k_pairs_path()
    )
    payload["target_smiles_strings_hash"] = "not-the-source-hash"

    with pytest.raises(ValueError, match="target SMILES hash"):
        datasets.validate_positive_pairs_payload(
            payload,
            target_smiles=datasets.read_paroutes_n1_d2_f0_smiles(),
            pattern_smarts=datasets.read_compatible_template_smarts(),
        )

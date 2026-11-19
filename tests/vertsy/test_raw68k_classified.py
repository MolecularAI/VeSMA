"""The classified raw68k templates match their manifest, and their counts are the real ones.

These pin the dataset rather than exercise code. The artifacts are committed, so the thing
that can go wrong is not a crash but a silent drift: a rebuild that changes the file, a
classifier rerun that shifts the labels, a column renamed out from under the loader. Every
assertion here is a number the README also states, so the two cannot disagree for long.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from tests.datasets import (
    CLASSIFIED_COLUMNS,
    raw68k_classified_path,
    raw68k_verdict_path,
    read_classified_templates,
)

# Pinned from the committed artifacts. A rebuild must reproduce these byte for byte, which is
# only possible because the build pins gzip's mtime to 0.
SHA256 = {
    "raw68k-classified.tsv.gz": "45113f4edcbcd2c99061583a8173771886d70c60ac56b87d88d94d98921fe6dd",
}

#: The curated projection is a build product, not a committed file, so it is only asserted
#: when it happens to be present. Skipping rather than failing keeps a fresh checkout green;
#: the message says which target produces it.
CURATED_MISSING = "run `make raw68k-curated` to build the projection"

SMARTS = 67_557
HASHES = 42_554
IN_USPTO42K = 42_554
IN_USED14K = 13_978
IN_BOTH = 9_725
IN_NEITHER = 20_750

VERDICTS = frozenset(
    {"agree", "unconfirmed", "mapping-conflict", "zenodo-conflict", "unclassified"}
)


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    return read_classified_templates(raw68k_classified_path())


@pytest.fixture(scope="module")
def dataset_dir() -> Path:
    return raw68k_classified_path().parent.parent


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_artifacts_match_their_checksums(subtests) -> None:
    """The committed bytes are the ones the README and the manifest describe."""

    for path in (raw68k_classified_path(),):
        with subtests.test(file=path.name):
            assert sha256_of(path) == SHA256[path.name]


def test_every_raw68k_template_is_present(rows) -> None:
    assert len(rows) == SMARTS
    assert len({row["template_hash"] for row in rows}) == HASHES


def test_the_header_is_the_contract(rows) -> None:
    """`read_classified_templates` raises on drift, so reaching here means it held."""

    assert tuple(rows[0]) == CLASSIFIED_COLUMNS


def test_every_template_carries_a_label() -> None:
    """Coverage is total, and that is a property of the data rather than a coincidence.

    raw68k's hash set and the label files' hash set are identical, so labelling 42,554 hashes
    labels all 67,557 spellings. If a future relabelling covered fewer hashes this would be the
    first thing to fail, which is why it is asserted rather than assumed.
    """

    rows = read_classified_templates(raw68k_classified_path())
    unlabelled = [row for row in rows if not row["rc_code"] or not row["ri_name"]]

    assert unlabelled == []


def test_the_subsets_are_the_measured_ones(rows) -> None:
    """uspto42k and used14k overlap without either containing the other.

    The interesting number is `IN_NEITHER`: 20,750 spellings that uspto42k did not pick and no
    reference route used. They are the reason raw68k is the library to curate.
    """

    uspto = sum(row["in_uspto42k"] == "True" for row in rows)
    used = sum(row["in_used14k"] == "True" for row in rows)
    both = sum(
        row["in_uspto42k"] == "True" and row["in_used14k"] == "True" for row in rows
    )
    neither = sum(
        row["in_uspto42k"] == "False" and row["in_used14k"] == "False" for row in rows
    )

    assert (uspto, used, both, neither) == (
        IN_USPTO42K,
        IN_USED14K,
        IN_BOTH,
        IN_NEITHER,
    )


def test_verdicts_come_from_a_closed_set(rows) -> None:
    """Adding a verdict should be a deliberate act, not a silent one."""

    assert {row["verdict"] for row in rows} <= VERDICTS


@pytest.mark.skipif(not raw68k_verdict_path().exists(), reason=CURATED_MISSING)
def test_the_curated_projection_is_exactly_the_agreeing_rows(rows) -> None:
    """The projection is the filter it claims to be, and keeps the full column set.

    Keeping every column is what makes it usable as a rule set without conversion: readers
    take `retro_template` and ignore the rest.
    """

    curated = read_classified_templates(raw68k_verdict_path())
    agreeing = [row for row in rows if row["verdict"] == "agree"]

    assert len(curated) == len(agreeing)
    assert {row["retro_template"] for row in curated} == {
        row["retro_template"] for row in agreeing
    }


def test_the_curated_set_covers_most_of_the_library_by_occurrence(rows) -> None:
    """Row count understates coverage, so the occurrence share is what gets asserted.

    The curated set keeps 63% of spellings but 74.6% of `library_occurence`, because the
    templates that survive curation are the ones that fire often.
    """

    total = sum(int(row["library_occurence"]) for row in rows)
    kept = sum(
        int(row["library_occurence"]) for row in rows if row["verdict"] == "agree"
    )

    assert kept / total == pytest.approx(0.746, abs=0.002)


def test_the_hand_assigned_rows_are_marked(rows) -> None:
    """The only labels not produced by a classifier are flagged as such.

    Three Balz-Schiemann templates that both classifiers return Unrecognized for, carrying four
    SMARTS spellings between them. `label_source` is what lets a consumer drop them.
    """

    manual = [row for row in rows if row["label_source"] == "manual"]

    assert len(manual) == 4
    assert len({row["template_hash"] for row in manual}) == 3
    assert {row["rc_code"] for row in manual} == {"9.1.5.12"}


def test_the_mapping_covers_every_class_it_is_asked_about(rows, dataset_dir) -> None:
    """A row can only be judged against a class the mapping knows.

    An unmapped class would make `mapped_ri_name` empty and send the row to
    mapping-conflict for a reason that is about the mapping, not the chemistry.
    """

    with open(dataset_dir / "processed" / "class-mapping.tsv", newline="") as handle:
        mapped = {row["rc_class"] for row in csv.DictReader(handle, delimiter="\t")}

    confirmed = [
        row
        for row in rows
        if row["rc_verified_votes"] not in ("", "0", "0.0")
        and not row["rc_code"].startswith("0.0")
        and row["label_source"] != "manual"
    ]
    classes = {".".join(row["rc_code"].split(".")[:2]) for row in confirmed}

    assert classes <= mapped


def test_the_manifest_agrees_with_the_data(rows, dataset_dir) -> None:
    manifest = json.loads(
        (dataset_dir / "processed" / "raw68k-classified.provenance.json").read_text()
    )
    counts = manifest["counts"]

    assert counts["smarts"] == len(rows)
    assert counts["hashes"] == len({row["template_hash"] for row in rows})
    assert counts["agree_smarts"] == sum(row["verdict"] == "agree" for row in rows)
    assert manifest["mapping_depth"] == 2


def test_the_vendored_labels_are_readable_and_keyed_on_hash(dataset_dir) -> None:
    """The committed label files are what the build reads; a fresh checkout has only these."""

    for name in ("reactionclassifier-labels.json.gz", "rxn-insight-labels.json.gz"):
        with gzip.open(dataset_dir / "vendored" / name, "rt") as handle:
            records = json.load(handle)

        assert len(records) == HASHES - 3, "the 3 both classifiers reject are absent"
        assert set(records[0]) == {
            "template_hash",
            "classification",
            "votes",
            "agreement",
            "verified_votes",
        }

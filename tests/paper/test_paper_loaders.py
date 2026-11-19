from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PAPER = Path(__file__).parents[2] / "docs" / "paper" / "paper"
sys.path.insert(0, str(PAPER))

from paperlib.loaders import (  # noqa: E402
    explode_records,
    load_benchmarks,
    load_json,
    require_audit_accounting,
)
from paperlib.paper_tables import (  # noqa: E402
    backend_label,
    fraction_cell,
    precursor_audit_agreement_table,
    precursor_audit_class_table,
    qtable,
    require_agreeing_records,
    require_one_record,
    require_unique_records,
)


def benchmark_document(extra_info: dict[str, object]) -> dict[str, object]:
    return {
        "machine_info": {"host_type": "test-host"},
        "benchmarks": [
            {
                "extra_info": extra_info,
                "stats": {"mean": 1.0, "data": [1.0]},
            }
        ],
    }


def write_document(path: Path, document: dict[str, object]) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")


def test_load_benchmarks_requires_current_fields(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    write_document(
        path,
        benchmark_document(
            {
                "experiment": "paper-validation",
                "backend": "cpu",
                "validation": "identity-agreement",
                "molecules_seen": 1,
                "molecules_encoded": 1,
                "identity_records": [],
                "encoding_failures": 0,
                "wl_false_merge_records": [],
                "rdkit_false_split_records": [],
            }
        ),
    )

    frame = load_benchmarks(tmp_path)

    assert frame.loc[0, "experiment"] == "paper-validation"
    assert frame.loc[0, "backend"] == "cpu"
    assert frame.loc[0, "stats.mean"] == 1.0
    assert frame.loc[0, "source_path"] == str(path.resolve())


def test_load_benchmarks_reads_a_gzipped_document(tmp_path: Path) -> None:
    """Correctness artifacts are written uncapped, so they are written compressed.

    They carry every divergence and refusal in full rather than a sample, which is what
    makes them worth keeping and also what makes them too large to store as plain text.
    """
    document = benchmark_document(
        {
            "experiment": "paper-validation",
            "backend": "cpu",
            "validation": "template-compatibility",
            "template_sources": [],
        }
    )
    path = tmp_path / "result.json.gz"
    path.write_bytes(gzip.compress(json.dumps(document).encode("utf-8")))

    frame = load_benchmarks(tmp_path)

    assert frame.loc[0, "validation"] == "template-compatibility"
    assert frame.loc[0, "source_path"] == str(path.resolve())


def test_require_one_record_lists_every_conflicting_json() -> None:
    frame = pd.DataFrame(
        {
            "source_path": ["/results/old.json", "/results/new.json"],
            "backend": ["cuda-device", "cuda-device"],
        }
    )

    with pytest.raises(ValueError) as error:
        require_one_record(frame, context="RTX 5090 CUDA-device stage record")

    message = str(error.value)
    assert "/results/old.json" in message
    assert "/results/new.json" in message
    assert "does not choose one implicitly" in message


def test_require_agreeing_records_accepts_repeated_audits() -> None:
    frame = pd.DataFrame(
        {
            "source_path": ["/results/mac.json", "/results/linux.json"],
            "result": [
                {"count": 2, "artifact": {"path": "/mac/input", "sha256": "abc"}},
                {
                    "count": 2,
                    "artifact": {"path": "/linux/input", "sha256": "abc"},
                },
            ],
        }
    )

    record = require_agreeing_records(
        frame,
        ["result"],
        context="rules and keys audit",
        ignore_nested_keys=frozenset({"path"}),
    )

    assert record["source_path"] == "/results/mac.json"


def test_require_agreeing_records_reports_differing_fields_and_files() -> None:
    frame = pd.DataFrame(
        {
            "source_path": ["/results/old.json", "/results/new.json"],
            "templates": [42_554, 42_553],
            "false_merges": [0, 1],
        }
    )

    with pytest.raises(ValueError) as error:
        require_agreeing_records(
            frame,
            ["templates", "false_merges"],
            context="rules and keys audit",
        )

    message = str(error.value)
    assert "/results/old.json" in message
    assert "/results/new.json" in message
    assert "templates, false_merges" in message


def test_require_unique_records_lists_duplicate_coordinates_and_files() -> None:
    frame = pd.DataFrame(
        {
            "source_path": ["/results/old.json", "/results/new.json"],
            "backend": ["cuda-device", "cuda-device"],
            "molecules": [2881, 2881],
        }
    )

    with pytest.raises(ValueError) as error:
        require_unique_records(
            frame,
            ["backend", "molecules"],
            context="match-plus-split scaling",
        )

    message = str(error.value)
    assert "backend='cuda-device', molecules=2881" in message
    assert "/results/old.json" in message
    assert "/results/new.json" in message


@pytest.mark.parametrize(
    "document, message",
    [
        ({}, "machine_info"),
        ({"machine_info": {}, "benchmarks": []}, "non-empty benchmarks"),
        (
            {"machine_info": {}, "benchmarks": [{"stats": {}}]},
            "extra_info",
        ),
        (
            benchmark_document({"experiment": "unknown", "backend": "cpu"}),
            "unsupported experiment",
        ),
        (
            benchmark_document({"experiment": "paper-validation", "backend": ""}),
            "unsupported backend",
        ),
        (
            benchmark_document(
                {"experiment": "paper-validation", "backend": "not-a-backend"}
            ),
            "unsupported backend",
        ),
    ],
)
def test_load_benchmarks_fails_on_missing_or_unsupported_data(
    tmp_path: Path, document: dict[str, object], message: str
) -> None:
    write_document(tmp_path / "result.json", document)

    with pytest.raises(ValueError, match=message):
        load_benchmarks(tmp_path)


def test_load_json_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.touch()

    with pytest.raises(ValueError, match="Empty benchmark JSON"):
        load_json(path)


def test_explode_records_requires_lists_of_mappings() -> None:
    with pytest.raises(ValueError, match="must contain lists"):
        explode_records(pd.DataFrame({"rows": [None]}), "rows", "row")

    with pytest.raises(ValueError, match="must contain dictionaries"):
        explode_records(pd.DataFrame({"rows": [[1]]}), "rows", "row")


def test_table_generation_uses_current_json() -> None:
    frame = load_benchmarks(PAPER.parent / "benchmarks" / "reaction-application")
    table = (
        frame[["backend", "stats.mean"]]
        .head(2)
        .rename(columns={"backend": "Backend", "stats.mean": "Seconds"})
    )

    rendered = qtable(table, caption="Smoke table", table_id="tbl-smoke")

    assert "#tbl-smoke" in rendered
    assert "Smoke table" in rendered


def test_unknown_backend_and_missing_fraction_fail() -> None:
    with pytest.raises(KeyError):
        backend_label("not-a-backend")
    with pytest.raises(ValueError, match="non-missing"):
        fraction_cell(None, 10)


def test_precursor_audit_publication_aggregates() -> None:
    agreement = precursor_audit_agreement_table()
    classes = precursor_audit_class_table()

    assert agreement.to_dict(orient="records") == [
        {
            "Comparison": "VeSMA-RDChiral",
            "All positive pairs": "755,047/768,015\\\n(98.3%)",
            "Comparable pairs": "754,780/767,732\\\n(98.3%)",
        },
        {
            "Comparison": "VeSMA-RDKit",
            "All positive pairs": "710,592/768,015\\\n(92.5%)",
            "Comparable pairs": "705,147/756,764\\\n(93.2%)",
        },
        {
            "Comparison": "RDKit-RDChiral",
            "All positive pairs": "711,864/768,015\\\n(92.7%)",
            "Comparable pairs": "706,189/756,749\\\n(93.3%)",
        },
    ]
    assert classes["Frequency in audit"].tolist() == [
        "Focused regression example",
        "29,215/2,468,381\\\n(1.18%)",
        "5,218/2,468,381\\\n(0.21%)",
        "3,936/2,468,381\\\n(0.16%)",
        "540/2,468,381\\\n(0.022%)",
        "49,750/768,015\\\n(6.48%)",
        "1,773/2,468,381\\\n(0.072%)",
        "5,026/768,015\\\n(0.65%)",
        "RDKit 17,140/2,468,494\\\n(0.69%); VeSMA 281/2,468,381\\\n(0.011%)",
        "Not exercised by this audit",
    ]


@pytest.mark.parametrize(
    "family",
    [
        "fanout-depth",
        "match-only-scaling",
        "reaction-application",
        "paper-validation",
        "route-recovery",
        "route-replay",
    ],
)
def test_retained_publication_json_uses_current_contract(family: str) -> None:
    frame = load_benchmarks(PAPER.parent / "benchmarks" / family)

    assert not frame.empty
    assert frame["experiment"].notna().all()
    assert frame["backend"].notna().all()


def test_an_empty_benchmark_json_is_skipped_with_a_visible_warning(
    tmp_path, capsys
) -> None:
    """A run in progress must not stop the paper from rendering.

    `pytest-benchmark` creates its output file when a run starts and fills it when the run ends,
    so any measurement in flight leaves an empty file under the tree. Refusing it meant
    `make paper-render` failed for as long as a benchmark was running -- which is exactly when
    someone wants to confirm the paper still builds.

    The message goes to stderr, which Quarto renders into the document beside the table it
    affects, so a draft shows *which* file was skipped rather than quietly dropping a backend.
    """
    (tmp_path / "live").mkdir()
    (tmp_path / "done").mkdir()
    (tmp_path / "live" / "inflight.json").write_text("")
    sample = next((PAPER.parent / "benchmarks" / "match-only-scaling").rglob("*.json"))
    (tmp_path / "done" / "done.json").write_text(sample.read_text(encoding="utf-8"))

    frame = load_benchmarks(tmp_path)

    assert not frame.empty
    assert "inflight.json" in capsys.readouterr().err


def test_a_tree_of_only_empty_json_is_still_an_error(tmp_path) -> None:
    """Tolerating one in-flight file must not quietly render a paper from no data at all."""
    (tmp_path / "all-empty.json").write_text("")

    with pytest.raises(ValueError, match="every benchmark JSON"):
        load_benchmarks(tmp_path)


def audit_extra_info(**overrides: object) -> dict[str, object]:
    record = {
        "experiment": "paper-validation",
        "backend": "cpu",
        "validation": "precursor-audit",
        "pairs_compared": 2,
        "agreement_records": [],
        "backend_records": [
            {
                "backend": "vesma-cpu",
                "applications_accepted": 7,
                "applications_refused": 2,
                "applications_unkeyable": 1,
            },
            {
                "backend": "rdkit",
                "applications_accepted": 9,
                "applications_refused": 0,
                "applications_unkeyable": 1,
            },
            {"backend": "rdchiral", "applications_accepted": 5},
        ],
        "vesma_rejections": {"delocalized_edge_not_cyclic": 2},
        "rdkit_dropped_outcomes": {"outcomes_total": 10, "outcomes_unsanitizable": 1},
        "divergences": [],
        "refusals": [],
        "unkeyable_precursors": [],
        "rdkit_discards": [{"molecule_index": 0, "rule_id": 0, "app_idx": 0}],
        "vesma_enumeration": {
            "applications_attempted": 10,
            "pairs_declined": 0,
            "pairs_search_incomplete": 0,
        },
    }
    record.update(overrides)
    return record


def test_precursor_audit_documents_load(tmp_path: Path) -> None:
    path = tmp_path / "audit.json.gz"
    document = benchmark_document(audit_extra_info())
    path.write_bytes(gzip.compress(json.dumps(document).encode("utf-8")))

    frame = load_benchmarks(tmp_path)

    assert frame.loc[0, "validation"] == "precursor-audit"


def test_audit_accounting_is_per_backend_not_across_them() -> None:
    """Each backend must account for what *it* enumerated, and nothing more.

    Vesma counts subgraph isomorphisms of the left-hand side; RDKit counts RunReactants
    outcomes. Several reaction centres can collapse to one outcome, and each backend
    decides for itself whether that is one result or several, so the totals are not
    comparable and must not be required to match. An earlier version required exactly
    that, on the strength of the two agreeing at size 100; over the full corpus they
    differ by 743 in 2.47 million.
    """
    differing = audit_extra_info(
        vesma_enumeration={
            "applications_attempted": 10,
            "pairs_declined": 3,
            "pairs_search_incomplete": 0,
        }
    )
    require_audit_accounting(differing, context="audit")

    short = audit_extra_info(
        vesma_enumeration={
            "applications_attempted": 12,
            "pairs_declined": 0,
            "pairs_search_incomplete": 0,
        }
    )
    with pytest.raises(ValueError, match="vesma accounts for"):
        require_audit_accounting(short, context="audit")


def test_audit_accounting_rejects_a_search_that_declined_to_answer() -> None:
    """A pair the matcher gave up on contributes to no total at all.

    "No match" is an answer and is accounted for. An exhausted search is not: those
    applications were never attempted, so they are missing from accepted, refused and
    unkeyable alike, and the account would look complete while being short.
    """
    gave_up = audit_extra_info(
        vesma_enumeration={
            "applications_attempted": 10,
            "pairs_declined": 1,
            "pairs_search_incomplete": 1,
        }
    )
    with pytest.raises(ValueError, match="declined to answer"):
        require_audit_accounting(gave_up, context="audit")


def test_audit_accounting_requires_rdkit_to_close_against_its_own_total() -> None:
    lost = audit_extra_info(
        backend_records=[
            {"backend": "vesma-cpu", "applications_accepted": 10},
            {"backend": "rdkit", "applications_accepted": 9},
        ]
    )
    with pytest.raises(ValueError, match="RDKit accounts for"):
        require_audit_accounting(lost, context="audit")


def test_audit_accounting_requires_a_row_per_discard() -> None:
    """A total without its rows cannot be reviewed chemically, only believed."""
    with pytest.raises(ValueError, match="discard rows recorded"):
        require_audit_accounting(audit_extra_info(rdkit_discards=[]), context="audit")


def test_chemistry_fingerprint_moves_when_the_chemistry_does(tmp_path: Path) -> None:
    """The packaged version cannot say which chemistry produced an artifact.

    ``vesma_version()`` stayed at 0.1.0 across every chemistry change in this work, so a
    cached molecule graph or a resumed sweep checkpoint could silently carry perception
    from code that has since been replaced. The fingerprint digests the source instead.
    """
    import sys

    sys.path.insert(0, str(PAPER.parents[2]))
    from tests.datasets import chemistry_fingerprint

    first = chemistry_fingerprint()

    assert first == chemistry_fingerprint(), "must be stable within a run"
    assert len(first) == 16

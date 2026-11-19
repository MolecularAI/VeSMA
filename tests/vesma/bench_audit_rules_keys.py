"""Rules-and-keys audit records used by the Vesma evaluation."""

from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.datadirs import external_data_root
from tests.vertsy.paper_provenance import source_artifact
from tests.vertsy.paroutes_reference import clean_reaction_smarts
from vesma.keyer import rdkit_token_from_smiles, short_ring_wl_keyer
from vesma.parsing.errors import SmartsError
from vesma.reaction import parse_reaction_smarts

pytestmark = [pytest.mark.perf, pytest.mark.paper_benchmark]

PAROUTES = external_data_root() / "external" / "paroutes" / "processed"
REMIX = external_data_root() / "external" / "remix"


def test_bench_template_compatibility(benchmark: Any) -> None:
    report = benchmark.pedantic(
        audit_template_compatibility,
        rounds=1,
        warmup_rounds=0,
        iterations=1,
    )
    benchmark.extra_info.update(
        {
            "experiment": "paper-validation",
            "backend": "cpu",
            "validation": "template-compatibility",
            **report,
        }
    )


def test_bench_identity_agreement(benchmark: Any) -> None:
    report = benchmark.pedantic(
        audit_identity_agreement,
        rounds=1,
        warmup_rounds=0,
        iterations=1,
    )
    benchmark.extra_info.update(
        {
            "experiment": "paper-validation",
            "backend": "cpu",
            "validation": "identity-agreement",
            **report,
        }
    )


def audit_template_compatibility() -> dict[str, Any]:
    sources = template_sources()
    outcomes: dict[str, tuple[str, str]] = {}
    records = []
    for name, source_rows, path, smarts in sources:
        unique_smarts = tuple(
            dict.fromkeys(clean_reaction_smarts(row) for row in smarts)
        )
        for text in unique_smarts:
            if text not in outcomes:
                outcomes[text] = parse_outcome(text)
        records.append(
            summarize_template_source(
                name,
                source_rows,
                unique_smarts,
                outcomes,
                source_artifact(path),
            )
        )
    return {"template_sources": records}


def template_sources() -> tuple[tuple[str, int, Path, tuple[str, ...]], ...]:
    used14k = PAROUTES / "used14k-reactions.tsv.gz"
    raw68k = PAROUTES / "raw68k-templates.tsv.gz"
    raw68k_provenance = PAROUTES / "raw68k-templates.provenance.json"
    uspto42k = PAROUTES / "uspto_unique_templates.csv.gz"
    compatible = REMIX / "uspto42k-compat" / "processed" / "uspto42k-compat.csv.gz"
    provenance = json.loads(raw68k_provenance.read_text(encoding="utf-8"))
    return (
        (
            "used14k",
            count_gzip_rows(used14k, delimiter="\t"),
            used14k,
            read_gzip_column(used14k, "retro_template", delimiter="\t"),
        ),
        (
            "raw68k",
            int(provenance["raw_reaction_rows"]),
            raw68k,
            read_gzip_column(raw68k, "retro_template", delimiter="\t"),
        ),
        (
            "uspto42k",
            count_gzip_rows(uspto42k, delimiter="\t"),
            uspto42k,
            read_gzip_column(uspto42k, "retro_template", delimiter="\t"),
        ),
        (
            "uspto42k-compat",
            count_gzip_rows(compatible, delimiter=","),
            compatible,
            read_gzip_column(compatible, "retro_template", delimiter=","),
        ),
    )


def count_gzip_rows(path: Path, *, delimiter: str) -> int:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle, delimiter=delimiter))


def read_gzip_column(path: Path, column: str, *, delimiter: str) -> tuple[str, ...]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if column not in tuple(reader.fieldnames or ()):
            raise ValueError(f"{path} does not contain required column {column!r}")
        return tuple(row[column] for row in reader)


def parse_outcome(smarts: str) -> tuple[str, str]:
    try:
        parse_reaction_smarts(smarts)
    except SmartsError as exc:
        return type(exc).__name__, str(exc)
    return "accepted", ""


def summarize_template_source(
    name: str,
    source_rows: int,
    smarts: tuple[str, ...],
    outcomes: dict[str, tuple[str, str]],
    artifact: dict[str, Any],
) -> dict[str, Any]:
    rejected: dict[str, int] = defaultdict(int)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    accepted = 0
    for index, text in enumerate(smarts):
        category, message = outcomes[text]
        if category == "accepted":
            accepted += 1
            continue
        rejected[category] += 1
        if len(examples[category]) < 10:
            examples[category].append(
                {"template_index": index, "message": message, "smarts": text}
            )
    return {
        "name": name,
        "source_rows": source_rows,
        "unique_templates": len(smarts),
        "accepted_templates": accepted,
        "rejected_templates": len(smarts) - accepted,
        "rejections_by_category": dict(sorted(rejected.items())),
        "rejection_examples": dict(sorted(examples.items())),
        "source_artifact": artifact,
    }


def audit_identity_agreement() -> dict[str, Any]:
    categorized = categorized_paroutes_smiles()
    codec = short_ring_wl_keyer
    records = []
    failures = []
    for index, (smiles, categories) in enumerate(sorted(categorized.items())):
        try:
            rdkit_token = str(rdkit_token_from_smiles(smiles))
            low, high = codec.from_smiles(smiles)
        except (ValueError, SmartsError) as exc:
            failures.append(
                {
                    "molecule_index": index,
                    "smiles": smiles,
                    "categories": sorted(categories),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue
        records.append(
            {
                "molecule_index": index,
                "smiles": smiles,
                "categories": sorted(categories),
                "rdkit_identity": rdkit_token,
                "wl_key": f"{high:016x}{low:016x}",
            }
        )

    summary = summarize_identity_records(records)
    return {
        "identity_codec": codec.name,
        "molecules_seen": len(categorized),
        "molecules_encoded": len(records),
        "identity_records": records,
        "encoding_failures": len(failures),
        "encoding_failure_records": failures,
        "category_counts": category_counts(categorized),
        "source_artifacts": identity_source_artifacts(),
        **summary,
    }


def categorized_paroutes_smiles() -> dict[str, set[str]]:
    categorized: dict[str, set[str]] = defaultdict(set)
    for route_set in ("n1", "n5"):
        routes_path = PAROUTES / f"{route_set}-routes.json.gz"
        with gzip.open(routes_path, "rt", encoding="utf-8") as handle:
            routes = json.load(handle)
        for route in routes:
            for smiles, category in iter_route_molecules(route, is_root=True):
                categorized[smiles].add(category)
        stock_path = PAROUTES / f"{route_set}-stock.smi.gz"
        with gzip.open(stock_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                smiles = line.strip()
                if smiles:
                    categorized[smiles].add("stock")
    return categorized


def iter_route_molecules(
    node: dict[str, Any], *, is_root: bool = False
) -> Iterator[tuple[str, str]]:
    children = node.get("children", [])
    if node.get("type") == "mol":
        smiles = node.get("smiles")
        if not isinstance(smiles, str) or not smiles:
            raise ValueError("PaRoutes molecule node requires a non-empty SMILES")
        category = "target" if is_root else ("intermediate" if children else "leaf")
        yield smiles, category
    for child in children:
        yield from iter_route_molecules(child)


def summarize_identity_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = tuple(records)
    reference_by_wl: dict[str, set[str]] = defaultdict(set)
    wl_by_reference: dict[str, set[str]] = defaultdict(set)
    examples_by_wl: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples_by_reference: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in rows:
        reference = str(record["rdkit_identity"])
        wl_key = str(record["wl_key"])
        reference_by_wl[wl_key].add(reference)
        wl_by_reference[reference].add(wl_key)
        examples_by_wl[wl_key].append(record)
        examples_by_reference[reference].append(record)

    false_merges = [key for key, values in reference_by_wl.items() if len(values) > 1]
    false_splits = [key for key, values in wl_by_reference.items() if len(values) > 1]
    return {
        "unique_rdkit_identities": len(wl_by_reference),
        "unique_wl_keys": len(reference_by_wl),
        "wl_false_merge_groups": len(false_merges),
        "rdkit_false_split_groups": len(false_splits),
        "wl_false_merge_records": [
            {
                "wl_key": key,
                "rdkit_identities": sorted(reference_by_wl[key]),
                "examples": examples_by_wl[key][:10],
            }
            for key in sorted(false_merges)
        ],
        "rdkit_false_split_records": [
            {
                "rdkit_identity": key,
                "wl_keys": sorted(wl_by_reference[key]),
                "examples": examples_by_reference[key][:10],
            }
            for key in sorted(false_splits)
        ],
    }


def category_counts(categorized: dict[str, set[str]]) -> dict[str, int]:
    categories = sorted(
        {category for values in categorized.values() for category in values}
    )
    return {
        category: sum(category in values for values in categorized.values())
        for category in categories
    }


def identity_source_artifacts() -> dict[str, dict[str, Any]]:
    return {
        name: source_artifact(PAROUTES / name)
        for name in (
            "n1-routes.json.gz",
            "n5-routes.json.gz",
            "n1-stock.smi.gz",
            "n5-stock.smi.gz",
        )
    }

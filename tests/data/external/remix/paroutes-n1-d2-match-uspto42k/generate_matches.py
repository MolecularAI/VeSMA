from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import FilterCatalog

DATASET_DIR = Path(__file__).resolve().parent
EXTERNAL_DIR = DATASET_DIR.parent.parent
DEFAULT_TARGETS = EXTERNAL_DIR / "paroutes" / "processed" / "n1-d2-f0.smi.gz"
DEFAULT_PATTERNS = (
    EXTERNAL_DIR / "remix" / "uspto42k-compat" / "processed" / "uspto42k-compat.csv.gz"
)
DEFAULT_OUTPUT = DATASET_DIR / "processed" / "n1-d2-f0-uspto42k-compat-matches.json.gz"
PATTERN_COLUMNS = ("retro_template",)


def main() -> None:
    args = parse_args()
    target_smiles = read_smiles(args.targets)
    pattern_smarts = read_pattern_smarts(args.patterns)
    pairs = positive_pairs(target_smiles, pattern_smarts)
    payload = {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "hash_encoding": "json-array-utf8-compact",
        "target_smiles_strings_hash": stable_string_tuple_hash(target_smiles),
        "patterns_smarts_strings_hash": stable_string_tuple_hash(pattern_smarts),
        "target_count": len(target_smiles),
        "pattern_count": len(pattern_smarts),
        "positive_pairs": pairs,
    }
    write_json_gz(args.output, payload)
    print(
        f"wrote {len(pairs)} positive pairs for "
        f"{len(target_smiles)} targets x {len(pattern_smarts)} patterns -> {args.output}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--patterns", type=Path, default=DEFAULT_PATTERNS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_smiles(path: Path) -> tuple[str, ...]:
    rows: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip().split()[0]
            if value:
                rows.append(value)
    return tuple(rows)


def read_pattern_smarts(path: Path) -> tuple[str, ...]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PATTERN_COLUMNS:
            raise ValueError(f"Unexpected pattern columns: {reader.fieldnames}")
        return tuple(row["retro_template"] for row in reader)


def stable_string_tuple_hash(strings: tuple[str, ...]) -> str:
    payload = json.dumps(
        strings,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def positive_pairs(
    target_smiles: tuple[str, ...],
    pattern_smarts: tuple[str, ...],
) -> list[list[int]]:
    catalog = filter_catalog(pattern_smarts)
    rows = FilterCatalog.RunFilterCatalog(catalog, list(target_smiles), numThreads=0)
    pairs = [
        [mol_index, rule_index_from_entry(entry)]
        for mol_index, row in enumerate(rows)
        for entry in row
    ]
    pairs.sort()
    return pairs


def filter_catalog(pattern_smarts: tuple[str, ...]) -> Any:
    params = FilterCatalog.FilterCatalogParams()
    catalog = FilterCatalog.FilterCatalog(params)
    for index, smarts in enumerate(pattern_smarts):
        query = Chem.MolFromSmarts(lhs_smarts(smarts))
        if query is None:
            raise ValueError(f"RDKit failed to parse LHS SMARTS at pattern {index}.")
        matcher = FilterCatalog.SmartsMatcher(f"R_{index}", query)
        if not matcher.IsValid():
            raise ValueError(f"RDKit FilterCatalog rejected query {index}.")
        catalog.AddEntry(FilterCatalog.FilterCatalogEntry(f"R_{index}", matcher))
    return catalog


def lhs_smarts(smarts: str) -> str:
    parts = smarts.partition(">>")
    if parts[1] != ">>":
        raise ValueError("Reaction SMARTS is missing >> separator.")
    return parts[0].strip()


def rule_index_from_entry(entry: Any) -> int:
    description = str(entry.GetDescription())
    if not description.startswith("R_"):
        raise ValueError(f"Unexpected FilterCatalog entry description: {description}")
    return int(description[2:])


def write_json_gz(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
    tmp.replace(path)


if __name__ == "__main__":
    main()

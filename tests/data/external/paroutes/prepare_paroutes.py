from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import shutil
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ZENODO_BASE = "https://zenodo.org/records/7341155/files"
DATASET_DIR = Path(__file__).resolve().parent
ROUTE_SETS = ("n1", "n5")


@dataclass(frozen=True)
class DownloadSpec:
    key: str
    url: str
    output: str
    mode: Literal["gzip", "copy"]


DOWNLOADS = (
    DownloadSpec(
        "n1-targets",
        f"{ZENODO_BASE}/targets_n1.txt?download=1",
        "n1-targets.smi.gz",
        "gzip",
    ),
    DownloadSpec(
        "n1-stock",
        f"{ZENODO_BASE}/stock_n1.txt?download=1",
        "n1-stock.smi.gz",
        "gzip",
    ),
    DownloadSpec(
        "n1-routes",
        f"{ZENODO_BASE}/ref_routes_n1.json?download=1",
        "n1-routes.json.gz",
        "gzip",
    ),
    DownloadSpec(
        "n5-targets",
        f"{ZENODO_BASE}/targets_n5.txt?download=1",
        "n5-targets.smi.gz",
        "gzip",
    ),
    DownloadSpec(
        "n5-stock",
        f"{ZENODO_BASE}/stock_n5.txt?download=1",
        "n5-stock.smi.gz",
        "gzip",
    ),
    DownloadSpec(
        "n5-routes",
        f"{ZENODO_BASE}/ref_routes_n5.json?download=1",
        "n5-routes.json.gz",
        "gzip",
    ),
    DownloadSpec(
        "uspto-raw-template-library",
        f"{ZENODO_BASE}/uspto_template_library.csv?download=1",
        "uspto_raw_template_library.csv.gz",
        "gzip",
    ),
    DownloadSpec(
        "uspto-unique-templates",
        f"{ZENODO_BASE}/uspto_unique_templates.csv.gz?download=1",
        "uspto_unique_templates.csv.gz",
        "copy",
    ),
)

REFERENCE_REACTION_COLUMNS = (
    "reaction_hash",
    "smiles",
    "rsmi",
    "id",
    "ring_breaker",
)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.only or [spec.key for spec in DOWNLOADS])
    for spec in DOWNLOADS:
        if spec.key not in selected:
            continue
        subdir = "raw" if spec.key == "uspto-raw-template-library" else "processed"
        output = dataset_dir / subdir / spec.output
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and args.skip_existing:
            print(f"skip {output}")
            continue
        download(spec, output)

    if args.derive_reference_reactions:
        for route_set in ROUTE_SETS:
            output_path = (
                dataset_dir / "processed" / f"{route_set}-reference-reactions.tsv.gz"
            )
            if output_path.exists() and args.skip_existing:
                print(f"skip {output_path}")
                continue
            write_reference_reactions(dataset_dir, route_set)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=DATASET_DIR,
        type=Path,
    )
    parser.add_argument(
        "--only",
        choices=tuple(spec.key for spec in DOWNLOADS),
        action="append",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--derive-reference-reactions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def download(spec: DownloadSpec, output: Path) -> None:
    print(f"download {spec.key} -> {output}")
    tmp = output.with_suffix(output.suffix + ".tmp")
    encoding = "identity" if spec.mode == "copy" else "gzip"
    request = urllib.request.Request(spec.url, headers={"Accept-Encoding": encoding})
    with urllib.request.urlopen(request) as response:
        if spec.mode == "copy":
            with tmp.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        elif response.headers.get("Content-Encoding") == "gzip":
            with tmp.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        else:
            with tmp.open("wb") as raw:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=raw,
                    mtime=0,
                ) as handle:
                    shutil.copyfileobj(response, handle)
        tmp.replace(output)


def write_reference_reactions(dataset_dir: Path, route_set: str) -> None:
    routes_path = dataset_dir / "processed" / f"{route_set}-routes.json.gz"
    output_path = dataset_dir / "processed" / f"{route_set}-reference-reactions.tsv.gz"
    print(f"derive {output_path}")

    with gzip.open(routes_path, "rt", encoding="utf-8") as handle:
        routes = json.load(handle)

    rows = {
        row_key(row): row
        for route in routes
        for row in iter_reference_reaction_rows(route)
    }
    # Write with mtime=0 (like download()) so re-deriving identical content produces
    # byte-identical output and does not churn the committed file.
    with output_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=REFERENCE_REACTION_COLUMNS,
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerows(rows[key] for key in sorted(rows))


def iter_reference_reaction_rows(node: dict) -> Iterator[dict[str, str]]:
    if node.get("type") == "reaction":
        metadata = node["metadata"]
        yield {
            "reaction_hash": str(metadata.get("reaction_hash", "")),
            "smiles": str(metadata.get("smiles", "")),
            "rsmi": str(metadata.get("rsmi", "")),
            "id": str(metadata.get("ID", "")),
            "ring_breaker": str(metadata.get("RingBreaker", "")),
        }
    for child in node.get("children", []):
        yield from iter_reference_reaction_rows(child)


def row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["reaction_hash"],
        row["smiles"],
        row["rsmi"],
        row["id"],
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the committed ``used14k-reactions.tsv.gz`` route-reaction subset.

Provenance / regeneration script. The full PaRoutes raw template library
(``raw/uspto_raw_template_library.csv.gz``, ~116 MB, 819,797 rows, tab-delimited
despite the ``.csv`` name) is downloaded on demand and gitignored. Route replay only
ever looks up the reactions that actually appear in the n1 and n5 reference routes —
56,657 distinct ``reaction_hash`` values — so we vendor just those rows, keeping only
the two columns replay needs (``reaction_hash``, ``retro_template``). The result is
~2 MB and committed, removing the 116 MB download from the route-replay benchmark.

Run (raw library must be present; ``make paroutes-raw-template-library`` fetches it):

    python tests/data/external/paroutes/build_used14k_reactions.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW_COLUMNS = (
    "reaction_smiles",
    "reaction_hash",
    "classification",
    "ring_breaker",
    "retro_template",
    "template_hash",
)
OUTPUT_COLUMNS = ("reaction_hash", "retro_template")


def route_reaction_hashes(routes_path: Path) -> set[str]:
    """Collect every ``reaction_hash`` referenced by the routes in ``routes_path``."""
    with gzip.open(routes_path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)

    hashes: set[str] = set()

    def walk(node: dict) -> None:
        metadata = node.get("metadata") or {}
        reaction_hash = metadata.get("reaction_hash") or metadata.get("hash")
        if reaction_hash:
            hashes.add(str(reaction_hash))
        for child in node.get("children") or []:
            walk(child)

    trees = data if isinstance(data, list) else [data]
    for tree in trees:
        walk(tree)
    return hashes


def build(raw_path: Path, route_paths: list[Path], output_path: Path) -> None:
    wanted: set[str] = set()
    for routes_path in route_paths:
        wanted |= route_reaction_hashes(routes_path)
    print(f"Route reaction hashes wanted: {len(wanted)}")

    found: set[str] = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        gzip.open(raw_path, "rt", encoding="utf-8", newline="") as src,
        gzip.open(output_path, "wt", encoding="utf-8", newline="") as dst,
    ):
        reader = csv.DictReader(src, delimiter="\t")
        if tuple(reader.fieldnames or ()) != RAW_COLUMNS:
            raise ValueError(f"Unexpected raw template columns: {reader.fieldnames}")
        writer = csv.writer(dst, delimiter="\t")
        writer.writerow(OUTPUT_COLUMNS)
        for row in reader:
            reaction_hash = row["reaction_hash"]
            if reaction_hash in wanted and reaction_hash not in found:
                writer.writerow((reaction_hash, row["retro_template"]))
                found.add(reaction_hash)

    missing = wanted - found
    print(f"Templates written: {len(found)}")
    if missing:
        raise SystemExit(
            f"Raw library is missing {len(missing)} route reaction hashes; "
            "cannot build a complete subset."
        )
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=HERE / "raw" / "uspto_raw_template_library.csv.gz",
    )
    parser.add_argument(
        "--routes",
        type=Path,
        nargs="+",
        default=[
            HERE / "processed" / "n1-routes.json.gz",
            HERE / "processed" / "n5-routes.json.gz",
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "processed" / "used14k-reactions.tsv.gz",
    )
    args = parser.parse_args()
    if not args.raw.exists():
        raise SystemExit(
            f"Raw library not found at {args.raw}. "
            "Run `make paroutes-raw-template-library` first."
        )
    build(args.raw, list(args.routes), args.output)


if __name__ == "__main__":
    main()

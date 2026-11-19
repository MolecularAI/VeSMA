#!/usr/bin/env python3
"""Build the committed ``raw68k-templates.tsv.gz`` unique-template library.

Provenance / regeneration script. The full PaRoutes raw template library
(``raw/uspto_raw_template_library.csv.gz``, ~116 MB, 819,797 rows, tab-delimited
despite the ``.csv`` name) is downloaded on demand and gitignored. Those 819,797 rows
are per-*reaction* (one ``reaction_hash`` each) and collapse to only 67,557 distinct
``retro_template`` SMARTS strings — the "raw68k" rule library. (They collapse further,
to 42,554 distinct ``template_hash`` values, which is the curated uspto42k count; the
hash canonicalises several raw SMARTS onto one template, so raw68k is the larger, less
curated library.)

We vendor those 67,557 unique templates, dropping ``reaction_hash`` (a rule library is
reaction-agnostic) and the bulky ``reaction_smiles``. The result is ~4.6 MB and
committed, so route replay and recovery can run against the full raw library right after
checkout without the 116 MB download. Route-recovery alignment is reconstructed by joining
the reference hashes retained in ``used14k-reactions.tsv.gz`` back to these rules by SMARTS.

Dedup is by the exact ``retro_template`` string; every distinct raw SMARTS is one rule.
``template_hash`` and ``library_occurence`` (how many raw reaction rows share the SMARTS)
are kept for provenance only. Output is sorted by ``retro_template`` so regeneration is
deterministic regardless of the raw library's row order.

Run (raw library must be present; ``make paroutes-raw-template-library`` fetches it):

    python tests/data/external/paroutes/build_raw68k_templates.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from datetime import date
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
OUTPUT_COLUMNS = ("retro_template", "template_hash", "library_occurence")

# csv keeps a long field per row (retrosynthesis SMARTS); lift the default 128 KB cap.
csv.field_size_limit(64 * 1024 * 1024)


def build(raw_path: Path, output_path: Path, manifest_path: Path) -> None:
    first_hash: dict[str, str] = {}
    occurrence: dict[str, int] = {}
    total_rows = 0
    with gzip.open(raw_path, "rt", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src, delimiter="\t")
        if tuple(reader.fieldnames or ()) != RAW_COLUMNS:
            raise ValueError(f"Unexpected raw template columns: {reader.fieldnames}")
        for row in reader:
            total_rows += 1
            retro_template = row["retro_template"]
            if retro_template not in first_hash:
                first_hash[retro_template] = row["template_hash"]
            occurrence[retro_template] = occurrence.get(retro_template, 0) + 1

    templates = sorted(first_hash)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as dst:
        writer = csv.writer(dst, delimiter="\t")
        writer.writerow(OUTPUT_COLUMNS)
        for retro_template in templates:
            writer.writerow(
                (retro_template, first_hash[retro_template], occurrence[retro_template])
            )

    manifest = {
        "file": output_path.name,
        "source": raw_path.name,
        "raw_reaction_rows": total_rows,
        "unique_retro_templates": len(templates),
        "unique_template_hashes": len(set(first_hash.values())),
        "dedup_key": "retro_template",
        "columns": list(OUTPUT_COLUMNS),
        "note": (
            "Full raw USPTO template library deduplicated to unique retro_template "
            "SMARTS (raw68k). reaction_hash is dropped here; route recovery maps the "
            "reference hashes retained in used14k-reactions.tsv.gz back by SMARTS."
        ),
        "computed_on": date.today().isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Raw reaction rows read: {total_rows}")
    print(f"Unique retro_template templates written: {len(templates)}")
    print(f"Unique template_hash values: {manifest['unique_template_hashes']}")
    print(f"Wrote {output_path}")
    print(f"Wrote {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=HERE / "raw" / "uspto_raw_template_library.csv.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "processed" / "raw68k-templates.tsv.gz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=HERE / "processed" / "raw68k-templates.provenance.json",
    )
    args = parser.parse_args()
    if not args.raw.exists():
        raise SystemExit(
            f"Raw library not found at {args.raw}. "
            "Run `make paroutes-raw-template-library` first."
        )
    build(args.raw, args.output, args.manifest)


if __name__ == "__main__":
    main()

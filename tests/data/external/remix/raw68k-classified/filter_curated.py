"""Project the classified templates down to the rows carrying one verdict.

    make raw68k-curated

A one-column filter, deliberately not committed: the input is versioned and this is a pure
projection of it, so tracking the output would be tracking a function of a tracked file.
Rebuild it after checkout; it takes about a second.

The result is a gzipped TSV with a `retro_template` column, which is what every rule-set
reader in this repository expects (`bench_host_aostar.read_templates`,
`paroutes_reference.raw68k_rule_index`). So it drops into a vertsy run as a rule set without
further conversion -- the extra columns ride along and are ignored.

`--verdict` takes any value the classified table uses, so this is not only "the good ones":
`--verdict unconfirmed` gives the templates ReactionClassifier could not symbolically confirm,
which is the set to look at when asking what curation actually removed.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent
PROCESSED = DATASET_DIR / "processed"

#: gzip stores an mtime, so pinning it to 0 keeps the projection byte-reproducible -- the same
#: reason the build that writes its input does.
GZIP_MTIME = 0


def project(source: Path, target: Path, verdict: str) -> tuple[int, int]:
    """Copy the rows whose `verdict` matches, returning (kept, total)."""

    with gzip.open(source, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or ())

        if "verdict" not in fieldnames:
            raise ValueError(f"{source.name} has no `verdict` column: {fieldnames}")

        rows = list(reader)

    kept = [row for row in rows if row["verdict"] == verdict]

    if not kept:
        # An empty rule set is never what the caller wanted, and a typo in --verdict is the
        # likely cause; silently writing a header-only file would surface far downstream.
        raise ValueError(
            f"no rows with verdict {verdict!r}; present: {sorted({r['verdict'] for r in rows})}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)

    # `gzip.open` has no `mtime`; only `GzipFile` does, so the text layer is added by hand.
    with gzip.GzipFile(target, "wb", compresslevel=9, mtime=GZIP_MTIME) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(kept)

    return len(kept), len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(PROCESSED / "raw68k-classified.tsv.gz"))
    parser.add_argument("--verdict", default="agree")
    parser.add_argument(
        "--output", default=None, help="default processed/raw68k-<verdict>.tsv.gz"
    )
    args = parser.parse_args()

    target = (
        Path(args.output)
        if args.output
        else PROCESSED / f"raw68k-{args.verdict}.tsv.gz"
    )
    kept, total = project(Path(args.source), target, args.verdict)

    print(f"{kept:,} of {total:,} rows with verdict {args.verdict!r} -> {target.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

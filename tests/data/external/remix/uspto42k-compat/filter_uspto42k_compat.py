from __future__ import annotations

import argparse
import csv
import gzip
import io
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rdkit.Chem import rdChemReactions

from vesma import SmartsError
from vesma.reaction import parse_reaction_smarts

# Remix dataset: PaRoutes USPTO unique templates filtered to the rows usable by
# both RDKit reaction parsing and Vesma's reaction/GMol parser.
DATASET_DIR = Path(__file__).resolve().parent
EXTERNAL_DIR = DATASET_DIR.parent.parent
DEFAULT_SOURCE = (
    EXTERNAL_DIR / "paroutes" / "processed" / "uspto_unique_templates.csv.gz"
)
DEFAULT_OUTPUT = DATASET_DIR / "processed" / "uspto42k-compat.csv.gz"
ATOM_CAP = 64
SOURCE_COLUMNS = (
    "template_code",
    "retro_template",
    "template_hash",
    "classification",
    "library_occurence",
)
OUTPUT_COLUMNS = ("retro_template",)


@dataclass(frozen=True)
class FilterResult:
    retro_template: str
    lhs_atom_count: int | None
    rejection: str | None


def main() -> None:
    args = parse_args()
    kept, rejected, lhs_atom_count_max = filter_template_file(
        source=args.source,
        output=args.output,
        atom_cap=args.atom_cap,
    )
    rejected_count = sum(rejected.values())
    print(f"scanned {kept + rejected_count} templates")
    print(f"kept {kept} compatible templates -> {args.output}")
    print(f"rejected {rejected_count} templates: {dict(sorted(rejected.items()))}")
    print(f"kept LHS atom count max: {lhs_atom_count_max}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--atom-cap", type=int, default=ATOM_CAP)
    return parser.parse_args()


def filter_template_file(
    *,
    source: Path,
    output: Path,
    atom_cap: int,
) -> tuple[int, Counter[str], int]:
    if atom_cap <= 0:
        raise ValueError("atom cap must be positive.")

    rejected: Counter[str] = Counter()
    kept_templates: list[str] = []
    kept_lhs_atom_counts: list[int] = []

    with gzip.open(
        source, "rt", encoding="utf-8", errors="replace", newline=""
    ) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
            raise ValueError(f"Unexpected template columns: {reader.fieldnames}")

        for row in reader:
            result = filter_template_row(row, atom_cap)
            if result.rejection is not None:
                rejected[result.rejection] += 1
                continue
            if result.lhs_atom_count is None:
                raise ValueError("Compatible template result is missing atom count.")
            kept_templates.append(result.retro_template)
            kept_lhs_atom_counts.append(result.lhs_atom_count)

    write_rows(output, kept_templates)
    return len(kept_templates), rejected, max(kept_lhs_atom_counts, default=0)


def filter_template_row(row: dict[str, str], atom_cap: int) -> FilterResult:
    retro_template = row["retro_template"]

    try:
        pattern = parse_reaction_smarts(retro_template)
    except SmartsError:
        return FilterResult(
            retro_template=retro_template,
            lhs_atom_count=None,
            rejection="vesma_parse",
        )

    if pattern.lhs.atom_count > atom_cap:
        return FilterResult(
            retro_template=retro_template,
            lhs_atom_count=pattern.lhs.atom_count,
            rejection="lhs_atom_cap",
        )

    try:
        rdkit_reaction = rdChemReactions.ReactionFromSmarts(retro_template)
    except Exception:
        return FilterResult(
            retro_template=retro_template,
            lhs_atom_count=pattern.lhs.atom_count,
            rejection="rdkit_reaction_parse",
        )
    if rdkit_reaction is None:
        return FilterResult(
            retro_template=retro_template,
            lhs_atom_count=pattern.lhs.atom_count,
            rejection="rdkit_reaction_parse",
        )

    return FilterResult(
        retro_template=retro_template,
        lhs_atom_count=pattern.lhs.atom_count,
        rejection=None,
    )


def write_rows(path: Path, retro_templates: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=OUTPUT_COLUMNS,
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(
                    {"retro_template": retro_template}
                    for retro_template in retro_templates
                )
    tmp.replace(path)


if __name__ == "__main__":
    main()

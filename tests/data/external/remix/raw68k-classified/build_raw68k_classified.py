"""Label every raw68k template, and mark which labels two classifiers agree on.

Remix dataset: the PaRoutes raw68k template library joined to reaction-class labels from
ReactionClassifier and Rxn-INSIGHT, with a per-template verdict saying whether the two
agree. Run via `make raw68k-classified`. Reads only committed files -- no network, no
classifier, seconds to run.

Why this exists: neither AiZynthFinder's nor PaRoutes' template file carries usable
labels. Both have a `classification` column and both fill it with `0.0 Unrecognized` for
every row, so there is no way to tell a well-formed, confidently classified template from
an ambiguous one without deriving the labels first.

Labels are keyed on `template_hash`, so every spelling of one template gets the same
label. That is correct -- they are one transformation -- and it is why coverage is total:
raw68k's hash set and the label files' hash set are identical, so all 67,557 SMARTS are
labelled through 42,554 hashes.
"""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATASET_DIR = Path(__file__).resolve().parent
EXTERNAL_DIR = DATASET_DIR.parent.parent
PAROUTES = EXTERNAL_DIR / "paroutes" / "processed"
VENDORED = DATASET_DIR / "vendored"
PROCESSED = DATASET_DIR / "processed"

UNCLASSIFIED = "0.0 Unrecognized"

#: gzip settings that make the artifacts byte-reproducible.
#:
#: gzip stores an mtime in its header, so the default writer produces a different file --
#: and a different sha256 -- on every run, which would make the manifest's checksums
#: unfalsifiable and the "regenerate and compare" check meaningless. Pinning mtime to 0 is
#: what lets the test assert the hashes.
GZIP = {"method": "gzip", "mtime": 0}

#: Depth at which a ReactionClassifier code is mapped onto a Rxn-INSIGHT name.
#:
#: 2, not 1. Level 1 is too coarse where the taxonomy is itself mixed -- superclass 5
#: lumps Deprotections, Reductions and Protections together, so no single Rxn-INSIGHT name
#: can represent it and its purity sits at 54%. Level 2 splits 5.1/5.2/5.3 and lifts
#: weighted purity to 0.84, taking survivors from 55.9% to 59.7% of hashes. Level 3 was
#: not tried: it would fragment the mapping into hundreds of classes, most with too few
#: templates to establish a majority.
MAPPING_DEPTH = 2

OUTPUT_COLUMNS = (
    "retro_template",
    "template_hash",
    "library_occurence",
    "rc_code",
    "rc_agreement",
    "rc_verified_votes",
    "ri_name",
    "mapped_ri_name",
    "zen_code",
    "verdict",
    "verdict_reason",
    "label_source",
    "in_uspto42k",
    "in_used14k",
)

#: Every value `verdict` can take. Pinned by the test, so adding one is a deliberate act.
VERDICTS = (
    "agree",
    "unconfirmed",
    "mapping-conflict",
    "zenodo-conflict",
    "unclassified",
)


@dataclass(frozen=True)
class Counts:
    """What the build saw, for the provenance manifest and the README."""

    smarts: int
    hashes: int
    labelled_hashes: int
    confirmed_hashes: int
    agree_hashes: int
    agree_smarts: int
    agree_occurrence: float
    manual: int
    mapping_classes: int
    mapping_purity: float


def read_labels(path: Path) -> pd.DataFrame:
    """One classifier's audit output: hash, label, and how its sampled reactions voted."""

    records = json.loads(gzip.open(path, "rt").read())

    return pd.DataFrame.from_records(records)


def truncate(code: str, depth: int) -> str:
    """The first `depth` components of a hierarchical code."""

    return ".".join(str(code).split(".")[:depth])


def build_mapping(confirmed: pd.DataFrame, depth: int) -> pd.DataFrame:
    """The Rxn-INSIGHT name each ReactionClassifier class most often coincides with.

    Derived from the confirmed subset only, because that is where the two vocabularies are
    known to be near-nested; including the unconfirmed neural fallback is what made an
    earlier attempt look incompatible.

    `purity` is the share of the class that takes the majority name, and it is written out
    rather than used as a threshold: a class at 36% is not evidence that the mapping is
    wrong, it is usually a taxonomy catch-all ("Other FGI", "Other Oxidation") that no
    single name can represent. The README lists them rather than silently dropping them.
    """

    level = confirmed.assign(
        rc_class=confirmed.rc_code.map(lambda c: truncate(c, depth))
    )
    grouped = level.groupby("rc_class").ri_name

    return pd.DataFrame(
        {
            "rc_class": grouped.agg(lambda s: s.value_counts().idxmax()).index,
            "ri_name": grouped.agg(lambda s: s.value_counts().idxmax()).values,
            "purity": grouped.agg(lambda s: s.value_counts().iloc[0] / len(s))
            .round(4)
            .values,
            "templates": grouped.size().values,
        }
    ).sort_values("rc_class")


def verdict_of(row: pd.Series) -> tuple[str, str]:
    """Whether this template's two labels agree, and why not when they do not.

    Order matters: the checks run cheapest-first and the first failure is the one reported,
    so "unconfirmed" beats "mapping-conflict" for a template that is both. A template the
    Zenodo corpus covers *and* contradicts is separated out rather than folded into
    mapping-conflict, because the two mean different things -- one is two vocabularies
    disagreeing, the other is the classifier's own authors disagreeing with our rerun.
    """

    if row.label_source == "manual":
        return "agree", "hand-assigned; see manual-overrides.tsv"

    if pd.isna(row.rc_code) or pd.isna(row.ri_name):
        return "unclassified", "absent from one or both label files"

    if str(row.rc_code).startswith("0.0") or str(row.ri_name).startswith("0.0"):
        return "unclassified", "one or both classifiers returned Unrecognized"

    if not row.rc_verified_votes:
        return "unconfirmed", "no ReactionClassifier vote passed symbolic confirmation"

    if row.mapped_ri_name != row.ri_name:
        return (
            "mapping-conflict",
            f"class maps to {row.mapped_ri_name!r}, observed {row.ri_name!r}",
        )

    if isinstance(row.zen_code, str) and truncate(
        row.zen_code, MAPPING_DEPTH
    ) != truncate(row.rc_code, MAPPING_DEPTH):
        return "zenodo-conflict", f"published corpus says {row.zen_code!r}"

    return "agree", "confirmed, and both vocabularies consistent"


def build(depth: int) -> Counts:
    """Join every label source onto raw68k, decide a verdict, and write the artifacts."""

    raw = pd.read_csv(PAROUTES / "raw68k-templates.tsv.gz", sep="\t")
    uspto42k = pd.read_csv(PAROUTES / "uspto_unique_templates.csv.gz", sep="\t")
    used14k = pd.read_csv(PAROUTES / "used14k-reactions.tsv.gz", sep="\t")

    rc = read_labels(VENDORED / "reactionclassifier-labels.json.gz").rename(
        columns={
            "classification": "rc_code",
            "agreement": "rc_agreement",
            "verified_votes": "rc_verified_votes",
        }
    )[["template_hash", "rc_code", "rc_agreement", "rc_verified_votes"]]

    ri = read_labels(VENDORED / "rxn-insight-labels.json.gz").rename(
        columns={"classification": "ri_name"}
    )[["template_hash", "ri_name"]]

    zen = pd.read_csv(VENDORED / "zenodo-labels.tsv.gz", sep="\t")[
        ["template_hash", "zen_code"]
    ]

    frame = raw.merge(rc, on="template_hash", how="left")
    frame = frame.merge(ri, on="template_hash", how="left")
    frame = frame.merge(zen, on="template_hash", how="left")

    frame["in_uspto42k"] = frame.retro_template.isin(set(uspto42k.retro_template))
    frame["in_used14k"] = frame.retro_template.isin(set(used14k.retro_template))

    overrides = pd.read_csv(DATASET_DIR / "manual-overrides.tsv", sep="\t", comment="#")
    manual = dict(zip(overrides.template_hash, overrides.code))
    manual_names = dict(zip(overrides.template_hash, overrides.name))

    frame["label_source"] = frame.template_hash.map(
        lambda h: "manual" if h in manual else "classifier"
    )
    frame.loc[frame.label_source == "manual", "rc_code"] = frame.template_hash.map(
        manual
    )
    frame.loc[frame.label_source == "manual", "ri_name"] = frame.template_hash.map(
        manual_names
    )

    hashes = frame.drop_duplicates("template_hash")
    labelled = hashes[hashes.rc_code.notna() & hashes.ri_name.notna()]
    confirmed = labelled[
        (labelled.rc_verified_votes.fillna(0) > 0)
        & ~labelled.rc_code.astype(str).str.startswith("0.0")
        & ~labelled.ri_name.astype(str).str.startswith("0.0")
    ]

    mapping = build_mapping(confirmed, depth)
    names = dict(zip(mapping.rc_class, mapping.ri_name))
    frame["mapped_ri_name"] = frame.rc_code.map(
        lambda c: names.get(truncate(c, depth)) if isinstance(c, str) else None
    )

    decided = frame.apply(verdict_of, axis=1, result_type="expand")
    frame["verdict"] = decided[0]
    frame["verdict_reason"] = decided[1]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(PROCESSED / "class-mapping.tsv", sep="\t", index=False)

    out = frame[list(OUTPUT_COLUMNS)].sort_values("retro_template")
    out.to_csv(
        PROCESSED / "raw68k-classified.tsv.gz", sep="\t", index=False, compression=GZIP
    )

    # The `verdict == "agree"` projection is NOT written here. It is a pure function of the
    # file above, so `filter_curated.py` produces it on demand (`make raw68k-curated`) and it
    # is gitignored: committing it would mean committing a function of a committed file, and
    # re-checksumming it on every change to the verdict rule.
    curated = out[out.verdict == "agree"]

    agreeing = hashes.template_hash.isin(set(curated.template_hash))

    return Counts(
        smarts=len(frame),
        hashes=len(hashes),
        labelled_hashes=len(labelled),
        confirmed_hashes=len(confirmed),
        agree_hashes=int(agreeing.sum()),
        agree_smarts=len(curated),
        agree_occurrence=round(
            float(curated.library_occurence.sum())
            / float(frame.library_occurence.sum()),
            4,
        ),
        manual=int((frame.label_source == "manual").sum()),
        mapping_classes=len(mapping),
        mapping_purity=round(
            float((mapping.purity * mapping.templates).sum() / mapping.templates.sum()),
            4,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=MAPPING_DEPTH)
    args = parser.parse_args()

    counts = build(args.depth)

    manifest = {
        "files": ["raw68k-classified.tsv.gz", "class-mapping.tsv"],
        "projections": "raw68k-<verdict>.tsv.gz, rebuilt by filter_curated.py; not committed",
        "sources": {
            "templates": "../../paroutes/processed/raw68k-templates.tsv.gz",
            "reactionclassifier": "vendored/reactionclassifier-labels.json.gz",
            "rxn_insight": "vendored/rxn-insight-labels.json.gz",
            "published_corpus": "vendored/zenodo-labels.tsv.gz",
            "overrides": "manual-overrides.tsv",
        },
        "mapping_depth": args.depth,
        "counts": counts.__dict__,
    }
    (PROCESSED / "raw68k-classified.provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print(f"raw68k SMARTS {counts.smarts:,} over {counts.hashes:,} hashes")
    print(f"  labelled by both classifiers : {counts.labelled_hashes:,} hashes")
    print(f"  symbolically confirmed       : {counts.confirmed_hashes:,} hashes")
    print(
        f"  mapping                      : {counts.mapping_classes} classes, "
        f"weighted purity {counts.mapping_purity}"
    )
    print(
        f"  agree                        : {counts.agree_hashes:,} hashes, "
        f"{counts.agree_smarts:,} SMARTS, {100 * counts.agree_occurrence:.1f}% of occurrence"
    )
    print(f"  hand-assigned rows           : {counts.manual}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

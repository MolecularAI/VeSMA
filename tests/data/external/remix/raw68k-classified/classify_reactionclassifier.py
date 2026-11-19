"""Classify AiZynthFinder's USPTO templates, producing `uspto_templates_reactionclassifier.csv.gz`.

This is the **USPTO side** of the reaction-class alignment; `classify_real_reactions.py` is the
REAL side and `make_reaction_class_set.py` joins them.  `REACTION_CLASSES.md` is the argument.

**Why this file has to exist at all.**  AiZynthFinder's own `uspto_templates.csv.gz` ships a
`classification` column that is **100% `0.0 Unrecognized`**, and so does PaRoutes' raw library,
which is where the reactions come from.  `ReactionClassMembershipScorer` matches on that column,
so until it is filled the scorer is a no-op that silently scores every route as in-set.  Nothing
in AiZynthFinder or PaRoutes flags this: the column is present and populated with a valid label.

**A template is labelled from real reactions, by majority vote.**  Up to three of the USPTO
reactions that PaRoutes recorded under that template hash, shortest first, and the label is the
most common of the three -- which is what NameRxn's own pipeline does for a template.  Shortest
first because within one template the shorter reaction is the same transformation carrying less
unrelated scaffold, so it is the cleaner example; three rather than one because the agreement
rate is then measurable, and it is reported.

**Earlier, worse attempts, recorded so the improvement is traceable.**  Joining through
PaRoutes' *route* files covered only 27% of templates, because uspto42k is deliberately held out
of the routes.  Adding synthetically instantiated templates -- fire each retro SMARTS on a
minimal molecule and classify the result -- reached 48.8% by count and 85.4% weighted by
`library_occurence`, but classifies a transformation stripped of its scaffold.  The raw library
makes both moot: its 819,797 reactions cover **all 42,554** curated templates.

    make raw68k-classified-classify

which expands to, from the repository root:

    OMP_NUM_THREADS=1 UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-project \
        --with 'reactionclassifier==0.1.1' --python ./.venv/bin/python \
        python tests/data/external/remix/raw68k-classified/classify_reactionclassifier.py

~8 min for the classification plus ~2 min to read the raw library.  `OMP_NUM_THREADS=1` because
the neural gate is torch and will otherwise take every core on the machine.

**Vendored** from the REINVENT4 script that produced the committed labels, repointed at this
repository's paths.  It is not on the build path: `build_raw68k_classified.py` reads the
committed `vendored/*-labels.json.gz` instead, so a fresh checkout never runs a classifier.
Kept so the vendored labels can be re-derived rather than merely trusted.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402  -- after warnings.filterwarnings, deliberately
from rdkit import Chem, RDLogger  # noqa: E402  -- rdkit warns loudly at import time

RDLogger.DisableLog("rdApp.*")

DATASET_DIR = Path(__file__).resolve().parent
PAROUTES = DATASET_DIR.parent.parent / "paroutes"

#: PaRoutes 2.0's raw template library: 819,797 USPTO reactions, each carrying the
#: `template_hash` of the curated template it instantiates.  Tab-delimited despite the `.csv`
#: name, ~106 MB gzipped.  Zenodo 10.5281/zenodo.7341155.  Gitignored: fetch with
#: `make raw68k-classified-download`.
RAW = PAROUTES / "raw" / "uspto_raw_template_library.csv.gz"

#: The curated one-spelling-per-hash template table, which supplies the hashes to label.
TEMPLATES = PAROUTES / "processed" / "uspto_unique_templates.csv.gz"

PER_TEMPLATE = 3

#: What an unclassified template gets, and it must stay exactly this string: it is what
#: AiZynthFinder ships, so a reader can tell "we could not label this" from "we never tried".
UNRECOGNIZED = "0.0 Unrecognized"


def load_classifier():
    """`ReactionClassifier().classify`, or a message saying how to get one.

    Imported inside the function so `--help` works in the plain venv, and called before the
    raw library is read so a missing dependency fails in a second rather than after two
    minutes of pandas.
    """

    try:
        from reactionclassifier import ReactionClassifier
    except ImportError as exc:  # pragma: no cover -- environment, not logic
        raise SystemExit(
            "reactionclassifier is not importable.  It is not a REINVENT dependency; run this "
            "script under\n\n"
            "    uv run --no-project --with 'reactionclassifier==0.1.1' "
            "--python ./.venv/bin/python ...\n\n"
            f"({exc})"
        ) from exc

    classifier = ReactionClassifier()

    def classify(reaction_smiles: str) -> tuple[str | None, bool]:
        """The code, and whether it was template-confirmed rather than guessed.

        `reaction_code` is `None` when the symbolic stage could not confirm the gate's
        prediction; the prediction is used anyway, and the confirmation rate is reported, so a
        set built largely from guesses is visible as such rather than implied to be verified.
        """

        result = classifier.classify(reaction_smiles)

        return (result.reaction_code or getattr(result, "neural_code", None)), bool(
            result.reaction_code
        )

    return classify


def two_part(reaction_smiles: str) -> str:
    """`reactants>agents>product` to `reactants>>product`.

    PaRoutes writes the three-part form.  Both classifiers were validated on the two-part one,
    so the agent field is dropped rather than assumed harmless: a solvent or catalyst on the
    reactant side is a different transformation as far as a template match is concerned.
    """

    parts = reaction_smiles.split(">")

    return f"{parts[0]}>>{parts[-1]}" if len(parts) == 3 else reaction_smiles


def strip_maps(reaction_smiles: str) -> str | None:
    """The same reaction with its atom-map numbers removed, or `None` if it will not parse.

    **Measured, and not cosmetic.**  ReactionClassifier's symbolic confirmation fires on 0 of
    60 mapped reactions and 48 of 60 once the maps are gone, because its templates are matched
    against unmapped structures.  Leaving the maps in silently demotes the tool to its
    unconfirmed neural head -- which is exactly the regime `REACTION_CLASSES.md` shows to be
    unreliable.  This one step moved confirmation across the whole annotation from 0% to 67%.
    """

    sides = []

    for side in reaction_smiles.split(">>"):
        fragments = []

        for fragment in side.split("."):
            mol = Chem.MolFromSmiles(fragment)

            if mol is None:
                return None

            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)

            fragments.append(Chem.MolToSmiles(mol))

        sides.append(".".join(fragments))

    return ">>".join(sides)


def pick_reactions(raw: Path, curated: set[str], per_template: int) -> list[dict]:
    """Up to `per_template` reactions per curated template, shortest first."""

    print(f"reading {raw}")
    frame = pd.read_csv(raw, sep="\t", usecols=["reaction_smiles", "template_hash"])
    frame = frame.dropna(subset=["reaction_smiles"])

    # Sort by length once and take the first N per hash, rather than grouping and sorting
    # within each of 42,554 groups.
    frame = frame.assign(length=frame.reaction_smiles.str.len()).sort_values("length")

    picked: dict[str, list[str]] = defaultdict(list)

    for template_hash, reaction_smiles in zip(
        frame.template_hash.values, frame.reaction_smiles.values
    ):
        if len(picked[template_hash]) < per_template:
            picked[template_hash].append(reaction_smiles)

    rows = [
        {"template_hash": key, "reactions": value}
        for key, value in picked.items()
        if key in curated
    ]

    print(f"raw reactions: {len(frame)}, templates in raw: {len(picked)}")
    print(
        f"curated templates covered: {len(rows)}/{len(curated)} ({len(rows) / len(curated):.1%})"
    )
    print(f"reactions to classify: {sum(len(row['reactions']) for row in rows)}")

    return rows


def vote(reactions: list[str], classify) -> tuple[str | None, int, float, int]:
    """The majority label over one template's reactions, and how much they agreed."""

    votes: list[str] = []
    confirmed = 0

    for reaction_smiles in reactions:
        clean = strip_maps(two_part(reaction_smiles))

        if clean is None:
            continue

        try:
            code, was_confirmed = classify(clean)
        except Exception:
            # One reaction the classifier chokes on must not cost the template its other two.
            continue

        if code:
            votes.append(code)
            confirmed += bool(was_confirmed)

    if not votes:
        return None, 0, 0.0, 0

    code, count = Counter(votes).most_common(1)[0]

    return code, len(votes), count / len(votes), confirmed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw", default=str(RAW), help=f"PaRoutes raw template library [{RAW}]"
    )
    ap.add_argument(
        "--templates",
        default=str(TEMPLATES),
        help=f"AiZynthFinder templates to annotate [{TEMPLATES}]",
    )
    ap.add_argument("--per-template", type=int, default=PER_TEMPLATE)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="annotate only this many templates, for a smoke test",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="annotated templates copy [raw/uspto_templates_reactionclassifier.csv.gz]",
    )
    args = ap.parse_args()

    templates = Path(args.templates)
    # Into gitignored `raw/`: this annotated copy is a redundant intermediate. Everything the
    # build needs is in the labels file below, which is what gets committed.
    out = (
        Path(args.out)
        if args.out
        else DATASET_DIR / "raw" / "uspto_templates_reactionclassifier.csv.gz"
    )

    if out == templates:
        # Overwriting the input would destroy the only unannotated copy, and every re-run after
        # that would annotate an already-annotated file.
        raise SystemExit(f"refusing to overwrite the source templates file {templates}")

    classify = load_classifier()

    frame = pd.read_csv(templates, sep="\t")
    unlabelled = (
        (frame.classification != UNRECOGNIZED).sum()
        if "classification" in frame
        else -1
    )
    print(
        f"{len(frame)} templates in {templates.name}, "
        f"{unlabelled} already carrying a classification"
    )

    rows = pick_reactions(Path(args.raw), set(frame.template_hash), args.per_template)

    if args.limit:
        rows = rows[: args.limit]
        print(f"--limit {args.limit}: annotating {len(rows)} template(s) only")

    labels: dict[str, str] = {}
    audit: list[dict] = []
    start = time.perf_counter()

    for index, row in enumerate(rows):
        code, votes, agreement, confirmed = vote(row["reactions"], classify)

        if code:
            labels[row["template_hash"]] = code
            audit.append(
                {
                    "template_hash": row["template_hash"],
                    "classification": code,
                    "votes": votes,
                    "agreement": agreement,
                    "verified_votes": confirmed,
                }
            )

        if (index + 1) % 2000 == 0:
            rate = (time.perf_counter() - start) / (index + 1)
            print(
                f"  {index + 1}/{len(rows)}, {len(labels)} labelled, "
                f"eta {(len(rows) - index - 1) * rate / 60:.0f} min",
                flush=True,
            )

    frame["classification"] = frame.template_hash.map(labels).fillna(UNRECOGNIZED)

    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, sep="\t", index=False, compression="gzip")
    # The committed artifact: `template_hash`, `classification`, `votes`, `agreement`,
    # `verified_votes` -- everything `build_raw68k_classified.py` reads. Gzipped, because it is
    # 7.3 MB raw and 1.9 MB compressed.
    audit_path = DATASET_DIR / "vendored" / "reactionclassifier-labels.json.gz"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    # `gzip.open` has no `mtime`; only `GzipFile` does, and pinning it keeps the committed
    # labels byte-reproducible.
    with gzip.GzipFile(audit_path, "wb", compresslevel=9, mtime=0) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8") as handle:
            json.dump(audit, handle)

    done = (frame.classification != UNRECOGNIZED).sum()
    occurrence = frame.library_occurence
    weighted = occurrence[frame.classification != UNRECOGNIZED].sum() / occurrence.sum()
    unanimous = sum(1 for entry in audit if entry["agreement"] == 1.0)
    confirmed = sum(entry["verified_votes"] for entry in audit)
    total = sum(entry["votes"] for entry in audit)

    print(
        f"\nclassified {done}/{len(frame)} ({done / len(frame):.1%} by count, "
        f"{weighted:.1%} by library_occurence)"
    )
    print(f"distinct leaf classes: {frame.classification.nunique() - 1}")
    print(
        f"unanimous over its reactions: {unanimous}/{len(audit)} "
        f"({unanimous / max(len(audit), 1):.0%})"
    )
    print(
        f"symbolically confirmed votes: {confirmed}/{total} ({confirmed / max(total, 1):.0%})"
    )
    print(f"\nwrote {out} in {(time.perf_counter() - start) / 60:.1f} min")
    print(f"wrote {audit_path}")
    print("next: docs/synthinvent/scripts/make_reaction_class_set.py --depth 1")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

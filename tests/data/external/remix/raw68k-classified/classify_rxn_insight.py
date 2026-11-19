"""Fill AiZynthFinder's `classification` column without NameRxn.

The public AiZynthFinder release ships templates whose `classification` is
``"0.0 Unrecognized"`` on every row, and the reaction nodes of every route inherit
that. Three things downstream depend on the field: SynthSense's `cazp` endpoint reads
a "reaction class membership" score, `RoutePopularityEndpoint` builds its route
signature from a frozenset of reaction classes, and `RouteSimilarityEndpoint` builds
the tree string it compares with APTED the same way. With one constant class, the
popularity signature is identical for every route and the edit distance sees only
tree shape, so both are inert.

`NAMERXN_SETUP.md` solves this with NextMove's NameRxn, which is commercial. This
does the same job with an open classifier, and produces a file with the same schema,
so it is a drop-in for the `expansion` entry of an AiZynthFinder config.

Two classifiers are supported and they are not equivalent -- see PUBLIC_DATA_SETUP.md
for the measured comparison:

  reactionclassifier   MIT, neuro-symbolic, hierarchical codes ("2.1.2.1"), 8 ms
  rxn-insight          rule-based, class names ("Acylation"), 99 ms

Reactions come from the PaRoutes raw template library, which carries a
`template_hash` on every one of its 819,797 USPTO reactions and covers all 42,554
curated templates. Each template is labelled by majority vote over up to three of its
own reactions, which is what the NameRxn pipeline does.

Usage:

    python annotate_templates.py \\
        --templates  uspto_templates.csv.gz \\
        --raw-library uspto_raw_template_library.csv.gz \\
        --classifier reactionclassifier \\
        --output     uspto_templates_classified.csv.gz

The classifier must be importable, so run this inside a virtualenv that has it --
both pin dependencies that conflict with REINVENT's, which is why they live in their
own environments and this is a standalone script rather than a plugin.
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

UNCLASSIFIED = "0.0 Unrecognized"


def load_classifier(name: str):
    if name == "rxn-insight":
        from rxn_insight.reaction import Reaction

        def classify(smiles: str):
            return (Reaction(smiles).get_reaction_info().get("CLASS") or None), None

    else:
        from reactionclassifier import ReactionClassifier

        clf = ReactionClassifier()

        def classify(smiles: str):
            result = clf.classify(smiles)
            # `reaction_code` is set only when a retrosynthetic template confirms the
            # neural guess; `neural_code` is the unverified fallback. Both are used,
            # and which one it was is recorded so a caller can filter on it.
            return (result.reaction_code or getattr(result, "neural_code", None)), bool(
                result.reaction_code
            )

    return classify


def normalise(smiles: str) -> str | None:
    """Two-part, unmapped reaction SMILES -- both parts matter.

    PaRoutes writes ``reactants>agents>product``; the classifiers were validated on
    the two-part form. And the atom maps have to go: ReactionClassifier matches its
    confirming templates against unmapped structures, so with maps left in its
    symbolic check fired on 0 of 60 sampled reactions against 48 of 60 without them.
    That is the difference between a verified label and a bare neural guess, and it
    is invisible in the output.
    """
    parts = smiles.split(">")
    smiles = f"{parts[0]}>>{parts[-1]}" if len(parts) == 3 else smiles
    sides = []
    for side in smiles.split(">>"):
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


def reactions_per_template(raw_library: str, keep: set[str], per_template: int):
    """Up to `per_template` reactions for each wanted template, shortest first.

    Within one template the shorter reactions are the same transformation carrying
    less unrelated scaffold, which is the cleaner thing to classify.
    """
    frame = pd.read_csv(
        raw_library, sep="\t", usecols=["reaction_smiles", "template_hash"]
    )
    frame = frame.dropna(subset=["reaction_smiles"])
    frame = frame.assign(length=frame.reaction_smiles.str.len()).sort_values("length")

    picked: dict[str, list[str]] = defaultdict(list)
    for template, smiles in zip(
        frame.template_hash.values, frame.reaction_smiles.values
    ):
        if template in keep and len(picked[template]) < per_template:
            picked[template].append(smiles)
    return picked


DATASET_DIR = Path(__file__).resolve().parent
PAROUTES = DATASET_DIR.parent.parent / "paroutes"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--templates",
        default=str(PAROUTES / "processed" / "uspto_unique_templates.csv.gz"),
    )
    parser.add_argument(
        "--raw-library",
        default=str(PAROUTES / "raw" / "uspto_raw_template_library.csv.gz"),
        help="gitignored; fetch with `make raw68k-classified-download`",
    )
    parser.add_argument(
        "--output",
        default=str(DATASET_DIR / "raw" / "uspto_templates_rxn-insight.csv.gz"),
        help="annotated templates copy; a redundant intermediate, so gitignored",
    )
    parser.add_argument(
        "--labels",
        default=str(DATASET_DIR / "vendored" / "rxn-insight-labels.json.gz"),
        help="the committed artifact the dataset build actually reads",
    )
    parser.add_argument(
        "--classifier",
        default="rxn-insight",
        choices=("reactionclassifier", "rxn-insight"),
    )
    parser.add_argument(
        "--per-template",
        type=int,
        default=3,
        help="reactions to vote over per template",
    )
    args = parser.parse_args()

    classify = load_classifier(args.classifier)
    templates = pd.read_csv(args.templates, sep="\t")
    picked = reactions_per_template(
        args.raw_library, set(templates.template_hash), args.per_template
    )
    print(
        f"{len(picked)}/{len(templates)} templates have at least one reaction",
        flush=True,
    )

    labels, verified_votes, total_votes = {}, 0, 0
    # Per-template, not just global counters: `agreement` is how the dataset build tells a
    # template whose three reactions agreed from one whose votes were split, and
    # `verified_votes` is what its "symbolically confirmed" filter keys on. Aggregating these
    # away would make the committed labels file unreproducible from this script.
    audit = []
    started = time.perf_counter()
    for done, (template, smiles_list) in enumerate(picked.items(), start=1):
        votes, confirmed = [], 0
        for smiles in smiles_list:
            clean = normalise(smiles)
            if clean is None:
                continue
            try:
                code, verified = classify(clean)
            except Exception:
                continue
            if code:
                votes.append(code)
                total_votes += 1
                confirmed += bool(verified)
                verified_votes += bool(verified)
        if votes:
            winner, count = Counter(votes).most_common(1)[0]
            labels[template] = winner
            audit.append(
                {
                    "template_hash": template,
                    "classification": winner,
                    "votes": len(votes),
                    "agreement": count / len(votes),
                    "verified_votes": confirmed,
                }
            )
        if done % 2000 == 0:
            rate = (time.perf_counter() - started) / done
            print(
                f"  {done}/{len(picked)}  eta {(len(picked) - done) * rate / 60:.0f} min",
                flush=True,
            )

    templates["classification"] = templates.template_hash.map(labels).fillna(
        UNCLASSIFIED
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    templates.to_csv(args.output, sep="\t", index=False, compression="gzip")

    labels_path = Path(args.labels)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    # `gzip.open` has no `mtime`; only `GzipFile` does, and pinning it keeps the committed
    # labels byte-reproducible.
    with gzip.GzipFile(labels_path, "wb", compresslevel=9, mtime=0) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8") as handle:
            json.dump(audit, handle)

    classified = (templates.classification != UNCLASSIFIED).sum()
    occurrence = templates.library_occurence
    weighted = (
        occurrence[templates.classification != UNCLASSIFIED].sum() / occurrence.sum()
    )
    print(
        f"\nclassified {classified}/{len(templates)} "
        f"({classified / len(templates):.1%} by count, {weighted:.1%} by library_occurence)"
    )
    print(f"distinct classes: {templates.classification.nunique() - 1}")
    if total_votes:
        print(
            f"symbolically verified votes: {verified_votes}/{total_votes} "
            f"({verified_votes / total_votes:.0%})"
        )
    print(f"wrote {args.output} in {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()

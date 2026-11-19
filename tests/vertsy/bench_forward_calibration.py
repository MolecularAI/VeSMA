"""Calibrate the forward rule table against reactions that demonstrably worked.

The forward probe has been checked on hand-built cases, where the answer was known because it was
chosen. This measures it against **31,785 recorded literature reactions** instead, and it needs no
search at all: a reference record already carries its precursors *and* its product, so the answer is
known for every one of them without a single template being applied in the retro direction.

Two numbers come out, and they are the two the probe can be wrong in.

**Recall.** Does any forward rule reproduce the recorded product from the recorded precursors? A
reaction the table cannot express is one the probe will read as *unpaired* or *silent* -- charged
nothing, so harmless, but also invisible. This is the honest coverage number for the table against
real chemistry rather than against the retro library it was paired to.

**Precision, which is the number that matters.** For the reactions it *does* reproduce, how many
other products does the same pot admit? A literature reaction worked and was reported, so a step the
probe calls a mixture is a step the probe is wrong about -- or one where the chemist controlled the
selectivity by something a product count cannot see. Either way the rate is the probe's
false-positive rate, and it cannot be estimated from hand-picked examples.

**Independent of the retro library by construction.** The pairing that `sites` normally reads from
`templates-forward.csv` is supplied here by the data: whichever rules reproduce the recorded product
*are* the intended reaction for that record. So this measures the forward table on its own terms and
says nothing about used14k, raw68k or any other retro collection -- which is the point, since those
are what a route comparison would actually be measuring.
"""

from __future__ import annotations

import csv
import gzip
import itertools
import os
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from tests.datadirs import external_data_root
from vertsy.selectivity import template_arities
from vesma import cmol_from_smiles, reaction_patterns_from_smarts
from vesma.cmol import cmol_union
from vesma.cpu_matcher import CpuReactionMatcher
from vesma.keyer import cmol_smiles_keyer

TEMPLATES = external_data_root() / "templates" / "templates-forward.csv"
REFERENCE = (
    external_data_root()
    / "external"
    / "paroutes"
    / "processed"
    / "n1-reference-reactions.tsv.gz"
)
# One pass over a few hundred records answers both questions to a couple of percent, and the whole
# corpus takes minutes rather than days. Raise it to tighten the estimate, not to change it.
RECORDS = int(os.environ.get("FORWARD_CALIBRATION_RECORDS", "").strip() or 400)
# Skip records whose combined precursors exceed what a device batch could hold anyway, so the
# number quoted here is the number the deployed probe would produce.
ATOM_CAP = 64


@dataclass(frozen=True)
class Record:
    """One literature reaction: what went in, what came out, and where it was reported."""

    precursors: tuple[str, ...]
    product: str
    source: str


def records(limit: int) -> Iterator[Record]:
    """Reference reactions as precursor/product SMILES, skipping the ones with no product."""
    with gzip.open(REFERENCE, "rt") as handle:
        for row in itertools.islice(csv.DictReader(handle, delimiter="\t"), limit):
            left, _, right = row["smiles"].partition(">>")
            if not right or not left:
                continue
            yield Record(tuple(left.split(".")), right, row["id"])


def molecule(smiles: str) -> Any:
    built, _info = cmol_from_smiles(smiles)
    return built


@dataclass(frozen=True)
class Outcome:
    """What the forward table did with one record."""

    reproduced: bool
    """Some rule made the recorded product from the recorded precursors."""
    sites: int
    """Distinct products from the rules that reproduced it -- what ``site_probe`` would report."""
    competitors: int
    """Distinct products from the *other* multi-component rules, excluding the intended ones."""
    by_rule: frozenset[int]
    """Which rules reproduced it, so a miss or a false positive can be traced to a row."""


def outcome_of(
    record: Record,
    matcher: CpuReactionMatcher,
    arities: Sequence[int],
) -> Outcome | None:
    """Run every forward rule on one record's pot, and read both channels off the result.

    ``None`` when the record cannot be built or is larger than a device batch holds, so the quoted
    rates describe records the deployed probe would actually have measured.
    """
    try:
        parts = [molecule(smiles) for smiles in record.precursors]
        wanted = cmol_smiles_keyer.encode(molecule(record.product))
    except Exception:
        return None
    if sum(part.atom_count for part in parts) > ATOM_CAP:
        return None
    try:
        pot = cmol_union(parts)
        # One molecule against 79 rules is far below the grain where forking a worker pool pays:
        # the default `-1` spawned ten processes per record and made the sweep an order of
        # magnitude slower than the work in it.
        products, hits = matcher.match([pot], None, n_jobs=1)
    except Exception:
        return None

    rows = np.asarray(hits, dtype=np.int64).reshape(-1, 4)
    if not rows.shape[0]:
        return Outcome(False, 0, 0, frozenset())
    keys = [cmol_smiles_keyer.encode(each) for each in products]
    # An outcome is a whole application's component multiset, the same way the probe counts them.
    components: dict[tuple[int, int], list[Any]] = {}
    for _mol, rule, app, precursor in rows.tolist():
        components.setdefault((rule, app), []).append(keys[precursor])
    outcomes: dict[int, set[tuple[Any, ...]]] = {}
    for (rule, _app), parts_of in components.items():
        outcomes.setdefault(rule, set()).add(tuple(sorted(parts_of, key=repr)))

    intended = frozenset(
        rule
        for rule, made in outcomes.items()
        if any(wanted in outcome for outcome in made)
    )
    sites = {outcome for rule in intended for outcome in outcomes[rule]}
    competitors = {
        outcome
        for rule, made in outcomes.items()
        if rule not in intended and arities[rule] >= 2
        for outcome in made
    }
    return Outcome(bool(intended), len(sites), len(competitors - sites), intended)


def forward_rules() -> tuple[tuple[str, ...], tuple[str, ...]]:
    with TEMPLATES.open() as handle:
        rows = list(csv.DictReader(handle))
    return (
        tuple(row["forward_template"] for row in rows),
        tuple(row["forward_code"] for row in rows),
    )


@pytest.mark.perf
def test_forward_rules_against_reactions_that_worked(capsys) -> None:
    """Measure recall and the false-positive rate, and print the table a reader needs.

    Asserted loosely on purpose. What is worth failing a build over is that the experiment *ran* and
    found the table expressible against real chemistry at all; the rates themselves are a
    measurement to read, not a threshold to defend, and pinning them would turn every honest
    improvement to the table into a broken test.
    """
    smarts, codes = forward_rules()
    arities = template_arities(smarts)
    matcher = CpuReactionMatcher(reaction_patterns_from_smarts(smarts, cache=True))

    considered = skipped = 0
    reproduced = 0
    sites = Counter[int]()
    competitors = Counter[int]()
    credited = Counter[str]()
    for record in records(RECORDS):
        outcome = outcome_of(record, matcher, arities)
        if outcome is None:
            skipped += 1
            continue
        considered += 1
        if not outcome.reproduced:
            continue
        reproduced += 1
        sites[outcome.sites] += 1
        competitors[min(outcome.competitors, 5)] += 1
        for rule in outcome.by_rule:
            credited[codes[rule]] += 1

    with capsys.disabled():
        print(
            f"\n=== forward table against {considered} literature reactions "
            f"({skipped} skipped: unparseable or over the {ATOM_CAP}-atom cap)"
        )
        share = reproduced / considered if considered else 0.0
        print(f"reproduced by some rule: {reproduced}/{considered} ({share:.1%})")
        if reproduced:
            clean = sites.get(1, 0)
            print(
                f"\n  sites -- of the {reproduced} reproduced, how many products the intended"
            )
            print("           rule(s) make. 1 means the probe calls the step clean.")
            for count in sorted(sites):
                rows = sites[count]
                print(
                    f"    {count:>2} product(s): {rows:5}  ({rows / reproduced:5.1%})"
                )
            print(
                f"  => the probe calls {clean / reproduced:.1%} of reactions that worked clean, "
                f"and flags {(reproduced - clean) / reproduced:.1%} as a mixture"
            )
            print(
                "\n  competitors -- other multi-component rules firing on the same pot"
            )
            for count in sorted(competitors):
                rows = competitors[count]
                label = f"{count}" if count < 5 else "5+"
                print(
                    f"    {label:>2} product(s): {rows:5}  ({rows / reproduced:5.1%})"
                )
            print("\n  rules doing the reproducing (top 12):")
            for code, rows in credited.most_common(12):
                print(f"    {code:16s} {rows:5}")

    assert considered > 0, "no reference reaction could be read at all"
    assert reproduced > 0, "the forward table expresses no literature reaction"

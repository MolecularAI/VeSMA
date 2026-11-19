"""Which pairings go silent on real molecules, and how often.

`test_forward_templates.py` composes each pairing against **one** example per forward row, which can
only catch two halves that disagree about *that* example. Silence in a run is per substrate, and it
is common: a pairing whose forward rule assumed an aldehyde where its retro partner emits a ketone
passes every one-sided check and then reports nothing on every real application it meets.

So this runs the paired retro rules over real reference products and asks, per application, whether
*any* of that application's paired forward rules rebuilds the product. The union and not each rule
alone, because that is what :func:`vertsy.selectivity.site_probe` measures -- a forward family split
by amine class or leaving group has exactly one member that applies, and judging each member on its
own would call the whole table broken when nothing is wrong with it.

What comes out is a worklist ordered by volume: the pairings that are *always* silent are defects
until shown otherwise, and the ones that are sometimes silent name a substrate class nobody wrote a
rule for. Both are curation, not scoring -- which is the point, since a silent probe is charged as
low confidence (`vertsy.costs.doubted`) and a broken pairing therefore costs a route real money.
"""

from __future__ import annotations

import csv
import gzip
import itertools
import os
from collections import Counter
from collections.abc import Iterator

import numpy as np
import pytest

from tests.datadirs import external_data_root
from tests.vertsy.bench_host_aostar import (
    TEMPLATE_FILES,
    load_forward_library,
    load_merged_library,
)
from vesma import cmol_from_smiles, reaction_patterns_from_smarts
from vesma.cmol import cmol_union
from vesma.cpu_matcher import CpuReactionMatcher
from vesma.keyer import cmol_smiles_keyer

REFERENCE = (
    external_data_root()
    / "external"
    / "paroutes"
    / "processed"
    / "n1-reference-reactions.tsv.gz"
)
PRODUCTS = int(os.environ.get("SILENT_PAIRING_PRODUCTS", "").strip() or 600)
# The rule sets the forward table was written beside. raw68k has no pairings at all, so including
# it would add tens of thousands of unprobed rules and not one probed application.
RULE_SETS = ("hartenfeller", "medchem", "protecting-groups")
# Beyond what a device batch holds, so the quoted rate describes applications the probe would really
# have measured rather than ones it would have declared unbuildable.
ATOM_CAP = 64


def recorded_products(limit: int) -> Iterator[str]:
    with gzip.open(REFERENCE, "rt") as handle:
        for row in itertools.islice(csv.DictReader(handle, delimiter="\t"), limit):
            _left, _sep, right = row["smiles"].partition(">>")
            if right:
                yield right


@pytest.mark.perf
def test_which_pairings_go_silent_on_real_molecules(capsys) -> None:
    """Measure the silence rate and print the worklist behind it.

    Asserted only loosely, and deliberately. A rate is a measurement to read and act on, not a
    threshold to defend: pinning it would fail the build on every rule added to the forward table,
    including the ones that lower it.
    """
    merged = load_merged_library(
        tuple((name, TEMPLATE_FILES[name]) for name in RULE_SETS)
    )
    library = load_forward_library(merged.sources)
    name_of = {
        rule: source.columns.get("template_code", f"#{rule}")
        for rule, source in enumerate(merged.sources)
    }
    retro = CpuReactionMatcher(reaction_patterns_from_smarts(merged.smarts, cache=True))
    forward = CpuReactionMatcher(
        reaction_patterns_from_smarts(library.smarts, cache=True)
    )
    paired = sorted(library.index)

    seen = Counter[int]()
    silent = Counter[int]()
    wrong = Counter[int]()
    examined = 0
    for smiles in recorded_products(PRODUCTS):
        try:
            product, _info = cmol_from_smiles(smiles)
            precursors, hits = retro.match(
                [product], [(0, rule) for rule in paired], n_jobs=1
            )
        except Exception:
            continue
        rows = np.asarray(hits, dtype=np.int64).reshape(-1, 4)
        if not rows.shape[0]:
            continue
        wanted = cmol_smiles_keyer.encode(product)
        groups: dict[tuple[int, int], list[int]] = {}
        for _mol, rule, app, precursor in rows.tolist():
            groups.setdefault((rule, app), []).append(precursor)
        for (rule, _app), members in groups.items():
            parts = [precursors[index] for index in members]
            if sum(part.atom_count for part in parts) > ATOM_CAP:
                continue
            try:
                pot = cmol_union(parts)
            except Exception:
                continue
            examined += 1
            seen[rule] += 1
            made: set[object] = set()
            for forward_rule in library.index[rule]:
                try:
                    products, _hits = forward.match(
                        [pot], [(0, forward_rule)], n_jobs=1
                    )
                except Exception:
                    continue
                made |= {cmol_smiles_keyer.encode(each) for each in products}
            if not made:
                silent[rule] += 1
            elif wanted not in made:
                wrong[rule] += 1

    total = sum(silent.values())
    with capsys.disabled():
        print(
            f"\n=== {len(paired)} paired retro rules over {PRODUCTS} recorded products"
        )
        print(
            f"{examined} applications probed: {total} silent "
            f"({total / max(examined, 1):.1%}), "
            f"{sum(wrong.values())} rebuilt something other than the recorded product"
        )
        always = [rule for rule in seen if silent[rule] == seen[rule]]
        print(
            f"\nalways silent -- a defect until shown otherwise ({len(always)} rules):"
        )
        for rule, count in seen.most_common():
            if rule in always:
                codes = ", ".join(library.codes[f] for f in library.index[rule])
                print(f"  {count:6d}  {name_of[rule]:22s} -> {codes}")
        print("\nsometimes silent -- a substrate class with no rule (top 12):")
        shown = 0
        for rule, count in seen.most_common():
            if 0 < silent[rule] < count and shown < 12:
                codes = ", ".join(library.codes[f] for f in library.index[rule])
                print(f"  {silent[rule]:5d}/{count:<6d} {name_of[rule]:22s} -> {codes}")
                shown += 1

    assert examined > 0, "no application could be built from any recorded product"
    assert total < examined, "every single application went silent"

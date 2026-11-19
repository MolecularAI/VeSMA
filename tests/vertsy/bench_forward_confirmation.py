"""What changes if every retro application is probed by the *whole* forward table.

`bench_silent_pairings` asks whether any of an application's **paired** forward rules rebuilds the
product. The pairing is a curated column, `probes`, and it is a 1:1-ish claim: this forward row is
that retro row's inverse. The general reading drops the column from the decision -- run every
forward rule on the precursors, confirm the step if *any* of them makes the intended product, and
send everything else to the mixture report. Pairing then stops being a gate and becomes an
expectation, which a run can measure rather than trust.

Three things have to be true for that to be an improvement, and each is measured below.

**It must confirm at least as many steps.** A step the pairing calls silent is charged as low
confidence (`vertsy.costs.doubted`), so every application the general probe rescues is a route that
was paying for a curation gap rather than for its chemistry.

**It must not flood the mixture.** Every forward rule that fires and does *not* make the intended
product becomes a reported peak. The principled filter for those is conditions -- a reaction needs
its reagents, and a rule whose medium the intended reaction does not supply cannot run in that pot
-- so the flood is measured both raw and after `conditions_allow`.

**It must be affordable.** The paired probe runs a handful of `(pot, rule)` pairs per application;
the general one runs the whole table. That is a denser cross-product, which suits a batched matcher
better than the sparse version does, so the ratio is worth measuring rather than assuming.
"""

from __future__ import annotations

import csv
import gzip
import itertools
import os
import pathlib
import time
from collections import Counter
from collections.abc import Iterator, Sequence

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
from vesma.keyer import short_ring_wl_keyer

REFERENCE = (
    external_data_root()
    / "external"
    / "paroutes"
    / "processed"
    / "n1-reference-reactions.tsv.gz"
)
PRODUCTS = int(os.environ.get("FORWARD_CONFIRMATION_PRODUCTS", "").strip() or 400)
REPORT = os.environ.get("FORWARD_CONFIRMATION_REPORT", "").strip()
RULE_SETS = ("hartenfeller", "medchem", "protecting-groups")
ATOM_CAP = 64


def recorded_products(limit: int) -> Iterator[str]:
    with gzip.open(REFERENCE, "rt") as handle:
        for row in itertools.islice(csv.DictReader(handle, delimiter="\t"), limit):
            _left, _sep, right = row["smiles"].partition(">>")
            if right:
                yield right


def made_by(
    forward: CpuReactionMatcher, pot: object, rules: Sequence[int]
) -> dict[object, set[int]]:
    """Product key -> the forward rules that made it, in one batched call.

    One call rather than one per rule, which is how a run would do it and the only way the general
    probe's cost is a fair comparison: the whole point of running the table is that the matcher sees
    a dense set of `(pot, rule)` pairs at once.
    """
    if not rules:
        return {}
    products, hits = forward.match([pot], [(0, rule) for rule in rules], n_jobs=1)
    rows = np.asarray(hits, dtype=np.int64).reshape(-1, 4)
    found: dict[object, set[int]] = {}
    for _mol, rule, _app, product in rows.tolist():
        found.setdefault(short_ring_wl_keyer.encode(products[product]), set()).add(rule)
    return found


@pytest.mark.perf
def test_what_probing_the_whole_forward_table_changes(capsys) -> None:
    """Measured, not asserted beyond sanity: this is a design question, not a threshold."""
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
    everything = tuple(range(len(library.smarts)))

    examined = 0
    confirmed_paired = confirmed_all = 0
    rescued_by = Counter[str]()
    rescued_retro = Counter[str]()
    mixture_raw = mixture_gated = 0
    # Split by the candidate's own molecularity, because `competitor_probe` admits only rules that
    # *state* they need two starting materials -- a unimolecular rule firing on a precursor needs
    # its own reagent, which the pot does not contain. Counting those as peaks measures molecular
    # complexity, not a competition.
    mixture_multi = mixture_multi_gated = 0
    paired_seconds = all_seconds = 0.0

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
        wanted = short_ring_wl_keyer.encode(product)
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

            started = time.perf_counter()
            try:
                narrow = made_by(forward, pot, library.index[rule])
            except Exception:
                narrow = {}
            paired_seconds += time.perf_counter() - started

            started = time.perf_counter()
            try:
                broad = made_by(forward, pot, everything)
            except Exception:
                broad = {}
            all_seconds += time.perf_counter() - started

            confirmed_paired += wanted in narrow
            confirmed_all += wanted in broad
            if wanted in broad and wanted not in narrow:
                rescued_retro[name_of[rule]] += 1
                for forward_rule in broad[wanted]:
                    rescued_by[library.codes[forward_rule]] += 1

            # Every other product is a reported peak. The conditions gate is the principled
            # filter: a candidate needing a medium the intended reaction does not supply cannot
            # run in that pot, whoever else is in it.
            supplied = library.conditions.get(rule, frozenset())
            for key, makers in broad.items():
                if key == wanted:
                    continue
                mixture_raw += 1
                allowed = [m for m in makers if library.requires[m] <= supplied]
                if allowed:
                    mixture_gated += 1
                if any(library.declared[m] >= 2 for m in makers):
                    mixture_multi += 1
                    if any(library.declared[m] >= 2 for m in allowed):
                        mixture_multi_gated += 1

    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)

    emit(
        f"\n=== {len(paired)} paired retro rules, {len(everything)} forward rules, "
        f"{PRODUCTS} recorded products"
    )
    emit(f"{examined} applications probed\n")
    emit(
        f"confirmed by its paired rules : {confirmed_paired:6d} "
        f"({confirmed_paired / max(examined, 1):.1%})"
    )
    emit(
        f"confirmed by the whole table  : {confirmed_all:6d} "
        f"({confirmed_all / max(examined, 1):.1%})"
    )
    emit(
        f"rescued -- silent pairing, real chemistry: {confirmed_all - confirmed_paired}"
    )
    emit(
        f"\nmixture peaks, whole table : {mixture_raw} raw, {mixture_gated} gated "
        f"({mixture_gated / max(mixture_raw, 1):.1%} kept), "
        f"{mixture_gated / max(examined, 1):.1f} per application"
    )
    emit(
        f"mixture peaks, multi only  : {mixture_multi} raw, {mixture_multi_gated} gated "
        f"({mixture_multi_gated / max(mixture_multi, 1):.1%} kept), "
        f"{mixture_multi_gated / max(examined, 1):.1f} per application"
    )
    emit(
        f"\ncost: paired {paired_seconds:.2f}s, whole table {all_seconds:.2f}s "
        f"({all_seconds / max(paired_seconds, 1e-9):.1f}x)"
    )
    emit("\nforward rules that rescued a step they are not paired to (top 12):")
    for code, count in rescued_by.most_common(12):
        emit(f"  {count:6d}  {code}")
    emit("\nretro rules whose pairing the table already answers (top 12):")
    for code, count in rescued_retro.most_common(12):
        emit(f"  {count:6d}  {code}")

    report_text = "\n".join(lines)
    with capsys.disabled():
        print(report_text)
    if REPORT:
        pathlib.Path(REPORT).write_text(report_text + "\n")

    assert examined > 0, "no application could be built from any recorded product"
    # The general probe is a superset of the paired one, so it cannot confirm fewer.
    assert confirmed_all >= confirmed_paired

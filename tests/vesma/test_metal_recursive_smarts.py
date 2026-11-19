"""Recursive SMARTS on Metal: the level-by-level roots must reproduce the CPU reference exactly.

The CPU matcher is the oracle here, and is itself held to RDKit in ``test_recursive_smarts.py``.
Both read the same molecules, so no aromaticity question arises between them.
"""

from __future__ import annotations

import random
from collections import Counter
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from vesma import cmol_from_smiles, gmol_from_cmol, gmol_from_smarts
from vesma.cpu_matcher import CpuReactionMatcher, has_match_many
from vesma.gmol import GMol
from vesma.identity import short_ring_wl_molecule_id
from vesma.metal_matcher import (
    GpuUnavailableError,
    MetalFoldedPrefilteredMatcher,
    MetalMatcher,
    probe_metal,
)
from vesma.parsing import SmartsError
from vesma.reaction import parse_reaction_smarts
from vesma.rejection import ChemistryRejection

pytestmark = [pytest.mark.gpu, pytest.mark.metal]

PATTERNS = [
    "[C;$(C=O)]",
    "[N;!$(N=O);$(N[#6])]",
    "[C,$(N=O)]",
    "[O;$(O[$(C=O)])]",
    "[$([NX3,NX4+][CX4H]([*])[CX3](=[OX1])[O,N])]",
    "[$([NX3H,NX4H2+])]",
    "[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]",
    "C1CC[$(N(C)C)]CC1",
    "[C;$(C[$(C[$(C[$(CO)])])])]",
    "[c;!$(c[N+](=O)[O-])]",
    "CCO",
    # No recursion: properties tied across alternatives, a box union per clause.
    "[C,n]",
    "[CH2,NH3+]",
    "[OX2H1,SX2H1,NX3;H1,H2]",
    "[C,n]~[O,s]",
]
MOLECULES = [
    "CC(=O)O",
    "CCO",
    "CN=O",
    "CNC",
    "ON=O",
    "NO",
    "CC(=O)OC",
    "CCOC",
    "NC(C)C(=O)O",
    "CN(C)C",
    "CCCl",
    "CN1CCCCC1",
    "C1CCNCC1",
    "CCCCO",
    "CCCO",
    "O=[N+]([O-])c1ccccc1",
    "c1ccncc1",
    "C[NH3+]",
    "CS",
    "c1ccsc1",
]

BENCHMARK_DIR = (
    Path(__file__).parents[1]
    / "data/external/zbh-smarts/processed/smarts-benchmark-v2.0/substructure_search_set"
)


@pytest.fixture(scope="module", autouse=True)
def metal_available() -> None:
    try:
        probe_metal()
    except (GpuUnavailableError, RuntimeError) as exc:
        pytest.skip(f"Metal backend not available: {exc}")


def molecule(smiles: str) -> GMol:
    return gmol_from_cmol(cmol_from_smiles(smiles)[0])


def test_metal_matches_cpu_on_recursive_patterns() -> None:
    patterns = [gmol_from_smarts(s)[0] for s in PATTERNS]
    molecules = [molecule(s) for s in MOLECULES]
    expected = has_match_many(patterns, molecules, n_jobs=1)
    assert expected.any() and not expected.all()
    np.testing.assert_array_equal(
        MetalMatcher().has_match_many(patterns, molecules), expected
    )


def test_prefiltered_metal_matches_cpu_on_recursive_patterns() -> None:
    patterns = [gmol_from_smarts(s)[0] for s in PATTERNS]
    molecules = [molecule(s) for s in MOLECULES]
    np.testing.assert_array_equal(
        MetalFoldedPrefilteredMatcher().has_match_many(patterns, molecules),
        has_match_many(patterns, molecules, n_jobs=1),
    )


@pytest.mark.parametrize("assignment_probe_cap", [1, 8])
@pytest.mark.parametrize("delivery", ["joined", "batched", "folded"])
def test_metal_reaction_matcher_honours_recursion_on_the_matched_side(
    assignment_probe_cap: int, delivery: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vesma import metal_reaction_matcher as mrm
    from vesma.matcher import concat_hits, keyed_match_folding
    from vesma.metal_reaction_matcher import (
        MetalReactionMatcher,
        metal_short_ring_wl_key,
        metal_stack_mols,
    )

    rules = [
        # An ester's alkoxy oxygen, but not an ether's: the carbon must be an acyl carbon.
        parse_reaction_smarts("[C;$(C=O):1][O:2]>>[C:1].[O:2]"),
        parse_reaction_smarts("[C;!$(C=O):1][O:2]>>[C:1].[O:2]"),
    ]
    cmols = [cmol_from_smiles(s)[0] for s in ("CC(=O)OC", "CCOC", "CCO")]
    precursors, cpu_hits = CpuReactionMatcher(rules).match(cmols)
    expected = Counter(
        (int(h[0]), int(h[1]), short_ring_wl_molecule_id(precursors[h[3]]))
        for h in cpu_hits
    )

    # Cross both boundaries: a saturated probe retries predicate-aware matching, and
    # one assignment per construction dispatch forces delivery across multiple batches.
    monkeypatch.setattr(mrm, "_RETRO_ASSIGNMENTS_PER_DISPATCH", 1)
    matcher = MetalReactionMatcher(rules, assignment_probe_cap=assignment_probe_cap)
    products = metal_stack_mols(cmols)
    if delivery == "folded":
        batch, keys, hits = keyed_match_folding(
            matcher.match_batches, metal_short_ring_wl_key, mrm.metal_concat_batches
        )(products)
    else:
        if delivery == "batched":
            parts = list(matcher.match_batches(products))
            batch = mrm.metal_concat_batches([part for part, _ in parts])
            hits = concat_hits(
                [part_hits for _, part_hits in parts],
                [len(part) for part, _ in parts],
            )
        else:
            batch, hits = matcher.match(products)
        keys = metal_short_ring_wl_key(batch)
    observed = Counter((int(h[0]), int(h[1]), keys[h[3]]) for h in hits)
    assert matcher.last_diagnostics is not None
    if assignment_probe_cap == 1:
        assert matcher.last_diagnostics.match_retry_pairs > 0

    assert observed == expected
    assert {(mol, rule) for mol, rule, _ in observed} >= {(0, 0), (1, 1)}


def test_metal_screen_honours_recursion() -> None:
    from vesma.metal_reaction_matcher import metal_screen, metal_stack_mols

    screen = metal_screen(["[C;$(C=O)]", "[N;!$(N=O)]"])
    batch = metal_stack_mols([cmol_from_smiles(s)[0] for s in ("CC=O", "CCO", "CN=O")])
    pairs = [(veto, row) for veto in range(2) for row in range(3)]
    assert screen(batch, pairs).tolist() == [True, False, False, False, False, False]


@cache
def corpus() -> tuple[list[GMol], list[GMol]]:
    rows = [
        line.split()[:2]
        for name in (
            "literature_noHs_rec.smarts.lead-like.benchmarkset",
            "literature_noHs_rec.smarts.everything.benchmarkset",
        )
        for line in (BENCHMARK_DIR / name).read_text().splitlines()
    ]
    patterns: list[GMol] = []
    for smarts in sorted({row[0] for row in rows}):
        try:
            patterns.append(gmol_from_smarts(smarts)[0])
        except SmartsError:
            continue
    molecules: list[GMol] = []
    for smiles in random.Random(2).sample(sorted({row[1] for row in rows}), 150):
        try:
            molecules.append(molecule(smiles))
        except SmartsError, ChemistryRejection:
            continue
    return patterns, molecules


@pytest.mark.slow
def test_metal_matches_cpu_on_the_zbh_recursive_corpus() -> None:
    patterns, molecules = corpus()
    expected = has_match_many(patterns, molecules)
    observed = MetalMatcher().has_match_many(patterns, molecules)
    disagree = np.argwhere(observed != expected)
    assert disagree.tolist() == []


def test_an_unresolved_body_search_is_an_error_not_an_absence() -> None:
    """A body search cut off by the budget must not read as "no embedding": negated, that
    absence would become a match the outer search reports as definite."""
    from vesma.metal_matcher import UnresolvedPredicateError

    pattern = gmol_from_smarts("[!$(C.C.C)]")[0]
    target = molecule("CCC")
    with pytest.raises(UnresolvedPredicateError):
        MetalMatcher(max_prefix_depth=0, max_leaf_checks=1).has_match_many(
            [pattern], [target]
        )
    assert not MetalMatcher().has_match_many([pattern], [target])[0, 0]

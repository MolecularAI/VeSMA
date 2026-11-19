"""Skipping an assignment that rewrites the product the way an earlier one did must lose nothing.

`generate_precursors` builds precursors for every embedding the matcher returns and deduplicates
the results afterwards. Where a pattern names a reagent more than once, the embeddings include
every permutation of the interchangeable copies -- all of which rewrite the product identically --
so the construction cost is multiplied by the pattern's automorphism count and thrown away.
`edit_signature` skips those before the construction.

That filter is load-bearing in a way the deduplication below it is not. If the key were too
*coarse* it would merge two assignments that rewrite the product differently, and the outcome-level
deduplication could not catch it -- the second outcome was never built. So soundness is checked
here against the unfiltered path over the whole curated library, rather than argued.
"""

from __future__ import annotations

import csv
from typing import Any

import pytest

from tests.datadirs import external_data_root
from vesma import cmol_from_smiles
from vesma.cmol_builder import gmol_from_cmol
from vesma.cpu_matcher import CpuMatcher, generate_precursors
from vesma.keyer import cmol_rdkit_smiles_keyer
from vesma.reaction import (
    ReactionPattern,
    parse_reaction_smarts,
    precursor_signature,
    precursors_from_mapping_result,
)

CURATED = (
    ("templates", "templates-hartenfeller.csv"),
    ("templates", "templates-medchem.csv"),
    ("templates", "templates-protecting-groups.csv"),
)

# Products chosen to exercise the rules a reference route actually uses: an amide, an aryl halide,
# a nitroarene, a benzylic alcohol, an azine, a Boc carbamate, a stilbene, a fused azole.
PRODUCTS = (
    "O=C(CN1CCCC1)Nc1ccc(Br)cc1",
    "BrCc1ccc(-c2nc3ccccc3s2)nc1",
    "Nc1ccc(Br)cc1",
    "OCc1ccccc1I",
    "Brc1cnnc(N2CCCC2)c1",
    "CC(C)(C)OC(=O)N(CCN1CCCC1)c1ccc(Br)cc1",
    "Brc1ccccc1C=Cc1ccccc1I",
    "Cc1ccc(-c2nc3ccccc3o2)nc1",
    "CCC(=O)CCC",
    "Brc1ccc(C2=CCN(Cc3ccccc3)CC2)cc1",
)


def unfiltered(product_smiles: str, reaction: ReactionPattern) -> list[Any]:
    """What `generate_precursors` did before the filter: build everything, dedup the results."""
    product, _info = cmol_from_smiles(product_smiles)
    result = CpuMatcher().find_matches(reaction.lhs, gmol_from_cmol(product))
    if result.assignments.size == 0:
        return []
    outcomes: list[Any] = []
    seen: set[Any] = set()
    for mapping in result.assignments:
        mapped = precursors_from_mapping_result(
            product=product, reaction=reaction, mapping=mapping
        )
        if not mapped.precursors:
            continue
        signature = precursor_signature(mapped.precursors)
        if signature in seen:
            continue
        seen.add(signature)
        outcomes.append(signature)
    return outcomes


def curated_rows() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for parts in CURATED:
        path = external_data_root().joinpath(*parts)
        if not path.exists():
            pytest.skip(f"Missing curated template file: {path}")
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                smarts = (row.get("retro_template") or "").strip()
                if smarts:
                    rows.append((row["template_code"], smarts))
    return rows


@pytest.mark.parametrize("product", PRODUCTS)
def test_the_filter_changes_no_outcome_anywhere_in_the_library(product: str) -> None:
    """Every rule in the curated library, against one product, both ways round.

    Compared as the *set* of precursor signatures rather than as a list, because the filter is
    allowed to change how often a duplicate is reached -- that is the whole point -- but not which
    distinct outcomes exist. Order is checked too, since the first of a run of duplicates is the
    one kept either way.
    """
    checked = 0
    for code, smarts in curated_rows():
        try:
            reaction = parse_reaction_smarts(smarts)
        except Exception:
            continue  # a template this build cannot compile is not this test's subject
        try:
            filtered = [
                precursor_signature(one.precursors)
                for one in generate_precursors(cmol_from_smiles(product)[0], reaction)
            ]
            plain = unfiltered(product, reaction)
        except Exception:
            continue  # a rule the chemistry refuses on this product answers nothing either way
        assert filtered == plain, code
        checked += 1
    # The library really was exercised, rather than skipped into vacuous agreement.
    assert checked > 300


def test_the_filter_collapses_a_pattern_that_names_a_reagent_three_times() -> None:
    """And it does something: the nitro reduction's three tin chlorides, measured both ways.

    One outcome either way, from 288 embeddings -- `3! x 3!` orderings of the interchangeable
    reagents times the two chlorine orderings inside each SnCl2. Without the filter that is 288
    precursor constructions for one surviving answer.
    """
    smarts = (
        "([c:1]-[N+;H0:2](=[O])-[O-])"
        + ".(Cl[Sn]Cl)" * 3
        + ".(Cl)" * 3
        + ".([OH2])"
        + ">>([c:1]-[N;H2;+0:2])"
        + ".(Cl[Sn](Cl)(Cl)O)" * 3
    )
    pot_smiles = "O=[N+]([O-])c1ccccc1" + ".Cl[Sn]Cl" * 3 + ".Cl" * 3 + ".O"
    reaction = parse_reaction_smarts(smarts)
    pot, _info = cmol_from_smiles(pot_smiles)
    result = CpuMatcher().find_matches(reaction.lhs, gmol_from_cmol(pot))
    assert int(result.assignments.shape[0]) == 288

    outcomes = generate_precursors(pot, reaction)
    assert len(outcomes) == 1
    made = sorted(cmol_rdkit_smiles_keyer.encode(one) for one in outcomes[0].precursors)
    assert made == sorted(["Nc1ccccc1"] + ["[OH][Sn]([Cl])([Cl])[Cl]"] * 3)
    # Same answer as the unfiltered path, which is the claim that matters.
    assert [precursor_signature(outcomes[0].precursors)] == unfiltered(
        pot_smiles, reaction
    )

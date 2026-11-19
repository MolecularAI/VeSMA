"""The CUDA predicate path asks the same questions as the CPU and Metal suites."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import pytest

from tests.vesma.test_metal_recursive_smarts import MOLECULES, PATTERNS
from vesma import cmol_from_smiles, gmol_from_cmol, gmol_from_smarts
from vesma.cpu_matcher import CpuReactionMatcher, has_match_many
from vesma.cuda_matcher import (
    CudaFoldedPrefilteredMatcher,
    CudaFrontierMode,
    CudaMatcher,
    CudaReactionMatcher,
    cuda_concat_mols,
    cuda_short_ring_wl_key,
)
from vesma.gpu_predicates import UnresolvedPredicateError
from vesma.identity import short_ring_wl_molecule_id
from vesma.matcher import HitArray, concat_hits, keyed_match_folding
from vesma.reaction import parse_reaction_smarts

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


@pytest.fixture(scope="module", autouse=True)
def cuda_available() -> None:
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CUDA is unavailable")


@pytest.mark.parametrize("prefilter", [False, True])
def test_cuda_recursive_patterns_agree_with_cpu(prefilter: bool) -> None:
    patterns = [gmol_from_smarts(s)[0] for s in PATTERNS]
    molecules = [gmol_from_cmol(cmol_from_smiles(s)[0]) for s in MOLECULES]
    expected = has_match_many(patterns, molecules, n_jobs=1)
    matcher = CudaFoldedPrefilteredMatcher() if prefilter else CudaMatcher()
    np.testing.assert_array_equal(matcher.has_match_many(patterns, molecules), expected)


def triples(hits: HitArray, keys: Sequence[tuple[int, int]]) -> Counter:
    return Counter((int(h[0]), int(h[1]), keys[h[3]]) for h in hits)


@pytest.mark.parametrize("frontier_mode", ["host", "device"])
@pytest.mark.parametrize("delivery", ["joined", "batched", "folded", "native-keyed"])
def test_recursive_reaction_delivery_preserves_cpu_outcomes(
    frontier_mode: CudaFrontierMode, delivery: str
) -> None:
    rules = [
        parse_reaction_smarts("[C;$(C=O):1][O:2]>>[C:1].[O:2]"),
        parse_reaction_smarts("[C;!$(C=O):1][O:2]>>[C:1].[O:2]"),
    ]
    molecules = [cmol_from_smiles(s)[0] for s in ("CC(=O)OC", "CCOC", "CCO")]
    cpu, cpu_hits = CpuReactionMatcher(rules).match(molecules, n_jobs=1)
    expected = triples(cpu_hits, [short_ring_wl_molecule_id(m) for m in cpu])
    matcher = CudaReactionMatcher(
        rules, frontier_mode=frontier_mode, max_pairs_per_launch=1
    )
    if delivery == "native-keyed":
        _, keys, hits = matcher.match_keyed(molecules)
    elif delivery == "folded":
        _, keys, hits = keyed_match_folding(
            matcher.match_batches, cuda_short_ring_wl_key, cuda_concat_mols
        )(molecules)
    else:
        if delivery == "batched":
            parts = list(matcher.match_batches(molecules))
            precursors = cuda_concat_mols([part for part, _ in parts])
            hits = concat_hits(
                [part_hits for _, part_hits in parts], [len(part) for part, _ in parts]
            )
        else:
            precursors, hits = matcher.match(molecules)
        keys = cuda_short_ring_wl_key(precursors)
    assert triples(hits, keys) == expected
    assert expected


def test_unresolved_body_search_does_not_become_a_negated_match() -> None:
    pattern = gmol_from_smarts("[!$(C.C.C)]")[0]
    molecule = gmol_from_cmol(cmol_from_smiles("CCC")[0])
    with pytest.raises(UnresolvedPredicateError):
        CudaMatcher(max_prefix_depth=0, max_leaf_checks=1).has_match_many(
            [pattern], [molecule]
        )
    assert not CudaMatcher().has_match_many([pattern], [molecule])[0, 0]

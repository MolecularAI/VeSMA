from __future__ import annotations

import sys
from unittest.mock import Mock

import numpy as np
import pytest

import vesma.metal_matcher as metal_matcher
from vesma import cmol_from_smiles, gmol_from_cmol, gmol_from_smarts
from vesma.cpu_matcher import CpuMatcher
from vesma.metal_matcher import (
    GpuUnavailableError,
    MetalMatcher,
    pair_index_array,
    probe_metal,
)

pytestmark = [pytest.mark.gpu, pytest.mark.metal]


def pattern(smarts: str):
    return gmol_from_smarts(smarts)[0]


def molecule(smiles: str):
    return gmol_from_cmol(cmol_from_smiles(smiles)[0])


def skip_if_no_metal_toolchain() -> None:
    try:
        probe_metal()
    except (GpuUnavailableError, RuntimeError) as exc:
        pytest.skip(f"Metal backend not available: {exc}")


@pytest.mark.skipif(sys.platform == "darwin", reason="Requires a non-macOS host.")
def test_probe_metal_rejects_non_macos_before_loading_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_library = Mock(side_effect=AssertionError("Metal library load attempted."))
    monkeypatch.setattr(metal_matcher, "_load_metal_library", load_library)

    with pytest.raises(
        GpuUnavailableError,
        match="Metal backend is only available on macOS",
    ):
        probe_metal()

    load_library.assert_not_called()


def test_pair_index_array_accepts_empty_pairs() -> None:
    pairs = pair_index_array(
        [],
        first_count=2,
        second_count=3,
        first_name="pattern",
        second_name="molecule",
    )
    assert pairs.shape == (0, 2)
    assert pairs.dtype == np.int64


def test_pair_index_array_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="\\(N, 2\\)"):
        pair_index_array(
            np.array([0, 1]),
            first_count=2,
            second_count=3,
            first_name="pattern",
            second_name="molecule",
        )


@pytest.mark.parametrize(
    ("pair_indices", "message"),
    [
        ([(-1, 0)], "pattern index -1"),
        ([(2, 0)], "pattern index 2"),
        ([(0, -1)], "molecule index -1"),
        ([(0, 3)], "molecule index 3"),
    ],
)
def test_pair_index_array_rejects_out_of_bounds_pairs(
    pair_indices: list[tuple[int, int]], message: str
) -> None:
    with pytest.raises(IndexError, match=message):
        pair_index_array(
            pair_indices,
            first_count=2,
            second_count=3,
            first_name="pattern",
            second_name="molecule",
        )


@pytest.fixture(scope="module")
def matcher():
    skip_if_no_metal_toolchain()

    try:
        probe_metal()
        return MetalMatcher()
    except (GpuUnavailableError, RuntimeError) as exc:
        pytest.skip(f"Metal backend not available: {exc}")


def test_metal_matcher_basic_match(matcher):
    p = pattern("CO")
    m = molecule("CCO")
    result = matcher.find_matches(p, m)
    assert result.status == "match"
    assert result.assignments.shape[0] > 0


def test_metal_matcher_no_match(matcher):
    p = pattern("CN")
    m = molecule("CCO")
    result = matcher.find_matches(p, m)
    assert result.status == "no_match"
    assert result.assignments.shape[0] == 0


def test_metal_matcher_multiple_matches(matcher):
    p = pattern("CC")
    m = molecule("CCC")
    # CCC has two C-C bonds, so multiple matches expected
    result = matcher.find_matches(p, m)
    assert result.status == "match"
    # [0,1], [1,0], [1,2], [2,1] - actually 4 if we consider directions
    assert result.assignments.shape[0] >= 2


def test_metal_matcher_has_match_many(matcher):
    patterns = [pattern("CO"), pattern("CN"), pattern("CC")]
    molecules = [molecule("CCO"), molecule("CCC"), molecule("N")]

    # Expected results:
    # CO in CCO: True
    # CO in CCC: False
    # CO in N: False
    # CN in CCO: False
    # CN in CCC: False
    # CN in N: False (N is just an atom, CN needs C and N)
    # CC in CCO: True
    # CC in CCC: True
    # CC in N: False

    expected = np.array(
        [[True, False, False], [False, False, False], [True, True, False]]
    )

    results = matcher.has_match_many(patterns, molecules)
    assert np.array_equal(results, expected)


@pytest.mark.parametrize(
    ("smarts", "smiles", "expected"),
    [
        ("C@C", "C1CCCCC1", True),  # ring single matches a ring bond
        ("C!@C", "C1CCCCC1", False),  # chain single rejects a ring bond
        ("C@C", "CCCCCC", False),  # ring single rejects a chain bond
        ("C!@C", "CCCCCC", True),  # chain single matches a chain bond
        ("c@c", "c1ccccc1", True),  # ring aromatic matches a ring bond
        ("c-!@c", "c1ccccc1-c2ccccc2", True),  # biphenyl bridge is a chain single
        ("c-@c", "c1ccccc1-c2ccccc2", False),  # no ring single c-c bond exists
    ],
)
def test_metal_matcher_enforces_ring_chain_topology(
    matcher, smarts: str, smiles: str, expected: bool
) -> None:
    result = matcher.find_matches(pattern(smarts), molecule(smiles))
    assert (result.status == "match") is expected


def test_metal_matcher_respects_max_assignments(matcher):
    p = pattern("CC")
    m = molecule("CCC")
    result = matcher.find_matches(p, m, max_assignments=1)
    assert result.status == "match"
    assert result.assignments.shape[0] == 1


# A highly symmetric polycyclic aromatic with three fused six-membered rings.
# A bare benzene pattern has a huge number of candidate placements here, so the
# refinement-then-leaf-enumeration path alone blows past max_leaf_checks. The DFS
# backtracking middle phase is what collapses the search to a few leaf checks.
_TRIPHENYLENE = "c1ccc2c(c1)c1ccccc1c1ccccc21"


def test_metal_matcher_symmetric_three_rings_needs_dfs(matcher):
    p = pattern("c1ccccc1")
    m = molecule(_TRIPHENYLENE)

    cpu_result = CpuMatcher().find_matches(p, m, max_assignments=4096)
    assert cpu_result.status == "match"

    # With DFS (default depth) Metal resolves the match exactly and cheaply.
    result = matcher.find_matches(p, m, max_assignments=4096)
    assert result.status == "match"
    assert result.assignments.shape[0] == cpu_result.assignments.shape[0]
    # DFS collapses the candidate space: far fewer leaf checks than the budget.
    assert result.leaves_explored < 1024

    # Without the DFS middle phase the same query exhausts the leaf-check budget
    # and cannot conclude (status "unknown") -- this is the regression DFS fixes.
    no_dfs = MetalMatcher(max_prefix_depth=0).find_matches(p, m, max_assignments=4096)
    assert no_dfs.status == "unknown"


def test_metal_matcher_cpu_parity(matcher):
    cpu = CpuMatcher()
    patterns = [
        pattern("c1ccccc1"),
        pattern("[c]:[c]-[c]:[c]"),
        pattern("c1ccc2ccccc2c1"),
        pattern("CO"),
        pattern("CN"),
    ]
    molecules = [
        molecule(_TRIPHENYLENE),
        molecule("c1ccc2cc3ccccc3cc2c1"),  # anthracene
        molecule("CCO"),
        molecule("c1ccccc1CO"),
    ]

    metal_hits = matcher.has_match_many(patterns, molecules)
    cpu_hits = cpu.has_match_many(patterns, molecules)
    assert np.array_equal(metal_hits, cpu_hits)

    # Assignment counts must also agree on the positive pairs.
    for pi, p in enumerate(patterns):
        for mi, m in enumerate(molecules):
            if not cpu_hits[pi, mi]:
                continue
            cpu_result = cpu.find_matches(p, m, max_assignments=4096)
            metal_result = matcher.find_matches(p, m, max_assignments=4096)
            assert metal_result.status == "match"
            assert metal_result.assignments.shape[0] == cpu_result.assignments.shape[0]

from __future__ import annotations

import pytest

from vesma import cmol_from_smiles
from vesma.cpu_matcher import CpuReactionMatcher
from vesma.cuda_matcher import (
    CudaFoldedPrefilteredReactionMatcher,
    CudaMoleculeHostBatch,
    CudaReactionMatcher,
    cuda_short_ring_wl_key,
    cuda_stack_mols,
)
from vesma.cuda_retro import CudaMoleculeBatch
from vesma.keyer import short_ring_wl_keyer
from vesma.matcher import all_vs_all, keyed_match
from vesma.prefilter import (
    PrefilteredReactionMatcher,
    cpu_sparse_path_feature_provider,
    cuda_host_sparse_path_feature_provider,
)
from vesma.reaction import parse_reaction_smarts, reaction_patterns_from_smarts

RULES: dict[int, str] = {
    0: "[C:1][C:2]>>[C:1].[C:2]",
    1: "[C:1][O:2]>>[C:1].[O:2]",
    2: "[C:1][N:2]>>[C:1].[N:2]",
    3: "[C:1]=[O:2]>>[C:1].[O:2]",
}
TARGETS = ("CCO", "CCN", "CCCC", "CC(=O)O", "OCC(N)CO")


def build_layer(frontier_keys, hits, precursor_keys):
    groups: dict[tuple[int, int, int], list] = {}
    layer: dict = {key: set() for key in frontier_keys}
    for hit in hits:
        product_idx, rule_id, app_idx, precursor_idx = hit.tolist()
        groups.setdefault((product_idx, rule_id, app_idx), []).append(
            precursor_keys[precursor_idx]
        )
    for (target_idx, rule_id, _), keys in groups.items():
        layer[frontier_keys[target_idx]].add((rule_id, frozenset(keys)))
    return {key: frozenset(steps) for key, steps in layer.items()}


def patterns():
    return reaction_patterns_from_smarts(
        tuple(RULES.values()),
        n_jobs=1,
        cache=False,
    )


def test_cpu_prefilter_matches_unfiltered_layer() -> None:
    codec = short_ring_wl_keyer
    products = [cmol_from_smiles(smiles)[0] for smiles in TARGETS]
    frontier_keys = [codec.from_smiles(smiles) for smiles in TARGETS]
    compiled = patterns()

    unfiltered = CpuReactionMatcher(compiled)
    unfiltered_precursors, unfiltered_hits = unfiltered.match(products)
    unfiltered_keys = [codec.encode(molecule) for molecule in unfiltered_precursors]

    filtered = PrefilteredReactionMatcher(
        CpuReactionMatcher(compiled),
        compiled,
        feature_provider=cpu_sparse_path_feature_provider,
    )
    filtered_precursors, filtered_hits = filtered.match(products)
    filtered_keys = [codec.encode(molecule) for molecule in filtered_precursors]

    assert build_layer(frontier_keys, filtered_hits, filtered_keys) == build_layer(
        frontier_keys,
        unfiltered_hits,
        unfiltered_keys,
    )
    assert filtered.last_diagnostics is not None
    assert filtered.last_diagnostics.candidate_pair_count < len(TARGETS) * len(RULES)


def test_cpu_prefilter_composes_with_keyed_match() -> None:
    codec = short_ring_wl_keyer
    products = [cmol_from_smiles(smiles)[0] for smiles in TARGETS[:2]]
    compiled = patterns()
    filtered = PrefilteredReactionMatcher(
        CpuReactionMatcher(compiled),
        compiled,
        feature_provider=cpu_sparse_path_feature_provider,
    )
    match = keyed_match(
        all_vs_all(filtered), lambda mols: [codec.encode(m) for m in mols]
    )
    precursors, keys, hits = match(products)

    assert len(precursors) == len(keys) == len(hits)
    assert filtered.last_diagnostics is not None
    assert filtered.last_diagnostics.dense_pair_count == len(products) * len(RULES)


def test_prefilter_respects_explicit_pairs() -> None:
    ethanol = cmol_from_smiles("CCO")[0]
    compiled = patterns()
    filtered = PrefilteredReactionMatcher(
        CpuReactionMatcher(compiled),
        compiled,
        feature_provider=cpu_sparse_path_feature_provider,
    )

    precursors, hits = filtered.match([ethanol], pairs=[(0, 1), (0, 2)])

    assert len(hits)
    assert {rule_id for _, rule_id, _, _ in hits} == {1}
    assert len(precursors) == len(hits)
    assert filtered.last_diagnostics is not None
    assert filtered.last_diagnostics.dense_pair_count == 2
    assert filtered.last_diagnostics.candidate_pair_count == 1


def skip_if_no_cuda() -> None:
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CuPy cannot access CUDA on this machine.")


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.slow
def test_cuda_prefilter_matches_unfiltered_layer() -> None:
    skip_if_no_cuda()
    codec = short_ring_wl_keyer
    products = [cmol_from_smiles(smiles)[0] for smiles in TARGETS]
    frontier_keys = [codec.from_smiles(smiles) for smiles in TARGETS]
    compiled = patterns()

    unfiltered = CudaReactionMatcher(compiled)
    unfiltered_mols = cuda_stack_mols(products)
    unfiltered_precursors, unfiltered_hits = unfiltered.match(unfiltered_mols)
    unfiltered_keys = cuda_short_ring_wl_key(unfiltered_precursors)

    filtered = PrefilteredReactionMatcher(
        CudaReactionMatcher(compiled),
        compiled,
        feature_provider=cuda_host_sparse_path_feature_provider,
    )
    filtered_mols = cuda_stack_mols(products)
    filtered_precursors, filtered_hits = filtered.match(filtered_mols)
    filtered_keys = cuda_short_ring_wl_key(filtered_precursors)

    assert build_layer(frontier_keys, filtered_hits, filtered_keys) == build_layer(
        frontier_keys,
        unfiltered_hits,
        unfiltered_keys,
    )
    assert filtered.last_diagnostics is not None
    assert filtered.last_diagnostics.candidate_pair_count < len(TARGETS) * len(RULES)


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_folded_prefilter_matches_unfiltered_layer() -> None:
    skip_if_no_cuda()
    codec = short_ring_wl_keyer
    products = [cmol_from_smiles(smiles)[0] for smiles in TARGETS]
    frontier_keys = [codec.from_smiles(smiles) for smiles in TARGETS]
    compiled = patterns()

    unfiltered = CudaReactionMatcher(compiled)
    unfiltered_mols = cuda_stack_mols(products)
    unfiltered_precursors, unfiltered_hits = unfiltered.match(unfiltered_mols)
    unfiltered_keys = cuda_short_ring_wl_key(unfiltered_precursors)

    filtered = CudaFoldedPrefilteredReactionMatcher(CudaReactionMatcher(compiled))
    filtered_mols = cuda_stack_mols(products)
    filtered_precursors, filtered_hits = filtered.match(filtered_mols)
    filtered_keys = cuda_short_ring_wl_key(filtered_precursors)

    assert build_layer(frontier_keys, filtered_hits, filtered_keys) == build_layer(
        frontier_keys,
        unfiltered_hits,
        unfiltered_keys,
    )
    assert filtered.last_diagnostics is not None
    assert filtered.last_diagnostics.pair_count < len(TARGETS) * len(RULES)


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.parametrize("frontier_mode", ["host", "device"])
def test_cuda_folded_prefilter_keyed_matches_unfiltered_frontier(
    frontier_mode: str,
) -> None:
    skip_if_no_cuda()
    codec = short_ring_wl_keyer
    products = [cmol_from_smiles(smiles)[0] for smiles in TARGETS]
    frontier_keys = [codec.from_smiles(smiles) for smiles in TARGETS]
    compiled = patterns()

    unfiltered = CudaReactionMatcher(compiled, frontier_mode=frontier_mode)
    filtered = CudaFoldedPrefilteredReactionMatcher(
        CudaReactionMatcher(compiled, frontier_mode=frontier_mode)
    )
    unfiltered_result = unfiltered.match_keyed(cuda_stack_mols(products))
    filtered_result = filtered.match_keyed(cuda_stack_mols(products))
    unfiltered_precursors, unfiltered_keys, unfiltered_hits = unfiltered_result
    filtered_precursors, filtered_keys, filtered_hits = filtered_result

    assert build_layer(frontier_keys, filtered_hits, filtered_keys) == build_layer(
        frontier_keys, unfiltered_hits, unfiltered_keys
    )
    assert len(filtered_precursors) == len(filtered_keys)
    assert len(filtered_precursors) <= len(filtered_hits)
    expected_type = (
        CudaMoleculeHostBatch if frontier_mode == "host" else CudaMoleculeBatch
    )
    assert isinstance(filtered_precursors, expected_type)
    diagnostics = filtered.last_prefilter_diagnostics
    assert diagnostics is not None
    assert diagnostics.dense_pair_count == len(TARGETS) * len(RULES)
    assert diagnostics.candidate_pair_count < diagnostics.dense_pair_count
    assert diagnostics.prefilter_seconds >= 0.0


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_folded_prefilter_keeps_short_ring_single_bond_match() -> None:
    skip_if_no_cuda()
    product = cmol_from_smiles("CN1CCC(Oc2ccc3[nH]nc(S(=O)(=O)c4cccc5ccccc45)c3c2)C1")[
        0
    ]
    compiled = (
        parse_reaction_smarts(
            "[C:2]-[CH2;D2;+0:1]-[C:3]>>I-[CH;D3;+0:1](-[C:2])-[C:3]"
        ),
    )

    unfiltered = CudaReactionMatcher(compiled)
    filtered = CudaFoldedPrefilteredReactionMatcher(CudaReactionMatcher(compiled))
    mols = cuda_stack_mols([product])

    unfiltered_precursors, unfiltered_hits = unfiltered.match(mols)
    filtered_precursors, filtered_hits = filtered.match(mols)

    frontier_keys = [short_ring_wl_keyer(product)]
    assert build_layer(
        frontier_keys,
        filtered_hits,
        cuda_short_ring_wl_key(filtered_precursors),
    ) == build_layer(
        frontier_keys,
        unfiltered_hits,
        cuda_short_ring_wl_key(unfiltered_precursors),
    )

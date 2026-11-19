from __future__ import annotations

import time

import numpy as np
import pytest

from vesma import cmol_from_smiles, gmol_from_cmol, gmol_from_smarts
from vesma.cpu_matcher import CpuMatcher
from vesma.metal_matcher import GpuUnavailableError, MetalMatcher, probe_metal

pytestmark = [pytest.mark.perf, pytest.mark.gpu, pytest.mark.metal]


def pattern(smarts: str):
    return gmol_from_smarts(smarts)[0]


def molecule(smiles: str):
    return gmol_from_cmol(cmol_from_smiles(smiles)[0])


@pytest.mark.perf
def test_perf_metal_batch_cross_product():
    try:
        probe_metal()
        metal = MetalMatcher()
    except GpuUnavailableError, RuntimeError:
        pytest.skip("Metal not available")

    cpu = CpuMatcher()

    # 10 patterns x 1000 molecules = 10,000 matches
    patterns_smarts = [
        "CO",
        "CN",
        "CC",
        "C1CC1",
        "C(=O)O",
        "c1ccccc1",
        "N",
        "O",
        "S",
        "Cl",
    ]
    molecules_smiles = [
        "CCN(CC)CCNC(=O)c1ccc(OC)cc1",
        "Cc1ccc(S(=O)(=O)N2CCN(C)CC2)cc1",
        "O=C(Nc1ccccc1)c1ccccc1",
        "CN1CCN(c2ccc(NC(=O)c3cc(Cl)cc(Cl)c3)cc2)CC1",
        "COc1ccc(NC(=O)Cc2cccc(Cl)c2)cc1",
    ] * 200  # 1000 molecules

    patterns = [pattern(s) for s in patterns_smarts]
    molecules = [molecule(s) for s in molecules_smiles]

    print(
        f"\nBatch Size: {len(patterns)} patterns x {len(molecules)} molecules = {len(patterns) * len(molecules)} pairs"
    )

    # Warmup
    metal.has_match_many(patterns[:1], molecules[:10])

    start = time.perf_counter()
    metal_results = metal.has_match_many(patterns, molecules)
    end = time.perf_counter()
    metal_time = end - start

    print(
        f"Metal batch time: {metal_time:.4f}s ({metal_time * 1000 / (len(patterns) * len(molecules)):.4f} ms per pair)"
    )

    start = time.perf_counter()
    cpu_results = cpu.has_match_many(patterns, molecules)
    end = time.perf_counter()
    cpu_time = end - start

    print(
        f"CPU batch time:   {cpu_time:.4f}s ({cpu_time * 1000 / (len(patterns) * len(molecules)):.4f} ms per pair)"
    )
    print(f"Speedup: {cpu_time / metal_time:.2f}x")

    if not np.array_equal(metal_results, cpu_results):
        diff_indices = np.where(metal_results != cpu_results)
        print(f"Differences found at: {list(zip(*diff_indices))[:10]}")
        for i, j in list(zip(*diff_indices))[:5]:
            print(
                f"Pair ({i}, {j}): Metal={metal_results[i, j]}, CPU={cpu_results[i, j]}"
            )
            print(f"  Pattern: {patterns_smarts[i]}")
            print(f"  Molecule: {molecules_smiles[j]}")

    assert np.array_equal(metal_results, cpu_results)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-m", "perf"])

from __future__ import annotations

import time

import numpy as np
import pytest

from vesma import SmartsError, cmol_from_smiles, gmol_from_cmol, gmol_from_smarts
from vesma.cpu_matcher import CpuMatcher
from vesma.metal_matcher import GpuUnavailableError, MetalMatcher, probe_metal

pytestmark = [pytest.mark.perf, pytest.mark.gpu, pytest.mark.metal]


# Simplified Hartenfeller-like SMARTS for quick benchmarking
# In a real run we would load from patterns.tsv
BENCHMARK_PATTERNS = [
    ("[C:1](=[O:2])[OH].[N:3]", "Amide Coupling (Acid)"),
    ("[C:1]=[O:2].[N:3]", "Reductive Amination"),
    ("[c:1][B](O)O.[c:2][Cl,Br,I]", "Suzuki Coupling"),
    ("[S:1](=[O:2])(=[O:3])[Cl].[N:4]", "N-Sulfonylation"),
]

# Random ZINC-like SMILES
BENCHMARK_MOLECULES = [
    "CCN(CC)CCNC(=O)c1ccc(OC)cc1",
    "Cc1ccc(S(=O)(=O)N2CCN(C)CC2)cc1",
    "O=C(Nc1ccccc1)c1ccccc1",
    "CN1CCN(c2ccc(NC(=O)c3cc(Cl)cc(Cl)c3)cc2)CC1",
    "COc1ccc(NC(=O)Cc2cccc(Cl)c2)cc1",
]


class RdkitMatcher:
    def __init__(self):
        self.chem = pytest.importorskip("rdkit.Chem")

    def has_match(self, pattern_gmol, molecule_gmol):
        # We need SMILES/SMARTS or RDKit objects.
        # This is a bit complex for a generic Matcher if we only have GMols.
        # For perf script, we'll bypass and use RDKit directly on its own mols.
        pass


@pytest.mark.perf
def test_perf_metal_vs_cpu_vs_rdkit():
    try:
        probe_metal()
        metal = MetalMatcher()
    except GpuUnavailableError, RuntimeError:
        pytest.skip("Metal not available")

    cpu = CpuMatcher()
    chem = pytest.importorskip("rdkit.Chem")

    patterns = []
    for smarts, name in BENCHMARK_PATTERNS:
        try:
            vg = gmol_from_smarts(smarts)[0]
            rd = chem.MolFromSmarts(smarts)
            patterns.append((vg, rd, name))
        except SmartsError:
            continue

    molecules = []
    for smiles in BENCHMARK_MOLECULES:
        vg = gmol_from_cmol(cmol_from_smiles(smiles)[0])
        rd = chem.MolFromSmiles(smiles)
        molecules.append((vg, rd, smiles))

    print(
        f"\n{'Reaction':<25} | {'CPU (ms)':<10} | {'Metal (ms)':<10} | {'RDKit (ms)':<10} | Speedup"
    )
    print("-" * 75)

    for vp, rp, name in patterns:
        cpu_times = []
        metal_times = []
        rdkit_times = []

        for vm, rm, _smiles in molecules:
            # Warmup
            cpu.has_match(vp, vm)
            metal.has_match(vp, vm)
            rm.HasSubstructMatch(rp)

            # Bench
            start = time.perf_counter_ns()
            cpu.has_match(vp, vm)
            cpu_times.append(time.perf_counter_ns() - start)

            start = time.perf_counter_ns()
            metal.has_match(vp, vm)
            metal_times.append(time.perf_counter_ns() - start)

            start = time.perf_counter_ns()
            rm.HasSubstructMatch(rp)
            rdkit_times.append(time.perf_counter_ns() - start)

        avg_cpu = np.mean(cpu_times) / 1e6
        avg_metal = np.mean(metal_times) / 1e6
        avg_rdkit = np.mean(rdkit_times) / 1e6
        speedup = avg_cpu / avg_metal

        print(
            f"{name:<25} | {avg_cpu:10.3f} | {avg_metal:10.3f} | {avg_rdkit:10.3f} | {speedup:8.2f}x"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-m", "perf"])

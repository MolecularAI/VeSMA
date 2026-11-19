from __future__ import annotations

import csv
import gzip
import time

import numpy as np
import pytest

from tests.datadirs import external_data_root
from vesma import SmartsError, cmol_from_smiles, gmol_from_cmol, gmol_from_smarts
from vesma.cpu_matcher import CpuMatcher
from vesma.metal_matcher import GpuUnavailableError, MetalMatcher, probe_metal

pytestmark = [pytest.mark.perf, pytest.mark.gpu, pytest.mark.metal]


# Data paths
MOLECULES_GZ = (
    external_data_root() / "external" / "chembl" / "raw" / "chembl16_50K_sss.txt.gz"
)
PATTERNS_GZ = (
    external_data_root()
    / "external"
    / "paroutes"
    / "processed"
    / "uspto_unique_templates.csv.gz"
)

# Benchmark settings
PATTERN_LIMIT = 50
MOLECULE_LIMIT = 2000
MAX_ATOM_COUNT = 64  # MetalMatcher limit


def _load_benchmark_data():
    chem = pytest.importorskip("rdkit.Chem")

    # Load patterns
    patterns = []
    with gzip.open(
        PATTERNS_GZ, "rt", encoding="utf-8", errors="replace", newline=""
    ) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            smarts = row["retro_template"].split(">>", 1)[0].strip()
            try:
                vg = gmol_from_smarts(smarts)[0]
                if vg.atom_count > MAX_ATOM_COUNT:
                    continue
                rd = chem.MolFromSmarts(smarts)
                if rd is None:
                    continue
                patterns.append((vg, rd, smarts))
            except SmartsError:
                continue
            if len(patterns) >= PATTERN_LIMIT:
                break

    # Load molecules
    molecules = []
    with gzip.open(MOLECULES_GZ, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(maxsplit=1)
            if len(parts) < 2:
                continue
            chembl_id, smiles = parts
            try:
                cmol = cmol_from_smiles(smiles)[0]
                if cmol.atom_count > MAX_ATOM_COUNT:
                    continue
                vg = gmol_from_cmol(cmol)
                rd = chem.MolFromSmiles(smiles)
                if rd is None:
                    continue
                molecules.append((vg, rd, smiles))
            except SmartsError:
                continue
            if len(molecules) >= MOLECULE_LIMIT:
                break

    return patterns, molecules


@pytest.mark.perf
def test_perf_metal_vs_cpu_vs_rdkit_retro():
    try:
        probe_metal()
        metal = MetalMatcher()
    except GpuUnavailableError, RuntimeError:
        pytest.skip("Metal not available")

    cpu = CpuMatcher()

    print("\nLoading data...")
    patterns, molecules = _load_benchmark_data()

    pair_count = len(patterns) * len(molecules)
    print(
        f"Benchmark: {len(patterns)} patterns x {len(molecules)} molecules = {pair_count} pairs"
    )

    vesma_patterns = [p[0] for p in patterns]
    rd_patterns = [p[1] for p in patterns]
    vesma_molecules = [m[0] for m in molecules]
    rd_molecules = [m[1] for m in molecules]

    # --- RDKit ---
    print("Running RDKit Substructure Match...")
    start = time.perf_counter_ns()
    rdkit_hits = 0
    for rp in rd_patterns:
        for rm in rd_molecules:
            if rm.HasSubstructMatch(rp):
                rdkit_hits += 1
    rdkit_duration_ms = (time.perf_counter_ns() - start) / 1e6

    # --- CPU (Vesma) ---
    print("Running Vesma CpuMatcher (looped)...")
    start = time.perf_counter_ns()
    # has_match_many currently loops internally, but let's be explicit
    cpu_hits = cpu.has_match_many(vesma_patterns, vesma_molecules)
    cpu_duration_ms = (time.perf_counter_ns() - start) / 1e6

    # --- Metal (Vesma) ---
    print("Running Vesma MetalMatcher (batch)...")
    # Warmup
    metal.has_match_many(vesma_patterns[:1], vesma_molecules[:1])

    start = time.perf_counter_ns()
    metal_hits = metal.has_match_many(vesma_patterns, vesma_molecules)
    metal_duration_ms = (time.perf_counter_ns() - start) / 1e6

    # --- Results ---
    print("\n" + "=" * 80)
    print(f"{'Backend':<20} | {'Duration (ms)':<15} | {'Pairs/sec':<15} | {'Hits':<10}")
    print("-" * 80)
    print(
        f"{'RDKit':<20} | {rdkit_duration_ms:15.2f} | {pair_count / (rdkit_duration_ms / 1000):15.2f} | {rdkit_hits:<10}"
    )
    print(
        f"{'Vesma CPU':<20} | {cpu_duration_ms:15.2f} | {pair_count / (cpu_duration_ms / 1000):15.2f} | {np.sum(cpu_hits):<10}"
    )
    print(
        f"{'Vesma Metal':<20} | {metal_duration_ms:15.2f} | {pair_count / (metal_duration_ms / 1000):15.2f} | {np.sum(metal_hits):<10}"
    )
    print("=" * 80)

    print(f"\nSpeedup (Metal vs CPU):   {cpu_duration_ms / metal_duration_ms:.2f}x")
    print(f"Speedup (Metal vs RDKit): {rdkit_duration_ms / metal_duration_ms:.2f}x")

    # Check agreement
    # Note: RDKit and Vesma might disagree on some edge cases (aromaticity, etc.)
    # but they should be close.
    agreement_cpu_metal = np.mean(cpu_hits == metal_hits)
    print(f"\nAgreement (CPU vs Metal): {agreement_cpu_metal:.2%}")


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-m", "perf"])

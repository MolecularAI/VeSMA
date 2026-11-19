from __future__ import annotations

import shutil
import sys
import time
from statistics import median

import pytest

from vesma import cmol_from_smiles, gmol_from_cmol, gmol_from_smarts
from vesma.gmol import GMol
from vesma.metal_matcher import (
    GpuUnavailableError,
    has_match_cross_product_gpu,
    has_match_cross_product_reference,
)

pytestmark = [pytest.mark.perf, pytest.mark.gpu, pytest.mark.metal]


def pattern(smarts: str) -> GMol:
    return gmol_from_smarts(smarts)[0]


def molecule(smiles: str) -> GMol:
    return gmol_from_cmol(cmol_from_smiles(smiles)[0])


def _skip_if_no_metal_toolchain() -> None:
    if sys.platform != "darwin":
        pytest.skip("Metal backend is only available on macOS.")
    if shutil.which("clang++") is None:
        pytest.skip("clang++ is required for the Metal backend.")


def _measure(label: str, fn, repeat: int = 5) -> tuple[object, float]:
    timings_ms: list[float] = []
    result = fn()
    for _ in range(repeat):
        started = time.perf_counter_ns()
        result = fn()
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        timings_ms.append(elapsed_ms)
    print(f"{label}: median_ms={median(timings_ms):.3f}")
    return result, median(timings_ms)


@pytest.mark.perf
def test_perf_gpu_cross_product_small() -> None:
    _skip_if_no_metal_toolchain()
    chem = pytest.importorskip("rdkit.Chem")

    pattern_smarts = ["CO", "CC", "C1CC1", "N", "C=O", "O"]
    molecule_smiles = ["CCO", "CCC", "CC=O", "C1CCC1", "CCN", "O", "C1CC1", "CCCO"]

    patterns = [pattern(smarts) for smarts in pattern_smarts]
    molecules = [molecule(smiles) for smiles in molecule_smiles]
    rdkit_patterns = [chem.MolFromSmarts(smarts) for smarts in pattern_smarts]
    rdkit_molecules = [chem.MolFromSmiles(smiles) for smiles in molecule_smiles]
    assert all(query is not None for query in rdkit_patterns)
    assert all(mol is not None for mol in rdkit_molecules)

    reference, cpu_ms = _measure(
        "cpu_cross_product",
        lambda: has_match_cross_product_reference(patterns, molecules),
    )
    try:
        gpu, gpu_ms = _measure(
            "gpu_cross_product",
            lambda: has_match_cross_product_gpu(
                patterns,
                molecules,
                max_prunings=32,
                max_leaf_checks=4096,
            ),
        )
    except GpuUnavailableError as exc:
        pytest.skip(str(exc))

    assert gpu.shape == (len(patterns), len(molecules))
    assert (gpu == reference).all()
    _, rdkit_ms = _measure(
        "rdkit_cross_product",
        lambda: [
            [mol.HasSubstructMatch(query) for mol in rdkit_molecules]
            for query in rdkit_patterns
        ],
    )
    print(f"pairs={gpu.size}")
    print(f"cpu_pairs_per_sec={gpu.size / (cpu_ms / 1000):.2f}")
    print(f"gpu_pairs_per_sec={gpu.size / (gpu_ms / 1000):.2f}")
    print(f"rdkit_pairs_per_sec={gpu.size / (rdkit_ms / 1000):.2f}")
    print(f"speedup={cpu_ms / gpu_ms:.3f}x")
    print(f"gpu_vs_rdkit_speedup={rdkit_ms / gpu_ms:.3f}x")
    print(f"cpu_vs_rdkit_speedup={rdkit_ms / cpu_ms:.3f}x")

"""Standalone short-ring WL kernel microbenchmark.

Times the WL hash kernel in isolation from hash transfer, Python token construction,
host deduplication, and retained-molecule transfer. Per case it verifies the kernel
against ``short_ring_wl_molecule_id`` and measures pure-kernel throughput via CUDA events
(median of ``CUDA_WL_PROFILE_SAMPLES`` runs), recording the detail in ``benchmark.extra_info``
(pytest-benchmark), written out with ``--benchmark-json``.

Run: ``make bench-cuda-wl-key``.
"""

from __future__ import annotations

import os
from statistics import median
from typing import Any

import pytest

from vesma import cmol_from_smiles, gmol_from_cmol
from vesma.cuda_retro import (
    CudaMoleculeBatch,
    get_wl_hash_kernel,
    prepare_cuda_molecule_batch,
    wl_hash_kernel,
)
from vesma.identity import short_ring_wl_molecule_id

pytestmark = [
    pytest.mark.perf,
    pytest.mark.gpu,
    pytest.mark.cuda,
    pytest.mark.paper_benchmark,
]

CASES = {
    "linear-30": "C" * 30,
    "linear-46": "C" * 46,
    "polycycle-24": "C1CCC(CC1)C2CCC(CC2)C3CCC(CC3)C4CCC(CC4)",
    "aromatic-30": "c1ccccc1c2ccccc2c3ccccc3c4ccccc4c5ccccc5",
}


def ensure_cuda() -> Any:
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CuPy cannot access CUDA on this machine.")
    return cp


def repeated_batch(cp: Any, smiles: str, count: int) -> CudaMoleculeBatch:
    molecule = cmol_from_smiles(smiles)[0]
    base = prepare_cuda_molecule_batch([((0, 0), gmol_from_cmol(molecule))])
    return CudaMoleculeBatch(
        tokens=(),
        atom_counts_device=cp.repeat(base.atom_counts_device, count, axis=0),
        atoms_device=cp.repeat(base.atoms_device, count, axis=0),
        bonds_device=cp.repeat(base.bonds_device, count, axis=0),
        rows_device=cp.repeat(base.rows_device, count, axis=0),
    )


def measure_kernel_seconds(cp: Any, batch: CudaMoleculeBatch) -> float:
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()
    wl_hash_kernel(batch)
    stop.record()
    stop.synchronize()
    return float(cp.cuda.get_elapsed_time(start, stop) / 1000.0)


def run_sweep(cp: Any, count: int, samples: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for name, smiles in CASES.items():
        molecule = cmol_from_smiles(smiles)[0]
        expected = short_ring_wl_molecule_id(molecule)
        batch = repeated_batch(cp, smiles, count)
        observed_lo, observed_hi = wl_hash_kernel(batch)
        cp.cuda.Stream.null.synchronize()
        assert (int(observed_lo.get()[0]), int(observed_hi.get()[0])) == expected

        timings = [measure_kernel_seconds(cp, batch) for _ in range(samples)]
        middle = median(timings)
        results.append(
            {
                "name": name,
                "atom_count": molecule.atom_count,
                "molecule_count": count,
                "samples_seconds": timings,
                "median_seconds": middle,
                "molecules_per_second": count / middle,
                "microseconds_per_molecule": middle * 1e6 / count,
            }
        )
        del batch, observed_lo, observed_hi
        cp.get_default_memory_pool().free_all_blocks()
    return results


def test_perf_cuda_wl_key(benchmark: Any) -> None:
    cp = ensure_cuda()
    count = int(os.environ.get("CUDA_WL_PROFILE_MOLECULE_COUNT", "50000"))
    samples = int(os.environ.get("CUDA_WL_PROFILE_SAMPLES", "5"))
    if count <= 0 or samples <= 0:
        raise ValueError("CUDA WL profile counts must be positive.")

    def target() -> list[dict[str, Any]]:
        return run_sweep(cp, count, samples)

    results = benchmark.pedantic(target, rounds=1, warmup_rounds=0, iterations=1)
    print_table(results)
    benchmark.extra_info.update(
        wl_key_extra_info(cp, results, count=count, samples=samples)
    )


def wl_key_extra_info(
    cp: Any,
    results: list[dict[str, Any]],
    *,
    count: int,
    samples: int,
) -> dict[str, Any]:
    return {
        "experiment": "cuda-short-ring-wl-key",
        "backend": "cuda",
        "molecule_count": count,
        "samples": samples,
        "case_names": list(CASES),
        "kernel_attributes": {
            key: int(value) for key, value in get_wl_hash_kernel(cp).attributes.items()
        },
        "per_case": results,
    }


def print_table(results: list[dict[str, Any]]) -> None:
    print(
        f"{'case':>13} {'atoms':>6} {'mols':>8} "
        f"{'median_s':>10} {'mol/s':>14} {'us/mol':>9}"
    )
    for row in results:
        print(
            f"{row['name']:>13} {row['atom_count']:>6} {row['molecule_count']:>8} "
            f"{row['median_seconds']:>10.6f} {row['molecules_per_second']:>14.0f} "
            f"{row['microseconds_per_molecule']:>9.4f}"
        )

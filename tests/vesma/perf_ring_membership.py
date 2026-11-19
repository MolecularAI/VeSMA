"""What would exact SMARTS `R`/`r` cost next to the short-ring check the pipeline runs today?

Answers a proposal to perceive `R` and `r` on the GPU with Horton candidates and size-tiered GF(2)
elimination (relevant cycles). Three measurements, on the n1-d2 targets and a stock sample:

* correctness -- the Horton prototype against exact relevant cycles (every simple cycle,
  tiered GF(2)) on cages and bridged scaffolds, and every perception against RDKit's `R`/`r`;
* CPU cost per molecule and word operations per molecule, for the current short-ring BFS,
  the BFS-distance `r`, the Horton/GF(2) `R`+`r`, and the forest `R`/`r` the CPU uses today;
* Metal kernel time per molecule for the same three device algorithms, one thread per molecule.

Run: ``make bench-ring-membership`` (the Metal case needs the sandbox disabled).
"""

from __future__ import annotations

import ctypes
import gzip
import os
import platform
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pytest

from tests.vesma.ring_membership_prototypes import (
    RingMembership,
    adjacency_rows,
    brute_relevant_ring_info,
    cycle_rank,
    horton_relevant_ring_info,
    ring_rows,
    short_ring_edge_rows,
    smallest_cycle_r,
)
from vesma.adjacency import BoolBondMatrix
from vesma.rings import ring_edge_mask, ring_info_from_presence, short_cycle_edge_mask

Chem = pytest.importorskip("rdkit.Chem")

pytestmark = [pytest.mark.perf]

PROCESSED = Path(__file__).parents[1] / "data/external/paroutes/processed"
METAL_SOURCE = Path(__file__).with_name("perf_ring_membership_metal.mm")

STRESS = {
    "cubane": "C12C3C4C1C5C2C3C45",
    "prismane": "C12C3C1C4C2C34",
    "adamantane": "C1C2CC3CC1CC(C2)C3",
    "norbornane": "C1CC2CCC1C2",
    "bicyclo222octane": "C1CC2CCC1CC2",
    "bicyclo111pentane": "C1C2CC1C2",
    "quinuclidine": "C1CN2CCC1CC2",
    "twistane": "C1CC2CC3CC1CC2C3",
    "basketane": "C1=CC2C3C1C4C2C=CC34",
    "cholesterol": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",
    "dodecahedrane": "C12C3C4C5C1C6C7C2C8C3C9C4C%10C5C6C%11C7C8C9C%10%11",
    "strychnine": "O=C7N2c1ccccc1C64C2C3C(OCC=C5C3CC6N(CC4)C5)C7",
    "morphine": "CN1CCC23c4c5ccc(O)c4OC2C(O)C=CC3C1C5",
    "coronene": "c1cc2ccc3ccc4ccc5ccc6ccc1c7c2c3c4c5c67",
    "triptycene": "c1ccc2c(c1)C1c3ccccc3C2c2ccccc21",
    "paracyclophane22": "c1cc2ccc1CCc1ccc(cc1)CC2",
}


def load_smiles(path: Path, limit: int | None) -> list[str]:
    with gzip.open(path, "rt") as handle:
        smiles = [line.split()[0] for line in handle if line.strip()]
    if limit is not None:
        random.Random(0).shuffle(smiles)
        smiles = smiles[:limit]
    return smiles


def corpus(name: str) -> list[Any]:
    sources = {
        "n1-d2-f0": (PROCESSED / "n1-d2-f0.smi.gz", None),
        "stock3k": (PROCESSED / "n1-stock.smi.gz", 3000),
        "stress": None,
    }
    if name == "stress":
        smiles = list(STRESS.values())
    else:
        path, limit = sources[name]
        smiles = load_smiles(path, limit)
    molecules = [Chem.MolFromSmiles(entry) for entry in smiles]
    return [m for m in molecules if m is not None and m.GetNumAtoms() <= 64]


def adjacency(molecule: Any) -> BoolBondMatrix:
    return Chem.GetAdjacencyMatrix(molecule).astype(bool)


def rdkit_membership(molecule: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    info = molecule.GetRingInfo()
    atoms = range(molecule.GetNumAtoms())
    return (
        tuple(info.NumAtomRings(atom) for atom in atoms),
        tuple(info.MinAtomRingSize(atom) for atom in atoms),
    )


@pytest.mark.parametrize("name", list(STRESS))
def test_horton_prototype_is_exact_relevant_cycles(name: str) -> None:
    molecule = Chem.MolFromSmiles(STRESS[name])
    matrix = adjacency(molecule)
    prototype = horton_relevant_ring_info(ring_rows(matrix))
    exact = brute_relevant_ring_info(matrix)
    count, smallest = rdkit_membership(molecule)
    assert prototype.count == exact.count
    assert prototype.smallest == exact.smallest
    assert prototype.smallest == smallest
    assert smallest_cycle_r(ring_rows(matrix))[0] == smallest
    assert prototype.count == count


def agreement(molecules: Sequence[Any]) -> dict[str, float]:
    """Fraction of atoms each perception agrees with RDKit on (forest values capped at 7)."""
    totals = dict.fromkeys(("horton_R", "horton_r", "bfs_r", "forest_R", "forest_r"), 0)
    atoms = 0
    for molecule in molecules:
        matrix = adjacency(molecule)
        rows = ring_rows(matrix)
        horton = horton_relevant_ring_info(rows)
        bfs = smallest_cycle_r(rows)[0]
        forest = ring_info_from_presence(matrix)
        count, smallest = rdkit_membership(molecule)
        for atom in range(molecule.GetNumAtoms()):
            atoms += 1
            totals["horton_R"] += horton.count[atom] == count[atom]
            totals["horton_r"] += horton.smallest[atom] == smallest[atom]
            totals["bfs_r"] += bfs[atom] == smallest[atom]
            totals["forest_R"] += int(forest.membership_count[atom]) == min(
                count[atom], 7
            )
            totals["forest_r"] += int(forest.smallest_ring_size[atom]) == min(
                smallest[atom], 7
            )
    return {key: value / atoms for key, value in totals.items()}


CPU_METHODS: dict[str, tuple[str, Callable[[Any], Any]]] = {
    # input kind, function
    "bridge_ring_edge_mask": ("matrix", ring_edge_mask),
    "short_bfs_all_edges": ("rows", short_ring_edge_rows),
    "short_bfs_ring_edges": ("ring_rows", short_ring_edge_rows),
    "bfs_r": ("ring_rows", smallest_cycle_r),
    "horton_R_r": ("ring_rows", horton_relevant_ring_info),
    "forest_R_r": ("matrix", ring_info_from_presence),
    "cpu_3_8_enumerator": ("matrix", short_cycle_edge_mask),
}


def ops_of(result: Any) -> int:
    if isinstance(result, RingMembership):
        return result.ops
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result[1]
    return 0


@pytest.mark.parametrize("corpus_name", ["n1-d2-f0", "stock3k", "stress"])
@pytest.mark.parametrize("method", list(CPU_METHODS))
def test_perf_cpu_ring_membership(
    benchmark: Any, corpus_name: str, method: str
) -> None:
    molecules = corpus(corpus_name)
    matrices = [adjacency(m) for m in molecules]
    kind, function = CPU_METHODS[method]
    inputs = {
        "matrix": matrices,
        "rows": [adjacency_rows(m) for m in matrices],
        "ring_rows": [ring_rows(m) for m in matrices],
    }[kind]

    def target() -> list[Any]:
        return [function(item) for item in inputs]

    results = benchmark.pedantic(target, rounds=3, warmup_rounds=1, iterations=1)
    ops = np.array([ops_of(result) for result in results])
    benchmark.extra_info.update(
        {
            "experiment": "ring-membership-cpu",
            "corpus": corpus_name,
            "method": method,
            "molecules": len(molecules),
            "mean_atoms": float(np.mean([m.GetNumAtoms() for m in molecules])),
            "microseconds_per_molecule": benchmark.stats.stats.min
            * 1e6
            / len(molecules),
            "ops_mean": float(ops.mean()),
            "ops_p99": float(np.percentile(ops, 99)),
            "ops_max": int(ops.max()),
        }
    )
    if method == "horton_R_r":
        mus = np.array([cycle_rank(rows) for rows in inputs])
        candidates = np.array([result.candidates for result in results])
        benchmark.extra_info.update(
            {
                "mu_mean": float(mus.mean()),
                "mu_max": int(mus.max()),
                "candidates_mean": float(candidates.mean()),
                "candidates_max": int(candidates.max()),
                "rdkit_agreement": agreement(molecules),
            }
        )


def load_bench_library() -> ctypes.CDLL:
    if platform.system() != "Darwin":
        pytest.skip("Metal is only available on macOS.")
    from vesma.metal_runtime import load_metal_dylib

    library = load_metal_dylib(METAL_SOURCE)
    library.ring_bench_run.restype = ctypes.c_int
    library.ring_bench_run.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    library.ring_bench_last_error.restype = ctypes.c_char_p
    return library


def pack(rows_list: Sequence[tuple[int, ...]]) -> tuple[np.ndarray, np.ndarray]:
    rows = np.zeros((len(rows_list), 64), dtype=np.uint64)
    counts = np.zeros(len(rows_list), dtype=np.uint32)
    for index, rows_of in enumerate(rows_list):
        rows[index, : len(rows_of)] = rows_of
        counts[index] = len(rows_of)
    return rows, counts


KERNELS = {
    "short": 0,
    "bfs_r": 1,
    "horton": 2,
    "short_coop": 3,
    "horton_coop": 4,
    "horton_lean": 5,
}


def run_kernel(
    library: ctypes.CDLL,
    kernel: str,
    rows: np.ndarray,
    counts: np.ndarray,
    repeats: int,
    threads_per_group: int,
) -> dict[str, np.ndarray]:
    mol_count = len(counts)
    out_rows = np.zeros((mol_count, 64), dtype=np.uint64)
    out_count = np.zeros((mol_count, 64), dtype=np.uint8)
    out_r = np.zeros((mol_count, 64), dtype=np.uint8)
    out_status = np.zeros(mol_count, dtype=np.uint32)
    seconds = np.zeros(repeats, dtype=np.float64)
    code = library.ring_bench_run(
        KERNELS[kernel],
        rows.ctypes.data,
        counts.ctypes.data,
        mol_count,
        threads_per_group,
        repeats,
        out_rows.ctypes.data,
        out_count.ctypes.data,
        out_r.ctypes.data,
        out_status.ctypes.data,
        seconds.ctypes.data,
    )
    if code != 0:
        raise RuntimeError(library.ring_bench_last_error().decode())
    return {
        "rows": out_rows,
        "count": out_count,
        "r": out_r,
        "status": out_status,
        "seconds": seconds,
    }


def rows_from_bits(bits: np.ndarray, atom_count: int) -> tuple[int, ...]:
    return tuple(int(bits[atom]) for atom in range(atom_count))


def verify_metal(
    outputs: dict[str, dict[str, np.ndarray]], ring_rows_list: Sequence[tuple[int, ...]]
) -> None:
    for index, rows in enumerate(ring_rows_list):
        n = len(rows)
        edges, _ = short_ring_edge_rows(rows)
        expected_rows = [0] * n
        for left, right in edges:
            expected_rows[left] |= 1 << right
            expected_rows[right] |= 1 << left
        for kernel in ("short", "short_coop"):
            observed = rows_from_bits(outputs[kernel]["rows"][index], n)
            assert observed == tuple(expected_rows), kernel
        assert tuple(outputs["bfs_r"]["r"][index, :n]) == smallest_cycle_r(rows)[0]
        horton = horton_relevant_ring_info(rows)
        for kernel in ("horton", "horton_coop", "horton_lean"):
            assert outputs[kernel]["status"][index] == 0, kernel
            assert tuple(outputs[kernel]["count"][index, :n]) == horton.count, kernel
            assert tuple(outputs[kernel]["r"][index, :n]) == horton.smallest, kernel


@pytest.mark.gpu
@pytest.mark.metal
@pytest.mark.parametrize("corpus_name", ["n1-d2-f0", "stock3k", "stress"])
def test_perf_metal_ring_membership(benchmark: Any, corpus_name: str) -> None:
    library = load_bench_library()
    target_count = int(os.environ.get("RING_BENCH_MOLECULES", "200000"))
    repeats = int(os.environ.get("RING_BENCH_REPEATS", "7"))
    threads_per_group = int(os.environ.get("RING_BENCH_THREADS", "64"))
    unique = [ring_rows(adjacency(m)) for m in corpus(corpus_name)]
    tiles = -(-target_count // len(unique))
    rows, counts = pack(unique)
    rows = np.ascontiguousarray(np.tile(rows, (tiles, 1))[:target_count])
    counts = np.ascontiguousarray(np.tile(counts, tiles)[:target_count])

    def target() -> dict[str, dict[str, np.ndarray]]:
        return {
            kernel: run_kernel(
                library, kernel, rows, counts, repeats, threads_per_group
            )
            for kernel in KERNELS
        }

    started = time.perf_counter()
    outputs = benchmark.pedantic(target, rounds=1, warmup_rounds=0, iterations=1)
    verify_metal(outputs, unique)
    per_kernel = {
        kernel: {
            "median_seconds": float(median(result["seconds"][1:])),
            "nanoseconds_per_molecule": float(
                median(result["seconds"][1:]) * 1e9 / len(counts)
            ),
        }
        for kernel, result in outputs.items()
    }
    short = per_kernel["short"]["median_seconds"]
    for kernel in per_kernel:
        per_kernel[kernel]["ratio_to_short"] = (
            per_kernel[kernel]["median_seconds"] / short
        )
    print()
    for kernel, entry in per_kernel.items():
        print(
            f"{corpus_name:>9} {kernel:>7}: {entry['nanoseconds_per_molecule']:8.2f} ns/mol "
            f"x{entry['ratio_to_short']:.2f} vs short"
        )
    benchmark.extra_info.update(
        {
            "experiment": "ring-membership-metal",
            "corpus": corpus_name,
            "unique_molecules": len(unique),
            "molecules": len(counts),
            "repeats": repeats,
            "threads_per_group": threads_per_group,
            "wall_seconds": time.perf_counter() - started,
            "per_kernel": per_kernel,
        }
    )

from __future__ import annotations

import numpy as np
import pytest

from tests.vesma.bench_matcher import _require_metal_reaction_matcher
from vesma import cmol_from_smiles
from vesma.cpu_matcher import CpuReactionMatcher
from vesma.reaction import parse_reaction_smarts
from vesma.rejection import by_name, no_rejections

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("metal", marks=pytest.mark.metal),
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize("added,charge", [("N", 0), ("[NH3+]", 1), ("[O-]", -1)])
def test_added_atoms_have_concrete_charge(
    backend: str, added: str, charge: int
) -> None:
    if backend == "metal":
        matcher_class = _require_metal_reaction_matcher()
    else:
        cp = pytest.importorskip("cupy")
        if not cp.cuda.is_available():
            pytest.skip("CUDA is unavailable")
        from vesma.cuda_matcher import CudaReactionMatcher

        matcher_class = CudaReactionMatcher
    rules = [parse_reaction_smarts(f"[C:1]>>[C:1]{added}")]
    mols = [cmol_from_smiles("C")[0]]
    cpu_batch, _ = CpuReactionMatcher(rules).match(mols, n_jobs=1)
    batch, _ = matcher_class(rules).match(mols)
    assert len(batch) == len(cpu_batch) == 1
    words = batch.atoms[0] if backend == "metal" else batch.atoms_device[0].get()
    masks = words.reshape(64, 14)[:2, 5]
    np.testing.assert_array_equal(masks, [1 << 3, 1 << (charge + 3)])

    probes = [
        parse_reaction_smarts(rule) for rule in ("[N+:1]>>[N+:1]", "[O-:1]>>[O-:1]")
    ]
    cpu = CpuReactionMatcher(probes)
    _, cpu_hits = cpu.match(cpu_batch, n_jobs=1)
    gpu = matcher_class(probes)
    _, gpu_hits = gpu.match(batch)
    assert len(gpu_hits) == len(cpu_hits)
    assert by_name(
        tuple(gpu.last_diagnostics.rejections) or no_rejections()
    ) == by_name(cpu.last_diagnostics.rejections or no_rejections())

"""CUDA isotope matching and nuclear conservation use the CPU's semantics."""

from __future__ import annotations

import numpy as np
import pytest

from tests.vesma.test_metal_isotopes import MOLECULES, PATTERNS
from vesma import cmol_from_smiles, gmol_from_cmol, gmol_from_smarts
from vesma.cpu_matcher import CpuReactionMatcher, has_match_many
from vesma.cuda_matcher import CudaMatcher, CudaReactionMatcher
from vesma.gpu_atom_masks import ISOTOPE_SHIFT, ISOTOPE_WORD
from vesma.reaction import parse_reaction_smarts
from vesma.rejection import RejectionReason

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


@pytest.fixture(scope="module", autouse=True)
def cuda_available() -> None:
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CUDA is unavailable")


def test_isotope_matching_agrees_with_cpu() -> None:
    patterns = [gmol_from_smarts(s)[0] for s in PATTERNS]
    molecules = [gmol_from_cmol(cmol_from_smiles(s)[0]) for s in MOLECULES]
    np.testing.assert_array_equal(
        CudaMatcher().has_match_many(patterns, molecules),
        has_match_many(patterns, molecules, n_jobs=1),
    )


@pytest.mark.parametrize(
    ("rule", "smiles", "expected_masses"),
    [
        ("[*:1]>>[*:1]", "[13CH3][18OH]", [13, 18]),
        ("[O:1]>>[O:1][13CH3]", "CO", [0, 0, 13]),
        ("[O:1]>>[O:1]C", "CO", [0, 0, 0]),
    ],
)
def test_generated_atoms_keep_or_take_the_correct_isotope(
    rule: str, smiles: str, expected_masses: list[int]
) -> None:
    matcher = CudaReactionMatcher([parse_reaction_smarts(rule)])
    generated, _ = matcher.match([cmol_from_smiles(smiles)[0]])
    assert len(generated) > 0
    words = generated.atoms_device.get().reshape(-1, 64, 14)
    counts = generated.atom_counts_device.get()
    for row, count in zip(words, counts, strict=True):
        observed = (row[:count, ISOTOPE_WORD] >> ISOTOPE_SHIFT).tolist()
        assert sorted(observed) == expected_masses


def test_mapped_atom_cannot_change_isotope() -> None:
    rules = [parse_reaction_smarts("[C:1]>>[13C:1]")]
    molecules = [cmol_from_smiles("CO")[0]]
    cpu, cpu_hits = CpuReactionMatcher(rules).match(molecules, n_jobs=1)
    matcher = CudaReactionMatcher(rules)
    generated, hits = matcher.match(molecules)
    assert len(cpu) == len(cpu_hits) == len(generated) == len(hits) == 0
    assert matcher.last_diagnostics is not None
    assert (
        matcher.last_diagnostics.rejections[
            RejectionReason.NUCLEAR_CONSERVATION_VIOLATED
        ]
        == 1
    )

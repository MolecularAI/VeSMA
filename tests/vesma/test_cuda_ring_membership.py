"""CUDA-generated R/r descriptors agree with CPU perception, including tied paths."""

from __future__ import annotations

import numpy as np
import pytest

from tests.vesma.test_ring_membership import CAGES
from vesma import cmol_from_smiles
from vesma.cuda_matcher import CudaReactionMatcher
from vesma.interop import cmol_from_mol
from vesma.reaction import parse_reaction_smarts

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


@pytest.fixture(scope="module", autouse=True)
def cuda_available() -> None:
    cp = pytest.importorskip("cupy")
    if not cp.cuda.is_available():
        pytest.skip("CUDA is unavailable")


@pytest.mark.parametrize("smiles", [*CAGES.values(), "CC", "C1CCCCCCCCCCC1"])
@pytest.mark.parametrize("reverse", [False, True])
def test_generated_ring_membership_matches_cpu(smiles: str, reverse: bool) -> None:
    from rdkit import Chem

    product = cmol_from_smiles(smiles)[0]
    if reverse:
        mol = Chem.MolFromSmiles(smiles)
        product = cmol_from_mol(
            Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        )
    matcher = CudaReactionMatcher([parse_reaction_smarts("[*:1]>>[*:1]")])
    generated, _ = matcher.match([product])
    assert len(generated) > 0
    words = generated.atoms_device.get().reshape(-1, 64, 14)
    for row in words:
        np.testing.assert_array_equal(row[: product.atom_count, 11], 1 << product.R)
        np.testing.assert_array_equal(row[: product.atom_count, 12], 1 << product.r)

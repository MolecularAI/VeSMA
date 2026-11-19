from __future__ import annotations

import pytest

from vesma import cmol_from_smiles, gmol_from_cmol, gmol_from_smarts
from vesma.cpu_matcher import has_match

KNOEVENAGEL_LIKE_LHS = "".join(
    (
        r"[#7;a:4]:[c:3]:[c:2]/[CH;D2;+0:1]=[C;H0;D3;+0:6]",
        r"(\[#8:5])-[C:7]=[O;D1;H0:8]",
    )
)
CHROMONE_LIKE_TARGET = "".join(
    (
        r"CC(C)(C)OC(=O)N1CCN(Cc2c(O)ccc3c2O/C(=C\c2cn(S(C)(=O)=O)",
        r"c4ccccc24)C3=O)CC1",
    )
)


def test_knoevenagel_chromone_aromaticity_matches_rdkit() -> None:

    Chem = pytest.importorskip("rdkit.Chem")

    rdkit_pattern = Chem.MolFromSmarts(KNOEVENAGEL_LIKE_LHS)
    rdkit_molecule = Chem.MolFromSmiles(CHROMONE_LIKE_TARGET)
    assert rdkit_pattern is not None
    assert rdkit_molecule is not None

    pattern = gmol_from_smarts(KNOEVENAGEL_LIKE_LHS)[0]
    molecule = gmol_from_cmol(cmol_from_smiles(CHROMONE_LIKE_TARGET)[0])
    concrete = cmol_from_smiles(CHROMONE_LIKE_TARGET)[0]

    assert rdkit_molecule.GetSubstructMatches(rdkit_pattern) == (
        (24, 23, 22, 21, 20, 19, 35, 36),
    )
    assert concrete.A[[19, 20, 35]].tolist() == [1, 1, 1]
    assert has_match(pattern, molecule)

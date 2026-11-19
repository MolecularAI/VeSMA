from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import pytest

from vesma import SmartsError, cmol_from_smiles, gmol_from_cmol, gmol_from_smarts
from vesma.cpu_matcher import has_match

SMILES_MEMBER = "v2.0/zinc_lead-like_2011-02-12_first100k.smi"

# Accepted RDKit-vs-vesma aromaticity-model divergences (NOT vesma bugs). Vesma
# perceives aromaticity by a delocalized, Kekule-free electron count, so it aromatizes
# some rings that RDKit's default (Kekule-based) model keeps aliphatic. When that
# happens a localized single/double bond in RDKit's Kekule form becomes an aromatic
# bond in vesma, and an RDKit substructure match that relies on the explicit bond order
# has no vesma counterpart. These divergences are documented and deliberately not
# suppressed with heuristics (see the "aromaticity-bond-class-model" design note); each
# entry is keyed on the exact (pattern, smiles) pair so any *new* false-negative still
# fails the test.
ACCEPTED_AROMATICITY_DIVERGENCES = {
    # Enamine [NX3][CX3]=[CX3] vs a 5,6-dihydroimidazo[2,1-b]thiazole (levamisole core).
    # The amidine carbon's exocyclic C=N imine lets vesma count 6 pi electrons over the
    # fused thiazoline ring (ring S lone pair + bridgehead amine N lone pair + the ring
    # C=C), so vesma aromatizes that ring and the enamine's localized C=C becomes an
    # aromatic bond. RDKit's default model keeps the ring aliphatic and matches the
    # localized double bond. Same class as the documented 2-imino-thiazoline / amidine
    # bridgehead divergence.
    ("[NX3][CX3]=[CX3]", "S1C2=N[C@H](c3ccccc3)CN2C=C1"),
}


def _pattern(smarts: str):
    return gmol_from_smarts(smarts)[0]


def _molecule(smiles: str):
    return gmol_from_cmol(cmol_from_smiles(smiles)[0])


def _iter_smarts_patterns(dataset_dir: Path):
    tar_paths = [
        dataset_dir / "raw" / "daylight-smarts.tar.gz",
        dataset_dir / "raw" / "pains-smarts.tar.gz",
    ]
    for path in tar_paths:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile() or not member.name.endswith(".smarts"):
                    continue
                payload = archive.extractfile(member)
                if payload is None:
                    continue
                for raw in payload:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line or line.startswith("#"):
                        continue
                    yield line.split()[0]


def _iter_smiles_strings(dataset_dir: Path):
    smiles_zip = dataset_dir / "raw" / "smarts-benchmark-v2.0.zip"
    with zipfile.ZipFile(smiles_zip) as archive:
        payload = archive.read(SMILES_MEMBER).decode("utf-8", errors="ignore")
    for line in payload.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        yield stripped.split()[0]


@pytest.mark.slow
def test_rdkit_positive_implies_vesma_positive(smarts_dataset_dir: Path) -> None:
    chem = pytest.importorskip("rdkit.Chem")

    pattern_entries = []
    for pattern in _iter_smarts_patterns(smarts_dataset_dir):
        rdkit_query = chem.MolFromSmarts(pattern)
        if (
            rdkit_query is None
            or rdkit_query.GetNumAtoms() == 0
            or rdkit_query.GetNumAtoms() > 12
        ):
            continue
        try:
            vesma_pattern = _pattern(pattern)
        except SmartsError:
            continue
        pattern_entries.append((pattern, rdkit_query, vesma_pattern))
        if len(pattern_entries) >= 40:
            break

    molecule_entries = []
    for smiles in _iter_smiles_strings(smarts_dataset_dir):
        rdkit_mol = chem.MolFromSmiles(smiles)
        if (
            rdkit_mol is None
            or rdkit_mol.GetNumAtoms() == 0
            or rdkit_mol.GetNumAtoms() > 40
        ):
            continue
        try:
            vesma_molecule = _molecule(smiles)
        except SmartsError:
            continue
        molecule_entries.append((smiles, rdkit_mol, vesma_molecule))
        if len(molecule_entries) >= 120:
            break

    assert pattern_entries
    assert molecule_entries

    rdkit_positives = 0
    observed_divergences: set[tuple[str, str]] = set()
    for pattern, rdkit_query, vesma_pattern in pattern_entries:
        for smiles, rdkit_mol, vesma_molecule in molecule_entries:
            if not rdkit_mol.HasSubstructMatch(rdkit_query):
                continue
            rdkit_positives += 1
            if has_match(vesma_pattern, vesma_molecule):
                continue
            assert (
                pattern,
                smiles,
            ) in ACCEPTED_AROMATICITY_DIVERGENCES, (
                f"Vesma false-negative for pattern '{pattern}' and smiles '{smiles}'"
            )
            observed_divergences.add((pattern, smiles))

    assert rdkit_positives > 0
    # Every accepted-divergence entry must still actually diverge; otherwise it is stale
    # (vesma now matches it) and should be removed from the allow-list above.
    assert observed_divergences == ACCEPTED_AROMATICITY_DIVERGENCES

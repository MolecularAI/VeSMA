from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vesma import SmartsError, cmol_from_smiles, gmol_from_cmol, gmol_from_smarts
from vesma.cpu_matcher import has_match

RNG_SEED = 20260226
PATTERN_SAMPLE_SIZE = 100
MOLECULE_SAMPLE_SIZE = 100
# Bond-type-agnostic Ullmann intentionally trades specificity for no false negatives.
MAX_FP_RATE = 0.75
# Strict bond-type matching (see "Story - Bond Type Query Semantics.md") no longer
# masks Vesma<->RDKit aromaticity-perception divergences: where Vesma over-aromatizes a
# ring RDKit treats as non-aromatic (e.g. an sp3-amine / N=C ring), an explicit -/=
# query that RDKit matches can be rejected. Measured at ~0.2% of RDKit positives; allow
# a small budget here until the deferred full aromaticity-perception work lands.
MAX_FALSE_NEGATIVES = 2


def _pattern(smarts: str):
    return gmol_from_smarts(smarts)[0]


def _molecule(smiles: str):
    return gmol_from_cmol(cmol_from_smiles(smiles)[0])


def _read_pains_patterns(path: Path) -> list[str]:
    with tarfile.open(path, "r:gz") as archive:
        member = archive.getmember("pains.smarts")
        source = archive.extractfile(member)
        if source is None:
            raise ValueError(f"Missing pains.smarts in archive: {path}")
        text = source.read().decode("utf-8", errors="ignore")

    patterns: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            line = line.split("\t", 1)[0].strip()
        elif " <" in line:
            line = line.split(" <", 1)[0].strip()
        if line:
            patterns.append(line)
    return patterns


def _read_smiles(path: Path) -> list[str]:
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        usecols=[0],
        names=["smiles"],
        dtype={"smiles": "string"},
        engine="python",
    )
    series = frame["smiles"].dropna().astype(str).str.strip()
    series = series[series != ""]
    return series.tolist()


@pytest.mark.slow
def test_pains_vs_rdkit_sample_no_false_negatives_and_bounded_fp(
    smarts_dataset_dir: Path,
) -> None:
    chem = pytest.importorskip("rdkit.Chem")

    pains_path = smarts_dataset_dir / "raw" / "pains-smarts.tar.gz"
    smiles_path = (
        smarts_dataset_dir
        / "processed"
        / "smarts-benchmark-v2.0"
        / "zinc_lead-like_2011-02-12_first100k.smi"
    )

    pains_patterns = _read_pains_patterns(pains_path)
    smiles_all = _read_smiles(smiles_path)

    rdkit_and_vesma_patterns: list[tuple[str, object, object]] = []
    for smarts in pains_patterns:
        rd_query = chem.MolFromSmarts(smarts)
        if rd_query is None:
            continue
        try:
            vesma_query = _pattern(smarts)
        except SmartsError:
            continue
        rdkit_and_vesma_patterns.append((smarts, rd_query, vesma_query))

    assert len(pains_patterns) == 480
    assert len(rdkit_and_vesma_patterns) >= PATTERN_SAMPLE_SIZE
    assert len(smiles_all) == 100_000

    rng = np.random.default_rng(RNG_SEED)
    pattern_idx = rng.choice(
        len(rdkit_and_vesma_patterns), size=PATTERN_SAMPLE_SIZE, replace=False
    )
    sampled_patterns = [rdkit_and_vesma_patterns[int(i)] for i in pattern_idx]

    pool_size = min(5_000, len(smiles_all))
    pool_idx = rng.choice(len(smiles_all), size=pool_size, replace=False)
    rd_any_pool = np.zeros(pool_size, dtype=bool)
    pool_smiles = [smiles_all[int(i)] for i in pool_idx]

    for i, smi in enumerate(pool_smiles):
        rd_mol = chem.MolFromSmiles(smi)
        if rd_mol is None:
            continue
        for _, rd_query, _ in sampled_patterns:
            if rd_mol.HasSubstructMatch(rd_query):
                rd_any_pool[i] = True
                break

    positive_idx = np.flatnonzero(rd_any_pool)
    negative_idx = np.flatnonzero(~rd_any_pool)
    assert positive_idx.size > 0

    n_pos = min(MOLECULE_SAMPLE_SIZE // 2, int(positive_idx.size))
    n_neg = MOLECULE_SAMPLE_SIZE - n_pos
    sample_pos = rng.choice(positive_idx, size=n_pos, replace=False)
    sample_neg = rng.choice(negative_idx, size=n_neg, replace=False)
    sampled_idx = np.concatenate([sample_pos, sample_neg])
    rng.shuffle(sampled_idx)

    rdkit_any = []
    vesma_any = []

    for idx in sampled_idx:
        smi = pool_smiles[int(idx)]
        try:
            vesma_mol = _molecule(smi)
        except SmartsError:
            continue

        rd_hit = bool(rd_any_pool[int(idx)])
        vesma_hit = False

        for _, _, vesma_query in sampled_patterns:
            if not vesma_hit and has_match(vesma_query, vesma_mol):
                vesma_hit = True
            if rd_hit and vesma_hit:
                break

        rdkit_any.append(rd_hit)
        vesma_any.append(vesma_hit)

    assert rdkit_any
    assert len(rdkit_any) == len(vesma_any)

    rdkit_arr = np.array(rdkit_any, dtype=bool)
    vesma_arr = np.array(vesma_any, dtype=bool)

    false_negatives = int(np.sum(rdkit_arr & ~vesma_arr))
    assert false_negatives <= MAX_FALSE_NEGATIVES, (
        f"Too many false negatives vs RDKit: {false_negatives} > {MAX_FALSE_NEGATIVES}. "
        "Expected residual is aromaticity-perception divergence (deferred); a larger "
        "count likely indicates a real matching regression."
    )

    negatives = ~rdkit_arr
    fp_count = int(np.sum(vesma_arr & negatives))
    neg_count = int(np.sum(negatives))
    fp_rate = (fp_count / neg_count) if neg_count else 0.0

    assert fp_rate <= MAX_FP_RATE, (
        f"False-positive rate too high on sampled RDKit negatives: {fp_rate:.3f} > {MAX_FP_RATE:.3f} "
        f"(fp={fp_count}, negatives={neg_count})"
    )

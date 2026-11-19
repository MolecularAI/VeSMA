from __future__ import annotations

import tarfile
from functools import lru_cache
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vesma import SmartsError, cmol_from_smiles, gmol_from_cmol, gmol_from_smarts
from vesma.cpu_matcher import has_match

RNG_SEED = 20260226
PATTERN_SAMPLE_SIZE = 100
MOLECULE_POOL_SIZE = 5000
MOLECULE_SAMPLE_SIZE = 100

GZ_PATTERN_FILE_NAMES = [
    "brics-smarts.tar.gz",
    "daylight-smarts.tar.gz",
    "hann-smarts.tar.gz",
    "olah-smarts.tar.gz",
    "walters-smarts.tar.gz",
]


def _pattern(smarts: str):
    return gmol_from_smarts(smarts)[0]


def _molecule(smiles: str):
    return gmol_from_cmol(cmol_from_smiles(smiles)[0])


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


def _read_patterns_from_tar_gz(path: Path) -> list[str]:
    patterns: list[str] = []
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(".smarts"):
                continue
            payload = archive.extractfile(member)
            if payload is None:
                continue
            lines = [
                raw.decode("utf-8", errors="ignore")
                for raw in payload
                if raw.strip() and not raw.lstrip().startswith(b"#")
            ]
            if not lines:
                continue
            frame = pd.read_csv(
                StringIO("".join(lines)),
                sep=r"\s+",
                header=None,
                usecols=[0],
                names=["smarts"],
                dtype={"smarts": "string"},
                engine="python",
            )
            series = frame["smarts"].dropna().astype(str).str.strip()
            series = series[series != ""]
            patterns.extend(series.tolist())
    return patterns


@pytest.fixture
def smiles_pool(smarts_dataset_dir: Path) -> tuple[list[str], list[object]]:
    pytest.importorskip("rdkit.Chem")
    smiles_path = (
        smarts_dataset_dir
        / "processed"
        / "smarts-benchmark-v2.0"
        / "zinc_lead-like_2011-02-12_first100k.smi"
    )
    return _build_smiles_pool_cached(str(smiles_path))


@lru_cache(maxsize=1)
def _build_smiles_pool_cached(smiles_path: str) -> tuple[list[str], list[object]]:
    from rdkit import Chem

    smiles_all = _read_smiles(Path(smiles_path))
    assert len(smiles_all) == 100_000

    rng = np.random.default_rng(RNG_SEED)
    pool_idx = rng.choice(len(smiles_all), size=MOLECULE_POOL_SIZE, replace=False)
    pool_smiles = [smiles_all[int(i)] for i in pool_idx]
    pool_rd_mols = [Chem.MolFromSmiles(smi) for smi in pool_smiles]
    return pool_smiles, pool_rd_mols


@pytest.mark.slow
@pytest.mark.parametrize("pattern_file_name", GZ_PATTERN_FILE_NAMES)
def test_gz_smarts_sample_has_no_false_negatives_vs_rdkit(
    pattern_file_name: str,
    smarts_dataset_dir: Path,
    smiles_pool: tuple[list[str], list[object]],
) -> None:
    chem = pytest.importorskip("rdkit.Chem")
    pool_smiles, pool_rd_mols = smiles_pool
    pattern_file = smarts_dataset_dir / "raw" / pattern_file_name

    patterns = _read_patterns_from_tar_gz(pattern_file)
    assert patterns

    valid: list[tuple[str, object, object]] = []
    for smarts in patterns:
        rd_query = chem.MolFromSmarts(smarts)
        if rd_query is None:
            continue
        try:
            vesma_query = _pattern(smarts)
        except SmartsError:
            continue
        valid.append((smarts, rd_query, vesma_query))

    assert valid

    seed = RNG_SEED + sum(pattern_file.name.encode("utf-8"))
    rng = np.random.default_rng(seed)
    pattern_idx = rng.choice(
        len(valid), size=PATTERN_SAMPLE_SIZE, replace=len(valid) < PATTERN_SAMPLE_SIZE
    )
    sampled_patterns = [valid[int(i)] for i in pattern_idx]

    rd_any_pool = np.zeros(len(pool_smiles), dtype=bool)
    for _, rd_query, _ in sampled_patterns:
        for idx, rd_mol in enumerate(pool_rd_mols):
            if rd_any_pool[idx] or rd_mol is None:
                continue
            if rd_mol.HasSubstructMatch(rd_query):
                rd_any_pool[idx] = True

    positive_idx = np.flatnonzero(rd_any_pool)
    negative_idx = np.flatnonzero(~rd_any_pool)
    if positive_idx.size == 0:
        pytest.skip(f"No RDKit positives in molecule pool for {pattern_file.name}")

    if negative_idx.size == 0:
        sampled_idx = rng.choice(
            positive_idx,
            size=MOLECULE_SAMPLE_SIZE,
            replace=positive_idx.size < MOLECULE_SAMPLE_SIZE,
        )
    else:
        n_pos = min(MOLECULE_SAMPLE_SIZE // 2, int(positive_idx.size))
        n_neg = MOLECULE_SAMPLE_SIZE - n_pos
        sample_pos = rng.choice(positive_idx, size=n_pos, replace=False)
        sample_neg = rng.choice(
            negative_idx, size=n_neg, replace=len(negative_idx) < n_neg
        )
        sampled_idx = np.concatenate([sample_pos, sample_neg])
    rng.shuffle(sampled_idx)

    false_negatives = 0
    examples: list[dict[str, object]] = []

    for idx in sampled_idx:
        smi = pool_smiles[int(idx)]
        try:
            vesma_mol = _molecule(smi)
        except SmartsError:
            continue

        rd_hit = bool(rd_any_pool[int(idx)])
        vesma_hit = False
        for _, _, vesma_query in sampled_patterns:
            if has_match(vesma_query, vesma_mol):
                vesma_hit = True
                if rd_hit:
                    break

        if rd_hit and not vesma_hit:
            false_negatives += 1
            if len(examples) < 10:
                examples.append(
                    {
                        "smiles": smi,
                        "pattern_causing_fn": [
                            pat
                            for pat, rdq, vsq in sampled_patterns
                            if pool_rd_mols[int(idx)].HasSubstructMatch(rdq)
                            and not has_match(vsq, vesma_mol)
                        ],
                    }
                )

    detail = f"False negatives for {pattern_file.name}: {false_negatives}. Examples: {examples}"
    assert false_negatives == 0, detail

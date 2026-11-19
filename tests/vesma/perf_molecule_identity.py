from __future__ import annotations

import os
import time
import zipfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tests.datadirs import external_data_root
from vesma import (
    CMol,
    WlToken,
    cmol_canonicalish_smiles,
    cmol_from_mol,
    rdkit_canonical_smiles,
    rdkit_inchi,
    rdkit_inchi_key,
    short_ring_wl_molecule_id,
    short_ring_wl_molecule_ids,
)

DATASET_DIR = external_data_root() / "external" / "zbh-smarts" / "raw"
DEFAULT_SMILES_ZIP = DATASET_DIR / "smarts-benchmark-v2.0.zip"
DEFAULT_SMILES_MEMBER = "v2.0/zinc_lead-like_2011-02-12_first100k.smi"
DEFAULT_MOLECULE_LIMIT = 5_000
DEFAULT_REPEATS = 3


@dataclass(frozen=True)
class IdentityMethod:
    name: str
    encode: Callable[[CMol], Any]
    decodable: bool
    note: str
    encode_many: Callable[[Sequence[CMol]], Sequence[Any]] | None = None


@dataclass(frozen=True)
class StockIdentityMethod:
    name: str
    encode: Callable[[str], Any]
    decodable: bool
    note: str


@dataclass(frozen=True)
class IdentityTiming:
    method: str
    token_ms: float
    set_ms: float
    molecules_per_second: float
    average_token_bytes: float
    unique_tokens: int
    split_count: int
    clash_count: int


@dataclass(frozen=True)
class IdentityCorpus:
    smiles: tuple[str, ...]
    molecules: tuple[CMol, ...]


def identity_methods() -> tuple[IdentityMethod, ...]:
    return (
        IdentityMethod(
            name="rdkit-canonical-smiles",
            encode=rdkit_canonical_smiles,
            decodable=True,
            note="current RDKit-backed expansion-map codec",
        ),
        IdentityMethod(
            name="rdkit-inchi",
            encode=rdkit_inchi,
            decodable=True,
            note="portable, longer token; decodable through RDKit InChI",
        ),
        IdentityMethod(
            name="rdkit-inchikey",
            encode=rdkit_inchi_key,
            decodable=False,
            note="fixed-width hash; needs side table to decode",
        ),
        IdentityMethod(
            name="cmol-wl-ir-smiles",
            encode=cmol_canonicalish_smiles,
            decodable=True,
            note="in-repo WL refinement plus individualization/certificate search",
        ),
        IdentityMethod(
            name="short-ring-wl128",
            encode=short_ring_wl_molecule_id,
            encode_many=short_ring_wl_molecule_ids,
            decodable=False,
            note="Z/C/H/degree WL with flexible bonds in 3-8 membered rings",
        ),
    )


def stock_identity_methods() -> tuple[StockIdentityMethod, ...]:
    return (
        StockIdentityMethod(
            name="rdkit-canonical-smiles",
            encode=rdkit_canonical_smiles_from_smiles,
            decodable=True,
            note="RDKit SMILES parse, then RDKit canonical SMILES",
        ),
        StockIdentityMethod(
            name="rdkit-inchi",
            encode=rdkit_inchi_from_smiles,
            decodable=True,
            note="RDKit SMILES parse, then RDKit InChI",
        ),
        StockIdentityMethod(
            name="rdkit-inchikey",
            encode=rdkit_inchi_key_from_smiles,
            decodable=False,
            note="RDKit SMILES parse, then RDKit InChIKey",
        ),
        StockIdentityMethod(
            name="cmol-wl-ir-smiles",
            encode=cmol_wl_ir_smiles_from_smiles,
            decodable=True,
            note="RDKit SMILES parse to CMol, then in-repo WL/IR token",
        ),
        StockIdentityMethod(
            name="short-ring-wl128",
            encode=short_ring_wl_from_smiles,
            decodable=False,
            note="RDKit SMILES parse to CMol, then short-ring WL token",
        ),
    )


def rdkit_identity_mol_from_smiles(smiles: str) -> Any:
    from rdkit import Chem

    rd_mol = Chem.MolFromSmiles(smiles)
    if rd_mol is None:
        raise ValueError("Expected RDKit-parseable SMILES.")
    for atom in rd_mol.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.RemoveStereochemistry(rd_mol)
    return rd_mol


def rdkit_canonical_smiles_from_smiles(smiles: str) -> str:
    from rdkit import Chem

    return Chem.MolToSmiles(
        rdkit_identity_mol_from_smiles(smiles),
        canonical=True,
        isomericSmiles=False,
    )


def rdkit_inchi_from_smiles(smiles: str) -> str:
    from rdkit.Chem import rdinchi

    inchi, return_code, message, log, _ = rdinchi.MolToInchi(
        rdkit_identity_mol_from_smiles(smiles)
    )
    if int(return_code) not in {0, 1}:
        details = str(message or log or "unknown error")
        raise ValueError(f"RDKit InChI generation failed: {details}")
    return str(inchi)


def rdkit_inchi_key_from_smiles(smiles: str) -> str:
    from rdkit.Chem import rdinchi

    key = rdinchi.MolToInchiKey(rdkit_identity_mol_from_smiles(smiles))
    if not key:
        raise ValueError("RDKit InChIKey generation failed.")
    return str(key)


def cmol_wl_ir_smiles_from_smiles(smiles: str) -> str:
    return cmol_canonicalish_smiles(
        cmol_from_mol(rdkit_identity_mol_from_smiles(smiles))
    )


def short_ring_wl_from_smiles(smiles: str) -> WlToken:
    return short_ring_wl_molecule_id(
        cmol_from_mol(rdkit_identity_mol_from_smiles(smiles))
    )


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def iter_smiles() -> Sequence[str]:
    path = os.environ.get("VESMA_IDENTITY_BENCH_SMILES")
    if path is not None:
        return tuple(iter_smiles_file(Path(path)))
    return tuple(iter_smiles_zip(DEFAULT_SMILES_ZIP, DEFAULT_SMILES_MEMBER))


def iter_smiles_file(path: Path) -> Sequence[str]:
    with path.open("r", encoding="utf-8") as handle:
        return tuple(smiles_from_lines(handle))


def iter_smiles_zip(path: Path, member: str) -> Sequence[str]:
    with zipfile.ZipFile(path) as archive:
        payload = archive.read(member).decode("utf-8", errors="ignore")
    return tuple(smiles_from_lines(payload.splitlines()))


def smiles_from_lines(lines: Iterable[str]) -> list[str]:
    smiles: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        smiles.append(stripped.split()[0])
    return smiles


def load_molecules(
    smiles_values: Sequence[str],
    limit: int,
    Chem: Any,
) -> IdentityCorpus:
    loaded_smiles: list[str] = []
    molecules: list[CMol] = []
    for smiles in smiles_values:
        rd_mol = Chem.MolFromSmiles(smiles)
        if rd_mol is None:
            continue
        loaded_smiles.append(smiles)
        molecules.append(cmol_from_mol(rd_mol))
        if len(molecules) >= limit:
            return IdentityCorpus(tuple(loaded_smiles), tuple(molecules))
    return IdentityCorpus(tuple(loaded_smiles), tuple(molecules))


def time_encoded_values(
    encode: Callable[[Any], Any],
    values: Sequence[Any],
    repeats: int,
) -> tuple[float, tuple[Any, ...]]:
    best_ns: int | None = None
    best_tokens: tuple[Any, ...] = ()
    for _ in range(repeats):
        started = time.perf_counter_ns()
        tokens = tuple(encode(value) for value in values)
        elapsed_ns = time.perf_counter_ns() - started
        if best_ns is None or elapsed_ns < best_ns:
            best_ns = elapsed_ns
            best_tokens = tokens
    if best_ns is None:
        raise ValueError("Identity benchmark requires at least one repeat.")
    return best_ns / 1_000_000, best_tokens


def time_identity_method_values(
    method: IdentityMethod,
    values: Sequence[CMol],
    repeats: int,
) -> tuple[float, tuple[Any, ...]]:
    if method.encode_many is None:
        return time_encoded_values(method.encode, values, repeats)

    best_ns: int | None = None
    best_tokens: tuple[Any, ...] = ()
    for _ in range(repeats):
        started = time.perf_counter_ns()
        tokens = tuple(method.encode_many(values))
        elapsed_ns = time.perf_counter_ns() - started
        if best_ns is None or elapsed_ns < best_ns:
            best_ns = elapsed_ns
            best_tokens = tokens
    if best_ns is None:
        raise ValueError("Identity benchmark requires at least one repeat.")
    return best_ns / 1_000_000, best_tokens


def time_set_insert(tokens: Sequence[Any], repeats: int) -> float:
    best_ns: int | None = None
    for _ in range(repeats):
        started = time.perf_counter_ns()
        set(tokens)
        elapsed_ns = time.perf_counter_ns() - started
        if best_ns is None or elapsed_ns < best_ns:
            best_ns = elapsed_ns
    if best_ns is None:
        raise ValueError("Identity benchmark requires at least one repeat.")
    return best_ns / 1_000_000


def identity_group_deltas(
    reference_tokens: Sequence[Any],
    candidate_tokens: Sequence[Any],
) -> tuple[int, int]:
    reference_to_candidate: dict[Any, set[Any]] = {}
    candidate_to_reference: dict[Any, set[Any]] = {}
    for reference, candidate in zip(reference_tokens, candidate_tokens, strict=True):
        reference_to_candidate.setdefault(reference, set()).add(candidate)
        candidate_to_reference.setdefault(candidate, set()).add(reference)

    split_count = sum(len(values) - 1 for values in reference_to_candidate.values())
    clash_count = sum(len(values) - 1 for values in candidate_to_reference.values())
    return split_count, clash_count


def benchmark_method(
    method: IdentityMethod,
    molecules: Sequence[CMol],
    reference_tokens: Sequence[Any],
    repeats: int,
) -> tuple[IdentityTiming, tuple[Any, ...]]:
    token_ms, tokens = time_identity_method_values(method, molecules, repeats)
    set_ms = time_set_insert(tokens, repeats)
    split_count, clash_count = identity_group_deltas(reference_tokens, tokens)
    return (
        IdentityTiming(
            method=method.name,
            token_ms=token_ms,
            set_ms=set_ms,
            molecules_per_second=len(tokens) / (token_ms / 1_000),
            average_token_bytes=sum(token_bytes(token) for token in tokens)
            / len(tokens),
            unique_tokens=len(set(tokens)),
            split_count=split_count,
            clash_count=clash_count,
        ),
        tokens,
    )


def benchmark_stock_method(
    method: StockIdentityMethod,
    smiles_values: Sequence[str],
    reference_tokens: Sequence[Any],
    repeats: int,
) -> tuple[IdentityTiming, tuple[Any, ...]]:
    token_ms, tokens = time_encoded_values(method.encode, smiles_values, repeats)
    set_ms = time_set_insert(tokens, repeats)
    split_count, clash_count = identity_group_deltas(reference_tokens, tokens)
    return (
        IdentityTiming(
            method=method.name,
            token_ms=token_ms,
            set_ms=set_ms,
            molecules_per_second=len(tokens) / (token_ms / 1_000),
            average_token_bytes=sum(token_bytes(token) for token in tokens)
            / len(tokens),
            unique_tokens=len(set(tokens)),
            split_count=split_count,
            clash_count=clash_count,
        ),
        tokens,
    )


def print_timing_rows(
    title: str,
    methods: Sequence[IdentityMethod | StockIdentityMethod],
    timings: Sequence[IdentityTiming],
    reference_ms: float,
) -> None:
    print(title)
    for method, timing in zip(methods, timings, strict=True):
        print(
            {
                "method": timing.method,
                "token_ms": round(timing.token_ms, 3),
                "set_ms": round(timing.set_ms, 3),
                "molecules_per_second": round(timing.molecules_per_second),
                "ratio_vs_rdkit_smiles": round(timing.token_ms / reference_ms, 3),
                "average_token_bytes": round(timing.average_token_bytes, 1),
                "unique_tokens": timing.unique_tokens,
                "splits_vs_rdkit_smiles": timing.split_count,
                "clashes_vs_rdkit_smiles": timing.clash_count,
                "decodable": method.decodable,
                "note": method.note,
            }
        )


def token_bytes(token: Any) -> int:
    if isinstance(token, str):
        return len(token.encode("utf-8"))
    if isinstance(token, tuple) and all(isinstance(part, int) for part in token):
        return 8 * len(token)
    return len(repr(token).encode("utf-8"))


@pytest.mark.perf
def test_perf_molecule_identity_tokens() -> None:
    Chem = pytest.importorskip("rdkit.Chem")
    pytest.importorskip("rdkit.Chem.rdinchi")

    limit = env_int("VESMA_IDENTITY_BENCH_LIMIT", DEFAULT_MOLECULE_LIMIT)
    repeats = env_int("VESMA_IDENTITY_BENCH_REPEATS", DEFAULT_REPEATS)
    corpus = load_molecules(iter_smiles(), limit, Chem)
    if not corpus.molecules:
        raise ValueError("Identity benchmark loaded no molecules.")

    methods = identity_methods()
    reference_method = methods[0]
    reference_ms, reference_tokens = time_identity_method_values(
        reference_method,
        corpus.molecules,
        repeats,
    )
    baseline = IdentityTiming(
        method=reference_method.name,
        token_ms=reference_ms,
        set_ms=time_set_insert(reference_tokens, repeats),
        molecules_per_second=len(corpus.molecules) / (reference_ms / 1_000),
        average_token_bytes=sum(token_bytes(token) for token in reference_tokens)
        / len(reference_tokens),
        unique_tokens=len(set(reference_tokens)),
        split_count=0,
        clash_count=0,
    )

    timings = [baseline]
    for method in methods[1:]:
        timing, _ = benchmark_method(
            method,
            corpus.molecules,
            reference_tokens,
            repeats,
        )
        timings.append(timing)

    stock_methods = stock_identity_methods()
    stock_reference_method = stock_methods[0]
    stock_reference_ms, stock_reference_tokens = time_encoded_values(
        stock_reference_method.encode,
        corpus.smiles,
        repeats,
    )
    stock_baseline = IdentityTiming(
        method=stock_reference_method.name,
        token_ms=stock_reference_ms,
        set_ms=time_set_insert(stock_reference_tokens, repeats),
        molecules_per_second=len(corpus.smiles) / (stock_reference_ms / 1_000),
        average_token_bytes=sum(token_bytes(token) for token in stock_reference_tokens)
        / len(stock_reference_tokens),
        unique_tokens=len(set(stock_reference_tokens)),
        split_count=0,
        clash_count=0,
    )
    stock_timings = [stock_baseline]
    for method in stock_methods[1:]:
        timing, _ = benchmark_stock_method(
            method,
            corpus.smiles,
            stock_reference_tokens,
            repeats,
        )
        stock_timings.append(timing)

    print(
        {
            "molecules": len(corpus.molecules),
            "repeats": repeats,
            "dataset": os.environ.get(
                "VESMA_IDENTITY_BENCH_SMILES", str(DEFAULT_SMILES_ZIP)
            ),
        }
    )
    print_timing_rows("cmol_token_methods:", methods, timings, reference_ms)
    print_timing_rows(
        "stock_smiles_token_methods:",
        stock_methods,
        stock_timings,
        stock_reference_ms,
    )

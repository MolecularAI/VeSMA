"""Loaders for the committed test/benchmark datasets.

`tests/datadirs.py` resolves *where* the data lives; this module *loads* it. Two
datasets need typed loaders:

- the RDKit/GMol-compatible USPTO templates (`remix/uspto42k-compat/`), and
- the pinned retro-benchmark corpus (`remix/retro-benchmark/`).

Parsing and caching are delegated to `src` (`vesma.reaction.reaction_patterns_from_smarts`
caches parsed patterns on disk via joblib), so these loaders stay thin. Filtering is a
build-time concern baked into the committed files, never applied here — see the dataset
READMEs and `tests/README.md`.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import platform
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from tests.datadirs import external_data_root
from vesma import SmartsError, cmol_from_smiles, gmol_from_cmol, gmol_from_smarts
from vesma.gmol import GMol
from vesma.interop import cmol_from_mol
from vesma.loading import cached_function, parallel_map, resolve_cache
from vesma.reaction import ReactionTemplate, reaction_patterns_from_smarts

_TEMPLATE_COLUMNS = ("retro_template",)
MATCH_MOLECULE_CACHE_VERSION = 2


@cache
def chemistry_fingerprint() -> str:
    """A digest of the chemistry source, so nothing outlives the code that made it.

    ``vesma_version()`` is the packaged version and does not move between commits, so a
    cached molecule graph, or a resumed sweep checkpoint, can silently carry perception
    from chemistry that has since changed. Hashing the source is reproducible on another
    machine given the same checkout, which a git revision is not once anything is
    uncommitted.
    """
    import hashlib

    import vesma

    digest = hashlib.blake2b(digest_size=8)
    for path in sorted(Path(vesma.__file__).parent.rglob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


# --- compatible USPTO templates -------------------------------------------------


def compatible_templates_path() -> Path:
    return (
        external_data_root()
        / "external"
        / "remix"
        / "uspto42k-compat"
        / "processed"
        / "uspto42k-compat.csv.gz"
    )


CLASSIFIED_COLUMNS = (
    "retro_template",
    "template_hash",
    "library_occurence",
    "rc_code",
    "rc_agreement",
    "rc_verified_votes",
    "ri_name",
    "mapped_ri_name",
    "zen_code",
    "verdict",
    "verdict_reason",
    "label_source",
    "in_uspto42k",
    "in_used14k",
)


def raw68k_classified_path() -> Path:
    """Every raw68k template with its two reaction-class labels and an agreement verdict."""

    return (
        external_data_root()
        / "external"
        / "remix"
        / "raw68k-classified"
        / "processed"
        / "raw68k-classified.tsv.gz"
    )


def raw68k_verdict_path(verdict: str = "agree") -> Path:
    """One verdict's projection of the classified templates.

    Not committed: it is a pure function of `raw68k_classified_path()`, produced by
    `make raw68k-curated` in about a second. Still an explicit dataset rather than a runtime
    slice -- the filter runs at build time and writes a file, which is what
    tests/README.md's rule is about. Callers that need it should skip when it is absent and
    say which target builds it.

    The result is a gzipped TSV carrying `retro_template`, so it is directly loadable as a
    rule set by `bench_host_aostar.read_templates`.
    """

    return (
        external_data_root()
        / "external"
        / "remix"
        / "raw68k-classified"
        / "processed"
        / f"raw68k-{verdict}.tsv.gz"
    )


def read_classified_templates(path: Path) -> list[dict[str, str]]:
    """Rows of a classified-template TSV, with the header checked against the contract.

    Raises rather than returning a partial frame: a renamed or reordered column means the
    artifact and this loader have stopped describing the same thing, and every caller
    downstream would read the wrong field silently.
    """

    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        header = tuple(reader.fieldnames or ())

        if header != CLASSIFIED_COLUMNS:
            raise ValueError(
                f"{path.name} has columns {header}, expected {CLASSIFIED_COLUMNS}"
            )

        return list(reader)


def paroutes_n1_d2_f0_smiles_path() -> Path:
    return (
        external_data_root() / "external" / "paroutes" / "processed" / "n1-d2-f0.smi.gz"
    )


def read_paroutes_n1_d2_f0_smiles() -> tuple[str, ...]:
    return read_smiles_column(paroutes_n1_d2_f0_smiles_path())


def read_compatible_template_smarts() -> tuple[str, ...]:
    with gzip.open(
        compatible_templates_path(),
        "rt",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _TEMPLATE_COLUMNS:
            raise ValueError(
                f"Unexpected compatible template columns: {reader.fieldnames}"
            )
        return tuple(row["retro_template"] for row in reader)


def load_compatible_templates() -> tuple[ReactionTemplate, ...]:
    """Load every RDKit/GMol-compatible USPTO template as a ReactionTemplate.

    The dataset is pre-filtered at build time (compatibility + atom cap baked in), so
    every row is parseable; a parse failure here signals dataset corruption. Callers
    should treat this as the full compatible template set.
    """
    smarts = read_compatible_template_smarts()
    patterns = reaction_patterns_from_smarts(smarts, n_jobs=-1, cache=True)
    return tuple(
        ReactionTemplate(
            id=str(index),
            smarts=text,
            pattern=pattern,
            metadata={},
        )
        for index, (text, pattern) in enumerate(zip(smarts, patterns))
    )


# --- n1-d2 x uspto42k match pairs ----------------------------------------------


def paroutes_n1_d2_match_uspto42k_dir() -> Path:
    return external_data_root() / "external" / "remix" / "paroutes-n1-d2-match-uspto42k"


def paroutes_n1_d2_match_uspto42k_pairs_path() -> Path:
    return (
        paroutes_n1_d2_match_uspto42k_dir()
        / "processed"
        / "n1-d2-f0-uspto42k-compat-matches.json.gz"
    )


@dataclass(frozen=True)
class MatchMolecule:
    smiles: str
    rd_mol: Any
    cmol: Any
    gmol: GMol


@dataclass(frozen=True)
class PositivePairsArtifact:
    target_smiles_strings_hash: str
    patterns_smarts_strings_hash: str
    target_count: int
    pattern_count: int
    positive_pairs: np.ndarray


def load_paroutes_n1_d2_match_uspto42k_positive_pairs(
    molecule_indices: tuple[int, ...],
) -> list[tuple[int, int]]:
    artifact = load_paroutes_n1_d2_match_uspto42k_artifact()
    selected = np.asarray(molecule_indices, dtype=np.int64)
    if selected.size == 0:
        return []
    if np.any(selected < 0) or np.any(selected >= artifact.target_count):
        raise ValueError("Molecule selection contains an index outside the artifact.")
    if len(set(molecule_indices)) != len(molecule_indices):
        raise ValueError("Molecule selection must contain unique indices.")

    selected_pairs = artifact.positive_pairs[
        np.isin(artifact.positive_pairs[:, 0], selected)
    ]
    local_index = {
        molecule_index: local_index
        for local_index, molecule_index in enumerate(molecule_indices)
    }
    remapped = selected_pairs.copy()
    remapped[:, 0] = np.fromiter(
        (local_index[int(molecule_index)] for molecule_index in selected_pairs[:, 0]),
        dtype=np.int64,
        count=len(selected_pairs),
    )
    order = np.lexsort((remapped[:, 1], remapped[:, 0]))
    remapped = remapped[order]
    return [tuple(map(int, row)) for row in remapped.tolist()]


def load_paroutes_n1_d2_match_uspto42k_artifact() -> PositivePairsArtifact:
    target_smiles = read_paroutes_n1_d2_f0_smiles()
    pattern_smarts = read_compatible_template_smarts()
    payload = read_positive_pairs_json(paroutes_n1_d2_match_uspto42k_pairs_path())
    validate_positive_pairs_payload(
        payload,
        target_smiles=target_smiles,
        pattern_smarts=pattern_smarts,
    )
    pairs = positive_pairs_array(payload["positive_pairs"])
    return PositivePairsArtifact(
        target_smiles_strings_hash=payload["target_smiles_strings_hash"],
        patterns_smarts_strings_hash=payload["patterns_smarts_strings_hash"],
        target_count=int(payload["target_count"]),
        pattern_count=int(payload["pattern_count"]),
        positive_pairs=pairs,
    )


def read_positive_pairs_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Required positive-pairs file is missing: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("Positive-pairs payload must be a JSON object.")
    return payload


def validate_positive_pairs_payload(
    payload: dict[str, Any],
    *,
    target_smiles: tuple[str, ...],
    pattern_smarts: tuple[str, ...],
) -> None:
    required = {
        "schema_version",
        "hash_algorithm",
        "hash_encoding",
        "target_smiles_strings_hash",
        "patterns_smarts_strings_hash",
        "target_count",
        "pattern_count",
        "positive_pairs",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Positive-pairs payload is missing keys: {missing}")
    if payload["schema_version"] != 1:
        raise ValueError("Unsupported positive-pairs schema version.")
    if payload["hash_algorithm"] != "sha256":
        raise ValueError("Unsupported positive-pairs hash algorithm.")
    if payload["hash_encoding"] != "json-array-utf8-compact":
        raise ValueError("Unsupported positive-pairs hash encoding.")
    if int(payload["target_count"]) != len(target_smiles):
        raise ValueError("Positive-pairs target count does not match source SMILES.")
    if int(payload["pattern_count"]) != len(pattern_smarts):
        raise ValueError("Positive-pairs pattern count does not match source SMARTS.")
    if payload["target_smiles_strings_hash"] != stable_string_tuple_hash(target_smiles):
        raise ValueError("Positive-pairs target SMILES hash does not match source.")
    if payload["patterns_smarts_strings_hash"] != stable_string_tuple_hash(
        pattern_smarts
    ):
        raise ValueError("Positive-pairs SMARTS hash does not match source.")

    pairs = positive_pairs_array(payload["positive_pairs"])
    if pairs.size == 0:
        return
    if np.any(pairs[:, 0] < 0) or np.any(pairs[:, 0] >= len(target_smiles)):
        raise ValueError("Positive-pairs molecule index outside source range.")
    if np.any(pairs[:, 1] < 0) or np.any(pairs[:, 1] >= len(pattern_smarts)):
        raise ValueError("Positive-pairs pattern index outside source range.")
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    if not np.array_equal(order, np.arange(len(pairs))):
        raise ValueError("Positive pairs must be sorted by molecule_idx, pattern_idx.")


def stable_string_tuple_hash(strings: tuple[str, ...]) -> str:
    payload = json.dumps(
        strings,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def positive_pairs_array(rows: Any) -> np.ndarray:
    pairs = np.asarray(rows)
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("Positive pairs must be a 2-column array.")
    if not np.issubdtype(pairs.dtype, np.integer):
        raise TypeError("Positive pairs must contain integer indices.")
    return pairs.astype(np.int64, copy=False)


def read_smiles_column(path: Path) -> tuple[str, ...]:
    rows: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip().split()[0]
            if value:
                rows.append(value)
    return tuple(rows)


def load_match_molecules(
    smiles: tuple[str, ...],
    *,
    n_jobs: int | None = -1,
    cache: bool | None = True,
) -> tuple[MatchMolecule, ...]:
    rows = tuple(smiles)
    if not resolve_cache(cache):
        return parse_match_molecule_items(rows, n_jobs=n_jobs)

    cached = cached_function(parse_match_molecule_items, ignore=("n_jobs",))
    return cached(
        rows,
        MATCH_MOLECULE_CACHE_VERSION,
        platform.python_version(),
        n_jobs,
        chemistry_fingerprint(),
    )


def parse_match_molecule_items(
    smiles: tuple[str, ...],
    cache_version: int = MATCH_MOLECULE_CACHE_VERSION,
    python_version: str = "",
    n_jobs: int | None = -1,
    fingerprint: str = "",
) -> tuple[MatchMolecule, ...]:
    del python_version, fingerprint
    if cache_version != MATCH_MOLECULE_CACHE_VERSION:
        raise ValueError("Unsupported match molecule cache version.")
    return parallel_map(parse_match_molecule, smiles, n_jobs=n_jobs)


def parse_match_molecule(smiles: str) -> MatchMolecule:
    rd_mol = Chem.MolFromSmiles(smiles)
    if rd_mol is None:
        raise ValueError(f"Invalid benchmark SMILES: {smiles}")
    cmol = cmol_from_mol(rd_mol)
    return MatchMolecule(
        smiles=smiles,
        rd_mol=rd_mol,
        cmol=cmol,
        gmol=gmol_from_cmol(cmol),
    )


# --- retro-benchmark corpus -----------------------------------------------------


DEFAULT_RETRO_BENCHMARK_DIR = (
    external_data_root() / "external" / "remix" / "retro-benchmark"
)
_PATTERN_COLUMNS = (
    "template_code",
    "template_hash",
    "classification",
    "library_occurence",
    "retro_template",
    "lhs_smarts",
    "lhs_atom_count",
    "hit_class",
)
_MOLECULE_COLUMNS = ("chembl_id", "smiles", "atom_count")


@dataclass(frozen=True)
class RetroBenchmarkPattern:
    template_code: str
    template_hash: str
    classification: str
    library_occurence: int
    retro_template: str
    lhs_smarts: str
    lhs_atom_count: int
    hit_class: str
    query: GMol


@dataclass(frozen=True)
class RetroBenchmarkMolecule:
    chembl_id: str
    smiles: str
    atom_count: int
    molecule: GMol


@dataclass(frozen=True)
class RetroBenchmarkCorpus:
    patterns: tuple[RetroBenchmarkPattern, ...]
    molecules: tuple[RetroBenchmarkMolecule, ...]


def load_retro_benchmark_corpus(
    dataset_dir: Path = DEFAULT_RETRO_BENCHMARK_DIR,
) -> RetroBenchmarkCorpus:
    """Load the full pinned corpus. Callers slice (`corpus.molecules[:n]`) for subsets."""
    return RetroBenchmarkCorpus(
        patterns=_load_patterns(dataset_dir / "patterns.tsv"),
        molecules=_load_molecules(dataset_dir / "molecules.tsv"),
    )


def _load_patterns(path: Path) -> tuple[RetroBenchmarkPattern, ...]:
    rows = _read_tsv_rows(path, _PATTERN_COLUMNS)
    patterns: list[RetroBenchmarkPattern] = []
    for row in rows:
        lhs_smarts = row["lhs_smarts"]
        try:
            query = gmol_from_smarts(lhs_smarts)[0]
        except SmartsError as exc:
            raise ValueError(f"{path} contains an invalid lhs_smarts value.") from exc

        lhs_atom_count = int(row["lhs_atom_count"])
        if query.atom_count != lhs_atom_count:
            raise ValueError(f"{path} lhs_atom_count does not match parsed query size.")

        hit_class = row["hit_class"]
        if hit_class not in {"positive", "negative"}:
            raise ValueError(f"{path} contains an unsupported hit_class value.")

        patterns.append(
            RetroBenchmarkPattern(
                template_code=row["template_code"],
                template_hash=row["template_hash"],
                classification=row["classification"],
                library_occurence=int(row["library_occurence"]),
                retro_template=row["retro_template"],
                lhs_smarts=lhs_smarts,
                lhs_atom_count=lhs_atom_count,
                hit_class=hit_class,
                query=query,
            )
        )
    return tuple(patterns)


def _load_molecules(path: Path) -> tuple[RetroBenchmarkMolecule, ...]:
    rows = _read_tsv_rows(path, _MOLECULE_COLUMNS)
    molecules: list[RetroBenchmarkMolecule] = []
    for row in rows:
        smiles = row["smiles"]
        try:
            molecule = gmol_from_cmol(cmol_from_smiles(smiles)[0])
        except SmartsError as exc:
            raise ValueError(f"{path} contains an invalid smiles value.") from exc

        atom_count = int(row["atom_count"])
        if molecule.atom_count != atom_count:
            raise ValueError(f"{path} atom_count does not match parsed molecule size.")

        molecules.append(
            RetroBenchmarkMolecule(
                chembl_id=row["chembl_id"],
                smiles=smiles,
                atom_count=atom_count,
                molecule=molecule,
            )
        )
    return tuple(molecules)


def _read_tsv_rows(
    path: Path, expected_columns: tuple[str, ...]
) -> list[dict[str, str]]:
    if not path.exists():
        raise ValueError(f"Required benchmark file is missing: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = tuple(reader.fieldnames or ())
        if fieldnames != expected_columns:
            raise ValueError(f"{path} has unexpected columns: {fieldnames!r}")
        return [dict(row) for row in reader]

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tests.vertsy import paper_provenance as prov
from tests.vertsy.paroutes_reference import (
    reference_route_root,
    reference_template_set,
)
from vesma.reaction import ReactionTemplate


@dataclass(frozen=True)
class SmallPaRoutesCorpus:
    directory: Path
    route_set: str
    reference_routes: list[dict[str, Any]]
    reference_expansions: dict[str, frozenset]
    target_smiles: tuple[str, ...]
    stock_smiles: frozenset[str]


def pytest_benchmark_update_machine_info(
    config: Any, machine_info: dict[str, Any]
) -> None:
    machine_info["cpu_detail"] = prov.cpu_provenance()
    machine_info["cuda"] = prov.cuda_provenance()
    machine_info["metal"] = prov.metal_provenance(
        probe_backend=os.environ.get("BENCH_PROBE_METAL") == "1"
    )
    machine_info["software"] = prov.software_provenance()
    machine_info["host_type"] = (
        os.environ.get("BENCH_HOST_LABEL", "") or prov.host_slug()
    )


def pytest_benchmark_update_commit_info(
    config: Any, commit_info: dict[str, Any]
) -> None:
    commit_info["dirty"] = not prov.repo_is_clean(paths=("src", "tests"))
    commit_info["clean_scope"] = "src,tests"
    commit_info["source_trees"] = prov.git_provenance()["source_trees"]


def merge_expansions(
    left: dict[str, frozenset],
    right: dict[str, frozenset],
) -> dict[str, frozenset]:
    merged = dict(left)
    for key, value in right.items():
        merged[key] = merged.get(key, frozenset()) | value
    return merged


def extract_expansions(node: dict[str, Any]) -> dict[str, frozenset]:
    node_type = node.get("type")
    if node_type == "mol" or (node_type is None and "smiles" in node):
        molecule = node["smiles"]
        expansions = {molecule: frozenset()}
        for child in node.get("children", []):
            reaction_id = child["metadata"]["ID"]
            precursors = frozenset(c["smiles"] for c in child.get("children", []))
            expansions[molecule] |= frozenset([(reaction_id, precursors)])
            expansions = merge_expansions(expansions, extract_expansions(child))
        return expansions
    if node_type == "reaction":
        expansions: dict[str, frozenset] = {}
        for child in node.get("children", []):
            expansions = merge_expansions(expansions, extract_expansions(child))
        return expansions
    return {}


@pytest.fixture
def small_paroutes_corpus(readonly_datadir: Path) -> SmallPaRoutesCorpus:
    return load_small_paroutes_corpus(
        readonly_datadir,
        route_set="n1",
        subset_dir="subset-retro-benchmark-small-n1-1",
    )


@pytest.fixture
def deep_paroutes_corpus(readonly_datadir: Path) -> SmallPaRoutesCorpus:
    return load_small_paroutes_corpus(
        readonly_datadir,
        route_set="n5",
        subset_dir="subset-retro-benchmark-small-n5-depth3-4-1",
    )


def load_small_paroutes_corpus(
    readonly_datadir: Path,
    *,
    route_set: str,
    subset_dir: str,
) -> SmallPaRoutesCorpus:
    directory = readonly_datadir / f"external/paroutes/processed/{subset_dir}"
    if not directory.exists():
        pytest.skip(
            f"Small corpus not found. Run extract_small_retro_corpus.py for {route_set} first."
        )

    with gzip.open(
        directory / f"{route_set}-routes.json.gz", "rt", encoding="utf-8"
    ) as handle:
        reference_routes = json.load(handle)

    expansions: dict[str, frozenset] = {}
    targets: list[str] = []
    for row in reference_routes:
        smiles = row["smiles"]
        targets.append(smiles)
        node = {"type": "mol", "smiles": smiles, "children": row.get("children", [])}
        expansions = merge_expansions(expansions, extract_expansions(node))

    with gzip.open(
        directory / f"{route_set}-stock.smi.gz", "rt", encoding="utf-8"
    ) as handle:
        stock = frozenset(line.strip().split()[0] for line in handle if line.strip())

    return SmallPaRoutesCorpus(
        directory=directory,
        route_set=route_set,
        reference_routes=reference_routes,
        reference_expansions=expansions,
        target_smiles=tuple(targets),
        stock_smiles=stock,
    )


@pytest.fixture
def small_paroutes_templates(
    small_paroutes_corpus: SmallPaRoutesCorpus,
) -> tuple[ReactionTemplate, ...]:
    routes = tuple(
        reference_route_root(row) for row in small_paroutes_corpus.reference_routes
    )
    return reference_template_set(
        small_paroutes_corpus.directory,
        routes,
    ).templates

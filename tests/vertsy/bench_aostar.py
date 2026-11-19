"""Heuristic AO* search over PaRoutes n1, against the RDKit backend.

Reports how much of the graph AO* has to build to reach a route, which is the only claim a
heuristic search makes: fewer molecules expanded than exhaustive search needs, for the same
route. Timings are incidental — the interesting number is `molecules_expanded`, because it
is a property of the search rather than of the host it ran on.

Both arms use identical chemistry (same templates, same stock, same keyer) and differ only
in the heuristic, so the comparison isolates the thing under test:

- `zero`  — no estimate. Admissible, so the route is the cheapest within the depth budget.
- `stock` — Jaccard distance to the nearest stock molecule, weighted. At weight <= 1 this is
  still admissible; above it, weighted A*: usually fewer expansions, no optimality claim.

Configure with `BENCH_AOSTAR_TARGETS`, `BENCH_AOSTAR_HEURISTICS`, `BENCH_AOSTAR_WEIGHT`,
`BENCH_AOSTAR_WIDTH`, `BENCH_AOSTAR_PASSES`. Run it with `make bench-aostar`.
"""

from __future__ import annotations

import csv
import gzip
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("rdkit.Chem")

from rdkit import Chem

from tests.datadirs import external_data_root
from vertsy.aostar import aostar, constant_step
from vertsy.expansions import Expansions
from vertsy.heuristics import zero_heuristic
from vertsy.interner import DictInterner
from vertsy.routes import reaction_count
from vertsy.search import concat_object_batches, expansion, object_mol_array
from vertsy.similarity import stock_similarity
from vertsy.store.numpy_store import NumpyExpansionStore
from vesma.keyer import Keyer, rdkit_mol_canonical_smiles, rdkit_smiles_keyer
from vesma.matcher import all_vs_all, keyed_match
from vesma.rdkit_matcher import (
    RdkitFilterCatalogReactionMatcher,
    rdkit_reactions_from_smarts,
)

pytestmark = [pytest.mark.perf, pytest.mark.external_data]

PAROUTES_DIR = external_data_root() / "external" / "paroutes"
STOCK_FILE = PAROUTES_DIR / "processed" / "n1-stock.smi.gz"
TARGETS_FILE = PAROUTES_DIR / "processed" / "n1-d2-f0.smi.gz"
TEMPLATES_FILE = PAROUTES_DIR / "processed" / "used14k-reactions.tsv.gz"


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, "") or default)


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, "") or default)


def read_smiles(path: Path, limit: int | None = None) -> tuple[str, ...]:
    if not path.exists():
        pytest.skip(f"Missing benchmark data file: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = (line.strip().split()[0] for line in handle if line.strip())
        return tuple(row for _, row in zip(range(limit or 10**9), rows))


@lru_cache
def load_stock_smiles() -> tuple[str, ...]:
    return read_smiles(STOCK_FILE)


@lru_cache
def stock_keys(codec: Keyer[Chem.Mol, str]) -> frozenset[str]:
    """Stock in the key space precursors are generated in.

    Not the raw file SMILES: a precursor is keyed stereo-free, so raw stock hides every
    stereochemical entry from the search — 9.5% of n1 — and routes through them never solve.
    """
    return frozenset(codec.from_smiles(smiles) for smiles in load_stock_smiles())


@lru_cache
def load_templates() -> tuple[str, ...]:
    if not TEMPLATES_FILE.exists():
        pytest.skip(f"Missing benchmark template file: {TEMPLATES_FILE}")
    with gzip.open(TEMPLATES_FILE, "rt", encoding="utf-8") as handle:
        return tuple(
            row["retro_template"] for row in csv.DictReader(handle, delimiter="\t")
        )


@lru_cache
def build_match() -> Any:
    templates = load_templates()
    matcher = RdkitFilterCatalogReactionMatcher(
        rdkit_reactions_from_smarts(templates),
        [smarts.split(">>")[0] for smarts in templates],
    )
    return expansion(
        keyed_match(
            all_vs_all(matcher),
            lambda mols: [rdkit_mol_canonical_smiles(mol) for mol in mols],
        )
    )


@lru_cache
def build_heuristic(kind: str):
    if kind == "zero":
        return zero_heuristic
    if kind == "stock":
        return stock_similarity(
            load_stock_smiles(), weight=env_float("BENCH_AOSTAR_WEIGHT", 1.0)
        )
    raise ValueError(f"Unknown heuristic {kind!r}; expected 'zero' or 'stock'.")


def heuristics() -> list[str]:
    configured = os.environ.get("BENCH_AOSTAR_HEURISTICS", "") or "zero,stock"
    return [kind.strip() for kind in configured.split(",") if kind.strip()]


@pytest.fixture(params=heuristics())
def heuristic_kind(request: Any) -> str:
    return str(request.param)


def test_bench_aostar_n1(benchmark: Any, heuristic_kind: str) -> None:
    stock = stock_keys(rdkit_smiles_keyer)
    match = build_match()
    heuristic = build_heuristic(heuristic_kind)
    targets = read_smiles(TARGETS_FILE, env_int("BENCH_AOSTAR_TARGETS", 5))
    width = env_int("BENCH_AOSTAR_WIDTH", 32)
    passes = env_int("BENCH_AOSTAR_PASSES", 16)
    # Both weights stated, at the ratio `aostar` recommends: a reaction costs about a tenth of what
    # the whole fingerprint range is worth, which is what the depth-5 sweep settled on.
    price = constant_step(env_float("BENCH_AOSTAR_STEP", 0.105))

    def run() -> list[dict[str, Any]]:
        records = []
        for smiles in targets:
            expansions = Expansions(NumpyExpansionStore(), DictInterner())
            routes = aostar(
                rdkit_smiles_keyer.from_smiles(smiles),
                target_mol=Chem.MolFromSmiles(smiles),
                stock=stock,
                expansions=expansions,
                heuristic=heuristic,
                match=match,
                stack_mols=object_mol_array,
                concat_mols=concat_object_batches,
                price=price,
                width=width,
                passes=passes,
            )
            records.append(
                {
                    "target": smiles,
                    "solved": bool(routes),
                    "routes": len(routes),
                    "best_reaction_count": (
                        min(reaction_count(route) for route in routes)
                        if routes
                        else None
                    ),
                    # What the heuristic is actually judged on: how much graph it had to
                    # build. Wall clock varies with the host; this does not.
                    "molecules_expanded": int(expansions.store.expanded().size),
                    "molecules_seen": int(expansions.sid_count()),
                }
            )
        return records

    records = benchmark.pedantic(run, rounds=env_int("BENCH_ROUNDS", 1), iterations=1)

    solved = [record for record in records if record["solved"]]
    benchmark.extra_info.update(
        {
            "heuristic": heuristic_kind,
            "weight": env_float("BENCH_AOSTAR_WEIGHT", 1.0),
            "width": width,
            "passes": passes,
            "expansion_budget": width * passes,
            "stock_size": len(stock),
            "template_count": len(load_templates()),
            "targets": len(targets),
            "solved_targets": len(solved),
            "expanded_per_target_mean": (
                float(np.mean([r["molecules_expanded"] for r in records]))
                if records
                else 0.0
            ),
            "expanded_per_solved_mean": (
                float(np.mean([r["molecules_expanded"] for r in solved]))
                if solved
                else None
            ),
            "records": records,
        }
    )

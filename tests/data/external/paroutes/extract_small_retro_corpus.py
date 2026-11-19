from __future__ import annotations

import argparse
import gzip
import io
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

from vertsy import cmol_rdkit_smiles_keyer
from vertsy.expansions import Expansions
from vertsy.interner import DictInterner
from vertsy.interop import route_from_aizynth_dict
from vertsy.replay import replay_search
from vertsy.routes import (
    Route,
    encode_route,
    iter_solved_routes,
    leaves,
    reaction_depth,
)
from vertsy.search import object_mol_array
from vertsy.store.numpy_store import NumpyExpansionStore
from vesma import SmartsError, cmol_from_smiles
from vesma.cpu_matcher import CpuReactionMatcher
from vesma.keyer import rdkit_mol_canonical_smiles
from vesma.rdkit_matcher import (
    RdkitReactionMatcher,
    rdkit_reactions_from_smarts,
)
from vesma.reaction import ReactionPattern, parse_reaction_smarts

RAW_TEMPLATE_LIBRARY = "uspto_raw_template_library.csv.gz"
RAW_TEMPLATE_LIBRARY_SEPARATOR = "\t"
DEFAULT_ROUTE_SET = "n1"
DEFAULT_SUBSET_DIR = "subset-retro-benchmark-small-n1-1"
DEFAULT_MIN_DEPTH = 2
DEFAULT_MAX_DEPTH = 2
DEFAULT_LIMIT = 10


@contextmanager
def deterministic_gzip_text(path: Path) -> Iterator[TextIO]:
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            mtime=0,
        ) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8") as text:
                yield text


def depth(node: dict[str, Any]) -> int:
    if node["type"] == "reaction":
        return 1 + max((depth(c) for c in node.get("children", [])), default=0)
    if node["type"] == "mol" and "children" in node:
        return max((depth(c) for c in node.get("children", [])), default=0)
    return 0


def startmat(node: dict[str, Any]) -> list[str]:
    if node["type"] == "mol" and "children" not in node:
        return [node["smiles"]]
    return sum([startmat(c) for c in node.get("children", [])], [])


def reaction_hashes(node: dict[str, Any]) -> list[str]:
    if node["type"] == "reaction":
        return [node["metadata"]["reaction_hash"]] + sum(
            [reaction_hashes(c) for c in node.get("children", [])], []
        )
    if node["type"] == "mol" and "children" in node:
        return sum([reaction_hashes(c) for c in node.get("children", [])], [])
    return []


def main() -> None:
    args = parse_args()
    paroutes_dir = Path(__file__).resolve().parent
    processed_dir = paroutes_dir / "processed"
    out_dir = processed_dir / args.subset_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    routes_path = processed_dir / f"{args.route_set}-routes.json.gz"
    print(f"Loading {routes_path.name}...")
    with gzip.open(routes_path, "rt", encoding="utf-8") as handle:
        routes_data = json.load(handle)

    print(f"Loading {RAW_TEMPLATE_LIBRARY}...")
    raw_library_path = paroutes_dir / "raw" / RAW_TEMPLATE_LIBRARY
    if not raw_library_path.exists():
        raise FileNotFoundError(
            f"{raw_library_path} is missing. Run prepare_paroutes.py with "
            "--only uspto-raw-template-library first."
        )
    lib_df = pd.read_csv(raw_library_path, sep=RAW_TEMPLATE_LIBRARY_SEPARATOR)

    library_rows = {str(row["reaction_hash"]): row for _, row in lib_df.iterrows()}
    print(f"Loaded {len(library_rows)} reaction templates.")

    selected_routes = select_routes(
        routes_data,
        library_rows,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        limit=args.limit,
        require_route_replay=args.require_route_replay,
    )

    if len(selected_routes) < args.limit:
        raise RuntimeError(f"Only found {len(selected_routes)} suitable routes.")

    print(f"Found {len(selected_routes)} suitable routes.")

    required_stock: set[str] = set()
    required_targets: set[str] = set()
    used_reaction_hashes: set[str] = set()

    for r in selected_routes:
        required_targets.add(r["smiles"])
        for sm in startmat(r):
            required_stock.add(sm)
        used_reaction_hashes.update(reaction_hashes(r))

    print(f"Required stock: {len(required_stock)}, targets: {len(required_targets)}")

    mask = lib_df["reaction_hash"].isin(used_reaction_hashes)
    subset_lib_df = lib_df[mask]
    print(f"Extracted {len(subset_lib_df)} templates for the selected routes.")

    with deterministic_gzip_text(out_dir / f"{args.route_set}-routes.json.gz") as f:
        routes_by_target: dict[str, list[dict[str, Any]]] = {}
        for r in selected_routes:
            smiles = r["smiles"]
            if smiles not in routes_by_target:
                routes_by_target[smiles] = []
            routes_by_target[smiles].extend(r["children"])

        output_routes: list[dict[str, Any]] = []
        for smiles, children in routes_by_target.items():
            output_routes.append({"smiles": smiles, "children": children})

        json.dump(output_routes, f)

    with deterministic_gzip_text(out_dir / f"{args.route_set}-targets.smi.gz") as f:
        for t in sorted(required_targets):
            f.write(t + "\n")

    with deterministic_gzip_text(out_dir / f"{args.route_set}-stock.smi.gz") as f:
        for s in sorted(required_stock):
            f.write(s + "\n")

    with deterministic_gzip_text(out_dir / RAW_TEMPLATE_LIBRARY) as f:
        subset_lib_df.to_csv(f, sep=RAW_TEMPLATE_LIBRARY_SEPARATOR, index=False)

    print("Done!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-set", default=DEFAULT_ROUTE_SET)
    parser.add_argument("--subset-dir", default=DEFAULT_SUBSET_DIR)
    parser.add_argument("--min-depth", type=int, default=DEFAULT_MIN_DEPTH)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--require-route-replay", action="store_true")
    return parser.parse_args()


def select_routes(
    routes_data: list[dict[str, Any]],
    library_rows: dict[str, pd.Series],
    *,
    min_depth: int,
    max_depth: int,
    limit: int,
    require_route_replay: bool,
) -> list[dict[str, Any]]:
    selected_routes: list[dict[str, Any]] = []
    for row in routes_data:
        target_smiles = row["smiles"]
        for child in row.get("children", []):
            route_node = {"type": "mol", "smiles": target_smiles, "children": [child]}
            route_depth = depth(route_node)
            if route_depth < min_depth or route_depth > max_depth:
                continue
            route_reaction_hashes = reaction_hashes(route_node)
            if not all(h in library_rows for h in route_reaction_hashes):
                continue
            if not all(
                reaction_template_parses(library_rows[h]["retro_template"])
                for h in route_reaction_hashes
            ):
                continue
            if require_route_replay and not route_replays_reference(
                route_node,
                library_rows,
            ):
                continue
            selected_routes.append(route_node)
            if len(selected_routes) == limit:
                return selected_routes
    return selected_routes


def reaction_template_parses(smarts: str) -> bool:
    try:
        parse_reaction_smarts(clean_reaction_smarts(smarts))
    except SmartsError:
        return False
    return True


def route_replays_reference(
    route: dict[str, Any],
    library_rows: dict[str, pd.Series],
) -> bool:
    try:
        smiles_route, smarts, patterns = replay_inputs(route, library_rows)
        expected = encode_route(smiles_route, cmol_rdkit_smiles_keyer)
        from rdkit import Chem

        target = Chem.MolFromSmiles(route["smiles"])
        if target is None:
            return False
        rdkit_replayed = rdkit_replay_solved_routes(
            expected,
            target,
            RdkitReactionMatcher(rdkit_reactions_from_smarts(smarts)),
        )
        if not route_matches_reference(rdkit_replayed, expected):
            return False
        vesma_replayed = vesma_replay_solved_routes(
            expected,
            cmol_from_smiles(route["smiles"])[0],
            CpuReactionMatcher(patterns),
        )
        return route_matches_reference(vesma_replayed, expected)
    except SmartsError, ValueError:
        return False


def rdkit_replay_solved_routes(
    mapped_route: Route,
    target: Any,
    matcher: RdkitReactionMatcher,
) -> frozenset[Route]:
    target_key = rdkit_mol_canonical_smiles(target)
    stock = leaves(mapped_route)

    def match_fn(mols, pairs):
        precursors, hits = matcher.match(mols, pairs)
        return (
            object_mol_array(precursors),
            [rdkit_mol_canonical_smiles(mol) for mol in precursors],
            hits,
        )

    expansions = replay_search(
        mapped_route,
        [target_key],
        object_mol_array([target]),
        stock,
        reaction_depth(mapped_route),
        Expansions(NumpyExpansionStore(), DictInterner()),
        match=match_fn,
    )
    return frozenset(iter_solved_routes(target_key, expansions, stock))


def vesma_replay_solved_routes(
    mapped_route: Route,
    target: Any,
    matcher: CpuReactionMatcher,
) -> frozenset[Route]:
    target_key = cmol_rdkit_smiles_keyer.encode(target)
    stock = leaves(mapped_route)

    def match_fn(mols, pairs):
        precursors, hits = matcher.match(mols, pairs)
        return (
            object_mol_array(precursors),
            [cmol_rdkit_smiles_keyer.encode(mol) for mol in precursors],
            hits,
        )

    expansions = replay_search(
        mapped_route,
        [target_key],
        object_mol_array([target]),
        stock,
        reaction_depth(mapped_route),
        Expansions(NumpyExpansionStore(), DictInterner()),
        match=match_fn,
    )
    return frozenset(iter_solved_routes(target_key, expansions, stock))


def replay_inputs(
    route: dict[str, Any],
    library_rows: dict[str, pd.Series],
) -> tuple[Route, tuple[str, ...], tuple[ReactionPattern, ...]]:
    reaction_hashes_seen: dict[str, int] = {}
    smarts: list[str] = []
    patterns: list[ReactionPattern] = []
    for reaction_hash in reaction_hashes(route):
        if reaction_hash in reaction_hashes_seen:
            continue
        rule_id = len(smarts)
        reaction_hashes_seen[reaction_hash] = rule_id
        cleaned = clean_reaction_smarts(library_rows[reaction_hash]["retro_template"])
        smarts.append(cleaned)
        patterns.append(parse_reaction_smarts(cleaned))
    smiles_route = route_from_aizynth_dict(
        route,
        lambda metadata: reaction_hashes_seen[str(metadata["reaction_hash"])],
    )
    return smiles_route, tuple(smarts), tuple(patterns)


def route_matches_reference(candidates: frozenset[Route], expected: Route) -> bool:
    return expected in candidates


def clean_reaction_smarts(smarts: str) -> str:
    lhs, rhs = smarts.split(">>")
    clean_lhs = ".".join(
        clean_smarts_component(component) for component in lhs.split(".")
    )
    clean_rhs = ".".join(
        clean_smarts_component(component) for component in rhs.split(".")
    )
    return f"{clean_lhs}>>{clean_rhs}"


def clean_smarts_component(component: str) -> str:
    if not (component.startswith("(") and component.endswith(")")):
        return component

    depth = 0
    for idx, char in enumerate(component):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if depth == 0 and idx < len(component) - 1:
            return component
    return component[1:-1]


if __name__ == "__main__":
    main()

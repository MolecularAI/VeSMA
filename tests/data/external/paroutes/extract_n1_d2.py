"""Extract PaRoutes targets whose reference route has a given depth.

Each n1 route row is one target with its reference route tree; a target is selected when that
route's depth (longest reaction chain) equals ``--depth``. Defaults to depth 2 (the original
n1-d2 set) so existing callers are unchanged; pass ``--depth 3`` for the n1-d3 set (targets that
genuinely need three steps, so a depth-3 expansion is productive rather than over-expansion).
"""

import argparse
import gzip
import json


def depth(node):
    if node["type"] == "reaction":  # reaction
        return 1 + max((depth(c) for c in node.get("children", [])), default=0)
    elif node["type"] == "mol" and "children" in node:  # intermediate
        return max((depth(c) for c in node.get("children", [])), default=0)
    else:  # starting material
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", required=True)
    parser.add_argument("--out-targets", required=True)
    parser.add_argument("--depth", type=int, default=2, help="route depth to select")
    args = parser.parse_args()

    print(f"Loading {args.routes}...")
    with gzip.open(args.routes, "rt") as f:
        routes_data = json.load(f)

    targets = set()
    for row in routes_data:
        target_smiles = row["smiles"]
        for child in row.get("children", []):
            route_node = {"type": "mol", "smiles": target_smiles, "children": [child]}
            if depth(route_node) == args.depth:
                targets.add(target_smiles)
                break  # target has at least one route of the requested depth

    print(f"Found {len(targets)} targets with depth {args.depth} routes.")

    with gzip.open(args.out_targets, "wt") as f:
        for smi in sorted(targets):
            f.write(smi + "\n")


if __name__ == "__main__":
    main()

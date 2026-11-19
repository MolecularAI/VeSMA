from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .core import K
from .routes import Route, route_root

__all__ = [
    "aizynth_depth",
    "aizynth_reaction_metadata",
    "aizynth_starting_smiles",
    "route_from_aizynth_dict",
    "route_to_aizynth_dict",
]


def aizynth_depth(node: Mapping[str, Any]) -> int:
    """Return reaction depth of an AiZynthFinder route node."""

    if node["type"] == "reaction":
        return 1 + max(
            (aizynth_depth(child) for child in node.get("children", [])),
            default=0,
        )
    if node["type"] == "mol" and "children" in node:
        return max(
            (aizynth_depth(child) for child in node["children"]),
            default=0,
        )
    return 0


def aizynth_starting_smiles(node: Mapping[str, Any]) -> tuple[str, ...]:
    """Return starting-material SMILES from an AiZynthFinder route node."""

    if node["type"] == "mol" and "children" not in node:
        smiles = node["smiles"]
        if not isinstance(smiles, str):
            raise TypeError("AiZynthFinder molecule SMILES must be a string.")
        return (smiles,)

    starting: list[str] = []
    for child in node["children"]:
        starting.extend(aizynth_starting_smiles(child))
    return tuple(starting)


def aizynth_reaction_metadata(node: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return reaction metadata records in depth-first route order."""

    if node["type"] == "reaction":
        metadata = node["metadata"]
        if not isinstance(metadata, Mapping):
            raise TypeError("AiZynthFinder reaction metadata must be a mapping.")
        records = [dict(metadata)]
    else:
        records = []

    for child in node.get("children", []):
        records.extend(aizynth_reaction_metadata(child))
    return tuple(records)


def route_from_aizynth_dict(
    node: Mapping[str, Any],
    rule_id_from_reaction_metadata: Callable[[dict[str, Any]], int],
) -> Route:
    """Convert an AiZynthFinder route dict to a Vertsy route.

    Molecules are returned as their SMILES strings. Map the resulting route with
    a codec when a different molecule identity is needed.
    """

    if node["type"] != "mol":
        raise ValueError("AiZynthFinder route root must be a molecule node.")
    return route_from_aizynth_mol(node, rule_id_from_reaction_metadata)


def route_from_aizynth_mol(
    node: Mapping[str, Any],
    rule_id_from_reaction_metadata: Callable[[dict[str, Any]], int],
) -> Route:
    if node["type"] != "mol":
        raise ValueError("AiZynthFinder molecule route node must have type 'mol'.")

    product = node["smiles"]
    if not isinstance(product, str):
        raise TypeError("AiZynthFinder molecule SMILES must be a string.")

    children = tuple(node.get("children", ()))
    if not children:
        return Route(product, None)
    if len(children) != 1 or children[0]["type"] != "reaction":
        raise ValueError(
            "AiZynthFinder molecule route node must have one reaction child."
        )

    reaction = children[0]
    precursor_nodes = tuple(reaction.get("children", ()))
    if any(child["type"] != "mol" for child in precursor_nodes):
        raise ValueError("AiZynthFinder reaction children must be molecule nodes.")

    metadata = reaction["metadata"]
    if not isinstance(metadata, Mapping):
        raise TypeError("AiZynthFinder reaction metadata must be a mapping.")

    rule_id = rule_id_from_reaction_metadata(dict(metadata))
    children = tuple(
        route_from_aizynth_mol(child, rule_id_from_reaction_metadata)
        for child in precursor_nodes
    )
    return Route(product, rule_id, children)


def route_to_aizynth_dict(
    route: Route,
    mol_to_smiles: Callable[[K], str],
    reaction_metadata_from_rule_id: Callable[[int], Mapping[str, Any]],
    target: K | None = None,
) -> dict[str, Any]:
    """Convert a Vertsy route to an AiZynthFinder-like route dict."""
    if target is not None and target != route_root(route):
        raise ValueError("Tree routes can only be exported from their root product.")
    return aizynth_mol_from_route(route, mol_to_smiles, reaction_metadata_from_rule_id)


def aizynth_mol_from_route(
    route: Route,
    mol_to_smiles: Callable[[K], str],
    reaction_metadata_from_rule_id: Callable[[int], Mapping[str, Any]],
) -> dict[str, Any]:
    node: dict[str, Any] = {
        "type": "mol",
        "smiles": mol_to_smiles(route.product),
    }
    if route.rule is None:
        return node

    node["children"] = [
        {
            "type": "reaction",
            "metadata": dict(reaction_metadata_from_rule_id(route.rule)),
            "children": [
                aizynth_mol_from_route(
                    child,
                    mol_to_smiles,
                    reaction_metadata_from_rule_id,
                )
                for child in route.precursors
            ],
        }
    ]
    return node

from __future__ import annotations

import csv
import gzip
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vertsy.interop import aizynth_reaction_metadata
from vesma import CMol, SmartsError, cmol_from_smiles
from vesma.cpu_matcher import generate_precursors
from vesma.keyer import Keyer, cmol_rdkit_smiles_keyer
from vesma.reaction import ReactionTemplate, parse_reaction_smarts

TemplateRows = dict[str, dict[str, str]]

RAW_TEMPLATE_COLUMNS = (
    "reaction_smiles",
    "reaction_hash",
    "classification",
    "ring_breaker",
    "retro_template",
    "template_hash",
)

ROUTE_TEMPLATE_COLUMNS = ("reaction_hash", "retro_template")
RAW68K_TEMPLATE_COLUMNS = ("retro_template", "template_hash", "library_occurence")


@dataclass(frozen=True)
class ReferenceTemplateSet:
    templates: tuple[ReactionTemplate, ...]
    rule_id_by_reaction_hash: dict[str, int]


@dataclass(frozen=True)
class RouteRecoveryRuleIndex:
    smarts: tuple[str, ...]
    rule_id_by_reaction_hash: dict[str, int]


__all__ = [
    "RAW68K_TEMPLATE_COLUMNS",
    "RAW_TEMPLATE_COLUMNS",
    "ROUTE_TEMPLATE_COLUMNS",
    "ReferenceTemplateSet",
    "RouteRecoveryRuleIndex",
    "TemplateRows",
    "clean_reaction_smarts",
    "load_raw_template_rows",
    "load_route_template_rows",
    "raw68k_rule_index",
    "rdkit_precursor_sets",
    "reference_route_root",
    "reference_rule_id_by_hash",
    "reference_template_set",
    "reference_templates",
    "route_recovery_rule_index",
    "route_recovery_rule_index_from_rows",
    "uspto42k_rule_index",
    "vesma_precursor_sets",
]


def reference_route_root(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "mol",
        "smiles": row["smiles"],
        "children": row.get("children", []),
    }


def load_raw_template_rows(directory: Path) -> TemplateRows:
    path = directory / "uspto_raw_template_library.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != RAW_TEMPLATE_COLUMNS:
            raise ValueError(
                f"Unexpected small PaRoutes template columns: {reader.fieldnames}"
            )
        return {row["reaction_hash"]: dict(row) for row in reader}


def load_route_template_rows(path: Path) -> TemplateRows:
    """Load the committed ``(reaction_hash, retro_template)`` route-template subset.

    This is the trimmed library committed under ``processed/`` (e.g.
    ``used14k-reactions.tsv.gz``) that replaces the multi-hundred-MB raw library.
    """
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != ROUTE_TEMPLATE_COLUMNS:
            raise ValueError(f"Unexpected route-template columns: {reader.fieldnames}")
        return {
            row["reaction_hash"]: {"retro_template": row["retro_template"]}
            for row in reader
        }


def route_recovery_rule_index(path: Path) -> RouteRecoveryRuleIndex:
    """Intern route-recovery SMARTS strings and map route hashes to matcher ids."""
    return route_recovery_rule_index_from_rows(load_route_template_rows(path))


def route_recovery_rule_index_from_rows(rows: TemplateRows) -> RouteRecoveryRuleIndex:
    smarts: list[str] = []
    rule_id_by_smarts: dict[str, int] = {}
    rule_id_by_reaction_hash: dict[str, int] = {}
    for reaction_hash, row in rows.items():
        cleaned = clean_reaction_smarts(row["retro_template"])
        rule_id = rule_id_by_smarts.get(cleaned)
        if rule_id is None:
            rule_id = len(smarts)
            rule_id_by_smarts[cleaned] = rule_id
            smarts.append(cleaned)
        rule_id_by_reaction_hash[reaction_hash] = rule_id
    return RouteRecoveryRuleIndex(
        smarts=tuple(smarts),
        rule_id_by_reaction_hash=rule_id_by_reaction_hash,
    )


def raw68k_rule_index(
    path: Path,
    reference_reactions_path: Path,
) -> RouteRecoveryRuleIndex:
    """Deduplicated SMARTS index for the raw68k full USPTO template library.

    The 67,557 unique ``retro_template`` SMARTS of the raw PaRoutes template library, vendored
    by ``build_raw68k_templates.py``. The compact raw68k artifact drops reaction hashes, so map
    the hashes retained by ``used14k-reactions.tsv.gz`` back onto the deduplicated raw rules by
    cleaned SMARTS. raw68k is a superset of used14k; a missing reference template is an invalid
    benchmark artifact and fails fast.
    """
    smarts: list[str] = []
    rule_id_by_smarts: dict[str, int] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != RAW68K_TEMPLATE_COLUMNS:
            raise ValueError(f"Unexpected raw68k template columns: {reader.fieldnames}")
        for row in reader:
            cleaned = clean_reaction_smarts(row["retro_template"])
            if cleaned not in rule_id_by_smarts:
                rule_id_by_smarts[cleaned] = len(smarts)
                smarts.append(cleaned)
    reference_rows = load_route_template_rows(reference_reactions_path)
    missing: list[str] = []
    rule_id_by_reaction_hash: dict[str, int] = {}
    for reaction_hash, row in reference_rows.items():
        cleaned = clean_reaction_smarts(row["retro_template"])
        rule_id = rule_id_by_smarts.get(cleaned)
        if rule_id is None:
            missing.append(reaction_hash)
        else:
            rule_id_by_reaction_hash[reaction_hash] = rule_id
    if missing:
        raise ValueError(
            f"raw68k is missing {len(missing)} reference reaction templates."
        )
    return RouteRecoveryRuleIndex(
        smarts=tuple(smarts),
        rule_id_by_reaction_hash=rule_id_by_reaction_hash,
    )


def uspto42k_rule_index() -> RouteRecoveryRuleIndex:
    """Deduplicated SMARTS index for the uspto42k (uspto_unique_templates) rule set.

    These templates carry their own ids, unaligned with the PaRoutes route reaction hashes, so
    ``rule_id_by_reaction_hash`` is empty: only the reaction-agnostic comparators (precursors,
    starting materials, solved-at-all) are meaningful for this rule set.
    """
    from tests.datasets import load_compatible_templates

    smarts: list[str] = []
    rule_id_by_smarts: dict[str, int] = {}
    for template in load_compatible_templates():
        cleaned = clean_reaction_smarts(template.smarts)
        if cleaned not in rule_id_by_smarts:
            rule_id_by_smarts[cleaned] = len(smarts)
            smarts.append(cleaned)
    return RouteRecoveryRuleIndex(smarts=tuple(smarts), rule_id_by_reaction_hash={})


def reference_rule_id_by_hash(node: Mapping[str, Any]) -> dict[str, int]:
    rule_ids: dict[str, int] = {}
    for rule_id, metadata in enumerate(aizynth_reaction_metadata(node)):
        rule_ids.setdefault(str(metadata["reaction_hash"]), rule_id)
    return rule_ids


def reference_templates(
    directory: Path,
    node: Mapping[str, Any],
) -> tuple[ReactionTemplate, ...]:
    rows = load_raw_template_rows(directory)
    templates: list[ReactionTemplate] = []
    for rule_id, metadata in enumerate(aizynth_reaction_metadata(node)):
        row = rows[str(metadata["reaction_hash"])]
        cleaned = clean_reaction_smarts(row["retro_template"])
        templates.append(
            ReactionTemplate(
                id=str(rule_id),
                smarts=cleaned,
                pattern=parse_reaction_smarts(cleaned),
                metadata={
                    "reaction_hash": row["reaction_hash"],
                    "template_hash": row["template_hash"],
                    "classification": row["classification"],
                },
            )
        )
    return tuple(templates)


def reference_template_set(
    directory: Path,
    nodes: Iterable[Mapping[str, Any]],
) -> ReferenceTemplateSet:
    required_hashes = frozenset(
        str(metadata["reaction_hash"])
        for node in nodes
        for metadata in aizynth_reaction_metadata(node)
    )
    templates: list[ReactionTemplate] = []
    rule_id_by_reaction_hash: dict[str, int] = {}
    rule_id_by_template_hash: dict[str, int] = {}
    for row in load_raw_template_rows(directory).values():
        reaction_hash = row["reaction_hash"]
        template_hash = row["template_hash"]
        if template_hash in rule_id_by_template_hash:
            rule_id_by_reaction_hash[reaction_hash] = rule_id_by_template_hash[
                template_hash
            ]
            continue

        try:
            cleaned = clean_reaction_smarts(row["retro_template"])
            reaction = parse_reaction_smarts(cleaned)
        except SmartsError:
            continue

        rule_id = len(templates)
        rule_id_by_template_hash[template_hash] = rule_id
        rule_id_by_reaction_hash[reaction_hash] = rule_id
        templates.append(
            ReactionTemplate(
                id=template_hash,
                smarts=cleaned,
                pattern=reaction,
                metadata={
                    "reaction_hash": reaction_hash,
                    "template_hash": template_hash,
                    "classification": row["classification"],
                },
            )
        )

    missing = required_hashes - frozenset(rule_id_by_reaction_hash)
    if missing:
        raise ValueError(
            f"Small PaRoutes template subset is missing {len(missing)} reference reactions."
        )

    return ReferenceTemplateSet(
        templates=tuple(templates),
        rule_id_by_reaction_hash=rule_id_by_reaction_hash,
    )


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


def rdkit_precursor_sets(
    product_smiles: str,
    retro_template: str,
    identity_codec: Keyer[CMol, str] = cmol_rdkit_smiles_keyer,
) -> frozenset[frozenset[str]]:
    from rdkit import Chem
    from rdkit.Chem import rdChemReactions

    product = Chem.MolFromSmiles(product_smiles)
    if product is None:
        raise ValueError(f"Invalid product SMILES: {product_smiles}")
    reaction = rdChemReactions.ReactionFromSmarts(retro_template)
    return frozenset(
        frozenset(
            identity_codec.from_smiles(Chem.MolToSmiles(precursor))
            for precursor in outcome
        )
        for outcome in reaction.RunReactants((product,))
    )


def vesma_precursor_sets(
    product_smiles: str,
    retro_template: str,
    identity_codec: Keyer[CMol, str] = cmol_rdkit_smiles_keyer,
    max_mappings: int = 256,
) -> frozenset[frozenset[str]]:
    product = cmol_from_smiles(product_smiles)[0]
    reaction = parse_reaction_smarts(retro_template)
    return frozenset(
        frozenset(identity_codec.encode(precursor) for precursor in outcome.precursors)
        for outcome in generate_precursors(
            product,
            reaction,
            max_mappings=max_mappings,
        )
    )

"""Strict loaders for the paper's committed pytest-benchmark JSON artifacts."""

from __future__ import annotations

import gzip
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd

# Every backend name a bench target can write, not only the ones a table publishes. An unlisted
# name is a *hard* render failure rather than an ignored row, so an ablation arm whose records are
# meant for `output/` still belongs here: a record that lands in the paper tree by accident should
# show up as a surprising table row someone notices, not as a paper that cannot be built at all.
# `metal-joined` / `metal-prefilter-joined` are such arms (see `bench-route-recovery-metal`).
SUPPORTED_BACKENDS = frozenset(
    {
        "cpu",
        "cpu-parallel",
        "cpu-parallel-gpu-parity",
        "cpu-serial",
        "cpu-serial-gpu-parity",
        "cuda",
        "cuda-device",
        "cuda-device-prefilter",
        "cuda-host",
        "cuda-host-prefilter",
        "cuda-raw",
        "metal",
        "metal-joined",
        "metal-prefilter",
        "metal-prefilter-joined",
        "metal-raw",
        "rdkit",
        "rdkit-filtercatalog-rules",
        "rdkit-serial",
        "rdkit-parallel",
        "rdkit-substructlibrary-frontier",
    }
)
SUPPORTED_EXPERIMENTS = frozenset(
    {
        "fanout-depth",
        "match-only",
        "match-plus-split",
        "paper-validation",
        "reaction-application",
        "route-recovery",
        "route-replay",
    }
)

EXPERIMENT_FIELDS = {
    "reaction-application": {
        "contract": str,
        "size": int,
        "molecules": int,
        "rules": int,
        "positive_pairs": int,
        "precursors": int,
        "pair_sha256": str,
        "smiles_sha256": str,
        "templates_sha256": str,
        "precursor_identity": str,
        "delivery": str,
    },
    "fanout-depth": {
        "rule_set": str,
        "rules": int,
        "num_targets": int,
        "max_depth": int,
        "any_truncated": bool,
        "targets": list,
    },
    "match-only": {
        "size": int,
        "molecules": int,
        "rules": int,
        "pairs": int,
        "matches": int,
    },
    "match-plus-split": {
        "size": int,
        "molecules": int,
        "rules": int,
        "positive_pairs": int,
        "precursors": int,
    },
    "route-recovery": {
        "rule_set": str,
        "num_targets": int,
        "max_depth": int,
        "any_solved_per_target": list,
        "total_search_seconds": (int, float),
    },
    "route-replay": {
        "mode": str,
        "routes": int,
        "strict_reference_recoveries_list": list,
        "loose_reference_recoveries_list": list,
        "library_reference_recoveries_list": list,
        "strict_reference_missing_indices": list,
        "loose_reference_missing_indices": list,
        "library_reference_missing_indices": list,
        "diagnostics": dict,
    },
}

VALIDATION_FIELDS = {
    "identity-agreement": {
        "molecules_seen": int,
        "molecules_encoded": int,
        "identity_records": list,
        "encoding_failures": int,
        "wl_false_merge_records": list,
        "rdkit_false_split_records": list,
    },
    "template-compatibility": {"template_sources": list},
    "precursor-audit": {
        "pairs_compared": int,
        "agreement_records": list,
        "backend_records": list,
        "vesma_rejections": dict,
        "rdkit_dropped_outcomes": dict,
        "divergences": list,
        "refusals": list,
        "unkeyable_precursors": list,
        "rdkit_discards": list,
    },
}


def audit_application_totals(extra_info: dict[str, Any]) -> dict[str, int]:
    """Applications attempted, per backend, from an audit record.

    Each backend's accepted, refused, and unkeyable counts must close against that
    backend's own enumeration. Totals need not agree across backends because they enumerate
    and deduplicate reaction applications differently.
    """
    backends = {row["backend"]: row for row in extra_info["backend_records"]}
    return {
        backend: (
            int(row.get("applications_accepted", 0))
            + int(row.get("applications_refused", 0))
            + int(row.get("applications_unkeyable", 0))
        )
        for backend, row in backends.items()
    }


def require_audit_accounting(extra_info: dict[str, Any], *, context: str) -> None:
    """Fail unless each backend accounts for everything *it* enumerated.

    The invariant is per backend and internal. It is **not** that the backends agree on
    how many applications there were: vesma enumerates subgraph isomorphisms of the
    left-hand side, RDKit enumerates ``RunReactants`` outcomes, and the two count
    different things. Several distinct reaction centres can also collapse to one
    outcome, and each backend decides on its own whether that is one result or several.
    The only quantity comparable across backends is the *set of distinct outcomes* per
    pair, which is what the audit actually compares.

    """
    totals = audit_application_totals(extra_info)
    dropped = extra_info["rdkit_dropped_outcomes"]
    outcomes_total = int(dropped["outcomes_total"])
    discards = int(dropped["outcomes_unsanitizable"])

    if totals.get("rdkit") != outcomes_total:
        raise ValueError(
            f"{context}: RDKit accounts for {totals.get('rdkit')} applications against "
            f"{outcomes_total} outcomes it reports enumerating. Accepted plus discarded "
            "must equal the total."
        )
    if len(extra_info["rdkit_discards"]) != discards:
        raise ValueError(
            f"{context}: {len(extra_info['rdkit_discards'])} discard rows recorded "
            f"against {discards} counted. A total without its rows cannot be reviewed."
        )

    enumeration = extra_info.get("vesma_enumeration")
    if enumeration is None:
        return  # recorded from 2026-08-15; older artifacts cannot be checked this way
    attempted = int(enumeration["applications_attempted"])
    if totals.get("vesma-cpu") != attempted:
        raise ValueError(
            f"{context}: vesma accounts for {totals.get('vesma-cpu')} applications "
            f"against {attempted} its matcher enumerated. Accepted plus refused plus "
            "unkeyable must equal what was attempted."
        )
    if int(enumeration["pairs_search_incomplete"]):
        raise ValueError(
            f"{context}: {enumeration['pairs_search_incomplete']} pairs where the "
            "matcher declined to answer rather than answering 'no'. Those applications "
            "were never attempted and are absent from every total."
        )


def read_benchmark_text(path: Path) -> str:
    """Read a benchmark document, transparently decompressing a ``.json.gz``.

    Correctness artifacts are written uncapped and compressed: they carry every
    divergence and refusal in full, which is the point of them and also what makes them
    too large to keep as plain text.
    """
    if path.suffix == ".gz":
        return gzip.decompress(path.read_bytes()).decode("utf-8")
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    """Parse one non-empty pytest-benchmark JSON document or raise."""
    text = read_benchmark_text(path)
    if not text.strip():
        raise ValueError(f"Empty benchmark JSON: {path}")
    document = json.loads(text)
    if not isinstance(document, dict):
        raise ValueError(f"Benchmark JSON root must be an object: {path}")
    require_mapping(document, "machine_info", path)
    benchmarks = document.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ValueError(f"Benchmark JSON requires a non-empty benchmarks list: {path}")
    return document


def require_mapping(
    record: dict[str, Any], key: str, context: object
) -> dict[str, Any]:
    value = record.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{context}: {key!r} must be an object")
    return value


def require_fields(
    record: dict[str, Any],
    fields: dict[str, type | tuple[type, ...]],
    context: object,
) -> None:
    for key, expected in fields.items():
        value = record.get(key)
        if not isinstance(value, expected):
            names = (
                ", ".join(item.__name__ for item in expected)
                if isinstance(expected, tuple)
                else expected.__name__
            )
            raise ValueError(f"{context}: {key!r} must be {names}")


def validate_extra_info(extra_info: dict[str, Any], context: object) -> None:
    experiment = extra_info["experiment"]
    if experiment == "paper-validation":
        validation = extra_info.get("validation")
        if validation not in VALIDATION_FIELDS:
            raise ValueError(
                f"{context}: unsupported paper validation {validation!r}; "
                f"expected one of {sorted(VALIDATION_FIELDS)}"
            )
        require_fields(extra_info, VALIDATION_FIELDS[validation], context)
        return
    require_fields(extra_info, EXPERIMENT_FIELDS[experiment], context)
    if (
        experiment == "match-plus-split"
        and extra_info["backend"] == "cpu-parallel-gpu-parity"
    ):
        require_fields(
            extra_info,
            {"reaction_processing": dict, "cpu_retro_diagnostics": dict},
            context,
        )


def iter_records(root: Path) -> Iterator[tuple[Path, dict, dict, dict]]:
    """Yield validated ``(source_path, machine_info, extra_info, stats)`` records.

    An **empty** file is skipped with a warning naming it, rather than failing the render.
    `pytest-benchmark` creates its output file when a run starts and fills it when the run ends,
    so a benchmark in flight always leaves one behind -- and refusing it meant the paper could not
    be rendered at all while any measurement was running, which is exactly when someone wants to
    check the paper still builds. The warning is deliberate: an empty file is still a fact worth
    seeing, either a run in progress or one that died, and silently ignoring it would let a
    half-finished sweep look like a complete one.

    Every other defect is still fatal. A file that parses but carries the wrong experiment,
    backend or fields is a *wrong* record rather than an absent one, and publishing from it would
    be worse than not rendering.
    """
    paths = sorted({*root.rglob("*.json"), *root.rglob("*.json.gz")})
    if not paths:
        raise ValueError(f"No benchmark JSON found under {root}")

    usable = 0
    for path in paths:
        if not read_benchmark_text(path).strip():
            # Plain stderr rather than `warnings.warn`. Quarto renders a cell's stderr into the
            # document, so this lands as a visible block beside the table it affects -- which is
            # where someone checking a draft will actually see it. `warnings.warn` reaches the
            # same place but prefixes the module path and trails a line of this function's own
            # source, which reads as a malfunction rather than a note when it appears in a PDF.
            print(
                f"WARNING: skipping empty benchmark JSON "
                f"(a run in progress, or one that died): {path}",
                file=sys.stderr,
                flush=True,
            )
            continue
        usable += 1
        document = load_json(path)
        machine = require_mapping(document, "machine_info", path)
        for index, entry in enumerate(document["benchmarks"]):
            context = f"{path}: benchmark {index}"
            if not isinstance(entry, dict):
                raise ValueError(f"{context} must be an object")
            extra_info = require_mapping(entry, "extra_info", context)
            stats = require_mapping(entry, "stats", context)
            experiment = extra_info.get("experiment")
            if experiment not in SUPPORTED_EXPERIMENTS:
                raise ValueError(
                    f"{context}: unsupported experiment {experiment!r}; "
                    f"expected one of {sorted(SUPPORTED_EXPERIMENTS)}"
                )
            backend = extra_info.get("backend")
            if backend not in SUPPORTED_BACKENDS:
                raise ValueError(
                    f"{context}: unsupported backend {backend!r}; "
                    f"expected one of {sorted(SUPPORTED_BACKENDS)}"
                )
            require_fields(stats, {"mean": (int, float), "data": list}, context)
            validate_extra_info(extra_info, context)
            yield path.resolve(), machine, extra_info, stats

    if not usable:
        raise ValueError(
            f"every benchmark JSON under {root} was empty; "
            "nothing to render from, so this is a failure rather than a warning"
        )


def load_benchmarks(root: Path) -> pd.DataFrame:
    """Load one normalized row per strictly validated benchmark record."""
    records = [
        {
            "source_path": str(source_path),
            "machine": machine,
            **extra_info,
            "stats": stats,
        }
        for source_path, machine, extra_info, stats in iter_records(root)
    ]
    return pd.json_normalize(records)


def require_columns(
    frame: pd.DataFrame, columns: Iterable[str], *, context: str
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{context}: missing required columns {missing}")


def explode_records(df: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    """Expand a required list-of-dicts column, preserving its parent columns."""
    require_columns(df, [column], context=f"explode {prefix}")
    invalid_lists = [
        index for index, value in df[column].items() if not isinstance(value, list)
    ]
    if invalid_lists:
        raise ValueError(
            f"explode {prefix}: {column!r} must contain lists; invalid rows "
            f"{invalid_lists[:10]}"
        )
    invalid_children = [
        index
        for index, values in df[column].items()
        if any(not isinstance(value, dict) for value in values)
    ]
    if invalid_children:
        raise ValueError(
            f"explode {prefix}: {column!r} must contain dictionaries; invalid rows "
            f"{invalid_children[:10]}"
        )

    populated = df[df[column].map(bool)]
    if populated.empty:
        return populated.drop(columns=[column]).reset_index(drop=True)
    exploded = populated.explode(column, ignore_index=True)
    child_frame = pd.json_normalize(exploded[column]).add_prefix(f"{prefix}.")
    parents = exploded.drop(columns=[column]).reset_index(drop=True)
    return pd.concat([parents, child_frame], axis=1)

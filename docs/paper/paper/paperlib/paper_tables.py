"""Render benchmark DataFrames into format-agnostic Quarto and Markdown tables.

Extraction lives in :mod:`paperlib.loaders`; this module holds the composable table
rendering mini-library (``TableCell``, ``Table``, ``table_from_dataframe``,
``list_table``, ``pipe_table``, ``qtable``, ``compact_multiindex_rows``), the cell
formatters, record validators, and backend label metadata.

Tables render to format-independent Pandoc/Quarto markup rather than backend-specific
raw blocks: ``list_table`` (default) emits Quarto ``.list-table`` divs supporting
multi-line cells, row/column spans, multi-level headers, alignments, and relative column
widths across Typst, LaTeX, and DOCX; ``pipe_table`` emits standard Markdown pipe tables
for flat single-line tables. Any ``Callable[[Table], str]`` renderer can be injected
into ``qtable(..., render=...)``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import groupby
from math import floor, isfinite, log10
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from paperlib.loaders import load_benchmarks, require_columns

__all__ = [
    # loaders (re-exported from paperlib.loaders)
    "load_benchmarks",
    "require_columns",
    # record validation and selection
    "require_agreeing_records",
    "require_one_record",
    "require_unique_records",
    "select_records",
    "with_rates",
    "load_application",
    # labels and metadata
    "backend_label",
    "backend_order",
    "host_label",
    # cell formatters
    "fraction_cell",
    "format_duration",
    "format_throughput",
    "format_significant",
    "format_microseconds",
    "scaling_cell",
    # benchmark table builders
    "precursor_audit_agreement_table",
    "precursor_audit_class_table",
    "scaling_table",
    "compact_multiindex_rows",
    # composable table rendering mini-library
    "Alignment",
    "TableCell",
    "Table",
    "TableRenderer",
    "cell_text",
    "parse_alignments",
    "table_from_dataframe",
    "list_table",
    "pipe_table",
    "qtable",
]

type Alignment = Literal["l", "c", "r", "d"]

VALID_ALIGNMENTS: dict[str, Alignment] = {
    "l": "l",
    "c": "c",
    "r": "r",
    "d": "d",
}
PIPE_ALIGNMENT_RULES: dict[Alignment, str] = {
    "l": ":---",
    "c": ":---:",
    "r": "---:",
    "d": "---",
}

MULTIPLIERS: dict[str, float] = {
    "": 1,
    "k": 1e3,
    "M": 1e6,
}

# Publication aggregates from the completed full CPU precursor audit at the frozen
# chemistry boundary. Keeping the small reduction here lets the main article and SI render
# from one tracked source without turning the 4.6 MB application-level audit into a paper
# build dependency.
PRECURSOR_AUDIT_PAIRS = 768_015
PRECURSOR_AUDIT_VESMA_APPLICATIONS = 2_468_381
PRECURSOR_AUDIT_RDKIT_OUTCOMES = 2_468_494
PRECURSOR_AUDIT_AGREEMENTS = (
    ("VeSMA-RDChiral", 755_047, 283, 754_780),
    ("VeSMA-RDKit", 710_592, 11_251, 705_147),
    ("RDKit-RDChiral", 711_864, 11_266, 706_189),
)

BACKEND_LABELS: dict[str, str] = {
    "cpu": "Vesma CPU",
    "cpu-parallel": "Vesma CPU parallel",
    "cpu-serial-gpu-parity": "Vesma CPU serial (GPU parity)",
    "cpu-parallel-gpu-parity": "Vesma CPU parallel (GPU parity)",
    "cpu-serial": "Vesma CPU serial",
    "cuda": "CUDA",
    "cuda-host": "CUDA host",
    "cuda-device": "CUDA device",
    "cuda-host-prefilter": "CUDA host + prefilter",
    "cuda-device-prefilter": "CUDA device + prefilter",
    "cuda-raw": "CUDA raw",
    "metal": "Metal",
    "metal-joined": "Metal, joined precursors",
    "metal-prefilter": "Metal w/ prefilter",
    "metal-prefilter-joined": "Metal w/ prefilter, joined precursors",
    "metal-raw": "Metal raw",
    "rdkit": "RDKit",
    "rdkit-serial": "RDKit",
    "rdkit-parallel": "RDKit parallel",
    "rdkit-filtercatalog-rules": "RDKit FilterCatalog",
    "rdkit-substructlibrary-frontier": "RDKit SubstructLibrary",
}

BACKEND_ORDER: dict[str, int] = {
    "rdkit-filtercatalog-rules": 0,
    "rdkit-substructlibrary-frontier": 1,
    "rdkit": 2,
    "rdkit-serial": 2,
    "rdkit-parallel": 2,
    "cpu": 3,
    "cpu-serial": 3,
    "cpu-serial-gpu-parity": 3,
    "cpu-parallel": 4,
    "cpu-parallel-gpu-parity": 4,
    "metal": 5,
    "metal-joined": 5,
    "metal-prefilter": 5,
    "metal-prefilter-joined": 5,
    "metal-raw": 6,
    "cuda-device": 7,
    "cuda-device-prefilter": 8,
    "cuda-host": 9,
    "cuda-host-prefilter": 10,
    "cuda": 11,
    "cuda-raw": 12,
}


# --------------------------------------------------------------------------- #
# Labels and metadata
# --------------------------------------------------------------------------- #


def backend_label(backend: str) -> str:
    return BACKEND_LABELS[backend]


def backend_order(backend: str) -> int:
    return BACKEND_ORDER[backend]


def host_label(record: pd.Series, joiner: str = "plus") -> str:
    """Build a display label from the hardware metadata in a benchmark record."""
    cpu = str(
        record.get("machine.cpu.brand_raw") or record.get("machine.host_type") or ""
    )
    for noise in (" 64-Core Processor", " 32-Core Processor", " Processor"):
        cpu = cpu.replace(noise, "")
    cpu = " ".join(cpu.split())
    device = record.get("machine.cuda.device_name")
    if not device or pd.isna(device):
        return cpu
    return f"{cpu} {joiner} {device}"


# --------------------------------------------------------------------------- #
# Record validation and selection
# --------------------------------------------------------------------------- #


def require_one_record(frame: pd.DataFrame, *, context: str) -> pd.Series:
    """Return the sole selected record or identify every conflicting JSON file."""
    if len(frame) == 1:
        return frame.iloc[0]
    if frame.empty:
        raise ValueError(f"{context}: expected exactly one record, found none")
    if "source_path" not in frame.columns:
        raise ValueError(
            f"{context}: expected exactly one record, found {len(frame)}; "
            "the selected rows do not include source_path"
        )
    counts = frame["source_path"].value_counts().sort_index()
    conflicts = "\n".join(
        f"- {path}" + (f" ({count} matching records)" if count > 1 else "")
        for path, count in counts.items()
    )
    raise ValueError(
        f"{context}: expected exactly one record, found {len(frame)}.\n"
        f"Conflicting benchmark JSON files:\n{conflicts}\n"
        "Keep the intended publication record and delete or move every other listed "
        "JSON file; the paper deliberately does not choose one implicitly."
    )


def require_agreeing_records(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    context: str,
    ignore_nested_keys: frozenset[str] = frozenset(),
) -> pd.Series:
    """Return one representative row when all selected result fields agree."""
    value_columns = list(columns)
    require_columns(
        frame,
        ["source_path", *value_columns],
        context=context,
    )
    if frame.empty:
        raise ValueError(f"{context}: expected at least one record, found none")

    def comparable(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: comparable(item)
                for key, item in sorted(value.items())
                if key not in ignore_nested_keys
            }
        if isinstance(value, list):
            return [comparable(item) for item in value]
        if isinstance(value, tuple):
            return tuple(comparable(item) for item in value)
        item_fn = getattr(value, "item", None)
        if callable(item_fn):
            return item_fn()
        return value

    def encoded(value: object) -> str:
        return json.dumps(
            comparable(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    reference = frame.iloc[0]
    reference_path = reference["source_path"]
    disagreements = []
    for _, record in frame.iloc[1:].iterrows():
        differing_fields = [
            column
            for column in value_columns
            if encoded(reference[column]) != encoded(record[column])
        ]
        if differing_fields:
            disagreements.append(
                f"- {record['source_path']}: {', '.join(differing_fields)}"
            )
    if disagreements:
        details = "\n".join(disagreements)
        raise ValueError(
            f"{context}: retained audit records disagree with "
            f"{reference_path}.\nDiffering records and fields:\n{details}"
        )
    return reference


def require_unique_records(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    context: str,
) -> None:
    """Reject duplicate experimental coordinates and list their source files."""
    keys = list(columns)
    require_columns(frame, [*keys, "source_path"], context=context)
    duplicates = frame[frame.duplicated(keys, keep=False)]
    if duplicates.empty:
        return
    conflicts = []
    for values, group in duplicates.groupby(keys, sort=True, dropna=False):
        coordinates = values if isinstance(values, tuple) else (values,)
        selector = ", ".join(
            f"{key}={value!r}" for key, value in zip(keys, coordinates, strict=True)
        )
        paths = "\n".join(
            f"  - {path}" for path in sorted(group["source_path"].unique())
        )
        conflicts.append(f"- {selector}\n{paths}")
    raise ValueError(
        f"{context}: duplicate benchmark records were selected:\n"
        + "\n".join(conflicts)
        + "\nKeep the intended publication record for each coordinate and delete or "
        "move the other listed JSON files; the paper deliberately does not "
        "choose one implicitly."
    )


def with_rates(records: pd.DataFrame) -> pd.DataFrame:
    """Add each record's positive-pair throughput and its time per raw precursor, in seconds."""
    require_columns(
        records, ["positive_pairs", "precursors", "stats.mean"], context="rates"
    )
    return records.assign(
        pairs_per_second=records["positive_pairs"] / records["stats.mean"],
        seconds_per_precursor=records["stats.mean"] / records["precursors"],
    )


def load_application(root: Path) -> pd.DataFrame:
    """Load the streamed host-output workload used by the main performance table."""
    records = load_benchmarks(root)
    require_columns(
        records,
        ["contract", "precursor_identity", "delivery", "stats.rounds"],
        context="reaction application",
    )
    for column, expected in (
        ("experiment", "reaction-application"),
        ("contract", "streamed-unkeyed-host-v1"),
        ("precursor_identity", "none"),
        ("delivery", "host-batches-released"),
        ("stats.rounds", 3),
    ):
        if not records[column].eq(expected).all():
            raise ValueError(f"Reaction application requires {column}={expected!r}")
    require_unique_records(
        records,
        ["backend", "machine.host_type", "molecules"],
        context="reaction application",
    )
    for size, same_size in records.groupby("molecules"):
        require_agreeing_records(
            same_size,
            [
                "rules",
                "positive_pairs",
                "pair_sha256",
                "smiles_sha256",
                "templates_sha256",
            ],
            context=f"reaction-application inputs at {size} targets",
        )
        vesma = same_size[~same_size["backend"].str.startswith("rdkit")]
        require_agreeing_records(
            vesma,
            ["precursors", "precursor_atom_cap", "reaction_processing.after_split"],
            context=f"reaction-application VeSMA outputs at {size} targets",
        )
    return with_rates(records)


def select_records(
    records: pd.DataFrame, keys: Iterable[tuple[Any, ...]], columns: list[str]
) -> pd.DataFrame:
    """Return the sole record for each key, in the order the keys are given.

    ``keys`` are value tuples for ``columns``. A key matching no record or several raises,
    so a table never silently lacks a row or picks between duplicates.
    """
    wanted = pd.DataFrame(list(keys), columns=pd.Index(columns))
    merged = wanted.merge(records, on=columns, how="left", indicator=True)
    matches = merged.groupby(columns, sort=False)["_merge"].agg(
        lambda found: int((found == "both").sum())
    )
    if (matches != 1).any():
        raise ValueError(
            f"Expected exactly one record per key; matches found:\n{matches[matches != 1]}"
        )
    return merged.drop(columns="_merge")


# --------------------------------------------------------------------------- #
# Cell formatters
# --------------------------------------------------------------------------- #


def fraction_cell(recovered: Any, total: Any, decimals: int = 1) -> str:
    """Render a required recovered/total count with its percentage."""
    if pd.isna(recovered) or pd.isna(total):
        raise ValueError(
            "fraction_cell requires non-missing recovered and total values"
        )
    recovered = int(recovered)
    total = int(total)
    pct = 0.0 if total == 0 else 100.0 * recovered / total
    return f"{recovered:,}/{total:,}\\\n({pct:.{decimals}f}%)"


def format_duration(seconds: float, decimals: int = 1) -> str:
    """Format a wall-clock duration, auto-scaling to s / ms / µs.

    Used for the per-route and per-target average-time columns, whose values span
    milliseconds (a GPU or parallel-CPU replay) to hundreds of seconds (a single-target
    full-library search), so a fixed unit would either lose precision or read awkwardly.
    """
    if not isfinite(seconds):
        return f"{seconds} s"
    if seconds >= 1:
        return f"{seconds:.{decimals}f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.{decimals}f} ms"
    return f"{seconds * 1e6:.{decimals}f} µs"


def format_throughput(
    value: float,
    scale: str = "",
    unit: str = "",
    decimals: int = 1,
) -> str:
    """Format throughput using a fixed scale, e.g. ``3.5M pairs/s``.

    For scaled values >= 1, uses ``decimals`` decimal places.
    For scaled values < 1, uses enough decimals to avoid displaying ``0.0M``.
    """
    if scale not in MULTIPLIERS:
        raise ValueError(
            f"Unknown scale {scale!r}. Expected one of {list(MULTIPLIERS)}"
        )

    if not isfinite(value):
        return f"{value} {unit}"

    scaled = value / MULTIPLIERS[scale]

    if scaled == 0:
        text = f"{0:.{decimals}f}"
    elif abs(scaled) >= 1:
        text = f"{scaled:.{decimals}f}"
    else:
        extra_decimals = -floor(log10(abs(scaled)))
        n_decimals = max(decimals, extra_decimals)
        text = f"{scaled:.{n_decimals}f}"

    return f"{text}{scale} {unit}"


def format_significant(value: float, digits: int = 3) -> str:
    """Round to ``digits`` significant figures, with thousands separators: 6781 -> 6,780."""
    if value == 0 or not isfinite(value):
        return f"{value:,}"
    ndigits = digits - 1 - floor(log10(abs(value)))
    return f"{round(value, ndigits):,.{max(0, ndigits)}f}"


def format_microseconds(seconds: float, decimals: int = 1) -> str:
    """Format a duration as a bare number of microseconds, for a column headed with the unit."""
    return f"{seconds * 1e6:,.{decimals}f}"


def scaling_cell(record: pd.Series) -> str:
    """Wall time over throughput, as one table cell."""
    time = format_duration(float(record["stats.mean"]))
    rate = format_throughput(record["pairs_per_second"], "k", "p.pairs/s")
    return f"{time}\\\n({rate})"


# --------------------------------------------------------------------------- #
# Benchmark table builders
# --------------------------------------------------------------------------- #


def precursor_audit_agreement_table() -> pd.DataFrame:
    """Pairwise exact outcome agreement in the completed full precursor audit."""
    return pd.DataFrame(
        [
            {
                "Comparison": comparison,
                "All positive pairs": fraction_cell(equal, PRECURSOR_AUDIT_PAIRS),
                "Comparable pairs": fraction_cell(
                    equal_comparable,
                    PRECURSOR_AUDIT_PAIRS - incomparable,
                ),
            }
            for comparison, equal, incomparable, equal_comparable in (
                PRECURSOR_AUDIT_AGREEMENTS
            )
        ]
    )


def precursor_audit_class_table() -> pd.DataFrame:
    """Major semantic classes retained for interpretation in the SI."""
    return pd.DataFrame(
        [
            {
                "Class": "Circuit-relative aromaticity",
                "Frequency in audit": "Focused regression example",
                "Chemical or semantic interpretation": (
                    "VeSMA and RDKit make different explicit graph-model choices for a "
                    "fused-ring bond. This is a difference between aromaticity models, "
                    "not a measurement of physical aromaticity."
                ),
            },
            {
                "Class": "Delocalized bond released from its circuit",
                "Frequency in audit": fraction_cell(
                    29_215,
                    PRECURSOR_AUDIT_VESMA_APPLICATIONS,
                    decimals=2,
                ),
                "Chemical or semantic interpretation": (
                    "VeSMA stores delocalized bonds categorically, without a hidden "
                    "single or double order to reveal after editing. It therefore refuses "
                    "an under-specified edit instead of choosing a localization."
                ),
            },
            {
                "Class": "Other typed chemistry refusal",
                "Frequency in audit": fraction_cell(
                    5_218,
                    PRECURSOR_AUDIT_VESMA_APPLICATIONS,
                    decimals=2,
                ),
                "Chemical or semantic interpretation": (
                    "The edited graph asks for a circuit outside automatic discovery or "
                    "for an undeclared, incompatible, or invalid closed-shell state. "
                    "An explicit refusal preserves the model's stated domain."
                ),
            },
            {
                "Class": "Derived RHS property differs",
                "Frequency in audit": fraction_cell(
                    3_936,
                    PRECURSOR_AUDIT_VESMA_APPLICATIONS,
                    decimals=2,
                ),
                "Chemical or semantic interpretation": (
                    "The RHS is applied as a graph rewrite. Derived SMARTS properties are "
                    "recomputed from the result, so the record is neutral telemetry rather "
                    "than a rejection or silent coercion."
                ),
            },
            {
                "Class": "Component has no RHS atom",
                "Frequency in audit": fraction_cell(
                    540,
                    PRECURSOR_AUDIT_VESMA_APPLICATIONS,
                    decimals=3,
                ),
                "Chemical or semantic interpretation": (
                    "The template does not anchor that disconnected material on the RHS. "
                    "VeSMA omits the unanchored component and reports the event instead of "
                    "inventing its intended role."
                ),
            },
            {
                "Class": "Multi-component RHS meets a ring",
                "Frequency in audit": fraction_cell(
                    49_750, PRECURSOR_AUDIT_PAIRS, decimals=2
                ),
                "Chemical or semantic interpretation": (
                    "The match lies outside the usual intermolecular reading of the "
                    "template. VeSMA conserves atoms while opening the ring; RDKit realizes "
                    "the declared product-component topology. Both are consistent readings "
                    "of an under-specified ring match."
                ),
            },
            {
                "Class": "Hydrogen completion after a local edit",
                "Frequency in audit": fraction_cell(
                    1_773,
                    PRECURSOR_AUDIT_VESMA_APPLICATIONS,
                    decimals=3,
                ),
                "Chemical or semantic interpretation": (
                    "VeSMA completes hydrogen from the atom's preserved closed-shell "
                    "configuration after a local edit. The event is commoner than a visible "
                    "outcome difference: only four applications differ solely for it."
                ),
            },
            {
                "Class": "RDChiral stereochemical suppression",
                "Frequency in audit": fraction_cell(
                    5_026, PRECURSOR_AUDIT_PAIRS, decimals=2
                ),
                "Chemical or semantic interpretation": (
                    "RDChiral applies stereochemical checks that VeSMA deliberately omits. "
                    "The count reflects different supported semantics, not disagreement on "
                    "the underlying non-stereochemical graph edit."
                ),
            },
            {
                "Class": "Outcome outside the shared comparison key",
                "Frequency in audit": (
                    "RDKit "
                    f"{fraction_cell(17_140, PRECURSOR_AUDIT_RDKIT_OUTCOMES, decimals=2)}; "
                    "VeSMA "
                    f"{fraction_cell(281, PRECURSOR_AUDIT_VESMA_APPLICATIONS, decimals=3)}"
                ),
                "Chemical or semantic interpretation": (
                    "The shared key requires RDKit sanitization and SMILES serialization. "
                    "Failure at that exchange boundary makes an outcome incomparable; it is "
                    "not, by itself, a verdict on the native chemistry result."
                ),
            },
            {
                "Class": "Open-shell molecule literal",
                "Frequency in audit": "Not exercised by this audit",
                "Chemical or semantic interpretation": (
                    "A bracketed atom in a SMILES literal is read as OpenSMILES "
                    "specifies, so [N] is atomic nitrogen rather than ammonia and TEMPO "
                    "is a nitroxide radical. RDKit and RDChiral read it the same way; "
                    "the divergence is that VeSMA's closed-shell model then refuses the "
                    "radical by name rather than building it. The audit imports "
                    "molecules through RDKit and therefore does not exercise the "
                    "literal path."
                ),
            },
        ]
    )


def scaling_table(
    records: pd.DataFrame, *, index_name: str = "Backend and machine"
) -> pd.DataFrame:
    """Backends over machines as rows and workload sizes as columns, one cell per record.

    ``records`` need the ``with_rates`` columns. A backend-machine pair lacking a size
    leaves that cell empty.
    """
    require_unique_records(
        records, ["backend", "machine.host_type", "molecules"], context="scaling table"
    )
    labelled = records.assign(
        Backend=records["backend"].map(backend_label),
        Machine=records["machine.host_type"],
        cell=records.apply(scaling_cell, axis=1),
    )
    in_order = labelled.assign(order=labelled["backend"].map(backend_order))
    labelled["Backend"] = pd.Categorical(
        labelled["Backend"],
        categories=in_order.sort_values(["order", "Backend"])["Backend"].unique(),
        ordered=True,
    )
    table = labelled.pivot(
        index=["Backend", "Machine"], columns="molecules", values="cell"
    )
    table.columns = [f"{int(size):,}" for size in table.columns]
    return compact_multiindex_rows(table, index_name=index_name)


def compact_multiindex_rows(
    df: pd.DataFrame,
    *,
    index_name: str = "Backend and Machine",
    indent: str = "\u00a0\u00a0\u00a0\u00a0",
) -> pd.DataFrame:
    """Convert a MultiIndex-row DataFrame into a flat outline in a single first column.

    Parent index rows get blank data cells. Leaf index rows carry the actual DataFrame
    values, indented by level with non-breaking spaces so indentation survives Markdown,
    Typst, LaTeX, and DOCX rendering.
    """
    if not isinstance(df.index, pd.MultiIndex):
        out = df.reset_index()
        return out.rename(columns={out.columns[0]: index_name})

    df = df.sort_index()
    rows = []
    nlevels = df.index.nlevels
    data_columns = list(df.columns)
    previous_key: tuple[Any, ...] | None = None

    for key, values in df.iterrows():
        key_tuple = key if isinstance(key, tuple) else (key,)
        for level in range(nlevels - 1):
            prefix = key_tuple[: level + 1]
            if previous_key is None or previous_key[: level + 1] != prefix:
                rows.append(
                    {
                        index_name: f"{indent * level}{key_tuple[level]}",
                        **dict.fromkeys(data_columns, ""),
                    }
                )
        rows.append(
            {
                index_name: f"{indent * (nlevels - 1)}{key_tuple[-1]}",
                **dict(zip(data_columns, values, strict=True)),
            }
        )
        previous_key = key_tuple

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Composable table rendering mini-library
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TableCell:
    """One cell in a normalized table, optionally spanning rows or columns."""

    text: str
    rowspan: int = 1
    colspan: int = 1


@dataclass(frozen=True)
class Table:
    """Format-independent table with validated layout metadata."""

    headers: tuple[tuple[TableCell, ...], ...]
    rows: tuple[tuple[TableCell, ...], ...]
    aligns: tuple[Alignment, ...]
    colwidths: tuple[int | float, ...] | None = None
    caption: str | None = None
    table_id: str | None = None

    def __post_init__(self) -> None:
        if not self.aligns:
            raise ValueError("Cannot render a table with zero columns")
        invalid = [a for a in self.aligns if a not in VALID_ALIGNMENTS]
        if invalid:
            raise ValueError(
                f"Unknown alignment specifier(s) {invalid!r}; "
                f"expected one of {sorted(VALID_ALIGNMENTS)}"
            )
        if self.colwidths is not None:
            if len(self.colwidths) != len(self.aligns):
                raise ValueError(
                    f"Expected {len(self.aligns)} column widths, "
                    f"got {len(self.colwidths)}: {self.colwidths!r}"
                )
            for width in self.colwidths:
                if isinstance(width, bool) or not isfinite(width) or width <= 0:
                    raise ValueError(
                        f"Column widths must be finite positive numbers, got {width!r}"
                    )
        if self.caption is not None and "\n\n" in self.caption:
            raise ValueError(
                "Table captions must be a single paragraph (no blank lines)"
            )
        if self.table_id is not None and (
            not self.table_id
            or self.table_id.startswith("#")
            or any(ch.isspace() for ch in self.table_id)
        ):
            raise ValueError(
                "table_id must be a bare identifier without '#' or whitespace, "
                f"got {self.table_id!r}"
            )

    @property
    def column_count(self) -> int:
        return len(self.aligns)


type TableRenderer = Callable[[Table], str]


def cell_text(value: object) -> str:
    """Convert a scalar cell value to text, mapping missing values to an empty string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    missing = pd.isna(value)
    return "" if isinstance(missing, bool) and missing else str(value)


def parse_alignments(
    aligns: str | Sequence[Alignment] | None,
    column_count: int,
) -> tuple[Alignment, ...]:
    """Validate and normalize column alignment specs, defaulting to left alignment.

    Defaulting to explicit left alignment (``"l"``) keeps Typst, LaTeX, and DOCX output
    consistent: Pandoc's DOCX writer otherwise centers cells whose alignment is left at
    ``AlignDefault``.
    """
    if aligns is None:
        return ("l",) * column_count

    if isinstance(aligns, str):
        raw = (
            [part.strip() for part in aligns.split(",")]
            if "," in aligns
            else list(aligns.strip())
        )
    else:
        raw = list(aligns)

    if len(raw) != column_count:
        raise ValueError(
            f"Expected {column_count} column alignments, got {len(raw)}: {aligns!r}"
        )

    invalid = [token for token in raw if token not in VALID_ALIGNMENTS]
    if invalid:
        raise ValueError(
            f"Unknown alignment specifier(s) {invalid!r}; "
            f"expected one of {sorted(VALID_ALIGNMENTS)}"
        )
    return tuple(VALID_ALIGNMENTS[token] for token in raw)


def table_from_dataframe(
    df: pd.DataFrame,
    *,
    caption: str | None = None,
    table_id: str | None = None,
    colwidths: Sequence[int | float] | None = None,
    aligns: str | Sequence[Alignment] | None = None,
    include_index: bool = False,
) -> Table:
    """Build a validated, format-independent ``Table`` from a flat or MultiIndex DataFrame.

    Supports both flat and ``MultiIndex`` columns (emitting ``colspan`` across repeated
    parent levels) and, when ``include_index=True``, single-level or ``MultiIndex`` row
    indices (emitting ``rowspan`` across consecutive identical index prefixes).
    """
    col_index = df.columns
    col_nlevels = col_index.nlevels
    row_index = df.index
    row_nlevels = row_index.nlevels if include_index else 0
    column_count = row_nlevels + len(col_index)

    if column_count == 0:
        raise ValueError("Cannot render a table with zero columns")

    row_names = [cell_text(name) for name in row_index.names] if include_index else []

    if isinstance(col_index, pd.MultiIndex):
        col_tuples = list(col_index)
        header_rows: list[tuple[TableCell, ...]] = []
        for level in range(col_nlevels):
            cells = (
                [TableCell(name, rowspan=col_nlevels) for name in row_names]
                if level == 0
                else []
            )
            cells.extend(
                TableCell(cell_text(prefix[-1]), colspan=sum(1 for _ in group))
                for prefix, group in groupby(col_tuples, key=lambda t: t[: level + 1])
            )
            header_rows.append(tuple(cells))
    else:
        header_rows = [
            tuple(
                [TableCell(name) for name in row_names]
                + [TableCell(cell_text(col)) for col in col_index]
            )
        ]

    values = df.to_numpy(dtype=object)
    if include_index:
        row_tuples = (
            list(row_index)
            if isinstance(row_index, pd.MultiIndex)
            else [(item,) for item in row_index]
        )
        span_starts: list[dict[int, int]] = []
        for level in range(row_nlevels):
            starts: dict[int, int] = {}
            pos = 0
            for _, group in groupby(row_tuples, key=lambda t: t[: level + 1]):
                length = sum(1 for _ in group)
                starts[pos] = length
                pos += length
            span_starts.append(starts)

        body_rows = tuple(
            tuple(
                [
                    TableCell(
                        cell_text(idx_tuple[level]), rowspan=span_starts[level][i]
                    )
                    for level in range(row_nlevels)
                    if i in span_starts[level]
                ]
                + [TableCell(cell_text(val)) for val in values[i]]
            )
            for i, idx_tuple in enumerate(row_tuples)
        )
    else:
        body_rows = tuple(
            tuple(TableCell(cell_text(val)) for val in row_values)
            for row_values in values
        )

    clean_caption = caption.strip() if caption is not None else None
    return Table(
        headers=tuple(header_rows),
        rows=body_rows,
        aligns=parse_alignments(aligns, column_count),
        colwidths=tuple(colwidths) if colwidths is not None else None,
        caption=clean_caption or None,
        table_id=table_id or None,
    )


def list_table(table: Table) -> str:
    """Render a ``Table`` as a Quarto ``.list-table`` fenced div.

    Uses Quarto's native ``list-table`` attributes (``widths``, ``aligns``, and
    ``header-rows``), which translate into Pandoc AST table metadata honored consistently
    by Typst, LaTeX, and DOCX writers—including for tables with row/column spans.
    """
    attrs: list[str] = []
    if table.table_id:
        attrs.append(f"#{table.table_id}")
    attrs.append(".list-table")
    if len(table.headers) > 1:
        attrs.append(f"header-rows={len(table.headers)}")
    if table.colwidths is not None:
        widths = ",".join(
            f"{w:g}" if isinstance(w, float) else str(w) for w in table.colwidths
        )
        attrs.append(f'widths="{widths}"')
    attrs.append(f'aligns="{",".join(table.aligns)}"')

    out = [f"::: {{{' '.join(attrs)}}}"]
    if table.caption:
        out.extend([table.caption, ""])

    def emit_cell(prefix: str, cell: TableCell) -> None:
        span_attrs: list[str] = []
        if cell.rowspan > 1:
            span_attrs.append(f"rowspan={cell.rowspan}")
        if cell.colspan > 1:
            span_attrs.append(f"colspan={cell.colspan}")
        span_prefix = f"[]{{{' '.join(span_attrs)}}}" if span_attrs else ""
        content = f"{span_prefix}{cell.text}"
        if not content:
            out.append(prefix.rstrip())
            return
        lines = content.splitlines()
        out.append(prefix + lines[0])
        for line in lines[1:]:
            out.append("    " + line if line else "")

    def emit_row(cells: tuple[TableCell, ...]) -> None:
        if not cells:
            return
        emit_cell("- - ", cells[0])
        for cell in cells[1:]:
            emit_cell("  - ", cell)
        out.append("")

    for row in (*table.headers, *table.rows):
        emit_row(row)

    out.append(":::")
    return "\n".join(out)


def pipe_table(table: Table) -> str:
    """Render a flat, single-line ``Table`` as a generic Pandoc/Quarto pipe table.

    Fails fast if the table requires multi-row headers, row/column spans, or multi-line
    cells, which Markdown pipe tables cannot represent portably across Typst, LaTeX, and
    DOCX.
    """
    if len(table.headers) != 1:
        raise ValueError(
            "Markdown pipe tables support only a single header row; "
            "use list_table for MultiIndex columns"
        )

    def pipe_cell(cell: TableCell) -> str:
        if cell.rowspan > 1 or cell.colspan > 1:
            raise ValueError(
                "Markdown pipe tables do not support rowspan or colspan; use list_table"
            )
        if "\n" in cell.text:
            raise ValueError(
                "Markdown pipe tables do not support multi-line cells; use list_table"
            )
        return cell.text.replace("|", r"\|")

    def format_row(cells: tuple[TableCell, ...]) -> str:
        return "| " + " | ".join(pipe_cell(cell) for cell in cells) + " |"

    out = [
        format_row(table.headers[0]),
        "|" + "|".join(PIPE_ALIGNMENT_RULES[a] for a in table.aligns) + "|",
        *(format_row(row) for row in table.rows),
    ]

    attrs: list[str] = []
    if table.table_id:
        attrs.append(f"#{table.table_id}")
    if table.colwidths is not None:
        widths = ",".join(
            f"{w:g}" if isinstance(w, float) else str(w) for w in table.colwidths
        )
        attrs.append(f'tbl-colwidths="[{widths}]"')

    if table.caption or attrs:
        attr_block = f" {{{' '.join(attrs)}}}" if attrs else ""
        caption_body = f" {table.caption}" if table.caption else ""
        out.extend(["", f":{caption_body}{attr_block}"])

    return "\n".join(out)


def qtable(
    df: pd.DataFrame,
    *,
    caption: str | None = None,
    table_id: str | None = None,
    colwidths: Sequence[int | float] | None = None,
    aligns: str | Sequence[Alignment] | None = None,
    include_index: bool = False,
    render: TableRenderer = list_table,
) -> str:
    """Build a validated ``Table`` from ``df`` and render it with ``render``."""
    return render(
        table_from_dataframe(
            df,
            caption=caption,
            table_id=table_id,
            colwidths=colwidths,
            aligns=aligns,
            include_index=include_index,
        )
    )
